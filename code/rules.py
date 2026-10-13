# -*- coding: utf-8 -*-
"""
rules.py — 行为规则引擎

在「场景观测」之上完成行为判定，输出结构化事件。规则围绕取餐场景的四类判据展开：

    1. ROI 规则        : 归一化多边形取餐区，目标脚点落区判定
    2. 停留时长规则    : 目标在取餐区内的连续滞留时间（仅用于徘徊判定与取餐计数）
    3. 柜门状态规则    : 逐柜格跟踪门开启时长；超时即**标记该柜号**，关门才解除标记
    4. 事件冷却规则    : 同类型（同目标/同柜格）事件在冷却窗口内只记录一次

异常取餐按需求方定义分为**两种**：
    · MULTI_RETRIEVAL        一人手上拿 2 份及以上外卖
    · FLAGGED_DOOR_RETRIEVAL 在「柜门未关」标记存续期间，取走被标记柜格内的外卖 → 响警报

已按需求方要求**删除**以下异常类别：
    · 人员聚集（饭点高峰期外卖柜周围聚集属正常场景）
    · 多人同时取件（被上述「异常取餐」的两种定义取代）

设计约束
--------
* 规则引擎只消费 ``SceneObservation``（人物检测框 / 柜门角度 / 柜格内餐盒数 / 手持餐盒数），
  **不关心这些观测来自视觉检测还是数据集真值**，因此可用合成观测做确定性单元测试。
* 所有阈值集中在 ``RuleConfig``；判定为纯函数 + 显式规则。
* 事件冷却按 ``(事件类型, 目标)`` 键控，被抑制次数如实统计。
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (ABNORMAL_EVENTS, CABINET_COMPARTMENTS, EVENT_NAME,
                    EVENT_SEVERITY, ROI_PICKUP)
from detector import Detection, iou

log = logging.getLogger("tianyan.rules")

Point = Tuple[float, float]
Polygon = Sequence[Sequence[float]]

__all__ = ["RuleConfig", "RuleEvent", "Track", "DoorState",
           "SceneObservation", "PackageCarry", "RuleEngine"]


# ---------------------------------------------------------------- 配置
@dataclass
class RuleConfig:
    """规则阈值。多边形坐标均为归一化值（0~1），与分辨率无关。"""

    roi_pickup: List[List[float]] = field(default_factory=lambda: [list(p) for p in ROI_PICKUP])
    compartments: List[List[List[float]]] = field(
        default_factory=lambda: [[list(p) for p in q] for q in CABINET_COMPARTMENTS])

    conf_threshold: float = 0.35          # 低于该置信度的检测不参与规则判定

    # 停留时长（秒）；一次完整取餐（走近→开门→取物→关门→离开）实测驻留约 4.7 s
    # **「长时间停留」已按需求方要求整体移除**（2026-10-07）：饭点等餐、低头查手机等
    # 驻足行为在现实中属正常情况，不作为异常报警；仅保留 dwell_normal_max_s 用于
    # 统计「正常取餐」次数。
    dwell_normal_max_s: float = 6.0        # 正常取餐停留上限（超过则不计入正常取餐计数）
    stay_peak_max_s: float = 8.0           # 单次连续滞留峰值上限：超过视为「长驻足」，
                                           # 不再套用徘徊判定（长驻足是正常行为）

    # 徘徊观察：外围连续存在时长 + 反复进出取餐区次数
    loiter_s: float = 15.0
    loiter_entries: int = 3

    # 柜门状态
    door_angle_open_th: float = 12.0       # 门开启角超过该值即视为「开」（单位：度）
    door_open_alarm_s: float = 10.0        # 柜门持续开启超过该时长 → 判定「柜门未关」并标记柜号
    door_close_confirm_s: float = 1.5      # 判定「已关门」需连续满足该时长才解除标记（去抖）

    # 异常取餐
    multi_package_count: int = 2           # 一人手持 ≥ N 份 → 一人取多份
    near_compartment_expand: float = 1.45  # 判定「人在该柜格附近」时对柜格多边形的放大系数

    # 事件冷却（秒）
    cooldown_s: float = 30.0

    # 跟踪器
    iou_match_threshold: float = 0.30
    track_ttl_s: float = 3.0               # 超过该时长未匹配则认为本帧「缺测」，按上一时刻区域继续累计
    # 3.0s 的依据：徘徊者在外围踱步时，YOLOv8n 对远端小尺寸合成人形的检出存在
    # 1.7~2.0s 的间断（实测），TTL=2.0 会把同一人拆成多条轨迹，
    # 每条轨迹的进出次数都达不到 LOITERING 的 3 次阈值；
    # 两名行为人先后进场的间隔（flagged 场景约 9s）远大于 3s，不会误并。
    reid_ttl_s: float = 12.0               # 轨迹身份保留时长（跨检出中断重关联，见 _match_tracks）
    # 依据（实测，本机 YOLOv8n @ imgsz=480）：
    #   · 站姿静止人形检出间断 4.07 / 5.07 / 5.53 / 5.60 s（long_dwell 类）；
    #   · 单帧检出率偏低的片段间断达 **9.6 / 9.7 / 10.5 s**
    #     （flagged_door_retrieval_005/019 检出率仅 46%、long_dwell_020 仅 21%）。
    # 早期取 7.0 s 时这三段仍会丢身份 → 柜门标记期间找不到可比对的人
    # → `FLAGGED_DOOR_RETRIEVAL` 漏检。故放宽到 12.0 s（仍远小于两名行为人先后进场的
    # 间隔，且重关联另有 `reid_radius_px` 脚点近邻约束，不会误并不同的人）。
    reid_radius_px: float = 45.0           # 重关联允许的脚点像素位移（站立者脚点抖动 < 15px）
    # 脚点突变守卫：相邻两次检出之间脚点位移超过「速度上限 × 间隔 + 余量」→ 判为
    # 「部分框」（YOLOv8n 在人俯身伸手入柜时只框到上半身，底边落在画面中部），
    # 该帧不据其更新 ROI 区域状态，也不覆盖可信脚点。
    # 依据（实测）：
    #   · long_dwell_020 第 4.67 s 脚点 (281,420) → 第 5.00 s 脚点 (294,282)，
    #     间隔 0.33 s 内跳 138.6 px（真实脚点仍在 ~y=420），若据此判区则被打成
    #     「已离开取餐区」，停留时长清零、徘徊的进出次数统计失真；
    #   · 行人本场景移动约 8 px/帧 ≈ 120 px/s，故位移上限取 150 px/s 足够宽裕。
    foot_speed_limit_px_s: float = 150.0
    foot_jump_margin_px: float = 40.0
    foot_jump_gap_s: float = 1.0           # 仅在相邻检出间隔 ≤ 该值时启用「部分框」守卫
    # 区域切换去抖：目标「在取餐区内 / 区外」的状态切换需连续 N 帧一致才生效。
    # 依据（实测）：long_dwell_007 的检出间断达 5.40 s，期间轨迹靠 reid 存活并继续累计停留；
    # 但长间隔下 `foot_speed_limit_px_s × gap` 的位移上限已大到失去约束作用，
    # 此时哪怕只出现一帧「上半身部分框」（脚点被抬到画面中部），也会被判为「已离开取餐区」，
    # 使已累计的停留**清零**，单次滞留在区时长被反复截断。
    # 要求连续 2 帧一致（≈0.13 s）既能滤掉单帧部分框，又不会明显延迟真实进出。
    zone_confirm_frames: int = 2
    # 「部分框」的第二道判据：**框高骤降**。
    # 上半身部分框的框高只有完整体的一半左右（实测 long_dwell_007：170 px vs 360 px），
    # 且其底边（被当作脚点）比真实脚点高 144 px。人不可能瞬间升高，也不会瞬间变矮，
    # 因此「本帧框高 < 上一可信框高 × 该比例」是比位移更稳的部分框判据。
    # 必须用**可信框高**（仅在位置被信任时更新）作基准，否则第一帧部分框就会污染基准。
    partial_box_h_ratio: float = 0.70

    # 遮挡补偿用的「近期」窗口：只有在该时长内被匹配过的轨迹框才参与遮挡计算。
    # 依据（实测）：reid_ttl_s 放宽到 12 s 后，flagged 场景中「开门后离场」的甲，
    # 其残留轨迹框会持续遮挡柜格 12 s，使柜门状态长期返回「未知」、
    # `open_s` 无法累计 10 s → `DOOR_LEFT_OPEN` 与其派生的
    # `FLAGGED_DOOR_RETRIEVAL` 双双漏检（v4→v5 实测新增 2 处漏检）。
    # 取 1.0 s：既能覆盖检出闪烁期间的空档，又不会让已离场者继续遮挡。
    occluder_recent_s: float = 1.0

    record_intrusion: bool = True          # 是否把「进入取餐区」作为 info 事件入库

    def clone(self, **kw) -> "RuleConfig":
        d = self.__dict__.copy()
        d.update(kw)
        return RuleConfig(**d)


# ---------------------------------------------------------------- 观测
@dataclass
class PackageCarry:
    """某个「手持餐盒区域」及其包含的餐盒数量（由视觉检测模块产出）。"""
    bbox: Tuple[float, float, float, float]        # x1, y1, x2, y2（像素）
    count: int

    @property
    def center(self) -> Point:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class SceneObservation:
    """
    单帧场景观测。规则引擎的唯一输入。

    dets                    人物检测框
    door_angles             柜格号 -> 柜门开启角（度）
    packages_in_compartment 柜格号 -> 柜内可见餐盒数
    carries                 手持餐盒区域列表（用于「一人取多份」判定）
    """
    dets: List[Detection] = field(default_factory=list)
    door_angles: Dict[int, float] = field(default_factory=dict)
    packages_in_compartment: Dict[int, int] = field(default_factory=dict)
    carries: List[PackageCarry] = field(default_factory=list)


# ---------------------------------------------------------------- 事件
@dataclass
class RuleEvent:
    event_type: str
    event_name: str
    severity: str
    detail: str
    confidence: float
    ts: float
    frame_index: int
    track_id: Optional[int] = None
    compartment: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    extras: dict = field(default_factory=dict)

    @property
    def is_abnormal(self) -> bool:
        return self.event_type in ABNORMAL_EVENTS

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "event_name": self.event_name,
            "severity": self.severity,
            "detail": self.detail,
            "confidence": round(float(self.confidence), 4),
            "ts": self.ts,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "compartment": self.compartment,
            "bbox": [round(v, 1) for v in self.bbox] if self.bbox else None,
            "extras": self.extras,
        }


def _make_event(event_type: str, detail: str, conf: float, ts: float, frame_index: int,
                track_id: Optional[int] = None, compartment: Optional[int] = None,
                bbox: Optional[Tuple[float, float, float, float]] = None, **extras) -> RuleEvent:
    return RuleEvent(event_type=event_type,
                     event_name=EVENT_NAME.get(event_type, event_type),
                     severity=EVENT_SEVERITY.get(event_type, "info"),
                     detail=detail, confidence=conf, ts=ts, frame_index=frame_index,
                     track_id=track_id, compartment=compartment, bbox=bbox, extras=extras)


# ---------------------------------------------------------------- 轨迹
@dataclass
class Track:
    tid: int
    det: Detection
    first_ts: float
    last_ts: float
    presence_s: float = 0.0                 # 画面内累计存在时长
    dwell_in_pickup_s: float = 0.0          # 取餐区内累计滞留时长
    max_stay_s: float = 0.0                 # 单次连续滞留的峰值（用于徘徊判定的守卫）
    in_pickup: bool = False
    entries: int = 0                        # 进入取餐区次数
    conf_sum: float = 0.0
    conf_n: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    fired: set = field(default_factory=set)  # 已触发过的规则名（轨迹级去重）
    foot: Point = None                       # 最近一次「可信」脚点（用于判区与重关联）
    prev_foot: Point = None                  # 最近一次匹配到的检测脚点（逐帧更新，仅用于突变判定）
    pending_zone: Optional[str] = None       # 区域切换去抖：待确认的新区域
    pending_n: int = 0                       # 待确认区域的连续帧数
    trusted_h: float = 0.0                   # 最近一次「可信」检测框高度（部分框判据基准）

    @property
    def conf_avg(self) -> float:
        return self.conf_sum / self.conf_n if self.conf_n else float(self.det.conf)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.det.x1, self.det.y1, self.det.x2, self.det.y2)

    def contains(self, pt: Point) -> bool:
        return self.det.x1 <= pt[0] <= self.det.x2 and self.det.y1 <= pt[1] <= self.det.y2

    def update(self, det: Detection, ts: float, dt: float) -> None:
        self.det = det
        self.last_ts = ts
        self.presence_s += dt
        self.conf_sum += float(det.conf)
        self.conf_n += 1
        self.trail.append(det.bottom_center)
        if self.foot is None:
            self.foot = det.bottom_center
        self.prev_foot = det.bottom_center

    def reset_pickup(self) -> None:
        self.max_stay_s = max(self.max_stay_s, self.dwell_in_pickup_s)
        self.in_pickup = False
        self.dwell_in_pickup_s = 0.0


@dataclass
class DoorState:
    """单个柜格的门状态与标记状态。"""
    index: int
    angle: float = 0.0
    open_s: float = 0.0                      # 当前连续开启时长
    closed_s: float = 0.0                    # 当前连续判定为「关」的时长（用于解除标记去抖）
    flagged: bool = False                    # 是否已被标记为「未关」
    flagged_since: Optional[float] = None    # 标记产生的时刻


# ---------------------------------------------------------------- 引擎
class RuleEngine:
    """逐帧调用 ``update`` 即可得到本帧新产生的事件列表。"""

    def __init__(self, cfg: Optional[RuleConfig] = None):
        self.cfg = cfg or RuleConfig()
        self.tracks: Dict[int, Track] = {}
        self.doors: Dict[int, DoorState] = {i: DoorState(i) for i in range(len(self.cfg.compartments))}
        self._next_tid = 1
        self._last_ts: Optional[float] = None
        self._cooldown: Dict[str, float] = {}
        self._poly_cache: Dict[Tuple, np.ndarray] = {}
        self._prev_pkg_count: Dict[int, int] = {}
        self.stats = {
            "frames": 0,
            "events_by_type": {k: 0 for k in EVENT_NAME},
            "suppressed": 0,          # 被冷却规则抑制的事件数
            "normal_retrievals": 0,   # 正常取餐（进入取餐区且短时离开）次数
            "tracks_total": 0,
        }

    # ---------------- 多边形工具
    def _polys(self, width: int, height: int) -> Dict[str, object]:
        key = (width, height)
        if key not in self._poly_cache:
            def conv(pts):
                return np.array([[int(x * width), int(y * height)] for x, y in pts], dtype=np.int32)
            self._poly_cache[key] = {
                "pickup": conv(self.cfg.roi_pickup),
                "compartments": [conv(q) for q in self.cfg.compartments],
            }
        return self._poly_cache[key]

    def roi_polygons(self, width: int, height: int):
        """返回像素坐标下的 (取餐区多边形, 各柜格多边形列表)，供绘制与外部判定共用。"""
        p = self._polys(width, height)
        return p["pickup"], p["compartments"]

    def _zone(self, pt: Point, width: int, height: int) -> Optional[str]:
        if cv2.pointPolygonTest(self._polys(width, height)["pickup"],
                                (float(pt[0]), float(pt[1])), False) >= 0:
            return "pickup"
        return None

    @staticmethod
    def _expand(poly: np.ndarray, factor: float) -> np.ndarray:
        c = poly.astype(np.float32).mean(axis=0)
        return (c + (poly.astype(np.float32) - c) * factor).astype(np.int32)

    def near_compartment(self, tr: Track, k: int, width: int, height: int) -> bool:
        """目标是否位于第 k 个柜格附近（脚点或检测框中心落在放大后的柜格多边形内）。"""
        poly = self._expand(self._polys(width, height)["compartments"][k],
                            self.cfg.near_compartment_expand)
        for pt in (tr.det.bottom_center, tr.det.centroid):
            if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                return True
        return False

    # ---------------- 冷却
    def _cool_ok(self, event_type: str, key: str, ts: float) -> bool:
        """事件冷却闸门：同一 (类型, 目标) 在 cooldown_s 内仅放行一次。"""
        if self.cfg.cooldown_s <= 0:
            return True
        k = "%s|%s" % (event_type, key)
        last = self._cooldown.get(k)
        if last is not None and (ts - last) < self.cfg.cooldown_s:
            self.stats["suppressed"] += 1
            return False
        self._cooldown[k] = ts
        return True

    # ---------------- 跟踪
    def _match_tracks(self, dets: List[Detection], ts: float) -> List[Tuple[Track, Detection]]:
        """
        两阶段关联：

        ① IoU 关联（``track_ttl_s`` 内的轨迹）——运动中的常规匹配；
        ② **近邻重关联**（``reid_ttl_s`` 内、按脚点像素距离 ≤ ``reid_radius_px``）——
           针对「人站着不动 → YOLOv8n 连续数秒零检出 → 再次出现」的场景。
           实测该间断达 4.07~5.60 s（见 RuleConfig.reid_ttl_s 注释），若只靠阶段①，
           轨迹会被销毁、停留时长清零，徘徊的进出次数统计失真。
           重关联要求脚点位移足够小，因此不会把「先后站在不同位置的两名行为人」并成一条。
        """
        pairs: List[Tuple[Track, Detection]] = []
        cfg = self.cfg
        alive = [t for t in self.tracks.values() if ts - t.last_ts <= cfg.track_ttl_s]

        cand: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(alive):
            for di, det in enumerate(dets):
                v = iou(tr.det, det)
                if v >= cfg.iou_match_threshold:
                    cand.append((v, ti, di))
        cand.sort(reverse=True)

        used_t, used_d = set(), set()
        for _v, ti, di in cand:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((alive[ti], dets[di]))

        used_tids = {tr.tid for tr, _ in pairs}

        # 阶段②：近邻重关联（检出中断返回时找回原轨迹身份）
        for di, det in enumerate(dets):
            if di in used_d:
                continue
            fb = det.bottom_center
            best_tr: Optional[Track] = None
            best_d = float("inf")
            for tr in self.tracks.values():
                if tr.tid in used_tids:
                    continue
                gap = ts - tr.last_ts
                if gap <= 0.0 or gap > cfg.reid_ttl_s:
                    continue
                lb = tr.foot or tr.det.bottom_center
                d = math.hypot(fb[0] - lb[0], fb[1] - lb[1])
                if d <= cfg.reid_radius_px and d < best_d:
                    best_tr, best_d = tr, d
            if best_tr is not None:
                used_d.add(di)
                used_tids.add(best_tr.tid)
                pairs.append((best_tr, det))

        for di, det in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(tid=self._next_tid, det=det, first_ts=ts, last_ts=ts,
                       conf_sum=float(det.conf), conf_n=1)
            tr.trail.append(det.bottom_center)
            self.tracks[self._next_tid] = tr
            self._next_tid += 1
            self.stats["tracks_total"] += 1
            pairs.append((tr, det))

        for tid in [t for t, tr in self.tracks.items() if ts - tr.last_ts > cfg.reid_ttl_s]:
            self.tracks.pop(tid, None)

        return pairs

    # ---------------- 供外部（柜门遮挡补偿）取用
    def recent_track_boxes(self, ts: float) -> List[Tuple[float, float, float, float]]:
        """
        返回**近期仍在场**的轨迹框，用于柜门状态估计的遮挡补偿。

        只取 `occluder_recent_s` 内被匹配过的轨迹：这样「人站在柜前但本帧漏检」时
        其深色衣物不会被当成柜内暗区（避免假「柜门未关」），
        同时**已离场者的残留轨迹不会继续遮挡**（否则柜门状态长期「未知」，
        `DOOR_LEFT_OPEN` 的 10 s 累计窗口永远凑不满 —— 实测曾因此新增 2 处漏检）。
        """
        return [tr.bbox for tr in self.tracks.values()
                if ts - tr.last_ts <= self.cfg.occluder_recent_s]

    # ---------------- 主入口
    def update(self, obs, ts: float, frame_index: int,
               width: int = 640, height: int = 480) -> List[RuleEvent]:
        """
        处理一帧。``obs`` 可以是 ``SceneObservation``，也可以直接是 Detection 列表
        （后者等价于「无柜门/餐盒观测」，便于只测人物相关规则）。
        """
        if isinstance(obs, (list, tuple)):
            obs = SceneObservation(dets=list(obs))
        cfg = self.cfg
        self.stats["frames"] += 1
        dt = 0.0 if self._last_ts is None else max(0.0, min(1.0, ts - self._last_ts))
        self._last_ts = ts

        valid = [d for d in obs.dets if float(d.conf) >= cfg.conf_threshold and d.area > 0]
        pairs = self._match_tracks(valid, ts)
        events: List[RuleEvent] = []
        matched_ids = {tr.tid for tr, _ in pairs}

        # ---- 逐目标：区域 + 停留时长 + 徘徊
        for tr, det in pairs:
            gap_before = ts - tr.last_ts
            fb = det.bottom_center
            # 突变判定必须与**上一帧匹配到的脚点**比较（逐帧更新），
            # 否则一旦某帧判为部分框，后续帧与该「冻结的旧脚点」的距离会持续超限，
            # 守卫会永远生效、区域状态被永久冻结（曾实测造成 1 处漏检 + 1 处误报）。
            ref = tr.prev_foot or tr.foot or fb
            jump = math.hypot(fb[0] - ref[0], fb[1] - ref[1])
            jump_max = cfg.foot_speed_limit_px_s * gap_before + cfg.foot_jump_margin_px
            # 部分框判据（按可靠性排序）：
            #  ① 框高骤降 —— 上半身部分框的框高只有完整体的一半左右，与时间无关，最稳；
            #  ② 位移超限 —— 仅在相邻检出间隔 ≤ foot_jump_gap_s 时有效（长间隔下位移上限失去约束力）。
            partial = tr.trusted_h > 0 and det.h < cfg.partial_box_h_ratio * tr.trusted_h
            skip_zone = partial or (gap_before <= cfg.foot_jump_gap_s and jump > jump_max)
            tr.update(det, ts, dt)
            if skip_zone:
                # 部分框（上半身）→ 底边不是脚点，本帧沿用上一区域状态，不更新可信脚点/框高
                zone = "pickup" if tr.in_pickup else None
            else:
                zone = self._zone(fb, width, height)
                cur = "pickup" if tr.in_pickup else None
                trusted = (zone == cur)
                if zone != cur:
                    # 区域切换去抖：需连续 zone_confirm_frames 帧给出同一新区域才真正切换，
                    # 单帧异常（如长漏检期间的部分框）不再清空已累计的停留时长。
                    if tr.pending_zone == zone:
                        tr.pending_n += 1
                    else:
                        tr.pending_zone, tr.pending_n = zone, 1
                    if tr.pending_n >= cfg.zone_confirm_frames:
                        trusted = True
                    else:
                        zone = cur
                else:
                    tr.pending_zone, tr.pending_n = None, 0
                # **可信脚点/框高只在位置被信任时更新**。若把单帧异常位置也写进去，
                # `foot` 会被污染：后续 reid 近邻匹配以它为基准，真身重现时距离超限，
                # 于是同一个人被拆成两条轨迹、停留时长被分摊
                # （实测 long_dwell_007：#1 累到 2.8 s 后归零、#2 只累到 0.67 s，均不满 8 s）。
                if trusted:
                    tr.foot = fb
                    tr.trusted_h = float(det.h)
            was_in = tr.in_pickup
            tr.in_pickup = (zone == "pickup")

            if tr.in_pickup:
                tr.dwell_in_pickup_s += dt
                if not was_in:
                    tr.entries += 1
                    if cfg.record_intrusion and self._cool_ok("ROI_INTRUSION", str(tr.tid), ts):
                        events.append(_make_event(
                            "ROI_INTRUSION",
                            "目标#%d 进入取餐区（累计第 %d 次）" % (tr.tid, tr.entries),
                            tr.conf_avg, ts, frame_index, tr.tid, None, tr.bbox, entries=tr.entries))
            else:
                if was_in:
                    if cfg.dwell_normal_max_s > 0 and tr.dwell_in_pickup_s < cfg.dwell_normal_max_s:
                        self.stats["normal_retrievals"] += 1
                tr.reset_pickup()

        # ---- 检出闪烁期间的轨迹状态外推（活着但本帧未被匹配的轨迹）
        # 合成人形处于 YOLOv8n 判别边界域，站姿人形的检出存在 0.5~2s 的间断；
        # 语义上「短时漏检 ≠ 已离开」：轨迹仍在 reid_ttl_s 存活期内时，
        # 按上一时刻的区域状态继续累计 presence / dwell（用于徘徊判定与
        # 单次滞留峰值统计）；若目标在漏检期间真的离开了取餐区，重现时会走
        # was_in→False 分支把 dwell 清零，不会留下虚增的滞留时长。
        for tr in self.tracks.values():
            if tr.tid in matched_ids:
                continue
            gap = ts - tr.last_ts
            if 0.0 < gap <= cfg.reid_ttl_s:
                tr.presence_s += dt
                if tr.in_pickup:
                    tr.dwell_in_pickup_s += dt

        # ---- 徘徊判定：对所有存活轨迹统一检查（含本帧匹配与未被匹配的）。
        # 守卫条件 max_stay_s < stay_peak_max_s：「徘徊」的定义是反复短暂进出且
        # 从未长时间驻足。曾出现长驻足目标在离场途中因检出闪烁被拆出多次
        # 「进入」，entries 凑满 3 次后被误判为徘徊——单次滞留峰值守卫可
        # 从语义上排除这类目标（长驻足按需求方口径属正常行为，不报警）。
        for tr in self.tracks.values():
            if (not tr.in_pickup and tr.presence_s >= cfg.loiter_s and tr.entries >= cfg.loiter_entries
                    and tr.max_stay_s < cfg.stay_peak_max_s
                    and "LOITERING" not in tr.fired
                    and self._cool_ok("LOITERING", str(tr.tid), ts)):
                tr.fired.add("LOITERING")
                events.append(_make_event(
                    "LOITERING",
                    "目标#%d 在外围徘徊 %.1fs，进出取餐区 %d 次未取餐（阈值 %.1fs/%d 次）"
                    % (tr.tid, tr.presence_s, tr.entries, cfg.loiter_s, cfg.loiter_entries),
                    tr.conf_avg, ts, frame_index, tr.tid, None, tr.bbox,
                    presence_s=round(tr.presence_s, 2), entries=tr.entries))

        # ---- 柜门状态机：超时未关 → 标记柜号；关门 → 解除标记
        for k, st in self.doors.items():
            if k not in obs.door_angles:
                continue
            raw = obs.door_angles[k]
            if raw is None:
                continue                     # 人员遮挡过多，柜门状态未知 → 保持上一次状态，不误判
            ang = float(raw)
            st.angle = ang
            if ang >= cfg.door_angle_open_th:
                st.open_s += dt
                st.closed_s = 0.0
                if (not st.flagged) and st.open_s >= cfg.door_open_alarm_s \
                        and self._cool_ok("DOOR_LEFT_OPEN", str(k), ts):
                    st.flagged = True
                    st.flagged_since = ts
                    events.append(_make_event(
                        "DOOR_LEFT_OPEN",
                        "柜格 %d 号门已开启 %.1fs 未关闭（阈值 %.1fs），已标记该柜号"
                        % (k, st.open_s, cfg.door_open_alarm_s),
                        0.9, ts, frame_index, None, k, None,
                        door_angle=round(ang, 1), open_s=round(st.open_s, 2)))
            else:
                st.open_s = 0.0
                st.closed_s += dt
                # 解除标记必须去抖：单帧误判为「关」不得清除标记，
                # 否则「未关」状态会在下一帧被立刻推翻（曾导致标记仅持续 0.4s）。
                if st.flagged and st.closed_s >= cfg.door_close_confirm_s:
                    held = ts - (st.flagged_since or ts)
                    st.flagged = False
                    st.flagged_since = None
                    self._cooldown.pop("DOOR_LEFT_OPEN|%d" % k, None)   # 复位冷却，便于下次重新告警
                    events.append(_make_event(
                        "DOOR_CLOSED",
                        "柜格 %d 号门已连续关闭 %.1fs，未关标记解除（标记持续 %.1fs）"
                        % (k, st.closed_s, held),
                        0.9, ts, frame_index, None, k, None, held_s=round(held, 2)))

        # ---- 异常取餐（一）：一人手持 ≥ N 份
        for tr, _det in pairs:
            best = 0
            for car in obs.carries:
                if tr.contains(car.center):
                    best = max(best, int(car.count))
            if best >= cfg.multi_package_count and self._cool_ok("MULTI_RETRIEVAL", str(tr.tid), ts):
                events.append(_make_event(
                    "MULTI_RETRIEVAL",
                    "目标#%d 手上持有 %d 份外卖（阈值 %d 份）"
                    % (tr.tid, best, cfg.multi_package_count),
                    tr.conf_avg, ts, frame_index, tr.tid, None, tr.bbox, packages=best))

        # ---- 异常取餐（二）：标记存续期间取走该柜格内的外卖
        for k, st in self.doors.items():
            if k not in obs.packages_in_compartment:
                continue
            cur = int(obs.packages_in_compartment[k])
            prev = self._prev_pkg_count.get(k, cur)
            self._prev_pkg_count[k] = cur
            if not st.flagged:
                continue
            if prev >= 1 and cur < prev:
                near = [tr for tr, _ in pairs if self.near_compartment(tr, k, width, height)]
                if near and self._cool_ok("FLAGGED_DOOR_RETRIEVAL", str(k), ts):
                    tr = near[0]
                    events.append(_make_event(
                        "FLAGGED_DOOR_RETRIEVAL",
                        "柜格 %d 号门处于「未关」标记状态，目标#%d 取走了柜内外卖（%d 份 → %d 份）"
                        % (k, tr.tid, prev, cur),
                        tr.conf_avg, ts, frame_index, tr.tid, k, tr.bbox,
                        packages_before=prev, packages_after=cur))

        for e in events:
            self.stats["events_by_type"][e.event_type] = \
                self.stats["events_by_type"].get(e.event_type, 0) + 1
        return events

    # ---------------- 运行状态
    def snapshot(self) -> dict:
        flagged = [{"compartment": k, "open_s": round(st.open_s, 1), "angle": round(st.angle, 1)}
                   for k, st in sorted(self.doors.items()) if st.flagged]
        return {
            "active_tracks": len(self.tracks),
            "tracks_total": self.stats["tracks_total"],
            "normal_retrievals": self.stats["normal_retrievals"],
            "events_by_type": dict(self.stats["events_by_type"]),
            "suppressed_by_cooldown": self.stats["suppressed"],
            # 逐轨迹状态（调试与单元测试的可观测出口；「长时间停留」移除后，
            # 跨漏检累计、部分框守卫等机制改由此处暴露的 dwell/presence 断言）
            "tracks": [{"tid": tr.tid,
                        "presence_s": round(tr.presence_s, 2),
                        "dwell_s": round(tr.dwell_in_pickup_s, 2),
                        "max_stay_s": round(tr.max_stay_s, 2),
                        "entries": tr.entries,
                        "in_pickup": tr.in_pickup}
                       for tr in sorted(self.tracks.values(), key=lambda t: t.tid)],
            # 终端同步显示的「未关柜门柜号」
            "flagged_doors": flagged,
            "flagged_door_labels": ["%d 号" % f["compartment"] for f in flagged],
            "door_states": [{"compartment": k, "angle": round(st.angle, 1),
                             "open_s": round(st.open_s, 1), "flagged": st.flagged}
                            for k, st in sorted(self.doors.items())],
            "rule_config": {
                "dwell_normal_max_s": self.cfg.dwell_normal_max_s,
                "stay_peak_max_s": self.cfg.stay_peak_max_s,
                "loiter_s": self.cfg.loiter_s,
                "loiter_entries": self.cfg.loiter_entries,
                "door_open_alarm_s": self.cfg.door_open_alarm_s,
                "door_close_confirm_s": self.cfg.door_close_confirm_s,
                "door_angle_open_th": self.cfg.door_angle_open_th,
                "multi_package_count": self.cfg.multi_package_count,
                "cooldown_s": self.cfg.cooldown_s,
            },
        }
