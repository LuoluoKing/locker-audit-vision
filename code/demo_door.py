# -*- coding: utf-8 -*-
"""
code/demo_door.py — AI 演示素材专用「门态判定」引擎（**演示口径**）

定位
----
`real_video/` 三段素材为 AI 生成视频（非实拍），柜体几何与机位均不稳定，
本引擎提供一条与主链路隔离的「演示档案」判定通道：

    单应稳定化 → 逐格「与基准关闭帧的一致性」(CLAHE-NCC, 双尺度, 扩窗对齐)
    → 逐格状态机（滞回 + 阶跃 + 持续去抖） → 事件（柜门未关 / 取走未关柜门外卖 / 柜门已关）

硬性边界（与文档、界面声明一致）
--------------------------------
* 本引擎的判定**只用于功能演示**，不进入 `data/simulated/` 实验数据集，
  不计入论文任何准确率/误报率指标；
* 「门态变化」的工程含义是「该格口与基准关闭状态的视觉一致性显著下降」，
  并非工业级门磁信号；
* 事件阈值属于**演示档参数**（在 `config.AI_DEMO_DOOR_PROFILE` 中逐素材声明），
  与论文 E1 使用的全局阈值相互独立。

依据与实测记录见 `docs/09_阶段3证据/AI演示素材标定与门态判据记录.md`。
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config as C
from rules import RuleEvent, _make_event


def _clahe(img: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return c.apply(img)


def _to_px(poly: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    return np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)


def _bbox_of(poly_px: np.ndarray) -> Tuple[int, int, int, int]:
    x0, y0 = poly_px.min(axis=0)
    x1, y1 = poly_px.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


class _CellState:
    """单格口状态机（滞回 + 去抖 + 阶跃检查 + 标记/取走判定）。"""

    __slots__ = ("idx", "door_id", "state", "ncc_hist", "changed_n", "open_since",
                 "flagged", "flagged_ts", "closed_confirm_n", "touch_since",
                 "touch_track", "pre_touch_tpl", "last_ncc", "last_ncc2", "retrieved")

    def __init__(self, idx: int, door_id: str):
        self.idx = idx
        self.door_id = door_id
        self.state = "CLOSED"          # CLOSED / OPEN
        self.ncc_hist: deque = deque(maxlen=64)
        self.changed_n = 0             # 连续「变化」帧数（去抖）
        self.open_since: Optional[float] = None
        self.flagged = False
        self.flagged_ts: Optional[float] = None
        self.closed_confirm_n = 0
        self.touch_since: Optional[float] = None
        self.touch_track: Optional[int] = None
        self.pre_touch_tpl: Optional[np.ndarray] = None
        self.last_ncc = 1.0
        self.last_ncc2 = 1.0
        self.retrieved = False             # 本次「开门—关闭」周期内是否已报过取走


class DemoDoorEngine:
    """
    参数见 `config.AI_DEMO_DOOR_PROFILE[clip_name]`；本类不做任何文件路径猜测，
    校准帧从 `real_video/<clip_name>` 读取。
    """

    def __init__(self, clip_name: str, profile: dict,
                 detector=None, log: Optional[Callable[[str], None]] = None):
        self.clip_name = clip_name
        self.profile = profile
        self.log = log or (lambda m: None)
        self.detector = detector                    # 提供 person 框（用于遮挡剔除/靠近判定）
        self.p = dict(profile.get("params", {}))
        self.cells = [c for c in profile["cells"]]

        self._ref_bgr: Optional[np.ndarray] = None
        self._ref_gray: Optional[np.ndarray] = None
        self._orb = cv2.ORB_create(800)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._kref = None
        self._dref = None
        self._mask: Optional[np.ndarray] = None

        # 逐格基准模板（CLAHE 灰度）与可用性
        self._tpl: Dict[int, np.ndarray] = {}
        self._tpl_box: Dict[int, Tuple[int, int, int, int]] = {}
        self._baseline_ok: Dict[int, bool] = {}
        self._cell_bbox_px: Dict[int, Tuple[int, int, int, int]] = {}

        self._states: Dict[int, _CellState] = {}
        # 稳定化开关：机位静态的素材（实测缩放/平移接近恒等）可关闭，省去
        # ORB 特征匹配与整帧 warp（≈25~40ms/帧），提升检测帧率。
        self._stab_on = bool(profile.get("stabilize", True))
        # 操作屏锚定：素材为**数字变焦**（屏幕中心近乎不动、尺寸单向缩放）时，
        # 「以操作屏为锚的相似变换」是精确模型（比 ORB 全图估计稳、快，且天然
        # 带对齐自检：检测不到屏幕 = 无法对齐 → 跳过该帧判定）。
        self._screen_ref = None           # 参考帧的操作屏 (cx, cy, w, h)
        self._screen_tpl = None           # 操作屏灰度模板（多尺度模板匹配用）
        self._cur_scale = None            # 屏幕当前尺度（跟踪状态；None=需全扫重锁）
        self._scr_w_hist = deque(maxlen=9)   # 屏幕宽度历史（稳定性闸门用）
        self._scr_c_hist = deque(maxlen=9)   # 屏幕中心历史
        self._scr_w_ema = None             # 屏幕宽度 EMA（尺度跟踪主力）
        self._anchor_screen = bool(profile.get("screen_anchor", False))
        self._calibrated = False
        self._calib_error = ""
        self._Hinv = None                 # 最近一次估计的单应（逆）
        self._n_updates = 0               # 已处理帧数（决定何时重估单应）
        self._last_thumb = None           # 32x18 灰度缩略图（用于场景跳变检测）
        self._frame_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------ 校准
    def calibrate(self) -> bool:
        """
        离线校准：读取候选参考帧 → 用检测器剔除被人物遮挡的候选 → 取**最早的两个**
        未被遮挡候选做共识基线（中位模板）。
        「最早两个」保证基线偏向片头（门初始关闭）状态；不足两个可用候选的格口
        标记为 `baseline_ok=False`，演示中不参与门态判定。
        """
        src = C.REAL_DEMO_DIR / self.clip_name
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            self._calib_error = "无法打开演示素材：%s" % src
            return False
        # 候选参考帧：优先读档案**顶层**的 base_candidates（params 中的兼容保留）
        cands = list(self.profile.get("base_candidates",
                                      self.p.get("base_candidates", [30, 75, 120, 160])))
        frames: Dict[int, np.ndarray] = {}
        for fi in cands:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                frames[fi] = fr
        # 参考帧（用于格口坐标系与稳像）
        ref_idx = int(self.profile.get("ref_frame", cands[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ok, ref = cap.read()
        cap.release()
        if not ok or not frames:
            self._calib_error = "参考帧读取失败"
            return False
        self._ref_bgr = ref
        h, w = ref.shape[:2]
        self._frame_size = (w, h)
        if self._anchor_screen:
            det = self._detect_screen(ref)
            if det is None:
                self._calib_error = "screen_anchor 已启用但在参考帧未检测到操作屏"
                return False
            self._screen_ref = det
            cx, cy, sw, sh = det
            m = 4
            x0 = max(0, int(cx - sw / 2) - m); x1 = int(cx + sw / 2) + m
            y0 = max(0, int(cy - sh / 2) - m); y1 = int(cy + sh / 2) + m
            self._screen_tpl = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            self.log("操作屏锚定：参考帧屏幕 (%.0f,%.0f %.0fx%.0f)，模板 %dx%d"
                     % (cx, cy, sw, sh, self._screen_tpl.shape[1], self._screen_tpl.shape[0]))
        self._ref_gray = _clahe(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        mask = np.zeros(self._ref_gray.shape, np.uint8)
        x0, y0, x1, y1 = self.profile.get("stab_roi", [0.10, 0.10, 0.80, 0.99])
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
        self._mask = mask
        if self._stab_on:
            self._kref, self._dref = self._orb.detectAndCompute(
                cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), mask)
            if self._dref is None or len(self._kref) < 20:
                self._calib_error = "参考帧特征过少（%d）" % (0 if self._kref is None else len(self._kref))
                return False

        # 候选帧先**稳定化**到参考坐标系再取块——素材镜头存在持续推近，
        # 未对齐的原始帧之间取块互不重合，会让共识基线全部失效（实测教训）。
        # 静态机位素材（stabilize=False）跳过本步骤。
        stab_frames: Dict[int, np.ndarray] = {}
        for fi, fr in frames.items():
            if self._anchor_screen:
                ok_s, sf = self._screen_warp(fr)
                stab_frames[fi] = sf if ok_s else fr
            elif self._stab_on:
                self._Hinv = None
                ok_s, sf = self._stabilize(fr, force=True)
                stab_frames[fi] = sf if ok_s else fr
            else:
                stab_frames[fi] = fr

        # 逐帧的人物框（用于遮挡剔除；在原始帧上检测）
        occ: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for fi, fr in frames.items():
            boxes: List[Tuple[float, float, float, float]] = []
            if self.detector is not None:
                try:
                    boxes = [tuple(d.xyxy()) for d in self.detector.detect(fr)]
                except Exception as exc:            # 检测失败不阻塞校准，仅记录
                    self.log("演示档校准：候选帧 f%d 人物检测失败 %s" % (fi, exc))
            occ[fi] = boxes

        exp = float(self.p.get("occ_expand", 0.15))
        for idx, cell in enumerate(self.cells):
            if cell.get("skip"):
                continue
            door_id = self._door_id(cell)
            self._states[idx] = _CellState(idx, door_id)
            poly = _to_px(cell["poly"], w, h)
            bx0, by0, bx1, by1 = _bbox_of(poly)
            bw, bh = bx1 - bx0, by1 - by0
            # 内缩 6% 再取块（与构建 cells 时的 inset 一致，略再收一点避开边缝）
            bx0i = int(bx0 + 0.06 * bw); bx1i = int(bx1 - 0.06 * bw)
            by0i = int(by0 + 0.06 * bh); by1i = int(by1 - 0.06 * bh)
            self._tpl_box[idx] = (bx0i, by0i, bx1i, by1i)
            self._cell_bbox_px[idx] = (bx0, by0, bx1, by1)

            picks: List[np.ndarray] = []
            for fi in sorted(stab_frames.keys()):
                x0c, y0c = bx0 - exp * bw, by0 - exp * bh
                x1c, y1c = bx1 + exp * bw, by1 + exp * bh
                hit = any(not (bx + bw2 < x0c or bx > x1c or by + bh2 < y0c or by > y1c)
                          for (bx, by, bw2, bh2) in occ[fi])
                if hit:
                    continue
                patch = _clahe(cv2.cvtColor(stab_frames[fi], cv2.COLOR_BGR2GRAY))[by0i:by1i, bx0i:bx1i]
                if patch.size == 0:
                    continue
                picks.append(patch)
                if len(picks) >= 2:                 # 最早的两个未被遮挡候选
                    break
            if len(picks) < 2:
                self._baseline_ok[idx] = False
                continue
            if picks[0].shape != picks[1].shape:
                self._baseline_ok[idx] = False
                continue
            # 一致性用**扩窗对齐**后的 NCC（残差 2~3px 属常态，等尺寸直接比会低估）
            cpad = int(self.p.get("consensus_pad", 6))
            ph, pw = picks[0].shape
            w0 = picks[0][max(0, 0):ph, 0:pw]
            search = np.zeros((ph + 2 * cpad, pw + 2 * cpad), picks[0].dtype)
            search[cpad:cpad + ph, cpad:cpad + pw] = picks[0]
            search[0:cpad, cpad:cpad + pw] = picks[0][0:1, :]
            search[cpad + ph:, cpad:cpad + pw] = picks[0][-1:, :]
            search[:, 0:cpad] = search[:, cpad:cpad + 1]
            search[:, cpad + pw:] = search[:, cpad + pw - 1:cpad + pw]
            n01 = float(cv2.matchTemplate(search, picks[1], cv2.TM_CCOEFF_NORMED).max())
            if n01 < float(self.p.get("consensus_ncc", 0.72)):
                self._baseline_ok[idx] = False        # 两个候选彼此不一致 → 基线不可用
                continue
            tpl = np.uint8((picks[0].astype(np.uint16) + picks[1].astype(np.uint16)) // 2)
            self._tpl[idx] = tpl
            self._baseline_ok[idx] = True

        usable = sum(1 for v in self._baseline_ok.values() if v)
        self.log("演示档（%s）校准完成：可用格口 %d / %d，参考帧 f%d，候选 %s"
                 % (self.clip_name, usable, len(self._states), ref_idx, sorted(frames.keys())))
        for idx, st in self._states.items():
            if not self._baseline_ok.get(idx):
                self.log("  格口 %s：基线不可用（候选帧均被遮挡或彼此不一致），演示中不判定" % st.door_id)
        self._calibrated = True
        return True

    # ------------------------------------------------------------ 主循环
    def update(self, frame_bgr: np.ndarray, ts: float,
               person_boxes: Sequence[Tuple[float, float, float, float]],
               track_of_box: Optional[Sequence[int]] = None) -> List[RuleEvent]:
        """处理一帧（**应在检测帧上调用**，与主链路同一节拍）。返回本帧产生的事件。"""
        if not self._calibrated:
            return []
        events: List[RuleEvent] = []
        h, w = frame_bgr.shape[:2]
        if self._frame_size != (w, h):
            return []

        # ---- 0) 场景跳变检测（播放循环回到首帧 / 硬切）：
        # 跳变瞬间旧单应矩阵失效（镜头推近后的参数被复用到首帧），若不处理会把
        # 「画面跳变」误判为整片柜门的门态变化（实测：循环边界左列 4 格集体假开）。
        # 处理：强制重估单应、清零逐格去抖计数，且**跳变帧本身不参与判定**。
        small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (32, 18),
                           interpolation=cv2.INTER_AREA)
        jump = (self._last_thumb is not None
                and float(np.mean(cv2.absdiff(small, self._last_thumb)))
                > float(self.p.get("scene_jump_th", 22.0)))
        self._last_thumb = small
        if jump:
            self._Hinv = None
            self._cur_scale = None            # 播放循环/硬切：屏幕跟踪重锁
            self._scr_w_hist.clear()
            self._scr_c_hist.clear()
            self._scr_w_ema = None
            for _st in self._states.values():
                _st.changed_n = 0
                _st.closed_confirm_n = 0
                _st.touch_since = None
                _st.touch_track = None
                _st.pre_touch_tpl = None
                _st.ncc_hist.clear()
            self.log("场景跳变（播放循环/硬切）：重锚定稳定化，跳变帧跳过判定")
            return events

        # ---- 1) 稳定化（失败则整帧跳过判定，避免在未对齐画面上误判）
        self._n_updates += 1
        if self._anchor_screen:
            ok_s, frame = self._screen_warp(frame_bgr)
            if not ok_s:
                return events                # 屏幕不可见 = 无法验证对齐 → 本帧不判
        elif self._stab_on:
            stab_ok, frame = self._stabilize(frame_bgr)
            if not stab_ok:
                return events
        else:
            frame = frame_bgr
        gc = _clahe(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # 半数分辨率版（双尺度一致性：小错位在低分辨率下会被吸收）
        gh = cv2.resize(gc, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        pad = int(self.p.get("search_pad", 16))
        p2 = max(4, pad // 2)
        changed_n_th = int(self.p.get("confirm_frames", 4))     # ≈0.3s@12fps
        closed_n_th = int(self.p.get("closed_frames", 6))

        for idx, st in self._states.items():
            if not self._baseline_ok.get(idx):
                continue
            # 被人物遮挡的格口**冻结门态更新**：人在柜前时格口不可观测，
            # 若拿遮挡画面去比基准会把「有人经过」误判为「门态变化」（实测教训：
            # 甲站在左列前 1~7s，左列 4 格集体假开、假标记）。
            # 但**取走判定**恰需在遮挡期内工作（人员接触），故先跑取走逻辑再跳过门态。
            occluded = self._person_overlaps(
                st, person_boxes, expand=float(self.p.get("occ_expand_runtime", 0.05)))
            if st.flagged:
                self._update_retrieval(st, idx, ts, gc, person_boxes, occluded, track_of_box, events)
            if occluded:
                continue
            tpl = self._tpl[idx]
            bx0, by0, bx1, by1 = self._tpl_box[idx]
            win = gc[max(0, by0 - pad):min(h, by1 + pad), max(0, bx0 - pad):min(w, bx1 + pad)]
            if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
                continue
            ncc = float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max())
            # 低分辨率同窗比较（模板同步降采样）
            th, tw = tpl.shape
            tpl2 = cv2.resize(tpl, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
            win2 = gh[max(0, (by0 - pad) // 2):min(h // 2, (by1 + pad) // 2),
                      max(0, (bx0 - pad) // 2):min(w // 2, (bx1 + pad) // 2)]
            ncc2 = float(cv2.matchTemplate(win2, tpl2, cv2.TM_CCOEFF_NORMED).max()) \
                if win2.shape[0] >= tpl2.shape[0] and win2.shape[1] >= tpl2.shape[1] else ncc
            st.last_ncc, st.last_ncc2 = ncc, ncc2
            st.ncc_hist.append((ts, ncc, ncc2))

            open_th = float(self.p.get("open_th", 0.55))
            closed_th = float(self.p.get("closed_th", 0.72))
            changed = (ncc < open_th) and (ncc2 < open_th + 0.08)

            # ---- 状态迁移
            if st.state == "CLOSED":
                if changed:
                    st.changed_n += 1
                else:
                    st.changed_n = max(0, st.changed_n - 1)
                if st.changed_n >= changed_n_th:
                    # 阶跃检查（可选）：历史上用于排除「缓慢漂移/光照渐变」，
                    # 但在多圈循环工况下，持续开门会被滑动窗口误判为缓变而漏检
                    # （实测：第 2/3 圈 changed_n 卡在阈值-1、报警消失）。
                    # 现已有更强防线（共识基线 / 双尺度 / 遮挡冻结），故默认关闭；
                    # 如需启用可设 require_step=True（不建议与循环播放同用）。
                    if self.p.get("require_step", False):
                        ref_ts = ts - float(self.p.get("step_window_s", 0.8))
                        old = [n for (t, n, n2) in st.ncc_hist if t <= ref_ts]
                        if old and (float(np.median(old)) - ncc) < float(self.p.get("step_drop", 0.15)):
                            st.changed_n = changed_n_th - 1
                            continue
                    st.state = "OPEN"
                    st.open_since = ts
                    st.closed_confirm_n = 0
                    st.retrieved = False        # 新一次开门周期：允许再报取走
            else:  # OPEN
                if (ncc >= closed_th) and (ncc2 >= closed_th - 0.06):
                    st.closed_confirm_n += 1
                else:
                    st.closed_confirm_n = 0
                if st.closed_confirm_n >= closed_n_th:
                    st.state = "CLOSED"
                    st.changed_n = 0
                    if st.flagged:
                        st.flagged = False
                        events.append(_make_event(
                            "DOOR_CLOSED",
                            "柜格 %s 门已关闭，解除未关标记" % st.door_id,
                            float(np.clip(ncc, 0, 1)), ts, 0,
                            compartment=idx,
                            door_id=st.door_id, demo=True))

            # ---- 未关柜门（演示档阈值 + 人已离开）
            if st.state == "OPEN" and st.open_since is not None and not st.flagged:
                open_s = ts - st.open_since
                if open_s >= float(self.p.get("door_left_open_s", 6.0)):
                    if not self._person_overlaps(st, person_boxes, expand=0.05):
                        st.flagged = True
                        st.flagged_ts = ts
                        events.append(_make_event(
                            "DOOR_LEFT_OPEN",
                            "柜格 %s 门已开启 %.1fs 未关闭（演示档阈值 %.1fs），已标记该柜号"
                            % (st.door_id, open_s, float(self.p.get("door_left_open_s", 6.0))),
                            0.9, ts, 0, compartment=idx,
                            door_id=st.door_id, open_s=round(open_s, 1), demo=True))
                        self.log("[演示档] 标记柜格 %s 未关（已开启 %.1fs）" % (st.door_id, open_s))

            # ---- 取走未关柜门外卖：见循环顶部的 _update_retrieval（遮挡期内也要工作）
        return events

    # ------------------------------------------------------------ 内部
    def _detect_screen(self, frame: np.ndarray):
        """检测操作屏（深蓝面板）；返回 (cx, cy, w, h) 或 None。"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (100, 40, 30), (135, 255, 150))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            cx, cy = cent[i]
            if (area > 1200 and 0.2 < cy / h < 0.75 and 0.2 < cx / w < 0.8
                    and 0.8 < ww / max(hh, 1) < 2.6):
                if best is None or area > best[4]:
                    best = (cx, cy, float(ww), float(hh), area)
        return None if best is None else (best[0], best[1], best[2], best[3])

    def _detect_screen_tm(self, frame: np.ndarray):
        """多尺度模板匹配定位操作屏（比色域检测稳：不吃阴影/深色衣物干扰）。

        粗扫（步长 0.10）→ 细扫（步长 0.025）；仅搜索屏幕的合理走廊区域。
        匹配度 < 0.5 或尺度超出 [0.9, 1.9] → 视为不可用（返回 None）。
        """
        if self._screen_tpl is None:
            return None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self._screen_tpl.shape
        rx0, rx1 = int(0.28 * w), int(0.72 * w)
        ry0, ry1 = int(0.34 * h), int(0.60 * h)
        win = gray[ry0:ry1, rx0:rx1]
        best = None                        # (ncc, x, y, sw, sh)

        def scan(scales):
            nonlocal best
            for sc in scales:
                sw_, sh_ = int(round(tw * sc)), int(round(th * sc))
                if sw_ < 8 or sh_ < 8 or sw_ > win.shape[1] or sh_ > win.shape[0]:
                    continue
                t = cv2.resize(self._screen_tpl, (sw_, sh_), interpolation=cv2.INTER_AREA)
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, v, _, loc = cv2.minMaxLoc(r)
                if best is None or v > best[0]:
                    best = (float(v), loc[0], loc[1], sw_, sh_)

        if self._cur_scale is None:
            # 全扫重锁（启动 / 场景跳变后）：粗扫 + 细扫
            scan(np.arange(0.90, 1.90, 0.10))
            if best is None or best[0] < 0.45:
                return None
            c = best[0]
            scan(np.arange(max(0.90, c - 0.085), min(1.90, c + 0.086), 0.025))
        else:
            # 跟踪：尺度只能缓变（约束窗口），避免跨尺度错配（实测振荡 60↔94px）
            c = self._cur_scale
            scan(np.arange(max(0.90, c - 0.14), min(1.90, c + 0.15), 0.03))
        if best is None or best[0] < 0.55:
            return None
        v, x, y, sw_, sh_ = best
        self._cur_scale = sw_ / float(tw)
        return (rx0 + x + sw_ / 2.0, ry0 + y + sh_ / 2.0, float(sw_), float(sh_))

    def _screen_warp(self, frame: np.ndarray):
        """按操作屏锚点把当前帧对齐回参考坐标系（缩放+平移的相似变换）。

        数字变焦素材的正确模型；检测不到操作屏（被人物遮挡等）→ 返回失败，
        调用方跳过该帧判定（宁可不判、不可错判）。
        """
        cur = self._detect_screen(frame)
        if cur is None:
            return False, frame
        if self._screen_ref is None:
            return False, frame
        rcx, rcy, rw, _rh = self._screen_ref
        # 走廊过滤：单帧报告中心离参考位置过远（深色衣物/阴影混入造成）→ 丢弃该样本
        if abs(cur[0] - rcx) > 80 or abs(cur[1] - rcy) > 80:
            return False, frame
        # 尺度估计：带离群拒斥的 EMA（α=0.35，滞后 ~2 帧）。
        # 变焦是连续斜坡——纯中位滤波滞后 ~0.4s，会造成远离锚点的底排查格瞬态错位
        # （实测：R7C2 在人物入场段被光影变化顶出假「开门」）；EMA 滞后小且平滑。
        cur_w = float(cur[2])
        self._scr_w_hist.append(cur_w)
        self._scr_c_hist.append((float(cur[0]), float(cur[1])))
        if len(self._scr_w_hist) < 4:      # 预热：样本不足不判（循环重启后 ~0.3s）
            self._scr_w_ema = cur_w
            return False, frame
        if self._scr_w_ema is None:
            self._scr_w_ema = float(np.median(self._scr_w_hist))
        if abs(cur_w - self._scr_w_ema) > 0.28 * self._scr_w_ema:
            return False, frame           # 单帧离群（阴影/衣物混入）→ 该帧弃权
        # 稳定性闸门：近 1s 内宽度波动过大（人物交互/半遮挡期）→ 对齐不可信，弃权
        if (max(self._scr_w_hist) - min(self._scr_w_hist)) > 0.30 * self._scr_w_ema:
            return False, frame
        self._scr_w_ema = 0.65 * self._scr_w_ema + 0.35 * cur_w
        wmed = self._scr_w_ema
        cxm = float(np.median([c[0] for c in self._scr_c_hist]))
        cym = float(np.median([c[1] for c in self._scr_c_hist]))
        if wmed <= 1:
            return False, frame
        s = rw / wmed                     # 当前帧 → 参考坐标系 的缩放
        if not (0.4 < s < 2.5):
            return False, frame
        M = np.array([[s, 0.0, rcx - s * cxm], [0.0, s, rcy - s * cym]], np.float32)
        h, w = frame.shape[:2]
        return True, cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

    def _door_id(self, cell: dict) -> str:
        return "R%dC%d" % (int(cell["row"]) + 1, int(cell["col"]) + 1)

    def _estimate_homography(self, frame: np.ndarray) -> bool:
        """对当前帧估计「→参考坐标系」的单应。成功则更新 self._Hinv。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kf, df = self._orb.detectAndCompute(gray, self._mask)
        if df is None or len(kf) < 12:
            return False
        matches = self._bf.knnMatch(self._dref, df, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return False
        src = np.float32([self._kref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl is None or int(inl.sum()) < 10:
            return False
        try:
            self._Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return False
        return True

    def _stabilize(self, frame: np.ndarray, force: bool = False) -> Tuple[bool, np.ndarray]:
        """稳定化到参考坐标系。

        性能：knn 全对全匹配是重负载（实测 2500 特征 ≈500ms/帧），故**每
        `reestimate_every` 帧才重估一次单应**，其间复用上一结果——镜头运动平滑，
        残差由逐格 ±16px 扩窗搜索吸收；实测复用不影响判定（见标定记录）。
        """
        if force or self._Hinv is None or (self._n_updates % int(self.p.get("reestimate_every", 4)) == 0):
            self._estimate_homography(frame)
        if self._Hinv is None:
            return False, frame
        h, w = frame.shape[:2]
        return True, cv2.warpPerspective(frame, self._Hinv, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)

    def _person_overlaps(self, st: _CellState, boxes, expand: float) -> bool:
        bx0, by0, bx1, by1 = self._cell_bbox_px[st.idx]
        ew, eh = (bx1 - bx0) * expand, (by1 - by0) * expand
        x0c, y0c, x1c, y1c = bx0 - ew, by0 - eh, bx1 + ew, by1 + eh
        for b in boxes:
            px0, py0, px1, py1 = b[:4]
            if not (px1 < x0c or px0 > x1c or py1 < y0c or py0 > y1c):
                return True
        return False

    def _update_retrieval(self, st: _CellState, idx: int, ts: float, gc: np.ndarray,
                          person_boxes, occluded: bool, track_of_box,
                          events: List[RuleEvent]) -> None:
        """
        标记存续期间的「取走未关柜门外卖」判定（演示口径）。

        物理过程：格口开着（标记中）→ 有人**接触**（检测框持续覆盖格口）→ 人离开 →
        格口内容与「接触前的干净快照」相比发生变化 ⇒ 取走。
        接触期间格口被人体遮挡、无法直接观测，因此比较发生在**人走开之后**；
        接触前的干净快照取「最近一帧未被遮挡的格口画面」。
        """
        x0, y0, x1, y1 = self._tpl_box[idx]
        patch = cv2.resize(gc[y0:y1, x0:x1], (0, 0), fx=0.5, fy=0.5)

        if st.retrieved:
            return                          # 本开门周期已报过取走，避免重复告警
        if occluded:
            # 接触中：冻结快照（保留接触前最后一帧干净画面），记录接触起点
            if st.touch_since is None:
                st.touch_since = ts
                st.touch_track = None
            return

        # 未遮挡
        if st.touch_since is not None:
            touch_dur = ts - st.touch_since
            # 接触刚结束：与接触前快照比对内容变化（短暂擦过不触发）
            if (touch_dur >= float(self.p.get("retrieval_touch_s", 1.2))
                    and st.pre_touch_tpl is not None
                    and patch.shape == st.pre_touch_tpl.shape):
                ncc = float(cv2.matchTemplate(patch, st.pre_touch_tpl,
                                              cv2.TM_CCOEFF_NORMED)[0, 0])
                self.log("[演示档] %s：接触结束（持续 %.1fs），内容一致性 NCC=%.3f"
                         % (st.door_id, touch_dur, ncc))
                if ncc < float(self.p.get("retrieval_change_th", 0.62)):
                    events.append(_make_event(
                        "FLAGGED_DOOR_RETRIEVAL",
                        "目标在标记柜格 %s 前接触 %.1fs 后离开，格口内容发生变化（演示口径：人员接触 + 内容变化）"
                        % (st.door_id, touch_dur),
                        0.85, ts, 0, compartment=idx,
                        door_id=st.door_id, demo=True, content_ncc=round(ncc, 3)))
                    st.retrieved = True
                    self.log("[演示档] 柜格 %s：标记存续期间检测到取走动作（内容变化 NCC=%.2f）"
                             % (st.door_id, ncc))
            st.touch_since = None
            st.touch_track = None
        # 未遮挡且未在接触中：持续刷新「接触前干净快照」（遮挡到来时自然冻结为最后一帧干净画面）
        st.pre_touch_tpl = patch

    def frame_size(self):
        """校准参考帧的原生尺寸 (w, h)；未校准时为 None。"""
        return self._frame_size

    # ------------------------------------------------------------ 状态
    def snapshot(self) -> dict:
        return {
            "calibrated": self._calibrated,
            "calib_error": self._calib_error,
            "cells_total": len(self._states),
            "cells_usable": sum(1 for v in self._baseline_ok.values() if v),
            "doors": [{"idx": st.idx, "door_id": st.door_id, "state": st.state,
                       "ncc": round(st.last_ncc, 3), "ncc_half": round(st.last_ncc2, 3),
                       "flagged": st.flagged}
                      for st in sorted(self._states.values(), key=lambda s: s.idx)],
            "flagged_doors": [st.door_id for st in self._states.values() if st.flagged],
        }
