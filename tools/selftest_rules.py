# -*- coding: utf-8 -*-
"""
tools/selftest_rules.py — 规则引擎单元测试

对 rules.RuleEngine 注入**合成场景观测**（人物框 / 柜门角 / 柜内餐盒数 / 手持餐盒数），逐项验证：

    T1 进入取餐区事件
    T2 长时间驻足不报警（需求方口径）
    T3 徘徊观察 + 入区事件冷却
    T4 事件冷却（同一目标不重复记录）
    T5 正常取餐不产生任何异常
    T6 柜门未关 → 触发事件并**标记柜号**
    T7 柜门关闭 → 解除标记
    T8 一人手持 ≥2 份 → 一人取多份
    T9 标记存续期间取走柜内外卖 → 报警
    T10 阴性对照：未标记柜格被取物**不得**报警（防误报）
    T11 检出中断 5.5s 后的轨迹重关联（身份不丢、停留时长不清零）
    T12 手持餐盒关联的空间扩展判据（横举手臂）+ 层板餐盒守卫 + 双色合并

说明：本测试使用人工构造的观测，用于验证**判定逻辑**的正确性，
不产生任何实验数据，也不用于论文指标计算。
所有测试点位均由 `config` 的 ROI 配置**运行时推导**，不硬编码像素坐标——
否则 ROI 随场景重标定后测试会静默失真。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402
from detector import Detection  # noqa: E402
from rules import (PackageCarry, RuleConfig, RuleEngine,  # noqa: E402
                   SceneObservation)

W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
FPS = 10.0
DT = 1.0 / FPS


def _poly(pts) -> np.ndarray:
    return np.array([[int(x * W), int(y * H)] for x, y in pts], np.int32)


P_PICKUP = _poly(C.ROI_PICKUP)
COMPARTMENTS = [_poly(q) for q in C.CABINET_COMPARTMENTS]


def _in(poly, pt) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def _points_in(region: str, n: int, min_gap: int = 95) -> List[Tuple[int, int]]:
    """从 ROI 配置自动推导测试点位。region: 'pickup' | 'outside'"""
    found: List[Tuple[int, int]] = []
    for y in range(4, H - 4, 3):
        for x in range(4, W - 4, 3):
            p = (x, y)
            ok = _in(P_PICKUP, p) if region == "pickup" else not _in(P_PICKUP, p)
            if not ok:
                continue
            if all(abs(x - q[0]) >= min_gap or abs(y - q[1]) >= min_gap for q in found):
                found.append(p)
            if len(found) >= n:
                return found
    raise RuntimeError("ROI 配置 %s 内找不到 %d 个满足间距要求的测试点" % (region, n))


PICKUP_PTS = _points_in("pickup", 2)
OUTSIDE_PTS = _points_in("outside", 4)
IN_PICKUP, IN_PICKUP_2 = PICKUP_PTS
OUTSIDE = OUTSIDE_PTS[0]
# 各柜格中心点（用于构造「人在某柜格附近」的观测）
COMPARTMENT_CENTERS = [tuple(p.astype(float).mean(axis=0)) for p in COMPARTMENTS]


def det(cx: float, cy: float, conf: float = 0.85, bw: int = 70, bh: int = 200) -> Detection:
    """以「脚点 + 框尺寸」构造检测框。"""
    return Detection(cx - bw / 2, cy - bh, cx + bw / 2, cy, "person", conf)


class Checker:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, note: str = "") -> None:
        self.rows.append((name, bool(cond), note))
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + note) if note else ""))

    def section(self, title: str) -> None:
        print("\n" + title)

    @property
    def failed(self) -> int:
        return sum(1 for _n, ok, _ in self.rows if not ok)


def run(engine: RuleEngine, frames, collect: List[str], start_t: float = 0.0) -> float:
    """按 10 fps 逐帧投喂观测，返回结束时刻。"""
    t = start_t
    for obs in frames:
        for ev in engine.update(obs, t, 0, W, H):
            collect.append(ev.event_type)
        t += DT
    return t


def person_frames(pos: Tuple[float, float], n: int, **kw) -> List[SceneObservation]:
    return [SceneObservation(dets=[det(pos[0], pos[1], **kw)]) for _ in range(n)]


def move_frames(a: Tuple[float, float], b: Tuple[float, float], step_px: int = 24,
                **kw) -> List[SceneObservation]:
    """
    从 a 渐变移动到 b，逐帧位置按 step_px 插值。

    必须渐变：位置突变会使相邻帧 IoU 归零，跟踪器会判为新目标，
    导致 dwell/entries 无法在同一轨迹上累计（曾因此使 T3/T5 误报失败）。
    步长 24px 时相邻帧 IoU≈0.49，高于匹配阈值 0.30。
    """
    dist = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(dist // step_px))
    out = []
    for s in range(1, n + 1):
        cx = a[0] + (b[0] - a[0]) * s / n
        cy = a[1] + (b[1] - a[1]) * s / n
        out.append(SceneObservation(dets=[det(cx, cy, **kw)]))
    return out


def door_frames(k: int, angle: float, n: int, pkg: Optional[int] = None,
                person: Optional[Tuple[float, float]] = None,
                carries: Optional[List[PackageCarry]] = None) -> List[SceneObservation]:
    out = []
    for _ in range(n):
        obs = SceneObservation(dets=[], door_angles={k: angle})
        if pkg is not None:
            obs.packages_in_compartment = {k: pkg}
        if person is not None:
            obs.dets = [det(person[0], person[1])]
        if carries:
            obs.carries = list(carries)
        out.append(obs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="规则引擎单元测试")
    ap.add_argument("--save-log", action="store_true", default=True)
    a = ap.parse_args()

    ck = Checker()
    print("=" * 78)
    print("天眼哨兵 · 规则引擎单元测试")
    print("画面 %dx%d  仿真帧率 %.1f fps" % (W, H, FPS))
    print("由 config 自动推导的测试点位：取餐区内 %s；两区之外 %s" % (PICKUP_PTS, OUTSIDE))
    print("事件类别：%s" % "、".join(C.EVENT_NAME.values()))
    print("=" * 78)

    # ---------------- T1/T2 入区 + 长驻足不报警（2026-10-07 新口径） ----------------
    ck.section("T1/T2 进入取餐区 + 长时间驻足【不再报警】（需求方口径：长驻足属正常）")
    eng = RuleEngine(RuleConfig()); evs: List[str] = []
    run(eng, person_frames(IN_PICKUP, int(12 * FPS)), evs)
    ck.check("T1 ROI_INTRUSION 触发一次", evs.count("ROI_INTRUSION") == 1, "实际 %d 次" % evs.count("ROI_INTRUSION"))
    ck.check("T2 驻留 12s 不产生任何异常事件", not (set(evs) & C.ABNORMAL_EVENTS),
             "异常事件 %s" % sorted(set(evs) & C.ABNORMAL_EVENTS))
    ck.check("T2 事件集合仅含进入取餐区", set(evs) <= {"ROI_INTRUSION"}, "事件集合 %s" % set(evs))
    trs = eng.snapshot()["tracks"]
    ck.check("T2 停留时长正常累计（dwell ≥ 11s）",
             trs and trs[0]["dwell_s"] >= 11.0, "dwell=%s" % (trs[0]["dwell_s"] if trs else "无轨迹"))

    # ---------------- T3 徘徊观察 + 入区冷却 ----------------
    ck.section("T3 徘徊观察：3 次进出取餐区后在外围滞留（阈值 15s / 3 次）")
    eng = RuleEngine(RuleConfig()); evs = []
    t3 = 0.0
    for _ in range(3):
        t3 = run(eng, move_frames(OUTSIDE, IN_PICKUP) + person_frames(IN_PICKUP, 6)
                 + move_frames(IN_PICKUP, OUTSIDE), evs, start_t=t3)
    run(eng, person_frames(OUTSIDE, int(17 * FPS)), evs, start_t=t3)
    ck.check("T3 LOITERING 触发一次", evs.count("LOITERING") == 1, "实际 %d 次" % evs.count("LOITERING"))
    ck.check("T3 入区事件冷却生效", evs.count("ROI_INTRUSION") == 1, "实际 %d 次" % evs.count("ROI_INTRUSION"))

    # ---------------- T4 事件冷却（改用入区事件观测） ----------------
    ck.section("T4 事件冷却：冷却窗内再次入区被抑制（不再借长时间停留观测）")
    eng = RuleEngine(RuleConfig()); evs = []
    t4 = run(eng, person_frames(IN_PICKUP, int(4 * FPS)), evs)          # 一次 4s 驻留（新口径：不报警）
    t4 = run(eng, person_frames(IN_PICKUP, int(8 * FPS)), evs, start_t=t4)    # 继续停留，仍不报警
    t4 = run(eng, move_frames(IN_PICKUP, OUTSIDE) + person_frames(OUTSIDE, 6)
             + move_frames(OUTSIDE, IN_PICKUP), evs, start_t=t4)     # 冷却窗内再次进入
    run(eng, person_frames(IN_PICKUP, 4), evs, start_t=t4)
    ck.check("T4 长驻足全程零异常事件", not (set(evs) & C.ABNORMAL_EVENTS),
             "异常事件 %s" % sorted(set(evs) & C.ABNORMAL_EVENTS))
    ck.check("T4 冷却抑制计数 > 0（再次入区被冷却拦截）",
             eng.snapshot()["suppressed_by_cooldown"] > 0,
             "抑制 %d 次" % eng.snapshot()["suppressed_by_cooldown"])

    # ---------------- T5 正常取餐 ----------------
    ck.section("T5 正常取餐：进入取餐区约 3s 后离开（< 6.0s）")
    eng = RuleEngine(RuleConfig()); evs = []
    t5 = run(eng, move_frames(OUTSIDE, IN_PICKUP), evs)
    t5 = run(eng, person_frames(IN_PICKUP, int(3.0 * FPS)), evs, start_t=t5)
    run(eng, move_frames(IN_PICKUP, OUTSIDE) + person_frames(OUTSIDE, int(5 * FPS)), evs, start_t=t5)
    ck.check("T5 未产生任何异常事件", not (set(evs) & C.ABNORMAL_EVENTS), "事件集合 %s" % set(evs))
    ck.check("T5 正常取餐计数 +1", eng.snapshot()["normal_retrievals"] == 1,
             "实际 %d 次" % eng.snapshot()["normal_retrievals"])

    # ---------------- T6/T7 柜门未关 → 标记 → 关门解除 ----------------
    ck.section("T6/T7 柜门未关标记状态机（阈值 10.0s）")
    K_DOOR = 2
    eng = RuleEngine(RuleConfig()); evs = []
    run(eng, door_frames(K_DOOR, 86.0, int(4 * FPS)), evs)                 # 开门 4s：未超时
    ck.check("T6 未超时不标记", not eng.snapshot()["flagged_doors"], "标记列表为空")
    run(eng, door_frames(K_DOOR, 86.0, int(8 * FPS)), evs)                 # 再 8s：累计 12s > 10s
    ck.check("T6 DOOR_LEFT_OPEN 触发一次", evs.count("DOOR_LEFT_OPEN") == 1,
             "实际 %d 次" % evs.count("DOOR_LEFT_OPEN"))
    snap = eng.snapshot()
    ck.check("T6 已标记该柜号供终端显示",
             [f["compartment"] for f in snap["flagged_doors"]] == [K_DOOR],
             "标记柜号 %s" % snap["flagged_door_labels"])
    run(eng, door_frames(K_DOOR, 0.0, int(2 * FPS)), evs)                  # 关门
    snap = eng.snapshot()
    ck.check("T7 关门后标记解除", not snap["flagged_doors"], "标记列表已清空")
    ck.check("T7 记录一次柜门已关闭事件", evs.count("DOOR_CLOSED") == 1,
             "实际 %d 次" % evs.count("DOOR_CLOSED"))

    # ---------------- T8 一人取多份 ----------------
    ck.section("T8 一人取多份：手持 2 份（阈值 2 份）")
    eng = RuleEngine(RuleConfig()); evs = []
    cx, cy = IN_PICKUP
    carry2 = PackageCarry(bbox=(cx - 30, cy - 170, cx + 30, cy - 90), count=2)
    run(eng, [SceneObservation(dets=[det(cx, cy)], carries=[carry2]) for _ in range(int(3 * FPS))], evs)
    ck.check("T8 MULTI_RETRIEVAL 触发一次", evs.count("MULTI_RETRIEVAL") == 1,
             "实际 %d 次" % evs.count("MULTI_RETRIEVAL"))
    eng2 = RuleEngine(RuleConfig()); evs2 = []
    carry1 = PackageCarry(bbox=(cx - 30, cy - 170, cx + 30, cy - 90), count=1)
    run(eng2, [SceneObservation(dets=[det(cx, cy)], carries=[carry1]) for _ in range(int(3 * FPS))], evs2)
    ck.check("T8 阴性对照：仅持 1 份不告警", "MULTI_RETRIEVAL" not in evs2, "事件集合 %s" % set(evs2))

    # ---------------- T9/T10 标记柜格被取物 → 报警；未标记柜格不报警 ----------------
    ck.section("T9/T10 取走「未关柜门」柜格内外卖（阳性 + 阴性对照）")
    K_DOOR = 2
    near_pt = COMPARTMENT_CENTERS[K_DOOR]
    # 先让门标记生效
    eng = RuleEngine(RuleConfig()); evs = []
    run(eng, door_frames(K_DOOR, 86.0, int(12 * FPS), pkg=1, person=near_pt), evs)
    ck.check("T9 前置：柜门已标记", bool(eng.snapshot()["flagged_doors"]), "标记已生成")
    run(eng, door_frames(K_DOOR, 86.0, int(3 * FPS), pkg=0, person=near_pt), evs)   # 柜内餐盒 1→0
    ck.check("T9 FLAGGED_DOOR_RETRIEVAL 触发", evs.count("FLAGGED_DOOR_RETRIEVAL") == 1,
             "实际 %d 次" % evs.count("FLAGGED_DOOR_RETRIEVAL"))

    eng = RuleEngine(RuleConfig()); evs = []
    run(eng, door_frames(3, 0.0, int(12 * FPS), pkg=1, person=COMPARTMENT_CENTERS[3]), evs)  # 门始终关闭
    run(eng, door_frames(3, 0.0, int(3 * FPS), pkg=0, person=COMPARTMENT_CENTERS[3]), evs)
    ck.check("T10 阴性对照：未标记柜格被取物不报警",
             evs.count("FLAGGED_DOOR_RETRIEVAL") == 0 and evs.count("DOOR_LEFT_OPEN") == 0,
             "事件集合 %s" % set(evs))

    # ---------------- T11 检出中断后的轨迹重关联 ----------------
    ck.section("T11 轨迹重关联：站姿静止目标被漏检 5.5s 后重现（实测间断 4.07~5.60s）")
    eng = RuleEngine(RuleConfig()); evs = []
    t11 = run(eng, person_frames(IN_PICKUP, int(2 * FPS)), evs)            # 先有 2s 正常检出
    run(eng, [SceneObservation(dets=[]) for _ in range(int(5.5 * FPS))], evs, start_t=t11)  # 5.5s 零检出
    run(eng, person_frames(IN_PICKUP, int(4 * FPS)), evs, start_t=t11 + 5.5)
    snap = eng.snapshot()
    ck.check("T11 漏检期间不新建轨迹", snap["tracks_total"] == 1, "轨迹总数 %d" % snap["tracks_total"])
    ck.check("T11 身份保持：只记一次进入取餐区", evs.count("ROI_INTRUSION") == 1,
             "实际 %d 次" % evs.count("ROI_INTRUSION"))
    trs11 = eng.snapshot()["tracks"]
    ck.check("T11 漏检期间轨迹状态继续累计（dwell ≥ 11s：2 + 漏检 5.5 + 复现 4）",
             trs11 and trs11[0]["dwell_s"] >= 11.0,
             "dwell=%s" % (trs11[0]["dwell_s"] if trs11 else "无轨迹"))

    # ---------------- T12 手持餐盒关联的空间扩展判据 ----------------
    ck.section("T12 手持餐盒关联：横举手臂伸出检测框（外扩 35%/15%）+ 层板守卫 + 双色合并")
    from detector import Blob, count_carried_packages  # noqa: E402
    box = (300.0, 100.0, 430.0, 460.0)                  # 实测帧几何：宽 130、高 360

    def blob_at(cx: float, cy: float, half: float = 15.0) -> Blob:
        return Blob(cx - half, cy - half, cx + half, cy + half, (2 * half) ** 2)

    # 阳性：手持餐盒在身体侧方，中心 (456,260) 落在框右外 26px（< 外扩 45.5px）且不在任何柜格内
    py = (box[1] + box[3]) / 2.0
    n = count_carried_packages([blob_at(400, 260), blob_at(456, py)], [box], COMPARTMENTS)[0][1]
    ck.check("T12 横举手臂的餐盒被计入手持（1+1=2）", n == 2, "实测计数 %d" % n)

    # 阴性守卫：中心落在外扩区但不在原框内，且位于柜格多边形内 → 属层板餐盒，不得计入手持
    pt_comp = COMPARTMENT_CENTERS[5]
    bx2 = pt_comp[0] - 5.0
    guard_box = (bx2 - 130.0, pt_comp[1] - 100.0, bx2, pt_comp[1] + 100.0)
    ck.check("T12 前置：测试点位于柜格 5 内且在原框外",
             _in(COMPARTMENTS[5], pt_comp) and not (guard_box[0] <= pt_comp[0] <= guard_box[2]),
             "柜格中心 (%.0f,%.0f)" % pt_comp)
    n = count_carried_packages([blob_at((guard_box[0] + guard_box[2]) / 2.0, pt_comp[1]),
                                blob_at(pt_comp[0], pt_comp[1])], [guard_box], COMPARTMENTS)[0][1]
    ck.check("T12 阴性守卫：层板餐盒不计入手持", n == 1, "实测计数 %d" % n)

    # 双色同一份餐盒（紧邻双 blob）→ 合并为 1 份
    blob_l = blob_at(340, 300)
    blob_r = Blob(340 + 15 + 5, 285, 340 + 15 + 5 + 20, 315, 400)
    n = count_carried_packages([blob_l, blob_r], [box], COMPARTMENTS)[0][1]
    ck.check("T12 双色餐盒（间隙 5px）合并为 1 份", n == 1, "实测计数 %d" % n)

    # ---------------- T13 脚点突变守卫（部分框不得翻转区域判定） ----------------
    ck.section("T13 部分框守卫：单帧「上半身框」使脚点突跳（实测 long_dwell_020 跳 143px）")
    eng = RuleEngine(RuleConfig()); evs = []
    cx, cy = IN_PICKUP
    t13 = run(eng, person_frames(IN_PICKUP, int(2 * FPS), bh=360), evs)          # 取餐区内 2s（全身框）
    # 部分框：顶边不变、底边抬到画面中部（IoU≈0.48 仍高于匹配阈值 0.30，脚点突跳 188px）
    part = Detection(cx - 35, cy - 360, cx + 35, cy - 188, "person", 0.39)
    t13 = run(eng, [SceneObservation(dets=[part])], evs, start_t=t13)
    run(eng, person_frames(IN_PICKUP, int(7 * FPS), bh=360), evs, start_t=t13)   # 继续在区内停留
    snap13 = eng.snapshot()
    ck.check("T13 部分框不翻转区域状态（不重记进入）", evs.count("ROI_INTRUSION") == 1,
             "实际 %d 次" % evs.count("ROI_INTRUSION"))
    trs13 = eng.snapshot()["tracks"]
    ck.check("T13 部分框不清零停留时长（dwell ≥ 8.5s，非被重置后的 7s）",
             trs13 and trs13[0]["dwell_s"] >= 8.5,
             "dwell=%s" % (trs13[0]["dwell_s"] if trs13 else "无轨迹"))
    ck.check("T13 未产生额外轨迹", snap13["tracks_total"] == 1,
             "轨迹总数 %d" % snap13["tracks_total"])

    # ---------------- T14 部分框（框高骤降）不得翻转区域状态 ----------------
    ck.section("T14 部分框判据：本帧框高骤降至可信高度的 53%（实测 long_dwell_007 170 vs 360 px）")
    eng = RuleEngine(RuleConfig()); evs = []
    cx, cy = IN_PICKUP
    t14 = run(eng, person_frames(IN_PICKUP, int(2 * FPS), bh=360), evs)      # 完整体 2s
    # 上半身部分框：顶边不变、底边抬到画面中部 → 框高 190（360 的 53%），脚点被抬到区外
    part14 = Detection(cx - 35, cy - 360, cx + 35, cy - 170, "person", 0.55)
    t14 = run(eng, [SceneObservation(dets=[part14]) for _ in range(3)], evs, start_t=t14)
    run(eng, person_frames(IN_PICKUP, int(7 * FPS), bh=360), evs, start_t=t14)
    snap14 = eng.snapshot()
    trs14 = snap14["tracks"]
    ck.check("T14 部分框不清零停留时长（dwell ≥ 8.5s）",
             trs14 and trs14[0]["dwell_s"] >= 8.5,
             "dwell=%s" % (trs14[0]["dwell_s"] if trs14 else "无轨迹"))
    ck.check("T14 部分框不重记进入取餐区", evs.count("ROI_INTRUSION") == 1,
             "实际 %d 次" % evs.count("ROI_INTRUSION"))
    ck.check("T14 部分框不拆分轨迹（可信脚点未被污染）",
             snap14["tracks_total"] == 1, "轨迹总数 %d" % snap14["tracks_total"])

    # ---------------- 汇总 ----------------
    total = len(ck.rows); passed = total - ck.failed
    print("\n" + "=" * 78)
    print("自检结果：%d / %d 项通过" % (passed, total))
    print("=" * 78)

    if a.save_log:
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        p = C.LOG_DIR / ("selftest_rules_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        lines = ["天眼哨兵 · 规则引擎单元测试记录",
                 "时间：%s" % datetime.now().isoformat(timespec="seconds"),
                 "环境：%s" % sys.version.replace("\n", " "),
                 "画面：%dx%d  仿真帧率 %.1f fps" % (W, H, FPS),
                 "事件类别：%s" % "、".join(C.EVENT_NAME.values()),
                 "结果：%d / %d 项通过" % (passed, total), ""]
        lines += ["[%s] %s%s" % ("PASS" if ok else "FAIL", n, ("  — " + t) if t else "")
                  for n, ok, t in ck.rows]
        p.write_text("\n".join(lines), encoding="utf-8")
        print("测试记录已保存：%s" % p)

    return 1 if ck.failed else 0


if __name__ == "__main__":
    sys.exit(main())
