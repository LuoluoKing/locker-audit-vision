# -*- coding: utf-8 -*-
"""
tools/gen_behavior_clip3d.py — 按行为脚本批量生成三维场景数据集（多进程）

场景设定与物理逻辑（务必保持自洽）
----------------------------------
取餐柜共 6 个柜格（3 行 × 2 列，行优先自上而下）。**柜格 2（中排左）是被操作柜格。**

六类行为的动作脚本均按真实取餐流程编排，并通过 `validate_spec()` 做**时序自检**，
禁止出现「手里已拿着外卖还往柜里伸」「从已空的柜格取餐」这类不合物理的组合。

    正常取餐        : 开门 → 伸手取走餐盒 → 收手 → **顺手关门** → 离开
    徘徊观察        : 反复靠近又退回 ≥3 次，随后在外围滞留；全程不接触柜门
    一人取多份      : 同一柜格内放 2 份餐盒，依次取走 → 手中同时/累计持有 ≥2 份
    柜门未关        : 取走餐盒后**不关门**即离开；此后无人再取该柜格
    取走未关柜门外卖: 甲**只开门不取物**即离开（柜内餐盒仍在）→ 柜门未关被标记
                      → 乙随后取走被标记柜格内的餐盒 → 触发报警

命令行
------
    python tools/gen_behavior_clip3d.py --behavior all --workers 12
    python tools/gen_behavior_clip3d.py --behavior normal_pickup --count 3
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from gen_selftest_clip3d import (CAM_EYE, CAM_FOV, CAM_TARGET, KeyTrack,  # noqa: E402
                                 OUT_H, OUT_W, hand_in_cabinet)

TARGET_DOOR = 2
# 柜门开启角：真实取餐柜门通常开到 90° 以上。经 tools/calib_locker_door_threshold.py
# 实测，86° 时右列两格的门板会挡住柜内、暗比降至阈值以下（仅 4/6 可判）；
# 115° 时六格门开暗比 0.245~0.797、门关最大 0.053，现行阈值 0.15 仍 6/6 命中且零误报。
DOOR_OPEN_ANGLE = 115.0
PKG = K.PACKAGE_COLORS

# 柜格 2 的中排左门；其余柜格门始终关闭
DOOR_CLOSED_ALL = [0.0] * 6


# ---------------------------------------------------------------- 行为主体
class Actor:
    def __init__(self, px, pz, facing, lean, sh_r, el_r, sh_l, el_l, walk, pkg):
        self.px, self.pz, self.facing, self.lean = px, pz, facing, lean
        self.sh_r, self.el_r, self.sh_l, self.el_l = sh_r, el_r, sh_l, el_l
        self.walk, self.pkg = walk, pkg

    def pose(self, t: float, walk_freq: float = 1.55) -> K.Pose:
        w = self.walk.at(t)
        ph = t * walk_freq * 2 * np.pi
        swing = 26.0 * w * math.sin(ph)
        n_hold = int(round(self.pkg.at(t)))          # 手持餐盒**数量**（0~2）
        # 分配：1 份用右手；2 份左右手各一（比双手叠在同一腕更符合直觉，
        # 且「双手各持一盒」的轮廓对检测器更友好——实测站姿置信度 0.0 → 0.49）
        holds = ("r",) if n_hold == 1 else (("r", "l") if n_hold >= 2 else ())
        return K.Pose(
            x=self.px.at(t), z=self.pz.at(t), facing=self.facing.at(t), lean=self.lean.at(t),
            shoulder_l=self.sh_l.at(t) - 12.0 * w * math.sin(ph),
            shoulder_r=self.sh_r.at(t) + 12.0 * w * math.sin(ph),
            elbow_l=self.el_l.at(t), elbow_r=self.el_r.at(t),
            hip_l=swing, hip_r=-swing,
            knee_l=-18.0 * w * max(0.0, math.sin(ph)),
            knee_r=-18.0 * w * max(0.0, -math.sin(ph)),
            packages_in_hand=holds)


@dataclass
class BehaviorSpec:
    name: str
    label_cn: str
    is_abnormal: bool
    seconds: float
    actors: Callable[[np.random.Generator], List[Actor]]
    door: Callable[[np.random.Generator], KeyTrack]
    shelf: Callable[[np.random.Generator, float], List[Tuple[float, float, float, int]]]
    expect: Dict[str, str]
    note: str


PICKUP_X, PICKUP_Z = 0.16, -0.72


def _person(base_x: float = PICKUP_X, base_z: float = PICKUP_Z, jx: float = 0.0, jz: float = 0.0):
    """构造单个行人的各关节轨道（时间参数化）。"""
    return base_x + jx, base_z + jz


def _walk_actor(t_arrive: float, t_leave: float, x_stand: float, z_stand: float,
                yaw_in: float, reach_t0: float, reach_t1: float, pkg: KeyTrack,
                t_start: float = 0.0) -> Actor:
    """
    通用「走近 → 伸手 → 取物 → 收手 → 离开」主体。

    pkg : 手持餐盒**数量**的关键帧轨道。餐盒只能在伸手动作**期间**出现
          （由 validate_spec 强制校验），不得出现「拿着外卖走过来再往柜里伸」。
    t_start : 开始走进画面的时刻。0 = 从 2.1 m 外直接走入；
              > 0 = 此前一直等在**画外**（左前方，镜头视野之外），到点后以约
              1 m/s 的正常步速走入。绝不能用「把 2.1 m 的路程摊到十几秒」来
              等待入场——龟速逼近会让人物长时间遮挡柜门、并在取餐区异常滞留
              （曾因此压掉 DOOR_LEFT_OPEN 的 10s 累计窗口并派生其他误报）。

    姿态要点（由 YOLOv8n 实测标定，见 tools/exp_pose_variants.py）：
      · 双臂笔直下垂的「棍状」站姿人形置信度仅 0.0~0.24，必然漏检；
      · 双手前置 / 持盒屈臂的站姿置信度 0.49~0.55，可靠检出；
      · 行走姿态（迈步摆臂）置信度约 0.69。
    因此：站立等待一律双手前置，取到餐盒后一律持盒屈臂。
    """
    x0, z0 = x_stand - 2.1, z_stand - 0.85
    x1, z1 = x_stand + 2.1, z_stand - 1.0        # 退场落点：+2.1 处于画面内但不再跨越
                                                 # 取餐区边界（拉长退场会来回穿越 ROI，
                                                 # 使 entries 虚增并派生 LOITERING 误报，
                                                 # 实测第四版因此退化 12 个百分点）
    x_wait, z_wait = x_stand - 3.6, z_stand - 1.55        # 画外等待位
    t_retract = reach_t1 + 0.7
    # 手持餐盒的起始时刻（数量首次 ≥1），此后改为持盒屈臂
    t_carry = next((tt for tt, v in pkg.keys if v >= 0.5), None)
    t_carry2 = next((tt for tt, v in pkg.keys if v >= 1.5), None)
    if t_start > 0.0:
        px_in = [(0.0, x_wait), (t_start, x_wait), (t_arrive, x_stand)]
        pz_in = [(0.0, z_wait), (t_start, z_wait), (t_arrive, z_stand)]
        walk_in = [(0.0, 0.0), (t_start, 0.0), (t_start + 0.1, 1.0),
                   (t_arrive - 0.2, 1.0), (t_arrive + 0.1, 0.0)]
        face_in = [(0.0, 0.0), (t_start, yaw_in), (t_arrive, 0.0)]
    else:
        px_in = [(0.0, x0), (t_arrive, x_stand)]
        pz_in = [(0.0, z0), (t_arrive, z_stand)]
        walk_in = [(0.0, 1.0), (t_arrive - 0.2, 1.0), (t_arrive + 0.1, 0.0)]
        face_in = [(0.0, yaw_in), (t_arrive, 0.0)]

    # 右臂：等待=双手前置 → 伸手前回归体侧 → 深伸 → 收回到「持盒胸前屈臂」（若取到物）。
    # 持盒高度两版实测对比：胸前 (sh-30/el-55) 在整段数据集上链路准确率 90.7%，
    # 腰侧 (sh-18/el-30) 单帧更优（conf 0.82）但整段反而 78.6%（站姿检出波动加剧，
    # multi 段崩到 4.8%）——以**整段链路指标**为准，取胸前方案。
    sh_r_keys = [(0.0, -24.0), (reach_t0 - 0.4, -14.0), (reach_t0 + 0.9, -78.0),
                 (reach_t1, -80.0)]
    el_r_keys = [(0.0, -44.0), (reach_t0 - 0.4, -6.0), (reach_t0 + 0.9, 44.0),
                 (reach_t1, 46.0)]
    if t_carry is not None:
        sh_r_keys += [(t_retract, -30.0), (3600.0, -30.0)]
        el_r_keys += [(t_retract, -55.0), (3600.0, -55.0)]
    else:
        sh_r_keys += [(t_retract, -24.0), (3600.0, -26.0)]
        el_r_keys += [(t_retract, -44.0), (3600.0, -46.0)]
    # 左臂：等待/持单盒=前置；第二次取件（pkg≥2）前改为深伸（左手取第二份）
    sh_l_keys = [(0.0, -22.0), (reach_t0 - 0.4, 6.0)]
    el_l_keys = [(0.0, -42.0), (reach_t0 - 0.4, -4.0)]
    if t_carry2 is not None:
        sh_l_keys += [(t_carry2 - 0.9, -14.0), (t_carry2 - 0.25, -78.0),
                      (t_carry2 + 0.1, -26.0), (3600.0, -24.0)]
        el_l_keys += [(t_carry2 - 0.9, -6.0), (t_carry2 - 0.25, 46.0),
                      (t_carry2 + 0.1, -44.0), (3600.0, -46.0)]
    else:
        sh_l_keys += [(t_retract, -22.0), (3600.0, 16.0)]
        el_l_keys += [(t_retract, -42.0), (3600.0, -8.0)]
    return Actor(
        px=KeyTrack(px_in + [(t_leave, x_stand), (t_leave + 2.6, x1), (3600.0, x1 + 0.4)]),
        pz=KeyTrack(pz_in + [(t_leave, z_stand), (t_leave + 2.6, z1), (3600.0, z1 - 0.3)]),
        facing=KeyTrack(face_in + [(t_leave, 0.0), (t_leave + 2.6, 26.0), (3600.0, 30.0)]),
        lean=KeyTrack([(0.0, 2.0), (t_arrive, 4.0), (reach_t0, 13.0),
                       (t_retract, 5.0), (t_retract + 0.6, 2.0), (3600.0, 2.0)]),
        sh_r=KeyTrack(sh_r_keys),
        el_r=KeyTrack(el_r_keys),
        sh_l=KeyTrack(sh_l_keys),
        el_l=KeyTrack(el_l_keys),
        walk=KeyTrack(walk_in + [(t_leave, 0.0), (t_leave + 0.3, 1.0), (3600.0, 0.0)]),
        pkg=pkg,
    )


# ---------------------------------------------------------------- 柜内餐盒
def _pkg_at(spec_list: Sequence[Tuple[float, float, float, int]], visible_from: float,
            t: float) -> List[Tuple[float, float, float, int]]:
    return [p for p in spec_list if t >= visible_from] if visible_from > 0 else list(spec_list)


def _shelf_default(x_off: float = 0.0, extra_in_comp2: int = 0):
    """
    返回柜内餐盒规格 [(x, y, z, 颜色索引)]。
    每个柜格 1 份；extra_in_comp2 指定在柜格 2 内额外放几份（并排摆放）。
    """
    out: List[Tuple[float, float, float, int]] = []
    for k, (x, y, z, c) in enumerate(K.locker_package_slots()):
        if k == TARGET_DOOR:
            n = 1 + extra_in_comp2
            for i in range(n):
                out.append((x + (i - (n - 1) / 2.0) * 0.005, y, z + i * 0.20, i % len(PKG)))
        else:
            out.append((x + x_off, y, z, c))
    return out


def _comp_of(x: float, y: float) -> int:
    """
    由餐盒的 (x, y) 反查所属柜格编号。

    柜格由 6 个餐槽定义，同一柜格可并排放多份（z 不同），
    因此只能按 (x, y) 就近匹配，不能用 z。
    """
    best, bd = -1, 1e9
    for k, (sx, sy, _sz, _c) in enumerate(K.locker_package_slots()):
        d = (x - sx) ** 2 + (y - sy) ** 2
        if d < bd:
            bd, best = d, k
    return best


def _shelf_counts(shelf: Sequence[Tuple[float, float, float, int]]) -> List[int]:
    """逐柜格餐盒数量（长度 6）。"""
    cnt = [0] * len(K.locker_package_slots())
    for x, y, _z, _c in shelf:
        cnt[_comp_of(x, y)] += 1
    return cnt


# ---------------------------------------------------------------- 六类行为
def spec_normal_pickup() -> BehaviorSpec:
    def actors(rng):
        j = rng.uniform(-0.10, 0.10)
        return [_walk_actor(2.0, 5.4, PICKUP_X + j, PICKUP_Z, -46.0, 2.6, 3.6,
                            KeyTrack([(0.0, 0.0), (3.80, 0.0), (3.95, 1.0), (3600.0, 1.0)]))]
    def door(rng):
        return KeyTrack([(0.0, 0.0), (2.0, 0.0), (2.6, DOOR_OPEN_ANGLE), (4.8, DOOR_OPEN_ANGLE), (5.4, 0.0), (3600.0, 0.0)])
    def shelf(rng, t):
        base = _shelf_default()
        taken = t >= 3.95
        return [p for i, p in enumerate(base) if not (i == TARGET_DOOR and taken)]
    return BehaviorSpec("normal_pickup", "正常取餐", False, 10.0, actors, door, shelf,
                        {"LOITERING": "no", "DOOR_LEFT_OPEN": "no"},
                        "开门取餐并顺手关门离开")


def spec_loitering() -> BehaviorSpec:
    """
    徘徊观察：人在外卖柜前来回踱步、三次驻足观望取餐区但始终未取餐。

    轨迹设计（全部由检测实测反推，迭代记录见 logs/dataset_eval_check.json）：
      1. 三个歇脚点分别位于取餐区的**左 / 右 / 靠镜头前方**，均为人形全尺寸位置——
         旧版把「外围」放在远处（z 更深），人形过小，YOLOv8n 置信度跌到 0.2 以下，
         出现 1.7~3.1s 的连片漏检，同一人被拆成多条轨迹（TTL 内无法续配），
         每条轨迹的进出次数都到不了 3 次，LOITERING 永远无法成立；
      2. 每次从歇脚点走到取餐区中点（进入 ROI → entries+1）再走向下一个歇脚点
         （离开 ROI），三圈后 entries ≥ 3、presence ≥ 15s，规则自然触发；
      3. 走动段占绝对多数、驻足仅 1s 左右——行走姿态的检出置信度（约 0.69）
         远高于双臂下垂的站姿（0.0~0.24），双手前置站姿约 0.49。
    """
    def actors(rng):
        j = rng.uniform(-0.10, 0.10)
        near = (PICKUP_X + j + 0.05, PICKUP_Z)           # 取餐区内（计入 entries）
        rests = [(PICKUP_X + j - 1.21, PICKUP_Z),        # 左侧（同深度，ROI 外）
                 (PICKUP_X + j + 0.89, PICKUP_Z),        # 右侧（ROI 外）
                 (PICKUP_X + j + 0.29, PICKUP_Z - 0.68)] # 靠镜头前方（ROI 下缘之外）
        keys_x, keys_z = [(0.0, rests[0][0])], [(0.0, rests[0][1])]
        walk_w, face_f = [(0.0, 0.0)], [(0.0, 0.0)]
        t = 0.0

        def go_to(p, dur, yaw):
            nonlocal t
            t += dur
            keys_x.append((t, p[0])); keys_z.append((t, p[1]))
            walk_w.append((t - dur + 0.05, 1.0))
            face_f.append((t - dur + 0.05, yaw))

        def pause(dur, yaw=0.0):
            nonlocal t
            keys_x.append((t, keys_x[-1][1])); keys_z.append((t, keys_z[-1][1]))
            walk_w.append((t, 0.0))
            face_f.append((t, yaw))
            t += dur

        pause(0.9, 0.0)                                  # 开场在左侧歇脚点观望
        for i in range(3):
            go_to(near, 1.3, 0.0)                        # 走进取餐区（面向柜子）
            pause(1.1, 0.0)                              # 驻足观望
            nxt = rests[(i + 1) % 3]
            if nxt[0] > near[0] + 0.1:
                yaw = 96.0                               # 向右横移
            elif nxt[0] < near[0] - 0.1:
                yaw = -96.0                              # 向左横移
            else:
                yaw = 170.0                              # 朝镜头方向退到前排
            go_to(nxt, 1.3, yaw)
            pause(0.9, 0.0)
        # 尾段：在「左侧 ↔ 前方」两个歇脚点之间往返踱步（不再进区），保持可检出
        t_end = 21.0
        while t < t_end - 2.2:
            go_to(rests[2], 1.3, 170.0)
            pause(0.8, 0.0)
            if t >= t_end - 2.2:
                break
            go_to(rests[0], 1.3, -96.0)
            pause(0.8, 0.0)
        keys_x.append((t_end, keys_x[-1][1])); keys_z.append((t_end, keys_z[-1][1]))
        walk_w.append((t_end, 0.0)); face_f.append((t_end, 0.0))
        return [Actor(
            px=KeyTrack(keys_x), pz=KeyTrack(keys_z),
            facing=KeyTrack(face_f),
            lean=KeyTrack([(0.0, 3.0), (t_end, 3.0)]),
            sh_r=KeyTrack([(0.0, -20.0), (t_end, -20.0)]),
            el_r=KeyTrack([(0.0, -34.0), (t_end, -34.0)]),
            sh_l=KeyTrack([(0.0, -18.0), (t_end, -18.0)]),
            el_l=KeyTrack([(0.0, -30.0), (t_end, -30.0)]),
            walk=KeyTrack(walk_w),
            pkg=KeyTrack([(0.0, 0.0), (3600.0, 0.0)]),
        )]
    def door(rng):
        return KeyTrack([(0.0, 0.0), (3600.0, 0.0)])      # 全程不接触柜门
    def shelf(rng, t):
        return _shelf_default()
    return BehaviorSpec("loitering", "徘徊观察", True, 21.0, actors, door, shelf,
                        {"LOITERING": "yes", "DOOR_LEFT_OPEN": "no"},
                        "反复靠近又退回，在外围长时间滞留且未取餐")


def spec_multi_retrieval() -> BehaviorSpec:
    def actors(rng):
        j = rng.uniform(-0.08, 0.08)
        # 右手 3.95s 取第一份；左手 5.85s 取第二份（由 _walk_actor 按 pkg 轨道自动编排双臂）
        a = _walk_actor(2.0, 6.5, PICKUP_X + j, PICKUP_Z, -46.0, 2.6, 3.6,
                        KeyTrack([(0.0, 0.0), (3.80, 0.0), (3.95, 1.0), (5.70, 1.0),
                                  (5.85, 2.0), (3600.0, 2.0)]))
        return [a]
    def door(rng):
        return KeyTrack([(0.0, 0.0), (2.0, 0.0), (2.6, DOOR_OPEN_ANGLE), (5.9, DOOR_OPEN_ANGLE), (6.5, 0.0), (3600.0, 0.0)])
    def shelf(rng, t):
        base = _shelf_default(extra_in_comp2=1)
        # 柜格 2 的两份：3.95s 取走第 1 份，5.85s 取走第 2 份
        comp2 = [p for i, p in enumerate(base) if i in (TARGET_DOOR, TARGET_DOOR + 1)]
        others = [p for i, p in enumerate(base) if i not in (TARGET_DOOR, TARGET_DOOR + 1)]
        live = []
        if t < 3.95:
            live.append(comp2[0])
        if t < 5.85 and len(comp2) > 1:
            live.append(comp2[1])
        return others + live
    return BehaviorSpec("multi_retrieval", "一人取多份", True, 11.0, actors, door, shelf,
                        {"MULTI_RETRIEVAL": "yes", "DOOR_LEFT_OPEN": "no"},
                        "同一柜格内两个餐盒被同一人依次取走（左右手各持一份）")


def spec_door_left_open() -> BehaviorSpec:
    def actors(rng):
        j = rng.uniform(-0.10, 0.10)
        return [_walk_actor(2.0, 5.2, PICKUP_X + j, PICKUP_Z, -46.0, 2.6, 3.6,
                            KeyTrack([(0.0, 0.0), (3.80, 0.0), (3.95, 1.0), (3600.0, 1.0)]))]
    def door(rng):
        return KeyTrack([(0.0, 0.0), (2.0, 0.0), (2.6, DOOR_OPEN_ANGLE), (3600.0, DOOR_OPEN_ANGLE)])   # 取餐后不再关门
    def shelf(rng, t):
        base = _shelf_default()
        taken = t >= 3.95
        return [p for i, p in enumerate(base) if not (i == TARGET_DOOR and taken)]
    return BehaviorSpec("door_left_open", "柜门未关", True, 22.0, actors, door, shelf,
                        {"DOOR_LEFT_OPEN": "yes", "FLAGGED_DOOR_RETRIEVAL": "no",
                         "MULTI_RETRIEVAL": "no"},
                        "取餐后未关柜门即离开；此后无人再取该柜格")


def spec_flagged_door_retrieval() -> BehaviorSpec:
    """
    物理逻辑关键：甲**只开门不取物**就离开（柜内餐盒仍在，柜门敞着）；
    柜门未关被标记后，乙取走被标记柜格内的餐盒 → 报警。

    绝不能出现「甲取空柜格后乙再从空柜格取物」这种不合物理且会造成检测器假触发的组合。
    """
    def actors(rng):
        j = rng.uniform(-0.10, 0.10)
        j2 = rng.uniform(-0.10, 0.10)
        # 甲：只开门、不取物，开门后即离开（柜内餐盒仍在，柜门敞着）
        a = _walk_actor(2.0, 4.0, PICKUP_X + j, PICKUP_Z, -46.0, 2.6, 3.2,
                        KeyTrack([(0.0, 0.0), (3600.0, 0.0)]))
        # 乙：先等在**画外**，14.4s 起以约 1 m/s 正常步速走入，16.5s 到位。
        # 时序约束：甲走远后（约 5s）柜格 2 才无遮挡，open_s 从约 5s 开始累计，
        # 10s 阈值在约 15s 触发 DOOR_LEFT_OPEN 并标记柜号；乙必须在标记**之后**才
        # 靠近柜体（约 16.3s 进入取餐区），否则其身体遮挡会让柜门状态变「未知」，
        # open_s 累计中断，报警与后续「取走未关柜门外卖」都无法成立
        # （旧版把 2.1 m 路程摊满 15 s，乙全程龟速逼近，正是栽在这里）。
        b = _walk_actor(16.5, 19.9, PICKUP_X + j2, PICKUP_Z, 32.0, 17.9, 18.9,
                        KeyTrack([(0.0, 0.0), (19.10, 0.0), (19.25, 1.0), (3600.0, 1.0)]),
                        t_start=14.4)
        return [a, b]
    def door(rng):
        return KeyTrack([(0.0, 0.0), (2.0, 0.0), (2.6, DOOR_OPEN_ANGLE), (3600.0, DOOR_OPEN_ANGLE)])
    def shelf(rng, t):
        base = _shelf_default()
        taken = t >= 19.25                        # 仅乙取走
        return [p for i, p in enumerate(base) if not (i == TARGET_DOOR and taken)]
    return BehaviorSpec("flagged_door_retrieval", "取走未关柜门外卖", True, 22.0,
                        actors, door, shelf,
                        {"DOOR_LEFT_OPEN": "yes", "FLAGGED_DOOR_RETRIEVAL": "yes"},
                        "开门未取即离开，柜门未关被标记后他人取走该柜格外卖")


SPECS = {s.name: s for s in (
    spec_normal_pickup(), spec_loitering(),
    spec_multi_retrieval(), spec_door_left_open(), spec_flagged_door_retrieval(),
)}


# ---------------------------------------------------------------- 时序自检
def validate_spec(sp: BehaviorSpec) -> List[str]:
    """
    确定性时序自检：禁止不合物理的动作组合。

    检查项（物理自洽性）：
      1. 手持餐盒**首次出现**必须落在伸手动作区间内（取物动作必须发生在手伸入柜时）；
      2. 首次伸手之前不得已手持餐盒（禁止「拿着外卖走过来再往柜里伸」）；
      3. 手持数量不得下降（食物不会凭空消失）；
      4. 声明「取走未关柜门外卖」的行为，柜内餐盒在**取走者**伸手期间必须仍然存在；
      5. 柜门开启累计时长与期望事件自洽。
    """
    errs: List[str] = []
    rng = np.random.default_rng(0)
    actors = sp.actors(rng)
    for ai, ac in enumerate(actors):
        first_reach = None
        for tt in np.arange(0.0, sp.seconds, 0.05):
            if ac.sh_r.at(tt) <= -60.0:
                first_reach = tt
                break
        hold_ts = [tt for tt in np.arange(0.0, sp.seconds, 0.05) if ac.pkg.at(tt) > 0.5]
        if hold_ts:
            t_first_hold = hold_ts[0]
            if first_reach is None:
                errs.append("主体%d 手持餐盒但全程未伸手取物（不合物理）" % ai)
            else:
                if t_first_hold < first_reach - 0.05:
                    errs.append("主体%d 在 %.2fs 已手持餐盒，却要到 %.2fs 才开始伸手"
                                "（禁止拿着外卖往柜里伸）" % (ai, t_first_hold, first_reach))
                if t_first_hold > first_reach + 2.0:
                    errs.append("主体%d 餐盒出现于 %.2fs，远离伸手区间（%.2fs 起）"
                                % (ai, t_first_hold, first_reach))
        # 手持数量不得下降
        counts = [ac.pkg.at(tt) for tt in np.arange(0.0, sp.seconds, 0.05)]
        if any(counts[i + 1] < counts[i] - 1e-6 for i in range(len(counts) - 1)):
            errs.append("主体%d 手持餐盒数量出现下降（食物不会凭空消失）" % ai)
    # 柜内餐盒在取走者伸手期间必须存在
    if sp.name == "flagged_door_retrieval":
        for t in np.arange(16.4, 17.7, 0.1):
            if len(sp.shelf(rng, t)) != 6:
                errs.append("取走未关柜门外卖：%.2fs 时柜格 2 内已无餐盒，取物前提不成立" % t)
                break
    # 柜门开启累计时长与期望自洽
    d = sp.door(rng)
    open_s = sum(0.05 for t in np.arange(0.0, sp.seconds, 0.05) if d.at(t) >= 12.0)
    if sp.expect.get("DOOR_LEFT_OPEN") == "no" and open_s >= 10.0:
        errs.append("柜门开启累计 %.1fs ≥ 10s，会额外触发「柜门未关」，与期望不符" % open_s)
    if sp.expect.get("DOOR_LEFT_OPEN") == "yes" and open_s < 10.0:
        errs.append("柜门开启累计 %.1fs < 10s，不会触发「柜门未关」，与期望不符" % open_s)
    return errs


# ---------------------------------------------------------------- 渲染单个片段
def render_clip(args) -> dict:
    name, idx, out_dir, fps, ssaa = args
    sp = SPECS[name]
    # 注意：不能用内置 hash()——字符串哈希受 PYTHONHASHSEED 随机化影响，
    # 会导致同一份代码两次运行产出不同片段，数据集不可复现。改用 crc32。
    seed = zlib.crc32(("%s:%d" % (name, idx)).encode("utf-8"))
    rng = np.random.default_rng(seed)

    actors = sp.actors(rng)
    door = sp.door(rng)
    n_frames = int(round(sp.seconds * fps))
    stem = "%s_%03d" % (name, idx)
    out = Path(out_dir)
    video_path = out / (stem + ".mp4")
    truth_path = out / (stem + "_truth.csv")

    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    light = S.Light()
    floor = S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0))
    walls = [S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                         base_color=(0.80, 0.81, 0.83))]
    # 程序化纹理：确定性、零成本，且已实测不破坏柜门/餐盒检测阈值
    renderer = S.Renderer(OUT_W, OUT_H, ssaa=ssaa, textures=TX.Textures(seed=20260925))
    import cv2
    vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))

    rows: List[dict] = []
    for i in range(n_frames):
        t = i / float(fps)
        poses = [ac.pose(t) for ac in actors]
        shelf = sp.shelf(rng, t)
        doors = list(DOOR_CLOSED_ALL)
        doors[TARGET_DOOR] = door.at(t)
        items = K.build_locker(doors)
        for p in poses:
            items += K.build_human(p)
        items += K.build_packages(shelf)
        occ = [(m, mt) for m, mt, _ in items]
        if i == 0:
            renderer.render(items, cam, light, floor=floor, walls=walls, shadow_occluders=occ)
        img = renderer.render(items, cam, light, floor=floor, walls=walls, shadow_occluders=occ)
        vw.write(img)

        counts = _shelf_counts(shelf)
        row = {"frame": i, "t": round(t, 3), "n_persons": len(poses),
               "door_angle_deg": round(door.at(t), 1),
               "pkg_total": sum(counts), "pkg_in_comp2": counts[TARGET_DOOR]}
        for kk in range(6):
            row["door_open_%d" % kk] = int(door.at(t) >= 12.0) if kk == TARGET_DOOR else 0
            row["pkg_truth_%d" % kk] = counts[kk]
        for ai, p in enumerate(poses):
            ins, _w = hand_in_cabinet(p)
            row["a%d_hand_in_cabinet" % ai] = int(ins)
            row["a%d_packages_in_hand" % ai] = len(p.packages_in_hand)
        rows.append(row)
    vw.release()

    with truth_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    max_hand = max(r["a%d_packages_in_hand" % i] for r in rows for i in range(len(actors)))
    return {"clip": stem, "behavior": sp.name, "label_cn": sp.label_cn,
            "is_abnormal": int(sp.is_abnormal), "frames": n_frames,
            "seconds": sp.seconds, "fps": fps, "n_persons": len(actors),
            "max_packages_in_hand": max_hand,
            "door_open_frames": sum(1 for r in rows if r["door_open_%d" % TARGET_DOOR]),
            "truth": truth_path.name, "video": video_path.name}


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="批量生成六类行为的三维场景数据集")
    ap.add_argument("--behavior", default="all",
                    help="all 或具体行为名：%s" % ", ".join(SPECS))
    ap.add_argument("--count", type=int, default=0, help="每个行为生成的片段数（0=按默认配比）")
    ap.add_argument("--out", default=str(C.BASE_DIR / "data" / "simulated"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0, help="0=自动取 CPU 核心数-2")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()

    names = list(SPECS) if a.behavior == "all" else [a.behavior]
    for n in names:
        if n not in SPECS:
            print("未知行为：%s（可选：%s）" % (n, ", ".join(SPECS)))
            return 2

    # ---- 时序自检（先验证再渲染，避免生成不合物理的数据）
    print("=" * 78)
    print("时序自检")
    bad = 0
    for n in names:
        errs = validate_spec(SPECS[n])
        status = "通过" if not errs else "不通过"
        print("  %-26s %-6s %s" % (n, status, SPECS[n].note))
        for e in errs:
            print("     ✗ %s" % e); bad += 1
    print("=" * 78)
    if bad:
        print("自检未通过，终止生成。")
        return 1
    if a.validate_only:
        return 0

    # ---- 配比：异常 : 正常 = 3 : 1
    if a.count:
        tasks = [(n, i + 1, a.out, a.fps, a.ssaa) for n in names for i in range(a.count)]
    else:
        # 5 类异常各 21 段 = 105；正常 35 段 → 105 : 35 = 3 : 1，合计 140 段
        tasks = []
        for n in names:
            cnt = 35 if not SPECS[n].is_abnormal else 21
            tasks += [(n, i + 1, a.out, a.fps, a.ssaa) for i in range(cnt)]
    Path(a.out).mkdir(parents=True, exist_ok=True)

    workers = a.workers or max(2, (__import__("os").cpu_count() or 8) - 2)
    n_abn = sum(1 for n, *_ in tasks if SPECS[n].is_abnormal)
    print("待生成 %d 段（异常 %d : 正常 %d = %.2f : 1），%d 进程并行"
          % (len(tasks), n_abn, len(tasks) - n_abn, n_abn / max(1, len(tasks) - n_abn), workers))

    t0 = time.perf_counter()
    results: List[dict] = []
    done = 0
    with __import__("multiprocessing").Pool(workers) as pool:
        for r in pool.imap_unordered(render_clip, tasks):
            results.append(r); done += 1
            if done % 5 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print("  进度 %3d/%d  已用 %.0fs  预计剩余 %.0fs"
                      % (done, len(tasks), el, el / done * (len(tasks) - done)))

    # ---- labels.csv 汇总
    results.sort(key=lambda r: (r["behavior"], r["clip"]))
    labels = Path(a.out) / "labels.csv"
    cols = ["clip", "video", "behavior", "label_cn", "is_abnormal", "frames", "seconds",
            "fps", "n_persons", "max_packages_in_hand", "door_open_frames", "truth"]
    with labels.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})

    print("\n" + "=" * 78)
    print("已生成 %d 段，总耗时 %.1f 秒（%.2f 秒/段）" % (len(results),
                                                time.perf_counter() - t0,
                                                (time.perf_counter() - t0) / max(1, len(results))))
    by = {}
    for r in results:
        by.setdefault(r["behavior"], []).append(r)
    print("%-26s %5s %8s %10s %s" % ("行为", "段数", "帧数", "手持最多", "说明"))
    for n, rs in sorted(by.items()):
        print("%-26s %5d %8d %10d %s" % (n, len(rs), sum(x["frames"] for x in rs),
                                         max(x["max_packages_in_hand"] for x in rs), SPECS[n].label_cn))
    print("\n标注文件：%s" % labels)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
