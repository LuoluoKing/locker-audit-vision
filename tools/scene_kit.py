# -*- coding: utf-8 -*-
"""
tools/scene_kit.py — 取餐场景的三维构件库（柜体 / 铰链柜门 / 关节式人形 / 外卖餐盒）

构件说明
--------
* **取餐柜**：柜体 + 3×2 共 6 扇**铰链柜门**（绕外侧竖边旋转，角度 0~95°）+ 层板
* **人形**：12 段骨架的**关节式**模型（骨盆→躯干→头 / 肩→上臂→前臂→手 / 髋→大腿→小腿→脚），
  用前向运动学（FK）按关节点逐级求解世界变换，因此可做出**伸手取物、抬手、屈肘**等动作
* **餐盒**：小尺寸彩色盒体，可放在层板上或**握在手中**（用于「一人取多份」的可观测证据）

坐标与旋转约定（与 tools/scene3d.py 一致）
------------------------------------------
* 左手系：+X 右、+Y 上、+Z 向前（远离摄像机）
* 行向量约定，v_world = v_local @ M
* **局部旋转必须写在平移之前**：`compose(rotate, translate)`；
  写成 `compose(translate, rotate)` 会变成绕世界原点旋转（本模块已按正确顺序封装）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import scene3d as S

Mat4 = np.ndarray
RGB = Tuple[float, float, float]

# ---------------------------------------------------------------- 配色
C_CABINET = (0.50, 0.53, 0.58)
C_CABINET_TOP = (0.58, 0.61, 0.66)
C_DOOR = (0.72, 0.74, 0.78)
C_DOOR_OPEN_SIDE = (0.60, 0.62, 0.66)
C_HANDLE = (0.30, 0.32, 0.36)
C_SHELF = (0.64, 0.66, 0.70)
C_INTERIOR = (0.26, 0.28, 0.32)          # 柜内暗色，作为「柜门开启」的视觉证据
C_SKIN = (0.80, 0.64, 0.52)
C_HAIR = (0.16, 0.14, 0.13)
C_SHIRT = (0.22, 0.34, 0.56)
C_SHIRT2 = (0.55, 0.28, 0.30)
C_PANTS = (0.17, 0.20, 0.28)
C_SHOE = (0.11, 0.12, 0.15)
C_BOX_A = (0.90, 0.86, 0.30)             # 餐盒配色（高饱和，便于视觉检测）
C_BOX_B = (0.88, 0.46, 0.24)
C_BOX_C = (0.30, 0.72, 0.52)
PACKAGE_COLORS = (C_BOX_A, C_BOX_B, C_BOX_C)

# ---------------------------------------------------------------- 几何尺寸（米）
CAB_W, CAB_H, CAB_D = 1.60, 1.90, 0.52
DOOR_ROWS, DOOR_COLS = 3, 2
DOOR_GAP = 0.012
DOOR_H = (CAB_H - 0.10) / DOOR_ROWS - DOOR_GAP
DOOR_W = (CAB_W - 0.06) / DOOR_COLS - DOOR_GAP
PKG_W, PKG_H, PKG_D = 0.20, 0.12, 0.16

SEG = dict(
    pelvis=0.26, chest=0.46, waist_gap=0.05, head=0.23, neck=0.07,
    upper_arm=0.30, fore_arm=0.28, thigh=0.44, shin=0.42, foot_h=0.08, foot_d=0.24,
    hip_w=0.19, shoulder_w=0.20,
    w_pelvis=0.30, w_chest=0.34, d_body=0.20,
    r_arm=0.055, r_hand=0.062, r_leg=0.072, r_foot=0.075,
)


# ================================================================ 柜体
def build_cabinet(door_angles: Sequence[float], base: Tuple[float, float, float] = (0, 0, 0)
                  ) -> List[Tuple[S.Mesh, Mat4, RGB]]:
    """
    构建取餐柜。

    door_angles : 6 个柜门开启角（度），顺序为 行优先 从上到下、每行从左到右。
                  0 = 关闭（贴合柜面），正值 = 绕外侧竖边向外开启。
    """
    bx, by, bz = base
    out: List[Tuple[S.Mesh, Mat4, RGB]] = []
    cube = _cube()

    # 柜体外壳：用 5 块板拼成（缺正面，正面由柜门承担），这样开门后能看到暗色内部
    t = 0.035
    back = S.compose(S.translate(bx, by + CAB_H / 2, bz + CAB_D / 2 - t / 2))
    out.append((cube, S.compose(S.scale_m(CAB_W, CAB_H, t), back), C_CABINET))
    left = S.compose(S.scale_m(t, CAB_H, CAB_D),
                     S.translate(bx - CAB_W / 2 + t / 2, by + CAB_H / 2, bz))
    out.append((cube, left, C_CABINET))
    right = S.compose(S.scale_m(t, CAB_H, CAB_D),
                      S.translate(bx + CAB_W / 2 - t / 2, by + CAB_H / 2, bz))
    out.append((cube, right, C_CABINET))
    top = S.compose(S.scale_m(CAB_W, t, CAB_D),
                    S.translate(bx, by + CAB_H - t / 2, bz))
    out.append((cube, top, C_CABINET_TOP))
    bottom = S.compose(S.scale_m(CAB_W, t, CAB_D),
                       S.translate(bx, by + t / 2, bz))
    out.append((cube, bottom, C_CABINET))

    # 内部背板（暗色）：柜门开启时最直接的视觉证据
    out.append((cube, S.compose(S.scale_m(CAB_W - 2 * t, CAB_H - 2 * t, 0.012),
                                S.translate(bx, by + CAB_H / 2, bz + CAB_D / 2 - t - 0.008)),
                C_INTERIOR))

    # 每层隔板
    for r in range(DOOR_ROWS + 1):
        y = by + 0.05 + (CAB_H - 0.10) * r / DOOR_ROWS
        out.append((cube, S.compose(S.scale_m(CAB_W - 2 * t, 0.022, CAB_D - t), S.translate(bx, y, bz)),
                    C_SHELF))

    # 竖隔板（分左右两列）
    out.append((cube, S.compose(S.scale_m(0.022, CAB_H - 0.10, CAB_D - t),
                                S.translate(bx, by + CAB_H / 2, bz)), C_SHELF))

    # 铰链柜门
    door = _cube()
    for i, ang in enumerate(door_angles[:DOOR_ROWS * DOOR_COLS]):
        r, c = divmod(i, DOOR_COLS)
        sign = -1.0 if c == 0 else 1.0                       # 左门向外左开，右门向外右开
        hinge_x = bx + sign * CAB_W / 2
        hy = by + 0.05 + (CAB_H - 0.10) * (DOOR_ROWS - 1 - r) / DOOR_ROWS + DOOR_H / 2 + DOOR_GAP / 2
        hz = bz - CAB_D / 2 - 0.006
        a = math.radians(ang) * -sign                        # 开启方向
        # 门板先沿 -sign 方向平移半个门宽（使其以铰链边为原点），再绕铰链旋转，最后放到铰链位置
        m = S.compose(S.scale_m(DOOR_W, DOOR_H, 0.026),
                      S.translate(-sign * DOOR_W / 2, 0, 0),
                      S.rotate_y(a),
                      S.translate(hinge_x, hy, hz))
        color = C_DOOR if ang < 3.0 else C_DOOR_OPEN_SIDE
        out.append((door, m, color))
        # 门把手
        hp = S.compose(S.scale_m(0.05, 0.10, 0.05),
                       S.translate(-sign * (DOOR_W - 0.10), 0, -0.03),
                       S.rotate_y(a),
                       S.translate(hinge_x, hy, hz))
        out.append((cube, hp, C_HANDLE))

    return out


# ================================================================ 人形（关节式）
@dataclass
class Pose:
    """人形关节角（单位：度）。所有角度均以「绕局部 X 轴前后摆动」为主。"""
    x: float = 0.0                    # 世界位置
    z: float = 0.0
    facing: float = 0.0               # 朝向角（绕 Y，0 = 面向 +Z 即背对摄像机）
    lean: float = 0.0                 # 躯干前倾
    shoulder_l: float = 0.0           # 上臂前摆
    shoulder_r: float = 0.0
    elbow_l: float = 0.0              # 肘部弯曲
    elbow_r: float = 0.0
    hip_l: float = 0.0                # 大腿前摆
    hip_r: float = 0.0
    knee_l: float = 0.0
    knee_r: float = 0.0
    stride: float = 0.0               # 双腿分开的横向偏移（用于站姿微调）
    shirt: RGB = C_SHIRT
    packages_in_hand: Tuple[str, ...] = ()   # 手持餐盒：'l' 左手 / 'r' 右手


def human_joints(pose: "Pose", base_y: float = 0.0) -> Dict[str, Mat4]:
    """
    前向运动学：返回具名关节的世界矩阵（骨盆/脊柱/胸顶/头基/肩肘腕/髋膝）。

    单独抽出关节解算，是为了让「手腕是否进入柜格」这类**真值判定**按关节名取矩阵，
    而不是依赖 build_human 输出列表中的下标——后者会随手持餐盒数量变化而错位
    （曾因此把「手在柜内」的真值恒判为 0）。
    """
    seg = SEG
    root = S.compose(S.rotate_y(math.radians(pose.facing)),
                     S.translate(pose.x, base_y, pose.z),
                     S.translate(0, seg["thigh"] + seg["shin"] + seg["foot_h"] + seg["pelvis"] / 2, 0))
    spine = S.compose(S.rotate_x(math.radians(pose.lean)), root)
    chest_top = S.compose(S.translate(0, seg["pelvis"] / 2 + seg["chest"], 0), spine)
    head_base = S.compose(S.translate(0, seg["neck"], 0), chest_top)

    J: Dict[str, Mat4] = {"root": root, "spine": spine, "chest_top": chest_top, "head_base": head_base}
    sh_y = seg["pelvis"] / 2 + seg["chest"] - 0.04
    for side, ang_sh, ang_el in (("l", pose.shoulder_l, pose.elbow_l),
                                 ("r", pose.shoulder_r, pose.elbow_r)):
        sx = (-1 if side == "l" else 1) * seg["shoulder_w"]
        shoulder = S.compose(S.rotate_x(math.radians(ang_sh)),
                             S.translate(sx, sh_y, 0), spine)
        elbow = S.compose(S.rotate_x(math.radians(ang_el)),
                          S.translate(0, -seg["upper_arm"], 0), shoulder)
        wrist = S.compose(S.translate(0, -seg["fore_arm"] - 0.02, 0), elbow)
        J["shoulder_" + side] = shoulder
        J["elbow_" + side] = elbow
        J["wrist_" + side] = wrist
    for side, ang_hip, ang_knee in (("l", pose.hip_l, pose.knee_l), ("r", pose.hip_r, pose.knee_r)):
        sgn = -1 if side == "l" else 1
        hip = S.compose(S.rotate_x(math.radians(ang_hip)),
                        S.translate(sgn * (seg["hip_w"] + pose.stride), -seg["pelvis"] / 2, 0), root)
        J["hip_" + side] = hip
        J["knee_" + side] = S.compose(S.rotate_x(math.radians(ang_knee)),
                                      S.translate(0, -seg["thigh"], 0), hip)
    return J


def build_human(pose: Pose, base_y: float = 0.0) -> List[Tuple[S.Mesh, Mat4, RGB]]:
    """
    按前向运动学构建关节式人形。返回 (网格, 世界矩阵, 颜色) 列表。

    **矩阵顺序约定（行向量，务必遵守）**
        v_world = v_local @ M_local @ M_parent
    即 `S.compose(局部变换..., 父变换)` —— 局部在**左**、父变换在**右**。
    若写成 `S.compose(父变换, 局部变换)`（父在左），局部缩放会连同父变换的平移一起被缩放，
    典型症状是「人形被压扁并贴到原点附近」。本模块所有调用均按上述顺序书写，并在
    `verify_fk()` 中做数值自校验。

    局部链内部顺序：先 `scale_m` 定义几何尺寸，再 `translate` 摆到关节处；
    关节旋转写在平移**之前**（`compose(rotate, translate)`）才是绕关节自身转动。
    """
    seg = SEG
    out: List[Tuple[S.Mesh, Mat4, RGB]] = []
    cube, ball = _cube(), _ball()
    J = human_joints(pose, base_y)            # 关节解算的唯一来源，避免两处 FK 产生漂移
    root, spine, chest_top, head_base = J["root"], J["spine"], J["chest_top"], J["head_base"]

    # 骨盆
    out.append((cube, S.compose(S.scale_m(seg["w_pelvis"], seg["pelvis"], seg["d_body"]), root),
                C_PANTS))
    # 躯干
    out.append((cube, S.compose(
        S.scale_m(seg["w_chest"], seg["chest"], seg["d_body"]),
        S.translate(0, seg["pelvis"] / 2 + seg["chest"] / 2, 0), spine), pose.shirt))
    # 颈 / 头 / 发
    out.append((cube, S.compose(S.scale_m(0.09, seg["neck"], 0.09),
                                S.translate(0, seg["neck"] / 2, 0), chest_top), C_SKIN))
    out.append((ball, S.compose(S.scale_m(seg["head"] * 0.46, seg["head"] * 0.52, seg["head"] * 0.44),
                                S.translate(0, seg["head"] * 0.52, 0), head_base), C_SKIN))
    out.append((ball, S.compose(S.scale_m(seg["head"] * 0.47, seg["head"] * 0.34, seg["head"] * 0.45),
                                S.translate(0, seg["head"] * 0.80, -0.01), head_base), C_HAIR))

    # 双臂：上臂 → 前臂 → 手 →（可选）手持餐盒
    for side, tag in (("l", "l"), ("r", "r")):
        shoulder, elbow, wrist = J["shoulder_" + side], J["elbow_" + side], J["wrist_" + side]
        out.append((cube, S.compose(
            S.scale_m(seg["r_arm"] * 2, seg["upper_arm"], seg["r_arm"] * 2),
            S.translate(0, -seg["upper_arm"] / 2, 0), shoulder), pose.shirt))
        out.append((cube, S.compose(
            S.scale_m(seg["r_arm"] * 1.7, seg["fore_arm"], seg["r_arm"] * 1.7),
            S.translate(0, -seg["fore_arm"] / 2, 0), elbow), C_SKIN))
        out.append((cube, S.compose(S.scale_m(seg["r_hand"] * 2.1, 0.10, seg["r_hand"] * 1.5),
                                    S.translate(0, -0.05, 0), wrist), C_SKIN))
        for k in range(sum(1 for x in pose.packages_in_hand if x == tag)):
            out.append((cube, S.compose(
                S.scale_m(PKG_W, PKG_H, PKG_D),
                S.translate(0, -0.10 - k * (PKG_H + 0.02), 0.03), wrist),
                PACKAGE_COLORS[k % len(PACKAGE_COLORS)]))

    # 双腿：大腿 → 小腿 → 脚
    for side in ("l", "r"):
        hip, knee = J["hip_" + side], J["knee_" + side]
        out.append((cube, S.compose(
            S.scale_m(seg["r_leg"] * 2.1, seg["thigh"], seg["r_leg"] * 2.1),
            S.translate(0, -seg["thigh"] / 2, 0), hip), C_PANTS))
        out.append((cube, S.compose(
            S.scale_m(seg["r_leg"] * 1.8, seg["shin"], seg["r_leg"] * 1.8),
            S.translate(0, -seg["shin"] / 2, 0), knee), C_PANTS))
        out.append((cube, S.compose(
            S.scale_m(seg["r_foot"] * 2.0, seg["foot_h"], seg["foot_d"]),
            S.translate(0, -seg["shin"] - seg["foot_h"] / 2 + 0.01, seg["foot_d"] * 0.22), knee),
            C_SHOE))

    return out


def verify_fk(verbose: bool = True) -> Dict[str, float]:
    """
    前向运动学的确定性自校验：给定静止站姿，检查关键部位的世界坐标是否落在合理范围。

    返回各项实测值；任一超出容差即视为姿态解算错误（供单元测试断言使用）。
    """
    pose = Pose(x=0.0, z=0.0, facing=0.0)
    parts = build_human(pose)
    centers = [m.transformed(mt).verts.mean(axis=0) for m, mt, _ in parts]
    pelvis = centers[0]
    head_top = centers[4] if len(centers) > 4 else centers[3]
    feet = centers[-1]
    seg = SEG
    expect_hip_y = seg["thigh"] + seg["shin"] + seg["foot_h"]
    expect_pelvis_y = expect_hip_y + seg["pelvis"] / 2
    expect_head_y = expect_pelvis_y + seg["pelvis"] / 2 + seg["chest"] + seg["neck"] + seg["head"] * 0.52
    got = {
        "pelvis_y": float(pelvis[1]),
        "head_y": float(centers[3][1]),          # 索引 3 = 头部球心（4 是头发，非头部）
        "foot_y": float(feet[1]),
        "expect_pelvis_y": expect_pelvis_y,
        "expect_head_y": expect_head_y,
    }
    got["ok"] = float(
        abs(got["pelvis_y"] - expect_pelvis_y) < 0.02 and
        abs(got["head_y"] - expect_head_y) < 0.03 and
        abs(got["foot_y"] - seg["foot_h"] / 2) < 0.03)
    if verbose:
        print("FK 自校验：骨盆高 %.3f（期望 %.3f）｜头顶高 %.3f（期望 %.3f）｜脚高 %.3f（期望 %.3f）→ %s"
              % (got["pelvis_y"], expect_pelvis_y, got["head_y"], expect_head_y,
                 got["foot_y"], seg["foot_h"] / 2, "通过" if got["ok"] else "不通过"))
    return got


# ================================================================ 餐盒
def build_packages(specs: Sequence[Tuple[float, float, float, int]]
                   ) -> List[Tuple[S.Mesh, Mat4, RGB]]:
    """货架上的餐盒。specs = [(x, y, z, 颜色索引), ...]"""
    cube = _cube()
    return [(cube, S.compose(S.scale_m(PKG_W, PKG_H, PKG_D), S.translate(x, y, z)),
             PACKAGE_COLORS[c % len(PACKAGE_COLORS)]) for x, y, z, c in specs]


def shelf_package_slots(base: Tuple[float, float, float] = (0, 0, 0)) -> List[Tuple[float, float, float, int]]:
    """给出 6 个格位中餐盒的标准摆放位置（每格靠近门口，便于被取走）。"""
    bx, by, bz = base
    slots = []
    for r in range(DOOR_ROWS):
        for c in range(DOOR_COLS):
            x = bx + (-1 if c == 0 else 1) * (CAB_W / 4)
            y = by + 0.05 + (CAB_H - 0.10) * (DOOR_ROWS - 1 - r) / DOOR_ROWS + DOOR_H * 0.45
            slots.append((x, y, bz - CAB_D / 2 + 0.16, r * DOOR_COLS + c))
    return slots


# ================================================================ 真实智能取餐柜（参数化）
# 配色（**必须避开 PackageDetector 的 HSV 判据**：H∈[15,95] 且 S≥130 且 V≥90 且面积∈[60,1200]）
# 教训：初版把格口编号牌设为橙黄 (0.94,0.72,0.20)、扫码灯设为绿 (0.30,0.78,0.46)，
# 两者都落进餐盒色域；实测编号牌（约 10×10 px、面积 88）落在格口多边形**之外**
# 但在人物外扩框**之内**，被 `count_carried_packages` 计为手持餐盒，
# 导致演示素材凭空出现「一人取多份」误报（tools/ 下 demo 校验脚本实测）。
# 现改为：编号牌用柜体同款深蓝、品牌条用浅灰、扫码灯用低饱和浅青，
# 三者的色相或饱和度都在餐盒色域之外。
LOCKER_STYLE = {
    "body": (0.13, 0.26, 0.46),          # 柜体钢制深蓝（H≈108）
    "body_light": (0.20, 0.35, 0.56),    # 立柱/边框亮蓝（H≈108）
    "door": (0.88, 0.89, 0.90),          # 门板浅灰白（S≈5）
    "door_open": (0.74, 0.76, 0.78),
    "handle": (0.60, 0.63, 0.66),        # 银灰把手（S≈23）
    "shelf": (0.55, 0.57, 0.60),         # 层板/隔板（S≈21）
    "base": (0.20, 0.21, 0.23),          # 底座踢脚（S≈35）
    "canopy": (0.10, 0.20, 0.38),        # 顶部雨棚/品牌条底（H≈108）
    "brand_strip": (0.90, 0.91, 0.92),   # 品牌条文字（浅灰，S≈5）
    "screen_panel": (0.24, 0.26, 0.29),  # 操作屏面板
    "screen_glow": (0.26, 0.62, 0.88),   # 屏幕亮蓝（H≈103）
    "screen_scan": (0.55, 0.90, 0.95),   # 扫码区提示灯（浅青，S≈108 < 130）
    "label": (0.10, 0.20, 0.38),         # 格口编号牌（深蓝，H≈109，避开餐盒色域）
}

LOCKER_W, LOCKER_H, LOCKER_D = 1.62, 2.00, 0.50
SCREEN_COL_W = 0.28                     # 最右侧操作屏列宽（不设柜门）
LOCKER_BASE_H = 0.12                    # 底座踢脚高
LOCKER_CANOPY_H = 0.10                  # 顶部品牌条高


def locker_geometry(rows: int, cols: int) -> dict:
    """由行/列数推出柜门与格口的几何量（米），供柜体构建与真值导出共用。"""
    inner_w = LOCKER_W - SCREEN_COL_W - 0.06
    inner_h = LOCKER_H - LOCKER_BASE_H - LOCKER_CANOPY_H - 0.04
    door_w = inner_w / cols - DOOR_GAP
    door_h = inner_h / rows - DOOR_GAP
    return {"rows": rows, "cols": cols, "inner_w": inner_w, "inner_h": inner_h,
            "door_w": door_w, "door_h": door_h, "x0": -inner_w / 2.0,
            "y0": LOCKER_BASE_H + 0.02, "col_pitch": inner_w / cols,
            "row_pitch": inner_h / rows}


def locker_door_center(geo: dict, r: int, c: int) -> Tuple[float, float]:
    """第 (r, c) 扇门的中心（世界 x, y），r 自上而下、c 自左而右。"""
    cx = geo["x0"] + (c + 0.5) * geo["col_pitch"]
    cy = geo["y0"] + (geo["rows"] - 0.5 - r) * geo["row_pitch"]
    return cx, cy


def cabinet_door_quad(r: int, c: int, inset: float = 0.06,
                      base: Tuple[float, float, float] = (0, 0, 0)) -> List[List[float]]:
    """
    **关闭状态**下第 (r, c) 扇柜门门面的四角世界坐标（顺时针）。

    这是柜门几何的**单一来源**：`build_cabinet` 的铰链位置与门板尺寸、以及
    `config.CABINET_COMPARTMENTS` 的标定，都必须与本函数一致。
    换几何后若只改一处，格口多边形就会与门面错位 —— 曾因此使左列三格
    （编号 0/2/4）把柜体左侧暗色立板算进格口，暗比恒高 0.21~0.89，
    在长停留片段里凭空触发「柜门未关」。

    inset：向格口中心内缩的比例，避免把门缝与立柱算进格口区域。
    """
    bx, by, bz = base
    sign = -1.0 if c == 0 else 1.0
    hinge_x = bx + sign * CAB_W / 2
    hy = by + 0.05 + (CAB_H - 0.10) * (DOOR_ROWS - 1 - r) / DOOR_ROWS + DOOR_H / 2 + DOOR_GAP / 2
    hz = bz - CAB_D / 2 - 0.006
    hw = DOOR_W / 2.0 * (1.0 - inset)
    hh = DOOR_H / 2.0 * (1.0 - inset)
    cx = hinge_x - sign * DOOR_W / 2.0                      # 门板中心（由铰链向内推半个门宽）
    return [
        [cx - hw, hy + hh, hz],
        [cx + hw, hy + hh, hz],
        [cx + hw, hy - hh, hz],
        [cx - hw, hy - hh, hz],
    ]


def build_locker(door_angles: Sequence[float], rows: int = DOOR_ROWS, cols: int = DOOR_COLS,
                 base: Tuple[float, float, float] = (0, 0, 0),
                 style: dict = None) -> List[Tuple[S.Mesh, Mat4, RGB]]:
    """
    构建**校园智能取餐柜**：真实外观，但**柜门几何与 ``build_cabinet`` 完全一致**。

    设计取舍（重要）
    ----------------
    真实感靠「配色 + 加装件」实现，而不是改柜体比例：

    * 配色改为钢制深蓝柜体 + 浅灰白门板 + 银灰把手 + 黄色格口编号牌；
    * 加装**右侧操作屏列**（触控屏 + 扫码区），位于柜体右侧**外侧**，不遮挡任何门面；
    * 加装**顶部品牌条**，位于柜顶上方，不遮挡门面；
    * 格口编号牌贴在门板上、随手开门一起动。

    为什么不改柜体比例：第一版把柜体加高加深、格口重排后，
    用 `tools/calib_locker_door_threshold.py` 实测「门开 / 门关」的暗比**分离度仅 +0.002**
    （原柜体为 +0.26），即敞开的门板把格口区域挡死，`DoorStateEstimator` 的
    「柜内暗区」前提不再成立。保持门面几何不变，则已标定的
    `config.CABINET_COMPARTMENTS` 与 `DoorStateEstimator` 阈值继续有效，
    数据集与评估结果可比，不会因换外观而推翻整套标定。

    rows / cols 仅为兼容签名，目前必须等于 ``DOOR_ROWS`` / ``DOOR_COLS``。
    """
    if (rows, cols) != (DOOR_ROWS, DOOR_COLS):
        raise ValueError("build_locker 目前仅支持 %d×%d 格口（改格口数需重做柜格标定与阈值标定）"
                         % (DOOR_ROWS, DOOR_COLS))
    st = dict(LOCKER_STYLE)
    if style:
        st.update(style)

    items = build_cabinet(door_angles, base)
    bx, by, bz = base
    cube = _cube()
    door_face_z = bz - CAB_D / 2.0

    # ---- 重新着色：柜体外壳/门板/把手/编号牌（几何未动，仅换材质）
    n_shell = 5                     # build_cabinet 前 5 项为外壳板
    recol = []
    for i, (mesh, m, col) in enumerate(items):
        if col == C_INTERIOR:
            recol.append((mesh, m, col))
        elif col == C_SHELF:
            recol.append((mesh, m, st["shelf"]))
        elif col in (C_DOOR, C_DOOR_OPEN_SIDE):
            recol.append((mesh, m, st["door"] if col == C_DOOR else st["door_open"]))
        elif col == C_HANDLE:
            recol.append((mesh, m, st["handle"]))
        elif col == C_CABINET_TOP:
            recol.append((mesh, m, st["body_light"]))
        else:
            recol.append((mesh, m, st["body"]))
    items = recol

    # ---- 加装：格口编号牌（贴门板，随门开合一起动）
    for i, ang in enumerate(list(door_angles)[:DOOR_ROWS * DOOR_COLS]):
        r, c = divmod(i, DOOR_COLS)
        sign = -1.0 if c == 0 else 1.0
        hinge_x = bx + sign * CAB_W / 2
        hy = by + 0.05 + (CAB_H - 0.10) * (DOOR_ROWS - 1 - r) / DOOR_ROWS + DOOR_H / 2 + DOOR_GAP / 2
        hz = bz - CAB_D / 2 - 0.006
        a = math.radians(ang) * -sign
        lb = S.compose(S.scale_m(0.055, 0.026, 0.008),
                       S.translate(sign * DOOR_W * 0.30, DOOR_H * 0.32, -0.016),
                       S.rotate_y(a), S.translate(hinge_x, hy, hz))
        items.append((cube, lb, st["label"]))

    # ---- 加装：顶部品牌条（在柜顶之上，不遮挡门面）
    items.append((cube, S.compose(S.scale_m(CAB_W + 0.06, LOCKER_CANOPY_H, CAB_D + 0.06),
                                  S.translate(bx, by + CAB_H + LOCKER_CANOPY_H / 2, bz)),
                  st["canopy"]))
    items.append((cube, S.compose(S.scale_m(CAB_W * 0.30, 0.045, 0.02),
                                  S.translate(bx - CAB_W * 0.22,
                                              by + CAB_H + LOCKER_CANOPY_H / 2,
                                              door_face_z - 0.02)), st["brand_strip"]))

    # ---- 加装：右侧操作屏列（在柜体右侧**外侧**，不遮挡任何门面）
    scx = bx + CAB_W / 2 + SCREEN_COL_W / 2
    items.append((cube, S.compose(S.scale_m(SCREEN_COL_W, CAB_H, CAB_D),
                                  S.translate(scx, by + CAB_H / 2, bz)), st["body"]))
    items.append((cube, S.compose(S.scale_m(SCREEN_COL_W * 0.80, CAB_H * 0.42, 0.03),
                                  S.translate(scx, by + CAB_H * 0.60,
                                              door_face_z - 0.005)), st["screen_glow"]))
    items.append((cube, S.compose(S.scale_m(SCREEN_COL_W * 0.30, CAB_H * 0.055, 0.03),
                                  S.translate(scx, by + CAB_H * 0.36,
                                              door_face_z - 0.005)), st["screen_scan"]))
    items.append((cube, S.compose(S.scale_m(SCREEN_COL_W * 0.20, CAB_H * 0.030, 0.03),
                                  S.translate(scx, by + CAB_H * 0.86,
                                              door_face_z - 0.005)), st["label"]))
    return items


def locker_package_slots(rows: int = DOOR_ROWS, cols: int = DOOR_COLS,
                         base: Tuple[float, float, float] = (0, 0, 0)
                         ) -> List[Tuple[float, float, float, int]]:
    """
    各格口内餐盒的标准摆放位置（每格靠近门口，便于被取走）；索引 = r*cols + c。

    **必须与 ``build_locker`` 的实际柜体几何一致**：``build_locker`` 为保证
    「柜内暗区」这一门状态判据成立，刻意沿用了 ``build_cabinet`` 的门面几何，
    因此餐盒槽位也必须取 ``shelf_package_slots``。
    曾因这里仍按「加高加深版」柜体算槽位（x 偏 8 cm、y 偏高 9 cm），
    使餐盒与格口错位、柜内计数与「取走柜内餐盒」判定一并受损。
    """
    return shelf_package_slots(base)


# ================================================================ 内部
def _cube() -> S.Mesh:
    return S.box_mesh(1.0, 1.0, 1.0)

def _ball() -> S.Mesh:
    return S.sphere_mesh(1.0, rings=4, segments=10)
