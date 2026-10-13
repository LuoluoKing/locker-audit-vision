# -*- coding: utf-8 -*-
"""
tools/gen_selftest_clip3d.py — 用三维渲染管线生成「链路自检 / 网页演示」视频

与旧版 2D 剪影素材的区别
------------------------
旧素材是平面几何图形，看不出柜门开合，也没有手部取物动作，检测器与时序规则都无法被有效触发。
本脚本改用 tools/scene3d.py + tools/scene_kit.py 的**真三维**管线，并**逐帧驱动关节与柜门角度**，
因此画面中可以清楚看到：

    走近取餐柜 → 柜门开启 → 手臂前伸入柜 → 取走餐盒 → 手臂收回 → 柜门关闭 → 转身离开

同时导出**逐帧真值**（柜门角度、手是否在柜内、手持餐盒数、人员位置），
为阶段3 的指标计算与阶段2 的数据集标注打基础。

命令行
------
    python tools/gen_selftest_clip3d.py                     # 默认 12s @ 15fps
    python tools/gen_selftest_clip3d.py --seconds 10 --fps 25 --out data/selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402

OUT_W, OUT_H, SSAA = 640, 480, 2

# 场景坐标：柜体位于原点，柜门面朝 -Z（摄像机方向）
# 侧向 3/4 机位：正对柜体时人物会几乎完全遮挡柜面，侧视才能同时看清
# 柜体形态、柜门开合角度与手部取物动作（对照图见 docs/08_阶段2证据/_机位对比.png）
CAM_EYE = (2.35, 1.92, -3.25)
CAM_TARGET = (-0.15, 1.02, 0.10)
CAM_FOV = 48.0

# 被操作的是「中排左门」= 门索引 2（r=1, c=0）
TARGET_DOOR = 2

# 柜门开启角（度）。真实取餐柜门通常开到 90° 以上；经 tools/calib_locker_door_threshold.py
# 实测，86° 时右列两格的门板会挡住柜内、暗比落到阈值 0.15 之下（仅 4/6 格可判）；
# 115° 时六格门开暗比 0.245~0.797、门关最大 0.053，现行阈值下 6/6 命中且零误报。
DOOR_OPEN_ANGLE_S = 115.0


# ---------------------------------------------------------------- 插值工具
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def seg(t: float, t0: float, t1: float) -> float:
    """把 t 映射到 [t0,t1] 区间内的 0~1 进度（自带截断）。"""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return max(0.0, min(1.0, (t - t0) / (t1 - t0)))


def ease(t: float) -> float:
    """平滑起止（3t²-2t³），避免动作出现硬拐点。"""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class KeyTrack:
    """单个标量量的关键帧轨道：(时刻, 值) 列表 + 缓动插值。"""

    def __init__(self, keys: Sequence[Tuple[float, float]]):
        self.keys = sorted(keys)

    def at(self, t: float) -> float:
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for i in range(len(ks) - 1):
            t0, v0 = ks[i]
            t1, v1 = ks[i + 1]
            if t0 <= t <= t1:
                return lerp(v0, v1, ease(seg(t, t0, t1)))
        return ks[-1][1]


# ---------------------------------------------------------------- 行为时间线
def make_timeline() -> Dict[str, KeyTrack]:
    """
    「正常取餐」时间线（单位：秒）。所有时刻均以片段起点为 0。

    0.0–2.2  从左侧走近取餐柜
    2.2–2.8  柜门开启
    2.8–4.2  右臂前伸入柜
    4.2–4.7  抓取餐盒（餐盒从层板转移到手中）
    4.7–5.7  手臂收回
    5.7–6.5  柜门关闭
    6.5–8.8  转身离开
    8.8–末   空场景（用于核对「柜门已关」）
    """
    return {
        "px":      KeyTrack([(0.0, -1.95), (2.0, 0.16), (5.4, 0.16), (8.0, 1.95), (12.0, 2.30)]),
        "pz":      KeyTrack([(0.0, -1.55), (2.0, -0.72), (5.4, -0.72), (8.0, -1.70), (12.0, -2.05)]),
        "facing":  KeyTrack([(0.0, -46.0), (2.0, 0.0), (5.2, 0.0), (8.0, 22.0), (12.0, 26.0)]),
        "lean":    KeyTrack([(0.0, 2.0), (2.0, 4.0), (3.2, 13.0), (4.6, 6.0), (5.4, 2.0), (8.0, 2.0)]),
        "sh_r":    KeyTrack([(0.0, -12.0), (2.6, -14.0), (3.6, -78.0), (4.0, -80.0),
                             (4.8, -16.0), (5.4, -10.0), (8.0, -18.0)]),
        "el_r":    KeyTrack([(0.0, -6.0), (2.6, -6.0), (3.6, 44.0), (4.0, 46.0),
                             (4.8, 6.0), (5.4, 2.0), (8.0, 4.0)]),
        "sh_l":    KeyTrack([(0.0, 14.0), (2.0, 6.0), (4.8, 6.0), (8.0, 18.0)]),
        "el_l":    KeyTrack([(0.0, 8.0), (2.0, -4.0), (8.0, 6.0)]),
        "walk":    KeyTrack([(0.0, 1.0), (1.9, 1.0), (2.2, 0.0), (5.3, 0.0), (5.7, 1.0), (8.1, 1.0)]),
        "pkg":     KeyTrack([(0.0, 0.0), (3.85, 0.0), (4.10, 1.0)]),
        "grip":    KeyTrack([(0.0, 0.0), (3.85, 0.0), (4.05, 1.0), (5.5, 1.0), (5.7, 0.0)]),
    }


def make_door_track() -> KeyTrack:
    """被操作柜门的开启角（度）：关闭 → 全开 → 保持 → 关闭。"""
    return KeyTrack([(0.0, 0.0), (2.0, 0.0), (2.6, DOOR_OPEN_ANGLE_S), (4.8, DOOR_OPEN_ANGLE_S), (5.4, 0.0), (12.0, 0.0)])


def build_pose(tl: Dict[str, KeyTrack], t: float, walk_freq: float = 1.55) -> K.Pose:
    """按时间线合成某一时刻的人体姿态（行走相由 walk 权重驱动）。"""
    w = tl["walk"].at(t)
    ph = t * walk_freq * 2 * math.pi
    swing = 26.0 * w * math.sin(ph)
    return K.Pose(
        x=tl["px"].at(t), z=tl["pz"].at(t), facing=tl["facing"].at(t), lean=tl["lean"].at(t),
        shoulder_l=tl["sh_l"].at(t) - 12.0 * w * math.sin(ph),
        shoulder_r=tl["sh_r"].at(t) + 12.0 * w * math.sin(ph),
        elbow_l=tl["el_l"].at(t), elbow_r=tl["el_r"].at(t),
        hip_l=swing, hip_r=-swing,
        knee_l=-18.0 * w * max(0.0, math.sin(ph)), knee_r=-18.0 * w * max(0.0, -math.sin(ph)),
        packages_in_hand=("r",) if tl["pkg"].at(t) > 0.5 else (),
    )


# ---------------------------------------------------------------- 渲染
def render_frame(cam: S.Camera, light: S.Light, floor: S.FloorPlane,
                 walls: Sequence[S.WallPlane], door_angle: float, pose: K.Pose,
                 shelf_pkgs: Sequence[Tuple[float, float, float, int]]):
    doors = [0.0] * (K.DOOR_ROWS * K.DOOR_COLS)
    doors[TARGET_DOOR] = door_angle
    items = K.build_locker(doors)
    items += K.build_human(pose)
    items += K.build_packages(shelf_pkgs)
    return items


def hand_in_cabinet(pose: K.Pose, side: str = "r") -> Tuple[bool, Tuple[float, float, float]]:
    """
    按**关节名**取手腕世界坐标，判定其是否进入柜门开口区域（用于导出逐帧真值）。

    柜门开口：x 在柜宽范围内、y 在该门高度范围内、z 越过门面（> -CAB_D/2）。
    注意：不可依赖 build_human 输出列表的下标——手部件的位置会随手持餐盒数量变化。
    """
    J = K.human_joints(pose)
    w = J["wrist_" + side]
    c = np.array([0.0, -K.SEG["fore_arm"] - 0.02, 0.0, 1.0]) @ w
    x, y, z = float(c[0]), float(c[1]), float(c[2])
    inside = (abs(x) < K.CAB_W / 2 and -K.CAB_D / 2 - 0.08 < z < K.CAB_D / 2
              and 0.30 < y < K.CAB_H)
    return bool(inside), (x, y, z)


def main() -> int:
    ap = argparse.ArgumentParser(description="用三维渲染管线生成自检/演示视频")
    ap.add_argument("--out", default=str(C.BASE_DIR / "data" / "selftest"))
    ap.add_argument("--name", default="selftest_scene.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ssaa", type=int, default=SSAA)
    ap.add_argument("--label-out", default=str(C.BASE_DIR / "data" / "selftest" / "selftest_truth.csv"))
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / a.name

    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    light = S.Light()
    floor = S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0))
    walls = [S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                          base_color=(0.80, 0.81, 0.83))]

    tl = make_timeline()
    door = make_door_track()
    slots = K.locker_package_slots()

    n_frames = int(round(a.seconds * a.fps))
    vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    # 与行为数据集统一启用程序化纹理，保证网页演示素材与数据集观感一致
    renderer = S.Renderer(OUT_W, OUT_H, ssaa=a.ssaa,
                          textures=TX.Textures(seed=20260925))
    truth: List[dict] = []
    t_all0 = time.perf_counter()
    for i in range(n_frames):
        t = i / float(a.fps)
        pkgs_taken = tl["pkg"].at(t) > 0.5
        shelf = [p for j, p in enumerate(slots) if not (pkgs_taken and j == TARGET_DOOR)]
        pose = build_pose(tl, t)
        items = render_frame(cam, light, floor, walls, door.at(t), pose, shelf)

        if i == 0:      # 首帧构建静态背景缓存（后续帧复用）
            renderer.render(items, cam, light, floor=floor, walls=walls,
                            shadow_occluders=[(m, mt) for m, mt, _ in items])
        img = renderer.render(items, cam, light, floor=floor, walls=walls,
                              shadow_occluders=[(m, mt) for m, mt, _ in items])
        vw.write(img)

        inside, wrist = hand_in_cabinet(pose)
        row = {
            "frame": i, "t": round(t, 3),
            "door_angle_deg": round(door.at(t), 1),
            "person_x": round(pose.x, 3), "person_z": round(pose.z, 3),
            "hand_in_cabinet": int(inside),
            "wrist_x": round(wrist[0], 3), "wrist_y": round(wrist[1], 3), "wrist_z": round(wrist[2], 3),
            "packages_in_hand": len(pose.packages_in_hand),
        }
        # 逐柜格真值（供视觉检测器评估：柜门是否开、柜内餐盒数）
        for kk in range(K.DOOR_ROWS * K.DOOR_COLS):
            row["door_open_%d" % kk] = int(door.at(t) >= 12.0) if kk == TARGET_DOOR else 0
            row["pkg_truth_%d" % kk] = sum(1 for j, _p in enumerate(slots)
                                           if j == kk and not (pkgs_taken and j == TARGET_DOOR))
        truth.append(row)
        if (i + 1) % max(1, n_frames // 6) == 0:
            print("  渲染进度 %3d/%d 帧  (%.1fs)" % (i + 1, n_frames, time.perf_counter() - t_all0))
    vw.release()

    # 逐帧真值导出（阶段2 数据集标注的原型）
    import csv
    lp = Path(a.label_out)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0].keys()))
        w.writeheader()
        w.writerows(truth)

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fr = cap.read()[1]
    cap.release()
    h, w = (fr.shape[:2] if fr is not None else (0, 0))
    print("\n已生成：%s" % video_path)
    print("  分辨率 %dx%d  帧率 %.1f  帧数 %d  时长 %.1fs" % (w, h, fps, n, n / fps if fps else 0))
    print("  逐帧真值：%s（%d 行）" % (lp, len(truth)))
    print("  总耗时 %.1fs（%.0f ms/帧）" % (time.perf_counter() - t_all0,
                                        (time.perf_counter() - t_all0) / max(1, n_frames) * 1000))
    print("  真值统计：柜门开启帧数 %d，手在柜内帧数 %d，手持餐盒帧数 %d"
          % (sum(1 for r in truth if r["door_angle_deg"] > 5),
             sum(1 for r in truth if r["hand_in_cabinet"]),
             sum(1 for r in truth if r["packages_in_hand"] > 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
