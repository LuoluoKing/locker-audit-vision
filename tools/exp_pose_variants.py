# -*- coding: utf-8 -*-
"""
tools/exp_pose_variants.py — 姿态变体 × 门板幻影 × YOLO 置信度实验

两个待解问题（数据集评估暴露）：
  A. 手臂笔直下垂的站姿人形置信度跌到 0.16~0.24（阈值 0.35 之下）；
  B. 敞开的柜门板单独入画时被 YOLO 误检为 person（幻影人），
     导致柜门状态被「人员遮挡」逻辑压成未知 → DOOR_LEFT_OPEN 漏检。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from gen_selftest_clip3d import CAM_EYE, CAM_FOV, CAM_TARGET  # noqa: E402
from detector import build_detector  # noqa: E402

OUT = Path(C.BASE_DIR / "logs" / "pose_variants")


def scene(doors, humans, packages=True):
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    light = S.Light()
    floor = S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0))
    walls = [S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                         base_color=(0.80, 0.81, 0.83))]
    items = K.build_cabinet(doors)
    items += humans
    if packages:
        items += K.build_packages([(x, y, z, c) for k, (x, y, z, c)
                                   in enumerate(K.shelf_package_slots()) if k != 2])
    r = S.Renderer(640, 480, ssaa=2, textures=TX.Textures(seed=20260925))
    return r.render(items, cam, light, floor=floor, walls=walls,
                    shadow_occluders=[(m, mt) for m, mt, _ in items])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                         imgsz=480, conf_threshold=0.05)

    print("== A. 站姿变体（柜门 2 开启 86°，人站在取餐位）==")
    poses = {
        "P0_双臂笔直下垂":  dict(shoulder_l=0.0, shoulder_r=0.0, elbow_l=0.0, elbow_r=0.0),
        "P1_微前摆":       dict(shoulder_l=6.0, shoulder_r=-14.0, elbow_l=-6.0, elbow_r=-6.0),
        "P2_P1加分腿":     dict(shoulder_l=6.0, shoulder_r=-14.0, elbow_l=-6.0, elbow_r=-6.0,
                                stride=0.07),
        "P3_P1加屈肘":     dict(shoulder_l=6.0, shoulder_r=-14.0, elbow_l=-22.0, elbow_r=-22.0),
        "P4_分腿加屈肘":   dict(shoulder_l=8.0, shoulder_r=-16.0, elbow_l=-24.0, elbow_r=-24.0,
                                stride=0.07),
        "P5_行走中间态":   dict(shoulder_l=18.0, shoulder_r=-18.0, elbow_l=-12.0, elbow_r=-12.0,
                                hip_l=16.0, hip_r=-14.0, knee_l=6.0, knee_r=18.0, stride=0.05),
    }
    doors = [0.0] * 6
    doors[2] = 86.0
    for name, kw in poses.items():
        pose = K.Pose(x=0.16, z=-0.72, facing=0.0, lean=4.0, **kw)
        img = scene(doors, K.build_human(pose))
        dets = det.detect(img)
        confs = [(d.label, round(d.conf, 3)) for d in dets]
        cv2.imencode(".png", img)[1].tofile(str(OUT / ("a_%s.png" % name)))
        print("  %-14s %s" % (name, confs[:3]))

    print("== B. 门板幻影（无人，柜门 2 开启 86°）==")
    door_styles = {
        "D0_现行浅灰门板":  ((0.72, 0.74, 0.78), (0.60, 0.62, 0.66)),
        "D1_深板岩门板":    ((0.36, 0.40, 0.48), (0.28, 0.31, 0.38)),
        "D2_蓝灰门板":      ((0.42, 0.50, 0.62), (0.34, 0.41, 0.52)),
    }
    for name, (cd, co) in door_styles.items():
        K.C_DOOR, K.C_DOOR_OPEN_SIDE = cd, co
        img = scene(doors, [])
        dets = det.detect(img)
        confs = [(d.label, round(d.conf, 3)) for d in dets]
        cv2.imencode(".png", img)[1].tofile(str(OUT / ("b_%s.png" % name)))
        print("  %-16s %s" % (name, confs[:3]))
    K.C_DOOR, K.C_DOOR_OPEN_SIDE = (0.72, 0.74, 0.78), (0.60, 0.62, 0.66)
    print("变体图已保存：%s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
