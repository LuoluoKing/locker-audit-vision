# -*- coding: utf-8 -*-
"""
tools/exp_render_variants.py — 渲染变体 × YOLO 置信度对照实验

目的
----
行为数据集评估暴露两类检测失败：
  1) 合成人形在部分片段置信度跌到 0.16~0.24（阈值 0.35 之下）→ 人物漏检；
  2) 敞开的柜门板被 YOLO 误检为「人」→ 柜门状态被人员遮挡逻辑判为未知，
     DOOR_LEFT_OPEN 漏检，并派生其他误报。
本实验对同一机位、同一姿态渲染多种视觉变体，用真实 YOLOv8n 量化每种变体的
「人形置信度」与「门板幻影置信度」，为渲染改进提供数据依据。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

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

RGB = Tuple[float, float, float]


def outline_variants(human_items, grow: float = 1.06, color: RGB = (0.05, 0.05, 0.06)):
    """给人形套一层深色描边：每个部件先画一个放大的深色底层，再画本体。"""
    out = []
    for mesh, m, col in human_items:
        grown = m.copy()
        grown[:3, :3] *= grow
        out.append((mesh, grown, color))
    return out


def render_variant(name: str, shirt: RGB, use_outline: bool, door_style: str,
                   out_dir: Path, px: float = 0.16, pz: float = -0.72) -> None:
    doors = [0.0] * 6
    doors[2] = 86.0
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    light = S.Light()
    floor = S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0))
    walls = [S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                         base_color=(0.80, 0.81, 0.83))]
    tex = TX.Textures(seed=20260925)

    if door_style == "dark":
        K.C_DOOR = (0.38, 0.42, 0.50)          # 深板岩色门板
        K.C_DOOR_OPEN_SIDE = (0.30, 0.33, 0.40)
    else:
        K.C_DOOR = (0.72, 0.74, 0.78)
        K.C_DOOR_OPEN_SIDE = (0.60, 0.62, 0.66)

    items = K.build_cabinet(doors)
    pose = K.Pose(x=px, z=pz, facing=0.0, lean=4.0, shirt=shirt)
    humans = K.build_human(pose)
    if use_outline:
        items += outline_variants(humans)
    items += humans
    items += K.build_packages([(x, y, z, c) for (x, y, z, c), k in
                               zip(_slots(), range(6)) if k != 2])

    renderer = S.Renderer(640, 480, ssaa=2, textures=tex)
    occ = [(m, mt) for m, mt, _ in items]
    img = renderer.render(items, cam, light, floor=floor, walls=walls, shadow_occluders=occ)
    p = out_dir / ("v_%s.png" % name)
    cv2.imencode(".png", img)[1].tofile(str(p))


def _slots():
    import scene_kit as KK
    return KK.shelf_package_slots()


def main() -> int:
    out_dir = Path(C.BASE_DIR / "logs" / "render_variants")
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("v0_baseline",   K.C_SHIRT,               False, "light"),
        ("v1_redshirt",   (0.72, 0.13, 0.13),      False, "light"),
        ("v2_outline",    K.C_SHIRT,               True,  "light"),
        ("v3_darkdoor",   K.C_SHIRT,               False, "dark"),
        ("v4_outline_darkdoor", K.C_SHIRT,         True,  "dark"),
        ("v5_red_outline_darkdoor", (0.72, 0.13, 0.13), True, "dark"),
    ]
    for name, shirt, ol, ds in variants:
        render_variant(name, shirt, ol, ds, out_dir)

    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                         imgsz=480, conf_threshold=0.05)
    print("%-24s %-30s %-30s" % ("变体", "人形检出(conf/框)", "最高非人框(conf/位置)"))
    for name, *_ in variants:
        img = cv2.imdecode(np.fromfile(str(out_dir / ("v_%s.png" % name)), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        dets = det.detect(img)
        confs = sorted((round(d.conf, 3) for d in dets), reverse=True)
        boxes = [(d.label, round(d.conf, 2), tuple(int(v) for v in d.xyxy())) for d in dets]
        print("%-24s conf=%-22s %s" % (name, confs[:3], boxes[:2]))
    print("变体图已保存：%s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
