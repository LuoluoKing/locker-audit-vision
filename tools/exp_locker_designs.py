# -*- coding: utf-8 -*-
"""
tools/exp_locker_designs.py — 取餐柜外观与格口方案对照渲染（供人工选定）

背景
----
现行渲染件（`scene_kit.build_cabinet`）是 3 行 × 2 列共 6 格口的浅灰柜体，
形态更像办公文件柜，与校园真实「智能取餐柜」（深蓝钢制、多格口、右侧操作屏、
带底座与顶部品牌条）差距明显。本脚本渲染 4 个方案的同机位静帧，
用于**人工看图选定**后再批量生成数据集。

方案
----
* D0 现行柜体（3×2，浅灰）—— 对照组
* D1 真实取餐柜外观（3×2，深蓝钢制 + 操作屏 + 底座 + 品牌条）
* D2 真实取餐柜外观 + 2 列 × 5 行 = 10 格口
* D3 真实取餐柜外观 + 3 列 × 6 行 = 18 格口

命令行
------
    python tools/exp_locker_designs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from gen_selftest_clip3d import CAM_EYE, CAM_FOV, CAM_TARGET  # noqa: E402

OUT = C.BASE_DIR / "logs" / "locker_designs"


def _scene_common():
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    light = S.Light()
    floor = S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0))
    walls = [S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                         base_color=(0.80, 0.81, 0.83))]
    return cam, light, floor, walls


def render_design(name: str, rows: int, cols: int, realistic: bool,
                  open_index: int = None, px: float = -0.18, pz: float = -1.05) -> Path:
    cam, light, floor, walls = _scene_common()
    tex = TX.Textures(seed=20260925)
    door_angles = [0.0] * (rows * cols)
    if open_index is not None and 0 <= open_index < len(door_angles):
        door_angles[open_index] = 86.0

    if realistic:
        items = K.build_locker(door_angles, rows=rows, cols=cols)
        slots = K.locker_package_slots(rows, cols)
    else:
        items = K.build_cabinet(door_angles)          # 仅支持 3×2（现行件）
        slots = K.shelf_package_slots()

    # 餐盒：只摆在「未被取走」的格口里（除被打开的那格）
    pkgs = [(x, y, z, c) for (x, y, z, c) in slots if open_index is None or c != open_index]
    items += K.build_packages(pkgs)

    # 取餐者：站在柜前、右臂前伸（便于判断人与柜体的尺度关系）
    pose = K.Pose(x=px, z=pz, facing=0.0, lean=6.0,
                  shoulder_r=52.0, elbow_r=-34.0, shoulder_l=8.0)
    items += K.build_human(pose)

    renderer = S.Renderer(640, 480, ssaa=2, textures=tex)
    occ = [(m, mt) for m, mt, _ in items]
    img = renderer.render(items, cam, light, floor=floor, walls=walls, shadow_occluders=occ)
    p = OUT / ("locker_%s.png" % name)
    cv2.imencode(".png", img)[1].tofile(str(p))
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    designs = [
        ("D0_现行3x2浅灰",  3, 2, False, 2),
        ("D1_真实外观3x2",  3, 2, True,  2),
        ("D2_真实外观2x5",  5, 2, True,  7),
        ("D3_真实外观3x6",  6, 3, True, 13),
    ]
    for name, rows, cols, real, oi in designs:
        p = render_design(name, rows, cols, real, oi)
        geo = K.locker_geometry(rows, cols) if real else None
        extra = ("格口 %.0f×%.0f cm，共 %d 格" % (geo["door_w"] * 100, geo["door_h"] * 100, rows * cols)
                 if geo else "格口 6 格（现行）")
        print("已渲染 %-18s %s  ->  %s" % (name, extra, p))
    print("输出目录：%s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())