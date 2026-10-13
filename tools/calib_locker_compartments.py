# -*- coding: utf-8 -*-
"""
tools/calib_locker_compartments.py — 按相机投影重算柜格格口多边形（换柜体后必须重标定）

为什么需要
----------
`code/config.py` 的 `CABINET_COMPARTMENTS` 是「6 个格口门面在关闭状态下的画面四边形」，
用于「柜门状态判定」与「取走柜内餐盒」判定。该坐标是由机位投影反算得到的，
**柜体几何一变（高度、底座、正面 z、格口位置）就必须重算**，否则格口区域会与
真实门面错位，柜门状态估计与「取走被标记柜格外卖」判定都会失准。

本工具做三件事
--------------
1. 用与渲染完全相同的机位，把新的格口门面四角投影到像素并归一化，打印可直接粘贴进
   `config.CABINET_COMPARTMENTS` 的坐标；
2. 把投影结果画在实测渲染帧上，输出对照图（人工核对格口是否贴合门面）；
3. 校验取餐站位脚点是否仍落在 `config.ROI_PICKUP` 内，并输出其归一化坐标。

命令行
------
    python tools/calib_locker_compartments.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from gen_selftest_clip3d import CAM_EYE, CAM_FOV, CAM_TARGET  # noqa: E402

W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
OUT = C.BASE_DIR / "logs" / "locker_designs"


def project(cam: S.Camera, pts: np.ndarray) -> np.ndarray:
    """世界坐标 (N,3) -> 像素坐标 (N,2)。行向量约定：d = X - eye，按相机基投影。

    与 scene3d.Camera.ray_dirs 的推导同源（后者是像素 -> 世界的反方向），
    不再走投影矩阵，避免与渲染器的行/列向量约定不一致（历史上曾因此多转置一次）。
    """
    eye = np.asarray(cam.eye, dtype=np.float64)
    tgt = np.asarray(cam.target, dtype=np.float64)
    up = np.asarray(getattr(cam, "up", (0.0, 1.0, 0.0)), dtype=np.float64)
    f = tgt - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = pts - eye
    x = d @ r
    y = d @ u
    z = d @ f
    aspect = W / float(H)
    tan_half = np.tan(np.radians(cam.fov_y_deg) / 2.0)
    ndc_x = (x / z) / (tan_half * aspect)
    ndc_y = (y / z) / tan_half
    px = (ndc_x + 1.0) / 2.0 * W
    py = (1.0 - ndc_y) / 2.0 * H
    return np.stack([px, py], axis=1)


def door_quad(geo: dict, r: int, c: int, inset: float = 0.06) -> np.ndarray:
    """
    第 (r,c) 扇门在**关闭**状态下的门面四角（世界坐标）。

    统一委托 ``scene_kit.cabinet_door_quad``（柜门几何的单一来源）。
    注意：**不要**用 ``locker_geometry`` 自行推算 —— `build_locker` 为保证
    「柜内暗区」判据成立，沿用的是 `build_cabinet` 的门面几何；
    曾因本函数按「加高加深版」柜体推算，导致标定出的格口与真实门面错位。
    """
    return np.asarray(K.cabinet_door_quad(r, c, inset), dtype=np.float64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cols = K.DOOR_ROWS, K.DOOR_COLS
    geo = K.locker_geometry(rows, cols)
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)

    print("=" * 78)
    print("柜格格口多边形重标定（柜体：build_locker，%d 行 × %d 列，共 %d 格）"
          % (rows, cols, rows * cols))
    print("格口尺寸 %.1f × %.1f cm；柜体 %.2f(W) × %.2f(H) × %.2f(D) m，底座 %.2f m，屏幕列 %.2f m"
          % (geo["door_w"] * 100, geo["door_h"] * 100, K.LOCKER_W, K.LOCKER_H, K.LOCKER_D,
             K.LOCKER_BASE_H, K.SCREEN_COL_W))
    print("=" * 78)

    polys_norm: List[List[List[float]]] = []
    for r in range(rows):
        for c in range(cols):
            px = project(cam, door_quad(geo, r, c))
            norm = [[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in px]
            polys_norm.append(norm)
            print("    格口 %d（行%d 列%d）: %s" % (r * cols + c, r, c, norm))

    print("\n---- 可直接粘贴进 code/config.py 的 CABINET_COMPARTMENTS ----")
    print("CABINET_COMPARTMENTS = [")
    names = []
    for i, p in enumerate(polys_norm):
        r, c = divmod(i, cols)
        names.append("    # 柜格 %d（%s排%s）" % (i, {0: "上", 1: "中", 2: "下"}.get(r, str(r)),
                                              "左" if c == 0 else "右"))
        print("%s\n    [%s]," % (names[-1], ", ".join("[%.4f, %.4f]" % (a, b) for a, b in p)))
    print("]")

    # ---- 取餐站位脚点校验（站位未变，但柜体变高变深，需确认 ROI 仍成立）
    foot = np.array([[0.16, 0.0, -0.72]], dtype=np.float64)   # PICKUP_X, 地面, PICKUP_Z
    pxf = project(cam, foot)[0]
    roi = np.array([[int(x * W), int(y * H)] for x, y in C.ROI_PICKUP], np.int32)
    inside = cv2.pointPolygonTest(roi, (float(pxf[0]), float(pxf[1])), False) >= 0
    print("\n取餐站位脚点世界 (0.16, 0, -0.72) -> 像素 (%.0f, %.0f)，归一化 (%.4f, %.4f)"
          % (pxf[0], pxf[1], pxf[0] / W, pxf[1] / H))
    print("  是否仍落在现行 ROI_PICKUP 内：%s" % ("是 ✓（ROI 无需改动）" if inside else "否 ✗（必须重标定 ROI）"))

    # ---- 实测渲染帧 + 投影叠加对照图
    blank_doors = [0.0] * (rows * cols)
    items = K.build_locker(blank_doors)
    items += K.build_packages(K.locker_package_slots(rows, cols))
    items += K.build_human(K.Pose(x=0.16, z=-0.72, shoulder_l=14.0))
    renderer = S.Renderer(W, H, ssaa=2, textures=TX.Textures(seed=20260925))
    occ = [(m, mt) for m, mt, _ in items]
    img = renderer.render(items, cam, S.Light(),
                          floor=S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0)),
                          walls=[S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0,
                                             v_lo=-0.6, v_hi=5.2, base_color=(0.80, 0.81, 0.83))],
                          shadow_occluders=occ)
    for i, p in enumerate(polys_norm):
        pts = np.array([[int(x * W), int(y * H)] for x, y in p], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cxm, cym = pts.mean(axis=0).astype(int)
        cv2.putText(img, "C%d" % i, (cxm - 10, cym + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [roi], True, (255, 0, 0), 2)
    p = OUT / "calib_compartments_overlay.png"
    cv2.imencode(".png", img)[1].tofile(str(p))
    print("对照图已保存：%s（红框=柜格多边形，蓝框=取餐区 ROI，需人工核对是否贴合门面）" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())