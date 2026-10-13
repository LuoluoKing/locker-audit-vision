# -*- coding: utf-8 -*-
"""
tools/fit_demo_grid.py — 演示素材格口网格的**边缘导向自动精配准**

为什么不用亮度法
----------------
旧版按「格心亮度 − 格线亮度」拟合，在黄色柜体上失效：金色编号牌本身偏暗、
阴影与反光都会产生亮度极小值，噪声把目标函数淹没了（实测只从 2.1 提到 2.5）。

本版改用**边缘导向**目标：
    score = 网格内部缝线上的边缘强度均值 − 格口内部边缘强度均值
缝线（门与门之间的暗缝）是强边缘带，而门板中心（含编号牌以外的区域）相对平坦。
横向缝只统计 Sobel-y，纵向缝只统计 Sobel-x，方向对齐可进一步降噪。

标定完成后用 `tools/overlay_demo_grid.py` 叠加目视核对。

命令行
------
    python tools/fit_demo_grid.py --clip 异常取餐 --frame 180 ^
        --corners "0.326,0.352 0.676,0.362 0.674,0.782 0.326,0.792" --rows 7 --cols 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402

OUT = C.BASE_DIR / "logs" / "real_check"
_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def _H(corners):
    return cv2.getPerspectiveTransform(_UNIT, np.array(corners, np.float32))


def _sample_bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,2) 像素坐标 → 双线性采样。"""
    x = np.clip(pts[:, 0], 0, img.shape[1] - 1.001)
    y = np.clip(pts[:, 1], 0, img.shape[0] - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy).ravel()


def _to_px(u, v, Hm, w, h):
    q = cv2.perspectiveTransform(np.stack([u, v], axis=1).astype(np.float32).reshape(-1, 1, 2), Hm)
    q = q.reshape(-1, 2)
    return np.stack([q[:, 0] * w, q[:, 1] * h], axis=1)


def grid_score(edge_x: np.ndarray, edge_y: np.ndarray, corners, rows: int, cols: int):
    """见模块 docstring。返回 (分数, 缝线边缘均值, 格内边缘均值)。"""
    h, w = edge_x.shape
    Hm = _H(corners)
    us = np.linspace(0.06, 0.94, 48)
    vs = np.linspace(0.04, 0.96, 80)

    seam_vals = []
    # 纵向缝（列间）：统计 Sobel-x
    for c in range(1, cols):
        pts = _to_px(np.full_like(us, c / cols), us, Hm, w, h)
        seam_vals.append(_sample_bilinear(edge_x, pts))
    # 横向缝（行间）：统计 Sobel-y
    for r in range(1, rows):
        pts = _to_px(vs, np.full_like(vs, r / rows), Hm, w, h)
        seam_vals.append(_sample_bilinear(edge_y, pts))
    seam = float(np.concatenate(seam_vals).mean())

    # 格内（含编号牌区域，取格心 60% 范围）：两方向都算，取平均
    in_parts = []
    for r in range(rows):
        for c in range(cols):
            cu, cv = (c + 0.5) / cols, (r + 0.5) / rows
            uu = np.linspace(cu - 0.30 / (cols * 1.0), cu + 0.30 / cols, 13)
            vv = np.linspace(cv - 0.30 / rows, cv + 0.30 / rows, 20)
            UU, VV = np.meshgrid(uu, vv)
            pts = _to_px(UU.ravel(), VV.ravel(), Hm, w, h)
            in_parts.append(_sample_bilinear(edge_x, pts))
            in_parts.append(_sample_bilinear(edge_y, pts))
    inner = float(np.concatenate(in_parts).mean())
    return seam - inner, seam, inner


def refine(edge_x, edge_y, corners, rows, cols, steps=(0.010, 0.005, 0.0025, 0.0012)):
    cur = [list(map(float, p)) for p in corners]
    s0 = grid_score(edge_x, edge_y, cur, rows, cols)[0]
    for st in steps:
        improved = True
        guard = 0
        while improved and guard < 60:
            improved = False; guard += 1
            for i in range(4):
                for j in range(2):
                    base = grid_score(edge_x, edge_y, cur, rows, cols)[0]
                    for d in (st, -st):
                        trial = [list(p) for p in cur]
                        trial[i][j] = min(0.999, max(0.001, trial[i][j] + d))
                        # 保持四边形不自交：TL.x<TR.x, BL.x<BR.x, 上边在上
                        if not (trial[0][0] < trial[1][0] and trial[3][0] < trial[2][0]
                                and trial[0][1] < trial[3][1] and trial[1][1] < trial[2][1]):
                            continue
                        sc = grid_score(edge_x, edge_y, trial, rows, cols)[0]
                        if sc > base + 1e-4:
                            cur = trial; improved = True
                            break
    return cur, s0, grid_score(edge_x, edge_y, cur, rows, cols)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="演示素材格口网格自动精配准（边缘导向）")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--corners", required=True, help="TL TR BR BL：8 个数（x,y 各 4 组）")
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()

    parts = a.corners.replace(",", " ").split()
    if len(parts) != 8:
        print("corners 需要 8 个数")
        return 2
    corners = [[float(parts[i]), float(parts[i + 1])] for i in range(0, 8, 2)]

    src = C.REAL_DEMO_DIR / (a.clip + ".mp4")
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读取失败")
        return 2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    ex = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    ey = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    cur, s0, s1 = refine(ex, ey, corners, a.rows, a.cols)
    print("=" * 78)
    print("%s f=%d  网格 %dx%d" % (a.clip, a.frame, a.rows, a.cols))
    print("边缘贴合分：%.1f -> %.1f" % (s0, s1))
    print("corners = [[%s]]" % "], [".join(", ".join("%.4f" % v for v in p) for p in cur))
    # 收敛后打印每条缝的位置（归一化，便于对照）
    Hm = _H(cur)
    for c in range(1, a.cols):
        pts = _to_px(np.array([c / a.cols]), np.array([0.5]), Hm, frame.shape[1], frame.shape[0])
        print("  列缝 %d: x≈%.4f" % (c, pts[0][0] / frame.shape[1]))
    for r in range(1, a.rows):
        pts = _to_px(np.array([0.5]), np.array([r / a.rows]), Hm, frame.shape[1], frame.shape[0])
        print("  行缝 %d: y≈%.4f" % (r, pts[0][1] / frame.shape[0]))
    print("提示：用 tools/overlay_demo_grid.py 叠加目视复核")
    return 0


if __name__ == "__main__":
    sys.exit(main())
