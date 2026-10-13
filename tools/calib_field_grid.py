# -*- coding: utf-8 -*-
"""tools/calib_field_grid.py — 实拍画面里的柜体格口网格标定（四角插值法）

用于把「真实拍摄画面里的一台柜体」标定成逐格多边形（归一化坐标），供
DoorStateEstimator / 规则引擎在实拍素材上使用。

方法：操作者给出柜体面板的四个角点（左上/右上/右下/左下，按正视中的顺时针），
工具在该四边形内按双线性插值生成 R×C 网格，并渲染叠加图供视觉核对；
不满意则调外框/行列数/边距，直至网格与门缝对齐。

输出：
  <out.json>     {"corners": [...], "rows": R, "cols": C, "compartments": [[[x,y],...], ...]}
                 （归一化到 0~1，可直接喂给 RuleConfig.compartments）
  <out.png>      叠加核对图

用法（在本项目内）：
  python tools/calib_field_grid.py --image <帧.png> --corners "x1,y1;x2,y2;x3,y3;x4,y4" \
      --rows 5 --cols 5 --out logs/field_calib/v04_grid.json
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def bilinear(quad: np.ndarray, u: float, v: float):
    """四边形内双线性插值：quad = [TL, TR, BR, BL]，u,v ∈ [0,1]。"""
    tl, tr, br, bl = quad
    top = tl * (1 - u) + tr * u
    bot = bl * (1 - u) + br * u
    return top * (1 - v) + bot * v


def grid_polys(quad: np.ndarray, rows: int, cols: int,
               margin_u: float = 0.0, margin_v: float = 0.0):
    """在四边形内生成 rows×cols 个格口多边形（归一化）。"""
    polys = []
    for r in range(rows):
        for c in range(cols):
            u0 = c / cols + margin_u / cols
            u1 = (c + 1) / cols - margin_u / cols
            v0 = r / rows + margin_v / rows
            v1 = (r + 1) / rows - margin_v / rows
            pts = [bilinear(quad, u0, v0), bilinear(quad, u1, v0),
                   bilinear(quad, u1, v1), bilinear(quad, u0, v1)]
            polys.append([[float(x), float(y)] for x, y in pts])
    return polys


def parse_corners(s: str):
    pts = []
    for chunk in s.split(";"):
        x, y = chunk.split(",")
        pts.append([float(x), float(y)])
    if len(pts) != 4:
        raise ValueError("需要 4 个角点：x,y;x,y;x,y;x,y")
    return np.array(pts, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--corners", required=True,
                    help="TL;TR;BR;BL 四角，像素坐标 x,y;x,y;x,y;x,y")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--margin-u", type=float, default=6.0, help="水平内缩（格数单位）")
    ap.add_argument("--margin-v", type=float, default=6.0, help="垂直内缩（格数单位）")
    ap.add_argument("--out", required=True, help="输出 json 路径（同名 .png 为核对图）")
    a = ap.parse_args()

    img = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取图像：", a.image)
        return 2
    h, w = img.shape[:2]
    quad = parse_corners(a.corners)
    poly_px = grid_polys(quad, a.rows, a.cols, a.margin_u, a.margin_v)

    # 归一化
    comps = [[[round(x / w, 6), round(y / h, 6)] for x, y in q] for q in poly_px]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    data = {"image": os.path.basename(a.image), "width": w, "height": h,
            "corners": quad.tolist(), "rows": a.rows, "cols": a.cols,
            "margin_u": a.margin_u, "margin_v": a.margin_v,
            "compartments": comps}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

    vis = img.copy()
    for i, q in enumerate(poly_px):
        pts = np.array(q, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 1)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vis, str(i), (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (0, 0, 255), 1)
    cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 0, 0), 2)
    out_png = os.path.splitext(a.out)[0] + ".png"
    cv2.imencode(".png", vis)[1].tofile(out_png)
    print("格口数:", len(comps), " -> ", a.out)
    print("核对图:", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
