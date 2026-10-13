# -*- coding: utf-8 -*-
"""
tools/overlay_demo_grid.py — 演示素材格口网格叠加核对（标定迭代工具）

用途
----
给定「门面矩形四角 + 行/列数 + 跳过格」，把格口多边形画到参考帧上，
用于**目视核对与逐角微调**（每次只动一个自由度，对齐后再动下一个）。
同时打印每格的平均亮度与暗比，帮助判断格口是否落在门板上。

命令行
------
    python tools/overlay_demo_grid.py --clip 异常取餐 --frame 180 ^
        --corners "0.324,0.310 0.700,0.310 0.700,0.800 0.324,0.800" --inset 0.08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402

OUT = C.BASE_DIR / "logs" / "real_check"
_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def cell_polys(corners: List[List[float]], rows: int, cols: int, inset: float):
    Hm = cv2.getPerspectiveTransform(_UNIT, np.array(corners, np.float32))

    def tf(u, v):
        q = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), Hm)[0, 0]
        return float(q[0]), float(q[1])

    polys, keys = [], []
    for r in range(rows):
        for c in range(cols):
            u0, u1 = c / cols, (c + 1) / cols
            v0, v1 = r / rows, (r + 1) / rows
            cu, cvv = (u0 + u1) / 2, (v0 + v1) / 2
            pts = []
            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                uu = cu + (u - cu) * (1 - inset)
                vv = cvv + (v - cvv) * (1 - inset)
                x, y = tf(uu, vv)
                pts.append([x, y])
            polys.append(pts)
            keys.append((r, c))
    return polys, keys


def main() -> int:
    ap = argparse.ArgumentParser(description="演示素材格口叠加核对")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--corners", required=True,
                    help="TL TR BR BL 四组 x,y，空格分隔；也可只给 4 个数=轴对齐矩形 x0 y0 x1 y1")
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--skip", default="1,1", help="列,行 逗号分隔，可多个用分号")
    ap.add_argument("--inset", type=float, default=0.08)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # 解析角点
    parts = a.corners.replace(",", " ").split()
    if len(parts) == 4:
        x0, y0, x1, y1 = map(float, parts)
        corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    elif len(parts) == 8:
        corners = [[float(parts[i]), float(parts[i + 1])] for i in range(0, 8, 2)]
    else:
        print("corners 需要 4 或 8 个数")
        return 2

    skip = set()
    for seg in a.skip.split(";"):
        if seg.strip():
            c, r = seg.split(",")
            skip.add((int(r.strip()), int(c.strip())))

    src = C.REAL_DEMO_DIR / (a.clip + ".mp4")
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读取失败：%s" % src)
        return 2
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = float(np.median(gray))

    polys, keys = cell_polys(corners, a.rows, a.cols, a.inset)
    vis = frame.copy()
    idx = 0
    lines = []
    for poly, (r, c) in zip(polys, keys):
        px = np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)
        if (r, c) in skip:
            cv2.polylines(vis, [px], True, (255, 0, 255), 2)
            cx, cy = px.mean(axis=0).astype(int)
            cv2.putText(vis, "SKIP", (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1, cv2.LINE_AA)
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [px], 1)
        vals = gray[mask > 0]
        ratio = float((vals < 0.62 * med).mean()) if vals.size else 0.0
        col = (0, 0, 255) if ratio >= 0.15 else (0, 210, 0)
        cv2.polylines(vis, [px], True, col, 2)
        cx, cy = px.mean(axis=0).astype(int)
        cv2.putText(vis, "%d" % idx, (cx - 10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        lines.append((idx, r, c, ratio))
        idx += 1

    # 角点标记
    for name, (x, y), col in zip("ABCD", corners, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]):
        p = (int(round(x * w)), int(round(y * h)))
        cv2.drawMarker(vis, p, col, cv2.MARKER_CROSS, 24, 3)
        cv2.putText(vis, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

    dst = OUT / (a.out or ("ovl_%s_f%d.png" % (a.clip, a.frame)))
    cv2.imencode(".png", vis)[1].tofile(str(dst))
    print("=" * 78)
    print("%s f=%d 网格 %dx%d 有效格口 %d 已跳过 %d" %
          (a.clip, a.frame, a.rows, a.cols, len(lines), len(keys) - len(lines)))
    print("corners: %s" % " ".join("%.4f,%.4f" % (x, y) for x, y in corners))
    print("对照图：%s" % dst)
    opened = [(i, r, c, v) for i, r, c, v in lines if v >= 0.15]
    print("参考帧上判为「开」的格口：%s" % ("无" if not opened else
          ", ".join("#%d(行%d列%d)=%.2f" % t for t in opened)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
