# -*- coding: utf-8 -*-
"""
tools/calib_demo_locker.py — 为「AI 生成演示素材」标定取餐柜格口多边形

背景
----
`config.CABINET_COMPARTMENTS` 是为三维合成场景（6 格深蓝柜体、640x480 横屏）标定的，
与 `real_video/` 演示素材（黄色柜体，3 列 x 7 行，720x1280 竖屏）**完全不同**，
直接套用会让格口落在柜体之外或跨到相邻门板上，柜门状态判定与「取走标记柜格外卖」判定全部失准。

本工具的做法
------------
人工在参考帧上给出「门面矩形」的四个角点（TL/TR/BR/BL，归一化坐标），
再用**单应变换**（cv2.getPerspectiveTransform）把单位方格映射到该四边形，
按 `rows x cols` 等分，得到每扇门的四边形（含 inset 收缩，避免压到门缝与边框）。

单应而非双线性插值：柜体为等距网格，透视下近端格口宽、远端窄，
只有投影变换才能正确反映这种非均匀间距（实测三列间距 0.17 / 0.20 不等）。

本工具输出
----------
1. 叠加对照图（`logs/real_check/calib_<素材>.png`）——红网格=格口、黄字=格号，
   用于人工核对是否贴合门板；
2. 可直接粘贴进 `code/config.py` 的坐标；
3. 参考帧上逐格口的**暗比**（与 `detector.DoorStateEstimator` 同一判据），
   用于确认「关闭的门板」暗比确实低于开门阈值（黄色柜体的门板若本身偏暗则需重新选点）。

命令行
------
    python tools/calib_demo_locker.py                # 全部素材
    python tools/calib_demo_locker.py 未关柜门        # 单个素材（按名字前缀匹配）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402
from detector import DoorStateEstimator  # noqa: E402

OUT = C.BASE_DIR / "logs" / "real_check"

# 门面矩形四角（归一化，TL / TR / BR / BL）——人工目视标定，可用叠加图迭代微调
# rows/cols 为柜体网格；skip 为「不是柜门」的格子（中间列第 2 行是操作屏）
PROFILES: Dict[str, dict] = {
    "未关柜门": {
        "clip": "未关柜门.mp4", "frame": 226, "rows": 7, "cols": 3, "skip": [(1, 1)],
        "corners": [[0.268, 0.292], [0.792, 0.278], [0.792, 0.958], [0.268, 0.972]],
    },
    "异常取餐": {
        "clip": "异常取餐.mp4", "frame": 180, "rows": 7, "cols": 3, "skip": [(1, 1)],
        "corners": [[0.286, 0.300], [0.745, 0.300], [0.745, 0.930], [0.286, 0.930]],
    },
    "正常取餐": {
        "clip": "正常取餐.mp4", "frame": 180, "rows": 7, "cols": 3, "skip": [(1, 1)],
        "corners": [[0.246, 0.248], [0.812, 0.248], [0.812, 0.968], [0.246, 0.968]],
    },
}

INSET = 0.10          # 每格向中心收缩比例，避免压到门缝、边框与铰链

_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def _homography(corners) -> np.ndarray:
    return cv2.getPerspectiveTransform(_UNIT, np.array(corners, np.float32))


def _sample(gray: np.ndarray, Hm: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """把归一化 (u,v) 经单应映射到像素并取灰度（向量化，越界按边缘裁剪）。"""
    h, w = gray.shape[:2]
    pts = np.stack([us, vs], axis=1).astype(np.float32).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
    xs = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
    vs_out = gray[ys, xs]
    inside = (q[:, 0] >= 0) & (q[:, 0] <= w - 1) & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)
    return vs_out, inside


def grid_score(gray: np.ndarray, corners, rows: int, cols: int):
    """
    网格贴合度：门板中心亮度均值 − 内部格线亮度均值（越大越贴合）。

    依据：柜门门板受光偏亮，门缝/铰链/柜体阴影偏暗，因此「格线落在门缝上、
    格心落在门板中央」时该差值最大。只取**内部**格线——外边界落在柜体边框上，
    边框本身是亮黄色，纳入会干扰。
    """
    Hm = _homography(corners)
    ul = np.linspace(0.03, 0.97, 60)
    vl = np.linspace(0.02, 0.98, 150)
    seam_parts = []
    for c in range(1, cols):
        vals, _ = _sample(gray, Hm, np.full_like(ul, c / cols), ul)
        seam_parts.append(vals)
    for r in range(1, rows):
        vals, _ = _sample(gray, Hm, vl, np.full_like(vl, r / rows))
        seam_parts.append(vals)
    seam = float(np.concatenate(seam_parts).mean())
    cu = (np.arange(cols) + 0.5) / cols
    cvv = (np.arange(rows) + 0.5) / rows
    UU, VV = np.meshgrid(cu, cvv)
    cvals, _ = _sample(gray, Hm, UU.ravel(), VV.ravel())
    centers = float(cvals.mean())
    return centers - seam, centers, seam


def refine_corners(gray: np.ndarray, corners, rows: int, cols: int,
                   steps=(0.012, 0.006, 0.003, 0.0015)):
    """坐标下降：逐个角点坐标微调，使 grid_score 最大。返回 (新角点, 起始分, 结束分)。"""
    cur = [list(map(float, p)) for p in corners]
    s0 = grid_score(gray, cur, rows, cols)[0]
    for st in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for j in range(2):
                    base = grid_score(gray, cur, rows, cols)[0]
                    for d in (st, -st):
                        trial = [list(p) for p in cur]
                        trial[i][j] += d
                        # 防止角点跑出画面或四边形自交（保持合理的正序）
                        if not (0.0 <= trial[i][j] <= 1.0):
                            continue
                        sc = grid_score(gray, trial, rows, cols)[0]
                        if sc > base + 1e-6:
                            cur = trial
                            base = sc
                            improved = True
    return cur, s0, grid_score(gray, cur, rows, cols)[0]


def cell_polys(corners: List[List[float]], rows: int, cols: int, inset: float = INSET
               ) -> Tuple[List[List[List[float]]], List[Tuple[int, int]]]:
    """返回 (归一化格口多边形列表, 对应 (row, col) 列表)。"""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
    dst = np.array(corners, np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)

    def tf(u: float, v: float) -> Tuple[float, float]:
        p = np.array([[[u, v]]], np.float32)
        q = cv2.perspectiveTransform(p, Hm)[0, 0]
        return float(q[0]), float(q[1])

    polys, keys = [], []
    for r in range(rows):
        for c in range(cols):
            u0, u1 = c / cols, (c + 1) / cols
            v0, v1 = r / rows, (r + 1) / rows
            cu, cvv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
            k = inset
            pts = []
            for (u, v) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                uu = cu + (u - cu) * (1.0 - k)
                vv = cvv + (v - cvv) * (1.0 - k)
                x, y = tf(uu, vv)
                pts.append([round(x, 4), round(y, 4)])
            polys.append(pts)
            keys.append((r, c))
    return polys, keys


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    for tag, cfg in PROFILES.items():
        if want and not tag.startswith(want):
            continue
        src = C.REAL_DEMO_DIR / cfg["clip"]
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cfg["frame"]))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("!! 读取失败：%s" % src)
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        corners, s0, s1 = refine_corners(gray, cfg["corners"], cfg["rows"], cfg["cols"])
        print("角点微调：贴合度 %.1f -> %.1f" % (s0, s1))
        print("  refined corners = [[%s]]"
              % "], [".join(", ".join("%.4f" % v for v in p) for p in corners))
        polys, keys = cell_polys(corners, cfg["rows"], cfg["cols"])

        keep, keep_keys = [], []
        for p, k in zip(polys, keys):
            if k in cfg["skip"]:
                continue
            keep.append(p)
            keep_keys.append(k)

        px = [np.array([[int(round(x * w)), int(round(y * h))] for x, y in p], np.int32)
              for p in keep]
        est = DoorStateEstimator(px)
        angles = est.estimate(frame, [])
        med = float(np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        thr = 0.62 * med

        img = frame.copy()
        for i, (poly, k) in enumerate(zip(px, keep_keys)):
            ratio = est.last_dark_ratios[i]
            col = (0, 0, 255) if ratio >= 0.15 else (0, 200, 0)
            cv2.polylines(img, [poly], True, col, 2)
            cx, cy = poly.mean(axis=0).astype(int)
            cv2.putText(img, "%d:%.2f" % (i, ratio), (cx - 26, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        dst = OUT / ("calib_%s.png" % tag)
        cv2.imencode(".png", img)[1].tofile(str(dst))

        print("=" * 84)
        print("%s  f=%d  画幅=%dx%d  网格=%dx%d  有效格口=%d"
              % (cfg["clip"], cfg["frame"], w, h, cfg["rows"], cfg["cols"], len(keep)))
        print("中位亮度=%.1f 开门判据阈值=%.1f（暗比>=0.15 判为开）" % (med, thr))
        bad = [(i, keep_keys[i], est.last_dark_ratios[i]) for i in range(len(keep))
               if est.last_dark_ratios[i] >= 0.15]
        print("参考帧上被判为「开」的格口：%s"
              % ("无（全部闭合，标定可信）" if not bad else
                 ", ".join("C%d(行%d列%d)=%.2f" % (i, k[0], k[1], v) for i, k, v in bad)))
        print("对照图：%s" % dst)
        print("---- 可直接粘贴进 config.py ----")
        print('    "": {  # %s' % tag)
        print('        "roi_pickup": [...],  # 取餐区另行标定')
        print('        "compartments": [')
        for p, k in zip(keep, keep_keys):
            print("            [%s],   # C%d（行%d 列%d）"
                  % (", ".join("[%.4f, %.4f]" % (a, b) for a, b in p),
                     keep_keys.index(k), k[0], k[1]))
        print("        ],")
        print("    },")
    return 0


if __name__ == "__main__":
    sys.exit(main())