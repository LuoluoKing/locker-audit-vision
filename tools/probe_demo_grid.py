# -*- coding: utf-8 -*-
"""
tools/probe_demo_grid.py — 用「黄色门缝」自动探测演示素材柜体的格线位置（标定辅助）

思路
----
AI 演示素材里的黄色柜体，门与门之间是**明亮的黄色边框/门缝**，与白色门板、
金色编号牌在色相上可分离。对画面按行/列统计「黄色像素占比」，剖面的峰值
即对应格线（水平缝 = 行间缝，竖直缝 = 列间缝）。把探测到的缝线位置打印出来，
用于给 `calib_demo_locker.py` 的四角提供锚点（替代纯目视读点）。

注意：本工具只做**探测与打印**，不改任何标定值；最终标定仍以叠加对照图人工核对为准。

用法
----
    python tools/probe_demo_grid.py --clip 异常取餐 --frame 60
    python tools/probe_demo_grid.py --clip 异常取餐 --frame 60 --xrange 0.42 0.58 --yrange 0.30 0.72
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


def yellow_mask(bgr: np.ndarray) -> np.ndarray:
    """黄色掩膜：H 18~40（OpenCV 0-180），S >= 120，V >= 120。AI 画面偏亮，阈值取宽松些。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (18, 120, 120), (40, 255, 255))


def _peaks(profile: np.ndarray, min_gap: int = 8, min_val: float = 0.15, top: int = 12):
    """找剖面的局部强峰：先在窗口内取最大，抑制邻近重复。返回 [(位置, 值)] 降序。"""
    p = profile.astype(np.float32).copy()
    cand = []
    order = np.argsort(-p)
    taken = np.zeros(len(p), bool)
    for idx in order:
        if p[idx] < min_val:
            break
        if taken[max(0, idx - min_gap): idx + min_gap].any():
            continue
        taken[idx] = True
        # 3 点加权质心，取亚像素
        a, b, c = p[max(0, idx - 1)], p[idx], p[min(len(p) - 1, idx + 1)]
        wsum = a + b + c
        pos = idx + (c - a) / wsum if wsum > 0 else float(idx)
        cand.append((float(pos), float(b)))
        if len(cand) >= top:
            break
    return sorted(cand)


def main() -> int:
    ap = argparse.ArgumentParser(description="演示素材格线自动探测")
    ap.add_argument("--clip", required=True, help="素材名（不含 .mp4）")
    ap.add_argument("--frame", type=int, default=60)
    ap.add_argument("--xrange", type=float, nargs=2, default=None,
                    help="统计水平缝时限制的 x 归一化范围（默认自动取黄色主体左右界）")
    ap.add_argument("--yrange", type=float, nargs=2, default=None,
                    help="统计竖直缝时限制的 y 归一化范围")
    a = ap.parse_args()

    src = C.REAL_DEMO_DIR / (a.clip + ".mp4")
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读取失败：%s" % src)
        return 2
    h, w = frame.shape[:2]
    mask = yellow_mask(frame)

    # ---- 自动取「柜体门面区」的粗略范围：黄色掩膜的列/行投影，取中段稳定区间
    colsum = (mask > 0).sum(axis=0).astype(np.float32) / h
    rowsum = (mask > 0).sum(axis=1).astype(np.float32) / w
    # 柜体门面区（排除顶部雨棚与地面黄黑条）：先找到行投影中连续的强区
    strong_rows = np.nonzero(rowsum > 0.25)[0]
    if len(strong_rows) < 10:
        print("黄色像素过少（%.1f%%），可能选帧不佳" % (100 * (mask > 0).mean()))
        return 1
    y_lo, y_hi = int(strong_rows[0]), int(strong_rows[-1])
    strong_cols = np.nonzero((mask > 0).sum(axis=0) > 0.02 * h)[0]
    if len(strong_cols) < 10:
        print("黄色列范围异常")
        return 1
    x_lo, x_hi = int(strong_cols[0]), int(strong_cols[-1])

    # 收缩到门面区内部，避开雨棚（顶部）与地面警示条（底部）
    y0 = y_lo + int(0.05 * (y_hi - y_lo))
    y1 = y_hi - int(0.05 * (y_hi - y_lo))

    xs = a.xrange if a.xrange else (0.30, 0.58)      # 默认取中右列（多数帧不被人物遮挡）
    x0p, x1p = int(xs[0] * w), int(xs[1] * w)

    # ---- 水平缝剖面：在 [x0p,x1p] 内逐行统计黄色占比
    sub = mask[y0:y1, x0p:x1p]
    hprof = (sub > 0).mean(axis=1)
    hp = [(p + y0, v) for p, v in _peaks(hprof, min_gap=10, min_val=0.12, top=16)]

    ys = a.yrange if a.yrange else (0.34, 0.70)
    y0p, y1p = int(ys[0] * h), int(ys[1] * h)
    sub2 = mask[y0p:y1p, :]
    vprof = (sub2 > 0).mean(axis=0)
    vp = [(p, v) for p, v in _peaks(vprof, min_gap=12, min_val=0.12, top=12)]

    print("=" * 78)
    print("%s  f=%d  画幅=%dx%d" % (a.clip, a.frame, w, h))
    print("黄色像素占比 = %.1f%%" % (100 * (mask > 0).mean()))
    print("黄色行范围 y=%d..%d（归一化 %.3f..%.3f）" % (y_lo, y_hi, y_lo / h, y_hi / h))
    print("黄色列范围 x=%d..%d（归一化 %.3f..%.3f）" % (x_lo, x_hi, x_lo / w, x_hi / w))
    print("-" * 78)
    print("水平缝候选（y_px, 黄色占比）：在 x %d..%d 内统计" % (x0p, x1p))
    for p, v in hp:
        print("   y=%7.1f  (%.4f)  %.2f" % (p, p / h, v))
    print("竖直缝候选（x_px, 黄色占比）：在 y %d..%d 内统计" % (y0p, y1p))
    for p, v in vp:
        print("   x=%7.1f  (%.4f)  %.2f" % (p, p / w, v))

    # ---- 调试叠加图
    vis = frame.copy()
    for p, _v in hp:
        cv2.line(vis, (0, int(p)), (w - 1, int(p)), (0, 0, 255), 1)
    for p, _v in vp:
        cv2.line(vis, (int(p), 0), (int(p), h - 1), (0, 200, 0), 1)
    cv2.rectangle(vis, (x0p, y0), (x1p, y1), (255, 128, 0), 1)
    cv2.rectangle(vis, (x_lo, y0p), (x_hi, y1p), (128, 0, 255), 1)
    dst = OUT / ("probe_%s_f%d.png" % (a.clip, a.frame))
    cv2.imencode(".png", vis)[1].tofile(str(dst))
    print("调试图：%s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
