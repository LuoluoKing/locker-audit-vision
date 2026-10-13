# -*- coding: utf-8 -*-
"""tools/field_door_probe.py — 实拍场景的「柜门暗比」标定与探测（E8-D）

针对一台实拍取餐柜：
  1. 在若干参考帧上自动检测「深色门窗口」（自适应阈值 + 轮廓）
  2. 合并去重 → 生成候选格口矩形（窗口外扩成门面）
  3. 在整段视频上采样（默认每 0.5 s 一帧）计算每个格口的 dark_ratio 时间序列
  4. 输出统计（分位数）+ 若干超阈时刻的取证帧，供论文的现场柜门判定分析

注意：这只是**探测**，不做开门/关门断言——实拍素材无真值，结论以时间序列与
取证帧呈现，由人去核对。

输出：
  logs/field_calib/<key>_doors.json    格口列表 + 逐格口时间序列
  logs/field_calib/<key>_doors.png     格口叠加核对图
  logs/field_calib/<key>_spike_*.png   暗比异常升高的取证帧
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))


def detect_windows(frame, region, area_lo=150, area_hi=4500):
    x0, y0, x1, y1 = region
    roi = frame[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h
        if area_lo < a < area_hi and 0.45 < w / max(h, 1) < 2.4:
            boxes.append((x + x0, y + y0, w, h))
    return boxes


def merge_boxes(all_boxes, iou_th=0.35):
    """跨参考帧合并同一窗口的检测（按中心距聚类）。"""
    merged = []
    for b in all_boxes:
        bx = (b[0], b[1], b[0] + b[2], b[1] + b[3])
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        hit = False
        for m in merged:
            mx, my = m["cx"], m["cy"]
            if abs(cx - mx) < 0.55 * max(b[2], m["w"]) and abs(cy - my) < 0.55 * max(b[3], m["h"]):
                m["n"] += 1
                hit = True
                break
        if not hit:
            merged.append({"cx": cx, "cy": cy, "w": b[2], "h": b[3], "n": 1})
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--region", required=True, help="x0,y0,x1,y1")
    ap.add_argument("--refs", default="10,60,120,200", help="参考帧时刻（秒，逗号分隔）")
    ap.add_argument("--step-s", type=float, default=0.5, help="时间序列采样间隔")
    ap.add_argument("--expand", type=float, default=6.0, help="窗口外扩像素（生成门面）")
    ap.add_argument("--outdir", default=str(ROOT / "logs" / "field_calib"))
    a = ap.parse_args()

    region = [int(v) for v in a.region.split(",")]
    refs = [float(v) for v in a.refs.split(",")]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("打不开视频")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw = []
    for t in refs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        raw += detect_windows(fr, region)
    merged = merge_boxes(raw)
    print("合并后格口窗口数：", len(merged))

    # 窗口外扩 → 格口矩形
    comps = []
    for m in sorted(merged, key=lambda d: (round(d["cy"] / 24), d["cx"])):
        x1 = m["cx"] - m["w"] / 2 - a.expand
        y1 = m["cy"] - m["h"] / 2 - a.expand
        comps.append([x1, y1, x1 + m["w"] + 2 * a.expand, y1 + m["h"] + 2 * a.expand])

    outdir = Path(a.outdir)
    os.makedirs(outdir, exist_ok=True)

    # 叠加核对图（用 refs[0] 帧）
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(refs[0] * fps))
    ok, ref_frame = cap.read()
    vis = ref_frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(comps):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
        cv2.putText(vis, str(i), (int(x1) + 2, int(y1) + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1)
    cv2.imencode(".png", vis)[1].tofile(str(outdir / ("%s_doors.png" % a.key)))

    # 时间序列
    step = max(1, int(a.step_s * fps))
    series = {i: [] for i in range(len(comps))}
    times = []
    i = 0
    k = 0
    while i < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        thr = 0.62 * med
        ts = i / fps
        times.append(round(ts, 2))
        for ci, (x1, y1, x2, y2) in enumerate(comps):
            g = gray[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if g.size == 0:
                series[ci].append(None)
                continue
            ratio = float(np.count_nonzero(g < thr)) / float(g.size)
            series[ci].append(round(ratio, 4))
        k += 1
        if k % 100 == 0:
            print("  已处理 %.0fs" % ts)
        i += step
    cap.release()

    # 统计 + 找尖峰
    stats = {}
    for ci, vals in series.items():
        vv = np.array([v for v in vals if v is not None])
        if vv.size == 0:
            stats[ci] = None
            continue
        stats[ci] = {"median": round(float(np.median(vv)), 3),
                     "p90": round(float(np.percentile(vv, 90)), 3),
                     "p99": round(float(np.percentile(vv, 99)), 3),
                     "max": round(float(vv.max()), 3)}
    # 取证帧：任一格口暗比超过自身中位数 + 0.25 的时刻
    spikes = []
    for ti, t in enumerate(times):
        for ci, vals in series.items():
            v = vals[ti]
            md = stats[ci]["median"] if stats[ci] else None
            if v is not None and md is not None and v > md + 0.25:
                spikes.append((t, ci, v, md))
    spike_times = sorted({s[0] for s in spikes})[:6]
    for t in spike_times:
        cap2 = cv2.VideoCapture(a.video)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap2.read()
        cap2.release()
        if ok:
            vis = fr.copy()
            for ci, (x1, y1, x2, y2) in enumerate(comps):
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            for tt, ci, v, md in [s for s in spikes if s[0] == t]:
                x1, y1, x2, y2 = comps[ci]
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(vis, "#%d %.2f->%.2f" % (ci, md, v), (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imencode(".png", vis)[1].tofile(str(outdir / ("%s_spike_%.0fs.png" % (a.key, t))))

    out = {"video": os.path.basename(a.video), "size": [w, h], "fps": round(fps, 2),
           "step_s": a.step_s, "compartments_px": [[round(v, 1) for v in c] for c in comps],
           "times": times, "series": {str(k): v for k, v in series.items()},
           "stats": {str(k): v for k, v in stats.items()},
           "spikes": [{"t": s[0], "comp": s[1], "ratio": s[2], "median": s[3]} for s in spikes][:40]}
    with open(outdir / ("%s_doors.json" % a.key), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("系列点：%d，尖峰时刻：%s" % (len(times), spike_times))
    print("输出 ->", outdir / ("%s_doors.json" % a.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
