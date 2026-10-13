# -*- coding: utf-8 -*-
"""tools/field_study.py — 现场边界研究（论文第 7 章）测量日志的**统一产出脚本**

背景：论文「现场研究」用到四份测量日志，但此前项目里只有读它们的绘图脚本、
没有产它们的脚本（见 paper/applsci_mdpi/终稿审计报告.md 的 M2 条）。本脚本把
这些日志从既有中间产物 / 公开素材**可复现地**重算出来，使论文的 Data
Availability「随代码一并发布」这一声明成立。

产出（写入 logs/）：
  1) exp_field_summary.json        表 3：逐段时长/分辨率/亮度/运动/事件/人员帧
                                   合并字段：field_survey.json（分析时
                                   analyze_field_videos.py 产出）、
                                   field_color_audit.json（field_color_audit.py）、
                                   field_eval/v*.json（eval_field.py）。
                                   纯数值合并，无需重新解码视频。
  2) exp_field_v04_eps.json        v04「校园柜排」12 个格口内胆的暗比时间序列
                                   （阈值 = 0.62×全画面灰度中位数；每 2 s 采样）。
  3) exp_field_v04_eps_masked.json 同上 + 遮挡掩膜（人员框覆盖 >50% 格口面积记为
                                   遮挡）、η 扫描与基线分位数（n=1152）。
  4) exp_field_door.json           v05「丰巢关门公益片」真实开门事件的门态序列：
                                   O_open（开门格口）+ 三个邻接对照区
                                   C_above / C_right / C_below。
  5) exp_field_door_masked.json    同一格口组合、更长的窗口与更粗的采样（0.2 s），
                                   带可见度掩膜（vis_frac<0.5 记 unknown，值置 null）。

复现口径与实测吻合度（由 --check 与既有日志逐字段比对得到）：
  * exp_field_summary.json            —— **逐字段完全一致**（纯日志合并，无解码）。
  * exp_field_v04_eps.json            —— 1440/1452 字段一致；不一致的 12 处全部是
    stats[*].occluded（原始运行的遮挡计数用了与本文不同的口径，且与其自身 n 字段
    不自洽：其 occluded 之和 392 ≠ 1452−1152）。序列/中位/p90 完全一致。
  * exp_field_v04_eps_masked.json     —— cells/基线中位 0.292/基线 p90 0.328/n=1152
    完全一致；仅 eta_scan 相差约 1%（本文 1148/1152 对原始 1136/1152），差值来自
    遮挡掩膜口径不同。
  * exp_field_door.json               —— 374/376 字段一致；例外 2 处（C_below[1][2]）
    为 ±0.005 的解码定位抖动。
  * exp_field_door_masked.json        —— 序列 300/304 一致；vis_frac 系统性相差约
    ±0.02（原始运行人员框略大），仅影响临界样本的 null 判定。

注意：论文正文引用的现场数字（如 1136/1152、292、328）取自**原始运行**写入的
logs/*.json，是权威记录。本脚本默认 `--check` 模式**只重算、不写盘**；`--all`
会覆盖既有日志，请仅在确认要替换原始记录时使用。

用法：
  python tools/field_study.py --check          # 只重算并与现有日志逐字段比对（默认）
  python tools/field_study.py --all            # 重算并写盘（会覆盖既有日志，慎用）
  python tools/field_study.py --only summary   # 只做其中一项
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

LOGS = ROOT / "logs"
VID_DIR = ROOT / "EXAMPLE" / "bilibili-vedio"

# ------------------------------------------------------------------
# 常量：逐段缩写标题（作者定义，非测量量）+ 现场研究的采样口径
# ------------------------------------------------------------------
LABELS = {
    "v01": "Fudan Univ., courier deposit (locker full scenario)",
    "v02": "Fudan Univ., pickup",
    "v03": "Fudan Univ., courier deposit (long)",
    "v04": "Campus locker bank, meal storage/retrieval (static)",
    "v05": "Fengchao express locker, door-closing PSA",
    "v06": "Generic food locker (clip 1)",
    "v07": "Indoor food locker room, multi-user",
    "v08": "Changshu Inst. Tech., campus lockers",
    "v09": "Xintian smart cabinet, usage flow",
    "v10": "Express locker demo (vertical)",
    "v11": "TV report, contactless pickup in pandemic",
    "v12": "Smart handover cabinet demo",
    "v13": "Smart express locker demo",
    "v14": "Wuhu college, food lockers",
    "v15": "West-Bldg-6 canteen pickup cabinet",
}

DARK_FACTOR = 0.62          # 暗比阈值 = DARK_FACTOR × 全画面灰度中位数
V04_STEP_S = 2.0            # v04 暗比采样间隔
V04_START_F = 136           # 门态序列起始帧（v05 25 fps；与原始运行一致）
V04_STEP_F = 2
V05_DOOR_F0 = 136           # exp_field_door.json 起始帧
V05_DOOR_STEP = 2
V05_MASK_F0 = 150           # exp_field_door_masked.json 起始帧（t=6.0s）
V05_MASK_STEP = 5           # 0.2 s @ 25 fps
OCC_THRESH = 0.50           # 人员框覆盖超过格口面积的此比例 → 记遮挡
VIS_THRESH = 0.50           # 可见度低于此 → 记 unknown（值置 null）
ETA_GRID = [0.15, 0.45, 0.55, 0.65, 0.75]


# ------------------------------------------------------------------
# 1) 表 3：现场语料盘点（纯日志合并）
# ------------------------------------------------------------------
def build_summary():
    survey = {r["key"]: r for r in json.load(open(LOGS / "field_survey.json", encoding="utf-8"))}
    audit = {r["key"]: r for r in json.load(open(LOGS / "field_color_audit.json", encoding="utf-8"))}
    rows = []
    for k in sorted(survey, key=lambda s: int(s[1:])):
        s, a = survey[k], audit[k]
        e = json.load(open(LOGS / "field_eval" / (k + ".json"), encoding="utf-8"))
        counts = e.get("counts") or {}
        rows.append({
            "key": k,
            "label": LABELS[k],
            "file": s["file"],
            "dur": s["duration_s"],
            "size": "%dx%d" % (s["w"], s["h"]),
            "bright": s["mean_brightness"],
            "motion": s["motion_mean"],
            "person_frames": a["person_frames"],
            "collision_rate": a["collision_rate"],
            "apparel_med": a["apparel_domain_median"],
            "max_blobs": a["max_blobs_in_person"],
            "events": counts,
            "events_total": (sum(counts.values()) if counts else None),
        })
    return {"exp_field_summary.json": rows}


# ------------------------------------------------------------------
# 2/3) v04 格口暗比序列 + 遮挡掩膜统计
# ------------------------------------------------------------------
def _video_path_of(key):
    files = sorted(f for f in os.listdir(VID_DIR) if f.lower().endswith(".mp4"))
    return VID_DIR / files[int(key[1:]) - 1]


def build_v04_eps():
    ref = json.load(open(LOGS / "exp_field_v04_eps.json", encoding="utf-8"))
    cells, times = ref["cells"], ref["times"]
    cap = cv2.VideoCapture(str(_video_path_of("v04")))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    raw = {str(ci): [] for ci in range(len(cells))}   # 4 位精度，供掩膜统计
    series = {str(ci): [] for ci in range(len(cells))}  # 3 位精度，与既有日志一致
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        thr = DARK_FACTOR * float(np.median(gray))
        for ci, (x1, y1, x2, y2) in enumerate(cells):
            g = gray[y1:y2, x1:x2]
            v = float(np.count_nonzero(g < thr)) / g.size
            raw[str(ci)].append(v)
            series[str(ci)].append(round(v, 3))
    cap.release()

    # 逐格口统计（stats 覆盖全部样本，含遮挡；frac_open 用 η=0.15）
    stats = {}
    for ci in range(len(cells)):
        v = np.array(series[str(ci)], dtype=float)
        stats[str(ci)] = {
            "median": round(float(np.median(v)), 3),
            "p90": round(float(np.percentile(v, 90)), 3),
            "max": round(float(v.max()), 3),
            "frac_open": round(float((v >= ETA_GRID[0]).mean()), 3),
            "occluded": None,   # 由掩膜阶段回填
            "n": int(v.size),
        }
    full = {"video": os.path.basename(str(_video_path_of("v04"))),
            "cells": cells, "times": times, "series": series, "stats": stats}

    # 掩膜：逐样本用 YOLO 人员框算格口被覆盖比例
    import config as C
    from detector import build_detector
    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=480)
    cap = cv2.VideoCapture(str(_video_path_of("v04")))
    ratio = {str(ci): [] for ci in range(len(cells))}
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            break
        b = det.detect(fr)
        for ci, (x1, y1, x2, y2) in enumerate(cells):
            area = float((x2 - x1) * (y2 - y1))
            mx = 0.0
            for d in b:
                ix1, iy1 = max(int(d.x1), x1), max(int(d.y1), y1)
                ix2, iy2 = min(int(d.x2), x2), min(int(d.y2), y2)
                if ix2 > ix1 and iy2 > iy1:
                    mx = max(mx, (ix2 - ix1) * (iy2 - iy1) / area)
            ratio[str(ci)].append(mx)
    cap.release()

    for ci in range(len(cells)):
        stats[str(ci)]["occluded"] = int(sum(1 for r in ratio[str(ci)] if r > OCC_THRESH))

    kept = []
    for ci in range(len(cells)):
        for i, r in enumerate(ratio[str(ci)]):
            if r <= OCC_THRESH:
                kept.append(raw[str(ci)][i])
    kept = np.array(kept, dtype=float)
    masked = {
        "cells": cells,
        "eta_scan": {str(e): round(float((kept >= e).mean()), 4) for e in ETA_GRID},
        "baseline_median": round(float(np.median(kept)), 3),
        "baseline_p90": round(float(np.percentile(kept, 90)), 3),
        "n": int(kept.size),
    }
    return {"exp_field_v04_eps.json": full, "exp_field_v04_eps_masked.json": masked}


# ------------------------------------------------------------------
# 4/5) v05 门态序列（O_open + 三个对照区）
# ------------------------------------------------------------------
def build_door():
    ref = json.load(open(LOGS / "exp_field_door.json", encoding="utf-8"))
    comps = {k: list(v) for k, v in ref["comps"].items()}
    times = ref["times"]
    cap = cv2.VideoCapture(str(_video_path_of("v05")))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    series = {k: [] for k in comps}
    for i in range(len(times)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, V05_DOOR_F0 + V05_DOOR_STEP * i)
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        thr = DARK_FACTOR * float(np.median(gray))
        for k, (x1, y1, x2, y2) in comps.items():
            g = gray[y1:y2, x1:x2]
            series[k].append(round(float(np.count_nonzero(g < thr)) / g.size, 3))
    cap.release()

    # 分段统计（作者按帧时刻人工划定的四个阶段；8–10、47–49、68–75 为开/关
    # 过渡样本，不计入任一稳定段）
    seg = {"closed_before": (0, 8), "open": (10, 47), "man": (49, 68), "closed_after": (75, 94)}
    segments = {}
    for name, (a, b) in seg.items():
        segments[name] = {}
        for k in comps:
            v = np.array(series[k][a:b], dtype=float)
            segments[name][k] = {"mean": round(float(v.mean()), 3),
                                 "min": round(float(v.min()), 3),
                                 "max": round(float(v.max()), 3), "n": int(v.size)}
    full = {"comps": {k: [int(x) for x in v] for k, v in comps.items()},
            "segments": segments, "times": times, "series": series}

    # 可见度掩膜版：更长窗口、更粗采样
    ref2 = json.load(open(LOGS / "exp_field_door_masked.json", encoding="utf-8"))
    times2 = ref2["times"]
    import config as C
    from detector import build_detector
    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=480)
    cap = cv2.VideoCapture(str(_video_path_of("v05")))
    series2 = {k: [] for k in comps}
    vis2 = {k: [] for k in comps}
    for i in range(len(times2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, V05_MASK_F0 + V05_MASK_STEP * i)
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        thr = DARK_FACTOR * float(np.median(gray))
        b = det.detect(fr)
        for k, (x1, y1, x2, y2) in comps.items():
            box = gray[y1:y2, x1:x2]
            cov = np.zeros(box.shape, dtype=bool)
            for d in b:
                ix1, iy1 = max(int(d.x1), x1), max(int(d.y1), y1)
                ix2, iy2 = min(int(d.x2), x2), min(int(d.y2), y2)
                if ix2 > ix1 and iy2 > iy1:
                    cov[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = True
            vis = 1.0 - float(cov.mean())
            vis2[k].append(round(vis, 2))
            if vis < VIS_THRESH:
                series2[k].append(None)
            else:
                free = box[~cov]
                series2[k].append(round(float(np.count_nonzero(free < thr)) / max(1, free.size), 3))
    cap.release()
    masked = {"comps": {k: [int(x) for x in v] for k, v in comps.items()},
              "times": times2, "series": series2, "vis_frac": vis2,
              "note": "camera settles at ~6.2s; samples 6.0-6.2 may show the settling view"}
    return {"exp_field_door.json": full, "exp_field_door_masked.json": masked}


# ------------------------------------------------------------------
# 比对 / 落盘
# ------------------------------------------------------------------
def _agree(got, want):
    """返回 (一致布尔, 差异列表)。数值容差 1e-6；None 必须对齐。"""
    diffs = []

    def walk(g, w, path):
        if isinstance(w, dict):
            if not isinstance(g, dict):
                diffs.append("%s: type" % path); return
            for k in set(list(w.keys()) + list(g.keys())):
                if k not in g: diffs.append("%s.%s missing in got" % (path, k))
                elif k not in w: diffs.append("%s.%s extra in got" % (path, k))
                else: walk(g[k], w[k], "%s.%s" % (path, k))
        elif isinstance(w, list):
            if not isinstance(g, list) or len(g) != len(w):
                diffs.append("%s: len %s vs %s" % (path, len(g) if isinstance(g, list) else "?", len(w))); return
            for i, (gi, wi) in enumerate(zip(g, w)):
                walk(gi, wi, "%s[%d]" % (path, i))
        else:
            if w is None or g is None:
                if w is not g: diffs.append("%s: %r vs %r" % (path, g, w))
            elif isinstance(w, (int, float)) and isinstance(g, (int, float)):
                if abs(float(g) - float(w)) > 1e-6:
                    diffs.append("%s: %r vs %r" % (path, g, w))
            elif g != w:
                diffs.append("%s: %r vs %r" % (path, g, w))

    walk(got, want, "$")
    return (len(diffs) == 0), diffs


def emit(products, write):
    report = {}
    for name, obj in products.items():
        p = LOGS / name
        if p.exists():
            want = json.load(open(p, encoding="utf-8"))
            ok, diffs = _agree(obj, want)
            report[name] = {"exact": ok, "ndiff": len(diffs), "sample": diffs[:6]}
            print("[%s] %s" % ("OK " if ok else "DIFF", name),
                  "" if ok else "(差异 %d 处)" % len(diffs))
            if not ok:
                for d in diffs[:6]:
                    print("    ", d)
        else:
            report[name] = {"exact": None, "ndiff": None, "sample": []}
            print("[NEW] %s（无既有日志可比对）" % name)
        if write:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="重算并写盘")
    ap.add_argument("--check", action="store_true", help="只重算、与现有日志比对")
    ap.add_argument("--only", default="", choices=["", "summary", "v04", "door"])
    a = ap.parse_args()
    if not (a.all or a.check or a.only):
        a.check = True

    products = {}
    if a.only in ("", "summary"):
        products.update(build_summary())
    if a.only in ("", "v04"):
        products.update(build_v04_eps())
    if a.only in ("", "door"):
        products.update(build_door())

    emit(products, write=a.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
