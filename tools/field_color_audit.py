# -*- coding: utf-8 -*-
"""tools/field_color_audit.py — 部署前「服装-餐盒色域碰撞」审计（E8-E）

动机（来自实拍素材实测）：餐盒检测器是 HSV 色域代理（H∈[15,95], S≥130,
V≥90, area∈[60,1200]）。实拍中高饱和暖色衣物（黄/橙/绿系工装）会落入同一
色域，被「手持计数」误计 → 凭空触发 `MULTI_RETRIEVAL`。

本工具把这一风险量化为**安装前的审计指标**：对每段实拍素材周期采样，
在每个人的检测框内统计落入餐盒色域的连通域个数：
    · collision_frames  ：框内 ≥2 个餐盒状 blobs 的人员帧占比（= 误报风险）
    · apparel_risk      ：人员框内落入色域的像素占比（中位数）
输出 logs/field_color_audit.json，用于论文的「部署前审计」程序说明。

用法：python tools/field_color_audit.py [--step-s 1.0]
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

import config as C  # noqa: E402
from detector import PackageDetector, build_detector  # noqa: E402

VID_DIR = ROOT / "EXAMPLE" / "bilibili-vedio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / "field_color_audit.json"))
    a = ap.parse_args()

    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=480)
    pkg = PackageDetector()

    files = sorted(f for f in os.listdir(VID_DIR) if f.lower().endswith(".mp4"))
    audit = []
    for vi, fn in enumerate(files, 1):
        key = "v%02d" % vi
        cap = cv2.VideoCapture(str(VID_DIR / fn))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(a.step_s * fps))
        person_frames = 0
        collision_frames = 0
        apparel_ratios = []
        max_blobs = 0
        i = 0
        while i < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            dets = det.detect(fr)
            if dets:
                blobs = pkg.detect(fr)
                hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                lo = np.array([pkg.hue_min, pkg.sat_min, pkg.val_min])
                hi = np.array([pkg.hue_max, 255, 255])
                dom = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))
                for d in dets:
                    x1, y1, x2, y2 = [int(max(0, v)) for v in (d.x1, d.y1, d.x2, d.y2)]
                    x2 = min(x2, fr.shape[1]); y2 = min(y2, fr.shape[0])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    person_frames += 1
                    box = dom[y1:y2, x1:x2]
                    apparel_ratios.append(round(float(np.count_nonzero(box)) / box.size, 4))
                    inside = sum(1 for b in blobs
                                 if x1 <= b.center[0] <= x2 and y1 <= b.center[1] <= y2)
                    max_blobs = max(max_blobs, inside)
                    if inside >= 2:
                        collision_frames += 1
            i += step
        cap.release()
        rec = {
            "key": key, "file": fn, "person_frames": person_frames,
            "collision_frames": collision_frames,
            "collision_rate": round(collision_frames / max(1, person_frames), 4),
            "apparel_domain_median": round(float(np.median(apparel_ratios)), 4) if apparel_ratios else None,
            "apparel_domain_p90": round(float(np.percentile(apparel_ratios, 90)), 4) if apparel_ratios else None,
            "max_blobs_in_person": max_blobs,
        }
        audit.append(rec)
        print("%s %-52s 人员帧=%4d  碰撞率=%6.1f%%  框内色域占比中位=%.3f  最多blob=%d" % (
            key, fn[:52], person_frames, 100 * rec["collision_rate"],
            rec["apparel_domain_median"] or -1, max_blobs))

    json.dump(audit, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total_pf = sum(r["person_frames"] for r in audit)
    total_cf = sum(r["collision_frames"] for r in audit)
    print("\n合计：人员帧 %d，其中碰撞帧 %d（%.1f%%）" %
          (total_pf, total_cf, 100 * total_cf / max(1, total_pf)))
    print("输出 ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
