# -*- coding: utf-8 -*-
"""
tools/exp_behavior_arms.py — 阶段3 行为级三组对照实验（E1）

三组对照
--------
* **A1 仅检测器**      : 判据 = 整段中任一帧检测到行人 → 该段判为「异常」。
                         用于量化「检测器本身不含行为语义」这一基线。
* **A2 YOLOv8n + 规则** : 完整系统（阶段2 全量评估报告 `logs/dataset_eval_final3_shard*.json`）。
* **A3 HOG + 规则**     : CPU 回退路径（`logs/dataset_eval_final3_hog_shard*.json`）。

指标
----
段级二分类（正类 = 异常片段）：TP/FP/FN/TN、Precision、Recall、F1、准确率、误报率、漏检率；
事件级：按事件类型的 TP/FP/FN 与 micro/macro P/R/F1。

命令行
------
    python tools/exp_behavior_arms.py --mode baseline --shard 0/4     # A1 逐帧检测（分片）
    python tools/exp_behavior_arms.py --mode aggregate                # 汇总 A1/A2/A3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from detector import build_detector  # noqa: E402

ANOMALIES = sorted(C.ABNORMAL_EVENTS)
BASELINE_JSON = "exp_a1_baseline.json"
OUT_CSV = "exp_stage3_behavior.csv"
OUT_JSON = "exp_stage3_behavior.json"


def _labels() -> List[dict]:
    p = C.SIM_DIR / "labels.csv"
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def run_baseline(shard: str, backend: str, imgsz: int) -> int:
    """A1：逐帧跑检测器，记录「整段是否至少检出一次行人」。"""
    det = build_detector(backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=imgsz)
    rows = _labels()
    if shard:
        idx, tot = (int(v) for v in shard.split("/"))
        rows = [r for i, r in enumerate(rows) if i % tot == idx]
    out: Dict[str, dict] = {}
    t0 = time.perf_counter()
    for n, r in enumerate(rows, 1):
        cap = cv2.VideoCapture(str(C.SIM_DIR / r["video"]))
        hit_frames = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if det.detect(frame):
                hit_frames += 1
            i += 1
        cap.release()
        out[r["clip"]] = {"frames": i, "frames_with_person": hit_frames,
                          "is_abnormal": int(r["is_abnormal"]),
                          "predicted_abnormal": int(hit_frames > 0)}
        print("  [%3d/%d] %-30s 检出帧 %4d/%4d  预测异常=%d"
              % (n, len(rows), r["clip"], hit_frames, i, int(hit_frames > 0)))
    p = C.LOG_DIR / (BASELINE_JSON if not shard
                     else BASELINE_JSON.replace(".json", "_shard%s.json" % shard.split("/")[0]))
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("A1 基线（%s）已保存：%s  用时 %.1fs" % (backend, p, time.perf_counter() - t0))
    return 0


def _load_eval(pattern: str) -> Dict[str, dict]:
    """读取某组评估的分片报告，合并为 clip -> 结果。"""
    merged: Dict[str, dict] = {}
    for p in sorted(C.LOG_DIR.glob(pattern)):
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            merged[r["clip"]] = r
    return merged


def _segment_metrics(results: Dict[str, dict], predicted_abnormal) -> dict:
    tp = fp = fn = tn = 0
    for clip, r in results.items():
        truth = int(r["is_abnormal"])
        pred = int(predicted_abnormal(clip, r))
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = tp + fp + fn + tn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / n if n else 0.0, 4),
            "false_alarm_rate": round(fp / (fp + tn) if fp + tn else 0.0, 4),
            "miss_rate": round(fn / (fn + tp) if fn + tp else 0.0, 4)}


def _event_metrics(results: Dict[str, dict]) -> dict:
    """事件级：期望事件类型 vs 实际触发的异常事件类型。"""
    per_type: Dict[str, dict] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    for _clip, r in results.items():
        fired = set(r.get("fires", [])) & set(ANOMALIES)
        expect = set(r.get("expect_yes", [])) & set(ANOMALIES)
        for t in expect & fired:
            per_type[t]["TP"] += 1
        for t in expect - fired:
            per_type[t]["FN"] += 1
        for t in fired - expect:
            per_type[t]["FP"] += 1
    rows = []
    for t, c in sorted(per_type.items()):
        p = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
        rc = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
        rows.append({"event_type": t, "TP": c["TP"], "FP": c["FP"], "FN": c["FN"],
                     "precision": round(p, 4), "recall": round(rc, 4),
                     "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0})
    tp = sum(r["TP"] for r in rows); fp = sum(r["FP"] for r in rows); fn = sum(r["FN"] for r in rows)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    return {"per_type": rows,
            "micro": {"TP": tp, "FP": fp, "FN": fn, "precision": round(mp, 4),
                      "recall": round(mr, 4), "f1": round(2 * mp * mr / (mp + mr), 4) if mp + mr else 0.0},
            "macro_f1": round(macro_f1, 4)}


def aggregate() -> int:
    arms: Dict[str, dict] = {}

    # ---- 只统计当前数据集中仍存在的片段（按 data/simulated/labels.csv）----
    # 数据集会随需求变更（如 2026-10-07 移除「长时间停留」21 段），
    # 逐段评估的原始记录保持不动，聚合时按当前 labels 过滤，保证指标与现行数据集一致。
    cur_clips = None
    try:
        labels_path = C.BASE_DIR / "data" / "simulated" / "labels.csv"
        with labels_path.open(encoding="utf-8-sig") as f:
            cur_clips = {row["clip"] for row in csv.DictReader(f)}
    except OSError:
        cur_clips = None

    def _keep(d):
        return {k: v for k, v in d.items() if cur_clips is None or k in cur_clips}

    # ---- A1 仅检测器（合并分片）----
    a1: Dict[str, dict] = {}
    for p in sorted(C.LOG_DIR.glob(BASELINE_JSON.replace(".json", "_shard*.json"))):
        a1.update(json.loads(p.read_text(encoding="utf-8")))
    if not a1 and (C.LOG_DIR / BASELINE_JSON).exists():
        a1 = json.loads((C.LOG_DIR / BASELINE_JSON).read_text(encoding="utf-8"))
    a1 = _keep(a1)
    if a1:
        labeled = {c: {"is_abnormal": v["is_abnormal"]} for c, v in a1.items()}
        arms["A1_仅YOLOv8n检测（无规则）"] = {
            "detector_only": True,
            "segments": _segment_metrics(labeled, lambda c, r: a1[c]["predicted_abnormal"]),
            "person_detected_segments": sum(1 for v in a1.values() if v["predicted_abnormal"]),
        }

    # ---- A2 YOLOv8n + 规则 ----
    a2 = _keep(_load_eval("dataset_eval_final3_shard*.json"))
    if a2:
        arms["A2_YOLOv8n_规则引擎"] = {
            "segments": _segment_metrics(a2, lambda c, r: bool(set(r["fires"]) & set(ANOMALIES))),
            "events": _event_metrics(a2),
            "clips": len(a2),
        }

    # ---- A3 HOG + 规则 ----
    a3 = _keep(_load_eval("dataset_eval_final3_hog_shard*.json"))
    if a3:
        arms["A3_HOG_规则引擎"] = {
            "segments": _segment_metrics(a3, lambda c, r: bool(set(r["fires"]) & set(ANOMALIES))),
            "events": _event_metrics(a3),
            "clips": len(a3),
        }

    print("=" * 96)
    print("%-30s %6s %6s %6s %6s %8s %8s %8s %8s" %
          ("实验组", "TP", "FP", "FN", "TN", "精确率", "召回率", "F1", "准确率"))
    print("-" * 96)
    for name, d in arms.items():
        s = d["segments"]
        print("%-30s %6d %6d %6d %6d %8.4f %8.4f %8.4f %8.4f" %
              (name, s["TP"], s["FP"], s["FN"], s["TN"], s["precision"], s["recall"], s["f1"], s["accuracy"]))
    print("=" * 96)
    for name, d in arms.items():
        if "events" in d:
            print("\n%s 事件级（micro: P=%.4f R=%.4f F1=%.4f；macro-F1=%.4f）" %
                  (name, d["events"]["micro"]["precision"], d["events"]["micro"]["recall"],
                   d["events"]["micro"]["f1"], d["events"]["macro_f1"]))
            for r in d["events"]["per_type"]:
                print("    %-26s TP=%3d FP=%2d FN=%2d  P=%.3f R=%.3f F1=%.3f" %
                      (r["event_type"], r["TP"], r["FP"], r["FN"], r["precision"], r["recall"], r["f1"]))

    (C.LOG_DIR / OUT_JSON).write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    with (C.LOG_DIR / OUT_CSV).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "TP", "FP", "FN", "TN", "precision", "recall", "f1",
                    "accuracy", "false_alarm_rate", "miss_rate"])
        for name, d in arms.items():
            s = d["segments"]
            w.writerow([name, s["TP"], s["FP"], s["FN"], s["TN"], s["precision"], s["recall"],
                        s["f1"], s["accuracy"], s["false_alarm_rate"], s["miss_rate"]])
    print("\n汇总已保存：%s / %s" % (C.LOG_DIR / OUT_JSON, C.LOG_DIR / OUT_CSV))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段3 行为级三组对照（E1）")
    ap.add_argument("--mode", choices=["baseline", "aggregate"], required=True)
    ap.add_argument("--shard", default="")
    ap.add_argument("--backend", default="yolov8n")
    ap.add_argument("--imgsz", type=int, default=480)
    a = ap.parse_args()
    if a.mode == "baseline":
        return run_baseline(a.shard, a.backend, a.imgsz)
    return aggregate()


if __name__ == "__main__":
    sys.exit(main())