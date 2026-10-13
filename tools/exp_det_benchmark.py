# -*- coding: utf-8 -*-
"""
tools/exp_det_benchmark.py — 检测器级基准（真实图像 + 官方标注框）

数据集
------
COCO val2017 的 person 子集（真实图像，官方 GT 边界框，`category_id=1`）。
抽样固定随机种子，清单落盘 `logs/exp_coco_sample.csv`，可逐条复现。
（`iscrowd=1` 的拥挤标注不参与匹配，与常见做法一致，见实验报告说明。）

对比对象
--------
* **YOLOv8n**：COCO 预训练权重，`classes=[0]`（person），本机 CPU、`imgsz=480`；
* **HOG + 线性 SVM**：OpenCV 内置 INRIA 行人检测器，`resize_width=320`。

指标
----
* AP@0.5（VOC 全点插值，IoU≥0.5 贪心匹配）
* 最佳 F1 阈值处的 Precision / Recall / F1（阈值在该子集上扫描，报告取值）
* 固定工作阈值（0.35，与系统默认一致）处的 Precision / Recall / F1
* 单图推理耗时（均值 / 中位数 / P95）

命令行
------
    python tools/exp_det_benchmark.py --backend yolov8n --n-pos 400 --n-neg 100
    python tools/exp_det_benchmark.py --backend hog --n-pos 400 --n-neg 100
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402
from detector import Detection, build_detector, iou  # noqa: E402

COCO_DIR = C.DATA_DIR / "public_downloads" / "coco"
IMAGES = COCO_DIR / "val2017"
ANN = COCO_DIR / "annotations" / "instances_val2017.json"
SEED = 20260925


def load_val_persons() -> Tuple[Dict[int, List[Tuple[float, float, float, float]]], List[int], List[int]]:
    """返回 (image_id -> person 框列表, 含人图像 id 列表, 无人图像 id 列表)。"""
    with ANN.open(encoding="utf-8") as f:
        data = json.load(f)
    persons: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for a in data["annotations"]:
        if a["category_id"] != 1 or a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        persons.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    pos, neg = [], []
    for img in data["images"]:
        (pos if persons.get(img["id"]) else neg).append(img["id"])
    return persons, sorted(pos), sorted(neg)


def sample(n_pos: int, n_neg: int, seed: int = SEED):
    persons, pos, neg = load_val_persons()
    rnd = random.Random(seed)
    p = sorted(rnd.sample(pos, min(n_pos, len(pos))))
    n = sorted(rnd.sample(neg, min(n_neg, len(neg))))
    return persons, p, n


def evaluate(backend: str, sample_ids: Sequence[int],
             persons: Dict[int, List[Tuple[float, float, float, float]]],
             conf_threshold: float, imgsz: int) -> dict:
    det = build_detector(backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                         conf_threshold=conf_threshold, imgsz=imgsz)
    per_image = []                      # [{gt, preds:[(conf, tp)], ms}]
    for image_id in sample_ids:
        path = IMAGES / ("%012d.jpg" % image_id)
        # 中文路径兼容：cv2.imread 在本机对含中文的路径会失败（实测，
        # 且导入 ultralytics 后行为又不同），统一改用 fromfile + imdecode。
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t0 = time.perf_counter()
        dets = det.detect(img)
        ms = (time.perf_counter() - t0) * 1000.0
        gts = list(persons.get(image_id, []))
        preds = sorted([(float(d.conf), d.xyxy()) for d in dets], key=lambda t: -t[0])

        used = [False] * len(gts)
        matched: List[Tuple[float, bool]] = []
        for conf, box in preds:
            pd = Detection(box[0], box[1], box[2], box[3], "person", conf)
            best, best_i = 0.0, -1
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = iou(pd, Detection(g[0], g[1], g[2], g[3], "person", 1.0))
                if v > best:
                    best, best_i = v, i
            ok = best >= 0.5 and best_i >= 0
            if ok:
                used[best_i] = True
            matched.append((conf, ok))
        per_image.append({"image_id": image_id, "n_gt": len(gts), "preds": matched, "ms": ms})

    total_gt = sum(r["n_gt"] for r in per_image)
    all_preds: List[Tuple[float, bool]] = []
    for r in per_image:
        all_preds += r["preds"]
    all_preds.sort(key=lambda t: -t[0])

    def prf(th: float) -> Tuple[float, float, float]:
        tp = sum(1 for c, ok in all_preds if c >= th and ok)
        fp = sum(1 for c, ok in all_preds if c >= th and not ok)
        fn = total_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return p, rc, f1

    # AP@0.5：全点插值
    tp_c = fp_c = 0
    recs, precs = [], []
    for _c, ok in all_preds:
        tp_c += 1 if ok else 0
        fp_c += 0 if ok else 1
        recs.append(tp_c / total_gt if total_gt else 0.0)
        precs.append(tp_c / (tp_c + fp_c))
    ap = 0.0
    mrec = [0.0] + recs + [1.0]
    mpre = [0.0] + precs + [0.0]
    for i in range(len(mpre) - 2, -1, -1):          # 精度包络（VOC 全点插值）
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))
    ap = round(ap, 4)

    # 最佳 F1 阈值（在观测到的置信度上扫描）
    best = (0.0, 0.0, 0.0, 0.0, 0.0)   # f1, th, p, r
    for th in sorted({round(c, 3) for c, _ in all_preds} | {conf_threshold}):
        p, r, f1 = prf(th)
        if f1 > best[0]:
            best = (f1, th, p, r)
    p35, r35, f135 = prf(conf_threshold)
    # PR 曲线（按步长抽样保存，供绘图；同时保存 IoU≥0.5 的逐预测记录供复核）
    curve = []
    step = max(1, len(all_preds) // 200)
    for i in range(0, len(all_preds), step):
        th = all_preds[i][0]
        p, r, _f = prf(th)
        curve.append([round(th, 4), round(r, 4), round(p, 4)])
    ms = sorted(r["ms"] for r in per_image)
    return {
        "backend": det.backend_name,
        "display_name": det.display_name,
        "images": len(per_image),
        "gt_boxes": total_gt,
        "predictions": len(all_preds),
        "AP@0.5": ap,
        "pr_curve": curve,
        "best_f1": {"f1": round(best[0], 4), "threshold": round(best[1], 3),
                    "precision": round(best[2], 4), "recall": round(best[3], 4)},
        "at_conf_%.2f" % conf_threshold: {"precision": round(p35, 4),
                                          "recall": round(r35, 4), "f1": round(f135, 4)},
        "infer_ms": {"mean": round(sum(ms) / len(ms), 2), "median": round(ms[len(ms) // 2], 2),
                     "p95": round(ms[int(len(ms) * 0.95)], 2)},
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="检测器级基准（COCO val2017 person 子集）")
    ap.add_argument("--backend", default="yolov8n")
    ap.add_argument("--n-pos", type=int, default=400)
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    if not ANN.exists():
        print("未找到 COCO 标注：%s（先解压 annotations_trainval2017.zip）" % ANN)
        return 2
    persons, pos_ids, neg_ids = sample(a.n_pos, a.n_neg)
    ids = pos_ids + neg_ids
    print("=" * 84)
    print("检测器级基准：COCO val2017 person 子集  后端=%s" % a.backend)
    print("抽样：含人 %d 张 + 无人 %d 张（种子 %d，清单 logs/exp_coco_sample.csv）"
          % (len(pos_ids), len(neg_ids), SEED))
    print("=" * 84)

    manifest = C.LOG_DIR / "exp_coco_sample.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "n_person_gt", "group"])
        for i in pos_ids:
            w.writerow([i, "%012d.jpg" % i, len(persons[i]), "with_person"])
        for i in neg_ids:
            w.writerow([i, "%012d.jpg" % i, 0, "no_person"])

    t0 = time.perf_counter()
    res = evaluate(a.backend, ids, persons, a.conf, a.imgsz)
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    res["sample_seed"] = SEED
    print(json.dumps(res, ensure_ascii=False, indent=2))

    rp = Path(a.report) if a.report else C.LOG_DIR / ("exp_det_benchmark_%s.json" % res["backend"])
    rp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已保存：%s" % rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())