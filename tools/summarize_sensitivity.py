# -*- coding: utf-8 -*-
"""tools/summarize_sensitivity.py — 汇总阈值敏感性实验（door_open_alarm_s）

输入：logs/_sens_d{5,10,15,20}_shard{0,1}.json（tools/eval_dataset.py --set 产生）
输出：logs/exp_threshold_sensitivity.json
      每个阈值下的段级混淆矩阵与事件级逐类 TP/FP/FN、micro P/R/F1

段级判定口径与 logs/exp_stage3_behavior.json 一致：异常段只要命中任一期望事件即 TP。
"""
import json
import io
import glob
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")
THRESHOLDS = [5, 10, 15, 20]
ANOMALIES = {"LOITERING", "DOOR_LEFT_OPEN", "FLAGGED_DOOR_RETRIEVAL", "MULTI_RETRIEVAL"}


def load_th(th):
    """合并该阈值下的所有分片结果。"""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(LOG, "_sens_d%d_shard*.json" % th)))
    rows = []
    for p in files:
        rows += json.load(io.open(p, encoding="utf-8"))["results"]
    return rows or None


def summarize(rows):
    TP = FP = FN = TN = 0
    ev = defaultdict(lambda: [0, 0, 0])
    for x in rows:
        if x["is_abnormal"]:
            if x["hit"]:
                TP += 1
            else:
                FN += 1
        else:
            if x["fp"]:
                FP += 1
            else:
                TN += 1
        for e in x["hit"]:
            ev[e][0] += 1
        for e in x["fp"]:
            ev[e][1] += 1
        for e in x["miss"]:
            ev[e][2] += 1

    def prf(tp, fp, fn):
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p, r, f = prf(TP, FP, FN)
    mt = mf = mn = 0
    per_type = {}
    for k in sorted(ev):
        t, fp, fn = ev[k]
        pk, rk, fk = prf(t, fp, fn)
        mt += t; mf += fp; mn += fn
        per_type[k] = {"TP": t, "FP": fp, "FN": fn, "precision": pk, "recall": rk, "f1": fk}
    mp, mr, mfl = prf(mt, mf, mn)
    return {
        "clips": len(rows),
        "segments": {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "precision": p, "recall": r, "f1": f},
        "events": {"per_type": per_type,
                   "micro": {"TP": mt, "FP": mf, "FN": mn,
                             "precision": mp, "recall": mr, "f1": mfl}},
    }


def main():
    out = {"threshold_parameter": "door_open_alarm_s",
           "backend": "yolov8n",
           "note": "all other parameters at defaults; 119 clips, same-distribution",
           "points": {}}
    for th in THRESHOLDS:
        rows = load_th(th)
        if rows is None:
            print("缺少阈值 %d 的结果文件（实验未完成？）" % th)
            continue
        s = summarize(rows)
        out["points"][str(th)] = s
        seg, ev = s["segments"], s["events"]
        print("tau=%2ds  段级 P/R/F1 = %.3f/%.3f/%.3f  (TP=%d FP=%d FN=%d TN=%d)  "
              "事件 micro P/R/F1 = %.3f/%.3f/%.3f"
              % (th, seg["precision"], seg["recall"], seg["f1"],
                 seg["TP"], seg["FP"], seg["FN"], seg["TN"],
                 ev["micro"]["precision"], ev["micro"]["recall"], ev["micro"]["f1"]))
        for k, v in ev["per_type"].items():
            print("      %-24s TP=%3d FP=%3d FN=%3d  R=%.3f" % (k, v["TP"], v["FP"], v["FN"], v["recall"]))

    p = os.path.join(LOG, "exp_threshold_sensitivity.json")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n已保存：%s" % p)


if __name__ == "__main__":
    main()