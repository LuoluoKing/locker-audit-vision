# -*- coding: utf-8 -*-
"""tools/field_event_eval.py — 现场素材上「校准配置事件级 P/R」的标注与评分工具（s8）

用途
----
论文「现场边界研究」目前只报告系统*行为*（报了多少事件），没有事件级的
precision / recall，因为现场素材没有逐帧真值。本工具把这个缺口变成可执行的
标注流程，但**不代替人**：真值必须由作者看过视频后填写，工具只负责

  1) `--worksheet`：把某次校准配置运行的候选事件导出成待标注表
     （候选事件 CSV + 真值模板 CSV + 说明），供人工逐条判定；
  2) `--score`：在作者填好真值后，计算事件级 TP/FP/FN 与 P/R/F1
     （按事件类型 + 时间容差匹配）。

在作者确认真值之前，本工具**不会**产出任何准确率数字。

真值模板约定（`*_truth.csv`）
----------------------------
    t,type,note
    12.4,DOOR_LEFT_OPEN,柜门确实没关
    35.1,MULTI_RETRIEVAL,一人拿走两袋
每条 = 一个**真实发生**的事件（作者判定）。`type` 取系统六类事件名之一：
    DOOR_LEFT_OPEN, MULTI_RETRIEVAL, FLAGGED_DOOR_RETRIEVAL, LOITERING,
    DOOR_CLOSED, ROI_INTRUSION
只统计 ABNORMAL_EVENTS（异常类）的事件级 P/R；`DOOR_CLOSED`/`ROI_INTRUSION`
等非异常类不参与 P/R（可留空）。

用法
----
    python tools/field_event_eval.py --log logs/field_eval/v04_cal055.json --worksheet
    python tools/field_event_eval.py --log logs/field_eval/v04_cal055.json \
        --truth logs/field_eval/v04_truth.csv --score
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import config as C  # noqa: E402

ANOMALY = {"DOOR_LEFT_OPEN", "MULTI_RETRIEVAL", "FLAGGED_DOOR_RETRIEVAL", "LOITERING"}
ALL_TYPES = sorted(ANOMALY | {"DOOR_CLOSED", "ROI_INTRUSION"})


def _load_log(path: Path):
    return json.load(open(path, encoding="utf-8"))


def worksheet(log_path: Path, tol_s: float):
    d = _load_log(log_path)
    key = log_path.stem          # 含 tag，避免不同配置互相覆盖
    events = d.get("events", [])
    out_dir = log_path.parent
    cand_csv = out_dir / ("%s_candidates.csv" % key)
    truth_csv = out_dir / ("%s_truth_TEMPLATE.csv" % key)

    with open(cand_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "t", "type", "abnormal", "detail"])
        for i, e in enumerate(events, 1):
            w.writerow([i, e["t"], e["type"], int(e["type"] in ANOMALY), e.get("detail", "")])

    if not truth_csv.exists():
        with open(truth_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "type", "note"])
    print("候选事件表 ->", cand_csv, "（%d 条）" % len(events))
    print("真值模板   ->", truth_csv, "（请作者逐段看视频后填写；当前为空）")
    print("\n说明：真值 = 视频里**真实发生**的异常事件；非异常类型可不填。")
    print("填好后运行：--truth %s --score" % truth_csv.name)
    return 0


def score(log_path: Path, truth_path: Path, tol_s: float):
    d = _load_log(log_path)
    key = log_path.stem
    cands = [e for e in d.get("events", []) if e["type"] in ANOMALY]
    rows = [r for r in csv.DictReader(open(truth_path, encoding="utf-8-sig"))
            if r.get("type", "").strip()]
    truth = [{"t": float(r["t"]), "type": r["type"].strip()} for r in rows]
    if not truth:
        print("真值表为空：请先填写后重跑（本工具不会凭空给数）。")
        return 2
    for t in truth:
        if t["type"] not in ANOMALY:
            print("警告：真值类型 %r 不在异常类集合内，将被忽略" % t["type"])

    used = [False] * len(cands)
    tp = []
    fn = []
    for g in truth:
        if g["type"] not in ANOMALY:
            continue
        best, bd = -1, 1e9
        for i, c in enumerate(cands):
            if used[i] or c["type"] != g["type"]:
                continue
            dd = abs(c["t"] - g["t"])
            if dd < bd:
                bd, best = dd, i
        if best >= 0 and bd <= tol_s:
            used[best] = True
            tp.append((g, cands[best], round(bd, 2)))
        else:
            fn.append(g)
    fp = [c for i, c in enumerate(cands) if not used[i]]

    P = len(tp) / max(1, len(tp) + len(fp))
    R = len(tp) / max(1, len(tp) + len(fn))
    F = 2 * P * R / max(1e-9, P + R)
    out = {"key": key, "tol_s": tol_s, "n_truth": len(truth), "n_candidates": len(cands),
           "TP": len(tp), "FP": len(fp), "FN": len(fn),
           "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
           "matches": [{"truth_t": g["t"], "type": g["type"], "cand_t": c["t"],
                        "dt": dt} for g, c, dt in tp],
           "false_positives": [{"t": c["t"], "type": c["type"]} for c in fp],
           "false_negatives": [{"t": g["t"], "type": g["type"]} for g in fn]}
    op = log_path.parent / ("%s_event_pr.json" % key)
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f（容差 ±%.1fs）"
          % (len(tp), len(fp), len(fn), P, R, F, tol_s))
    print("输出 ->", op)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="某次运行的 field_eval 日志")
    ap.add_argument("--worksheet", action="store_true", help="导出候选表 + 真值模板")
    ap.add_argument("--truth", default="", help="作者填好的真值 CSV")
    ap.add_argument("--score", action="store_true", help="按真值计算 P/R/F1")
    ap.add_argument("--tol", type=float, default=3.0, help="事件匹配时间容差（秒）")
    a = ap.parse_args()
    log_path = Path(a.log)
    if not log_path.exists():
        print("找不到日志：", log_path)
        return 2
    if a.score:
        if not a.truth:
            print("--score 需要 --truth")
            return 2
        return score(log_path, Path(a.truth), a.tol)
    return worksheet(log_path, a.tol)


if __name__ == "__main__":
    raise SystemExit(main())
