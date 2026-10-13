# -*- coding: utf-8 -*-
"""tools/wilson_ci.py — 论文中各比例/均值区间的一键复算（全部从 logs/*.json 读取）

背景：论文为每个比例都报告了 Wilson 95% 区间，均值报告正态近似 95% 区间（n<30）。
审稿人要求「可复现」，故把区间的计算与所用 (k, n) 全部落到脚本里，数字只来自日志。

用法：
    python tools/wilson_ci.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
Z = 1.959963984540054


def wilson(k: int, n: int):
    """Wilson score 95% interval for k successes in n trials."""
    if n == 0:
        return (0.0, 100.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, (c - h)) * 100, min(1.0, (c + h)) * 100)


def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    h = Z * sd / math.sqrt(n)
    return m, sd, m - h, m + h


def jload(name):
    return json.load(open(LOGS / name, encoding="utf-8"))


def main():
    print("=" * 96)
    print("Wilson 95% CI（比例） —— 全部由 logs/*.json 复算")
    print("=" * 96)

    # 1) 现场语料碰撞率
    au = jload("field_color_audit.json")
    pf = sum(r["person_frames"] for r in au)
    cf = sum(r["collision_frames"] for r in au)
    print("corpus collision           %4d/%-5d  %.2f%%  CI %.1f--%.1f%%" %
          (cf, pf, 100 * cf / pf, *wilson(cf, pf)))
    for r in au:
        if r["key"] in ("v04", "v14"):
            print("  %s                       %4d/%-5d  %.2f%%  CI %.1f--%.1f%%" %
                  (r["key"], r["collision_frames"], r["person_frames"],
                   100 * r["collision_rate"], *wilson(r["collision_frames"], r["person_frames"])))

    # 2) v04 门态审计
    m = jload("exp_field_v04_eps_masked.json")
    n = int(m["n"])
    for eta in ("0.15", "0.45", "0.75"):
        k = int(round(m["eta_scan"][eta] * n))
        print("v04 open @eta=%-4s         %4d/%-5d  %.2f%%  CI %.1f--%.1f%%" %
              (eta, k, n, 100 * k / n, *wilson(k, n)))

    # 3) 渲染基准三臂（段级）与事件级
    beh = jload("exp_stage3_behavior.json")
    a2 = beh["A2_YOLOv8n_规则引擎"]["segments"]
    print("benchmark clips correct     %3d/%-3d  CI %.1f--%.1f%%" %
          (a2["TP"] + a2["TN"], a2["TP"] + a2["TN"] + a2["FP"] + a2["FN"],
           *wilson(a2["TP"] + a2["TN"], a2["TP"] + a2["TN"] + a2["FP"] + a2["FN"])))
    nrm = a2["TN"] + a2["FP"]
    print("benchmark normal correct    %3d/%-3d  CI %.1f--%.1f%%" %
          (a2["TN"], nrm, *wilson(a2["TN"], nrm)))
    for key, arm in beh.items():
        s = arm["segments"]
        tp, fp, fn, tn = s["TP"], s["FP"], s["FN"], s["TN"]
        print("%-26s recall   %3d/%-3d  %.3f  CI %.1f--%.1f%%" %
              (key[:26], tp, tp + fn, tp / max(1, tp + fn), *wilson(tp, tp + fn)))
        print("%-26s precision%3d/%-3d  %.3f  CI %.1f--%.1f%%" %
              ("", tp, tp + fp, tp / max(1, tp + fp), *wilson(tp, tp + fp)))

    # 4) 消融
    ab = jload("exp_ablation.json")
    for cfg, v in ab["configs"].items():
        s = v["segments"]
        print("ablation %-10s recall   %3d/%-3d  CI %.1f--%.1f%%" %
              (cfg, s["TP"], s["TP"] + s["FN"], *wilson(s["TP"], s["TP"] + s["FN"])))

    # 5) Avenue 帧级「无缺检出」
    ds = jload("exp_stage3_domain_shift.json")
    tot = ds["yolov8n"]["total_frames"]
    miss = sum(v["frames_without_person"] for v in ds["yolov8n"]["per_video"])
    print("Avenue frames w/o detection %d/%d  CI %.2f--%.2f%% (upper bound %.2f%%)" %
          (miss, tot, *wilson(miss, tot), wilson(miss, tot)[1]))

    print("\n" + "=" * 96)
    print("均值 95% CI（正态近似） —— 现场真实开门段（未遮挡样本）")
    print("=" * 96)
    d = jload("exp_field_door_masked.json")
    # 论文声明的开门窗 6.3--9.2 s；掩膜日志采样 0.2 s（t = 6.0 + 0.2 i）→ 索引 i∈[2,16]
    op = [v for v in d["series"]["O_open"][2:17] if v is not None]
    mm, sd, lo, hi = mean_ci(op)
    print("O_open open segment  n=%d  mean=%.4f  sd=%.4f  CI %.4f--%.4f  range %.3f--%.3f"
          % (len(op), mm, sd, lo, hi, min(op), max(op)))
    print("（论文用 13 个未遮挡样本；若此处 n 与之不同，以日志为准）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
