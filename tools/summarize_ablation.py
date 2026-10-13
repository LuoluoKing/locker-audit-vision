# -*- coding: utf-8 -*-
"""tools/summarize_ablation.py — 汇总消融实验（组件贡献归因）

输入：logs/ablation/{full,noreid,nooccl,noguard}.json
      （由 tools/eval_dataset.py --per-behavior 4 --set ... 产生，每配置同一 20 段）
输出：logs/exp_ablation.json（段级混淆 + 事件级逐类 TP/FP/FN + 相对 full 的增减）
"""
import io
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")

CONFIGS = [
    ("full", "complete system"),
    ("noreid", "w/o re-association"),
    ("nooccl", "w/o occlusion look-back"),
    ("noguard", "w/o partial-box guards"),
]


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
    return {"segments": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "events": {k: {"TP": v[0], "FP": v[1], "FN": v[2]} for k, v in ev.items()}}


def main():
    out = {"clips_per_config": None, "configs": {}}
    base = None
    print("%-26s %6s %6s %6s | %s" % ("configuration", "TP", "FP", "FN",
                                      "event TP per class"))
    print("-" * 100)
    for key, label in CONFIGS:
        p = os.path.join(LOG, "ablation", key + ".json")
        if not os.path.exists(p):
            print("缺少结果：%s（实验未完成？）" % key)
            continue
        rows = json.load(io.open(p, encoding="utf-8"))["results"]
        s = summarize(rows)
        out["clips_per_config"] = len(rows)
        seg = s["segments"]
        ev = s["events"]
        tot_ev_tp = sum(v["TP"] for v in ev.values())
        if key == "full":
            base = (seg, ev)
        print("%-26s %6d %6d %6d | total TP=%d  %s"
              % (label, seg["TP"], seg["FP"], seg["FN"], tot_ev_tp,
                 {k: v["TP"] for k, v in sorted(ev.items())}))
        out["configs"][key] = {"label": label, **s}

    # 相对 full 的增减
    if base:
        bseg, bev = base
        print("\n相对完整系统的损失（段级 FN 增加 / 事件级 TP 减少）：")
        for key, label in CONFIGS:
            if key == "full" or key not in out["configs"]:
                continue
            c = out["configs"][key]
            d_fn = c["segments"]["FN"] - bseg["FN"]
            d_tp = sum(v["TP"] for v in c["events"].values()) - sum(v["TP"] for v in bev.values())
            per = {k: bev[k]["TP"] - c["events"][k]["TP"]
                   for k in bev if bev[k]["TP"] != c["events"][k]["TP"]}
            print("  %-26s ΔFN=%+d  ΔeventTP=%+d  %s" % (label, d_fn, d_tp, per))
            out["configs"][key]["delta_vs_full"] = {"segments_FN": d_fn,
                                                    "event_TP": d_tp,
                                                    "per_event_TP_lost": per}

    p = os.path.join(LOG, "exp_ablation.json")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n已保存：%s" % p)


if __name__ == "__main__":
    main()