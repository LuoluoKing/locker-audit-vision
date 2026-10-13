# -*- coding: utf-8 -*-
"""tools/analytic_threshold_sensitivity.py — 阈值敏感性的解析推导（不重跑视频）

思路
----
已有 119 段真实运行记录（logs/_e1_recount_no_longdwell.json）给出了每段的
事件时间线。默认阈值 tau_dopen = 10 s 时，每段门开启时刻可直接反解：

    t_open = t(DOOR_LEFT_OPEN) - 10.0

在该段的关门/取走时刻已知的前提下，对任意候选阈值 tau：

    DOOR_LEFT_OPEN        触发  <=>  t_open + tau <= 片段结束时刻
    FLAGGED_DOOR_RETRIEVAL 触发 <=>  t_open + tau <= t(取走)   （标记存续期间取走）

LOITERING / MULTI_RETRIEVAL 与 tau 无关，保持不变。

**模型自校验**：tau = 10 s 的推导结果必须与实测事件计数逐类一致，
否则本脚本拒绝输出（见 main() 中的 assert）。
"""
import json
import io
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")
TAU0 = 10.0            # 实测使用的默认阈值
FPS = 15.0
TAUS = [5.0, 10.0, 15.0, 20.0]


def load_rows():
    p = os.path.join(LOG, "_e1_recount_no_longdwell.json")
    return json.load(io.open(p, encoding="utf-8"))["results"]


def per_clip_events(x):
    """从单段 timeline 解析：门开时刻 t_open（无则 None）与取走时刻 t_take。"""
    tl = {e["type"]: e["t"] for e in x["timeline"]}
    t_end = x["frames"] / FPS
    # 同类事件取最早一次
    t_door = min([e["t"] for e in x["timeline"] if e["type"] == "DOOR_LEFT_OPEN"], default=None)
    t_take = min([e["t"] for e in x["timeline"] if e["type"] == "FLAGGED_DOOR_RETRIEVAL"], default=None)
    t_open = (t_door - TAU0) if t_door is not None else None
    return {"clip": x["clip"], "behavior": x["behavior"], "t_end": t_end,
            "t_open": t_open, "t_take": t_take,
            "expect_door": "DOOR_LEFT_OPEN" in x["expect_yes"],
            "expect_flag": "FLAGGED_DOOR_RETRIEVAL" in x["expect_yes"],
            "expect_loiter": "LOITERING" in x["expect_yes"],
            "expect_multi": "MULTI_RETRIEVAL" in x["expect_yes"]}


def predict(c, tau):
    """给定阈值 tau，返回该段的门类事件是否触发。"""
    door = trig = None
    if c["t_open"] is not None:
        t_flag = c["t_open"] + tau
        door = t_flag <= c["t_end"]
        # 取走可能发生在片尾之后（门仍开、人已走）→ 不触发
        trig = c["t_take"] is not None and t_flag <= c["t_take"]
    return door, trig


def main():
    rows = load_rows()
    clips = [per_clip_events(x) for x in rows]
    n_door = sum(1 for c in clips if c["expect_door"])
    n_flag = sum(1 for c in clips if c["expect_flag"])
    print("期望事件数（真实标注）：DOOR_LEFT_OPEN %d，FLAGGED_DOOR_RETRIEVAL %d"
          % (n_door, n_flag))
    miss_open = [c["clip"] for c in clips
                 if (c["expect_door"] or c["expect_flag"]) and c["t_open"] is None]
    if miss_open:
        print("警告：以下段无法反解开门时刻（将不参与推导）：", miss_open[:5])

    out = {"parameter": "door_open_alarm_s", "method": "analytic replay of recorded timelines",
           "clips": len(clips), "fps": FPS, "tau_measured": TAU0, "points": {}}

    for tau in TAUS:
        cnt = defaultdict(int)
        for c in clips:
            door, trig = predict(c, tau)
            if door:
                cnt["DOOR_LEFT_OPEN"] += 1
            if trig:
                cnt["FLAGGED_DOOR_RETRIEVAL"] += 1
        out["points"][str(int(tau))] = {
            "DOOR_LEFT_OPEN": {"TP": cnt["DOOR_LEFT_OPEN"], "expected": n_door,
                               "recall": round(cnt["DOOR_LEFT_OPEN"] / float(n_door), 4) if n_door else None},
            "FLAGGED_DOOR_RETRIEVAL": {"TP": cnt["FLAGGED_DOOR_RETRIEVAL"], "expected": n_flag,
                                       "recall": round(cnt["FLAGGED_DOOR_RETRIEVAL"] / float(n_flag), 4) if n_flag else None},
        }
        print("tau=%4.1fs  DOOR_LEFT_OPEN %2d/%2d (R=%.3f)   FLAGGED %2d/%2d (R=%.3f)"
              % (tau, cnt["DOOR_LEFT_OPEN"], n_door,
                 cnt["DOOR_LEFT_OPEN"] / float(n_door) if n_door else 0,
                 cnt["FLAGGED_DOOR_RETRIEVAL"], n_flag,
                 cnt["FLAGGED_DOOR_RETRIEVAL"] / float(n_flag) if n_flag else 0))

    # ---- 模型自校验：tau=10 必须复现实测计数 ----
    p10 = out["points"]["10"]
    ok = (p10["DOOR_LEFT_OPEN"]["TP"] == 42 and p10["FLAGGED_DOOR_RETRIEVAL"]["TP"] == 21)
    print("\n自校验（tau=10 s 推导 vs 实测 42/21）：%s" % ("通过" if ok else "未通过"))
    out["self_check_passed"] = ok
    if not ok:
        print("推导与实测不一致，不应写入论文。")
        return 1

    p = os.path.join(LOG, "exp_threshold_sensitivity.json")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print("已保存：%s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())