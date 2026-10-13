# -*- coding: utf-8 -*-
"""tools/verify_paper_numbers.py — 终稿数字一致性核对

逐项检查论文（Springer 版）中的关键数字是否与原始实验 JSON 一致。
只做「论文里是否出现该数字」的存在性核对，不修改稿件。
"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "submission_springer", "main.tex")


def main():
    tex = io.open(TEX, encoding="utf-8").read()
    D = lambda *p: json.load(io.open(os.path.join(ROOT, "logs", *p), encoding="utf-8"))

    beh = D("exp_stage3_behavior.json")
    dom = D("exp_stage3_domain_shift.json")
    ab = D("exp_ablation.json")
    th = D("exp_threshold_sensitivity.json")

    seg = beh["A2_YOLOv8n_规则引擎"]["segments"]
    yolo = dom["yolov8n"]
    hog = dom["hog"]
    p10 = th["points"]["10"]
    p15 = th["points"]["15"]
    p20 = th["points"]["20"]

    checks = [
        # (名称, 论文中应出现的字符串, 数据来源说明)
        ("消融 完整系统行", "16 & 0 & 0 & 20/20", "ablation full"),
        ("消融 noreid 行", "12 & 0 & 4 & 16/20", "ablation noreid"),
        ("阈值 15s flagged 归零", "$15$\\,s & $42/42$ & $0/21$", "τ=15s keeps door, loses flagged"),
        ("阈值 20s door 归零", "$0/42$ & $0/21$", "τ=20s"),
        ("门开时长下界", "15.7", "door span min"),
        ("门开时长上界", "18.3", "door span max"),
        ("真实域总帧", "15\\,328", "15328 frames"),
        ("真实域墙钟", "535", "wall seconds"),
        ("真实域处理帧率", "28.6", "proc fps"),
        ("真实域推理中位", "33.0", "infer median"),
        ("告警间隔", "one alarm every $28$\\,s", "1 alarm / 28 s"),
        ("E4 YOLO 均值", "8.13", "yolo persons/frame"),
        ("E4 HOG 检出占比", "7.2\\%", "hog frames with detection"),
        ("E4 HOG 均值", "0.11", "hog persons/frame"),
        ("数据集规模", "119", "119 clips"),
        ("异常:正常", "84", "84 abnormal"),
        ("E2 YOLO AP@0.5", "0.5135", "AP"),
        ("E2 HOG AP@0.5", "0.0051", "AP"),
        ("E2 HOG 召回", "0.019", "recall"),
        ("E1 clip 级召回", "1.000", "A2 P/R/F1"),
    ]

    ok = 0
    miss = []
    for name, token, src in checks:
        hit = token in tex
        ok += hit
        print(("  OK   " if hit else "  MISS ") + "%-20s [%s]  %s" % (name, src, token))
        if not hit:
            miss.append(name)

    print()
    print("核对结果：%d/%d 项在论文中正确出现" % (ok, len(checks)))

    # 数据源侧的自洽性再确认（防止 JSON 本身被改动）
    print()
    print("数据源自洽性：")
    print("  A2 段级混淆 = %s" % json.dumps(seg, ensure_ascii=False)[:120])
    print("  阈值自校验 passed =", th.get("self_check_passed"))
    print("  τ=10 door/flagged = %d/%d，与实测 42/21 一致 = %s"
          % (p10["DOOR_LEFT_OPEN"]["TP"], p10["FLAGGED_DOOR_RETRIEVAL"]["TP"],
             p10["DOOR_LEFT_OPEN"]["TP"] == 42 and p10["FLAGGED_DOOR_RETRIEVAL"]["TP"] == 21))
    print("  τ=15 flagged TP = %d；τ=20 door TP = %d"
          % (p15["FLAGGED_DOOR_RETRIEVAL"]["TP"], p20["DOOR_LEFT_OPEN"]["TP"]))
    print("  Avenue yolo 总帧 = %d，空检出帧 = %d"
          % (yolo["total_frames"], sum(v["frames_without_person"] for v in yolo["per_video"])))
    print("  Avenue hog 总帧 = %d，空检出帧 = %d"
          % (hog["total_frames"], sum(v["frames_without_person"] for v in hog["per_video"])))
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())