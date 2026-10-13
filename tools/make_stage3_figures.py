# -*- coding: utf-8 -*-
"""
tools/make_stage3_figures.py — 生成阶段3 论文用图（全部由真实实验结果绘制，不做任何修饰）

输入（全部由本机真实运行产出）
------------------------------
* logs/exp_stage3_behavior.json       三组对照（A1/A2/A3）段级与事件级指标
* logs/exp_det_benchmark_yolov8n.json COCO val2017 person 子集检测指标（YOLOv8n）
* logs/exp_det_benchmark_hog.json      同上（HOG + 线性 SVM）
* logs/exp_stage3_realtime.json        实时链路实测帧率/单帧耗时

输出 → docs/09_阶段3证据/
-------------------------
    fig_confusion_matrix.png   段级混淆矩阵（完整系统 A2）
    fig_arms_bar.png           三组对照 P/R/F1/准确率
    fig_event_bars.png         事件级各类别 F1（A2/A3）
    fig_pr_curve_coco.png      COCO person 子集 PR 曲线（两后端）
    fig_realtime_bar.png       实时性能对比（显示帧率 / 检测帧率 / 单帧耗时）

命令行
------
    python tools/make_stage3_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402

OUT = C.BASE_DIR / "docs" / "09_阶段3证据"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160


def _load(name: str) -> dict:
    p = C.LOG_DIR / name
    if not p.exists():
        print("缺少输入，跳过：%s" % p)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def fig_confusion(arms: dict) -> None:
    a2 = arms.get("A2_YOLOv8n_规则引擎", {}).get("segments")
    if not a2:
        return
    m = np.array([[a2["TP"], a2["FN"]], [a2["FP"], a2["TN"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%d" % m[i, j], ha="center", va="center",
                    fontsize=15, color="white" if m[i, j] > m.max() * 0.6 else "#1a1a1a")
    ax.set_xticks([0, 1], ["判为异常", "判为正常"])
    ax.set_yticks([0, 1], ["实际异常", "实际正常"])
    ax.set_xlabel("系统输出")
    ax.set_ylabel("数据集真值")
    ax.set_title("段级混淆矩阵（YOLOv8n + 规则引擎，140 段）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion_matrix.png")
    plt.close(fig)
    print("已生成 fig_confusion_matrix.png")


def fig_arms_bar(arms: dict) -> None:
    names = ["A1_仅YOLOv8n检测（无规则）", "A2_YOLOv8n_规则引擎", "A3_HOG_规则引擎"]
    labels = ["A1 仅检测器\n（无规则）", "A2 YOLOv8n\n+ 规则", "A3 HOG\n+ 规则"]
    keys = ["precision", "recall", "f1", "accuracy", "false_alarm_rate"]
    cn = ["精确率", "召回率", "F1", "准确率", "误报率"]
    data = []
    used_labels = []
    for n, lb in zip(names, labels):
        if n in arms:
            data.append([arms[n]["segments"][k] for k in keys])
            used_labels.append(lb)
    if not data:
        return
    data = np.array(data)
    x = np.arange(len(keys))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i in range(len(data)):
        off = x + i * w - 0.4 + w / 2
        ax.bar(off, data[i], w, label=used_labels[i])
        for xi, v in zip(off, data[i]):
            ax.text(xi, v + 0.015, "%.3f" % v, ha="center", fontsize=7.5)
    ax.set_xticks(x, cn)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("指标值")
    ax.set_title("行为级三组对照（合成数据集 140 段，段级指标）", pad=34)
    # 图例置于标题下方、坐标区上方，避免遮挡柱体、数值标注与标题
    ax.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_arms_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig_arms_bar.png")


def fig_event_bars(arms: dict) -> None:
    series = []
    for name, tag in (("A2_YOLOv8n_规则引擎", "A2 YOLOv8n+规则"), ("A3_HOG_规则引擎", "A3 HOG+规则")):
        ev = arms.get(name, {}).get("events")
        if ev:
            series.append((tag, {r["event_type"]: r for r in ev["per_type"]}))
    if not series:
        return
    types = sorted({t for _tag, d in series for t in d})
    cn = {"LOITERING": "徘徊观察", "DOOR_LEFT_OPEN": "柜门未关",
          "MULTI_RETRIEVAL": "一人取多份", "FLAGGED_DOOR_RETRIEVAL": "取走未关柜门外卖"}
    x = np.arange(len(types))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for i, (tag, d) in enumerate(series):
        off = x + i * w - 0.4 + w / 2
        vals = [d.get(t, {}).get("f1", 0.0) for t in types]
        ax.bar(off, vals, w, label=tag)
        for xi, v in zip(off, vals):
            ax.text(xi, v + 0.025, "%.2f" % v, ha="center", fontsize=8)
    ax.set_xticks(x, [cn.get(t, t) for t in types])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("事件级 F1")
    ax.set_title("异常事件分类 F1（事件级，合成数据集）", pad=30)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_event_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig_event_bars.png")


def fig_pr_curve() -> None:
    y = _load("exp_det_benchmark_yolov8n.json")
    h = _load("exp_det_benchmark_hog.json")
    if not y or not h:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for d, tag, color in ((y, "YOLOv8n", "#1f77b4"), (h, "HOG + 线性SVM", "#d62728")):
        cur = d.get("pr_curve") or []
        if not cur:
            continue
        rec = [c[1] for c in cur]
        pre = [c[2] for c in cur]
        ax.plot(rec, pre, "-", color=color, lw=2,
                label="%s（AP@0.5=%.4f）" % (tag, d["AP@0.5"]))
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("COCO val2017 person 子集 PR 曲线（真实图像，500 张）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pr_curve_coco.png")
    plt.close(fig)
    print("已生成 fig_pr_curve_coco.png")


def fig_realtime() -> None:
    rt = _load("exp_stage3_realtime.json").get("runs")
    if not rt:
        return
    labels = ["%s\n%s" % (r["backend"], r.get("source_kind", "")) for r in rt]
    x = np.arange(len(rt))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    ax.bar(x - 0.2, [r["display_fps"] for r in rt], 0.4, label="显示帧率")
    ax.bar(x + 0.2, [r["detect_fps"] for r in rt], 0.4, label="检测帧率")
    for xi, r in zip(x, rt):
        ax.text(xi - 0.2, r["display_fps"] + 0.3, "%.1f" % r["display_fps"], ha="center", fontsize=7.5)
        ax.text(xi + 0.2, r["detect_fps"] + 0.3, "%.1f" % r["detect_fps"], ha="center", fontsize=7.5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, max(r["display_fps"] for r in rt) * 1.35)
    ax.set_ylabel("帧率 (fps)")
    ax.set_title("实时链路帧率（CPU，服务运行态）")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    ax.bar(x, [r["infer_ms"] for r in rt], 0.5, color="#ff7f0e")
    for xi, r in zip(x, rt):
        ax.text(xi, r["infer_ms"] + 0.8, "%.1f ms" % r["infer_ms"], ha="center", fontsize=7.5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, max(r["infer_ms"] for r in rt) * 1.3)
    ax.set_ylabel("单帧推理耗时 (ms)")
    ax.set_title("单帧推理耗时（CPU）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_realtime_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig_realtime_bar.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = _load("exp_stage3_behavior.json")
    if arms:
        fig_confusion(arms)
        fig_arms_bar(arms)
        fig_event_bars(arms)
    fig_pr_curve()
    fig_realtime()
    print("图表目录：%s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())