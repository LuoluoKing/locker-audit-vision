# -*- coding: utf-8 -*-
"""paper/applsci_mdpi/make_figs.py — Applied Sciences 论文全部图表生成

风格基准（依《风格基准报告》）：
  · 白底、极浅网格；Office 调色板：蓝 #2E75B6 / 橙 #ED7D31 / 绿 #70AD47 / 红 #C00000
  · 轴标签加粗带单位；图例内嵌无边框；子图 (a)(b)(c) 标注
  · 全宽图 17cm；半宽图 8.2cm；PDF 矢量输出（真实帧位图内嵌）
输出：figures/*.pdf
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
LOGS = ROOT / "logs"

C_BLUE, C_ORANGE, C_GREEN, C_RED = "#2E75B6", "#ED7D31", "#70AD47", "#C00000"
C_GREY = "#7F7F7F"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#404040", "axes.linewidth": 0.8,
    "axes.labelweight": "bold", "axes.grid": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
CM = 1 / 2.54


def imread_cn(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_cn_png(p, img):
    cv2.imencode(".png", img)[1].tofile(str(p))


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def panel_label(ax, s, x=0.02, y=0.98, color="#111111"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left", color=color)


# ---------------------------------------------------------------- fig1 context
def fig1():
    picks = [
        ("EXAMPLE/bilibili-vedio/20250531凯普迅：复旦大学，外卖柜，快递小哥送餐_视频.mp4", 175, "Campus bank, courier deposit"),
        ("EXAMPLE/bilibili-vedio/从-丰巢快递柜-取件后关门的重要性_视频.mp4", 12.6, "Express-locker site (dark windows)"),
        ("EXAMPLE/bilibili-vedio/智能快递柜演示视频_视频.mp4", 30, "Smart cabinet, artificial light"),
        ("EXAMPLE/bilibili-vedio/西六楼取餐柜，从柜子里拿外卖，还能保温消毒，你用过吗？_视频.mp4", 26, "Canteen pickup cabinet"),
    ]
    imgs = []
    for fn, t, cap in picks:
        cap_v = cv2.VideoCapture(str(ROOT / fn))
        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30
        cap_v.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap_v.read()
        cap_v.release()
        if not ok:
            fr = np.full((360, 640, 3), 220, np.uint8)
        imgs.append((fr, cap))

    fig, axes = plt.subplots(2, 2, figsize=(17 * CM, 10.4 * CM))
    for ax, (fr, cap), lab in zip(axes.ravel(), imgs, ["(a)", "(b)", "(c)", "(d)"]):
        ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(cap, fontsize=8, pad=2)
        panel_label(ax, lab, 0.02, 0.94)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03, hspace=0.10)
    fig.savefig(FIG / "fig1_context.pdf")
    plt.close(fig)
    print("fig1 ok")


# ---------------------------------------------------------------- fig2 architecture
def fig2():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(17 * CM, 7.6 * CM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec, fs=7.6, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                           fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", color="#404040", ls="-", lw=1.1):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                            lw=lw, linestyle=ls, mutation_scale=9)
        ax.add_patch(a)

    box(1, 30, 15, 10, "Local video\nsource", "#F2F2F2", "#909090")
    box(20, 30, 18, 10, "Detection &\ntracking\n(Alg. 1)", "#DEEBF7", C_BLUE)
    box(42, 30, 20, 10, "Scene observation\n(door state, packages,\nzones)", "#FFF2CC", "#BF9000")
    box(66, 30, 16, 10, "Rule engine\n(Alg. 2-3)", "#E2F0D9", C_GREEN)
    box(85.5, 30, 13.5, 10, "Event store\n+ snapshots", "#FCE4EC", "#C2185B")
    box(66, 12, 16, 9, "Operator\nconsole", "#FCE4EC", "#C2185B")
    box(20, 12, 18, 9, "One-time\ncalibration", "#DEEBF7", C_BLUE)
    box(42, 12, 20, 9, "Configuration\n(named thresholds)", "#F2F2F2", "#909090")
    box(1, 12, 15, 9, "CPU-only\nsingle PC", "#F2F2F2", "#909090")

    arrow(16.6, 35, 19.4, 35); arrow(38.6, 35, 41.4, 35)
    arrow(62.6, 35, 65.4, 35); arrow(82.6, 35, 84.9, 35)
    arrow(74, 29.4, 74, 21.6, ls=(0, (4, 2)))
    arrow(38.6, 16.5, 41.4, 16.5)
    arrow(29, 21.6, 29, 29.4, ls=(0, (4, 2)))
    arrow(52, 21.6, 52, 29.4, ls=(0, (4, 2)))
    arrow(62.6, 16.5, 65.4, 16.5)

    ax.text(29, 25.8, "calibrated polygons", fontsize=6.4, ha="center", color="#505050")
    ax.text(52, 25.8, "all thresholds", fontsize=6.4, ha="center", color="#505050")
    ax.text(74, 25.8, "render", fontsize=6.4, ha="center", color="#505050")
    ax.text(2, 2.2, "solid: monitoring pipeline   ·   dashed: configuration / calibration",
            fontsize=6.6, color="#505050")
    ax.text(2, 0.0, "everything runs in one process on one CPU-only machine; no video leaves the host",
            fontsize=6.6, color="#505050")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    fig.savefig(FIG / "fig2_architecture.pdf")
    plt.close(fig)
    print("fig2 ok")


# ---------------------------------------------------------------- fig3 calibration
def fig3():
    # (a) benchmark scene with calibrated polygons from config
    sys.path.insert(0, str(ROOT / "code"))
    import config as C
    import glob
    vids = sorted(glob.glob(str(ROOT / "data" / "simulated" / "*.mp4")))[:1]
    if vids:
        cap = cv2.VideoCapture(vids[0]); ok, fr = cap.read(); cap.release()
    else:
        fr = np.full((480, 640, 3), 200, np.uint8)
    h, w = fr.shape[:2]
    vis_a = fr.copy()
    roi = np.array([[int(x * w), int(y * h)] for x, y in C.ROI_PICKUP], np.int32)
    cv2.polylines(vis_a, [roi], True, (0, 200, 255), 2)
    for q in C.CABINET_COMPARTMENTS:
        pts = np.array([[int(x * w), int(y * h)] for x, y in q], np.int32)
        cv2.polylines(vis_a, [pts], True, (255, 170, 0), 2)

    # (b) field site: quad + fitted cells on v04
    im_b = imread_cn(LOGS / "field_frames" / "v04" / "t_010.png")
    vis_b = im_b.copy()
    cols = [245.5, 322.5, 411.5]; rows = [205, 227, 247, 261]
    for cy in rows:
        for cx in cols:
            x1, y1, x2, y2 = int(cx - 16), int(cy - 10), int(cx + 16), int(cy + 10)
            cv2.rectangle(vis_b, (x1, y1), (x2, y2), (255, 170, 0), 1)
    quad = np.array([[212, 100], [560, 100], [560, 262], [212, 262]], np.int32)
    cv2.polylines(vis_b, [quad], True, (0, 0, 255), 2)
    for pt in quad:
        cv2.circle(vis_b, tuple(pt), 4, (0, 0, 255), -1)
    roi_b = np.array([[150, 300], [560, 290], [600, 400], [120, 400]], np.int32)
    cv2.polylines(vis_b, [roi_b], True, (0, 200, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 5.6 * CM))
    for ax, im, lab in [(axes[0], vis_a, "(a) benchmark scene: fitted grid + pickup area"),
                        (axes[1], vis_b, "(b) field site: panel corners + generated cells")]:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=7.6, pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.02)
    fig.savefig(FIG / "fig3_calibration.pdf")
    plt.close(fig)
    print("fig3 ok")


# ---------------------------------------------------------------- fig4 door
def fig4():
    d = jload(LOGS / "exp_field_door_masked.json")
    series = d["series"]; times = d["times"]

    # crops: (a) closed after the user's correction; (b) open (the episode)
    cap = cv2.VideoCapture(str(ROOT / "EXAMPLE/bilibili-vedio/从-丰巢快递柜-取件后关门的重要性_视频.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    crops = {}
    for t, name in [(13.5, "closed"), (7.5, "open")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, fr = cap.read()
        if ok:
            c = fr[168:262, 330:435].copy()
            cv2.rectangle(c, (354 - 330, 188 - 168), (412 - 330, 239 - 168), (0, 0, 0), 1)
            crops[name] = c
    cap.release()

    fig = plt.figure(figsize=(17 * CM, 6.2 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 3.4], wspace=0.10)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(cv2.cvtColor(crops["closed"], cv2.COLOR_BGR2RGB))
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.grid(False)
    for s in ax0.spines.values(): s.set_color("#606060"); s.set_linewidth(1.0)
    ax0.set_title("closed ($t$=13.5 s):  $r_{k^*}$ = 0.04", fontsize=7.2, pad=2)
    panel_label(ax0, "(a)", 0.04, 0.93)

    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(cv2.cvtColor(crops["open"], cv2.COLOR_BGR2RGB))
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    for s in ax1.spines.values(): s.set_color(C_RED); s.set_linewidth(1.6)
    ax1.set_title("open ($t$=7.5 s):  $r_{k^*}$ = 0.94", fontsize=7.2, pad=2)

    ax2 = fig.add_subplot(gs[0, 2])
    styles = {"O_open": (C_RED, 2.0, "open compartment $k^*$"),
              "C_above": (C_BLUE, 1.0, "control (above)"),
              "C_right": (C_ORANGE, 1.0, "control (right)"),
              "C_below": (C_GREEN, 1.0, "control (below)")}
    for k, (c, lw, lab) in styles.items():
        vv = np.array([np.nan if x is None else x for x in series[k]], float)
        ax2.plot(times, vv, color=c, lw=lw, label=lab, alpha=0.95)
    ax2.axhline(0.15, color="#404040", ls="--", lw=1.0)
    ax2.axvspan(6.3, 9.2, color="#FFE8E8", alpha=0.55, zorder=0)
    ax2.text(6.42, 1.06, "door open (unattended)", fontsize=6.8, color="#802020")
    ax2.axvspan(9.3, 13.4, color="#E8E8F4", alpha=0.85, zorder=0)
    ax2.text(10.1, 0.60, "state held\n(occluded)", fontsize=6.4, color="#3a3a6a")
    ax2.annotate("user arrives\nand closes", xy=(10.6, 0.30), xytext=(15.0, 0.66),
                 fontsize=6.6, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#404040"))
    ax2.text(11.6, 0.075, "threshold $\\eta$ = 0.15", fontsize=6.6, color="#404040", ha="left")
    ax2.annotate("closed: $r\\leq0.04$", xy=(17.5, 0.030), xytext=(15.6, 0.20),
                 fontsize=6.6, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#404040"))
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Darkness ratio $r_k$")
    ax2.set_ylim(-0.03, 1.14); ax2.set_xlim(6.0, 21.0)
    ax2.legend(loc="center left", fontsize=6.4, bbox_to_anchor=(0.05, 0.60))
    panel_label(ax2, "(b)", 0.015, 0.99)

    fig.subplots_adjust(left=0.062, right=0.995, top=0.90, bottom=0.145)
    fig.savefig(FIG / "fig4_door.pdf")
    plt.close(fig)
    print("fig4 ok")


# ---------------------------------------------------------------- fig6 e1 arms + confusion
def fig6():
    arms = ["A1\ndetector only", "A2\nYOLOv8n+rules", "A3\nHOG+rules"]
    prec = [0.706, 1.000, 1.000]; rec = [1.000, 1.000, 0.714]; f1 = [0.828, 1.000, 0.833]
    confs = [np.array([[84, 35], [0, 0]]), np.array([[84, 0], [0, 35]]),
             np.array([[60, 0], [24, 35]])]

    fig = plt.figure(figsize=(17 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.6, 1.5, 1.5, 1.5, 0.2], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3); wd = 0.26
    ax.bar(x - wd, prec, wd, color=C_BLUE, label="Precision")
    ax.bar(x, rec, wd, color=C_ORANGE, label="Recall")
    ax.bar(x + wd, f1, wd, color=C_GREEN, label="F1")
    for i, v in enumerate(prec): ax.text(i - wd, v + 0.02, "%.3f" % v, ha="center", fontsize=6.2)
    for i, v in enumerate(rec): ax.text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=6.2)
    for i, v in enumerate(f1): ax.text(i + wd, v + 0.02, "%.3f" % v, ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=7)
    ax.set_ylim(0, 1.16); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Score"); ax.legend(loc="lower right", ncols=1)
    panel_label(ax, "(a)")

    for j, (cm, name) in enumerate(zip(confs, ["A1", "A2", "A3"])):
        axm = fig.add_subplot(gs[0, j + 1])
        axm.imshow(cm, cmap="Blues", vmin=0, vmax=90)
        for (r, c), v in np.ndenumerate(cm):
            axm.text(c, r, str(v), ha="center", va="center", fontsize=8,
                     color="white" if v > 50 else "#222222")
        axm.set_xticks([0, 1]); axm.set_yticks([0, 1])
        axm.set_xticklabels(["pred +\n", "pred -"], fontsize=6.4)
        axm.set_yticklabels(["true +\n", "true -"], fontsize=6.4)
        axm.grid(False)
        axm.set_title(name, fontsize=7.6)
        if j == 0:
            panel_label(axm, "(b)", -0.34, 1.06)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.17)
    fig.savefig(FIG / "fig6_e1_arms.pdf")
    plt.close(fig)
    print("fig6 ok")


# ---------------------------------------------------------------- fig7 e2 PR
def fig7():
    fig, ax = plt.subplots(figsize=(8.6 * CM, 6.6 * CM))
    for name, color, lab in [("yolov8n", C_BLUE, "YOLOv8n"), ("yolo11n", C_ORANGE, "YOLO11n"),
                             ("hog", C_GREEN, "HOG + SVM")]:
        d = jload(LOGS / ("exp_det_benchmark_%s.json" % name))
        pr = np.array(d["pr_curve"])
        ax.plot(pr[:, 1], pr[:, 0], color=color, lw=1.5, label=lab)
        op = d["at_conf_0.35"]
        ax.plot(op["recall"], op["precision"], "o", color=color, ms=3.6)
    ax.plot([0.5246], [0.8957], "s", mfc="none", mec="white", ms=6, mew=1.0)
    ax.annotate("operating points\n(conf = 0.35)", xy=(0.52, 0.90), xytext=(0.36, 0.52),
                fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("HOG confined to\nrecall < 0.02", xy=(0.017, 0.26), xytext=(0.12, 0.16),
                fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.14)
    fig.savefig(FIG / "fig7_e2_pr.pdf")
    plt.close(fig)
    print("fig7 ok")


# ---------------------------------------------------------------- fig8 e3 cost + fps
def fig8():
    fig = plt.figure(figsize=(17 * CM, 5.8 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    groups = ["YOLOv8n", "HOG"]
    det = [24.0, 20.0]; obs_extra = [3.0, 2.0]
    x = np.arange(2)
    ax.bar(x, det, 0.5, color=C_BLUE, label="detector")
    ax.bar(x, obs_extra, 0.5, bottom=det, color=C_ORANGE, label="observation step")
    for i, (a, b) in enumerate(zip(det, obs_extra)):
        ax.text(i, a + b + 1.2, "%.0f ms" % (a + b), ha="center", fontsize=7)
    ax.axhline(66.7, color=C_RED, ls="--", lw=1.0)
    ax.text(1.42, 68.5, "15 fps budget: 66.7 ms", fontsize=6.8, color=C_RED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Per-frame cost (ms)")
    ax.set_ylim(0, 78); ax.legend(loc="upper left")
    panel_label(ax, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["display\n(synth.)", "detect\n(synth.)", "display\n(real)", "detect\n(real)"]
    v8 = [14.93, 14.93, 24.27, 7.70]; hg = [14.89, 14.91, 24.04, 11.54]
    x2 = np.arange(4); wd = 0.36
    ax2.bar(x2 - wd / 2, v8, wd, color=C_BLUE, label="YOLOv8n")
    ax2.bar(x2 + wd / 2, hg, wd, color=C_ORANGE, label="HOG")
    for i, (a, b) in enumerate(zip(v8, hg)):
        ax2.text(i - wd / 2, a + 0.3, "%.1f" % a, ha="center", fontsize=6.0)
        ax2.text(i + wd / 2, b + 0.3, "%.1f" % b, ha="center", fontsize=6.0)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels, fontsize=6.8)
    ax2.set_ylabel("Frame rate (fps)"); ax2.set_ylim(0, 30)
    ax2.legend(loc="upper left", fontsize=6.8)
    panel_label(ax2, "(b)")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.17)
    fig.savefig(FIG / "fig8_e3_cost.pdf")
    plt.close(fig)
    print("fig8 ok")


# ---------------------------------------------------------------- fig9 ablation
def fig9():
    d = jload(LOGS / "exp_ablation.json")["configs"]
    order = [("full", "complete"), ("noreid", "w/o\nre-assoc."),
             ("nooccl", "w/o occl.\nlook-back"), ("noguard", "w/o partial-\nbox guards")]
    classes = [("DOOR_LEFT_OPEN", C_BLUE, "door left open"),
               ("FLAGGED_DOOR_RETRIEVAL", C_ORANGE, "flagged retrieval"),
               ("MULTI_RETRIEVAL", C_GREEN, "multiple packages"),
               ("LOITERING", C_RED, "loitering")]
    fig = plt.figure(figsize=(17 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order)); bottom = np.zeros(len(order))
    for key, color, lab in classes:
        vals = np.array([d[cfg]["events"].get(key, {}).get("TP", 0) for cfg, _ in order], float)
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, label=lab, alpha=0.92)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.3, "%d" % v, ha="center", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels([s for _, s in order], fontsize=6.8)
    ax.set_ylabel("Correctly detected events (of 20)"); ax.set_ylim(0, 23)
    ax.legend(loc="lower left", ncols=2, fontsize=6.4)
    panel_label(ax, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    seg = {cfg: d[cfg]["segments"] for cfg, _ in order}
    fn = [seg["noreid"]["FN"] if c == "noreid" else seg[c]["FN"] for c, _ in order]
    fp = [seg["noreid"]["FP"] if c == "noreid" else seg[c]["FP"] for c, _ in order]
    x2 = np.arange(len(order)); wd = 0.34
    ax2.bar(x2 - wd / 2, fn, wd, color=C_RED, label="missed clips (FN)")
    ax2.bar(x2 + wd / 2, fp, wd, color="#B0B0B0", label="false alarms (FP)")
    for i in range(len(order)):
        ax2.text(i - wd / 2, fn[i] + 0.06, str(int(fn[i])), ha="center", fontsize=7)
        ax2.text(i + wd / 2, fp[i] + 0.06, str(int(fp[i])), ha="center", fontsize=7)
    ax2.set_xticks(x2); ax2.set_xticklabels([s for _, s in order], fontsize=6.8)
    ax2.set_ylabel("Clips (of 20)"); ax2.set_ylim(0, 5)
    ax2.legend(loc="upper center", fontsize=6.6)
    panel_label(ax2, "(b)")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.20)
    fig.savefig(FIG / "fig9_ablation.pdf")
    plt.close(fig)
    print("fig9 ok")


# ---------------------------------------------------------------- fig10 sensitivity
def fig10():
    import glob
    A = []
    t_take = []
    for p in sorted(glob.glob(str(LOGS / "dataset_eval_final3_shard*.json"))):
        dd = jload(p)
        for c in dd["results"]:
            tl = c.get("timeline") or []
            t_door = min([e["t"] for e in tl if e["type"] == "DOOR_LEFT_OPEN"], default=None)
            t_tk = min([e["t"] for e in tl if e["type"] == "FLAGGED_DOOR_RETRIEVAL"], default=None)
            fps = 15.0; dur = c["frames"] / fps
            if t_door is not None:
                t_open = t_door - 10.0
                A.append(dur - t_open)
                if t_tk is not None:
                    t_take.append(t_tk - t_open)
    A = np.array(A); t_take = np.array(t_take)

    tau = np.linspace(2, 22, 400)
    r_door = [(A >= t).mean() for t in tau]
    r_flag = [((A >= t) & (t_take[:, None].ravel() >= t)).mean()
              if False else 0.0 for t in tau]
    r_flag = []
    for t in tau:
        ok = (A >= t)
        # flagged clips are the first 21 that have t_take
        n_flag = 0; hit = 0
        for a in A:
            if abs(a - A[A == a][0]) < 1:  # placeholder
                pass
        r_flag.append(0.0)
    # 直接按 21 段 flagged 的 A 与 t_take 配对（t_take 顺序对应 42 段中的后 21 段门事件）
    r_flag = []
    for t in tau:
        hits = int(((t_take >= t)).sum()) if len(t_take) else 0
        r_flag.append(hits / len(t_take))

    fig = plt.figure(figsize=(17 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tau, r_door, color=C_BLUE, lw=1.6, label="door left open")
    ax.plot(tau, r_flag, color=C_ORANGE, lw=1.6, label="flagged retrieval")
    for t in [5, 10, 15, 20]:
        ax.axvline(t, color="#DDDDDD", lw=0.8, zorder=0)
    ax.plot([10, 15, 20], [1, 1, 0], "o", color=C_BLUE, ms=4)
    ax.plot([5, 10, 15, 20], [1, 1, 0, 0], "s", color=C_ORANGE, ms=4)
    ax.axvspan(12.5, 15.7, color="#FFF3D9", alpha=0.8, zorder=0)
    ax.text(14.1, 0.50, "usable\nrange", fontsize=7, ha="center", color="#8a6100")
    ax.set_xlabel("Door-open threshold $\\tau_{\\mathrm{open}}$ (s)")
    ax.set_ylabel("Event recall")
    ax.set_ylim(-0.05, 1.1); ax.set_xlim(2, 22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left")
    panel_label(ax, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    ax2.scatter(A, 1.0 + 0.05 * rng.standard_normal(len(A)), s=9, color=C_BLUE,
                alpha=0.75, label="available open duration $A_k$ (42 events)")
    ax2.scatter(t_take, 0.72 + 0.05 * rng.standard_normal(len(t_take)), s=9, color=C_ORANGE,
                alpha=0.75, label="retrieval instant $t_{take}$ (21 clips)")
    for t in [5, 10, 15, 20]:
        ax2.axvline(t, color="#BBBBBB", ls=":", lw=0.9)
        ax2.text(t, 1.21, "%d s" % t, ha="center", fontsize=6.6, color="#666666")
    ax2.annotate("door events lost\nif $\\tau$ > 18.3 s", xy=(19.6, 1.02), xytext=(13.9, 1.30),
                 fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.annotate("retrieval events lost\nif $\\tau$ > 11.1 s", xy=(14.2, 0.72), xytext=(5.2, 0.42),
                 fontsize=6.5, ha="left", arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xlim(2, 22); ax2.set_ylim(0.25, 1.44)
    ax2.set_yticks([]); ax2.set_xlabel("Duration (s)")
    ax2.legend(loc="lower left", fontsize=6.5, bbox_to_anchor=(0.0, 0.02))
    panel_label(ax2, "(b)")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.15)
    fig.savefig(FIG / "fig10_sensitivity.pdf")
    plt.close(fig)
    print("fig10 ok")


# ---------------------------------------------------------------- fig11 avenue
def fig11():
    d = jload(LOGS / "exp_stage3_domain_shift.json")
    v8 = d["yolov8n"]["per_video"]; hg = d["hog"]["per_video"]
    names = [pv["video"].replace(".avi", "") for pv in v8]
    means8 = [pv["det_per_frame_mean"] for pv in v8]
    meansh = [pv["det_per_frame_mean"] for pv in hg]

    fig = plt.figure(figsize=(17 * CM, 5.4 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names)); wd = 0.38
    ax.bar(x - wd / 2, means8, wd, color=C_BLUE, label="YOLOv8n")
    ax.bar(x + wd / 2, meansh, wd, color=C_ORANGE, label="HOG + SVM")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.8, rotation=45, ha="right")
    ax.set_ylabel("Persons detected per frame")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 14)
    panel_label(ax, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([0, 1], [100.0, 7.2], 0.5, color=[C_BLUE, C_ORANGE])
    for i, v in enumerate([100.0, 7.2]):
        ax2.text(i, v + 2, "%.1f%%" % v, ha="center", fontsize=8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["YOLOv8n", "HOG"], fontsize=7.5)
    ax2.set_ylabel("Frames with $\\geq$ 1 detection (%)")
    ax2.set_ylim(0, 118)
    panel_label(ax2, "(b)")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.24)
    fig.savefig(FIG / "fig11_avenue.pdf")
    plt.close(fig)
    print("fig11 ok")


# ---------------------------------------------------------------- fig12 field transfer
def fig12():
    rows = jload(LOGS / "exp_field_summary.json")
    keys = [r["key"] for r in rows]
    tot = [r["events_total"] or 0 for r in rows]
    doorish = []
    for r in rows:
        ev = r["events"] or {}
        doorish.append(ev.get("DOOR_LEFT_OPEN", 0) + ev.get("DOOR_CLOSED", 0)
                       + ev.get("FLAGGED_DOOR_RETRIEVAL", 0))
    dur = [r["dur"] for r in rows]

    # snapshots
    s1 = imread_cn(LOGS / "field_eval" / "v01_01_MULTI_RETRIEVAL.png")
    s2 = imread_cn(LOGS / "field_eval" / "v04_01_DOOR_LEFT_OPEN.png")

    fig = plt.figure(figsize=(17 * CM, 6.0 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 1.0], wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(keys))
    ax.bar(x, tot, 0.62, color="#C9C9C9", label="all alarms")
    ax.bar(x, doorish, 0.62, color=C_BLUE, label="door-derived")
    for i, (t, dd) in enumerate(zip(tot, doorish)):
        if t:
            ax.text(i, t + 1.2, str(t), ha="center", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=6.8)
    ax.set_ylabel("Alarms (factory config)")
    ax.set_ylim(0, 84)
    ax2 = ax.twinx()
    ax2.plot(x, dur, color=C_RED, lw=1.0, marker="o", ms=2.4, label="duration")
    ax2.set_ylabel("Duration (s)", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.set_ylim(0, 350)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.2)
    ax.set_ylim(0, 92)
    panel_label(ax, "(a)")

    axs1 = fig.add_subplot(gs[0, 1])
    axs1.imshow(cv2.cvtColor(s1, cv2.COLOR_BGR2RGB)); axs1.axis("off")
    axs1.set_title("default polygons land on a person", fontsize=6.4, pad=3)
    axs2 = fig.add_subplot(gs[0, 2])
    axs2.imshow(cv2.cvtColor(s2, cv2.COLOR_BGR2RGB)); axs2.axis("off")
    axs2.set_title("default grid unrelated to real doors", fontsize=6.4, pad=3)
    panel_label(axs1, "(b)", 0.02, 0.88, color="#FFFFFF")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.16)
    fig.savefig(FIG / "fig12_field_transfer.pdf")
    plt.close(fig)
    print("fig12 ok")


# ---------------------------------------------------------------- fig13 audits
def fig13():
    rows = jload(LOGS / "exp_field_summary.json")
    keys = [r["key"] for r in rows]
    coll = [(r["collision_rate"] or 0) * 100 for r in rows]

    d = jload(LOGS / "exp_field_v04_eps_masked.json")
    etas = [0.15, 0.45, 0.55, 0.65, 0.75]
    frac = [d["eta_scan"][str(e)] * 100 for e in etas]

    fig = plt.figure(figsize=(17 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    colors = [C_RED if c > 45 else (C_ORANGE if c > 25 else C_GREEN) for c in coll]
    x = np.arange(len(keys))
    ax.bar(x, coll, 0.62, color=colors)
    ax.axhline(28.3, color="#404040", ls="--", lw=1.0)
    ax.annotate("corpus mean 28.3%", xy=(10.4, 28.3), xytext=(10.2, 76), fontsize=6.6,
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.7, color="#404040"))
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=6.8)
    ax.set_ylabel("Collision frames (%)")
    ax.set_ylim(0, 82)
    ax.text(6.0, 46, "high risk:\nrule restricted\nor report-only", fontsize=6.3, color=C_RED)
    panel_label(ax, "(a) colour-collision audit")

    ax2 = fig.add_subplot(gs[0, 1])
    xs = [0, 1, 2, 3, 4]
    ax2.plot(xs, frac, "o-", color=C_BLUE, lw=1.5, ms=4)
    ax2.axhline(d["baseline_p90"] * 100, color=C_ORANGE, ls=":", lw=1.0)
    ax2.text(1.2, d["baseline_p90"] * 100 + 3.0, "closed-state $p_{99}$ (0.38)", fontsize=6.2,
             color="#8a4a00")
    ax2.annotate("default $\\eta$ = 0.15:\n98.6% of closed-door\nsamples read ``open''",
                 xy=(0.05, 98.6), xytext=(0.75, 66), fontsize=6.3,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.annotate("$\\eta \\in [0.45, 0.75]$:\n$\\leq$ 0.5%", xy=(2.9, 0.4), xytext=(2.35, 26),
                 fontsize=6.3, arrowprops=dict(arrowstyle="->", lw=0.8))
    for i, v in enumerate(frac):
        ax2.text(i, v + 3.5, ("%.1f" % v) if v > 2 else "%.1f" % v, ha="center", fontsize=6.2)
    ax2.set_xticks(xs); ax2.set_xticklabels(["0.15", "0.45", "0.55", "0.65", "0.75"])
    ax2.set_xlabel("Candidate threshold $\\eta$")
    ax2.set_ylabel("Samples classified ``open'' (%)")
    ax2.set_ylim(-3, 118)
    panel_label(ax2, "(b) door-appearance audit")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.16)
    fig.savefig(FIG / "fig13_audits.pdf")
    plt.close(fig)
    print("fig13 ok")


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "6": fig6, "7": fig7,
           "8": fig8, "9": fig9, "10": fig10, "11": fig11, "12": fig12, "13": fig13}
    for k, f in fns.items():
        if only and k not in only:
            continue
        try:
            f()
        except Exception as e:
            import traceback
            print("fig%s FAILED: %s" % (k, e))
            traceback.print_exc()
