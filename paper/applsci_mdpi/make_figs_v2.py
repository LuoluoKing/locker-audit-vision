# -*- coding: utf-8 -*-
"""paper/applsci_mdpi/make_figs_v2.py — 图表全面重做（v2，参照用户提供范例图）

风格基准（参照 EXAMPLE/ 范例图）：
  · 真实图像直接嵌入流程图/对比图（不是纯文字方框）
  · 粗浅色箭头（#8FC7E8 系）+ 虚线圆角容器分组
  · 面板标签 (a)(b)(c) 统一放在每个面板**下方居中**（无正文压图）
  · 图像面板：细黑边框、紧密网格、白色间隔
  · 图表：白底、极浅网格、Office 调色板、轴标签加粗带单位、图例不压数据
  · 输出：PDF（矢量，供正文）+ PNG 300dpi、≥2200px 宽（供 figures.zip 与 GA）
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
LOGS = ROOT / "logs"

C_BLUE, C_ORANGE, C_GREEN, C_RED = "#2E75B6", "#ED7D31", "#70AD47", "#C00000"
C_LBLUE, C_LGREEN, C_LORANGE = "#DEEBF7", "#E2F0D9", "#FFF2CC"
C_ARR = "#7FB6E0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.8,
    "axes.labelweight": "bold", "axes.grid": True,
    "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
CM = 1 / 2.54


def imread_cn(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def grab_frame(rel, t):
    cap = cv2.VideoCapture(str(ROOT / rel))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else np.full((360, 640, 3), 210, np.uint8)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, png=True):
    fig.savefig(FIG / (name + ".pdf"))
    if png:
        fig.savefig(FIG / (name + ".png"), dpi=300)
    plt.close(fig)
    print(name, "ok")


def label_below(ax, s, dy=-0.14):
    """把 (a)/(b)/... 标签放在面板下方居中（参照范例图样式）。"""
    ax.text(0.5, dy, s, transform=ax.transAxes, fontsize=8.2,
            ha="center", va="top", color="#222222")


def img_panel(ax, im, border=True):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(border)
        sp.set_color("#111111"); sp.set_linewidth(0.8)


# ================================================================ fig1 context
def fig1():
    picks = [
        ("EXAMPLE/bilibili-vedio/智能快递柜演示视频_视频.mp4", 40,
         "(a) smart cabinet in use"),
        ("EXAMPLE/bilibili-vedio/芜湖高校流行智能外卖柜_视频.mp4", 8,
         "(b) campus locker row"),
        ("EXAMPLE/bilibili-vedio/常熟理工学院两校区外卖柜已到位！_视频.mp4", 25,
         "(c) campus deployment"),
        ("EXAMPLE/bilibili-vedio/无接触取餐 外卖“快递柜”扩大铺设，你接受吗？_视频.mp4", 35,
         "(d) corridor pickup cabinets"),
    ]

    def crop169(fr):
        h, w = fr.shape[:2]
        target = 16 / 9
        if w / h > target:                      # 过宽：裁两侧
            nw = int(h * target); x0 = (w - nw) // 2; return fr[:, x0:x0 + nw]
        else:                                   # 过高：裁上下（居中偏上，保留柜体）
            nh = int(w / target); y0 = int((h - nh) * 0.35); return fr[y0:y0 + nh, :]

    fig, axes = plt.subplots(1, 4, figsize=(17.4 * CM, 4.6 * CM))
    for ax, (rel, t, cap) in zip(axes, picks):
        fr = crop169(grab_frame(rel, t))
        img_panel(ax, fr)
        ax.text(0.5, -0.06, cap, transform=ax.transAxes, fontsize=7.4,
                ha="center", va="top", color="#222222")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.105, wspace=0.035)
    save(fig, "fig1_context", png=True)


# ================================================================ fig2 architecture
def fig2():
    fig, ax = plt.subplots(figsize=(17.4 * CM, 8.6 * CM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off"); ax.grid(False)

    def thumb(im, x, y, w):
        """把真实帧作为缩略图嵌入画布（保持比例，细边框）。"""
        h = w * im.shape[0] / im.shape[1]
        a = fig.add_axes([x / 100, y / 50, w / 100, h / 50])
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color("#111111"); sp.set_linewidth(0.8)
        return h

    def rbox(x, y, w, h, fc, ec, ls="solid", lw=1.1, r=1.4):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.5,rounding_size=%s" % r,
                           fc=fc, ec=ec, lw=lw, linestyle=ls)
        ax.add_patch(p)

    def arrow(x1, y1, x2, y2, lw=3.2, color=C_ARR):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="simple,head_width=7,head_length=5",
                            color=color, lw=0, mutation_scale=1)
        ax.add_patch(a)

    def txt(x, y, s, fs=7.6, weight="bold", color="#1a1a1a", ha="center"):
        ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color, ha=ha,
                va="center", linespacing=1.4)

    # 缩略图素材：合成基准场景的一帧
    import glob as _glob
    _vids = sorted(_glob.glob(str(ROOT / "data" / "simulated" / "*.mp4")))
    fr_src = grab_frame(str(Path(_vids[0]).relative_to(ROOT)).replace("\\", "/"), 2.0) if _vids \
        else np.full((480, 640, 3), 205, np.uint8)

    # 上排：监控流水线（容器内：标签在顶部，缩略图在下方，互不重叠）
    rbox(1, 30, 13, 15.5, "#F4F4F4", "#8a8a8a")
    txt(7.5, 42.6, "local video\nsource", fs=7.2)
    thumb(fr_src, 2.6, 31.4, 9.8)

    arrow(15.5, 37.8, 19.0, 37.8)
    rbox(19.5, 30, 15, 15.5, C_LBLUE, C_BLUE)
    txt(27, 43.0, "detection &\ntracking", fs=7.3)
    det_vis = fr_src.copy()
    h_, w_ = det_vis.shape[:2]
    cv2.rectangle(det_vis, (int(w_*0.38), int(h_*0.25)), (int(w_*0.62), int(h_*0.9)), (0, 200, 0), 2)
    thumb(det_vis, 21.4, 31.4, 10.2)

    arrow(35.5, 37.8, 39.0, 37.8)
    rbox(39.5, 30, 17, 15.5, C_LORANGE, "#BF9000")
    txt(48, 43.4, "scene observation", fs=7.0)
    txt(48, 41.3, "door, package and", fs=6.2)
    txt(48, 39.7, "zone evidence", fs=6.2)
    obs = fr_src.copy()
    hh, ww = obs.shape[:2]
    cv2.rectangle(obs, (int(ww*0.30), int(hh*0.30)), (int(ww*0.70), int(hh*0.70)), (255, 170, 0), 2)
    thumb(obs, 43.0, 31.4, 9.0)

    arrow(57.5, 37.8, 61.0, 37.8)
    rbox(61.5, 30, 15, 15.5, C_LGREEN, C_GREEN)
    txt(69, 41.8, "rule engine", fs=7.6)
    txt(69, 36.2, "6 event types\nexplicit thresholds", fs=6.4, weight="normal")

    arrow(77.5, 37.8, 81.0, 37.8)
    rbox(81.5, 30, 17.5, 15.5, "#FCE4EC", "#C2185B")
    txt(90.2, 41.8, "event store", fs=7.6)
    txt(90.2, 36.0, "audited alarm:\ncompartment + quantity\n+ threshold + snapshot",
        fs=6.3, weight="normal")

    # 下排：标定与配置
    rbox(19.5, 8, 15, 12.5, C_LBLUE, C_BLUE, ls=(0, (4, 2)))
    txt(27, 18.4, "one-time\ncalibration", fs=7.0)
    cal = fr_src.copy()
    hh, ww = cal.shape[:2]
    for gy in range(3):
        for gx in range(3):
            x1 = int(ww*(0.32+0.12*gx)); y1 = int(hh*(0.30+0.20*gy))
            cv2.rectangle(cal, (x1, y1), (x1+int(ww*0.10), y1+int(hh*0.17)), (255, 170, 0), 1)
    thumb(cal, 21.4, 9.2, 10.2)

    rbox(39.5, 8, 17, 12.5, "#F4F4F4", "#8a8a8a", ls=(0, (4, 2)))
    txt(48, 14.2, "configuration\n(named thresholds)", fs=7.0)

    rbox(61.5, 8, 15, 12.5, "#FCE4EC", "#C2185B")
    txt(69, 14.2, "operator\nconsole", fs=7.2)

    # 虚线连接
    a = FancyArrowPatch((69, 20.8), (69, 29.4), arrowstyle="-|>", color="#8a8a8a",
                        lw=1.0, linestyle=(0, (3, 2)), mutation_scale=8)
    ax.add_patch(a)
    txt(70.6, 25.1, "render", fs=6.3, weight="normal", ha="left")

    arrow(35.5, 14.2, 39.0, 14.2, lw=2.6)
    arrow(57.5, 14.2, 61.0, 14.2, lw=2.6)
    a2 = FancyArrowPatch((27, 20.8), (27, 29.4), arrowstyle="-|>", color="#8a8a8a",
                         lw=1.0, linestyle=(0, (3, 2)), mutation_scale=8)
    ax.add_patch(a2)
    txt(28.4, 25.1, "calibrated polygons", fs=6.3, weight="normal", ha="left")
    a3 = FancyArrowPatch((48, 20.8), (48, 29.4), arrowstyle="-|>", color="#8a8a8a",
                         lw=1.0, linestyle=(0, (3, 2)), mutation_scale=8)
    ax.add_patch(a3)
    txt(49.4, 25.1, "all thresholds", fs=6.3, weight="normal", ha="left")

    txt(1, 3.2, "solid: monitoring pipeline   ·   dashed: calibration / configuration   ·   "
                "one process, one CPU-only machine, no video leaves the host",
        fs=6.4, weight="normal", color="#555555", ha="left")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "fig2_architecture", png=True)


# ================================================================ fig3 calibration
def fig3():
    sys.path.insert(0, str(ROOT / "code"))
    import config as C
    import glob
    vids = sorted(glob.glob(str(ROOT / "data" / "simulated" / "*.mp4")))
    fr = grab_frame(str(Path(vids[0]).relative_to(ROOT)).replace("\\", "/"), 2.0) if vids else \
        np.full((480, 640, 3), 200, np.uint8)
    h, w = fr.shape[:2]

    a = fr.copy()
    b = fr.copy()
    cv2.polylines(b, [np.array([[int(x*w), int(y*h)] for x, y in C.ROI_PICKUP], np.int32)],
                  True, (0, 200, 255), 2)
    c = b.copy()
    for q in C.CABINET_COMPARTMENTS:
        cv2.polylines(c, [np.array([[int(x*w), int(y*h)] for x, y in q], np.int32)],
                      True, (255, 170, 0), 2)

    fb = imread_cn(LOGS / "field_frames" / "v04" / "t_010.png")
    fh, fw = fb.shape[:2]
    nw = int(fh * 4 / 3)                    # 与合成素材同为 4:3，避免面板高矮不一
    fx = (fw - nw) // 2
    fb = fb[:, fx:fx + nw]
    d = fb.copy()
    # 坐标随裁剪平移
    quad = np.array([[212 - fx, 100], [560 - fx, 100], [560 - fx, 262], [212 - fx, 262]], np.int32)
    d = fb.copy()
    cv2.polylines(d, [quad], True, (0, 0, 255), 2)
    for pt in quad:
        cv2.circle(d, tuple(pt), 4, (0, 0, 255), -1)
    for cy_ in [205, 227, 247, 261]:
        for cx_ in [245.5, 322.5, 411.5]:
            cv2.rectangle(d, (int(cx_-16-fx), int(cy_-10)), (int(cx_+16-fx), int(cy_+10)), (255, 170, 0), 1)

    fig, axes = plt.subplots(1, 4, figsize=(17.4 * CM, 5.0 * CM))
    for ax, im, cap in zip(axes, [a, b, c, d],
                           ["(a) input frame", "(b) + pickup area", "(c) + compartment grid",
                            "(d) field site: corners + cells"]):
        img_panel(ax, im)
        label_below(ax, cap)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.115, wspace=0.03)
    save(fig, "fig3_calibration", png=True)


# ================================================================ fig4 door
def fig4():
    d = jload(LOGS / "exp_field_door_masked.json")
    series = d["series"]; times = d["times"]
    rel = "EXAMPLE/bilibili-vedio/从-丰巢快递柜-取件后关门的重要性_视频.mp4"

    f_open = grab_frame(rel, 7.5)
    f_closed = grab_frame(rel, 13.5)
    f_occ = grab_frame(rel, 10.2)

    def mark(im, color):
        v = im.copy()
        cv2.rectangle(v, (354, 188), (412, 239), color, 2)
        return v[150:280, 300:465]

    fig = plt.figure(figsize=(17.4 * CM, 8.6 * CM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.30], hspace=0.50, wspace=0.20)

    for j, (im, cap, col) in enumerate([
            (mark(f_open, (0, 0, 230)), "(a) door open", (0, 0, 230)),
            (mark(f_closed, (60, 60, 60)), "(b) closed afterwards", None),
            (mark(f_occ, (200, 120, 0)), "(c) occluded (state held)", None)]):
        ax = fig.add_subplot(gs[0, j])
        img_panel(ax, im)
        label_below(ax, cap)

    ax = fig.add_subplot(gs[1, :])
    styles = {"O_open": (C_RED, 2.2, "open compartment $k^*$"),
              "C_above": (C_BLUE, 1.1, "control (above)"),
              "C_right": (C_ORANGE, 1.1, "control (right)"),
              "C_below": (C_GREEN, 1.1, "control (below)")}
    for k, (c, lw, lab) in styles.items():
        vv = np.array([np.nan if x is None else x for x in series[k]], float)
        ax.plot(times, vv, color=c, lw=lw, label=lab)
    ax.axhline(0.15, color="#404040", ls="--", lw=1.0)
    ax.axvspan(6.3, 9.2, color="#FFE8E8", alpha=0.55, zorder=0)
    ax.axvspan(9.3, 13.4, color="#E9E9F6", alpha=0.9, zorder=0)
    ax.text(7.05, 1.24, "door open,\nunattended", fontsize=6.6, color="#802020",
            ha="center", va="top")
    ax.text(11.3, 1.24, "occluded: state held", fontsize=6.6, color="#3a3a6a",
            ha="center", va="top")
    ax.text(17.3, 0.245, "closed: $r\\leq$ 0.04", fontsize=6.6, color="#333333", ha="center")
    ax.text(20.6, 0.30, "threshold $\\eta$ = 0.15", fontsize=6.4, color="#404040",
            ha="right", bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Darkness ratio $r_k$")
    ax.set_ylim(-0.03, 1.26); ax.set_xlim(6.0, 21.0)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20])
    ax.legend(loc="center left", fontsize=6.6, bbox_to_anchor=(0.24, 0.60))
    fig.text(0.5, 0.015, "(d) darkness ratio: open compartment vs. three controls",
             fontsize=8.2, ha="center", va="bottom", color="#222222")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.985, bottom=0.185)
    save(fig, "fig4_door", png=True)


# ================================================================ fig6 e1
def fig6():
    arms = ["A1", "A2", "A3"]
    prec = [0.706, 1.000, 1.000]; rec = [1.000, 1.000, 0.714]; f1 = [0.828, 1.000, 0.833]
    confs = [np.array([[84, 35], [0, 0]]), np.array([[84, 0], [0, 35]]),
             np.array([[60, 0], [24, 35]])]

    fig = plt.figure(figsize=(17.4 * CM, 6.0 * CM))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.15, 1.3, 1.3, 1.3], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3); wd = 0.26
    ax.bar(x - wd, prec, wd, color=C_BLUE, label="Precision")
    ax.bar(x, rec, wd, color=C_ORANGE, label="Recall")
    ax.bar(x + wd, f1, wd, color=C_GREEN, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=7.6)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Clip-level score"); ax.legend(loc="upper left", fontsize=6.6, ncols=1)
    label_below(ax, "(a) precision / recall / F1", dy=-0.20)

    for j, (cm, name) in enumerate(zip(confs, ["A1", "A2", "A3"])):
        axm = fig.add_subplot(gs[0, j + 1])
        axm.imshow(cm, cmap="Blues", vmin=0, vmax=90)
        for (r, c), v in np.ndenumerate(cm):
            axm.text(c, r, str(v), ha="center", va="center", fontsize=10,
                     fontweight="bold", color="white" if v > 50 else "#222222")
        axm.set_xticks([0, 1]); axm.set_yticks([0, 1])
        axm.set_xticklabels(["pred +", "pred $-$"], fontsize=6.6)
        axm.set_yticklabels(["true +", "true $-$"], fontsize=6.6)
        axm.tick_params(axis="both", pad=1.5)
        axm.grid(False)
        label_below(axm, "(%s) %s" % ("bcd"[j], name), dy=-0.20)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.225)
    save(fig, "fig6_e1_arms", png=True)


# ================================================================ fig7 e2 PR
def fig7():
    fig, ax = plt.subplots(figsize=(9.6 * CM, 7.4 * CM))
    for name, color, lab in [("yolov8n", C_BLUE, "YOLOv8n"), ("yolo11n", C_ORANGE, "YOLO11n"),
                             ("hog", C_GREEN, "HOG + SVM")]:
        d = jload(LOGS / ("exp_det_benchmark_%s.json" % name))
        pr = np.array(d["pr_curve"])
        ax.plot(pr[:, 1], pr[:, 0], color=color, lw=1.6, label=lab)
        op = d["at_conf_0.35"]
        ax.plot(op["recall"], op["precision"], "o", color=color, ms=4)
    ax.annotate("operating points\n(conf = 0.35)", xy=(0.515, 0.90), xytext=(0.30, 0.60),
                fontsize=6.8, bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=2),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("HOG confined to\nrecall < 0.02", xy=(0.017, 0.27), xytext=(0.14, 0.14),
                fontsize=6.8, bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=2),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(-0.02, 1.06)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.145, right=0.975, top=0.975, bottom=0.125)
    save(fig, "fig7_e2_pr", png=True)


# ================================================================ fig8 e3
def fig8():
    fig = plt.figure(figsize=(17.4 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    det = [24.0, 20.0]; extra = [3.0, 2.0]
    x = np.arange(2)
    ax.bar(x, det, 0.46, color=C_BLUE, label="detector")
    ax.bar(x, extra, 0.46, bottom=det, color=C_ORANGE, label="observation stage")
    for i, (a, b) in enumerate(zip(det, extra)):
        ax.text(i, a + b + 1.0, "%.0f ms" % (a + b), ha="center", fontsize=7.4, fontweight="bold")
    ax.axhline(66.7, color=C_RED, ls="--", lw=1.0)
    ax.text(1.4, 69.5, "15 fps budget (66.7 ms)", fontsize=6.6, color=C_RED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["YOLOv8n", "HOG + SVM"])
    ax.set_ylabel("Per-frame cost (ms)"); ax.set_ylim(0, 80)
    ax.legend(loc="center right", fontsize=7.0)
    label_below(ax, "(a) per-frame cost decomposition", dy=-0.18)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["display\n(synthetic)", "detect\n(synthetic)", "display\n(real)", "detect\n(real)"]
    v8 = [14.93, 14.93, 24.27, 7.70]; hg = [14.89, 14.91, 24.04, 11.54]
    x2 = np.arange(4); wd = 0.34
    ax2.bar(x2 - wd/2, v8, wd, color=C_BLUE, label="YOLOv8n")
    ax2.bar(x2 + wd/2, hg, wd, color=C_ORANGE, label="HOG")
    for i in range(4):
        ax2.text(i - wd/2, v8[i] + 0.35, "%.1f" % v8[i], ha="center", fontsize=5.9)
        ax2.text(i + wd/2, hg[i] + 0.35, "%.1f" % hg[i], ha="center", fontsize=5.9)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels, fontsize=6.6)
    ax2.set_ylabel("Frame rate (fps)"); ax2.set_ylim(0, 31)
    ax2.legend(loc="upper left", fontsize=6.8)
    label_below(ax2, "(b) display and detection rates", dy=-0.18)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.225)
    save(fig, "fig8_e3_cost", png=True)


# ================================================================ fig9 ablation
def fig9():
    d = jload(LOGS / "exp_ablation.json")["configs"]
    order = [("full", "complete"), ("noreid", "w/o\nre-association"),
             ("nooccl", "w/o\nocclusion"), ("noguard", "w/o\nbox guards")]
    classes = [("DOOR_LEFT_OPEN", C_BLUE, "door left open"),
               ("FLAGGED_DOOR_RETRIEVAL", C_ORANGE, "flagged retrieval"),
               ("MULTI_RETRIEVAL", C_GREEN, "multiple packages"),
               ("LOITERING", C_RED, "loitering")]
    fig = plt.figure(figsize=(17.4 * CM, 5.4 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order)); bottom = np.zeros(len(order))
    for key, color, lab in classes:
        vals = np.array([d[cfg]["events"].get(key, {}).get("TP", 0) for cfg, _ in order], float)
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, label=lab, alpha=0.92)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.35, "%d" % v, ha="center", fontsize=7.4, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s for _, s in order], fontsize=6.7)
    ax.set_ylabel("Correct events (of 20)"); ax.set_ylim(0, 24)
    # 图例移到坐标区**下方**：原先 loc="lower left" 会压在 complete 柱上（数据被图例遮挡）
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=4, fontsize=6.3,
              frameon=False, handlelength=1.2, columnspacing=1.0)
    label_below(ax, "(a) correctly detected events per configuration", dy=-0.62)

    ax2 = fig.add_subplot(gs[0, 1])
    fn = [d[c]["segments"]["FN"] for c, _ in order]
    x2 = np.arange(len(order))
    ax2.bar(x2, fn, 0.42, color=C_RED, label="missed clips (FN)")
    for i, v in enumerate(fn):
        ax2.text(i, v + 0.09, str(int(v)), ha="center", fontsize=7.4, fontweight="bold")
    ax2.set_xticks(x2); ax2.set_xticklabels([s for _, s in order], fontsize=6.7)
    ax2.set_ylabel("Missed clips (of 20)"); ax2.set_ylim(0, 5)
    ax2.legend(loc="upper right", fontsize=6.8)
    label_below(ax2, "(b) clip-level misses", dy=-0.20)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.32)
    save(fig, "fig9_ablation", png=True)


# ================================================================ fig10 sensitivity
def fig10():
    import glob
    A, t_take = [], []
    for p in sorted(glob.glob(str(LOGS / "dataset_eval_final3_shard*.json"))):
        dd = jload(p)
        for c in dd["results"]:
            tl = c.get("timeline") or []
            t_door = min([e["t"] for e in tl if e["type"] == "DOOR_LEFT_OPEN"], default=None)
            t_tk = min([e["t"] for e in tl if e["type"] == "FLAGGED_DOOR_RETRIEVAL"], default=None)
            dur = c["frames"] / 15.0
            if t_door is not None:
                t_open = t_door - 10.0
                A.append(dur - t_open)
                if t_tk is not None:
                    t_take.append(t_tk - t_open)
    A = np.array(A); t_take = np.array(t_take)

    tau = np.linspace(3, 22, 400)
    r_door = [(A >= t).mean() for t in tau]
    r_flag = [(t_take >= t).mean() for t in tau]

    fig = plt.figure(figsize=(17.4 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tau, r_door, color=C_BLUE, lw=1.8, label="door left open")
    ax.plot(tau, r_flag, color=C_ORANGE, lw=1.8, label="flagged retrieval")
    for t in [5, 10, 15, 20]:
        ax.axvline(t, color="#E2E2E2", lw=0.8, zorder=0)
    ax.plot([10, 15, 20], [1, 1, 0], "o", color=C_BLUE, ms=4.5)
    ax.plot([5, 10, 15, 20], [1, 1, 0, 0], "s", color=C_ORANGE, ms=4.5)
    ax.axvspan(12.5, 15.7, color="#FFF3D9", alpha=0.85, zorder=0)
    ax.text(14.1, 0.42, "operable\nrange", fontsize=6.8, ha="center", color="#8a6100")
    ax.text(10, 1.055, "as deployed", fontsize=6.6, ha="center", color="#333333")
    ax.set_xlabel("Door-open threshold $\\tau_{\\mathrm{open}}$ (s)")
    ax.set_ylabel("Event recall")
    ax.set_ylim(-0.06, 1.14); ax.set_xlim(3, 22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower left")
    label_below(ax, "(a) recall vs. door-open threshold", dy=-0.26)

    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    ax2.scatter(A, 1.0 + 0.05*rng.standard_normal(len(A)), s=10, color=C_BLUE, alpha=0.7,
                label="available open duration $A_k$ (42 events)")
    ax2.scatter(t_take, 0.70 + 0.05*rng.standard_normal(len(t_take)), s=10, color=C_ORANGE,
                alpha=0.7, label="retrieval instant $t_{take}$ (21 clips)")
    ax2.axvspan(12.5, 15.7, color="#FFF3D9", alpha=0.85, zorder=0)
    ax2.annotate("door events survive\nup to 18.3 s", xy=(17.7, 1.05), xytext=(10.6, 1.24),
                 fontsize=6.5, ha="left", bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=1.6),
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.text(11.7, 0.50, "retrieval instant $t_{take}$ (21)", fontsize=6.4, ha="center",
             color="#7a3c00", bbox=dict(fc="white", ec="none", pad=1.0))
    for t in [5, 10, 15, 20]:
        ax2.axvline(t, color="#DDDDDD", ls=":", lw=0.9)
        ax2.text(t, 1.42, "%d" % t, ha="center", fontsize=6.3, color="#777777")
    ax2.set_xlim(3, 22); ax2.set_ylim(0.20, 1.50)
    ax2.set_yticks([]); ax2.set_xlabel("Duration (s)")
    ax2.text(17.0, 0.86, "available open duration $A_k$ (42)", fontsize=6.4, ha="center",
             color="#1a4a73", bbox=dict(fc="white", ec="none", pad=1.0))
    label_below(ax2, "(b) recorded durations (s)", dy=-0.26)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.265)
    save(fig, "fig10_sensitivity", png=True)


# ================================================================ fig11 avenue
def fig11():
    d = jload(LOGS / "exp_stage3_domain_shift.json")
    v8 = d["yolov8n"]["per_video"]; hg = d["hog"]["per_video"]
    names = [pv["video"].replace(".avi", "") for pv in v8]
    m8 = [pv["det_per_frame_mean"] for pv in v8]
    mh = [pv["det_per_frame_mean"] for pv in hg]
    fig = plt.figure(figsize=(17.4 * CM, 5.4 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 0.9], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names)); wd = 0.38
    ax.bar(x - wd/2, m8, wd, color=C_BLUE, label="YOLOv8n")
    ax.bar(x + wd/2, mh, wd, color=C_ORANGE, label="HOG + SVM")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.8, rotation=45, ha="right")
    ax.set_ylabel("Persons detected per frame")
    ax.legend(loc="upper right"); ax.set_ylim(0, 14)
    label_below(ax, "(a) per-sequence detection density (CUHK Avenue, 15,328 frames)", dy=-0.26)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([0, 1], [100.0, 7.2], 0.5, color=[C_BLUE, C_ORANGE])
    for i, v in enumerate([100.0, 7.2]):
        ax2.text(i, v + 2.5, "%.1f%%" % v, ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["YOLOv8n", "HOG"], fontsize=7.2)
    ax2.set_ylabel("Frames with $\\geq 1$ detection (%)")
    ax2.set_ylim(0, 120)
    label_below(ax2, "(b) frame coverage", dy=-0.26)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.27)
    save(fig, "fig11_avenue", png=True)


# ================================================================ fig12 field transfer
def fig12():
    rows = jload(LOGS / "exp_field_summary.json")
    keys = [r["key"] for r in rows]
    tot = [r["events_total"] or 0 for r in rows]
    doorish = []
    for r in rows:
        ev = r["events"] or {}
        doorish.append(ev.get("DOOR_LEFT_OPEN", 0) + ev.get("DOOR_CLOSED", 0)
                       + ev.get("FLAGGED_DOOR_RETRIEVAL", 0))

    s1 = imread_cn(LOGS / "field_eval" / "v01_01_MULTI_RETRIEVAL.png")
    s2 = imread_cn(LOGS / "field_eval" / "v04_01_DOOR_LEFT_OPEN.png")

    fig = plt.figure(figsize=(17.4 * CM, 10.2 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.10)

    # (a) 全宽柱状图；x 刻度只保留关键站点，避免拥挤
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(keys))
    ax.bar(x, tot, 0.62, color="#CFCFCF", label="all events")
    ax.bar(x, doorish, 0.62, color=C_BLUE, label="door-derived events")
    for i, t in enumerate(tot):
        if t:
            ax.text(i, t + 1.5, str(t), ha="center", fontsize=6.4)
    keep = [k for k in keys if k in ("v01", "v04", "v07", "v10", "v12", "v15")]
    ax.set_xticks(x); ax.set_xticklabels([k if k in keep else "" for k in keys],
                                         fontsize=8.0)
    ax.set_ylabel("Alarms")
    ax.set_ylim(0, 96)
    ax.set_title("(a) events per recording with the factory configuration "
                 "(grey: all; blue: door-derived)", fontsize=8.4, pad=5)
    ax.legend(loc="upper right", fontsize=7.0)

    # (b)(c) 取证快照：等宽大图，置于下一行
    for j, (im, cap) in enumerate([(s1, "(b) default polygons land on a person"),
                                   (s2, "(c) default grid unrelated to doors")]):
        axs = fig.add_subplot(gs[1, j])
        img_panel(axs, im)
        label_below(axs, cap, dy=-0.055)
    fig.subplots_adjust(left=0.062, right=0.99, top=0.925, bottom=0.085)
    save(fig, "fig12_field_transfer", png=True)


# ================================================================ fig13 audits
def fig13():
    rows = jload(LOGS / "exp_field_summary.json")
    keys = [r["key"] for r in rows]
    coll = [(r["collision_rate"] or 0) * 100 for r in rows]

    d = jload(LOGS / "exp_field_v04_eps_masked.json")
    etas = [0.15, 0.45, 0.55, 0.65, 0.75]
    frac = [d["eta_scan"][str(e)] * 100 for e in etas]

    fig = plt.figure(figsize=(17.4 * CM, 5.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    colors = [C_RED if c > 45 else (C_ORANGE if c > 25 else C_GREEN) for c in coll]
    x = np.arange(len(keys))
    ax.bar(x, coll, 0.62, color=colors)
    ax.axhline(28.3, color="#404040", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=6.6)
    ax.set_ylabel("Collision frames (%)")
    ax.set_ylim(0, 96)
    ax.text(0.15, 88, "bands: green $<$25%; orange 25--45%;\nred $>$45%: rule restricted or report-only",
            fontsize=6.0, color=C_RED, va="top")
    ax.text(14.55, 91, "corpus mean 28.3%", fontsize=6.4, ha="right", va="top",
            color="#333333")
    label_below(ax, "(a) colour-collision audit per recording", dy=-0.26)

    ax2 = fig.add_subplot(gs[0, 1])
    xs = [0, 1, 2, 3, 4]
    ax2.plot(xs, frac, "o-", color=C_BLUE, lw=1.6, ms=4.5)
    ax2.axhspan(0.27, 0.5, xmin=0.32, color="#E9F5EC", zorder=0)
    ax2.axhline(d["baseline_p90"] * 100, color=C_ORANGE, ls=":", lw=1.1)
    # 该虚线画的是日志中实存的 p90；原文案把它写成 "p99 = 0.38"（硬编码、且与所画数值不符）。
    # 现改为按日志数值标注 p90，使图与 logs/exp_field_v04_eps_masked.json 逐字对应。
    ax2.text(4.15, d["baseline_p90"]*100 + 5.5,
             "closed-state $p_{90}$ = %.2f" % d["baseline_p90"],
             fontsize=6.2, color="#8a4a00", ha="right")
    ax2.annotate("default $\\eta$ = 0.15:\n98.6% of closed-door\nsamples read ``open''",
                 xy=(0.04, 98.6), xytext=(0.62, 61), fontsize=6.3,
                 bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=1.6),
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.annotate("$\\eta \\in [0.45, 0.75]$:  $\\leq$ 0.5%", xy=(2.0, 0.6), xytext=(1.55, 30),
                 fontsize=6.3, bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=1.6),
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xticks(xs); ax2.set_xticklabels(["0.15", "0.45", "0.55", "0.65", "0.75"])
    ax2.set_xlabel("Candidate threshold $\\eta$")
    ax2.set_ylabel("Classified ``open'' (%)")
    ax2.set_ylim(-4, 112)
    label_below(ax2, "(b) door-appearance audit", dy=-0.26)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.27)
    save(fig, "fig13_audits", png=True)


# ================================================================ graphical abstract
def fig_ga():
    """Graphical Abstract：1100x560 px（推荐规格），PNG
    左：问题（真实开门帧+红框）；中：方法（流水线小图 + 标定/审计）；右：两条关键结论数字
    """
    W, H = 1100, 560
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off"); ax.grid(False)

    def rounded(x, y, w, h, fc, ec, lw=1.4, ls="solid", rad=12):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%d" % rad,
                           fc=fc, ec=ec, lw=lw, linestyle=ls)
        ax.add_patch(p)

    def arrow(x1, y1, x2, y2, color=C_ARR, w=9):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="simple,head_width=14,head_length=12",
                            color=color, lw=0)
        ax.add_patch(a)

    # 左：问题
    rounded(18, 60, 250, 440, "#FAFAFA", "#B0B0B0")
    ax.text(143, 474, "The problem", fontsize=15, fontweight="bold", ha="center")
    fr = grab_frame("EXAMPLE/bilibili-vedio/从-丰巢快递柜-取件后关门的重要性_视频.mp4", 7.5)
    crop = fr[120:330, 270:520].copy()
    cv2.rectangle(crop, (354-270, 188-120), (412-270, 239-120), (0, 0, 255), 3)
    a = fig.add_axes([36/W, 240/H, 214/W, (214*crop.shape[0]/crop.shape[1])/H])
    a.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); a.axis("off")
    ax.text(143, 210, "door left open", fontsize=12.5, fontweight="bold", ha="center", color="#8a1010")
    ax.text(143, 180, "multiple packages ·\nretrieval from an\nopen compartment",
            fontsize=11, ha="center", va="top", color="#333333", linespacing=1.5)
    ax.text(143, 86, "no active supervision", fontsize=11.5, style="italic",
            ha="center", color="#555555")

    arrow(276, 280, 316, 280)

    # 中：方法
    rounded(322, 60, 300, 440, "#F7FBFE", C_BLUE, ls=(0, (5, 3)))
    ax.text(472, 474, "Calibrated audit-first pipeline", fontsize=13.5, fontweight="bold", ha="center")
    rounded(345, 366, 118, 66, C_LBLUE, C_BLUE)
    ax.text(404, 399, "detect + track", fontsize=11, ha="center", fontweight="bold")
    rounded(479, 366, 118, 66, C_LORANGE, "#BF9000")
    ax.text(538, 399, "door state\n(+occlusion)", fontsize=10.5, ha="center", fontweight="bold")
    rounded(345, 276, 252, 62, C_LGREEN, C_GREEN)
    ax.text(471, 307, "explicit rules  $\\rightarrow$  audited alarms", fontsize=11.5,
            ha="center", fontweight="bold")
    arrow(463, 399, 477, 399, w=7)
    ax.add_patch(FancyArrowPatch((471, 364), (471, 340), arrowstyle="simple,head_width=12,head_length=10",
                                 color=C_ARR, lw=0))
    rounded(330, 168, 284, 78, "#FCE4EC", "#C2185B")
    ax.text(471, 226, "pre-deployment audit", fontsize=11.5, ha="center", fontweight="bold")
    ax.text(471, 200, "geometry coverage  +  colour collision\n+  per-site door threshold",
            fontsize=9.4, ha="center", va="center", linespacing=1.5)
    ax.text(471, 118, "one-time calibration ·\nCPU-only · local", fontsize=11,
            ha="center", va="center", color="#333333", linespacing=1.5)

    arrow(634, 280, 674, 280)

    # 右：结果
    rounded(680, 60, 402, 440, "#FAFAFA", "#B0B0B0")
    ax.text(881, 474, "Measured outcome", fontsize=15, fontweight="bold", ha="center")
    rounded(700, 356, 172, 92, "#FFFFFF", C_BLUE)
    ax.text(786, 424, "real open door", fontsize=10.5, ha="center", color="#333333")
    ax.text(786, 396, "0.94", fontsize=24, fontweight="bold", ha="center", color=C_BLUE)
    ax.text(786, 372, "vs threshold 0.15", fontsize=10, ha="center", color="#555555")
    rounded(888, 356, 172, 92, "#FFFFFF", C_GREEN)
    ax.text(974, 424, "controls (closed)", fontsize=10.5, ha="center", color="#333333")
    ax.text(974, 396, "$\\leq$ 0.04", fontsize=24, fontweight="bold", ha="center", color=C_GREEN)
    ax.text(974, 372, "over the same interval", fontsize=10, ha="center", color="#555555")
    rounded(700, 236, 360, 100, "#FFF8F0", C_ORANGE)
    ax.text(880, 306, "uncalibrated deployment is unusable:", fontsize=11, ha="center", color="#333333")
    ax.text(880, 272, "1 event / 4.4 s", fontsize=21, fontweight="bold", ha="center", color=C_ORANGE)
    ax.text(880, 248, "(197 alarms / 1145 s)  $\\rightarrow$  audit predicts and repairs",
            fontsize=9.2, ha="center", color="#555555")
    ax.text(881, 130, "same-distribution benchmark:\n119/119 clips, zero false\nalarms (correctness, not\ngeneralisation \u2014 stated as such)",
            fontsize=10.5, ha="center", va="center", color="#333333", linespacing=1.55)
    fig.savefig(FIG / "_graphical_abstract_raw.png", dpi=100)
    plt.close(fig)
    # 精确缩放到 1100x560
    im = cv2.imdecode(np.fromfile(str(FIG / "_graphical_abstract_raw.png"), np.uint8), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (1100, 560), interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", im)[1].tofile(str(FIG / "graphical_abstract.png"))
    os.remove(FIG / "_graphical_abstract_raw.png")
    print("graphical_abstract ok", im.shape)


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "6": fig6, "7": fig7,
           "8": fig8, "9": fig9, "10": fig10, "11": fig11, "12": fig12, "13": fig13,
           "ga": fig_ga}
    for k, f in fns.items():
        if only and k not in only:
            continue
        try:
            f()
        except Exception as e:
            import traceback
            print("fig%s FAILED: %s" % (k, e))
            traceback.print_exc()
