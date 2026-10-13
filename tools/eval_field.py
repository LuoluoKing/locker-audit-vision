# -*- coding: utf-8 -*-
"""tools/eval_field.py — 实拍现场素材上的系统行为评估（E8）

对 `EXAMPLE/bilibili-vedio/` 的实拍视频逐帧跑真实链路
（YOLOv8n 检测 → 柜门状态估计 → 餐盒检测 → 规则引擎），记录全部触发事件、
保存事件时刻的标注快照，供论文的「现场素材边界研究」使用。

与 `eval_dataset.py` 的区别：
  * 无标注（实拍素材没有期望事件），产出是**行为记录**而非对错统计；
  * 使用**视频自身分辨率**做归一化坐标 → 像素坐标映射（min/max 适配任意画幅）；
  * 支持三种配置模式：
      default        —— 沿用系统出厂标定（我们合成场景的 ROI + 柜格多边形）
      no-compartment —— 仅人物类规则（柜格为空，等价于「柜门类事件不可用」）
      file:<path>    —— 从 JSON 加载自定义标定（roi_pickup / compartments，归一化）
  * 支持片段区间（--start / --end，秒），用于只跑视频的某个稳定片段。

输出：
  logs/field_eval/<key>.json     逐视频事件时间线 + 统计
  logs/field_eval/<key>_<n>_<ev>.png  事件快照（首见 + 每次新的冷却放行）

用法：
  python tools/eval_field.py --all --mode default
  python tools/eval_field.py --video v04 --start 0 --end 60 --mode file:logs/field_calib/v04.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tools"))

import config as C  # noqa: E402
from detector import (DoorStateEstimator, PackageDetector, build_detector,  # noqa: E402
                      count_carried_packages, count_packages_in_compartments)
from rules import PackageCarry, RuleConfig, RuleEngine, SceneObservation  # noqa: E402

VID_DIR = ROOT / "EXAMPLE" / "bilibili-vedio"
OUT_DIR = ROOT / "logs" / "field_eval"


def imwrite_cn(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def load_cfg(mode: str):
    """返回 (RuleConfig, eta)。eta 为可选的开门暗比阈值覆盖（None=用出厂 0.15）。"""
    cfg = RuleConfig()
    if mode == "default":
        return cfg, None
    if mode == "no-compartment":
        cfg.compartments = []
        return cfg, None
    if mode.startswith("file:"):
        p = mode.split(":", 1)[1]
        d = json.load(open(p, encoding="utf-8"))
        if "roi_pickup" in d:
            cfg.roi_pickup = d["roi_pickup"]
        if "compartments" in d:
            cfg.compartments = d["compartments"]
        return cfg, (float(d["eta"]) if "eta" in d else None)
    raise ValueError("未知模式：%s" % mode)


def ascii_only(s: str) -> str:
    """cv2.putText 不支持中文字形：快照标签只保留 ASCII 可见字符。"""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)


def draw_overlay(frame, dets, roi_px, comps_px, label):
    vis = frame.copy()
    if roi_px is not None and len(roi_px):
        cv2.polylines(vis, [np.array(roi_px, np.int32)], True, (0, 200, 255), 2)
    for q in comps_px:
        cv2.polylines(vis, [np.array(q, np.int32)], True, (255, 160, 0), 1)
    for d in dets:
        cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      (0, 220, 0), 2)
        fx, fy = int(d.bottom_center[0]), int(d.bottom_center[1])
        cv2.circle(vis, (fx, fy), 3, (0, 0, 255), -1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, ascii_only(label)[:110], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2)
    return vis


def eval_video(path: Path, key: str, mode: str, backend: str = "yolov8n",
               imgsz: int = 480, start_s: float = 0.0, end_s: float = 0.0,
               snapshot_max: int = 3, verbose: bool = True, eta_override: float = 0.0,
               tag: str = ""):
    cfg, eta_cfg = load_cfg(mode)
    eta = eta_override if eta_override else (eta_cfg or 0.15)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "无法打开 %s" % path}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f0 = int(start_s * fps)
    f1 = int(end_s * fps) if end_s > 0 else n_total
    if f0 > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    det = build_detector(backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                         imgsz=imgsz)
    rules = RuleEngine(cfg)
    roi_px, comps_px = rules.roi_polygons(w, h)
    door_est = DoorStateEstimator(comps_px, open_dark_ratio=eta)
    pkg_det = PackageDetector()

    events = []
    snap_count = {}
    i = f0
    t0 = time.time()
    while i < f1:
        ok, frame = cap.read()
        if not ok:
            break
        ts = i / float(fps)
        dets = det.detect(frame)
        boxes = [d.xyxy() for d in dets]
        d_ang = door_est.estimate(frame, boxes + rules.recent_track_boxes(ts))
        blobs = pkg_det.detect(frame)
        obs = SceneObservation(
            dets=dets, door_angles=d_ang,
            packages_in_compartment=count_packages_in_compartments(blobs, comps_px, boxes),
            carries=[PackageCarry(bbox=b, count=c)
                     for b, c in count_carried_packages(blobs, boxes, comps_px)])
        for ev in rules.update(obs, ts, i, w, h):
            rec = {"t": round(ts, 2), "type": ev.event_type, "detail": ev.detail}
            events.append(rec)
            c = snap_count.get(ev.event_type, 0) + 1
            snap_count[ev.event_type] = c
            if c <= snapshot_max:
                label = "%s | t=%.1fs | %s" % (ev.event_type, ts,
                                               str(ev.detail)[:80])
                vis = draw_overlay(frame, dets, roi_px, comps_px, label)
                imwrite_cn(OUT_DIR / ("%s_%02d_%s.png" % (key, c, ev.event_type)), vis)
        if verbose and (i - f0) % max(1, int(5 * fps)) == 0:
            el = time.time() - t0
            done = i - f0
            print("  [%s] %.0fs / %.0fs（%.1f fps）事件 %d" % (
                key, done / fps, (f1 - f0) / fps, done / max(el, 1e-6), len(events)))
        i += 1
    cap.release()

    snap = rules.snapshot()
    summary = {
        "key": key, "file": path.name, "mode": mode, "eta": eta,
        "range_s": [start_s, round((f1 - f0) / fps, 1)],
        "size": [w, h], "frames": i - f0,
        "wall_s": round(time.time() - t0, 1),
        "counts": {k: int(v) for k, v in sorted(
            {t: sum(1 for e in events if e["type"] == t)
             for t in {e["type"] for e in events}}.items())},
        "events": events,
        "normal_retrievals": int(snap.get("normal_retrievals", 0)),
        "stats": {k: v for k, v in snap.items()
                  if isinstance(v, (int, float, str)) and k != "tracks"},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s" % (key, tag) if tag else key
    with open(OUT_DIR / ("%s.json" % stem), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", default="", help="单段：v01..v15 或文件名子串")
    ap.add_argument("--mode", default="default")
    ap.add_argument("--backend", default="yolov8n")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--snapshots", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.0,
                    help="覆盖开门暗比阈值（0=用标定文件/出厂值）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖既有日志")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(VID_DIR) if f.lower().endswith(".mp4"))
    keys = {("v%02d" % (i + 1)): f for i, f in enumerate(files)}

    todo = []
    if a.all:
        todo = list(keys.items())
    elif a.video:
        if a.video in keys:
            todo = [(a.video, keys[a.video])]
        else:
            todo = [(k, f) for k, f in keys.items() if a.video in f]
    if not todo:
        print("无匹配视频")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, fn in todo:
        print("=" * 90)
        print("评估 %s：%s  模式=%s" % (key, fn, a.mode))
        s = eval_video(VID_DIR / fn, key, a.mode, backend=a.backend,
                       imgsz=a.imgsz, start_s=a.start, end_s=a.end,
                       snapshot_max=a.snapshots, eta_override=a.eta, tag=a.tag)
        if "error" in s:
            print("  错误：", s["error"])
            continue
        print("  完成：%d 帧 / %.1fs 墙钟；事件统计：%s；正常取餐计数：%d" % (
            s["frames"], s["wall_s"], s["counts"] or "无", s["normal_retrievals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
