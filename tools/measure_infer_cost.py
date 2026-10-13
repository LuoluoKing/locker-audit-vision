# -*- coding: utf-8 -*-
"""tools/measure_infer_cost.py — 单帧推理与场景观测耗时分解（E3 口径统一用）

背景
----
阶段3 的 E3 数据（logs/exp_stage3_realtime.json，2026-09-25）中 infer_ms 是
**在线服务运行时的 status 读数**（含显示/推流并发负载、完整观测步），与 E2 的
**纯检测器离线计时**口径不同，不能直接比较。本脚本给出**口径明确、可复现**的
耗时分解测量：

  1. 检测器单独（视频帧，640x480）：YOLOv8n（imgsz=480）与 HOG（resize 320）
  2. 场景观测模块单独：餐盒色块检测 + 逐格门态估计
  3. 完整观测步 = 检测器 + 观测模块（离线、无线程并发）

产出：logs/exp_infer_cost.json（含每个口径的 median / mean / n）
用法：python tools/measure_infer_cost.py [--n 30]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import config as C                       # noqa: E402
from detector import (DoorStateEstimator, HogDetector, PackageDetector,  # noqa: E402
                      YoloDetector)
from rules import RuleConfig, RuleEngine  # noqa: E402


def _timed(fn, n: int):
    """跑 n 次，返回 (median, mean) 毫秒。"""
    ts = []
    for _ in range(n):
        tb = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - tb) * 1000.0)
    return round(statistics.median(ts), 1), round(statistics.mean(ts), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="每项重复次数")
    a = ap.parse_args()

    # 取一张真实模拟帧（640x480）
    vids = sorted(glob.glob(str(C.SIM_DIR / "**" / "*.mp4"), recursive=True))
    cap = cv2.VideoCapture(vids[0])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("无法读取测试帧：", vids[0])
        return 1
    h, w = frame.shape[:2]

    out = {"frame": {"source": Path(vids[0]).name, "size": f"{w}x{h}"},
           "n": a.n, "note": "offline single-thread timing; see module docstring"}

    # -------- 1. YOLOv8n 纯检测 --------
    y = YoloDetector(model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                     conf_threshold=0.35, iou_threshold=0.50,
                     imgsz=480, device="cpu", allow_download=False)
    y.warmup()
    y.detect(frame)
    med, mean = _timed(lambda: y.detect(frame), a.n)
    out["yolov8n_detect_ms"] = {"median": med, "mean": mean}
    print(f"YOLOv8n 纯检测: median {med} ms / mean {mean} ms")

    # -------- 2. HOG 纯检测（resize 320，与 E2 同口径）--------
    h320 = HogDetector(conf_threshold=0.35, resize_width=320)
    h320.warmup()
    med, mean = _timed(lambda: h320.detect(frame), a.n)
    out["hog320_detect_ms"] = {"median": med, "mean": mean}
    print(f"HOG(resize320) 纯检测: median {med} ms / mean {mean} ms")

    # -------- 3. 完整观测步（detector + 餐盒 + 门态）--------
    pkg = PackageDetector()
    _, comps = RuleEngine(RuleConfig()).roi_polygons(C.FRAME_WIDTH, C.FRAME_HEIGHT)
    door = DoorStateEstimator(comps)

    def obs_y():
        dets = y.detect(frame)
        boxes = [d.xyxy() for d in dets]
        pkg.detect(frame)
        door.estimate(frame, boxes)

    med, mean = _timed(obs_y, a.n)
    out["yolov8n_full_step_ms"] = {"median": med, "mean": mean}
    print(f"YOLOv8n 完整观测步: median {med} ms / mean {mean} ms")

    def obs_h():
        dets = h320.detect(frame)
        boxes = [d.xyxy() for d in dets]
        pkg.detect(frame)
        door.estimate(frame, boxes)

    med, mean = _timed(obs_h, a.n)
    out["hog320_full_step_ms"] = {"median": med, "mean": mean}
    print(f"HOG(resize320) 完整观测步: median {med} ms / mean {mean} ms")

    # -------- 4. 观测模块单独（从完整步 - 纯检测粗略可得，另测精确值）--------
    def obs_only():
        pkg.detect(frame)
        door.estimate(frame, [])

    med, mean = _timed(obs_only, a.n)
    out["observation_only_ms"] = {"median": med, "mean": mean}
    print(f"观测模块单独: median {med} ms / mean {mean} ms")

    path = C.LOG_DIR / "exp_infer_cost.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
