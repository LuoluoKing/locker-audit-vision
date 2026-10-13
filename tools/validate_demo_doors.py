# -*- coding: utf-8 -*-
"""
tools/validate_demo_doors.py — 离线验证「演示档案门态判定」（AI 演示素材）

以与运行时相同的节拍（每 2 帧≈12fps）回放素材，运行 `code/demo_door.py` 引擎，
打印事件时间线与逐格口最终状态；可选导出带叠加的验证视频。

用法
----
    python tools/validate_demo_doors.py --clip 未关柜门
    python tools/validate_demo_doors.py --clip 异常取餐 --video logs/real_check/demo_异常取餐.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402
from detector import build_detector  # noqa: E402
from demo_door import DemoDoorEngine, _to_px  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="演示档案门态判定离线验证")
    ap.add_argument("--clip", required=True, help="素材名（不含 .mp4）")
    ap.add_argument("--every", type=int, default=2, help="抽样步长（2≈12fps@24fps）")
    ap.add_argument("--video", default="", help="可选：输出叠加视频路径")
    a = ap.parse_args()

    clip_name = a.clip + ".mp4"
    profile = C.AI_DEMO_DOOR_PROFILE.get(clip_name)
    if profile is None:
        print("config.AI_DEMO_DOOR_PROFILE 中没有该素材：%s" % clip_name)
        return 2

    detector = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"),
                              imgsz=480, conf_threshold=0.35)
    engine = DemoDoorEngine(clip_name, profile, detector=detector, log=lambda m: print("   " + m))
    if not engine.calibrate():
        print("校准失败：%s" % engine._calib_error)
        return 1

    src = C.REAL_DEMO_DIR / clip_name
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    writer = None
    if a.video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(a.video, fourcc, fps / a.every, (C.FRAME_WIDTH, C.FRAME_HEIGHT))

    events = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % a.every == 0:
            ts = fi / fps
            dets = detector.detect(frame)
            boxes = [tuple(d.xyxy()) for d in dets]
            evs = engine.update(frame, ts, boxes, track_of_box=None)
            for ev in evs:
                events.append((ts, ev.event_type, ev.detail))
                print("  %6.2fs  %-24s %s" % (ts, ev.event_type, ev.detail))
            if writer is not None:
                vis = frame.copy()
                snap = engine.snapshot()
                for d in snap["doors"]:
                    cell = next(c for c in profile["cells"]
                                if c["row"] == int(d["door_id"][1]) - 1
                                and c["col"] == int(d["door_id"][3]) - 1)
                    px = _to_px(cell["poly"], C.FRAME_WIDTH, C.FRAME_HEIGHT)
                    if cell.get("skip"):
                        continue
                    color = (0, 0, 255) if d["flagged"] else ((0, 210, 0) if d["state"] == "CLOSED" else (0, 165, 255))
                    cv2.polylines(vis, [px], True, color, 2)
                    cx, cy = px.mean(axis=0).astype(int)
                    cv2.putText(vis, "%s %.2f" % (d["door_id"], d["ncc"]), (cx - 34, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                cv2.putText(vis, "t=%.1fs" % ts, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 1, cv2.LINE_AA)
                writer.write(vis)
        fi += 1
    cap.release()
    if writer is not None:
        writer.release()

    snap = engine.snapshot()
    print("=" * 78)
    print("%s：事件 %d 条；校准格口 %d/%d 可用" %
          (clip_name, len(events), snap["cells_usable"], snap["cells_total"]))
    print("最终标记柜号：%s" % (snap["flagged_doors"] or "无"))
    toggled = [d for d in snap["doors"] if d["state"] == "OPEN"]
    print("末态为「开」的格口：%s" % ([d["door_id"] for d in toggled] or "无"))
    if a.video:
        print("叠加视频：%s" % a.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
