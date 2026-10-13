# -*- coding: utf-8 -*-
"""
tools/diag_obs.py — 逐帧观测诊断器

用途
----
数据集评估不通过时，把指定片段、指定时刻的**中间观测量**打印出来：
人员检测框 / 脚点 / 所属区域 / 柜门暗比与状态估计 / 柜内餐盒数 / 手持餐盒数。
用于区分「检测器输出错」还是「规则判定错」。

命令行
------
    python tools/diag_obs.py --clip multi_retrieval_001 --times 4.5 6.0
    python tools/diag_obs.py --clip flagged_door_retrieval_001 --times 12.8 13.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from detector import (DoorStateEstimator, PackageDetector, build_detector,  # noqa: E402
                      count_carried_packages, count_packages_in_compartments)
from rules import PackageCarry, RuleConfig, RuleEngine, SceneObservation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="逐帧观测诊断")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--backend", default="yolov8n")
    ap.add_argument("--imgsz", type=int, default=480)
    a = ap.parse_args()

    video = C.BASE_DIR / "data" / "simulated" / (a.clip + ".mp4")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("无法打开 %s" % video)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    det = build_detector(a.backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=a.imgsz)
    eng = RuleEngine(RuleConfig())
    _, comps = eng.roi_polygons(C.FRAME_WIDTH, C.FRAME_HEIGHT)
    pickup_poly, _ = eng.roi_polygons(C.FRAME_WIDTH, C.FRAME_HEIGHT)
    door_est = DoorStateEstimator(comps)
    pkg_det = PackageDetector()

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for tq in a.times:
        fi = min(n_frames - 1, int(round(tq * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print("t=%.2f 读帧失败" % tq)
            continue
        dets = det.detect(frame)
        boxes = [d.xyxy() for d in dets]
        d_ang = door_est.estimate(frame, boxes)
        blobs = pkg_det.detect(frame)
        in_comp = count_packages_in_compartments(blobs, comps, boxes)
        carries = count_carried_packages(blobs, boxes, comps)
        print("== %s  t=%.2fs (frame %d)" % (a.clip, fi / fps, fi))
        print("   检测到人物 %d 个:" % len(dets))
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.xyxy()
            bx, by = (x1 + x2) / 2.0, y2
            zone = eng._zone((bx, by), C.FRAME_WIDTH, C.FRAME_HEIGHT)
            print("     #%d conf=%.2f box=(%d,%d,%d,%d) 脚点=(%d,%d) 区域=%s"
                  % (i, d.conf, x1, y1, x2, y2, bx, by, zone))
        print("   柜门角度估计: %s" % ("None" if d_ang is None else
              {k: (round(v, 1) if v is not None else None) for k, v in sorted(d_ang.items())}))
        ratios = getattr(door_est, "last_ratios", None)
        if ratios is not None:
            print("   柜门暗比: %s" % ["%.3f" % r for r in ratios])
        print("   餐盒 blob %d 个, 各柜格数量: %s" % (len(blobs), in_comp))
        for i, (b, n) in enumerate(carries):
            print("     人物%d 手持=%d" % (i, n))
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
