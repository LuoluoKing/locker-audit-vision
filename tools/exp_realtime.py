# -*- coding: utf-8 -*-
"""
tools/exp_realtime.py — 阶段3 实时性（E3）与真实监控视频域外测试（E4）

E3 实时性
---------
用**真实服务链路**（app.create_app + VideoWorker，读帧→推理→规则→截图→入库）跑固定时长，
读取 `/api/status` 上报的**显示帧率 / 检测帧率 / 单帧推理耗时 / 检测间隔**，两后端各测一遍。
在合成素材（640×480）与真实监控素材（CUHK Avenue，640×360）上分别测量。

E4 域外测试（CUHK Avenue 真实监控视频）
--------------------------------------
Avenue 数据集的 16 段 **训练视频均为正常行为**。本系统按**部署场景**的 ROI/柜格多边形
直接套用到该场景（未标定），用于量化：
  · 逐帧处理吞吐（真实素材上的推理耗时）；
  · 行人检出连续性（每帧检出数、检出间断）；
  · 异常事件触发情况 —— **不作为误报率指标**，而是用于说明「异常判定依赖部署场景标定」
    这一系统边界（论文中如实表述）。

命令行
------
    python tools/exp_realtime.py --mode realtime --seconds 30
    python tools/exp_realtime.py --mode domain --limit 16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402

SELFTEST_CLIP = C.BASE_DIR / "data" / "selftest" / "selftest_scene.mp4"
AVENUE_TRAIN = C.BASE_DIR / "data" / "videos" / "avenue" / "Avenue Dataset" / "training_videos"
OUT_REALTIME = "exp_stage3_realtime.json"
OUT_DOMAIN = "exp_stage3_domain_shift.json"


def run_realtime(source: Path, backend: str, seconds: float, detect_fps: float) -> dict:
    """走真实 Web 服务链路测量实时性能（与部署形态一致）。"""
    from app import create_app
    from config import Settings

    tmp = C.BASE_DIR / "data" / "selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    # 每次运行使用**独立**数据库与截图目录：同一文件在上一次运行结束后仍可能被
    # SQLite 连接持有（Windows 上表现为 PermissionError: WinError 32），
    # 复用会中断整轮实验（曾实测发生）。
    tag = "%s_%s_%d" % (backend, source.stem, int(time.time() * 1000) % 100000)
    db = tmp / ("exp_realtime_%s.db" % tag)
    snap = tmp / ("exp_snapshots_%s" % tag)
    snap.mkdir(parents=True, exist_ok=True)

    settings = Settings(source=str(source), backend=backend, autostart=False,
                        display_fps=30.0, detect_fps=detect_fps, loop_playback=True,
                        db_path=str(db), snapshot_dir=str(snap), device_id="exp-node",
                        device_name="实验节点")
    app, rt = create_app(settings)
    client = app.test_client()
    r = client.post("/api/control", json={"action": "start", "source": str(source),
                                          "backend": backend, "detect_fps": detect_fps,
                                          "loop": True})
    ok = bool(r.get_json().get("ok"))
    time.sleep(2.0)
    t0 = time.time()
    status = {}
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        status = client.get("/api/status").get_json() or {}
    client.post("/api/control", json={"action": "stop"})
    out = {
        "source": source.name,
        "source_resolution": "%dx%d" % (status.get("frame_width", 0), status.get("frame_height", 0)),
        "src_fps": round(status.get("src_fps", 0) or 0, 2),
        "backend": status.get("backend"),
        "started": ok,
        "display_fps": round(status.get("fps", 0) or 0, 2),
        "detect_fps": round(status.get("detect_fps", 0) or 0, 2),
        "infer_ms": round(status.get("infer_ms", 0) or 0, 2),
        "detect_interval": status.get("detect_interval"),
        "detect_skipped": status.get("detect_skipped"),
        "processed_frames": status.get("frame_index", 0),
        "events": (client.get("/api/events?size=1").get_json() or {}).get("total", 0),
    }
    try:
        rt.stop()
    except Exception:
        pass
    return out


def run_domain(limit: int, backend: str, imgsz: int) -> dict:
    """E4：真实监控视频逐帧跑完整链路，统计吞吐、检出连续性、事件触发情况。"""
    from detector import (DoorStateEstimator, PackageDetector, build_detector,
                          count_carried_packages, count_packages_in_compartments)
    from rules import PackageCarry, RuleConfig, RuleEngine, SceneObservation

    videos = sorted(AVENUE_TRAIN.glob("*.avi"))[:limit]
    if not videos:
        print("未找到 Avenue 训练视频：%s" % AVENUE_TRAIN)
        return {}
    det = build_detector(backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=imgsz)
    _, comps = RuleEngine(RuleConfig()).roi_polygons(C.FRAME_WIDTH, C.FRAME_HEIGHT)
    door = DoorStateEstimator(comps)
    pkg = PackageDetector()

    per_video: List[dict] = []
    for n, v in enumerate(videos, 1):
        rules = RuleEngine(RuleConfig())
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        infer_ms: List[float] = []
        n_det: List[int] = []
        events: List[dict] = []
        i = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = i / float(fps)
            tb = time.perf_counter()
            dets = det.detect(frame)
            boxes = [d.xyxy() for d in dets]
            blobs = pkg.detect(frame)
            obs = SceneObservation(
                dets=dets, door_angles=door.estimate(frame, boxes),
                packages_in_compartment=count_packages_in_compartments(blobs, comps, boxes),
                carries=[PackageCarry(bbox=b, count=c)
                         for b, c in count_carried_packages(blobs, boxes, comps)])
            infer_ms.append((time.perf_counter() - tb) * 1000.0)
            n_det.append(len(dets))
            for ev in rules.update(obs, ts, i, C.FRAME_WIDTH, C.FRAME_HEIGHT):
                if ev.is_abnormal:
                    events.append({"t": round(ts, 2), "type": ev.event_type, "detail": ev.detail})
            i += 1
        cap.release()
        wall = time.perf_counter() - t0
        gaps, run_len, prev = [], 0, None
        for k, c in enumerate(n_det):
            if c == 0:
                run_len += 1
            else:
                if run_len > 8:
                    gaps.append(round(run_len / float(fps), 2))
                run_len = 0
        rec = {"video": v.name, "frames": i, "fps_src": round(fps, 1),
               "wall_s": round(wall, 1), "proc_fps": round(i / wall, 2),
               "infer_ms_median": round(statistics.median(infer_ms), 1),
               "det_per_frame_mean": round(sum(n_det) / len(n_det), 2),
               "frames_without_person": sum(1 for c in n_det if c == 0),
               "detection_gaps_s": gaps[:6],
               "abnormal_events": len(events),
               "event_types": sorted({e["type"] for e in events})}
        per_video.append(rec)
        print("  [%2d/%d] %-8s 帧 %5d  %.2f fps  检出/帧 %.2f  无检出帧 %4d  异常事件 %d %s"
              % (n, len(videos), v.name, i, rec["proc_fps"], rec["det_per_frame_mean"],
                 rec["frames_without_person"], len(events), rec["event_types"]))

    out = {"backend": backend, "dataset": "CUHK Avenue (training_videos, 均为正常行为)",
           "videos": len(per_video),
           "total_frames": sum(r["frames"] for r in per_video),
           "proc_fps_mean": round(sum(r["proc_fps"] for r in per_video) / len(per_video), 2),
           "infer_ms_median": round(statistics.median([r["infer_ms_median"] for r in per_video]), 1),
           "det_per_frame_mean": round(sum(r["det_per_frame_mean"] for r in per_video) / len(per_video), 2),
           "abnormal_events_total": sum(r["abnormal_events"] for r in per_video),
           "per_video": per_video}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段3：实时性（E3）与域外测试（E4）")
    ap.add_argument("--mode", choices=["realtime", "domain"], required=True)
    ap.add_argument("--backend", default="")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--detect-fps", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--real-source", default="", help="E3 真实素材路径（默认 Avenue 首段）")
    a = ap.parse_args()

    if a.mode == "realtime":
        real = Path(a.real_source) if a.real_source else (
            sorted(AVENUE_TRAIN.glob("*.avi"))[0] if AVENUE_TRAIN.exists() else None)
        sources = [("合成素材", SELFTEST_CLIP)]
        if real and real.exists():
            sources.append(("真实监控素材", real))
        results = []
        for backend in ([a.backend] if a.backend else ["yolov8n", "hog"]):
            for label, src in sources:
                if not src.exists():
                    print("素材不存在，跳过：%s" % src)
                    continue
                r = run_realtime(src, backend, a.seconds, a.detect_fps)
                r["source_kind"] = label
                results.append(r)
                print("  %-6s %-12s 显示 %5.2f fps | 检测 %5.2f fps | 单帧 %6.1f ms | 间隔 %s 帧 | 帧数 %s"
                      % (backend, label, r["display_fps"], r["detect_fps"], r["infer_ms"],
                         r["detect_interval"], r["processed_frames"]))
        (C.LOG_DIR / OUT_REALTIME).write_text(
            json.dumps({"runs": results, "detect_fps_requested": a.detect_fps},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("E3 结果已保存：%s" % (C.LOG_DIR / OUT_REALTIME))
        return 0

    results = {}
    for backend in ([a.backend] if a.backend else ["yolov8n", "hog"]):
        r = run_domain(a.limit, backend, a.imgsz)
        if r:
            results[backend] = r
    (C.LOG_DIR / OUT_DOMAIN).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("E4 结果已保存：%s" % (C.LOG_DIR / OUT_DOMAIN))
    return 0


if __name__ == "__main__":
    sys.exit(main())