# -*- coding: utf-8 -*-
"""
tools/eval_dataset.py — 数据集离线评估（检出率 / 误报率 / 逐类混淆）

工作方式
--------
对 `data/simulated/labels.csv` 中的每个片段，**逐帧**跑真实的
「YOLOv8n 检测 → 柜门状态估计 → 餐盒检测 → 规则引擎」链路，统计每段实际触发了哪些事件，
再与该行为的**期望事件**（由 tools/gen_behavior_clip3d.SPECS 声明）对比：

    命中   HIT  : 期望触发的异常事件确实触发
    漏检   MISS : 期望触发但没有触发
    误报   FP   : 不期望的异常事件被触发

时间基准使用**源视频时基**（frame/fps），不按墙钟节流，因此评估速度远快于实时，
且结果与运行机器性能无关（可复现）。

命令行
------
    python tools/eval_dataset.py --per-behavior 2      # 每类抽查 2 段
    python tools/eval_dataset.py --all                 # 全量评估
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from detector import (DoorStateEstimator, PackageDetector, build_detector,  # noqa: E402
                      count_carried_packages, count_packages_in_compartments)
from rules import PackageCarry, RuleConfig, RuleEngine, SceneObservation  # noqa: E402
from gen_behavior_clip3d import SPECS  # noqa: E402

ANOMALIES = sorted(C.ABNORMAL_EVENTS)


def eval_clip(path: Path, backend: str = "yolov8n", imgsz: int = 480,
              cfg: Optional[RuleConfig] = None) -> Dict[str, object]:
    """对单个片段跑完整链路，返回触发的事件类型集合与逐帧记录。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "无法打开 %s" % path}
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    det = build_detector(backend, model_path=str(C.MODEL_DIR / "yolov8n.pt"), imgsz=imgsz)
    _, comps = RuleEngine(cfg or RuleConfig()).roi_polygons(C.FRAME_WIDTH, C.FRAME_HEIGHT)
    door_est = DoorStateEstimator(comps)
    pkg_det = PackageDetector()
    rules = RuleEngine(cfg or RuleConfig())

    fired: List[str] = []
    timeline: List[dict] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = i / float(fps)                         # 源视频时基，保证可复现
        dets = det.detect(frame)
        boxes = [d.xyxy() for d in dets]
        # 柜门状态估计的遮挡补偿：把**近期仍在场**的轨迹框也一并传入。
        # 轨迹在 reid_ttl_s 内即使本帧漏检仍然存活，因此「人站在柜前但被漏检」时，
        # 其深色衣物不会填满格口区域被误判成「柜门开启」
        # （曾使长驻足片段凭空触发 DOOR_LEFT_OPEN，并连带污染其他判定）。
        # 注意只用 `recent_track_boxes`（窗口 occluder_recent_s）：已离场者的残留框继续
        # 遮挡会让柜门状态长期「未知」，反而使 DOOR_LEFT_OPEN 漏检。
        d_ang = door_est.estimate(frame, boxes + rules.recent_track_boxes(ts))
        blobs = pkg_det.detect(frame)
        obs = SceneObservation(
            dets=dets, door_angles=d_ang,
            packages_in_compartment=count_packages_in_compartments(
                blobs, comps, [d.xyxy() for d in dets]),
            carries=[PackageCarry(bbox=b, count=n)
                     for b, n in count_carried_packages(blobs, [d.xyxy() for d in dets], comps)])
        for ev in rules.update(obs, ts, i, C.FRAME_WIDTH, C.FRAME_HEIGHT):
            fired.append(ev.event_type)
            timeline.append({"t": round(ts, 2), "type": ev.event_type, "detail": ev.detail})
        i += 1
    cap.release()

    return {"frames": i, "fired": sorted(set(fired)), "timeline": timeline,
            "normal_retrievals": rules.snapshot()["normal_retrievals"],
            "flagged_ever": bool(rules.snapshot().get("flagged_doors", [])) or
                            any(t["type"] == "DOOR_LEFT_OPEN" for t in timeline)}


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集离线评估")
    ap.add_argument("--labels", default=str(C.BASE_DIR / "data" / "simulated" / "labels.csv"))
    ap.add_argument("--per-behavior", type=int, default=0, help="每类抽查片段数（0 表示全量）")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default="", help="分片评估，形如 0/4（多进程并行用）")
    ap.add_argument("--backend", default="yolov8n")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--report", default="")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖 RuleConfig 字段（阈值敏感性实验用），可重复："
                         "--set door_open_alarm_s=5")
    ap.add_argument("--config", default="",
                    help="可选：几何配置 JSON（含 roi_pickup / compartments），"
                         "用于非同分布场景（如 tools/exp_variant_dataset.py 的重标定输出）；"
                         "留空则沿用 config.py 的默认标定，行为与旧版完全一致")
    a = ap.parse_args()

    # 阈值覆盖：按字段当前值的类型转换，保持类型正确（int 阈值不会被写成 float）
    cfg_override = RuleConfig()
    for kv in a.set:
        if "=" not in kv:
            print("--set 需要 key=value 形式：%s" % kv)
            return 2
        k, v = kv.split("=", 1)
        if not hasattr(cfg_override, k):
            print("RuleConfig 无此字段：%s" % k)
            return 2
        cur = getattr(cfg_override, k)
        if isinstance(cur, bool):
            new_v = v.strip().lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            new_v = int(float(v))
        elif isinstance(cur, float):
            new_v = float(v)
        else:
            new_v = v
        setattr(cfg_override, k, new_v)
        print("阈值覆盖：%s = %r（原 %r）" % (k, new_v, cur))

    # 几何配置载入（纯新增，默认不启用 → 与旧版行为一致）
    if a.config:
        cpath = Path(a.config)
        if not cpath.exists():
            print("未找到几何配置文件：%s" % cpath)
            return 2
        geo = json.loads(cpath.read_text(encoding="utf-8"))
        if "roi_pickup" in geo:
            cfg_override.roi_pickup = [list(p) for p in geo["roi_pickup"]]
        if "compartments" in geo:
            cfg_override.compartments = [[list(p) for p in q] for q in geo["compartments"]]
        print("已载入几何配置：%s（取餐区 %d 点，柜格 %d 个）"
              % (cpath, len(cfg_override.roi_pickup), len(cfg_override.compartments)))

    labels = Path(a.labels)
    if not labels.exists():
        print("未找到标注文件：%s" % labels)
        return 2
    rows = list(csv.DictReader(labels.open(encoding="utf-8-sig")))
    base = labels.parent
    by_beh: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_beh[r["behavior"]].append(r)

    picked: List[dict] = []
    if a.all or a.per_behavior <= 0:
        picked = rows
    else:
        for b, rs in by_beh.items():
            picked += rs[:a.per_behavior]

    shard_tag = ""
    if a.shard:
        idx, tot = (int(v) for v in a.shard.split("/"))
        picked = [r for i, r in enumerate(picked) if i % tot == idx]
        shard_tag = "_shard%d" % idx

    report = Path(a.report) if a.report else C.LOG_DIR / ("dataset_eval%s.json" % shard_tag)

    print("=" * 84)
    print("数据集离线评估：共 %d 段（标注总数 %d）%s"
          % (len(picked), len(rows), ("，分片 %s" % a.shard) if a.shard else ""))
    print("每段逐帧运行 YOLOv8n + 柜门估计 + 餐盒检测 + 规则引擎；时间基准为源视频时基")
    print("=" * 84)

    results = []
    t0 = time.perf_counter()
    for n, r in enumerate(picked, 1):
        spec = SPECS.get(r["behavior"])
        path = base / r["video"]
        res = eval_clip(path, a.backend, a.imgsz, cfg_override)
        fired: Set[str] = set(res.get("fired", []))
        expect: Dict[str, str] = spec.expect if spec else {}
        want_yes = {k for k, v in expect.items() if v == "yes"}
        want_no = {k for k, v in expect.items() if v == "no"}
        hit = sorted(want_yes & fired)
        miss = sorted(want_yes - fired)
        fp = sorted((fired & (set(ANOMALIES) - want_yes)) | (fired & want_no))
        rec = {"clip": r["clip"], "behavior": r["behavior"], "label_cn": r["label_cn"],
               "is_abnormal": int(r["is_abnormal"]), "frames": res.get("frames", 0),
               "fires": sorted(fired), "expect_yes": sorted(want_yes),
               "hit": hit, "miss": miss, "fp": fp,
               "normal_retrievals": res.get("normal_retrievals", 0),
               "timeline": res.get("timeline", [])}
        results.append(rec)
        flag = "✓" if not miss and not fp else ("漏检" if miss else "误报")
        print("  [%3d/%d] %-30s %-6s 触发=%-34s %s"
              % (n, len(picked), r["clip"], "异常" if int(r["is_abnormal"]) else "正常",
                 ",".join(sorted(fired)) or "无", flag))
        for t in res.get("timeline", []):
            print("           %.1fs  %-10s %s" % (t["t"], t["type"], t["detail"][:60]))

    # ---- 汇总（按行为）----
    print("\n" + "=" * 84)
    print("%-26s %4s %6s %6s %6s %8s" % ("行为", "段数", "命中", "漏检", "误报", "准确段数"))
    print("-" * 84)
    tot_hit = tot_miss = tot_fp = tot_ok = 0
    for b in sorted(by_beh):
        rs = [x for x in results if x["behavior"] == b]
        if not rs:
            continue
        h = sum(1 for x in rs if x["hit"]); m = sum(1 for x in rs if x["miss"])
        f = sum(1 for x in rs if x["fp"]); ok = sum(1 for x in rs if not x["miss"] and not x["fp"])
        tot_hit += h; tot_miss += m; tot_fp += f; tot_ok += ok
        print("%-26s %4d %6d %6d %6d %8d" % (b, len(rs), h, m, f, ok))
    print("-" * 84)
    print("%-26s %4d %6d %6d %6d %8d  （段级准确率 %.1f%%）"
          % ("合计", len(results), tot_hit, tot_miss, tot_fp, tot_ok,
             100.0 * tot_ok / max(1, len(results))))
    print("=" * 84)
    print("总耗时 %.1fs（%.2f s/段）" % (time.perf_counter() - t0,
                                     (time.perf_counter() - t0) / max(1, len(results))))

    rp = Path(report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"results": results,
                              "summary": {"clips": len(results), "hit": tot_hit,
                                          "miss": tot_miss, "fp": tot_fp, "ok": tot_ok}},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print("评估报告已保存：%s" % rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
