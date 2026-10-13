# -*- coding: utf-8 -*-
"""
tools/exp_variant_dataset.py — 第二套「非同分布」渲染测试集（跨机位 / 跨光照）+ 几何重标定

背景
----
119 段基准（data/simulated/）与生成器、标定几何、规则阈值**同分布自洽**，因此 clip/event
指标全满分（1.000）。审稿人要求补充**独立渲染场景**：至少改变相机位姿 / 光照 / 柜体布局，
并在**与新场景匹配的标定**下给出真实指标。

本工具只做两件事，且**不触碰任何既有行为**：
  1. calib：用新机位把柜格门面四角投影成归一化多边形（复用 scene_kit.cabinet_door_quad
     这一几何单一来源），并把基准 ROI_PICKUP 的四个画面角**反投影到地面**再用新机位
     重投影，得到「同一物理取餐区、新机位标定」的取餐区；同时输出柜门暗比分离度复核。
  2. gen  ：复用 gen_behavior_clip3d.SPECS 的六类行为脚本，仅在**渲染环节**替换机位与光源，
     在**新目录**渲染出第二套数据集与 labels.csv。

为什么必须重标定
----------------
`detector.DoorStateEstimator` 需要用「柜格门面在**本机位**下的画面四边形」来统计暗比；
`rules.RuleEngine` 的取餐区判定同理。换机位后若沿用旧的 config 坐标，格口会与真实门面错位，
指标将衡量「标定错误」而不是「规则迁移」。本工具产出的 JSON 由 eval_dataset.py --config 载入。

机位/光照差异（相对基准）
------------------------
基准：eye=(2.35,1.92,-3.25) target=(-0.15,1.02,0.10) fov=48；Light 默认（key 0.78 / amb 0.34）。
变体：eye=(-1.95,1.62,-2.85) target=(0.30,1.00,-0.05) fov=57；Light key 0.60 / amb 0.46 等。
三项相机参数（eye/target/fov）与光照四项（方向/主光强/补光/环境）均改变；柜体仍为 3×2
（scene_kit.build_locker 目前仅支持 3×2，改行列数须改 scene_kit，不在本任务范围）。

命令行
------
    python tools/exp_variant_dataset.py --mode calib --config-out logs/variant_v2_geometry.json
    python tools/exp_variant_dataset.py --mode gen --out data/simulated_v2 --count 4 --workers 5
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from gen_selftest_clip3d import (CAM_EYE, CAM_FOV, CAM_TARGET, OUT_H, OUT_W)  # noqa: E402
from calib_locker_compartments import door_quad  # noqa: E402  （仅取世界几何，不用其 project）
import gen_behavior_clip3d as G  # noqa: E402

# ================================================================ 变体机位 / 光照
VARIANT_CAM_EYE = (1.05, 2.30, -2.60)         # 基准 (2.35,1.92,-3.25)：更矮更正、更高、更近
VARIANT_CAM_TARGET = (-0.05, 1.05, 0.12)      # 基准 (-0.15,1.02,0.10)
VARIANT_CAM_FOV = 62.0                        # 基准 48.0：视野更宽
VARIANT_LIGHT = dict(
    key_dir=(-0.48, -1.0, -0.12),             # 基准 (0.34,-1.0,-0.22)：主光改从另一侧来
    key_intensity=0.60,                       # 基准 0.78
    fill_intensity=0.16,                      # 基准 0.30
    ambient=0.46,                             # 基准 0.34
    shadow_strength=0.30,                     # 基准 0.42
)

W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
DOOR_DARK_TH = 0.15                            # 与 detector.DoorStateEstimator.open_dark_ratio 一致

BASE_CALIB = C.BASE_DIR / "logs" / "variant_v2_geometry.json"


def variant_camera() -> S.Camera:
    return S.Camera(eye=VARIANT_CAM_EYE, target=VARIANT_CAM_TARGET, fov_y_deg=VARIANT_CAM_FOV)


def variant_light() -> S.Light:
    return S.Light(**VARIANT_LIGHT)


def project_px(cam: S.Camera, pts: np.ndarray) -> np.ndarray:
    """World (N,3) -> 像素 (N,2)，与 scene3d.Renderer 内部投影**完全同源**（V@P）。

    关键：**不要**复用 calib_locker_compartments.project —— 它取 r = cross(f, up)，
    而渲染器（scene3d.look_at）取 xaxis = cross(up, f)，二者互为相反数，
    故该工具输出相对真实渲染画面**左右镜像**（这正是现行 config.CABINET_COMPARTMENTS
    与该工具输出不一致的原因：config 用的是渲染器一致的约定）。此处照搬渲染器变换。
    """
    V = cam.view()
    P = cam.projection(W / float(H))
    VP = np.asarray(V, np.float64) @ np.asarray(P, np.float64)
    vh = np.hstack([np.asarray(pts, dtype=np.float64), np.ones((len(pts), 1))])
    clip = vh @ VP
    w = clip[:, 3:4]
    ndc = clip[:, :3] / w
    sx = (ndc[:, 0] * 0.5 + 0.5) * W
    sy = (0.5 - ndc[:, 1] * 0.5) * H
    return np.stack([sx, sy], axis=1)


# ================================================================ 几何重标定
def _unproject_ground(norm_pts: Sequence[Sequence[float]], cam: S.Camera) -> List[np.ndarray]:
    """把一批归一化画面点按相机反投影到地面 y=0，得到世界坐标（同一物理区域）。"""
    rd = cam.ray_dirs(W, H)
    eye = np.asarray(cam.eye, dtype=np.float64)
    out: List[np.ndarray] = []
    for nx, ny in norm_pts:
        px = int(min(W - 1, max(0, round(nx * W))))
        py = int(min(H - 1, max(0, round(ny * H))))
        d = rd[py, px].astype(np.float64)
        t = (0.0 - eye[1]) / d[1]
        out.append(eye + t * d)
    return out


def calibrate() -> dict:
    """用变体机位重算 6 个柜格多边形与取餐区 ROI（返回归一化坐标）。"""
    cam = variant_camera()

    # ---- 柜格：直接投影 scene_kit.cabinet_door_quad（柜门几何单一来源）
    rows, cols = K.DOOR_ROWS, K.DOOR_COLS
    geo = K.locker_geometry(rows, cols)
    comps: List[List[List[float]]] = []
    for r in range(rows):
        for c in range(cols):
            px = project_px(cam, door_quad(geo, r, c))
            comps.append([[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in px])

    # ---- 取餐区：把基准 ROI 的四角反投影到地面，再用变体机位重投影（保持同一物理区域）
    cam_base = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    ground = _unproject_ground(C.ROI_PICKUP, cam_base)
    px = project_px(cam, np.asarray(ground, dtype=np.float64))
    roi = [[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in px]

    # ---- 站位脚点校验
    foot_world = np.array([[G.PICKUP_X, 0.0, G.PICKUP_Z]], dtype=np.float64)
    foot_px = project_px(cam, foot_world)[0]
    import cv2
    roi_px = np.array([[int(x * W), int(y * H)] for x, y in roi], np.int32)
    inside = cv2.pointPolygonTest(roi_px, (float(foot_px[0]), float(foot_px[1])), False) >= 0

    return {"camera": {"eye": list(VARIANT_CAM_EYE), "target": list(VARIANT_CAM_TARGET),
                       "fov": VARIANT_CAM_FOV},
            "light": VARIANT_LIGHT,
            "roi_pickup": roi,
            "compartments": comps,
            "foot_px_norm": [round(float(foot_px[0]) / W, 4), round(float(foot_px[1]) / H, 4)],
            "foot_in_roi": bool(inside),
            "n_compartments": len(comps)}


# ================================================================ 投影一致性自检
def baseline_sanity() -> dict:
    """用基准机位 + project_px 重投影柜门，应与现行 config.CABINET_COMPARTMENTS 一致。

    这是 project_px 正确性的硬证据：若与 config 的角点最大偏差 < 1px 量级，
    说明本工具的投影约定与渲染器/config 完全一致（而非 calib_locker_compartments 的镜像约定）。
    """
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    rows, cols = K.DOOR_ROWS, K.DOOR_COLS
    geo = K.locker_geometry(rows, cols)
    devs = []
    for r in range(rows):
        for c in range(cols):
            px = project_px(cam, door_quad(geo, r, c))
            a = np.array([[x / W, y / H] for x, y in px])
            b = np.array(C.CABINET_COMPARTMENTS[r * cols + c], dtype=np.float64)
            # 角点顺序可能不同：每个投影角点到 config 角点集的最近距离取最大
            d = max(float(min(np.linalg.norm(pa - pb) for pb in b)) for pa in a)
            devs.append(round(d, 4))
    return {"max_corner_deviation": max(devs), "per_compartment": devs}


# ================================================================ 柜门暗比复核（变体场景）
def _render_scene_frame(open_index: int, angle: float = 115.0) -> np.ndarray:
    doors = [0.0] * (K.DOOR_ROWS * K.DOOR_COLS)
    if open_index is not None:
        doors[open_index] = float(angle)
    items = K.build_locker(doors)
    items += K.build_packages(K.locker_package_slots())
    cam = variant_camera()
    renderer = S.Renderer(W, H, ssaa=2, textures=TX.Textures(seed=20260925))
    occ = [(m, mt) for m, mt, _ in items]
    return renderer.render(items, cam, variant_light(),
                           floor=S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0)),
                           walls=[S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0,
                                              v_lo=-0.6, v_hi=5.2, base_color=(0.80, 0.81, 0.83))],
                           shadow_occluders=occ)


def write_overlay(calib: dict, path: Path) -> None:
    """把变体标定（柜格多边形=红、取餐区 ROI=蓝）叠加在变体渲染帧上，供人工核对。"""
    import cv2
    img = _render_scene_frame(None, 0.0)
    for i, q in enumerate(calib["compartments"]):
        pts = np.array([[int(x * W), int(y * H)] for x, y in q], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(img, "C%d" % i, (int(c[0]) - 10, int(c[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    roi = np.array([[int(x * W), int(y * H)] for x, y in calib["roi_pickup"]], np.int32)
    cv2.polylines(img, [roi], True, (255, 0, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(path))


def door_threshold_check(comps: Sequence[Sequence[float]]) -> dict:
    """用变体机位/光照 + 新标定，实测「门开/门关」暗比分离度（决定性硬证据）。"""
    from detector import DoorStateEstimator
    N = len(comps)
    comps_px = [np.array([[int(x * W), int(y * H)] for x, y in q], np.int32) for q in comps]

    est = DoorStateEstimator(comps_px)
    est.estimate(_render_scene_frame(None, 0.0))
    base = list(est.last_dark_ratios)
    mat = np.zeros((N, N))
    for k in range(N):
        e = DoorStateEstimator(comps_px)
        e.estimate(_render_scene_frame(k, 115.0))
        mat[k] = e.last_dark_ratios
    diag = int(np.sum(np.argmax(mat, axis=1) == np.arange(N)))
    open_vals = np.diag(mat)
    closed_vals = np.concatenate([np.delete(mat, np.arange(N), axis=1).ravel(), np.array(base)])
    thr = (float(np.min(open_vals)) + float(np.max(closed_vals))) / 2.0
    return {
        "closed_base": [round(float(v), 3) for v in base],
        "open_dark_ratios": [round(float(v), 3) for v in open_vals],
        "diag_dominant": diag, "n": N,
        "open_min": round(float(np.min(open_vals)), 3),
        "closed_max": round(float(np.max(closed_vals)), 3),
        "separation": round(float(np.min(open_vals) - np.max(closed_vals)), 3),
        "recommended_thr": round(thr, 3),
        "hit_at_0.15": int(np.sum(open_vals >= DOOR_DARK_TH)),
        "fp_at_0.15": int(np.sum(closed_vals >= DOOR_DARK_TH)),
        "matrix": [[round(float(v), 3) for v in row] for row in mat],
    }


# ================================================================ 渲染（复用 gen_behavior_clip3d）
def _init_worker(eye, target, fov, light_kw) -> None:
    """worker 进程内替换 gen_behavior_clip3d 的**模块级机位绑定**与 scene3d.Light 默认参数。

    说明：gen_behavior_clip3d 里 `from gen_selftest_clip3d import CAM_EYE` 是模块级绑定，
    改 gen_selftest_clip3d.CAM_EYE 无效，必须 patch gen_behavior_clip3d.CAM_EYE。
    Light 则是 render_clip 内 `S.Light()` 无参构造，故把 scene3d.Light 换成一个带默认值的
    partial（G 与 S 指向同一 scene3d 模块对象，patch 一次即生效）。
    """
    G.CAM_EYE = tuple(eye)
    G.CAM_TARGET = tuple(target)
    G.CAM_FOV = float(fov)
    _BaseLight = S.Light
    S.Light = functools.partial(_BaseLight, **light_kw)   # type: ignore[assignment]
    G.S.Light = S.Light


def gen(out_dir: str, count: int, workers: int, fps: int, ssaa: int,
        behaviors: str = "all") -> int:
    names = list(G.SPECS) if behaviors in ("", "all") else [b for b in behaviors.split(",") if b]
    for b in names:
        if b not in G.SPECS:
            print("未知行为：%s（可选：%s）" % (b, ", ".join(G.SPECS)))
            return 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(n, i + 1, str(out), fps, ssaa) for n in names for i in range(count)]

    # 变体标定必须先落盘（数据集与评估所用的几何要一致）
    calib = calibrate()
    cfg_path = BASE_CALIB
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 84)
    print("第二套（变体）数据集渲染：%d 类 × %d 段 = %d 段，%d 进程" %
          (len(names), count, len(tasks), workers))
    print("机位 eye=%s target=%s fov=%.1f（基准 eye=%s fov=%.1f）"
          % (VARIANT_CAM_EYE, VARIANT_CAM_TARGET, VARIANT_CAM_FOV, CAM_EYE, CAM_FOV))
    print("光照 %s" % VARIANT_LIGHT)
    print("重标定：%d 个柜格；站位脚点归一化 %s，落在取餐区内=%s"
          % (calib["n_compartments"], calib["foot_px_norm"], calib["foot_in_roi"]))
    print("几何配置已写入：%s" % cfg_path)
    print("=" * 84)

    t0 = time.perf_counter()
    results: List[dict] = []
    done = 0
    with multiprocessing.Pool(workers, initializer=_init_worker,
                              initargs=(VARIANT_CAM_EYE, VARIANT_CAM_TARGET,
                                        VARIANT_CAM_FOV, VARIANT_LIGHT)) as pool:
        for r in pool.imap_unordered(G.render_clip, tasks):
            results.append(r); done += 1
            if done % 3 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print("  进度 %3d/%d  已用 %.0fs  预计剩余 %.0fs"
                      % (done, len(tasks), el, el / done * (len(tasks) - done)))

    results.sort(key=lambda r: (r["behavior"], r["clip"]))
    cols = ["clip", "video", "behavior", "label_cn", "is_abnormal", "frames", "seconds",
            "fps", "n_persons", "max_packages_in_hand", "door_open_frames", "truth"]
    labels = out / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})

    print("\n已生成 %d 段，用时 %.1fs" % (len(results), time.perf_counter() - t0))
    print("标注文件：%s" % labels)
    return 0


# ================================================================ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="第二套非同分布渲染测试集 + 几何重标定")
    ap.add_argument("--mode", choices=["sanity", "calib", "gen"], required=True)
    ap.add_argument("--out", default=str(C.BASE_DIR / "data" / "simulated_v2"))
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--behaviors", default="all", help="逗号分隔的行为名，或 all")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--config-out", default=str(BASE_CALIB))
    a = ap.parse_args()

    if a.mode == "sanity":
        s = baseline_sanity()
        print("project_px 与基准 config 一致性自检：最大角点偏差 %.4f（归一化）" % s["max_corner_deviation"])
        print("  逐格偏差: %s" % s["per_compartment"])
        print("  结论：%s" % ("一致（投影约定与渲染器相符）✓"
                            if s["max_corner_deviation"] < 0.01 else "不一致 ✗"))
        return 0

    if a.mode == "calib":
        calib = calibrate()
        p = Path(a.config_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        chk = door_threshold_check(calib["compartments"])
        calib["door_threshold_check"] = chk
        p.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        ov = C.LOG_DIR / "variant_v2_calib_overlay.png"
        write_overlay(calib, ov)
        print("变体标定对照图已保存：%s（红=柜格多边形，蓝=取餐区 ROI）" % ov)
        print("=" * 88)
        print("变体几何重标定（机位 eye=%s target=%s fov=%.1f）"
              % (VARIANT_CAM_EYE, VARIANT_CAM_TARGET, VARIANT_CAM_FOV))
        print("柜格数 %d；站位脚点归一化 %s，落在取餐区内=%s"
              % (calib["n_compartments"], calib["foot_px_norm"], calib["foot_in_roi"]))
        for i, q in enumerate(calib["compartments"]):
            print("  柜格 %d: %s" % (i, q))
        print("取餐区 ROI: %s" % calib["roi_pickup"])
        print("-" * 88)
        print("柜门暗比复核（变体机位/光照 + 新标定）：")
        print("  门全关基线暗比: %s" % chk["closed_base"])
        print("  各格门开暗比  : %s" % chk["open_dark_ratios"])
        print("  对角占优 %d/%d；门开最小 %.3f，门关最大 %.3f，分离度 %+.3f"
              % (chk["diag_dominant"], chk["n"], chk["open_min"], chk["closed_max"],
                 chk["separation"]))
        print("  推荐阈值 %.3f；现行 0.15 下命中 %d/%d、误报 %d/%d"
              % (chk["recommended_thr"], chk["hit_at_0.15"], chk["n"],
                 chk["fp_at_0.15"], chk["n"] + chk["n"] * (chk["n"] - 1)))
        ok = chk["diag_dominant"] == chk["n"] and chk["separation"] > 0.0
        print("  结论：%s" % ("变体场景下柜门判别仍可分 ✓" if ok
                            else "变体场景下柜门判别不可分 ✗（评估将如实反映该退化）"))
        print("几何配置已写入：%s" % p)
        print("=" * 88)
        return 0

    return gen(a.out, a.count, a.workers, a.fps, a.ssaa, a.behaviors)


if __name__ == "__main__":
    sys.exit(main())
