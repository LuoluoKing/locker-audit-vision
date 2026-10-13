# -*- coding: utf-8 -*-
"""
tools/calib_locker_door_threshold.py — 换柜体后重标定「柜门暗比」判别阈值

原理
----
`detector.DoorStateEstimator` 用「柜格区域内显著暗于全画面中位亮度的像素占比」
（dark_ratio）判断门是否开启。该阈值与**柜体格口尺寸 / 材质 / 机位**强相关，
换柜体后必须重新实测标定。

本工具做**决定性的几何+阈值双重检验**：
  对每个柜格 k 单独渲染「只有第 k 扇门开启」的帧，计算**全部 6 个格口**的暗比，
  形成 6×6 矩阵。
  * 若格口多边形与真实门面**对齐**，则矩阵应为**对角占优**（第 k 门开启时第 k 格暗比最高）；
    若出现错位或左右镜像，会表现为非对角元偏高 —— 这是判定标定是否正确的硬证据。
  * 由「门关」样本（全部门关闭）与「门开」样本（该格开启）的暗比分布，
    给出推荐阈值（取两类分布的分界中点）并报告分类准确率。

命令行
------
    python tools/calib_locker_door_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import scene3d as S  # noqa: E402
import scene_kit as K  # noqa: E402
import textures as TX  # noqa: E402
from detector import DoorStateEstimator  # noqa: E402
from gen_selftest_clip3d import CAM_EYE, CAM_FOV, CAM_TARGET  # noqa: E402
from rules import RuleConfig, RuleEngine  # noqa: E402

W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
N = K.DOOR_ROWS * K.DOOR_COLS
C_DOOR_DARK_TH = 0.15       # 与 detector.DoorStateEstimator 的 open_dark_ratio 默认值一致


def render_frame(open_index: int = None, angle: float = 86.0) -> np.ndarray:
    doors = [0.0] * N
    if open_index is not None:
        doors[open_index] = float(angle)
    items = K.build_locker(doors)
    items += K.build_packages(K.locker_package_slots())
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    renderer = S.Renderer(W, H, ssaa=2, textures=TX.Textures(seed=20260925))
    occ = [(m, mt) for m, mt, _ in items]
    return renderer.render(items, cam, S.Light(),
                           floor=S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0)),
                           walls=[S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0,
                                              v_lo=-0.6, v_hi=5.2, base_color=(0.80, 0.81, 0.83))],
                           shadow_occluders=occ)


def main() -> int:
    """扫描开门角，校验新柜体外观下「门开/门关」的暗比是否可分，并给出推荐阈值。"""
    eng = RuleEngine(RuleConfig())
    _, comps = eng.roi_polygons(W, H)
    est0 = DoorStateEstimator(comps)
    est0.estimate(render_frame(None))
    base = np.array(est0.last_dark_ratios)
    print("=" * 92)
    print("柜门暗比校验与开门角择优：柜体 build_locker（外观已换、门面几何与旧件一致）")
    print("门全关时各格暗比: %s" % ["%.3f" % v for v in base])
    print("=" * 92)

    best = None
    for angle in (86.0, 100.0, 115.0):
        mat = np.zeros((N, N))
        for k in range(N):
            e = DoorStateEstimator(comps)
            e.estimate(render_frame(k, angle))
            mat[k] = e.last_dark_ratios
        diag = int(np.sum(np.argmax(mat, axis=1) == np.arange(N)))
        open_vals = np.diag(mat)
        closed_vals = np.concatenate([np.delete(mat, np.arange(N), axis=1).ravel(), base])
        sep = float(open_vals.mean() - closed_vals.mean())
        lo, hi = float(open_vals.min()), float(closed_vals.max())
        thr_pick = (lo + hi) / 2.0
        print("\n开门角 %.0f°：对角占优 %d/%d  门开均值 %.3f（最小 %.3f）  "
              "门关均值 %.3f（最大 %.3f）  分离度 %+.3f"
              % (angle, diag, N, open_vals.mean(), lo, closed_vals.mean(), hi, sep))
        print("     各格门开暗比: %s" % ["%.3f" % v for v in open_vals])
        print("     建议阈值（门开最小与门关最大之中点）= %.3f；现行 0.15 下命中 %d/%d、误报 %d/%d"
              % (thr_pick, int(np.sum(open_vals >= C_DOOR_DARK_TH)), N,
                 int(np.sum(closed_vals >= C_DOOR_DARK_TH)), closed_vals.size))
        score = (diag, lo - hi)
        if best is None or score > best[0]:
            best = (score, angle, thr_pick, mat.copy(), lo, hi)

    score, angle, thr_pick, mat, lo, hi = best
    print("\n" + "=" * 92)
    print("最优开门角 = %.0f°（对角占优 %d/%d，门开最小 %.3f 与门关最大 %.3f 间距 %.3f）"
          % (angle, score[0], N, lo, hi, lo - hi))
    print("=" * 92)
    print("      " + "".join("%10s" % ("格口%d" % j) for j in range(N)))
    for k in range(N):
        cells = ["%10.3f" % v for v in mat[k]]
        cells[k] = "%10s" % ("[%.3f]" % mat[k][k])
        print("开%d  " % k + "".join(cells))

    ok = (score[0] == N and (lo - hi) > 0.05)
    print("\n结论：%s" % ("柜门判别有效 ✓（推荐 open_dark_ratio = %.2f）"
                        % max(0.05, round((hi + lo) / 2.0 - 0.01, 2)) if ok
                        else "仍有格口不可分 ✗，不要继续生成数据集"))
    return 0


def _legacy_main() -> int:
    eng = RuleEngine(RuleConfig())
    _, comps = eng.roi_polygons(W, H)
    est = DoorStateEstimator(comps)
    closed_obj = DoorStateEstimator(comps)      # 用「关」阈值下的基线（同一实现）
    print("=" * 84)
    print("柜门暗比标定：柜体 build_locker，格口数 %d（多边形来自 config.CABINET_COMPARTMENTS）" % N)
    print("=" * 84)

    # 门全关基线
    base = closed_obj.estimate(render_frame(None))
    base_ratio = list(closed_obj.last_dark_ratios)
    print("门全关时各格暗比: %s" % ["%.3f" % v for v in base_ratio])

    mat = np.zeros((N, N), dtype=float)
    vis = np.zeros((N, N), dtype=float)
    for k in range(N):
        est2 = DoorStateEstimator(comps)
        est2.estimate(render_frame(k))
        mat[k] = est2.last_dark_ratios
        vis[k] = est2.last_visible

    print("\n6×6 暗比矩阵（行 = 实际开启的柜格，列 = 各格口实测暗比）")
    print("      " + "".join("%9s" % ("格口%d" % j) for j in range(N)))
    for k in range(N):
        marks = ["%9.3f" % v for v in mat[k]]
        marks[k] = "%9s" % ("[%.3f]" % mat[k][k])          # 对角线加中括号标记
        print("开%d  " % k + "".join(marks))

    diag_ok = int(np.sum(np.argmax(mat, axis=1) == np.arange(N)))
    print("\n对角占优检验：%d / %d 行最大值落在对角线 → %s"
          % (diag_ok, N, "格口多边形与真实门面对齐 ✓" if diag_ok == N
             else "存在错位/镜像，必须先修正 CABINET_COMPARTMENTS ✗"))

    open_vals = np.diag(mat)
    closed_vals = np.delete(mat, np.arange(N), axis=1).ravel()
    closed_vals = np.concatenate([closed_vals, np.array(base_ratio)])
    thr_mid = float((open_vals.mean() + closed_vals.mean()) / 2.0)
    print("\n门开时该格暗比: 均值 %.3f，最小 %.3f（样本 %d）" % (open_vals.mean(), open_vals.min(), N))
    print("门关时该格暗比: 均值 %.3f，最大 %.3f（样本 %d）"
          % (closed_vals.mean(), closed_vals.max(), closed_vals.size))
    print("推荐阈值（两类均值中点）= %.3f；现行 config 使用 0.15" % thr_mid)

    for thr in (0.15, round(thr_mid, 2)):
        tp = int(np.sum(open_vals >= thr))
        fp = int(np.sum(closed_vals >= thr))
        acc = (tp + int(np.sum(closed_vals < thr))) / float(N + closed_vals.size)
        print("  阈值 %.2f：门开命中 %d/%d，门关误报 %d/%d，准确率 %.4f"
              % (thr, tp, N, fp, closed_vals.size, acc))

    err = float(np.max(np.abs(mat - np.diag(open_vals))))
    print("\n最大非对角暗比 = %.3f（越小说明串扰越小）" % err)
    print("最低可见率 = %.3f（低于 DoorStateEstimator.min_visible 时该格返回「未知」）" % vis.min())
    return 0


if __name__ == "__main__":
    sys.exit(main())