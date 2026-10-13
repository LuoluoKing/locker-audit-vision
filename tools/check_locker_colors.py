# -*- coding: utf-8 -*-
"""
tools/check_locker_colors.py — 校验取餐柜装饰配色是否落在餐盒检测色域内（防误报回归）

背景
----
`detector.PackageDetector` 用 HSV 色域分割找餐盒：H∈[15,95] 且 S≥130 且 V≥90，
再按面积 [60,1200] 与长宽比过滤。**柜体上的任何装饰色若落进该色域**，
就可能被计为餐盒：实测初版橙黄编号牌（约 10×10 px、面积 88）虽然位于格口多边形
之外，却落在人物检测框的外扩区之内，被 `count_carried_packages` 当成「手持餐盒」，
使演示素材凭空出现「一人取多份」误报。

本工具对柜体全部配色逐一换算 HSV，并给出「是否会进餐盒色域」的判定；
另做一次**真实渲染 + 餐盒检测**的端到端校验（关掉人物，只留柜体），
确认空场景下餐盒 blob 数为 0。

命令行
------
    python tools/check_locker_colors.py
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
import cv2  # noqa: E402
from detector import PackageDetector  # noqa: E402
from gen_selftest_clip3d import CAM_EYE, CAM_FOV, CAM_TARGET  # noqa: E402

W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT


def to_hsv(rgb) -> tuple:
    """RGB(0~1) -> OpenCV HSV（H 0~180, S/V 0~255），与 detector 的判据同一口径。"""
    px = np.uint8([[[int(round(c * 255)) for c in rgb[::-1]]]])      # RGB -> BGR
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def main() -> int:
    pd = PackageDetector()
    print("=" * 88)
    print("餐盒检测判据：H∈[%d,%d]，S≥%d，V≥%d，面积∈[%d,%d]，长宽比≤%.1f"
          % (pd.hue_min, pd.hue_max, pd.sat_min, pd.val_min, pd.area_min, pd.area_max, pd.aspect_max))
    print("=" * 88)

    bad = []
    print("%-16s %-22s %5s %5s %5s  %s" % ("配色项", "RGB", "H", "S", "V", "判定"))
    for name, rgb in sorted(K.LOCKER_STYLE.items()):
        h, s, v = to_hsv(rgb)
        in_hue = pd.hue_min <= h <= pd.hue_max
        in_sat = s >= pd.sat_min
        in_val = v >= pd.val_min
        risky = in_hue and in_sat and in_val
        verdict = "⚠ 落入餐盒色域（风险）" if risky else "安全"
        if risky:
            bad.append(name)
        print("%-16s %-22s %5d %5d %5d  %s"
              % (name, "(%.2f, %.2f, %.2f)" % rgb, h, s, v, verdict))

    # 餐盒自身配色（应当落进食盒色域，作为判据有效性的阳性对照）
    print("\n阳性对照（餐盒配色，**应当**落进食盒色域）：")
    for i, rgb in enumerate(K.PACKAGE_COLORS):
        h, s, v = to_hsv(rgb)
        ok = (pd.hue_min <= h <= pd.hue_max) and s >= pd.sat_min and v >= pd.val_min
        print("  餐盒色%d %-22s H=%3d S=%3d V=%3d  %s"
              % (i, "(%.2f, %.2f, %.2f)" % rgb, h, s, v, "命中 ✓" if ok else "未命中 ✗（判据可能失效）"))

    # ---- 端到端校验（**必须看渲染像素，不能只看配色表**）
    # 教训：配色表里的 screen_glow 是蓝色 (0.26,0.62,0.88)，但渲染器早先会用
    # `tex.infer_material()` 把它替换成材质表里「最接近」的颜色 —— 恰好是绿色餐盒色
    # `pkg_c`，于是操作屏被渲染成绿色，进而被餐盒检测器当成餐盒。
    # 因此**只校验配色表的数值是不够的**，必须在真实渲染帧上复检。
    doors = [0.0] * (K.DOOR_ROWS * K.DOOR_COLS)
    cam = S.Camera(eye=CAM_EYE, target=CAM_TARGET, fov_y_deg=CAM_FOV)
    tex = TX.Textures(seed=20260925)
    scene = dict(floor=S.FloorPlane(x_range=(-14.0, 14.0), z_range=(-6.0, 16.0)),
                 walls=[S.WallPlane(axis="z", pos=7.0, lo=-18.0, hi=18.0, v_lo=-0.6, v_hi=5.2,
                                    base_color=(0.80, 0.81, 0.83))])

    cases = [("空场景（仅柜体）", K.build_locker(doors))]
    cases.append(("含遮挡人物", K.build_locker(doors) + K.build_human(
        K.Pose(x=0.16, z=-0.72, shoulder_r=52.0, elbow_r=-34.0))))
    # 阳性对照：**必须把某扇门打开**——餐盒放在柜内，门关着就被门板挡住，
    # 渲染帧里根本看不到（曾因用全关门做对照，误以为餐盒检测失效）。
    doors_open = list(doors)
    doors_open[2] = 115.0
    cases.append(("含货架餐盒（阳性对照，柜格2开门）",
                  K.build_locker(doors_open) + K.build_packages(K.locker_package_slots())))

    ok_all = True
    tags = ("empty", "person", "packages")
    for ci, (label, items) in enumerate(cases):
        renderer = S.Renderer(W, H, ssaa=2, textures=tex)
        occ = [(m, mt) for m, mt, _ in items]
        img = renderer.render(items, cam, S.Light(), shadow_occluders=occ, **scene)
        blobs = pd.detect(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([pd.hue_min, pd.sat_min, pd.val_min], np.uint8),
                           np.array([pd.hue_max, 255, 255], np.uint8))
        n_px = int(np.count_nonzero(mask))
        print("\n端到端校验 · %s：命中餐盒色域像素 %d，检出 blob %d 个" % (label, n_px, len(blobs)))
        for b in blobs:
            cx, cy = int(b.center[0]), int(b.center[1])
            h, s, v = hsv[cy, cx]
            print("   blob c=(%d,%d) area=%.0f 中心HSV=(%d,%d,%d)"
                  % (cx, cy, b.area, h, s, v))
        out = C.BASE_DIR / "logs" / "locker_designs" / ("color_check_%s.png" % tags[ci])
        cv2.imencode(".png", img)[1].tofile(str(out))
        expect_zero = "阳性对照" not in label
        if expect_zero and (blobs or n_px > 0):
            ok_all = False
            print("   ✗ 期望无餐盒信号，但存在（会引发假「一人取多份」/柜内计数错误）")
        elif expect_zero:
            print("   ✓ 无任何餐盒信号")
        else:
            print("   %s 阳性对照应能检出餐盒" % ("✓" if blobs else "✗"))

    print("\n结论：%s" % ("配色安全，不会引起餐盒误检 ✓" if not bad and ok_all
                        else "存在风险配色 %s 或渲染后仍落进餐盒色域，需调整" % (bad or "（见上）")))
    return 0


if __name__ == "__main__":
    sys.exit(main())