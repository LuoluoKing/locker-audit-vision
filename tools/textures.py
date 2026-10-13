# -*- coding: utf-8 -*-
"""
tools/textures.py — 程序化纹理生成（纯 NumPy，确定性、零成本、可复现）

为什么用程序化纹理而不是图像生成模型
------------------------------------
1. **确定性**：同一 seed 每次产出逐像素相同的纹理，数据集可复现；
2. **零成本**：不消耗任何额度；
3. **不影响真值**：纹理只改变外观，逐帧真值（柜门角、手部坐标、手持餐盒数）完全不受影响；
4. **不破坏可检测性**：纹理对比度可控，柜门明暗判据与餐盒颜色判据仍成立。

纹理以「世界坐标 → 像素」的方式采样，因此地面与墙面得到的是**透视正确的真实纹理**，
而不是贴图拉伸。刚体表面采用逐面中心采样（平面着色 + 材质色），用于消除纯色塑料感。

使用方式
--------
    tex = Textures(seed=7)
    tex.floor_rgb(x_world, z_world)     -> (H,W,3) float32 RGB
    tex.wall_rgb(u, v)                  -> (H,W,3) float32 RGB
    tex.face_color(mat_name, u, v)      -> (3,) float32 RGB
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = ["Textures", "MATERIALS"]

# 材质名 -> (基色 RGB, 纹理强度, 纹理尺度(每米重复次数))
MATERIALS: Dict[str, Tuple[Tuple[float, float, float], float, float]] = {
    "cabinet":   ((0.50, 0.53, 0.58), 0.10, 3.5),   # 喷涂钢板
    "door":      ((0.72, 0.74, 0.78), 0.07, 3.0),   # 门板
    "door_dark": ((0.60, 0.62, 0.66), 0.07, 3.0),
    "interior":  ((0.26, 0.28, 0.32), 0.05, 2.0),   # 柜内暗区
    "shelf":     ((0.64, 0.66, 0.70), 0.06, 2.5),
    "console":   ((0.30, 0.32, 0.36), 0.12, 6.0),   # 把手
    "pkg_a":     ((0.90, 0.86, 0.30), 0.05, 5.0),   # 餐盒（高饱和，检测依赖其色相）
    "pkg_b":     ((0.88, 0.46, 0.24), 0.05, 5.0),
    "pkg_c":     ((0.30, 0.72, 0.52), 0.05, 5.0),
    "shirt":     ((0.22, 0.34, 0.56), 0.10, 7.0),   # 织物
    "shirt2":    ((0.55, 0.28, 0.30), 0.10, 7.0),
    "pants":     ((0.17, 0.20, 0.28), 0.09, 8.0),
    "skin":      ((0.80, 0.64, 0.52), 0.03, 6.0),
    "hair":      ((0.16, 0.14, 0.13), 0.05, 8.0),
    "shoe":      ((0.11, 0.12, 0.15), 0.06, 8.0),
}


class Textures:
    """程序化纹理库。所有随机量均由固定 seed 的生成器产生，逐次运行完全一致。"""

    def __init__(self, seed: int = 20260925, floor_tile: float = 0.62,
                 strength: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.floor_tile = float(floor_tile)
        self.strength = float(strength)
        # 预生成若干张平铺噪声图，采样时按世界坐标取模索引
        self._noise = self._make_noise(256, 4)
        self._stripe = self._make_stripes(256, 12)

    # ------------------------------------------------------------ 基础噪声
    def _make_noise(self, size: int, octaves: int) -> np.ndarray:
        """多倍频值噪声（[0,1]），用于材质微起伏与污渍。"""
        acc = np.zeros((size, size), np.float32)
        amp = 1.0
        total = 0.0
        for o in range(octaves):
            n = max(4, size >> (octaves - 1 - o))
            base = self.rng.random((n, n)).astype(np.float32)
            acc += amp * cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
            total += amp
            amp *= 0.5
        acc /= max(total, 1e-6)
        return np.clip(acc, 0.0, 1.0)

    def _make_stripes(self, size: int, period: int) -> np.ndarray:
        """横向细条纹，用于金属拉丝质感。"""
        x = np.arange(size, dtype=np.float32)
        s = 0.5 + 0.5 * np.sin(2 * math.pi * x / period)
        s = np.tile(s[None, :], (size, 1))
        s += 0.12 * self.rng.standard_normal((size, size)).astype(np.float32)
        return np.clip(s, 0.0, 1.0)

    @staticmethod
    def _tile_index(coord: np.ndarray, scale: float, size: int) -> np.ndarray:
        return (np.abs(coord) * scale * size).astype(np.int32) % size

    # ------------------------------------------------------------ 地面 / 墙面
    def floor_rgb(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        地面材质：浅灰玻化砖 + 砖缝 + 污渍。
        输入为**世界坐标**，因此透视关系由上层射线求交天然保证，不存在贴图拉伸。
        """
        t = self.floor_tile
        tx = np.floor(x / t)
        tz = np.floor(z / t)
        fx = np.abs(x / t - tx - 0.5) * 2.0
        fz = np.abs(z / t - tz - 0.5) * 2.0
        grout = np.maximum(fx, fz) > 0.92

        ix = self._tile_index(x, 0.35, 256)
        iz = self._tile_index(z, 0.35, 256)
        n = self._noise[iz, ix]

        # 每块砖的确定性明度差（避免整片死板）
        wob = (np.sin(tx * 12.9898 + tz * 78.233) * 0.5 + 0.5) * 0.05
        base = 0.705 + wob + (n - 0.5) * 0.075 * self.strength
        col = np.stack([base * 1.00, base * 1.015, base * 1.045], axis=-1)
        g = 0.545 + (n - 0.5) * 0.05
        col = np.where(grout[..., None], np.stack([g, g * 1.01, g * 1.03], axis=-1), col)
        return col.astype(np.float32)

    def wall_rgb(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """墙面材质：乳胶漆 + 竖向分缝 + 踢脚线。"""
        iu = self._tile_index(u, 0.5, 256)
        iv = self._tile_index(v, 0.5, 256)
        n = self._noise[iv, iu]
        base = 0.845 + (n - 0.5) * 0.035 * self.strength
        base = base + 0.030 * np.clip((v - (-0.6)) / 5.8, 0.0, 1.0)
        col = np.stack([base * 0.995, base, base * 1.020], axis=-1)
        seam = np.abs((u / 1.6) - np.floor(u / 1.6) - 0.5) * 2.0 > 0.965
        col = np.where(seam[..., None], col * 0.94, col)
        skirt = (v - (-0.6)) < 0.12
        col = np.where(skirt[..., None], col * 0.86, col)
        return col.astype(np.float32)

    # ------------------------------------------------------------ 刚体面采样
    def face_color(self, material: str, u: float, v: float,
                   base_override: Tuple[float, float, float] = None) -> np.ndarray:
        """
        按材质的局部归一化坐标 (u,v) ∈ [0,1]² 取一个面颜色（平面着色 + 材质变化）。

        base_override：**以调用方给出的真实材质色为基色**，只沿用该材质的
        纹理变化幅度与图案类型（条纹 / 噪点）。必须传：否则 `infer_material`
        会把任意新配色替换成固定材质表里「最接近」的那个颜色，
        导致渲染出的物体颜色与设定不符（曾使蓝色操作屏渲染成绿色，
        并被餐盒检测器当成餐盒 → 误报「一人取多份」）。
        """
        base, s, scale = MATERIALS.get(material, ((0.6, 0.6, 0.6), 0.08, 3.0))
        if base_override is not None:
            base = tuple(float(x) for x in base_override)
        ix = int(abs(u) * scale * 256) % 256
        iy = int(abs(v) * scale * 256) % 256
        if material in ("cabinet", "door", "door_dark", "shelf", "console", "interior"):
            k = float(self._stripe[iy, ix]) - 0.5
        else:
            k = float(self._noise[iy, ix]) - 0.5
        f = 1.0 + k * s * 2.0 * self.strength
        return np.clip(np.asarray(base, np.float32) * f, 0.0, 1.0)

    # ------------------------------------------------------------ 材质推断
    @staticmethod
    def infer_material(rgb: Tuple[float, float, float]) -> str:
        """
        按基色反查**纹理图案类型**（条纹 / 噪点）与变化幅度。

        注意：返回值只用于挑选纹理图案，**不得**用它的颜色覆盖调用方的基色
        （见 `face_color` 的 base_override 说明）。
        """
        best, bestd = "cabinet", 1e9
        for name, (c, _s, _sc) in MATERIALS.items():
            d = sum((c[i] - rgb[i]) ** 2 for i in range(3))
            if d < bestd:
                best, bestd = name, d
        return best
