# -*- coding: utf-8 -*-
"""
tools/scene3d.py — 极简软件三维渲染器（纯 NumPy + OpenCV，无 GPU、无第三方渲染引擎）

为什么自研而非使用现成引擎
--------------------------
本项目硬性约束为「单机 CPU + 全程免费工具」，且阶段3 需要**逐帧真值**（关节角、柜门角、
手持餐盒数）与**可复现性**（固定随机种子、逐帧确定）。任何基于神经网络的视频生成模型
都同时违反这三点（额度消耗、结果不可复现、无法导出真值）。因此这里实现一个**几何意义上的
真三维渲染器**：三维网格 → 世界变换 → 透视投影 → 深度缓冲光栅化 → 光照着色 → 超采样抗锯齿。

需要如实说明的能力边界
----------------------
本渲染器可以做到：正确的透视投影、平面/曲面着色（Lambert + 补光 + 环境光）、正确的遮挡关系、
逐帧阴影投影、地面透视网格、超采样抗锯齿、可选的深度雾。
它**做不到**照片级真实感（无纹理映射、无全局光照、无法线贴图、无运动模糊）。
产出的是「三维渲染合成数据」，不是实拍视频，论文中必须如此表述。

主要近似（已在代码中标注）
--------------------------
1. 逐三角形**常量深度**：以三角形重心的视空间 z 作为该三角形的深度。
   在三角形屏幕尺寸约 30px 量级时无可见误差，可省去逐像素重心插值。
2. 阴影为**平行光投影**（沿单一光照方向把顶点投到地面平面求凸包后填充），非软阴影。
3. 无背面剔除：闭合网格由深度缓冲保证遮挡正确，代价是三角形数量翻倍。

坐标系约定（左手系，便于理解）
------------------------------
    +X 向右, +Y 向上, +Z 向前（远离摄像机）
    摄像机位于 -Z 侧，朝 +Z 方向观察
    视图空间 z_view 为正表示在摄像机前方，直接作为深度值使用
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "Mesh", "box_mesh", "sphere_mesh", "prism_mesh",
    "translate", "rotate_x", "rotate_y", "rotate_z", "scale_m", "compose", "look_at",
    "Camera", "Light", "Renderer", "FloorPlane", "WallPlane",
]

Vec3 = Sequence[float]
Mat4 = np.ndarray


# ============================================================ 变换
def translate(x: float, y: float, z: float) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[3, 0], m[3, 1], m[3, 2] = x, y, z
    return m


def scale_m(sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> Mat4:
    if sy is None:
        sy = sz = sx
    if sz is None:
        sz = sy
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def rotate_x(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, s, -s, c
    return m


def rotate_y(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, -s, s, c
    return m


def rotate_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, s, -s, c
    return m


def compose(*mats: Mat4) -> Mat4:
    """按参数顺序左乘组合（行向量约定：先施加第一个矩阵）。"""
    out = np.eye(4, dtype=np.float32)
    for m in mats:
        out = out @ m
    return out


def look_at(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> Mat4:
    """
    左手系 look-at 视图矩阵（**行向量**约定：v_view = v_world @ M）。

    注意基底排布：行向量约定下 M 的前三行必须是基底的**列**形式
    （即 M[:3,:3] = [xaxis, yaxis, zaxis] 的转置），否则等价于使用了转置矩阵，
    相机方向会完全错误。此处曾因此出错，已修正并保留说明。
    """
    e = np.asarray(eye, np.float32)
    t = np.asarray(target, np.float32)
    u = np.asarray(up, np.float32)
    zaxis = t - e
    zaxis /= np.linalg.norm(zaxis)
    xaxis = np.cross(u, zaxis)
    n = np.linalg.norm(xaxis)
    if n < 1e-6:                       # 视线与 up 平行时的退化处理
        xaxis = np.array([1.0, 0.0, 0.0], np.float32)
    else:
        xaxis /= n
    yaxis = np.cross(zaxis, xaxis)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.stack([xaxis, yaxis, zaxis], axis=1)      # 列式摆放
    m[3, :3] = -np.array([xaxis @ e, yaxis @ e, zaxis @ e], np.float32)
    return m


# ============================================================ 网格
@dataclass
class Mesh:
    verts: np.ndarray                       # (N,3) float32
    faces: np.ndarray                       # (M,3) int32
    tag: str = ""

    def transformed(self, m: Mat4) -> "Mesh":
        """
        应用世界变换。行向量约定：v_world = [v,1] @ M（**不要转置 M**）。
        本项目所有变换构造器（translate/rotate_*/scale_m/compose）均按行向量约定建立，
        此处若写成 vh @ m.T 即等价于按列向量使用，会得到错误的缩放与旋转结果。
        """
        vh = np.concatenate([self.verts, np.ones((len(self.verts), 1), np.float32)], axis=1)
        return Mesh((vh @ m)[:, :3].astype(np.float32), self.faces, self.tag)

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _quad_faces(quads: List[Tuple[int, int, int, int]]) -> np.ndarray:
    tris: List[Tuple[int, int, int]] = []
    for a, b, c, d in quads:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return np.asarray(tris, np.int32)


def box_mesh(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Mesh:
    """以原点为中心的立方体，12 个三角形。"""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], np.float32)
    q = [(0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7), (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0)]
    return Mesh(v, _quad_faces(q), "box")


def prism_mesh(radius: float = 0.5, height: float = 1.0, segments: int = 10,
               top_scale: float = 1.0) -> Mesh:
    """竖直棱柱（圆柱的低多边形近似），可指定顶面缩放形成锥台。侧面 + 上下盖。"""
    h = height / 2
    ring = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    cs, sn = np.cos(ring), np.sin(ring)
    bottom = np.stack([radius * cs, np.full(segments, -h), radius * sn], axis=1)
    top = np.stack([radius * top_scale * cs, np.full(segments, h), radius * top_scale * sn], axis=1)
    verts = np.vstack([bottom, top]).astype(np.float32)

    faces: List[Tuple[int, int, int]] = []
    for i in range(segments):                                # 侧面（逐四边形三角化）
        j = (i + 1) % segments
        faces.append((i, j, segments + j))
        faces.append((i, segments + j, segments + i))
    for i in range(1, segments - 1):                         # 底盖扇形
        faces.append((0, i + 1, i))
    for i in range(1, segments - 1):                         # 顶盖扇形
        faces.append((segments, segments + i, segments + i + 1))
    return Mesh(verts, np.asarray(faces, np.int32), "prism")


def sphere_mesh(radius: float = 0.5, rings: int = 4, segments: int = 8) -> Mesh:
    """低多边形球（rings×segments 四边形带），用于头部等圆润部件。"""
    verts: List[List[float]] = [[0.0, radius, 0.0]]
    for r in range(1, rings):
        phi = math.pi * r / rings
        y = radius * math.cos(phi)
        rr = radius * math.sin(phi)
        for s in range(segments):
            th = 2 * math.pi * s / segments
            verts.append([rr * math.cos(th), y, rr * math.sin(th)])
    verts.append([0.0, -radius, 0.0])
    top, bottom = 0, len(verts) - 1
    faces: List[Tuple[int, int, int]] = []
    for s in range(segments):
        faces.append((top, 1 + (s + 1) % segments, 1 + s))
    for r in range(rings - 2):
        base0 = 1 + r * segments
        base1 = 1 + (r + 1) * segments
        for s in range(segments):
            a = base0 + s
            b = base0 + (s + 1) % segments
            c = base1 + (s + 1) % segments
            d = base1 + s
            faces.append((a, b, c)); faces.append((a, c, d))
    base0 = 1 + (rings - 2) * segments
    for s in range(segments):
        faces.append((bottom, base0 + s, base0 + (s + 1) % segments))
    return Mesh(np.asarray(verts, np.float32), np.asarray(faces, np.int32), "sphere")


# ============================================================ 摄像机 / 光源
@dataclass
class Camera:
    eye: Vec3 = (0.0, 1.90, -3.20)
    target: Vec3 = (0.0, 1.00, 0.30)
    up: Vec3 = (0.0, 1.0, 0.0)
    fov_y_deg: float = 52.0
    near: float = 0.15
    far: float = 60.0

    def view(self) -> Mat4:
        return look_at(self.eye, self.target, self.up)

    def view_inv3(self) -> np.ndarray:
        """视图矩阵左上 3x3 的逆（正交矩阵即转置），用于像素反投影求射线。"""
        return self.view()[:3, :3].T.copy()

    def projection(self, aspect: float) -> Mat4:
        f = 1.0 / math.tan(math.radians(self.fov_y_deg) / 2)
        m = np.zeros((4, 4), np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = self.far / (self.far - self.near)
        m[3, 2] = -self.far * self.near / (self.far - self.near)
        m[2, 3] = 1.0
        return m

    def ray_dirs(self, w: int, h: int) -> np.ndarray:
        """
        返回形状 (h,w,3) 的**单位化**世界空间射线方向，供地面 / 墙面求交使用。

        推导（行向量约定）：v_view = v_world @ M3 + t，M3 = V[:3,:3] 为世界→视图。
        视图空间射线 d_view = (ndc.x*aspect/f, ndc.y/f, 1)，
        则世界方向 = d_view @ inv(M3) = d_view @ M3.T。
        注意 view_inv3() 返回的正是 M3.T，**不要再转置一次**（曾因此多转置一次导致背景层垂直错位）。
        """
        f = 1.0 / math.tan(math.radians(self.fov_y_deg) / 2)
        aspect = w / float(h)
        xs = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
        ys = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0
        gx, gy = np.meshgrid(xs, ys)
        d = np.stack([gx * aspect / f, gy / f, np.ones_like(gx)], axis=-1)
        d = d @ self.view_inv3()
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        return (d / np.maximum(n, 1e-9)).astype(np.float32)


@dataclass
class Light:
    key_dir: Vec3 = (0.34, -1.0, -0.22)      # 主光方向（从光源指向场景）
    key_intensity: float = 0.78
    fill_intensity: float = 0.30             # 补光：来自摄像机方向
    ambient: float = 0.34
    shadow_strength: float = 0.42


# ============================================================ 背景平面
@dataclass
class FloorPlane:
    """地面：由像素反投影射线与 y=0 平面求交，逐像素解析计算颜色。"""
    y: float = 0.0
    x_range: Tuple[float, float] = (-7.0, 7.0)
    z_range: Tuple[float, float] = (-4.0, 12.0)
    tile: float = 0.62
    base_color: Tuple[float, float, float] = (0.72, 0.735, 0.755)
    grout_color: Tuple[float, float, float] = (0.55, 0.565, 0.585)
    grid_fade: float = 0.0

    def shade(self, x: np.ndarray, z: np.ndarray, shadow: np.ndarray) -> np.ndarray:
        """shadow 为已乘过强度系数的遮蔽量（0~1）。"""
        tx = np.floor(x / self.tile)
        tz = np.floor(z / self.tile)
        fx = np.abs(x / self.tile - tx - 0.5) * 2.0
        fz = np.abs(z / self.tile - tz - 0.5) * 2.0
        edge = np.maximum(fx, fz)
        is_grout = edge > 0.90
        # 每块砖的确定性明度扰动，强化透视与材质感
        wob = (np.sin(tx * 12.9898 + tz * 78.233) * 0.5 + 0.5) * 0.055
        col = np.empty(x.shape + (3,), np.float32)
        for i in range(3):
            c = self.base_color[i] + wob - 0.018
            col[..., i] = np.where(is_grout, self.grout_color[i], c)
        col *= (1.0 - np.clip(shadow, 0.0, 0.85))[..., None]
        return col


@dataclass
class WallPlane:
    """后墙 / 侧墙：同样用射线-平面求交解析着色。"""
    axis: str = "z"                          # 'z' 后墙，'x' 侧墙
    pos: float = 9.5
    lo: float = -7.0
    hi: float = 7.0
    v_lo: float = -0.5
    v_hi: float = 4.2
    base_color: Tuple[float, float, float] = (0.845, 0.855, 0.87)
    panel: float = 1.6
    skirt_h: float = 0.12

    def shade(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        col = np.empty(u.shape + (3,), np.float32)
        base = np.array(self.base_color, np.float32)
        # 竖直方向渐变（上方略亮）
        grad = 1.0 + 0.055 * np.clip((v - self.v_lo) / max(self.v_hi - self.v_lo, 1e-6), 0, 1)
        col[:] = base
        col *= grad[..., None]
        # 竖向分缝
        fu = np.abs((u / self.panel) - np.floor(u / self.panel) - 0.5) * 2.0
        col *= np.where(fu > 0.965, 0.93, 1.0)[..., None]
        # 踢脚线
        col *= np.where((v - self.v_lo) < self.skirt_h, 0.86, 1.0)[..., None]
        return col


# ============================================================ 渲染器
class Renderer:
    """
    软件光栅化渲染器。

    参数
    ----
    width, height : 输出分辨率
    ssaa          : 超采样倍率（2 = 内部 2 倍分辨率渲染后降采样，用于抗锯齿）
    fog           : 深度雾系数（0 关闭）；远处物体向背景色靠拢，增强空间感
    """

    SHADOW_SAMPLE_DOWN = 4          # 地面阴影按 1/4 分辨率采样后上采样

    def __init__(self, width: int = 640, height: int = 480, ssaa: int = 2,
                 fog: float = 0.055, bg_top: Vec3 = (0.90, 0.91, 0.93),
                 bg_bottom: Vec3 = (0.80, 0.815, 0.835), shadow_map_res: int = 640,
                 shadow_extent: Tuple[float, float, float, float] = (-6.0, 6.0, -3.0, 11.0),
                 textures=None, cull_backface: bool = True):
        self.w, self.h = int(width), int(height)
        self.ssaa = max(1, int(ssaa))
        self.rw, self.rh = self.w * self.ssaa, self.h * self.ssaa
        self.fog = float(fog)
        self.bg_top = np.asarray(bg_top, np.float32)
        self.bg_bottom = np.asarray(bg_bottom, np.float32)
        self.shadow_res = int(shadow_map_res)
        self.shadow_extent = shadow_extent
        self.tex = textures            # tools.textures.Textures 实例；None 则使用纯色着色
        self.cull_backface = bool(cull_backface)
        self._ray_cache: Dict[Tuple, np.ndarray] = {}
        self._bg_cache_key = None
        self._bg_cache: Optional[Tuple] = None
        self.stats = {"faces_total": 0, "faces_drawn": 0}

    # -------------------------------------------------------- 静态背景缓存
    def _bg_key(self, camera: Camera, floor, walls, W: int, H: int) -> Tuple:
        return (W, H, tuple(camera.eye), tuple(camera.target), camera.fov_y_deg,
                repr(floor), tuple(repr(w) for w in walls), self.fog)

    def _build_background(self, camera: Camera, floor, walls, W: int, H: int):
        """
        计算背景层（天空渐变 + 地面 + 墙面 + 深度）。

        这一层只与摄像机、地面/墙面参数有关，**与逐帧的阴影无关**，因此缓存起来，
        每帧只对地面像素做一次阴影调制即可（否则每帧重算 123 万像素的着色会拖慢 10 倍以上）。
        返回：(bg_color, bg_depth, floor_base, floor_mask, floor_x, floor_z)
        """
        g = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
        color = self.bg_top[None, None, :] * (1 - g) + self.bg_bottom[None, None, :] * g
        color = np.repeat(color, W, axis=1).astype(np.float32)
        depth = np.full((H, W), np.inf, np.float32)
        floor_base = floor_mask = floor_x = floor_z = None

        rd = self._rays(W, H, camera)
        if floor is not None:
            oy = float(camera.eye[1])
            dy = rd[..., 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (floor.y - oy) / dy
            hit = (dy < -1e-6) & (t > 0)
            px = camera.eye[0] + t * rd[..., 0]
            pz = camera.eye[2] + t * rd[..., 2]
            hit &= (px >= floor.x_range[0]) & (px <= floor.x_range[1])
            hit &= (pz >= floor.z_range[0]) & (pz <= floor.z_range[1])
            if self.tex is not None:
                col = self.tex.floor_rgb(px, pz)
            else:
                col = floor.shade(px, pz, np.zeros_like(px))         # 不带阴影的基色
            col *= (1.0 - np.clip(self.fog * t, 0, 0.55))[..., None]
            color[hit] = col[hit]
            depth[hit] = t[hit]
            floor_base, floor_mask = col, hit
            floor_x, floor_z = px, pz

        for wall in walls:
            if wall.axis == "z":
                dz = rd[..., 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (wall.pos - camera.eye[2]) / dz
                hit = (dz > 1e-6) & (t > 0)
                u = camera.eye[0] + t * rd[..., 0]
                vv = camera.eye[1] + t * rd[..., 1]
            else:
                dx = rd[..., 0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (wall.pos - camera.eye[0]) / dx
                hit = (dx > 1e-6) & (t > 0)
                u = camera.eye[2] + t * rd[..., 2]
                vv = camera.eye[1] + t * rd[..., 1]
            hit &= (u >= wall.lo) & (u <= wall.hi) & (vv >= wall.v_lo) & (vv <= wall.v_hi)
            hit &= t < depth
            wcol = self.tex.wall_rgb(u, vv) if self.tex is not None else wall.shade(u, vv)
            wcol *= (1.0 - np.clip(self.fog * t, 0, 0.55))[..., None]
            color[hit] = wcol[hit]
            depth[hit] = t[hit]

        return color, depth, floor_base, floor_mask, floor_x, floor_z

    def _floor_lowres(self, floor_x: np.ndarray, floor_z: np.ndarray, floor_mask: np.ndarray):
        """
        把地面世界坐标的采样网格降到 1/4 分辨率缓存。

        阴影是平滑的大面积量，按 1/4 分辨率采样再上采样即可，可把全画幅阴影采样开销降低约 16 倍。
        """
        k = self.SHADOW_SAMPLE_DOWN
        if k <= 1:
            return floor_x, floor_z, floor_mask, None
        H, W = floor_x.shape
        h2, w2 = max(1, H // k), max(1, W // k)
        fx = cv2.resize(floor_x, (w2, h2), interpolation=cv2.INTER_NEAREST)
        fz = cv2.resize(floor_z, (w2, h2), interpolation=cv2.INTER_NEAREST)
        fm = cv2.resize(floor_mask.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
        return fx, fz, fm, (w2, h2)

    @staticmethod
    def _floor_rect(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """地面掩膜的外接矩形，用于把逐帧更新限制在该矩形内（切片 + 布尔掩膜）。"""
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) == 0 or len(cols) == 0:
            return None
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    # -------------------------------------------------------- 阴影图
    def make_shadow_map(self, occluders: List[Tuple[Mesh, Mat4]], light: Light) -> np.ndarray:
        sm = np.zeros((self.shadow_res, self.shadow_res), np.float32)
        L = np.asarray(light.key_dir, np.float32)
        L = L / np.linalg.norm(L)
        if abs(L[1]) < 1e-4:
            return sm
        x0, x1, z0, z1 = self.shadow_extent
        for mesh, mat in occluders:
            wm = mesh.transformed(mat)
            v = wm.verts
            t = -v[:, 1] / L[1]                    # 沿光照方向投到 y=0
            p = v + t[:, None] * L
            pts = np.stack([(p[:, 0] - x0) / (x1 - x0) * (self.shadow_res - 1),
                            (p[:, 2] - z0) / (z1 - z0) * (self.shadow_res - 1)], axis=1)
            if len(pts) < 3:
                continue
            hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
            m = np.zeros_like(sm, np.uint8)
            cv2.fillConvexPoly(m, hull, 1)
            sm = np.maximum(sm, m.astype(np.float32))
        if self.ssaa > 1 or True:
            k = max(3, int(self.shadow_res / 220) | 1)
            sm = cv2.GaussianBlur(sm, (k, k), 0)
        return sm

    def sample_shadow(self, sm: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x0, x1, z0, z1 = self.shadow_extent
        u = (x - x0) / (x1 - x0) * (self.shadow_res - 1)
        v = (z - z0) / (z1 - z0) * (self.shadow_res - 1)
        ui = np.clip(np.round(u).astype(np.int32), 0, self.shadow_res - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, self.shadow_res - 1)
        out = sm[vi, ui].copy()
        out[(u < 0) | (u > self.shadow_res - 1) | (v < 0) | (v > self.shadow_res - 1)] = 0.0
        return out

    # -------------------------------------------------------- 主渲染
    def render(self, items: List[Tuple[Mesh, Mat4, Vec3]], camera: Camera, light: Light,
               floor: Optional[FloorPlane] = None,
               walls: Sequence[WallPlane] = (),
               shadow_occluders: Optional[List[Tuple[Mesh, Mat4]]] = None) -> np.ndarray:
        """
        items : [(mesh, world_matrix, rgb_color)]，颜色取值 0~1
        返回 BGR uint8 图像
        """
        W, H = self.rw, self.rh
        aspect = W / float(H)
        V = camera.view()
        P = camera.projection(aspect)
        VP = V @ P

        # ---- 背景层（天空+地面+墙面）：静态部分缓存，逐帧只叠加阴影与物体
        key = self._bg_key(camera, floor, walls, W, H)
        if key != self._bg_cache_key:
            bg_color, bg_depth, floor_base, floor_mask, floor_x, floor_z = \
                self._build_background(camera, floor, walls, W, H)
            fx_s, fz_s, fm_s, shape_s = (None, None, None, None)
            rect = None
            if floor is not None and floor_mask is not None and floor_mask.any():
                fx_s, fz_s, fm_s, shape_s = self._floor_lowres(floor_x, floor_z, floor_mask)
                rect = self._floor_rect(floor_mask)
            self._bg_cache = {
                "bg_color": bg_color, "bg_depth": bg_depth,
                "floor_base": floor_base, "floor_mask": floor_mask, "floor_rect": rect,
                "fx_s": fx_s, "fz_s": fz_s, "fm_s": fm_s, "shape_s": shape_s,
                "work_color": bg_color.copy(), "work_depth": bg_depth.copy(),
            }
            self._bg_cache_key = key
        c = self._bg_cache
        color, depth = c["work_color"], c["work_depth"]
        bg_color, bg_depth = c["bg_color"], c["bg_depth"]

        # ---- 还原上一帧被三维物体覆盖的像素（比整幅背景拷贝便宜得多：
        #      物体像素 = 深度小于背景深度的像素）
        restore = depth < bg_depth
        if restore.any():
            color[restore] = bg_color[restore]
            depth[restore] = bg_depth[restore]

        # ---- 阴影：低分辨率采样 → 上采样 → 在地面外接矩形内按掩膜更新
        if floor is not None and c["floor_base"] is not None and shadow_occluders and c["floor_rect"]:
            smap = self.make_shadow_map(shadow_occluders, light)
            sh_s = self.sample_shadow(smap, c["fx_s"], c["fz_s"]) * light.shadow_strength
            sh_full = cv2.resize(sh_s, (W, H), interpolation=cv2.INTER_LINEAR)
            r0, r1, c0, c1 = c["floor_rect"]
            msub = c["floor_mask"][r0:r1, c0:c1]
            np.multiply(c["floor_base"][r0:r1, c0:c1],
                        (1.0 - np.clip(sh_full[r0:r1, c0:c1], 0.0, 0.85))[..., None],
                        out=color[r0:r1, c0:c1], where=msub[..., None])

        # ---- 三维物体
        L = np.asarray(light.key_dir, np.float32)
        L = L / np.linalg.norm(L)
        self.stats["faces_total"] = 0
        self.stats["faces_drawn"] = 0
        for mesh, mat, rgb in items:
            self._draw_mesh(mesh, mat, np.asarray(rgb, np.float32), camera, VP, W, H,
                            color, depth, L, light)

        img = np.clip(color, 0.0, 1.0)
        if self.ssaa > 1:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # 内部颜色缓冲为 RGB；此处**只做一次** RGB→BGR 转换。
        # 若同时再写一次 img[:, :, ::-1] 就会交换两次通道，导致红蓝互换
        # （症状：蓝色上衣渲染成橙褐色、绿色餐盒显示成青色）。
        return cv2.cvtColor((img * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------- 内部
    def _rays(self, W: int, H: int, camera: Camera) -> np.ndarray:
        key = (W, H, tuple(camera.eye), tuple(camera.target), camera.fov_y_deg)
        if key not in self._ray_cache:
            self._ray_cache[key] = camera.ray_dirs(W, H)
        return self._ray_cache[key]

    def _draw_mesh(self, mesh: Mesh, mat: Mat4, base_color: np.ndarray, camera: Camera,
                   VP: Mat4, W: int, H: int, color: np.ndarray, depth: np.ndarray,
                   L: np.ndarray, light: Light) -> None:
        wm = mesh.transformed(mat)
        wv = wm.verts
        if len(wv) == 0:
            return
        vh = np.concatenate([wv, np.ones((len(wv), 1), np.float32)], axis=1)
        clip = vh @ VP                       # 行向量约定：v_clip = v_world @ V @ P
        w = clip[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc = clip[:, :3] / w[:, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * W
        sy = (0.5 - ndc[:, 1] * 0.5) * H
        sz = w

        f = mesh.faces
        self.stats["faces_total"] += len(f)
        # 向量化剔除：三点中任一在近裁剪面之后，或整体落在屏幕外，则跳过
        vis = (sz[f[:, 0]] > camera.near) & (sz[f[:, 1]] > camera.near) & (sz[f[:, 2]] > camera.near)
        fx = sx[f]; fy = sy[f]
        fxmin, fxmax = fx.min(axis=1), fx.max(axis=1)
        fymin, fymax = fy.min(axis=1), fy.max(axis=1)
        vis &= (fxmax >= 0) & (fxmin <= W - 1) & (fymax >= 0) & (fymin <= H - 1)
        idx = np.nonzero(vis)[0]
        self.stats["faces_drawn"] += len(idx)
        if len(idx) == 0:
            return

        # 面法线（世界空间），朝向摄像机翻转以实现双面着色
        a = wv[f[idx, 0]]; b = wv[f[idx, 1]]; c = wv[f[idx, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        cen = (a + b + c) / 3.0
        view_vec = np.asarray(camera.eye, np.float32) - cen
        view_vec /= np.maximum(np.linalg.norm(view_vec, axis=1, keepdims=True), 1e-9)
        back = np.einsum("ij,ij->i", n, view_vec) < 0        # 背对摄像机

        if self.cull_backface:
            # 背面剔除：闭合网格中背对相机的面必然被前方遮挡，光栅化它们纯属浪费。
            # 实测可少画约一半三角形，端到端提速约 1.6~1.8 倍。
            # 仅对**闭合且绕向一致**的网格安全；本项目的立方体/棱柱/球体均满足。
            # 注意：所有并行数组（idx/a/b/c/n/cen/view_vec）必须同步过滤，漏掉任一个都会广播失败。
            front = ~back
            if not front.any():
                return
            idx, a, b, c = idx[front], a[front], b[front], c[front]
            n, cen, view_vec = n[front], cen[front], view_vec[front]
            back = back[front]
        n = np.where(back[:, None], -n, n)

        key = np.clip(n @ L, 0.0, 1.0) * light.key_intensity
        fill = np.clip(np.einsum("ij,ij->i", n, view_vec), 0.0, 1.0) * light.fill_intensity
        shade = np.clip(light.ambient + key + fill, 0.0, 1.35)

        if self.tex is not None:
            # 逐面用**局部**归一化坐标采样材质，消除纯色塑料感。
            # 仅按面中心取一个颜色（平面着色），不做逐像素 UV 插值——在约 30px 的面尺度下
            # 与逐像素贴图的观感差异极小，而开销低一个数量级。
            lcen = mesh.verts[f[idx]].mean(axis=1)
            lo, hi = mesh.verts.min(axis=0), mesh.verts.max(axis=0)
            ext = np.maximum(hi - lo, 1e-6)
            uv = (lcen[:, :2] - lo[:2]) / ext[:2]
            mat = self.tex.infer_material(tuple(float(x) for x in base_color))
            # base_override 必须传：纹理只提供图案与明暗变化，基色一律用调用方给的材质色，
            # 否则 infer_material 会把新配色替换成固定材质表里最接近的颜色（曾使蓝色操作屏渲染成绿色）。
            face_base = np.array([self.tex.face_color(mat, float(u), float(v), base_color)
                                  for u, v in uv], np.float32)
        else:
            face_base = np.repeat(base_color[None, :], len(idx), axis=0)
        face_rgb = np.clip(face_base * shade[:, None], 0.0, 1.0)

        tri_s = np.stack([sx[f[idx]], sy[f[idx]]], axis=2).astype(np.float32)   # (K,3,2)
        tri_z = sz[f[idx]].mean(axis=1)                                        # 常量深度近似
        order = np.argsort(-tri_z)                                             # 远到近，减少无效写入

        for k in order:
            tri = tri_s[k]
            z = tri_z[k]
            x0 = int(max(0, math.floor(tri[:, 0].min())))
            x1 = int(min(W - 1, math.ceil(tri[:, 0].max())))
            y0 = int(max(0, math.floor(tri[:, 1].min())))
            y1 = int(min(H - 1, math.ceil(tri[:, 1].max())))
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if bw <= 0 or bh <= 0:
                continue
            local = (tri - (x0, y0)).astype(np.float32)
            m = np.zeros((bh, bw), np.uint8)
            cv2.fillConvexPoly(m, np.round(local).astype(np.int32), 1, cv2.LINE_8)
            # 用「子矩形切片 + 布尔掩膜」写入，而不是双索引数组的 fancy indexing：
            # 前者作用在连续视图上，对大面积三角形可快 2~3 倍。
            dsub = depth[y0:y1 + 1, x0:x1 + 1]
            sel = (m != 0) & (z < dsub)
            if not sel.any():
                continue
            color[y0:y1 + 1, x0:x1 + 1][sel] = face_rgb[k]
            dsub[sel] = z

    # -------------------------------------------------------- 工具
    @staticmethod
    def bgr_to_png(path: str, img: np.ndarray) -> None:
        """中文路径安全的写图。"""
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(path)
