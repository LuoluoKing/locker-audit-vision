# -*- coding: utf-8 -*-
"""
detector.py — 目标检测后端（YOLOv8n 优先，HOG 自动回退）

设计要点
--------
1. 上层只依赖统一接口 ``BaseDetector.detect(frame) -> List[Detection]``，
   不感知底层是 YOLOv8n 还是 HOG；
2. YOLOv8n 依赖 ultralytics + torch。若依赖缺失、模型下载/加载失败、或
   warmup 推理抛异常，``build_detector`` 会自动回退到 OpenCV 内置的
   HOG + 线性 SVM 行人检测器，保证系统在纯 OpenCV 环境下也能跑通；
3. 所有耗时均为真实测量值（``last_infer_ms``），不使用任何模拟数据。

仅使用 person (COCO class 0) 类别，与校园外卖取餐场景一致。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("tianyan.detector")

PERSON = "person"
__all__ = ["Detection", "BaseDetector", "YoloDetector", "HogDetector", "build_detector"]


# ---------------------------------------------------------------- 数据结构
@dataclass
class Detection:
    """单个目标框。坐标使用图像像素坐标，浮点存储，绘制时取整。"""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = PERSON
    conf: float = 0.0          # HOG 后端为等价置信度（由 SVM 权重映射），非概率值

    # ---- 便捷属性
    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """脚点。判定“人站在某区域”时比中心点更符合透视关系。"""
        return (self.x1 + self.x2) / 2.0, self.y2

    def xyxy(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def to_dict(self) -> dict:
        return {"bbox": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
                "label": self.label, "conf": round(float(self.conf), 4)}


def iou(a: Detection, b: Detection) -> float:
    """两个检测框的交并比。"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------- 基类
class BaseDetector:
    backend_name = "base"
    display_name = "未初始化"

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = float(conf_threshold)
        self.last_infer_ms = 0.0
        self.infer_count = 0

    # 子类实现
    def _infer(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        dets = self._infer(frame)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self.infer_count += 1
        return dets

    def warmup(self, size: Tuple[int, int] = (480, 640)) -> None:
        """用合成帧跑一次真实推理，验证后端确实可用（也是 CPU 可用性的初步验证）。"""
        h, w = size
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        blank[:] = 40
        self.detect(blank)

    def describe(self) -> dict:
        return {"backend": self.backend_name, "display_name": self.display_name,
                "conf_threshold": self.conf_threshold, "infer_count": self.infer_count,
                "last_infer_ms": round(self.last_infer_ms, 2)}


# ---------------------------------------------------------------- YOLOv8n
class YoloDetector(BaseDetector):
    """Ultralytics YOLOv8 检测后端（CPU 推理，仅取 person 类）。"""

    backend_name = "yolov8n"

    def __init__(self, model_path: str, conf_threshold: float = 0.35, iou_threshold: float = 0.5,
                 imgsz: int = 640, device: str = "cpu", allow_download: bool = True,
                 torch_threads: int = 0):
        super().__init__(conf_threshold)
        from ultralytics import YOLO  # 延迟导入：缺失时由工厂捕获并回退

        self.model_path = str(model_path)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device
        self.allow_download = allow_download

        # CPU 推理线程数：实测 8 线程平台最优（43.4ms vs 2线程 52.3ms），
        # 超过物理核心数后无增益甚至反降，故默认取物理核心数并封顶 8。
        import os
        n_threads = 0
        if str(device).lower().startswith("cpu"):
            try:
                import torch
                if torch_threads and torch_threads > 0:
                    n_threads = int(torch_threads)
                else:
                    n_threads = min(8, os.cpu_count() or 4)
                torch.set_num_threads(n_threads)
            except Exception:
                n_threads = 0
        self.torch_threads = n_threads

        import os
        exists = os.path.exists(self.model_path)
        if not exists and not allow_download:
            raise FileNotFoundError("本地模型不存在且禁止下载：%s" % self.model_path)

        self.model = YOLO(self.model_path)      # 文件不存在时 ultralytics 会尝试自动下载
        self.model.to(self.device)
        # 后端名与展示名按权重文件名判定：同一实现同时覆盖 YOLOv8n / YOLO11n
        # （对照实验用；两条路径在下游完全一致，仅权重不同）。
        stem = os.path.splitext(os.path.basename(self.model_path))[0].lower()
        if stem.startswith("yolo11"):
            self.backend_name = "yolo11n"
            pretty = "YOLO11n"
        else:
            self.backend_name = "yolov8n"
            pretty = "YOLOv8n"
        self.display_name = "%s（Ultralytics, %s, imgsz=%d）" % (pretty, self.device.upper(), self.imgsz)
        self.classes = [0]                       # COCO: 0 -> person

        try:
            import ultralytics, torch
            self.versions = {"ultralytics": getattr(ultralytics, "__version__", "?"),
                             "torch": getattr(torch, "__version__", "?")}
        except Exception:
            self.versions = {}

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        dets = [Detection(float(x1), float(y1), float(x2), float(y2), PERSON, float(c))
                for (x1, y1, x2, y2), c in zip(xyxy, confs)]
        return _suppress_nested_boxes(dets)

    def describe(self) -> dict:
        d = super().describe()
        d["model"] = self.model_path
        d["imgsz"] = self.imgsz
        d["torch_threads"] = getattr(self, "torch_threads", 0)
        d["versions"] = getattr(self, "versions", {})
        return d


def _suppress_nested_boxes(dets: List[Detection], contain_ratio: float = 0.70) -> List[Detection]:
    """
    抑制「嵌套框」：被同类更大框包含 ≥ contain_ratio 的框视为重复检出丢弃。

    依据：YOLO 的 NMS 按 IoU 抑制，但「上半身框嵌在全身框内」这类部分重叠的
    双框 IoU 只有 0.3~0.4（＜0.5），不会被 NMS 消掉；同一人因此占两条轨迹，
    使 entries / 停留时长被分摊，LOITERING、MULTI_RETRIEVAL 等判定全部失灵
    （实测曾使同一人 1.3s/2.5s 各记一次「进入取餐区」）。
    """
    if len(dets) <= 1:
        return dets
    keep: List[Detection] = []
    for i, a in enumerate(dets):
        aa = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
        nested = False
        for j, b in enumerate(dets):
            if i == j or b.conf < a.conf:
                continue
            ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
            ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if aa > 0 and inter / aa >= contain_ratio:
                nested = True
                break
        if not nested:
            keep.append(a)
    return keep


# ---------------------------------------------------------------- HOG 回退
class HogDetector(BaseDetector):
    """
    OpenCV 内置 HOG + 线性 SVM 行人检测器（回退后端）。

    为控制 CPU 耗时，先把画面等比缩放到 ``resize_width`` 再检测，得到框后
    按缩放比映射回原分辨率。SVM 决策值经 Logistic 函数映射为 0~1 的等价
    置信度，便于与 YOLO 的 conf 使用同一阈值逻辑。
    """

    backend_name = "hog"

    def __init__(self, conf_threshold: float = 0.35, resize_width: int = 320,
                 win_stride: Tuple[int, int] = (8, 8), padding: Tuple[int, int] = (16, 16),
                 scale: float = 1.05, min_area_ratio: float = 0.004, nms_threshold: float = 0.45):
        super().__init__(conf_threshold)
        self.resize_width = int(resize_width)
        self.win_stride = tuple(win_stride)
        self.padding = tuple(padding)
        self.scale = float(scale)
        self.min_area_ratio = float(min_area_ratio)
        self.nms_threshold = float(nms_threshold)
        self.display_name = "HOG + 线性SVM 行人检测（OpenCV 内置, CPU）"

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _svm_to_conf(weight: float) -> float:
        """SVM 决策值 -> (0,1) 等价置信度。权重越大越像行人。"""
        return float(1.0 / (1.0 + np.exp(-float(weight))))

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        h0, w0 = frame.shape[:2]
        if self.resize_width and w0 > self.resize_width:
            k = self.resize_width / float(w0)
            small = cv2.resize(frame, (self.resize_width, max(1, int(h0 * k))), interpolation=cv2.INTER_AREA)
        else:
            k = 1.0
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects, weights = self.hog.detectMultiScale(
            gray, winStride=self.win_stride, padding=self.padding, scale=self.scale)

        dets: List[Detection] = []
        min_area = self.min_area_ratio * gray.shape[0] * gray.shape[1]
        for (x, y, w, h), wt in zip(rects, weights):
            if w * h < min_area:
                continue
            dets.append(Detection(x / k, y / k, (x + w) / k, (y + h) / k, PERSON, self._svm_to_conf(wt)))

        dets = [d for d in dets if d.conf >= self.conf_threshold]
        return self._nms(dets)

    def _nms(self, dets: List[Detection]) -> List[Detection]:
        if len(dets) < 2:
            return dets
        boxes = [[d.x1, d.y1, d.w, d.h] for d in dets]
        scores = [d.conf for d in dets]
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        if idx is None or len(idx) == 0:
            return []
        return [dets[int(i)] for i in np.array(idx).flatten()]


# ================================================================ 场景状态观测
# 下列两个估计器把「画面」转换为 rules.SceneObservation 所需的观测项：
#   DoorStateEstimator  -> 各柜格门是否开启（二值代理量）
#   PackageDetector     -> 各柜格内可见餐盒数、以及每个人手持餐盒数
# 两者均为经典计算机视觉方法（亮度对比 / 颜色分割 + 连通域），不依赖任何训练数据。


@dataclass
class Blob:
    """颜色连通域检测到的一个目标区域。"""
    x1: float
    y1: float
    x2: float
    y2: float
    area: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DoorStateEstimator:
    """
    柜门状态视觉估计（二值：开 / 关）。

    原理：柜门关闭时，柜格区域显示的是**受光的门板**；柜门开启后，露出的是**柜内暗区**。
    因此统计柜格区域内「显著暗于全画面中位亮度」的像素占比（`dark_ratio`），
    高于阈值即判为「门开」。以全画面中位亮度为参考，可抵消整体光照变化。

    **必须排除人员遮挡像素**：实测发现站在柜前的人（深色衣物）会把柜格区域整体压暗，
    使门开与门关时的暗像素占比都接近 0.6，判定恒为「开」（准确率仅 82%）。
    因此本实现先扣掉人员检测框覆盖的像素；若剩余可见面积不足柜格面积的 `min_visible`，
    则返回 ``None`` 表示**状态未知**，由上层保持上一次状态，不产生误判。

    输出的是**二值代理量**（0 或 90 度），不是真实开启角——真实角度存在于数据集的逐帧真值中，
    仅用于阶段3 的评估，不参与系统判定。这一点在论文中必须如实说明。
    """

    def __init__(self, compartments_px: Sequence[np.ndarray], dark_threshold_ratio: float = 0.62,
                 open_dark_ratio: float = 0.15, min_visible: float = 0.50):
        self.compartments = [np.asarray(c, np.int32) for c in compartments_px]
        self.dark_threshold_ratio = float(dark_threshold_ratio)
        self.open_dark_ratio = float(open_dark_ratio)
        self.min_visible = float(min_visible)
        self.last_dark_ratios: List[float] = [0.0] * len(self.compartments)
        self.last_visible: List[float] = [1.0] * len(self.compartments)

    def estimate(self, frame: np.ndarray, person_boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None
                 ) -> Dict[int, Optional[float]]:
        """返回 柜格号 -> 0.0(关) / 90.0(开) / None(人员遮挡过多，状态未知)。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        thr = self.dark_threshold_ratio * med
        h, w = gray.shape[:2]

        occ = np.zeros((h, w), np.uint8)
        for box in (person_boxes or []):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(occ, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), 1, -1)

        out: Dict[int, Optional[float]] = {}
        for k, poly in enumerate(self.compartments):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            total = int(np.count_nonzero(mask))
            if total == 0:
                out[k] = None
                continue
            mask[occ > 0] = 0                      # 扣除人员遮挡
            vals = gray[mask > 0]
            vis = vals.size / float(total)
            self.last_visible[k] = vis
            if vis < self.min_visible:             # 可见面积不足 → 状态未知
                out[k] = None
                continue
            ratio = float(np.count_nonzero(vals < thr)) / float(vals.size)
            self.last_dark_ratios[k] = ratio
            out[k] = 90.0 if ratio >= self.open_dark_ratio else 0.0
        return out


class PackageDetector:
    """
    外卖餐盒检测（HSV 色域分割 + 连通域）。

    判据来自实测（本机三维素材采样的 HSV 分布，OpenCV 色相 H ∈ [0,180]）：
        ｜目标       ｜ H     ｜ S     ｜ 说明
        ｜餐盒-黄    ｜ 20–45 ｜170–190｜黄/橙/绿三色餐盒，**高饱和**
        ｜餐盒-绿    ｜  76   ｜  149  ｜（曾因色相上限设 62 而漏检绿盒）
        ｜皮肤       ｜ 13–17 ｜ 84–124｜手/头，**饱和度不足**
        ｜蓝色衣物   ｜  110  ｜151–158｜饱和度虽高但**色相在蓝区**
        ｜柜体地面   ｜  —    ｜ <60   ｜低饱和灰系
    因此用色相区间 [15,95] + `sat_min=130` 可同时覆盖三种餐盒并排除皮肤与蓝色衣物；
    再用面积区间排除大面积衣物与噪声小块。
    参数可在 config 侧按新素材重新标定，不得写死到调用处。
    """

    def __init__(self, hue_min: int = 15, hue_max: int = 95, sat_min: int = 130,
                 val_min: int = 90, area_min: int = 60, area_max: int = 1200,
                 aspect_max: float = 2.6):
        self.hue_min, self.hue_max = int(hue_min), int(hue_max)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.area_min = int(area_min)
        self.area_max = int(area_max)
        self.aspect_max = float(aspect_max)

    def detect(self, frame: np.ndarray) -> List[Blob]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([self.hue_min, self.sat_min, self.val_min], np.uint8)
        hi = np.array([self.hue_max, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs: List[Blob] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.area_min or area > self.area_max:
                continue
            if max(bw / max(bh, 1), bh / max(bw, 1)) > self.aspect_max:
                continue
            blobs.append(Blob(float(x), float(y), float(x + bw), float(y + bh), float(area)))
        return blobs


def count_packages_in_compartments(blobs: Sequence[Blob],
                                   compartments_px: Sequence[np.ndarray],
                                   person_boxes: Sequence[Tuple[float, float, float, float]] = ()
                                   ) -> Dict[int, int]:
    """
    统计各柜格多边形内可见餐盒数（以餐盒中心点是否落在柜格内为准）。

    **人物框内的餐盒一律不计入柜格**：人站在柜前时，手中餐盒的投影会落进柜格
    四边形（曾在实测中把「手持 1 份」计成「柜格 3 内存 1 份」）。
    餐盒 blob 落在人员检测框内 → 视为手持，归 ``count_carried_packages`` 管。
    """
    out: Dict[int, int] = {k: 0 for k in range(len(compartments_px))}
    for b in blobs:
        c = b.center
        if any(x1 <= c[0] <= x2 and y1 <= c[1] <= y2
               for x1, y1, x2, y2 in person_boxes):
            continue
        for k, poly in enumerate(compartments_px):
            if cv2.pointPolygonTest(np.asarray(poly, np.int32), (float(c[0]), float(c[1])), False) >= 0:
                out[k] += 1
                break
    return out


def count_carried_packages(blobs: Sequence[Blob], person_boxes: Sequence[Tuple[float, float, float, float]],
                           compartments_px: Sequence[np.ndarray],
                           expand_w: float = 0.35, expand_h: float = 0.15
                           ) -> List[Tuple[Tuple[float, float, float, float], int]]:
    """
    统计每个人**手持**的餐盒数。

    判定为「手持」：中心落在**外扩后**的人员检测框内即算，其中框的左右各外扩
    ``expand_w`` × 框宽、上下各外扩 ``expand_h`` × 框高。

    外扩依据（实测）：人**横举手臂**把餐盒拿到身体侧方时，YOLOv8n 的检测框并不
    覆盖伸出的手——实测 `multi_retrieval_006` 第 7.60 s 帧，餐盒中心 (462,213)、
    检测框右边界 430，用「中心在框内」判定只数到 1 份，导致「一人取多份」漏检
    （该类 21 段中 6 段漏检）。外扩后该餐盒被正确计入。

    为防止外扩把**层板上的餐盒**误算成手持，追加一条守卫：中心落在外扩区但
    **不在原框内**的 blob，只有在**不位于任何柜格多边形内**时才计入手持。
    手中餐盒的投影落进柜格四边形的情况由「在原框内」这一支覆盖，行为不变。

    **近邻 blob 合并**：同一份餐盒若是双色（如黄盒身 + 橙色顶盖），颜色分割会
    拆成两个紧挨的连通域（实测间隙约 6px），按个数计会把「手持 1 份」误判成
    「手持 2 份」。因此先按「外接框间隙 ≤ merge_gap_px」合并同一人框内的相邻
    blob，再计数；柜格计数不受影响（层板上并排两份餐盒间距远大于该阈值）。
    返回 [(人员框, 手持餐盒数), ...]。
    """
    merge_gap_px = 10.0
    res = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        ex1, ex2 = x1 - expand_w * (x2 - x1), x2 + expand_w * (x2 - x1)
        ey1, ey2 = y1 - expand_h * (y2 - y1), y2 + expand_h * (y2 - y1)
        inside: List[Blob] = []
        for b in blobs:
            cx, cy = b.center
            if not (ex1 <= cx <= ex2 and ey1 <= cy <= ey2):
                continue
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):        # 落在「外扩区但非原框」内
                if any(cv2.pointPolygonTest(np.asarray(q, np.int32), (float(cx), float(cy)), False) >= 0
                       for q in compartments_px):
                    continue                                   # 属于层板上的餐盒，不算手持
            inside.append(b)
        # 近邻合并：按 x 排序，间隙小于阈值视为同一份餐盒
        inside.sort(key=lambda b: b.center[0])
        merged: List[List[Blob]] = []
        for b in inside:
            if merged and b.x1 - merged[-1][-1].x2 <= merge_gap_px:
                merged[-1].append(b)
            else:
                merged.append([b])
        res.append((box, len(merged)))
    return res


# ---------------------------------------------------------------- 工厂 + 自动回退
def build_detector(backend: str = "auto", *, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35,
                   iou_threshold: float = 0.5, imgsz: int = 640, device: str = "cpu",
                   hog_resize: int = 320, allow_download: bool = True,
                   torch_threads: int = 0) -> BaseDetector:
    """
    构建检测后端。

    backend:
      - "auto"    : 尝试 YOLOv8n，失败自动回退 HOG（默认）
      - "yolov8n" : 指定 YOLOv8n，失败同样回退（并记录告警），保证系统不停机
      - "yolo11n" : 指定 YOLO11n（对照实验用；权重 models/yolo11n.pt，缺失即回退）
      - "hog"     : 直接使用 HOG

    返回的实例必定可用（``warmup`` 已通过）。
    """
    backend = (backend or "auto").lower()
    errors: List[str] = []

    if backend == "yolo11n":
        # 若调用方仍传了 yolov8n 的路径，替换为同目录下的 yolo11n 权重（调用点无需改动）
        if "11" not in os.path.basename(str(model_path)):
            model_path = os.path.join(os.path.dirname(str(model_path)), "yolo11n.pt")

    if backend in ("auto", "yolov8n", "yolo11n", "yolo", "yolov8"):
        try:
            det = YoloDetector(model_path=model_path, conf_threshold=conf_threshold,
                               iou_threshold=iou_threshold, imgsz=imgsz, device=device,
                               allow_download=allow_download, torch_threads=torch_threads)
            det.warmup()
            log.info("检测后端就绪：%s", det.describe())
            return det
        except Exception as exc:                       # 依赖缺失 / 权重下载失败 / 推理异常
            msg = "%s: %s" % (type(exc).__name__, exc)
            errors.append("%s 不可用 -> %s" % (backend, msg))
            log.warning("%s 后端初始化失败，将回退 HOG。原因：%s", backend, msg)

    try:
        det = HogDetector(conf_threshold=conf_threshold, resize_width=hog_resize)
        det.warmup()
        det.fallback_errors = errors
        log.info("检测后端就绪：%s", det.describe())
        return det
    except Exception as exc:
        errors.append("HOG 不可用 -> %s: %s" % (type(exc).__name__, exc))
        raise RuntimeError("所有检测后端均不可用：\n  " + "\n  ".join(errors))
