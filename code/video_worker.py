# -*- coding: utf-8 -*-
"""
video_worker.py — 视频处理工作线程（本地视频文件 / 图片序列，不依赖摄像头）

职责
----
1. 从 mp4/avi/mkv 等视频文件或图片序列目录顺序读帧（严格拒绝摄像头输入）；
2. 按设定帧率上限调用 detector + rules 完成推理与行为判定；
3. 命中事件时保存截图（data/snapshots/）、写入 SQLite、写运行日志；
4. 维护一帧“带标注的最新画面”供 Flask 以 MJPEG 推流；
5. 周期性上报设备状态、会话统计（供管理后台展示）。

路径兼容：Windows 中文路径下 OpenCV 视频读取可能失败，
``_open_capture`` 会自动把源复制到 ASCII 临时目录后重试；
图片读写统一使用 ``imdecode/imencode + tofile``，避免中文路径乱码。
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (AI_DEMO_DOOR_PROFILE, AI_DEMO_FRAME_BANNER, FRAME_HEIGHT, FRAME_WIDTH,
                    JPEG_QUALITY, SNAP_DIR, is_ai_demo_source)
from database import Database
from detector import (BaseDetector, DoorStateEstimator, PackageDetector,
                      count_carried_packages, count_packages_in_compartments)
from rules import PackageCarry, RuleEngine, RuleEvent, SceneObservation

log = logging.getLogger("tianyan.worker")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")

# 标注配色（BGR）
C_ROI_PICKUP = (0, 140, 255)     # 取餐区：橙
C_DOOR_CLOSED = (170, 175, 185)  # 柜格（门关）：灰
C_DOOR_OPEN = (40, 200, 240)     # 柜格（门开）：琥珀
C_BOX = (80, 200, 80)            # 普通目标框：绿
C_BOX_ALERT = (60, 60, 235)      # 告警目标框：红
C_HUD = (240, 240, 240)
C_HUD_ALERT = (60, 60, 255)
C_DEMO_NOTE = (0, 210, 255)      # AI 生成演示素材的声明条：琥珀


# ---------------------------------------------------------------- 路径工具
def _ascii_tmp_dir() -> Path:
    """返回可写且路径为 ASCII 的临时目录（用于规避 Windows 中文路径限制）。"""
    cands: List[str] = []
    if os.environ.get("TIANYAN_TMP"):
        cands.append(os.environ["TIANYAN_TMP"])
    cands += [str(tempfile.gettempdir()), "D:/tianyan_tmp", "C:/tianyan_tmp",
              str(Path.home() / ".tianyan_tmp")]
    for c in cands:
        try:
            p = Path(c)
            p.mkdir(parents=True, exist_ok=True)
            if str(p).isascii():
                return p
        except Exception:
            continue
    raise RuntimeError("找不到可用的 ASCII 临时目录，请设置环境变量 TIANYAN_TMP")


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    for sub in sorted([d for d in folder.iterdir() if d.is_dir()]):
        files += [p for p in sorted(sub.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return files


def _read_image(path: Path) -> Optional[np.ndarray]:
    """中文路径安全的图片读取。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        log.warning("图片读取失败 %s: %s", path, exc)
        return None


def _write_image(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    """中文路径安全的图片写入（JPEG）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:
        log.warning("图片写入失败 %s: %s", path, exc)
        return False


# ---------------------------------------------------------------- 工作线程
class VideoWorker:
    """
    单实例顺序处理一个视频源。

    线程模型与帧率解耦（重要）
    --------------------------
    显示回路与检测回路**解耦**，这是解决「画面卡顿、帧率延迟高」的关键：

      · 显示回路按**源视频帧率**均匀出帧（上限 display_fps），保证画面平滑且
        播放速度与源一致（不会播快/播慢）；
      · 检测回路按 detect_fps 独立触发，未触发检测的帧复用上一次的检测结果与
        规则判定结果做标注。这会使目标框最多滞后一个检测周期（≈1/detect_fps 秒），
        属工程上常见的折中，已在界面与文档中如实标注；
      · **自适应降频**：当单帧推理耗时超过帧预算时，自动增大检测间隔以保住显示
        流畅度；推理变快后再自动收紧。跳过的检测次数逐帧统计，不隐藏。

    实测依据（本机 CPU，640x480 源帧，YOLOv8n）：
      imgsz=640 → 66.2 ms/帧；imgsz=480 → 44.7 ms/帧；标注 0.61ms；JPEG 编码 1.44ms。
      因此 imgsz=480 时可在 15 fps 源上做到**逐帧检测**（44.7+2 < 66.7ms）。
    """

    def __init__(self, source: str, db: Database, detector: BaseDetector, rule_engine: RuleEngine,
                 *, device_id: str = "edge-01", device_name: str = "边缘节点-01",
                 snapshot_dir: Path | str = SNAP_DIR, loop: bool = True,
                 display_fps: float = 30.0, detect_fps: float = 15.0,
                 size: Tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)):
        src = Path(str(source))
        is_dir = src.is_dir()
        if str(source).isdigit():
            raise ValueError("禁止使用摄像头输入（索引 %s）" % source)
        if not src.exists():
            raise FileNotFoundError("视频源不存在：%s" % source)
        if is_dir and not _list_images(src):
            raise FileNotFoundError("图片序列目录为空或格式不支持：%s" % source)

        self.source = str(src)
        self.source_name = src.name if src.name else str(src)
        self.is_sequence = is_dir
        # AI 生成演示素材（real_video/）：竖屏 720x1280、机位未标定，
        # 需要等比适配而非拉伸，且必须关闭未标定的柜门/餐盒判定（详见 config 说明）
        self.ai_demo = is_ai_demo_source(self.source)
        # 演示档门态引擎：仅当该素材在 AI_DEMO_DOOR_PROFILE 中登记时启用。
        # 判定为**演示口径**（单应稳定化 + 逐格基准一致性），与主链路完全隔离，
        # 不进入实验数据集、不计入论文指标（详见 code/demo_door.py 与 docs/09 记录）。
        self.demo_engine = None
        _prof = AI_DEMO_DOOR_PROFILE.get(self.source_name)
        if _prof is not None:
            try:
                from demo_door import DemoDoorEngine
                eng = DemoDoorEngine(self.source_name, _prof,
                                     detector=detector, log=lambda m: log.info("[演示档] %s", m))
                eng.calibrate()
                self.demo_engine = eng
            except Exception as exc:
                log.exception("演示档门态引擎初始化失败（不影响主链路）：%s", exc)
        self.db = db
        self.detector = detector
        self.rules = rule_engine
        self.device_id = device_id
        self.device_name = device_name
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.loop = bool(loop)
        self.display_fps = float(display_fps) if display_fps and display_fps > 0 else 30.0
        self.detect_fps = float(detect_fps) if detect_fps and detect_fps > 0 else 15.0
        self.size = (int(size[0]), int(size[1]))

        # 运行时状态
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._jpeg: Optional[bytes] = None
        self._new_frame = threading.Event()            # 新帧通知，避免推流端忙等轮询
        self._frames: List[Path] = []
        self._cap: Optional[cv2.VideoCapture] = None
        self._tmp_copy: Optional[Path] = None
        self._src_fps = 0.0                            # 源视频帧率
        self._out_fps = 0.0                            # 实际显示帧率
        self._fps_ema = 0.0                            # 实测显示帧率
        self._detect_ema = 0.0                         # 实测检测帧率
        self._infer_ema = 0.0
        self._prev_loop_ts: Optional[float] = None
        self._detect_interval = 1
        self._detect_interval_min = 1
        self._detect_interval_target = 1
        self._infer_n = 0
        self._since_detect = 0
        self._detect_count = 0
        self._detect_skipped = 0
        self._last_dets: List = []
        self._last_events: List = []
        # 显示与检测解耦：显示线程只负责「出帧 + 发布最新帧」，检测线程持续消费。
        # 推理在 CPU 上约 50~90 ms，若在显示循环里同步执行，24 fps 的显示节拍会被拖到
        # 20 fps 以下；而旧的自适应降频会把检测间隔顶到 6~10 帧，画面上的目标框看起来
        # 「卡住不动」。解耦后显示帧率≈源帧率，检测帧率≈1000/推理耗时（目标框持续更新）。
        self._det_thread: Optional[threading.Thread] = None
        self._det_running = False
        self._det_lock = threading.Lock()
        self._det_event = threading.Event()
        self._det_frame: Optional[np.ndarray] = None
        self._det_native: Optional[np.ndarray] = None
        self._det_ts = 0.0
        self._det_index = 0
        self._published = 0                 # 显示线程发布的帧数（检测未覆盖的即为跳检数）
        self._last_det_wall = 0.0           # 上一次检测的墙钟时刻（用于「检测帧率上限」节流）
        self._canvas: Optional[np.ndarray] = None   # letterbox 画布（复用，避免每帧重新分配）
        # 场景状态观测器：柜门开合（亮度对比）+ 餐盒（颜色分割）——首次出帧时按分辨率初始化
        self._door_est: Optional[DoorStateEstimator] = None
        self._pkg_det = PackageDetector()
        self._compartment_polys: Optional[List] = None
        self._last_door_angles: dict = {}
        self._last_pkg_counts: dict = {}
        self._frame_index = 0
        self._event_count = 0
        self._abnormal_count = 0
        self._total_frames = 0
        self._start_ts: Optional[float] = None
        self._session_id: Optional[int] = None
        self._last_hb = 0.0
        self._flash_until = 0.0
        self._last_event: Optional[dict] = None
        self._error: str = ""
        self._status_text = "已停止"
        self._recent: List[dict] = []

    # ------------------------------------------------------------ 资源
    def _open_capture(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            log.warning("直接打开失败（可能是中文路径），尝试复制到 ASCII 临时目录：%s", self.source)
            tmp_dir = _ascii_tmp_dir() / "tianyan_src"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            dst = tmp_dir / ("src" + Path(self.source).suffix.lower())
            shutil.copy2(self.source, dst)
            self._tmp_copy = dst
            cap = cv2.VideoCapture(str(dst))
        if not cap.isOpened():
            cap.release()
            raise IOError("无法打开视频文件：%s" % self.source)
        self._cap = cap
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._src_fps = fps if 1.0 <= fps <= 240.0 else 0.0

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tmp_copy is not None and self._tmp_copy.exists():
            try:
                self._tmp_copy.unlink()
            except Exception:
                pass
            self._tmp_copy = None

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._running:
            return
        if self.is_sequence:
            self._frames = _list_images(Path(self.source))
            if not self._frames:
                raise FileNotFoundError("图片序列为空：%s" % self.source)
        else:
            self._open_capture()

        res = self._resolution_hint()
        # 显示帧率不得高于源帧率，否则会把视频播快；图片序列无固有帧率，用 display_fps
        self._out_fps = min(self.display_fps, self._src_fps) if self._src_fps > 0 else self.display_fps
        # 检测间隔：源帧率 / 检测帧率，向上取整；源帧率未知时按 1 处理（逐帧检测）
        if self._src_fps > 0:
            self._detect_interval_min = max(1, int(round(self._out_fps / self.detect_fps)))
        else:
            self._detect_interval_min = 1
        self._detect_interval = self._detect_interval_min
        self._since_detect = self._detect_interval        # 首帧立即检测

        self._session_id = self.db.insert_session(
            source=self.source_name, backend=self.detector.backend_name,
            resolution="%dx%d" % res, started_at=datetime.now().isoformat(timespec="seconds"))
        self._start_ts = time.time()
        self._running = True
        self._status_text = "运行中"
        self._thread = threading.Thread(target=self._run, name="tianyan-video-worker", daemon=True)
        self._thread.start()
        self._det_running = True
        self._det_thread = threading.Thread(target=self._detect_loop, name="tianyan-detect", daemon=True)
        self._det_thread.start()
        log.info("工作线程启动：source=%s backend=%s 源帧率=%.1f 显示=%.1f 检测≈%.1f（间隔 %d 帧）",
                 self.source_name, self.detector.backend_name, self._src_fps, self._out_fps,
                 self._out_fps / self._detect_interval, self._detect_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        self._det_running = False
        self._det_event.set()                      # 唤醒检测线程，避免其阻塞在 wait 上
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        if self._det_thread is not None and self._det_thread.is_alive():
            self._det_thread.join(timeout=timeout)
        self._det_thread = None
        self._release()
        self._status_text = "已停止"
        self._close_session()
        log.info("工作线程已停止：source=%s", self.source_name)

    def _close_session(self) -> None:
        if self._session_id is None or self._start_ts is None:
            return
        elapsed = time.time() - self._start_ts
        try:
            self.db.update_session(
                self._session_id,
                ended_at=datetime.now().isoformat(timespec="seconds"),
                total_frames=self._total_frames,
                processed_frames=self._frame_index,
                elapsed_s=round(elapsed, 2),
                avg_fps=round(self._fps_ema, 2),
                avg_infer_ms=round(self._infer_ema, 2),
                event_count=self._event_count,
                note="loop=%s backend=%s" % (self.loop, self.detector.backend_name))
        except Exception as exc:
            log.warning("会话记录写入失败：%s", exc)

    def _resolution_hint(self) -> Tuple[int, int]:
        if self.is_sequence and self._frames:
            img = _read_image(self._frames[0])
            if img is not None:
                return img.shape[1], img.shape[0]
            return self.size
        if self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.size[0]), \
                   int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.size[1])
        return self.size

    # ------------------------------------------------------------ 检测线程
    def _detect_loop(self) -> None:
        """
        独立检测线程：持续消费显示线程发布的**最新帧**，执行
        「YOLOv8n 检测 → 柜门/餐盒场景观测 → 规则引擎 → 事件落库与截图」。

        与显示回路彻底解耦后：
          · 显示帧率不再受推理耗时影响（推理慢只会降低检测帧率）；
          · 检测帧率 ≈ 1000 / 单帧推理耗时（本机 imgsz=480 约 50~90 ms → 11~20 fps），
            画面上的目标框基本每帧都在更新，不再出现「框卡住不动」；
          · 被跳过的帧仅表现为「检测更新慢一点」，如实计入 `detect_skipped`。
        事件落库只在本线程、每个被检测的帧上发生一次，避免重复写库与重复截图。
        """
        t0 = time.time()
        n0 = self._detect_count
        try:
            while self._det_running:
                got = self._det_event.wait(0.5)
                if not self._det_running:
                    break
                if not got:
                    continue
                self._det_event.clear()
                with self._det_lock:
                    frame, ts, fidx = self._det_frame, self._det_ts, self._det_index
                    native = self._det_native if self.demo_engine is not None else None
                if frame is None:
                    continue

                # 遵守界面上的「检测帧率上限」：推理本身够快时按上限节流，
                # 避免 24 fps 素材上把 CPU 全吃掉（上限设为 15 即每秒最多 15 次推理）
                min_gap = 1.0 / max(self.detect_fps, 0.1)
                now = time.time()
                if now - self._last_det_wall < min_gap:
                    continue
                self._last_det_wall = now

                dets = self.detector.detect(frame)
                self._last_dets = dets
                events = self.rules.update(
                    self._build_observation(frame, dets, ts),
                    ts, fidx, self.size[0], self.size[1])
                if self.demo_engine is not None and native is not None:
                    # 演示档：柜门类事件由 demo_door 引擎产生；
                    # 事件白名单确保该素材上只出现「与演示相关的告警」
                    # （进入取餐区 / 柜门未关 / 取走未关柜门外卖 / 柜门已关）。
                    events = list(events) + self.demo_engine.update(
                        native, ts, self._native_boxes([tuple(d.xyxy()) for d in dets]))
                    allowed = {"ROI_INTRUSION", "DOOR_CLOSED", "DOOR_LEFT_OPEN",
                               "FLAGGED_DOOR_RETRIEVAL"}
                    events = [e for e in events if e.event_type in allowed]
                self._last_events = events
                self._detect_count += 1

                last_ms = float(self.detector.last_infer_ms)
                self._infer_n += 1
                if self._infer_n <= 1:
                    pass                       # 丢弃冷启动首帧，避免把初始化耗时当作稳态耗时
                elif self._infer_n == 2:
                    self._infer_ema = last_ms
                else:
                    self._infer_ema = 0.85 * self._infer_ema + 0.15 * last_ms

                if events:
                    annotated = self._annotate(frame, dets, events)
                    for ev in events:
                        self._handle_event(ev, annotated, ts)

                el = ts - t0
                if el >= 1.0:                  # 1 秒滑动窗口统计检测帧率，比 EMA 更稳
                    self._detect_ema = (self._detect_count - n0) / el
                    t0 = ts
                    n0 = self._detect_count
        except Exception as exc:               # 线程内异常必须落到状态里，不能静默
            self._error = "%s: %s" % (type(exc).__name__, exc)
            log.exception("检测线程异常终止：%s", exc)
            self._running = False

    # ------------------------------------------------------------ 主循环
    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        把源帧适配到内部画布。

        AI 生成演示素材为 720x1280 竖屏，若直接 resize 到 640x480，画面会被横向
        拉伸 2.4 倍，人物与柜体严重变形。故对这类素材改用「等比缩放 + 居中黑边」
        （letterbox）。其余素材维持原 resize 行为不变——它们的既有指标（含公开数据集
        上的实测数据）都是在该行为下测得的，不应被本次改动影响。
        """
        if not self.ai_demo:
            return cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        tw, th = self.size
        h, w = frame.shape[:2]
        if (w, h) == (tw, th):
            return frame
        scale = min(tw / float(w), th / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        # 画布预分配复用：原先每帧 np.zeros 分配 640x480x3 并整块赋值，
        # 在 24 fps 下是纯粹的无谓开销（每帧多一次 ~0.9 MB 的分配与清零）
        if self._canvas is None or self._canvas.shape[:2] != (th, tw):
            self._canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas = self._canvas
        x0, y0 = (tw - nw) // 2, (th - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _run(self) -> None:
        seq_i = 0
        frame_period = 1.0 / max(self._out_fps, 1.0)
        try:
            while self._running:
                t_loop = time.perf_counter()

                frame = self._next_frame(seq_i)
                if frame is None:
                    if self.loop and not self.is_sequence:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.005)     # 避免重播瞬间形成不睡眠的忙循环
                        self._prev_loop_ts = None   # 该次迭代不出帧，不计入帧率统计
                        continue
                    if self.loop and self.is_sequence:
                        seq_i = 0
                        self._prev_loop_ts = None
                        continue
                    self._status_text = "播放结束"
                    break
                if self.is_sequence:
                    seq_i += 1

                _native = frame if self.demo_engine is not None else None
                frame = self._fit_frame(frame)
                self._frame_index += 1
                ts = time.time()

                # 实测显示帧率 = 相邻两次**成功出帧**的循环起点间隔倒数（含限速休眠）。
                # 必须在确认出帧后再统计：否则重播/跳帧等不睡眠的迭代会把 EMA 抬到数百 fps。
                if self._prev_loop_ts is not None:
                    dt_full = t_loop - self._prev_loop_ts
                    if dt_full > 0:
                        inst = 1.0 / dt_full
                        self._fps_ema = inst if self._fps_ema == 0 else (0.8 * self._fps_ema + 0.2 * inst)
                self._prev_loop_ts = t_loop

                # ---- 检测回路：只把最新帧发布给独立检测线程，显示循环不再被推理阻塞
                # 必须发布一份**拷贝**：letterbox 画布是复用的，而标注（ROI/目标框/HUD）
                # 会在显示线程里就地改写画面；若共享同一块内存，检测线程可能读到被标注过的帧。
                with self._det_lock:
                    self._det_frame = frame.copy()
                    self._det_ts = ts
                    self._det_index = self._frame_index
                    if _native is not None:
                        # 演示档引擎按**原生分辨率**标定（720x1280），必须拿到未
                        # letterbox 的原始帧；人物框在检测线程内映射回原生坐标
                        self._det_native = _native.copy()
                self._published += 1
                self._det_event.set()

                dets = self._last_dets
                events = self._last_events

                # 先标注、再落盘：事件截图保留 ROI 区域、目标框与事件横幅，
                # 便于人工复核与论文取证（此前保存的是未标注原始帧，已修正）。
                annotated = self._annotate(frame, dets, events)

                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                    self._new_frame.set()          # 通知推流端有新帧，避免忙等

                self._heartbeat(ts)

                spare = frame_period - (time.perf_counter() - t_loop)
                if spare > 0:
                    time.sleep(spare)
        except Exception as exc:                     # 线程内异常必须落到状态里，不能静默
            self._error = "%s: %s" % (type(exc).__name__, exc)
            self._status_text = "异常"
            log.exception("工作线程异常终止：%s", exc)
        finally:
            self._running = False
            self._det_running = False
            self._det_event.set()                    # 一并结束检测线程
            self._new_frame.set()                    # 唤醒推流端，避免其阻塞在 wait 上
            self._release()
            self._close_session()

    def _next_frame(self, seq_i: int) -> Optional[np.ndarray]:
        if self.is_sequence:
            if seq_i >= len(self._frames):
                return None
            return _read_image(self._frames[seq_i])
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    # ------------------------------------------------------------ 事件落库
    def _handle_event(self, ev: RuleEvent, frame: np.ndarray, ts: float) -> None:
        stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        fname = "%s_%s_t%s_f%d.jpg" % (stamp, ev.event_type, ev.track_id if ev.track_id is not None else "scene",
                                       self._frame_index)
        snap_path = self.snapshot_dir / fname
        saved = _write_image(snap_path, frame)
        rel = os.path.relpath(str(snap_path), str(Path(__file__).resolve().parent.parent)).replace("\\", "/")

        try:
            self.db.insert_event(
                ts=datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
                event_type=ev.event_type, event_name=ev.event_name, severity=ev.severity,
                detail=ev.detail, snapshot=rel if saved else None, confidence=ev.confidence,
                track_id=ev.track_id, frame_index=ev.frame_index, source=self.source_name,
                duration_s=ev.extras.get("duration_s") or ev.extras.get("dwell_s") or ev.extras.get("presence_s"),
                persons=ev.extras.get("persons"), is_abnormal=ev.is_abnormal)
        except Exception as exc:
            log.error("事件入库失败：%s", exc)

        self._event_count += 1
        if ev.is_abnormal:
            self._abnormal_count += 1
            self._flash_until = ts + 3.0
        rec = {"ts": datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
               "event_type": ev.event_type, "event_name": ev.event_name,
               "severity": ev.severity, "detail": ev.detail,
               "confidence": round(float(ev.confidence), 3),
               "snapshot": rel if saved else None, "track_id": ev.track_id}
        with self._lock:
            self._last_event = rec
            self._recent.insert(0, rec)
            del self._recent[20:]
        log.info("[事件] %s | %s | conf=%.3f | track=%s", ev.event_name, ev.detail,
                 ev.confidence, ev.track_id)

    def _native_boxes(self, boxes):
        """把 letterbox 画布坐标的人物框映射回素材原生坐标（演示档引擎使用）。

        仅对启用演示档引擎的素材调用；映射参数由「原生尺寸 vs 画布尺寸」推导，
        与 `_fit_frame` 的 letterbox 逻辑保持同一套公式。
        """
        fs = self.demo_engine.frame_size() if self.demo_engine is not None else None
        if not fs:
            return boxes
        nw_, nh_ = fs
        tw, th = self.size
        scale = min(tw / float(nw_), th / float(nh_))
        ox = (tw - int(round(nw_ * scale))) // 2
        oy = (th - int(round(nh_ * scale))) // 2
        out = []
        for x1, y1, x2, y2 in boxes:
            out.append(((x1 - ox) / scale, (y1 - oy) / scale,
                        (x2 - ox) / scale, (y2 - oy) / scale))
        return out

    # ------------------------------------------------------------ 场景观测
    def _build_observation(self, frame: np.ndarray, dets, ts: float) -> SceneObservation:
        """
        把当前帧转换为规则引擎所需的场景观测：
            人物检测框（由 detector 给出）
            各柜格门状态（DoorStateEstimator，亮度对比）
            各柜格内可见餐盒数、每人手持餐盒数（PackageDetector，颜色分割 + 连通域）

        ts 为源视频时基的时刻，用于取「近期仍在场」的轨迹框做遮挡补偿。
        """
        if self.ai_demo:
            # 演示素材（AI 生成，非实拍）未做柜格标定：柜格多边形为空，
            # 若仍跑柜门亮度对比与餐盒颜色分割，只会在真实柜体画面上得到
            # 无依据的「柜门未关 / 取走未关柜门外卖 / 一人取多份」报警。
            # 故此处只返回人物检测框——该素材上只演示「进入取餐区」等人物类事件。
            return SceneObservation(dets=list(dets))
        if self._compartment_polys is None:
            _, self._compartment_polys = self.rules.roi_polygons(self.size[0], self.size[1])
            self._door_est = DoorStateEstimator(self._compartment_polys)
        # 柜门状态估计的遮挡补偿：除本帧检测框外，**一并传入近期仍在场的轨迹框**。
        # 轨迹在 reid_ttl_s 内即使本帧漏检仍然存活，因此「人站在柜前但被漏检」时，
        # 其深色衣物不会被当成柜内暗区而误判为「柜门开启」
        # （曾使长驻足片段凭空触发 DOOR_LEFT_OPEN，并连带污染其他判定）。
        # 注意只用 `recent_track_boxes`：已离场者的残留框继续遮挡会让柜门状态长期「未知」，
        # 反而使 DOOR_LEFT_OPEN 的 10 s 累计窗口凑不满（实测曾因此新增 2 处漏检）。
        self._last_door_angles = self._door_est.estimate(
            frame, [d.xyxy() for d in dets] + self.rules.recent_track_boxes(ts))
        blobs = self._pkg_det.detect(frame)
        person_boxes = [d.xyxy() for d in dets]
        pkg_in_comp = count_packages_in_compartments(blobs, self._compartment_polys, person_boxes)
        self._last_pkg_counts = pkg_in_comp
        carries = [PackageCarry(bbox=box, count=n) for box, n in
                   count_carried_packages(blobs, person_boxes, self._compartment_polys)]
        return SceneObservation(dets=list(dets), door_angles=self._last_door_angles,
                                packages_in_compartment=pkg_in_comp, carries=carries)

    # ------------------------------------------------------------ 画面标注
    def _annotate(self, frame: np.ndarray, dets, events) -> np.ndarray:
        h, w = frame.shape[:2]
        p1, comps = self.rules.roi_polygons(w, h)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [p1], (0, 90, 190))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        cv2.polylines(frame, [p1], True, C_ROI_PICKUP, 2)
        # 画面 HUD 使用 ASCII，避免 cv2.putText 无法渲染中文导致乱码
        cv2.putText(frame, "PICKUP ZONE", (p1[:, 0].min(), max(14, p1[:, 1].min() - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ROI_PICKUP, 2)

        # 柜格与柜门状态：已标记「未关」的柜格用红色加粗并标注柜号
        flagged = {f["compartment"] for f in self.rules.snapshot().get("flagged_doors", [])}
        for k, poly in enumerate(comps):
            is_flag = k in flagged
            _da = self._last_door_angles.get(k)
            is_open = (_da is not None) and (float(_da) >= self.rules.cfg.door_angle_open_th)
            col = C_BOX_ALERT if is_flag else (C_DOOR_OPEN if is_open else C_DOOR_CLOSED)
            cv2.polylines(frame, [poly], True, col, 2 if is_flag else 1)
            anchor = (int(poly[:, 0].min()), int(poly[:, 1].min()) - 4)
            label = "C%d%s" % (k, " OPEN" if (is_flag or is_open) else "")
            cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # 演示档（AI 素材）：逐格门态叠加（灰=闭合 / 橙=开 / 红=已标记未关）
        if self.demo_engine is not None:
            from demo_door import _to_px as _demo_to_px
            for d in self.demo_engine.snapshot()["doors"]:
                cell = self.demo_engine.cells[d["idx"]]
                if cell.get("skip"):
                    continue
                _poly = _demo_to_px(cell["poly"], w, h)
                if d["flagged"]:
                    _col, _th = C_BOX_ALERT, 2
                elif d["state"] == "OPEN":
                    _col, _th = C_DOOR_OPEN, 2
                else:
                    _col, _th = (140, 140, 140), 1
                cv2.polylines(frame, [_poly], True, _col, _th)
                if d["flagged"] or d["state"] == "OPEN":
                    _cx, _cy = _poly.mean(axis=0).astype(int)
                    cv2.putText(frame, "%s OPEN" % d["door_id"], (_cx - 34, _cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _col, 1, cv2.LINE_AA)

        alerting = bool(events)
        for d in dets:
            in_zone = cv2.pointPolygonTest(p1, (float(d.bottom_center[0]), float(d.bottom_center[1])), False) >= 0
            color = C_BOX_ALERT if (alerting and in_zone) else C_BOX
            x1, y1, x2, y2 = d.xyxy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "person %.2f" % d.conf, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        info = self.rules.snapshot()
        # 画面内文本必须为 ASCII：cv2.putText 不支持中文字形，中文会渲染成 "???"
        flagged = ["C%d" % f["compartment"] for f in info.get("flagged_doors", [])]
        if self.demo_engine is not None:
            flagged += list(self.demo_engine.snapshot()["flagged_doors"])
        lines = [
            "tianyan v1.0 | backend=%s" % self.detector.backend_name,
            "frame=%d  disp=%.1ffps  det=%.1ffps  infer=%.0fms"
            % (self._frame_index, self._fps_ema, self._detect_ema, self._infer_ema),
            "src=%.1ffps  detect_every=%d  skipped=%d"
            % (self._src_fps, self._detect_interval, self._detect_skipped),
            "persons=%d  events=%d  abnormal=%d" % (len(dets), self._event_count, self._abnormal_count),
            "doors_flagged_open=%s" % (",".join(flagged) if flagged else "none"),
            "normal_retrievals=%d  suppressed=%d" % (info["normal_retrievals"], info["suppressed_by_cooldown"]),
        ]
        # HUD 加深色半透明底衬：浅色素材上纯浅色文字不可读，深色素材上纯深色文字同样不可读
        hud_h = 14 + len(lines) * 18
        hud = frame.copy()
        cv2.rectangle(hud, (6, 6), (338, hud_h), (18, 22, 28), -1)
        cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (14, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_HUD, 1,
                        cv2.LINE_AA)

        if self.ai_demo:
            # 画面内固定声明（英文，cv2.putText 无法渲染中文）：该素材为 AI 生成，非实拍。
            # 落盘的事件截图同样带此声明，避免截图脱离上下文后被误当作实拍证据。
            _txt = AI_DEMO_FRAME_BANNER
            _sc, _thk = 0.44, 1
            (_tw, _th), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, _sc, _thk)
            _x0 = max(8, (w - _tw) // 2)
            _y0 = hud_h + 8 + _th
            _band = frame.copy()
            cv2.rectangle(_band, (_x0 - 8, _y0 - _th - 6), (_x0 + _tw + 8, _y0 + 6), (18, 22, 28), -1)
            cv2.addWeighted(_band, 0.62, frame, 0.38, 0, frame)
            cv2.putText(frame, _txt, (_x0, _y0), cv2.FONT_HERSHEY_SIMPLEX, _sc, C_DEMO_NOTE, _thk,
                        cv2.LINE_AA)

        if time.time() < self._flash_until:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_HUD_ALERT, 6)
            cv2.putText(frame, "ALERT", (w // 2 - 45, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_HUD_ALERT, 3)
        if events:
            # 画面内文本使用英文事件枚举（cv2.putText 不支持中文字形）
            names = " | ".join([e.event_type for e in events])
            band = frame.copy()
            cv2.rectangle(band, (6, h - 28), (w - 6, h - 4), (18, 22, 28), -1)
            cv2.addWeighted(band, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, names, (14, h - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_HUD_ALERT, 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------ 心跳
    def _heartbeat(self, ts: float) -> None:
        if ts - self._last_hb < 2.0:
            return
        self._last_hb = ts
        try:
            self.db.upsert_device_status(
                device_id=self.device_id, device_name=self.device_name, source=self.source_name,
                backend=self.detector.backend_name, status=self._status_text,
                fps=self._fps_ema, infer_ms=self._infer_ema, frame_index=self._frame_index,
                event_count=self._event_count)
        except Exception as exc:
            log.warning("设备状态上报失败：%s", exc)

        # 同步刷新会话进度：否则进程被强制结束（崩溃/直接关窗）时会话记录会停留在
        # 「0 帧 / 运行中」，导致阶段3 的实验复现核对失去依据。
        if self._session_id is not None and self._start_ts is not None:
            try:
                self.db.update_session(
                    self._session_id,
                    processed_frames=self._frame_index,
                    elapsed_s=round(ts - self._start_ts, 2),
                    avg_fps=round(self._fps_ema, 2),
                    avg_infer_ms=round(self._infer_ema, 2),
                    event_count=self._event_count)
            except Exception as exc:
                log.warning("会话进度上报失败：%s", exc)

    # ------------------------------------------------------------ 对外接口
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def wait_new_frame(self, timeout: float = 1.0) -> bool:
        """阻塞等待新帧（由工作线程置位），用于 MJPEG 推流端零忙等出帧。"""
        got = self._new_frame.wait(timeout)
        if got:
            self._new_frame.clear()
        return got

    def status(self) -> dict:
        info = self.rules.snapshot()
        _demo_snap = self.demo_engine.snapshot() if self.demo_engine is not None else None
        pos = 0.0
        if self._cap is not None and self._total_frames > 0:
            pos = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        # 检测帧率：滑动窗口尚未成熟时，退化为「累计检测次数 / 已运行时长」，避免上报 0 造成误判
        detect_fps = self._detect_ema
        if detect_fps <= 0 and self._detect_count > 0 and self._start_ts:
            el = time.time() - self._start_ts
            if el > 0.3:
                detect_fps = self._detect_count / el
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "source": self.source_name,
            "source_path": self.source,
            "source_type": "图片序列" if self.is_sequence else "视频文件",
            # True 表示当前素材是 real_video/ 下的 AI 生成演示视频（非实拍），
            # 界面据此显示声明横幅；该素材不参与论文的任何实验数据统计
            "ai_demo": self.ai_demo,
            "running": self._running,
            "status": self._status_text,
            "backend": self.detector.backend_name,
            "backend_name": self.detector.display_name,
            "rule_fallback": getattr(self.detector, "fallback_errors", []),
            "frame_index": self._frame_index,
            "total_frames": self._total_frames,
            "progress": round(pos / self._total_frames, 4) if self._total_frames else 0.0,
            "fps": round(self._fps_ema, 2),                      # 实测显示帧率
            "detect_fps": round(detect_fps, 2),            # 实测检测帧率
            "infer_ms": round(self._infer_ema, 2),
            "src_fps": round(self._src_fps, 2),
            "display_fps": round(self._out_fps, 2),
            # 显示与检测已解耦：检测线程持续消费最新帧，故「检测间隔」恒为 1 帧发布；
            # 未参与检测的帧数 = 发布帧数 − 实际检测帧数（含按上限节流与推理来不及）
            "detect_interval": 1,
            "detect_interval_target": 1,
            "detect_count": self._detect_count,
            "detect_skipped": max(0, self._published - self._detect_count),
            "resolution": "%dx%d" % self.size,
            "uptime_s": round(time.time() - self._start_ts, 1) if self._start_ts else 0.0,
            "event_count": self._event_count,
            "abnormal_count": self._abnormal_count,
            "active_tracks": info["active_tracks"],
            "normal_retrievals": info["normal_retrievals"],
            "suppressed_by_cooldown": info["suppressed_by_cooldown"],
            "events_by_type": info["events_by_type"],
            # 终端同步显示的「未关柜门柜号」（门关闭后自动从该列表移除）。
            # 演示档（AI 素材）的标记柜号使用位置编号（如 R5C2），与规则链路同字段展示。
            "flagged_doors": info.get("flagged_doors", []),
            "flagged_door_labels": (["%s 格" % d for d in _demo_snap["flagged_doors"]]
                                    if _demo_snap is not None
                                    else info.get("flagged_door_labels", [])),
            "demo_doors": ([{"door_id": d["door_id"], "state": d["state"],
                             "flagged": d["flagged"], "ncc": d["ncc"]}
                            for d in _demo_snap["doors"]] if _demo_snap is not None else []),
            "demo_calibrated": (None if _demo_snap is None else _demo_snap["calibrated"]),
            "demo_calib_error": (None if _demo_snap is None else _demo_snap["calib_error"]),
            "door_states": info.get("door_states", []),
            "door_dark_ratios": [round(v, 3) for v in (self._door_est.last_dark_ratios
                                                       if self._door_est else [])],
            "packages_in_compartment": dict(self._last_pkg_counts),
            "loop": self.loop,
            "error": self._error,
            "detector": self.detector.describe(),
            "rules": info["rule_config"],
        }

    def recent_events(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return list(self._recent[:limit])
