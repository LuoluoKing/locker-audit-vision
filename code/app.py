# -*- coding: utf-8 -*-
"""
app.py — 天眼哨兵 · Flask Web 服务与云端管理后台

启动示例
--------
    python app.py --source ../data/simulated/normal_001.mp4 --backend auto --port 5000
    python app.py --backend hog --source ../data/videos/UCF-Crime/

页面
----
    /            实时监控台（MJPEG 视频流 + 状态面板 + 事件流）
    /admin       云端管理后台（设备状态 / 事件统计 / 历史查询 / CSV 导出）

接口
----
    GET  /video_feed           MJPEG 视频流
    GET  /api/frame            单帧 JPEG
    GET  /api/status           运行状态（含检测后端、规则参数、实时指标）
    GET  /api/events           事件查询（类型/时间/关键词/分页）
    GET  /api/events/<id>      单条事件详情
    GET  /api/stats            统计聚合
    GET  /api/devices          设备状态列表
    GET  /api/sources          可用视频素材列表
    POST /api/control          启停/重启/切换源与后端
    GET  /api/export.csv       导出事件 CSV
"""
from __future__ import annotations

import argparse
import atexit
import io
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

import config as C
from config import Settings, resolve_source, setup_logging
from database import Database
from detector import build_detector
from rules import RuleConfig, RuleEngine
from video_worker import VideoWorker

log = logging.getLogger("tianyan.app")


# ================================================================= 运行态
class Runtime:
    """进程内单例运行态：数据库 / 检测器 / 规则引擎 / 工作线程。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.db = Database(settings.db_path)
        self.snapshot_dir = Path(settings.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.detector = None            # type: ignore
        self.rules: Optional[RuleEngine] = None
        self.worker: Optional[VideoWorker] = None
        self.started_at = time.time()
        self.last_error: str = ""
        self._placeholder: Optional[bytes] = None
        self._viewers = 0

    # ---------------------------------------------------------- 占位帧
    def placeholder_jpeg(self) -> bytes:
        if self._placeholder is None:
            img = np.full((C.FRAME_HEIGHT, C.FRAME_WIDTH, 3), 32, dtype=np.uint8)
            cv2.putText(img, "NO SIGNAL", (176, 232), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            cv2.putText(img, "worker stopped - press START", (146, 268),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            self._placeholder = buf.tobytes() if ok else b""
        return self._placeholder

    # ---------------------------------------------------------- 生命周期
    def start(self, source: Optional[str] = None, backend: Optional[str] = None,
              max_fps: Optional[float] = None, loop: Optional[bool] = None,
              imgsz: Optional[int] = None, display_fps: Optional[float] = None) -> dict:
        with self.lock:
            self.stop()
            s = self.settings
            try:
                src = resolve_source(source if source is not None else s.source)
            except Exception as exc:
                self.last_error = str(exc)
                raise

            if imgsz:
                s.imgsz = int(imgsz)
            self.detector = build_detector(
                backend or s.backend, model_path=s.model_path, conf_threshold=s.conf_threshold,
                iou_threshold=s.iou_threshold, imgsz=s.imgsz, device=s.device,
                hog_resize=s.hog_resize, torch_threads=s.torch_threads)

            # AI 生成演示素材按「演示档案」启动：取餐区换用针对该机位粗标定的多边形，
            # 且**不提供柜格多边形**——柜门/餐盒类判定在该素材上未标定，必须关闭，
            # 否则会在真实柜体画面上给出无依据的报警（详见 config 演示素材档案说明）。
            if C.is_ai_demo_source(src):
                self.rules = RuleEngine(RuleConfig(
                    conf_threshold=s.conf_threshold,
                    roi_pickup=[list(p) for p in C.DEMO_ROI_PICKUP],
                    compartments=[]))
                log.warning("素材属于 AI 生成演示视频（非实拍）：启用演示档案，"
                            "仅保留取餐区相关规则，柜门/餐盒类判定已关闭。source=%s", src)
            else:
                self.rules = RuleEngine(RuleConfig(conf_threshold=s.conf_threshold))
            self.worker = VideoWorker(
                source=src, db=self.db, detector=self.detector, rule_engine=self.rules,
                device_id=s.device_id, device_name=s.device_name,
                snapshot_dir=self.snapshot_dir,
                loop=(s.loop_playback if loop is None else bool(loop)),
                display_fps=(s.display_fps if display_fps is None else float(display_fps)),
                detect_fps=(s.detect_fps if max_fps is None else float(max_fps)))
            self.worker.start()
            s.source = src
            self.last_error = ""
            return self.worker.status()

    def stop(self) -> None:
        with self.lock:
            if self.worker is not None:
                try:
                    self.worker.stop()
                except Exception as exc:
                    log.warning("停止工作线程出错：%s", exc)
                self.worker = None

    def status(self) -> dict:
        with self.lock:
            base = {
                "app": C.APP_NAME, "version": C.APP_VERSION, "subtitle": C.APP_SUBTITLE,
                "server_uptime_s": round(time.time() - self.started_at, 1),
                "viewers": self._viewers,
                "last_error": self.last_error,
                "db_path": self.settings.db_path,
                "device_id": self.settings.device_id,
                "device_name": self.settings.device_name,
            }
            if self.worker is None:
                base.update({"running": False, "status": "已停止" if not self.last_error else "错误",
                             "backend": (self.detector.backend_name if self.detector else "-"),
                             "backend_name": (self.detector.display_name if self.detector else "未初始化"),
                             "event_count": 0, "abnormal_count": 0, "active_tracks": 0,
                             "normal_retrievals": 0, "suppressed_by_cooldown": 0,
                             "events_by_type": {}, "rules": {}, "source": self.settings.source})
                return base
            base.update(self.worker.status())
            return base

    def rebuild_detector(self, backend: str, imgsz: Optional[int] = None) -> dict:
        """仅切换检测后端 / 推理尺寸（用于阶段3 的 YOLOv8n / HOG / 输入尺寸对比实验）。"""
        s = self.settings
        if imgsz:
            s.imgsz = int(imgsz)
        self.detector = build_detector(backend, model_path=s.model_path,
                                       conf_threshold=s.conf_threshold, iou_threshold=s.iou_threshold,
                                       imgsz=s.imgsz, device=s.device, hog_resize=s.hog_resize,
                                       torch_threads=s.torch_threads)
        return self.detector.describe()


# ================================================================= 工具
def _is_output_dir(name: str) -> bool:
    """
    判断目录是否为系统输出目录（截图/缓存/临时），不应作为可播放素材。

    采用模式化规则而非硬编码名单：目录名含 snapshot/snap（截图输出）、
    或以 _ 开头（临时目录）、或属常见缓存目录，一律排除。
    """
    low = name.lower()
    if low in {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}:
        return True
    if name.startswith("_"):
        return True
    return ("snapshot" in low) or ("snap" == low)


def list_sources() -> list:
    """
    扫描 data/simulated、data/videos、data/selftest 与 real_video 下可用素材。
    截图输出目录与临时目录会被排除，避免被误判为「图片序列」。
    `real_video/` 为 AI 生成演示视频（非实拍），条目带 ai_demo 标记，供界面显示声明。
    """
    exts = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v")
    out = []
    scan = (("模拟视频", C.SIM_DIR), ("数据集视频", C.VIDEO_DIR),
            ("链路自检素材", C.BASE_DIR / "data" / "selftest"),
            (C.AI_DEMO_GROUP, C.REAL_DEMO_DIR))
    for label, d in scan:
        if not d.exists():
            continue
        is_demo_dir = (d == C.REAL_DEMO_DIR)
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                if any(_is_output_dir(part) for part in f.relative_to(d).parts[:-1]):
                    continue
                try:
                    rel = f.relative_to(C.BASE_DIR)
                except ValueError:
                    rel = f
                out.append({"group": label, "name": f.name, "path": str(rel).replace("\\", "/"),
                            "size_mb": round(f.stat().st_size / 1048576, 2),
                            "ai_demo": is_demo_dir})
        # 图片序列目录（排除输出目录）
        for sub in sorted([p for p in d.iterdir() if p.is_dir()]):
            if _is_output_dir(sub.name):
                continue
            imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
            if imgs:
                out.append({"group": label + "(图片序列)", "name": sub.name,
                            "path": str(sub.relative_to(C.BASE_DIR)).replace("\\", "/"),
                            "size_mb": round(sum(i.stat().st_size for i in imgs) / 1048576, 2),
                            "ai_demo": is_demo_dir})
    return out


def _abs_source(p: str) -> str:
    """把相对项目根目录的路径还原为绝对路径。"""
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    cand = C.BASE_DIR / pp
    return str(cand if cand.exists() else pp)


# ================================================================= 应用工厂
def create_app(settings: Settings) -> Flask:
    app = Flask(__name__, template_folder=str(C.CODE_DIR / "templates"),
                static_folder=str(C.CODE_DIR / "static"))
    app.config["JSON_AS_ASCII"] = False
    rt = Runtime(settings)
    app.extensions["tianyan"] = rt

    # ---------------------------------------------------------- 页面
    @app.route("/")
    def page_index():
        return render_template("index.html", app_name=C.APP_NAME, version=C.APP_VERSION,
                               subtitle=C.APP_SUBTITLE, settings=settings.to_dict(),
                               event_catalog=C.EVENT_CATALOG, ai_demo_notice=C.AI_DEMO_NOTICE)

    @app.route("/admin")
    def page_admin():
        return render_template("admin.html", app_name=C.APP_NAME, version=C.APP_VERSION,
                               subtitle=C.APP_SUBTITLE,
                               db_path=settings.db_path, device_name=settings.device_name)

    # ---------------------------------------------------------- 视频流
    def _gen_frames():
        """
        MJPEG 推流：由工作线程的新帧事件驱动，零忙等。
        仅在新帧到达时写出，避免重复推送同一帧造成浏览器端缓冲堆积与额外延迟。
        """
        rt._viewers += 1
        boundary = b"--frame\r\n"
        try:
            last = None
            while True:
                worker = rt.worker
                if worker is None:
                    jpg = rt.placeholder_jpeg()
                    if jpg is not last:
                        yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.25)
                    last = None
                    continue
                worker.wait_new_frame(timeout=1.0)
                jpg = worker.get_jpeg()
                if jpg is None or jpg is last:
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                last = jpg
        except GeneratorExit:
            pass
        finally:
            rt._viewers = max(0, rt._viewers - 1)

    @app.route("/video_feed")
    def video_feed():
        return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/frame")
    def api_frame():
        jpg = rt.worker.get_jpeg() if rt.worker is not None else None
        jpg = jpg or rt.placeholder_jpeg()
        return Response(jpg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store, max-age=0"})

    # ---------------------------------------------------------- 状态与统计
    @app.route("/api/status")
    def api_status():
        return jsonify(rt.status())

    @app.route("/api/stats")
    def api_stats():
        start = request.args.get("start") or None
        end = request.args.get("end") or None
        return jsonify({
            "summary": rt.db.stats_summary(start=start, end=end),
            "by_type": rt.db.stats_by_type(start=start, end=end),
            "by_day": rt.db.stats_by_day(int(request.args.get("days", 14))),
            "catalog": {k: {"name": v[0], "severity": v[1]} for k, v in C.EVENT_CATALOG.items()},
            "live": rt.status(),
        })

    @app.route("/api/devices")
    def api_devices():
        return jsonify({"devices": rt.db.list_devices()})

    @app.route("/api/sessions")
    def api_sessions():
        return jsonify({"sessions": rt.db.recent_sessions(int(request.args.get("limit", 10)))})

    @app.route("/api/sources")
    def api_sources():
        import os
        # current 同时给出完整路径与文件名：前端下拉用文件名匹配选中项，
        # 否则绝对路径与选项 value/name 都不相等，浏览器回退到第一项造成「选中项与
        # 实际播放素材不一致」的假象。
        return jsonify({"sources": list_sources(), "current": rt.settings.source,
                        "current_name": os.path.basename(rt.settings.source or ""),
                        "current_ai_demo": C.is_ai_demo_source(rt.settings.source)})

    # ---------------------------------------------------------- 事件
    @app.route("/api/events")
    def api_events():
        args = request.args
        page = max(1, int(args.get("page", 1)))
        size = min(200, max(1, int(args.get("size", 20))))
        filters = dict(
            event_type=(args.get("type") or None),
            start=(args.get("start") or None),
            end=(args.get("end") or None),
            keyword=(args.get("q") or None),
            only_abnormal=(args.get("abnormal", "0") == "1"),
        )
        total = rt.db.count_events(**filters)
        items = rt.db.query_events(limit=size, offset=(page - 1) * size, **filters)
        return jsonify({"total": total, "page": page, "size": size, "items": items})

    @app.route("/api/events/<int:eid>")
    def api_event_detail(eid: int):
        row = rt.db.get_event(eid)
        if not row:
            abort(404)
        return jsonify(row)

    @app.route("/api/export.csv")
    def api_export():
        csv_text = rt.db.export_csv(
            event_type=(request.args.get("type") or None),
            start=(request.args.get("start") or None),
            end=(request.args.get("end") or None),
            keyword=(request.args.get("q") or None),
            only_abnormal=(request.args.get("abnormal", "0") == "1"))
        fname = "tianyan_events_%s.csv" % time.strftime("%Y%m%d_%H%M%S")
        return Response(csv_text, mimetype="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=%s" % fname})

    # ---------------------------------------------------------- 截图
    @app.route("/snapshots/<path:name>")
    def snapshots(name: str):
        safe = Path(name).name                      # 防目录穿越
        target = (rt.snapshot_dir / safe).resolve()
        if not str(target).startswith(str(rt.snapshot_dir.resolve())) or not target.exists():
            abort(404)
        return send_file(str(target), mimetype="image/jpeg")

    # ---------------------------------------------------------- 控制
    @app.route("/api/control", methods=["POST"])
    def api_control():
        payload = request.get_json(silent=True) or {}
        action = (payload.get("action") or "start").lower()
        try:
            if action == "start":
                src = payload.get("source")
                st = rt.start(source=_abs_source(src) if src else None,
                              backend=payload.get("backend") or None,
                              max_fps=payload.get("detect_fps") or payload.get("max_fps") or None,
                              display_fps=payload.get("display_fps") or None,
                              imgsz=payload.get("imgsz") or None,
                              loop=payload.get("loop"))
                return jsonify({"ok": True, "action": action, "status": st})
            if action == "stop":
                rt.stop()
                return jsonify({"ok": True, "action": action, "status": rt.status()})
            if action == "restart":
                src = payload.get("source") or rt.settings.source
                st = rt.start(source=_abs_source(src),
                              backend=payload.get("backend") or None,
                              max_fps=payload.get("detect_fps") or payload.get("max_fps") or None,
                              display_fps=payload.get("display_fps") or None,
                              imgsz=payload.get("imgsz") or None,
                              loop=payload.get("loop"))
                return jsonify({"ok": True, "action": action, "status": st})
            if action == "backend":
                desc = rt.rebuild_detector(payload.get("backend") or "auto",
                                           imgsz=payload.get("imgsz") or None)
                return jsonify({"ok": True, "action": action, "detector": desc,
                                "note": "后端已重建，重启视频源后生效"})
            return jsonify({"ok": False, "error": "未知操作：%s" % action}), 400
        except Exception as exc:
            rt.last_error = "%s: %s" % (type(exc).__name__, exc)
            log.exception("控制接口失败：%s", exc)
            return jsonify({"ok": False, "error": rt.last_error}), 500

    @app.route("/api/health")
    def api_health():
        return jsonify({"ok": True, "ts": time.time(), "version": C.APP_VERSION})

    @app.errorhandler(404)
    def _404(_e):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "not found"}), 404
        return render_template("404.html", app_name=C.APP_NAME), 404

    return app, rt


# ================================================================= 端口守卫
def has_live_listener(host: str, port: int, timeout: float = 0.8) -> bool:
    """
    检测 host:port 上是否已有进程在监听（用 TCP 连接探测）。

    为什么必须显式检测：Windows 的 SO_REUSEADDR 语义与 Unix 不同，
    它**允许**第二个进程绑定同一端口而**不报错**，但实际只会有一个 socket 真正
    接收连接。结果是「新实例启动日志显示成功，页面却打不开或画面不刷新」。
    因此这里在启动前主动探测并拒绝，避免出现两个实例静默共存。
    """
    probe_host = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    try:
        with socket.create_connection((probe_host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


# ================================================================= 入口
def parse_args() -> Settings:
    p = argparse.ArgumentParser(description="天眼哨兵校园外卖无感安防系统 · Web 服务")
    p.add_argument("--source", default="", help="视频文件或图片序列目录（留空自动挑选 data 下素材）")
    p.add_argument("--backend", default="auto", choices=["auto", "yolov8n", "hog"], help="检测后端")
    p.add_argument("--model", default=str(C.MODEL_DIR / "yolov8n.pt"), help="YOLOv8 权重路径")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--display-fps", type=float, default=30.0,
                   help="显示/推流帧率上限（实际不会超过源视频帧率）")
    p.add_argument("--detect-fps", "--max-fps", dest="detect_fps", type=float, default=15.0,
                   help="检测回路帧率上限；推理跟不上时自动降频保显示流畅")
    p.add_argument("--imgsz", type=int, default=480,
                   help="推理输入尺寸。实测 640→66ms、480→45ms、416→34ms（本机 CPU）")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="CPU 推理线程数，0 = 自动取物理核心数（封顶 8）")
    p.add_argument("--no-loop", action="store_true", help="视频播放结束不重播")
    p.add_argument("--no-autostart", action="store_true", help="启动后不自动开始处理视频")
    p.add_argument("--db", default=str(C.DB_PATH), help="SQLite 数据库路径")
    p.add_argument("--snapshot-dir", default=str(C.SNAP_DIR), help="事件截图保存目录")
    p.add_argument("--device-id", default="edge-01")
    p.add_argument("--device-name", default="边缘节点-01（本机 CPU）")
    p.add_argument("--server", default="flask", choices=["flask", "waitress"], help="WSGI 服务器")
    a = p.parse_args()
    return Settings(host=a.host, port=a.port, source=a.source, loop_playback=not a.no_loop,
                    display_fps=a.display_fps, detect_fps=a.detect_fps, autostart=not a.no_autostart,
                    backend=a.backend, imgsz=a.imgsz, torch_threads=a.torch_threads,
                    model_path=a.model, db_path=a.db, device_id=a.device_id,
                    snapshot_dir=a.snapshot_dir, device_name=a.device_name, server=a.server)


def main() -> None:
    settings = parse_args()
    setup_logging()

    # ---- 端口守卫：拒绝与已有实例静默共存（Windows SO_REUSEADDR 陷阱）
    if has_live_listener(settings.host, settings.port):
        msg = (
            "\n" + "!" * 74 + "\n"
            "端口 %d 已被占用：检测到 %s:%d 上已有服务在监听。\n"
            "\n"
            "这通常说明已有一个「天眼哨兵」实例正在运行。Windows 允许两个进程绑定同一端口\n"
            "而不报错，但只有一个会真正响应请求，于是表现为「启动日志显示成功，页面却打不开\n"
            "或画面一直不刷新」。为避免这种情况，本程序拒绝重复启动。\n"
            "\n"
            "请任选一种处理方式：\n"
            "  1) 关闭已打开的那个命令行窗口（或在该窗口按 Ctrl+C），再重新启动本程序；\n"
            "  2) 改用其他端口启动，例如： python code/app.py --port 5050\n"
            % (settings.port, settings.host, settings.port)
        )
        log.error(msg)
        print(msg)
        raise SystemExit(2)

    log.info("=" * 78)
    log.info("%s %s 启动", C.APP_NAME, C.APP_VERSION)
    cfg_file = settings.dump()
    log.info("运行配置已存档：%s", cfg_file)

    app, rt = create_app(settings)

    if settings.autostart:
        try:
            st = rt.start()
            log.info("视频源已自动启动：%s (%s)", st["source"], st["backend"])
        except Exception as exc:
            rt.last_error = "%s: %s" % (type(exc).__name__, exc)
            log.error("自动启动失败：%s（可在页面手动选择素材后点击启动）", exc)

    atexit.register(rt.stop)

    log.info("监控台  http://%s:%d/", settings.host, settings.port)
    log.info("管理后台 http://%s:%d/admin", settings.host, settings.port)

    if settings.server == "waitress":
        try:
            from waitress import serve
            serve(app, host=settings.host, port=settings.port, threads=8)
        except ImportError:
            log.warning("未安装 waitress，回退 Flask 开发服务器")
            app.run(host=settings.host, port=settings.port, threaded=True, debug=False)
    else:
        app.run(host=settings.host, port=settings.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
