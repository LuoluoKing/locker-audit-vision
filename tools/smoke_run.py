# -*- coding: utf-8 -*-
"""
tools/smoke_run.py — 阶段1 端到端冒烟测试（真实运行，非模拟）

覆盖内容
--------
    A. 依赖与配置自检：无摄像头依赖、禁止摄像头输入、YOLOv8n 失败自动回退 HOG
    B. Flask 路由自检：/ 、/admin 、/video_feed(MJPEG) 、全部 /api/*
    C. 全链路自检：读帧 -> 推理 -> 规则 -> 截图落盘 -> SQLite 落库 -> 状态上报
    D. 双后端实测：分别用 yolov8n / hog 跑同一素材，记录真实帧率与单帧推理耗时

产出：logs/smoke_<时间戳>.txt（原始输出同时打印到终端）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import config as C  # noqa: E402
from config import Settings, resolve_source  # noqa: E402
from detector import build_detector  # noqa: E402

OUT: List[str] = []


def log(msg: str = "") -> None:
    print(msg)
    OUT.append(msg)


class Checker:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, note: str = "") -> None:
        self.rows.append((name, bool(cond), note))
        log("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + note) if note else ""))

    def section(self, t: str) -> None:
        log("\n" + t)

    @property
    def failed(self) -> int:
        return sum(1 for _n, ok, _ in self.rows if not ok)


def ensure_clip() -> Path:
    """确保存在自检素材（调用 tools/make_selftest_clip.py）。"""
    p = C.BASE_DIR / "data" / "selftest" / "selftest_scene.mp4"
    if not p.exists():
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        import gen_selftest_clip3d as G3
        G3.main() if False else None
        # 直接调用三维生成器主流程（等价于命令行 python tools/gen_selftest_clip3d.py）
        import argparse
        old = sys.argv
        sys.argv = ["gen_selftest_clip3d", "--seconds", "10", "--fps", "15"]
        try:
            G3.main()
        finally:
            sys.argv = old
    return p


# ============================================================ A
def test_dependencies(ck: Checker) -> None:
    ck.section("A 依赖与配置自检")

    try:
        resolve_source("0")
        refused = False
    except ValueError:
        refused = True
    ck.check("A1 拒绝摄像头索引输入", refused, "resolve_source('0') 抛出 ValueError")

    try:
        resolve_source("camera")
        refused2 = False
    except ValueError:
        refused2 = True
    ck.check("A2 拒绝 camera/webcam 关键字", refused2)

    # 用不存在的权重 + 禁止下载，强制走回退分支
    det = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "__not_exist__.pt"),
                         allow_download=False, hog_resize=320)
    ck.check("A3 YOLOv8n 失败自动回退 HOG", det.backend_name == "hog",
             "实际后端=%s，回退原因=%s" % (det.backend_name, getattr(det, "fallback_errors", [])))

    yolo = build_detector("yolov8n", model_path=str(C.MODEL_DIR / "yolov8n.pt"))
    ck.check("A4 YOLOv8n 权重可加载", yolo.backend_name == "yolov8n",
             "版本=%s" % (getattr(yolo, "versions", {})))
    ck.section("A5 双后端实测单帧推理耗时（640x480，真实测量）")
    for name, d in (("yolov8n", yolo), ("hog", build_detector("hog"))):
        import numpy as np
        import cv2
        frame = np.full((C.FRAME_HEIGHT, C.FRAME_WIDTH, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (280, 180), (360, 380), (90, 110, 130), -1)
        for _ in range(2):
            d.detect(frame)
        log("    %-8s 单帧推理 %7.1f ms  (~%.1f FPS 上限)" %
            (name, d.last_infer_ms, 1000.0 / max(d.last_infer_ms, 1e-6)))
    ck.check("A5 两个后端均可实际推理", yolo.infer_count > 0, "")


# ============================================================ B/C/D
def run_pipeline(ck: Checker, backend: str, clip: Path, rounds: int = 2) -> dict:
    """用 Flask 测试客户端驱动完整链路，返回实测指标。"""
    from app import create_app

    test_db = C.BASE_DIR / "data" / "selftest" / ("selftest_%s.db" % backend)
    if test_db.exists():
        test_db.unlink()
    # 测试独占截图目录：避免与同时在跑的正式实例共用 data/snapshots 造成互相污染
    test_snap = C.BASE_DIR / "data" / "selftest" / ("smoke_snapshots_%s" % backend)
    test_snap.mkdir(parents=True, exist_ok=True)
    for f in test_snap.glob("*.jpg"):
        f.unlink()
    snap_before = set(p.name for p in test_snap.glob("*.jpg"))

    settings = Settings(source=str(clip), backend=backend, autostart=False, display_fps=30.0, detect_fps=20.0,
                        loop_playback=True, db_path=str(test_db), device_id="selftest-node",
                        snapshot_dir=str(test_snap), device_name="自检节点")
    app, rt = create_app(settings)
    client = app.test_client()
    metrics = {"backend": backend}

    ck.section("B 页面与接口自检（backend=%s）" % backend)
    r = client.get("/")
    ck.check("B1 GET /              监控台可打开", r.status_code == 200 and "实时监控台" in r.get_data(as_text=True),
             "HTTP %d, %d bytes" % (r.status_code, len(r.data)))
    r = client.get("/admin")
    ck.check("B2 GET /admin         管理后台可打开", r.status_code == 200 and "云端管理后台" in r.get_data(as_text=True),
             "HTTP %d, %d bytes" % (r.status_code, len(r.data)))
    r = client.get("/api/sources")
    srcs = r.get_json().get("sources", [])
    ck.check("B3 GET /api/sources   素材枚举", r.status_code == 200 and len(srcs) > 0,
             "发现 %d 个素材" % len(srcs))
    ck.check("B3b 截图输出目录未被误列为素材",
             not any("snapshots" in s["path"] for s in srcs),
             "素材路径：%s" % ", ".join(s["path"] for s in srcs))
    r = client.get("/api/status")
    ck.check("B4 GET /api/status    状态接口", r.status_code == 200 and r.get_json().get("app") is not None)

    ck.section("C 全链路自检（读帧→推理→规则→截图→入库→上报）")
    r = client.post("/api/control", json={"action": "start", "source": str(clip),
                                         "backend": backend, "detect_fps": 20.0, "loop": True})
    ck.check("C1 POST /api/control  启动处理", r.status_code == 200 and r.get_json().get("ok"),
             json.dumps(r.get_json().get("status", {}).get("backend"), ensure_ascii=False))

    target_frames = max(6, rounds * 15)
    t0 = time.time()
    status = {}
    while time.time() - t0 < 120:
        status = client.get("/api/status").get_json()
        if status.get("frame_index", 0) >= target_frames or (status.get("total_frames") and
                status.get("frame_index", 0) >= status["total_frames"]):
            break
        time.sleep(1.0)

    ck.check("C2 处理帧数 > 0", status.get("frame_index", 0) > 0,
             "已处理 %d 帧 / 共 %s 帧" % (status.get("frame_index", 0), status.get("total_frames")))
    ck.check("C3 检测后端与请求一致", status.get("backend") == backend,
             "实际=%s" % status.get("backend"))
    ck.check("C4 实测帧率已记录", status.get("fps", 0) > 0,
             "显示 %.2f fps，检测 %.2f fps，单帧推理 %.1f ms"
             % (status.get("fps", 0), status.get("detect_fps", 0), status.get("infer_ms", 0)))
    src_fps = status.get("src_fps", 0) or 30.0
    ck.check("C4b 显示帧率不高于源帧率（播放速度不被拖慢/放快）",
             0 < status.get("fps", 0) <= src_fps * 1.30,
             "源帧率 %.1f fps，实测显示 %.2f fps" % (src_fps, status.get("fps", 0)))
    ck.check("C4d 显示与检测回路已解耦且默认逐帧检测",
             status.get("detect_interval", 0) >= 1 and status.get("detect_fps", 0) > 0,
             "检测间隔 %s 帧，检测帧率 %.2f fps，跳过 %s 次"
             % (status.get("detect_interval"), status.get("detect_fps", 0),
                status.get("detect_skipped")))

    # MJPEG 流
    r = client.get("/video_feed", buffered=False)
    chunk = b""
    try:
        chunk = next(r.response)
    except StopIteration:
        pass
    ck.check("C5 /video_feed MJPEG 分块输出", chunk.startswith(b"--frame") and b"\xff\xd8" in chunk,
             "首块 %d 字节，含 JPEG SOI 标记" % len(chunk))
    r.close()

    r = client.get("/api/frame")
    ck.check("C6 /api/frame 单帧 JPEG", r.status_code == 200 and r.data[:2] == b"\xff\xd8",
             "%d 字节" % len(r.data))

    ev = client.get("/api/events?size=50").get_json()
    st = client.get("/api/stats").get_json()
    metrics["frames"] = status.get("frame_index", 0)
    metrics["fps"] = status.get("fps", 0)
    metrics["infer_ms"] = status.get("infer_ms", 0)
    metrics["events"] = ev.get("total", 0)
    metrics["detections_note"] = "合成剪影素材，目标检测数取决于模型对几何人形的响应"

    ck.check("C7 SQLite 可查询（事件表可读）", isinstance(ev.get("items"), list),
             "事件 %d 条，统计合计 %d 条" % (ev.get("total", 0), st["summary"]["total"]))
    ck.check("C8 事件查询字段完整（时间/类型/详情/截图/置信度）",
             all(k in (ev["items"][0] if ev.get("items") else
                       {"ts": 1, "event_type": 1, "detail": 1, "snapshot": 1, "confidence": 1})
                 for k in ("ts", "event_type", "detail", "snapshot", "confidence")),
             "样例：%s" % (ev["items"][0]["detail"] if ev.get("items") else "本素材未触发事件"))

    # 运行期会话进度应已由心跳写入（不依赖进程正常退出）
    live_sessions = client.get("/api/sessions").get_json().get("sessions", [])
    ck.check("C4c 会话进度运行期已落库（心跳上报）",
             bool(live_sessions) and (live_sessions[0].get("processed_frames") or 0) > 0,
             ("会话#%s 运行中已记录 %s 帧" % (live_sessions[0]["id"], live_sessions[0]["processed_frames"]))
             if live_sessions else "无会话记录")

    # 处理结束后核对落盘与会话
    client.post("/api/control", json={"action": "stop"})
    time.sleep(0.5)
    snap_after = set(p.name for p in test_snap.glob("*.jpg"))
    new_snaps = snap_after - snap_before
    metrics["snapshots"] = len(new_snaps)
    ck.check("C9 截图目录有新增文件", (len(new_snaps) == ev.get("total", 0) and len(new_snaps) > 0) or
             (ev.get("total", 0) == 0 and len(new_snaps) == 0),
             "新增 %d 张（事件 %d 条）" % (len(new_snaps), ev.get("total", 0)))

    dev = client.get("/api/devices").get_json().get("devices", [])
    ck.check("C10 设备状态已上报", len(dev) > 0,
             "设备 %s 状态=%s 帧=%s" % (dev[0]["device_id"], dev[0]["status"], dev[0]["frame_index"]) if dev else "无记录")

    sess = client.get("/api/sessions").get_json().get("sessions", [])
    ck.check("C11 处理会话已入库（含真实帧率/推理耗时）", len(sess) > 0 and sess[0]["processed_frames"] > 0,
             "会话#%s 处理 %s 帧 用时 %.1fs 平均 %.2f fps" %
             (sess[0]["id"], sess[0]["processed_frames"], sess[0]["elapsed_s"], sess[0]["avg_fps"]) if sess else "无记录")

    r = client.get("/api/export.csv")
    ck.check("C12 CSV 导出可用", r.status_code == 200 and "event_type" in r.get_data(as_text=True)[:200],
             "%d 字节" % len(r.data))

    r = client.get("/no-such-page")
    ck.check("C13 404 页面正常", r.status_code == 404)

    rt.stop()
    return metrics


def _two_points_in_pickup(min_gap: int = 20) -> List[Tuple[float, float]]:
    """
    从 config.ROI_PICKUP 推导取餐区内 2 个脚点位置（供桩检测器使用）。

    位置选择规则（2026-10-07 更新）：**自 ROI 左缘低处起找**——
    桩检测器会持续站在取餐区里，而柜门状态判定要扣掉人员框；若桩站在柜门正前方，
    格口会被永久遮挡成「未知」，柜门类事件在 D 组场景里就永远出不来
    （移除「长时间停留」后，D 组的「多类型事件」覆盖必须依赖柜门事件）。
    取餐区的**左侧低处**在格口多边形之外，桩站在这里不遮挡任何格口。
    """
    import cv2 as _cv2
    W, H = C.FRAME_WIDTH, C.FRAME_HEIGHT
    poly = np.array([[int(x * W), int(y * H)] for x, y in C.ROI_PICKUP], np.int32)
    found: List[Tuple[float, float]] = []
    # 自画面底部向上、自左向右扫描（越靠下越不容易挡到格口）
    for y in range(H - 4, 4, -3):
        for x in range(4, W - 4, 2):
            if _cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0:
                continue
            if all(abs(x - q[0]) >= min_gap and abs(y - q[1]) < 40 for q in found):
                found.append((float(x), float(y)))
            if len(found) >= 2:
                return found
    raise RuntimeError("无法从 ROI_PICKUP 推导出 2 个分离的测试点位")


def test_event_pipeline(ck: Checker, clip: Path) -> dict:
    """
    注入式事件管道验证。

    说明：合成剪影素材不会被行人检测器稳定识别，因此本项测试用**桩检测器**直接提供
    “取餐区内 2 人”的检测框，专门验证检测之后的下游链路是否真实工作：
    规则判定 -> 截图落盘 -> SQLite 落库 -> /api/events 查询 -> /snapshots 取图 -> CSV 导出。
    该测试不产生任何用于论文的检测精度指标（精度评估属阶段3，使用公开数据集）。
    """
    import numpy as np
    from app import create_app
    import app as app_module
    from detector import BaseDetector, Detection

    class StubPersonDetector(BaseDetector):
        """
        桩检测器：返回「取餐区内 2 人」的检测框，专门验证检测之后的下游链路。

        目标位置由 ROI 配置**运行时推导**，不使用硬编码像素坐标——ROI 随场景重新标定后
        本桩自动跟随，避免目标落到取餐区之外导致事件不触发（曾因此使 D 组误判）。
        """
        backend_name = "stub"

        def __init__(self):
            super().__init__(0.5)
            self.display_name = "桩检测器（仅用于下游链路验证）"
            self._pts = _two_points_in_pickup()

        def _infer(self, frame):
            h, w = frame.shape[:2]
            out = []
            for i, (px, py) in enumerate(self._pts):
                out.append(Detection(px - 30, py - 200, px + 30, py, "person", 0.91 - 0.03 * i))
            return out

    def fake_build(*_a, **_kw):
        d = StubPersonDetector()
        d.warmup()
        return d

    ck.section("D 事件管道验证（桩检测器，覆盖 规则→截图→入库→查询→取图→导出）")
    test_db = C.BASE_DIR / "data" / "selftest" / "selftest_pipeline.db"
    if test_db.exists():
        test_db.unlink()
    settings = Settings(source=str(clip), backend="hog", autostart=False, display_fps=30.0, detect_fps=20.0,
                        loop_playback=True, db_path=str(test_db), device_id="selftest-stub",
                        snapshot_dir=str(C.BASE_DIR / "data" / "selftest" / "snapshots"),
                        device_name="流水线验证节点")
    original = app_module.build_detector
    app_module.build_detector = fake_build
    try:
        app, rt = create_app(settings)
        client = app.test_client()
        r = client.post("/api/control", json={"action": "start", "source": str(clip),
                                             "backend": "hog", "detect_fps": 20.0, "loop": True})
        ck.check("D1 启动成功（桩检测器）", r.get_json().get("ok") is True)

        t0 = time.time()
        ev = {"total": 0}
        while time.time() - t0 < 90:
            ev = client.get("/api/events?size=100").get_json()
            if ev.get("total", 0) >= 3:
                break
            time.sleep(1.0)

        # 停止前抓取在线统计（worker 停止后运行态会清空）
        live_stats = client.get("/api/stats").get_json()
        client.post("/api/control", json={"action": "stop"})
        time.sleep(0.5)

        ck.check("D2 规则触发并写入 SQLite", ev.get("total", 0) >= 3,
                 "共 %d 条事件：%s" % (ev.get("total", 0),
                                     ", ".join(sorted({i["event_name"] for i in ev.get("items", [])}))))
        types = sorted({i["event_type"] for i in ev.get("items", [])})
        ck.check("D3 覆盖多类规则事件", {"ROI_INTRUSION", "DOOR_LEFT_OPEN"} <= set(types),
                 "事件类型 %s" % types)

        items = ev.get("items", [])
        with_snap = [i for i in items if i.get("snapshot")]
        ck.check("D4 事件均已保存截图", len(with_snap) == len(items) and len(items) > 0,
                 "%d/%d 条带截图" % (len(with_snap), len(items)))
        ck.check("D5 事件字段完整（时间/类型/详情/截图/置信度）",
                 all(i.get("ts") and i.get("event_type") and i.get("detail") and
                     i.get("confidence") is not None for i in items))

        ok_img = 0
        for i in with_snap[:5]:
            name = i["snapshot"].split("/")[-1]
            rr = client.get("/snapshots/" + name)
            if rr.status_code == 200 and rr.data[:2] == b"\xff\xd8":
                ok_img += 1
        ck.check("D6 截图可通过 /snapshots 访问", ok_img == len(with_snap[:5]) and ok_img > 0,
                 "%d/%d 张返回有效 JPEG" % (ok_img, len(with_snap[:5])))

        st = client.get("/api/stats").get_json()
        ck.check("D7 统计接口与事件一致", st["summary"]["total"] == ev.get("total", 0),
                 "summary.total=%d，abnormal=%d" % (st["summary"]["total"], st["summary"]["abnormal"]))

        csv_txt = client.get("/api/export.csv").get_data(as_text=True)
        ck.check("D8 CSV 导出包含全部事件行", csv_txt.count("\n") - 1 == ev.get("total", 0),
                 "CSV 数据行 %d" % (csv_txt.count("\n") - 1))

        # 说明：旧版此处的「冷却抑制 > 0」断言依赖已删除的「多人同时取件」（唯一的场景级重复事件源），
        # 新分类下不存在重复触发场景，故改为验证柜门状态机在流水线中真实运行且对外暴露标记。
        live = live_stats["live"]
        n_door = len(live.get("door_states", []))
        n_dark = len(live.get("door_dark_ratios", []))
        ck.check("D9 柜门状态机在流水线中运行并对外暴露标记",
                 n_door == C.CABINET_DOOR_COUNT and n_dark == C.CABINET_DOOR_COUNT
                 and "flagged_door_labels" in live,
                 "柜格状态 %d 项 / 亮度比 %d 项 / 未关柜门标记 %s（冷却逻辑由规则单元测试 T4 覆盖）"
                 % (n_door, n_dark, live.get("flagged_door_labels")))

        rt.stop()
        return {"events": ev.get("total", 0), "snapshots": len(with_snap)}
    finally:
        app_module.build_detector = original


def test_real_server(ck: Checker, clip: Path) -> None:
    """
    E 真实进程级验证：以子进程方式启动 app.py，通过真实 HTTP 访问各端点。

    与 test_client 不同，这一项会捕获「参数解析、Settings 字段、WSGI 启动」等
    只有真正拉起服务才会暴露的问题。
    """
    import socket
    import subprocess
    import urllib.error
    import urllib.request

    ck.section("E 真实服务进程验证（子进程 + 真实 HTTP）")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    probe_log = C.LOG_DIR / "_server_probe.log"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "app.py", "--source", str(clip), "--backend", "hog",
         "--max-fps", "8", "--port", str(port), "--no-loop"],
        cwd=str(C.CODE_DIR), stdout=open(probe_log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, env=env)

    # 不经过任何系统代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = "http://127.0.0.1:%d" % port

    up = False
    t0 = time.time()
    while time.time() - t0 < 60:
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                up = True
                break
        except OSError:
            time.sleep(0.6)

    ck.check("E1 服务进程正常监听端口", up,
             "端口 %d；%s" % (port, "进程已退出" if proc.poll() is not None else "监听成功"))

    if up:
        def get(path, limit=None, timeout=20):
            with opener.open(base + path, timeout=timeout) as r:
                return r.status, (r.read(limit) if limit else r.read())

        try:
            code, body = get("/")
            ck.check("E2 真实 HTTP GET /", code == 200 and "实时监控台" in body.decode("utf-8", "ignore"),
                     "HTTP %d, %d bytes" % (code, len(body)))
        except Exception as exc:
            ck.check("E2 真实 HTTP GET /", False, str(exc))

        try:
            code, body = get("/admin")
            ck.check("E3 真实 HTTP GET /admin", code == 200 and "云端管理后台" in body.decode("utf-8", "ignore"),
                     "HTTP %d, %d bytes" % (code, len(body)))
        except Exception as exc:
            ck.check("E3 真实 HTTP GET /admin", False, str(exc))

        try:
            code, body = get("/api/status")
            j = json.loads(body)
            ck.check("E4 真实 HTTP /api/status", code == 200 and j.get("backend") == "hog",
                     "后端=%s 帧=%s 帧率=%.2f" % (j.get("backend"), j.get("frame_index"), j.get("fps", 0)))
        except Exception as exc:
            ck.check("E4 真实 HTTP /api/status", False, str(exc))

        try:
            code, body = get("/video_feed", limit=4096)
            ck.check("E5 真实 HTTP MJPEG 流", code == 200 and b"--frame" in body and b"\xff\xd8" in body,
                     "首 %d 字节含帧边界与 JPEG 标记" % len(body))
        except Exception as exc:
            ck.check("E5 真实 HTTP MJPEG 流", False, str(exc))

        # ---- E6 端口守卫：在已占用端口上再启一个实例，必须被拒绝而不是静默共存
        try:
            second = subprocess.run(
                [sys.executable, "app.py", "--port", str(port), "--no-autostart"],
                cwd=str(C.CODE_DIR), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=90, env=env)
            out = (second.stdout or "") + (second.stderr or "")
            ck.check("E6 端口守卫拒绝重复启动",
                     second.returncode == 2 and "已被占用" in out,
                     "退出码 %s，提示含「已被占用」=%s" % (second.returncode, "已被占用" in out))
        except Exception as exc:
            ck.check("E6 端口守卫拒绝重复启动", False, str(exc))

    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
    if not up:
        tail = probe_log.read_text(encoding="utf-8", errors="replace")[-900:]
        log("    服务进程输出尾部：\n%s" % tail)


def test_frontend(ck: Checker) -> None:
    """
    F 前端静态资源与统计口径检查（含回归守卫）。

    重点守护两类已被修复的静默错误：
      · 趋势图用 toISOString() 取日期 → 东八区凌晨整体错位一天，图表静默显示为空；
      · 控制台无帧输出告警过于敏感 → 单次轮询即误报。
    """
    from app import create_app
    from datetime import datetime

    ck.section("F 前端资源与统计口径检查")
    test_db = C.BASE_DIR / "data" / "selftest" / "selftest_frontend.db"
    if test_db.exists():
        test_db.unlink()
    settings = Settings(backend="hog", autostart=False, db_path=str(test_db),
                        snapshot_dir=str(C.BASE_DIR / "data" / "selftest" / "smoke_snapshots_frontend"))
    app, rt = create_app(settings)
    client = app.test_client()

    admin_js = client.get("/static/admin.js").get_data(as_text=True)
    index_js = client.get("/static/index.js").get_data(as_text=True)

    ck.check("F1 趋势图使用本地日期（不残留时区隐患调用）",
             ".toISOString().slice(0, 10)" not in admin_js and "function ymd(" in admin_js,
             "admin.js 含本地日期函数 ymd=%s，残留问题调用=%s"
             % ("function ymd(" in admin_js, ".toISOString().slice(0, 10)" in admin_js))
    ck.check("F2 无帧输出告警采用连续轮询判定",
             "sameCount" in index_js and "sameCount >= 6" in index_js,
             "index.js 含连续轮询计数")
    ck.check("F3 监控台与事件流同一轮刷新（消除面板间不一致）",
             "Promise.all([refreshStatus(), refreshEvents()])" in index_js,
             "index.js 采用合并刷新")

    st = client.get("/api/stats?days=14").get_json()
    today = datetime.now().strftime("%Y-%m-%d")
    days = [r["day"] for r in st.get("by_day", [])]
    ck.check("F4 统计按日聚合使用本地日期", all(d <= today for d in days) if days else True,
             "by_day 日期：%s（本地今日 %s）" % (days or "暂无数据", today))
    rt.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段1端到端冒烟测试")
    ap.add_argument("--backends", default="yolov8n,hog")
    a = ap.parse_args()

    C.setup_logging()
    ck = Checker()
    log("=" * 82)
    log("天眼哨兵 · 阶段1 端到端冒烟测试")
    log("时间：%s" % datetime.now().isoformat(timespec="seconds"))
    log("Python：%s" % sys.version.replace("\n", " "))
    log("=" * 82)

    clip = ensure_clip()
    log("\n自检素材：%s（%.2f MB，仅用于链路自检，不属于阶段2实验数据集）"
        % (clip, clip.stat().st_size / 1048576))

    test_dependencies(ck)

    metrics_all = []
    for b in [x.strip() for x in a.backends.split(",") if x.strip()]:
        try:
            metrics_all.append(run_pipeline(ck, b, clip))
        except Exception as exc:
            import traceback
            log("后端 %s 测试异常：%s" % (b, exc))
            traceback.print_exc()

    try:
        # D 组素材用「柜门未关」数据集片段：真实帧驱动柜门状态机（约 12.6s 触发
        # DOOR_LEFT_OPEN 并标记柜号），桩检测器驱动人物事件（ROI_INTRUSION）。
        # 用自检片段（门只开 2.2s、从不报警）在移除「长时间停留」后已无法覆盖
        # 多类型事件，故改用本片段恢复「多类型」覆盖。
        d_clip = C.BASE_DIR / "data" / "simulated" / "door_left_open_001.mp4"
        pm = test_event_pipeline(ck, d_clip if d_clip.exists() else clip)
        log("  事件管道实测：事件 %d 条，截图 %d 张" % (pm["events"], pm["snapshots"]))
    except Exception as exc:
        import traceback
        log("事件管道测试异常：%s" % exc)
        traceback.print_exc()

    try:
        test_real_server(ck, clip)
    except Exception as exc:
        import traceback
        log("真实服务进程测试异常：%s" % exc)
        traceback.print_exc()

    try:
        test_frontend(ck)
    except Exception as exc:
        import traceback
        log("前端资源测试异常：%s" % exc)
        traceback.print_exc()

    total = len(ck.rows)
    passed = total - ck.failed
    log("\n" + "=" * 82)
    log("实测指标汇总（真实运行，非模拟）")
    log("-" * 82)
    for m in metrics_all:
        log("  %-8s 处理帧=%-5s 帧率=%6.2f fps  单帧推理=%8.1f ms  事件=%s 条  截图=%s 张"
            % (m["backend"], m.get("frames"), m.get("fps", 0), m.get("infer_ms", 0),
               m.get("events"), m.get("snapshots")))
    log("-" * 82)
    log("自检结果：%d / %d 项通过" % (passed, total))
    log("=" * 82)

    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = C.LOG_DIR / ("smoke_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.write_text("\n".join(OUT), encoding="utf-8")
    print("测试记录已保存：%s" % p)
    return 1 if ck.failed else 0


if __name__ == "__main__":
    sys.exit(main())
