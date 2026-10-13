# -*- coding: utf-8 -*-
"""
database.py — SQLite 数据层

事件表字段（对应阶段1 要求）：
    ts(时间) / event_type(类型) / detail(详情) / snapshot(截图路径) / confidence(置信度)
另附会话表与设备状态表，供 Web 管理后台做设备状态、事件统计与历史查询。

线程安全：``check_same_thread=False`` + 全局可重入锁 + WAL 模式，
使视频工作线程与 Flask 请求线程可安全并发读写。
"""
from __future__ import annotations

import csv
import io
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("tianyan.db")

__all__ = ["Database", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_DDL = [
    # ---------------- 事件表
    """
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT    NOT NULL,               -- 事件时间 ISO8601
        event_type  TEXT    NOT NULL,               -- 类型（英文枚举）
        event_name  TEXT    NOT NULL,               -- 类型（中文名）
        severity    TEXT    NOT NULL DEFAULT 'info',
        detail      TEXT,                           -- 详情
        snapshot    TEXT,                           -- 截图相对路径
        confidence  REAL,                           -- 置信度 0~1
        track_id    INTEGER,
        frame_index INTEGER,
        source      TEXT,                           -- 视频源文件名
        duration_s  REAL,                           -- 关联持续时长
        persons     INTEGER,                        -- 关联人数
        is_abnormal INTEGER NOT NULL DEFAULT 0,     -- 是否属于异常事件
        created_at  TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_abnormal  ON events(is_abnormal)",

    # ---------------- 处理会话表
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        source       TEXT,
        backend      TEXT,
        resolution   TEXT,
        started_at   TEXT,
        ended_at     TEXT,
        total_frames INTEGER DEFAULT 0,
        processed_frames INTEGER DEFAULT 0,
        elapsed_s    REAL DEFAULT 0,
        avg_fps      REAL DEFAULT 0,
        avg_infer_ms REAL DEFAULT 0,
        event_count  INTEGER DEFAULT 0,
        note         TEXT
    )
    """,

    # ---------------- 设备状态表
    """
    CREATE TABLE IF NOT EXISTS device_status (
        device_id   TEXT PRIMARY KEY,
        device_name TEXT,
        source      TEXT,
        backend     TEXT,
        status      TEXT,
        fps         REAL,
        infer_ms    REAL,
        frame_index INTEGER,
        event_count INTEGER,
        updated_at  TEXT
    )
    """,
]


class Database:
    """轻量 SQLite 封装。所有公开方法均可在多线程下调用。"""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            for ddl in _DDL:
                self._conn.execute(ddl)
            self._conn.commit()
        log.info("数据库就绪：%s", self.path)

    # ------------------------------------------------------------ 内部
    def _exec(self, sql: str, args: Tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _query(self, sql: str, args: Tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def _scalar(self, sql: str, args: Tuple = ()) -> Any:
        with self._lock:
            cur = self._conn.execute(sql, args)
            row = cur.fetchone()
            return None if row is None else row[0]

    # ------------------------------------------------------------ 事件
    def insert_event(self, *, ts: str, event_type: str, event_name: str, severity: str = "info",
                     detail: str = "", snapshot: Optional[str] = None, confidence: float = 0.0,
                     track_id: Optional[int] = None, frame_index: int = 0, source: str = "",
                     duration_s: Optional[float] = None, persons: Optional[int] = None,
                     is_abnormal: bool = False) -> int:
        cur = self._exec(
            """INSERT INTO events
               (ts, event_type, event_name, severity, detail, snapshot, confidence,
                track_id, frame_index, source, duration_s, persons, is_abnormal, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, event_type, event_name, severity, detail, snapshot, float(confidence),
             track_id, int(frame_index), source, duration_s, persons,
             1 if is_abnormal else 0, datetime.now().isoformat(timespec="seconds")))
        return int(cur.lastrowid)

    def query_events(self, *, event_type: Optional[str] = None, start: Optional[str] = None,
                     end: Optional[str] = None, keyword: Optional[str] = None,
                     only_abnormal: bool = False, limit: int = 50, offset: int = 0,
                     order: str = "DESC") -> List[Dict[str, Any]]:
        where, args = self._build_filter(event_type, start, end, keyword, only_abnormal)
        order = "ASC" if str(order).upper() == "ASC" else "DESC"
        sql = "SELECT * FROM events %s ORDER BY ts %s, id %s LIMIT ? OFFSET ?" % (where, order, order)
        return self._query(sql, args + (int(limit), int(offset)))

    def count_events(self, *, event_type: Optional[str] = None, start: Optional[str] = None,
                     end: Optional[str] = None, keyword: Optional[str] = None,
                     only_abnormal: bool = False) -> int:
        where, args = self._build_filter(event_type, start, end, keyword, only_abnormal)
        return int(self._scalar("SELECT COUNT(*) FROM events %s" % where, args) or 0)

    @staticmethod
    def _build_filter(event_type, start, end, keyword, only_abnormal) -> Tuple[str, Tuple]:
        cond, args = [], []
        if event_type:
            cond.append("event_type = ?")
            args.append(event_type)
        if start:
            cond.append("ts >= ?")
            args.append(start)
        if end:
            cond.append("ts <= ?")
            args.append(end)
        if keyword:
            cond.append("(detail LIKE ? OR event_name LIKE ? OR source LIKE ?)")
            kw = "%" + keyword + "%"
            args += [kw, kw, kw]
        if only_abnormal:
            cond.append("is_abnormal = 1")
        return ("WHERE " + " AND ".join(cond) if cond else ""), tuple(args)

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM events WHERE id = ?", (int(event_id),))
        return rows[0] if rows else None

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM events ORDER BY ts DESC, id DESC LIMIT ?", (int(limit),))

    # ------------------------------------------------------------ 统计
    def stats_by_type(self, *, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        where, args = self._build_filter(None, start, end, None, False)
        return self._query(
            """SELECT event_type, event_name, COUNT(*) AS cnt, MAX(ts) AS last_ts,
                      SUM(is_abnormal) AS abnormal_cnt
               FROM events %s GROUP BY event_type, event_name ORDER BY cnt DESC""" % where, args)

    def stats_by_day(self, days: int = 14) -> List[Dict[str, Any]]:
        start = (datetime.now() - timedelta(days=int(days) - 1)).strftime("%Y-%m-%d")
        return self._query(
            """SELECT substr(ts,1,10) AS day,
                      COUNT(*) AS total,
                      SUM(CASE WHEN is_abnormal=1 THEN 1 ELSE 0 END) AS abnormal
               FROM events WHERE substr(ts,1,10) >= ?
               GROUP BY day ORDER BY day ASC""", (start,))

    def stats_summary(self, *, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        where, args = self._build_filter(None, start, end, None, False)
        row = self._query(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN is_abnormal=1 THEN 1 ELSE 0 END) AS abnormal,
                      SUM(CASE WHEN is_abnormal=0 THEN 1 ELSE 0 END) AS normal,
                      AVG(confidence) AS avg_conf,
                      COUNT(DISTINCT source) AS sources
               FROM events %s""" % where, args)
        d = dict(row[0]) if row else {}
        d["total"] = int(d.get("total") or 0)
        d["abnormal"] = int(d.get("abnormal") or 0)
        d["normal"] = int(d.get("normal") or 0)
        d["avg_conf"] = round(float(d.get("avg_conf") or 0.0), 4)
        d["sources"] = int(d.get("sources") or 0)
        d["first_ts"] = self._scalar("SELECT MIN(ts) FROM events") or ""
        d["last_ts"] = self._scalar("SELECT MAX(ts) FROM events") or ""
        return d

    # ------------------------------------------------------------ 会话
    def insert_session(self, *, source: str, backend: str, resolution: str, started_at: str) -> int:
        cur = self._exec(
            """INSERT INTO sessions (source, backend, resolution, started_at, ended_at,
                                     total_frames, processed_frames, elapsed_s, avg_fps,
                                     avg_infer_ms, event_count, note)
               VALUES (?,?,?,?,NULL,0,0,0,0,0,0,NULL)""",
            (source, backend, resolution, started_at))
        return int(cur.lastrowid)

    def update_session(self, session_id: int, **fields) -> None:
        allowed = {"ended_at", "total_frames", "processed_frames", "elapsed_s",
                   "avg_fps", "avg_infer_ms", "event_count", "note"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        sql = "UPDATE sessions SET " + ", ".join("%s = ?" % k for k in sets) + " WHERE id = ?"
        self._exec(sql, tuple(sets.values()) + (int(session_id),))

    def recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (int(limit),))

    # ------------------------------------------------------------ 设备状态
    def upsert_device_status(self, *, device_id: str, device_name: str, source: str, backend: str,
                             status: str, fps: float, infer_ms: float, frame_index: int,
                             event_count: int) -> None:
        self._exec(
            """INSERT INTO device_status
                 (device_id, device_name, source, backend, status, fps, infer_ms, frame_index, event_count, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(device_id) DO UPDATE SET
                 device_name=excluded.device_name, source=excluded.source, backend=excluded.backend,
                 status=excluded.status, fps=excluded.fps, infer_ms=excluded.infer_ms,
                 frame_index=excluded.frame_index, event_count=excluded.event_count,
                 updated_at=excluded.updated_at""",
            (device_id, device_name, source, backend, status, float(fps), float(infer_ms),
             int(frame_index), int(event_count), datetime.now().isoformat(timespec="seconds")))

    def list_devices(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM device_status ORDER BY device_id ASC")

    # ------------------------------------------------------------ 导出
    def export_csv(self, **filters) -> str:
        """把查询结果导出为 CSV 文本（UTF-8 BOM，Excel 直接可读）。"""
        rows = self.query_events(limit=100000, offset=0, order="DESC", **filters)
        buf = io.StringIO()
        cols = ["id", "ts", "event_type", "event_name", "severity", "detail",
                "confidence", "track_id", "frame_index", "source", "snapshot", "duration_s", "persons"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return "\ufeff" + buf.getvalue()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
