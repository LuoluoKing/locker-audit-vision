# -*- coding: utf-8 -*-
"""
tools/measure_stream.py — MJPEG 推流平滑度实测

用途
----
平均帧率（display_fps）无法反映「一卡一卡」的观感问题：平均 15fps 也可能意味着
「连续几帧挤在 20ms 内吐出，然后停顿 300ms」。本工具按**帧到达间隔**统计分布，
直接量化浏览器端看到的抖动。

指标
----
  p50 / p95 / p99 / max : 帧到达间隔分位数（ms）
  超期间隔占比          : > 1.5×、> 2×、> 3× 名义帧周期的比例
  最长停顿              : 单次最长间隔及其出现时刻

命令行
------
    python tools/measure_stream.py --url http://127.0.0.1:5000/video_feed --seconds 12
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import urllib.request


def measure(url: str, seconds: float, nominal_fps: float) -> dict:
    req = urllib.request.Request(url)
    # 分块必须足够小：若一次 read 拿到 64KB（可能含 2 帧），两帧会被记为同一时刻到达，
    # 人为制造 "0ms + 2×周期" 的假抖动。4KB 在 1MB/s 下约 4ms 分辨率，足够分辨 66ms 周期。
    chunk_size = 4096
    with urllib.request.urlopen(req, timeout=seconds + 15) as resp:
        buf = b""
        stamps: list = []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            now = time.perf_counter()
            buf += chunk
            # 每遇到一个 JPEG 结束标记即视为一帧到达，时间戳取该分块的到达时刻
            while True:
                i = buf.find(b"\xff\xd9")
                if i < 0:
                    break
                stamps.append(now)
                buf = buf[i + 2:]
    if len(stamps) < 3:
        return {"frames": len(stamps)}
    d = [1000.0 * (stamps[i + 1] - stamps[i]) for i in range(len(stamps) - 1)]
    d_sorted = sorted(d)
    period = 1000.0 / nominal_fps

    def q(p: float) -> float:
        return d_sorted[min(len(d_sorted) - 1, int(round(p * (len(d_sorted) - 1))))]

    worst = max(range(len(d)), key=lambda i: d[i])
    return {
        "frames": len(stamps),
        "wall_s": round(stamps[-1] - stamps[0], 2),
        "fps_avg": round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2),
        "nominal_period_ms": round(period, 1),
        "p50_ms": round(q(0.50), 1),
        "p95_ms": round(q(0.95), 1),
        "p99_ms": round(q(0.99), 1),
        "max_ms": round(d_sorted[-1], 1),
        "min_ms": round(d_sorted[0], 1),
        "stdev_ms": round(statistics.pstdev(d), 1),
        "over_1p5x_pct": round(100.0 * sum(1 for x in d if x > 1.5 * period) / len(d), 1),
        "over_2x_pct": round(100.0 * sum(1 for x in d if x > 2.0 * period) / len(d), 1),
        "over_3x_pct": round(100.0 * sum(1 for x in d if x > 3.0 * period) / len(d), 1),
        "worst_gap_at_s": round(stamps[worst] - stamps[0], 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MJPEG 推流平滑度实测")
    ap.add_argument("--url", default="http://127.0.0.1:5000/video_feed")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--nominal-fps", type=float, default=15.0)
    a = ap.parse_args()

    print("=" * 72)
    print("MJPEG 推流平滑度实测：%s（采样 %.0f s）" % (a.url, a.seconds))
    print("=" * 72)
    r = measure(a.url, a.seconds, a.nominal_fps)
    if r.get("frames", 0) < 3:
        print("采样帧数过少（%s），无法统计" % r.get("frames"))
        return 1
    print("  实际帧数 %d，时长 %.2f s，平均 %.2f fps（名义周期 %.1f ms）"
          % (r["frames"], r["wall_s"], r["fps_avg"], r["nominal_period_ms"]))
    print("  帧间隔分位数：p50 %.1f ms / p95 %.1f ms / p99 %.1f ms / max %.1f ms / min %.1f ms"
          % (r["p50_ms"], r["p95_ms"], r["p99_ms"], r["max_ms"], r["min_ms"]))
    print("  间隔标准差 %.1f ms" % r["stdev_ms"])
    print("  抖动占比：>%.1f ms(1.5×) %.1f%%  >%.1f ms(2×) %.1f%%  >%.1f ms(3×) %.1f%%"
          % (1.5 * r["nominal_period_ms"], r["over_1p5x_pct"],
             2.0 * r["nominal_period_ms"], r["over_2x_pct"],
             3.0 * r["nominal_period_ms"], r["over_3x_pct"]))
    print("  最长停顿出现在第 %.2f s" % r["worst_gap_at_s"])
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
