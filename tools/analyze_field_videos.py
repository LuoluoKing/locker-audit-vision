# -*- coding: utf-8 -*-
"""tools/analyze_field_videos.py — 实拍现场素材（B站下载）的场域盘点工具

用途：对 EXAMPLE/bilibili-vedio/ 下的实拍视频逐段抽帧、生成联系表（contact
sheet）与亮度/运动统计，供论文的「部署域特征」分析与素材选用。

产出：
  logs/field_frames/<video_key>/f000.png ... 抽帧
  logs/field_frames/<video_key>/sheet.png    该视频联系表
  logs/field_survey.json                     逐视频统计（分辨率/时长/亮度/
                                             夜昼判定/帧间差分运动量）

注意：本工具只读素材、只写分析产物，不修改任何源视频。
"""
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID_DIR = os.path.join(ROOT, "EXAMPLE", "bilibili-vedio")
OUT_DIR = os.path.join(ROOT, "logs", "field_frames")
SURVEY = os.path.join(ROOT, "logs", "field_survey.json")

N_SAMPLES = 6          # 每段视频抽帧数
SHEET_COLS = 3


def imwrite_cn(path, img):
    """中文路径安全写图：cv2.imwrite 在中文路径下会静默失败。"""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok


def ascii_key(name):
    """把中文文件名转成稳定短键：v01..v15（按排序）"""
    return name


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(VID_DIR) if f.lower().endswith(".mp4"))
    survey = []
    for vi, fn in enumerate(files, 1):
        key = "v%02d" % vi
        path = os.path.join(VID_DIR, fn)
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = n / fps if fps > 0 else 0.0

        vdir = os.path.join(OUT_DIR, key)
        os.makedirs(vdir, exist_ok=True)

        # 抽帧 + 亮度
        frames, brights = [], []
        for si in range(N_SAMPLES):
            idx = int(n * (si + 0.5) / N_SAMPLES)
            fr = read_frame(cap, idx)
            if fr is None:
                continue
            frames.append(fr)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            brights.append(float(np.mean(gray)))
            imwrite_cn(os.path.join(vdir, "f%03d.png" % idx), fr)

        # 运动量：连续 60 帧的帧间差分均值（取中段）
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2 - 30))
        prev, diffs = None, []
        for _ in range(60):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(g, prev))))
            prev = g
        cap.release()

        # 联系表
        if frames:
            tile_w, tile_h = 400, 225
            tiles = []
            for fr in frames:
                fh, fw = fr.shape[:2]
                sc = min(tile_w / fw, tile_h / fh)
                r = cv2.resize(fr, (max(1, int(fw * sc)), max(1, int(fh * sc))))
                c = np.full((tile_h, tile_w, 3), 245, np.uint8)
                y0 = (tile_h - r.shape[0]) // 2
                x0 = (tile_w - r.shape[1]) // 2
                c[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
                tiles.append(c)
            while len(tiles) % SHEET_COLS:
                tiles.append(np.full((tile_h, tile_w, 3), 245, np.uint8))
            rows = [np.hstack(tiles[i:i + SHEET_COLS])
                    for i in range(0, len(tiles), SHEET_COLS)]
            sheet = np.vstack(rows)
            cv2.putText(sheet, key, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            imwrite_cn(os.path.join(vdir, "sheet.png"), sheet)

        rec = {
            "key": key, "file": fn, "w": w, "h": h, "fps": round(fps, 2),
            "frames": n, "duration_s": round(dur, 1),
            "mean_brightness": round(float(np.mean(brights)), 1) if brights else None,
            "day_night": ("night" if brights and np.mean(brights) < 70 else "day"),
            "motion_mean": round(float(np.mean(diffs)), 2) if diffs else None,
        }
        survey.append(rec)
        print("%s  %-58s %4dx%-5d %6.1fs  bright=%5.1f  %s" % (
            key, fn[:58], w, h, dur, rec["mean_brightness"] or -1, rec["day_night"]))

    with open(SURVEY, "w", encoding="utf-8") as f:
        json.dump(survey, f, ensure_ascii=False, indent=1)
    print("\n调查表 ->", SURVEY)
    print("帧与联系表 ->", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
