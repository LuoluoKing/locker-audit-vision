# -*- coding: utf-8 -*-
"""
config.py — 天眼哨兵系统全局配置

集中管理路径、默认参数与事件字典，供 app / video_worker / detector / rules 复用。
所有路径均相对项目根目录（tianyan_project/）解析，便于整体迁移。
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------- 路径

BASE_DIR = Path(__file__).resolve().parent.parent          # tianyan_project/
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"                            # 公开数据集视频
SIM_DIR = DATA_DIR / "simulated"                           # 阶段2 模拟视频
SNAP_DIR = DATA_DIR / "snapshots"                          # 事件截图
REAL_DEMO_DIR = BASE_DIR / "real_video"                    # AI 生成演示视频（非实拍，见下）
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
PAPER_DIR = BASE_DIR / "paper"
SC_DIR = BASE_DIR / "software_copyright"
PROMPT_DIR = BASE_DIR / "ai_prompts"

DB_PATH = DATA_DIR / "tianyan.db"

for _d in (DATA_DIR, VIDEO_DIR, SIM_DIR, SNAP_DIR, LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 版本

APP_NAME = "天眼哨兵校园外卖无感安防系统"
APP_VERSION = "V1.0"
APP_SUBTITLE = "基于多模态感知与边缘智能的校园外卖无感安防系统"

# ---------------------------------------------------------------- 视频规格

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 78
MAX_STREAM_FPS = 60.0          # MJPEG 推流轮询上限（仅用于兜底，正常由新帧事件驱动）

# ---------------------------------------------------------------- 默认 ROI
# 归一化坐标（0~1），与分辨率无关。
# 当前标定对应三维自检场景的机位（侧向 3/4 视角，见 tools/gen_selftest_clip3d.py）：
#   取餐区 = 取餐柜正前方的站立地面区域，实测「取餐站位脚点」投影于 (0.444, 0.860)。
# 换素材/换机位后需重新标定；标定方法见 docs/08_阶段2证据/README.md。
ROI_PICKUP = [
    [0.340, 0.775],
    [0.565, 0.795],
    [0.620, 0.950],
    [0.260, 0.940],
]
# 等候区已移除：饭点高峰期外卖柜周围聚集人员属正常场景，
# 「人员聚集」不再作为异常类别（需求方明确要求删除）。
CABINET_DOOR_COUNT = 6          # 柜门/柜格数量（3 行 × 2 列，行优先、自上而下）

# 6 个柜格门面在**关闭状态**下的画面四边形（归一化，顺时针）。
# 由机位与柜体几何投影反算得到，用于「柜门状态判定」与「取走柜内餐盒」判定。
# 当前坐标对应 `scene_kit.build_cabinet` 的柜门几何；`scene_kit.build_locker`
# （真实取餐柜外观）**刻意保持门面几何完全一致**，仅换配色并外挂操作屏/品牌条/编号牌，
# 因此本坐标对两者均有效。换几何或换机位后必须用
# `tools/calib_locker_compartments.py` 重新生成，并用
# `tools/calib_locker_door_threshold.py` 做「对角占优 + 暗比分离度」复核。
CABINET_COMPARTMENTS = [
    [[0.3557, 0.4348], [0.3515, 0.2790], [0.4703, 0.2815], [0.4712, 0.4550]],   # 柜格 0（上排左）
    [[0.6350, 0.4836], [0.6401, 0.2850], [0.4852, 0.2818], [0.4857, 0.4575]],   # 柜格 1（上排右）
    [[0.3598, 0.5848], [0.3558, 0.4379], [0.4713, 0.4584], [0.4722, 0.6210]],   # 柜格 2（中排左）
    [[0.6302, 0.6719], [0.6349, 0.4875], [0.4857, 0.4610], [0.4861, 0.6255]],   # 柜格 3（中排右）
    [[0.3637, 0.7265], [0.3599, 0.5877], [0.4722, 0.6242], [0.4730, 0.7768]],   # 柜格 4（下排左）
    [[0.6256, 0.8470], [0.6301, 0.6755], [0.4862, 0.6287], [0.4866, 0.7831]],   # 柜格 5（下排右）
]

# ---------------------------------------------------------------- 演示素材档案
# real_video/ 下的三段素材（异常取餐 / 未关柜门 / 正常取餐）**经逐帧核验为 AI 生成视频**
# （画面左上角有「AI生成」水印、右下角有「豆包AI生成」水印），并非实拍素材。
# 因此：
#   1. 它们只在界面中作为「AI 生成演示视频 · 非实拍」标注使用，
#      **不得进入论文的任何实验数据或指标统计**；
#   2. 该素材的机位、分辨率（720x1280 竖屏）与柜体几何（黄色柜体，约 5 行 x 3 列）
#      与合成数据集（6 格深蓝柜体、640x480 横屏）完全不同，
#      故使用 `DEMO_ROI_PICKUP` 作为**粗标定**取餐区，且**不提供柜格多边形**——
#      柜门开合 / 取走未关柜门外卖 / 手持多份 等依赖精细标定的判定在演示模式下关闭。
AI_DEMO_GROUP = "AI 生成演示视频（非实拍）"
AI_DEMO_NOTICE = (
    "当前素材为 AI 生成演示视频（非实拍）：画面左上角「AI生成」、右下角「豆包AI生成」水印可核验。"
    "该素材启用「演示档案」：柜门状态由「与基准帧的视觉一致性」判定（演示口径，"
    "逐格阈值见 config.AI_DEMO_DOOR_PROFILE），可演示「柜门未关」与「取走未关柜门外卖」告警；"
    "演示档阈值为演示专用、与论文实验阈值相互独立；"
    "本素材及其判定结果不参与论文的任何实验数据统计。"
)
AI_DEMO_FRAME_BANNER = "AI-GENERATED DEMO CLIP - NOT FIELD FOOTAGE"


# ---------------------------------------------------------------- AI 演示素材·门态演示档案
# 用途：为 `real_video/` 的 AI 生成演示素材提供「柜门未关 / 取走未关柜门外卖」的**演示口径**判定。
# 实现：`code/demo_door.py`（单应稳定化 + 逐格基准一致性 NCC + 状态机）。
# 边界：仅用于功能演示，**不进入实验数据集、不计入论文指标**；
#       阈值属演示档参数，与论文 E1 的全局阈值相互独立（见 docs/09_阶段3证据/ 判定记录）。
# 格口坐标来源：logs/real_check/cells_*.json（实测行界直接构造，见标定记录）。参考帧：
#       未关柜门=f30（7 行：上2高+下5矮），异常取餐=f30（8 行：上2高+下6矮）。
AI_DEMO_DOOR_PROFILE = {
    "未关柜门.mp4": {
        "ref_frame": 30,
        "stab_roi": [0.10, 0.10, 0.78, 0.99],
        "base_candidates": [2, 5, 10, 30, 75, 120, 160, 200, 240],
        "params": {
            "open_th": 0.70,            # 与基准的一致性 NCC 低于该值 → 疑似门态变化（实测：闭合≈0.87 / 敞开≈0.55~0.67）
            "closed_th": 0.82,          # 回滞：恢复到该值以上并持续 → 判回关闭
            "confirm_frames": 4,        # 连续「变化」帧数去抖（≈0.3s@12fps）
            "closed_frames": 6,
            "step_window_s": 0.8,       # 阶跃检查窗口（开门是快变、光照漂移不得触发）
            "step_drop": 0.15,
            "door_left_open_s": 3.0,    # 【演示档阈值】开启超过该秒数且人已离开 → 报警并标记柜号
            "search_pad": 16,           # 扩窗对齐（吸收稳定化残差）
            "occ_expand": 0.15,         # 校准期人物遮挡判定外扩比例
            "consensus_ncc": 0.72, "consensus_pad": 6,      # 两个候选参考帧的一致性下限
            "retrieval_touch_s": 1.2,   # 标记存续期间「人员接触」判定所需持续时长
            "retrieval_change_th": 0.62,# 接触前后格口内容变化（NCC 下限）
        },
        "cells": [
            {"row": 0, "col": 0, "skip": False, "poly": [[0.2694, 0.3540], [0.4076, 0.3540], [0.4076, 0.4480], [0.2694, 0.4480]]},
            {"row": 0, "col": 1, "skip": False, "poly": [[0.4166, 0.3540], [0.5614, 0.3540], [0.5614, 0.4480], [0.4166, 0.4480]]},
            {"row": 0, "col": 2, "skip": False, "poly": [[0.5690, 0.3540], [0.6620, 0.3540], [0.6620, 0.4480], [0.5690, 0.4480]]},
            {"row": 1, "col": 0, "skip": False, "poly": [[0.2694, 0.4540], [0.4076, 0.4540], [0.4076, 0.5470], [0.2694, 0.5470]]},
            {"row": 1, "col": 1, "skip": True, "poly": [[0.4166, 0.4540], [0.5614, 0.4540], [0.5614, 0.5470], [0.4166, 0.5470]]},
            {"row": 1, "col": 2, "skip": False, "poly": [[0.5690, 0.4540], [0.6620, 0.4540], [0.6620, 0.5470], [0.5690, 0.5470]]},
            {"row": 2, "col": 0, "skip": False, "poly": [[0.2694, 0.5519], [0.4076, 0.5519], [0.4076, 0.6101], [0.2694, 0.6101]]},
            {"row": 2, "col": 1, "skip": False, "poly": [[0.4166, 0.5519], [0.5614, 0.5519], [0.5614, 0.6101], [0.4166, 0.6101]]},
            {"row": 2, "col": 2, "skip": False, "poly": [[0.5690, 0.5519], [0.6620, 0.5519], [0.6620, 0.6101], [0.5690, 0.6101]]},
            {"row": 3, "col": 0, "skip": False, "poly": [[0.2694, 0.6139], [0.4076, 0.6139], [0.4076, 0.6721], [0.2694, 0.6721]]},
            {"row": 3, "col": 1, "skip": False, "poly": [[0.4166, 0.6139], [0.5614, 0.6139], [0.5614, 0.6721], [0.4166, 0.6721]]},
            {"row": 3, "col": 2, "skip": False, "poly": [[0.5690, 0.6139], [0.6620, 0.6139], [0.6620, 0.6721], [0.5690, 0.6721]]},
            {"row": 4, "col": 0, "skip": False, "poly": [[0.2694, 0.6761], [0.4076, 0.6761], [0.4076, 0.7419], [0.2694, 0.7419]]},
            {"row": 4, "col": 1, "skip": False, "poly": [[0.4166, 0.6761], [0.5614, 0.6761], [0.5614, 0.7419], [0.4166, 0.7419]]},
            {"row": 4, "col": 2, "skip": False, "poly": [[0.5690, 0.6761], [0.6620, 0.6761], [0.6620, 0.7419], [0.5690, 0.7419]]},
            {"row": 5, "col": 0, "skip": False, "poly": [[0.2694, 0.7461], [0.4076, 0.7461], [0.4076, 0.8129], [0.2694, 0.8129]]},
            {"row": 5, "col": 1, "skip": False, "poly": [[0.4166, 0.7461], [0.5614, 0.7461], [0.5614, 0.8129], [0.4166, 0.8129]]},
            {"row": 5, "col": 2, "skip": False, "poly": [[0.5690, 0.7461], [0.6620, 0.7461], [0.6620, 0.8129], [0.5690, 0.8129]]},
            {"row": 6, "col": 0, "skip": False, "poly": [[0.2694, 0.8171], [0.4076, 0.8171], [0.4076, 0.8829], [0.2694, 0.8829]]},
            {"row": 6, "col": 1, "skip": False, "poly": [[0.4166, 0.8171], [0.5614, 0.8171], [0.5614, 0.8829], [0.4166, 0.8829]]},
            {"row": 6, "col": 2, "skip": False, "poly": [[0.5690, 0.8171], [0.6620, 0.8171], [0.6620, 0.8829], [0.5690, 0.8829]]},
        ],
    },
    "正常取餐.mp4": {
        "ref_frame": 6,
        # 数字变焦 + 人物遮挡频繁 → 对齐容差加大、状态确认更保守（宁可不判、不可错判）。
        # 右下 4 格（R6/R7×C2/C3）位于人物站位与阴影扫过区，标记为不判定（余 15 格全程监控）。
        # 该素材为**数字变焦**（屏幕中心近乎不动、尺寸 60→103px 单向放大），
        # 以操作屏为锚的相似变换即精确模型；屏幕不可见/半遮挡 → 跳过该帧判定。
        "screen_anchor": True,
        "stab_roi": [0.24, 0.28, 0.80, 0.90],
        "base_candidates": [2, 5, 8],
        "params": {
            "open_th": 0.70,
            "closed_th": 0.82,
            "occ_expand_runtime": 0.12,
            "confirm_frames": 6,
            "closed_frames": 6,
            "step_window_s": 0.8,
            "step_drop": 0.15,
            "door_left_open_s": 3.0,
            "search_pad": 28,
            "occ_expand": 0.15,
            "consensus_ncc": 0.72, "consensus_pad": 6,
            "retrieval_touch_s": 1.2,
            "retrieval_change_th": 0.62,
        },
        "cells": [
            {"row": 0, "col": 0, "skip": False, "poly": [[0.3390, 0.3719], [0.4640, 0.3719], [0.4640, 0.4931], [0.3390, 0.4931]]},
            {"row": 0, "col": 1, "skip": False, "poly": [[0.4717, 0.3719], [0.5883, 0.3719], [0.5883, 0.4931], [0.4717, 0.4931]]},
            {"row": 0, "col": 2, "skip": False, "poly": [[0.5960, 0.3719], [0.7229, 0.3719], [0.7229, 0.4931], [0.5960, 0.4931]]},
            {"row": 1, "col": 0, "skip": False, "poly": [[0.3390, 0.4984], [0.4640, 0.4984], [0.4640, 0.5436], [0.3390, 0.5436]]},
            {"row": 1, "col": 1, "skip": False, "poly": [[0.4717, 0.4984], [0.5883, 0.4984], [0.5883, 0.5436], [0.4717, 0.5436]]},
            {"row": 1, "col": 2, "skip": False, "poly": [[0.5960, 0.4984], [0.7229, 0.4984], [0.7229, 0.5436], [0.5960, 0.5436]]},
            {"row": 2, "col": 0, "skip": False, "poly": [[0.3390, 0.5465], [0.4640, 0.5465], [0.4640, 0.5935], [0.3390, 0.5935]]},
            {"row": 2, "col": 1, "skip": False, "poly": [[0.4717, 0.5465], [0.5883, 0.5465], [0.5883, 0.5935], [0.4717, 0.5935]]},
            {"row": 2, "col": 2, "skip": False, "poly": [[0.5960, 0.5465], [0.7229, 0.5465], [0.7229, 0.5935], [0.5960, 0.5935]]},
            {"row": 3, "col": 0, "skip": False, "poly": [[0.3390, 0.5965], [0.4640, 0.5965], [0.4640, 0.6435], [0.3390, 0.6435]]},
            {"row": 3, "col": 1, "skip": False, "poly": [[0.4717, 0.5965], [0.5883, 0.5965], [0.5883, 0.6435], [0.4717, 0.6435]]},
            {"row": 3, "col": 2, "skip": False, "poly": [[0.5960, 0.5965], [0.7229, 0.5965], [0.7229, 0.6435], [0.5960, 0.6435]]},
            {"row": 4, "col": 0, "skip": False, "poly": [[0.3390, 0.6465], [0.4640, 0.6465], [0.4640, 0.6935], [0.3390, 0.6935]]},
            {"row": 4, "col": 1, "skip": False, "poly": [[0.4717, 0.6465], [0.5883, 0.6465], [0.5883, 0.6935], [0.4717, 0.6935]]},
            {"row": 4, "col": 2, "skip": False, "poly": [[0.5960, 0.6465], [0.7229, 0.6465], [0.7229, 0.6935], [0.5960, 0.6935]]},
            {"row": 5, "col": 0, "skip": False, "poly": [[0.3390, 0.6965], [0.4640, 0.6965], [0.4640, 0.7435], [0.3390, 0.7435]]},
            {"row": 5, "col": 1, "skip": True, "poly": [[0.4717, 0.6965], [0.5883, 0.6965], [0.5883, 0.7435], [0.4717, 0.7435]]},
            {"row": 5, "col": 2, "skip": True, "poly": [[0.5960, 0.6965], [0.7229, 0.6965], [0.7229, 0.7435], [0.5960, 0.7435]]},
            {"row": 6, "col": 0, "skip": False, "poly": [[0.3390, 0.7463], [0.4640, 0.7463], [0.4640, 0.7887], [0.3390, 0.7887]]},
            {"row": 6, "col": 1, "skip": True, "poly": [[0.4717, 0.7463], [0.5883, 0.7463], [0.5883, 0.7887], [0.4717, 0.7887]]},
            {"row": 6, "col": 2, "skip": True, "poly": [[0.5960, 0.7463], [0.7229, 0.7463], [0.7229, 0.7887], [0.5960, 0.7887]]},
        ],
    },
    "异常取餐.mp4": {
        "ref_frame": 30,
        # 实测该素材机位静态（缩放 0.996~0.999、平移 ≤1.6px），关闭稳定化以提速
        "stabilize": False,
        "stab_roi": [0.10, 0.10, 0.78, 0.99],
        "base_candidates": [2, 5, 10, 30, 75, 120, 160, 200, 240],
        "params": {
            "open_th": 0.70,
            "closed_th": 0.82,
            "confirm_frames": 4,
            "closed_frames": 6,
            "step_window_s": 0.8,
            "step_drop": 0.15,
            "door_left_open_s": 3.0,
            "search_pad": 16,
            "occ_expand": 0.15,
            "consensus_ncc": 0.72, "consensus_pad": 6,
            "retrieval_touch_s": 1.2,
            "retrieval_change_th": 0.62,
        },
        "cells": [
            {"row": 0, "col": 0, "skip": False, "poly": [[0.3319, 0.3753], [0.4525, 0.3753], [0.4525, 0.4333], [0.3319, 0.4333]]},
            {"row": 0, "col": 1, "skip": False, "poly": [[0.4597, 0.3753], [0.5637, 0.3753], [0.5637, 0.4333], [0.4597, 0.4333]]},
            {"row": 0, "col": 2, "skip": False, "poly": [[0.5701, 0.3753], [0.6689, 0.3753], [0.6689, 0.4333], [0.5701, 0.4333]]},
            {"row": 1, "col": 0, "skip": False, "poly": [[0.3319, 0.4373], [0.4525, 0.4373], [0.4525, 0.5018], [0.3319, 0.5018]]},
            {"row": 1, "col": 1, "skip": True, "poly": [[0.4597, 0.4373], [0.5637, 0.4373], [0.5637, 0.5018], [0.4597, 0.5018]]},
            {"row": 1, "col": 2, "skip": False, "poly": [[0.5701, 0.4373], [0.6689, 0.4373], [0.6689, 0.5018], [0.5701, 0.5018]]},
            {"row": 2, "col": 0, "skip": False, "poly": [[0.3319, 0.5053], [0.4525, 0.5053], [0.4525, 0.5494], [0.3319, 0.5494]]},
            {"row": 2, "col": 1, "skip": False, "poly": [[0.4597, 0.5053], [0.5637, 0.5053], [0.5637, 0.5494], [0.4597, 0.5494]]},
            {"row": 2, "col": 2, "skip": False, "poly": [[0.5701, 0.5053], [0.6689, 0.5053], [0.6689, 0.5494], [0.5701, 0.5494]]},
            {"row": 3, "col": 0, "skip": False, "poly": [[0.3319, 0.5522], [0.4525, 0.5522], [0.4525, 0.5963], [0.3319, 0.5963]]},
            {"row": 3, "col": 1, "skip": False, "poly": [[0.4597, 0.5522], [0.5637, 0.5522], [0.5637, 0.5963], [0.4597, 0.5963]]},
            {"row": 3, "col": 2, "skip": False, "poly": [[0.5701, 0.5522], [0.6689, 0.5522], [0.6689, 0.5963], [0.5701, 0.5963]]},
            {"row": 4, "col": 0, "skip": False, "poly": [[0.3319, 0.5991], [0.4525, 0.5991], [0.4525, 0.6439], [0.3319, 0.6439]]},
            {"row": 4, "col": 1, "skip": False, "poly": [[0.4597, 0.5991], [0.5637, 0.5991], [0.5637, 0.6439], [0.4597, 0.6439]]},
            {"row": 4, "col": 2, "skip": False, "poly": [[0.5701, 0.5991], [0.6689, 0.5991], [0.6689, 0.6439], [0.5701, 0.6439]]},
            {"row": 5, "col": 0, "skip": False, "poly": [[0.3319, 0.6467], [0.4525, 0.6467], [0.4525, 0.6908], [0.3319, 0.6908]]},
            {"row": 5, "col": 1, "skip": False, "poly": [[0.4597, 0.6467], [0.5637, 0.6467], [0.5637, 0.6908], [0.4597, 0.6908]]},
            {"row": 5, "col": 2, "skip": False, "poly": [[0.5701, 0.6467], [0.6689, 0.6467], [0.6689, 0.6908], [0.5701, 0.6908]]},
            {"row": 6, "col": 0, "skip": False, "poly": [[0.3319, 0.6936], [0.4525, 0.6936], [0.4525, 0.7377], [0.3319, 0.7377]]},
            {"row": 6, "col": 1, "skip": False, "poly": [[0.4597, 0.6936], [0.5637, 0.6936], [0.5637, 0.7377], [0.4597, 0.7377]]},
            {"row": 6, "col": 2, "skip": False, "poly": [[0.5701, 0.6936], [0.6689, 0.6936], [0.6689, 0.7377], [0.5701, 0.7377]]},
            {"row": 7, "col": 0, "skip": False, "poly": [[0.3319, 0.7403], [0.4525, 0.7403], [0.4525, 0.7777], [0.3319, 0.7777]]},
            {"row": 7, "col": 1, "skip": False, "poly": [[0.4597, 0.7403], [0.5637, 0.7403], [0.5637, 0.7777], [0.4597, 0.7777]]},
            {"row": 7, "col": 2, "skip": False, "poly": [[0.5701, 0.7403], [0.6689, 0.7403], [0.6689, 0.7777], [0.5701, 0.7777]]},
        ],
    },
}


# 演示素材的取餐区（归一化坐标，上边为柜体底部地面线，下边为画面底边）。
# 标定方式：在 3 段素材的抽样帧上人工目视框定「柜前站立地面」。
# 横向范围按 letterbox 后的**画面内容**收窄：720x1280 等比缩放到 640x480 画布后，
# 内容仅占横向 0.289~0.711（270x480 居中），若沿用 0~1 会把多边形画到黑边上。
# 素材镜头在拍摄中存在平移/推拉（实测 `未关柜门` 平移最大 (12,22) px、
# `正常取餐` 最大 (93,24) px），故该多边形为**粗标定**，只用于定性演示，
# 不用于任何量化结论。
DEMO_ROI_PICKUP = [
    [0.330, 0.790],
    [0.670, 0.790],
    [0.712, 1.000],
    [0.288, 1.000],
]


def is_ai_demo_source(path) -> bool:
    """
    判断素材路径是否位于 `real_video/`（AI 生成演示视频目录）下。

    素材路径既可能是绝对路径，也可能是相对项目根目录的路径（前端下拉给的就是后者），
    因此两种解释都试一遍，避免因工作目录不同而漏判。
    """
    if not path:
        return False
    try:
        p = Path(str(path))
        cands = [p] if p.is_absolute() else [p, BASE_DIR / p]
        root = REAL_DEMO_DIR.resolve()
    except OSError:
        return False
    for c in cands:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r == root or root in r.parents:
            return True
    return False


# ---------------------------------------------------------------- 事件字典
# key -> (中文名, 严重级别)
# 注：「长时间停留」已按需求方要求**整体移除**（2026-10-07）：
#     饭点等餐、驻足查手机等长驻足行为在现实中属正常情况，不作为异常报警，
#     其对应数据集素材（long_dwell_*.mp4，21 段）已一并移除。
EVENT_CATALOG = {
    "ROI_INTRUSION":          ("进入取餐区",       "info"),
    "DOOR_CLOSED":            ("柜门已关",         "info"),
    "LOITERING":              ("徘徊观察",         "warning"),
    "DOOR_LEFT_OPEN":         ("柜门未关",         "warning"),
    "MULTI_RETRIEVAL":        ("一人取多份",       "critical"),
    "FLAGGED_DOOR_RETRIEVAL": ("取走未关柜门外卖", "critical"),
}
EVENT_NAME = {k: v[0] for k, v in EVENT_CATALOG.items()}
EVENT_SEVERITY = {k: v[1] for k, v in EVENT_CATALOG.items()}
# 判定为“异常”的事件类型（用于检出率/误报率统计）
ABNORMAL_EVENTS = {"LOITERING", "DOOR_LEFT_OPEN",
                   "MULTI_RETRIEVAL", "FLAGGED_DOOR_RETRIEVAL"}


# ---------------------------------------------------------------- 运行设置
@dataclass
class Settings:
    """运行时设置。命令行参数覆盖默认值，可序列化写入 logs/run_config.json 以便实验复现。"""

    host: str = "127.0.0.1"
    port: int = 5000

    source: str = ""                    # 视频文件 / 图片序列目录；留空自动挑选 data 下第一个可用视频
    loop_playback: bool = True          # 播放结束后自动重播
    display_fps: float = 30.0           # 显示/推流帧率上限（实际取值不会超过源视频帧率）
    detect_fps: float = 15.0            # 检测回路帧率上限；推理跟不上时自动降频保显示流畅
    autostart: bool = True              # 启动 Web 服务后自动开始处理视频

    backend: str = "auto"               # auto | yolov8n | hog
    model_path: str = str(MODEL_DIR / "yolov8n.pt")
    device: str = "cpu"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.50
    imgsz: int = 480                    # 推理输入尺寸；480 实测约 45ms/帧，可支撑 15fps 源逐帧检测
    torch_threads: int = 0              # 0 = 自动取物理核心数
    hog_resize: int = 320               # HOG 推理前缩放宽度，用于控制 CPU 耗时

    db_path: str = str(DB_PATH)
    snapshot_dir: str = str(SNAP_DIR)
    device_id: str = "edge-01"
    device_name: str = "边缘节点-01（本机 CPU）"
    server: str = "flask"               # flask | waitress

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: Optional[Path] = None) -> Path:
        path = path or (LOG_DIR / "run_config.json")
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------- 日志
def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """统一日志配置：控制台 + 文件（UTF-8）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = log_file or (LOG_DIR / "tianyan.log")

    root = logging.getLogger()
    if root.handlers:                      # 避免重复添加 handler
        return logging.getLogger("tianyan")

    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    try:                                   # Windows 控制台默认 GBK，中文日志可能报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("tianyan")


def resolve_source(raw: str) -> str:
    """
    解析视频源。
    严格禁止摄像头输入（本项目不依赖任何外部采集硬件），
    因此纯数字（摄像头索引）或常见设备字符串一律拒绝。
    """
    s = (raw or "").strip()
    if not s:
        return pick_default_source()
    if s.isdigit():
        raise ValueError("禁止使用摄像头输入（索引 %s）。请提供 mp4/avi 视频文件或图片序列目录。" % s)
    if s.lower() in ("camera", "cam", "webcam", "device", "0"):
        raise ValueError("禁止使用摄像头输入（%s）。请提供 mp4/avi 视频文件或图片序列目录。" % s)
    return s


def pick_default_source() -> str:
    """
    按优先级挑选默认素材：
        AI 生成演示素材 real_video/正常取餐.mp4（需求方指定为网页首屏演示素材）
        -> 链路自检演示素材 data/selftest/selftest_scene.mp4
        -> 阶段2 模拟视频 data/simulated/
        -> 公开数据集 data/videos/
        -> 其它自检素材 data/selftest/

    首屏素材按需求方要求改为 `real_video/正常取餐.mp4`（界面会同步显示「AI 生成演示视频·非实拍」
    声明，并按演示档案运行）。该素材不存在时自动回退到自检演示素材，
    保证无参启动在任何机器上都拿得到一段「正常取餐」完整时间线。
    """
    exts = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v")
    demo_ai = REAL_DEMO_DIR / "正常取餐.mp4"
    if demo_ai.exists():
        return str(demo_ai)
    demo = BASE_DIR / "data" / "selftest" / "selftest_scene.mp4"
    if demo.exists():
        return str(demo)
    for d in (SIM_DIR, VIDEO_DIR, BASE_DIR / "data" / "selftest"):
        if not d.exists():
            continue
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in exts])
        if files:
            return str(files[0])
        subdirs = sorted([p for p in d.iterdir() if p.is_dir()])
        for sd in subdirs:                       # 数据集常见：data/videos/UCF-Crime/xxx.mp4
            f2 = sorted([p for p in sd.rglob("*") if p.suffix.lower() in exts])
            if f2:
                return str(f2[0])
    raise FileNotFoundError(
        "未找到可用视频素材。请把 mp4/avi 放入 data/videos/ 或 data/simulated/，"
        "或使用 --source 指定路径。"
    )
