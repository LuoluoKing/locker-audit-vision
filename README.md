# Auditable Anomaly Detection for Unattended Delivery Lockers

Code, configuration, calibration/audit tools, experiment drivers and evaluation logs
accompanying the manuscript:

> **Auditable Anomaly Detection for Unattended Delivery Lockers:
> A Calibrated Rule Pipeline with Field-Footage Boundaries and a Pre-Deployment Audit**
> (submitted to *Applied Sciences*, MDPI)

中文说明见本文末尾。

---

## 1. What the system does

A CPU-only, fully local, open-source monitoring layer for grid-structured self-service
cabinets (campus food-delivery lockers). It estimates each compartment's open/closed state
from calibrated image regions, tracks users, and raises six event types through
per-compartment rules whose every threshold is a **named parameter**, so each alarm names a
compartment, a measured quantity and a threshold and can be audited after the fact.

* **Input**: local video files only. No camera, no board, no external sensor.
* **Inference**: Ultralytics **YOLOv8n** on CPU; falls back automatically to OpenCV
  **HOG + linear SVM** if the deep model or weights are unavailable.
* **Service**: Flask MJPEG stream + status panel + event query; SQLite event store.

## 2. Repository layout

```
code/                     system source (app.py, detector.py, rules.py, video_worker.py, database.py, config.py)
tools/                    calibration, audit, experiment and evaluation scripts
  ├─ calib_locker_compartments.py / calib_locker_door_threshold.py / calib_field_grid.py
  ├─ field_color_audit.py          colour-collision audit
  ├─ field_door_probe.py           door-state (darkness-ratio) probe
  ├─ field_study.py                reproduces the field-study logs (see §4)
  ├─ field_event_eval.py           event-level annotation worksheet + scoring
  ├─ eval_field.py                 per-recording event timelines (calibrated/uncalibrated)
  ├─ eval_dataset.py               rendered-benchmark evaluation (--config accepts a geometry JSON)
  ├─ exp_variant_dataset.py        second, cross-scene rendered set + geometric recalibration
  ├─ analytic_threshold_sensitivity.py / wilson_ci.py
  └─ gen_behavior_clip3d.py, gen_selftest_clip3d.py, scene3d.py, scene_kit.py  (scene generator)
logs/                     evaluation logs (JSON) from which the paper's numbers are derived
data/simulated/           rendered-benchmark labels (labels.csv + per-clip truth CSVs)
data/simulated_v2/        second, cross-scene rendered set: labels.csv
paper/applsci_mdpi/       manuscript LaTeX source, figures, build/main.pdf
```

## 3. What is **not** included (and why)

* **Third-party field recordings** (`EXAMPLE/`) — publicly available videos published on
  video-sharing platforms, used strictly as analysis input and **not redistributed**; the
  paper's Data Availability lists their published titles.
* **Frames / snapshots derived from those recordings** — same reason.
* **Model weights** (`models/*.pt`) — download `yolov8n.pt` from Ultralytics.
* **Public datasets** (COCO val2017, CUHK Avenue) — obtainable from their distributors.
* **Rendered video files** (`data/simulated/**/*.mp4`) — regenerable with the scene
  generator in `tools/`.
* Virtual environments, build intermediates, internal project documents.

## 4. Reproducing the field-study measurements

The field-study logs in `logs/` can be recomputed from the analysis products of the four
tools above:

```bash
# corpus survey + per-recording event timelines + audits (need the recordings in EXAMPLE/)
python tools/analyze_field_videos.py                 # -> logs/field_survey.json
python tools/eval_field.py --all --mode default      # -> logs/field_eval/v*.json
python tools/field_color_audit.py                    # -> logs/field_color_audit.json

# unified recomputation + field-by-field check against the stored logs
python tools/field_study.py --check                  # summary: exact; video-derived: see note
```

`tools/field_study.py` defaults to `--check` (**recompute only, does not overwrite**); `--all`
overwrites the stored logs and should be used deliberately. Video-derived series reproduce to
within a small decoder-seek tolerance (a few samples differ by ±0.005); the paper quotes the
**original** run's values.

Audit closed loop (Section 7.3.2 of the paper):

```bash
python tools/eval_field.py --video v04 --mode file:logs/field_calib/v04.json --eta 0.55 --tag cal055
```

Threshold intervals: `python tools/wilson_ci.py` recomputes every Wilson 95% interval and
the door-segment mean interval reported in the paper, reading only `logs/*.json`.

## 5. Environment

Python 3.11, OpenCV, NumPy, PyTorch 2.14.0+cpu, Ultralytics 8.4.161, Flask 3.1.3
(`code/requirements.txt`). All components are free and open-source.

---

## 中文说明

本仓库是论文《Auditable Anomaly Detection for Unattended Delivery Lockers》(Applied
Sciences 投稿) 的配套代码与日志：**系统源码**（`code/`）、**标定/审计/实验/评估脚本**
（`tools/`）、**评测日志**（`logs/*.json`）、**渲染基准标签**（`data/simulated/`、
`data/simulated_v2/`）与**论文 LaTeX 源件**（`paper/applsci_mdpi/`）。

**未包含**：第三方公开录像及其派生帧（论文承诺仅作分析输入、不再分发）、模型权重、
公开数据集（COCO / CUHK Avenue）、可重新生成的渲染视频、虚拟环境与内部文档。

现场研究日志的复现在 §4；`tools/field_study.py --check` 会逐字段回算并与既有日志比对
（默认只算不写盘）。论文中的比例区间可用 `tools/wilson_ci.py` 一键复算。
