#!/usr/bin/env bash
# 2026-07-19 v23: 新 baseline = 数据 v2 (train=642 全正, val=80 全正)
#         + hyper-yolon.pt + 强 aug + 250 ep
# 与 v22 基线唯一变量: 数据 (旧 545 → 新 642, 全正样本)
set -uo pipefail
cd /home/pi/projects/hyperyolo

PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  dfl=1.5 box=1.5 cls=0.5 \
  epochs=250 patience=80 \
  lr0=0.01 lrf=0.0001 \
  imgsz=1024 batch=8 device=0 \
  close_mosaic=15 warmup_epochs=3.0 \
  bbox_noise_scale=0.8,1.2 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=10.0 scale=0.5 \
  flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  name=v23_baseline_train642_strong_aug_250ep \
  exist_ok=True \
  project=runs/baseline_v2

echo "✅ v23 baseline (train=642 全正, 强 aug) 启动 at $(date '+%H:%M:%S')"
ps -ef | grep "yolo detect train" | grep -v grep | head -1 | awk '{print "PID:", $2}'