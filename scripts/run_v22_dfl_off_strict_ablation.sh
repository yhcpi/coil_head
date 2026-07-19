#!/usr/bin/env bash
# 2026-07-16 v22: DFL-off 严格消融（与 baseline 唯一变量对照）
# baseline 配置: hyper-yolon.pt + 强 aug + dfl=on + 250 ep → mAP50=0.8888
# v22 配置:     hyper-yolon.pt + 强 aug + dfl=off + 250 ep ← 唯一变量 = dfl off
# 目的: 验证 DFL-off 单独贡献 (排除起点/aug/隐含变量干扰)
set -uo pipefail
cd /home/pi/projects/hyperyolo

PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  dfl=0.0 box=1.5 cls=0.5 \
  epochs=250 patience=80 \
  lr0=0.01 lrf=0.0001 \
  imgsz=1024 batch=8 device=0 \
  close_mosaic=15 warmup_epochs=3.0 \
  bbox_noise_scale=0.8,1.2 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=10.0 scale=0.5 \
  flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  name=v22_dfl_off_strict_ablation_250ep \
  exist_ok=True \
  project=runs/dfl_off

echo "✅ v22 严格消融启动 at $(date '+%H:%M:%S')"
ps -ef | grep "yolo detect train" | grep -v grep | head -1 | awk '{print "PID:", $2}'