#!/usr/bin/env bash
# 2026-07-16 v21: v20 (DFL-off) + v18.3 (HN crop) 三合一 250 ep
# 假设: 学术 mAP50 + 部署 F1 双赢
# 起点: baseline best.pt (mAP50=0.8888)
# 数据: 545 + 33 (HN crop) = 578 张
set -uo pipefail
cd /home/pi/projects/hyperyolo

PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt \
  data=data/coil/data.yaml \
  dfl=0.0 box=1.5 cls=0.5 \
  epochs=250 patience=80 \
  lr0=0.005 lrf=0.0005 \
  imgsz=1024 batch=8 device=0 \
  degrees=0.0 scale=0.0 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 \
  name=v21_dfl_off_hn_250ep \
  exist_ok=True \
  project=runs/dfl_off

echo "✅ v21 full 启动 at $(date '+%H:%M:%S')"
ps -ef | grep "yolo detect train" | grep -v grep | head -1 | awk '{print "PID:", $2}'