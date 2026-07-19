#!/usr/bin/env bash
# 2026-07-15 H21:18 smoke B 假设: rect=False 比 rect=True 更合适
# (rect=True 对 2560x1440 -> 1024x576 letterbox 压缩可能让目标过小)
set -e
cd /home/pi/projects/hyperyolo

NAME="smoke_rect_off_30ep"

echo "===== smoke B ${NAME} ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/ultralytics/ultralytics/cfg/models/coil_exp/yolov8n_baseline.yaml \
  data=data/coil/data.yaml \
  epochs=30 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/smoke name="${NAME}" exist_ok=True \
  pretrained=True cls_remap=False single_cls=True optimizer=SGD verbose=True seed=0 deterministic=True \
  rect=False cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  degrees=10.0 translate=0.1 scale=0.5 shear=0.0 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
test -f runs/smoke/${NAME}/results.csv && \
  awk -F, 'NR>1 {printf "ep=%-3d box=%-7.4f cls=%-7.4f mAP50=%-7.4f P=%-6.3f R=%-6.3f\n",$1,$3,$4,$8,$6,$7}' \
  runs/smoke/${NAME}/results.csv | tail -10
