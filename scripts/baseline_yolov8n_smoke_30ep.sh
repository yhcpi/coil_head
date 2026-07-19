#!/usr/bin/env bash
# 2026-07-15 H21:05 quick smoke 验证 A 假设:
# + single_cls=True + yolov8n.pt + 强 aug + 30 epoch
# 预期: ep10 mAP50 > 0.1 否则 A 路线不对.
set -e
cd /home/pi/projects/hyperyolo

NAME="smoke_baseline_a_30ep"

echo "===== smoke A ${NAME} ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/ultralytics/ultralytics/cfg/models/coil_exp/yolov8n_baseline.yaml \
  data=data/coil/data.yaml \
  epochs=30 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/smoke name="${NAME}" exist_ok=True \
  pretrained=True single_cls=True nwd=True nwd_constant=12.0 optimizer=SGD verbose=True seed=0 deterministic=True \
  rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  degrees=10.0 translate=0.1 scale=0.5 shear=0.0 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
test -f runs/smoke/${NAME}/results.csv && \
  awk -F, 'NR>1 {printf "ep=%-3d box=%-7.4f cls=%-7.4f mAP50=%-7.4f P=%-6.3f R=%-6.3f\n",$1,$3,$4,$8,$6,$7}' \
  runs/smoke/${NAME}/results.csv | tail -5
