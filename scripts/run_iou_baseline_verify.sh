#!/bin/bash
# IoU baseline smoke（验证 loss.py 修改没破坏 IoU 分支）
# 关键：D3 验证历史 box_loss=4.2436，本脚本应能复现
# 用法: bash scripts/run_iou_baseline_verify.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_iou_baseline_verify"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=1 patience=0 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v8_verify name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=False seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=false coverage=false \
  nbs=64 lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 pose=12.0 kobj=1.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  iou=0.7 max_det=300 conf=0.001 plots=False \
  2>&1 | tail -3

echo "===== box_loss epoch 1 ====="
awk -F',' 'NR>1 {printf "box=%s  cls=%s\n", $2, $3}' \
  runs/v8_verify/${NAME}/results.csv 2>/dev/null
