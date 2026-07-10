#!/bin/bash
# v8 NWD v1 纯 CLI 250 epoch full — save_period=10 保住 best.pt
# 与 v8_nwd_v1_pure_cli_full 完全相同，只改 save=True + save_period=10
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_v1_pure_cli_full_save"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=1 start_val_epoch=0 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  nwd=true nwd_constant=12.0 \
  coverage=false coverage_weight=0.5 \
  degrees=0.0 translate=0.1 scale=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log