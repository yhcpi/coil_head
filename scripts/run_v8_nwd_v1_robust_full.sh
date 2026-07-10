#!/bin/bash
# v8 NWD v1 250 epoch full — 加入多尺度鲁棒性增强
# 相比 v8_nwd_v1_pure_cli_full 的改动：
#   save=True, save_period=10    (保住 best.pt)
#   multi_scale=0.25             (以后输入尺寸变化时鲁棒)
#   rect=False                   (避免 multi_scale collate 崩溃)
#   flipud=0.5, degrees=10.0     (用户要求的鲁棒性增强)
#   copy_paste=0.2               (用户要求的鲁棒性增强)
# nwd_constant=12.0 保持 (后续 30ep A/B 验证 6/12/20/30)
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_v1_robust_full"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=1 start_val_epoch=0 \
  cache=False device=0 workers=0 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=False cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  nwd=true nwd_constant=12.0 \
  coverage=false coverage_weight=0.5 \
  degrees=10.0 translate=0.1 scale=0.5 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.25 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log