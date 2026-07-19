#!/bin/bash
# v9 NBBoxNoise full 训练 — 250 epoch + TTA 后处理目标
# baseline: hyp_v8_coil_nwd_v1_repro.yaml (weak_aug + NWD, F1=0.929)
# 唯一变量: bbox_noise=true (训练期 GT bbox 加 [0.5,1.5] scale + ±0.1 shift 噪声)
set -u
cd /home/pi/projects/hyperyolo

NAME="v9_nbbox_noise_full"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  cfg=data/coil/hyp_v9_nbbox_noise.yaml \
  epochs=250 patience=80 batch=16 imgsz=1024 \
  save=True save_period=20 val_period=1 start_val_epoch=0 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log