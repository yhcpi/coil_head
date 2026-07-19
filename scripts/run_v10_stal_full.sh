#!/bin/bash
# v10 STAL full 训练 — 250 epoch + 期望 TTA 后处理
# baseline: hyp_v8_coil_nwd.yaml (weak_aug + NWD, F1=0.929)
# 唯一变量: stal_area_thr=400 / stal_topk=13 / stal_expand=0.2 (通过 cfg= yaml 注入)
# 期望: recall 0.974 → 1.00 (push F1 higher); mAP50 持平或微升
set -u
cd /home/pi/projects/hyperyolo

NAME="v10_stal_full"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  cfg=data/coil/hyp_v10_stal.yaml \
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