#!/bin/bash
# NWD full 250 epoch 用 T1 cfg（强 aug + cls=0.5）
# 已知：T1 50 epoch best mAP50=0.0528 (vs v1 弱 aug 同 epoch ≈ 0.55)
# 期望：250 epoch 强 aug 长期泛化能追平/超 v1 0.894
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_full_T1_cls05"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_loss_ablation name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd_tune_cls.yaml \
  2>&1 | tail -10