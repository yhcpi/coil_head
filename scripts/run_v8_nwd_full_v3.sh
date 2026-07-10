#!/bin/bash
# NWD 250 epoch full（修后 cfg：box=1.5/cls=1/copy_paste=0.2 等与 v4 baseline 一致）
# vs 之前的 v8_nwd_full（cfg 错：box=1.5/cls=0.5/copy_paste=0）mAP50=0.894
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_full_v3"

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
  cfg=data/coil/hyp_v8_coil_nwd.yaml \
  2>&1 | tail -10