#!/bin/bash
# v8 NWD v1 30 epoch smoke — 验证 albumentations 装上后 mAP50 涨速是否回到历史水平
# 关键对比点：
#   - 历史 v8_nwd_full ep 5: mAP50=0.042（缓慢涨）
#   - 这次（无 alb）v8_nwd_v1_repro ep 5: mAP50=0.159（涨得过快）
#   - 装上 alb 后预期：mAP50 ep 5 ≈ 0.04（涨速回归正常）
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_v1_smoke_with_alb"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=30 patience=0 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd_v1_repro.yaml \
  2>&1 | tee /tmp/${NAME}.log
