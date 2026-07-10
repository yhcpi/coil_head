#!/bin/bash
set -e
cd /home/pi/projects/hyperyolo
echo "==== smoke test: v4 cmd line + cfg=hyp_v6_bayes_prior.yaml 启动 5 epoch ===="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=5 patience=50 batch=16 imgsz=1024 \
  save=True val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/smoke_test name=v4cmd_v6cfg exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 cfg=data/coil/hyp_v6_bayes_prior.yaml \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  iou=0.7 max_det=300 conf=0.001 plots=False 2>&1 | tail -40
