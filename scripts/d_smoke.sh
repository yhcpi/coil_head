#!/bin/bash
# D-1 smoke test：paaug=motion 5 epoch 验证模块链路无报错
set -e
cd /home/pi/projects/hyperyolo
echo "==== D smoke test (paaug=motion, 5 epoch) 启动: $(date) ===="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=5 patience=5 batch=16 imgsz=1024 \
  save=True val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_loss_ablation name=11_paaug_motion_smoke exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 cfg=data/coil/hyp_v7_paaug_motion.yaml \
  box=5.0 cls=1.0 dfl=1.5 label_smoothing=0.02 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=5.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0005 \
  flipud=0.5 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  iou=0.7 max_det=300 conf=0.001 plots=True \
  2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_loss_ablation/11_paaug_motion_smoke.log
echo "==== D smoke test 完成: $(date) ===="