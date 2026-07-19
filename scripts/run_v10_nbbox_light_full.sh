#!/bin/bash
# v10 NBBoxNoise 轻量版 full — 250 epoch（主线程启动, 此处 PID 留空）
# baseline: hyp_v8_coil_nwd_v1_repro.yaml (weak_aug + NWD)
# 唯一变量: bbox_noise=true (scale/shift/p 由 default.yaml 注入轻量值)
# 复现路径: v8 NWD weak (mAP50=0.869 + TTA F1=0.929) → v10 + 轻量 NBBoxNoise
set -u
cd /home/pi/projects/hyperyolo

NAME="v10_nbbox_light_full"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v10 nbbox-light full (250 epoch) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=hyper-yolon.yaml \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=10 start_val_epoch=10 \
  cache=False device=0 workers=0 \
  project=runs/v10_nbbox_light name="${NAME}" exist_ok=True \
  pretrained=False optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v10_nbbox_light.yaml \
  2>&1 | tee /tmp/${NAME}.log