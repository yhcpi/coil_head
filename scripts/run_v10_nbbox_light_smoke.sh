#!/bin/bash
# v10 NBBoxNoise 轻量版 smoke — 10 epoch 验证 transform 不爆
# baseline: hyp_v8_coil_nwd_v1_repro.yaml (weak_aug + NWD)
# 唯一变量: bbox_noise=true (scale/shift/p 由 default.yaml 注入轻量值)
set -u
cd /home/pi/projects/hyperyolo

NAME="v10_nbbox_light_smoke"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v10 nbbox-light smoke (10 epoch) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=hyper-yolon.yaml \
  data=data/coil/data.yaml \
  epochs=10 patience=0 batch=16 imgsz=1024 \
  save=False val_period=2 start_val_epoch=2 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v10_nbbox_light name="${NAME}" exist_ok=True \
  pretrained=False optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v10_nbbox_light.yaml \
  2>&1 | tail -30

echo ""
echo "===== smoke 结果 ====="
test -f runs/v10_nbbox_light/v10_nbbox_light_smoke/results.csv && \
  awk -F, 'NR==1{next} NR>=3 {printf "ep%s box=%s P=%s R=%s mAP50=%s\n", $1, $2, $9, $10, $11}' runs/v10_nbbox_light/v10_nbbox_light_smoke/results.csv | head -10 || \
  echo "(no results.csv yet)"