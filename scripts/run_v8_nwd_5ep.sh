#!/bin/bash
# v8 NWD 调参实验：5 epoch smoke
# 目的：box_gain=1.5 + nwd=true 时 box_loss 收敛轨迹是否健康（应降到 < 10）
# 数据：train=556, val=102
#
# 用法: bash scripts/run_v8_nwd_5ep.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_smoke_5ep_g1.5"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=5 patience=0 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/v8_smoke name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd.yaml \
  2>&1 | tail -10

echo "===== epoch 收敛轨迹 ====="
awk -F',' 'NR>1 {printf "epoch %d  box=%s  cls=%s  dfl=%s  P=%s  R=%s  mAP50=%s\n", $1, $2, $3, $4, $8, $9, $10}' \
  runs/v8_smoke/${NAME}/results.csv
