#!/bin/bash
# NWD smoke v3 — 用修好的 cfg（aug/loss 字段与 v4 baseline 一致，box=1.5 是 NWD 替换 IoU 必须降）
# 关键差异 vs 之前的 v8 NWD full：v8 NWD full (mAP50=0.894) 用的是 box=1.5/cls=0.5/copy_paste=0 等弱化版 cfg，
# 现重跑用 v4 baseline 的强化 aug/loss，看是否能追平 v4 Coverage (mAP50=0.877)
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_smoke_v3_30ep"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=30 patience=0 batch=16 imgsz=1024 \
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