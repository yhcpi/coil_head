#!/bin/bash
# Coverage smoke v3 — 用修好的 cfg（aug/loss 字段与 v4 baseline 一致）
# 关键差异 vs v2：cfg 字段 box=5/cls=1/copy_paste=0.2/degrees=5/flipud=0.5/bbox_shrink 0.8-1.2/multi_scale=0.2
# 预期：模型能学到东西（mAP50 > 0），val_box_loss 健康下降
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_coverage_smoke_v3_30ep"

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
  cfg=data/coil/hyp_v8_coil_coverage.yaml \
  2>&1 | tail -10

echo "===== epoch 收敛轨迹 ====="
awk -F',' 'NR>1 {printf "epoch %d  box=%s  cls=%s  dfl=%s  P=%s  R=%s  mAP50=%s\n", $1, $2, $3, $4, $8, $9, $10}' \
  runs/v8_smoke/${NAME}/results.csv