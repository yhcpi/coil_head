#!/bin/bash
# Coverage smoke（验证 loss.py 修复后 Coverage loss 量级是否合理）
# 关键前置：loss.py 修复 + IoU baseline smoke 通过
# 修复前 Coverage box_loss=370+（爆炸），修复后期望 ~8-10（健康量级）
# 用法: bash scripts/run_v8_coverage_smoke.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_coverage_smoke_5ep"

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
  cfg=data/coil/hyp_v8_coil_coverage.yaml \
  2>&1 | tail -10

echo "===== epoch 收敛轨迹 ====="
awk -F',' 'NR>1 {printf "epoch %d  box=%s  cls=%s  dfl=%s  P=%s  R=%s  mAP50=%s\n", $1, $2, $3, $4, $8, $9, $10}' \
  runs/v8_smoke/${NAME}/results.csv
