#!/bin/bash
# Coverage smoke v2（验证 coverage_loss NaN 修复）
# 关键：30 epoch（vs 5 epoch）看修复后是否会再崩
# 修复前 epoch 5 后崩；修复后期望：30 epoch 不崩 + box_loss 持续下降
# 用法: bash scripts/run_v8_coverage_smoke_v2.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_coverage_smoke_v2_30ep"

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

echo "===== 是否崩了 ====="
LAST_BOX=$(awk -F',' 'NR>1 {print $2}' runs/v8_smoke/${NAME}/results.csv | tail -1)
if [ "$LAST_BOX" = "0" ] || [ -z "$LAST_BOX" ]; then
  echo "❌ 崩了！box_loss=0"
else
  echo "✓ 没崩，最后 box_loss=$LAST_BOX"
fi
