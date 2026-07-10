#!/bin/bash
# v8 NWD+Coverage 250 epoch full 训练（loss.py 修复后）
# 关键前置：NWD full + Coverage full 都训完，确认无回归
# 用法: bash scripts/run_v8_nwd_coverage_full.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_coverage_full"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=True val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_loss_ablation name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd_coverage.yaml \
  2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}.log
echo "$(date) NWD+Coverage full 训练完成"

awk -F',' 'NR>1 && NR%25==1 {printf "epoch %d  box=%s  cls=%s  mAP50=%s\n", $1, $2, $3, $11}' \
  /home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}/results.csv 2>/dev/null
