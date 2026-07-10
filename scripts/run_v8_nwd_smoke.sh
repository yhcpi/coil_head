#!/bin/bash
# v8 NWD 调参实验 smoke（1 epoch 看 box_loss 量级）
# 目的：NWD 替换 IoU 后，box_gain=1.5 / nwd_constant=12.0 配置下 box_loss 是否合理
# 数据：train=556, val=102
#
# 用法: bash scripts/run_v8_nwd_smoke.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_smoke_g1.5"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=1 patience=0 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v8_smoke name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd.yaml \
  2>&1 | tail -30

echo "===== args.yaml 实际生效字段 ====="
grep -E "^(box|cls|nwd|coverage|nwd_constant|copy_paste|degrees|mosaic|cfg):" \
  runs/v8_smoke/${NAME}/args.yaml

echo "===== results.csv epoch 1 ====="
head -2 runs/v8_smoke/${NAME}/results.csv
