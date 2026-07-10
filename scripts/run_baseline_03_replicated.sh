#!/bin/bash
# 03 baseline 复现命令（D3 验证 = 03_nwd_v5B_N_model 历史快照 box_loss=4.2436）
#
# 关键发现：
#   03/04 baseline 命名误导（nwd_only_v1 / nwd_coverage_v5C），实际是标准 IoU + box=7.5 cls=0.5
#   args.yaml 实际生效字段（不是 cfg 字段）：
#     box: 7.5, cls: 0.5, dfl: 1.5, label_smoothing: 0.0
#     nwd: false, coverage: false
#     copy_paste: 0.0, mosaic: 0.0, degrees: 0.0, perspective: 0.0, flipud: 0.0
#     bbox_shrink_min: 1.0（默认关闭）, multi_scale: 0.0
#
# 用法: bash scripts/run_baseline_03_replicated.sh [epoch=250]
set -u
EPOCHS="${1:-250}"
PATIENCE="${2:-50}"
NAME="${3:-03_baseline_replicated}"
cd /home/pi/projects/hyperyolo

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs="${EPOCHS}" patience="${PATIENCE}" batch=16 imgsz=1024 \
  save=True val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_v5 name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=false coverage=false \
  nbs=64 lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 pose=12.0 kobj=1.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  iou=0.7 max_det=300 conf=0.001 plots=True \
  2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_v5/${NAME}.log

echo "[$(date)] ${NAME} 训练完成"
awk -F',' 'NR>1 && NR%25==1 {printf "epoch %d  box=%s  cls=%s  mAP50=%s\n", $1, $2, $3, $11}' \
  /home/pi/projects/hyperyolo/runs/coil_v5/${NAME}/results.csv 2>/dev/null
