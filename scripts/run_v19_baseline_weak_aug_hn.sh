#!/usr/bin/env bash
# 2026-07-16 H04:00 v19 = baseline best.pt + v18.3 范式 (weak aug + lr=0.005 + hard neg crop)
# baseline 学术 mAP50=0.8888 (新 SOTA), 但部署 F1 0.8736 输 v18.3 0.9286 (强 aug 让 FP 多)
# v18.3 范式: 弱 aug (degrees=0/scale=0/flipud=0/cp=0) + hard neg crop → 针对性消除 FP
# 关键: baseline 已含 11 张 hard neg (data/coil/images/train/hn{1,2,3}_*.png × 3 副本)
set -e
cd /home/pi/projects/hyperyolo

NAME="v19_baseline_weak_aug_hn_100ep"

echo "===== 启动 ${NAME} ====="
echo "pretrain from: runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt (mAP50=0.8888)"
echo "范式: 弱 aug + lr=0.005 + 100ep + hard neg 已在 data"
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt \
  data=data/coil/data.yaml \
  epochs=100 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val=True \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.005 lrf=0.0005 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.05 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=True nwd_constant=12.0 \
  coverage=False coverage_weight=0.5 coverage_sigma=20.0 \
  degrees=0.0 translate=0.05 scale=0.0 shear=0.0 flipud=0.0 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
test -f runs/cfg_truth_repro/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|lr0|lrf|warmup|box|cls|nwd|coverage|degrees|translate|scale|flipud|mosaic|copy_paste|multi_scale|pretrained|seed):" \
  runs/cfg_truth_repro/${NAME}/args.yaml