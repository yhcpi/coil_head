#!/bin/bash
# NBBox Aug v1 训练 - smoke 30 epoch (验证 box_loss 不崩)
# baseline: 03_nwd_v5B_N_model (NWD-only)
# 单变量：cfg=hyp_v5_nwd_only.yaml 已加 nbbox 段（其余字段 100% 继承 03）
# 用法: bash scripts/run_nbbox_v1.sh [phase: smoke|full]
#  - smoke: 30 epoch (~9 分钟) — 验证 box_loss 不爆
#  - full:  250 epoch (~77 分钟) — 实际实验
set -u
PHASE="${1:-smoke}"

case "$PHASE" in
  smoke)
    EPOCHS=30
    PATIENCE=10
    NAME="12b_nbbox_aug_v1_smoke_fix5px"
    ;;
  full)
    EPOCHS=250
    PATIENCE=50
    NAME="12b_nbbox_aug_v1"
    ;;
  *)
    echo "phase 必须 smoke 或 full"; exit 1 ;;
esac

set -u
cd /home/pi/projects/hyperyolo

# WSL2 SIGABRT 风险对策：setsid + nohup + 单 worker + plots=false 已用过 (reflection v5 验证)
# 但 NBBox 是数据 transform 端，不增加 GPU 压力；保留默认 plots=True

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs="${EPOCHS}" patience="${PATIENCE}" batch=16 imgsz=1024 \
  save=True val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_loss_ablation name="${NAME}" exist_ok=True \
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
  cfg=data/coil/hyp_v5_nwd_only.yaml \
  2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}.log
echo "$(date) ${PHASE} 训练完成"
