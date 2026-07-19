#!/usr/bin/env bash
# 2026-07-16 v19 续训: 从 ep26 best.pt resume + 加大 patience + +60ep
# v19 ep31 被早停 (patience=30 触顶), 续训策略:
#   - 起点: v19 ep26 best.pt (mAP50=0.8698)
#   - epochs: 100 → 160 (从原 ep31 续到 ep91)
#   - patience: 30 → 80 (避免再早停)
#   - lr0: 0.005 → 0.003 (微调第二阶段更细)
#   - 验证: 同样弱 aug + HN + NWD
set -e
cd /home/pi/projects/hyperyolo

NAME="v19_baseline_weak_aug_hn_100ep_resume"

echo "===== 启动 ${NAME} ====="
echo "从 v19 ep26 best.pt resume, +60ep, patience=80"
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/cfg_truth_repro/v19_baseline_weak_aug_hn_100ep/weights/best.pt \
  data=data/coil/data.yaml \
  epochs=160 patience=80 batch=16 imgsz=1024 \
  save=True save_period=10 val=True \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.003 lrf=0.0003 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=2.0 warmup_momentum=0.8 warmup_bias_lr=0.05 \
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