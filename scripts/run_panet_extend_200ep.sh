#!/usr/bin/env bash
# 2026-07-16 H01:00 Coil-PANet 续训脚本 (chain 完后自动启动)
# 续训策略: resume panet last.pt + lr=0.005 (half lr, 治震荡) + patience=120 (延长早停)
# 上限 +200 ep, 优先保留 panet 末段已学到的 Detect2 权重
set -e
cd /home/pi/projects/hyperyolo

NAME="v0_panet_hyper_yolon_250ep"

echo "===== panet 续训 (resume last.pt, lr=0.005, +200ep) ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/baseline/${NAME}/weights/last.pt \
  data=data/coil/data.yaml \
  epochs=450 patience=120 batch=16 imgsz=1024 \
  save=True save_period=25 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/baseline name="${NAME}_extend" exist_ok=True \
  pretrained=False optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.005 lrf=0.005 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=2.0 warmup_momentum=0.8 warmup_bias_lr=0.05 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=True nwd_constant=12.0 \
  coverage=False coverage_weight=0.5 coverage_sigma=20.0 \
  degrees=10.0 translate=0.1 scale=0.5 shear=0.0 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}_extend.log

echo ""
test -f runs/baseline/${NAME}_extend/args.yaml && \
  echo "panet 续训完成: $(ls runs/baseline/${NAME}_extend/weights/best.pt 2>&1)"