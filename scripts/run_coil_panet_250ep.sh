#!/usr/bin/env bash
# 2026-07-15 H22:50 Coil-PANet 实验: model=hyper_yolon_panet.yaml (Detect2 nl=2, P5 dropped) + 强 aug + 250 ep
# 用 hyper-yolon backbone (verified baseline), Detect2 子类去掉 P5 detection head (小目标够用, 减参 373K)
set -e
cd /home/pi/projects/hyperyolo

NAME="v0_panet_hyper_yolon_250ep"

echo "===== 启动 ${NAME} ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/Hyper-YOLO/ultralytics/cfg/models/coil_exp/hyper_yolon_panet.yaml \
  data=data/coil/data.yaml \
  epochs=250 patience=80 batch=16 imgsz=1024 \
  save=True save_period=25 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/baseline name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=True nwd_constant=12.0 \
  coverage=False coverage_weight=0.5 coverage_sigma=20.0 \
  degrees=10.0 translate=0.1 scale=0.5 shear=0.0 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
test -f runs/baseline/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|box|cls|nwd|degrees|flipud|mosaic|copy_paste|multi_scale|pretrained|seed|single_cls):" \
  runs/baseline/${NAME}/args.yaml