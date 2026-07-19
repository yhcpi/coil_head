#!/usr/bin/env bash
# 2026-07-15 DySample-Tip 创新点 (yolov8n baseline 上 +
# 把 nn.Upsample 换成 DySample 占位 placeholder)
# 同 baseline 全配置, 只换 model=yolov8n_dy.yaml.
set -e
cd /home/pi/projects/hyperyolo

NAME="v0_dysample_tip_yolov8n_250ep"

echo "===== 现有进程检查 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || true

echo "===== 启动 ${NAME} ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/ultralytics/ultralytics/cfg/models/coil_exp/yolov8n_dy.yaml \
  data=data/coil/data.yaml \
  epochs=250 patience=80 batch=16 imgsz=1024 \
  save=True save_period=25 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/baseline name="${NAME}" exist_ok=True \
  pretrained=True single_cls=True optimizer=SGD verbose=True seed=0 deterministic=True \
  rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  degrees=10.0 translate=0.1 scale=0.5 shear=0.0 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== args.yaml 实际生效字段 ====="
test -f runs/baseline/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|box|cls|degrees|translate|scale|flipud|fliplr|mosaic|mixup|copy_paste|multi_scale|pretrained|seed|single_cls):" \
  runs/baseline/${NAME}/args.yaml
