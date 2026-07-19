#!/usr/bin/env bash
# 2026-07-15 22:00 baseline verification (no innovation).
# Plan B: yolov8n.pt + standard yolov8n PANet head + coil data + 强 aug + 250 epoch.
# 严格按用户 22:10 决定: mosaic 开, mixup 关, copy_paste=0.2.
# 与 V12 baseline 关键差异: 这跑的是 标准 YOLOv8n PANet (不是 Hyper-YOLO 自定义 MANet).
# 目的: 用现有架构 + 数据 + 训练配置估个真实训时间 + smoke verify 配置正确.
# 预期: train=578 + val=99 + 250 ep + 强 aug = 2 小时左右.
set -e
cd /home/pi/projects/hyperyolo

NAME="v0_baseline_yolov8n_strong_aug_250ep"

echo "===== 现有进程检查 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || true

echo "===== 启动 ${NAME} ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=repos/ultralytics/ultralytics/cfg/models/coil_exp/yolov8n_baseline.yaml \
  data=data/coil/data.yaml \
  epochs=250 patience=80 batch=16 imgsz=1024 \
  save=True save_period=25 val=True \
  cache=False device=0 workers=2 \
  project=/home/pi/projects/hyperyolo/runs/baseline name="${NAME}" exist_ok=True \
  pretrained=True single_cls=True nwd=True nwd_constant=12.0 optimizer=SGD verbose=True seed=0 deterministic=True \
  rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
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
