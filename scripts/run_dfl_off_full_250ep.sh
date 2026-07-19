#!/usr/bin/env bash
# 2026-07-16 v20: DFL-off + v18.3 范式 + 250 ep full 训练
# Smoke (30 ep) 结果: mAP50=0.8702 (-1.9pp vs baseline 0.8888), mAP50-95=0.4191 (+3.6pp)
# 假设: 250 ep + v18.3 弱 aug 范式能进一步涨 mAP50-95, 部署 F1 可能破 v18.3 0.9286
# 起点: baseline best.pt (mAP50=0.8888) — DFL 已带过来, 但 dfl_loss=0 不更新
set -uo pipefail
cd /home/pi/projects/hyperyolo

# 数据已含 33 张 hard neg (hn{1,2,3}_*.png 副本) — v18.3 范式
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt \
  data=data/coil/data.yaml \
  dfl=0.0 box=1.5 cls=0.5 \
  epochs=250 patience=80 \
  lr0=0.005 lrf=0.0005 \
  imgsz=1024 batch=8 device=0 \
  degrees=0.0 scale=0.0 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 \
  name=v20_dfl_off_full_250ep \
  exist_ok=True \
  project=runs/dfl_off

echo "✅ v20 full 启动 at $(date '+%H:%M:%S')"
echo "PID: $!"