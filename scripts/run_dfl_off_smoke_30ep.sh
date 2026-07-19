#!/usr/bin/env bash
# 2026-07-16 DFL-off smoke (30 ep)
# Hypothesis: DFL 是为紧标注设计，钢卷 labelme 输出精度 ~3 px，DFL 在学噪声
# 设 dfl=0.0 让 DFL loss 不贡献梯度，bbox 回归改用 L1/CIoU
# 起点: baseline best.pt (mAP50=0.8888)
set -uo pipefail
cd /home/pi/projects/hyperyolo

PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/yolo detect train \
  model=runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt \
  data=data/coil/data.yaml \
  dfl=0.0 box=1.5 cls=0.5 \
  epochs=30 patience=20 \
  lr0=0.003 lrf=0.0001 \
  imgsz=1024 batch=8 device=0 \
  degrees=0.0 scale=0.0 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 \
  name=dfl_off_smoke_30ep \
  exist_ok=True \
  project=runs/dfl_off

echo "✅ smoke 启动 at $(date '+%H:%M:%S')"
echo "PID: $!"

# 输出最后 mAP50
sleep 5
ls -la runs/dfl_off/dfl_off_smoke_30ep/ 2>&1 | head -5