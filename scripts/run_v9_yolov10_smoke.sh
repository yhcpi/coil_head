#!/bin/bash
# v9 YOLOv10-inspired 大核 backbone smoke (10 epoch 看架构 forward + box_loss 量级)
# 目的: 验证 yolov10_backbone.yaml 架构不爆, NWD loss 链路正常
# 数据: train=312, val=43 (refresh 后, 2026-07-10)
# 期望:
#   - 10 epoch 跑通, 无 shape mismatch / OOM
#   - box_loss 起点 ≈ 0.8-1.5 (NWD 替换 IoU, box_gain=1.5)
#   - val mAP50 在 epoch 5+ 起步 ≥ 0.3 (验证 Detect head 工作)
#
# 用法: bash scripts/run_v9_yolov10_smoke.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v9_yolov10_smoke"

# 训练前先查重, 避免两个 train 共享 GPU 写 results.csv
echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

# 由于 backbone stem (k=7) + P3/P5 MANet (k=7) shape 与 hyper-yolon.pt 不匹配,
# 必须 pretrained=False 让新模块 fresh init; neck 部分会自动 init
# (Strict load 会爆 shape mismatch; 即便 strict=False 也只能加载 70-80% keys, 反而干扰)
echo "===== 启动 v9 yolov10 smoke (10 epoch) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=data/coil/yolov10_backbone.yaml \
  data=data/coil/data.yaml \
  epochs=10 patience=0 batch=16 imgsz=1024 \
  save=False val_period=2 start_val_epoch=2 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v9_yolov10 name="${NAME}" exist_ok=True \
  pretrained=False optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_nwd.yaml \
  2>&1 | tail -40

echo ""
echo "===== args.yaml 实际生效字段 ====="
grep -E "^(model|cfg|nwd|coverage|box|cls|degrees|copy_paste|imgsz|batch|mosaic|scale|flipud|multi_scale|lr0|warmup|epochs|pretrained):" \
  runs/v9_yolov10/${NAME}/args.yaml 2>/dev/null | head -20

echo ""
echo "===== results.csv (epoch 1-10 box/cls/mAP) ====="
head -11 runs/v9_yolov10/${NAME}/results.csv 2>/dev/null | awk -F, '{printf "%-4s box=%-7s cls=%-7s dfl=%-7s mAP50=%-7s mAP50-95=%s\n", $1, $2, $3, $4, $7, $8}'

echo ""
echo "===== 架构 forward 验证 (最后一层输出 shape) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -c "
import sys
sys.path.insert(0, 'repos/Hyper-YOLO')
from ultralytics import YOLO
import torch
m = YOLO('data/coil/yolov10_backbone.yaml').model
m.eval()
n_params = sum(p.numel() for p in m.parameters())
print(f'Total params: {n_params/1e6:.2f}M')
x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    y = m(x)
print(f'Output shape (Detect head): {[t.shape for t in y if isinstance(t, torch.Tensor)]}')" \
  2>&1 | tail -10