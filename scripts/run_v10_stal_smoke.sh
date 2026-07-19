#!/bin/bash
# v10 STAL (Small-Target-Aware Label assignment) smoke — 10 epoch 验证 STAL 接入 + box_loss 量级
# baseline: hyp_v8_coil_nwd.yaml (weak_aug + NWD, F1=0.929)
# 唯一变量: stal_area_thr=400 / stal_topk=13 / stal_expand=0.2 (通过 cfg= yaml 注入)
# 期望:
#   - 10 epoch 跑通, 无 shape mismatch / OOM
#   - box_loss 起点 ≈ 1.0-1.5 (NWD 替换 IoU)
#   - val mAP50 在 epoch 5+ 起步 ≥ 0.3 (STAL 不应该破坏 baseline)
#
# 用法: bash scripts/run_v10_stal_smoke.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v10_stal_smoke"

# 训练前先查重, 避免两个 train 共享 GPU 写 results.csv
echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v10 stal smoke (10 epoch) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  cfg=data/coil/hyp_v10_stal.yaml \
  epochs=10 patience=0 batch=16 imgsz=1024 \
  save=False val_period=2 start_val_epoch=2 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v10_stal name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  nbs=64 \
  2>&1 | tail -30

echo ""
echo "===== args.yaml 实际生效字段 ====="
grep -E "^(model|cfg|nwd|coverage|box|cls|degrees|copy_paste|imgsz|batch|mosaic|scale|flipud|multi_scale|lr0|warmup|epochs|pretrained|stal):" \
  runs/v10_stal/${NAME}/args.yaml 2>/dev/null | head -25

echo ""
echo "===== smoke 结果 ====="
test -f runs/v10_stal/${NAME}/results.csv && \
  awk -F, 'NR==1{next} NR>=3 {printf "ep%s box=%s cls=%s dfl=%s P=%s R=%s mAP50=%s mAP50-95=%s\n", $1, $2, $3, $4, $9, $10, $11, $12}' runs/v10_stal/${NAME}/results.csv | head -10 || \
  echo "(no results.csv yet)"