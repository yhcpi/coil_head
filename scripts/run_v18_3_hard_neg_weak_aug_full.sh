#!/bin/bash
# v18.3 = v12 baseline + 33 hard neg + 弱 aug (避免 aug 漂白副本) + 中等 lr fine-tune
# 改动 (相对 v18.2):
#   - degrees:    10.0 → 0.0  (不旋转，避免副本漂白)
#   - translate:  0.1  → 0.05 (极小抖动)
#   - scale:      0.5  → 0.0  (不缩放)
#   - flipud:     0.5  → 0.0  (不垂直翻转)
#   - copy_paste: 0.2  → 0.0  (不复制粘贴)
#   - lr0:        0.001 → 0.005 (更积极的 fine-tune)
#   - epochs:     80   → 100 (更长)
#   - fliplr=0.5, hsv_* 保留（水平翻转 + 颜色扰动不漂白图像语义）
#   - mosaic=0, mixup=0, multi_scale=0, nwd/coverage/box/cls 全部沿用 v12
set -u
cd /home/pi/projects/hyperyolo

NAME="v18_3_hard_neg_weak_aug_full"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v18.3 (hard neg + 弱 aug + lr=0.005 + 100ep) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=runs/cfg_truth_repro/v12_strong_aug_flipud_300ep/weights/best.pt \
  data=data/coil/data.yaml \
  epochs=100 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=1 start_val_epoch=0 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.005 lrf=0.0005 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.05 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  nwd=true nwd_constant=12.0 \
  coverage=false coverage_weight=0.5 \
  degrees=0.0 translate=0.05 scale=0.0 flipud=0.0 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== args.yaml 实际生效字段 ====="
test -f runs/cfg_truth_repro/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|lr0|lrf|warmup|box|cls|nwd|coverage|degrees|translate|scale|flipud|mosaic|copy_paste|multi_scale|pretrained|seed):" \
  runs/cfg_truth_repro/${NAME}/args.yaml

echo ""
echo "===== 训练结果 ====="
test -f runs/cfg_truth_repro/${NAME}/results.csv && \
  awk -F, 'NR>1{if($11+0>max){max=$11+0; line=$0}} END{print "Best mAP50: "max"\n@ "line}' runs/cfg_truth_repro/${NAME}/results.csv