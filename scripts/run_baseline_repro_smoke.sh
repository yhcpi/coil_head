#!/bin/bash
# baseline 复现 smoke (50 epoch) — 严格 1:1 复制 run_v8_nwd_v1_weak_aug_full.sh
# 目的: 验证历史 best.pt (mAP50=0.869 @ ep208) 训练用的实际配置
# 比 pure_cli_smoke 多 flipud=0.0（之前 smoke 漏了这个字段！）
# 比对: ep10/30/50 box_loss+P+R+mAP50 与历史 training 一致 → 配置 100% 正确
set -u
cd /home/pi/projects/hyperyolo

NAME="baseline_repro_smoke"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 baseline 复现 smoke (50 epoch, 严格 1:1 复制 full) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=50 patience=0 batch=16 imgsz=1024 \
  save=False val_period=5 start_val_epoch=5 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  nwd=true nwd_constant=12.0 \
  coverage=false coverage_weight=0.5 \
  degrees=0.0 translate=0.1 scale=0.5 flipud=0.0 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== args.yaml 实际生效字段（验证） ====="
test -f runs/cfg_truth_repro/${NAME}/args.yaml && \
  grep -E "^(degrees|translate|scale|flipud|fliplr|box|cls|nwd|coverage|copy_paste|mosaic|multi_scale|epochs|pretrained):" \
  runs/cfg_truth_repro/${NAME}/args.yaml

echo ""
echo "===== smoke 50 epoch 结果 vs 历史 baseline 对照 ====="
echo ""
echo "字段          | baseline full ep10 | baseline full ep30 | baseline full ep50"
echo "------------- | ------------------ | ------------------ | ------------------"
echo "训练 ep10 来自历史 results.csv"
test -f runs/cfg_truth_repro/${NAME}/results.csv && \
  awk -F, 'NR>1 && (NR-1==10 || NR-1==30 || NR-1==50) {printf "  ep%s: box=%s P=%s R=%s mAP50=%s\n", $1, $2, $9, $10, $11}' runs/cfg_truth_repro/${NAME}/results.csv