#!/bin/bash
# v13 = v11 baseline + 3 张真漏标 (166/183/400) 增量重训 250 epoch
# 改动: data/coil/labels/train/{166,183,400}.txt 新增 1 个 GT（共 +3 正 / -3 负）
# 其他 100% 复用 run_v11_baseline_strong_aug_full.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v13_hard_neg_added_full"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v13 (= v11 + 3 张真漏标增量) 250 epoch ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=80 batch=16 imgsz=1024 \
  save=True save_period=20 val_period=1 start_val_epoch=0 \
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
  degrees=10.0 translate=0.1 scale=0.5 flipud=0.0 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== args.yaml 实际生效字段 ====="
test -f runs/cfg_truth_repro/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|box|cls|nwd|coverage|degrees|translate|scale|flipud|mosaic|copy_paste|multi_scale|pretrained|seed):" \
  runs/cfg_truth_repro/${NAME}/args.yaml

echo ""
echo "===== 训练结果 ====="
test -f runs/cfg_truth_repro/${NAME}/results.csv && \
  awk -F, 'NR>1{if($11+0>max){max=$11+0; line=$0}} END{print "Best mAP50: "max"\n@ "line}' runs/cfg_truth_repro/${NAME}/results.csv