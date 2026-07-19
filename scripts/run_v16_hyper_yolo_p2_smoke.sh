#!/bin/bash
# v16 = Hyper-YOLO-P2 (新增 P2 输出层) smoke 测试
# 目的：验证 P2 四尺度架构能正常训练（loss 下降、梯度正常）
# 改动 (相对 v12 baseline):
#   - model: hyper-yolon.pt → repos/Hyper-YOLO/ultralytics/cfg/models/hyper-yolo/hyper-yolo-p2.yaml
#   - epochs: 300 → 30 (smoke)
#   - 其他 100% 沿用 v12
# 预期：训练 30 epoch，box_loss 从 ~20 降到 ~5-8，best.pt 应在 ep30
set -u
cd /home/pi/projects/hyperyolo

NAME="v16_hyper_yolo_p2_smoke"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 启动 v16 smoke (hyper-yolo-p2 30 epoch) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/ultralytics/cfg/models/hyper-yolo/hyper-yolo-p2.yaml \
  data=data/coil/data.yaml \
  epochs=30 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=1 start_val_epoch=0 \
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
  degrees=10.0 translate=0.1 scale=0.5 flipud=0.5 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.2 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== 训练结果 ====="
test -f runs/cfg_truth_repro/${NAME}/results.csv && \
  awk -F, 'NR>1{if($11+0>max){max=$11+0; line=$0}} END{print "Best mAP50: "max"\n@ "line}' runs/cfg_truth_repro/${NAME}/results.csv