#!/bin/bash
# v4 baseline 复现（强 aug + IoU+Coverage + cls=1.0，目标 mAP50=0.877）
# 验证 cfg 真相：cfg= 在命令末尾时 cfg 字段是否覆盖 CLI 字段
# v4 baseline 用 hyp_aug.yaml（coverage=true），等效于 IoU loss（详见 [coverage-true-root-cause]）
set -u
cd /home/pi/projects/hyperyolo

NAME="v4_baseline_repro"

# CLI 命令不包含 box/cls/nwd/coverage 等与 cfg 冲突的字段
# cfg=hyp_aug.yaml 在命令末尾，cfg 文件字段生效：
#   box=5.0, cls=1.0, coverage=true, coverage_weight=0.5
#   degrees=5.0, translate=0.1, scale=0.5, perspective=0.0005, flipud=0.5
#   copy_paste=0.2, label_smoothing=0.02
#   bbox_shrink=0.8/1.2/1.0, multi_scale=0.2
#   nwd=false

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_aug.yaml \
  2>&1 | tee /tmp/${NAME}.log
