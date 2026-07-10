#!/bin/bash
# v8 NWD v1 复现（弱 aug + NWD + cls=0.5，目标 mAP50=0.894）
# 验证 cfg 真相：cfg= 在命令末尾时 cfg 字段是否真的生效
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_nwd_v1_repro"

# Python 模拟 cfg 合并顺序：先覆盖非 cfg 字段，再加载 cfg 文件
# 因为 cfg 在命令末尾，cfg 字段会覆盖 CLI 字段（详见 [cfg-merge-truth]）
# 但本次命令 CLI 无 box/cls/nwd/coverage 等与 cfg 冲突的字段，所以 cfg 直接生效

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
  cfg=data/coil/hyp_v8_coil_nwd_v1_repro.yaml \
  2>&1 | tee /tmp/${NAME}.log
