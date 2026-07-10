#!/bin/bash
# v8 方式 B（纯 cfg 文件）验证脚本
# 目的：确认 cfg=path.yaml 在命令末尾时，cfg yaml 字段是否覆盖 CLI 默认
# 关键：训练 1 epoch 后看 args.yaml 实际生效字段
#
# 用法: bash scripts/run_v8_cfg_verify.sh
set -u
cd /home/pi/projects/hyperyolo

NAME="v8_cfg_verify"

# 用 coverage cfg，命令**末尾** 传 cfg=，不传 box/nwd/coverage 等字段
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=1 patience=0 batch=16 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v8_verify name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=False seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v8_coil_coverage.yaml \
  2>&1 | tail -40

echo "===== args.yaml 实际生效字段 ====="
grep -E "^(box|cls|nwd|coverage|copy_paste|degrees|label_smoothing|mosaic|translate|scale|bbox_shrink_min|multi_scale|cfg):" \
  runs/v8_verify/${NAME}/args.yaml
