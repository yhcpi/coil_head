#!/usr/bin/env bash
# train_with_pseudo_v1.sh — weak_aug baseline + pseudo-label 联合训练
# 用法: bash scripts/train_with_pseudo_v1.sh [conf_thr] [name_suffix]
# 依赖: data/coil/pseudo_labels/ 已生成；当前无其他 ultralytics 训练
# 回退: rm -rf data/coil/pseudo_labels/ data/coil_combined/ runs/cfg_truth_repro/pseudo_v1_*/

set -euo pipefail
CONF_THR=${1:-0.30}
NAME_SUFFIX=${2:-c${CONF_THR//./}}
ROOT=/home/pi/projects/hyperyolo
PSEUDO=$ROOT/data/coil/pseudo_labels
COMBINED=$ROOT/data/coil_combined
NAME="pseudo_v1_${NAME_SUFFIX}"

[ -d "$PSEUDO" ] || { echo "[error] $PSEUDO 不存在，先跑 pseudo_label_v1.py"; exit 1; }
pgrep -f "ultralytics.*train" >/dev/null && echo "[warn] 检测到 ultralytics 训练在跑，会共享 GPU"

# combined 目录: symlink 原 train + pseudo
echo "[1/3] build $COMBINED"
rm -rf $COMBINED && mkdir -p $COMBINED/images/train $COMBINED/labels/train
ln -s $ROOT/data/coil/images/train/* $COMBINED/images/train/
ln -s $ROOT/data/coil/labels/train/*.txt $COMBINED/labels/train/
ln -s $PSEUDO/*.png $COMBINED/images/train/
ln -s $PSEUDO/*.txt $COMBINED/labels/train/
printf "path: %s\ntrain: images/train\nval: %s\nnc: 1\nnames: ['coil_head']\n" \
  "$COMBINED" "$ROOT/data/coil/images/val" > $COMBINED/data.yaml

# 训练: v8 NWD v1 弱 aug cfg + 250 epoch
echo "[2/3] train $NAME"
cd $ROOT
python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=$COMBINED/data.yaml cfg=data/coil/hyp_v8_coil_nwd_v1_repro.yaml \
  epochs=250 patience=50 batch=16 imgsz=1024 device=0 workers=2 \
  project=runs/cfg_truth_repro name=$NAME exist_ok=True \
  optimizer=SGD seed=0 deterministic=True single_cls=False rect=True \
  cos_lr=True close_mosaic=15 amp=True fraction=1.0 \
  val=True save=True save_period=10 val_period=1 cache=False pretrained=True verbose=True

echo "[3/3] done. eval:"
echo "  python scripts/eval_center_distance.py runs/cfg_truth_repro/$NAME/weights/best.pt"
