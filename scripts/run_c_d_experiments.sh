#!/bin/bash
# 串行跑 PA-Aug 4 个组件 ablation（GPU 1 个，避免 OOM）
# 每个 run 250 epoch，~1.3h，总计 ~5.2h
# 必须等 C 项 09_bayes_prior 完成后再启动

set -e
cd /home/pi/projects/hyperyolo

declare -A COMPONENTS=(
    [motion]=10_paaug_motion
    [reflection]=11_paaug_reflection
    [occlusion]=12_paaug_occlusion
    [noise]=13_paaug_noise
)

for c in motion reflection occlusion noise; do
    name=${COMPONENTS[$c]}
    echo "==== [$c] 启动训练 (project=runs/coil_loss_ablation, name=$name) ===="
    /home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
      task=detect mode=train \
      model=repos/Hyper-YOLO/hyper-yolon.pt \
      data=data/coil/data.yaml \
      epochs=250 patience=50 batch=16 imgsz=1024 \
      save=True val_period=1 start_val_epoch=0 save_period=-1 \
      cache=False device=0 workers=2 \
      project=runs/coil_loss_ablation name=$name exist_ok=True \
      pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
      single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
      fraction=1.0 cfg=data/coil/hyp_v7_paaug_${c}.yaml \
      lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 warmup_epochs=3.0 \
      warmup_momentum=0.8 warmup_bias_lr=0.1 box=5.0 cls=1.0 dfl=1.5 \
      label_smoothing=0.02 nbs=64 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
      degrees=5.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0005 \
      flipud=0.5 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.2 \
      iou=0.7 max_det=300 conf=0.001 plots=True 2>&1 | tail -50
    echo "==== [$c] 完成 ===="
done

echo "==== PA-Aug 4 组件 ablation 全部完成 ===="
