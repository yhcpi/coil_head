#!/bin/bash
# critical A/B 测试 - 隔离 NBBoxNoise vs 其他 hook 对 box_loss 的影响
# A: nbbox=false  → 应回到 baseline ~4.24
# B: nbbox=true   → 已观察 box_loss=40+（确认 NBBox 单独贡献）
# C: nbbox=true + bbox_shrink=true → 协同效应
# 每个 10 epoch ~3 分钟
set -u
PHASE="${1:-A}"
cd /home/pi/projects/hyperyolo

case "$PHASE" in
  A)  # nbbox 关
    NBBX=false; SHRINK_P=0.0
    NAME="ab_A_nbbox_off"
    DESC="A: nbbox=false, shrink_p=0 → baseline 复现"
    ;;
  B)  # 只 nbbox 开
    NBBX=true; SHRINK_P=0.0
    NAME="ab_B_nbbox_only"
    DESC="B: nbbox=true, shrink_p=0 → NBBox 单独贡献"
    ;;
  C)  # nbbox + shrink 都开
    NBBX=true; SHRINK_P=1.0
    NAME="ab_C_nbbox_shrink"
    DESC="C: nbbox=true, shrink_p=1 → 协同效应"
    ;;
  *)
    echo "phase 必须 A / B / C"; exit 1 ;;
esac

# 临时覆盖 cfg
cat > /tmp/hyp_ab_${PHASE}.yaml << EOF
# A/B 测试：仅 nbbox + bbox_shrink 两个变量，其它全同 03 baseline
box: 5.0
cls: 1.0
dfl: 1.5
label_smoothing: 0.02
nwd: true
nwd_constant: 12.0
coverage: false
coverage_weight: 0.5
coverage_sigma: 20.0
multi_scale: 0.0
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
degrees: 5.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0005
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
nbbox: ${NBBX}
nbbox_p: 0.5
nbbox_scale_min: 0.7
nbbox_scale_max: 1.3
nbbox_shift: 0.1
bbox_shrink_min: 0.8
bbox_shrink_max: 1.2
bbox_shrink_p: ${SHRINK_P}
EOF

echo "[$(date)] ${DESC}"
echo "  cfg: /tmp/hyp_ab_${PHASE}.yaml"

/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=10 patience=10 batch=16 imgsz=1024 \
  save=True val_period=10 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=2 \
  project=runs/coil_loss_ablation name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  box=7.5 cls=0.5 dfl=1.5 label_smoothing=0.0 nwd=false coverage=false \
  nbs=64 lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 pose=12.0 kobj=1.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=0.0 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
  flipud=0.0 fliplr=0.5 mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  iou=0.7 max_det=300 conf=0.001 plots=True \
  cfg=/tmp/hyp_ab_${PHASE}.yaml \
  2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}.log

# 抓关键 epoch box/cls/dfl
echo ""
echo "=== ${NAME} 训练完成 ==="
echo "epoch 1 (box/cls/dfl/mAP50):"
awk -F',' 'NR==2 {printf "  box=%s  cls=%s  dfl=%s  mAP50=%s\n", $2, $3, $4, $11}' runs/coil_loss_ablation/${NAME}/results.csv
echo "epoch $(($(wc -l < runs/coil_loss_ablation/${NAME}/results.csv) - 1)) (末 epoch box/cls/dfl/mAP50):"
tail -1 runs/coil_loss_ablation/${NAME}/results.csv | awk -F',' '{printf "  box=%s  cls=%s  dfl=%s  mAP50=%s\n", $2, $3, $4, $11}'
