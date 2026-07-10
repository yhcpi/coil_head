#!/bin/bash
# baseline 退化诊断 - 定位 hook 改动链引入的 box_loss 起点 60 退化
# 设计：
#   D0: 03 baseline 原始 cfg（应回到 box_loss=4.24）
#   D1: D0 + bbox_shrink_min=1.0/max=1.0 (关闭 BBoxRandomShrink)
#   D2: D1 + MosaicNeg 关闭（需 monkey-patch patch_mosaic_neg(0,0,0)）
# 每个 5 epoch ~ 1.5 分钟
set -u
PHASE="${1:-D0}"
cd /home/pi/projects/hyperyolo

case "$PHASE" in
  D0)
    CFG="data/coil/hyp_v5_nwd_only.yaml"
    NAME="diag_D0_03_baseline_cfg"
    DESC="D0: 03 baseline 原始 cfg"
    WRAPPER=""
    ;;
  D1)
    CFG="/tmp/hyp_diag_D1.yaml"
    NAME="diag_D1_no_shrink"
    DESC="D1: 03 baseline cfg + bbox_shrink_min=1.0 (关闭 shrink)"
    cat > "$CFG" << 'YAML'
# D1: 03 baseline + 关闭 BBoxRandomShrink
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
bbox_shrink_min: 1.0
bbox_shrink_max: 1.0
bbox_shrink_p: 1.0
YAML
    WRAPPER=""
    ;;
  D2)
    CFG="/tmp/hyp_diag_D1.yaml"
    NAME="diag_D2_no_shrink_no_mosaic"
    DESC="D2: D1 + MosaicNeg monkey-patch 关闭"
    WRAPPER='patch_mosaic_neg(neg_p=0.0, neg_min=0, neg_max=0); '
    ;;
  *)
    echo "phase 必须 D0/D1/D2"; exit 1 ;;
esac

echo "[$(date)] ${DESC}"
echo "  cfg: ${CFG}"
echo "  name: ${NAME}"

# 用 python -c 直接调用 entrypoint()
/home/pi/anaconda3/envs/hyper-yolo/bin/python -c "
import sys
sys.path.insert(0, 'repos/Hyper-YOLO')
${WRAPPER}
from ultralytics.cfg import entrypoint
sys.argv = [
    '-m', 'ultralytics.models.yolo.detect.train',
    'task=detect', 'mode=train',
    'model=repos/Hyper-YOLO/hyper-yolon.pt',
    'data=data/coil/data.yaml',
    'epochs=5', 'patience=10', 'batch=16', 'imgsz=1024',
    'save=False', 'val_period=5', 'start_val_epoch=0',
    'cache=False', 'device=0', 'workers=2',
    'project=runs/coil_loss_ablation', 'name=${NAME}', 'exist_ok=True',
    'pretrained=True', 'optimizer=SGD', 'verbose=True', 'seed=0', 'deterministic=True',
    'single_cls=False', 'rect=True', 'cos_lr=True', 'close_mosaic=15', 'resume=False', 'amp=True',
    'fraction=1.0',
    'box=7.5', 'cls=0.5', 'dfl=1.5', 'label_smoothing=0.0', 'nwd=false', 'coverage=false',
    'nbs=64', 'lr0=0.01', 'lrf=0.01', 'momentum=0.937', 'weight_decay=0.0005',
    'warmup_epochs=3.0', 'warmup_momentum=0.8', 'warmup_bias_lr=0.1', 'pose=12.0', 'kobj=1.0',
    'hsv_h=0.015', 'hsv_s=0.7', 'hsv_v=0.4',
    'degrees=0.0', 'translate=0.1', 'scale=0.5', 'shear=0.0', 'perspective=0.0',
    'flipud=0.0', 'fliplr=0.5', 'mosaic=0.0', 'mixup=0.0', 'copy_paste=0.0',
    'iou=0.7', 'max_det=300', 'conf=0.001', 'plots=False',
    'cfg=${CFG}',
]
entrypoint()
" 2>&1 | tee /home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}.log

echo ""
echo "=== ${NAME} 关键 epoch box_loss ==="
awk -F',' 'NR>1 {printf "%-3d box=%-8s cls=%-8s dfl=%-6s mAP50=%s\n", $1, $2, $3, $4, $11}' runs/coil_loss_ablation/${NAME}/results.csv 2>/dev/null