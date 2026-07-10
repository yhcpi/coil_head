"""C 项 (09_bayes_prior) vs v4 baseline 预测对比：每张图的预测数量分布 + conf 分布"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

from ultralytics import YOLO

VAL_DIR = Path('/home/pi/projects/hyperyolo/data/coil/images/val')

# 加载两个模型
models = {
    'v4_baseline': '/home/pi/projects/hyperyolo/runs/coil_loss_ablation/02_coverage_v4_N_model/weights/best.pt',
    'C_item_0.2':  '/home/pi/projects/hyperyolo/runs/coil_loss_ablation/09_bayes_prior/weights/best.pt',
}

print(f"{'img':<6} {'GT':<4} ", end='')
for name in models:
    print(f"{name+'#':<14} {name+'conf_max':<14} {name+'#>=0.05':<14}", end=' ')
print()

results = {name: [] for name in models}

val_imgs = sorted(VAL_DIR.glob('*.png'))
print(f"val={len(val_imgs)} 张")

for img_idx, img_p in enumerate(val_imgs):
    # 读 GT
    gt_p = Path('/home/pi/projects/hyperyolo/data/coil/labels/val') / f'{img_p.stem}.txt'
    n_gt = 0
    if gt_p.exists():
        with open(gt_p) as f:
            n_gt = sum(1 for line in f if line.strip())

    row = f"{img_idx:<6} {n_gt:<4} "
    for name, weights in models.items():
        result = YOLO(weights).predict(str(img_p), conf=0.001, imgsz=1024, verbose=False, rect=True)[0]
        n_pred = len(result.boxes) if result.boxes is not None else 0
        max_conf = 0.0
        if result.boxes is not None and len(result.boxes) > 0:
            max_conf = float(result.boxes.conf.max())
        n_at_005 = sum(1 for c in result.boxes.conf) if result.boxes is not None else 0
        results[name].append((img_idx, n_gt, n_pred, max_conf, n_at_005))
        row += f"{n_pred:<14} {max_conf:<14.4f} {n_at_005:<14} "
    print(row)

# 统计
print("\n=== 汇总 ===")
for name, rs in results.items():
    n_pred_arr = np.array([r[2] for r in rs])
    max_conf_arr = np.array([r[3] for r in rs])
    n_at_005_arr = np.array([r[4] for r in rs])
    print(f"\n[{name}]")
    print(f"  n_pred per image:  mean={n_pred_arr.mean():.2f}  median={np.median(n_pred_arr):.0f}  max={n_pred_arr.max()}  p90={np.percentile(n_pred_arr, 90):.0f}")
    print(f"  max_conf per img:  mean={max_conf_arr.mean():.4f}  p25={np.percentile(max_conf_arr, 25):.4f}  p50={np.percentile(max_conf_arr, 50):.4f}  p75={np.percentile(max_conf_arr, 75):.4f}")
    print(f"  #preds>=0.05/img:  mean={n_at_005_arr.mean():.2f}  total={n_at_005_arr.sum()}")
    # 看哪些图的 max_conf 很低（说明模型完全没把握）
    low_conf_imgs = sum(1 for c in max_conf_arr if c < 0.05)
    print(f"  #img max_conf<0.05: {low_conf_imgs}/{len(max_conf_arr)}")