#!/usr/bin/env python3
"""2026-07-16 v20 部署 F1 评估 vs v18.3 + baseline + v19r
v20 配置: baseline best.pt + dfl=0.0 + v18.3 弱 aug 范式 + 250 ep
评估口径: TTA-builtin + per-image top1 + conf sweep [0.10, 0.15, 0.20] + center dist ≤ 30 px
"""
import sys, json
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from ultralytics import YOLO
from PIL import Image

REPO = Path('/home/pi/projects/hyperyolo')
VAL_LBL_DIR = REPO / 'data/coil/labels/val'
VAL_IMG_DIR = REPO / 'data/coil/images/val'

RUNS = [
    {
        'name': 'v21_dfl_off_hn_250ep',
        'weights': REPO / 'runs/dfl_off/v21_dfl_off_hn_250ep/weights/best.pt',
    },
    {
        'name': 'v20_dfl_off_full_250ep',
        'weights': REPO / 'runs/dfl_off/v20_dfl_off_full_250ep/weights/best.pt',
    },
    {
        'name': 'v18_3_hard_neg_weak_aug_full',
        'weights': REPO / 'runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt',
    },
    {
        'name': 'v0_baseline_hyper_yolon_strong_aug_250ep',
        'weights': REPO / 'runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt',
    },
]


def load_gts_by_image():
    """读 val 标签 → 还原成原图 (W,H) 空间的 xyxy 坐标。"""
    gts = {}
    for txt in sorted(VAL_LBL_DIR.glob('*.txt')):
        img_stem = txt.stem
        img_p = VAL_IMG_DIR / f"{img_stem}.png"
        if not img_p.exists():
            continue
        W, H = Image.open(img_p).size
        boxes = []
        for line in open(txt):
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                boxes.append((x1, y1, x2, y2))
        gts[img_stem] = boxes
    return gts


def val_imgs():
    return sorted(VAL_IMG_DIR.glob('*.png')) + sorted(VAL_IMG_DIR.glob('*.jpg'))


def predict_top1_per_image(model, val_imgs, conf, imgsz=1024, max_det=1, tta=False):
    out = []
    for img_p in val_imgs:
        try:
            r = model.predict(
                source=str(img_p), imgsz=imgsz, conf=conf, iou=0.5, max_det=max_det,
                augment=tta, save=False, verbose=False,
            )[0]
        except RuntimeError as e:
            if tta:
                r = model.predict(
                    source=str(img_p), imgsz=imgsz, conf=conf, iou=0.5, max_det=max_det,
                    augment=False, save=False, verbose=False,
                )[0]
            else:
                raise
        if r.boxes is None or len(r.boxes) == 0:
            out.append(None); continue
        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
        out.append((float(best.conf[0]), x1, y1, x2, y2))
    return out


def center_dist(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def eval_top1_dist(top1_preds, gts_by_img, val_img_list, dist_thr=30.0):
    tp = fp = fn = tn = 0
    for img_p, pred in zip(val_img_list, top1_preds):
        gts = gts_by_img.get(img_p.stem, [])
        if pred is None and not gts: tn += 1
        elif pred is None and gts: fn += 1
        elif pred is not None and not gts: fp += 1
        else:
            best_d = min(center_dist(pred[1:5], g) for g in gts)
            if best_d < dist_thr: tp += 1
            else: fn += 1; fp += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'p': p, 'r': r, 'f1': f1}


def main():
    data_yaml = REPO / 'data/coil/data.yaml'
    gts_by_img = load_gts_by_image()
    val_img_list = val_imgs()
    confs = [0.10, 0.15, 0.20]
    print(f"val 集: {len(val_img_list)} 张 ({sum(1 for v in gts_by_img.values() if v)} 正 / {sum(1 for v in gts_by_img.values() if not v)} 负)")

    results = []
    for r in RUNS:
        weights = r['weights']
        if not weights.exists():
            print(f"  [SKIP] {r['name']}: {weights}")
            continue
        print(f"\n=== {r['name']} ===")
        m = YOLO(str(weights))
        try:
            m_val = m.val(data=str(data_yaml), imgsz=1024, conf=0.001, iou=0.6, max_det=1, verbose=False)
            mAP50 = float(m_val.box.map50)
            mAP50_95 = float(m_val.box.map)
        except Exception as e:
            print(f"  m.val() FAIL: {e}")
            mAP50 = mAP50_95 = -1
        print(f"  学术 mAP50 = {mAP50:.4f}, mAP50-95 = {mAP50_95:.4f}", flush=True)

        deploy = {}
        best_f1 = -1; best_conf = None
        for c in confs:
            top1 = predict_top1_per_image(m, val_img_list, conf=c, imgsz=1024, max_det=1, tta=True)
            ev = eval_top1_dist(top1, gts_by_img, val_img_list, dist_thr=30.0)
            key = f"c{int(c*100)}"
            deploy[f"f1_{key}"] = round(ev['f1'], 4)
            deploy[f"tp_{key}"] = ev['tp']
            deploy[f"fp_{key}"] = ev['fp']
            deploy[f"fn_{key}"] = ev['fn']
            print(f"  部署 conf={c:.2f} TTA dist≤30: F1={ev['f1']:.4f} TP={ev['tp']} FP={ev['fp']} FN={ev['fn']}")
            if ev['f1'] > best_f1: best_f1, best_conf = ev['f1'], c

        results.append({
            'exp_name': r['name'],
            'mAP50': round(mAP50, 4),
            'mAP50_95': round(mAP50_95, 4),
            'deploy_best_f1': round(best_f1, 4) if best_f1 >= 0 else None,
            'deploy_best_conf': best_conf,
            'deploy': deploy,
        })

    print("\n" + "=" * 95)
    print(f"  {'run':<45} {'mAP50':<8} {'mAP50-95':<10} | {'c0.10':<7} {'c0.15':<7} {'c0.20':<7}")
    print("-" * 95)
    for r in results:
        d = r['deploy']
        print(f"  {r['exp_name']:<45} {r['mAP50']:<8.4f} {r['mAP50_95']:<10.4f} | {d.get('f1_c10', 'N/A'):<7} {d.get('f1_c15', 'N/A'):<7} {d.get('f1_c20', 'N/A'):<7}")
    print("=" * 95)

    out = {
        'val_imgs': len(val_img_list),
        'dist_thr': 30.0,
        'conf_sweep': confs,
        'runs': {r['exp_name']: r for r in results},
    }
    out_json = REPO / 'runs/dfl_off/v20_vs_v18_3_summary.json'
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nJSON 已写入: {out_json}")

    v20 = next((r for r in results if 'v20' in r['exp_name']), None)
    v183 = next((r for r in results if 'v18' in r['exp_name']), None)
    if v20 and v183:
        delta = v20['deploy_best_f1'] - v183['deploy_best_f1']
        delta95 = v20['mAP50_95'] - v183['mAP50_95']
        print(f"\n结论: v20 部署 F1 {v20['deploy_best_f1']:.4f} vs v18.3 {v183['deploy_best_f1']:.4f} → Δ = {delta:+.4f}")
        print(f"       v20 mAP50-95 {v20['mAP50_95']:.4f} vs v18.3 {v183['mAP50_95']:.4f} → Δ = {delta95:+.4f}")
        if delta > 0:
            print("🎉 v20 部署破纪录! 建议立刻归档")
        else:
            print(f"v20 部署未破 v18.3 纪录 ({abs(delta):.4f})")


if __name__ == '__main__':
    main()