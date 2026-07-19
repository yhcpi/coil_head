#!/usr/bin/env python3
"""2026-07-16 v19 vs v18.3 vs baseline 三方对比脚本.
- v19: runs/cfg_truth_repro/v19_baseline_weak_aug_hn_100ep/weights/best.pt
- v18.3: runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt (saved from previous v18.3 train)
- baseline: runs/baseline/v0_baseline_hyper_yolon_strong_aug_250ep/weights/best.pt

评估口径: TTA-builtin + per-image top1 + conf sweep [0.10, 0.15, 0.20] + center dist ≤ 30 px
输出: 学术 mAP50 (model.val) + 部署 F1
"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

REPO = Path('/home/pi/projects/hyperyolo')

RUNS = [
    {
        'name': 'v19_baseline_weak_aug_hn_100ep_resume',
        'weights': REPO / 'runs/cfg_truth_repro/v19_baseline_weak_aug_hn_100ep_resume/weights/best.pt',
    },
    {
        'name': 'v19_baseline_weak_aug_hn_100ep',
        'weights': REPO / 'runs/cfg_truth_repro/v19_baseline_weak_aug_hn_100ep/weights/best.pt',
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

VAL_LBL_DIR = REPO / 'data/coil/labels/val'
VAL_IMG_DIR = REPO / 'data/coil/images/val'

import json
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO


def find_best(exp_name: str) -> Path:
    """兼容 v18.3 在 deploy_best 的特殊情况."""
    for r in RUNS:
        if r['name'] == exp_name:
            return r['weights']
    return None


def load_gts_by_image():
    """读 val 标签 → 还原成原图 (W,H) 空间的 xyxy 坐标（与 predict 输出坐标空间一致）。"""
    from PIL import Image
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
            out.append(None)
            continue
        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
        out.append((float(best.conf[0]), x1, y1, x2, y2))
    return out


def center_dist(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def eval_top1_dist(top1_preds, gts_by_img, dist_thr=30.0, imgsz=1024):
    tp = fp = fn = tn = 0
    for img_p, pred in zip(val_imgs(), top1_preds):
        gts = gts_by_img.get(img_p.stem, [])
        if pred is None and not gts:
            tn += 1
        elif pred is None and gts:
            fn += 1
        elif pred is not None and not gts:
            fp += 1
        else:
            best_d = min(center_dist(pred[1:5], g) for g in gts)
            if best_d < dist_thr:
                tp += 1
            else:
                fn += 1
                fp += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'p': p, 'r': r, 'f1': f1}


def eval_one(exp_name: str, weights: Path, val_imgs, gts_by_img, confs, data_yaml):
    if not weights.exists():
        print(f"  [SKIP] {exp_name}: weights 不存在 {weights}")
        return None
    print(f"\n=== {exp_name} ===")
    print(f"  weights: {weights}")
    m = YOLO(str(weights))
    try:
        m_val = m.val(data=str(data_yaml), imgsz=1024, conf=0.001, iou=0.6, max_det=1, verbose=False)
        mAP50 = float(m_val.box.map50)
    except Exception as e:
        print(f"  m.val() FAIL: {e}")
        mAP50 = -1.0
    print(f"  学术 mAP50 (conf=0.001, iou=0.6, max_det=1) = {mAP50:.4f}", flush=True)

    deploy = {}
    best_f1 = -1
    best_conf = None
    for c in confs:
        top1 = predict_top1_per_image(m, val_imgs, conf=c, imgsz=1024, max_det=1, tta=True)
        m_eval = eval_top1_dist(top1, gts_by_img, dist_thr=30.0)
        key = f"c{int(c*100)}"
        deploy[f"f1_{key}"] = round(m_eval['f1'], 4)
        deploy[f"tp_{key}"] = m_eval['tp']
        deploy[f"fp_{key}"] = m_eval['fp']
        deploy[f"fn_{key}"] = m_eval['fn']
        print(f"  部署 conf={c:.2f} TTA dist≤30: F1={m_eval['f1']:.4f} TP={m_eval['tp']} FP={m_eval['fp']} FN={m_eval['fn']}")
        if m_eval['f1'] > best_f1:
            best_f1 = m_eval['f1']
            best_conf = c

    return {
        'exp_name': exp_name,
        'weights': str(weights),
        'mAP50': round(mAP50, 4),
        'deploy_best_f1': round(best_f1, 4) if best_f1 >= 0 else None,
        'deploy_best_conf': best_conf,
        'deploy': deploy,
    }


def main():
    data_yaml = REPO / 'data/coil/data.yaml'
    gts_by_img = load_gts_by_image()
    val_img_list = val_imgs()
    confs = [0.10, 0.15, 0.20]
    print(f"val 集: {len(val_img_list)} 张 ({sum(1 for v in gts_by_img.values() if v)} 正 / {sum(1 for v in gts_by_img.values() if not v)} 负)")
    print(f"dist 阈值: 30.0 px")

    results = []
    for r in RUNS:
        res = eval_one(r['name'], r['weights'], val_img_list, gts_by_img, confs, data_yaml)
        if res is not None:
            results.append(res)

    print("\n" + "=" * 90)
    print(f"  {'run':<45} {'mAP50':<8} | {'c0.10':<7} {'c0.15':<7} {'c0.20':<7}")
    print("-" * 90)
    for res in results:
        d = res['deploy']
        print(f"  {res['exp_name']:<45} {res['mAP50']:<8.4f} | {d.get('f1_c10', 'N/A'):<7} {d.get('f1_c15', 'N/A'):<7} {d.get('f1_c20', 'N/A'):<7}")
    print("=" * 90)

    out = {
        'val_imgs': len(val_img_list),
        'val_pos': sum(1 for v in gts_by_img.values() if v),
        'val_neg': sum(1 for v in gts_by_img.values() if not v),
        'dist_thr': 30.0,
        'conf_sweep': confs,
        'runs': {res['exp_name']: res for res in results},
    }
    out_json = REPO / 'runs/cfg_truth_repro/v19_vs_v18_3_summary.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nJSON 已写入: {out_json}")

    # 与 v18.3 对比结论
    v183 = next((r for r in results if 'v18_3' in r['exp_name']), None)
    v19 = next((r for r in results if 'v19' in r['exp_name']), None)
    if v183 and v19:
        delta = v19['deploy_best_f1'] - v183['deploy_best_f1']
        print(f"\n结论: v19 部署 F1 {v19['deploy_best_f1']:.4f} vs v18.3 {v183['deploy_best_f1']:.4f} → Δ = {delta:+.4f}")
        if delta > 0:
            print("🎉 v19 破纪录! 建议立刻 save_repro_config 归档")
        else:
            print(f"v19 未破 v18.3 纪录 ({abs(delta):.4f})")


if __name__ == '__main__':
    main()