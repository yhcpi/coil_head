#!/usr/bin/env python
"""V5 v5 部署 F1 评估 — 单一 conf 评估 (Lenient-Match, dist_thresh=30)

直接复刻 V18.3 部署口径:
  conf=0.05, imgsz=1024, max_det=300
匹配: center_distance<=30px (Greedy top-1)

输出: TP/FP/FN 总数 + Per-image 详情
"""
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/ultralytics')
sys.path.insert(0, '/home/pi/projects/hyperyolo/src/hyper_yolo_patches')
import ultralytics
sys.modules['ultralytics'] = ultralytics  # bind to new version (含 C3k2)

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
from lenient_eval import compute_center_dist, yolo_to_xyxy

try:
    import yolo26_loss_extension  # noqa
except Exception:
    pass

from ultralytics import YOLO

WEIGHT = '/home/pi/projects/hyperyolo/repos/ultralytics/runs/detect/runs/yolo26_coil/v5_v5_nwd_only_soft_relative_pat100_350ep/weights/best.pt'
VAL_DIR = '/home/pi/projects/hyperyolo/data/coil/images/val'
GT_DIR = '/home/pi/projects/hyperyolo/data/coil/labels/val'
IMGSZ = 1024

OUT_DIR = Path('/home/pi/projects/hyperyolo/runs/yolo26_coil/v5_v5_deploy_eval')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def eval_at_conf(weights, conf, max_det=300, imgsz=1024):
    """单 conf 值评估, 返回详细 metrics"""
    print(f'\n{"="*60}')
    print(f'[eval] conf={conf} imgsz={imgsz}')
    print(f'{"="*60}')
    model = YOLO(weights)
    val_imgs = sorted(Path(VAL_DIR).glob('*.png'))
    gt_dir = Path(GT_DIR)

    total_tp = 0; total_fp = 0; total_fn = 0
    per_img = []
    for img_p in val_imgs:
        W, H = Image.open(img_p).size
        gt_p = gt_dir / f'{img_p.stem}.txt'
        gt_list = yolo_to_xyxy(gt_p, W, H)
        r = model.predict(str(img_p), conf=conf, imgsz=imgsz,
                          max_det=max_det, verbose=False, rect=True)[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                preds.append((float(box.conf[0]), x1, y1, x2, y2))
        pred_boxes = [(p[1], p[2], p[3], p[4], p[0]) for p in preds]
        pred_boxes.sort(key=lambda x: -x[4])
        matched_gt = set(); matched_pred = set()
        for pi, (x1, y1, x2, y2, c) in enumerate(pred_boxes):
            best_dist = 1e9; best_gi = -1
            for gi, (cls, gx1, gy1, gx2, gy2) in enumerate(gt_list):
                if gi in matched_gt: continue
                d = compute_center_dist((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if d < best_dist: best_dist = d; best_gi = gi
            if best_gi >= 0 and best_dist <= 30:
                matched_gt.add(best_gi); matched_pred.add(pi)
        tp = len(matched_pred); fp = len(pred_boxes) - tp; fn = len(gt_list) - len(matched_gt)
        total_tp += tp; total_fp += fp; total_fn += fn
        per_img.append({
            'img': img_p.name, 'gt': len(gt_list), 'pred': len(pred_boxes),
            'tp': tp, 'fp': fp, 'fn': fn,
        })

    prec = total_tp / (total_tp + total_fp + 1e-9)
    rec = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    print(f'TP={total_tp} FP={total_fp} FN={total_fn}  '
          f'P={prec:.4f} R={rec:.4f} F1={f1:.4f}')
    return {
        'conf': conf,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'per_image': per_img,
    }


def main():
    print(f'[V5 v5] {WEIGHT}')
    # 与 V18.3 部署口径一致: conf=0.05
    res = eval_at_conf(WEIGHT, conf=0.05)
    out_path = OUT_DIR / 'v5_v5_deploy_f1_c0.05.json'
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f'\n[save] {out_path}')


if __name__ == '__main__':
    main()
