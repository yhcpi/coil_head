"""Hard Negative Mining: use v11 best.pt to run inference on train set, find high-conf FP candidates"""
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import sys
sys.path.insert(0, 'scripts')
from tta_inference import yolo_to_xyxy

WEIGHTS = 'runs/cfg_truth_repro/v11_baseline_strong_aug_full/weights/best.pt'
TRAIN_IMG = Path('data/coil/images/train')
TRAIN_LBL = Path('data/coil/labels/train')
OUT = Path('/tmp/v11_train_predictions.json')

model = YOLO(WEIGHTS)
imgs = sorted(TRAIN_IMG.glob('*.png'))
print('predicting', len(imgs), 'train images with conf=0.001, imgsz=1024, max_det=300...')

results = []
for i, p in enumerate(imgs):
    if i % 50 == 0:
        print('  [{}/{}] {}'.format(i, len(imgs), p.name))
    W, H = Image.open(p).size
    gt_path = TRAIN_LBL / (p.stem + '.txt')
    gts = yolo_to_xyxy(gt_path, W, H)
    preds_raw = model.predict(str(p), conf=0.001, imgsz=1024, max_det=300, rect=True, verbose=False)[0]
    preds = []
    if preds_raw.boxes is not None and len(preds_raw.boxes) > 0:
        for box in preds_raw.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_v = float(box.conf[0])
            cls_v = int(box.cls[0])
            preds.append({'conf': conf_v, 'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2), 'cls': cls_v})
    results.append({
        'img': p.name,
        'W': W, 'H': H,
        'gt_count': len(gts),
        'gt_boxes': [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in gts],
        'pred_count': len(preds),
        'preds': preds,
    })

OUT.write_text(json.dumps(results, indent=1))
print('')
print('Saved to', OUT)
print('Total:', len(results), 'images')
total_preds = sum(r['pred_count'] for r in results)
total_gt = sum(r['gt_count'] for r in results)
print('Total predictions:', total_preds)
print('Total GT:', total_gt)
