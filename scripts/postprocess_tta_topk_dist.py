#!/usr/bin/env python
"""TTA + per-image top-2 + dist=50 后处理，在 val 集上评估 best.pt。

TTA: 原图 + 水平翻转 + 上下翻转 (3 次推理，结果合并)
后处理:
    - per-image top-2 (按 score 排序，最多保留 2 个)
    - dist=50 NMS (中心距离 < 50px 的合并)
"""
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
RUN = ROOT / "runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full"
GT_DIR = ROOT / "data/coil/labels/val"
IMG_DIR = ROOT / "data/coil/images/val"

TTA_CONFS = [0.05, 0.10, 0.25]
TOPK = 2
DIST_THR = 50
IOU_THR = 0.5


def load_gt(img_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = GT_DIR / (img_path.stem + ".txt")
    boxes = []
    if lbl.exists():
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) < 5: continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h
            x2 = (cx + bw/2)*w; y2 = (cy + bh/2)*h
            boxes.append([x1, y1, x2, y2])
    return img, h, w, boxes


def tta_predict(model, img):
    """TTA: 原图 + Hflip + Vflip。每次翻转后预测，再把翻转的预测翻回原图坐标。"""
    h, w = img.shape[:2]
    runs = []

    # 原图
    res = model.predict(img, conf=0.001, verbose=False, imgsz=1024)[0]
    runs.append(res)

    # H-flip
    img_h = cv2.flip(img, 1)
    res_h = model.predict(img_h, conf=0.001, verbose=False, imgsz=1024)[0]
    runs.append(res_h)

    # V-flip
    img_v = cv2.flip(img, 0)
    res_v = model.predict(img_v, conf=0.001, verbose=False, imgsz=1024)[0]
    runs.append(res_v)

    boxes, scores = [], []
    for res, flip in zip(runs, [None, 'h', 'v']):
        if res.boxes is None: continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        scr = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), s in zip(xyxy, scr):
            if flip == 'h':
                x1, x2 = w - x2, w - x1
            elif flip == 'v':
                y1, y2 = h - y2, h - y1
            boxes.append([x1, y1, x2, y2])
            scores.append(float(s))
    return boxes, scores


def topk_dist_nms(boxes, scores, k=2, dist_thr=50):
    """per-image top-k + 距离 NMS: 中心距离<dist_thr 的保留 score 最高的。"""
    if not boxes:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes])
    order = np.argsort(-scores)  # 降序
    kept_idx = []
    for i in order:
        cx_i, cy_i = centers[i]
        suppress = False
        for j in kept_idx:
            cx_j, cy_j = centers[j]
            if (cx_i - cx_j)**2 + (cy_i - cy_j)**2 < dist_thr**2:
                suppress = True
                break
        if not suppress:
            kept_idx.append(i)
        if len(kept_idx) >= k:
            break
    return boxes[kept_idx].tolist(), scores[kept_idx].tolist()


def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0


def evaluate(kept_by_name, gt_by_name, conf_thr, iou_thr=0.5):
    tg, tp_total, pp_total = 0, 0, 0
    miss = []
    for stem, (w, h, gts) in gt_by_name.items():
        ps = [p for p in kept_by_name.get(stem, []) if p['score'] >= conf_thr]
        tg += len(gts); pp_total += len(ps)
        mg = set()
        for p in ps:
            bi, bii = 0, -1
            for i, g in enumerate(gts):
                if i in mg: continue
                v = iou_xyxy(p['bbox'], g)
                if v > bi: bi, bii = v, i
            if bi >= iou_thr:
                tp_total += 1; mg.add(bii)
        for i in range(len(gts)):
            if i not in mg: miss.append(stem)
    p = tp_total/pp_total if pp_total else 0
    r = tp_total/tg if tg else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return p, r, f1, tp_total, pp_total, tg, miss


def main():
    print(f"[1/4] loading {BEST_PT}")
    model = YOLO(str(BEST_PT))

    # 1) 加载 GT
    gt_by_name = {}
    img_paths = sorted(IMG_DIR.glob('*.png'))
    for img_p in img_paths:
        img, h, w, boxes = load_gt(img_p)
        gt_by_name[img_p.stem] = (w, h, boxes)
    print(f"[2/4] {len(gt_by_name)} val images, {sum(len(v[2]) for v in gt_by_name.values())} GT boxes")

    # 2) TTA 推理 (单次，全 conf=0.001)
    print(f"[3/4] TTA predict (orig + hflip + vflip)...")
    raw_by_name = {}
    for img_p in img_paths:
        img = cv2.imread(str(img_p))
        boxes, scores = tta_predict(model, img)
        raw_by_name[img_p.stem] = [
            {'bbox': b, 'score': s} for b, s in zip(boxes, scores)
        ]
    avg_raw = np.mean([len(v) for v in raw_by_name.values()])
    print(f"    TTA avg preds per image: {avg_raw:.1f}")

    # 3) 应用 top-k + dist NMS
    kept_by_name = {}
    for stem, ps in raw_by_name.items():
        boxes = [p['bbox'] for p in ps]
        scores = [p['score'] for p in ps]
        kb, ks = topk_dist_nms(boxes, scores, k=TOPK, dist_thr=DIST_THR)
        kept_by_name[stem] = [{'bbox': b, 'score': s} for b, s in zip(kb, ks)]
    avg_kept = np.mean([len(v) for v in kept_by_name.values()])
    print(f"    after top-{TOPK}+dist{DIST_THR}: {avg_kept:.2f} preds/image")

    # 4) 评估
    print(f"\n[4/4] 评估结果:")
    print(f"{'conf':>6} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'Pred':>5} {'GT':>4}  miss")
    for c in TTA_CONFS:
        p, r, f1, tp, pp, tg, miss = evaluate(kept_by_name, gt_by_name, c)
        print(f"{c:>6.3f} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {tp:>4d} {pp:>5d} {tg:>4d}  miss={len(miss)}")
        if c == 0.05:
            print(f"    漏检图: {sorted(set(miss))}")

    # 输出后处理的 prediction JSON 备用
    out = []
    for stem, ps in kept_by_name.items():
        for p in ps:
            x1, y1, x2, y2 = p['bbox']
            out.append({
                'image_id': stem,
                'category_id': 1,
                'bbox': [x1, y1, x2-x1, y2-y1],
                'score': p['score'],
            })
    out_path = RUN / "tta_topk_dist_predictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n后处理结果 JSON: {out_path}")


if __name__ == "__main__":
    main()