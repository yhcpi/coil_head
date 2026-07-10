#!/usr/bin/env python
"""weak_aug best.pt + TTA + 后处理 (k=1, dist=30, conf=0.10) 可视化"""
import json, cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
RUN = ROOT / "runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full"
BEST_PT = RUN / "weights/best.pt"
GT_DIR = ROOT / "data/coil/labels/val"
IMG_DIR = ROOT / "data/coil/images/val"
OUT_DIR = RUN / "predict_viz_tta"

K = 1
DIST_THR = 30
CONF_THR = 0.10
GT_COLOR = (0, 255, 0)
PRED_OK = (0, 165, 255)     # 橙: 匹配
PRED_FP = (0, 0, 255)       # 红: 误检
MISS_COLOR = (255, 0, 255)  # 紫红: 漏检
IOU_THR = 0.5


def load_gt(img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = GT_DIR / (img_path.stem + ".txt")
    boxes = []
    if lbl.exists():
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) < 5: continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx-bw/2)*w; y1 = (cy-bh/2)*h
            x2 = (cx+bw/2)*w; y2 = (cy+bh/2)*h
            boxes.append([x1, y1, x2, y2])
    return img, boxes


def iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0


def tta_predict(model, img):
    h, w = img.shape[:2]
    boxes, scores = [], []
    for flip in [None, 'h', 'v']:
        im = img if flip is None else (cv2.flip(img, 1) if flip == 'h' else cv2.flip(img, 0))
        r = model.predict(im, conf=0.001, verbose=False, imgsz=1024)[0]
        if r.boxes is None: continue
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if flip == 'h': x1, x2 = w - x2, w - x1
            elif flip == 'v': y1, y2 = h - y2, h - y1
            boxes.append([x1, y1, x2, y2])
            scores.append(float(s))
    return boxes, scores


def topk_dist(boxes, scores, k, dist_thr):
    if not boxes: return [], []
    boxes = np.array(boxes); scores = np.array(scores)
    c = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes])
    order = np.argsort(-scores); kept = []
    for i in order:
        if len(kept) >= k: break
        ci = c[i]
        if any((ci - c[j]) @ (ci - c[j]) < dist_thr**2 for j in kept):
            continue
        kept.append(i)
    return boxes[kept].tolist(), scores[kept].tolist()


def main():
    print(f"[1/3] load {BEST_PT}")
    model = YOLO(str(BEST_PT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IMG_DIR.glob('*.png'))
    print(f"[2/3] TTA predict + top-{K}+dist{DIST_THR} on {len(img_paths)} imgs")

    summary = []
    for img_p in img_paths:
        img, gt = load_gt(img_p)
        boxes, scores = tta_predict(model, img)
        # 应用 conf
        boxes_c = [b for b, s in zip(boxes, scores) if s >= CONF_THR]
        scores_c = [s for s in scores if s >= CONF_THR]
        # top-k + dist
        kept_b, kept_s = topk_dist(boxes_c, scores_c, K, DIST_THR)

        vis = img.copy()
        matched = set()
        for b, s in zip(kept_b, kept_s):
            bi, bii = 0, -1
            for i, g in enumerate(gt):
                if i in matched: continue
                v = iou(b, g)
                if v > bi: bi, bii = v, i
            color = PRED_OK if bi >= IOU_THR else PRED_FP
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{s:.2f}", (x1, max(0, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if bii >= 0: matched.add(bii)
        for i, g in enumerate(gt):
            if i in matched: continue
            x1, y1, x2, y2 = map(int, g)
            cv2.rectangle(vis, (x1, y1), (x2, y2), MISS_COLOR, 2)
        tag = f"GT={len(gt)} Pred={len(kept_b)} Match={len(matched)} conf>={CONF_THR} TTA k={K} d={DIST_THR}"
        cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
        cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(str(OUT_DIR / img_p.name), vis)
        summary.append((img_p.name, len(gt), len(kept_b), len(matched)))

    # index.html
    print(f"[3/3] index.html")
    rows = ["<html><head><meta charset='utf-8'><title>TTA deploy viz</title>",
            "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}img{max-width:520px}</style></head><body>",
            f"<h2>v8_nwd_v1_weak_aug_full best.pt + TTA + top-{K} + dist={DIST_THR} + conf>={CONF_THR}</h2>",
            "<p>绿框=GT, 橙框=Pred匹配GT(IoU>=0.5), 红框=Pred误检, 紫红框=GT漏检, 数字=pred score</p>",
            "<table><tr><th>image</th><th>viz</th><th>summary</th></tr>"]
    for name, gt_n, pred_n, match_n in summary:
        status = "✅" if match_n == gt_n else ("⚠️" if match_n > 0 else "❌")
        rows.append(f"<tr><td>{name}</td><td><a href='{name}'><img src='{name}'></a></td>"
                    f"<td>{status} GT={gt_n} Pred={pred_n} Match={match_n}</td></tr>")
    rows.append("</table></body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(rows), encoding="utf-8")

    # 统计
    total_gt = sum(s[1] for s in summary)
    total_pred = sum(s[2] for s in summary)
    total_match = sum(s[3] for s in summary)
    p = total_match / total_pred if total_pred else 0
    r = total_match / total_gt if total_gt else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    miss = [s[0] for s in summary if s[3] < s[1]]
    fp_count = sum(1 for s in summary if s[2] > s[3])
    print(f"\n结果: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"   漏检图 {len(miss)}: {miss}")
    print(f"   误检图 {fp_count}")
    print(f"   Open: file://{OUT_DIR/'index.html'}")


if __name__ == "__main__":
    main()