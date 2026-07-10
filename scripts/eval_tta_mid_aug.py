#!/usr/bin/env python
"""mid_aug best.pt + TTA + top-k + dist NMS 扫参 + 最佳参数可视化"""
import json, cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
RUN = ROOT / "runs/cfg_truth_repro/v8_nwd_v1_mid_aug_full"
BEST_PT = RUN / "weights/best.pt"
GT_DIR = ROOT / "data/coil/labels/val"
IMG_DIR = ROOT / "data/coil/images/val"
OUT_DIR = RUN / "predict_viz_tta"


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
        if any((ci - c[j]) @ (ci - c[j]) < dist_thr**2 for j in kept): continue
        kept.append(i)
    return boxes[kept].tolist(), scores[kept].tolist()


def main():
    print(f"[1/3] load {BEST_PT}")
    model = YOLO(str(BEST_PT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IMG_DIR.glob('*.png'))
    gt_by_name = {p.stem: load_gt(p)[1] for p in img_paths}

    print(f"[2/3] TTA predict on {len(img_paths)} imgs")
    raw_by_name = {}
    for img_p in img_paths:
        img = cv2.imread(str(img_p))
        boxes, scores = tta_predict(model, img)
        raw_by_name[img_p.stem] = list(zip(boxes, scores))
    print(f"   avg raw preds/img: {np.mean([len(v) for v in raw_by_name.values()]):.1f}")

    def eval_at(kept_by_name, conf_thr):
        tg, tp, pp = 0, 0, 0
        for stem, gts in gt_by_name.items():
            ps = [(b, s) for b, s in kept_by_name.get(stem, []) if s >= conf_thr]
            tg += len(gts); pp += len(ps)
            mg = set()
            for b, s in ps:
                bi, bii = 0, -1
                for i, g in enumerate(gts):
                    if i in mg: continue
                    v = iou(b, g)
                    if v > bi: bi, bii = v, i
                if bi >= 0.5: tp += 1; mg.add(bii)
        p = tp/pp if pp else 0
        r = tp/tg if tg else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        return p, r, f1, tp, pp

    print(f"\n扫参 (k, dist, conf):")
    print(f"{'k':>3} {'dist':>5} {'conf':>5} {'P':>6} {'R':>6} {'F1':>6} {'Pred':>5}")
    best = (0, None)
    for k in [1, 2, 3]:
        for d in [30, 50, 70]:
            kb = {}
            for stem, ps in raw_by_name.items():
                bb, ss = topk_dist([b for b,s in ps], [s for b,s in ps], k, d)
                kb[stem] = list(zip(bb, ss))
            for c in [0.05, 0.10, 0.15, 0.25]:
                p, r, f1, tp, pp = eval_at(kb, c)
                mark = " ⭐" if f1 > best[0] else ""
                if f1 > best[0]: best = (f1, (k, d, c, p, r, tp, pp))
                print(f"{k:>3} {d:>5} {c:>5.2f} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {pp:>5d}{mark}")

    f1, (k, d, c, p, r, tp, pp) = best
    print(f"\nBest F1={f1:.3f} at k={k} dist={d} conf={c}: P={p:.3f} R={r:.3f} TP={tp} Pred={pp}")

    # 用最佳参数生成可视化
    print(f"\n[3/3] 用最佳参数生成可视化 (k={k}, dist={d}, conf={c})")
    GT_COLOR = (0, 255, 0)
    PRED_OK = (0, 165, 255)
    PRED_FP = (0, 0, 255)
    MISS_COLOR = (255, 0, 255)

    rows = ["<html><head><meta charset='utf-8'><title>mid_aug TTA viz</title>",
            "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}img{max-width:520px}</style></head><body>",
            f"<h2>v8_nwd_v1_mid_aug_full best.pt + TTA + k={k} + dist={d} + conf>={c}</h2>",
            f"<p>绿=GT, 橙=Pred匹配, 红=Pred误检, 紫红=GT漏检, 数字=pred score</p>",
            "<table><tr><th>image</th><th>viz</th><th>summary</th></tr>"]
    for img_p in img_paths:
        img = cv2.imread(str(img_p))
        gt = gt_by_name[img_p.stem]
        pred_list = raw_by_name[img_p.stem]  # list of (box, score)
        boxes_all = [b for b, s in pred_list]
        scores_all = [s for b, s in pred_list]
        kept_b, kept_s = topk_dist(boxes_all, scores_all, k, d)
        kept_b = [b for b, s in zip(kept_b, kept_s) if s >= c]
        kept_s = [s for s in kept_s if s >= c]

        vis = img.copy()
        matched = set()
        for b, s in zip(kept_b, kept_s):
            bi, bii = 0, -1
            for i, g in enumerate(gt):
                if i in matched: continue
                v = iou(b, g)
                if v > bi: bi, bii = v, i
            color = PRED_OK if bi >= 0.5 else PRED_FP
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{s:.2f}", (x1, max(0, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if bii >= 0: matched.add(bii)
        for i, g in enumerate(gt):
            if i in matched: continue
            x1, y1, x2, y2 = map(int, g)
            cv2.rectangle(vis, (x1, y1), (x2, y2), MISS_COLOR, 2)
        tag = f"GT={len(gt)} Pred={len(kept_b)} Match={len(matched)}"
        cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
        cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(str(OUT_DIR / img_p.name), vis)
        status = "✅" if len(matched) == len(gt) else ("⚠️" if len(matched) > 0 else "❌")
        rows.append(f"<tr><td>{img_p.name}</td><td><a href='{img_p.name}'><img src='{img_p.name}'></a></td>"
                    f"<td>{status} GT={len(gt)} Pred={len(kept_b)} Match={len(matched)}</td></tr>")
    rows.append("</table></body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(rows), encoding="utf-8")
    print(f"Done. Open: file://{OUT_DIR/'index.html'}")


if __name__ == "__main__":
    main()