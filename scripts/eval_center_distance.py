#!/usr/bin/env python
"""用'中心距离 < 阈值' 替代 IoU>=0.5 的评估方式。

适用场景: 钢卷 tip 小目标 + 宽松标注 (labelme 外接矩形)。
对比三个 best.pt 在不同评估标准下的 P/R/F1。
所有评估都是 per-image (一个 pred 只能匹配本图 GT)。
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
GT_DIR = ROOT / "data/coil/labels/val"
IMG_DIR = ROOT / "data/coil/images/val"


def load_gt(img_path):
    img = cv2.imread(str(img_path)); h, w = img.shape[:2]
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
    return img, boxes


def center_distance(a, b):
    return (( (a[0]+a[2])/2 - (b[0]+b[2])/2 )**2 +
            ( (a[1]+a[3])/2 - (b[1]+b[3])/2 )**2)**0.5


def iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0


def tta_predict(model, img):
    """orig + hflip + vflip"""
    h, w = img.shape[:2]
    boxes, scores = [], []
    for flip in [None, 'h', 'v']:
        im = img if flip is None else (cv2.flip(img, 1) if flip == 'h' else cv2.flip(img, 0))
        r = model.predict(im, conf=0.001, verbose=False, imgsz=1024)[0]
        if r.boxes is None: continue
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy()):
            if flip == 'h': x1, x2 = w - x2, w - x1
            elif flip == 'v': y1, y2 = h - y2, h - y1
            boxes.append([x1, y1, x2, y2])
            scores.append(float(s))
    return boxes, scores


def topk_dist(boxes, scores, k, dist_thr):
    """按 conf 排序，取 k 个; 中心距离 < dist_thr 的丢弃"""
    if not boxes: return [], []
    boxes = np.array(boxes); scores = np.array(scores)
    c = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes])
    order = np.argsort(-scores); kept = []
    for i in order:
        if len(kept) >= k: break
        ci = c[i]
        if any((ci-c[j])@(ci-c[j]) < dist_thr**2 for j in kept):
            continue
        kept.append(i)
    return boxes[kept].tolist(), scores[kept].tolist()


def eval_per_image(kept_by_name, gt_by_name, conf_thr, match_fn):
    """per-image 匹配, 统计全局 P/R/F1
    match_fn(pred, gt) -> bool
    """
    tg = tp = pp = 0
    for stem, gts in gt_by_name.items():
        ps = [(b, s) for b, s in kept_by_name.get(stem, []) if s >= conf_thr]
        tg += len(gts)
        pp += len(ps)
        matched = set()
        for b, s in ps:
            for i, g in enumerate(gts):
                if i in matched: continue
                if match_fn(b, g):
                    tp += 1; matched.add(i); break
    p = tp/pp if pp else 0
    r = tp/tg if tg else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return p, r, f1, tp, pp


MODELS = [
    ("weak_aug",   ROOT/"runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full/weights/best.pt"),
    ("mid_aug",    ROOT/"runs/cfg_truth_repro/v8_nwd_v1_mid_aug_full/weights/best.pt"),
    ("robust_aug", ROOT/"runs/cfg_truth_repro/v8_nwd_v1_robust_aug_full/weights/best.pt"),
]


def main():
    img_paths = sorted(IMG_DIR.glob('*.png'))
    gt_by_name = {p.stem: load_gt(p)[1] for p in img_paths}

    # TTA + 后处理在三个模型上分别跑
    results = {}  # model_name -> kept_by_name
    for model_name, best_pt in MODELS:
        print(f"\n[TTA] {model_name} ({best_pt.name})")
        if not best_pt.exists():
            print(f"  ❌ best.pt 不存在: {best_pt}")
            results[model_name] = None
            continue
        model = YOLO(str(best_pt))
        kept_by_name = {}
        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            boxes, scores = tta_predict(model, img)
            # TTA + top-1 + dist=30
            kb, ks = topk_dist(boxes, scores, k=1, dist_thr=30)
            kept_by_name[img_p.stem] = list(zip(kb, ks))
        results[model_name] = kept_by_name

    # === 在不同评估标准下比较 ===
    print(f"\n{'='*90}")
    print(f"对比 (TTA + top-1 + dist=30, conf>=0.10 后处理):")
    print(f"{'='*90}")

    # 评估方式 A: IoU >= 0.5 (传统/学术)
    print(f"\n[A] 传统评估 IoU >= 0.5")
    print(f"  {'model':>10} | {'P':>6} {'R':>6} {'F1':>6} {'Pred':>5}")
    base = {}
    for model_name, kept in results.items():
        if kept is None: continue
        p, r, f1, tp, pp = eval_per_image(kept, gt_by_name, 0.10,
                                          lambda b, g: iou(b, g) >= 0.5)
        print(f"  {model_name:>10} | {p:.3f} {r:.3f} {f1:.3f} {pp:>5d}")
        base[model_name] = f1

    # 评估方式 B: 中心距离 < thr
    print(f"\n[B] 中心距离评估 (center_distance < thr px)")
    dist_thrs = [5, 10, 15, 20, 30, 40, 50, 80]
    header = f"  {'model':>10} | " + " ".join(f"{'d<'+str(d):>6}" for d in dist_thrs)
    print(header)
    cd_table = {}
    for model_name, kept in results.items():
        if kept is None: continue
        line = f"  {model_name:>10} | "
        cd_table[model_name] = []
        for d in dist_thrs:
            p, r, f1, tp, pp = eval_per_image(kept, gt_by_name, 0.10,
                                              lambda b, g, dd=d: center_distance(b, g) < dd)
            line += f" {f1:>6.3f}"
            cd_table[model_name].append((d, f1, p, r))
        print(line)

    print(f"\n{'='*90}")
    print("结论:")
    if base and cd_table:
        winner_iou = max(base, key=base.get)
        for d in dist_thrs:
            fs = {m: [x[1] for x in cd_table[m] if x[0]==d][0] for m in cd_table}
            winner_cd = max(fs, key=fs.get)
            if winner_iou != winner_cd:
                print(f"  d<{d}px: 评估胜者从 IoU 的 '{winner_iou}' 变成 '{winner_cd}'")
            else:
                print(f"  d<{d}px: 评估胜者仍为 '{winner_iou}'")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
