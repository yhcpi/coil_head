#!/usr/bin/env python
"""在 val 集上跑 best.pt，把 GT（绿）+ Pred（红）画到原图上对比。

用法:
    /home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/visualize_bestpt_val.py

输出:
    runs/cfg_truth_repro/<run>/predict_viz/val_conf05/*.png
    runs/cfg_truth_repro/<run>/predict_viz/val_conf25/*.png
    runs/cfg_truth_repro/<run>/predict_viz/index.html  (汇总)
"""
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
DATA_YAML = ROOT / "data/coil/data.yaml"
VAL_IMG_DIR = ROOT / "data/coil/images/val"
VAL_LBL_DIR = ROOT / "data/coil/labels/val"
RUN = ROOT / "runs/cfg_truth_repro/v8_nwd_v1_robust_aug_full"
BEST_PT = RUN / "weights/best.pt"
OUT_DIR = RUN / "predict_viz"

CONFS = [0.05, 0.25]   # 部署阈值 + 学术阈值
GT_COLOR = (0, 255, 0)   # BGR 绿
PRED_COLOR = (0, 0, 255)  # BGR 红
MISS_COLOR = (255, 0, 255)  # 紫红 (GT 没被检出)


def load_gt_boxes(img_path: Path):
    """读 yolo 格式标签，转 xyxy 像素坐标。空文件返回 []。"""
    lbl = VAL_LBL_DIR / (img_path.stem + ".txt")
    if not lbl.exists():
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    boxes = []
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def draw(img, gt, preds, iou_thr=0.3):
    """画 GT（绿）+ Pred（红），如果 Pred 与某个 GT IoU>=thr 也算匹配（pred 红变橙）。"""
    out = img.copy()
    matched_gt = set()
    for x1, y1, x2, y2 in preds:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            iou = iou_xyxy((x1, y1, x2, y2), g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        color = (0, 165, 255) if best_iou >= iou_thr else PRED_COLOR   # 橙色 = 匹配
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if best_idx >= 0:
            matched_gt.add(best_idx)
    for i, (x1, y1, x2, y2) in enumerate(gt):
        if i in matched_gt:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), MISS_COLOR, 2)  # 漏检紫红
    return out, len(matched_gt)


def main():
    print(f"[1/3] loading model {BEST_PT}")
    model = YOLO(str(BEST_PT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    val_imgs = sorted(VAL_IMG_DIR.glob("*.png"))
    print(f"[2/3] predicting on {len(val_imgs)} val images")

    for conf in CONFS:
        sub = OUT_DIR / f"val_conf{int(conf*100):02d}"
        sub.mkdir(parents=True, exist_ok=True)
        print(f"  conf={conf} → {sub}")

    for img_path in val_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = load_gt_boxes(img_path)
        # 一次推理拿多 conf 结果
        for conf in CONFS:
            res = model.predict(img_path, conf=conf, verbose=False, imgsz=1024)[0]
            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                for x1, y1, x2, y2 in xyxy:
                    preds.append((x1, y1, x2, y2))
            vis, matched = draw(img, gt, preds)
            tag = f"GT={len(gt)} Pred={len(preds)} Match={matched}/{len(gt)}"
            cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4)
            cv2.putText(vis, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            sub = OUT_DIR / f"val_conf{int(conf*100):02d}"
            cv2.imwrite(str(sub / img_path.name), vis)
            summary.append((img_path.name, conf, len(gt), len(preds), matched))

    # 写 index.html (双栏对比表)
    print(f"[3/3] writing index.html")
    rows = []
    rows.append("<html><head><meta charset='utf-8'><title>predict_viz</title>")
    rows.append("<style>body{font-family:sans-serif}table{border-collapse:collapse} "
                "td,th{border:1px solid #999;padding:4px 8px} "
                "img{display:block;max-width:480px;height:auto}</style></head><body>")
    rows.append("<h2>v8_nwd_v1_robust_aug_full best.pt — val 可视化</h2>")
    rows.append("<p>绿框=GT, 红框=Pred(漏匹配), 橙框=Pred(IoU>=0.3匹配GT), 紫红框=GT漏检</p>")
    rows.append("<table><tr><th>image</th><th>conf=0.05</th><th>conf=0.25</th></tr>")
    for name in sorted({s[0] for s in summary}):
        s05 = next((s for s in summary if s[0] == name and s[1] == 0.05), None)
        s25 = next((s for s in summary if s[0] == name and s[1] == 0.25), None)
        rows.append(f"<tr><td>{name}<br>"
                    f"<small>GT={s05[2]} conf05=P{s05[3]}/M{s05[4]} conf25=P{s25[3]}/M{s25[4]}</small></td>")
        rows.append(f"<td><a href='val_conf05/{name}'><img src='val_conf05/{name}'></a></td>")
        rows.append(f"<td><a href='val_conf25/{name}'><img src='val_conf25/{name}'></a></td></tr>")
    rows.append("</table></body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(rows), encoding="utf-8")
    print(f"Done. {len(summary)} predictions → {OUT_DIR}")
    print(f"Open: file://{OUT_DIR/'index.html'}")


if __name__ == "__main__":
    main()