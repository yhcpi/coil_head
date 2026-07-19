#!/usr/bin/env python
"""pseudo_label_v1.py — CADT/STAC 思路的师生一致性 pseudo-label 生成 (钢卷场景)

思路 (借鉴 CADT: Confident Anchor-based Teacher-Student, STAC 2020):
  老师 weak_aug best.pt (mAP50=0.869, 部署 F1=0.929) → TTA (原图+hflip+vflip)
  → per-image top-k + dist NMS → conf>=阈值 → 出 YOLO txt；同时复制图像。
  conf<阈值 或 无检测 → 空 txt = 负样本。

回退: rm -rf data/coil/pseudo_labels/
"""
import argparse, json, shutil
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/pi/projects/hyperyolo")
DEFAULT_TEACHER = ROOT / "runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "data/coil/pseudo_labels"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--teacher", default=DEFAULT_TEACHER, type=Path)
    p.add_argument("--conf-thr", default=0.30, type=float)
    p.add_argument("--top-k", default=1, type=int)
    p.add_argument("--dist-thr", default=30, type=float)
    p.add_argument("--imgsz", default=1024, type=int)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT, type=Path)
    return p.parse_args()


def tta_predict(model, img, imgsz):
    """TTA: 原图+hflip+vflip，conf=0.001 拿全预测，返回原图坐标 xyxy。"""
    h, w = img.shape[:2]
    boxes, scores = [], []
    for flip in [None, "h", "v"]:
        im = img if flip is None else cv2.flip(img, 1 if flip == "h" else 0)
        r = model.predict(im, conf=0.001, verbose=False, imgsz=imgsz)[0]
        if r.boxes is None:
            continue
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.conf.cpu().numpy()):
            if flip == "h":
                x1, x2 = w - x2, w - x1
            elif flip == "v":
                y1, y2 = h - y2, h - y1
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            scores.append(float(s))
    return boxes, scores


def topk_dist(boxes, scores, k, dist_thr):
    """per-image top-k + 中心距离 NMS (像素平方)。"""
    if not boxes:
        return [], []
    b = np.array(boxes); s = np.array(scores)
    c = np.array([((b[i, 0] + b[i, 2]) / 2, (b[i, 1] + b[i, 3]) / 2) for i in range(len(b))])
    kept = []
    for i in np.argsort(-s):
        if len(kept) >= k:
            break
        ci = c[i]
        if any(np.sum((ci - c[j]) ** 2) < dist_thr ** 2 for j in kept):
            continue
        kept.append(i)
    return b[kept].tolist(), s[kept].tolist()


def xyxy_to_yolo(box, w, h):
    cx = max(0.0, min(1.0, ((box[0] + box[2]) / 2) / w))
    cy = max(0.0, min(1.0, ((box[1] + box[3]) / 2) / h))
    bw = max(0.0, min(1.0, (box[2] - box[0]) / w))
    bh = max(0.0, min(1.0, (box[3] - box[1]) / h))
    return cx, cy, bw, bh


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[teacher] {args.teacher}\n[gating] conf>={args.conf_thr} top-{args.top_k} "
          f"dist<{args.dist_thr}\n[output] {out_dir}")

    model = YOLO(str(args.teacher))
    images = sorted(p for p in args.images_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"[count] {len(images)} images")

    n_total = n_kept = n_empty = 0
    confs = []
    for img_p in images:
        n_total += 1
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"  [skip] {img_p.name}")
            continue
        h, w = img.shape[:2]
        boxes, scores = tta_predict(model, img, args.imgsz)
        kb, ks = topk_dist(boxes, scores, args.top_k, args.dist_thr)
        final = [(b, s) for b, s in zip(kb, ks) if s >= args.conf_thr]
        shutil.copy2(img_p, out_dir / img_p.name)  # self-contained
        out_txt = out_dir / (img_p.stem + ".txt")
        if not final:
            out_txt.write_text("")
            n_empty += 1
        else:
            lines = []
            for b, s in final:
                cx, cy, bw, bh = xyxy_to_yolo(b, w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines) + "\n")
            confs.extend(s for _, s in final)
            n_kept += 1
        if n_total % 20 == 0 or n_total == len(images):
            print(f"  [{n_total}/{len(images)}] kept={n_kept} empty={n_empty}")

    print(f"\n[done] total={n_total} pseudo={n_kept} ({n_kept/max(1,n_total)*100:.1f}%) empty={n_empty}")
    if confs:
        c = np.array(confs)
        print(f"  conf: min={c.min():.3f} median={np.median(c):.3f} max={c.max():.3f} mean={c.mean():.3f}")
    meta = {"teacher": str(args.teacher), "images_dir": str(args.images_dir),
            "conf_thr": args.conf_thr, "top_k": args.top_k, "dist_thr": args.dist_thr,
            "imgsz": args.imgsz, "n_total": n_total, "n_pseudo": n_kept, "n_empty": n_empty}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
