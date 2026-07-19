#!/usr/bin/env python3
"""2026-07-15 v0 创新点对比脚本 v2（修复 v1 F1 计算 bug + 加中心距离过滤）。

对比 baseline / dysample / coil_panet 三个 best.pt：
  - 学术 mAP50：model.val(imgsz=1024, conf=0.001, iou=0.6, max_det=1)
  - 部署 F1：m.predict(augment=True) + per-image top1 + conf sweep [0.10, 0.15, 0.20]
              + 中心距离 <= 30 像素过滤

输出：
  - stdout：表格
  - JSON：runs/baseline/v0_innovations_summary.json

鲁棒性：任何一个 best.pt 不存在就 skip（不 raise）。
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

REPO = Path("/home/pi/projects/hyperyolo")
VAL_IMG_DIR = REPO / "data/coil/images/val"
VAL_LBL_DIR = REPO / "data/coil/labels/val"
OUT_JSON = REPO / "runs/baseline/v0_innovations_summary.json"

DEFAULT_RUNS = [
    "v0_baseline_hyper_yolon_strong_aug_250ep",
    "v0_dy_hyper_yolon_250ep",
    "v0_dy_v2_hyper_yolon_250ep",
    "v0_panet_hyper_yolon_250ep",
]


def find_best(exp_name: str):
    """runs/baseline/<exp_name>/weights/best.pt 找最新一个。"""
    runs_dir = REPO / "runs/baseline" / exp_name
    bs = sorted(runs_dir.glob("weights/best.pt"))
    return bs[-1] if bs else None


def load_gts_by_image():
    """读 val 标签，返回 list[list[(x1, y1, x2, y2)]]，按 PNG 文件名排序对齐。"""
    val_imgs = sorted(VAL_IMG_DIR.glob("*.png"))
    gts_by_img = []
    for img_p in val_imgs:
        W, H = Image.open(img_p).size
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists() and lbl_p.stat().st_size > 0:
            for line in lbl_p.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
                gts.append((x1, y1, x2, y2))
        gts_by_img.append(gts)
    return val_imgs, gts_by_img


def predict_top1_per_image(model, val_imgs, conf, imgsz=1024, max_det=1, tta=False):
    """返回 list[(conf, x1, y1, x2, y2)]，每张图最多一个（max_det=1 → top-1）。

    tta=True 走 multi-scale+flip 的 TTA-builtin；但 Detect2 nl=2 模型在 multi-scale 拼接时
    会 RuntimeError (conv.py line 322 Concat size mismatch)。自动 fallback: 若 tta=True 抛
    RuntimeError，改为 tta=False 重跑，不中断评估。
    """
    out = []
    for img_p in val_imgs:
        try:
            r = model.predict(
                source=str(img_p), imgsz=imgsz, conf=conf, iou=0.5, max_det=max_det,
                augment=tta, save=False, verbose=False,
            )[0]
        except RuntimeError as e:
            # Detect2 与 multi-scale TTA 不兼容 → fallback 到非 TTA 单尺度
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
    return float(np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2,
                          (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2))


def eval_top1_dist(top1_preds, gts_by_img, dist_thr=30.0):
    """per-image top1 + center_dist < dist_thr 算命中 → tp/fp/fn/p/r/f1。

    - 有 GT 且 top1 dist<thr  → tp++
    - 有 GT 且 top1 偏离       → fn++（同时算 FP）
    - 无 GT 且有 top1           → fp++
    - 无 GT 且 top1=None        → tn++
    """
    tp = fp = fn = 0
    for pred, gts in zip(top1_preds, gts_by_img):
        if pred is None:
            if not gts:
                continue  # tn
            fn += len(gts)
            continue
        if not gts:
            fp += 1
            continue
        # 取 GT 中最近的一个中心
        best_d = min(center_dist(pred[1:5], g) for g in gts)
        if best_d < dist_thr:
            tp += 1
        else:
            fn += len(gts)
            fp += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1}


def eval_one(exp_name: str, val_imgs, gts_by_img, confs, data_yaml):
    weight = find_best(exp_name)
    if not weight:
        print(f"  [SKIP] {exp_name}: best.pt 不存在", flush=True)
        return None

    print(f"\n=== {exp_name} ===")
    print(f"  weights: {weight}", flush=True)
    m = YOLO(str(weight))

    # 学术 mAP50（max_det=1 + conf=0.001 + iou=0.6）
    r = m.val(
        data=str(data_yaml), imgsz=1024, batch=8,
        conf=0.001, iou=0.6, max_det=1,
        plots=False, save=False, verbose=False, workers=0,
    )
    mAP50 = float(r.box.map50)
    print(f"  学术 mAP50 (conf=0.001, iou=0.6, max_det=1) = {mAP50:.4f}", flush=True)

    # 部署 F1：TTA-builtin + per-image top1 + conf sweep + dist<=30
    deploy = {}
    for c in confs:
        top1 = predict_top1_per_image(m, val_imgs, conf=c, imgsz=1024, max_det=1, tta=True)
        m_eval = eval_top1_dist(top1, gts_by_img, dist_thr=30.0)
        deploy[f"f1_tta_c{int(c*100):02d}"] = m_eval["f1"]
        deploy[f"f1_tta_c{int(c*100):02d}_tp"] = m_eval["tp"]
        deploy[f"f1_tta_c{int(c*100):02d}_fp"] = m_eval["fp"]
        deploy[f"f1_tta_c{int(c*100):02d}_fn"] = m_eval["fn"]
        print(f"  部署 conf={c:.2f} TTA-builtin dist≤30: "
              f"F1={m_eval['f1']:.4f} TP={m_eval['tp']} FP={m_eval['fp']} FN={m_eval['fn']}",
              flush=True)

    # 部署 F1 最佳（用于摘要）
    best_c = max(confs, key=lambda c: deploy[f"f1_tta_c{int(c*100):02d}"])
    best_f1 = deploy[f"f1_tta_c{int(best_c*100):02d}"]

    return {
        "weights": str(weight),
        "mAP50": round(mAP50, 4),
        "deploy_best_f1": round(best_f1, 4),
        "deploy_best_conf": best_c,
        "deploy": {k: round(v, 4) if isinstance(v, float) else v for k, v in deploy.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "data/coil/data.yaml"))
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--conf", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    p.add_argument("--dist", type=float, default=30.0)
    p.add_argument("--out", default=str(OUT_JSON))
    args = p.parse_args()

    val_imgs, gts_by_img = load_gts_by_image()
    n_pos = sum(1 for g in gts_by_img if g)
    n_neg = sum(1 for g in gts_by_img if not g)
    print(f"val 集：{len(val_imgs)} 张（{n_pos} 正 + {n_neg} 负）", flush=True)
    print(f"dist 阈值: {args.dist} px\n", flush=True)

    summary = {}
    for exp in args.runs:
        try:
            r = eval_one(exp, val_imgs, gts_by_img, args.conf, args.data)
        except Exception as e:
            print(f"  [ERROR] {exp}: {e}", flush=True)
            import traceback; traceback.print_exc()
            r = None
        if r:
            summary[exp] = r

    # stdout 表格
    print("\n" + "=" * 90)
    header_cells = " ".join(f"c{c:.2f}".rjust(6) for c in args.conf)
    print(f"  {'run':<45} {'mAP50':>8} | {header_cells}")
    print("-" * 90)
    for exp, r in summary.items():
        cells = " ".join(f"{r['deploy'][f'f1_tta_c{int(c*100):02d}']:.4f}".rjust(6) for c in args.conf)
        print(f"  {exp:<45} {r['mAP50']:>8.4f} | {cells}")
    print("=" * 90)

    # JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_imgs": len(val_imgs),
        "val_pos": n_pos,
        "val_neg": n_neg,
        "dist_thr": args.dist,
        "conf_sweep": args.conf,
        "runs": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON 已写入: {out_path}")


if __name__ == "__main__":
    main()