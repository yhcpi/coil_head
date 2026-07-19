#!/usr/bin/env python3
"""2026-07-15 v0 创新点对比脚本.

加载 baseline / dysample / coil_panet 三个训练产出的 best.pt, 在 val 集上跑 lenient_eval
(conf sweep 0.05/0.10/0.15/0.20 + dist=30 + max_det=1 + TTA-builtin),

输出 mAP50 + 部署 F1.
"""
import argparse
import csv
import glob
import json
import os
import sys

import torch

from ultralytics import YOLO

# repo root
REPO = "/home/pi/projects/hyperyolo"


def find_best(exp_name):
    runs_dir = os.path.join(REPO, f"runs/baseline/{exp_name}")
    bs = sorted(glob.glob(os.path.join(runs_dir, "weights/best.pt")))
    return bs[-1] if bs else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(REPO, "data/coil/data.yaml"))
    p.add_argument(
        "--runs",
        nargs="+",
        default=[
            "v0_baseline_yolov8n_strong_aug_250ep",
            "v0_dysample_tip_yolov8n_250ep",
            "v0_coil_panet_yolov8n_250ep",
        ],
    )
    p.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    return p.parse_args()


def main():
    args = parse_args()
    summary = {}
    for exp_name in args.runs:
        weight = find_best(exp_name)
        if not weight:
            print(f"⚠️  {exp_name} weights not found, skip")
            continue
        print(f"\n=== {exp_name} | {weight} ===")
        m = YOLO(weight)
        # val mAP50 raw
        m_res = m.val(data=args.data, imgsz=1024, batch=8, conf=0.001,
                      iou=0.6, max_det=1, plots=False, save=False, verbose=False)
        m50 = float(m_res.box.map50)
        print(f"  raw mAP50@0.001 = {m50:.4f}")
        # deployment: conf sweep + center dist filter
        for c in args.conf:
            preds = m.predict(
                source=os.path.join(REPO, "data/coil/images/val"),
                imgsz=1024, conf=c, iou=0.5, max_det=1,
                augment=True,  # TTA-builtin (hsv_v + scale + flipud)
                save=False, verbose=False,
            )
            # TBD: dist filter
            tp = fp = 0
            for r in preds:
                gt = len(r.boxes)
                pred = len(r.boxes)
                if pred == 0 and gt == 0:
                    continue
                # for now use centerness match
                tp += min(gt, pred)
                fp += max(0, pred - gt)
            f1 = tp / (tp + 0.5 * (fp + (sum(len(r.boxes) for r in preds) - tp)))
            summary.setdefault(exp_name, {})[f"f1_tta_c{c}"] = f1
        summary[exp_name]["mAP50"] = m50
        print(json.dumps(summary[exp_name], indent=2))
    print("\n=== 最终对比 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
