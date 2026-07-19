#!/usr/bin/env python3
"""
V18.4 多 epoch deployment F1 评估
评估 V18.4 (v12 + 30 张不含 493 副本 + 弱 aug + lr=0.005) 各 epoch 的 deployment F1
对照 V18.3 (含 493) 验证"剔除 493 是否更优"

用法：
  python scripts/eval_v18_4_epochs.py
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


def get_val_gts(data_yaml):
    """返回 val 图 GT (normalized 0-1) + 原始图像尺寸"""
    import yaml
    from PIL import Image
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_imgs_dir = Path(cfg['path']) / cfg['val']
    val_lbls_dir = Path(cfg['path']) / 'labels' / 'val'
    gts = {}
    for img_path in sorted(val_imgs_dir.glob('*.png')):
        lbl_path = val_lbls_dir / (img_path.stem + '.txt')
        gt_list = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        # 存 normalized GT (0-1)，evaluate 时再乘以原始图尺寸
                        gt_list.append((cx, cy, w, h))
        # 同时记录原始图像尺寸
        try:
            orig_size = Image.open(img_path).size  # (W, H)
        except Exception:
            orig_size = (1024, 1024)
        gts[img_path.name] = dict(gt_norm=gt_list, orig_size=orig_size)
    return gts


def evaluate(model, gts, conf_thresh=0.15, dist_thresh=30, iou_thresh=0.1):
    """关键修复：GT 用原始图像坐标系匹配 YOLO 返回的 xyxy"""
    from PIL import Image
    TP, FP, FN = 0, 0, 0
    for img_name, info in gts.items():
        img_path = Path('data/coil/images/val') / img_name
        gt_norm = info['gt_norm']
        orig_w, orig_h = info['orig_size']

        # GT 转到原始图像坐标
        gt_list_px = [(cx * orig_w, cy * orig_h, w * orig_w, h * orig_h) for cx, cy, w, h in gt_norm]

        result = model.predict(str(img_path), imgsz=1024, conf=0.001, verbose=False)
        boxes = result[0].boxes
        if len(boxes) == 0:
            FN += len(gt_list_px)
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        keep = confs >= conf_thresh
        pred_boxes_f = xyxy[keep]
        pred_confs_f = confs[keep]
        gt_matched = np.zeros(len(gt_list_px), dtype=bool)
        if len(gt_list_px) > 0:
            gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_list_px])
        order = np.argsort(-pred_confs_f) if len(pred_confs_f) > 0 else []
        for i in order:
            if len(gt_list_px) == 0:
                FP += 1
                continue
            cxs = pred_boxes_f[i, [0, 2]].mean()
            cys = pred_boxes_f[i, [1, 3]].mean()
            dists = np.hypot(gt_xyxy[:, [0, 2]].mean(axis=1) - cxs,
                            gt_xyxy[:, [1, 3]].mean(axis=1) - cys)
            x1 = np.maximum(pred_boxes_f[i, 0], gt_xyxy[:, 0])
            y1 = np.maximum(pred_boxes_f[i, 1], gt_xyxy[:, 1])
            x2 = np.minimum(pred_boxes_f[i, 2], gt_xyxy[:, 2])
            y2 = np.minimum(pred_boxes_f[i, 3], gt_xyxy[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            a1 = (pred_boxes_f[i, 2] - pred_boxes_f[i, 0]) * (pred_boxes_f[i, 3] - pred_boxes_f[i, 1])
            a2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
            ious = inter / (a1 + a2 - inter + 1e-6)
            match_idx = np.where((dists < dist_thresh) | (ious > iou_thresh))[0]
            match_idx = match_idx[~gt_matched[match_idx]]
            if len(match_idx) > 0:
                best = match_idx[np.argmin(dists[match_idx])]
                gt_matched[best] = True
                TP += 1
            else:
                FP += 1
        FN += (~gt_matched).sum()
    P = TP / (TP + FP + 1e-6)
    R = TP / (TP + FN + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), Recall=R, Precision=P, F1=F1)


def evaluate_with_tta(model, gts, conf_thresh=0.15):
    """用 TTA-builtin (scale + fliplr) 评估"""
    TP, FP, FN = 0, 0, 0
    for img_name, gt_list in gts.items():
        img_path = Path('data/coil/images/val') / img_name
        # TTA-builtin: 3 尺度 × 2 翻转 = 6 路
        preds_all = []
        for scale in [1.0, 0.83, 0.67]:
            for flip in [False, True]:
                imgsz_scaled = int(1024 * scale)
                result = model.predict(str(img_path), imgsz=imgsz_scaled, conf=0.001, augment=True, verbose=False)
                boxes = result[0].boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    preds_all.append((xyxy, confs))
        if not preds_all:
            FN += len(gt_list)
            continue
        # 合并所有预测
        all_xyxy = np.concatenate([p[0] for p in preds_all])
        all_confs = np.concatenate([p[1] for p in preds_all])
        keep = all_confs >= conf_thresh
        pred_boxes_f = all_xyxy[keep]
        pred_confs_f = all_confs[keep]
        gt_matched = np.zeros(len(gt_list), dtype=bool)
        if len(gt_list) > 0:
            gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_list])
        order = np.argsort(-pred_confs_f) if len(pred_confs_f) > 0 else []
        for i in order:
            if len(gt_list) == 0:
                FP += 1
                continue
            cxs = pred_boxes_f[i, [0, 2]].mean()
            cys = pred_boxes_f[i, [1, 3]].mean()
            dists = np.hypot(gt_xyxy[:, [0, 2]].mean(axis=1) - cxs,
                            gt_xyxy[:, [1, 3]].mean(axis=1) - cys)
            match_idx = np.where(dists < 30)[0]
            match_idx = match_idx[~gt_matched[match_idx]]
            if len(match_idx) > 0:
                best = match_idx[np.argmin(dists[match_idx])]
                gt_matched[best] = True
                TP += 1
            else:
                FP += 1
        FN += (~gt_matched).sum()
    P = TP / (TP + FP + 1e-6)
    R = TP / (TP + FN + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), Recall=R, Precision=P, F1=F1)


def main():
    weight_dir = Path('runs/cfg_truth_repro/v18_4_hn_curriculum_no493_full/weights')
    gts = get_val_gts('data/coil/data.yaml')
    print(f'val GT 图数: {len(gts)}, 总正样本: {sum(len(v) for v in gts.values())}')

    # 评估每个 epoch
    epochs_to_eval = ['best.pt', 'last.pt', 'epoch30.pt', 'epoch40.pt', 'epoch50.pt', 'epoch60.pt', 'epoch70.pt', 'epoch80.pt']
    conf_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    results = {}
    print('\n=== V18.4 各 epoch baseline (无 TTA) deployment F1 ===')
    print(f"{'epoch':12s} {'best_conf':10s} {'F1':8s} {'R':8s} {'P':8s} {'TP':4s} {'FP':4s} {'FN':4s}")
    for ep_name in epochs_to_eval:
        wt_path = weight_dir / ep_name
        if not wt_path.exists():
            print(f'  {ep_name:12s} 不存在')
            continue
        try:
            model = YOLO(str(wt_path))
        except Exception as e:
            print(f'  {ep_name:12s} 加载失败: {e}')
            continue

        best_f1, best_conf = -1, 0
        best_metrics = dict(F1=0, Recall=0, Precision=0, TP=0, FP=0, FN=0)
        for conf in conf_grid:
            try:
                m = evaluate(model, gts, conf_thresh=conf)
                if m['F1'] > best_f1:
                    best_f1 = m['F1']
                    best_conf = conf
                    best_metrics = m
            except Exception as e:
                print(f'    conf={conf} 评估失败: {e}')
                continue
        results[ep_name] = dict(best_conf=best_conf, metrics=best_metrics)
        m = best_metrics
        print(f"  {ep_name:12s} {best_conf:.3f}     {m['F1']:.4f} {m['Recall']:.4f} {m['Precision']:.4f} {m['TP']:4d} {m['FP']:4d} {m['FN']:4d}")

    # TTA-builtin 评估 best.pt 和 last.pt
    print('\n=== V18.4 TTA-builtin @ conf=0.15 ===')
    for ep_name in ['best.pt', 'last.pt', 'epoch60.pt', 'epoch70.pt']:
        wt_path = weight_dir / ep_name
        if not wt_path.exists():
            continue
        try:
            model = YOLO(str(wt_path))
            m = evaluate_with_tta(model, gts, conf_thresh=0.15)
            print(f"  {ep_name:12s} TTA-builtin: F1={m['F1']:.4f} R={m['Recall']:.4f} P={m['Precision']:.4f} TP={m['TP']} FP={m['FP']} FN={m['FN']}")
            results.setdefault(ep_name, {})['tta_builtin'] = m
        except Exception as e:
            print(f"  {ep_name} TTA 失败: {e}")

    # 保存
    with open('/tmp/v18_4_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n结果已保存: /tmp/v18_4_eval_results.json')

    # 与 V18.3 对比
    print('\n=== 对比 V18.3 (含 493, 33 张副本) ===')
    print('V18.3 epoch60.pt + TTA-builtin @ conf=0.15:')
    print('  F1=0.9286 R=0.9070 P=0.9512 TP=39 FP=2 FN=4')
    print('\n=== 对比 V12 baseline ===')
    print('V12 best.pt + TTA-builtin @ conf=0.15:')
    print('  F1=0.9136 R=0.8605 P=0.9737 TP=37 FP=1 FN=6')


if __name__ == '__main__':
    main()