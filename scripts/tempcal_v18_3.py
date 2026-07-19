#!/usr/bin/env python3
"""
TempCal — Platt scaling 温度标定 + 按 bbox 面积分桶 conf 阈值
目标：在 V18.3 epoch60.pt 上验证 conf 分桶是否能突破 deployment F1=0.9286

算法：
  1. 在 val 99 张上推理，对每个 (pred_conf, pred_match_gt) 拟合 Platt scaling 温度 T
  2. 按 bbox 面积分 3 桶：小 (<100px²) / 中 (100-300) / 大 (>300)
  3. 对每桶用独立 conf 阈值扫 F1，最优阈值即为分桶阈值

输入：
  - runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt (V18.3 epoch60.pt)
  - data/coil/data.yaml

输出：
  - /tmp/tempcal_v18_3_results.json (含全局温度 + 每桶最优 conf + 全桶 F1)
  - 控制台打印最优分桶阈值表 + 对比 baseline
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# 设置 ultralytics
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy


def compute_iou_matrix(boxes1, boxes2):
    """boxes1: (N, 4) xyxy, boxes2: (M, 4) xyxy -> (N, M) IoU"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-6)


def compute_center_dist(boxes1, boxes2):
    """boxes1: (N, 4) xyxy, boxes2: (M, 4) xyxy -> (N, M) center distance"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.full((len(boxes1), len(boxes2)), 1e9)
    c1 = (boxes1[:, None, :2] + boxes1[:, None, 2:4]) / 2
    c2 = (boxes2[None, :, :2] + boxes2[None, :, 2:4]) / 2
    return np.linalg.norm(c1 - c2, axis=-1)


def get_gt_boxes_per_image(model, data_yaml):
    """返回每张 val 图的 GT boxes (xyxy pixel, 用原始图尺寸)"""
    import yaml
    from PIL import Image
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_imgs_dir = Path(cfg['path']) / cfg['val']
    lbls_dir = Path(cfg['path']) / 'labels' / 'val'
    gt_per_img = {}
    for img_path in sorted(val_imgs_dir.glob('*.png')):
        lbl_path = lbls_dir / (img_path.stem + '.txt')
        try:
            orig_w, orig_h = Image.open(img_path).size
        except Exception:
            orig_w, orig_h = 1024, 1024
        gt = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        gt.append((cx * orig_w, cy * orig_h, w * orig_w, h * orig_h))
        gt_per_img[img_path.name] = gt
    return gt_per_img


def run_val_inference(model, data_yaml, conf_thresh=0.001):
    """在 val 集上全推理，返回每张图的 (pred_boxes_xyxy, pred_confs, pred_areas)"""
    import yaml
    from PIL import Image
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_imgs_dir = Path(cfg['path']) / cfg['val']

    results = {}
    for img_path in sorted(val_imgs_dir.glob('*.png')):
        result = model.predict(str(img_path), imgsz=1024, conf=conf_thresh, verbose=False)
        boxes = result[0].boxes
        if len(boxes) == 0:
            results[img_path.name] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0))
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        results[img_path.name] = (xyxy, confs, areas)
    return results


def evaluate_deploy(per_img_preds, per_img_gt, conf_thresh, dist_thresh=30, iou_thresh=0.1):
    """
    部署口径 per-image top-1：每张图取 conf>thr 的所有预测，按 conf 降序，
    用 Lenient-Match (center dist<dist_thresh) 与 GT 匹配
    返回 (TP, FP, FN, TN, Recall, Precision, F1)
    """
    TP, FP, FN = 0, 0, 0
    for img_name, (pred_boxes, pred_confs, pred_areas) in per_img_preds.items():
        gt_boxes_xywh = per_img_gt.get(img_name, [])
        if len(gt_boxes_xywh) == 0:
            gt_xyxy = np.zeros((0, 4))
        else:
            gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_boxes_xywh])

        # 应用 conf 阈值
        keep = pred_confs >= conf_thresh
        pred_boxes_f = pred_boxes[keep]
        pred_confs_f = pred_confs[keep]
        gt_matched = np.zeros(len(gt_xyxy), dtype=bool)

        if len(pred_boxes_f) == 0:
            FN += len(gt_xyxy)
            continue

        # 按 conf 降序贪心匹配
        order = np.argsort(-pred_confs_f)
        for i in order:
            if len(gt_xyxy) == 0:
                FP += 1
                continue
            dists = compute_center_dist(pred_boxes_f[i:i+1], gt_xyxy)[0]
            ious = compute_iou_matrix(pred_boxes_f[i:i+1], gt_xyxy)[0]
            # Lenient-Match: dist<thresh OR IoU>thresh
            match_idx = np.where((dists < dist_thresh) | (ious > iou_thresh))[0]
            match_idx = match_idx[~gt_matched[match_idx]]
            if len(match_idx) > 0:
                # 取 dist 最小的
                best = match_idx[np.argmin(dists[match_idx])]
                gt_matched[best] = True
                TP += 1
            else:
                FP += 1
        FN += (~gt_matched).sum()
    Precision = TP / (TP + FP + 1e-6)
    Recall = TP / (TP + FN + 1e-6)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-6)
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), Recall=float(Recall),
                Precision=float(Precision), F1=float(F1))


def fit_platt_scaling(confs, labels):
    """用 numpy 简单拟合温度 T（最小化 NLL）"""
    # confs: (N,) 预测置信度, labels: (N,) 0/1 标签
    # 简化版：grid search T in [0.5, 3.0]
    best_T, best_nll = 1.0, 1e9
    for T in np.linspace(0.5, 3.0, 51):
        scaled = 1 / (1 + np.exp(-np.log(confs / (1 - confs + 1e-6) + 1e-6) / T))
        nll = -np.mean(labels * np.log(scaled + 1e-6) + (1 - labels) * np.log(1 - scaled + 1e-6))
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt')
    parser.add_argument('--data', default='data/coil/data.yaml')
    parser.add_argument('--out', default='/tmp/tempcal_v18_3_results.json')
    parser.add_argument('--conf_grid', default='0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30')
    args = parser.parse_args()

    print(f'=== TempCal: V18.3 conf 标定 + 分桶阈值 ===')
    print(f'模型: {args.model}')
    print(f'数据: {args.data}')

    # 1. 加载模型
    model = YOLO(args.model)
    gt_per_img = get_gt_boxes_per_image(model, args.data)
    print(f'val 集 GT 图数: {len(gt_per_img)}')

    # 2. 全推理 (conf=0.001 拿所有候选)
    print('正在推理 val 99 张 (conf=0.001)...')
    per_img_preds = run_val_inference(model, args.data, conf_thresh=0.001)
    print(f'推理完成，共 {sum(len(c) for _, c, _ in per_img_preds.values())} 个候选检测')

    # 3. 构建 (pred_conf, is_match_gt) 标签，用于 Platt 标定
    matched_labels = []
    matched_confs = []
    matched_areas = []
    for img_name, (pred_boxes, pred_confs, pred_areas) in per_img_preds.items():
        gt_boxes_xywh = gt_per_img.get(img_name, [])
        if len(gt_boxes_xywh) == 0:
            gt_xyxy = np.zeros((0, 4))
        else:
            gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_boxes_xywh])
        if len(pred_boxes) == 0:
            continue
        dists = compute_center_dist(pred_boxes, gt_xyxy) if len(gt_xyxy) > 0 else np.full((len(pred_boxes), 0), 1e9)
        ious = compute_iou_matrix(pred_boxes, gt_xyxy) if len(gt_xyxy) > 0 else np.zeros((len(pred_boxes), 0))
        for i in range(len(pred_boxes)):
            is_match = False
            if len(gt_xyxy) > 0:
                is_match = bool(((dists[i] < 30) | (ious[i] > 0.1)).any())
            matched_labels.append(int(is_match))
            matched_confs.append(float(pred_confs[i]))
            matched_areas.append(float(pred_areas[i]))

    matched_labels = np.array(matched_labels)
    matched_confs = np.array(matched_confs)
    matched_areas = np.array(matched_areas)

    print(f'候选数: {len(matched_confs)}, 匹配 GT 数: {matched_labels.sum()}')

    # 4. 拟合 Platt 温度
    T = fit_platt_scaling(matched_confs, matched_labels)
    print(f'Platt scaling 温度 T = {T:.3f} (T>1 → conf 偏低, T<1 → conf 偏高)')

    # 应用标定: conf' = sigmoid(logit(conf) / T)
    logits = np.log(matched_confs / (1 - matched_confs + 1e-6) + 1e-6)
    calibrated_confs = 1 / (1 + np.exp(-logits / T))

    # 5. 分桶：小 (<100px²) / 中 (100-300) / 大 (>300)
    small_mask = matched_areas < 100
    mid_mask = (matched_areas >= 100) & (matched_areas < 300)
    large_mask = matched_areas >= 300
    print(f'分桶数量: 小={small_mask.sum()}, 中={mid_mask.sum()}, 大={large_mask.sum()}')

    # 6. 全局 conf 阈值扫 F1 (baseline 对比)
    print('\n=== 全局 conf 阈值扫 F1 (标定前) ===')
    conf_grid = [float(c) for c in args.conf_grid.split(',')]
    baseline_results = {}
    for conf in conf_grid:
        # 重新构造 per_img_preds 应用 conf 阈值
        filtered = {}
        for img_name, (b, c, a) in per_img_preds.items():
            keep = c >= conf
            filtered[img_name] = (b[keep], c[keep], a[keep])
        m = evaluate_deploy(filtered, gt_per_img, conf_thresh=0.0)
        baseline_results[conf] = m
        print(f"  conf={conf:.3f}: F1={m['F1']:.4f} R={m['Recall']:.4f} P={m['Precision']:.4f} TP={m['TP']} FP={m['FP']} FN={m['FN']}")

    best_global = max(baseline_results.items(), key=lambda x: x[1]['F1'])
    print(f'  最优全局: conf={best_global[0]:.3f} F1={best_global[1]["F1"]:.4f}')

    # 7. 分桶 conf 阈值 (使用标定后的 conf)
    print('\n=== 分桶 conf 阈值 (标定后) ===')
    bucket_results = {}
    for bucket_name, mask in [('small', small_mask), ('mid', mid_mask), ('large', large_mask)]:
        if mask.sum() == 0:
            continue
        bk_confs = calibrated_confs[mask]
        bk_labels = matched_labels[mask]
        # 对每桶扫 conf 阈值
        best_conf, best_f1 = 0.0, 0.0
        for conf in conf_grid:
            tp = ((bk_confs >= conf) & (bk_labels == 1)).sum()
            fp = ((bk_confs >= conf) & (bk_labels == 0)).sum()
            fn = ((bk_confs < conf) & (bk_labels == 1)).sum()
            p = tp / (tp + fp + 1e-6)
            r = tp / (tp + fn + 1e-6)
            f1 = 2 * p * r / (p + r + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf
        bucket_results[bucket_name] = dict(threshold=best_conf, F1=best_f1, n=int(mask.sum()))
        print(f'  {bucket_name:6s} ({int(mask.sum()):3d} 个): 最优 conf={best_conf:.3f} F1={best_f1:.4f}')

    # 8. 整体 F1 (应用分桶 conf)
    bucket_thresh = {k: v['threshold'] for k, v in bucket_results.items()}
    print(f'\n=== 应用分桶阈值 {bucket_thresh} 评估整体 ===')

    def apply_bucket(pred_boxes, pred_confs, pred_areas):
        keep = np.zeros(len(pred_confs), dtype=bool)
        for i, (c, a) in enumerate(zip(pred_confs, pred_areas)):
            if a < 100:
                t = bucket_thresh.get('small', 0.15)
            elif a < 300:
                t = bucket_thresh.get('mid', 0.15)
            else:
                t = bucket_thresh.get('large', 0.15)
            keep[i] = c >= t
        return pred_boxes[keep], pred_confs[keep], pred_areas[keep]

    # 构造标定后的 preds
    per_img_calibrated = {}
    for img_name, (b, c, a) in per_img_preds.items():
        mask_in_img = None
        # 重新计算标定后的 conf
        if len(c) > 0:
            l = np.log(c / (1 - c + 1e-6) + 1e-6)
            cal_c = 1 / (1 + np.exp(-l / T))
        else:
            cal_c = c
        per_img_calibrated[img_name] = (b, cal_c, a)

    filtered = {}
    for img_name, (b, c, a) in per_img_calibrated.items():
        fb, fc, fa = apply_bucket(b, c, a)
        filtered[img_name] = (fb, fc, fa)
    bucket_metrics = evaluate_deploy(filtered, gt_per_img, conf_thresh=0.0)
    print(f"分桶评估: F1={bucket_metrics['F1']:.4f} R={bucket_metrics['Recall']:.4f} P={bucket_metrics['Precision']:.4f} TP={bucket_metrics['TP']} FP={bucket_metrics['FP']} FN={bucket_metrics['FN']}")

    # 9. 保存结果
    output = dict(
        model=args.model,
        platt_temperature=T,
        baseline_global=baseline_results,
        best_global=dict(conf=best_global[0], F1=best_global[1]['F1']),
        bucket_thresholds=bucket_results,
        bucket_metrics=bucket_metrics,
        n_candidates=int(len(matched_confs)),
        n_matched=int(matched_labels.sum()),
        improvement_F1=bucket_metrics['F1'] - best_global[1]['F1'],
    )
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n结果已保存到: {args.out}')
    print(f'\n=== 总结 ===')
    print(f'baseline (全局 conf sweep 最优): F1={best_global[1]["F1"]:.4f} @ conf={best_global[0]:.3f}')
    print(f'分桶 + Platt 标定:               F1={bucket_metrics["F1"]:.4f}')
    print(f'提升: {output["improvement_F1"]*100:+.2f}pp F1')


if __name__ == '__main__':
    main()