#!/usr/bin/env python3
"""
TipGeoPrior — 从训练集 GT 统计 bbox 几何分布，对推理候选做硬规则过滤
目标：消除"conjoined patch 误检为大 bbox"等极端 FP

算法：
  1. 解析 train 集 312 张正样本的所有 GT bbox
  2. 统计 w/h (aspect_ratio), area, center_x, center_y 的 5%-95% 分位数
  3. 推理时对每个 conf>0.05 的检测框检查 3 条硬规则，任一不满足则 conf *= 0.05
  4. 评估过滤前后的 deployment F1 对比

输入：
  - runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt
  - data/coil/data.yaml
  - imgsz=1024

输出：
  - /tmp/tip_geo_prior_stats.json (几何分布统计)
  - /tmp/tip_geo_prior_results.json (过滤前后 F1 对比)
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


def get_train_gts(data_yaml):
    """返回所有 train GT 的 normalized (cx, cy, w, h) in 0-1 + 对应原图尺寸"""
    import yaml
    from PIL import Image
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    train_lbls_dir = Path(cfg['path']) / 'labels' / 'train'
    train_imgs_dir = Path(cfg['path']) / cfg['train']

    gts = []
    for lbl_path in sorted(train_lbls_dir.glob('*.txt')):
        if lbl_path.stem.startswith('hn'):
            continue
        # 找对应原图（去掉可能的扩展名）
        img_path = train_imgs_dir / (lbl_path.stem + '.png')
        if not img_path.exists():
            continue
        try:
            orig_w, orig_h = Image.open(img_path).size
        except Exception:
            orig_w, orig_h = 1024, 1024
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, w, h = map(float, parts[:5])
                    # normalized GT (0-1)
                    gts.append((cx, cy, w, h, orig_w, orig_h))
    return gts


def compute_geo_stats(gts):
    """统计 normalized 几何特征分布"""
    arr = np.array(gts)  # (N, 6) = (cx, cy, w, h, orig_w, orig_h)
    cx, cy, w, h, ow, oh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    aspect = w / (h + 1e-6)
    rel_area = w * h  # 已经是 normalized（cx,cy,w,h 都是 0-1），w*h = 相对面积
    cx_rel = cx
    cy_rel = cy

    stats = dict(
        n=int(len(gts)),
        aspect=dict(
            p5=float(np.percentile(aspect, 5)),
            p50=float(np.percentile(aspect, 50)),
            p95=float(np.percentile(aspect, 95)),
            mean=float(aspect.mean()),
            std=float(aspect.std()),
        ),
        rel_area=dict(  # 相对原图面积比例（0-1）
            p5=float(np.percentile(rel_area, 5)),
            p50=float(np.percentile(rel_area, 50)),
            p95=float(np.percentile(rel_area, 95)),
            mean=float(rel_area.mean()),
            std=float(rel_area.std()),
        ),
        cx_rel=dict(  # normalized center x
            p5=float(np.percentile(cx_rel, 5)),
            p50=float(np.percentile(cx_rel, 50)),
            p95=float(np.percentile(cx_rel, 95)),
            mean=float(cx_rel.mean()),
            std=float(cx_rel.std()),
        ),
        cy_rel=dict(  # normalized center y
            p5=float(np.percentile(cy_rel, 5)),
            p50=float(np.percentile(cy_rel, 50)),
            p95=float(np.percentile(cy_rel, 95)),
            mean=float(cy_rel.mean()),
            std=float(cy_rel.std()),
        ),
    )
    return stats


def is_in_bounds(cx_norm, cy_norm, w_norm, h_norm, stats, n_sigma=2.0):
    """检查 normalized bbox 是否在训练集几何分布范围内"""
    aspect = w_norm / (h_norm + 1e-6)
    rel_area = w_norm * h_norm  # relative area (0-1)

    # aspect ratio
    if not (stats['aspect']['p5'] <= aspect <= stats['aspect']['p95']):
        return False, 'aspect_out'

    # relative area
    if not (stats['rel_area']['p5'] <= rel_area <= stats['rel_area']['p95']):
        return False, 'area_out'

    # center (normalized)
    cx_lo = stats['cx_rel']['mean'] - n_sigma * stats['cx_rel']['std']
    cx_hi = stats['cx_rel']['mean'] + n_sigma * stats['cx_rel']['std']
    if not (cx_lo <= cx_norm <= cx_hi):
        return False, 'cx_out'

    cy_lo = stats['cy_rel']['mean'] - n_sigma * stats['cy_rel']['std']
    cy_hi = stats['cy_rel']['mean'] + n_sigma * stats['cy_rel']['std']
    if not (cy_lo <= cy_norm <= cy_hi):
        return False, 'cy_out'

    return True, 'in_bounds'


def run_val_with_filter(model, data_yaml, stats, conf_thresh, dist_thresh=30, iou_thresh=0.1, suppress_factor=0.05):
    """推理 val + 应用几何过滤 + 评估 F1（GT 用原图坐标系）"""
    import yaml
    from PIL import Image
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_imgs_dir = Path(cfg['path']) / cfg['val']
    val_lbls_dir = Path(cfg['path']) / 'labels' / 'val'

    TP, FP, FN = 0, 0, 0

    for img_path in sorted(val_imgs_dir.glob('*.png')):
        lbl_path = val_lbls_dir / (img_path.stem + '.txt')
        # 读原图尺寸用于 GT 坐标转换
        try:
            orig_w, orig_h = Image.open(img_path).size
        except Exception:
            orig_w, orig_h = 1024, 1024
        gt_list = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        gt_list.append((cx * orig_w, cy * orig_h, w * orig_w, h * orig_h))

        result = model.predict(str(img_path), imgsz=1024, conf=0.001, verbose=False)
        boxes = result[0].boxes
        if len(boxes) == 0:
            FN += len(gt_list)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # 应用几何过滤（用 normalized 坐标系）
        filtered_confs = confs.copy()
        n_filtered = 0
        for i in range(len(xyxy)):
            if stats is None:
                continue
            # 归一化到原图坐标系 [0, 1]
            cx_norm = (xyxy[i, 0] + xyxy[i, 2]) / 2 / orig_w
            cy_norm = (xyxy[i, 1] + xyxy[i, 3]) / 2 / orig_h
            w_norm = (xyxy[i, 2] - xyxy[i, 0]) / orig_w
            h_norm = (xyxy[i, 3] - xyxy[i, 1]) / orig_h
            in_bounds, reason = is_in_bounds(cx_norm, cy_norm, w_norm, h_norm, stats)
            if not in_bounds:
                filtered_confs[i] *= suppress_factor
                n_filtered += 1

        # 应用 conf 阈值
        keep = filtered_confs >= conf_thresh
        pred_boxes_f = xyxy[keep]
        pred_confs_f = filtered_confs[keep]

        gt_matched = np.zeros(len(gt_list), dtype=bool)
        if len(gt_list) > 0:
            gt_xyxy = np.array([[cx - w/2, cy - h/2, cx + w/2, cy + h/2] for cx, cy, w, h in gt_list])

        # 按 conf 降序贪心匹配
        order = np.argsort(-pred_confs_f) if len(pred_confs_f) > 0 else []
        for i in order:
            if len(gt_list) == 0:
                FP += 1
                continue
            cxs = pred_boxes_f[i, [0, 2]].mean()
            cys = pred_boxes_f[i, [1, 3]].mean()
            dists = np.hypot(gt_xyxy[:, [0, 2]].mean(axis=1) - cxs,
                            gt_xyxy[:, [1, 3]].mean(axis=1) - cys)
            ious = compute_iou_single(pred_boxes_f[i], gt_xyxy)
            match_idx = np.where((dists < dist_thresh) | (ious > iou_thresh))[0]
            match_idx = match_idx[~gt_matched[match_idx]]
            if len(match_idx) > 0:
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
                Precision=float(Precision), F1=float(F1), n_filtered=int(n_filtered))


def compute_iou_single(box, boxes):
    """box: (4,), boxes: (M, 4)"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt')
    parser.add_argument('--data', default='data/coil/data.yaml')
    parser.add_argument('--out_stats', default='/tmp/tip_geo_prior_stats.json')
    parser.add_argument('--out_results', default='/tmp/tip_geo_prior_results.json')
    parser.add_argument('--conf_thresh', type=float, default=0.15)
    args = parser.parse_args()

    print(f'=== TipGeoPrior: 几何先验硬规则过滤 ===')
    print(f'模型: {args.model}')

    # 1. 统计 GT 几何分布
    print('统计 train GT 几何分布...')
    gts = get_train_gts(args.data)
    stats = compute_geo_stats(gts)
    print(f'GT 数量: {stats["n"]}')
    print(f'  aspect ratio (w/h):    p5={stats["aspect"]["p5"]:.3f}, p50={stats["aspect"]["p50"]:.3f}, p95={stats["aspect"]["p95"]:.3f}')
    print(f'  rel_area (w*h/全图):   p5={stats["rel_area"]["p5"]:.5f}, p50={stats["rel_area"]["p50"]:.5f}, p95={stats["rel_area"]["p95"]:.5f}')
    print(f'  cx_rel (normalized):   mean={stats["cx_rel"]["mean"]:.3f}, std={stats["cx_rel"]["std"]:.3f}')
    print(f'  cy_rel (normalized):   mean={stats["cy_rel"]["mean"]:.3f}, std={stats["cy_rel"]["std"]:.3f}')

    with open(args.out_stats, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  统计已保存: {args.out_stats}')

    # 2. 评估过滤前后 F1
    print(f'\n评估 V18.3 部署 F1 @ conf={args.conf_thresh} ...')
    model = YOLO(args.model)

    # baseline (无过滤)
    print('  baseline (无过滤)...')
    base_metrics = run_val_with_filter(model, args.data, stats=None, conf_thresh=args.conf_thresh, suppress_factor=1.0)
    print(f"  baseline: F1={base_metrics['F1']:.4f} R={base_metrics['Recall']:.4f} P={base_metrics['Precision']:.4f} TP={base_metrics['TP']} FP={base_metrics['FP']} FN={base_metrics['FN']}")

    # 应用几何过滤
    print('  + 几何过滤...')
    filtered_metrics = run_val_with_filter(model, args.data, stats=stats, conf_thresh=args.conf_thresh, suppress_factor=0.05)
    print(f"  + 几何过滤: F1={filtered_metrics['F1']:.4f} R={filtered_metrics['Recall']:.4f} P={filtered_metrics['Precision']:.4f} TP={filtered_metrics['TP']} FP={filtered_metrics['FP']} FN={filtered_metrics['FN']} (n_filtered={filtered_metrics['n_filtered']})")

    # 3. 保存结果
    output = dict(
        model=args.model,
        conf_thresh=args.conf_thresh,
        baseline=base_metrics,
        geo_filtered=filtered_metrics,
        improvement_F1=filtered_metrics['F1'] - base_metrics['F1'],
    )
    with open(args.out_results, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n结果已保存: {args.out_results}')
    print(f'\n=== 总结 ===')
    print(f'baseline: F1={base_metrics["F1"]:.4f}')
    print(f'+ 几何过滤: F1={filtered_metrics["F1"]:.4f}')
    print(f'提升: {output["improvement_F1"]*100:+.2f}pp F1 (过滤了 {filtered_metrics["n_filtered"]} 个候选)')


if __name__ == '__main__':
    main()