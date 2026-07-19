#!/usr/bin/env python3
"""
RelabelGT — V18.3 反向审计 545 张训练集 GT
目标：用 V18.3 在训练集上推理，找高 conf 但 GT 缺失的"漏标高可疑"图像

输出：
  - /tmp/relabel_gt_audit_candidates.json (A 类漏标候选 + B 类边界候选 + GT 文件冲突清单)
  - 控制台打印候选图像清单和统计

用法：
  python scripts/relabel_gt_audit.py \
    --model runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt \
    --data data/coil/data.yaml \
    --imgsz 1024 \
    --conf_thresh 0.30 \
    --dist_thresh 50
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


def get_train_images_and_gts(data_yaml, imgsz=1024):
    """返回 train 集所有图路径 + GT (xywh pixel) per 图"""
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    train_imgs_dir = Path(cfg['path']) / cfg['train']
    train_lbls_dir = Path(cfg['path']) / 'labels' / 'train'

    items = []
    for img_path in sorted(train_imgs_dir.glob('*.png')):
        # 跳过 hard neg 副本 (hn*_XXX)
        if img_path.stem.startswith('hn'):
            continue
        lbl_path = train_lbls_dir / (img_path.stem + '.txt')
        gt = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = map(float, parts[:5])
                        gt.append((cx * imgsz, cy * imgsz, w * imgsz, h * imgsz))
        items.append((img_path, lbl_path, gt))
    return items


def compute_center_dist(box, gt_list):
    """box: (cx, cy, w, h), gt_list: list of (cx, cy, w, h)"""
    if len(gt_list) == 0:
        return []
    bx, by = box[0], box[1]
    dists = []
    for cx, cy, _, _ in gt_list:
        dists.append(np.hypot(bx - cx, by - cy))
    return dists


def audit_train_set(model, items, conf_thresh=0.30, dist_thresh=50):
    """
    A 类：图像 GT=空 (gt_count=0) 且模型报高 conf (>=conf_thresh) → 漏标高可疑
    B 类：图像有 GT，模型在 GT 距离 >= dist_thresh 处另报高 conf (>=conf_thresh) → 边界可疑 / 冗余
    C 类：图像 GT 数为 0 但模型预测 conf < conf_thresh → 暂时放过
    """
    candidates = {'A': [], 'B': [], 'C': []}

    for img_path, lbl_path, gt_list in items:
        result = model.predict(str(img_path), imgsz=1024, conf=0.001, verbose=False)
        boxes = result[0].boxes
        if len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # 候选 = conf >= conf_thresh (默认 0.30)
        keep = confs >= conf_thresh
        if keep.sum() == 0:
            continue

        for i in np.where(keep)[0]:
            cx, cy = (xyxy[i, 0] + xyxy[i, 2]) / 2, (xyxy[i, 1] + xyxy[i, 3]) / 2
            w, h = xyxy[i, 2] - xyxy[i, 0], xyxy[i, 3] - xyxy[i, 1]

            dists = compute_center_dist((cx, cy, w, h), gt_list)
            min_dist = min(dists) if dists else 1e9

            if len(gt_list) == 0:
                # A 类：图无 GT，模型报高 conf → 真正的漏标高可疑
                candidates['A'].append({
                    'img': img_path.name,
                    'lbl': str(lbl_path),
                    'conf': float(confs[i]),
                    'box_xyxy': xyxy[i].tolist(),
                    'gt_count': 0,
                    'min_dist_to_gt': None,
                    'reason': 'gt_empty_but_model_high_conf',
                })
            elif min_dist >= dist_thresh:
                # B 类：图有 GT，但模型另报高 conf 在远处 → 边界可疑 / 多 tip / 误报
                candidates['B'].append({
                    'img': img_path.name,
                    'lbl': str(lbl_path),
                    'conf': float(confs[i]),
                    'box_xyxy': xyxy[i].tolist(),
                    'gt_count': len(gt_list),
                    'min_dist_to_gt': float(min_dist),
                    'reason': 'gt_exists_but_distant_high_conf',
                })
            # 否则 (gt_count > 0 且 min_dist < dist_thresh): 这是正确预测，跳过

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt')
    parser.add_argument('--data', default='data/coil/data.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--conf_thresh', type=float, default=0.30)
    parser.add_argument('--dist_thresh', type=int, default=50)
    parser.add_argument('--out', default='/tmp/relabel_gt_audit_candidates.json')
    args = parser.parse_args()

    print(f'=== RelabelGT: V18.3 反向审计 train 集 ===')
    print(f'模型: {args.model}')
    print(f'conf_thresh={args.conf_thresh}, dist_thresh={args.dist_thresh}')

    model = YOLO(args.model)
    items = get_train_images_and_gts(args.data, imgsz=args.imgsz)
    print(f'train 集 (排除 hn* 副本): {len(items)} 张')

    candidates = audit_train_set(model, items, conf_thresh=args.conf_thresh, dist_thresh=args.dist_thresh)

    print(f'\n=== 候选统计 ===')
    print(f'A 类 (空 GT + 模型高 conf → 真漏标高可疑): {len(candidates["A"])} 个')
    print(f'B 类 (有 GT 但远处另报高 conf → 边界可疑):   {len(candidates["B"])} 个')

    # 按图像去重
    A_imgs = sorted(set(c['img'] for c in candidates['A']))
    B_imgs = sorted(set(c['img'] for c in candidates['B']))
    print(f'\nA 类涉及图数: {len(A_imgs)}')
    print(f'B 类涉及图数: {len(B_imgs)}')

    if A_imgs:
        print(f'\n=== A 类候选图清单（建议优先人工审核）===')
        for img in A_imgs[:30]:
            cs = [c for c in candidates['A'] if c['img'] == img]
            max_conf = max(c['conf'] for c in cs)
            print(f'  {img}  ({len(cs)} 个 FP, max_conf={max_conf:.3f})')
        if len(A_imgs) > 30:
            print(f'  ... 还有 {len(A_imgs) - 30} 张')

    # 保存
    output = dict(
        model=args.model,
        conf_thresh=args.conf_thresh,
        dist_thresh=args.dist_thresh,
        n_train_imgs=len(items),
        stats=dict(
            A_count=len(candidates['A']),
            B_count=len(candidates['B']),
            A_imgs=len(A_imgs),
            B_imgs=len(B_imgs),
        ),
        A_candidates=candidates['A'],
        B_candidates=candidates['B'],
    )
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n结果已保存到: {args.out}')


if __name__ == '__main__':
    main()