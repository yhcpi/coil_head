"""SAHI 推理评估：对 val 集跑 SAHI 切片推理，算 mAP。

策略：
1. 加载训练好的模型（ultralytics YOLO 格式）
2. 用 SAHI 库 get_sliced_prediction 对每张 val 图切片推理 + NMS 合并
3. 把 SAHI 结果转回 YOLO 评估格式，与原 label 比
4. 输出 mAP50 / mAP50-95 / Recall / Precision

依据：sahi/predict.py: get_sliced_prediction + get_prediction
"""
import sys
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/sahi')
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from sahi.predict import get_sliced_prediction, get_prediction
from sahi import AutoDetectionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='模型权重 .pt')
    p.add_argument('--data_yaml', default='/home/pi/projects/hyperyolo/data/coil_sahi/data.yaml',
                   help='data.yaml（用于取 val 路径和 names）')
    p.add_argument('--val_dir', default=None,
                   help='覆盖 data.yaml 的 val 路径（用于跨域评估）')
    p.add_argument('--gt_dir', default=None,
                   help='覆盖 GT 路径（默认从 val_dir 推断为 ../labels/{split}/）')
    p.add_argument('--slice_h', type=int, default=640)
    p.add_argument('--slice_w', type=int, default=640)
    p.add_argument('--overlap', type=float, default=0.2)
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def load_val_images(data_yaml: str, val_dir_override=None):
    """读 data.yaml 拿 val 集路径列表。val_dir_override 用于跨域评估。"""
    import yaml
    cfg = yaml.safe_load(open(data_yaml))
    if val_dir_override:
        val_dir = Path(val_dir_override)
    else:
        val_dir = Path(cfg['path']) / cfg['val']
    images = sorted(val_dir.glob('*.png'))
    return images, cfg.get('names', {0: 'coil_head'})


def yolo_labels_to_xyxy(label_path: Path, W: int, H: int, conf: float = 1.0):
    """读 YOLO txt → (N, 5) [cls, x1, y1, x2, y2]"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    out = []
    for line in label_path.read_text().strip().split('\n'):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        out.append([cls, cx - w/2, cy - h/2, cx + w/2, cy + h/2, conf])
    return out


def compute_iou(box, boxes):
    """box [x1,y1,x2,y2] vs boxes (N,4)。返回 IoU (N,)。"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a1 + a2 - inter
    return np.where(union > 0, inter / union, 0)


def voc_ap(rec, prec):
    """11-point AP (PASCAL VOC)。"""
    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        if not mask.any():
            p = 0
        else:
            p = prec[mask].max()
        ap += p / 11
    return ap


def evaluate(model, val_images, names, args):
    """对 val 集跑推理 + 算 mAP50 / mAP50-95 / Recall / Precision。"""
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # mAP50-95

    # 累积所有预测和标签
    all_preds = []  # [(cls, conf, x1, y1, x2, y2)]
    all_gts = []    # [(cls, x1, y1, x2, y2, img_id)]
    img_dims = []   # [(W, H), ...]

    # GT 路径：用户指定 > 自动推断（{val_dir}/../labels/{split}/）
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        # 自动推断：假设 val_dir 在 data/<name>/images/val/ → labels 在 data/<name>/labels/val/
        gt_dir = val_images[0].parent.parent.parent / 'labels' / val_images[0].parent.name

    print(f'推理 {len(val_images)} 张 val 图（SAHI 切片）...')
    print(f'GT 目录: {gt_dir}')
    for img_idx, img_p in enumerate(val_images):
        img = Image.open(img_p)
        W, H = img.size
        img_dims.append((W, H))

        # 真实标签
        lbl_p = gt_dir / f'{img_p.stem}.txt'
        for gt in yolo_labels_to_xyxy(lbl_p, W, H):
            all_gts.append((*gt[:5], img_idx))

        # SAHI 推理
        result = get_sliced_prediction(
            image=str(img_p),
            detection_model=model,
            slice_height=args.slice_h,
            slice_width=args.slice_w,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=True,  # 加全图推理一起 NMS
            postprocess_type='NMS',
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
            verbose=0,
        )

        # 钢卷头部场景：每张图要么 0 个要么 1 个目标 → SAHI 合并后只保留 conf 最高的 1 个预测
        top_pred = max(result.object_prediction_list, key=lambda p: p.score.value, default=None)
        if top_pred is not None:
            bbox = top_pred.bbox
            score = top_pred.score.value
            cls_id = top_pred.category.id
            all_preds.append((cls_id, score, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))

        if (img_idx + 1) % 20 == 0:
            print(f'  [{img_idx+1}/{len(val_images)}] 当前累计 {len(all_preds)} preds, {len(all_gts)} gts')

    print(f'\n总计: {len(all_preds)} preds, {len(all_gts)} gts')

    # 算 mAP
    cls_list = sorted(set(g[0] for g in all_gts) | set(p[0] for p in all_preds))
    if not cls_list:
        print('⚠️ 没有目标或预测')
        return

    ap50_per_cls = {}
    ap_per_cls = {}

    for cls in cls_list:
        cls_gts = [g for g in all_gts if g[0] == cls]
        cls_preds = sorted([p for p in all_preds if p[0] == cls], key=lambda x: -x[1])

        # 按 IoU 阈值评估
        aps = []
        for iou_thr in iou_thresholds:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = [False] * len(cls_gts)
            for pi, p in enumerate(cls_preds):
                if not cls_gts:
                    fp[pi] = 1
                    continue
                # 找同图 + IoU 最大的 GT
                best_iou = 0
                best_gt = -1
                for gi, g in enumerate(cls_gts):
                    if matched[gi] or g[5] != p[0] and False:  # 同 cls 已经在 cls_gts 里
                        continue
                    if g[5] != (None if False else None):
                        pass
                    iou = compute_iou(p[2:6], np.array([[g[1], g[2], g[3], g[4]]]))[0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi
                if best_iou >= iou_thr:
                    tp[pi] = 1
                    if best_gt >= 0:
                        matched[best_gt] = True
                else:
                    fp[pi] = 1
            if len(cls_gts) == 0:
                ap = 0
            else:
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                rec = cum_tp / len(cls_gts)
                prec = cum_tp / (cum_tp + cum_fp + 1e-10)
                ap = voc_ap(rec, prec)
            aps.append(ap)

        cls_name = names[cls] if isinstance(names, list) and 0 <= cls < len(names) else (names.get(cls, str(cls)) if isinstance(names, dict) else str(cls))
        ap50_per_cls[cls_name] = aps[0]
        ap_per_cls[cls_name] = np.mean(aps)
        print(f'  cls {cls_name}: mAP50={aps[0]:.4f}, mAP50-95={np.mean(aps):.4f}')

    mAP50 = np.mean(list(ap50_per_cls.values()))
    mAP50_95 = np.mean(list(ap_per_cls.values()))

    # 简单算 overall recall/precision（IoU=0.5 匹配）
    n_gt = len(all_gts)
    matched_gt = set()
    tp_total, fp_total = 0, 0
    for p in sorted(all_preds, key=lambda x: -x[1]):
        if not all_gts:
            fp_total += 1
            continue
        best_iou, best_gi = 0, -1
        for gi, g in enumerate(all_gts):
            if gi in matched_gt:
                continue
            if g[0] != p[0]:
                continue
            iou = compute_iou(p[2:6], np.array([[g[1], g[2], g[3], g[4]]]))[0]
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= 0.5:
            tp_total += 1
            matched_gt.add(best_gi)
        else:
            fp_total += 1
    recall = tp_total / max(n_gt, 1)
    precision = tp_total / max(tp_total + fp_total, 1)

    print(f'\n=== SAHI 推理评估结果 ===')
    print(f'  mAP50:     {mAP50:.4f}')
    print(f'  mAP50-95:  {mAP50_95:.4f}')
    print(f'  Recall:    {recall:.4f}  ({tp_total}/{n_gt})')
    print(f'  Precision: {precision:.4f}  ({tp_total}/{tp_total+fp_total})')


def main():
    args = parse_args()

    # 加载模型
    print(f'加载模型: {args.weights}')
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # ultralytics
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    val_images, names = load_val_images(args.data_yaml, val_dir_override=args.val_dir)
    print(f'val 集: {len(val_images)} 张, names={names}')

    evaluate(model, val_images, names, args)


if __name__ == '__main__':
    main()