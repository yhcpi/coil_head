"""对比多个 best.pt 在同一 val 集上的逐图预测。

目的：验证三个 loss 配置是否真的训出了"等价模型"，
还是只是 TP/FP 整数统计巧合。
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')
from ultralytics import YOLO


def load_gt(path):
    if not Path(path).exists():
        return []
    out = []
    for line in open(path).read().strip().split('\n'):
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        out.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return out


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def center_dist(b1, b2):
    cx1 = (b1[0]+b1[2])/2; cy1 = (b1[1]+b1[3])/2
    cx2 = (b2[0]+b2[2])/2; cy2 = (b2[1]+b2[3])/2
    return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5


def predict_top1(model, img_path, imgsz, conf):
    """返回 top1 预测：[box(xyxy), conf] 或 None"""
    r = model.predict(str(img_path), imgsz=imgsz, conf=conf, max_det=10, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    return {'xyxy': boxes[best].tolist(), 'conf': float(confs[best])}


def predict_all_top1(model, val_imgs, imgsz, conf):
    """对每张图返回 top1（如果有），None if no detection"""
    out = []
    for img_p in val_imgs:
        r = predict_top1(model, img_p, imgsz, conf)
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', nargs='+', required=True,
                   help='要对比的 best.pt 列表（空格分隔）')
    p.add_argument('--names', nargs='+', required=True,
                   help='每份权重的名字（空格分隔）')
    p.add_argument('--val_dir', required=True)
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.05)
    args = p.parse_args()

    if len(args.weights) != len(args.names):
        print('weights 和 names 数量必须相等')
        return

    val_imgs = sorted(Path(args.val_dir).glob('*.png'))
    print(f'val 集: {len(val_imgs)} 张')
    print(f'conf 阈值: {args.conf}')

    # 加载所有模型
    models = []
    for w in args.weights:
        print(f'加载: {w}')
        models.append(YOLO(w))

    # 对每张图跑所有模型
    print('\n推理...')
    n_imgs = len(val_imgs)
    n_models = len(models)
    all_preds = [[None] * n_models for _ in range(n_imgs)]
    for i, img_p in enumerate(val_imgs):
        for j, model in enumerate(models):
            all_preds[i][j] = predict_top1(model, img_p, args.imgsz, args.conf)

    # 加载 GT
    gt_dir = Path(args.gt_dir)
    all_gts = []
    for img_p in val_imgs:
        lbl = gt_dir / (img_p.stem + '.txt')
        gts = load_gt(lbl)
        # 转 xyxy (像素)
        img = None
        # 仅一次读图
        all_gts.append(gts)

    # 每个模型分别算 TP/FP/FN
    print('\n=== 每个模型的逐图结果 ===')
    summaries = []
    for j, name in enumerate(args.names):
        tp = fp = fn = tn = 0
        for i, img_p in enumerate(val_imgs):
            gt = all_gts[i]
            pred = all_preds[i][j]
            if gt:
                # 有 GT
                gt_xyxy = [(g[1]-g[3]/2)*img_p.stat().st_size and 0  # 我们需要图尺寸
                           for g in gt]  # 不对，用 cv2
                # 简化：直接用 cv2 读尺寸
                import cv2
                img = cv2.imread(str(img_p))
                H, W = img.shape[:2]
                gt_xyxy = []
                for cls, cx, cy, w, h in gt:
                    gt_xyxy.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
                if pred is None:
                    fn += 1
                else:
                    # IoU 是否 ≥ 0.5 OR Lenient (center_dist < 30)
                    iou = box_iou(gt_xyxy[0], pred['xyxy'])
                    dist = center_dist(gt_xyxy[0], pred['xyxy'])
                    if iou >= 0.5 or dist < 30:
                        tp += 1
                    else:
                        fn += 1
            else:
                # 无 GT（负样本）
                if pred is None:
                    tn += 1
                else:
                    fp += 1
        rec = tp/(tp+fn) if tp+fn else 0
        prec = tp/(tp+fp) if tp+fp else 0
        f1 = 2*rec*prec/(rec+prec) if rec+prec else 0
        print(f'{name}: TP={tp} FP={fp} FN={fn} TN={tn} | R={rec:.4f} P={prec:.4f} F1={f1:.4f}')
        summaries.append({'name': name, 'tp': tp, 'fp': fp, 'fn': fn,
                          'rec': rec, 'prec': prec, 'f1': f1})

    # 找 3 个模型的 FN 案例集
    print('\n=== FN 交集分析（每个模型漏检哪些图）===')
    fn_sets = []
    for j, name in enumerate(args.names):
        fns = []
        for i, img_p in enumerate(val_imgs):
            if not all_gts[i]:
                continue
            pred = all_preds[i][j]
            if pred is None:
                fns.append(img_p.stem)
                continue
            import cv2
            img = cv2.imread(str(img_p)); H, W = img.shape[:2]
            cls, cx, cy, w, h = all_gts[i][0]
            gt_xyxy = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
            iou = box_iou(gt_xyxy, pred['xyxy'])
            dist = center_dist(gt_xyxy, pred['xyxy'])
            if iou < 0.5 and dist >= 30:
                fns.append(img_p.stem)
        fn_sets.append(set(fns))
        print(f'{name} ({len(fns)} FN): {sorted(fns)}')

    # 分析异同
    if len(fn_sets) >= 2:
        all_fn = set()
        for s in fn_sets:
            all_fn |= s
        common_fn = fn_sets[0]
        for s in fn_sets[1:]:
            common_fn &= s
        print(f'\n所有模型都漏检: {sorted(common_fn)} ({len(common_fn)} 个)')
        any_fn = all_fn - common_fn
        print(f'部分模型漏检: {sorted(any_fn)} ({len(any_fn)} 个)')

    # 分析预测是否完全一致（同图同 top1 预测框）
    if len(models) >= 2:
        print('\n=== 预测框对比（同图同 top1 框）===')
        same_box = 0; diff_box = 0
        for i in range(n_imgs):
            preds_i = [all_preds[i][j] for j in range(n_models)]
            if all(p is None for p in preds_i):
                continue
            # 看 conf 数值差（即使 box 一样 conf 可能不同）
            confs = [p['conf'] if p else -1 for p in preds_i]
            conf_max = max(confs); conf_min = min(c for c in confs if c >= 0)
            conf_range = conf_max - conf_min

            boxes = [p['xyxy'] if p else None for p in preds_i]
            if all(b is None for b in boxes):
                continue
            # 计算 IoU 最大 box pair
            from itertools import combinations
            iou_pairs = []
            for (bi, bj) in combinations(range(n_models), 2):
                if boxes[bi] and boxes[bj]:
                    iou_pairs.append(box_iou(boxes[bi], boxes[bj]))
            if iou_pairs:
                if min(iou_pairs) > 0.95:
                    same_box += 1
                else:
                    diff_box += 1

        print(f'图同（IoU>0.95）: {same_box}/{same_box+diff_box}')
        print(f'图不同: {diff_box}')

        # conf 范围分布
        conf_ranges = []
        for i in range(n_imgs):
            preds_i = [all_preds[i][j] for j in range(n_models)]
            confs = [p['conf'] if p else None for p in preds_i]
            valid = [c for c in confs if c is not None]
            if len(valid) >= 2:
                conf_ranges.append(max(valid) - min(valid))
        if conf_ranges:
            conf_ranges = np.array(conf_ranges)
            print(f'\n同图多模型 conf 范围:')
            print(f'  均值: {conf_ranges.mean():.4f}')
            print(f'  中位: {np.median(conf_ranges):.4f}')
            print(f'  最大: {conf_ranges.max():.4f}')
            print(f'  <0.05: {(conf_ranges < 0.05).sum()}/{len(conf_ranges)}')
            print(f'  <0.10: {(conf_ranges < 0.10).sum()}/{len(conf_ranges)}')


if __name__ == '__main__':
    main()
