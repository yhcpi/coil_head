"""Lenient-mAP 评估 + 4 口径 ablation 矩阵

一个脚本、一次推理，跑出 4 套指标对比：
  T0：标准 IoU-NMS + IoU-mAP（对照组）
  T1：标准 IoU-NMS + Lenient-mAP（创新点 6 单独）
  T2：Soft-NMS-Cov + IoU-mAP（创新点 5 单独）
  T3：Soft-NMS-Cov + Lenient-mAP（创新点 5+6 组合）

每个口径输出：mAP50、Recall、Precision、TP/FP/FN

数据：v3 val 集
权重：用户指定
"""
import argparse
import sys
import csv
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')
from lenient_nms import soft_nms_coverage, iou_nms
from ultralytics import YOLO


def yolo_to_xyxy(label_path, W, H):
    """读 YOLO txt → list of (cls, x1, y1, x2, y2) 像素坐标"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    out = []
    for line in label_path.read_text().strip().split('\n'):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
        out.append((cls, x1, y1, x2, y2))
    return out


def compute_iou(b1, b2):
    """IoU for two boxes (4-vec each)"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def compute_center_dist(b1, b2):
    """两个 bbox 中心点的像素距离"""
    c1x = (b1[0] + b1[2]) / 2; c1y = (b1[1] + b1[3]) / 2
    c2x = (b2[0] + b2[2]) / 2; c2y = (b2[1] + b2[3]) / 2
    return np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)


def compute_inter_area(b1, b2):
    """两个 bbox 的交集面积（像素²）"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_mac(b1, b2):
    """Min-Area Coverage：intersection / min(area_A, area_B)。

    含义：
      - 较小框完全被较大框包住 → MAC = 1.0
      - 两框不相交 → MAC = 0
      - 与 IoU 区别：IoU 用 union 作分母；MAC 用较小面积作分母 → 对"小框套大框"更宽容

    典型场景：
      - 预测紧贴目标（bbox 较小）+ GT 宽松包住（bbox 较大）
        IoU ≈ 0.4（差），MAC = 1.0（极好）
    """
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    inter = compute_inter_area(b1, b2)
    return inter / max(min(a1, a2), 1e-9)


def voc_ap(rec, prec):
    """11-point VOC AP"""
    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        p = prec[mask].max() if mask.any() else 0
        ap += p / 11
    return ap


def eval_top1_deploy(preds, gts, mode='lenient', iou_thresh=0.5, dist_thresh=50,
                     mac_thresh=0.5, conf_thresh=0.001):
    """Per-image top1 部署口径评估（针对"每张图最多 1 个目标"场景）。

    Args:
        preds: list of (img_idx, conf, x1, y1, x2, y2, cls) — 已经过 max_det=1 + conf_thresh 过滤
        gts: list of (img_idx, cls, x1, y1, x2, y2)
        mode:
          - 'iou'      : IoU >= iou_thresh 算命中
          - 'lenient'  : 中心距离 < dist_thresh 算命中（创新点 6 主推）
          - 'mac'      : Min-Area Coverage >= mac_thresh 算命中
          - 'combined' : lenient OR mac 任一满足即命中（最宽容）
        conf_thresh: 业务 conf 阈值（默认 0.001，低于它不算"有目标"）

    Returns:
        dict: {'tp', 'fp', 'fn', 'tn', 'recall', 'precision', 'f1', 'top1_conf_mean'}
    """
    # 按图分组
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p[0], []).append(p)
    gts_by_img = {}
    for g in gts:
        gts_by_img.setdefault(g[0], []).append(g)

    tp = fp = fn = tn = 0
    matched_top1_confs = []
    fp_top1_confs = []
    fn_no_pred = []  # 有 GT 但无 top1

    all_img_idx = sorted(set(list(preds_by_img.keys()) + list(gts_by_img.keys())))
    for img_idx in all_img_idx:
        img_preds = preds_by_img.get(img_idx, [])
        img_gts = gts_by_img.get(img_idx, [])

        # 选 top1（按 conf 降序取第一个）
        top1 = None
        if img_preds:
            top1 = max(img_preds, key=lambda x: x[1])

        if top1 is None:
            # 模型"无目标"判定
            if img_gts:
                fn += len(img_gts)  # 漏检
                fn_no_pred.append(img_idx)
            else:
                tn += 1  # 正确静默
            continue

        # top1 与 GT 比较
        matched_gt = False
        if mode == 'iou':
            best_score = -1
            for g in img_gts:
                iou = compute_iou(top1[2:6], g[2:6])
                if iou > best_score:
                    best_score = iou
            passed = best_score >= iou_thresh
        elif mode == 'mac':
            best_score = -1
            for g in img_gts:
                mac = compute_mac(top1[2:6], g[2:6])
                if mac > best_score:
                    best_score = mac
            passed = best_score >= mac_thresh
        elif mode == 'combined':
            # lenient OR mac：任一满足即命中
            best_dist = 1e9
            best_mac = -1
            for g in img_gts:
                d = compute_center_dist(top1[2:6], g[2:6])
                m = compute_mac(top1[2:6], g[2:6])
                if d < best_dist: best_dist = d
                if m > best_mac: best_mac = m
            passed = (best_dist < dist_thresh) or (best_mac >= mac_thresh)
        else:  # lenient（默认）
            best_score = 1e9
            for g in img_gts:
                d = compute_center_dist(top1[2:6], g[2:6])
                if d < best_score:
                    best_score = d
            passed = best_score < dist_thresh

        if img_gts:
            if passed:
                tp += 1
                matched_top1_confs.append(top1[1])
            else:
                fn += len(img_gts)  # 漏检（top1 没对上）
                fp_top1_confs.append(top1[1])  # 实际是漏但写成 FP
        else:
            if passed:
                # 实际无 GT 但 top1 命中（不可能 lenient+iou 撞框） → 算 FP
                fp += 1
                fp_top1_confs.append(top1[1])
            else:
                fp += 1  # 误报
                fp_top1_confs.append(top1[1])

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'recall': recall, 'precision': precision, 'f1': f1,
        'top1_conf_mean_tp': np.mean(matched_top1_confs) if matched_top1_confs else 0,
        'top1_conf_mean_fp': np.mean(fp_top1_confs) if fp_top1_confs else 0,
    }


def collect_preds_top1(raw_preds_by_image, nms_kind, dist_thresh_nms, iou_thresh):
    """Per-image top1：每张图最多 1 个预测（conf 最高）"""
    out = []
    for img_idx, raw in enumerate(raw_preds_by_image):
        if not raw:
            continue
        boxes = np.array([(r[1], r[2], r[3], r[4]) for r in raw])
        scores = np.array([r[0] for r in raw])
        if nms_kind == 'iou':
            kb, ks = iou_nms(boxes, scores, iou_thresh=iou_thresh, max_output=1)
        elif nms_kind == 'soft_coverage':
            kb, ks = soft_nms_coverage(boxes, scores, dist_thresh_px=dist_thresh_nms, max_output=1)
        else:
            kb, ks = boxes, scores
            order = np.argsort(-scores)[:1]
            kb, ks = kb[order], ks[order]
        for b, s in zip(kb, ks):
            out.append((img_idx, s, b[0], b[1], b[2], b[3], raw[0][5]))
    return out


def eval_one_mode(preds, gts, mode='iou', iou_thresh=0.5, dist_thresh=50):
    """单一评估口径，返回 (mAP50, Recall, Precision, TP, FP, FN)。"""
    cls_list = sorted(set(g[1] for g in gts) | set(p[6] for p in preds))
    ap_per_cls = {}

    for cls in cls_list:
        cls_gts = [g for g in gts if g[1] == cls]
        cls_preds = sorted([p for p in preds if p[6] == cls], key=lambda x: -x[1])

        if not cls_preds:
            ap_per_cls[cls] = 0.0
            continue
        if not cls_gts:
            ap_per_cls[cls] = 0.0
            continue

        tp = np.zeros(len(cls_preds))
        matched = [False] * len(cls_gts)
        for pi, p in enumerate(cls_preds):
            best_score = -1 if mode == 'iou' else 1e9
            best_gi = -1
            for gi, g in enumerate(cls_gts):
                if matched[gi] or g[0] != p[0]:
                    continue
                pbox = p[2:6]; gbox = g[2:6]
                if mode == 'iou':
                    s = compute_iou(pbox, gbox)
                    if s > best_score:
                        best_score = s; best_gi = gi
                else:  # lenient: 中心距离，越小越好
                    s = compute_center_dist(pbox, gbox)
                    if s < best_score:
                        best_score = s; best_gi = gi
            threshold = iou_thresh if mode == 'iou' else dist_thresh
            passed = (best_score >= threshold) if mode == 'iou' else (best_score < threshold)
            if passed:
                tp[pi] = 1
                if best_gi >= 0:
                    matched[best_gi] = True

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        rec = cum_tp / len(cls_gts)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap_per_cls[cls] = voc_ap(rec, prec)

    mAP50 = np.mean(list(ap_per_cls.values())) if ap_per_cls else 0

    # 总 TP/FP/FN（不严格按类，重在为单一类别给出 Recall/Precision）
    tp_total = 0; fp_total = 0; fn_total = 0
    # 按所有 GT 看是否被任何预测匹配
    matched_global = set()
    for p in sorted(preds, key=lambda x: -x[1]):
        if not gts:
            fp_total += 1
            continue
        best_score = -1 if mode == 'iou' else 1e9
        best_gi = -1
        for gi, g in enumerate(gts):
            if gi in matched_global:
                continue
            if g[1] != p[6]:
                continue
            pbox = p[2:6]; gbox = g[2:6]
            if mode == 'iou':
                s = compute_iou(pbox, gbox)
            else:
                s = compute_center_dist(pbox, gbox)
            score_is_better = (s > best_score) if mode == 'iou' else (s < best_score)
            if score_is_better:
                best_score = s; best_gi = gi
        threshold = iou_thresh if mode == 'iou' else dist_thresh
        passed = (best_score >= threshold) if mode == 'iou' else (best_score < threshold)
        if passed:
            tp_total += 1
            matched_global.add(best_gi)
        else:
            fp_total += 1
    fn_total = len(gts) - len(matched_global)

    recall = tp_total / max(tp_total + fn_total, 1)
    precision = tp_total / max(tp_total + fp_total, 1)
    return mAP50, recall, precision, tp_total, fp_total, fn_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--val_dir', default='/home/pi/projects/hyperyolo/data/coil/images/val')
    p.add_argument('--gt_dir', default='/home/pi/projects/hyperyolo/data/coil/labels/val')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou_thresh', type=float, default=0.5)
    p.add_argument('--dist_thresh_nms', type=float, default=30.0,
                   help='Soft-NMS-Cov 的距离衰减阈值（imgsz=1024 坐标系下像素）')
    p.add_argument('--dist_thresh_eval', type=float, default=50.0,
                   help='Lenient-mAP 的"中心距离 < X 像素算匹配"')
    p.add_argument('--mac_thresh', type=float, default=0.5,
                   help='MAC 模式的"intersection/min(areaA,areaB) >= X" 算命中（0~1）')
    p.add_argument('--max_det', type=int, default=300)
    p.add_argument('--mode', default='both',
                   choices=['pr', 'top1', 'both'],
                   help='pr=PR-curves 学术 mAP；top1=per-image 部署口径；both=两者都跑')
    p.add_argument('--top1_conf_thresh', type=float, default=0.001,
                   help='部署口径下的 conf 阈值（top1 conf < 此值视为"无目标"）')
    args = p.parse_args()

    print(f'加载模型: {args.weights}')
    model = YOLO(args.weights)

    val_imgs = sorted(Path(args.val_dir).glob('*.png'))
    gt_dir = Path(args.gt_dir)
    print(f'val 集: {len(val_imgs)} 张')

    # 全量预测（不做 NMS，让 eval 自己用不同 NMS 跑）
    raw_preds_by_image = []
    gts_by_image = []
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=args.conf, imgsz=args.imgsz,
                               max_det=args.max_det, verbose=False)[0]
        W, H = Image.open(img_p).size
        raw = []  # list of (score, x1, y1, x2, y2, cls)
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_v = float(box.conf[0])
                cls_v = int(box.cls[0])
                raw.append((conf_v, x1, y1, x2, y2, cls_v))
        raw_preds_by_image.append(raw)

        # GT
        gt_p = gt_dir / f'{img_p.stem}.txt'
        gts = yolo_to_xyxy(gt_p, W, H)
        gts_by_image.append([(img_idx, *g) for g in gts])

        if (img_idx + 1) % 30 == 0:
            n_pos_gt = sum(1 for g in gts_by_image[-1] if g)
            print(f'  [{img_idx+1}/{len(val_imgs)}]')

    print(f'\n总预测 (per image, raw): {sum(len(p) for p in raw_preds_by_image)}')

    # 把 raw 预测转成统一格式：list of (img_idx, score, x1, y1, x2, y2, cls)
    def collect_preds(nms_kind):
        out = []
        for img_idx, raw in enumerate(raw_preds_by_image):
            if not raw:
                continue
            boxes = np.array([(r[1], r[2], r[3], r[4]) for r in raw])
            scores = np.array([r[0] for r in raw])
            if nms_kind == 'iou':
                kb, ks = iou_nms(boxes, scores, iou_thresh=args.iou_thresh)
            elif nms_kind == 'soft_coverage':
                kb, ks = soft_nms_coverage(boxes, scores, dist_thresh_px=args.dist_thresh_nms)
            else:  # 'none'
                kb, ks = boxes, scores
            for b, s in zip(kb, ks):
                out.append((img_idx, s, b[0], b[1], b[2], b[3], raw[0][5]))  # cls
        return out

    all_gts = [g for gs in gts_by_image for g in gs]

    # ========== 学术 mAP 口径（PR-curves 11 点 VOC AP）==========
    modes_pr = [
        ('T0: 标准 IoU-NMS + IoU-mAP',     'iou',           'iou'),
        ('T1: 标准 IoU-NMS + Lenient-mAP', 'iou',           'lenient'),
        ('T2: Soft-NMS-Cov + IoU-mAP',     'soft_coverage', 'iou'),
        ('T3: Soft-NMS-Cov + Lenient-mAP', 'soft_coverage', 'lenient'),
    ]

    if args.mode in ['pr', 'both']:
        print('\n' + '=' * 80)
        print(f'[学术 mAP] 4 口径评估矩阵（weights={args.weights}）')
        print('=' * 80)
        print(f'{"模式":<38} {"mAP50":>8} {"Recall":>8} {"Precision":>10} {"TP":>4} {"FP":>4} {"FN":>4}')
        print('-' * 80)
        for name, nms_kind, eval_kind in modes_pr:
            preds = collect_preds(nms_kind)
            mAP, rec, prec, tp, fp, fn = eval_one_mode(
                preds, all_gts,
                mode=eval_kind,
                iou_thresh=args.iou_thresh,
                dist_thresh=args.dist_thresh_eval,
            )
            print(f'{name:<38} {mAP:>8.4f} {rec:>8.4f} {prec:>10.4f} {tp:>4} {fp:>4} {fn:>4}')

    # ========== 部署口径（per-image top1，每张图 ≤ 1 个目标）==========
    if args.mode in ['top1', 'both']:
        print('\n' + '=' * 80)
        print(f'[部署口径] Per-image top1（每张图最多 1 个目标，conf_thresh={args.top1_conf_thresh}）')
        print('=' * 80)

        modes_top1 = [
            ('N0: IoU-Match',                  'iou'),
            ('N1: Lenient-Match (创新点 6)',    'lenient'),
            ('N2: MAC-Match (新加指标)',        'mac'),
            ('N3: Lenient OR MAC (复合最宽容)', 'combined'),
        ]

        print(f'{"模式":<40} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
              f'{"Recall":>8} {"Precision":>10} {"F1":>8} '
              f'{"top1_conf_TP":>12} {"top1_conf_FP":>12}')
        print('-' * 120)
        for name, eval_kind in modes_top1:
            preds_top1 = collect_preds_top1(
                raw_preds_by_image, 'iou',  # NMS 在 top1 之后无关，固定用 iou
                dist_thresh_nms=args.dist_thresh_nms,
                iou_thresh=args.iou_thresh,
            )
            # 应用 conf 阈值
            preds_top1 = [p for p in preds_top1 if p[1] >= args.top1_conf_thresh]
            r = eval_top1_deploy(
                preds_top1, all_gts,
                mode=eval_kind,
                iou_thresh=args.iou_thresh,
                dist_thresh=args.dist_thresh_eval,
                mac_thresh=args.mac_thresh,
                conf_thresh=args.top1_conf_thresh,
            )
            print(f'{name:<40} {r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
                  f'{r["recall"]:>8.4f} {r["precision"]:>10.4f} {r["f1"]:>8.4f} '
                  f'{r["top1_conf_mean_tp"]:>12.4f} {r["top1_conf_mean_fp"]:>12.4f}')

    print('\n参数:')
    print(f'  IoU thresh (IoU-mAP):       {args.iou_thresh}')
    print(f'  Dist thresh NMS (创新点5):   {args.dist_thresh_nms} px (imgsz={args.imgsz} 坐标系)')
    print(f'  Dist thresh Eval (创新点6): {args.dist_thresh_eval} px')
    print(f'  top1 conf thresh:           {args.top1_conf_thresh}')


if __name__ == '__main__':
    main()
