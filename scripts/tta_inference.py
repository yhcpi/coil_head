"""TTA (Test-Time Augmentation) 推理 + 评估

对 v4 best.pt 在 val 102 张图上跑 3 种配置：
  - baseline：单次原图推理（imgsz=1024, rect=True）
  - tta_builtin：ultralytics 内置 augment=True（multiscale + flip）
  - tta_custom：自定义多尺度 + 水平翻转 + WBF 合并

输出：
  - tta_predictions.json：每张图的 TTA 合并后预测（与 lenient_eval.py 格式兼容）
  - 控制台打印：与 baseline 对比的 4 口径指标（IoU/Lenient/MAC/Combined）

回退：删脚本即可，不影响训练代码
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

from ultralytics import YOLO

from lenient_eval import (eval_top1_deploy, eval_one_mode, compute_iou,
                          compute_center_dist, compute_mac, voc_ap,
                          yolo_to_xyxy)


def predict_single(model, img_path, imgsz, conf, max_det, verbose=False):
    """单图推理，返回 list of (conf, x1, y1, x2, y2, cls)"""
    result = model.predict(str(img_path), conf=conf, imgsz=imgsz,
                           max_det=max_det, verbose=verbose, rect=True)[0]
    out = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_v = float(box.conf[0])
            cls_v = int(box.cls[0])
            out.append((conf_v, x1, y1, x2, y2, cls_v))
    return out


def predict_with_flip(model, img_path, imgsz, conf, max_det, verbose=False):
    """原图 + 水平翻转推理，flip 后坐标还原（W=原图宽）

    Returns merged list of (conf, x1, y1, x2, y2, cls) 去重（按中心距离 < 30px）
    """
    # 原图推理
    orig_preds = predict_single(model, img_path, imgsz, conf, max_det, verbose)

    # 水平翻转推理（用 PIL 翻转后保存临时文件，再调用 model.predict）
    img = Image.open(img_path)
    W, H = img.size
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

    import tempfile, os
    tmp_path = None
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        flipped.save(tmp.name)
        tmp_path = tmp.name

    try:
        flip_preds = predict_single(model, tmp_path, imgsz, conf, max_det, verbose)
    finally:
        os.unlink(tmp_path)

    # 还原 flipped 坐标：x_new = W - x
    flipped_back = []
    for conf_v, x1, y1, x2, y2, cls_v in flip_preds:
        nx1, nx2 = W - x2, W - x1
        flipped_back.append((conf_v, nx1, y1, nx2, y2, cls_v))

    # 合并：把 flipped 也加入候选
    all_preds = orig_preds + flipped_back
    return all_preds


def wbf_merge(boxes_list, scores_list, iou_thresh=0.55, weights=None,
              skip_box_thr=0.001):
    """Weighted Box Fusion (简化版) — 多路预测合并

    简化：每路 conf 权重相同，按 conf × IoU 合并重叠框
    boxes_list: list of np.array(N, 4) xyxy
    scores_list: list of np.array(N,)
    返回: (merged_boxes, merged_scores)
    """
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros((0,))

    all_boxes = np.concatenate(boxes_list, axis=0)
    all_scores = np.concatenate(scores_list, axis=0)

    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    # 按 conf 降序
    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    all_scores = all_scores[order]

    merged_boxes = []
    merged_scores = []
    used = np.zeros(len(all_boxes), dtype=bool)

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        if all_scores[i] < skip_box_thr:
            break
        # 当前最大 conf 框
        cur_box = all_boxes[i].copy()
        cur_score = all_scores[i]
        # 找与之 IoU > thresh 的所有框，合并坐标加权
        cluster_boxes = [cur_box]
        cluster_scores = [cur_score]
        used[i] = True

        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            iou = compute_iou(cur_box, all_boxes[j])
            if iou > iou_thresh:
                cluster_boxes.append(all_boxes[j])
                cluster_scores.append(all_scores[j])
                used[j] = True

        # 加权融合
        cluster_boxes = np.array(cluster_boxes)
        cluster_scores = np.array(cluster_scores)
        weights = cluster_scores / cluster_scores.sum()
        merged_box = (cluster_boxes * weights[:, None]).sum(axis=0)
        merged_score = cluster_scores.mean()  # 平均
        merged_boxes.append(merged_box)
        merged_scores.append(merged_score)

    return np.array(merged_boxes), np.array(merged_scores)


def wbf_from_candidates(candidates, wbf_iou):
    """对已有候选框做 WBF 合并，返回按 conf 降序的 (conf, x1, y1, x2, y2, cls) 列表"""
    if not candidates:
        return []
    boxes = np.array([(r[1], r[2], r[3], r[4]) for r in candidates])
    scores = np.array([r[0] for r in candidates])
    merged_boxes, merged_scores = wbf_merge([boxes], [scores], iou_thresh=wbf_iou)
    merged = [(float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]),
               candidates[0][5])
              for b, s in zip(merged_boxes, merged_scores)]
    return sorted(merged, key=lambda x: -x[0])


def run_tta_custom(model, val_imgs, gt_dir, imgsz, conf, max_det, wbf_iou=0.55):
    """自定义 TTA：3 路推理（original / 1.25x scale / hflip）→ WBF 合并

    Returns: (all_raw_preds, all_candidates) — 后者是 WBF 前的候选框，供离线 WBF-IoU sweep
    """
    all_raw_preds = []   # 整个 val 的合并后预测
    all_candidates = []  # 每张图 WBF 前的原始候选

    for img_idx, img_p in enumerate(val_imgs):
        W, H = Image.open(img_p).size

        # 路 1: 原图
        p1 = predict_single(model, str(img_p), imgsz, conf, max_det)

        # 路 2: imgsz ×1.25 (放大，适合小目标)
        p2 = predict_single(model, str(img_p), int(imgsz * 1.25), conf, max_det)

        # 路 3: 水平翻转 (predict_with_flip 已返回合并候选)
        p3 = predict_with_flip(model, str(img_p), imgsz, conf, max_det)

        # 合并所有候选
        candidates = p1 + p2 + p3
        all_candidates.append(candidates)
        all_raw_preds.append(wbf_from_candidates(candidates, wbf_iou))

    return all_raw_preds, all_candidates


def run_tta_builtin(model, val_imgs, conf, max_det):
    """ultralytics 内置 TTA: augment=True"""
    all_raw_preds = []
    for img_p in val_imgs:
        result = model.predict(str(img_p), conf=conf, augment=True,
                               max_det=max_det, verbose=False, rect=True)[0]
        out = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_v = float(box.conf[0])
                cls_v = int(box.cls[0])
                out.append((conf_v, x1, y1, x2, y2, cls_v))
        all_raw_preds.append(out)
    return all_raw_preds


def run_baseline(model, val_imgs, conf, max_det):
    """baseline：单次原图推理（imgsz=1024）"""
    return [predict_single(model, str(p), 1024, conf, max_det) for p in val_imgs]


def eval_predictions(raw_preds_by_image, gts_by_image, name):
    """用 lenient_eval.py 的口径评估"""
    all_gts = [g for gs in gts_by_image for g in gs]

    # 转统一格式：list of (img_idx, conf, x1, y1, x2, y2, cls)
    def collect():
        out = []
        for img_idx, raw in enumerate(raw_preds_by_image):
            for r in raw:
                out.append((img_idx, r[0], r[1], r[2], r[3], r[4], r[5]))
        return out

    preds = collect()

    # 学术 mAP 4 口径
    print(f'\n{"=" * 80}')
    print(f'[{name}] 学术 mAP 评估')
    print(f'{"=" * 80}')
    modes = [
        ('T0: IoU-mAP',     'iou',      'iou'),
        ('T1: Lenient-mAP', 'iou',      'lenient'),
    ]
    print(f'{"模式":<25} {"mAP50":>8} {"Recall":>8} {"Precision":>10} {"TP":>4} {"FP":>4} {"FN":>4}')
    print('-' * 80)
    for n, nms_kind, eval_kind in modes:
        m, r, p, tp, fp, fn = eval_one_mode(
            preds, all_gts,
            mode=eval_kind,
            iou_thresh=0.5,
            dist_thresh=50.0,
        )
        print(f'{n:<25} {m:>8.4f} {r:>8.4f} {p:>10.4f} {tp:>4} {fp:>4} {fn:>4}')

    # 部署口径 per-image top1
    print(f'\n[{name}] 部署口径 per-image top1（conf_thresh=0.001）')
    print(f'{"=" * 80}')
    modes_top1 = [
        ('IoU-Match',         'iou'),
        ('Lenient-Match',     'lenient'),
        ('MAC-Match',         'mac'),
        ('Lenient OR MAC',    'combined'),
    ]
    print(f'{"模式":<20} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
          f'{"Recall":>8} {"Precision":>10} {"F1":>8}')
    print('-' * 80)
    metrics = {}
    for n, kind in modes_top1:
        # 应用 conf 阈值 0.001（与部署口径一致）
        preds_top1 = [p for p in preds if p[1] >= 0.001]
        r = eval_top1_deploy(
            preds_top1, all_gts,
            mode=kind,
            iou_thresh=0.5,
            dist_thresh=30.0,
            mac_thresh=0.5,
        )
        print(f'{n:<20} {r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
              f'{r["recall"]:>8.4f} {r["precision"]:>10.4f} {r["f1"]:>8.4f}')
        metrics[n] = r
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--val_dir', default='/home/pi/projects/hyperyolo/data/coil/images/val')
    p.add_argument('--gt_dir', default='/home/pi/projects/hyperyolo/data/coil/labels/val')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--max_det', type=int, default=300)
    p.add_argument('--mode', default='all', choices=['baseline', 'builtin', 'custom', 'all'],
                   help='跑哪些配置；all=三个都跑并对比')
    p.add_argument('--wbf_iou', type=float, default=0.55,
                   help='custom TTA 的 WBF 合并 IoU 阈值')
    p.add_argument('--save_json', default='/home/pi/projects/hyperyolo/runs/coil_loss_ablation/tta_predictions.json',
                   help='保存 TTA 合并后预测的 JSON 路径')
    args = p.parse_args()

    print(f'[TTA] 加载模型: {args.weights}')
    model = YOLO(args.weights)

    val_imgs = sorted(Path(args.val_dir).glob('*.png'))
    gt_dir = Path(args.gt_dir)
    print(f'[TTA] val 集: {len(val_imgs)} 张')

    # 加载 GT
    gts_by_image = []
    for img_idx, img_p in enumerate(val_imgs):
        W, H = Image.open(img_p).size
        gt_p = gt_dir / f'{img_p.stem}.txt'
        gts = yolo_to_xyxy(gt_p, W, H)
        gts_by_image.append([(img_idx, *g) for g in gts])

    # ========== Baseline ==========
    baseline_metrics = None
    baseline_preds = None
    if args.mode in ['baseline', 'all']:
        print(f'\n{"#" * 80}')
        print(f'[#1] Baseline (无 TTA，单次推理 imgsz={args.imgsz})')
        print(f'{"#" * 80}')
        t0 = time.time()
        baseline_preds = run_baseline(model, val_imgs, args.conf, args.max_det)
        print(f'  推理耗时: {time.time()-t0:.1f}s')
        baseline_metrics = eval_predictions(baseline_preds, gts_by_image, 'Baseline')

    # ========== TTA builtin (ultralytics augment=True) ==========
    builtin_metrics = None
    builtin_preds = None
    if args.mode in ['builtin', 'all']:
        print(f'\n{"#" * 80}')
        print(f'[#2] TTA builtin (ultralytics augment=True: scale=[1, 0.83, 0.67] × flip=[None, lr])')
        print(f'{"#" * 80}')
        t0 = time.time()
        builtin_preds = run_tta_builtin(model, val_imgs, args.conf, args.max_det)
        print(f'  推理耗时: {time.time()-t0:.1f}s')
        builtin_metrics = eval_predictions(builtin_preds, gts_by_image, 'TTA-builtin')

    # ========== TTA custom (3 路：原图 + 1.25x + hflip) ==========
    custom_metrics = None
    custom_preds = None
    if args.mode in ['custom', 'all']:
        print(f'\n{"#" * 80}')
        print(f'[#3] TTA custom (3 路: scale=[1.0, 1.25] × flip=[None, lr] → WBF 合并)')
        print(f'{"#" * 80}')
        t0 = time.time()
        custom_preds, custom_candidates = run_tta_custom(
            model, val_imgs, gt_dir, args.imgsz, args.conf, args.max_det,
            wbf_iou=args.wbf_iou)
        print(f'  推理耗时: {time.time()-t0:.1f}s')
        custom_metrics = eval_predictions(custom_preds, gts_by_image, 'TTA-custom')

    # ========== 对比总结 ==========
    if args.mode == 'all':
        print(f'\n{"=" * 80}')
        print(f'[对比总结] 部署口径 Lenient-Match (dist_thresh=30)')
        print(f'{"=" * 80}')
        print(f'{"配置":<25} {"Recall":>8} {"Precision":>10} {"F1":>8} {"FN":>4}')
        print('-' * 60)
        for name, m in [('Baseline', baseline_metrics),
                        ('TTA-builtin', builtin_metrics),
                        ('TTA-custom', custom_metrics)]:
            if m:
                key = 'Lenient-Match'
                rec = m[key]['recall']
                prec = m[key]['precision']
                f1 = m[key]['f1']
                fn = m[key]['fn']
                print(f'{name:<25} {rec:>8.4f} {prec:>10.4f} {f1:>8.4f} {fn:>4}')

        # 保存所有模式预测 + custom WBF 前候选（供离线 conf/WBF-IoU sweep）
        def dump(raw):
            return [[{'conf': r[0], 'x1': r[1], 'y1': r[2], 'x2': r[3], 'y2': r[4],
                      'cls': r[5]} for r in img] for img in (raw or [])]

        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'config': f'custom TTA: scale=[1.0,1.25]×flip=[None,lr] → WBF(iou={args.wbf_iou})',
            'weights': str(args.weights),
            'wbf_iou': args.wbf_iou,
            'baseline': dump(baseline_preds),
            'builtin': dump(builtin_preds),
            'custom': dump(custom_preds),
            'custom_candidates': dump(custom_candidates),
        }
        with open(args.save_json, 'w') as f:
            json.dump(save_data, f, ensure_ascii=False, default=float)
        print(f'\n[保存] 全模式预测 + candidates 已写入: {args.save_json}')


if __name__ == '__main__':
    main()
