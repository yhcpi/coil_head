"""扫描 conf_thresh × 匹配阈值网格，找业务最优工作点。

支持 3 种匹配模式：
- 'lenient' : 中心距离 < dist_thr 算命中（创新点 6 主推）
- 'mac'     : Min-Area Coverage >= mac_thr 算命中（用户新加）
- 'combined': lenient OR mac 任一满足即命中（最宽容）

逻辑：
1. 一次性跑完 raw 预测（max_det=300, conf=0.001）
2. 每图取 top1（conf 最高）作为部署候选
3. 笛卡尔积扫 (conf_thresh, 匹配模式, 匹配阈值)，输出每组合的 TP/FP/FN/TN/Recall/Precision/F1
4. 写 markdown 报告
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
from lenient_eval import (
    yolo_to_xyxy, compute_iou, compute_center_dist, compute_mac,
)


def eval_top1_one(top1, img_gts, mode, iou_thresh, dist_thresh, mac_thresh):
    """top1: (conf, x1, y1, x2, y2) 或 None。返回 ('tp'|'fp'|'fn'|'tn')。"""
    if top1 is None:
        return ('fn' if img_gts else 'tn')
    if mode == 'iou':
        best = -1
        for g in img_gts:
            iou = compute_iou(top1[1:5], g[1:5])
            if iou > best:
                best = iou
        passed = best >= iou_thresh
    elif mode == 'lenient':
        best = 1e9
        for g in img_gts:
            d = compute_center_dist(top1[1:5], g[1:5])
            if d < best:
                best = d
        passed = best < dist_thresh
    elif mode == 'mac':
        best = -1
        for g in img_gts:
            m = compute_mac(top1[1:5], g[1:5])
            if m > best:
                best = m
        passed = best >= mac_thresh
    elif mode == 'combined':
        # Lenient OR MAC
        best_d, best_m = 1e9, -1
        for g in img_gts:
            d = compute_center_dist(top1[1:5], g[1:5])
            m = compute_mac(top1[1:5], g[1:5])
            if d < best_d: best_d = d
            if m > best_m: best_m = m
        passed = (best_d < dist_thresh) or (best_m >= mac_thresh)
    else:
        raise ValueError(f'unknown mode: {mode}')

    if img_gts:
        return ('tp' if passed else 'fn')
    else:
        return 'fp'


def stats_from_labels(n_img, top1_by_img, gts_by_img, conf_thr, mode,
                      iou_thresh, dist_thr, mac_thr):
    tp = fp = fn = tn = 0
    for img_idx in range(n_img):
        t1 = top1_by_img[img_idx]
        if t1 is not None and t1[0] < conf_thr:
            t1_eff = None
        else:
            t1_eff = t1
        label = eval_top1_one(
            t1_eff, gts_by_img[img_idx],
            mode=mode, iou_thresh=iou_thresh,
            dist_thresh=dist_thr, mac_thresh=mac_thr,
        )
        if label == 'tp': tp += 1
        elif label == 'fp': fp += 1
        elif label == 'fn': fn += 1
        else: tn += 1
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': recall, 'precision': precision, 'f1': f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--val_dir', default='/home/pi/projects/hyperyolo/data/coil/images/val')
    p.add_argument('--gt_dir', default='/home/pi/projects/hyperyolo/data/coil/labels/val')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--iou_thresh', type=float, default=0.5)
    p.add_argument('--max_det', type=int, default=300)
    p.add_argument('--conf', type=float, default=0.001)
    # 扫描范围
    p.add_argument('--conf_list', default='0.001,0.05,0.1,0.15,0.2,0.3,0.5,0.7',
                   help='top1 conf 阈值列表')
    p.add_argument('--dist_list', default='20,30,50,80,120',
                   help='Lenient 距离阈值列表')
    p.add_argument('--mac_list', default='0.3,0.5,0.7',
                   help='MAC 阈值列表')
    p.add_argument('--scan_modes', default='lenient,mac,combined',
                   help='要扫描的匹配模式（逗号分隔）')
    p.add_argument('--out_md', default='/home/pi/projects/hyperyolo/docs/scan_top1_thresholds_lastpt.md')
    args = p.parse_args()

    conf_list = [float(x) for x in args.conf_list.split(',')]
    dist_list = [float(x) for x in args.dist_list.split(',')]
    mac_list = [float(x) for x in args.mac_list.split(',')]
    scan_modes = args.scan_modes.split(',')

    print(f'加载模型: {args.weights}')
    model = YOLO(args.weights)
    val_imgs = sorted(Path(args.val_dir).glob('*.png'))
    gt_dir = Path(args.gt_dir)
    print(f'val 集: {len(val_imgs)} 张')

    # 1) raw 预测
    raw_by_img = []
    gts_by_img = []
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=args.conf, imgsz=args.imgsz,
                               max_det=args.max_det, verbose=False)[0]
        W, H = Image.open(img_p).size
        raw = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw.append((float(box.conf[0]), x1, y1, x2, y2, int(box.cls[0])))
        raw_by_img.append(raw)

        gt_p = gt_dir / f'{img_p.stem}.txt'
        gts = yolo_to_xyxy(gt_p, W, H)
        gts_by_img.append(gts)

        if (img_idx + 1) % 30 == 0:
            print(f'  [{img_idx+1}/{len(val_imgs)}]')

    n_img = len(val_imgs)
    n_pos = sum(1 for g in gts_by_img if g)
    n_neg = n_img - n_pos
    print(f'\n汇总: {n_img} 张 ({n_pos} 正样本 + {n_neg} 负样本)')

    # 2) 预计算每图的 top1
    top1_by_img = []
    for img_idx, raw in enumerate(raw_by_img):
        if not raw:
            top1_by_img.append(None)
        else:
            top1_by_img.append(max(raw, key=lambda x: x[0]))

    # 3) 扫描：每种模式独立跑
    all_results = {}  # mode → list of result dict
    for mode in scan_modes:
        if mode == 'lenient':
            grid = [(c, d, None) for c in conf_list for d in dist_list]
        elif mode == 'mac':
            grid = [(c, None, m) for c in conf_list for m in mac_list]
        elif mode == 'combined':
            grid = [(c, d, m) for c in conf_list for d in dist_list for m in mac_list]
        else:
            raise ValueError(f'unknown mode: {mode}')

        results = []
        for c_thr, d_thr, m_thr in grid:
            r = stats_from_labels(
                n_img, top1_by_img, gts_by_img,
                conf_thr=c_thr, mode=mode,
                iou_thresh=args.iou_thresh,
                dist_thr=d_thr if d_thr is not None else 30.0,
                mac_thr=m_thr if m_thr is not None else 0.5,
            )
            r['conf_thr'] = c_thr
            r['dist_thr'] = d_thr
            r['mac_thr'] = m_thr
            results.append(r)
        all_results[mode] = results

    # 4) 打印每个模式的最佳 + 完整表格
    bests = {}
    print('\n' + '=' * 110)
    print('扫描结果汇总（last.pt，每图 top1）')
    print('=' * 110)

    for mode, results in all_results.items():
        print(f'\n--- 模式: {mode} ---')
        if mode == 'lenient':
            print(f'{"conf_thr":>10} {"dist_thr":>10} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
                  f'{"Recall":>8} {"Precision":>10} {"F1":>8}')
            print('-' * 80)
            for r in results:
                print(f'{r["conf_thr"]:>10.3f} {r["dist_thr"]:>10.1f} '
                      f'{r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
                      f'{r["recall"]:>8.4f} {r["precision"]:>10.4f} {r["f1"]:>8.4f}')
        elif mode == 'mac':
            print(f'{"conf_thr":>10} {"mac_thr":>10} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
                  f'{"Recall":>8} {"Precision":>10} {"F1":>8}')
            print('-' * 80)
            for r in results:
                print(f'{r["conf_thr"]:>10.3f} {r["mac_thr"]:>10.2f} '
                      f'{r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
                      f'{r["recall"]:>8.4f} {r["precision"]:>10.4f} {r["f1"]:>8.4f}')
        elif mode == 'combined':
            print(f'{"conf_thr":>10} {"dist_thr":>10} {"mac_thr":>10} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} '
                  f'{"Recall":>8} {"Precision":>10} {"F1":>8}')
            print('-' * 100)
            for r in results:
                print(f'{r["conf_thr"]:>10.3f} {r["dist_thr"]:>10.1f} {r["mac_thr"]:>10.2f} '
                      f'{r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
                      f'{r["recall"]:>8.4f} {r["precision"]:>10.4f} {r["f1"]:>8.4f}')

        bests[mode] = max(results, key=lambda r: r['f1'])
        b = bests[mode]
        if mode == 'combined':
            print(f'\n>>> 最佳：conf={b["conf_thr"]}, dist={b["dist_thr"]}, '
                  f'mac={b["mac_thr"]} → F1={b["f1"]:.4f} '
                  f'(R={b["recall"]:.4f}, P={b["precision"]:.4f})')
        elif mode == 'lenient':
            print(f'\n>>> 最佳：conf={b["conf_thr"]}, dist={b["dist_thr"]} → '
                  f'F1={b["f1"]:.4f} (R={b["recall"]:.4f}, P={b["precision"]:.4f})')
        elif mode == 'mac':
            print(f'\n>>> 最佳：conf={b["conf_thr"]}, mac={b["mac_thr"]} → '
                  f'F1={b["f1"]:.4f} (R={b["recall"]:.4f}, P={b["precision"]:.4f})')

    # 5) IoU 对照
    r_iou = stats_from_labels(
        n_img, top1_by_img, gts_by_img,
        conf_thr=0.001, mode='iou',
        iou_thresh=args.iou_thresh, dist_thr=0, mac_thr=0,
    )
    print(f'\n>>> 对照 IoU>=0.5：TP={r_iou["tp"]} FP={r_iou["fp"]} '
          f'FN={r_iou["fn"]} TN={r_iou["tn"]} → F1={r_iou["f1"]:.4f} '
          f'(R={r_iou["recall"]:.4f}, P={r_iou["precision"]:.4f})')

    # 6) 写 markdown
    md = []
    md.append('# conf_thresh × 匹配阈值 扫描结果（last.pt，部署口径）\n')
    md.append(f'- 权重：`{args.weights}`\n')
    md.append(f'- val 集：{n_img} 张（{n_pos} 正样本 + {n_neg} 负样本）\n')
    md.append(f'- 每图 top1（conf 最高），仅与同图 GT 比较\n\n')
    md.append('## IoU 对照（conf_thr=0.001）\n')
    md.append(f'- TP={r_iou["tp"]} FP={r_iou["fp"]} FN={r_iou["fn"]} TN={r_iou["tn"]} | '
              f'R={r_iou["recall"]:.4f} P={r_iou["precision"]:.4f} F1={r_iou["f1"]:.4f}\n')

    for mode, results in all_results.items():
        md.append(f'\n## 模式 {mode}\n')
        if mode == 'lenient':
            md.append('| conf_thr | dist_thr | TP | FP | FN | TN | Recall | Precision | F1 |')
            md.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for r in results:
                md.append(f'| {r["conf_thr"]:.3f} | {r["dist_thr"]:.1f} | '
                          f'{r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | '
                          f'{r["recall"]:.4f} | {r["precision"]:.4f} | {r["f1"]:.4f} |')
        elif mode == 'mac':
            md.append('| conf_thr | mac_thr | TP | FP | FN | TN | Recall | Precision | F1 |')
            md.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for r in results:
                md.append(f'| {r["conf_thr"]:.3f} | {r["mac_thr"]:.2f} | '
                          f'{r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | '
                          f'{r["recall"]:.4f} | {r["precision"]:.4f} | {r["f1"]:.4f} |')
        elif mode == 'combined':
            md.append('| conf_thr | dist_thr | mac_thr | TP | FP | FN | TN | Recall | Precision | F1 |')
            md.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for r in results:
                md.append(f'| {r["conf_thr"]:.3f} | {r["dist_thr"]:.1f} | {r["mac_thr"]:.2f} | '
                          f'{r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | '
                          f'{r["recall"]:.4f} | {r["precision"]:.4f} | {r["f1"]:.4f} |')

        b = bests[mode]
        if mode == 'combined':
            md.append(f'\n**{mode} 最佳**：`conf={b["conf_thr"]}, dist={b["dist_thr"]}, '
                      f'mac={b["mac_thr"]}` → F1={b["f1"]:.4f} '
                      f'(R={b["recall"]:.4f}, P={b["precision"]:.4f})\n')
        elif mode == 'lenient':
            md.append(f'\n**{mode} 最佳**：`conf={b["conf_thr"]}, dist={b["dist_thr"]}` → '
                      f'F1={b["f1"]:.4f} (R={b["recall"]:.4f}, P={b["precision"]:.4f})\n')
        elif mode == 'mac':
            md.append(f'\n**{mode} 最佳**：`conf={b["conf_thr"]}, mac={b["mac_thr"]}` → '
                      f'F1={b["f1"]:.4f} (R={b["recall"]:.4f}, P={b["precision"]:.4f})\n')

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text('\n'.join(md))
    print(f'\n✓ Markdown 报告已写入：{args.out_md}')


if __name__ == '__main__':
    main()