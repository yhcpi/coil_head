"""分析 best.pt 在 val 集上的 FN（漏检）案例。

输出：
  - 每个 FN 的文件名
  - GT bbox 位置
  - best.pt 的 top1 预测（位置 + conf），如果有的话
  - 漏检类型分类（无预测 vs 位置偏移 vs conf 太低）
"""
import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.001, help='推理时的最低 conf')
    p.add_argument('--max_det', type=int, default=10)
    p.add_argument('--iou_thresh', type=float, default=0.5, help='IoU-mAP 阈值（区分 IoU-FN vs Lenient-match）')
    p.add_argument('--dist_thresh', type=int, default=30, help='中心距离阈值（imgsz=1024 坐标）')
    p.add_argument('--out_md', default=None)
    return p.parse_args()


def load_gt(path):
    """读取 YOLO 格式 label: cls cx cy w h (normalized)"""
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
    """bbox [x1,y1,x2,y2] IoU"""
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


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)
    gt_dir = Path(args.gt_dir)
    val_imgs = sorted(val_dir.glob('*.png')) + sorted(val_dir.glob('*.jpg'))
    print(f'val 集: {len(val_imgs)} 张')

    model = YOLO(args.weights)

    # 收集所有 FN 案例
    fns = []
    for img_path in val_imgs:
        gt_path = gt_dir / (img_path.stem + '.txt')
        gts = load_gt(gt_path)
        if not gts:
            continue  # 负样本不算 FN

        # 推理
        results = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf,
                                max_det=args.max_det, verbose=False)[0]
        H, W = results.orig_shape  # 原图尺寸

        # GT 转 xyxy (像素坐标)
        gt_xyxy = []
        for cls, cx, cy, w, h in gts:
            x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
            gt_xyxy.append([x1, y1, x2, y2, cx*W, cy*H, w*W, h*H])

        # 预测转 xyxy
        preds = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                preds.append({'xyxy': b.tolist(), 'conf': float(c)})

        # 给每个 GT 找最佳匹配（IoU >= 阈值算 match）
        matched_pred_idx = set()
        for g in gt_xyxy:
            g_box = g[:4]
            best_iou = 0; best_pred_idx = -1
            for i, p in enumerate(preds):
                if i in matched_pred_idx:
                    continue
                iou = box_iou(g_box, p['xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            if best_pred_idx >= 0:
                matched_pred_idx.add(best_pred_idx)

        # FN = 有 GT 但没被匹配
        for g_idx, g in enumerate(gt_xyxy):
            if g_idx in [i for i, _ in enumerate(gt_xyxy) if any(
                box_iou(g[:4], p['xyxy']) >= args.iou_thresh for p in preds
            )]:
                continue  # IoU >= 阈值，跳过

            # 找最近的预测（用于分析）
            best_dist = float('inf'); best_pred = None
            for p in preds:
                d = center_dist(g[:4], p['xyxy'])
                if d < best_dist:
                    best_dist = d; best_pred = p

            # 分类 FN 类型
            if best_pred is None:
                fn_type = 'NO_PRED'  # 模型完全没预测
                info = f'no prediction above conf={args.conf}'
            elif best_dist > args.dist_thresh:
                fn_type = 'FAR_OFF'   # 预测在远处
                info = f'nearest conf={best_pred["conf"]:.3f}, dist={best_dist:.0f}px (> {args.dist_thresh})'
            else:
                # 预测位置对但 IoU 不够
                iou = box_iou(g[:4], best_pred['xyxy'])
                fn_type = 'WRONG_BOX'  # 位置对但 bbox 形状不对
                info = f'nearest conf={best_pred["conf"]:.3f}, IoU={iou:.2f}, dist={best_dist:.0f}px'

            fns.append({
                'img': img_path.name,
                'gt_cx': g[4], 'gt_cy': g[5], 'gt_w': g[6], 'gt_h': g[7],
                'gt_box_xyxy': g[:4],
                'pred_conf': best_pred['conf'] if best_pred else None,
                'pred_xyxy': best_pred['xyxy'] if best_pred else None,
                'pred_dist': best_dist if best_pred else None,
                'pred_iou': box_iou(g[:4], best_pred['xyxy']) if best_pred else None,
                'type': fn_type, 'info': info,
            })

    print(f'\n=== 总共 {len(fns)} 个 FN ===')
    type_count = {}
    for fn in fns:
        type_count[fn['type']] = type_count.get(fn['type'], 0) + 1
    print(f'分类统计: {type_count}')

    # 按类型分组输出
    md_lines = [f'# FN 分析报告（best.pt on val 集）']
    md_lines.append(f'- 总 FN: {len(fns)}')
    md_lines.append(f'- FN 类型: {type_count}')
    md_lines.append(f'- IoU 阈值: {args.iou_thresh} (低于此判 FN)')
    md_lines.append(f'- Dist 阈值: {args.dist_thresh}px')
    md_lines.append('')

    for ftype in ['NO_PRED', 'FAR_OFF', 'WRONG_BOX']:
        items = [fn for fn in fns if fn['type'] == ftype]
        if not items:
            continue
        md_lines.append(f'## {ftype} ({len(items)} 个)')
        md_lines.append('')
        md_lines.append('| 文件 | GT 中心 (x,y) 像素 | GT 尺寸 (w×h) | 预测 conf | 距离 (px) | IoU | 备注 |')
        md_lines.append('|---|---|---|---|---|---|---|')
        for fn in items:
            pconf = f"{fn['pred_conf']:.3f}" if fn['pred_conf'] is not None else '—'
            pdist = f"{fn['pred_dist']:.0f}" if fn['pred_dist'] is not None else '—'
            piou = f"{fn['pred_iou']:.2f}" if fn['pred_iou'] is not None else '—'
            md_lines.append(
                f'| `{fn["img"]}` | ({fn["gt_cx"]:.0f}, {fn["gt_cy"]:.0f}) | '
                f'{fn["gt_w"]:.0f}×{fn["gt_h"]:.0f} | '
                f'{pconf} | {pdist} | {piou} | {fn["info"]} |'
            )
        md_lines.append('')

    out = '\n'.join(md_lines)
    print(out)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(out)
        print(f'\n✓ 报告已写入：{args.out_md}')

    return fns


if __name__ == '__main__':
    main()
