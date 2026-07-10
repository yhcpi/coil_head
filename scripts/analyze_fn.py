"""分析 FN 的 top1 特征：conf 分布 + top1 距离分布。

目的：找出 FN 的子模式，给训练改进提供数据支撑。

分类：
1. conf < 0.05：模型"看不见"该目标 → 召回能力差，要回到训练阶段
2. 0.05 ≤ conf < 0.3：模型"看见但不确定" → 可以降 conf_thresh 救回一些
3. conf ≥ 0.3：模型"看见了但位置错" → top1 离 GT > 30px → 定位问题
4. 没预测（top1=None）：模型完全没出框 → 完全召回失败

输出 markdown 报告 + 列出每张 FN 图的 top1 conf 和 top1 距离 GT 的距离。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
from lenient_eval import (
    yolo_to_xyxy, compute_center_dist,
)

WEIGHTS = '/home/pi/projects/hyperyolo/repos/Hyper-YOLO/runs/coil_v3_rect_imgsz1024_unfrozen_augv2/weights/last.pt'
VAL_DIR = '/home/pi/projects/hyperyolo/data/coil/images/val'
GT_DIR = '/home/pi/projects/hyperyolo/data/coil/labels/val'


def main():
    print(f'加载模型: {WEIGHTS}')
    model = YOLO(WEIGHTS)
    val_imgs = sorted(Path(VAL_DIR).glob('*.png'))
    gt_dir = Path(GT_DIR)
    print(f'val 集: {len(val_imgs)} 张')

    # 1) 跑所有 val 图：拿 top1 + GT
    rows = []
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=0.001, imgsz=1024,
                               max_det=300, verbose=False)[0]
        W, H = Image.open(img_p).size
        # top1
        if result.boxes is not None and len(result.boxes) > 0:
            top1_conf = float(result.boxes.conf[0])
            top1_box = result.boxes.xyxy[0].cpu().numpy()
        else:
            top1_conf = 0.0
            top1_box = None
        # GT
        gt_p = gt_dir / f'{img_p.stem}.txt'
        gts = yolo_to_xyxy(gt_p, W, H)
        has_gt = len(gts) > 0

        # 计算与最近 GT 的距离（如果有 GT）
        min_dist = None
        if top1_box is not None and gts:
            dists = [compute_center_dist(top1_box, g[1:5]) for g in gts]
            min_dist = min(dists)

        rows.append({
            'img_idx': img_idx,
            'stem': img_p.stem,
            'has_gt': has_gt,
            'top1_conf': top1_conf,
            'top1_box': top1_box,
            'min_dist_to_gt': min_dist,
        })

        if (img_idx + 1) % 30 == 0:
            print(f'  [{img_idx+1}/{len(val_imgs)}]')

    # 2) 分类 FN（按 conf_thr=0.05 的口径）
    FN = [r for r in rows if r['has_gt'] and (
        r['top1_conf'] < 0.05  # 模型看不见
    )]
    # 进一步细分 FN：conf≥0.05 但距离>30px 算"位置错"；conf<0.05 算"看不见"
    FN_blind = [r for r in FN if r['top1_conf'] < 0.05]
    # 用 Lenient 口径（dist<30）算 FN
    FN_LENIENT = [r for r in rows if r['has_gt'] and (
        r['top1_conf'] < 0.05 or
        (r['min_dist_to_gt'] is not None and r['min_dist_to_gt'] >= 30)
    )]
    FP = [r for r in rows if not r['has_gt'] and r['top1_conf'] >= 0.05]
    TN = [r for r in rows if not r['has_gt'] and r['top1_conf'] < 0.05]
    TP = [r for r in rows if r['has_gt'] and r['top1_conf'] >= 0.05 and
          (r['min_dist_to_gt'] is not None and r['min_dist_to_gt'] < 30)]

    print(f'\n=== 分类（conf_thr=0.05, dist_thr=30）===')
    print(f'TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}, TN: {len(TN)}')
    print(f'  其中 FN_blind (conf<0.05): {len(FN_blind)}')
    print(f'  其中 FN_loc (conf≥0.05 但距离>30px): {len(FN) - len(FN_blind)}')

    # 3) 输出 FN 详情
    print(f'\n=== 29 个 FN 详情（按 conf 升序）===')
    FN_sorted = sorted(FN, key=lambda r: r['top1_conf'])
    print(f'{"stem":<35} {"conf":>8} {"min_dist":>10}')
    print('-' * 60)
    for r in FN_sorted:
        md = f'{r["min_dist_to_gt"]:.1f}' if r['min_dist_to_gt'] is not None else 'no_pred'
        print(f'{r["stem"]:<35} {r["top1_conf"]:>8.4f} {md:>10}')

    # 4) 写到 markdown
    md = []
    md.append('# last.pt val FN 分析（部署口径 conf_thr=0.05, dist_thr=30）\n')
    md.append(f'- 权重：`{WEIGHTS}`\n')
    md.append(f'- val 集：{len(val_imgs)} 张（{sum(1 for r in rows if r["has_gt"])} 正样本 + '
              f'{sum(1 for r in rows if not r["has_gt"])} 负样本）\n\n')

    md.append('## 整体计数\n')
    md.append(f'- **TP={len(TP)}**，**FP={len(FP)}**，**FN={len(FN)}**，**TN={len(TN)}**\n')
    md.append(f'- Recall={len(TP)/max(len(TP)+len(FN),1):.4f}, '
              f'Precision={len(TP)/max(len(TP)+len(FP),1):.4f}, '
              f'F1={2*len(TP)/max(2*len(TP)+len(FP)+len(FN),1):.4f}\n\n')

    md.append('## FN 子分类\n')
    md.append(f'- **FN_blind**（模型完全看不见，conf<0.05）：{len(FN_blind)} 张\n')
    md.append(f'- **FN_loc**（看见了但定位错，conf≥0.05 但距离>30px）：{len(FN)-len(FN_blind)} 张\n\n')

    md.append('## 29 个 FN 列表（按 conf 升序）\n')
    md.append('| 图像名 | top1 conf | 与最近 GT 中心距离(像素) | 子类 |\n')
    md.append('|---|---:|---:|---|')
    for r in FN_sorted:
        md_v = f'{r["min_dist_to_gt"]:.1f}' if r['min_dist_to_gt'] is not None else 'no_top1'
        kind = 'blind' if r['top1_conf'] < 0.05 else 'loc'
        md.append(f'| {r["stem"]} | {r["top1_conf"]:.4f} | {md_v} | {kind} |')

    md.append('\n## FN 模式观察\n')
    # conf 分布
    confs = [r['top1_conf'] for r in FN]
    md.append(f'- FN conf 范围：[{min(confs):.4f}, {max(confs):.4f}]\n')
    md.append(f'- FN conf 中位数：{np.median(confs):.4f}\n')
    md.append(f'- FN 中 conf=0（无 top1）的有：{sum(1 for c in confs if c == 0)} 张\n')
    md.append(f'- FN 中 conf∈[0, 0.05) 的有：{sum(1 for c in confs if 0 <= c < 0.05)} 张\n')
    md.append(f'- FN 中 conf∈[0.05, 0.3) 的有：{sum(1 for c in confs if 0.05 <= c < 0.3)} 张\n')
    md.append(f'- FN 中 conf≥0.3 的有：{sum(1 for c in confs if c >= 0.3)} 张\n\n')

    md.append('## 训练改进建议（基于 FN 分析）\n')
    if len(FN_blind) >= len(FN) * 0.7:
        md.append('**主要瓶颈：FN_blind 占绝大多数（' +
                  f'{len(FN_blind)}/{len(FN)} = {len(FN_blind)/max(len(FN),1):.0%}）**\n')
        md.append('- 模型**根本看不见这些目标**——召回能力是核心问题\n')
        md.append('- **建议**：加训练数据（重点加这些 FN 图像）；考虑创新点 3（Heatmap Head）；多尺度训练\n')
    else:
        md.append(f'**主要瓶颈：FN_loc 较多（{len(FN)-len(FN_blind)}/{len(FN)}）**\n')
        md.append('- 模型看见了但定位错——定位精度是核心问题\n')
        md.append('- **建议**：创新点 2（Coverage Loss）；创新点 1（PBBR 概率回归）；NWD Loss\n')

    out_path = '/home/pi/projects/hyperyolo/docs/fn_analysis_lastpt.md'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text('\n'.join(md))
    print(f'\n✓ Markdown 报告已写入：{out_path}')


if __name__ == '__main__':
    main()