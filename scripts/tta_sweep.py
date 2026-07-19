"""离线 TTA sweep：从 tta_inference.py 保存的 JSON 读取预测，
对 baseline/builtin/custom 做 conf sweep，对 custom 做 WBF-IoU sweep。
不占 GPU（纯 CPU 后处理），评估口径 = Lenient-Match top1 (dist=30)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
from lenient_eval import eval_top1_deploy, yolo_to_xyxy
from tta_inference import wbf_from_candidates

VAL_DIR = Path('/home/pi/projects/hyperyolo/data/coil/images/val')
GT_DIR = Path('/home/pi/projects/hyperyolo/data/coil/labels/val')
CONF_GRID = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_gts():
    val_imgs = sorted(VAL_DIR.glob('*.png'))
    all_gts = []
    for i, p in enumerate(val_imgs):
        W, H = Image.open(p).size
        for g in yolo_to_xyxy(GT_DIR / f'{p.stem}.txt', W, H):
            all_gts.append((i, *g))
    return all_gts, len(val_imgs)


def to_preds(mode_json):
    """[[{conf,x1..}]] -> [(img_idx, conf, x1,y1,x2,y2,cls)]"""
    out = []
    for i, img in enumerate(mode_json):
        for r in img:
            out.append((i, r['conf'], r['x1'], r['y1'], r['x2'], r['y2'], int(r['cls'])))
    return out


def conf_sweep(preds, all_gts, label):
    print(f'\n[{label}] conf sweep (Lenient-Match, dist=30)')
    print(f'{"conf":>6} {"TP":>4} {"FP":>4} {"FN":>4} {"TN":>4} {"Recall":>8} {"Prec":>8} {"F1":>8}')
    best = None
    for thr in CONF_GRID:
        pf = [p for p in preds if p[1] >= thr]
        r = eval_top1_deploy(pf, all_gts, mode='lenient', dist_thresh=30.0)
        mark = ''
        if best is None or r['f1'] > best[1]['f1']:
            best = (thr, r)
        print(f'{thr:>6.3f} {r["tp"]:>4} {r["fp"]:>4} {r["fn"]:>4} {r["tn"]:>4} '
              f'{r["recall"]:>8.4f} {r["precision"]:>8.4f} {r["f1"]:>8.4f}{mark}')
    b = best[1]
    print(f'  >>> best F1={b["f1"]:.4f} @ conf={best[0]} (R={b["recall"]:.4f} P={b["precision"]:.4f} '
          f'TP={b["tp"]} FP={b["fp"]} FN={b["fn"]})')
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--wbf_sweep', action='store_true')
    args = ap.parse_args()

    d = json.load(open(args.json))
    all_gts, n_img = load_gts()
    print(f'val={n_img} 张, GT 正样本={len(all_gts)}, weights={d["weights"]}')

    results = {}
    for mode in ['baseline', 'builtin', 'custom']:
        results[mode] = conf_sweep(to_preds(d[mode]), all_gts, mode)

    if args.wbf_sweep and 'custom_candidates' in d:
        print('\n' + '=' * 60)
        print('WBF-IoU sweep (custom candidates 重新合并)')
        print('=' * 60)
        cands = [[(r['conf'], r['x1'], r['y1'], r['x2'], r['y2'], int(r['cls']))
                  for r in img] for img in d['custom_candidates']]
        for wbf_iou in [0.45, 0.55, 0.65]:
            merged = [wbf_from_candidates(c, wbf_iou) for c in cands]
            preds = []
            for i, img in enumerate(merged):
                for r in img:
                    preds.append((i, r[0], r[1], r[2], r[3], r[4], int(r[5])))
            results[f'custom_wbf{wbf_iou}'] = conf_sweep(
                preds, all_gts, f'custom WBF-IoU={wbf_iou}')

    print('\n' + '=' * 60)
    print('汇总：各配置最优 F1')
    print('=' * 60)
    print(f'{"配置":<24} {"bestF1":>8} {"conf":>6} {"Recall":>8} {"Prec":>8}')
    for k, (thr, r) in results.items():
        print(f'{k:<24} {r["f1"]:>8.4f} {thr:>6.3f} {r["recall"]:>8.4f} {r["precision"]:>8.4f}')


if __name__ == '__main__':
    main()
