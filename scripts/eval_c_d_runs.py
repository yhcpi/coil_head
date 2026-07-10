"""C+D 项 run 评估脚本：对每个 best.pt 跑 4 口径评估 + TTA + top-K 部署口径

输入：runs/coil_loss_ablation/{09_bayes_prior,10_paaug_motion,11_paaug_reflection,12_paaug_occlusion,13_paaug_noise}/weights/best.pt
输出：每个 run 一份 reports/{run_name}/eval.md
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
from lenient_eval import eval_top1_deploy, yolo_to_xyxy


VAL_DIR = Path('/home/pi/projects/hyperyolo/data/coil/images/val')
GT_DIR = Path('/home/pi/projects/hyperyolo/data/coil/labels/val')

# v4 baseline
V4_BEST = '/home/pi/projects/hyperyolo/runs/coil_loss_ablation/02_coverage_v4_N_model/weights/best.pt'

# C+D runs
C_D_RUNS = [
    ('v4_baseline', V4_BEST),
    ('09_bayes_prior',   'runs/coil_loss_ablation/09_bayes_prior/weights/best.pt'),
    ('10_paaug_motion',     'runs/coil_loss_ablation/10_paaug_motion/weights/best.pt'),
    ('11_paaug_reflection', 'runs/coil_loss_ablation/11_paaug_reflection/weights/best.pt'),
    ('12_paaug_occlusion',  'runs/coil_loss_ablation/12_paaug_occlusion/weights/best.pt'),
    ('13_paaug_noise',      'runs/coil_loss_ablation/13_paaug_noise/weights/best.pt'),
]


def load_gts():
    val_imgs = sorted(VAL_DIR.glob('*.png'))
    gts_by_image = []
    for img_idx, img_p in enumerate(val_imgs):
        W, H = Image.open(img_p).size
        gt_p = GT_DIR / f'{img_p.stem}.txt'
        gts = yolo_to_xyxy(gt_p, W, H)
        gts_by_image.append([(img_idx, *g) for g in gts])
    return val_imgs, [g for gs in gts_by_image for g in gs]


def predict_top1_per_image(model, val_imgs, imgsz=1024, conf=0.001, max_det=300, rect=True):
    """返回 dict[img_idx] = (conf, x1, y1, x2, y2, cls) - 仅 top1"""
    out = {}
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=conf, imgsz=imgsz, max_det=max_det,
                               verbose=False, rect=rect)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            best = max(result.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
            out[img_idx] = (float(best.conf[0]), float(x1), float(y1), float(x2), float(y2),
                            int(best.cls[0]))
        else:
            out[img_idx] = None
    return out


def predict_top_k_per_image(model, val_imgs, k=3, imgsz=1024, conf=0.001, max_det=300, rect=True):
    """返回 dict[img_idx] = list of (conf, x1, y1, x2, y2, cls) - top-K"""
    out = {}
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=conf, imgsz=imgsz, max_det=max_det,
                               verbose=False, rect=rect)[0]
        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                preds.append((float(box.conf[0]), float(x1), float(y1), float(x2), float(y2),
                              int(box.cls[0])))
        preds.sort(key=lambda x: -x[0])
        out[img_idx] = preds[:k]
    return out


def run_tta_custom(model, val_imgs, imgsz=1024, conf=0.001, max_det=300):
    """自定义 TTA：3 路 (原图/1.25x/hflip) → WBF 合并"""
    from scripts.tta_inference import run_tta_custom as _run
    return _run(model, [str(p) for p in val_imgs], GT_DIR, imgsz, conf, max_det)


def eval_topk_match(topk_preds, gts, k, conf_thresh, dist_thresh):
    """每张图 top-K 中任一命中 GT 算 TP"""
    tp = fn = fp = 0
    gts_by_img = {}
    for g in gts:
        gts_by_img.setdefault(g[0], []).append(g)
    for img_idx, preds in topk_preds.items():
        filtered = [p for p in preds if p[0] >= conf_thresh]
        if not filtered:
            if img_idx in gts_by_img:
                fn += 1
            else:
                fp += 1
            continue
        matched = False
        for p in filtered:
            cx, cy = (p[1]+p[3])/2, (p[2]+p[4])/2
            for g in gts_by_img.get(img_idx, []):
                gx, gy = (g[2]+g[4])/2, (g[3]+g[5])/2
                if ((cx-gx)**2 + (cy-gy)**2)**0.5 < dist_thresh:
                    matched = True; break
            if matched: break
        if img_idx in gts_by_img:
            if matched: tp += 1
            else: fn += 1
        else:
            fp += 1
    return tp, fp, fn


def eval_run(weights, val_imgs, all_gts, name):
    print(f'\n{"#" * 80}')
    print(f'[{name}] 加载模型: {weights}')
    print(f'{"#" * 80}')
    model = YOLO(weights)

    # === 1. Baseline 部署口径 (top-1, conf=0.05, dist=50) ===
    t0 = time.time()
    top1_preds = predict_top1_per_image(model, val_imgs, imgsz=1024, conf=0.001, max_det=300)
    t_baseline = time.time() - t0

    # 部署 top-1 @ conf=0.05
    top1_at_005 = {k: v for k, v in top1_preds.items() if v and v[0] >= 0.05}
    r_top1 = eval_top1_deploy(
        [(k, *v) for k, v in top1_at_005.items() if v],
        all_gts, mode='lenient', iou_thresh=0.5, dist_thresh=50)
    print(f'  [{name}] top-1 conf=0.05 dist=50: R={r_top1["recall"]:.4f} P={r_top1["precision"]:.4f} '
          f'F1={r_top1["f1"]:.4f}  TP={r_top1["tp"]} FP={r_top1["fp"]} FN={r_top1["fn"]}')

    # 部署 top-1 @ conf=0.10
    top1_at_010 = {k: v for k, v in top1_preds.items() if v and v[0] >= 0.10}
    r_top1_010 = eval_top1_deploy(
        [(k, *v) for k, v in top1_at_010.items() if v],
        all_gts, mode='lenient', iou_thresh=0.5, dist_thresh=50)
    print(f'  [{name}] top-1 conf=0.10 dist=50: R={r_top1_010["recall"]:.4f} P={r_top1_010["precision"]:.4f} '
          f'F1={r_top1_010["f1"]:.4f}  TP={r_top1_010["tp"]} FP={r_top1_010["fp"]} FN={r_top1_010["fn"]}')

    # === 2. Baseline 学术口径 IoU-mAP + Lenient-mAP ===
    # 复用一个内部函数：在 v8 全量预测上跑 eval_one_mode
    from lenient_eval import eval_one_mode
    full_preds = []
    for img_idx, img_p in enumerate(val_imgs):
        result = model.predict(str(img_p), conf=0.001, imgsz=1024, max_det=300, verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                full_preds.append((img_idx, float(box.conf[0]), float(x1), float(y1),
                                    float(x2), float(y2), int(box.cls[0])))

    mAP_iou, rec_iou, prec_iou, tp_iou, fp_iou, fn_iou = eval_one_mode(
        full_preds, all_gts, mode='iou', iou_thresh=0.5, dist_thresh=50)
    mAP_len, rec_len, prec_len, tp_len, fp_len, fn_len = eval_one_mode(
        full_preds, all_gts, mode='lenient', iou_thresh=0.5, dist_thresh=50)
    print(f'  [{name}] 学术 IoU-mAP={mAP_iou:.4f}  Lenient-mAP={mAP_len:.4f}')

    # === 3. TTA-custom (top-2 + conf=0.05 + dist=50) ===
    t0 = time.time()
    tta_raw = run_tta_custom(model, val_imgs, imgsz=1024, conf=0.001, max_det=300)
    t_tta = time.time() - t0
    top2_tta = {i: sorted(raw, key=lambda x: -x[0])[:2] for i, raw in enumerate(tta_raw)}

    tta_tp, tta_fp, tta_fn = eval_topk_match(top2_tta, all_gts, k=2,
                                              conf_thresh=0.05, dist_thresh=50)
    tta_recall = tta_tp / max(tta_tp + tta_fn, 1)
    tta_prec = tta_tp / max(tta_tp + tta_fp, 1)
    tta_f1 = 2 * tta_prec * tta_recall / max(tta_prec + tta_recall, 1e-9)
    print(f'  [{name}] TTA-custom top-2 conf=0.05 dist=50: R={tta_recall:.4f} P={tta_prec:.4f} '
          f'F1={tta_f1:.4f}  TP={tta_tp} FP={tta_fp} FN={tta_fn}  (TTA耗时={t_tta:.1f}s)')

    return {
        'name': name,
        'top1_conf005': r_top1,
        'top1_conf010': r_top1_010,
        'academic_iou_map': mAP_iou,
        'academic_lenient_map': mAP_len,
        'tta_top2_conf005': {'tp': tta_tp, 'fp': tta_fp, 'fn': tta_fn,
                              'recall': tta_recall, 'precision': tta_prec, 'f1': tta_f1},
        't_baseline_s': t_baseline,
        't_tta_s': t_tta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default=None, help='只评估指定 run (e.g. 09_bayes_prior)')
    p.add_argument('--output_dir', default='/home/pi/projects/hyperyolo/runs/coil_loss_ablation/cd_eval')
    args = p.parse_args()

    val_imgs, all_gts = load_gts()
    print(f'val={len(val_imgs)} 张，{len(all_gts)} 个 GT')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, weights in C_D_RUNS:
        if args.only and name != args.only:
            continue
        if not Path(weights).exists():
            print(f'[{name}] 跳过：{weights} 不存在')
            continue
        try:
            r = eval_run(weights, val_imgs, all_gts, name)
            results.append(r)
        except Exception as e:
            print(f'[{name}] 评估失败: {e}')
            import traceback; traceback.print_exc()

    # 写报告
    out_path = out_dir / f'{"all" if not args.only else args.only}.md'
    with open(out_path, 'w') as f:
        f.write(f'# C+D 项 Run 评估报告\n\n')
        f.write(f'> 模型：v4 / 09_bayes_prior / 10-13_paaug_*\n')
        f.write(f'> 数据：val 102 张，38 个 GT\n\n')
        f.write(f'## 学术 mAP\n\n')
        f.write(f'| Run | IoU-mAP | Lenient-mAP |\n')
        f.write(f'|-----|---------|-------------|\n')
        for r in results:
            f.write(f'| {r["name"]} | {r["academic_iou_map"]:.4f} | {r["academic_lenient_map"]:.4f} |\n')
        f.write(f'\n## 部署口径（per-image top-1, conf=0.05, dist=50）\n\n')
        f.write(f'| Run | TP | FP | FN | Recall | Precision | F1 |\n')
        f.write(f'|-----|----|----|----|--------|-----------|-----|\n')
        for r in results:
            t = r['top1_conf005']
            f.write(f'| {r["name"]} | {t["tp"]} | {t["fp"]} | {t["fn"]} | '
                    f'{t["recall"]:.4f} | {t["precision"]:.4f} | {t["f1"]:.4f} |\n')
        f.write(f'\n## 部署口径（per-image top-1, conf=0.10, dist=50）\n\n')
        f.write(f'| Run | TP | FP | FN | Recall | Precision | F1 |\n')
        f.write(f'|-----|----|----|----|--------|-----------|-----|\n')
        for r in results:
            t = r['top1_conf010']
            f.write(f'| {r["name"]} | {t["tp"]} | {t["fp"]} | {t["fn"]} | '
                    f'{t["recall"]:.4f} | {t["precision"]:.4f} | {t["f1"]:.4f} |\n')
        f.write(f'\n## 生产推荐：TTA-custom + top-2 + conf=0.05 + dist=50\n\n')
        f.write(f'| Run | TP | FP | FN | Recall | Precision | F1 |\n')
        f.write(f'|-----|----|----|----|--------|-----------|-----|\n')
        for r in results:
            t = r['tta_top2_conf005']
            f.write(f'| {r["name"]} | {t["tp"]} | {t["fp"]} | {t["fn"]} | '
                    f'{t["recall"]:.4f} | {t["precision"]:.4f} | {t["f1"]:.4f} |\n')
        f.write(f'\n## 推理耗时\n\n')
        f.write(f'| Run | baseline (s) | TTA (s) |\n')
        f.write(f'|-----|--------------|---------|\n')
        for r in results:
            f.write(f'| {r["name"]} | {r["t_baseline_s"]:.1f} | {r["t_tta_s"]:.1f} |\n')

    print(f'\n报告已写入: {out_path}')


if __name__ == '__main__':
    main()
