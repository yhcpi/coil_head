#!/usr/bin/env python3
"""2026-07-16 v21 best.pt 预标注脚本
- 输入: data/coil/captures_update_v2/*.png (648 张)
- 模型: runs/dfl_off/v21_dfl_off_hn_250ep/weights/best.pt (mAP50=0.9414)
- 输出:
  1. docs/pseudo_labels/captures_update_v2/*.png  ← 原图 + 绿框 + conf 文本
  2. data/coil/labels/pseudo_v2/*.txt            ← YOLO 格式伪标签 (人工微调)
  3. runs/dfl_off/v21_pseudo_v2_summary.json     ← 统计报告
"""
import sys, json, time
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/Hyper-YOLO')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REPO = Path('/home/pi/projects/hyperyolo')
SRC_DIR = REPO / 'data/coil/captures_update_v2'
VIZ_DIR = REPO / 'docs/pseudo_labels/captures_update_v2'
LBL_DIR = REPO / 'data/coil/labels/pseudo_v2'
WEIGHTS = REPO / 'runs/dfl_off/v21_dfl_off_hn_250ep/weights/best.pt'

# 推理配置
CONF_THR = 0.10      # 预标注用低阈值（漏检代价 > 误检代价）
IMGSZ = 1024
MAX_DET = 1          # 每图只保留 top-1
TTA = True           # 用 TTA 提升 recall

VIZ_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

print(f"加载模型: {WEIGHTS}")
m = YOLO(str(WEIGHTS))

# 设置 torch 线程数避免与 v22 训练抢 CPU
import torch
torch.set_num_threads(2)

img_paths = sorted(SRC_DIR.glob('*.png'))
print(f"待标注: {len(img_paths)} 张")
print(f"配置: conf={CONF_THR}, imgsz={IMGSZ}, max_det={MAX_DET}, tta={TTA}")
print(f"可视化输出: {VIZ_DIR}")
print(f"YOLO 标签输出: {LBL_DIR}")

stats = {
    'total': len(img_paths),
    'detected': 0,
    'no_detection': 0,
    'conf_high': 0,    # conf >= 0.5
    'conf_mid': 0,     # 0.3 <= conf < 0.5
    'conf_low': 0,     # 0.1 <= conf < 0.3
    'confs': [],
    'failed_viz': 0,
}

t0 = time.time()
for i, img_p in enumerate(img_paths, 1):
    stem = img_p.stem
    try:
        # 读原图尺寸
        with Image.open(img_p) as im:
            W, H = im.size
        # 推理（top-1）
        try:
            r = m.predict(source=str(img_p), imgsz=IMGSZ, conf=CONF_THR,
                           iou=0.5, max_det=MAX_DET, augment=TTA,
                           save=False, verbose=False)[0]
        except RuntimeError as e:
            # Detect2 类可能 RuntimeError，但 v21 是 Detect nl=3，理论上不会
            if TTA:
                r = m.predict(source=str(img_p), imgsz=IMGSZ, conf=CONF_THR,
                               iou=0.5, max_det=MAX_DET, augment=False,
                               save=False, verbose=False)[0]
            else:
                raise

        # 取 top-1
        if r.boxes is None or len(r.boxes) == 0:
            stats['no_detection'] += 1
            # 无检测：仍输出原图（无框），但 yolo 标签文件留空
            with Image.open(img_p) as im:
                im.save(VIZ_DIR / f"{stem}.png")
            (LBL_DIR / f"{stem}.txt").write_text("")
            continue

        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        conf = float(best.conf[0])
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()

        # YOLO 标签: class cx cy w h (归一化)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        (LBL_DIR / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # 统计
        stats['detected'] += 1
        stats['confs'].append(conf)
        if conf >= 0.5:
            stats['conf_high'] += 1
        elif conf >= 0.3:
            stats['conf_mid'] += 1
        else:
            stats['conf_low'] += 1

        # 可视化
        with Image.open(img_p) as im:
            im_draw = im.copy()
            draw = ImageDraw.Draw(im_draw)
            # 绿框 (line width 3)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
            # conf 文本
            label = f"coil_head {conf:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            # 文字背景
            bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            draw.rectangle(bbox, fill=(0, 220, 0))
            draw.text((x1, y1 - 22), label, fill=(255, 255, 255), font=font)
            im_draw.save(VIZ_DIR / f"{stem}.png")

    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        stats['failed_viz'] += 1
        continue

    if i % 50 == 0 or i == len(img_paths):
        elapsed = time.time() - t0
        eta = elapsed / i * (len(img_paths) - i)
        print(f"  [{i}/{len(img_paths)}] 检出={stats['detected']} 空={stats['no_detection']} "
              f"高={stats['conf_high']} 中={stats['conf_mid']} 低={stats['conf_low']} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

# 统计收尾
stats['conf_mean'] = sum(stats['confs']) / len(stats['confs']) if stats['confs'] else 0
stats['conf_median'] = sorted(stats['confs'])[len(stats['confs']) // 2] if stats['confs'] else 0
stats['conf_min'] = min(stats['confs']) if stats['confs'] else 0
stats['conf_max'] = max(stats['confs']) if stats['confs'] else 0
stats.pop('confs')

out_json = REPO / 'runs/dfl_off/v21_pseudo_v2_summary.json'
out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

print(f"\n{'='*60}")
print(f"✅ 完成 at {time.strftime('%H:%M:%S')}")
print(f"  总图: {stats['total']}")
print(f"  检出: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
print(f"  无检: {stats['no_detection']} ({stats['no_detection']/stats['total']*100:.1f}%)")
print(f"  conf 高 (≥0.5): {stats['conf_high']}")
print(f"  conf 中 (0.3-0.5): {stats['conf_mid']}")
print(f"  conf 低 (0.1-0.3): {stats['conf_low']}")
print(f"  conf 平均/中位: {stats['conf_mean']:.3f} / {stats['conf_median']:.3f}")
print(f"  conf 最小/最大: {stats['conf_min']:.3f} / {stats['conf_max']:.3f}")
print(f"  失败: {stats['failed_viz']}")
print(f"  报告: {out_json}")
print(f"{'='*60}")