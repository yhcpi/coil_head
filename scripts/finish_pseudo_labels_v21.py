#!/usr/bin/env python3
"""2026-07-16 补跑预标注剩下 52 张 (714_597 ~ 714_648)
脚本于 ~21:52 在 714_596 后中断，未生成 summary JSON
- 跳过已存在 txt 的图片
- 完成后重新统计全局 summary
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

CONF_THR = 0.10
IMGSZ = 1024
MAX_DET = 1
TTA = True

VIZ_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

print(f"加载模型: {WEIGHTS}")
m = YOLO(str(WEIGHTS))

import torch
torch.set_num_threads(2)

# 只跑已有 png 中、还没 txt 的
all_imgs = sorted(SRC_DIR.glob('*.png'))
pending = [p for p in all_imgs if not (LBL_DIR / f"{p.stem}.txt").exists()]
print(f"总图: {len(all_imgs)}, 已处理: {len(all_imgs) - len(pending)}, 待补: {len(pending)}")
print(f"配置: conf={CONF_THR}, imgsz={IMGSZ}, max_det={MAX_DET}, tta={TTA}")

stats = {
    'total': len(all_imgs),
    'processed': len(all_imgs) - len(pending),
    'newly_processed': 0,
    'detected': 0,
    'no_detection': 0,
    'conf_high': 0,
    'conf_mid': 0,
    'conf_low': 0,
    'confs': [],
    'failed_viz': 0,
}

t0 = time.time()
for i, img_p in enumerate(pending, 1):
    stem = img_p.stem
    try:
        with Image.open(img_p) as im:
            W, H = im.size
        try:
            r = m.predict(source=str(img_p), imgsz=IMGSZ, conf=CONF_THR,
                           iou=0.5, max_det=MAX_DET, augment=TTA,
                           save=False, verbose=False)[0]
        except RuntimeError as e:
            if TTA:
                r = m.predict(source=str(img_p), imgsz=IMGSZ, conf=CONF_THR,
                               iou=0.5, max_det=MAX_DET, augment=False,
                               save=False, verbose=False)[0]
            else:
                raise

        if r.boxes is None or len(r.boxes) == 0:
            stats['no_detection'] += 1
            with Image.open(img_p) as im:
                im.save(VIZ_DIR / f"{stem}.png")
            (LBL_DIR / f"{stem}.txt").write_text("")
            stats['newly_processed'] += 1
            continue

        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        conf = float(best.conf[0])
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        (LBL_DIR / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        stats['detected'] += 1
        stats['confs'].append(conf)
        if conf >= 0.5:
            stats['conf_high'] += 1
        elif conf >= 0.3:
            stats['conf_mid'] += 1
        else:
            stats['conf_low'] += 1

        with Image.open(img_p) as im:
            im_draw = im.copy()
            draw = ImageDraw.Draw(im_draw)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
            label = f"coil_head {conf:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            draw.rectangle(bbox, fill=(0, 220, 0))
            draw.text((x1, y1 - 22), label, fill=(255, 255, 255), font=font)
            im_draw.save(VIZ_DIR / f"{stem}.png")

    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        stats['failed_viz'] += 1
        continue

    stats['newly_processed'] += 1
    if i % 10 == 0 or i == len(pending):
        elapsed = time.time() - t0
        eta = elapsed / i * (len(pending) - i)
        print(f"  [{i}/{len(pending)}] 检出={stats['detected']} 空={stats['no_detection']} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

# 全局统计（基于 LBL_DIR 中所有非空 txt）
all_txts = list(LBL_DIR.glob('*.txt'))
total_processed = len(all_txts)
total_non_empty = sum(1 for t in all_txts if t.stat().st_size > 0)
total_empty = total_processed - total_non_empty

print(f"\n{'='*60}")
print(f"✅ 补跑完成 at {time.strftime('%H:%M:%S')}")
print(f"  本轮新增: {stats['newly_processed']} (检出={stats['detected']} 空={stats['no_detection']})")
print(f"  全局:")
print(f"    总图: {stats['total']}")
print(f"    已处理: {total_processed} ({total_processed/stats['total']*100:.1f}%)")
print(f"    检出 (非空 txt): {total_non_empty}")
print(f"    未检出 (空 txt): {total_empty}")
print(f"    未处理: {stats['total'] - total_processed}")
print(f"{'='*60}")

out_json = REPO / 'runs/dfl_off/v21_pseudo_v2_summary.json'
out_json.write_text(json.dumps({
    'total': stats['total'],
    'processed': total_processed,
    'non_empty_labels': total_non_empty,
    'empty_labels': total_empty,
    'unprocessed': stats['total'] - total_processed,
    'this_run': {
        'newly_processed': stats['newly_processed'],
        'detected': stats['detected'],
        'no_detection': stats['no_detection'],
        'conf_high': stats['conf_high'],
        'conf_mid': stats['conf_mid'],
        'conf_low': stats['conf_low'],
        'failed_viz': stats['failed_viz'],
    }
}, indent=2, ensure_ascii=False))
print(f"  报告: {out_json}")