#!/usr/bin/env python3
"""CP-Aug（宽容 Copy-Paste）—— 数据增强创新点 1

从已标 bbox 抠出 wire patch，随机 paste 到负样本（empty T）上，生成新的训练样本。

依据: augmentation_innovations.md 创新点 1（宽容 Copy-Paste 扩充）
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path('/home/pi/projects/hyperyolo/data/coil')
TRAIN_IMG = ROOT / 'images' / 'train'
TRAIN_LBL = ROOT / 'labels' / 'train'
OUT_AUG_IMG = ROOT / 'images_aug' / 'train'
OUT_AUG_LBL = ROOT / 'labels_aug' / 'train'
OUT_AUG_IMG.mkdir(parents=True, exist_ok=True)
OUT_AUG_LBL.mkdir(parents=True, exist_ok=True)

# 1. 收集正样本 patch（从所有训练集已标图抠）
positive_patches = []  # [(patch_img, scale_range, aspect)]
for img_p in TRAIN_IMG.glob('*.png'):
    lbl_p = TRAIN_LBL / f'{img_p.stem}.txt'
    if not lbl_p.exists() or lbl_p.stat().st_size == 0:
        continue
    img = Image.open(img_p).convert('RGB')
    W, H = img.size
    for line in lbl_p.read_text().strip().split('\n'):
        parts = line.split()
        if len(parts) < 5: continue
        cx, cy, w, h = map(float, parts[1:5])
        # 反归一化到像素坐标，加 padding 让 patch 大一点
        pad = 1.5
        px = int((cx - w/2 * pad) * W)
        py = int((cy - h/2 * pad) * H)
        pw = int(w * pad * W)
        ph = int(h * pad * H)
        px, py = max(0, px), max(0, py)
        px2, py2 = min(W, px + pw), min(H, py + ph)
        if px2 - px < 10 or py2 - py < 10: continue
        patch = img.crop((px, py, px2, py2))
        positive_patches.append({
            'patch': patch,
            'src': img_p.stem,
            'orig_size': (px2-px, py2-py),
        })

print(f'收集到 {len(positive_patches)} 个正样本 patch')

# 2. 找负样本
negatives = []
for img_p in TRAIN_IMG.glob('*.png'):
    lbl_p = TRAIN_LBL / f'{img_p.stem}.txt'
    if lbl_p.exists() and lbl_p.stat().st_size > 0:
        continue
    if not lbl_p.exists():
        continue
    img = Image.open(img_p).convert('RGB')
    negatives.append((img_p, img))

print(f'找到 {len(negatives)} 张负样本')

# 3. 对每张负样本生成 N 张增强图
N_PER_NEG = 5  # 每张负样本生成 5 张增强版
all_new_bboxes = []  # 记录每个新图的 bboxes（用于可视化验证）

aug_count = 0
for neg_idx, (neg_p, neg_img) in enumerate(negatives):
    W, H = neg_img.size
    for k in range(N_PER_NEG):
        out_img = neg_img.copy()
        new_bboxes = []  # 归一化 bbox (cls cx cy w h)

        # 随机 paste 1-3 个 patch
        n_paste = random.randint(1, 3)
        for _ in range(n_paste):
            p = random.choice(positive_patches)
            patch = p['patch']
            ow, oh = patch.size

            # 随机 scale (0.7-1.4)
            scale = random.uniform(0.7, 1.4)
            new_w = int(ow * scale)
            new_h = int(oh * scale)
            if new_w < 10 or new_h < 10: continue
            patch_resized = patch.resize((new_w, new_h), Image.LANCZOS)

            # 随机位置（不超出图边界）
            max_x = max(0, W - new_w)
            max_y = max(0, H - new_h)
            if max_x == 0 or max_y == 0: continue
            px = random.randint(0, max_x)
            py = random.randint(0, max_y)

            # 边缘羽化（避免硬边）
            mask = Image.new('L', (new_w, new_h), 0)
            mask_arr = np.array(mask)
            cx_m, cy_m = new_w // 2, new_h // 2
            Y, X = np.ogrid[:new_h, :new_w]
            dist = np.sqrt((X - cx_m)**2 + (Y - cy_m)**2)
            feather = max(2, min(new_w, new_h) // 4)
            mask_arr = np.clip(1.0 - dist / feather, 0, 1) * 255
            mask = Image.fromarray(mask_arr.astype(np.uint8))

            # Alpha paste
            out_img.paste(patch_resized, (px, py), mask)

            # 计算归一化 bbox（中心 + 宽高）
            cx_norm = (px + new_w/2) / W
            cy_norm = (py + new_h/2) / H
            w_norm = new_w / W
            h_norm = new_h / H
            new_bboxes.append(f'0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}')

        # 保存
        out_name = f'aug_{neg_p.stem}_v{k}.png'
        out_img.save(OUT_AUG_IMG / out_name, optimize=True)
        (OUT_AUG_LBL / f'{Path(out_name).stem}.txt').write_text('\n'.join(new_bboxes) + '\n')
        aug_count += 1
        all_new_bboxes.append((out_name, new_bboxes))

print(f'\n生成 {aug_count} 张 CP-Aug 增强图')
print(f'输出: {OUT_AUG_IMG}/')
print(f'输出: {OUT_AUG_LBL}/')

# 4. 统计
print(f'\n=== 数据集扩充效果 ===')
print(f'原 train 正样本: 35')
print(f'原 train 负样本: 16')
print(f'CP-Aug 新增 (全正样本): {aug_count}')
print(f'扩充后 train 总正样本: {35 + aug_count}（{aug_count/35:.1f}x）')
print(f'扩充后 train 总样本: {35 + aug_count + 16} = {35 + aug_count + 16}')