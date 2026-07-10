#!/usr/bin/env python3
"""钢卷数据集拆分：已标 train/val + 未标(负样本) train/val

支持多个图片源目录；labels_all/ 里有的视为已标，没有的视为负样本。
"""
import shutil
from pathlib import Path
import random

random.seed(42)  # 固定随机种子，结果可复现

# 多个图片源目录
SRC_IMG_DIRS = [
    Path('/home/pi/projects/hyperyolo/repos/Hyper-YOLO/粗钢卷/images'),
    Path('/home/pi/projects/hyperyolo/data/coil/new_images'),
    Path('/home/pi/projects/hyperyolo/data/coil/73images'),  # 2026-07-03: 565 张新批次
]
SRC_LBL = Path('/home/pi/projects/hyperyolo/data/coil/labels_all')

OUT_ROOT = Path('/home/pi/projects/hyperyolo/data/coil')
OUT_IMG = OUT_ROOT / 'images'
OUT_LBL = OUT_ROOT / 'labels'

# 清理旧数据
for sub in ['train', 'val']:
    (OUT_IMG / sub).mkdir(parents=True, exist_ok=True)
    (OUT_LBL / sub).mkdir(parents=True, exist_ok=True)

# 1. 列出所有图（有标 vs 无标）
all_imgs = []
for d in SRC_IMG_DIRS:
    if not d.exists():
        print(f'[WARN] 跳过不存在的目录: {d}')
        continue
    all_imgs.extend(sorted(d.glob('*.png')))
    all_imgs.extend(sorted(d.glob('*.PNG')))

labeled = [p for p in all_imgs if (SRC_LBL / (p.stem + '.txt')).exists()]
unlabeled = [p for p in all_imgs if not (SRC_LBL / (p.stem + '.txt')).exists()]
print(f'总图: {len(all_imgs)}')
print(f'已标: {len(labeled)}, 未标(空图/负样本): {len(unlabeled)}')

# 2. 拆已标的: 80/20 train/val
random.shuffle(labeled)
n_val_labeled = max(2, len(labeled) // 5)  # 至少 2 张 val
val_imgs = labeled[:n_val_labeled]
train_imgs = labeled[n_val_labeled:]

# 3. 未标: 80/20 train/val
random.shuffle(unlabeled)
n_val_unlabeled = max(1, len(unlabeled) // 5) if unlabeled else 0
val_unlabeled = unlabeled[:n_val_unlabeled]
train_unlabeled = unlabeled[n_val_unlabeled:]
train_imgs = train_imgs + train_unlabeled
val_imgs = val_imgs + val_unlabeled  # 修bug：val 也要含负样本

print(f'已标 val: {len(val_imgs)}, 已标 train: {len(labeled) - len(val_imgs)}')
print(f'未标 val: {len(val_unlabeled)}, 未标 train: {len(train_unlabeled)}')
print(f'最终: train={len(train_imgs)}, val={len(val_imgs)}')

# 4. 复制
def copy_pair(img_path, split):
    """复制图片 + 标签到目标 split 目录"""
    # 避免文件名冲突（不同源目录可能有同名文件）
    out_name = img_path.name
    dst_img = OUT_IMG / split / out_name
    if dst_img.exists():
        # 加源目录前缀避免冲突
        out_name = f'{img_path.parent.name}_{img_path.name}'
        dst_img = OUT_IMG / split / out_name

    shutil.copy(img_path, dst_img)

    txt_path = SRC_LBL / (img_path.stem + '.txt')
    dst_lbl = OUT_LBL / split / (dst_img.stem + '.txt')
    if txt_path.exists():
        shutil.copy(txt_path, dst_lbl)
    else:
        # 空图：写一个空 .txt（YOLO 允许空 txt 表示"无目标"）
        dst_lbl.write_text('')

for p in train_imgs:
    copy_pair(p, 'train')
for p in val_imgs:
    copy_pair(p, 'val')

print(f'\n=== 拆分完成 ===')
print(f'train: {len(train_imgs)} 张 → {OUT_IMG}/train/')
print(f'val:   {len(val_imgs)} 张 → {OUT_IMG}/val/')