#!/usr/bin/env python3
"""2026-07-19 删空标签 (负样本) → train/val 全正样本
- 删除空 txt 对应的 img 和 txt
- 备份到 data/coil/_deleted_negatives/ (可还原)
"""
import shutil
from pathlib import Path

REPO = Path('/home/pi/projects/hyperyolo')
IMG_DIR = REPO / 'data/coil/images'
LBL_DIR = REPO / 'data/coil/labels'
BAK = REPO / 'data/coil/_deleted_negatives'
BAK_IMG = BAK / 'images'
BAK_LBL = BAK / 'labels'
BAK_IMG.mkdir(parents=True, exist_ok=True)
BAK_LBL.mkdir(parents=True, exist_ok=True)

total_del = {'train': 0, 'val': 0}
for split in ['train', 'val']:
    for txt in (LBL_DIR / split).glob('*.txt'):
        if txt.stat().st_size == 0:  # 空标签 = 负样本
            stem = txt.stem
            img = (IMG_DIR / split) / f"{stem}.png"
            # 备份
            shutil.copy(str(txt), BAK_LBL / f"{split}_{stem}.txt")
            if img.exists():
                shutil.copy(str(img), BAK_IMG / f"{split}_{stem}.png")
                img.unlink()
            txt.unlink()
            total_del[split] += 1

print(f"✅ 删除空标签 (负样本):")
print(f"  train: {total_del['train']} 张图 + txt 移到 {BAK}")
print(f"  val:   {total_del['val']} 张图 + txt 移到 {BAK}")

# 重新统计
print(f"\n📊 删后规模:")
for split in ['train', 'val']:
    img = len(list((IMG_DIR / split).glob('*.png')))
    lbl = len(list((LBL_DIR / split).glob('*.txt')))
    non_empty = sum(1 for p in (LBL_DIR / split).glob('*.txt') if p.stat().st_size > 0)
    print(f"  {split}: {img} 张图 / {lbl} 个 txt (全正样本: {non_empty})")