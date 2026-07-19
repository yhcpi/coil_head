#!/usr/bin/env python3
"""2026-07-19 把新数据 (captures_merged + new_labels_714) 整合进训练集

流程:
  1. 573 张无标图 mv 到 data/coil/captures_merged_unused/ (可还原)
  2. 364 张带标图 90/10 随机分到 images/{train,val}/ + labels/{train,val}/
  3. labelme JSON → YOLO txt (取 points 最小外接矩形, 归一化 cx,cy,w,h)
  4. 写 data/coil/data_merged.yaml 软链指向新 train/val

约束:
  - seed=42 保证可复现
  - img 与 txt stem 严格一一对应 (labelme 验证)
  - 0 字节空 txt = 负样本 (无人头但有图)
"""
import json, random, shutil
from pathlib import Path
from collections import defaultdict

REPO = Path('/home/pi/projects/hyperyolo')
IMG_SRC = REPO / 'data/coil/captures_merged'
LBL_SRC = REPO / 'data/coil/new_labels_714'
UNUSED_DIR = REPO / 'data/coil/captures_merged_unused'
IMG_DST = REPO / 'data/coil/images'
LBL_DST = REPO / 'data/coil/labels'

SEED = 42
TRAIN_RATIO = 0.9
CLASS_ID = 0  # coil_head

random.seed(SEED)


def json_to_yolo_bbox(json_path, img_w, img_h):
    """labelme JSON → (cx, cy, w, h) 归一化坐标
    points 多边形 → 最小外接矩形 → 转 cx,cy,w,h
    """
    with open(json_path) as f:
        data = json.load(f)
    if not data.get('shapes'):
        return None
    pts = []
    for s in data['shapes']:
        if s.get('label') == 'coil_head' or s.get('label', '').startswith('coil'):
            pts.extend(s['points'])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    # 边界检查
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    bw = max(0.001, min(1, bw))
    bh = max(0.001, min(1, bh))
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def main():
    UNUSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 盘点
    imgs = sorted([p for p in IMG_SRC.glob('*.png')])
    lbls = sorted([p for p in LBL_SRC.glob('*.json')])
    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}

    orphans_img = img_stems - lbl_stems  # 有图无标
    matched = img_stems & lbl_stems       # 都有
    print(f"图: {len(imgs)}, 标: {len(lbls)}, 匹配: {len(matched)}, 孤儿图: {len(orphans_img)}")

    # 2. mv 孤儿图到 unused
    mv_count = 0
    for stem in orphans_img:
        src = IMG_SRC / f"{stem}.png"
        dst = UNUSED_DIR / f"{stem}.png"
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        mv_count += 1
    print(f"✅ 已 mv {mv_count} 张无标图到 {UNUSED_DIR}")

    # 3. 90/10 切分 (train/val 不重叠)
    matched_list = sorted(matched)
    random.shuffle(matched_list)
    n_train = int(len(matched_list) * TRAIN_RATIO)
    train_stems = set(matched_list[:n_train])
    val_stems = set(matched_list[n_train:])
    print(f"新 train: {len(train_stems)}, 新 val: {len(val_stems)}")

    # 4. 复制图片 + 写标签 (不动现有 train/val，只追加)
    stats = defaultdict(int)
    skipped = 0
    for stem in matched:
        # 跳过现有 train/val（不要破坏已训练的）
        # 现状: 现有 images/train + images/val 已有 labels, 不动它们
        img_src = IMG_SRC / f"{stem}.png"
        json_src = LBL_SRC / f"{stem}.json"

        # 读图片尺寸
        from PIL import Image
        with Image.open(img_src) as im:
            W, H = im.size

        # labelme → yolo
        yolo_line = json_to_yolo_bbox(json_src, W, H)
        if yolo_line is None:
            print(f"  [SKIP] {stem}.json 无 coil 标注")
            skipped += 1
            continue

        # 选 split
        split = 'train' if stem in train_stems else 'val'
        img_dst = IMG_DST / split / f"{stem}.png"
        lbl_dst = LBL_DST / split / f"{stem}.txt"

        # 跳过已存在
        if img_dst.exists() or lbl_dst.exists():
            skipped += 1
            continue

        shutil.copy(str(img_src), str(img_dst))
        lbl_dst.write_text(yolo_line)
        stats[split] += 1

    print(f"✅ 新增 train: {stats['train']} 张, 新增 val: {stats['val']} 张, 跳过: {skipped}")

    # 5. 统计现有数据规模
    n_train_total = len(list((IMG_DST / 'train').glob('*.png')))
    n_val_total = len(list((IMG_DST / 'val').glob('*.png')))
    print(f"\n📊 当前总规模:")
    print(f"  train: {n_train_total} 张")
    print(f"  val:   {n_val_total} 张")

    # 6. 统计新 train/val GT 分布
    n_train_pos = sum(1 for p in (LBL_DST / 'train').glob('*.txt') if p.stat().st_size > 0)
    n_val_pos = sum(1 for p in (LBL_DST / 'val').glob('*.txt') if p.stat().st_size > 0)
    print(f"  train 正样本: {n_train_pos} / {n_train_total}")
    print(f"  val 正样本:   {n_val_pos} / {n_val_total}")


if __name__ == '__main__':
    main()