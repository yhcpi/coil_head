#!/usr/bin/env python3
"""labelme JSON → YOLO txt 转换器（钢卷头尾专用）

读 labelme 格式的 JSON（rectangle shape），输出 YOLO 格式的 .txt。
每张图一个 .txt，每行: class_id cx cy w h（归一化 0-1）。

支持多个输入目录（合并到同一 labels_all/）。
"""
import json
from pathlib import Path

# ========== 配置（按需改）==========
SRC_DIRS = [
    Path('/home/pi/projects/hyperyolo/repos/Hyper-YOLO/粗钢卷/lables'),
    Path('/home/pi/projects/hyperyolo/data/coil/new_labels'),
    Path('/home/pi/projects/hyperyolo/data/coil/73labels'),  # 2026-07-03: 565 张新批次
]
OUT_DIR = Path('/home/pi/projects/hyperyolo/data/coil/labels_all')
CLASS_MAP = {'coil_head': 0}  # 类别名 → YOLO class id
# ===================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 只清空本次要处理的源目录对应的旧 txt（避免误删其他目录的标签）
processed_stems = set()
for SRC_DIR in SRC_DIRS:
    if SRC_DIR.exists():
        for jp in SRC_DIR.glob('*.json'):
            processed_stems.add(jp.stem)
for stem in processed_stems:
    old = OUT_DIR / f'{stem}.txt'
    if old.exists():
        old.unlink()

n_ok, n_skip, n_err = 0, 0, 0
for SRC_DIR in SRC_DIRS:
    if not SRC_DIR.exists():
        print(f'[WARN] 跳过不存在的目录: {SRC_DIR}')
        continue
    print(f'\n--- 处理目录: {SRC_DIR} ---')
    for jp in sorted(SRC_DIR.glob('*.json')):
        try:
            with open(jp) as f:
                data = json.load(f)

            # 1. 读图尺寸（用 JSON 自带字段，避免读图）
            W = data['imageWidth']
            H = data['imageHeight']

            # 2. 提取所有 rectangle bbox
            lines = []
            for shape in data.get('shapes', []):
                if shape['shape_type'] != 'rectangle':
                    continue
                label = shape['label']
                if label not in CLASS_MAP:
                    print(f'  [SKIP] {jp.name}: 未知类别 {label!r}')
                    n_skip += 1
                    continue
                cls_id = CLASS_MAP[label]

                # labelme rectangle: points = [[x1,y1],[x2,y2]]
                (x1, y1), (x2, y2) = shape['points']
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # 转 YOLO: 中心点 + 宽高（归一化 0-1）
                cx = ((x_min + x_max) / 2) / W
                cy = ((y_min + y_max) / 2) / H
                w  = (x_max - x_min) / W
                h  = (y_max - y_min) / H

                lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

            # 3. 写 YOLO txt（与 JSON 同名）
            out_path = OUT_DIR / (jp.stem + '.txt')
            with open(out_path, 'w') as f:
                f.write('\n'.join(lines) + '\n' if lines else '')

            n_ok += 1
            print(f'  [OK] {jp.stem}.json → {out_path.name} ({len(lines)} bbox)')

        except Exception as e:
            print(f'  [ERR] {jp.name}: {e}')
            n_err += 1

print(f'\n=== 完成: {n_ok} 成功, {n_skip} 跳过(未知类别), {n_err} 错误 ===')
print(f'输出目录: {OUT_DIR}')