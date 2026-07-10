"""SAHI 切片训练数据集生成（offline，v2 16:9）

把 2560×1440 大图切成 1024×576 子图（保持 16:9 长宽比，与训练 imgsz=1024 + rect 一致），
含 20% overlap，同时调整 bbox 到子图坐标系，丢弃无目标的子图。

子图尺寸 1024×576：
- 与训练时模型"看到的"几何完全一致（imgsz=1024 rect → 1024×576 输入）
- 切分：2560×1440 → 横向约 2~3 片、纵向约 2~3 片（约 4~9 子图/原图）
- 不让子图比训练单图更小，避免模型因"小图内目标占比过高"过度敏感

输出：
- coil_sahi_16x9/images/{train,val}/  16:9 切片
- coil_sahi_16x9/labels/{train,val}/  调整后的 YOLO txt
- coil_sahi_16x9/data.yaml

依据：SAHI 论文 https://arxiv.org/abs/2202.06934
"""
import sys
sys.path.insert(0, '/home/pi/projects/hyperyolo/repos/sahi')

import shutil
from pathlib import Path
from PIL import Image
from sahi.slicing import get_slice_bboxes

# ========== 配置（v2 16:9）==========
SLICE_H = 576
SLICE_W = 1024
OVERLAP = 0.2

SRC_ROOT = Path('/home/pi/projects/hyperyolo/data/coil')
SRC_IMG = {s: SRC_ROOT / 'images' / s for s in ['train', 'val']}
SRC_LBL = {s: SRC_ROOT / 'labels' / s for s in ['train', 'val']}

OUT_ROOT = Path('/home/pi/projects/hyperyolo/data/coil_sahi_16x9')
OUT_IMG = {s: OUT_ROOT / 'images' / s for s in ['train', 'val']}
OUT_LBL = {s: OUT_ROOT / 'labels' / s for s in ['train', 'val']}
# 清空
for d in list(OUT_IMG.values()) + list(OUT_LBL.values()):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

stats = {'src_imgs': 0, 'src_pos': 0, 'src_neg': 0, 'slices_total': 0,
         'slices_kept': 0, 'slices_empty': 0, 'bbox_in': 0, 'bbox_out': 0}


def read_yolo_labels(label_path: Path):
    """读 YOLO txt → list of (cls, cx, cy, w, h) (归一化 0-1)"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text().strip().split('\n'):
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def yolo_to_pixel(box, W, H):
    """(cls, cx_n, cy_n, w_n, h_n) → (cls, x1, y1, x2, y2) in pixel"""
    cls, cx, cy, w, h = box
    px = cx * W
    py = cy * H
    pw = w * W
    ph = h * H
    x1 = px - pw / 2
    y1 = py - ph / 2
    x2 = px + pw / 2
    y2 = py + ph / 2
    return (cls, x1, y1, x2, y2)


def pixel_to_yolo(cls, x1, y1, x2, y2, W, H):
    """像素 bbox → YOLO (cls, cx_n, cy_n, w_n, h_n)"""
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    # 裁剪到 [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'


def clip_bbox_to_slice(bbox, sx1, sy1, sx2, sy2):
    """bbox (cls, x1, y1, x2, y2) 与 slice 求交集（裁剪到 slice 边界）。
    返回 None 如果无交集。"""
    cls, x1, y1, x2, y2 = bbox
    cx1 = max(x1, sx1)
    cy1 = max(y1, sy1)
    cx2 = min(x2, sx2)
    cy2 = min(y2, sy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    # 过滤太小的框（< 4 像素），不然对损失贡献噪声
    if (cx2 - cx1) < 4 or (cy2 - cy1) < 4:
        return None
    return (cls, cx1, cy1, cx2, cy2)


def slice_one_image(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    """切一张图为若干 640×640 子图 + 调整标签。

    负样本（空 .txt）保留原图，不切——避免引入假负样本。
    """
    boxes = read_yolo_labels(lbl_path)
    # 负样本：原图复制，标签为空
    if not boxes:
        stats['src_neg'] += 1
        shutil.copy(img_path, out_img_dir / img_path.name)
        (out_lbl_dir / f'{img_path.stem}.txt').write_text('')
        return

    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    stats['src_pos'] += 1
    # 转像素
    pix_boxes = [yolo_to_pixel(b, W, H) for b in boxes]

    # 切片
    slices = get_slice_bboxes(
        image_height=H, image_width=W,
        slice_height=SLICE_H, slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
    )

    stem = img_path.stem
    for si, (sx1, sy1, sx2, sy2) in enumerate(slices):
        sw, sh = sx2 - sx1, sy2 - sy1
        if sw <= 0 or sh <= 0:
            continue
        # 切图
        crop = img.crop((sx1, sy1, sx2, sy2))
        # 标签裁剪
        slice_boxes = []
        for pb in pix_boxes:
            clipped = clip_bbox_to_slice(pb, sx1, sy1, sx2, sy2)
            if clipped is None:
                continue
            cls, cx1, cy1, cx2, cy2 = clipped
            # 转成切片坐标系
            nx1 = cx1 - sx1
            ny1 = cy1 - sy1
            nx2 = cx2 - sx1
            ny2 = cy2 - sy1
            slice_boxes.append(pixel_to_yolo(cls, nx1, ny1, nx2, ny2, sw, sh))
            stats['bbox_out'] += 1

        stats['slices_total'] += 1
        if not slice_boxes:
            # 丢弃无目标的子图（避免引入假负样本）
            stats['slices_empty'] += 1
            continue

        # 保存子图 + 标签
        out_name = f'{stem}_s{si}.png'
        crop.save(out_img_dir / out_name)
        (out_lbl_dir / f'{stem}_s{si}.txt').write_text('\n'.join(slice_boxes) + '\n')
        stats['slices_kept'] += 1


# 处理 train 和 val
for split in ['train', 'val']:
    src_img_d = SRC_IMG[split]
    src_lbl_d = SRC_LBL[split]
    out_img_d = OUT_IMG[split]
    out_lbl_d = OUT_LBL[split]
    img_files = sorted(src_img_d.glob('*.png'))
    print(f'\n[{split}] 处理 {len(img_files)} 张原图')
    for img_p in img_files:
        lbl_p = src_lbl_d / f'{img_p.stem}.txt'
        stats['src_imgs'] += 1
        slice_one_image(img_p, lbl_p, out_img_d, out_lbl_d)

# 输出统计
print('\n=== SAHI 切片统计 ===')
print(f'原图数: {stats["src_imgs"]}（正样本 {stats["src_pos"]} + 负样本 {stats["src_neg"]}）')
print(f'切片总数: {stats["slices_total"]}')
print(f'有目标保留: {stats["slices_kept"]}')
print(f'空切片丢弃: {stats["slices_empty"]}')
print(f'输出 bbox: {stats["bbox_out"]}')
print(f'正样本翻倍: {stats["src_pos"]} → {stats["slices_kept"]} ({stats["slices_kept"]/max(1,stats["src_pos"]):.1f}x)')

# 生成 data.yaml
data_yaml = f'''path: {OUT_ROOT}
train: images/train
val: images/val

names:
  0: coil_head
'''
(OUT_ROOT / 'data.yaml').write_text(data_yaml)
print(f'\n✓ data.yaml 写入: {OUT_ROOT}/data.yaml')