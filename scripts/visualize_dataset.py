#!/usr/bin/env python3
"""数据集可视化脚本：把 YOLO bbox 画回原图，输出到 runs/dataset_viz/。

输出结构：
  runs/dataset_viz/
    ├── train/
    │   ├── labeled/    # 有 bbox 的图（画了框）
    │   └── empty/      # 空标签的图（原图，提示"未标注"）
    ├── val/
    │   ├── labeled/
    │   └── empty/
    └── index.html      # 索引页，按 split × 类型分组展示所有图
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 路径配置（相对项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "coil"
OUT_ROOT = PROJECT_ROOT / "runs" / "dataset_viz"
SPLITS = ["train", "val"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 可视化样式
BOX_COLOR = (0, 255, 0)        # 绿色 bbox
BOX_WIDTH = 3                  # 线宽
TEXT_BG = (0, 200, 0)          # 标签背景
TEXT_COLOR = (255, 255, 255)   # 标签文字
LABEL_TEXT = "coil_head"       # 类别名（与 data.yaml 一致）
FONT_SIZE = 18


def get_font(size: int):
    """尝试加载系统字体，失败则 fallback 到 PIL default。"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """YOLO 格式 -> 像素坐标 (x1, y1, x2, y2)"""
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = (float(x) for x in parts[1:5])
            # 归一化 -> 像素
            x1 = int((x_c - w / 2) * img_w)
            y1 = int((y_c - h / 2) * img_h)
            x2 = int((x_c + w / 2) * img_w)
            y2 = int((y_c + h / 2) * img_h)
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def find_image(images_dir: Path, stem: str):
    """根据 stem (无扩展名) 找图片文件，支持多种扩展名。"""
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def draw_boxes(img: Image.Image, boxes, font) -> Image.Image:
    """在 PIL Image 上画 bbox + 标签"""
    draw = ImageDraw.Draw(img)
    for cls_id, x1, y1, x2, y2 in boxes:
        # 限制坐标在图内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width - 1, x2), min(img.height - 1, y2)
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
        # 标签文字 + 背景
        label = f"{LABEL_TEXT}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # 文字背景在 bbox 上方
        bg_y1 = max(0, y1 - text_h - 4)
        bg_y2 = y1
        draw.rectangle([x1, bg_y1, x1 + text_w + 6, bg_y2], fill=TEXT_BG)
        draw.text((x1 + 3, bg_y1 + 2), label, fill=TEXT_COLOR, font=font)
    return img


def add_watermark(img: Image.Image, text: str, font) -> Image.Image:
    """在左上角加角标（标注类型 / bbox 数 / 状态）"""
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 12
    text_h = bbox[3] - bbox[1] + 8
    # 半透明黑色背景
    draw.rectangle([0, 0, text_w, text_h], fill=(0, 0, 0, 180))
    draw.text((6, 4), text, fill=(255, 255, 0), font=font)
    return img


def process_split(split: str, stats: dict):
    """处理一个 split (train/val)，输出到 OUT_ROOT/{split}/labeled 和 empty/"""
    images_dir = DATA_ROOT / "images" / split
    labels_dir = DATA_ROOT / "labels" / split
    out_labeled = OUT_ROOT / split / "labeled"
    out_empty = OUT_ROOT / split / "empty"
    out_labeled.mkdir(parents=True, exist_ok=True)
    out_empty.mkdir(parents=True, exist_ok=True)

    # 枚举所有图片（可能存在 label 缺失）
    image_stems = sorted(
        p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    )

    font = get_font(FONT_SIZE)

    for stem in image_stems:
        img_path = find_image(images_dir, stem)
        if img_path is None:
            continue
        label_path = labels_dir / f"{stem}.txt"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] 跳过 {img_path.name}: {e}", file=sys.stderr)
            continue

        boxes = parse_yolo_label(label_path, img.width, img.height)
        has_box = len(boxes) > 0

        if has_box:
            img = draw_boxes(img, boxes, font)
            img = add_watermark(img, f"{split} | LABELED | {len(boxes)} bbox | {stem}", font)
            out_path = out_labeled / f"{stem}_labeled.jpg"
            stats["labeled"] += 1
        else:
            img = add_watermark(img, f"{split} | EMPTY/UNLABELED | {stem}", font)
            out_path = out_empty / f"{stem}_empty.jpg"
            stats["empty"] += 1
        img.save(out_path, "JPEG", quality=85)

        # 检测 img / label 不一致
        if not label_path.exists():
            stats["img_without_label"] += 1
        if not img_path.exists():
            stats["label_without_img"] += 1

    stats["total"] = stats["labeled"] + stats["empty"]


def write_index_html(stats: dict):
    """生成 runs/dataset_viz/index.html 便于浏览"""
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>coil dataset visualization</title>",
        "<style>",
        "body{font-family:sans-serif;margin:20px;background:#1a1a1a;color:#eee;}",
        "h1,h2{color:#4f9;}",
        "h2{margin-top:30px;border-bottom:1px solid #4f9;padding-bottom:5px;}",
        "table{border-collapse:collapse;margin:10px 0;}",
        "td,th{padding:5px 15px;border:1px solid #444;}",
        "th{background:#2a2a2a;color:#4f9;}",
        ".stat{color:#fc6;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin:10px 0;}",
        ".card{background:#2a2a2a;padding:8px;border-radius:4px;}",
        ".card img{width:100%;height:auto;display:block;}",
        ".card .name{font-size:11px;color:#aaa;margin-top:4px;word-break:break-all;}",
        "a{color:#4f9;text-decoration:none;}",
        "a:hover{text-decoration:underline;}",
        "</style></head><body>",
        "<h1>📦 coil 数据集可视化</h1>",
        "<p>每张图：绿色框 = YOLO bbox；左上角角标显示 split / 标注状态 / bbox 数量 / 文件名</p>",
        "<h2>📊 统计</h2>",
        "<table>",
        "<tr><th>split</th><th>有 bbox</th><th>空标签 (负样本)</th><th>合计</th></tr>",
    ]
    for split in SPLITS:
        s = stats[split]
        html.append(
            f"<tr><td>{split}</td>"
            f"<td class='stat'>{s['labeled']}</td>"
            f"<td class='stat'>{s['empty']}</td>"
            f"<td class='stat'>{s['total']}</td></tr>"
        )
    html.append("</table>")
    html.append(
        f"<p>图片/标签不一致警告："
        f"img 无 label = {stats.get('img_without_label', 0)}; "
        f"label 无 img = {stats.get('label_without_img', 0)}</p>"
    )

    for split in SPLITS:
        s = stats[split]
        html.append(f"<h2>📁 {split}/labeled（{s['labeled']} 张）</h2><div class='grid'>")
        labeled_dir = OUT_ROOT / split / "labeled"
        for img_path in sorted(labeled_dir.glob("*.jpg"))[:500]:
            rel = img_path.relative_to(OUT_ROOT)
            html.append(
                f"<div class='card'>"
                f"<a href='{rel}' target='_blank'>"
                f"<img src='{rel}' loading='lazy'>"
                f"</a>"
                f"<div class='name'>{img_path.name}</div>"
                f"</div>"
            )
        html.append("</div>")

        html.append(f"<h2>📁 {split}/empty（{s['empty']} 张）</h2><div class='grid'>")
        empty_dir = OUT_ROOT / split / "empty"
        for img_path in sorted(empty_dir.glob("*.jpg"))[:500]:
            rel = img_path.relative_to(OUT_ROOT)
            html.append(
                f"<div class='card'>"
                f"<a href='{rel}' target='_blank'>"
                f"<img src='{rel}' loading='lazy'>"
                f"</a>"
                f"<div class='name'>{img_path.name}</div>"
                f"</div>"
            )
        html.append("</div>")

    html.append("</body></html>")
    (OUT_ROOT / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stats = {s: {"labeled": 0, "empty": 0, "total": 0} for s in SPLITS}
    stats["img_without_label"] = 0
    stats["label_without_img"] = 0

    for split in SPLITS:
        print(f"处理 {split} ...")
        process_split(split, stats[split])
        s = stats[split]
        print(f"  {split}: {s['labeled']} labeled + {s['empty']} empty = {s['total']} 总")

    write_index_html(stats)
    print(f"\n完成。输出: {OUT_ROOT}")
    print(f"索引页: {OUT_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()