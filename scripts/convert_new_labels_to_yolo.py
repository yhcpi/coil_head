#!/usr/bin/env python3
"""把 data/new_label/ 下的 labelme JSON 转成 YOLO 格式，覆盖到 data/coil/labels/。

操作流程：
  1. 备份现有 labels 到 data/coil/labels_backup_<timestamp>/
  2. 解析每个 JSON（只支持 rectangle shape）
  3. 转成 YOLO 格式: class x_center y_center w h（归一化）
  4. 覆盖到 data/coil/labels/{train,val}/<stem>.txt
  5. 删除 data/coil/labels/{train.cache, val.cache} 让 ultralytics 重新读
"""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NEW_LABEL_DIR = PROJECT_ROOT / "data" / "new_label"
LABELS_DIR = PROJECT_ROOT / "data" / "coil" / "labels"
SPLITS = ["train", "val"]


def backup_labels() -> Path:
    """备份整个 data/coil/labels/ 到 data/coil/labels_backup_<timestamp>/"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_ROOT / "data" / "coil" / f"labels_backup_{ts}"
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    # 复制 labels 目录
    shutil.copytree(LABELS_DIR, backup_dir)
    # 移除 .cache 文件（备份里不需要）
    for cache in backup_dir.glob("*.cache"):
        cache.unlink()
    print(f"  备份到: {backup_dir}")
    return backup_dir


def json_to_yolo(json_path: Path, img_w: int, img_h: int) -> str:
    """解析单个 labelme JSON，返回 YOLO 格式字符串（多行）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        return ""

    lines = []
    for shape in shapes:
        if shape["shape_type"] != "rectangle":
            print(f"  [WARN] {json_path.name}: 非 rectangle shape ({shape['shape_type']})，跳过",
                  file=sys.stderr)
            continue
        pts = shape["points"]
        if len(pts) != 2:
            print(f"  [WARN] {json_path.name}: rectangle 期望 2 points，实际 {len(pts)}",
                  file=sys.stderr)
            continue
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        # 确保 (x1,y1) 是左上角，(x2,y2) 是右下角
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        # 归一化
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        # class_id = 0 (coil_head, 与 data.yaml 一致)
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def convert_split(split: str, stats: dict):
    """转换一个 split (train/val)"""
    src_dir = NEW_LABEL_DIR / split
    dst_dir = LABELS_DIR / split
    if not src_dir.exists():
        print(f"  [SKIP] {src_dir} 不存在")
        return
    for json_path in sorted(src_dir.glob("*.json")):
        stem = json_path.stem
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 优先用 JSON 里的 imageWidth/imageHeight
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            # 回退到 imagePath 找图片
            from PIL import Image
            # 解析 imagePath（Windows 风格 ..\..\coil\images\train\150.png）
            ip = data.get("imagePath", "").replace("\\", "/")
            # 找文件名
            img_name = Path(ip).name
            img_path = PROJECT_ROOT / "data" / "coil" / "images" / split / img_name
            if not img_path.exists():
                print(f"  [WARN] {json_path.name}: imageWidth/Height 缺失且图片 {img_path} 不存在",
                      file=sys.stderr)
                continue
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        # 转换
        yolo_content = json_to_yolo(json_path, img_w, img_h)
        if not yolo_content:
            print(f"  [SKIP] {json_path.name}: 转换结果为空（无有效 shape）")
            continue
        # 写覆盖
        dst_path = dst_dir / f"{stem}.txt"
        # 备份旧内容（如果存在）— 实际已经被整体备份了，这里只记一下
        if dst_path.exists():
            old = dst_path.read_text()
            if old.strip() != yolo_content.strip():
                stats["changed"] += 1
            else:
                stats["unchanged"] += 1
        else:
            stats["new"] += 1
        dst_path.write_text(yolo_content, encoding="utf-8")
        stats["converted"] += 1


def clear_cache():
    """删除 ultralytics 的 .cache 文件，下次训练会重新生成。"""
    for cache in (LABELS_DIR / "train.cache", LABELS_DIR / "val.cache"):
        if cache.exists():
            cache.unlink()
            print(f"  删除 cache: {cache}")


def main():
    print("=" * 60)
    print("labelme JSON → YOLO 格式转换")
    print("=" * 60)
    print()

    print("1. 备份现有 labels ...")
    backup_dir = backup_labels()
    print()

    stats = {"converted": 0, "changed": 0, "unchanged": 0, "new": 0}
    for split in SPLITS:
        print(f"2. 转换 {split}/ ...")
        convert_split(split, stats)
        print()

    print(f"3. 删除 .cache ...")
    clear_cache()
    print()

    print("=" * 60)
    print("完成")
    print(f"  总转换: {stats['converted']} 个文件")
    print(f"    - 内容变化: {stats['changed']}")
    print(f"    - 内容相同: {stats['unchanged']}")
    print(f"    - 新增: {stats['new']}")
    print(f"  备份目录: {backup_dir}")
    print(f"  原始 JSON: {NEW_LABEL_DIR}（未删，可回退）")
    print("=" * 60)


if __name__ == "__main__":
    main()