"""把 val_batch 4×2 网格图裁剪成单检测大图。

原图 1920×1108 = 4 列 × 2 行，每格 480×554
输出：每张裁剪图含 1-2 个检测结果，足够大能看清
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
RUN_DIR = ROOT / 'runs' / 'coil_loss_ablation' / '02_coverage_v4_N_model'
CROP_DIR = ROOT / 'docs' / 'crops'
CROP_DIR.mkdir(parents=True, exist_ok=True)

# 1920×1108 = 4 cols × 2 rows, cell ≈ 480×554
# 裁剪方案：每张裁剪图取 2 列 × 1 行 = 960×554

# batch 0：顶部一行 + 第二行（两张图各显示一组检测）
crops = [
    # (batch_file, crop_box (left, top, right, bot), out_name)
    ('val_batch0_pred.jpg', (0, 0, 960, 554), 'success_1.jpg'),
    ('val_batch0_pred.jpg', (960, 0, 1920, 554), 'success_2.jpg'),
    ('val_batch1_pred.jpg', (0, 0, 960, 554), 'success_3.jpg'),
    ('val_batch1_pred.jpg', (0, 554, 960, 1108), 'success_4.jpg'),
    ('val_batch2_pred.jpg', (960, 0, 1920, 554), 'success_5.jpg'),
]

for src, box, dst in crops:
    p = RUN_DIR / src
    out = CROP_DIR / dst
    with Image.open(p) as im:
        crop = im.crop(box)
        crop.save(out, 'JPEG', quality=92)
    print(f'  ✓ {dst}: {crop.size}')

print(f'\n裁剪图目录: {CROP_DIR}')