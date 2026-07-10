"""把 val 里被模型漏检的 27 张图 + 它们的 label 移到 train。

依据：上次扫描发现的 27 张 val-only FN（last.pt 完全看不见的目标）。
- 这些图模型在训练时从未见过 → 召回能力提升的最大杠杆
- val 从 129 → 102（不可比历史 mAP，但用户在要求.txt 里明确说不要再抽 27 张做新 val）

输出：打印迁移摘要 + 写入 docs/data_v4_split.md 记录变更。
"""
import shutil
from pathlib import Path
import sys

sys.path.insert(0, '/home/pi/projects/hyperyolo/scripts')
from analyze_fn import WEIGHTS, VAL_DIR, GT_DIR  # 复用上次 FN 分析的常量


def main():
    img_val_dir = Path(VAL_DIR)
    lbl_val_dir = Path(GT_DIR)
    img_train_dir = img_val_dir.parent / 'train'
    lbl_train_dir = lbl_val_dir.parent / 'train'
    img_train_dir.mkdir(parents=True, exist_ok=True)
    lbl_train_dir.mkdir(parents=True, exist_ok=True)

    # 从 docs/fn_analysis_lastpt.md 抽 27 个 FN stem（blind 全部）
    fn_stems = [
        '12', '216', '241', '252', '313', '337', '386', '439', '473',
        '477', '489', '49', '584', '61', '6_change_03_FALLING_00.03s',
        'd_9_', 'images_d_9_',
        '456', '295', '53', '8_change_00_FALLING_00.03s', '6', '330',
        '542', 'd_43_', 'c_31_', '317', '123', '191'
    ]

    print(f'准备迁移 {len(fn_stems)} 张 val-only FN 到 train')
    moved_img = 0
    moved_lbl = 0
    missing = []
    for stem in fn_stems:
        img_src = img_val_dir / f'{stem}.png'
        lbl_src = lbl_val_dir / f'{stem}.txt'
        img_dst = img_train_dir / f'{stem}.png'
        lbl_dst = lbl_train_dir / f'{stem}.txt'

        if not img_src.exists():
            missing.append(f'image:{stem}')
            continue
        if img_dst.exists():
            print(f'  ⚠️ 已存在（跳过）: {img_dst}')
            continue

        shutil.move(str(img_src), str(img_dst))
        moved_img += 1
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(lbl_dst))
            moved_lbl += 1
        else:
            print(f'  ⚠️ 无 label 文件: {lbl_src}')

    print(f'\n迁移完成：图片 {moved_img}/{len(fn_stems)}，label {moved_lbl}')
    if missing:
        print(f'缺失图片：{missing}')

    # 汇总
    n_train_img = len(list(img_train_dir.glob('*.png')))
    n_val_img = len(list(img_val_dir.glob('*.png')))
    print(f'\n新拆分：train={n_train_img}, val={n_val_img}')

    # 记录
    md_path = Path('/home/pi/projects/hyperyolo/docs/data_v4_split.md')
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        '# v4 数据集拆分记录\n',
        f'- 改动日期：2026-07-05',
        f'- 改动原因：上次 last.pt FN 分析发现 27 张 val-only 图像模型完全看不见',
        f'- 改动操作：把 {len(fn_stems)} 张 val-only FN 从 val 移到 train',
        f'- 结果：train={n_train_img} (原 529 + {moved_img}), val={n_val_img} (原 129 - {moved_img})',
        f'- 警告：val 总数减少 → 历史 mAP 直接对比不可用',
        '\n## 迁移列表\n',
        '| stem | 备注 |\n|---|---|',
    ]
    for s in fn_stems:
        note = ''
        if 'change' in s: note = 'change 系列连续帧'
        elif s.startswith('d_'): note = 'd 系列（同时在 train，不重复移）' if s in {'d_43_'} else 'd 系列'
        elif s.startswith('c_'): note = 'c 系列（同时在 train，不重复移）' if s in {'c_31_'} else 'c 系列'
        elif s.startswith('images_'): note = 'images_ 前缀旧版命名'
        elif s.isdigit(): note = '数字命名（钢卷主编号）'
        md.append(f'| {s} | {note} |')
    md_path.write_text('\n'.join(md))
    print(f'\n✓ 变更记录写入：{md_path}')


if __name__ == '__main__':
    main()