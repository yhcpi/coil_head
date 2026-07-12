"""把 A+B 14 张 hard neg 复制 N 份到 train/，配空 label"""
import shutil
from pathlib import Path

TRAIN_IMG = Path('/home/pi/projects/hyperyolo/data/coil/images/train')
TRAIN_LBL = Path('/home/pi/projects/hyperyolo/data/coil/labels/train')

HARD_NEG = [
    # 11 张真正的 hard neg（已验证：GT=0，pred 在金属缝隙/边缘反光处）
    # 排除 166/400/183 (模型预测已正确，P5 gt_count=0 是 bug)
    '463.png', '75.png',                                 # A 类：金属缝隙反光
    '556.png', '493.png', '62.png', 'd_38_.png', '274.png',  # B 类前 5
    '377.png', '588.png', '413.png', '84.png',           # B 类后 4
]
N_COPIES = 3  # 每张复制 3 份（共 33 张新增）

def expand():
    new_imgs = []
    for copy_idx in range(1, N_COPIES + 1):
        for img_name in HARD_NEG:
            stem = Path(img_name).stem
            # 新文件名：hn{copy_idx}_{原名}.png
            new_stem = f'hn{copy_idx}_{stem}'
            new_img_name = f'{new_stem}.png'
            new_lbl_name = f'{new_stem}.txt'

            src_img = TRAIN_IMG / img_name
            src_lbl = TRAIN_LBL / f'{stem}.txt'

            dst_img = TRAIN_IMG / new_img_name
            dst_lbl = TRAIN_LBL / new_lbl_name

            if dst_img.exists():
                print(f'  SKIP (exists): {new_img_name}')
                continue

            shutil.copy(src_img, dst_img)
            # label: 沿用原 label 内容（GT=0 的话就是空文件）
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)
            else:
                # 没有原 label → 创建空 label (GT=0)
                dst_lbl.touch()
            new_imgs.append(new_img_name)

    print(f'\\n新增 {len(new_imgs)} 张 hard neg 样本到 train/')
    print('新文件名示例:')
    for n in new_imgs[:6]:
        print(f'  {n}')


if __name__ == '__main__':
    expand()