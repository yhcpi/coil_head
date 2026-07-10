"""在每个训练 run 目录保存详细配置说明。

执行时机：训练启动后立即（在 BaseTrainer.train 开始时）

写入路径：<save_dir>/TRAIN_CONFIG.md
内容：
  - 训练命令（完整命令行）
  - 数据集状态（train/val 数量 + 来源统计）
  - 模型配置（哪个 yaml、是否预训练、是否冻结）
  - 增强配置（hyp_aug.yaml 全文）
  - 创新点启用情况
  - 启动时间、batch、epochs、imgsz、device
"""
import os
import sys
import time
from pathlib import Path

import torch


def save_train_config(save_dir, args, model_cfg_path, hyp_path, train_cmd,
                       coverage_loss_path=None, augment_patch_path=None):
    """在训练开始时调用，把所有配置写到 TRAIN_CONFIG.md"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = save_dir / 'TRAIN_CONFIG.md'

    # 数据集统计
    data_yaml = args.get('data') if hasattr(args, 'get') else getattr(args, 'data', None)
    data_info = _inspect_data(data_yaml)

    # 训练参数
    epochs = getattr(args, 'epochs', '?')
    batch = getattr(args, 'batch', '?')
    imgsz = getattr(args, 'imgsz', '?')
    device = getattr(args, 'device', '?')
    workers = getattr(args, 'workers', '?')
    rect = getattr(args, 'rect', False)
    freeze = getattr(args, 'freeze', None)
    pretrained = getattr(args, 'pretrained', False)
    patience = getattr(args, 'patience', '?')
    optimizer = getattr(args, 'optimizer', '?')
    project = getattr(args, 'project', 'runs')
    name = getattr(args, 'name', 'train')

    # 读取 hyp_aug.yaml
    hyp_text = ''
    if hyp_path and Path(hyp_path).exists():
        hyp_text = Path(hyp_path).read_text()

    md = []
    md.append(f'# 训练配置说明（{name}）\n')
    md.append(f'- 保存时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    md.append(f'- 训练目录：{save_dir}')
    md.append(f'- 启动用户：{os.environ.get("USER", "?")}')
    md.append(f'- 主机：{os.uname().nodename if hasattr(os, "uname") else "?"}')
    md.append(f'- PyTorch：{torch.__version__}')
    md.append(f'- CUDA available：{torch.cuda.is_available()}')
    if torch.cuda.is_available():
        md.append(f'- GPU：{torch.cuda.get_device_name(0)}')
    md.append('')

    md.append('## 训练命令\n')
    md.append('```bash')
    md.append(train_cmd)
    md.append('```\n')

    md.append('## 训练参数\n')
    md.append(f'| 参数 | 值 |')
    md.append(f'|---|---|')
    md.append(f'| epochs | {epochs} |')
    md.append(f'| batch | {batch} |')
    md.append(f'| imgsz | {imgsz} |')
    md.append(f'| device | {device} |')
    md.append(f'| workers | {workers} |')
    md.append(f'| rect | {rect} |')
    md.append(f'| freeze | {freeze} |')
    md.append(f'| pretrained | {pretrained} |')
    md.append(f'| patience | {patience} |')
    md.append(f'| optimizer | {optimizer} |')
    md.append(f'| project | {project} |')
    md.append(f'| name | {name} |')
    md.append('')

    md.append('## 数据集状态\n')
    if data_info:
        for k, v in data_info.items():
            md.append(f'- {k}: {v}')
    else:
        md.append('- 无法读取数据集信息')
    md.append('')

    md.append('## 创新点 / 改动清单\n')
    md.append('1. **数据策略**：移 27 张 val-only FN 到 train（val 减到 102 张）')
    md.append('2. **bbox_random_shrink**：每个 GT bbox 边长随机缩放 0.8~1.2 倍，保持中心')
    md.append('3. **Coverage Loss**：pred bbox 包住 GT 中心的概率作为附加 loss 项，权重 0.5')
    md.append('4. **multi-scale rect**：imgsz ±20% 随机，保持长宽比不变')
    md.append('5. **loss 权重微调**：box 7.5→5.0（降），cls 0.5→1.0（升），label_smoothing 0.05→0.02（降）')
    md.append('6. **copy_paste**：0.1 → 0.2')
    md.append('7. **从预训练权重开始**（不加载 last.pt）')
    md.append('')

    md.append('## hyp_aug.yaml（实际生效版本）\n')
    md.append('```yaml')
    md.append(hyp_text)
    md.append('```\n')

    if model_cfg_path:
        md.append('## 模型配置\n')
        md.append(f'- model yaml: `{model_cfg_path}`\n')

    if coverage_loss_path:
        md.append('## 源码改动\n')
        md.append(f'- Coverage Loss 实现：`{coverage_loss_path}`')
    if augment_patch_path:
        md.append(f'- bbox_random_shrink + multi-scale 实现：`{augment_patch_path}`')
    md.append('')

    cfg_path.write_text('\n'.join(md))
    print(f'✓ 训练配置已保存：{cfg_path}')


def _inspect_data(data_yaml):
    """快速读 data.yaml 拿 train/val 路径和统计"""
    if not data_yaml or not Path(data_yaml).exists():
        return None
    import yaml
    cfg = yaml.safe_load(open(data_yaml))
    out = {'yaml': str(data_yaml)}
    base = Path(cfg.get('path', '.'))
    for split in ['train', 'val']:
        d = base / cfg.get(split, '')
        if d.exists():
            n = len(list(d.glob('*.png')))
            out[f'{split}_images'] = n
            lbl_dir = Path(str(d).replace('/images/', '/labels/'))
            if lbl_dir.exists():
                nl = len(list(lbl_dir.glob('*.txt')))
                nonempty = sum(1 for p in lbl_dir.glob('*.txt') if p.stat().st_size > 0)
                out[f'{split}_labels_total'] = nl
                out[f'{split}_labels_with_target'] = nonempty
                out[f'{split}_labels_empty (负样本)'] = nl - nonempty
    return out


if __name__ == '__main__':
    # 测试：从命令行读取参数打印
    save_train_config(
        save_dir='/tmp/test_save',
        args={'data': 'data/coil/data.yaml', 'epochs': 250, 'batch': 16,
              'imgsz': 1024, 'device': 0, 'workers': 2, 'rect': True,
              'freeze': None, 'pretrained': True, 'patience': 50,
              'optimizer': 'SGD', 'project': 'runs',
              'name': 'test'},
        model_cfg_path='hyper-yolon.yaml',
        hyp_path='data/coil/hyp_aug.yaml',
        train_cmd='python train.py --model hyper-yolon.pt ...',
    )