"""训练完成后跑所有对比实验，汇总 mAP/Recall/Precision。

用法：
    /home/pi/anaconda3/envs/hyper-yolo/bin/python scripts/compare_all.py
"""
import sys
import subprocess
from pathlib import Path

WEIGHTS_BASELINE = '/home/pi/projects/hyperyolo/repos/Hyper-YOLO/runs/coil_v3_baseline/weights/best.pt'
WEIGHTS_SAHI = '/home/pi/projects/hyperyolo/repos/Hyper-YOLO/runs/coil_v3_sahi/weights/best.pt'

VAL_V3 = '/home/pi/projects/hyperyolo/data/coil/images/val'
GT_V3 = '/home/pi/projects/hyperyolo/data/coil/labels/val'
VAL_SAHI = '/home/pi/projects/hyperyolo/data/coil_sahi/images/val'
GT_SAHI = '/home/pi/projects/hyperyolo/data/coil_sahi/labels/val'
DATA_YAML_V3 = '/home/pi/projects/hyperyolo/data/coil/data_v3.yaml'
DATA_YAML_SAHI = '/home/pi/projects/hyperyolo/data/coil_sahi/data.yaml'


def run(cmd):
    print(f'\n>>> {" ".join(cmd)}')
    return subprocess.run(cmd, capture_output=True, text=True)


def run_inference(name, weights, val_dir, gt_dir, data_yaml, conf=0.05):
    """跑一次推理并捕获结果。"""
    if not Path(weights).exists():
        print(f'  ⚠️ 权重不存在: {weights}')
        return None
    cmd = [
        '/home/pi/anaconda3/envs/hyper-yolo/bin/python',
        '/home/pi/projects/hyperyolo/scripts/sahi_inference.py',
        '--weights', weights,
        '--data_yaml', data_yaml,
        '--val_dir', val_dir,
        '--gt_dir', gt_dir,
        '--conf', str(conf),
        '--overlap', '0.2',
    ]
    if 'SAHI' in name.upper():
        # 不传 --sahi_off 即开启 SAHI
        pass
    else:
        # 关掉 SAHI 切片（用更小的切片，不切）
        # 实际做法：把 slice_h/w 设成大值（超过 val 图）→ 等价于不切
        cmd += ['--slice_h', '9999', '--slice_w', '9999', '--overlap', '0.0']
    r = run(cmd)
    if r.returncode != 0:
        print('  STDERR:', r.stderr[-500:])
        return None
    # 解析输出
    out = r.stdout
    metrics = {}
    for line in out.split('\n'):
        if 'mAP50:' in line:
            metrics['mAP50'] = float(line.split(':')[1].strip())
        elif 'mAP50-95:' in line:
            metrics['mAP50-95'] = float(line.split(':')[1].strip())
        elif 'Recall:' in line:
            metrics['Recall'] = float(line.split(':')[1].split('(')[0].strip())
        elif 'Precision:' in line:
            metrics['Precision'] = float(line.split(':')[1].split('(')[0].strip())
    return metrics


def main():
    print('=' * 60)
    print('SAHI 创新点对比实验汇总')
    print('=' * 60)

    experiments = [
        ('1. baseline + 直接推理',         WEIGHTS_BASELINE, VAL_V3,    GT_V3,    DATA_YAML_V3,    'no-sahi'),
        ('2. baseline + SAHI 推理',         WEIGHTS_BASELINE, VAL_V3,    GT_V3,    DATA_YAML_V3,    'sahi'),
        ('3. SAHI train + 直接推理',        WEIGHTS_SAHI,     VAL_SAHI,  GT_SAHI,  DATA_YAML_SAHI,  'no-sahi'),
        ('4. SAHI train + 直接推理(v3原图)', WEIGHTS_SAHI,     VAL_V3,    GT_V3,    DATA_YAML_SAHI,  'no-sahi'),
        ('5. SAHI train + SAHI 推理(v3原图)', WEIGHTS_SAHI,    VAL_V3,    GT_V3,    DATA_YAML_SAHI,  'sahi'),
    ]

    results = {}
    for name, weights, val_dir, gt_dir, data_yaml, mode in experiments:
        print(f'\n--- {name} ---')
        if mode == 'sahi':
            metrics = run_inference(name, weights, val_dir, gt_dir, data_yaml, conf=0.05)
        else:
            # no-sahi: 用大切片=直接推理
            metrics = run_inference(name, weights, val_dir, gt_dir, data_yaml, conf=0.001)
        if metrics:
            results[name] = metrics
            print(f'  mAP50={metrics.get("mAP50", "N/A"):.4f}  Recall={metrics.get("Recall", "N/A"):.4f}  P={metrics.get("Precision", "N/A"):.4f}')

    print('\n' + '=' * 60)
    print('汇总表')
    print('=' * 60)
    print(f'{"实验":<35} {"mAP50":>8} {"mAP50-95":>10} {"Recall":>8} {"Precision":>10}')
    print('-' * 75)
    for name, m in results.items():
        print(f'{name:<35} {m.get("mAP50", 0):>8.4f} {m.get("mAP50-95", 0):>10.4f} {m.get("Recall", 0):>8.4f} {m.get("Precision", 0):>10.4f}')


if __name__ == '__main__':
    main()