# Hyper-YOLO 源码补丁清单

本目录保存相对于 [Hyper-YOLO](https://github.com/weihua13071/Hyper-YOLO) 上游的 **修改/新增** 文件，
**未修改** 的上游文件不放在这里，保持仓库精简。

## 使用方法

```bash
# 1. 克隆官方 Hyper-YOLO
git clone https://github.com/weihua13071/Hyper-YOLO.git
cd Hyper-YOLO

# 2. 覆盖本目录的文件
cp -r /path/to/this-repo/src/hyper_yolo_patches/ultralytics/* ultralytics/

# 3. 验证
python -c "from ultralytics.data.augment import BBoxRandomShrink; print('OK')"
```

## 修改/新增清单

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `ultralytics/data/pa_aug.py` | ~270 | PA-Aug (Physics-Aware Augmentation) 物理感知增强模块<br>4 个独立组件: motion / reflection / occlusion / noise<br>为钢卷头尾小目标场景特化设计 |
| `ultralytics/data/mosaic_neg.py` | ~85 | MosaicNeg: 在 Mosaic 中插入负样本（空图/纯背景）<br>解决 99 张图 batch 里负样本不足的问题 |

### 修改文件

| 文件 | 修改位置 | 上游 → 本项目 | 说明 |
|------|---------|--------------|------|
| `ultralytics/data/augment.py` | line 21 | 上游无 `from .pa_aug import ...` | 注册 PA-Aug 模块入口 |
| `ultralytics/data/augment.py` | line 961-1003 | 上游 Compose 末尾无 coil-specific hooks | 插入 `BBoxRandomShrink` + `RandomScaleRect` hook |
| `ultralytics/data/augment.py` | line 1173-1280 | 上游无 coil classes | 新增 class `BBoxRandomShrink` (类 Coverage 近似) + class `RandomScaleRect` (多尺度) |
| `ultralytics/utils/loss.py` | line 94-105 | 上游无 `coverage_loss()` | 新增 Coverage Loss 函数 (NWD-like 区域重叠度量) |
| `ultralytics/utils/loss.py` | line 186-260 | 上游 `v8DetectionLoss.__init__` 无 nwd/coverage 参数 | 集成 NWD + Coverage 到 loss 计算管线 |
| `ultralytics/models/yolo/detect/train.py` | line 121-137 | 上游 `train()` 函数只传 3 个 keys | 修复 cli args 丢失 bug (epochs/batch/name 全部保留) |
| `ultralytics/models/yolo/detect/train.py` | line 140-142 | 上游 `__main__` 走 entrypoint | 让 sys.argv 生效，不再 hardcode debug 参数 |

## 配套配置

`data/coil/hyp_*.yaml` 系列（17 个）是搭配本项目源码修改的训练超参配置：

- `hyp_v5_nwd_only.yaml` / `hyp_v5_nwd_coverage.yaml` — NWD/Coverage 调参起点
- `hyp_v6_*.yaml` / `hyp_v7_*.yaml` — 各创新点（Bayes prior / Heatmap aux / PA-Aug）
- `hyp_v8_coil_*.yaml` — 最终生产配置（NWD + cls=0.5 弱 aug）

`data/coil/data.yaml` / `data_v2.yaml` / `data_v3.yaml` 是数据集定义（不含图片，标签单独提供）。

## 上游兼容性

- 基于 Hyper-YOLO commit `e9b3fd6`（官方最新 main 分支）
- 适用于 `ultralytics>=8.0.0`，Python 3.12，PyTorch 2.x
- 修改遵循"外科手术式"原则，未改动 ultralytics 框架核心 API

## 项目论文/参考

- NWD: https://arxiv.org/abs/2110.13389
- Hyper-YOLO: https://arxiv.org/abs/2408.04884
- Coverage Loss (本项目创新点): 待发表
- PA-Aug (本项目创新点): 待发表
- NBBOX: https://arxiv.org/abs/2409.09424（已下载到 `repos/NBBOX/` 作为参考）