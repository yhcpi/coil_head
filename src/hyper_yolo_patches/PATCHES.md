# Hyper-YOLO + YOLO26 源码补丁清单

本目录保存相对于 [Hyper-YOLO](https://github.com/weihua13071/Hyper-YOLO) 上游的 **修改/新增** 文件，
**未修改** 的上游文件不放在这里，保持仓库精简。

YOLO26 部分用 **monkey-patch 注入**（不复制 ultralytics 源码到本仓库）——见 `yolo26_loss_extension.py`。

## 使用方法

### Hyper-YOLO（直接覆盖）
```bash
# 1. 克隆官方 Hyper-YOLO
git clone https://github.com/weihua13071/Hyper-YOLO.git
cd Hyper-YOLO

# 2. 覆盖本目录的文件
cp -r /path/to/this-repo/src/hyper_yolo_patches/ultralytics/* ultralytics/

# 3. 验证
python -c "from ultralytics.data.augment import BBoxRandomShrink; print('OK')"
```

### YOLO26（import hook 注入，不修改 ultralytics 源码）
```bash
# 训练入口脚本会在 ultralytics.cli.run() 之前 import 这个模块
import sys
sys.path.insert(0, '/path/to/this-repo/src/hyper_yolo_patches')
import yolo26_loss_extension  # 自动 install：替换 BboxLoss / v8DetectionLoss，
                               # 并在 cfg.check_dict_alignment 中加白名单
```

## 修改/新增清单

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `ultralytics/data/pa_aug.py` | ~270 | PA-Aug (Physics-Aware Augmentation) 物理感知增强模块<br>4 个独立组件: motion / reflection / occlusion / noise<br>为钢卷头尾小目标场景特化设计 |
| `ultralytics/data/mosaic_neg.py` | ~85 | MosaicNeg: 在 Mosaic 中插入负样本（空图/纯背景）<br>解决 99 张图 batch 里负样本不足的问题 |
| `ultralytics/nn/modules/spec_suppress.py` | ~150 | SpecSuppress: 结构感知反光抑制模块（创新点 v9）<br>Channel Attention + 高光 mask + 重建分支<br>自监督重建 loss (L1 + Sobel 梯度差)，无配对数据需求 |
| `yolo26_loss_extension.py` | ~290 | **YOLO26 BboxLoss 三合一扩展 (A+B+C)**<br>monkey-patch 注入，不修改 ultralytics 源码<br>**A**: `iou_loss_weight_nwd=0.7` + `iou_loss_weight_ciou=0.3` → 小目标友好回归（NWD 主导）<br>**B**: `reg_max=1` (YOLO26 默认) 自动 fallback 到 L1-normalized-by-imgsz，DFL=0 路径同样安全<br>**C**: `box_soft_sigma=2.0px` 给 FG target_bboxes 加 Gaussian noise 防 overfit，train-only<br>CLI 自定义 key 经 `cfg.check_dict_alignment` 白名单注入 |

### 修改文件

| 文件 | 修改位置 | 上游 → 本项目 | 说明 |
|------|---------|--------------|------|
| `ultralytics/data/augment.py` | line 21 | 上游无 `from .pa_aug import ...` | 注册 PA-Aug 模块入口 |
| `ultralytics/data/augment.py` | line 961-1003 | 上游 Compose 末尾无 coil-specific hooks | 插入 `BBoxRandomShrink` + `RandomScaleRect` hook |
| `ultralytics/data/augment.py` | line 1173-1280 | 上游无 coil classes | 新增 class `BBoxRandomShrink` (类 Coverage 近似) + class `RandomScaleRect` (多尺度) |
| `ultralytics/utils/loss.py` | line 94-105 | 上游无 `coverage_loss()` | 新增 Coverage Loss 函数 (NWD-like 区域重叠度量) |
| `ultralytics/utils/loss.py` | line 186-260 | 上游 `v8DetectionLoss.__init__` 无 nwd/coverage 参数 | 集成 NWD + Coverage 到 loss 计算管线 |
| `ultralytics/utils/loss.py` | line 309-360 | 上游 `v8DetectionLoss.__call__` 不叠加 recon loss | 创新点 v9: spec_recon_weight 控制的辅助 loss 通路 |
| `ultralytics/models/yolo/detect/train.py` | line 121-137 | 上游 `train()` 函数只传 3 个 keys | 修复 cli args 丢失 bug (epochs/batch/name 全部保留) |
| `ultralytics/models/yolo/detect/train.py` | line 140-142 | 上游 `__main__` 走 entrypoint | 让 sys.argv 生效，不再 hardcode debug 参数 |
| `ultralytics/models/yolo/detect/train.py` | `set_model_attributes` 末尾 | 上游无 spec_suppress hook | 创新点 v9: 默认关闭, hyp 开启时插入 SpecSuppress + 同步 head 层 i/f/save |
| `ultralytics/nn/modules/__init__.py` | 末尾 | 上游未导出 SpecSuppress | 注册 SpecSuppress 入口供 yaml/task.py 解析 |

## 配套配置

`data/coil/hyp_*.yaml` 系列（17+ 个）是搭配本项目源码修改的训练超参配置：

- `hyp_v5_nwd_only.yaml` / `hyp_v5_nwd_coverage.yaml` — NWD/Coverage 调参起点
- `hyp_v6_*.yaml` / `hyp_v7_*.yaml` — 各创新点（Bayes prior / Heatmap aux / PA-Aug）
- `hyp_v8_coil_*.yaml` — 最终生产配置（NWD + cls=0.5 弱 aug）
- `hyp_v9_spec_suppress.yaml` — 创新点 v9 起点（NWD weak + SpecSuppress off, 改 true 启用）

`data/coil/data.yaml` / `data_v2.yaml` / `data_v3.yaml` 是数据集定义（不含图片，标签单独提供）。

## YOLO26 Loss 扩展 (2026-07-14)

| 项 | 默认值 | 含义 | 论文/项目依据 |
|----|--------|------|--------------|
| `iou_loss_weight_nwd` | 0.7 | NWD 在 `loss_iou = (1 - sim)` 中的权重 | NWD 论文 (Wang et al. 2021, AI-TOD)，小目标距离度量 |
| `iou_loss_weight_ciou` | 0.3 | CIoU 在 `loss_iou` 中的权重 | 标准 YOLOv5/v8/v11 box_loss |
| `box_soft_sigma` | 0.0 (disabled) | FG target_bboxes 高斯软化 σ (像素) | label smoothing for box，防止 overfit |
| `box_soft_train_only` | True | noise 仅在 train mode 应用 | 不污染 eval 指标 |
| `nwd_constant` | 12.0 | NWD 指数缩放常数 (AI-TOD 推荐值) | NWD 论文 |

**B 路径（reg_max=1）兼容性**：YOLO26 默认 `reg_max=1`（论文 sec 2.3: "removal of DFL"），此时
`self.dfl_loss is None`。扩展保留上游 8.4.x 的 imgsz-normalized L1 路径，行为与官方一致。
若 `nwd` 分支开启，NWD 在 `loss_iou` 维度与 L1 在 `loss_dfl` 维度互不干扰。

**C 路径（box soft）stride-aware**：noise 按 per-anchor stride 等比缩放回像素坐标。
对 P3/P4/P5 三个尺度，相同像素 σ 在 feature space 的尺度分别是 σ/8、σ/16、σ/32。
这样无论 model 输出哪个尺度，扰动幅度在图上都一致。

**启用方式**：
- v3 训练入口：`repos/yolo26-coil/train_v3_nwd_soft_hard_neg.sh`
- v2 改造版：`repos/yolo26-coil/train_v2_hard_neg.sh`（已加 NWD/soft 参数 + 切到 launcher）
- launcher：`repos/yolo26-coil/launch_train_v3.py`（自动 install yolo26_loss_extension）

**关闭扩展**：把所有参数设为 0 或不传即可，patch 自动 fallback 到上游 CIoU+DFL 行为。

## 上游兼容性

- 基于 Hyper-YOLO commit `e9b3fd6`（官方最新 main 分支）
- 适用于 `ultralytics>=8.0.0`，Python 3.12，PyTorch 2.x
- 修改遵循"外科手术式"原则，未改动 ultralytics 框架核心 API
- 创新点 v9 默认关闭（spec_suppress=false），对现有 best.pt 推理零影响
- YOLO26 部分完全不修改 `repos/ultralytics/` 源码，monkey-patch 通过 `yolo26_loss_extension.install()` 在 import 时注入

## 项目论文/参考

- NWD: https://arxiv.org/abs/2110.13389
- Hyper-YOLO: https://arxiv.org/abs/2408.04884
- Coverage Loss (本项目创新点): 待发表
- PA-Aug (本项目创新点): 待发表
- NBBOX: https://arxiv.org/abs/2409.09424（已下载到 `repos/NBBOX/` 作为参考）
- Specular Highlight Removal (TII 2023, Chen et al., DOI 10.1109/TII.2023.3297613) — SpecSuppress 设计思想来源
- DHAN-SHR (arXiv 2407.12255) — Channel attention 灵感来源