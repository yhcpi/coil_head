# Hyper-YOLO 钢卷头尾小目标检测

基于 [Hyper-YOLO](https://github.com/weihua13071/Hyper-YOLO) 的钢卷头尾小目标检测项目。
针对工业场景中 ~20×20 像素的"tip"目标做了**框架级改造**和**部署优化**。

## 关键成果

| 指标 | 学术 mAP50 | 部署 F1 |
|------|-----------|---------|
| 上游 Hyper-YOLO baseline | 0.877 | — |
| **本项目 (v8 NWD weak aug + TTA)** | **0.869** | **0.929** |

部署 F1 超过学术 mAP 来自 **TTA (orig+hflip+vflip) + top-1 + dist=30 + conf=0.10** 后处理。

## 项目结构

```
hyperyolo/
├── src/hyper_yolo_patches/   # Hyper-YOLO 源码修改补丁（5 个文件）
│   ├── PATCHES.md            # 修改清单
│   └── ultralytics/
│       ├── data/
│       │   ├── augment.py    # (修改) 注册 BBoxRandomShrink/RandomScaleRect hooks
│       │   ├── pa_aug.py     # (新增) PA-Aug 物理感知增强 (motion/reflection/occlusion/noise)
│       │   └── mosaic_neg.py # (新增) MosaicNeg 负样本增强
│       ├── utils/
│       │   └── loss.py       # (修改) Coverage Loss + NWD 集成
│       └── models/yolo/detect/
│           └── train.py      # (修改) CLI args 保留修复
├── data/coil/                # 训练配置 + 标签（图片单独提供）
│   ├── data.yaml
│   ├── hyp_v8_coil_*.yaml    # 训练超参
│   └── labels/               # YOLO 格式标注
├── scripts/                  # 训练/评估/可视化脚本
│   ├── run_v8_nwd_v1_weak_aug_full.sh   # 当前最佳生产训练
│   ├── postprocess_tta_topk_dist.py     # TTA + top-k + dist NMS 后处理
│   ├── visualize_tta_best.py            # 部署可视化
│   └── labelme2yolo.py                  # labelme → YOLO 格式转换
├── docs/                     # 设计决策文档（postmortem/tutorial）
└── repos/                    # 第三方参考仓库（gitignore 排除）
```

## 快速复现

### 环境

- Python 3.12, PyTorch 2.x, CUDA 12.x
- ultralytics 8.0+

### 1. 克隆官方 Hyper-YOLO 并打补丁

```bash
git clone https://github.com/weihua13071/Hyper-YOLO.git
cd Hyper-YOLO
cp -r /path/to/hyperyolo/src/hyper_yolo_patches/ultralytics/* ultralytics/
pip install -e .
```

### 2. 准备数据

把图片放到 `data/coil/images/{train,val}/`（标签已在 `data/coil/labels/`）。
图片数据集来源见 `docs/INNOVATIONS_5_6_TUTORIAL.md`。

### 3. 训练（弱 aug + NWD，弱 aug 配置胜过强 aug）

```bash
bash scripts/run_v8_nwd_v1_weak_aug_full.sh
# 250 epoch, save_period=10 (保留 epoch ckpts + best.pt)
```

### 4. 部署推理（TTA + 后处理）

```python
from ultralytics import YOLO
from scripts.postprocess_tta_topk_dist import tta_predict, topk_dist_nms
# 见 scripts/visualize_tta_best.py
```

## 创新点 / 实验结论

详见 `docs/` 目录：

- `INNOVATIONS_5_6_TUTORIAL.md` — NWD + Coverage 创新点详解
- `nbbox_noise_failure_postmortem.md` — NBBoxNoise 失败分析
- `v5_deployment.md` — 部署策略与 hyper 选择

## 实验 run 状态

最佳生产模型：
- `runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full/weights/best.pt` (ep198, 学术 mAP50=0.869)
- 部署 F1=0.929 (TTA + top-1 + dist=30 + conf=0.10)

## License

本项目代码遵循上游 Hyper-YOLO 的 GPL-3.0 协议。

论文参考：
- NWD: arXiv:2110.13389
- Hyper-YOLO: arXiv:2408.04884
- NBBOX: arXiv:2409.09424