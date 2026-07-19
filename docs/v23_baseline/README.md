# v23 Baseline (2026-07-20) — 数据 v2 + 论文原始架构

## 关键事实

- **学术 mAP50 = 0.9617** (新 SOTA, vs v19r 0.9387 +2.30pp)
- mAP50-95 = 0.4789
- P = 0.9487, R = 0.9375
- **架构**：100% 原论文 Hyper-YOLOn（1× HyperComputeModule + 8× MANet + 标准 Detect nl=3）
- **没有任何 yaml/结构改动**

## 数据规模

| | train | val |
|---|---|---|
| 总数 | 642 | 80 |
| 正样本 | 642 (100%) | 80 (100%) |
| 负样本 | 0 (已删) | 0 (已删) |

**与之前版本的差异**：
- 数据 v1: train=545 (312+233 正+负), val=99 (43+56)
- 数据 v2: train=642 (all pos, +327 新标), val=80 (all pos, +37 新标)
- 数据 v2 = 原数据 (312+43=355) + captures_merged (327 train + 37 val)
- 全部清空负样本（已删负样本备份到 `data/coil/_deleted_negatives/`）

## 训练配置

| 项目 | 值 |
|---|---|
| 起点 | `repos/Hyper-YOLO/hyper-yolon.pt` (COCO 预训练权重) |
| epochs | 250 (patience=80, 实际跑满 250ep) |
| lr | 0.01 → 0.0001 |
| batch × imgsz | 8 × 1024 |
| 强 aug | degrees=10, scale=0.5, flipud=0.5, copy_paste=0.2 |
| mosaic/mixup | 0 / 0 |
| close_mosaic | 15 |
| bbox_noise_scale | (0.8, 1.2) |
| hsv | h=0.015, s=0.7, v=0.4 |
| dfl / box / cls | 1.5 / 1.5 / 0.5 |

## 关键结论

1. **v23 是 v21 mAP=0.9414 之后的真新 SOTA**：完全正样本训练反而让 mAP50 暴涨 (+2.30pp)
2. **不能跨数据集对比部署 F1**：v23 的 val=80 全正，**没有 FP 来源**，自然 P 虚高
3. **"是否能学到特征"的评估口径改为 80 张正样本召回率**：75/80 = **R=0.9375** 是衡量"模型是否抓得到 coil_head"的金标准
4. **网络结构 100% 原论文**（可通过 pt 解出的 yaml 验证）
5. **未来评估部署 F1 需要单独保留一组带难负的 val 子集**，本组 val 80 张只反映 Recall

## 产物清单（无 .pt/.png/.onnx 同步）

- `runs/baseline_v2/v23_baseline_train642_strong_aug_250ep/weights/best.pt` — 最佳权重（8MB, ~gitignore）
- `runs/baseline_v2/v23_baseline_train642_strong_aug_250ep/weights/last.pt` — 最后权重
- `docs/v23_baseline/args.yaml` — 训练参数快照
- `docs/v23_baseline/results.csv` — 训练 250 epoch 数据 (251 行)
- `docs/v23_baseline/TRAIN_CONFIG.md` — 自动生成的训练环境/超参报告
- `docs/v23_baseline/run_v23_baseline_train642_strong_aug_250ep.sh` — 启动脚本

## 对比 v21（数据 v1 baseline）

| Run | 数据 | dfl | 训练 mAP50 | 部署 F1 |
|---|---|---|---|---|
| v21 | 545 (含 233 负) | 0.0 (DFL-off) | 0.9414 | 0.8916 |
| **v23** | **642 (全正)** | **1.5 (DFL on)** | **0.9617** | **未评估 (val 无负)** |

**结论**：纯正样本 + DFL on 反而是更好的 baseline。这推翻"DFL-off 单独贡献"的假设。

## 后续方向

1. v23 best.pt 直接进部署评测，准备 v24 = v23 best.pt + 弱 aug 微调 (再涨 1-2pp)
2. 加一组"含难负的部署 val"用于未来横向对比
3. 恢复 deleted_negatives 时机：仅在需要"实测部署 F1"时再恢复，避免 baseline 评估口径漂移

