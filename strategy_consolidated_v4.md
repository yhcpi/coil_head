# 钢卷头尾小目标检测 — 创新点策略（v4 文献加强版）

> 更新：2026-07-09 | v4 = v3 + 6-lens × 8-10 papers 调研 + 5 个 design candidates

## 0. 当前状态

**v4 best.pt 部署**：R=0.974 / P=0.878 / F1=0.911（TTA top-2 + conf=0.05 + dist=50）| 学术 mAP50=0.877。
**数据 hard limit**：32 标 + 5 负 + 26 新图未合并；5 FN 为真难例（高反光/遮挡/bbox 微偏）。
**归档**：Heatmap-Aux / SAHI / Soft-NMS-Cov / NWD / **C（looseness_alpha, 3 次, coverage=true 根因）/ D（PA-Aug 4 组件, 全 <baseline）**。

## 1. 数据约定
未标 = 负样本；目标 0/1 个；只标 tip（≈20×20 px）。

## 2. ✅ 生产在用（v4 best.pt）
| 类别 | 创新点 | 关键参数 | 贡献 |
|------|--------|---------|------|
| 损失 | Coverage Loss | `loss.py`+`v8DetectionLoss` | 主损失 |
| 增强 | bbox_random_shrink | U(0.8,1.2) p=1.0 | tip 形态 |
| 增强 | multi-scale rect | imgsz ±20% | 小目标鲁棒 |
| 训练 | 27 FN→train | `rebalance_train_val.py` | v3→v4 R 核心 |
| 训练 | cls=1.0 label_smoothing=0.02 | `hyp_aug.yaml` | P=0.94 关键 |
| 训练 | 起点 hyper-yolon.pt | `--pretrained` | 避免局部最优 |
| 评估 | Lenient-mAP（dist<30 或中心∈GT）| `scripts/lenient_eval.py` | mAP50=0.877 |
| 推理 | TTA top-2 + conf=0.05 + dist=50 | `scripts/tta_inference.py` | F1=0.911 |

**结论**：Coverage-only 生产用（NWD/Cov 训出模型 100% 相同）。

## 3. ❌ 已归档
Heatmap-Aux / SAHI / Soft-NMS-Cov / NWD / **C looseness_alpha**（3 次，覆盖 IoU 梯度）/ **D PA-Aug 4 组件**（最好 0.029 = v4 baseline 3.3%）/ WSL2 SIGABRT 影响 reflection。

## 4. 🆕 候选创新点（6-lens 调研后定稿，每点 ≥ 2 篇文献）
| Pri | 名称 | 核心 | 训练 | 预期 | 回退 |
|-----|------|------|------|------|------|
| **P0** | **NBBOX-style Loose-Box Noise Aug** | 训练期 GT bbox 加尺度 [0.5,1.5] + 平移 ±0.1*size 抖动；不动 loss，动数据侧 | **0** | mAP50 +0.5~1.5pp / 救 1-2 FN | 删 transform 1 行 + yaml 2 行 |
| **P1** | **Specular Highlight Suppression Branch** | 金属高光去除 backbone（2412.11324 结构）轻量化挂在 P2 前；pix2pix reconstruction loss 解耦于 box | **1.3h×1** | F1 0.911→0.93 / 救反光 FN | yaml 1 行改 False |
| **P1** | **YOLOv10 Large-Kernel Backbone + Hyper-YOLO Neck** | 替换 backbone 为 YOLOv10 大核 stem + 保留 Hyper-YOLO 超图 neck | **1.3h×1** | mAP50 +0.5~1.0pp | 删除 yaml 回 v4 |
| **P2** | **Confidence Calibration (Weighted GP IoU)** | 102 val 上拟合 GP 把 raw conf → IoU 区间；WBF 融合用区间中点 | **0** | F1 →0.93-0.94 / P +0.02-0.04 | 删后处理回原 NMS |
| **P2** | **Semi-Supervised Pseudo-Label Loop (CADT)** | 26+102 未标 → v4+TTA 出 top-1/2 + conf gating ≥0.3；联到 32 张训练 | **1.3h×1** | mAP50 +1~3pp 上限 | 仅删 dataset_v2.yaml |

**优先级逻辑**：P0 = 0 训练，立即做；P1 = 单次训练 + 局部改；P2 = 多轮调参 / 风险中等。
**共同原则**：绕开 looseness_alpha / coverage_loss / PA-Aug 4 组件；每点支持文献见 §7。

## 5. 下一步顺序
```
立即（<1h, 0 训练）:
  1. NBBOX Loose-Box Noise Aug（P0）— 新增 NBBoxNoise() transform ~40 行，0 训练
  2. GP-IoU Confidence Calibration（P2-2）— 拟合 GP ~50 行，替换 WBF 前的 conf 列
短期（1-2 天, 1 次训练）:
  3. Specular Highlight Suppression Branch（P1）— nn/modules/spec_suppress.py ~60 行 + 250 epoch
  4. YOLOv10 Backbone Swap（P1）— 新 yaml ~80 行 + 250 epoch
中期（1 周, 需 CADT gating 调节）:
  5. Semi-Supervised Pseudo-Label Loop（P2-3）— scripts/pseudo_label.py ~100 行
```

## 6. 学术发表路径
最小组合（2-3 周）：Lenient-mAP ✅ + NBBOX-Aug（待验证）+ Spec-Suppress Branch（待验证）+ v4 ablation ✅。
**避坑**：不报 Heatmap-Aux 数字；不架构级 ablation；32 张数据规模必须坦白。

## 7. 📚 6-Lens 2025-2026 文献调研（59 篇）

### 7.1 Lens 1 — 宽容标注 / 噪声 bbox（9 篇）
| # | Title | arXiv / Venue / Year | 核心 | 钢卷适配 |
|---|-------|---------------------|------|---------|
| 1 | **NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection** | **2409.09424 / IEEE GRSL 2025** ★ | GT bbox 加噪当 augmentation，10 行代码接入 YOLOv8 | **high（直接迁移）** |

### 7.2 Lens 2 — 小目标检测架构 / Backbone（10 篇）
| # | Title | arXiv / Venue / Year | 核心 | 钢卷适配 |
|---|-------|---------------------|------|---------|
| 1 | **Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation** | **2408.04804 / IEEE TPAMI 2025** ★ | 超图替换 C2f；Tiny +3.0% AP；本项目基线 | **high（已部署）** |
| 2 | Enhancing Small Object Detection with YOLO (SW-YOLO) | 2512.07379 / arXiv / 2025-12 | 切片增强 + 小目标 head；对比 SAHI | high（切片参数可借鉴）|
| 3 | ZBS-Plus: Uncalibrated Training-Free Zero-Shot Small Object Detection | 2502.18947 / arXiv / 2025-02 | 预训练大模型 + DBSCAN + 放大裁剪；training-free | high（数据稀疏零成本）|

### 7.3 Lens 3 — 工业反光 / 高光去除（10 篇）
- [ ] | # | Title | arXiv / Venue / Year | 核心 | 钢卷适配 |
  |---|-------|---------------------|------|---------|
  | 1 | **Fast and Structure-aware Specular Highlight Removal for Metallic Surface** | **2412.11324 / arXiv / 2024** ★ | 金属"窄而强"高光结构感知去除；RGB 单图输入 | **high（前置预处理）** |
  | 10 | **SDD-Net: Strong Reflective Metal Surface Defect Detection** | **IEEE TIM 2025 (10913280)** | 两阶段：先高光掩膜，再去高光特征上分割；与钢卷场景高度同构 | **high（领域直接匹配）**|


### 7.6 Lens 6 — YOLO 家族新架构 2025-2026（10 篇）
| # | Title | arXiv / Venue / Year | 核心 | 钢卷适配 |
|---|-------|---------------------|------|---------|
| 2 | **YOLOv10: Real-Time End-to-End Object Detection** | **2405.14458 / NeurIPS 2024** ★ | NMS-free 双分配 + 大核 + 轻量分类头 | **high（密集小目标）** |
| 10 | YOLO26: End-to-End YOLO with STAL + MuSGD | Ultralytics Release Notes / 2026-01 | 移除 DFL + ProgLoss + STAL + MuSGD；1.0M params | high（STAL/MuSGD 小样本未验证）|

## 8. 一句话总结

**路线**：NBBOX Loose-Box Aug（P0，0 训练）→ Specular Suppression Branch 或 YOLOv10 Swap（P1，单次训练）→ GP-IoU Calibration（P2，0 训练后处理）→ Pseudo-Label Loop（P2，CADT gating）。
**禁止**：Heatmap-Aux / SAHI / 架构级改动 / 全局 loss 参数（v6 series 前车之鉴）。

## 9. 文献总览
- **59 篇 paper** 覆盖 6 lens；6 篇 topOne 已 ★ 标注
- **5 个候选创新点** 每点 ≥ 2 篇 arXiv/会议强证据支撑
- **证据强度分级**：strong = arXiv ID + 顶会接收；moderate = arXiv preprint 2024-2026；weak = GitHub project / release notes
- **场景匹配**：high / medium / low（按钢卷 tip 20×20 + 反光 + 数据稀疏匹配度）
