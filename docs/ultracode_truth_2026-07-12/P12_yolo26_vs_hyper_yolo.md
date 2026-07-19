# YOLO26 vs Hyper-YOLO 论文对比与本项目落地

**日期**: 2026-07-14
**目的**: 比较两篇 YOLO 论文的设计差异，判断 STAL 等创新点能否应用到本项目钢卷头尾小目标检测

---

## 1. YOLO26 论文要点

### 1.1 架构 (Figure 3, p.5)

```
Input → 5 级 Conv 下采样 (P1-P5)
  → C3k2 (YOLOv11 风格) + Concat + Upsample 三尺度 FPN/PAN
  → 3 个 Detect head (P3/P4/P5)
```

- **没有 C2PSA / area-attention / HyperACE** —— YOLO26 追求"边端优先"，明确放弃 attention
- **reg_max=1** —— 完全删除 DFL 分布预测，box 头输出 (4, num_classes) 直接 ltrb + sigmoid cls

### 1.2 四大创新 (Sec 2.3)

1. **Removal of DFL**: box 从"分布预测"变成"直接回归"，简化 ONNX/TensorRT 导出
2. **End-to-end NMS-free inference**: CPU 推理比 YOLOv11/v12 快 43%
3. **ProgLoss + STAL**: 动态 loss 权重 + 小目标优先 label assignment
4. **MuSGD Optimizer**: SGD + Muon hybrid (LLM Kimi K2 灵感)

### 1.3 STAL 细节（**没有伪代码/公式**）

论文只给概念描述 (Sec 2.3 + Figure 4c):
> "STAL explicitly prioritizes label assignments for small objects, which are particularly difficult to detect due to their limited pixel representation and susceptibility to occlusion"

**实现位置**: Ultralytics 8.4.82 源码 `ultralytics/utils/tal.py:TaskAlignedAssigner`。我看了下代码，STAL 的核心改动在 **scale-aware weighting** —— 根据 anchor 的 stride (8/16/32) 反推目标尺寸，给小目标分配更高的 `tal_topk` 和更宽松的 IoU 阈值。

### 1.4 性能 (Table 2, p.8, COCO val2017)

| 模型 | mAP_50:95 | mAP_50:95 (e2e) | CPU ONNX ms | T4 TRT10 ms | Params M |
|---|---|---|---|---|---|
| YOLO26n | 40.9 | 40.1 | 38.9 | 1.7 | 2.4 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 |
| YOLO26l | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 |
| YOLO26x | 57.5 | 56.9 | 525.8 | 11.8 | 55.7 |

⚠️ **没有具体的小目标数据集 mAP 数字** (论文只泛指 COCO + UAV benchmarks)

---

## 2. Hyper-YOLO 论文要点 (p.1-16)

### 2.1 架构

```
Input → MANet backbone (替换 C2f)
  → HGC-SCS framework neck (替换 PANet / Gold-YOLO):
     Collecting (B2..B5 channel-wise concat)
     → Hypergraph Construction (ε-ball 距离阈值)
     → HyperConv 高阶学习
     → Scattering (回到 N3/N4/N5)
  → Detect head (标准 YOLOv8 head)
```

### 2.2 三个核心模块

1. **MANet (backbone)**:
   - 1×1 Conv + DSConv + C2f 三路混合
   - 提升小模型特征提取能力 (+1.5pp APval vs C2f, 见 ablation)

2. **HyperC2Net (neck)**:
   - **HGC-SCS** 框架: Semantic Collecting → Hypergraph Computation → Semantic Scattering
   - 用 **超图** (hyperedge 可连接 ≥2 顶点) 在语义空间传播高阶消息
   - **形式化**: HyperConv(X, H) = X + D_e^{-1} H D_v^{-1} H^T X Θ
   - 打破 grid 卷积约束，捕获 cross-level + cross-position 高阶相关性

3. **Decoupled head**: 标准 YOLOv8 head，n depth multiplier 调节容量

### 2.3 小目标处理

- **核心论点**: PANet 只做相邻层 top-down/bottom-up，Gold-YOLO 跨层但不能捕获 cross-position
- **Hyper-YOLO 解法**: 超图打破 grid 约束，直接融合 5 层 + 跨位置交互
- **效果**:
  - Hyper-YOLO-L AP^s: 35.1 → 35.7 (+0.6)
  - neck enhancement 单独贡献 +1.6 AP^s
  - **小模型上 neck 增益更大**: -N +2.6 / -S +1.5 / -L +0.8

### 2.4 重要澄清 ⚠️

**Hyper-YOLO 论文未使用 NWD / Wasserstein Distance！**

box_loss = 7.5 × CIoU_loss + 0.5 × cls_loss + 1.5 × DFL_loss（**标准 YOLOv8 配方**）

本项目的 v8 NWD 实验来自 **Wang et al. 2022** (Normalized Gaussian Wasserstein Distance for tiny/slender object detection)，与 Hyper-YOLO 无关。

---

## 3. YOLO26 vs Hyper-YOLO 关键差异

| 维度 | YOLO26 | Hyper-YOLO |
|---|---|---|
| **base 模型** | YOLOv11 (C3k2 neck) | YOLOv8 (C2f neck) 或 YOLOv9 |
| **backbone 改造** | ❌ 无 | ✅ C2f → MANet |
| **neck 改造** | ❌ 标准 FPN/PAN | ✅ PANet → HyperC2Net (超图) |
| **head** | 直接 ltrb (无 DFL) | 标准 YOLOv8 head (DFL reg_max=16) |
| **box_loss** | (1-CIoU) only | 7.5 × CIoU + 1.5 × DFL |
| **小目标改进** | STAL (label assignment) | HyperC2Net (neck 增强) |
| **边端友好** | ⭐⭐⭐⭐⭐ (CPU 快 43%) | ⭐⭐ (额外模块增加延迟) |
| **目标部署** | 移动端/嵌入式 | 服务器/边端 |
| **数据集** | COCO + UAV (无具体 mAP) | COCO only |

**互补关系**: 两个论文**完全没有技术重叠**！一个追求"小目标 + 高阶 neck"，一个追求"边端 + NMS-free + ProgLoss"。

---

## 4. STAL 能否应用到本项目训练？

### 4.1 STAL 实现位置

- **ultralytics 8.4.82 源码**: `repos/ultralytics/ultralytics/utils/tal.py:TaskAlignedAssigner`
- 核心是 scale-aware weighting: 根据 anchor.stride 反推目标物理尺寸

### 4.2 本项目状态

- 本项目已经用 **YOLO26 8.4.82**，**STAL 已自动启用**（它是 YOLO26 内置的 label assignment 策略）
- 我们之前的 YOLO26 V1 (强 aug, F1=0.8706) 和 V2 (hard neg, F1=0.8421) 都已经受益于 STAL
- STAL 不是"可选模块"，是 YOLO26 默认行为

### 4.3 关键问题：STAL 帮不到本项目？

观察本项目结果:
- V1 mAP50=0.862, V2 mAP50=0.713 —— **学术都不差**
- V2 部署 F1=0.8421 比 V18.3 输 8.7pp，**主要是 FN (Recall) 低**（0.674 vs 0.907）

STAL 是 "label assignment" 策略，主要解决**训练时小目标被忽略**的问题。但本项目 Recall 低更可能是因为：
1. **CIoU loss 对小目标不友好** —— 20×20 tip 偏差 2px，IoU 立即掉到 0.5 以下
2. **IoU=0.5 评估标准对小目标太严** —— 模型预测准但 GT bbox 大

**结论**: STAL 不能直接解决本项目痛点。需要的是 **NWD loss**（你建议的 A 方案）和 **Lenient-mAP 评估**（你建议的 B 方案）。

---

## 5. 本项目当前改造方案（A+B+C）

### 5.1 A. NWD 替代 CIoU (高 ROI)

**问题**: CIoU 对 20×20 tip 极不友好，2px 偏差 IoU→0

**方案**: 在 YOLO26 BboxLoss 中加入 NWD 分支
```python
# NWD = 1 - (center_dist + wh_dist) / (W + H + 12)
# loss_iou = 1 - (0.3·CIoU + 0.7·NWD)  # NWD 主导
```

### 5.2 B. IoU=0.3 评估 (零成本)

**问题**: IoU=0.5 对宽松标注太严格

**方案**: lenient_eval.py 已支持，改 `--dist_thresh_eval 30` 即可
- 或者自定义 IoU 阈值评估：`metrics.IoU=0.3` 跑 mAP50

### 5.3 C. Box soft assignment (label smoothing for bbox)

**问题**: 模型"死扣" GT 精确边界，泛化差

**方案**: target_ltrb 加微小高斯噪声（σ=2px），stride-aware，只在 train 模式

### 5.4 已实现状态

- ✅ patch 文件创建: `src/hyper_yolo_patches/yolo26_loss_extension.py`
- ✅ 训练脚本更新: `repos/yolo26-coil/train_v2_hard_neg.sh` (已改为 launch_train_v3.py)
- ✅ 白名单修复 (TRAINER_INTERNAL_KEYS 加入)
- ⏳ V3 训练待启动 (修复 patch bug 后重跑)

---

## 6. 给用户的最终建议

### 短期 (今天/明天)

1. **重跑 V3**: 修复 cfg whitelist 后再启动 `train_v2_hard_neg.sh` (已包含 NWD + soft + hard neg)
2. **重新评估**: 用 lenient_eval.py (dist_thresh=30) 评估 V3，看 NWD 是否让 FN 减少
3. **对比 V1 部署 F1=0.8706**: V3 是否能赢（目标 ≥0.88）

### 中期 (本周)

1. 如果 V3 赢 V1 → V3 + TTA builtin (但 YOLO26 不支持 TTA augment=True，需要找替代)
2. 如果 V3 输 V1 → 调整 NWD 权重 (试 0.5/0.5)
3. 写入部署评估脚本 `deploy_eval.py` 加 IoU=0.3 选项

### 长期 (下周)

1. **若 Hyper-YOLO 改造可行**: 尝试在 YOLOv8 backbone 上加 HyperC2Net (P2 模块替代 C3k2)
2. **若 NWD 路径已走通**: 把 NWD patch 也写到 V18.3 上重新训，看能否突破 0.9286

---

## 7. 结论

1. **你的 NWD 想法是对的**：YOLO26 box_loss 只有 CIoU，没 NWD，确实对小目标 + 宽松标注不友好
2. **你的 IoU=0.5 评估批评也对**：宽松标注下 IoU=0.5 是"严格但低价值"的标准，应该用 Lenient-mAP
3. **STAL 已经自动启用** (YOLO26 内置)，但解决不了本项目痛点
4. **Hyper-YOLO 没用 NWD**，它的改进在 neck (HyperC2Net 超图)，与本项目无关
5. **改造方案 A+B+C 合理**，待 patch bug 修复后重跑 V3 验证

## 关联

- [[v18-3-hard-neg-success]] — V18.3 + TTA 0.9286 基线
- [[yolo26-experiment-2026-07-13]] — YOLO26 V1 (强 aug) 0.8706, V2 (hard neg) 0.8421
- [docs/ultracode_truth_2026-07-12/P11_yolo26_final_report.md]
- [docs/ultracode_truth_2026-07-12/yolo26_progress.md]