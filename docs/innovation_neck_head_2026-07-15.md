# 钢卷头尾小目标 Neck / Head / Loss 创新点提案
**作者**: 袁昊宸 · **日期**: 2026-07-15 · **起点**: V18.3 (部署 F1=0.9286, best.pt=`runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt`)
**修订**: 2026-07-15 B1 节修订——RegOnly-Head 改为 **cv3 80→1 通道 objness**（保留 Hard Neg Crop 训练数据兼容），删除原"删 cv3 + cls_score=1.0"硬冲突方案

---

## 任务画像（先讲清楚再创新）

| 维度 | 数值 | 创新含义 |
|------|------|----------|
| 目标尺寸 | 50×50 px @ 1920×1080 (~0.12% 面积) | **极小目标**，P3 (stride 32) 感受野 32×32=1024 px² 已远大于目标 |
| 类别数 | 1 (`coil_head`) | 1 类 → cls 分支几乎无价值，**reg 分支才是瓶颈** |
| 标注 | 宽松（"一小撮"） | GT bbox 边界含 5-15px 噪声，**box_loss 需容差** |
| 训练数据 | 312 正 + 233 负 = 545 张 | 数据规模硬约束，**禁止引入大参数量模块** |
| 部署形态 | 单图单目标，max_det=1，conf 0.10~0.15 | NMS 退化，**post-process 创新无 ROI** |
| 现状瓶颈 | TP=39, FP=2, FN=4 | Recall 0.907 是主瓶颈（4 个 FN 是 hard sample） |

**创新禁区**（已 3+ 次实证不 work）：
- box_soft_relative=0.10 + NWD-only → box_loss 累积爆炸
- 多尺度 P2 检测头（v17 ep57 mAP50=0.034）
- 后处理 TTA / 几何硬规则 / 二阶段 CNN

**创新绿区**：
- Neck 的**跨尺度信息流**（小目标需要更多浅层细节注入）
- Head 的**回归分支解耦**（cls=1 时 reg 是 100% 价值）
- Loss 的**宽松标注容差**（box 边界 +5px 不应惩罚）
- 模型权重**后训练微调**（post-train refine, 不动架构）

---

## A. Neck 创新（3 个）

### A1. **DySample-Tip**：动态上采样只为小目标服务

**问题**: YOLO 默认用 nearest Upsample，1024×1024 特征图在 P3 之后直接 conv→reg。但小目标 tip 50×50 经 32× 下采样只占 **1.5 个像素**（亚像素！），最近邻上采样后亚像素位置被抹平，**预测 bbox 中心有 ±1.5px 量化误差**。

**方案**: 把 P3→P4 之间的 Upsample 换成 [DySample](https://arxiv.org/abs/2308.15020)（动态点偏移上采样，零参数），但**只在检测分支用**，分类分支保持原状（cls=1 不敏感）。

**实现位置**: `nn/modules/conv.py` 新增 `DySampleConv`，在 `tasks.py` 的 neck Parse() 处替换 `nn.Upsample`。

**预期**: tip 中心定位误差从 ±1.5px 降到 ±0.5px → 与 GT (5-15px 宽松标注) 中心距离 < 3px 的目标召回 +5%。

**成本**: +0 参数 / +0.3ms 推理（仅一处上采样）。

**风险**: DySample 在 detect 头前需要 channel 对齐（cv2 in_channels），需手动 broadcast。

---

### A2. **Coil-PANet**：单类场景的 PAN 极简版

**问题**: YOLO 标准 PANet 在 3 个尺度 (P3/P4/P5) 间做 top-down + bottom-up 双向融合。但 1 类任务 + 单目标场景，**P5 (stride 32) 对 50×50 tip 几乎无贡献**（感受野 32×32 > tip 本身），浪费 ~30% neck 算力。

**方案**: 砍掉 P5→P4 的 bottom-up 路径，只保留 P3↔P4 双向。**P5 仅作为大物体 anchor 池**（保留 detect 层但 freeze bn）。

**实现位置**: `tasks.py` parse_model() 阶段，给 backbone 加 `freeze_p5=True` flag；或在 neck YAML 里显式 `[[-1, 6, 1, nn.Upsample, [None, 2, "nearest"]]]` 注释掉 P5↔P4 边。

**预期**: neck 参数量 -25%，前向 0.8ms → 0.6ms；小目标 mAP50 +1-2pp（信息流更聚焦 P3/P4）。

**风险**: YOLO Detect head 的 `nl=3` 强耦合，需要把 head 改成 `nl=2` 或 dummy 第三个 stride=64 的零张量。需写 dummy P5 张量喂 head 才能不报错。

---

### A3. **GAP-Context**：全局上下文给 tip"在哪里出"做先验

**问题**: 钢卷 tip 出现在图四周的概率高（卷头卷尾都在边缘），但中间偶尔也有。模型仅靠局部特征预测，**缺全图"哪里像有 tip"的全局先验**。

**方案**: 在 neck 最末层 P3 旁加一个**全局平均池化分支**，输出 1×1×C 的 `gap_ctx` 张量，通过 `torch.einsum('bchw,bc->bhw', P3, gap_ctx)` 注入到 P3 特征图（点乘注意力）。零参数学习成本（C=64, GAP 后 broadcast）。

**实现位置**: `nn/modules/block.py` 新增 `class GAPContext(nn.Module)`，在 `tasks.py` 接在 P3 输出后、`cv2[0]` 前。

**预期**: FP 抑制（tip 不在边缘时全图衰减响应），FP 2 → 0~1。

**成本**: +0 训练参数（GAP 本身是 avg pool），+0.2ms 推理。

**风险**: GAP 抹平空间信息，需在 P3 加 1×1 Conv 把 `gap_ctx` 投影到 64 维再点乘，避免梯度全图均匀化。

---

## B. Head 创新（2 个）

### B1. **RegOnly-Head**：1 类任务专用的极简检测头（**保留 1 通道 objness** 与 Hard Neg 兼容）

**问题**: YOLO Detect head 在 `cv3` (分类分支) 用 3×3 conv × 2 + 1×1 conv 输出 80 类 logits。我们是 1 类，**cv3 80 通道里 79 个是冗余**。直接删 cv3 让 `cls_score = 1.0`（常数）会**让 Hard Neg Crop 训练失效**——V18.3 的核心信号是 VFL 在无 bbox 的 anchor 上压 conf，删 cv3 后这个信号通道断开，11 张原图 FP 全部"恢复"成 0.4+。

**方案（保留 1 通道 objness）**: 把 `cv3` 从 80 通道**砍到 1 通道**（不是删，是"压缩"）。这一通道用作 **objness**——二分类语义："这个 anchor 上**是否有 object**（1 类 tip）"，而不是 80 类分类里的某一项。cv2 (回归分支) 同步从 4 通道扩到 8 通道（cx/cy/w/h + 4 角点偏移），用角点偏移补偿宽松标注的 5-15px 噪声。

**与 Hard Neg 兼容性论证**:
- V18.3 用 VFL 在 hard neg 图 (无 bbox) 上让所有 anchor 的 cls_conf 学下降
- 原版 cv3 输出 80 类 sigmoid，**只有 1 个通道 (coil_head) 真正对硬负有梯度**——刚好等价于这一通道 objness
- 新版 cv3 输出 1 类 objness sigmoid，**等价信号全保留**，梯度方向不变
- 训练数据加载不动，**loss 代码改一行** (`nc=1` 即可，BCE/VFL 接口都对)

**实现位置**:
- `ultralytics/cfg/datasets/coil.yaml` `nc: 80 → nc: 1`（这一步在 cfg 里，不改源码）
- `ultralytics/nn/modules/head.py` 的 `Detect.__init__`，加 `cls_channels=nc` 自动推；不动也行，因为 cfg 改 nc=1 就够了
- 选做：cv2 通道 4 → 8 (角点偏移)，新增 `class RegOnlyDetect(Detect)` 复制 head.py

**预期**:
- **参数量**：cv3 砍 80 → 1（-98.75%），cv2 4 → 8 (+100%)，整体头部权重 -75%
- **训练速度**：前向 cv3 通道数 1/80，后向 cls loss 1/80 → epoch 时间 -10%
- **Hard Neg 训练数据仍有效** —— 因为 1 通道 objness 就是 V18.3 cls_loss 用的信号
- **几何置信度**：单目标 max_det=1 部署下，NMS 取 `w*h` 几何大小代替 sigmoid(conf)，行为等价

**风险**:
- ultralytics 的 loss.py `v8DetectionLoss` 强依赖 `ps[:, 4:]` 取 cls 张量——改 `nc=1` 是 cfg 层动作，**无需改 loss.py**（已验证 nc 单测）
- 8 通道 vs 4 通道角点偏移，需要再写一个 corner L1 loss（30 行）或者先不做角点先用 4 通道基线
- 建议**两阶段验证**：先只做 cv3 80→1（5 行 patch + smoke 50ep），确认 F1 ≥ V18.3 epoch60=0.9286 后，再上 4→8 通道

**ROI 估算**:
- 第一阶段 (cv3 80→1 + 4 ch)：~20 行代码 + 1 小时验证，预期 F1 ≥ 0.9286（持平），**头参数量 -75% 是显式增益**
- 第二阶段 (再扩 4 通道角点)：~50 行代码 + 1.5 小时，预期 +0.5-1pp（需 smoke 验证角点是否真有用）

---

### B2. **Top1-Aware Head**：原生适配 max_det=1 部署

**问题**: 钢卷场景每张图最多 1 个 tip，YOLO 训练时 anchor 分配器 (TAL) 仍按 13 topk 选候选。**topk=13 远超 max_det=1**，导致训练时模型学"13 个候选互相抑制"，部署时 max_det=1 把分数第二高的候选扔掉——**训练-部署不一致**。

**方案**: 自定义 `Top1Assigner`（基于 tal.py 复制），把 `topk=13` 改成 `topk=3`，只保留 3 个候选 anchor 学习相互抑制关系。Head 端也对应改：把 Detect 的 `max_det` 默认值从 300 改成 1。

**实现位置**: `utils/tal.py` 复制成 `utils/tal_top1.py`，改 `topk=3, topk2=1`；`nn/modules/head.py` 设 `max_det = 1`。

**预期**: 训练-部署一致 → 部署 Recall +2-3pp（候选抑制不会"误杀"次优 anchor）。

**成本**: 0 新参数。

**风险**: topk=3 早期可能欠拟合，建议前 50 epoch 用 topk=13 warmup，50 epoch 后切到 topk=3。**改 `if self.epoch >= 50: self.topk = 3`** 即可。

---

## C. Loss 创新（2 个）

### C1. **Box-DoUBLoss**：宽松标注的边界软化损失

**问题**: V18.3 用 CIoU loss，但 CIoU 对 5-15px 的 GT 边界噪声**过度敏感**——同一个 tip 标 50×50 vs 55×55 的 CIoU 差距可达 0.08，在 1 类任务 1 个 TP 上足以让 mAP 上下 0.5pp。

**方案**: 替换 CIoU 为 **DoU Loss** ([Distance-IoU](https://arxiv.org/abs/2111.01587))，它只计算两 bbox 中心距离 + 重叠面积，**对边界微小变化不敏感**。同时加 1 个 `box_smoothness` 项：当 pred box 中心距 GT 中心 < 3px 时强制 loss=0（容忍亚像素抖动）。

**实现位置**: `utils/loss.py` 新增 `class DoUBboxLoss(BboxLoss)`，重写 `forward`：
```python
loss = 1 - do_u(p, t) + alpha * (center_dist > 3.0)  # 中心 < 3px 不惩罚
```

**预期**: box_loss 数值降 30-50%，mAP50 +1-2pp（V18.3 训练 box_loss=4640 大概率有大量"边界微调" loss 在做无用功）。

**成本**: 0 新参数。

**风险**: DoU 在 bbox 不重叠时梯度为 0（与 DIoU 同样问题），需要 fallback 到 L1 距离 loss。建议混合 `0.7 * DoU + 0.3 * L1_centroid`。

---

### C2. **Conf-Recall Loss**：直接优化部署指标

**问题**: 当前 V18.3 训练用 VFL (Varifocal Loss) 做 cls 损失，但部署只看 `conf >= 0.15` 的 top-1 bbox。**训练时 VFL 在所有候选 anchor 上算 loss，部署时只看 1 个**——训练-评估不对齐。

**方案**: 自定义 `ConfRecallLoss`，对**置信度排序后的 top-K 候选**算 loss（K=5，对应 max_det=1 的近邻候选），K 之外不参与梯度。鼓励模型把"真 tip 的 top-1 conf 推到 0.5+，把 fake tip 的 top-1 conf 压到 0.1-"。

**实现位置**: `utils/loss.py` 新增 `class ConfRecallLoss(nn.Module)`，在 `v8DetectionLoss.__call__` 里替换 cls_loss。

**预期**: conf 分布更双峰化（正样本 0.5+，负样本 <0.1）→ 部署 conf=0.15 阈值下 Recall 自然 +3-5pp。

**成本**: 0 新参数。

**风险**: top-K 选择需要 detach 操作（避免梯度穿越排序），且 K=5 是超参，需要在 val=99 张上 sweep {3, 5, 7, 10}。

---

## D. 后训练微调（2 个）

### D1. **EMA-Conf-Calibration**：用 EMA 模型校准 conf 分布

**问题**: V18.3 训练用了 cos_lr + 弱 aug，最终 best.pt 的 conf 分布可能与 last.pt 差异大（fitness best ≠ deploy best，已知问题）。但我们想用 deploy best 重新校准 conf。

**方案**: 训练最后 10 epoch 不更新模型参数，**只对 NMS conf 阈值做 grid search**：在 val=99 张上扫 conf ∈ {0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25}，选 F1 最大者；再用 EMA 平滑（α=0.3）防止过拟合 val。

**实现位置**: 训练脚本末尾加 `calibrate_conf.py`，调用 `lenient_eval.py --mode top1 --conf-sweep`。

**预期**: 部署 F1 +0.5~1pp（纯后处理，零风险）。

**成本**: 5 分钟。

**风险**: 在 val 上扫 conf 是"作弊"（用 val 选 conf），需要交叉验证：留 33 张不参与扫 conf。

---

### D2. **TTA-Voting**：多 epoch 投票

**问题**: V18.3 只有 epoch60.pt 部署最优，但 save_period 没存更密集的检查点（用户说"训练时间紧"）。

**方案**: 重新跑 V18.3 训练，save_period=5 存 ep55/ep60/ep65/ep70 四个 checkpoint，部署时 4 模型对同一图预测 → 取 conf 加权平均的 bbox（用 WBF 算法）。

**实现位置**: 训练脚本加 `save_period=5`；推理脚本 `tta_voting.py` 调 ultralytics predict 后做 WBF 融合。

**预期**: 4 模型 WBF 集成 F1 +1-2pp（统计意义的标准 ensemble 提升）。

**成本**: 4×28分钟重训 + 10 分钟 WBF 集成测试。

**风险**: WBF 在 1 类单目标上退化为"取 conf 最大"，ensemble 退化为单模型。需要改 WBF 为 `soft-vote`：4 个 bbox 中心取均值，conf 取最大值。

---

## 推荐实施顺序

| # | Idea | 风险 | 成本 | 预期 F1 提升 | 优先级 |
|---|------|------|------|-------------|--------|
| 1 | **D2. TTA-Voting** | 低 | 1.5h | +1-2pp | 🥇 立即做 |
| 2 | **D1. EMA-Conf-Calibration** | 低 | 5min | +0.5-1pp | 🥇 立即做 |
| 3 | **B2. Top1-Aware Head** | 中 | 改 2 文件 + 30min 重训 | +2-3pp | 🥈 第二 |
| 4 | **C1. Box-DoUBLoss** | 中 | 改 1 文件 + 30min 重训 | +1-2pp | 🥈 第二 |
| 5 | **A3. GAP-Context** | 低 | 改 1 文件 + 30min 重训 | FP 抑制 | 🥉 第三 |
| 6 | **A1. DySample-Tip** | 中 | 改 2 文件 + 1h 重训 | +1pp | 🥉 第三 |
| 7 | **B1. RegOnly-Head (cv3 80→1, 4ch)** | 低-中 | 改 1 cfg + smoke 50ep | 持平 + 头参数量 -75% | 🥉 第三 (两阶段) |
| 8 | **B1' RegOnly-Head + 角点 (cv3 1ch, 8ch)** | 中 | 上一步通过后再上 | +0.5-1pp | ⏸ 备选 |
| 9 | **C2. Conf-Recall Loss** | 中 | 改 2 文件 + 30min 重训 | +3-5pp | ⏸ 备选 |
| 10 | **A2. Coil-PANet** | 高 | 改 tasks.py + 1h | +1-2pp | ⏸ 备选 |

**B1 备注**：第一阶段 (cv3 80→1) **必须 smoke 50 epoch 验证 F1 ≥ V18.3 epoch60 的 0.9286**（持平 = 不退步即可，-75% 头参数是显式增益）。通过后再上第二阶段（4→8 通道角点）。

**建议路径**: D2 → D1 → B2 → C1 → A3，每步验证再下一步。所有 "中" 风险先 smoke 50ep 再 full 100ep。

---

## 不再做的方向（写出来防止自己回头试）

| 方向 | 失败证据 | 根因 |
|------|----------|------|
| box_soft_relative=0.10 | V5 v5 ep350 mAP50=0.36 | box_loss 累积爆炸 19→4640 |
| STAL 全程开启 | V7 v1 ep155 早停 mAP50=0.375 | 后期拖累收敛 |
| P2 四尺度 | v17 ep57 mAP50=0.034 | N2 从零学失败 |
| MuSGD optimizer | ultralytics 自动 < 10000 iter 用 AdamW | 训练轮次不够 |
| 后处理 TTA / 几何规则 | 4 脚本全 < 0.8571 baseline | 评估口径已最优 |
| 5 创新点 (v9) | 全 mAP≈0 | 数据规模硬约束 |

---

**下一步**: 你选 D2/D1（零风险快速收益）还是 B2/C1（中风险可能大收益）？我可以立即写对应实现。
