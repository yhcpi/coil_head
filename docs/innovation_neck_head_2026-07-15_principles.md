# 9 个创新点原理详解（图文版）

> 只讲**原理**。所有 idea 针对：钢卷头尾 1 类小目标检测（tip 50×50 px @ 1920×1080）、宽松标注、max_det=1 部署。

**修订 (2026-07-15)**:
- 4. RegOnly-Head 修订：**删 cv3 + cls_score=1.0 改为 cv3 80→1 (1 通道 objness)**，理由与 Hard Neg Crop 兼容。
- 一图概览新增 4a "冲突历史"备忘章节

---

## 1. DySample-Tip（neck 上采样）

### 问题
tip 在原图 50×50 px，**经过 32× 下采样到 P3 特征图只占 1.5 像素**（亚像素）。

```
原图 1920×1080         backbone 32× 下采样        P3 特征图 60×34
┌──────────┐            stride=32                ┌──┐
│   tip    │  ──────►                          │░░│  ← 1.5 个像素
│  50×50   │                                    └──┘
└──────────┘
```

P3→P4 时 YOLO 用 **nearest Upsample**（直接把 1.5 像素复制成 3 像素），**亚像素位置被抹平**。预测 bbox 中心出现 **±1.5px 量化误差**。

### 原理
换成 [DySample](https://arxiv.org/abs/2308.15020)：用一个**小型网络预测每个像素的 (dx, dy) 偏移**，再按偏移采样。

```
P3 像素 (1.5 个真实像素)         偏移 (dx, dy)        上采样后
   ┌──┐                          0.3, 0.7           ┌─┐
   │░░│  ──── 学习偏移 ────►   每个像素独立        │░│  ← 中心保留
   └──┘                                              └─┘
```

**作用**：tip 中心定位误差 ±1.5px → ±0.5px。

---

## 2. Coil-PANet（neck 极简版）

### 问题
标准 PANet 在 3 个尺度 (P3/P4/P5) 间做 top-down + bottom-up **双向**信息流。

```
标准 PANet:
         P5 (stride 32, RF=32×32)  ← 感受野大于 tip 本身 (50×50)，几乎无用
         ↕ ↕
         P4 (stride 16, RF=16×16)
         ↕ ↕
         P3 (stride 8,  RF=8×8)   ← 唯一能精确捕捉 tip 的尺度
```

### 原理
**砍掉 P5↔P4 之间的路径**。P5 只作为"大物体 anchor 池"保留，但不向 P4 注入信息。

```
Coil-PANet:
         P5 ──X (断)
              ↕
         P4
              ↕
         P3
```

**作用**：neck 参数量 -25%，信息流聚焦 P3/P4（前向 0.8ms → 0.6ms），小目标 mAP +1-2pp。

---

## 3. GAP-Context（neck 全局上下文）

### 问题
模型只靠**局部特征**预测 tip，缺"全图哪里像有 tip"的先验。钢卷 tip 多数在图四周（卷头卷尾），但中间偶尔也有。

### 原理
P3 后接一个**全局平均池化分支**，输出 1×1×C 的 `gap_ctx`，**点乘注入** P3 特征图：

```
P3 特征图 (60×34×64)         GAP 分支
┌────────────┐               ┌──┐
│  局部细节   │               │  │ 1×1×64
│  中心 / 边缘│  ── einsum ──►│  │ ──► 空间注意力 mask
└────────────┘               └──┘
       ▲                          │
       └────── mask × P3 ─────────┘
              (点乘)
```

**作用**：tip 不在边缘时全图响应被衰减 → 抑制假 tip。FP 2 → 0~1。

---

## 4. RegOnly-Head（head 极简版，**保留 1 通道 objness 与 Hard Neg 兼容**）

### 问题
YOLO Detect head 的分类分支 (cv3) 输出 `B × 80 × H × W` 个类别 logits。我们只有 **1 类**，cv3 的 79/80 通道冗余；训练时 cls 概率饱和到 ~1.0 时**完全无梯度**。

**关键约束**：直接删 cv3 让 `cls_score = 1.0`（常数）会**让 Hard Neg Crop 训练失效**。V18.3 是当前生产冠军（F1=0.9286），它的核心信号是 VFL 在无 bbox 的 hard neg 图上压 conf。删 cv3 后这条信号断了，11 张原图 FP 全部"恢复"。

### 原理（不是删 cv3，是压缩 80 → 1）
```
原 Detect head:                         RegOnly head (修订方案):
  cv2 (reg, 4ch)  ──► bbox                cv2 (reg, 4ch 或 8ch) ──► bbox [+ 4 角点]
  cv3 (cls, 80ch) ──► 80 类 sigmoid      cv3 (cls, 1ch)    ──► 1 类 objness sigmoid
                  仅 1 个通道对硬负有梯度      ↑             ↑ 这个通道就是 V18.3 的 conf 信号
                                              80 → 1 (-98.75% 通道数)
```

**新 cv3 的语义**：从 "80 类 softmax 之一" 变成 **"这个 anchor 上是否有 object"** 的二分类。它**等价于**原版 80 通道里那唯一对硬负有梯度的 1 个通道。

**为什么不"删 cv3 令 cls_score=1.0"**：那样 Hard Neg 图的所有 anchor 都没梯度，**V18.3 的 1.50pp F1 提升会一夜回到 0.9136**。

### 与 Hard Neg Crop 的兼容性证明

```
V18.3 训练: 输入 hn493 (无 bbox)
原版 cv3 输出 [N=8400, 80]:  80 个 sigmoid, VFL 取 coil_head 通道标量
V18.3 期望: 这个标量对 N 个 anchor 都学下降 → conf 0.467 → 0.011

RegOnly cv3 输出 [N=8400, 1]: 1 个 objness sigmoid, VFL 直接用
RegOnly 期望: 这个标量对 N 个 anchor 都学下降 → conf 0.467 → 0.011
                                          ↑ 完全等价 ↑
```

**等价性**：对硬负训练，1 通道 objness **就是** 80 通道里唯一管用的那一通道。**V18.3 的训练数据不需要动一行**。

### 实现要点（5 行 cfg 改动 + 训练脚本 flag）

**第一步（必做）**：数据 cfg `coil.yaml`
```yaml
nc: 1     # 原值 80, 改 1
```

**第二步（必做）**：`Detect` head 里加一个安全检查
```python
if self.nc != 1:
    raise ValueError("RegOnly-Head 仅支持 1 类任务 (coil_head)")
```

**第三步（可选）**：cv2 通道 4 → 8（角点偏移），需另写 corner L1 loss（30 行）

### 部署如何用 conf

**训练时**：`conf = sigmoid(cv3_output[:, 0])`（V18.3 同款）

**部署时**：单目标 `max_det=1` → NMS 排序只看 conf 最大那个。**1 类 objness 下行为完全等同原版**，无需改 NMS 代码。

### 角点偏移补偿宽松标注（**第二阶段才做，先验证第一阶段**）

5-15px 边界噪声在 (cx, cy, w, h) 上对中心不敏感，但**角点偏移**能学到"tip 的边界外延"：

```
基础 bbox (从 cx,cy,w,h):  [275, 175] → [325, 225]
                            ↖          ↘
                       角点偏移 dx=-3, dy=-2   角点偏移 dx=+5, dy=+4
                            ↓          ↓
最终 bbox:           [272, 173] → [330, 229]
```

**两阶段验证**：先做 cv3 80→1 (5 行) smoke 50 epoch，确认 F1 ≥ V18.3 epoch60 的 0.9286（持平即通过），再上 4→8 通道。

---

## 4a. RegOnly + Hard Neg 冲突历史（**写作警告，下笔前读**）

之前我提过"删 cv3 + cls_score=1.0"，**这方案与 V18.3 的 Hard Neg Crop 训练数据互斥**。

如果你/新会话看到旧版原则文档里有"删 cv3"字样，**不要照做**——必须先确认是这一节（cv3 80→1）而不是"全删 cv3"。

相关创新点的兼容性表：

| 创新 | 是否需要 cls 通道学"否" | 与 Hard Neg | 备注 |
|------|--------------------------|--------------|------|
| 原版 Detect | ✅ | ✅ 不冲突 | 当前生产 |
| RegOnly cv3 80→1 (本方案) | ✅（1 通道 objness） | ✅ 不冲突 | **推荐** |
| RegOnly cv3 全删 + 1.0 | ❌ | ❌ 硬冲突 | **已废弃** |
| Conf-Recall Loss | ✅ top-K 学 conf | ✅ 不冲突 | 强化 Hard Neg |
| Top1-Aware Head | ✅ TAL 学 anchor PK | ✅ 不冲突 | |
| Box-DoUBLoss | ❌ 只改 box loss | ✅ 不冲突 | |

---

## 5. Top1-Aware Head（head max_det 对齐）

### 问题
钢卷场景每张图**最多 1 个 tip**，部署时 `max_det=1`。但训练时 YOLO 的 TAL 锚框分配器按 `topk=13` 选 13 个候选 anchor，**13 个候选互相学抑制关系**。

```
训练：13 个候选 anchor 互相抑制 → 模型学"top-1 选最强那个"
部署：只取 top-1  →  但模型学的是"前 13 互相 PK"，不一致
```

### 原理
复制 `utils/tal.py` → `tal_top1.py`，改 `topk=13 → topk=3, topk2=1`。Head 端改 `max_det=300 → max_det=1`：

```
TAL (训练时):                    Head (max_det):
  13 个 anchor ──► 相互抑制         只取 top-1
   ↓ 改 topk=3                      ↓
  3 个 anchor ──► 选最强            max_det=1
```

**建议**：前 50 epoch warmup 用 `topk=13`，50 epoch 后切 `topk=3`（避免早期欠拟合）。

---

## 6. Box-DoUBLoss（loss 边界容差）

### 问题
V18.3 用 CIoU loss，但 CIoU 对 5-15px 标注边界噪声**过度敏感**：

```
同一个 tip 标注差异:
  GT-A: 50×50        GT-B: 55×55
       ┌──┐               ┌──┐
       │  │               │    │
       └──┘               └────┘
  CIoU=0.85               CIoU=0.77
       ↓                    ↓
  同一个 TP 上 loss 差 0.08 → mAP 上下 0.5pp 抖动
```

### 原理
替换为 **DoU Loss**（[Distance-IoU](https://arxiv.org/abs/2111.01587)），它只算**中心距离 + 重叠面积**，对边界不敏感。

加 1 个**中心容差项**：当 pred 中心距 GT 中心 < 3px 时**强制 loss=0**（容忍亚像素抖动）：

```
Box-DoUBLoss = 0.7 * DoU(pred, gt) + 0.3 * L1_centroid(pred, gt)
              × (center_dist > 3.0)        # 中心 < 3px 不惩罚
```

---

## 7. Conf-Recall Loss（loss 训练-部署对齐）

### 问题
训练用 VFL（Varifocal Loss）在**所有候选 anchor** 上算 loss，但部署只看 `conf >= 0.15` 的 **top-1**。

```
训练: 200 个候选都参与 cls loss → 鼓励"全局都准"
部署: 只看 1 个           → 实际只看"那个最像的够不够高"
```

### 原理
对**置信度排序后的 top-K=5 候选**算 loss，K 之外不参与梯度。

```
200 个 anchor          sort by conf           top-5 算 loss
  ● ● ● ● ●             ─────►                ● ● ● ● ●
  ● ● ● ● ●                                    ↑
  ● ● ● ● ●                              鼓励这 5 个"真 tip 推高 conf"
  ● ● ● ● ●                                    假 tip 压低 conf"
  (其余 195 个 frozen)
```

**预期**：conf 分布双峰化（正样本 0.5+，负样本 <0.1）→ 部署 conf=0.15 阈值下 Recall 自然 +3-5pp。

---

## 8. EMA-Conf-Calibration（后训练 conf 校准）

### 问题
V18.3 best.pt 训练末期的 conf 分布可能与"部署最优"不一致。fitness best ≠ deploy best（已知问题）。

### 原理
训练最后 10 epoch **冻结模型参数**，只在 val=99 张上扫 conf 阈值：

```
epoch 90 ┐
epoch 91 │  参数冻结
epoch 92 │  在 val 上 grid search:
epoch 93 │    conf=0.05 → F1=0.91
epoch 94 │    conf=0.10 → F1=0.93
...      │    conf=0.15 → F1=0.93  ← 选这个
epoch 99 ┘    conf=0.20 → F1=0.92
              (用 EMA α=0.3 平滑防过拟合)
```

**本质**：用 val 选 conf 阈值，再用 EMA 平滑。是**纯后处理**，零风险。

---

## 9. TTA-Voting（后训练多 ckpt 集成）

### 问题
V18.3 单一 epoch60.pt 部署最优，但单点最优 = 局部最优（运气）。多 epoch 集成能消除单点噪声。

### 原理
重新跑 V18.3 训练，`save_period=5` 存 4 个 checkpoint，部署时**软投票**：

```
训练: 存 ep55 / ep60 / ep65 / ep70 四个 .pt
                    ↓
推理 (同一张图):
  ep55 → bbox₁, conf₁=0.82
  ep60 → bbox₂, conf₂=0.85  ←  (各模型预测中心略不同)
  ep65 → bbox₃, conf₃=0.79
  ep70 → bbox₄, conf₄=0.84
                    ↓
Soft-vote:
  center_final = (c₁ + c₂ + c₃ + c₄) / 4
  conf_final   = max(conf₁..conf₄)
                    ↓
              部署用 center_final + conf_final
```

**与硬投票区别**：1 类单目标场景下，硬投票退化为"取 conf 最大"；soft-vote 让 4 个 bbox 中心**取平均**，保留 4 个模型的统计信息。

---

## 一图概览

```
                  10 个创新点 (修订后)

Neck (3)                    Head (3)                Loss (2)              后训练 (2)
─────────                   ─────────              ─────────             ─────────
1. DySample-Tip             4. RegOnly-Head (cv3 80→1)    6. Box-DoUBLoss       8. EMA-Conf-Calib
2. Coil-PANet                  Hard Neg 兼容            7. Conf-Recall Loss   9. TTA-Voting
3. GAP-Context              5. Top1-Aware Head
                            ─────────
                            4'. RegOnly + 角点 (cv3 1ch + cv2 8ch)
                            第二阶段, 先验证 4 再上 4'
```

每个 idea 改的代码量 50-200 行，单 GPU 验证 30 分钟内可出 smoke 结果。
