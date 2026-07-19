# YOLO26 三大创新点深度解析 — 预测头 / 标签分配 / MuSGD

**日期**: 2026-07-14
**目标**: 结合 `repos/ultralytics/ultralytics/` 8.4.95 源码,逐行解释 YOLO26 的 3 个核心创新点底层实现。
**适用读者**: 已读过 YOLO26 paper,需要把"概念描述"翻译成"代码能跑通的形式"。

---

## 0. YOLO26 创新的全景视图

| 创新点 | 论文定位 | ultralytics 8.4.95 代码入口 | 本质 |
|---|---|---|---|
| **End-to-end NMS-free 检测头** | Sec 2.3.1 + Sec 2.3.2 | `nn/modules/head.py: Detect` + `utils/loss.py: E2ELoss` | 双头 + 双 loss + top-k 后处理 |
| **ProgLoss 渐进式损失平衡** | Sec 2.3.3 | `utils/loss.py: E2ELoss.decay/update` | o2m 权重从 0.8 → 0.1 余弦退火 |
| **DFL Removal (reg_max=1)** | Sec 2.3.1 | `nn/modules/head.py: self.dfl = Identity` | box 从"分布预测"变成"直接回归" |
| **STAL 小目标感知标签分配** | Sec 2.3.4 + Fig 4c | `utils/tal.py: select_candidates_in_gts` (line 289-318) | `wh_mask` 把小目标 GT bbox 强制放大到 ≥ stride |
| **MuSGD 混合优化器** | Sec 2.3.5 | `optim/muon.py: MuSGD/Muon/zeropower_via_newtonschulz5` | SGD + Muon (Newton-Schulz 正交化) |

**5 个创新里,1+2+3 共享同一个"双头"架构**,所以并入第 1 章一起讲。第 2 章讲 STAL。第 3 章讲 MuSGD。

---

## 1. YOLO26 预测头 — 怎么"直接输出非冗余边界框"

### 1.1 传统 YOLO 头 vs YOLO26 双头

**传统 YOLOv8/v11 头** (`end2end=False`):

```
Backbone/Neck → 1 个共享 Detect head
  → 每 anchor 输出 (4*reg_max + nc) 通道
  → DFL 分布解码 + NMS 抑制重复框
```

**YOLO26 双头** (`end2end=True`, 8.4.95 `head.py:122-124`):

```python
if end2end:
    self.one2one_cv2 = copy.deepcopy(self.cv2)  # 整个 box 分支复制一份
    self.one2one_cv3 = copy.deepcopy(self.cv3)  # 整个 cls 分支复制一份
```

→ **同一张特征图过两份独立的 1×1 Conv**,得到 `one2many` 和 `one2one` 两组预测。参数量翻倍,但这是 NMS-free 的代价。

### 1.2 forward 的两条并行路径

`head.py:157-171` Detect.forward:

```python
def forward(self, x):
    preds = self.forward_head(x, **self.one2many)         # ① one2many 前向
    if self.end2end:
        x_detach = [xi.detach() for xi in x]              # ② 关键: 特征图截断梯度
        one2one = self.forward_head(x_detach, **self.one2one)  # ③ one2one 在 detached 特征上前向
        preds = {"one2many": preds, "one2one": one2one}
    if self.training:
        return preds                                      # 训练: 返回双 pred 给 loss
    y = self._inference(preds["one2one"] if self.end2end else preds)  # 推理: 只用 one2one
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))          # top-k 排序代替 NMS
    return y if self.export else (y, preds)
```

**关键设计**:

1. **训练时双 loss,推理时只用 one2one** — one2many 是"监督教师",只为算梯度;one2one 才是"产品"。
2. **`x_detach = [xi.detach() ...]`** — one2one 分支不参与反向传播的特征更新,只更新自己的 head 权重,避免两个 head 互相干扰。
3. **推理时 `preds["one2one"]`** — 选 one2one 而不是 one2many 是因为 one2one 训练时受 1-to-1 分配约束,天然无冗余。

### 1.3 E2ELoss:双 loss 怎么组合 + ProgLoss 衰减

`loss.py:1174-1206`:

```python
class E2ELoss:
    def __init__(self, model, loss_fn=v8DetectionLoss):
        self.one2many = loss_fn(model, tal_topk=10)         # ④ topk=10: 多对一,容错分配
        self.one2one  = loss_fn(model, tal_topk=7, tal_topk2=1)  # ⑤ topk=7, topk2=1: 一对一
        self.o2m = 0.8    # 初始 one2many loss 占比
        self.o2o = 0.2    # 初始 one2one  loss 占比
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1                                # 终态 one2many 占比(余弦退火到这)

    def __call__(self, preds, batch):
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_a = self.one2many.loss(one2many, batch)
        loss_o = self.one2one .loss(one2one,  batch)
        return loss_a[0] * self.o2m + loss_o[0] * self.o2o, loss_o[1]   # ⑥ 加权求和

    def update(self):
        self.updates += 1
        self.o2m = self.decay(self.updates)                  # ⑦ 每个 epoch 调一次
        self.o2o = max(1.0 - self.o2m, 0)

    def decay(self, x):
        # 线性插值: epoch=1 → o2m=0.8, epoch=epochs-1 → o2m=0.1
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) \
               * (self.o2m_copy - self.final_o2m) + self.final_o2m
```

**逐行解释**:

| 行号 | 含义 |
|---|---|
| ④ | one2many 给每个 GT 分配 **10 个** anchor 做正样本,监督信号密集,训练前期快速收敛 |
| ⑤ | one2one 给每个 GT 分配 **7 个候选 anchor**,再用 `tal_topk2=1` 二次筛选到 **1 个**,做"一对一"分配 |
| ⑥ | 总 loss = 0.8 × one2many + 0.2 × one2one(初始)|
| ⑦ | `update()` 在每个 epoch 结束由 `engine/trainer.py:539` 调用 |
| decay | 250 epoch 训练里 o2m 从 0.8 → 0.1,前 80% 训练以密集分配为主(快速学),后 20% 强约束 one2one(学会非冗余)|

**为什么 ProgLoss 能 work**:
- 训练前期:one2many 主导(0.8),相当于"先学识别"—— 让所有 anchor 都能预测 GT,提供密集监督信号
- 训练后期:one2one 主导(0.9),相当于"再学唯一"—— 每个 GT 必须恰好对应一个 anchor 出来最准的框,自然去重
- 整个过程不需要 hard switch,避免 loss 跳变

### 1.4 tal_topk=7, tal_topk2=1 是怎么"保证非冗余"的

`utils/tal.py:320-356` select_highest_overlaps:

```python
def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
    fg_mask = mask_pos.sum(-2)              # 每个 anchor 候选给几个 GT
    if fg_mask.max() > 1:                   # 同一个 anchor 被多个 GT 抢
        # 用 overlap 最大的 GT 赢,其它丢掉
        max_overlaps_idx = overlaps.argmax(1)
        is_max = torch.zeros_like(mask_pos)
        is_max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_max, mask_pos).float()

    if self.topk2 != self.topk:              # 这里! topk2=1 才进
        align_metric = align_metric * mask_pos
        # 二次 topk: 每个 GT 只保留 align_metric 最高的 topk2=1 个 anchor
        max_idx = torch.topk(align_metric, self.topk2, dim=-1).indices
        topk_idx = torch.zeros_like(mask_pos)
        topk_idx.scatter_(-1, max_idx, 1.0)
        mask_pos *= topk_idx
        fg_mask = mask_pos.sum(-2)

    target_gt_idx = mask_pos.argmax(-2)      # 每个 anchor 最终归属哪个 GT
    return target_gt_idx, fg_mask, mask_pos
```

**逻辑梳理**:

1. topk=10 时,每个 GT 在 anchor 里选 10 个候选(`mask_topk`),这些是"可能正样本"。
2. 再用 mask_in_gts × mask_gt 过滤到真正"在 GT 框内"的 anchor。
3. `topk2=1` 在这些候选里**只挑 1 个 align_metric 最高的**作为最终正样本 —— 这就是"一对一"的硬约束来源。
4. 一个 anchor 若被多个 GT 都挑中 → 用 overlap 最大的赢(冲突解决)。

**对比 one2many(topk=10, topk2=10 默认)**:
- topk2=10 → 上面 `if self.topk2 != self.topk` 不进,所有候选都保留。
- 同一 anchor 可以同时给多个 GT 提供回归信号(密集监督)。

**结论**: **one2one 的"非冗余"是 loss 函数层面强约束的**—— 每个 GT 最终只对应 1 个 anchor 出框,自然没有重复,所以推理时无需 NMS。

### 1.5 DFL Removal (reg_max=1) 为什么重要

`head.py:120`:

```python
self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
```

- `reg_max=16`(YOLOv8): 4×16=64 通道 → softmax → 期望值 → 4 个 ltrb。这是分布预测。
- `reg_max=1`(YOLO26): 4×1=4 通道 → 直接就是 ltrb。`dfl=Identity` 不过任何计算。

**为什么删 DFL**:
1. **ONNX 导出简化**: 没有 softmax + matmul,导出来只有 Conv。
2. **数值稳定**: softmax 在极端情况下会饱和,Identity 不会。
3. **和 one2one 配套**: one2one 既然要每个 GT 一个 anchor,精细的分布预测意义不大;直接回归更直接。

**副作用**: 失去了 DFL 的"边界不确定性建模"能力。但 YOLO26 paper 报告 COCO mAP 几乎不掉,说明 32× 像素以下的 box 在分布预测里本来就是无效的。

### 1.6 推理时的 NMS-free 怎么实现

`head.py:219-258` postprocess:

```python
def postprocess(self, preds):
    boxes, scores = preds.split([4, self.nc], dim=-1)
    scores, conf, idx = self.get_topk_index(scores, self.max_det)  # ① top-k 排序
    boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
    return torch.cat([boxes, scores, conf], dim=-1)

def get_topk_index(self, scores, max_det):
    k = max_det if self.export else min(max_det, anchors)   # ② 默认 max_det=300
    ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
    scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(k)
    idx = ori_index[batch_idx, index // nc]
    return scores[..., None], (index % nc)[..., None].float(), idx
```

**核心**: 用 **`topk(max_det=300)` 全局排序 + 类别竞争**,替代 NMS 的"按类/全局去重":
- 所有 anchor 按 max-class-score 排序 → 保留前 300 个。
- NMS 移除的是"高度重叠 + 同类高置信",而 one2one 已经保证每个 GT 一个 anchor,预测本身就唯一。
- **速度**: top-k 是 GPU 原生 sort,NMS 是迭代 IoU 计算 + mask,**论文报告 CPU 加速 43%** 就是这里来的。

---

## 2. STAL — 小目标感知标签分配

### 2.1 YOLO26 标签分配的两步流水线

`tal.py:14-356` TaskAlignedAssigner.forward:

```
每个 GT box → 找正样本 anchor
   步骤 1: select_candidates_in_gts    # 在 GT 框内的 anchor 是候选
   步骤 2: get_box_metrics              # 算 align_metric = s^α × IoU^β
   步骤 3: select_topk_candidates       # 每个 GT 选 align_metric 最高的 topk 个
   步骤 4: select_highest_overlaps      # 冲突解决 + topk2 二次筛选
```

### 2.2 STAL 的核心:select_candidates_in_gts

`tal.py:289-318`:

```python
def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
    gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
    wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]   # ← STAL 关键行
    gt_bboxes_xywh[..., 2:] = torch.where(
        (wh_mask * mask_gt).bool(),
        torch.tensor(self.stride_val, dtype=...),
        gt_bboxes_xywh[..., 2:],
    )
    gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    bbox_deltas = torch.cat((
        xy_centers[None] - lt,                          # 中心点距离 GT 左/上边
        rb - xy_centers[None],                          # 距离 GT 右/下边
    ), dim=2).view(bs, n_boxes, n_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)                 # 4 个距离都 > 0 → 在框内
```

**STAL 在哪**:
```python
wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
```

**这是论文 Fig 4c 的实现**:
- `self.stride` 默认 `[8, 16, 32]`(对应 P3/P4/P5 三层 feature map 步长)
- `self.stride[0] = 8` = P3 层 anchor 间距(像素)
- 如果 GT bbox 的 w 或 h **小于 8 像素**,意味着这个目标比一个 anchor cell 还小,标准做法下"框内 anchor 数 = 0",**小目标永远分不到正样本**。
- STAL 的 fix: 把这种 bbox 强制放大到 `(stride_val=16, stride_val=16)` —— 至少覆盖一个 P4 anchor cell。

**对 20×20 像素的钢丝头尾 tip**:
- 标准 YOLO: w=h=20 ≥ stride[0]=8 → **不触发 STAL**,但 20×20 框只能覆盖 ~6 个 P3 anchor(8px grid 中心 2.5 个),候选 anchor 很少。
- 极小目标(例如 4×4): 触发 STAL,bbox 被强制放大到 16×16,覆盖 4 个 P3 anchor → 不再"零正样本"。
- 论文"small object priority"的承诺 = 防止训练时小目标被"饿死"(零梯度)。

### 2.3 align_metric = s^α × IoU^β 的几何含义

`tal.py:172-203` get_box_metrics:

```python
def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    overlaps = torch.zeros([bs, n_max_boxes, na], dtype=...)
    bbox_scores = torch.zeros([bs, n_max_boxes, na], dtype=...)
    ind = torch.zeros([2, bs, n_max_boxes], dtype=torch.long)
    ind[0] = torch.arange(bs).view(-1, 1).expand(-1, n_max_boxes)
    ind[1] = gt_labels.squeeze(-1)
    bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

    pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gt]
    gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
    overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)  # CIoU

    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps
```

**逐行解释**:
- `bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]` — 在 GT 框内的 anchor,提取它们对 GT 类别预测的置信度。
- `overlaps[mask_gt]` — 在 GT 框内的 anchor,算 IoU(用 CIoU,`iou_calculation` 见 `tal.py:205-215`)。
- `align_metric = s^α × IoU^β`,**α=1, β=6**(默认超参) → IoU 的影响远大于 cls 分数。这是为了防止"分类准但定位差"的 anchor 抢到正样本。

**对比传统 IoU 分配**:
- 传统 anchor-based 分配: 只看 IoU(只看"位置像不像"),容易把"cls 差但位置准"的 anchor 当正样本 → cls 头学不到 GT 类别分布。
- TAL: 同时看 cls + IoU → 高质量正样本一定是"位置像 + 类别有把握"。
- α=1, β=6 是平衡点: IoU 太重要会过拟合 anchor 几何分布;cls 太重要会引入大量假正样本。

### 2.4 topk=13, topk2=1 的最终效果

- **topk=13**: 每个 GT 在"框内 + align_metric 排序"的 anchor 池里选前 13 个做正样本。13 这个数刚好是经验值:太少监督信号不足,太多引入噪声 anchor。
- **topk2=1**(one2one 分支) vs **topk2=13**(one2many 分支): 唯一区别是 one2one 用 topk2=1 强行只挑 1 个 anchor 出框。

**对小目标场景的效果**(钢卷 20×20 tip):
1. STAL wh_mask 把 <8px 的目标强制放大 → 防止零正样本
2. align_metric 的 IoU^6 → 小目标的 1px 偏移就让 IoU 从 0.8 → 0.5,对 align_metric 是 ~16x 衰减 → **很严,只有"贴得最近"的 anchor 入选**
3. topk=13 → 即使是 20×20 tip,周边 anchor 也会有 5-13 个候选,梯度信号够
4. 但**对小目标而言 STAL 的代价是 CIoU 计算对小目标太敏感**(1px 偏差 IoU 大幅衰减),所以本项目 v3 实验同时引入 NWD loss 来弥补这个几何不友好

---

## 3. MuSGD — 混合优化器的底层原理

### 3.1 为什么需要 Muon

**传统 SGD 的问题** (YOLO 训练默认 `optimizer=SGD, lr0=0.01, momentum=0.937`):

```
θ_{t+1} = θ_t - lr · (β · m_{t-1} + (1-β) · g_t)
```

- 梯度 g 是任意方向的向量,可能是病态(某些方向步长太大,某些方向步长太小)
- 矩阵参数(权重 W ∈ R^{m×n})的不同奇异值方向更新步长差异巨大 → 训练后期某些方向震荡,某些方向停滞
- 卷积核尤其严重:有 out_channel × in_channel × kH × kW 维,更新量分布极不均匀

**Muon 核心思想** (Kimi K2 / Bernstein & Newhouse 2024):

把动量更新"投影到正交基"上 → 所有方向步长一致。数学上等价于"用动量矩阵的 U·V^T 替代动量矩阵本身"(SVD 取最大奇异值方向)。

### 3.2 zeropower_via_newtonschulz5 — Newton-Schulz 正交化

`optim/muon.py:9-56`:

```python
def zeropower_via_newtonschulz5(G, eps=1e-7):
    """对 G 做 5 次 Newton-Schulz 迭代,得到近似 UV^T (SVD)。"""
    assert len(G.shape) == 2
    X = G.bfloat16()                       # ① 用 bfloat16 加速 + 数值稳定
    X /= X.norm() + eps                    # ② 归一化:让最大奇异值 ≤ 1
    if G.size(0) > G.size(1):
        X = X.T                            # ③ 转置让行 ≤ 列,后续迭代更快
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:  # ④ 5 次固定系数迭代
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X                 # ⑤ Newton-Schulz update
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

**Newton-Schulz 迭代的数学**:
- 想对 G 做 SVD: G = U·Σ·V^T,我们要的是 U·V^T(即 Σ=I,所有奇异值归一)。
- 直接 SVD 是 O(min(m,n)·m·n),太慢。
- Newton-Schulz 用一个二次/三次多项式迭代,**5 步就能收敛到 UV^T**,代价只是 matmul。
- 系数 (3.4445, -4.7750, 2.0315) 来自 Bernstein & Newhouse 2024,是"在 [0, 1] 区间最大化收敛斜率"的优化解。
- 输出是"近似正交矩阵":奇异值都接近 1,**所有方向步长被均匀化**。

### 3.3 muon_update:动量 + 正交化 + 4D 张量处理

`optim/muon.py:59-96`:

```python
def muon_update(grad, momentum, beta=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - beta)         # ① 动量 EMA: m = β·m + (1-β)·g
    update = grad.lerp(momentum, beta) if nesterov else momentum
                                           # ② Nesterov lookahead: u = g + β·m
    if update.ndim == 4:                   # ③ 4D 卷积核: (out, in, kH, kW) → (out, in*kH*kW)
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)   # ④ 正交化
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5  # ⑤ 形状补偿
    return update
```

**逐行解释**:
1. **动量 EMA**: `momentum.lerp_(grad, 1-β)` 是 PyTorch 的线性插值,等价于 `m = m + (1-β)·(g - m) = β·m + (1-β)·g`。
2. **Nesterov**: `grad.lerp(momentum, β)` = `g + β·(m - g)`,这是 Nesterov 加速梯度公式,等效于"先按动量方向往前看一步,再算梯度"。
3. **4D 张量 reshape**: 卷积核 shape `(out_channels, in_channels, kernel_h, kernel_w)` 展平成 `(out_channels, in_channels * kH * kW)`,把每个输出通道的整个 filter 视为一个"行向量"做正交化。**这是 Muon 能处理 CNN 的关键**。
4. **正交化**: 见 3.2,得到"所有方向均匀步长"的更新量。
5. **形状补偿**: `sqrt(max(1, dim[-2] / dim[-1]))` —— 矩阵"瘦高"(行 < 列)时,正交化后 Frobenius 范数偏小,补一个缩放因子补偿回来,让"瘦高矩阵"的更新量和"矮胖矩阵"一致。

### 3.4 MuSGD 的混合策略

`optim/muon.py:99-251` MuSGD:

```python
class MuSGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0,
                 nesterov=False, use_muon=False, muon=0.5, sgd=0.5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon = muon      # ⑥ Muon 分量缩放
        self.sgd = sgd        # ⑦ SGD 分量缩放

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["use_muon"]:    # ⑧ use_muon=True: 矩阵参数
                for p in group["params"]:
                    # Muon 部分
                    update = muon_update(p.grad, momentum_buf, ...)
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    # SGD 部分(同 lr 但独立 momentum buffer)
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)
                    sgd_buf.mul_(momentum).add_(grad)
                    sgd_update = grad.add(sgd_buf, alpha=momentum) if nesterov else sgd_buf
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:                    # ⑨ use_muon=False: 标量/向量参数(bias/norm/embedding)
                # 纯 SGD,标准实现
                ...
```

**MuSGD 的设计意图**:

| 参数类型 | 是否用 Muon | 原因 |
|---|---|---|
| Conv2d weight (4D) | ✅ | 矩阵参数,正交化收益大 |
| Linear weight (2D) | ✅ | 同上 |
| BatchNorm weight (1D) | ❌ | 1D 张量没法正交化,纯 SGD |
| Bias (1D) | ❌ | 同上 |
| Embedding (2D,但稀疏更新) | ❌ | 一次只更新一行,正交化无意义 |

**实际使用**: ultralytics 8.4.95 在 `engine/trainer.py` 里把模型参数按 shape 自动分组,1D 的走纯 SGD,2D+ 的走 `use_muon=True`。**用户不需要手动分组**。

**muon=0.5, sgd=0.5 的含义**:
- 对 use_muon=True 的参数,update 实际是 `0.5·lr·μ_update + 0.5·lr·sgd_update`。
- Muon 分量提供"方向均匀"的更新;SGD 分量提供"自适应缩放"和 weight decay 解耦。
- 两个分量共享 lr,系数 0.5/0.5 让总有效 lr 仍是 `lr`(相对于纯 SGD)。

### 3.5 Muon vs SGD:效果上差在哪

| 维度 | SGD | Muon |
|---|---|---|
| 更新量方向 | 任意 | 近似正交(均匀) |
| 不同奇异值步长 | 差异大 | 近似一致 |
| 矩阵参数收敛速度 | 慢,某些方向停滞 | 快 ~2x (paper claim) |
| 计算代价 | O(n) per param | O(n^2) for matmul + Newton-Schulz |
| 内存代价 | 1 个 momentum buffer | 2 个 buffer (muon + sgd) |
| 适用参数 | 所有 | 仅 2D+ |

**对 YOLO 训练的实际效果**:
- Conv weight 占 YOLO26 参数的 ~95% → Muon 主要加速这些
- BN/bias 占 ~5% → 纯 SGD 够用
- Paper 报告 MuSGD 训练速度比 SGD 快 ~30-50% 达到相同 mAP(在小模型上更明显)

### 3.6 本项目 V4 配置怎么启用 MuSGD

```bash
optimizer=MuSGD lr0=0.02 momentum=0.937  # 关键
```

`engine/trainer.py` 会在 `build_optimizer()` 里:
1. 识别 `MuSGD` 类,自动按参数 shape 分组
2. 2D/4D 参数设 `use_muon=True, momentum=0.95`
3. 1D 参数设 `use_muon=False, momentum=0.937`
4. 总 lr = 0.02 (Muon 和 SGD 各拿 0.02 * 0.5 = 0.01 有效 lr)

---

## 4. 三个创新点在本项目钢卷头尾小目标的落地

### 4.1 关键判断

| 创新点 | 适用场景 | 本项目相关性 | 本项目状态 |
|---|---|---|---|
| **End2end + DFL removal + ProgLoss** | 通用(端到端,易部署)| 中:CPU 推理加速但 NMS 已被本项目手工部署替代 | ✅ 已在 V1-V4 训练中启用 |
| **STAL (TAL scale-aware)** | 小目标 + 密集 anchor | 高:20×20 tip 容易零正样本 | ✅ YOLO26 内置,自动启用 |
| **MuSGD** | 大模型 + 矩阵参数主导 | 低:本项目 YOLO26n(2.4M params),数据 312 张,优化器不是瓶颈 | ✅ V4 启用,但 ROI 有限 |

### 4.2 V4 训练脚本里的实际配置

```bash
# train_v4_final.sh
optimizer=MuSGD lr0=0.02 momentum=0.937 \
iou_loss_weight_nwd=0.5 iou_loss_weight_ciou=0.5 \
box_soft_sigma=1.0 box_soft_train_only=True \
nwd_constant=12.0 \
degrees=10 translate=0.1 scale=0.2 flipud=0.5 \
copy_paste=0.2 mosaic=1.0 \
epochs=250 patience=30 close_mosaic=20
```

- ✅ 5 个创新点全部启用(End2end 双头 + DFL removal reg_max=1 + ProgLoss o2m 衰减 + STAL scale-aware + MuSGD)
- ✅ NWD 0.5 + CIoU 0.5 弥补 STAL 的几何不友好
- ✅ box_soft_sigma=1.0 给 target bbox 加小高斯噪声,提高泛化
- ⚠️ 强 aug(degrees=10, flipud=0.5, copy_paste=0.2)对小目标可能过度,但 scale=0.2 已温和化

### 4.3 三个创新点的"代码依赖关系"

```
Detection head (head.py: Detect)
   └─ end2end=True → one2one_cv2/cv3 存在
       └─ Loss (loss.py: E2ELoss)
           ├─ one2many = v8DetectionLoss(tal_topk=10)   ← ProgLoss 初始 0.8
           └─ one2one  = v8DetectionLoss(tal_topk=7, tal_topk2=1)  ← ProgLoss 终态 0.9
               └─ Assigner (tal.py: TaskAlignedAssigner)
                   ├─ select_candidates_in_gts  ← STAL wh_mask
                   ├─ get_box_metrics           ← α=1, β=6
                   └─ select_topk_candidates    ← topk=13
       └─ Optimizer (optim/muon.py: MuSGD)
           ├─ 2D/4D params → use_muon=True   ← Newton-Schulz
           └─ 1D params    → use_muon=False  ← 纯 SGD
```

**一句话总结**: YOLO26 用一个 **双头 + 双 loss + 正交化优化器** 的协同设计,实现了"非冗余边界框"—— 每个创新点单独看都有道理,但合起来才形成完整闭环。

---

## 5. 关键引用

- **`repos/ultralytics/ultralytics/nn/modules/head.py:89-258`** — Detect head 双头 + postprocess top-k
- **`repos/ultralytics/ultralytics/utils/loss.py:1174-1206`** — E2ELoss + ProgLoss 衰减
- **`repos/ultralytics/ultralytics/utils/tal.py:14-356`** — TaskAlignedAssigner(STAL scale-aware)
- **`repos/ultralytics/ultralytics/optim/muon.py:9-339`** — MuSGD / Muon / Newton-Schulz
- **`src/hyper_yolo_patches/yolo26_loss_extension.py`** — NWD+CIoU blend + box soft patch
- **`repos/yolo26-coil/train_v4_final.sh`** — V4 训练脚本(5 创新点 + 强 aug + NWD 0.5)
- 关联: [[yolo26-experiment-2026-07-13]] / [[v18-3-hard-neg-success]] / [[hyper-yolo-runs-status]]