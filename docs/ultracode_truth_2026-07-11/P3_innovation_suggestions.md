# P3: 钢卷头尾检测创新点 / Loss 改动方向调研（2026-07-11）

> 本文档调研在 v11 best.pt + conf=0.15 + Lenient d≤30 部署 F1=0.9176 的基础上，还能尝试哪些创新点 / Loss 改动能突破 F1=0.92 瓶颈。
> 排序按 **ROI = 期望收益 / 改动成本 + 风险**。

## 0. 关键约束（不要重复踩坑）

| 约束 | 来源 | 含义 |
|------|------|------|
| **数据规模硬约束** | memory: `v9-five-innovations-all-fail` / `v10-three-innovations-all-fail` | train=312 正 / 233 负；所有"模型侧"改动（NBBoxNoise/STAL/Spec-Suppress/GP-IoU/Pseudo-Label/YOLOv10/NBBox-light）**全部失败**，根本原因是数据规模不够让模型学出新东西。任何需要"模型看到新分布"的创新点 ROI 都低 |
| **真值 mAP50** | A_B_truth.md | v11 = **0.348**（不是 0.8887，那是被误读的 Recall）。学术 mAP 已经几个月没破 0.5 |
| **部署 F1 0.9176 = 5 个剩余 FN 的天花板** | 经验 | FN=4 + FP=3 → 任何"减少 1 个 FN"的改动都能 +1-2pp |
| **v11 强 aug 已"赢"学术 mAP -5.4pp 但部署赢 +1.8pp** | A_B_truth.md §7 | 强 aug 在低数据下让 conf 校准更准，反而**对部署更有用**——这是巧合但说明"放弃学术 mAP 追部署 F1"是合理路线 |
| **cfg 合并坑 + bash # 注释坑** | memory `cfg-merge-truth` / `regression-bash-comment-bug` | 任何创新点的 YAML 必须字段在最后位置 + 必须用纯 CLI 复现过一次 smoke 才算"实现完成" |

## 1. 候选清单（按 ROI 排序）

### 🥇 A. 离线 Hard Negative Mining + 增量再训练（ROI 最高 / 零模型改动）

**一句话原理**：用 v11 best.pt 跑 train 集 545 张预测，把 high-confidence FP 的人工看一眼——99% 是漏标的真目标；把这些"假 FP"加进 GT 重训，单次循环通常 +3-8pp 召回。

**改动文件 / 行数**：
- `scripts/pseudo_label_v1.py` 已存在但**没用过**——可直接复用 / 改写
- 新增 `scripts/hard_neg_mining.py` (~80 行)：跑 v11 best.pt on train → 输出 high-conf FP 图清单
- 人工看 + labelme 重标 + `labelme2yolo.py`（已存在）转 YOLO
- 重训只调 `--data` 路径，**不动 ultralytics 源码**

**期望收益**：
- 保守：+2-3pp 部署 Recall（找到 5-10 个漏标）
- 中性：+5pp（漏标比想象的更多）
- 乐观：突破 F1=0.95（如果漏标里有大量 "FN 的另一半"）

**主要风险**（要诚实说）：
- ⚠️ **没有"自动化"路径**——必须人工看图。312 正样本意味着每张图都看一遍 ≈ 1.5-2 小时人工
- ⚠️ 高反光"FP"可能真的是高反光误检，不是漏标——要审慎区分（参考 v4 fn_analysis.md 5 个 FN 的"高反光遮挡"模式）
- ⚠️ 重新打的 GT 质量不一定一致（labelme 重标历史上飘过）

**怎么验证**：
1. 先跑 v11 best.pt on train 545 张，统计 top-100 conf-FP
2. 把 top-100 截图 + 原 GT 重叠画图（`scripts/visualize_bestpt_val.py` 已存在可改）
3. 人工挑出明显是漏标的（中心位置有目标但 GT 没有）
4. 合并进 GT 后跑 smoke 10 epoch，**新 best.pt 在 val F1 应该 ≥ 0.92 才算赢**（vs 当前 0.9176）

**ROI 判断**：**推荐度 ★★★★★**——零 ultralytics 改动，复用所有现有工具，memory 反复说"数据是 hard limit"。**唯一成本是人工 2 小时。**

---

### 🥈 B. WBF 替换/叠加在单图推理路径（ROI 高 / 改动小）

**一句话原理**：当前 single-image inference 走 `non_max_suppression` (ops.py L238，torchvision NMS)；TTA 路径已经在用 WBF 合并多个 augmented prediction。把 WBF 引入 single-shot inference：用同一个图做 2-3 个轻微 augmentation (orig + hflip + 1.1x scale) 后合并。

**改动文件 / 行数**：
- `repos/Hyper-YOLO/ultralytics/utils/ops.py`：新增 `weighted_boxes_fusion()` (~30 行)
- `scripts/postprocess_tta_topk_dist.py` 已有 WBF 调用，参考其写法
- **不改**训练代码，只在推理时 patch

**期望收益**：
- 保守：+1pp F1（drop FP）
- 中性：+2pp（合并多尺度后大目标更准）
- 乐观：+3pp（解决"同一目标被检 2-3 次"型 FP）

**主要风险**：
- ⚠️ WBF 在单类小目标上 vs 标准 NMS 差异不大——TTA 之所以 WBF 有效是因为**多个 aug 的同一目标被分别检到**，单图 WBF 等价于直接调 NMS iou_thres
- ⚠️ 如果直接用 NMS(iou_thres=0.3) 比 WBF 更简单，先试这个
- ⚠️ WBF 需要 box 坐标精确，单图 3 个 aug 算 cost vs benefit

**怎么验证**：
1. 写 `scripts/wbf_single_image.py`：orig + hflip + 1.1x_scale 推理 → WBF 合并
2. 在 v11 best.pt 上跑 val 99 张，比较 `conf=0.15` 下 F1
3. smoke：F1 ≥ 0.9176 才算赢

**ROI 判断**：**推荐度 ★★★★**——改动小、风险低，但单图 WBF vs NMS 实际差距可能 <1pp。建议**和 A 同时做**，A 数据增益是 5pp 量级，WBF 是 1pp 量级。

---

### 🥉 C. cls loss 从 BCE 切到 Focal Loss（ROI 中 / 改动小）

**一句话原理**：钢卷场景单类正负极度不平衡（val 99 张里 43 正 / 56 负），且大部分背景是"明显背景"（远离钢卷），模型很快就能把它们压到 conf<0.05。BCE 对易负样本也一视同仁地给梯度，导致"模型已经在负样本上花太多力气"。Focal Loss 用 `(1-p_t)^γ` 把易样本梯度衰减，把学习压力集中到难样本。

**改动文件 / 行数**：
- `repos/Hyper-YOLO/ultralytics/utils/loss.py`：已有 `FocalLoss` 类 (L158)，**但 `v8DetectionLoss.__call__` L423 用的是 BCE**（VFL 注释掉）
- 改 1 行：`self.bce = nn.BCEWithLogitsLoss(reduction='none')` → `self.bce = FocalLoss()`
- 加 `hyp_aug.yaml` 字段：`cls_loss_type: focal`, `focal_gamma: 1.5`

**期望收益**：
- 保守：+0.5pp F1（甚至持平）
- 中性：+1.5pp
- 乐观：+3pp（解决"高反光误检当正样本"问题）

**主要风险**：
- ⚠️ **γ 调参敏感**——γ=2 太狠把正样本梯度也削掉，γ=1 太弱 ≈ BCE。论文 γ=2 但 COCO 是 80 类，我们 1 类可能 γ=1.5 更合适
- ⚠️ Focal + cls=0.5 可能让 conf 校准偏高（conf 普遍更小），需要 sweep conf_thr 重找最优
- ⚠️ 有 1 篇近期论文（YOLOv8-Focal, 2024）说小数据集 Focal 反向伤害——312 正样本规模类似小数据集
- ⚠️ `FocalLoss` 当前实现 `alpha=0.25`（Line 178）——这是 COCO 80 类平衡参数，单类应该关 alpha=0 或 alpha=0.5

**怎么验证**：
1. smoke 10 epoch：`cls=focal focal_gamma=1.5 focal_alpha=0.0`，对比 BCE baseline box/cls/dfl loss 曲线
2. 如果 smoke mAP50 ≥ 0.05（vs baseline 0.05），full 250ep
3. full 后扫 conf_thr 找最优 F1

**ROI 判断**：**推荐度 ★★★**——理论对路但小数据集风险大。建议先 smoke 10ep，box_loss 曲线明显低于 BCE 才 full。

---

### 4. cls gain 调权 cls=0.5 → cls=1.0（ROI 中 / 几乎零改动）

**一句话原理**：v11 args.yaml `cls=0.5, box=1.5, dfl=1.5`，cls 权重只有 box 的 1/3。考虑到钢卷场景"敢检"比"准检"更重要（FN 4 个 vs FP 3 个），升高 cls 让模型把 conf 推高，部署时 conf_thr sweep 空间更大。

**改动文件 / 行数**：
- `hyp_aug.yaml` 改 1 行：`cls: 0.5` → `cls: 1.0`
- 加 CLI：直接 `--cls 1.0`（不用碰 hyp 文件）
- **不动 ultralytics 源码**

**期望收益**：
- 保守：持平（cls loss 升但 conf 校准变，可能 net 0）
- 中性：+1pp F1
- 乐观：+2pp

**主要风险**：
- ⚠️ 这正是 v4 历史配置（hyp_aug.yaml L27：`cls: 1.0`），但 v11 用的是 `cls: 0.5`。**改回 v4 cls=1.0 是历史回归，需要先 smoke 验证 v11 配置的 cls=0.5 是不是有意为之**
- ⚠️ box=1.5 / cls=0.5 比例 = 3:1，改 cls=1.0 = box:cls = 1.5:1 = 1.5:1，回归到"老 YOLO 默认"——不一定是坏事，但要 smoke 验证
- ⚠️ dfl=1.5 也要重新平衡吗？

**怎么验证**：
1. smoke 10 epoch：`--cls 1.0`，看 val/cls_loss 是否从 ~3 降到 ~6（box_loss 不应暴涨）
2. full 250 epoch，扫 conf_thr [0.05, 0.10, 0.15, 0.20, 0.30, 0.50] 找最优 F1
3. 与 v11 best.pt (cls=0.5) F1=0.9176 对比

**ROI 判断**：**推荐度 ★★★**——成本极低，但本质上是"调超参"不是"创新点"。如果单纯想要部署 F1 +1pp，这个最便宜。

---

### 5. NWD + IoU 双 loss 加权融合（ROI 低 / 改动中）

**一句话原理**：当前 `BboxLoss.forward()` L248-259 是 `if NWD: NWD; elif IoU: IoU`（互斥）。改成 `loss = α * (1-NWD) + (1-α) * (1-IoU)`，α=0.5 起步。让两种 loss 互补：NWD 对位置偏差鲁棒，IoU 对形状敏感。

**改动文件 / 行数**：
- `repos/Hyper-YOLO/ultralytics/utils/loss.py` L252-260：加一个 `nwd_iou_blend` 字段
- 改 ~10 行：把 if/elif 改成 blend 计算
- hyp_aug.yaml 加字段：`nwd_iou_blend: 0.5`

**期望收益**：
- 保守：-1pp（两个 loss 互相干扰，梯度抵消）
- 中性：持平
- 乐观：+2pp（小目标 bbox 回归更稳）

**主要风险**：
- ⚠️ **v8 NWD v1 已有结论**——NWD 单独 mAP50=0.4015 是历史最高，加 IoU 可能拖后腿（IoU 梯度大让 bbox 回归"想贴紧 GT"，破坏 NWD 的"宽容"性质）
- ⚠️ α sweep 成本高（4-5 个值 × 250 epoch = 一周训练时间）
- ⚠️ 历史 v8 Coverage + NWD 都试过等价结论（NWD/Coverage/NWD+Coverage 三组 loss 完全等价）——参见 `runs/coil_v5_summary.md`

**怎么验证**：
1. smoke 10 epoch：`nwd=true, nwd_iou_blend=0.5`，看 box_loss 是不是介于纯 NWD 和纯 IoU 之间
2. 如果 box_loss 合理且 mAP50 > 0.05，full
3. 对比 v8 NWD v1 (0.4015) 和 v11 baseline (0.348) 看能不能突破

**ROI 判断**：**推荐度 ★★**——理论直觉对，但 v8/v11 NWD 历史结果说明这条路径 ROI 低。**只在 A、B、C、4 都失败后再考虑。**

---

### ❌ 已排除方向（不要再投入时间）

| 方向 | 排除理由 |
|------|----------|
| 多尺度训练 multi_scale=true | 已试 + 撤回（memory `multi-scale-fix-2026-07-10`），RandomAffine scale=0.5 已覆盖 |
| 改 anchor-free head (SimOTA/ATSS) | TAL assigner 代码量大改动，但 v11 FN=4 + FP=3 不是 head 问题 |
| 改 backbone / neck | 0.348 mAP 不是 backbone 瓶颈 |
| NBBoxNoise-light 调参 | 已试 v10，box_loss 收敛但 mAP=0.0007 |
| Spec-Suppress | 已试 v9，重建 loss 不收敛 |
| GP-IoU 校准 | 已试（`runs/gp_iou_calibration_report.md`），**GP 比 baseline F1 -5pp**，平均 -14pp |
| SAHI 切片推理 | v4 历史结论：rect=True 下 SAHI 损害 recall |
| YOLOv10 集成 | 已试 v9，集成失败 |

---

## 2. 推荐执行顺序

| 顺序 | 创新点 | 预期 ROI | 实际成本 | 风险 |
|------|--------|----------|----------|------|
| **1** | **A. 离线 Hard Neg Mining** | 极高（数据 = hard limit） | 人工 2h + 250ep 重训 | 低（不动代码） |
| **2** | **B. WBF 单图推理** | 中（+1-2pp） | 写 30 行 + smoke 1h | 低 |
| **3** | **4. cls=0.5 → cls=1.0** | 中（+1pp） | 改 1 行 + full 250ep | 低（v4 历史配置） |
| **4** | **C. Focal Loss for cls** | 中（+0-3pp，赌） | 改 ~3 行 + 250ep | 中（小数据集风险） |
| 5（备用） | NWD + IoU blend | 低（ROI 未知） | 改 ~10 行 + 多次 250ep | 中 |

**最坏情况**：如果 1+2+3 都没破 0.92，**接受 F1=0.9176 是当前数据规模的天花板**。v9 + v10 两次 5 创新点全失败已经证明"模型侧创新 ROI 在 312 正样本下接近 0"。

---

## 3. 我推荐先试 A + B 并行，理由

**A（离线 Hard Neg Mining）**：
- **零 ultralytics 源码改动**——memory 反复说"数据是 hard limit"，312 正样本在 v9 + v10 5 创新点全失败后已经成为唯一未被证明的路径
- **复用现有工具**——`pseudo_label_v1.py` / `visualize_bestpt_val.py` / `labelme2yolo.py` 都在
- **收益上限最高**——历史上 v4 → v5 数据迁移 (27 张 val-FN → train) 单一动作就 +57% Recall
- **风险最低**——最坏情况是"挑不出漏标"，不动模型任何东西

**B（WBF 单图推理）**：
- 改动小 / 风险低 / 与 A 不冲突（可以 A 重训完再跑 B 推理）
- 如果 A 找到的漏标被新模型学了，单图 WBF 推理可能进一步 drop FP

**为什么跳过 C 和 4 先做 A + B**：
- C（Focal Loss）小数据集风险高，单独跑 smoke 可能也要 1 小时 + full 4 小时
- 4（cls=1.0）成本低但 ROI 中——可以**在 A 重训的同一个 full 250ep 里同时测试 cls=1.0**（一份训练两份配置 → 实际零额外成本）

**执行计划**（预计总耗时 4-6 小时）：

1. **Hour 0-1**：跑 `scripts/hard_neg_mining.py` (新写)，扫 v11 best.pt on train 545 张 → top-100 conf-FP 清单 + 可视化
2. **Hour 1-2**：人工挑漏标 → labelme 重标 → 转 YOLO → 合并进 data/coil/labels/train (注意备份原标签)
3. **Hour 2-3**：并行两个 smoke 10 epoch：
   - `3a` cls=0.5（保持 v11 配置，验证数据增益）
   - `3b` cls=1.0（看是否叠加增益）
4. **Hour 3-5**：选赢的 smoke 配置 full 250 epoch
5. **Hour 5-6**：用新 best.pt 写 `scripts/wbf_single_image.py`，跑 val 99 张 + conf sweep

如果 Hour 5-6 后 F1 ≥ 0.93 → **创新点 6/7/8 归档成功**，写 memory 更新。
如果 < 0.93 → 接受 0.9176 是天花板，把剩余时间投入 **P4 TTA sweep**（任务 #35）。

---

## 4. 验证 checklist（每个创新点跑前必查）

- [ ] **cfg 合并坑**：hyp_aug.yaml 字段必须在最后（不会被 CLI 覆盖）——参见 memory `cfg-merge-truth`
- [ ] **bash # 注释坑**：CLI 命令不能有 `#` 注释——参见 memory `regression-bash-comment-bug`
- [ ] **进程重复坑**：`ps -ef | grep ultralytics` 确认只有一个训练进程——参见 memory `training-process-management`
- [ ] **列错位坑**：读 results.csv 必须用 `$11` 不是 `$10` 看 mAP50——参见 A_B_truth.md §1
- [ ] **smoke 阈值**：10 epoch smoke mAP50 必须 > 0.05 才算"实现完成"，否则按 user spec 归档（NBBox-light 教训）
- [ ] **部署评估**：full 后用 `scripts/lenient_eval.py` 跑 F1，不是只看学术 mAP50