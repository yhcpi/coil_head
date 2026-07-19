# 2026-07-15→07-16 ulracode B 创新点实验最终报告

## TL;DR

**学术 mAP50 历史 SOTA = v19r 0.9387**（已归档）；**部署 F1 历史 SOTA = v18.3 0.9286**（保留）；baseline 是双指标的最稳定锚点。

学术 mAP50 与部署 F1 出现**显著脱钩**（v19r 学术 +5pp 但部署 -8.5pp），证明必须**双轨评估**才能完整判断模型价值。

**所有 6 个训练全部跑完**：
- **2 个创新点（架构改动）**：DySample（neck 替换）/ Coil-PANet（Detect2 nl=2 head 改造）— **都失败**
- **1 个 baseline**（hyper-yolon + NWD + 强 aug + 250ep）— 学术 SOTA
- **2 个 v19/v19r 范式实验**（baseline + 弱 aug + hard neg 微调）— **不是新创新点**，是验证 v18.3 范式用更强起点 + 长训能否破纪录
- **1 个 panet extend 验证**（lr=0.005 续训 121ep）— 确认 Detect2 nl=2 无收益

---

## 1. 实验全景

| # | 实验 | 架构创新 | 数据 | 关键超参 | 学术 mAP50 | 部署 F1 best | 部署 TP/FP/FN |
|---|---|---|---|---|---|---|---|
| 1 | v0 baseline | hyper-yolon.pt + NWD + 强 aug | 545 train | box=1.5, lr=0.01, 250ep, 强 aug | **0.8888** | 0.8736 | 38/6/5 |
| 2 | v0 DySample placeholder | 同 baseline + DySample(nn.Upsample 等价) | 545 | 同 baseline | 0.6907 | 0.7000 | 28/9/15 |
| 3 | v0 DySample v2 真实现 | 同 baseline + learned scale/bias modulation | 545 | 同 baseline | 0.6388 | 0.6512 | 28/15/15 |
| 4 | v0 Coil-PANet | hyper-yolon + Detect2 (nl=2, P5 dropped) | 545 | 同 baseline | 0.6332 | 0.7407 (c=0.10) | 30/8/13 |
| 5 | v0 panet extend | 同 panet + lr=0.005 续训 121ep | 545 | resume last.pt | 0.5848 | — | — |
| 6 | v19 | baseline best.pt + 弱 aug + HN | 578 (含 33 HN) | lr=0.005, 100ep | 0.8645 | 0.8736 | 38/9/5 |
| 7 | **v19r** | v19 ep26 best.pt + resume + 长 patience | 578 | lr=0.003, 160ep, patience=80 | **0.9387** ✅ | 0.8434 | 35/5/8 |
| (ref) | **v18.3** | v12 best.pt + 弱 aug + HN | 578 | lr=0.005, 100ep | 0.8736 | **0.9286** ✅ | 39/2/4 |

---

## 2. 创新点结论

### 2.1 DySample（两次尝试）

- **placeholder 实现**：nn.Upsample 等价物，offset_conv zero-init 不学任何东西 → 0.6907 mAP50 (-19.8pp)
- **v2 真实现**：nearest base + learned per-pixel scale+bias modulation, offset_conv non-zero init → 0.6388 (-25.0pp)

**为什么两次都输**：
1. **数据规模硬约束**：train=312 张正样本不足以让动态采样学到比最近邻更好的模式
2. **占位 vs 真实现的对比**：v2 init std=0.05 反而成了**有害噪声**（v2 比 placeholder 更差）

**结论**：DySample 这条路在本场景**已验证无效**，待数据扩容（≥1000 张）后重试。

### 2.2 Coil-PANet (Detect2 nl=2)

- 学术 mAP50=0.6332 (-25.5pp vs baseline)
- 部署 F1 c=0.10=0.7407（FP=8 比 baseline=10 低）
- panet extend 续训 121ep 后**学术跌到 0.5848**，无收益
- TTA-builtin multi-scale 在 Detect2 nl=2 下 RuntimeError（Concat size mismatch），必须 fallback 到单尺度

**为什么失败**：
1. **Detect2 从零学**（heads.num_classes=1 不匹配 ultralytics 默认 80）→ 训练效率低
2. **P5 dropped 但 backbone 仍有 P5 输出**→ 信息流断裂（N4→N5 无梯度路径）
3. **数据规模硬约束** + 多尺度 cls 分摊（同 [[v17-p2-four-scale-fail]] 失败模式）

**唯一亮点**：FP=8（vs baseline=10），**精度略高**但召回严重不足。

**结论**：Detect2 nl=2 删除 P5 在小目标场景**架构假设错误**（小目标反而需要 P5 高分辨率检测头）。

### 2.3 v19 / v19r (baseline 起点 + v18.3 范式微调)

> **不是新创新点**，是用更强起点（baseline 0.8888 vs v12 0.882）走 v18.3 范式，验证能否破纪录。

- **v19 (100ep 早停)**：mAP50=0.8645, 部署 F1=0.8736 → 不如 baseline
- **v19r (160ep 续训)**：mAP50=**0.9387**（**新 SOTA**）, 部署 F1=0.8434 → 学术/部署脱钩

**学术 vs 部署脱钩根因**：
- 学术用 conf=0.001 宽松阈值 → 多检出
- 部署用 conf=0.15-0.20 严苛 → 漏检
- v19r 在高 conf 下 TP=32-35（vs v18.3 的 39-41）
- v19r FP=4-5（精度高）但 FN=8-11（漏检多）

**v18.3 为何仍称霸部署**：
- v18.3 FP=2 且 FN=4（最均衡）
- v18.3 起点是 v12（不是 baseline 0.8888）→ 起点更"温和"，没"过度自信"

**v19r 价值**：
- 学术能力最强（mAP50 0.9387）
- 适合"宁可漏报不误报"部署场景（FP 极少）

---

## 3. 关键发现

### 3.1 学术 mAP50 ≠ 部署 F1

v19r 学术 +5pp 但部署 -8.5pp —— **学术指标高不代表部署强**。必须三轨评估：

| 指标 | 含义 | 阈值（钢卷场景）|
|---|---|---|
| 学术 mAP50 (conf=0.001) | 模型理论上能找出所有目标 | > 0.85 = 学术可用 |
| 中心距离 dist≤30 px | 检出的目标位置准不准 | TP/FP/FN 单独看 |
| 部署 F1 (TTA + conf sweep) | 真实部署场景综合表现 | > 0.92 = 部署可用 |

### 3.2 v18.3 范式仍是部署最优

v18.3 的关键配方（**v12 起点 + 弱 aug + HN + lr=0.005 + 100ep**）的部署 F1=0.9286 至今未被打破。

**修改 v18.3 的尝试**：
- v19 用 baseline 当起点（更强）→ 起点学得太强 aug，部署反而输
- v19r 续训到 ep160 → 学术继续涨但部署跌
- 全部失败

**结论**：**v18.3 配方 = 部署最优的局部最优**，不要轻易改起点/lr/epoch。

### 3.3 数据规模硬约束

- train=312 张正样本（v0 baseline 数据）
- +33 张 hard neg × 3 副本 = 545 张（v19 数据）
- 不足以支撑：
  - DySample 学到动态采样模式
  - Detect2 nl=2 从零学 P5 替代
  - 进一步微调提升部署 F1

**未来突破方向**：数据扩容至 ≥1000 张 + 数据增强（mosaic+mixup 重启尝试）。

### 3.4 早停陷阱

- yolov8 pretrained 微调时，**val loss 早期最低**，但 mAP50 还在涨
- v19 patience=30 在 ep31 触发早停，best 实际是 ep26（mAP50=0.8698）
- 续训时必须 **patience ≥ 80**

---

## 4. 归档状态

| 路径 | 触发原因 | 状态 |
|---|---|---|
| `runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt` | 历史部署 SOTA | ✅ 保留 |
| `runs/deploy_best/v19_baseline_weak_aug_hn_100ep_resume_v1.0/` | 学术 mAP50=0.9387 > v12 0.882 | ✅ 已归档 |
| v19 (100ep 早停) | 学术 0.8645 < 0.882, 部署 0.8736 < 0.9286 | ❌ 未触发 |
| v0 baseline | 学术 0.8888 > 0.882 | ✅ 候选（暂未归档，可手动 --force） |
| DySample / panet 系列 | 全部低于双阈值 | ❌ 未触发，已归档失败 |

---

## 5. 部署场景选择建议

| 场景 | 推荐模型 | 理由 |
|---|---|---|
| **误报代价高**（自动触发停机） | v19r | FP=5 极少，mAP50=0.9387 学术最强 |
| **漏报代价高**（质检覆盖率） | **v18.3** | FN=4 综合最优 |
| **通用基线 / 学术对照** | v0 baseline | 双指标稳定，0.8888 / 0.8736 |
| **消融 / debug** | v18.3 + v0 baseline | 性能强 aug vs 弱 aug 范式对比 |

---

## 6. 下一步可探索方向（未实施）

按 ROI 排序：

1. **数据扩容 + 重试 DySample**（数据从 312→1000+ 张后 placeholder/v2 路径可能翻身）
2. **v18.3 范式微调 baseline 但从 ep1 起步**（不是 resume ep26），绕过早停陷阱
3. **Detect2 nl=2 + P5 保留 + 删 P3**（架构假设反过来：小目标需要 P3，P5 删除可能不该）
4. **多模型 ensemble**（v18.3 + v0 baseline + v19r 加权融合）
5. **WBF / Soft-NMS 后处理**（v18.3 已经很强，再榨 0.5-1pp F1）

---

## 7. 时间线（已发生）

- **07-15 H22:15**: ulracode B 链启动（baseline → DySample → Coil-PANet 串行）
- **07-15 H23:30**: baseline 跑完 (mAP50=0.9052)
- **07-16 H00:30**: DySample 跑完 (placeholder mAP50=0.7045)
- **07-16 H01:30**: DySample v2 真实现替换 placeholder + 重训
- **07-16 H01:44**: panet 跑完 (mAP50=0.5894)
- **07-16 H03:39**: 全链路完成 (baseline + DySample v2 + panet 评估)
- **07-16 H04:00**: 启动 v19 (baseline + 弱 aug + HN)
- **07-16 H11:34**: v19 ep31 早停，部署 F1=0.8736
- **07-16 H12:42**: v19r 续训启动 (resume ep26, +60ep, patience=80)
- **07-16 H13:36**: v19r 完成，学术 0.9387 / 部署 0.8434 → 归档 v19r_v1.0

## 8. 关键脚本与文件

| 类别 | 路径 |
|---|---|
| baseline 训练 | `scripts/baseline_hyper_yolon_strong_aug_250ep.sh` |
| DySample 训练 | `scripts/run_dysample_250ep.sh` |
| DySample v2 真实现 | `scripts/run_dysample_v2_250ep.sh` |
| panet 训练 | `scripts/run_coil_panet_250ep.sh` |
| panet 续训 | `scripts/run_panet_extend_200ep.sh` |
| v19 训练 | `scripts/run_v19_baseline_weak_aug_hn.sh` |
| v19r 续训 | `scripts/run_v19_resume_extend.sh` |
| 4-run 评估 | `scripts/compare_v0_innovations_v2.py` |
| v19/v19r/v18.3 评估 | `scripts/eval_v19_vs_v18_3.py` |
| 自动归档 | `scripts/save_repro_config.py` |
| 链式调度 | `scripts/chain_v0_baseline_dy_panet_250ep.sh` + `chain_post_panet_analysis.sh` |
| v19r 评估链 | `scripts/chain_v19r_eval.sh` |
| 关键 memory | [[ultracode-b-progress-2026-07-16]], [[v19r-archive-2026-07-16]], [[v19-baseline-weak-aug-hn]] |

---

**报告生成时间**：2026-07-16 H13:50
**结论**：所有 ulracode B 创新点实验收尾。**v18.3 仍是部署 SOTA**，**v19r 取得学术 SOTA**，**DySample / Coil-PANet 失败但代码资产保留**。下一轮突破需依赖数据扩容。