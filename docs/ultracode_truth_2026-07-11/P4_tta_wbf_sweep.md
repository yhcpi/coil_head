# P4：v11 TTA (builtin / custom) + WBF sweep

> 日期 2026-07-11 · 权重 `runs/cfg_truth_repro/v11_baseline_strong_aug_full/weights/best.pt`
> val=99 张（43 正 + 56 负）· 评估口径 = per-image top1 · Lenient-Match（中心距离 d≤30px）
> 脚本：`scripts/tta_inference.py`（生成全模式预测 + WBF 前候选 → `/tmp/v11_tta_all.json`）、`scripts/tta_sweep.py`（离线 conf/WBF-IoU sweep，纯 CPU）

## 结论（TL;DR）

**F1 没有到 0.93+。baseline（无 TTA）仍是最优，F1=0.9176。TTA 两种实现都跑输了。**

| 配置 | best F1 | @conf | Recall | Precision | TP/FP/FN |
|---|---|---|---|---|---|
| **baseline（无 TTA）** | **0.9176** | 0.15 | 0.9070 | 0.9286 | 39/3/4 |
| TTA-builtin（augment=True） | 0.8506 | 0.50 | 0.8605 | 0.8409 | 37/7/6 |
| TTA-custom（WBF IoU=0.55） | 0.9048 | 0.25 | 0.8837 | 0.9268 | 38/3/5 |

TTA 相对 baseline：builtin **−6.7pp**，custom **−1.3pp**。没有一个方向能推到 0.93。

---

## 1. baseline vs builtin vs custom（学术 mAP50，conf=0.001）

| mode | IoU-mAP50 | Lenient-mAP50 |
|---|---|---|
| baseline | 0.7841 | **0.9006** |
| builtin | 0.7725 | 0.8634 |
| custom | **0.8404** | 0.8882 |

- custom 的 **IoU-mAP50 最高**（0.8404，多尺度让框更贴合），但 **Lenient-mAP50 反而不如 baseline**——钢卷场景只看中心命中，定位精度红利用不上。
- builtin 两项都最低。

## 2. conf sweep（Lenient-Match d≤30，各配置逐点）

**baseline**（复现历史 0.9176，验证 pipeline 正确）：

| conf | TP | FP | FN | Recall | Prec | F1 |
|---|---|---|---|---|---|---|
| 0.001 | 39 | 32 | 4 | 0.9070 | 0.5493 | 0.6842 |
| 0.10 | 39 | 4 | 4 | 0.9070 | 0.9070 | 0.9070 |
| **0.15** | **39** | **3** | **4** | **0.9070** | **0.9286** | **0.9176** |
| 0.20 | 38 | 3 | 5 | 0.8837 | 0.9268 | 0.9048 |

**TTA-custom（WBF 0.55）**：

| conf | TP | FP | FN | Recall | Prec | F1 |
|---|---|---|---|---|---|---|
| 0.001 | 40 | 56 | 3 | 0.9302 | 0.4167 | 0.5755 |
| 0.15 | 39 | 7 | 4 | 0.9070 | 0.8478 | 0.8764 |
| 0.20 | 38 | 5 | 5 | 0.8837 | 0.8837 | 0.8837 |
| **0.25** | **38** | **3** | **5** | **0.8837** | **0.9268** | **0.9048** |
| 0.30 | 36 | 1 | 7 | 0.8372 | 0.9730 | 0.9000 |
| 0.40 | 33 | 0 | 10 | 0.7674 | 1.0000 | 0.8684 |

**TTA-builtin** 最优仅 0.8506 @conf=0.50（P 始终压不上去，FP 顽固）。

## 3. WBF-IoU sweep（custom，离线重合并候选）

| WBF IoU | best F1 | @conf | Recall | Prec | 备注 |
|---|---|---|---|---|---|
| 0.45 | 0.8941 | 0.20 | 0.8837 | 0.9048 | 合并更激进，略差 |
| **0.55（默认）** | **0.9048** | 0.25 | 0.8837 | 0.9268 | 最优 |
| 0.65 | 0.8974 | 0.40 | 0.8140 | 1.0000 | conf=0.4 时 FP=0/P=1.0，但 Recall 掉到 0.814 |

**不值得改 WBF IoU 阈值**：默认 0.55 已经是三者最优；0.45 更差，0.65 虽能拿到 Precision=1.0（FP=0）但要牺牲 8 个 TP，F1 更低。若业务是"零误报优先"，0.65 @conf=0.4（P=1.0 R=0.814）可作为保守档，但那是另一个目标函数。

## 4. 为什么 TTA 反而更差（根因）

- **正样本天花板几乎没动**：43 张正样本里，baseline FN=4、custom 低 conf 时 FN=3——TTA 多尺度只多救回 **1 张**难例，且一旦把 conf 提到可用区间（≥0.25）就又丢了。
- **负样本是杀手**：56 张负样本，conf=0.001 时 baseline FP=32，而 **custom FP=56（每张负样本都误触发）**、builtin FP=43。TTA 把每张图的检测数翻了几倍，负样本上的假阳性同步暴涨。
- 要压掉这些 FP 就得抬高 conf 阈值，但抬阈值又会连带杀掉真阳性（TP↓）→ Precision 涨的同时 Recall 崩，F1 净亏。这就是 custom 卡在 0.9048、builtin 卡在 0.8506 的机制。

一句话：**这个 val 上瓶颈是负样本误报，不是正样本漏检；而 TTA 恰好放大误报，方向相反。**

## 5. v12 状态

**未纳入对比**：v12（`v12_strong_aug_flipud_300ep`）当前仅训到 **epoch 9**（results.csv 10 行，训练进程 PID 2048966 仍在跑），best.pt 尚无参考价值，未达 task 要求的 epoch50+。待其训到 50+ 后可用同一套 `tta_inference.py --mode all` + `tta_sweep.py` 复跑对比。

## 6. 交付物 / 复现

```bash
# 1) 一次推理，产出全模式预测 + WBF 前候选
python scripts/tta_inference.py \
  --weights runs/cfg_truth_repro/v11_baseline_strong_aug_full/weights/best.pt \
  --mode all --save_json /tmp/v11_tta_all.json
# 2) 离线 conf sweep + WBF-IoU sweep（不占 GPU）
python scripts/tta_sweep.py --json /tmp/v11_tta_all.json --wbf_sweep
```

改动：`tta_inference.py` 增补保存全模式预测/候选 + `--wbf_iou` 参数（<30 行）；新增只读分析脚本 `tta_sweep.py`。未重训任何模型。

## 建议

- **保持 baseline 部署（conf=0.15，F1=0.9176），放弃 TTA。** TTA 在钢卷小目标+大量负样本场景是负收益。
- 想突破 0.93 的正确方向不是 TTA，而是**降负样本误报**：更强负样本训练 / 分类头置信度校准 / 后处理加"负样本静默"门槛（当前 4 口径 TN 全为 0，说明模型对负样本几乎从不静默，这才是真正的天花板）。
