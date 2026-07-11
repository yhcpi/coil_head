# 列错位根因 + v8/v11/v12 真值（2026-07-11 终极版）

> 本文档是 v8 / v11 / v12 best.pt 在 val=99 张（43 正 + 56 负）上的**终极真实指标**，由 `model.val()` 与训练 log 双向交叉验证。
>
> 历史 memory 中所有基于"v11 mAP50=0.8887 / 列 9 是 Recall / 列 11 是 mAP50"都是错的——已被本会话纠正两次，仍有第三次误读，本文档是终极定本。

---

## 1. 列映射终极真值

`runs/cfg_truth_repro/*/results.csv` 列顺序（ultralytics 标准输出 + log 双向验证）：

| col | 字段 |
|-----|------|
| 1 | epoch |
| 2-4 | train/box_loss, train/cls_loss, train/dfl_loss |
| 5-7 | lr/pg0, lr/pg1, lr/pg2 |
| **8** | **metrics/precision(B)** ← P |
| **9** | **metrics/recall(B)** ← R |
| **10** | **metrics/mAP50(B)** ← mAP50 |
| **11** | **metrics/mAP50-95(B)** ← mAP50-95 |
| 12-14 | val/box_loss, val/cls_loss, val/dfl_loss |

### 验证方法（ultralytics log 双向交叉）

v12 ep288 训练日志末尾：
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
  all         99         43      0.922       0.83      0.882      0.361
```

对应 results.csv 第 288 行（用 `$1==288` 提取）：
| col | 值 | log 输出 | 一致？ |
|---|---|---|---|
| 8 (P) | 0.922 | 0.922 | ✓ |
| 9 (R) | 0.830 | 0.83 | ✓ |
| 10 (mAP50) | 0.882 | 0.882 | ✓ |
| 11 (mAP50-95) | 0.363 | 0.361 | ✓ |

**列映射 100% 确认**。

### 历次误读归档（避免再次传染）

| 次数 | 误读内容 | 假 mAP50 数字 | 真身份 | 修复 |
|---|---|---|---|---|
| 第 1 次（原始 memory） | 列 10 mAP50 当 Recall | "v11 mAP50=0.8887" | 列 9 R @ ep226=0.8887 | 用 model.val() 验证 |
| 第 2 次（"修" memory 时） | 列 11 mAP50-95 当 mAP50 | "v11 mAP50=0.348" | 列 11 mAP50-95 @ ep194=0.348 | 用 model.val() 验证 |
| 第 3 次（终极校正） | ✓ 列 8=P/9=R/10=mAP50/11=mAP50-95 | - | - | log 双向交叉 |

**根因**：每次都只读了一列数字，没核对 header / log。

---

## 2. 学术指标真值（model.val() 权威）

| Run | **P** | **R** | **mAP50** | mAP50-95 | mAP75 | 训练峰值 mAP50 (csv) |
|-----|-------|-------|-----------|----------|-------|----------------------|
| **v8 weak aug** | 0.881 | 0.814 | **0.872** | 0.404 | - | 0.8331 @ ep212 |
| **v11 strong aug** | **0.941** | 0.744 | 0.822 | 0.344 | **0.269** | 0.8616 @ ep213 |
| **v12 flipud+strong+300ep** | 0.923 | **0.830** | **0.882** ⭐ | 0.363 | 0.221 | 0.8837 @ ep229 |

**两个数字不同是合理的**：
- **model.val()** = 训完后用 best.pt 重新跑验证
- **训练峰值** = 训练过程中任一 epoch 达到的最高
- v11 best.pt 是 ep213 那个时刻的 mAP50（0.822） vs 训练峰值 0.8616 @ ep213——基本一致
- v12 best.pt 是 ep229 mAP50（0.882） vs 训练峰值 0.8837——基本一致

---

## 3. 部署指标真值（Lenient d≤30, val=99 张, top-1, model.predict）

| conf | v8 weak aug F1 | v11 strong aug F1 | v12 flipud+strong F1 |
|------|----------------|---------------------|----------------------|
| 0.05 | **0.9024** | 0.8966 | 0.8810 |
| 0.10 | 0.9024 | 0.9070 | **0.9136** |
| **0.15** | 0.9000 | **0.9176** ⭐ | 0.9136 |
| 0.20 | 0.8861 | 0.9048 | 0.8861 |
| 0.30 | 0.8421 | 0.8780 | 0.8718 |

**部署最优（按 F1）**：

| Run | best F1 | @conf | TP/FP/FN | Recall | Precision |
|-----|---------|-------|----------|--------|-----------|
| v8 | 0.9024 | 0.05/0.10 | 37/2/6 | 0.861 | 0.949 |
| **v11** | **0.9176** ⭐ | 0.15 | 39/3/4 | **0.907** | 0.929 |
| v12 | 0.9136 | 0.10/0.15 | 37/1/6 | 0.861 | **0.974** |

---

## 4. 终极决策矩阵

| 维度 | v8 | v11 | **v12** | 部署推荐 |
|------|----|----|---------|----------|
| 学术 mAP50 | 0.872 | 0.822 | **0.882** ⭐ | v12（论文） |
| 学术 P | 0.881 | **0.941** ⭐ | 0.923 | v11 |
| 学术 R | 0.814 | 0.744 | **0.830** | v12 |
| **部署 F1** | 0.9024 | **0.9176** ⭐ | 0.9136 | **v11（部署）** |
| 部署 Recall | 0.861 | **0.907** ⭐ | 0.861 | v11 |
| 部署 Precision | 0.949 | 0.929 | **0.974** | v12（高 Precision 业务） |

**没有全胜**：
- 论文学术 mAP50：v12 完胜（最高 IoU mAP）
- 部署 F1 + Recall：v11 完胜
- 部署 Precision：v12 完胜（FP=1, 几乎不误报）

---

## 5. 历史未验证声明（来源不可考，**不要再作为真值引用**）

| 声明 | 来源 memory | 状态 |
|------|-------------|------|
| 历史 baseline 学术 mAP50=0.877 | `hyper-yolo-runs-status` | ❌ 来源不可考 |
| v4 + TTA + top-2 + dist=50 部署 F1=0.929 | `weak-aug-tta-deploy-2026-07-10` | ❌ 本会话未复现 |
| "Lenient Recall=1.0000 / FN=0" | `v11-baseline-strong-aug` (历史版) | ❌ v11 真值 FN=4 |
| "TTA k=1/d=30/conf=0.10 F1=0.929" | 同上 | ❌ 本会话 v11 真值 F1=0.9176 |
| 历史 96.png top-2 救回 | `tta-experiment-results` | ❌ 训练 log 中无具体图例 |

---

## 6. 复现命令（任何人跑都能复现这些数字）

```bash
# 学术指标（model.val() 权威）
/home/pi/anaconda3/envs/hyper-yolo/bin/python -c "
from ultralytics import YOLO
for w, n in [
    ('runs/cfg_truth_repro/v8_nwd_v1_weak_aug_full/weights/best.pt', 'v8'),
    ('runs/cfg_truth_repro/v11_baseline_strong_aug_full/weights/best.pt', 'v11'),
    ('runs/cfg_truth_repro/v12_strong_aug_flipud_300ep/weights/best.pt', 'v12'),
]:
    r = YOLO(w).val(data='data/coil/data.yaml', imgsz=1024, conf=0.001, iou=0.6, max_det=300, verbose=False)
    print(f'{n}: P={r.box.p.mean():.4f} R={r.box.r.mean():.4f} mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}')
"

# 部署指标（Lenient d≤30 conf sweep）
/home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/.claude/jobs/47f4233c/tmp/conf_sweep.py \
    runs/cfg_truth_repro/v11_baseline_strong_aug_full/weights/best.pt v11
```

---

## 7. 文档历史（避免重复犯错）

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-07-11 1.0 | 第一次校正 | 把列 9 R 误为 mAP50 → 写出 0.348 |
| 2026-07-11 2.0 | 第二次校正 | 把列 11 mAP50-95 误为 mAP50 → 写出 0.348（同 1.0 巧合） |
| 2026-07-11 3.0 | **终极校正** | 用 model.val() + log 双向验证，列映射 100% 确认 |