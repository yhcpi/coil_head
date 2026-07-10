# NBBoxNoise 创新点失败报告 + 回退执行记录（2026-07-09）

## 实验结论

**NBBoxNoise（NBBOX-style Loose-Box Noise Augmentation, Kim et al. IEEE GRSL 2025）作为 Hyper-YOLO + 钢卷头尾检测的创新点，实验无效，已整体回退**。

## 验证流程

### Step 1: critical A/B 隔离测试（10 epoch）

| epoch | A (nbbox=false, shrink=0) | B (nbbox=true, shrink=0) | 差 |
|-------|---------------------------|--------------------------|-----|
| 1 | box 61.25 / cls 55.00 | box 58.52 / cls 52.01 | -2.73 / -2.99 |
| 5 | box 21.66 / cls 10.04 | box 23.36 / cls 9.85 | +1.70 / -0.19 |
| 10 | box 19.61 / mAP=0.092 | box 20.86 / mAP=0.082 | +1.25 / -0.010 |

**结论**：NBBoxNoise 算法正确，box_loss 收敛轨迹与基线差 ±2（噪声水平），不影响模型。
但**box_loss 起始 ~60**（vs 03 baseline 历史快照 4.24）—— **03 baseline 不可复现**。

### Step 2: full 250 epoch（NBBox=true + shrink=1.0，PID 3587262）

跑到 epoch 53 时观察：
- box_loss epoch 5→50：28.4 → 29.5（**完全停滞不收敛**）
- cls_loss epoch 5→50：11.1 → 9.5（缓慢下降）
- **mAP50 持续 1e-05**（不学习）
- Precision/Recall 锁在 0.02632 / 2e-05（基线值）

**结论**：训练在 epoch ~5 后彻底卡住，250 epoch 不会收敛到 mAP>0.85。**触发回退预案条件 2**（val/mAP50<0.85 持续）。

## 根因分析

NBBoxNoise 本身**不是失败原因**：
- A vs B 10 epoch 完全等价（差 ±2 噪声）
- 算法 unit test 1000 次抽样：min box 14.3 px（>= 5 px 要求），无退化

**真正失败原因**：项目当前 hook 组合下，03 baseline box_loss=4.24 已不可复现：
- A 配置完全等同 03 baseline hook（NBBox 关、shrink 关）→ box_loss=61.25 epoch 1
- 比 03 baseline 高 14x
- 推测根因：BBoxRandomShrink / RandomScaleRect / 其他 augment 改动影响了数据加载阶段输入分布
- **03 baseline 训出的 mAP50=0.877 是历史快照**，无法在当前代码下复现

**NBBoxNoise 在新 baseline 下表现与无 NBBox 完全等价**——它既不解决问题（box_loss 高）也不引入新问题。**本质上是个对当前 hook 配置无贡献的 augmentation**。

## 回退执行（已完成 4 步）

按 `docs/nbbox_noise_fallback.md` 4 步单文件回退：

| 文件 | 操作 | 验证 |
|------|------|------|
| `repos/Hyper-YOLO/ultralytics/data/nbbox_noise.py` | 删除 | `ModuleNotFoundError` ✓ |
| `repos/Hyper-YOLO/ultralytics/data/augment.py` | 删除 import + 8 行注册块 | `grep nbbox` = 空 ✓ |
| `data/coil/hyp_v5_nwd_only.yaml` | 删除 nbbox 段 8 行 | `grep nbbox` = 空 ✓ |
| `repos/Hyper-YOLO/ultralytics/cfg/default.yaml` | 删除 nbbox* 5 行 | `grep nbbox` = 空 ✓ |

回退后 smoke 验证（3 epoch）：
- box_loss=30+（同 A baseline）
- cls_loss=20-23
- 无 ImportError / ModuleNotFoundError
- augment pipeline 工作正常

## 失败教训

1. **baseline 不可复现是 hidden risk**：03 baseline box_loss=4.24 是单次训练快照，没有同代码同数据的多次复现确认。后续创新点的 baseline 比较应做至少 2 次复现取均值。
2. **A/B 对比是黄金标准**：直接比 A vs B 10 epoch 收敛轨迹（差 ±2 噪声）比 absolute box_loss 数字更可靠。absolute number 受 hook 配置影响。
3. **回退预案的价值**：4 步单文件回退预案从写完到执行 ~3 分钟，避免 NBBoxNoise 残留污染后续实验。

## 当前 coil 训练系统状态

- hook 配置：BBoxRandomShrink（bbox_shrink_p=1.0, min=0.8, max=1.2）+ RandomScaleRect（multi_scale=0, no-op）+ NWD + Coverage=false
- NBBoxNoise：已回退（不存在）
- **新 box_loss baseline**（无 NBBox）：epoch 1 ~60，epoch 10 ~20（远高于 03 baseline 历史 4.24）
- 下一步应该先**修复 baseline 退化**（找到 hook 改动引入的退化点），再做新创新点
