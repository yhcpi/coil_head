# YOLO26 + 强 aug 训练进度 (250 epoch)

**模型**: YOLO26n (2.5M params)  
**配置**: imgsz=1024, batch=16, end2end=True, reg_max=1, 强 aug (degrees=10/flipud=0.5/copy_paste=0.2)  
**基线对比**: V18.3 + TTA-builtin F1=0.9286 / V12 + TTA F1=0.9136

## 每 50 epoch 进度

| epoch | mAP50 | mAP50-95 | P | R | train/box | val/box | train/dfl | time (s) | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.6954 | 0.2493 | 0.8339 | 0.5838 | 2.121 | 2.156 | 0.0029 | 657 | ✅ epoch50.pt (20.8MB) |
| 100 | 0.7283 | 0.2812 | 0.7233 | 0.6744 | 1.977 | 2.055 | 0.0029 | 1295 | ⏳ (save_period=50) |
| 150 | 0.8104 | 0.3406 | 0.9079 | 0.7209 | 1.907 | 2.079 | 0.0026 | 1958 | ⏳ (save_period=50) |
| 200 | 0.8118 | 0.3505 | 0.8223 | 0.7907 | 1.573 | 1.967 | 0.0022 | 2620 | ⏳ (save_period=50) |

## 训练终止 + 最终评估

**EarlyStopping at epoch 207** (patience=30, no improvement since epoch 177)
**Total time**: 0.753 hours (~45 min)
**Best.pt = epoch 177** (fitness best)

### 学术指标 (model.val() 真值)

| 指标 | YOLO26 best.pt | V18.3 epoch60 | V12 + flipud 300ep |
|---|---|---|---|
| mAP50 | **0.8619** ⭐ | 0.822 | 0.890 |
| mAP50-95 | 0.3728 | — | — |
| Precision | 0.8762 | — | — |
| Recall | 0.7907 | — | — |

### 部署指标 (Lenient-Match, top-1, dist_thresh=30)

| 配置 | F1 | R | P | TP/FP/FN |
|---|---|---|---|---|
| **YOLO26 best.pt** @ conf=0.30 (无 TTA) | **0.8706** | 0.8605 | 0.8810 | 37/5/6 |
| YOLO26 best.pt @ conf=0.25 | 0.8539 | 0.8837 | 0.8261 | 38/8/5 |
| YOLO26 best.pt @ conf=0.20 | 0.8172 | 0.8837 | 0.7600 | 38/12/5 |
| **V18.3 epoch60 + TTA-builtin** @ conf=0.15 | **0.9286** ⭐ | 0.9070 | 0.9512 | 39/2/4 |
| V12 + TTA-builtin @ conf=0.15 | 0.9136 | 0.8605 | 0.9737 | 37/1/6 |

### 结论

- **学术 mAP50 YOLO26 > V18.3** (0.862 vs 0.822, +4.0pp) — YOLO26 end2end + reg_max=1 训练质量更高
- **部署 F1 YOLO26 < V18.3** (0.8706 vs 0.9286, -5.8pp) — YOLO26 FP 较多（35 FP @ conf=0.05），无 hard neg 训练
- **V18.3 hard neg + 弱 aug 仍是部署最优**，但 YOLO26 学术性能已超越 V18.3

### YOLO26 关键发现

1. **收敛极快**: 50 epoch mAP50=0.695 已超过 v8 NWD 250 epoch 训练中的多个中间点（v8 NWD 弱 aug 250 ep 0.854）
2. **end2end + reg_max=1 训练稳定**: 207 epoch early stop, loss 无发散
3. **强 aug 可用**: 之前 V18.2 强 aug 漂白副本失败，YOLO26 强 aug 无漂白问题（end2end + L1 reg 更鲁棒）
4. **下次突破方向**: YOLO26 + hard neg 训练（移植 V18.3 的 33 张副本 + 弱 aug）
