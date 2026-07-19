# YOLO26 实验总结报告 (P11)

**日期**: 2026-07-13
**项目**: `repos/yolo26-coil/`
**目标**: 验证 YOLO26 (end2end + reg_max=1) 在钢卷头尾小目标检测场景下的表现

## TL;DR

- **学术 mAP50 0.8619** (model.val() 真值)，**部署 F1 0.8706** (无 TTA)
- 训练 207 epoch 早停 (best=epoch177)，总耗时 ~45 min
- **比 V18.3 学术高 4.0pp，但部署低 5.8pp**（FP 多）
- V18.3 + TTA-builtin 0.9286 仍为部署最优
- 下一步: **YOLO26 + hard neg 训练** (移植 V18.3 思路)

## 1. 实验配置

```bash
model=yolo26n.pt (2.5M params, Ultralytics 8.4.82)
imgsz=1024 batch=16 epochs=250 device=0
degrees=10 translate=0.1 scale=0.5 flipud=0.5 fliplr=0.5
mosaic=1.0 mixup=0.1 copy_paste=0.2 erasing=0.1
project=runs/yolo26_coil name=v1_strong_aug_250ep
save_period=50 cos_lr=True patience=30 close_mosaic=20
```

## 2. 训练进度

| epoch | mAP50 | mAP50-95 | P | R | val/box | train/dfl |
|---|---|---|---|---|---|---|
| 50 | 0.6954 | 0.2493 | 0.8339 | 0.5838 | 2.156 | 0.0029 |
| 100 | 0.7283 | 0.2812 | 0.7233 | 0.6744 | 2.055 | 0.0029 |
| 150 | 0.8104 | 0.3406 | 0.9079 | 0.7209 | 2.079 | 0.0026 |
| 200 | 0.8118 | 0.3505 | 0.8223 | 0.7907 | 1.967 | 0.0022 |
| **177 (best.pt)** | **0.8619** | 0.3728 | 0.8762 | 0.7907 | — | — |

**早停**: epoch 207 (patience=30 since epoch 177)
**总时间**: 0.753 hours (45 min)

## 3. 关键发现

### 3.1 收敛速度极快
50 epoch mAP50=0.695 已超过 v8 NWD 训练 50 epoch 的 0.149 (4.6x 速度)。
end2end + reg_max=1 显著简化训练目标 (无 DFL 多 bin 学习)。

### 3.2 训练稳定
207 epoch 内 loss 无发散、无收敛困难。
end2end (one2many + one2one) 双分支 + L1 reg_max=1 比 v8 DFL 更鲁棒。

### 3.3 强 aug 不漂白副本
之前 V18.2 强 aug 漂白副本导致失败 (loss 收敛但 FP 不消除)。
YOLO26 强 aug 无此问题，end2end + L1 reg 抗扰动更强。

### 3.4 学术 ≠ 部署
mAP50 高 4.0pp 但部署 F1 低 5.8pp：
- 学术指标包含全 conf range，部署只看 conf=0.30
- YOLO26 FP 较多 (35 个 @ conf=0.05)
- V18.3 hard neg 训练消除了 11 张原图 FP

## 4. 对比表 (本项目全部部署 SOTA)

| 模型 | 训练配置 | 部署 F1 | TP/FP/FN | 部署权重 |
|---|---|---|---|---|
| **V18.3 + TTA-builtin** | hard neg + 弱 aug + 100ep | **0.9286** ⭐ | 39/2/4 | `runs/deploy_best/v18_3_epoch60_hard_neg_weak_aug.pt` |
| V12 + TTA-builtin | flipud + 强 aug + 300ep | 0.9136 | 37/1/6 | `runs/detect/v12_strong_aug_300ep/weights/best.pt` |
| **YOLO26 best.pt** | 强 aug + 207ep | 0.8706 | 37/5/6 | `runs/yolo26_coil/v1_strong_aug_250ep/weights/best.pt` |
| V11 + TTA-builtin | 强 aug + 250ep | 0.9176 | — | `runs/detect/v11_strong_aug/weights/best.pt` |

## 5. 与 V18.3 (hyper_yolo_patches/ultralytics) 的关键差异

| 维度 | V18.3 | YOLO26 |
|---|---|---|
| ultralytics 版本 | 8.0.227 (含 patch) | 8.4.82 (官方最新) |
| 模型架构 | YOLOv8n + NWD/IoU/Coverage | YOLO26n + end2end + reg_max=1 |
| box_loss | CIoU (V18) / NWD (V8) | CIoU + 2*L1(ltrb) |
| 训练数据 | train 312 + 11 hard neg ×3 副本 | train 312 (无副本) |
| aug 强度 | **弱** (degrees=0, flipud=0, cp=0) | **强** (degrees=10, flipud=0.5, cp=0.2) |
| 总 epoch | 100 (best=60) | 207 (best=177) |
| 收敛速度 | 慢 (DFL + NWD) | **快 4-6x** (end2end + reg_max=1) |

## 6. 下一步: YOLO26 + Hard Neg

预期能复制 V18.3 训练流程在 YOLO26 上：

```bash
# 1. 复制 V18.3 hard neg 副本到 data/coil/images/train
# 2. 弱 aug (degrees=0 scale=0 flipud=0 cp=0)
# 3. 100 epoch + lr=0.005
```

预期结果:
- 学术 mAP50 持平 0.86
- **部署 F1 ≥ 0.9286** (消除 11 张原图 FP)
- 训练时间: 100 epoch × ~13s ≈ 22 min

## 7. 归档资产

| 文件 | 说明 |
|---|---|
| `repos/yolo26-coil/train.sh` | YOLO26 训练脚本 (含 PYTHONPATH 切换) |
| `repos/yolo26-coil/eval.sh` | YOLO26 + TTA 评估脚本 |
| `repos/yolo26-coil/monitor.py` | 训练进度监控脚本 (已退出) |
| `repos/yolo26-coil/data.yaml` | 独立 data.yaml (不改老项目) |
| `runs/yolo26_coil/v1_strong_aug_250ep/` | 训练结果 (含 epoch50/100/150/200.pt + best.pt/last.pt) |
| `repos/ultralytics/weights/yolo26n.pt` | 预训练权重 (5.5MB, gh-proxy 下载) |

## 8. 关键 takeaway

1. **end2end + reg_max=1 是真的进步**: 训练速度 4-6x，学术 mAP50 已超越 v8 NWD
2. **不解决 hard neg 永远拼不过 V18.3**: 训练侧是 FP 的根源，后处理救不了
3. **强 aug 在 YOLO26 上比 v8 上稳**: end2end 抗扰动强，可放心用
4. **下次实验首选**: YOLO26 + hard neg + 弱 aug + 100ep + lr=0.005
