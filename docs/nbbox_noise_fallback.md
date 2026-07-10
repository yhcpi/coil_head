# NBBoxNoise 失败回退预案

> 创建：2026-07-09（NBBoxNoise smoke 调试过程中）
> 适用：如果 NBBoxNoise 250 epoch 跑失败、或 box_loss 始终远高于基线 4.24，按此预案单文件回退。

## 触发条件（任一即回退）

1. **box_loss 不收敛**：250 epoch 后 val/box_loss 仍 > 10（基线 03 ≈ 3.0）
2. **mAP 退化**：val/mAP50 < 0.85（基线 03 学术 mAP50 = 0.877）
3. **cls_loss spike**：epoch 1 cls_loss > 30（基线 22.96）持续 5 epoch 不收敛
4. **PIL ValueError 频发**：plot_images 线程异常 > 100 次/epoch

## 单文件回退（4 个文件，互不依赖，可独立 revert）

### 1. 删除 NBBoxNoise 模块本身

```bash
rm /home/pi/projects/hyperyolo/repos/Hyper-YOLO/ultralytics/data/nbbox_noise.py
```

### 2. augment.py 顶部删除 import（一行）

文件：`repos/Hyper-YOLO/ultralytics/data/augment.py`
查找：`from .nbbox_noise import NBBoxNoise`
动作：删除该行

### 3. augment.py v8_transforms() 函数删除注册块

文件：`repos/Hyper-YOLO/ultralytics/data/augment.py`
查找（4 行）：

```python
if getattr(self, 'nbbox', False):
    transform.append(NBBoxNoise(
        p=getattr(self, 'nbbox_p', 0.5),
        scale_min=getattr(self, 'nbbox_scale_min', 0.5),
        scale_max=getattr(self, 'nbbox_scale_max', 1.5),
        shift=getattr(self, 'nbbox_shift', 0.1)))
```

动作：删除以上 5 行

### 4. hyp_v5_nwd_only.yaml 删除 nbbox 段

文件：`data/coil/hyp_v5_nwd_only.yaml`
查找（8 行含注释）：

```yaml
# === NBBox: NBBOX-style Loose-Box Noise Aug (Kim et al. IEEE GRSL 2025)
# 把 loose GT bbox 当 augmentation 加噪，让模型学「反去噪」以精准定位 20x20 tip
# 默认开启 (true)；禁用回退 = 设 nbbox: false 即可（augment.py 单变量注册）
nbbox: true                  # 启用 NBBoxNoise transform
nbbox_p: 0.5                 # 触发概率（NBBOX 论文推荐 0.5，过高易破坏 tip 形态）
nbbox_scale_min: 0.5         # bbox 长宽缩放下界（约 50%）
nbbox_scale_max: 1.5         # bbox 长宽缩放上界（约 150%）
nbbox_shift: 0.1             # 中心平移占新 bbox 大小的比例（约 ±10%）
```

动作：删除以上 8 行

### 5. default.yaml 删除 nbbox* 字段

文件：`repos/Hyper-YOLO/ultralytics/cfg/default.yaml`
查找（5 行）：

```yaml
nbbox: False  # (bool) NBBoxNoise (NBBOX-style loose-box aug, Kim et al. IEEE GRSL 2025) switch
nbbox_p: 0.5  # (float) NBBoxNoise trigger probability
nbbox_scale_min: 0.5  # (float) NBBoxNoise bbox 长宽缩放下界
nbbox_scale_max: 1.5  # (float) NBBoxNoise bbox 长宽缩放上界
nbbox_shift: 0.1  # (float) NBBoxNoise bbox 中心平移占新 bbox 大小的比例 (U(-shift, shift))
```

动作：删除以上 5 行

## 回退后状态

- 不破坏 03 baseline（NWD-only v1）任何能力，因为 03 baseline 训练时这些字段都不存在
- v6_v1/v6_v2/v6_v3/v6_heatmap 训练时也都不存在（coil_loss_ablation 全套无 NBBox hook）
- 仍保留：BBoxRandomShrink、RandomScaleRect、Coverage、NWD（其他 v4-v6 创新点）
- 训练命令完全兼容：把 `cfg=data/coil/hyp_v5_nwd_only.yaml` 改回原 03 baseline 命令即可

## 验证步骤

```bash
# 1. 确认代码无 nbbox 残留
grep -rn "nbbox\|NBBoxNoise" /home/pi/projects/hyperyolo/repos/Hyper-YOLO/ultralytics/ /home/pi/projects/hyperyolo/data/coil/

# 2. smoke 30 epoch 复现 03 baseline box_loss
bash /home/pi/projects/hyperyolo/scripts/run_nbbox_v1.sh smoke  # 若还在
# 或直接复跑 03 baseline 命令

# 3. 期望 box_loss epoch 1 ≈ 4.24，cls_loss ≈ 22.96
```

## git 视角

如果用了 git，回退更简单（按需要）：

```bash
git diff --name-only HEAD  # 应该只看到 augment.py / nbbox_noise.py / hyp*.yaml / default.yaml
git checkout HEAD -- augment.py default.yaml
rm -f ultralytics/data/nbbox_noise.py
```

但目前项目 root 无 `.git`，所以用上面 5 步文件级回退。
