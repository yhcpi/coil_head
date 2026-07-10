# 项目原始需求

来源：项目启动时袁昊宸手写笔记（`requirements_orig.txt`）

## 步骤 1：数据策略修正

移 27 张 FN 进 train，但是不要从现有 train 随机抽 27 张做新 val 了。

## 步骤 2：加一个轻量增强 + Coverage Loss

**bbox_random_shrink**：在数据加载时把每个 GT bbox 随机缩放到 0.8~1.2 倍，生成额外的伪 GT。

**Coverage Loss**：保留原有的 IoU 损失等不变，添加覆盖度损失作为一个新的损失参考项，给予一定权重。即使不理想，也可以删除，不影响已有的源码。

## 步骤 3（可选）：训练策略微调

- 不沿用现有 best，从预训练模型从头开始
- 新增 `--multi-scale`，但是得保持图片长宽比不变，无法保持就不加
- 微调 `hyp_aug.yaml`：
  - `copy_paste: 0.2`
  - 降 bbox loss 权重，升分类 loss 权重
  - `mosaic` 如果会改变训练图片长宽比就不加，不会改变才能加
- 每个训练结果文件夹里都要保存一份文件，详细说明本次训练的配置

## 实施情况

| 步骤 | 状态 | 落地位置 |
|------|------|----------|
| 步骤 1 数据策略 | ✅ | `scripts/split_dataset.py`（训练集 312 张，验证集 43 张） |
| 步骤 2 bbox_random_shrink | ✅ | `src/hyper_yolo_patches/ultralytics/data/augment.py:1173-1240`（class `BBoxRandomShrink`） |
| 步骤 2 Coverage Loss | ✅（但部署未启用）| `src/hyper_yolo_patches/ultralytics/utils/loss.py:94-105`（coverage_loss 函数） |
| 步骤 3 multi-scale | ⚠️ 撤回 | 详见 memory: multi-scale-fix-2026-07-10.md（RandomAffine scale=0.5 已覆盖） |
| 步骤 3 copy_paste=0.2 | ❌ 撤回 | 详见 memory: robust-aug-vs-pure-cli-2026-07-10.md（强 aug 反而降低 8.6pp mAP） |
| 步骤 3 配置记录 | ✅ | 每个 run 目录都有 `CLI_PARAMS.md` 或 `args.yaml` |