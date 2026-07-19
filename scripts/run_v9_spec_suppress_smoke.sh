#!/bin/bash
# v9 Specular Highlight Suppression smoke (10 epoch 验证模块生效 + loss 不爆炸)
# 目的: 验证 SpecSuppress 模块被正确插入, recon_loss 有梯度, 不破坏训练
# 数据: 复用 v8 NWD 配置 (data/coil/data.yaml)
#
# 预期:
#   - 训练能跑通 (无 shape/维度错误)
#   - args.yaml 中 spec_suppress=true, spec_recon_weight=0.1
#   - results.csv 有 recon_loss (从 v8DetectionLoss 输出)
#   - 不会破坏 box/cls/dfl 正常量级
#
# 用法: bash scripts/run_v9_spec_suppress_smoke.sh
set -u
cd /home/pi/projects/hyperyolo

# 先把 patches 同步到 repos/Hyper-YOLO/ultralytics/ (跟 PATCHES.md 流程一致)
cp src/hyper_yolo_patches/ultralytics/nn/modules/spec_suppress.py \
   repos/Hyper-YOLO/ultralytics/nn/modules/spec_suppress.py
cp src/hyper_yolo_patches/ultralytics/nn/modules/__init__.py \
   repos/Hyper-YOLO/ultralytics/nn/modules/__init__.py
cp src/hyper_yolo_patches/ultralytics/utils/loss.py \
   repos/Hyper-YOLO/ultralytics/utils/loss.py
cp src/hyper_yolo_patches/ultralytics/models/yolo/detect/train.py \
   repos/Hyper-YOLO/ultralytics/models/yolo/detect/train.py

NAME="v9_spec_suppress_smoke"

# 启动前查重 (避免 GPU 共享 + results.csv 覆盖)
echo "===== 现有 ultralytics 训练进程 ====="
ps -ef | grep "ultralytics.models.yolo.detect.train" | grep -v grep || echo "(无)"

# smoke 10 epoch, batch=8 (稳), imgsz=1024 (复用 v8 配置)
nohup /home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=repos/Hyper-YOLO/hyper-yolon.pt \
  data=data/coil/data.yaml \
  epochs=10 patience=0 batch=8 imgsz=1024 \
  save=False val_period=1 start_val_epoch=0 save_period=-1 \
  cache=False device=0 workers=0 \
  project=runs/v9_smoke name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  cfg=data/coil/hyp_v9_spec_suppress.yaml \
  > /tmp/v9_spec_smoke.log 2>&1 &

PID=$!
echo "训练 PID: ${PID}, 日志: /tmp/v9_spec_smoke.log"
echo "等 5 秒确认启动..."
sleep 5
if ps -p ${PID} > /dev/null; then
  echo "✓ 进程存活"
else
  echo "✗ 启动失败, 日志尾部:"
  tail -30 /tmp/v9_spec_smoke.log
  exit 1
fi

# 启动后立即验证 args.yaml 字段
echo "===== args.yaml 关键字段 ====="
if [ -f runs/v9_smoke/${NAME}/args.yaml ]; then
  grep -E "^(spec_suppress|spec_recon_weight|nwd|box|cls|cfg):" \
    runs/v9_smoke/${NAME}/args.yaml
else
  echo "(args.yaml 尚未生成, 等 30s)"
  sleep 30
  grep -E "^(spec_suppress|spec_recon_weight|nwd|box|cls|cfg):" \
    runs/v9_smoke/${NAME}/args.yaml
fi

# 等 1 分钟后看 1 个 epoch 的 loss
echo "等 60s 收集第 1 个 epoch..."
sleep 60
echo "===== results.csv 第 1-3 行 ====="
head -3 runs/v9_smoke/${NAME}/results.csv 2>/dev/null || echo "(尚未写入)"

# 等到 epoch 10 完成
echo "等训练完成 (PID=${PID})..."
wait ${PID} 2>/dev/null

echo "===== results.csv 最终 5 行 ====="
tail -5 runs/v9_smoke/${NAME}/results.csv

echo "===== train log 关键摘要 ====="
grep -E "spec_suppress|recon_loss|epoch.*GPU_mem" /tmp/v9_spec_smoke.log | head -20
