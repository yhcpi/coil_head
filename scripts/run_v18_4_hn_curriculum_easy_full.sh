#!/bin/bash
# V18.4 HN-Curriculum 实验 A：剔除 hn*_493 副本（验证 493 是否为毒样本）
# 假设：30 张不含 493 的副本可能比 V18.3 全部 33 张副本更适合作为稳定 baseline
# 数据：train 545 张 + 30 张 hn* 副本 (10 hard neg × 3) = 575 张
# 模型：v12 best.pt
# 配置：弱 aug + lr=0.005 + 100 epoch + patience=30
# 训练时间预计 ~28 分钟
# 评估：epoch60.pt 等多 epoch 评估 deployment F1 + 11 张原图 hard neg FP 消除率
# 训练完成后会自动恢复 hn*_493 文件
set -u
cd /home/pi/projects/hyperyolo

NAME="v18_4_hn_curriculum_no493_full"

echo "===== 检查现有 ultralytics 进程 ====="
ps -ef | grep -v grep | grep "ultralytics" | head -5 || echo "  (无现有进程)"

echo "===== 检查 hn*_493 副本是否已移出 ====="
ls data/coil/images/train/ | grep -E "hn[123]_493" && echo "WARN: 493 副本未移出!" && exit 1
echo "  已确认 493 副本已移出（剩余 30 张 hn* 副本）"

echo "===== 启动 V18.4 (剔除 493 副本) ====="
/home/pi/anaconda3/envs/hyper-yolo/bin/python -m ultralytics.models.yolo.detect.train \
  task=detect mode=train \
  model=runs/cfg_truth_repro/v12_strong_aug_flipud_300ep/weights/best.pt \
  data=data/coil/data.yaml \
  epochs=100 patience=30 batch=16 imgsz=1024 \
  save=True save_period=10 val_period=1 start_val_epoch=0 \
  cache=False device=0 workers=2 \
  project=runs/cfg_truth_repro name="${NAME}" exist_ok=True \
  pretrained=True optimizer=SGD verbose=True seed=0 deterministic=True \
  single_cls=False rect=True cos_lr=True close_mosaic=15 resume=False amp=True \
  fraction=1.0 \
  lr0=0.005 lrf=0.0005 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.05 \
  box=1.5 cls=0.5 dfl=1.5 label_smoothing=0.0 \
  nwd=true nwd_constant=12.0 \
  coverage=false coverage_weight=0.5 \
  degrees=0.0 translate=0.05 scale=0.0 flipud=0.0 fliplr=0.5 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  multi_scale=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  nbs=64 \
  2>&1 | tee /tmp/${NAME}.log

echo ""
echo "===== 恢复 hn*_493 副本 ====="
mv data/coil/v18_4_backup_493/hn1_493.png data/coil/images/train/
mv data/coil/v18_4_backup_493/hn2_493.png data/coil/images/train/
mv data/coil/v18_4_backup_493/hn3_493.png data/coil/images/train/
mv data/coil/v18_4_backup_493/hn1_493.txt data/coil/labels/train/
mv data/coil/v18_4_backup_493/hn2_493.txt data/coil/labels/train/
mv data/coil/v18_4_backup_493/hn3_493.txt data/coil/labels/train/
rmdir data/coil/v18_4_backup_493
echo "  hn*_493 副本已恢复（images 33 + labels 33）"

echo ""
echo "===== args.yaml 实际生效字段 ====="
test -f runs/cfg_truth_repro/${NAME}/args.yaml && \
  grep -E "^(model|epochs|batch|imgsz|lr0|lrf|warmup|box|cls|nwd|coverage|degrees|translate|scale|flipud|mosaic|copy_paste|multi_scale|pretrained|seed):" \
  runs/cfg_truth_repro/${NAME}/args.yaml

echo ""
echo "===== 训练结果 ====="
test -f runs/cfg_truth_repro/${NAME}/results.csv && \
  awk -F, 'NR>1{if($11+0>max){max=$11+0; line=$0}} END{print "Best mAP50: "max"\n@ "line}' runs/cfg_truth_repro/${NAME}/results.csv