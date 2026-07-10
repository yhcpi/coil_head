#!/bin/bash
# C+D 项训练进度追踪（任何时候跑都能看）
# 用法：bash scripts/c_d_status.sh [09_bayes_prior|10_paaug_motion|...]
cd /home/pi/projects/hyperyolo
NAME="${1:-09_bayes_prior}"
DIR="runs/coil_loss_ablation/$NAME"

echo "=========================================="
echo "[$NAME] 训练进度 ($(date '+%H:%M:%S'))"
echo "=========================================="

# 1. 进程
echo ""
echo "[1] 进程状态"
PIDS=$(pgrep -f "name=$NAME" | tr '\n' ',' | sed 's/,$//')
if [ -n "$PIDS" ]; then
  ps -p "$PIDS" -o pid,etime,pcpu,pmem,stat 2>&1 | head -10
else
  echo "  ⚠ 无 detect.train 进程在跑 $NAME"
fi

# 2. results.csv 当前 epoch
echo ""
echo "[2] 已完成 epoch（results.csv）"
if [ -f "$DIR/results.csv" ]; then
  echo "  $(tail -n +2 $DIR/results.csv | wc -l) 个 epoch 已记录"
  echo ""
  tail -3 $DIR/results.csv | awk -F',' 'NR==1 {print "  epoch box_loss cls_loss dfl_loss   P    R   mAP50  mAP50-95"} NR>1 {printf "  %5s  %7s  %7s  %7s  %5s %5s %6s %8s\n", $1, $2, $3, $4, $8, $9, $10, $11}'
else
  echo "  ⚠ $DIR/results.csv 不存在"
fi

# 3. best.pt
echo ""
echo "[3] best.pt"
if [ -f "$DIR/weights/best.pt" ]; then
  echo "  ✓ $(ls -lh $DIR/weights/best.pt | awk '{print $5}')  $(stat -c %y $DIR/weights/best.pt | cut -d. -f1)"
else
  echo "  ⚠ 未生成"
fi

# 4. GPU
echo ""
echo "[4] GPU 占用"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null | head -2

# 5. 日志尾部（仅 loss 行）
echo ""
echo "[5] 训练日志（最后 5 个 epoch 行）"
grep -oE '[0-9]+/250.*box_loss.*' "$DIR.log" 2>/dev/null | tail -3 | head -3