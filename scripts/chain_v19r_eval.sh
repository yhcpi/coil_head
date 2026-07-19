#!/usr/bin/env bash
# 2026-07-16 v19 resume 完跑自动评估 + 归档
# v19r 部署 F1 > v18.3 0.9286 → save_repro_config 自动归档 v19
set -uo pipefail
cd /home/pi/projects/hyperyolo

echo "===== 等 v19r (PID 29769) 完成 ====="
while ps -p 29769 > /dev/null 2>&1; do
  sleep 180
  ELAPSED=$(ps -p 29769 -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') v19r 仍在跑 [$ELAPSED]"
done
echo "✅ v19r 完成 at $(date '+%H:%M:%S')"

echo "===== 跑 v19r vs v18.3 vs v19 三方评估 ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/projects/hyperyolo/scripts/eval_v19_vs_v18_3.py

echo "🎉 v19r 评估完成 at $(date '+%H:%M:%S')"