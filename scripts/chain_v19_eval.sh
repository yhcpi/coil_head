#!/usr/bin/env bash
# 2026-07-16 H04:00 v19 完跑自动评估 + 归档
# v19 部署 F1 > v18.3 0.9286 → save_repro_config 自动归档 v19_aca
set -uo pipefail
cd /home/pi/projects/hyperyolo

echo "===== 等 v19 (PID 3765) 完成 ====="
while ps -p 3765 > /dev/null 2>&1; do
  sleep 180
  ELAPSED=$(ps -p 3765 -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') v19 仍在跑 [$ELAPSED]"
done
echo "✅ v19 完成 at $(date '+%H:%M:%S')"

echo "===== 跑 v19 vs v18.3 vs baseline 三方评估 ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/projects/hyperyolo/scripts/eval_v19_vs_v18_3.py

echo ""
echo "===== 尝试 save_repro_config 自动归档 ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/projects/hyperyolo/scripts/save_repro_config.py

echo "🎉 v19 全链完成 at $(date '+%H:%M:%S')"