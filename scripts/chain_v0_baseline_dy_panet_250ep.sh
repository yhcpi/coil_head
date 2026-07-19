#!/usr/bin/env bash
# 2026-07-15 H22:15 ulracode B chain: baseline -> DySample -> Coil-PANet sequential
# baseline 已经在跑 (PID 1224147)
# 等其完成 → 启动 DySample → 等其完成 → 启动 Coil-PANet
# 每个实验 250 ep, 总 ETA 4-5h (baseline 23:30 + dy 01:00 + panet 02:30)
set -uo pipefail
cd /home/pi/projects/hyperyolo

# 1. 等当前 baseline 完成
echo "===== 等 baseline (PID 1224147) 完成 ====="
BASELINE_PID=1224147
TARGET_NAME="v0_baseline_hyper_yolon_strong_aug_250ep"
while ps -p $BASELINE_PID > /dev/null 2>&1; do
  sleep 60
  ELAPSED=$(ps -p $BASELINE_PID -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') baseline 仍在跑 [$ELAPSED]"
done
echo "✅ baseline 完成 at $(date '+%H:%M:%S')"

# 检查 baseline 是否有效 (mAP50 > 0.5, 列 10 = metrics/mAP50(B), 旧 $8 = P 列错位)
CSV="runs/baseline/${TARGET_NAME}/results.csv"
if [ -f "$CSV" ]; then
  MAX_MAP=$(awk -F, 'NR>1 {if($10>m){m=$10}} END{printf "%.4f", m}' "$CSV")
  LAST_MAP=$(awk -F, 'NR>1 {m=$10} END{printf "%.4f", m}' "$CSV")
  echo "baseline max mAP50 = $MAX_MAP, 末 ep mAP50 = $LAST_MAP"
  if [ "$(echo "$MAX_MAP < 0.5" | bc)" = "1" ]; then
    echo "⚠️  baseline mAP50 < 0.5, DySample 仍继续但保留 baseline weight 备份"
  fi
fi

# 2. 启动 DySample
echo "===== 启动 DySample ====="
nohup bash scripts/run_dysample_250ep.sh > /tmp/dy.log 2>&1 &
DY_PID=$!
echo "DySample PID=$DY_PID"
while ps -p $DY_PID > /dev/null 2>&1; do
  sleep 60
  ELAPSED=$(ps -p $DY_PID -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') DySample 仍在跑 [$ELAPSED]"
done
echo "✅ DySample 完成 at $(date '+%H:%M:%S')"

CSV="runs/baseline/v0_dy_hyper_yolon_250ep/results.csv"
if [ -f "$CSV" ]; then
  MAX_MAP=$(awk -F, 'NR>1 {if($10>m){m=$10}} END{printf "%.4f", m}' "$CSV")
  LAST_MAP=$(awk -F, 'NR>1 {m=$10} END{printf "%.4f", m}' "$CSV")
  echo "DySample max mAP50 = $MAX_MAP, 末 ep mAP50 = $LAST_MAP"
fi

# 3. 启动 Coil-PANet (如果 script ready)
echo "===== 启动 Coil-PANet ====="
if [ ! -f scripts/run_coil_panet_250ep.sh ]; then
  echo "⚠️  scripts/run_coil_panet_250ep.sh 不存在, skip Coil-PANet (yaml/script 待完成)"
else
  nohup bash scripts/run_coil_panet_250ep.sh > /tmp/panet.log 2>&1 &
  PN_PID=$!
  echo "Coil-PANet PID=$PN_PID"
  while ps -p $PN_PID > /dev/null 2>&1; do
    sleep 60
    ELAPSED=$(ps -p $PN_PID -o etime= 2>/dev/null | xargs)
    echo "$(date '+%H:%M:%S') Coil-PANet 仍在跑 [$ELAPSED]"
  done
  echo "✅ Coil-PANet 完成 at $(date '+%H:%M:%S')"
fi

# 4. 评估
echo "===== 启动评估 ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/projects/hyperyolo/scripts/compare_v0_innovations_v2.py

echo "🎉 ulracode B 全部完成 at $(date '+%H:%M:%S')"
