#!/usr/bin/env bash
# 2026-07-16 H01:00 ulracode B 续跑: chain 完 panet 后自动接续
# 1. 跑评估 (compare_v0_innovations_v2.py)
# 2. 看 panet 末 30 ep 趋势: 未收敛则续跑 (lr=0.005 + 200ep)
# 3. 启动 DySample v2 真实现重训 (用户重点救)
# 4. 等 DySample v2 完跑评估, 总链归档
set -uo pipefail
cd /home/pi/projects/hyperyolo

echo "===== ulracode B 续跑入口 $(date '+%H:%M:%S') ====="

# Step 1: 等当前 chain (PID 1239375) 跑完所有阶段
echo "===== 等 chain (PID 1239375) 完成所有阶段 ====="
while ps -p 1239375 > /dev/null 2>&1; do
  sleep 120
  ELAPSED=$(ps -p 1239375 -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') chain 仍在跑 [$ELAPSED]"
done
echo "✅ chain 完成 at $(date '+%H:%M:%S')"

# Step 2: 看 panet 收敛性
PANET_CSV="runs/baseline/v0_panet_hyper_yolon_250ep/results.csv"
if [ -f "$PANET_CSV" ]; then
  PANET_MAX=$(awk -F, 'NR>1 {if($10>m){m=$10}} END{printf "%.4f", m}' "$PANET_CSV")
  PANET_LAST=$(awk -F, 'NR>1 {m=$10} END{printf "%.4f", m}' "$PANET_CSV")
  PANET_LAST_EP=$(awk -F, 'NR>1 {ep=$1} END{print ep}' "$PANET_CSV")
  echo "panet max mAP50=$PANET_MAX, last ep=$PANET_LAST_EP mAP50=$PANET_LAST"
fi

# Step 3: 看 panet 末 30 ep std (判收敛)
PANET_TAIL30_STD=$(awk -F, 'NR>1 {v[NR]=$10} END {
  n=NR-1; start=n-29; if(start<1)start=1
  sum=0; cnt=0
  for(i=start;i<=n;i++){sum+=v[i]; cnt++}
  mean=sum/cnt
  sq=0
  for(i=start;i<=n;i++){sq+=(v[i]-mean)^2}
  printf "%.4f", sqrt(sq/cnt)
}' "$PANET_CSV" 2>/dev/null)
echo "panet 末 30 ep mAP50 std = $PANET_TAIL30_STD (≤0.05 视为收敛)"

# Step 4: panet 决策
PN_NEED_EXTRA="false"
if [ "$(echo "$PANET_LAST_EP < 250" | bc 2>/dev/null)" = "1" ]; then
  echo "panet 训练被中断 (last_ep=$PANET_LAST_EP < 250), 不续训"
elif [ "$(echo "$PANET_TAIL30_STD > 0.05" | bc 2>/dev/null)" = "1" ]; then
  echo "⚠️  panet 末 30 ep std=$PANET_TAIL30_STD > 0.05, 仍在爬升 → 续训 200ep @ lr=0.005"
  PN_NEED_EXTRA="true"
elif [ "$(echo "$PANET_LAST < 0.6" | bc 2>/dev/null)" = "1" ]; then
  echo "⚠️  panet 末 ep mAP50=$PANET_LAST < 0.6, 已收敛但低 → 续训 200ep @ lr=0.005 + pretrain=baseline"
  PN_NEED_EXTRA="true"
else
  echo "✅ panet 末 ep mAP50=$PANET_LAST >= 0.6 且已收敛, 不续训"
fi

if [ "$PN_NEED_EXTRA" = "true" ]; then
  nohup bash scripts/run_panet_extend_200ep.sh > /tmp/panet_ext.log 2>&1 &
  PN_EXT_PID=$!
  echo "panet 续训 PID=$PN_EXT_PID"
  while ps -p $PN_EXT_PID > /dev/null 2>&1; do
    sleep 180
    ELAPSED=$(ps -p $PN_EXT_PID -o etime= 2>/dev/null | xargs)
    echo "$(date '+%H:%M:%S') panet 续训仍在跑 [$ELAPSED]"
  done
  echo "✅ panet 续训完成 at $(date '+%H:%M:%S')"
fi

# Step 5: 启动 DySample v2 真实现训练 (用户重点救)
echo "===== 启动 DySample v2 真实现 ====="
nohup bash scripts/run_dysample_v2_250ep.sh > /tmp/dy_v2.log 2>&1 &
DY_V2_PID=$!
echo "DySample v2 PID=$DY_V2_PID"
while ps -p $DY_V2_PID > /dev/null 2>&1; do
  sleep 180
  ELAPSED=$(ps -p $DY_V2_PID -o etime= 2>/dev/null | xargs)
  echo "$(date '+%H:%M:%S') DySample v2 仍在跑 [$ELAPSED]"
done
echo "✅ DySample v2 完成 at $(date '+%H:%M:%S')"

# Step 6: 总评估 + 归档
echo "===== 最终评估 (含 DySample v2) ====="
PYTHONPATH= /home/pi/anaconda3/envs/hyper-yolo/bin/python /home/pi/projects/hyperyolo/scripts/compare_v0_innovations_v2.py

echo "🎉 ulracode B + 续跑 全部完成 at $(date '+%H:%M:%S')"