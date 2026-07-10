#!/bin/bash
# NWD 主动轮询（每 10 分钟输出最新数据到界面）
NWD_DIR="runs/coil_loss_ablation/v8_nwd_full"
NWD_PID=3697858

echo "=== NWD 主动轮询启动（每 600 秒输出） ==="
echo "PID=$NWD_PID, 目标完成时间 ~21:45"

while true; do
    if ! ps -p $NWD_PID > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] NWD 进程已结束，轮询停止"
        exit 0
    fi
    /home/pi/anaconda3/envs/hyper-yolo/bin/python -c "
import csv
with open('$NWD_DIR/results.csv') as f:
    rows = [r for r in csv.reader(f) if r]
last = rows[-1]
best_m, best_e = 0, 0
for r in rows[1:]:
    m = float(r[9])
    if m > best_m: best_m, best_e = m, int(r[0])
cur_epoch = int(last[0])
remain_epoch = 250 - cur_epoch
remain_min = remain_epoch / 3.14
print(f'[$(date +%H:%M:%S)] epoch={cur_epoch}/250 box={last[1]} cls={last[2]} P={last[7]} R={last[8]} mAP50={last[9]} mAP50-95={last[10]}')
print(f'[$(date +%H:%M:%S)] best: epoch={best_e} mAP50={best_m:.4f}')
print(f'[$(date +%H:%M:%S)] remain: {remain_epoch} epoch (~{remain_min:.0f} min)')
"
    sleep 600
done
