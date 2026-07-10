#!/bin/bash
# NWD 训练监控（每 10 分钟查一次，输出到界面 + 日志）
NWD_DIR="runs/coil_loss_ablation/v8_nwd_full"
NWD_PID=3697858
LOG="/tmp/nwd_monitor.log"

echo "=== NWD 训练监控启动 ==="
echo "每 600 秒（10 分钟）输出一次"
echo "PID=$NWD_PID"

while true; do
    # 进程死了就退出
    if ! ps -p $NWD_PID > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] NWD 进程已结束，监控停止"
        exit 0
    fi
    if [ ! -f "$NWD_DIR/results.csv" ]; then
        echo "[$(date +%H:%M:%S)] results.csv 不存在，等待中..."
    else
        # 单行 python 脚本：解析最新 + 最佳 + 剩余时间
        /home/pi/anaconda3/envs/hyper-yolo/bin/python <<'PYEOF'
import csv
from datetime import datetime
results = '/home/pi/projects/hyperyolo/runs/coil_loss_ablation/v8_nwd_full/results.csv'
try:
    with open(results) as f:
        rows = [r for r in csv.reader(f) if r]
    header, *data = rows
    if not data:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] results.csv 无数据')
    else:
        last = data[-1]
        best_mAP50, best_epoch = 0, 0
        for r in data:
            try:
                mAP50 = float(r[9])
                if mAP50 > best_mAP50:
                    best_mAP50, best_epoch = mAP50, int(r[0])
            except (ValueError, IndexError):
                pass
        cur_epoch = int(last[0])
        remain_epoch = 250 - cur_epoch
        remain_min = remain_epoch / 3.14
        ts = datetime.now().strftime("%H:%M:%S")
        print(f'[{ts}] cur: epoch={cur_epoch} box={last[1]} cls={last[2]} P={last[7]} R={last[8]} mAP50={last[9]} mAP50-95={last[10]}')
        print(f'[{ts}] best: epoch={best_epoch} mAP50={best_mAP50:.4f}')
        print(f'[{ts}] remain: {remain_epoch} epoch (~{remain_min:.0f} min, ~{(remain_min/60):.1f} h)')
except Exception as e:
    print(f'[{datetime.now().strftime("%H:%M:%S")}] 解析失败: {e}')
PYEOF
    fi
    sleep 600
done
