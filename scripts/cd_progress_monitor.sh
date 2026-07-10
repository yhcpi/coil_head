#!/bin/bash
# C+D 项训练进度监控：每 ~60s 检查 epoch 数，达到 20 的倍数时输出当前 mAP 趋势
# 没进展（mAP 停滞或下跌）时打印 WARNING
# 用法：bash scripts/cd_progress_monitor.sh [09_bayes_prior|10_paaug_motion|...]
set -u
NAME="${1:-09_bayes_prior}"
DIR="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/$NAME"
CSV="$DIR/results.csv"
LOG="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}_monitor.log"
STATUS="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}/STATUS.md"
LAST_CHECK_EPOCH=0
LAST_BEST_MAP=0
NO_PROGRESS_COUNT=0

echo "[$(date '+%H:%M:%S')] 监控 $NAME 启动，每 60s 检查" > "$LOG"

while true; do
    sleep 60

    # 检查进程是否还活着
    if ! pgrep -f "name=$NAME" > /dev/null; then
        # 进程结束（可能完成或失败）
        if [ -f "$DIR/weights/best.pt" ]; then
            FINAL_EPOCH=$(tail -n +2 "$CSV" 2>/dev/null | tail -1 | cut -d, -f1)
            echo "[$(date '+%H:%M:%S')] ★ 训练结束，最终 epoch=$FINAL_EPOCH" >> "$LOG"
        else
            echo "[$(date '+%H:%M:%S')] ✗ 训练进程退出但无 best.pt，可能失败" >> "$LOG"
        fi
        break
    fi

    # 读当前 epoch
    if [ ! -f "$CSV" ]; then
        # 即使 CSV 不存在，也写一个 minimal STATUS.md 告知 main session 监控在线
        cat > "$STATUS" <<EOF
# ${NAME} 监控状态（$(date '+%Y-%m-%d %H:%M:%S')）

- 监控进程: 运行中（PID $$）
- 训练进程: 运行中（pgrep 命中 name=$NAME）
- results.csv: 尚未生成（等待第一个 epoch 写入）
- 最新检查: $(date '+%H:%M:%S')
EOF
        continue
    fi
    CUR_EPOCH=$(tail -n +2 "$CSV" 2>/dev/null | tail -1 | cut -d, -f1 | tr -d ' ')
    CUR_EPOCH=${CUR_EPOCH:-0}
    CUR_EPOCH=$((CUR_EPOCH + 0))

    # 当前 best mAP50（每次循环都算，STATUS.md 需要）
    CUR_BEST_MAP=$(tail -n +2 "$CSV" | cut -d, -f10 | sort -g | tail -1 2>/dev/null)
    CUR_BEST_MAP=${CUR_BEST_MAP:-0}
    PROCESS_ALIVE=$(pgrep -fc "name=$NAME" 2>/dev/null || echo 0)

    # 每 20 epoch 评估一次
    if [ $CUR_EPOCH -gt 0 ] && [ $((CUR_EPOCH % 20)) -eq 0 ] && [ $CUR_EPOCH -ne $LAST_CHECK_EPOCH ]; then
        LAST_CHECK_EPOCH=$CUR_EPOCH
        echo "" >> "$LOG"
        echo "========== [$(date '+%H:%M:%S')] epoch $CUR_EPOCH 检查 ==========" >> "$LOG"
        echo "[当前最近 5 个 epoch]" >> "$LOG"
        tail -n +2 "$CSV" | tail -5 | awk -F',' '{
            printf "  ep=%s  box=%s  cls=%s  dfl=%s  P=%s  R=%s  mAP50=%s  mAP50-95=%s\n",
                $1, $2, $3, $4, $8, $9, $10, $11
        }' >> "$LOG"

        # 当前 best mAP50（从 results.csv 中所有 epoch 中取 max）
        CUR_BEST_MAP=$(tail -n +2 "$CSV" | cut -d, -f10 | sort -g | tail -1)
        CUR_BEST_MAP=${CUR_BEST_MAP:-0}

        # 与上次比较
        DELTA=$(echo "$CUR_BEST_MAP - $LAST_BEST_MAP" | bc -l 2>/dev/null)
        if [ -z "$DELTA" ]; then DELTA=0; fi

        echo "[best mAP50=$CUR_BEST_MAP  Δ=$DELTA  (v4 baseline=0.877)]" >> "$LOG"

        # 进展判定：Δ < 0.01 算"无进展"
        IS_STALL=$(echo "$DELTA < 0.01" | bc -l 2>/dev/null)
        if [ "$IS_STALL" = "1" ]; then
            NO_PROGRESS_COUNT=$((NO_PROGRESS_COUNT + 1))
            echo "  ⚠⚠⚠ 第 $NO_PROGRESS_COUNT 次无进展（Δ < 0.01）⚠⚠⚠" >> "$LOG"
            if [ $NO_PROGRESS_COUNT -ge 3 ]; then
                echo "  ✗✗✗ 连续 3 次无进展，建议终止训练 ✗✗✗" >> "$LOG"
            fi
        else
            NO_PROGRESS_COUNT=0
        fi
        LAST_BEST_MAP=$CUR_BEST_MAP
    fi

    # 同时把最近 1 行写进 log（持续 trace）
    if [ $CUR_EPOCH -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] epoch=$CUR_EPOCH" >> "$LOG"
    fi

    # 覆盖写 STATUS.md（每 60s 一次，作为 main session 可直接 Read 的主动轮询源）
    if [ $CUR_EPOCH -gt 0 ]; then
        LAST3=$(tail -n +2 "$CSV" | tail -3 | awk -F',' '{
            printf "  ep=%s  box=%s  cls=%s  P=%s  R=%s  mAP50=%s\n",
                $1, $2, $3, $8, $9, $10
        }')
        LAST_20_CHECK=""
        if [ $((CUR_EPOCH % 20)) -eq 0 ]; then
            LAST_20_CHECK="⚠ 刚刚触发 epoch ${CUR_EPOCH} 检查：见 ${LOG}"
        fi
        STALL_INFO=""
        if [ $NO_PROGRESS_COUNT -ge 3 ]; then
            STALL_INFO="✗✗✗ 连续 ${NO_PROGRESS_COUNT} 次无进展（建议终止训练）"
        elif [ $NO_PROGRESS_COUNT -ge 1 ]; then
            STALL_INFO="⚠ 第 ${NO_PROGRESS_COUNT}/3 次无进展"
        fi
        cat > "$STATUS" <<EOF
# ${NAME} 监控状态（$(date '+%Y-%m-%d %H:%M:%S')）

- 监控进程: 运行中（PID $$）
- 训练进程: 运行中（${PROCESS_ALIVE} 个 worker）
- 当前 epoch: ${CUR_EPOCH} / 250
- best mAP50: ${CUR_BEST_MAP}（v4 baseline=0.877）
- 无进展计数: ${NO_PROGRESS_COUNT}/3
- 最近 3 个 epoch：
${LAST3}
- ${STALL_INFO:-(无停滞告警)}
- ${LAST_20_CHECK:-(非 20 倍数 epoch，无新检查)}
- 完整监控 log: ${LOG}
EOF
    fi
done
echo "[$(date '+%H:%M:%S')] 监控 $NAME 退出" >> "$LOG"