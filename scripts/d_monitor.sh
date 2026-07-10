#!/bin/bash
# D 项训练进度监控：每 ~60s 检查 epoch，达到 20 的倍数时输出 mAP 趋势
# 与 cd_progress_monitor.sh 一致逻辑，但支持任意 run name 列表（轮询切换）
# 用法：bash scripts/d_monitor.sh
set -u
DIR_BASE="/home/pi/projects/hyperyolo/runs/coil_loss_ablation"
LOG_BASE="/home/pi/projects/hyperyolo/runs/coil_loss_ablation"

# 4 个 PA-Aug runs 顺序监控
RUNS=(
    "11_paaug_motion"
    "11_paaug_reflection"
    "11_paaug_occlusion"
    "11_paaug_noise"
)

declare -A LAST_CHECK_EPOCH
declare -A LAST_BEST_MAP
declare -A NO_PROGRESS_COUNT
for r in "${RUNS[@]}"; do
    LAST_CHECK_EPOCH[$r]=0
    LAST_BEST_MAP[$r]=0
    NO_PROGRESS_COUNT[$r]=0
done

echo "[$(date '+%H:%M:%S')] D 项多 run 监控启动" > "$LOG_BASE/d_multi_monitor.log"

while true; do
    sleep 60
    ANY_ALIVE=0
    for NAME in "${RUNS[@]}"; do
        DIR="$DIR_BASE/$NAME"
        CSV="$DIR/results.csv"
        STATUS="$DIR/STATUS.md"
        LOG="$LOG_BASE/${NAME}_monitor.log"
        if ! pgrep -f "name=$NAME" > /dev/null; then
            # 此 run 未在跑（要么没开始，要么已结束）
            continue
        fi
        ANY_ALIVE=1

        if [ ! -f "$CSV" ]; then
            cat > "$STATUS" <<EOF
# ${NAME} 监控状态（$(date '+%Y-%m-%d %H:%M:%S')）

- 监控进程: 运行中
- 训练进程: 运行中（pgrep 命中 name=$NAME）
- results.csv: 尚未生成
- 最新检查: $(date '+%H:%M:%S')
EOF
            continue
        fi

        CUR_EPOCH=$(tail -n +2 "$CSV" 2>/dev/null | tail -1 | cut -d, -f1 | tr -d ' ')
        CUR_EPOCH=${CUR_EPOCH:-0}
        CUR_EPOCH=$((CUR_EPOCH + 0))
        CUR_BEST_MAP=$(tail -n +2 "$CSV" | cut -d, -f10 | sort -g | tail -1 2>/dev/null)
        CUR_BEST_MAP=${CUR_BEST_MAP:-0}
        PROCESS_ALIVE=$(pgrep -fc "name=$NAME" 2>/dev/null || echo 0)

        # 每 20 epoch 评估
        if [ $CUR_EPOCH -gt 0 ] && [ $((CUR_EPOCH % 20)) -eq 0 ] && [ $CUR_EPOCH -ne "${LAST_CHECK_EPOCH[$NAME]}" ]; then
            LAST_CHECK_EPOCH[$NAME]=$CUR_EPOCH
            echo "" >> "$LOG"
            echo "========== [$(date '+%H:%M:%S')] $NAME epoch $CUR_EPOCH 检查 ==========" >> "$LOG"
            tail -n +2 "$CSV" | tail -5 | awk -F',' '{
                printf "  ep=%s  box=%s  cls=%s  dfl=%s  P=%s  R=%s  mAP50=%s\n",
                    $1, $2, $3, $4, $8, $9, $10
            }' >> "$LOG"
            CUR_BEST_MAP=$(tail -n +2 "$CSV" | cut -d, -f10 | sort -g | tail -1)
            DELTA=$(echo "$CUR_BEST_MAP - ${LAST_BEST_MAP[$NAME]}" | bc -l 2>/dev/null)
            [ -z "$DELTA" ] && DELTA=0
            echo "[best mAP50=$CUR_BEST_MAP  Δ=$DELTA  (v4 baseline=0.877)]" >> "$LOG"
            IS_STALL=$(echo "$DELTA < 0.01" | bc -l 2>/dev/null)
            if [ "$IS_STALL" = "1" ]; then
                NO_PROGRESS_COUNT[$NAME]=$(( ${NO_PROGRESS_COUNT[$NAME]} + 1 ))
                echo "  ⚠⚠⚠ 第 ${NO_PROGRESS_COUNT[$NAME]} 次无进展 ⚠⚠⚠" >> "$LOG"
            else
                NO_PROGRESS_COUNT[$NAME]=0
            fi
            LAST_BEST_MAP[$NAME]=$CUR_BEST_MAP
        fi

        # 写 STATUS.md
        LAST3=$(tail -n +2 "$CSV" | tail -3 | awk -F',' '{
            printf "  ep=%s  box=%s  cls=%s  P=%s  R=%s  mAP50=%s\n",
                $1, $2, $3, $8, $9, $10
        }')
        STALL=""
        if [ "${NO_PROGRESS_COUNT[$NAME]}" -ge 3 ]; then
            STALL="✗✗✗ 连续 ${NO_PROGRESS_COUNT[$NAME]} 次无进展（建议终止）"
        elif [ "${NO_PROGRESS_COUNT[$NAME]}" -ge 1 ]; then
            STALL="⚠ 第 ${NO_PROGRESS_COUNT[$NAME]}/3 次无进展"
        fi
        cat > "$STATUS" <<EOF
# ${NAME} 监控状态（$(date '+%Y-%m-%d %H:%M:%S')）

- 监控进程: 运行中
- 训练进程: 运行中（${PROCESS_ALIVE} worker）
- 当前 epoch: ${CUR_EPOCH} / 250
- best mAP50: ${CUR_BEST_MAP}（v4 baseline=0.877）
- 无进展计数: ${NO_PROGRESS_COUNT[$NAME]}/3
- 最近 3 epoch：
${LAST3}
- ${STALL:-(无停滞告警)}
EOF
        echo "[$(date '+%H:%M:%S')] $NAME epoch=$CUR_EPOCH" >> "$LOG"
    done

    if [ $ANY_ALIVE -eq 0 ]; then
        # 所有 run 都结束（或还没开始）
        echo "[$(date '+%H:%M:%S')] 无活跃训练，监控进入空闲" >> "$LOG_BASE/d_multi_monitor.log"
        # 不退出，继续等待下一轮
    fi
done