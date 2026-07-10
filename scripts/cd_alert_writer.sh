#!/bin/bash
# C+D 项训练监控 + 主动通知脚本
# 每 60s 检查 epoch + 当条件满足时写 STATUS.md
# 条件：
#   1. epoch 跨过 20 倍数：写 STATUS.md "epoch $EPOCH"
#   2. 训练进程退出：写 STATUS.md "训练结束"
#   3. 出现 NaN：写 STATUS.md "训练失败 NaN"
#   4. best.pt 存在：写 STATUS.md "best.pt 已生成，可启动 D 项"
#
# 用法：bash scripts/cd_alert_writer.sh 09_bayes_prior [expected_best_epoch=200]
set -u
NAME="${1:-09_bayes_prior}"
EXPECTED_BEST="${2:-200}"
DIR="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/$NAME"
CSV="$DIR/results.csv"
STATUS="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/STATUS.md"
MONITOR_LOG="/home/pi/projects/hyperyolo/runs/coil_loss_ablation/${NAME}_monitor.log"

LAST_NOTIFIED_EPOCH=0
LAST_WRITTEN_AT=""

echo "[$(date '+%H:%M:%S')] alert_writer for $NAME 启动" >> "$MONITOR_LOG"

while true; do
    sleep 60

    # === 1. 训练进程退出：写入 best.pt 信息 ===
    if ! pgrep -f "name=$NAME" > /dev/null; then
        if [ -f "$DIR/weights/best.pt" ]; then
            cat > "$STATUS" <<EOF
# C 项 训练结束（$(date '+%Y-%m-%d %H:%M:%S')）

- run: $NAME
- best.pt: $DIR/weights/best.pt ($(stat -c %s $DIR/weights/best.pt) bytes)
- total epochs: $(tail -n +2 "$CSV" 2>/dev/null | tail -1 | cut -d, -f1)
- best mAP50: $(tail -n +2 "$CSV" 2>/dev/null | cut -d, -f10 | sort -g | tail -1)
- best Recall: $(tail -n +2 "$CSV" 2>/dev/null | cut -d, -f9 | sort -g | tail -1)

请启动 D 项 PA-Aug 4 组件 ablation。
EOF
            echo "[$(date '+%H:%M:%S')] ★ 写入 STATUS.md（训练结束）" >> "$MONITOR_LOG"
        fi
        break
    fi

    # === 2. NaN 检测 ===
    if [ -f "$CSV" ]; then
        if grep -q "nan" "$CSV" 2>/dev/null; then
            cat > "$STATUS" <<EOF
# ⚠ C 项训练失败（$(date '+%Y-%m-%d %H:%M:%S')）

- run: $NAME
- 失败原因：results.csv 出现 NaN
- NaN 起点 epoch：$(grep -n "nan" $CSV | head -1 | cut -d, -f1 | tr -d ' ')
- 需手动诊断
EOF
            echo "[$(date '+%H:%M:%S')] ✗ 写入 STATUS.md（NaN）" >> "$MONITOR_LOG"
            break
        fi
    fi

    # === 3. epoch 跨过 20 倍数：写 STATUS.md ===
    if [ -f "$CSV" ]; then
        CUR_EPOCH=$(tail -n +2 "$CSV" 2>/dev/null | tail -1 | cut -d, -f1 | tr -d ' ')
        CUR_EPOCH=$((CUR_EPOCH + 0))

        if [ $CUR_EPOCH -gt 0 ] && [ $((CUR_EPOCH % 5)) -eq 0 ] && [ $CUR_EPOCH -ne $LAST_NOTIFIED_EPOCH ]; then
            LAST_NOTIFIED_EPOCH=$CUR_EPOCH

            BEST_MAP=$(tail -n +2 "$CSV" | cut -d, -f10 | sort -g | tail -1)
            BEST_REC=$(tail -n +2 "$CSV" | cut -d, -f9 | sort -g | tail -1)
            CUR_BOX_LOSS=$(tail -n +2 "$CSV" | tail -1 | cut -d, -f2)
            CUR_VAL_LOSS=$(tail -n +2 "$CSV" | tail -1 | cut -d, -f12)

            cat > "$STATUS" <<EOF
# C 项 训练进度（$(date '+%Y-%m-%d %H:%M:%S')）

- run: $NAME
- epoch: $CUR_EPOCH/250
- best mAP50: $BEST_MAP  (v4 baseline=0.877)
- best Recall: $BEST_REC  (v4 baseline=0.85)
- 当前 train box_loss: $CUR_BOX_LOSS
- 当前 val box_loss: $CUR_VAL_LOSS

## 判定
$(if [ "${BEST_MAP%.*}" -lt 1 ] && [ $CUR_EPOCH -gt 100 ]; then
    echo "⚠⚠⚠ epoch $CUR_EPOCH 仍未出 mAP50 > 0.1，强烈建议终止 ⚠⚠⚠"
elif [ "${BEST_MAP%.*}" -lt 1 ]; then
    echo "⚠ mAP50 < 0.1, 继续等待 (epoch 100+ 才考虑终止)"
else
    echo "✓ 训练进展中"
fi)

下一步：
- 你回来时直接输入 prompt，我会主动读 STATUS.md 报告
- 或 `! bash scripts/c_d_status.sh $NAME` 手动查
- 监控位置：runs/coil_loss_ablation/${NAME}_monitor.log
EOF

            echo "[$(date '+%H:%M:%S')] 📝 写入 STATUS.md（epoch $CUR_EPOCH）" >> "$MONITOR_LOG"
            LAST_WRITTEN_AT="$(date '+%H:%M:%S')"
        fi
    fi

    # === 4. best.pt 已生成（提前于训练结束） ===
    if [ -f "$DIR/weights/best.pt" ] && [ $CUR_EPOCH -ge $EXPECTED_BEST ]; then
        LAST_BEST_MTIME=$(stat -c %Y "$DIR/weights/best.pt" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        AGE=$((NOW - LAST_BEST_MTIME))
        # best.pt 30 min 内新
        if [ $AGE -lt 1800 ]; then
            echo "[$(date '+%H:%M:%S')] best.pt 已生成 (epoch $CUR_EPOCH+)" >> "$MONITOR_LOG"
            # 不重复写 STATUS（跟每 20 epoch 的报告合并）
        fi
    fi
done
echo "[$(date '+%H:%M:%S')] alert_writer $NAME 退出" >> "$MONITOR_LOG"