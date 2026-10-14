#!/usr/bin/env bash
# 队列：等带帧单场景 run 跑完，自动接上跨场景（4 场景 × 13 回合）那一轮。
#
# 为什么排成队列而不是并行：InternNav 的 System-2 模型单实例占 ~21 GB 显存，
# GPU1 只有 48 GB（还和别人的 hospital-delivery-agent 共用），
# 同时起两个 evaluator 有 OOM 风险，而且上游 wrapper 也不鼓励并发。
#
# 用法：bash queue_multiscene_after_capture.sh [前置会话名] [跨场景标签]
set -euo pipefail

WAIT_FOR="${1:-r1_r2adapt_capture}"
LABEL="${2:-r1multi4}"
SCENES=4
PER_SCENE=13
EPISODES=$((SCENES * PER_SCENE))

REPO=/srv/agh-demo/InternNav
PY=/srv/agh-demo/habitat-env/bin/python
LOG_DIR=/srv/agh-demo/logs/route1
OUT="${REPO}/logs/habitat/route1_${LABEL}_seed20260826"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

echo "[queue] $(date -Is) 等待前置会话 ${WAIT_FOR} 结束…"
while tmux has-session -t "${WAIT_FOR}" 2>/dev/null; do
    sleep 60
done
echo "[queue] $(date -Is) 前置会话已结束。"

if [ -s "${OUT}/progress.json" ]; then
    echo "[queue] 目标已存在，放弃：${OUT}"
    exit 0
fi

cat > "${LOG_DIR}/_${LABEL}_inner.sh" <<INNER
#!/usr/bin/env bash
set -u
cd ${REPO}
export CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=20260826
export BUDGET_MODE=drift_recovery_periodic
export BUDGET_RUN_LABEL=${LABEL}
export EPISODE_LIMIT=${EPISODES}
export PERIODIC_INTERVAL=12
export PERIODIC_AFTER_RECOVERIES=2
export MAX_CONSECUTIVE_COLLISION_ACTIONS=0
export COLLISION_WINDOW_ACTIONS=8
export MAX_COLLISIONS_IN_WINDOW=6
export CAPTURE_MAX_EPISODE_ID=400
export MULTISCENE_PER_SCENE=${PER_SCENE}
export MULTISCENE_N_SCENES=${SCENES}
echo "START \$(date -Is) label=${LABEL} mode=drift_recovery_periodic scenes=${SCENES}x${PER_SCENE}=${EPISODES} split=val_unseen_multiscene"
${PY} -u scripts/eval/eval.py --config scripts/eval/configs/route1_multiscene_cfg.py
echo "EXIT=\$? \$(date -Is)"
INNER
chmod +x "${LOG_DIR}/_${LABEL}_inner.sh"

tmux new-session -d -s "r1_${LABEL}" \
    "bash ${LOG_DIR}/_${LABEL}_inner.sh 2>&1 | tee ${LOG_DIR}/${LABEL}.log"
echo "[queue] $(date -Is) 已启动 tmux 会话 r1_${LABEL}"
echo "  日志：${LOG_DIR}/${LABEL}.log"
echo "  产物：${OUT}"
