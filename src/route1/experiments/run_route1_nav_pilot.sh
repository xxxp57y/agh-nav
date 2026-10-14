#!/usr/bin/env bash
# 路线① 导航侧 pilot：带每决策帧捕获的闭环运行（语义 Monitor 的数据来源）
#
# 用法：
#   bash run_route1_nav_pilot.sh pilot20 20      # 20 回合，约 1 小时
#   bash run_route1_nav_pilot.sh full50 50       # 50 回合，约 2.5 小时
#
# 设计约束：
#   - 只用 GPU1（GPU0 有别人的常驻负载）
#   - 输出到 logs/habitat/route1_*，绝不覆盖上游既有 run（wrapper 也会拒绝）
#   - 长任务跑在 tmux 里（会话名 r1_<label>）
set -euo pipefail

LABEL="${1:-pilot20}"
EPISODES="${2:-20}"
MODE="${3:-budgeted_drift}"

REPO=/srv/agh-demo/InternNav
PY=/srv/agh-demo/habitat-env/bin/python
LOG_DIR=/srv/agh-demo/logs/route1
SEED=20260826
TMUX_SESSION="r1_${LABEL}"
OUT="${REPO}/logs/habitat/route1_${LABEL}_seed${SEED}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

if [ -s "${OUT}/progress.json" ]; then
  echo "拒绝覆盖已存在的 run：${OUT}"
  exit 1
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux 会话 ${TMUX_SESSION} 已存在，先处理它"
  exit 1
fi

echo "== 预检 =="
"${PY}" - <<'PY'
import importlib, os
for name in ("habitat", "torch", "internnav"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name}: OK")
    except Exception as exc:
        print(f"  {name}: FAIL {type(exc).__name__}: {exc}")
print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

cat > "${LOG_DIR}/_${LABEL}_inner.sh" <<INNER
#!/usr/bin/env bash
set -euo pipefail
cd ${REPO}
export CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=${SEED}
export BUDGET_MODE=${MODE}
export BUDGET_RUN_LABEL=${LABEL}
export EPISODE_LIMIT=${EPISODES}
export CAPTURE_MAX_EPISODE_ID=400
echo "START \$(date -Is) label=${LABEL} mode=${MODE} episodes=${EPISODES}"
${PY} -u scripts/eval/eval.py --config scripts/eval/configs/route1_capture_cfg.py
echo "EXIT=\$? \$(date -Is)"
INNER
chmod +x "${LOG_DIR}/_${LABEL}_inner.sh"

tmux new-session -d -s "${TMUX_SESSION}" \
  "bash ${LOG_DIR}/_${LABEL}_inner.sh 2>&1 | tee ${LOG_DIR}/${LABEL}.log"

echo
echo "已在 tmux 会话 ${TMUX_SESSION} 中启动。"
echo "  跟踪： tmux attach -t ${TMUX_SESSION}     （Ctrl-b d 脱离）"
echo "  日志： tail -f ${LOG_DIR}/${LABEL}.log"
echo "  产物： ${OUT}"
echo "  帧：   ${OUT}/early_replan_artifacts/<scene>_<episode>/"
