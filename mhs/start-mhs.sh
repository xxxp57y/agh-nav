#!/usr/bin/env bash
# Start the isolated native workbench. No existing navlive/Agent process is stopped.
set -euo pipefail
BASE="${AGH_MHS_BASE:?Set AGH_MHS_BASE to your isolated deployment directory}"
RUNTIME="${AGH_MHS_RUNTIME:-$BASE/agh-mhs-runtime}"
NAVROOT="${AGH_MHS_NAVROOT:-$BASE/agh-nav}"
export AGH_HOME="${AGH_MHS_HOME:-$BASE/agh-mhs-home}"
export AGNES_PROFILE=local-dev
export AGNES_HUB_LISTEN=127.0.0.1:43132
export AGNES_HUB_ORIGIN=http://127.0.0.1:43132
export AGNES_WEB_ORIGIN=http://127.0.0.1:43130
PY_HAB="${AGH_HABITAT_PYTHON:?Set AGH_HABITAT_PYTHON to a Habitat-enabled Python}"
HAB_REPO="${ROUTE1_REPO:?Set ROUTE1_REPO to your InternNav checkout}"
export NAVLIVE_REPO="$HAB_REPO"
export ROUTE1_REPO="$HAB_REPO"
export PYTHONPATH="$NAVROOT/src${PYTHONPATH:+:$PYTHONPATH}"
WORK="$BASE/agh-mhs-workspace"
LOGS="$BASE/agh-mhs-logs"
CLI="$RUNTIME/packages/cli/dist/local/agnes.mjs"
start() {
  local name="$1" command="$2"
  if tmux has-session -t "=$name" 2>/dev/null; then
    echo "$name already running"
  else
    tmux new-session -d -s "$name" "$command"
  fi
}
case "${1:-start}" in
  start|web|bootstrap)
    test -f "$CLI"
    test -f "$NAVROOT/mhs/nav_device.py"
    test -x "$NAVROOT/.venv/bin/python"
    mkdir -p "$WORK"
    mkdir -p "$LOGS"
    mkdir -p "$AGH_HOME"
    chmod 700 "$AGH_HOME"
    cp "$NAVROOT/mhs/AGENTS.md" "$WORK/AGENTS.md"
    start agh_mhs_daemon "cd '$WORK' && AGH_HOME='$AGH_HOME' AGNES_PROFILE=local-dev AGNES_HUB_LISTEN=127.0.0.1:43132 node '$RUNTIME/packages/cli/dist/local/daemon.mjs' --home '$AGH_HOME' --profile local-dev --workspace '$WORK' --local-web-addr 127.0.0.1:43134 --local-web-origin http://127.0.0.1:43130 > '$LOGS/daemon.log' 2>&1"
    for i in $(seq 1 40); do
      if [ -S "$AGH_HOME/data/daemon/agnesd.sock" ]; then break; fi
      sleep 0.25
    done
    if [ "${1:-start}" = bootstrap ]; then exit 0; fi
    start agh_mhs_web "cd '$WORK' && AGH_HOME='$AGH_HOME' AGNES_PROFILE=local-dev AGNES_HUB_LISTEN=127.0.0.1:43132 AGNES_HUB_ORIGIN=http://127.0.0.1:43132 AGNES_WEB_ORIGIN=http://127.0.0.1:43130 node '$CLI' serve --port 43130 > '$LOGS/web.log' 2>&1"
    if [ "${1:-start}" = web ]; then exit 0; fi
    start agh_mhs_navlive "cd '$HAB_REPO' && CUDA_VISIBLE_DEVICES=0 NAVLIVE_ENV_CACHE=2 PYTHONPATH='$NAVROOT/src' '$PY_HAB' -u '$NAVROOT/navlive/navlive_server.py' --port 43131 --open '${AGH_MHS_SCENE:-zsNo4HB9uLZ}' --preload > '$LOGS/navlive.log' 2>&1"
    # Wait for READY, not a fixed sleep that can return before the scene is loaded.
    for i in $(seq 1 100); do
      if curl -fsS --max-time 2 http://127.0.0.1:43131/health 2>/dev/null | grep -q '"env_loaded":true'; then break; fi
      sleep 1
    done
    if ! curl -fsS --max-time 3 http://127.0.0.1:43131/health | grep -q '"env_loaded":true'; then
      echo "Habitat scene did not become ready; see $LOGS/navlive.log"; exit 1
    fi
    start agh_mhs_device "PYTHONPATH='$RUNTIME/packages/mhs/python/src' '$NAVROOT/.venv/bin/python' -u '$NAVROOT/mhs/nav_device.py' --hub ws://127.0.0.1:43132 --navlive http://127.0.0.1:43131 --scene '${AGH_MHS_SCENE:-zsNo4HB9uLZ}' > '$LOGS/device.log' 2>&1"
    for i in $(seq 1 40); do
      if grep -q 'habitat-01: registered' "$LOGS/device.log" 2>/dev/null; then break; fi
      sleep 1
    done
    if ! grep -q 'habitat-01: registered' "$LOGS/device.log"; then
      echo "MHS device did not register; see $LOGS/device.log"; exit 1
    fi
    echo 'AGH: http://127.0.0.1:43130/  Devices: http://127.0.0.1:43132/?lang=zh-CN'
    ;;
  status)
    for name in agh_mhs_daemon agh_mhs_web agh_mhs_navlive agh_mhs_device; do
      tmux has-session -t "=$name" 2>/dev/null && echo "$name running" || echo "$name stopped"
    done
    curl -fsS --max-time 3 http://127.0.0.1:43131/health || true
    ;;
  stop)
    # Only the three dedicated sessions started by this script.
    for name in agh_mhs_device agh_mhs_navlive agh_mhs_web; do
      tmux kill-session -t "=$name" 2>/dev/null || true
    done
    AGH_HOME="$AGH_HOME" node "$CLI" daemon stop
    tmux kill-session -t '=agh_mhs_daemon' 2>/dev/null || true
    ;;
  *) echo 'Usage: start-mhs.sh start|status|stop'; exit 2 ;;
esac
