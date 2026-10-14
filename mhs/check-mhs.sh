#!/usr/bin/env bash
# Official protocol scenarios against the dedicated simulator; does not run motion scenarios.
set -euo pipefail
BASE="${AGH_MHS_BASE:?Set AGH_MHS_BASE to your deployment directory}"
PY="${AGH_MHS_NAVROOT:-$BASE/agh-nav}/.venv/bin/python"
export PYTHONPATH="${AGH_MHS_RUNTIME:-$BASE/agh-mhs-runtime}/packages/mhs/python/src"
LOGS="$BASE/agh-mhs-logs"
mkdir -p "$LOGS"
SID=$("$PY" - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:43131/state', timeout=3) as r:
    print(json.load(r)['session_id'])
PY
)
CHECK_PID=''
BRIDGE_PID=''
cleanup() {
  if [ -n "$BRIDGE_PID" ]; then kill "$BRIDGE_PID" 2>/dev/null || true; fi
  if [ -n "$CHECK_PID" ]; then kill "$CHECK_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
"$PY" -u -m agnes_mhs.check --port 43133 --profile core,streaming,maps,perception,derived \
  --no-motion --replace-wait 2 --wait 30 --report "$LOGS/conformance.json" > "$LOGS/conformance.log" 2>&1 &
CHECK_PID=$!
sleep 1
"$PY" -u "${AGH_MHS_NAVROOT:-$BASE/agh-nav}/mhs/nav_device.py" --hub ws://127.0.0.1:43133 \
  --navlive http://127.0.0.1:43131 --session "$SID" --device-id habitat-check \
  > "$LOGS/check-device.log" 2>&1 &
BRIDGE_PID=$!
set +e
wait "$CHECK_PID"
RESULT=$?
set -e
tail -50 "$LOGS/conformance.log"
exit "$RESULT"
