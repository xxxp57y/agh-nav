#!/usr/bin/env bash
# Public entry point starts only the isolated MHS workbench.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/mhs/start-mhs.sh" "${1:-start}"
