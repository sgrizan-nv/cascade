#!/usr/bin/env bash
# Repository-local workstation entry point. No motor commands on setup/check.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == setup ]]; then exec bash "$REPO/scripts/setup_rebot_oak.sh"; fi
PY="$REPO/.venv-rebot/bin/python"
[[ -x "$PY" ]] || { echo 'Run: bash scripts/rebot_oak.sh setup' >&2; exit 1; }
export PYTHONPATH="$REPO/src:$REPO"
export XDG_CACHE_HOME="$REPO/.runtime-rebot/cache"
exec "$PY" "$REPO/scripts/rebot_oak_workstation.py" "$@"
