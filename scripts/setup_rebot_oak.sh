#!/usr/bin/env bash
# Workstation environment only. Never opens a camera or CAN bus.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${CASCADE_UV:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi
[[ -n "$UV_BIN" ]] || { echo 'Install uv before running this setup.' >&2; exit 1; }
export UV_PYTHON_INSTALL_DIR="$REPO/.runtime-rebot/python"
export UV_CACHE_DIR="$REPO/.runtime-rebot/uv-cache"
mkdir -p "$UV_PYTHON_INSTALL_DIR"
"$UV_BIN" python install 3.12.16 --no-bin
REBOT_PYTHON="$("$UV_BIN" python find --managed-python 3.12.16)"
if [[ ! -x "$REPO/.venv-rebot/bin/python" ]]; then
    "$UV_BIN" venv --python "$REBOT_PYTHON" "$REPO/.venv-rebot"
fi
"$UV_BIN" pip install --python "$REPO/.venv-rebot/bin/python" \
    -e "$REPO[dev,kinematics,arm,camera-oak,occupancy,llm,perception]" \
    'motorbridge==0.4.9' 'pin==4.1.0' 'depthai==3.11.0' 'warp-lang==1.18.0' 'numpy==2.5.4'
"$UV_BIN" pip freeze --python "$REPO/.venv-rebot/bin/python" \
    > "$REPO/.runtime-rebot/installed-requirements.txt"
"$REPO/.venv-rebot/bin/python" -c \
    'import cascade, motorbridge, pinocchio, depthai, warp; print("Environment ready:", cascade.__file__)'
"$REPO/.venv-rebot/bin/python" "$REPO/scripts/rebot_oak_workstation.py" prepare
