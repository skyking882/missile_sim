#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/bootstrap_gui.py
elif command -v python >/dev/null 2>&1; then
  exec python scripts/bootstrap_gui.py
else
  echo "无法启动 GUI：未找到 Python 3.10 或更高版本。" >&2
  exit 1
fi
