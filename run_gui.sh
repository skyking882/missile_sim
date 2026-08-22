#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# `bash run_gui.sh` is non-login/non-interactive, so it skips ~/.bash_profile.
# macOS ships /usr/bin/python3 as 3.9; Homebrew must be searched first.
if [ -d /opt/homebrew/bin ]; then
  PATH="/opt/homebrew/bin:/opt/homebrew/sbin:${PATH}"
elif [ -x /usr/local/bin/brew ]; then
  PATH="/usr/local/bin:/usr/local/sbin:${PATH}"
fi
export PATH

python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

PYTHON=""
for candidate in \
  /opt/homebrew/bin/python3.14 \
  /opt/homebrew/bin/python3 \
  /usr/local/bin/python3.14 \
  /usr/local/bin/python3 \
  python3.14 python3.13 python3.12 python3.11 python3.10 \
  python3 python
do
  case "$candidate" in
    /*)
      if [ ! -x "$candidate" ]; then
        continue
      fi
      resolved=$candidate
      ;;
    *)
      if ! command -v "$candidate" >/dev/null 2>&1; then
        continue
      fi
      resolved=$(command -v "$candidate")
      ;;
  esac
  if python_is_supported "$resolved"; then
    PYTHON=$resolved
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "无法启动 GUI：未找到 Python 3.10 或更高版本。" >&2
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys; print("当前 python3 为 %s（%s）。" % (sys.version.split()[0], sys.executable))' >&2 || true
  fi
  echo "macOS 自带 python3 通常是 3.9。请安装较新版本：brew install python" >&2
  exit 1
fi

exec "$PYTHON" scripts/bootstrap_gui.py
