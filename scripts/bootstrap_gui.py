#!/usr/bin/env python3
"""Create/reuse .venv, verify optional dependencies, and launch the GUI."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path


MINIMUM = (3, 10)
PROJECT_DIR = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements-gui.txt"
READY_MARKER = VENV_DIR / ".gui-deps-ready"


def fail(message: str, detail: str | None = None) -> int:
    print(f"无法启动 GUI：{message}", file=sys.stderr)
    if detail:
        print(detail, file=sys.stderr)
    return 1


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def dependency_fingerprint() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def has_installable_requirements() -> bool:
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def main() -> int:
    if sys.version_info < MINIMUM:
        return fail(
            "Python 版本不满足要求。",
            f"需要 Python {MINIMUM[0]}.{MINIMUM[1]} 或更高版本；当前为 {sys.version.split()[0]}。",
        )
    if not REQUIREMENTS.is_file():
        return fail("缺少 requirements-gui.txt，无法检查 GUI 依赖。")
    try:
        if not venv_python().is_file():
            print("正在创建隔离环境 .venv …", flush=True)
            venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
        interpreter = venv_python()
        fingerprint = dependency_fingerprint()
        installed = READY_MARKER.read_text(encoding="utf-8").strip() if READY_MARKER.is_file() else ""
        if installed != fingerprint:
            if has_installable_requirements():
                print("正在安装 GUI 可选依赖 …", flush=True)
                completed = subprocess.run(
                    [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
                    cwd=PROJECT_DIR,
                    check=False,
                )
                if completed.returncode:
                    return fail("GUI 依赖安装失败。", "请检查终端中的 pip 错误，然后重试。")
            else:
                print("GUI 使用 Python 标准库，无额外包需要下载。", flush=True)
            READY_MARKER.write_text(fingerprint + "\n", encoding="utf-8")
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("创建环境或安装 GUI 依赖失败。", repr(exc))

    environment = os.environ.copy()
    source_dir = str(PROJECT_DIR / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_dir + (os.pathsep + existing if existing else "")
    print("正在启动本地 GUI …", flush=True)
    try:
        return subprocess.call([str(interpreter), "-m", "missile_gui.server"], cwd=PROJECT_DIR, env=environment)
    except OSError as exc:
        return fail("无法运行本地 GUI 服务。", repr(exc))


if __name__ == "__main__":
    raise SystemExit(main())
