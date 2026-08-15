#!/usr/bin/env python3
"""Small stdlib-only fallback for environments without pytest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    failures: list[tuple[str, str, str]] = []
    count = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        module_name = f"fallback_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            failures.append((path.name, "<module>", "could not load module"))
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - fallback reporting
            failures.append((path.name, "<module>", repr(exc)))
            continue
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            count += 1
            try:
                function()
            except Exception as exc:  # pragma: no cover - fallback reporting
                failures.append((path.name, name, repr(exc)))
    print(f"ran {count} test functions")
    if failures:
        for path, name, error in failures:
            print(f"FAIL {path}::{name}: {error}")
        return 1
    print("all fallback tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

