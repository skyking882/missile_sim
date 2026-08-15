"""Save a JSON payload copied from the StatShark browser session.

This is intentionally a small ingestion helper so browser captures remain raw
JSON artifacts while the capture itself is performed in the browser UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    destination = Path(args.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
