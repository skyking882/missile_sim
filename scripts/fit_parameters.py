#!/usr/bin/env python3
"""Scaffold for Phase 6; deliberately does not fit parameters yet."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 fitting scaffold")
    parser.add_argument("--allow", action="store_true", help="reserved; fitting remains disabled in milestone 1")
    parser.parse_args()
    print("Parameter fitting is intentionally disabled for milestone 1.")
    print("Reason: explicit parameters and model behavior must be validated before staged identification.")
    print("No StatShark calculation was performed and no parameters were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

