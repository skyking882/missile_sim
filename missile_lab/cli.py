"""Command line interface for the profile gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validator import validate_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="missile_lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-profiles", help="验证所有导弹 profile 和冻结回归")
    validate_parser.add_argument("--project-dir", type=Path, default=None)
    validate_parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if args.command == "validate-profiles":
        report = validate_project(args.project_dir)
        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            status = "PASS" if report["passed"] else "FAIL"
            print(f"PROFILE VALIDATION {status}")
            print(f"profiles: {report['profile_count']}")
            print(f"schema: {report['schema_passed']}/{report['profile_count']}")
            print(f"smoke: {report['smoke_passed']}/{report['profile_count']}")
            print(f"read-only unsupported profiles: {report.get('unsupported_profile_count', 0)}")
            print(f"AIM-120A frozen regression: {'PASS' if report['frozen_regression_passed'] else 'FAIL'}")
            for issue in report["issues"][:30]:
                print(f"- {issue}")
            if len(report["issues"]) > 30:
                print(f"- ... 还有 {len(report['issues']) - 30} 条错误")
        return 0 if report["passed"] else 1

    parser.error("未知命令")
    return 2
