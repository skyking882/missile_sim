"""Audit the final log only after the native parser process has exited.

Its asynchronous logger can finish writing after the in-process flush call.
Do not filter away E-level messages; separately count known include advisories.
"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
lines = (root / "parser.log").read_text(encoding="utf-8", errors="replace").splitlines()
errors = [line for line in lines if "[E]" in line]
advisories = [line for line in errors if "including binary file '" in line and "not fastest codepath" in line]
manifest.setdefault("in_process_parser_error_count", manifest["parser_error_count"])
manifest["parser_error_count"] = len(errors)
manifest["parser_error_examples"] = errors[:20]
manifest["binary_include_advisory_count"] = len(advisories)
manifest["other_parser_error_count"] = len(errors) - len(advisories)
manifest["weapon_sync_error_count"] = sum("Failed to read We payload" in line for line in errors)
manifest["log_audit_phase"] = "after parser process exit"
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"Final log: {len(errors)} E-level messages, including {len(advisories)} binary-include performance advisories")
