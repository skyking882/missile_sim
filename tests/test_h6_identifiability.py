import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from validate_h6_synthetic_identifiability import build_report  # noqa: E402


def test_synthetic_h6_gates_all_pass():
    report = build_report()
    assert report["status"] == "synthetic_identifiability_pass"
    assert all(report["acceptance_checks"].values())
