#!/usr/bin/env python3
"""Build a deterministic, allowlist-only v1.0.0 source release."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
RELEASE_NAME = f"aim120a-local-model-{VERSION}"
DIST = ROOT / "dist"
RELEASE_DIR = DIST / RELEASE_NAME
ZIP_PATH = DIST / f"{RELEASE_NAME}.zip"
ZIP_HASH_PATH = DIST / f"{RELEASE_NAME}.zip.sha256"

FILES = {
    ".gitignore": ".gitignore",
    "LICENSE_PENDING.md": "LICENSE_PENDING.md",
    "OPEN_SOURCE_README.md": "README.md",
    "VERSION": "VERSION",
    "V1_FREEZE.md": "MODEL_CARD.md",
    "pyproject.toml": "pyproject.toml",
    "requirements-gui.txt": "requirements-gui.txt",
    "run_gui.cmd": "run_gui.cmd",
    "run_gui.sh": "run_gui.sh",
    "configs/aim120a_v1.json": "configs/aim120a_v1.json",
    "configs/aim120a_v1_cases.json": "configs/aim120a_v1_cases.json",
    "config/defaults.json": "config/defaults.json",
    "config/profile_h2_runtime_defaults.json": "config/profile_h2_runtime_defaults.json",
    "scenarios/default_head_on.json": "scenarios/default_head_on.json",
    "schemas/missile_profile.schema.json": "schemas/missile_profile.schema.json",
    "data/aam_non_tvc_manifest.json": "data/aam_non_tvc_manifest.json",
    "examples/run_v1.py": "examples/run_v1.py",
    "scripts/import_datamine_missiles.py": "scripts/import_datamine_missiles.py",
    "scripts/run_test_functions.py": "scripts/run_test_functions.py",
    "scripts/bootstrap_gui.py": "scripts/bootstrap_gui.py",
    "missile_lab/__init__.py": "missile_lab/__init__.py",
    "missile_lab/__main__.py": "missile_lab/__main__.py",
    "missile_lab/cli.py": "missile_lab/cli.py",
    "missile_lab/validator.py": "missile_lab/validator.py",
    "src/aim120_model/__init__.py": "src/aim120_model/__init__.py",
    "src/aim120_model/aerodynamics.py": "src/aim120_model/aerodynamics.py",
    "src/aim120_model/atmosphere.py": "src/aim120_model/atmosphere.py",
    "src/aim120_model/config.py": "src/aim120_model/config.py",
    "src/aim120_model/control.py": "src/aim120_model/control.py",
    "src/aim120_model/drag_models.py": "src/aim120_model/drag_models.py",
    "src/aim120_model/dynamics.py": "src/aim120_model/dynamics.py",
    "src/aim120_model/effective_controller.py": "src/aim120_model/effective_controller.py",
    "src/aim120_model/events.py": "src/aim120_model/events.py",
    "src/aim120_model/guidance.py": "src/aim120_model/guidance.py",
    "src/aim120_model/h2_dynamics.py": "src/aim120_model/h2_dynamics.py",
    "src/aim120_model/h2_simulator.py": "src/aim120_model/h2_simulator.py",
    "src/aim120_model/math3d.py": "src/aim120_model/math3d.py",
    "src/aim120_model/metrics.py": "src/aim120_model/metrics.py",
    "src/aim120_model/propulsion.py": "src/aim120_model/propulsion.py",
    "src/aim120_model/profile_adapter.py": "src/aim120_model/profile_adapter.py",
    "src/aim120_model/public_api.py": "src/aim120_model/public_api.py",
    "src/aim120_model/simulator.py": "src/aim120_model/simulator.py",
    "src/aim120_model/target.py": "src/aim120_model/target.py",
    "src/aim120_model/units.py": "src/aim120_model/units.py",
    "src/missile_gui/__init__.py": "src/missile_gui/__init__.py",
    "src/missile_gui/library.py": "src/missile_gui/library.py",
    "src/missile_gui/server.py": "src/missile_gui/server.py",
    "src/missile_gui/static/app.css": "src/missile_gui/static/app.css",
    "src/missile_gui/static/app.js": "src/missile_gui/static/app.js",
    "src/missile_gui/static/index.html": "src/missile_gui/static/index.html",
    "tests/test_gui_v1.py": "tests/test_gui_v1.py",
    "tests/test_missile_profiles.py": "tests/test_missile_profiles.py",
    "tests/test_v1_regression.py": "tests/test_v1_regression.py"
}

# The profile set is generated from a versioned datamine checkout.  Keep the
# release allowlist explicit at build time while avoiding a hand-maintained
# profile-by-profile list.
for profile_path in sorted((ROOT / "missiles").glob("*.json")):
    relative = profile_path.relative_to(ROOT).as_posix()
    FILES[relative] = relative

FORBIDDEN_BYTE_PATTERNS = (
    b"cf-turnstile-response",
    b"token_sha256",
    b"edge_cdp_profile",
    b"network_evidence",
    b"authorization: bearer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_previous_release() -> None:
    dist_resolved = DIST.resolve()
    release_resolved = RELEASE_DIR.resolve()
    zip_resolved = ZIP_PATH.resolve()
    zip_hash_resolved = ZIP_HASH_PATH.resolve()
    if release_resolved.parent != dist_resolved or release_resolved.name != RELEASE_NAME:
        raise RuntimeError("refusing unsafe release-directory removal")
    if zip_resolved.parent != dist_resolved or zip_resolved.name != f"{RELEASE_NAME}.zip":
        raise RuntimeError("refusing unsafe release-archive removal")
    if zip_hash_resolved.parent != dist_resolved or zip_hash_resolved.name != f"{RELEASE_NAME}.zip.sha256":
        raise RuntimeError("refusing unsafe release-hash removal")
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    if ZIP_HASH_PATH.exists():
        ZIP_HASH_PATH.unlink()


def copy_allowlist() -> None:
    for source_name, destination_name in FILES.items():
        source = ROOT / source_name
        destination = RELEASE_DIR / destination_name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"required regular file missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def scan_release() -> None:
    for path in RELEASE_DIR.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is not allowed in release: {path}")
        if not path.is_file():
            continue
        content = path.read_bytes().lower()
        for pattern in FORBIDDEN_BYTE_PATTERNS:
            if pattern in content:
                raise RuntimeError(f"forbidden private-evidence pattern in {path}: {pattern!r}")


def write_manifest() -> Path:
    entries = []
    for path in sorted(item for item in RELEASE_DIR.rglob("*") if item.is_file()):
        relative = path.relative_to(RELEASE_DIR).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "release_name": RELEASE_NAME,
        "release_version": VERSION,
        "file_count_excluding_manifest": len(entries),
        "files": entries,
        "excluded_categories": [
            "raw external-service captures and browser state",
            "credentials and request tokens",
            "generated outputs and fit workspaces",
            "game installation files",
            "nonessential experiment scripts and notebooks"
        ]
    }
    path = RELEASE_DIR / "RELEASE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_deterministic_zip() -> None:
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in RELEASE_DIR.rglob("*") if item.is_file()):
            relative = path.relative_to(RELEASE_DIR).as_posix()
            info = ZipInfo(f"{RELEASE_NAME}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            mode = 0o100755 if relative == "run_gui.sh" else 0o100644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    if (ROOT / "VERSION").read_text(encoding="utf-8").strip() != VERSION:
        raise RuntimeError("VERSION file does not match release builder")
    DIST.mkdir(parents=True, exist_ok=True)
    remove_previous_release()
    RELEASE_DIR.mkdir(parents=True)
    copy_allowlist()
    scan_release()
    manifest = write_manifest()
    write_deterministic_zip()
    zip_digest = sha256(ZIP_PATH)
    ZIP_HASH_PATH.write_text(f"{zip_digest}  {ZIP_PATH.name}\n", encoding="ascii")
    print(f"release_dir={RELEASE_DIR}")
    print(f"manifest_sha256={sha256(manifest)}")
    print(f"zip={ZIP_PATH}")
    print(f"zip_sha256={zip_digest}")
    print(f"zip_sha256_file={ZIP_HASH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
