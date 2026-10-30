#!/usr/bin/env python3
"""Build a deterministic, allowlist-only source release."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
RELEASE_NAME = f"aim120a-local-model-{VERSION}"
DIST = ROOT / "dist"
RELEASE_DIR = DIST / RELEASE_NAME
ZIP_PATH = DIST / f"{RELEASE_NAME}.zip"
ZIP_HASH_PATH = DIST / f"{RELEASE_NAME}.zip.sha256"

FILES = {
    ".gitignore": ".gitignore",
    "LICENSE_PENDING.md": "LICENSE_PENDING.md",
    "README.md": "README.md",
    "README.en.md": "README.en.md",
    "VERSION": "VERSION",
    "V1_FREEZE.md": "V1_FREEZE.md",
    "docs/UPDATE_V1.0.1.md": "docs/UPDATE_V1.0.1.md",
    "pyproject.toml": "pyproject.toml",
    "requirements-gui.txt": "requirements-gui.txt",
    "run_gui.cmd": "run_gui.cmd",
    "run_gui.sh": "run_gui.sh",
    "configs/aim120a_v1.json": "configs/aim120a_v1.json",
    "configs/aim120a_v1_cases.json": "configs/aim120a_v1_cases.json",
    "config/defaults.json": "config/defaults.json",
    "config/profile_h2_runtime_defaults.json": "config/profile_h2_runtime_defaults.json",
    "config/missile_profile_catalog.json": "config/missile_profile_catalog.json",
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
    "src/aim120_model/alpha_servo.py": "src/aim120_model/alpha_servo.py",
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
    "src/aim120_model/observation.py": "src/aim120_model/observation.py",
    "src/aim120_model/propulsion.py": "src/aim120_model/propulsion.py",
    "src/aim120_model/profile_adapter.py": "src/aim120_model/profile_adapter.py",
    "src/aim120_model/profile_catalog.py": "src/aim120_model/profile_catalog.py",
    "src/aim120_model/public_api.py": "src/aim120_model/public_api.py",
    "src/aim120_model/effective_surface_core.py": "src/aim120_model/effective_surface_core.py",
    "src/aim120_model/surface_runtime.py": "src/aim120_model/surface_runtime.py",
    "config/effective_surface_defaults.json": "config/effective_surface_defaults.json",
    "tests/test_surface_runtime.py": "tests/test_surface_runtime.py",
    "src/aim120_model/radar_seeker.py": "src/aim120_model/radar_seeker.py",
    "src/aim120_model/simulator.py": "src/aim120_model/simulator.py",
    "src/aim120_model/target.py": "src/aim120_model/target.py",
    "src/aim120_model/tracking.py": "src/aim120_model/tracking.py",
    "src/aim120_model/trajectory.py": "src/aim120_model/trajectory.py",
    "src/aim120_model/units.py": "src/aim120_model/units.py",
    "src/missile_gui/__init__.py": "src/missile_gui/__init__.py",
    "src/missile_gui/library.py": "src/missile_gui/library.py",
    "src/missile_gui/server.py": "src/missile_gui/server.py",
    "src/missile_gui/static/app.css": "src/missile_gui/static/app.css",
    "src/missile_gui/static/app.js": "src/missile_gui/static/app.js",
    "src/missile_gui/static/index.html": "src/missile_gui/static/index.html",
    "tests/test_gui_v1.py": "tests/test_gui_v1.py",
    "tests/test_missile_profiles.py": "tests/test_missile_profiles.py",
    "tests/test_v1_regression.py": "tests/test_v1_regression.py",
}

# The profile set is generated from a versioned datamine checkout.  Keep the
# release allowlist explicit at build time while avoiding a hand-maintained
# profile-by-profile list.
for profile_path in sorted((ROOT / "missiles").glob("*.json")):
    relative = profile_path.relative_to(ROOT).as_posix()
    FILES[relative] = relative

# Both selectable profile sets are required by the packaged catalog.  Include
# their normalized profiles and source manifests, but never the raw datamine
# repositories: the default runtime does not load raw BLKX at startup.
for profile_set in ("2.57.1.107", "2.59.0.27"):
    profile_root = ROOT / "data" / "profile_sets" / profile_set
    manifest = profile_root / "data" / "aam_non_tvc_manifest.json"
    relative_manifest = manifest.relative_to(ROOT).as_posix()
    FILES[relative_manifest] = relative_manifest
    for profile_path in sorted((profile_root / "missiles").glob("*.json")):
        relative = profile_path.relative_to(ROOT).as_posix()
        FILES[relative] = relative

# The reconstructed candidate owns its arithmetic and data inside the package.
# Include only source and JSON-decoded resource formats, never local caches.
for candidate_path in sorted((ROOT / "src/aim120_model/reconstructed").rglob("*")):
    if candidate_path.is_file() and candidate_path.suffix in {".py", ".json", ".blk"}:
        relative = candidate_path.relative_to(ROOT).as_posix()
        FILES[relative] = relative
for test_name in ("test_reconstructed_infrared.py", "test_reconstructed_sarh.py", "test_reconstructed_runtime.py", "test_shared_responses.py"):
    relative = "tests/" + test_name
    FILES[relative] = relative

# MATLAB source, runnable models and pinned numerical fixtures.
for matlab_path in sorted((ROOT / "matlab").rglob("*")):
    relative = matlab_path.relative_to(ROOT)
    if any(part in {"outputs", "slprj", "__pycache__"} or part.startswith(".") for part in relative.parts):
        continue
    if matlab_path.is_file() and matlab_path.suffix in {".m", ".slx", ".json", ".py"}:
        FILES[relative.as_posix()] = relative.as_posix()

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
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", pyproject)
    project_version = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project_section.group(1), re.MULTILINE) if project_section else None
    if project_version is None or project_version.group(1) != VERSION:
        found = project_version.group(1) if project_version else "missing"
        raise RuntimeError(f"VERSION ({VERSION}) does not match [project].version ({found})")
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
