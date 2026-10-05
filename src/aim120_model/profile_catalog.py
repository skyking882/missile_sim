"""Resolve versioned missile profiles and their matching raw datamine source."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_RELATIVE_PATH = Path("config/missile_profile_catalog.json")


def _project_path(project_dir: Path, relative: str, label: str) -> Path:
    root = project_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within the project directory") from exc
    return path


@lru_cache(maxsize=8)
def _read_catalog(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported missile profile catalog")
    sets = value.get("profile_sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError("missile profile catalog has no profile sets")
    return value


def load_profile_catalog(project_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the active profile set and verify its source manifest identity."""

    catalog_path = _project_path(
        project_dir,
        CATALOG_RELATIVE_PATH.as_posix(),
        "catalog path",
    )
    catalog = _read_catalog(str(catalog_path))
    active_name = catalog.get("active_profile_set")
    profile_set = catalog["profile_sets"].get(active_name)
    if not isinstance(profile_set, dict):
        raise ValueError(f"active profile set {active_name!r} is not declared")
    profile_dir = _project_path(project_dir, str(profile_set["profiles_path"]), "profiles path")
    manifest_path = _project_path(project_dir, str(profile_set["manifest_path"]), "manifest path")
    if not profile_dir.is_dir():
        raise ValueError(f"active profile directory does not exist: {profile_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    if (
        source.get("version") != profile_set.get("source_version")
        or source.get("commit") != profile_set.get("source_commit")
    ):
        raise ValueError("active profile manifest version/commit does not match the catalog")
    resolved = dict(profile_set)
    resolved.update({"name": active_name, "profiles_dir": profile_dir, "manifest": manifest})
    return catalog, resolved


def datamine_path_for_profile(project_dir: Path, profile: dict[str, Any]) -> Path:
    """Resolve the raw checkout that matches a profile's recorded version.

    There is deliberately no fallback to the current checkout.  Unknown or
    mismatched provenance is an error instead of a silent cross-version read.
    """

    provenance = profile.get("provenance", {})
    version = provenance.get("source_version")
    commit = provenance.get("source_commit")
    catalog_path = _project_path(
        project_dir,
        CATALOG_RELATIVE_PATH.as_posix(),
        "catalog path",
    )
    catalog = _read_catalog(str(catalog_path))
    matched = [
        item
        for item in catalog["profile_sets"].values()
        if item.get("source_version") == version and item.get("source_commit") == commit
    ]
    if len(matched) != 1:
        raise ValueError(
            f"no unique raw datamine checkout matches profile source {version!r}/{commit!r}"
        )
    root = _project_path(project_dir, str(matched[0]["datamine_path"]), "datamine path")
    version_file = root / "version"
    if not version_file.is_file() or version_file.read_text(encoding="utf-8").strip() != version:
        raise ValueError(f"raw datamine version does not match profile source {version!r}")
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify raw datamine commit for {version!r}") from exc
    if actual_commit != commit:
        raise ValueError(f"raw datamine commit does not match profile source {commit}")
    return root


__all__ = ["datamine_path_for_profile", "load_profile_catalog"]
