"""Configuration loading for JSON-compatible YAML files and JSON references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_data_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON file or a JSON-compatible YAML file without external dependencies."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            import yaml  # type: ignore
        except ImportError as yaml_exc:  # pragma: no cover - only used for non-JSON YAML
            raise ValueError(
                f"{file_path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from yaml_exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Top-level value in {file_path} must be an object") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Top-level value in {file_path} must be an object")
    return data


def load_model_config(path: str | Path) -> dict[str, Any]:
    return load_data_file(path)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = load_data_file(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise ValueError(f"{path} must contain a list of case objects under 'cases'")
    return cases


def find_case(cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for case in cases:
        if case.get("name") == name:
            return case
    available = ", ".join(str(case.get("name")) for case in cases)
    raise KeyError(f"Unknown case {name!r}; available cases: {available}")

