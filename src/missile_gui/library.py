"""Read-only scanner for missiles/*.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aim120_model.config import load_model_config
from aim120_model.profile_adapter import (
    build_h2_candidate_config,
    load_runtime_defaults,
    unsupported_model_types,
)


VALID_STATUSES = {"Validated", "Experimental", "Unsupported physics"}
REQUIRED_TEXT = ("id", "name", "country", "series", "status")


class MissileLibraryError(ValueError):
    pass


def _load_one(path: Path, project_dir: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissileLibraryError(
            f"{path.name} 不是合法 JSON（第 {exc.lineno} 行，第 {exc.colno} 列）。"
        ) from exc
    if not isinstance(profile, dict):
        raise MissileLibraryError(f"{path.name} 的顶层必须是对象。")
    if "missile_id" in profile:
        missile_id = profile.get("missile_id")
        if not isinstance(missile_id, str) or not missile_id.strip():
            raise MissileLibraryError(f"{path.name} 缺少字符串字段 missile_id。")
        if path.stem != missile_id:
            raise MissileLibraryError(f"{path.name} 的文件名必须与 missile_id 一致。")
        family = profile.get("model_family")
        if not isinstance(family, dict):
            raise MissileLibraryError(f"{path.name} 缺少 model_family 对象。")
        runtime = profile.get("runtime")
        if not isinstance(runtime, dict):
            raise MissileLibraryError(f"{path.name} 缺少 runtime 对象。")
        unsupported = unsupported_model_types(profile)
        # Every supported library profile uses the same Python H2 model layer,
        # then overlays the selected missile JSON.  A historical frozen config
        # remains a regression artifact, not a per-missile GUI bypass.
        if not unsupported:
            try:
                defaults_path = (project_dir / "config" / "profile_h2_runtime_defaults.json").resolve()
                defaults = load_runtime_defaults(str(defaults_path))
                profile["_model_config"], profile["_runtime_assumptions"] = build_h2_candidate_config(profile, defaults)
                profile["_runtime_boundary"] = profile["_model_config"].get(
                    "reference", {}
                ).get("runtime_boundary", defaults["boundary"])
                profile["_runtime_adapter"] = profile["_model_config"].get(
                    "runtime_adapter", defaults["runtime_name"]
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise MissileLibraryError(f"{path.name} 无法接入 profile H2 runtime：{exc}") from exc
        profile["_runtime_unsupported"] = unsupported
        if unsupported:
            display_status = "Unsupported physics"
            status_reason = "不支持的模型类型：" + "、".join(unsupported)
        elif profile.get("_model_config") is not None:
            display_status = "Validated" if profile.get("model_status") == "validated" else "Experimental"
            status_reason = "已使用公共 Python H2 模型层并叠加当前导弹 JSON；公共层和缺失字段的来源保留在导出结果中，仍未端到端验证。"
        else:
            display_status = "Validated" if profile.get("model_status") == "validated" else "Experimental"
            status_reason = runtime.get("notes", "")
        legacy_ids = {
            "us_aim_120a": "aim-120a",
            "cn_pl12": "pl-12",
            "su_r_77": "r-77",
        }
        # In-memory aliases keep older GUI callers working; JSON files and API
        # responses continue to use the unit-safe missile_id.
        profile["id"] = legacy_ids.get(missile_id, missile_id)
        profile["name"] = profile["display_name"]
        profile["series"] = profile["display_name"]
        profile["status"] = display_status
        profile["status_reason"] = status_reason
        profile["description"] = profile.get("provenance", {}).get("notes", "")
        profile["_source_file"] = str(path)
        return profile
    for key in REQUIRED_TEXT:
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise MissileLibraryError(f"{path.name} 缺少字符串字段 {key}。")
    if profile["status"] not in VALID_STATUSES:
        raise MissileLibraryError(f"{path.name} 的模型状态无效：{profile['status']}。")
    physics = profile.get("physics")
    if not isinstance(physics, dict):
        raise MissileLibraryError(f"{path.name} 缺少 physics 对象。")
    config_relative = physics.get("config")
    if config_relative:
        config_path = (project_dir / str(config_relative)).resolve()
        try:
            config_path.relative_to(project_dir.resolve())
        except ValueError as exc:
            raise MissileLibraryError(f"{path.name} 的模型路径超出项目目录。") from exc
        if not config_path.is_file():
            raise MissileLibraryError(f"{path.name} 引用的模型参数不存在：{config_relative}。")
        try:
            profile["_model_config"] = load_model_config(config_path)
        except (OSError, ValueError) as exc:
            raise MissileLibraryError(f"{path.name} 的模型参数无法读取：{exc}") from exc
    profile["_source_file"] = str(path)
    return profile


def scan_library(missiles_dir: Path, project_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    profiles: list[dict[str, Any]] = []
    errors: list[str] = []
    ids: set[str] = set()
    if not missiles_dir.is_dir():
        return [], [f"导弹目录不存在：{missiles_dir}"]
    for path in sorted(missiles_dir.glob("*.json")):
        try:
            profile = _load_one(path, project_dir)
            profile_id = profile.get("missile_id", profile.get("id"))
            if profile_id in ids:
                raise MissileLibraryError(f"导弹 ID 重复：{profile_id}。")
            ids.add(profile_id)
            profiles.append(profile)
        except (OSError, MissileLibraryError) as exc:
            errors.append(str(exc))
    profiles.sort(key=lambda item: (not bool(item.get("_model_config")), item.get("missile_id", item.get("id", ""))))
    return profiles, errors


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if "missile_id" in profile:
        family = profile["model_family"]
        unsupported = profile.get("_runtime_unsupported", [])
        runtime = profile.get("runtime", {})
        implemented = not unsupported and bool(profile.get("_model_config"))
        if unsupported:
            status = "Unsupported physics"
            status_reason = "不支持的模型类型：" + "、".join(unsupported)
        elif not implemented:
            status = "Unsupported physics"
            status_reason = runtime.get("notes", "当前本地求解器尚未接入该 profile。")
        else:
            status = "Validated" if profile.get("model_status") == "validated" else "Experimental"
            status_reason = profile.get("status_reason", runtime.get("notes", ""))
        config = profile.get("_model_config")
        parameters: dict[str, Any] = {}
        if isinstance(config, dict):
            parameters = {
                "model_label": config.get("model_label"),
                "aero_model_version": config.get("aero_model_version"),
                "control_model_version": config.get("control_model_version"),
                "lifetime_s": config.get("performance", {}).get("lifetime_s"),
                "maximum_range_m": config.get("performance", {}).get("maximum_distance_m"),
                "loft_enabled": config.get("guidance", {}).get("lofting_enabled"),
                "engine_stages": len(config.get("propulsion", {}).get("stages", [])),
            }
        return {
            "id": profile["missile_id"],
            "name": profile["display_name"],
            "country": profile["country"],
            "series": profile["display_name"],
            "status": status,
            "model_status": profile.get("model_status"),
            "status_reason": status_reason,
            "description": profile.get("provenance", {}).get("notes", ""),
            "runnable": implemented,
            "physics": {
                "engine_type": family.get("propulsion"),
                "control_type": family.get("control"),
                "guidance_type": family.get("guidance"),
            },
            "parameters": parameters,
            "runtime_adapter": profile.get("_runtime_adapter", "frozen_config" if runtime.get("implemented") else None),
            "runtime_assumption_count": len(profile.get("_runtime_assumptions", [])),
        }
    config = profile.get("_model_config")
    parameters: dict[str, Any] = {}
    if isinstance(config, dict):
        parameters = {
            "model_label": config.get("model_label"),
            "aero_model_version": config.get("aero_model_version"),
            "control_model_version": config.get("control_model_version"),
            "lifetime_s": config.get("performance", {}).get("lifetime_s"),
            "maximum_range_m": config.get("performance", {}).get("maximum_distance_m"),
            "loft_enabled": config.get("guidance", {}).get("lofting_enabled"),
            "engine_stages": len(config.get("propulsion", {}).get("stages", [])),
        }
    return {
        "id": profile["id"],
        "name": profile["name"],
        "country": profile["country"],
        "series": profile["series"],
        "status": profile["status"],
        "status_reason": profile.get("status_reason", ""),
        "description": profile.get("description", ""),
        "runnable": bool(profile.get("_model_config")) and profile["status"] != "Unsupported physics",
        "physics": {
            "engine_type": profile["physics"].get("engine_type"),
            "control_type": profile["physics"].get("control_type"),
        },
        "parameters": parameters,
    }


__all__ = ["MissileLibraryError", "public_profile", "scan_library"]
