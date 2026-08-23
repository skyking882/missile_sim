"""Dependency-free profile validator and offline smoke runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = {
    "dynamics": {"h2_reduced_order"},
    "propulsion": {"staged_solid_rocket"},
    "aerodynamics": {"conventional_fin"},
    "control": {"aerodynamic_fin"},
    "guidance": {"pn", "pn_loft"},
}
RECOGNIZED_MODEL_TYPES = {
    "dynamics": {"h2_reduced_order", "full_6dof"},
    "propulsion": {"staged_solid_rocket", "ramjet"},
    "aerodynamics": {"conventional_fin", "grid_fin"},
    "control": {"aerodynamic_fin", "thrust_vector", "jet_vane"},
    "guidance": {"pn", "pn_loft", "beam_riding", "command_guidance"},
}
SOURCE_KINDS = {"datamine", "identified", "assumed", "derived", "calibrated"}
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
FORBIDDEN_CANONICAL_KEYS = {
    "mass",
    "massEnd",
    "massEnd1",
    "length",
    "caliber",
    "wingAreaMult",
    "finsAoaHor",
    "finsAoaVer",
    "finsLatAccel",
    "timeFire",
    "timeFire1",
    "force",
    "force1",
    "timeLife",
    "maxDistance",
}


class SchemaError(ValueError):
    pass


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/"):
        raise SchemaError(f"不支持的 schema 引用：{ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaError(f"schema 引用不是对象：{ref}")
    return node


def _schema_check(value: Any, schema: dict[str, Any], path: str, root: dict[str, Any]) -> list[str]:
    schema = _resolve_ref(schema, root)
    issues: list[str] = []
    if "const" in schema and value != schema["const"]:
        issues.append(f"{path}: 必须等于 {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        issues.append(f"{path}: 不在允许值 {schema['enum']!r} 中")
    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(value, item) for item in expected_types):
            return [f"{path}: 类型错误，期望 {expected_types!r}，实际 {type(value).__name__}"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            issues.append(f"{path}: 数值必须有限")
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(f"{path}: 不能小于 {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            issues.append(f"{path}: 必须大于 {schema['exclusiveMinimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(f"{path}: 不能大于 {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(f"{path}: 字符串不能为空")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            issues.append(f"{path}: 不符合模式 {schema['pattern']!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(f"{path}: 数组至少需要 {schema['minItems']} 项")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(encoded) != len(set(encoded)):
                issues.append(f"{path}: 数组项目必须唯一")
        if "items" in schema:
            for index, item in enumerate(value):
                issues.extend(_schema_check(item, schema["items"], f"{path}[{index}]", root))
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                issues.append(f"{path}: 缺少必填字段 {required}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value:
                if key not in properties:
                    issues.append(f"{path}: 不允许字段 {key}")
        for key, child in properties.items():
            if key in value:
                issues.extend(_schema_check(value[key], child, f"{path}.{key}", root))
        if isinstance(additional, dict):
            for key, child_value in value.items():
                if key not in properties:
                    issues.extend(_schema_check(child_value, additional, f"{path}.{key}", root))
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            issues.append(f"{path}: 至少需要 {schema['minProperties']} 个字段")
    return issues


def _finite_walk(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: 包含 NaN/Infinity"]
    if isinstance(value, dict):
        issues: list[str] = []
        for key, item in value.items():
            issues.extend(_finite_walk(item, f"{path}.{key}"))
        return issues
    if isinstance(value, list):
        issues = []
        for index, item in enumerate(value):
            issues.extend(_finite_walk(item, f"{path}[{index}]"))
        return issues
    return []


def _source_map_check(source_map: Any, path: str) -> list[str]:
    if not isinstance(source_map, dict) or not source_map:
        return [f"{path}: 来源说明不能为空"]
    issues: list[str] = []
    for key, entry in source_map.items():
        if not isinstance(entry, dict):
            issues.append(f"{path}.{key}: 来源说明必须是对象")
            continue
        if entry.get("kind") not in SOURCE_KINDS:
            issues.append(f"{path}.{key}: kind 必须是 {sorted(SOURCE_KINDS)} 之一")
        if "source_field" not in entry or "notes" not in entry or not str(entry.get("notes", "")).strip():
            issues.append(f"{path}.{key}: 必须有 source_field 和非空 notes")
    return issues


def _custom_checks(profile: dict[str, Any], path: Path) -> list[str]:
    issues: list[str] = []
    if not ID_PATTERN.fullmatch(str(profile.get("missile_id", ""))):
        issues.append(f"{path.name}: missile_id 只能使用小写字母、数字和下划线")
    if path.stem != profile.get("missile_id"):
        issues.append(f"{path.name}: 文件名必须与 missile_id 一致")
    family = profile.get("model_family", {})
    unsupported_axes: list[str] = []
    for axis, allowed in SUPPORTED_MODEL_TYPES.items():
        value = family.get(axis)
        if value not in RECOGNIZED_MODEL_TYPES[axis]:
            issues.append(f"{path.name}: 不支持的模型类型 {axis}={family.get(axis)!r}")
        elif value not in allowed:
            unsupported_axes.append(f"{axis}={value}")
    runtime = profile.get("runtime", {})
    if unsupported_axes and (runtime.get("implemented") or profile.get("model_status") == "validated"):
        issues.append(f"{path.name}: 未接入模型类型不能标记为可运行或 validated：{'、'.join(unsupported_axes)}")
    loft_enabled = profile.get("guidance", {}).get("lofting_enabled")
    expected_guidance = "pn_loft" if loft_enabled else "pn"
    if family.get("guidance") in {"pn", "pn_loft"} and family.get("guidance") != expected_guidance:
        issues.append(f"{path.name}: model_family.guidance 与 guidance.lofting_enabled 不一致")
    if profile.get("guidance", {}).get("guidance_type") != family.get("guidance"):
        issues.append(f"{path.name}: guidance.guidance_type 必须与 model_family.guidance 一致")
    if profile.get("model_status") != profile.get("validation", {}).get("status"):
        issues.append(f"{path.name}: model_status 与 validation.status 不一致")
    geometry = profile.get("geometry", {})
    initial_mass = geometry.get("initial_mass_kg")
    stages = profile.get("propulsion", {}).get("stages", [])
    if isinstance(initial_mass, (int, float)) and not isinstance(initial_mass, bool):
        total_mass_lost = sum(float(stage.get("mass_lost_kg", 0.0)) for stage in stages if isinstance(stage, dict))
        if total_mass_lost < 0 or total_mass_lost >= initial_mass:
            issues.append(f"{path.name}: 发动机总失重必须为非负且小于初始质量")
    forbidden_seen: list[str] = []

    def scan_keys(value: Any, location: str, raw_allowed: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not raw_allowed and key in FORBIDDEN_CANONICAL_KEYS:
                    forbidden_seen.append(f"{location}.{key}")
                scan_keys(child, f"{location}.{key}", raw_allowed)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan_keys(child, f"{location}[{index}]", raw_allowed)

    scan_keys(profile, "$", False)
    if forbidden_seen:
        issues.append(f"{path.name}: 存在未带单位的标准字段 {', '.join(forbidden_seen[:5])}")
    for field in ("aerodynamics", "guidance", "control", "performance"):
        issues.extend(_source_map_check(profile.get(field, {}).get("parameter_sources"), f"{path.name}.{field}.parameter_sources"))
    for index, stage in enumerate(stages):
        issues.extend(_source_map_check(stage.get("parameter_sources"), f"{path.name}.propulsion.stages[{index}].parameter_sources"))
    for index, stage in enumerate(stages):
        if stage.get("thrust_n", 0) < 0 or stage.get("mass_lost_kg", 0) < 0 or stage.get("duration_s", 0) <= 0:
            issues.append(f"{path.name}: stage {index} 的时间/推力/失重不合法")
    return issues


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_profile_smoke(profile: dict[str, Any], defaults: dict[str, Any]) -> tuple[bool, str]:
    """Run a deliberately small contract smoke, not a trajectory validation."""

    try:
        dt = float(defaults["time_step_s"])
        gravity = float(defaults["gravity_mps2"])
        duration = min(float(defaults["smoke_case"]["duration_s"]), float(profile["performance"]["lifetime_s"]))
        mass = float(profile["geometry"]["initial_mass_kg"])
        speed = float(defaults["smoke_case"]["initial_speed_mps"])
        distance = 0.0
        elapsed = 0.0
        stages = profile["propulsion"]["stages"]
        for stage in stages:
            duration_s = float(stage["duration_s"])
            thrust = float(stage["thrust_n"])
            mass_lost = float(stage["mass_lost_kg"])
            steps = max(1, int(math.ceil(min(duration_s, duration) / dt)))
            for _ in range(steps):
                if elapsed >= duration:
                    break
                step = min(dt, duration - elapsed, duration_s)
                acceleration = thrust / max(mass, 1e-9)
                speed += acceleration * step
                distance += speed * step
                mass -= mass_lost * (step / duration_s)
                elapsed += step
        if elapsed < duration:
            distance += speed * (duration - elapsed)
        if not all(math.isfinite(x) for x in (mass, speed, distance, gravity)):
            return False, "smoke 产生非有限数值"
        if mass <= 0 or speed <= 0 or distance <= 0:
            return False, "smoke 的质量/速度/位移未保持正值"
        return True, "profile_contract_smoke"
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return False, f"smoke 异常：{exc}"


def _frozen_regression(project_dir: Path) -> tuple[bool, str]:
    src_dir = project_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        config_module = importlib.import_module("aim120_model.config")
        simulator_module = importlib.import_module("aim120_model.h2_simulator")
        config = config_module.load_model_config(project_dir / "configs" / "aim120a_v1.json")
        cases = config_module.load_cases(project_dir / "configs" / "aim120a_v1_cases.json")
        expected = {"head_on_10deg_12km": 10.553325, "head_on_38deg_15km": 17.242967}
        simulator = simulator_module.H2Simulator(config)
        for case in cases:
            result = simulator.run(case)
            if result.get("event_type") != "fuse":
                return False, f"{case['name']}: 终止事件变为 {result.get('event_type')!r}"
            if abs(float(result["terminal_time_s"]) - expected[case["name"]]) >= 0.001:
                return False, f"{case['name']}: 冻结终点时间发生变化"
            if abs(float(result["samples"][-1]["distance_to_target_m"]) - 12.0) >= 0.001:
                return False, f"{case['name']}: 冻结终点距离发生变化"
        return True, "frozen_v1_trajectory_anchors"
    except Exception as exc:  # pragma: no cover - diagnostic boundary
        return False, f"冻结回归无法运行：{exc}"


def validate_project(project_dir: Path | None = None) -> dict[str, Any]:
    project_dir = (project_dir or Path(__file__).resolve().parents[1]).resolve()
    schema_path = project_dir / "schemas" / "missile_profile.schema.json"
    missiles_dir = project_dir / "missiles"
    defaults_path = project_dir / "config" / "defaults.json"
    issues: list[str] = []
    schema_passed = 0
    smoke_passed = 0
    profile_count = 0
    unsupported_profile_count = 0
    try:
        schema = _load_json(schema_path)
        defaults = _load_json(defaults_path)
    except Exception as exc:
        return {
            "passed": False,
            "profile_count": 0,
            "schema_passed": 0,
            "smoke_passed": 0,
            "frozen_regression_passed": False,
            "issues": [f"无法读取校验依赖：{exc}"],
        }
    paths = sorted(missiles_dir.glob("*.json")) if missiles_dir.is_dir() else []
    profile_count = len(paths)
    ids: set[str] = set()
    for path in paths:
        try:
            profile = _load_json(path)
        except Exception as exc:
            issues.append(f"{path.name}: JSON 读取失败：{exc}")
            continue
        if not isinstance(profile, dict):
            issues.append(f"{path.name}: 顶层必须是对象")
            continue
        profile_issues = _schema_check(profile, schema, path.name, schema)
        profile_issues.extend(_finite_walk(profile))
        profile_issues.extend(_custom_checks(profile, path))
        family = profile.get("model_family", {})
        if any(family.get(axis) not in SUPPORTED_MODEL_TYPES[axis] for axis in SUPPORTED_MODEL_TYPES):
            unsupported_profile_count += 1
        if profile.get("missile_id") in ids:
            profile_issues.append(f"{path.name}: missile_id 重复")
        ids.add(profile.get("missile_id"))
        if profile_issues:
            issues.extend(profile_issues)
        else:
            schema_passed += 1
        passed_smoke, smoke_message = run_profile_smoke(profile, defaults)
        if passed_smoke:
            smoke_passed += 1
        else:
            issues.append(f"{path.name}: {smoke_message}")
    frozen_passed, frozen_message = _frozen_regression(project_dir)
    if not frozen_passed:
        issues.append(frozen_message)
    return {
        "passed": not issues and profile_count > 0,
        "profile_count": profile_count,
        "schema_passed": schema_passed,
        "smoke_passed": smoke_passed,
        "unsupported_profile_count": unsupported_profile_count,
        "frozen_regression_passed": frozen_passed,
        "issues": issues,
    }


__all__ = ["validate_project", "run_profile_smoke"]
