"""Batch-import non-TVC air-to-air missile profiles from War Thunder Datamine.

The importer intentionally keeps the raw checkout outside the game files.  It
deduplicates launcher/default/platform variants by ``rocket.bulletName`` while
retaining every source path and hash in provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


G0 = 9.80665
AIR_TO_AIR_ICON_PATTERN = re.compile(r"^missile_type_[a-z]_air_to_air")
AIR_TO_AIR_ICON_PREFIX = "missile_air_to_air"
UNSUPPORTED_FEATURE_PATTERNS = (
    ("thrust_vector", re.compile(r"thrust.?vector|vector.?thrust", re.I)),
    ("jet_vane", re.compile(r"jet.?vane", re.I)),
    ("grid_fin", re.compile(r"grid.?fin", re.I)),
    ("ramjet", re.compile(r"ramjet", re.I)),
    ("beam_riding", re.compile(r"beam.?riding", re.I)),
    ("command_guidance", re.compile(r"command.?guidance", re.I)),
    ("full_6dof", re.compile(r"6.?dof|full.?six.?dof", re.I)),
)
COUNTRY_PREFIXES = {
    "ar": "Argentina",
    "br": "Brazil",
    "cn": "China",
    "de": "Germany",
    "fr": "France",
    "il": "Israel",
    "ir": "Iran",
    "it": "Italy",
    "jp": "Japan",
    "r": "South Africa",
    "ro": "Romania",
    "su": "USSR / Russia",
    "swd": "Sweden",
    "sws": "Sweden",
    "uk": "United Kingdom",
    "us": "USA",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _source(kind: str, field: str | None, notes: str) -> dict[str, Any]:
    return {"kind": kind, "source_field": field, "notes": notes}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"顶层不是对象：{path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo: Path) -> tuple[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    subject = run("show", "-s", "--format=%s", "HEAD")
    return commit, subject


def _icon_type(document: dict[str, Any], rocket: dict[str, Any]) -> str | None:
    value = rocket.get("iconType", document.get("iconType"))
    return value if isinstance(value, str) else None


def _is_air_to_air(document: dict[str, Any], rocket: dict[str, Any]) -> bool:
    """Recognize current and legacy air-to-air entries without trusting one icon."""

    icon = _icon_type(document, rocket)
    if isinstance(icon, str) and (
        icon.startswith(AIR_TO_AIR_ICON_PREFIX)
        or AIR_TO_AIR_ICON_PATTERN.match(icon)
    ):
        return True
    # A few historical entries have no AAM icon but retain a guided AAM type.
    return str(rocket.get("bulletType", "")).lower() == "aam" and isinstance(rocket.get("guidance"), dict)


def _safe_id(value: str) -> str:
    """Convert a raw BLK name to the v1 lower-case/underscore identifier."""

    safe = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unknown_missile"


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _unsupported_features(rocket: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for key in _walk_keys(rocket):
        for label, pattern in UNSUPPORTED_FEATURE_PATTERNS:
            if pattern.search(key) and label not in features:
                features.append(label)
    guidance = rocket.get("guidance")
    guidance_type = str(rocket.get("guidanceType", "")).lower()
    if guidance_type in {"saclos", "command", "command_guidance"} and "command_guidance" not in features:
        features.append("command_guidance")
    if not isinstance(guidance, dict) and "command_guidance" not in features:
        # X-4 and R.20 entries are present as AAMs but expose no PN/seeker
        # block in this datamine.  Keep them visible as unsupported instead
        # of silently assigning a PN model with empty gains.
        features.append("command_guidance")
    if isinstance(guidance, dict) and guidance.get("beamRider") is True and "beam_riding" not in features:
        features.append("beam_riding")
    return sorted(features)


def _model_family(unsupported_features: list[str], loft_enabled: bool) -> dict[str, str]:
    family = {
        "dynamics": "h2_reduced_order",
        "propulsion": "staged_solid_rocket",
        "aerodynamics": "conventional_fin",
        "control": "aerodynamic_fin",
        "guidance": "pn_loft" if loft_enabled else "pn",
    }
    if "full_6dof" in unsupported_features:
        family["dynamics"] = "full_6dof"
    if "ramjet" in unsupported_features:
        family["propulsion"] = "ramjet"
    if "grid_fin" in unsupported_features:
        family["aerodynamics"] = "grid_fin"
    if "jet_vane" in unsupported_features:
        family["control"] = "jet_vane"
    if "beam_riding" in unsupported_features:
        family["guidance"] = "beam_riding"
    elif "command_guidance" in unsupported_features:
        family["guidance"] = "command_guidance"
    if "thrust_vector" in unsupported_features:
        family["control"] = "thrust_vector"
    return family


def _autopilot(guidance: dict[str, Any]) -> tuple[dict[str, Any], str]:
    direct = guidance.get("guidanceAutopilot")
    if isinstance(direct, dict):
        return direct, "rocket.guidance.guidanceAutopilot"
    line_of_sight = guidance.get("lineOfSightAutopilot")
    if isinstance(line_of_sight, dict):
        accel_control = line_of_sight.get("accelControl")
        if isinstance(accel_control, dict):
            return accel_control, "rocket.guidance.lineOfSightAutopilot.accelControl"
        return line_of_sight, "rocket.guidance.lineOfSightAutopilot"
    return {}, "rocket.guidance.guidanceAutopilot"


def _variant_score(path: Path, bullet_name: str) -> tuple[int, str]:
    stem = path.stem
    score = 0
    if stem != bullet_name:
        score += 100
    if "_default" in stem:
        score += 50
    if "_bol_pod" in stem:
        score += 25
    for suffix in ("_switzerland", "_mirage_2000", "_tornado", "_f_14", "_f_15", "_f_16", "_fa_18"):
        if stem.endswith(suffix):
            score += 10
    return score, path.name


def _country_for(canonical_path: Path, bullet_name: str) -> str:
    stem = canonical_path.stem
    prefix = stem.split("_", 1)[0]
    if bullet_name.startswith("cn_"):
        return "China"
    if prefix == "su" and bullet_name.startswith(("su_pl", "pl_")):
        return "China"
    return COUNTRY_PREFIXES.get(prefix, "Unknown")


def _display_name(bullet_name: str) -> str:
    explicit = {
        "atam_mistral": "Mistral",
        "atam_mistral_a129": "Mistral (A129)",
        "br_maa_1": "MAA-1 Piranha",
        "cn_ld10": "LD-10",
        "cn_pf10": "PL-10",
        "cn_pl11": "PL-11",
        "cn_pl12": "PL-12",
        "cn_pl12a": "PL-12A",
        "cn_pl15": "PL-15",
        "cn_pl8b": "PL-8B",
        "cn_pl9": "PL-9",
        "cn_sd10a": "SD-10A",
        "cn_ty_90": "TY-90",
        "fr_matra_super_530d": "Super 530D",
        "fr_matra_super_530f": "Super 530F",
        "fr_mica_em": "MICA EM",
        "fr_aa20": "AA.20",
        "fr_r_511_matra": "R.511",
        "fr_r_530_matra_radar": "R.530 Radar",
        "fr_r_550_magic": "R.550 Magic",
        "fr_r_550_magic_2": "R.550 Magic 2",
        "il_aspide_1a": "Aspide 1A",
        "il_derby": "Derby",
        "il_pyton_3": "Python 3",
        "il_pyton_4": "Python 4",
        "ir_fakour_90": "Fakour-90",
        "ir_sedjil": "Sedjil",
        "jp_aam4": "AAM-4",
        "jp_aam3": "AAM-3",
        "pl_5e2": "PL-5E2",
        "r_darter": "R-Darter",
        "su_pl5b": "PL-5B",
        "su_pl5c": "PL-5C",
        "su_pl7": "PL-7",
        "su_pl8": "PL-8",
        "su_r13m1": "R-13M1",
        "su_r_24r": "R-24R",
        "su_r_27er": "R-27ER",
        "su_r_27er1": "R-27ER1",
        "su_r_27r": "R-27R",
        "su_r_27r1": "R-27R1",
        "su_r_40rd": "R-40RD",
        "su_r_60": "R-60",
        "su_r_60m": "R-60M",
        "su_r_60mk": "R-60MK",
        "su_r_73": "R-73",
        "su_r_73e": "R-73E",
        "su_r_77": "R-77",
        "su_r_77_1": "R-77-1",
        "su_rvv_ae": "RVV-AE",
        "swd_rb24j": "RB24J",
        "swd_rb71": "RB71",
        "swd_rb74": "RB74",
        "swd_rb74m": "RB74M",
        "swd_rb99": "RB99",
        "sws_flz_lwf_63_75": "Flz Lwf 63/75",
        "sws_flz_lwf_ll_64": "Flz Lwf LL 64",
        "us_aim_26b": "AIM-26B Falcon",
        "us_aim4f_falcon": "AIM-4F Falcon",
        "us_aim4g_falcon": "AIM-4G Falcon",
        "us_fim_92b": "FIM-92B Stinger",
        "us_starstreak": "Starstreak",
        "uk_fireflash": "Fireflash",
        "uk_firestreak": "Firestreak",
        "uk_redtop": "Red Top",
        "uk_skyflash_aim_7": "Skyflash",
        "uk_skyflash_aim_7_dogfight": "Skyflash Dogfight",
        "uk_skyflash_temp": "Skyflash (temporary)",
        "us_aim7e_sparrow": "AIM-7E Sparrow",
        "us_aim7e2_dogfight_sparrow": "AIM-7E-2 Sparrow",
        "us_aim7f_sparrow": "AIM-7F Sparrow",
        "us_aim7m_sparrow": "AIM-7M Sparrow",
        "us_aim7p_sparrow": "AIM-7P Sparrow",
        "us_aim9e_sidewinder": "AIM-9E Sidewinder",
        "us_aim9g_sidewinder": "AIM-9G Sidewinder",
        "us_aim9h_sidewinder": "AIM-9H Sidewinder",
        "us_aim9l_i_1_sidewinder": "AIM-9Li-1 Sidewinder",
        "us_aim9l_i_sidewinder": "AIM-9Li Sidewinder",
        "us_aim9l_sidewinder": "AIM-9L Sidewinder",
        "us_aim9m_sidewinder": "AIM-9M Sidewinder",
        "us_aim9p4_sidewinder": "AIM-9P-4 Sidewinder",
        "us_aim9p_sidewinder": "AIM-9P Sidewinder",
        "us_aim_120a": "AIM-120A",
        "us_aim_120b": "AIM-120B",
        "us_aim_120c_5": "AIM-120C-5",
        "us_aim_120c_7": "AIM-120C-7",
        "us_aim_120d": "AIM-120D",
        "us_aim_54a": "AIM-54A Phoenix",
        "us_aim_54b": "AIM-54B Phoenix",
        "us_aim_54c": "AIM-54C Phoenix",
        "us_aim_54c_plus": "AIM-54C+ Phoenix",
    }
    if bullet_name in explicit:
        return explicit[bullet_name]
    return bullet_name.replace("_", " ").upper()


def _seeker(rocket: dict[str, Any]) -> tuple[dict[str, Any], str]:
    guidance = rocket.get("guidance") or {}
    for key in ("radarSeeker", "infraredSeeker", "opticalSeeker", "seeker"):
        value = guidance.get(key)
        if isinstance(value, dict):
            return value, f"rocket.guidance.{key}"
    for key, value in guidance.items():
        if isinstance(value, dict) and key.lower().endswith("seeker"):
            return value, f"rocket.guidance.{key}"
    return {}, "rocket.guidance"


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _list_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    values = [_number(item) for item in value]
    return [float(item) for item in values if item is not None] if all(item is not None for item in values) else None


def _sensor_model(
    guidance: dict[str, Any],
    seeker: dict[str, Any],
    seeker_path: str,
) -> dict[str, Any] | None:
    """Map radar/inertial fields without inventing values for other seekers."""

    if not seeker or not seeker_path.lower().endswith("radarseeker"):
        return None
    inertial_guidance = guidance.get("inertialGuidance")
    inertial_guidance = inertial_guidance if isinstance(inertial_guidance, dict) else {}
    transmitter = seeker.get("transmitter")
    transmitter = transmitter if isinstance(transmitter, dict) else {}
    transmitter_antenna = transmitter.get("antenna")
    transmitter_antenna = transmitter_antenna if isinstance(transmitter_antenna, dict) else {}
    receiver = seeker.get("receiver")
    receiver = receiver if isinstance(receiver, dict) else {}
    receiver_antenna = receiver.get("antenna")
    receiver_antenna = receiver_antenna if isinstance(receiver_antenna, dict) else {}
    doppler_gate = seeker.get("dopplerSpeedGate")
    doppler_gate = doppler_gate if isinstance(doppler_gate, dict) else {}
    dist_gate = seeker.get("distGate")
    dist_gate = dist_gate if isinstance(dist_gate, dict) else {}
    doppler_speed = seeker.get("dopplerSpeed")
    doppler_speed = doppler_speed if isinstance(doppler_speed, dict) else {}
    distance = seeker.get("distance")
    distance = distance if isinstance(distance, dict) else {}

    radar_model = {
        "band": _number(seeker.get("band")),
        "active": _bool_or_none(seeker.get("active")),
        "angle_gate_rate_deg_s": _number(seeker.get("angleGateRate")),
        "prolongation_time_max_s": _number(seeker.get("prolongationTimeMax")),
        "multipath_effect": _list_or_none(seeker.get("multipathEffect")),
        "side_lobes_attenuation_db": _number(seeker.get("sideLobesAttenuation")),
        "lock_angle_max_deg": _number(seeker.get("lockAngleMax")),
        "angle_max_deg": _number(seeker.get("angleMax")),
        "rate_max_deg_s": _number(seeker.get("rateMax")),
        "doppler_speed_gate": {
            "filter_alpha": _number(doppler_gate.get("filterAlpha")),
            "filter_beta": _number(doppler_gate.get("filterBetta")),
            "search_range_mps": _number(doppler_gate.get("dopplerSpeedGateSearchRange")),
        },
        "dist_gate": {
            "filter_alpha": _number(dist_gate.get("filterAlpha")),
            "filter_beta": _number(dist_gate.get("filterBetta")),
            "search_range_m": _number(dist_gate.get("distGateSearchRange")),
        },
        "transmitter": {
            "power": _number(transmitter.get("power")),
            "antenna": {
                "angle_half_sens_deg": _number(transmitter_antenna.get("angleHalfSens")),
                "side_lobes_sensitivity_db": _number(transmitter_antenna.get("sideLobesSensitivity")),
            },
        },
        "receiver": {
            "rcs_m2": _number(receiver.get("rcs")),
            "range_m": _number(receiver.get("range")),
            "range_max_m": _number(receiver.get("rangeMax")),
            "time_gain_control": _bool_or_none(receiver.get("timeGainControl")),
            "antenna": {
                "angle_half_sens_deg": _number(receiver_antenna.get("angleHalfSens")),
                "side_lobes_sensitivity_db": _number(receiver_antenna.get("sideLobesSensitivity")),
            },
        },
        "doppler_speed": {
            "presents": _bool_or_none(doppler_speed.get("presents")),
            "min_mps": _number(doppler_speed.get("minValue")),
            "max_mps": _number(doppler_speed.get("maxValue")),
            "width_mps": _number(doppler_speed.get("width")),
            "ref_width_mps": _number(doppler_speed.get("refWidth")),
            "signal_width_min_mps": _number(doppler_speed.get("signalWidthMin")),
        },
        "distance": {
            "presents": _bool_or_none(distance.get("presents")),
            "min_m": _number(distance.get("minValue")),
            "max_m": _number(distance.get("maxValue")),
            "width_m": _number(distance.get("width")),
            "signal_width_min_m": _number(distance.get("signalWidthMin")),
            "ref_width_m": _number(distance.get("refWidth")),
        },
    }
    return {
        "active_radar": _bool_or_none(seeker.get("active")),
        "inertial_navigation": _bool_or_none(guidance.get("inertialNavigation")),
        "use_target_velocity": _bool_or_none(guidance.get("useTargetVel")),
        "lock_after_launch": _bool_or_none(guidance.get("lockAfterLaunch")),
        "lock_timeout_s": _number(guidance.get("lockTimeOut")),
        "break_lock_max_time_s": _number(guidance.get("breakLockMaxTime")),
        "inertial_drift_speed_mps": _number(inertial_guidance.get("inertialNavigationDriftSpeed")),
        "datalink": _bool_or_none(inertial_guidance.get("datalink")),
        "reconnect_datalink": False,
        "radar_seeker": radar_model,
        "parameter_sources": {
            "active_radar": _source("datamine", f"{seeker_path}.active", "直接读取主动雷达标志。"),
            "inertial_navigation": _source("datamine", "rocket.guidance.inertialNavigation", "直接读取惯性导航标志。"),
            "use_target_velocity": _source("datamine", "rocket.guidance.useTargetVel", "直接读取目标速度使用标志。"),
            "lock_after_launch": _source("datamine", "rocket.guidance.lockAfterLaunch", "直接读取发射后锁定标志。"),
            "lock_timeout_s": _source("datamine", "rocket.guidance.lockTimeOut", "直接读取锁定超时字段；本阶段只保留。"),
            "break_lock_max_time_s": _source("datamine", "rocket.guidance.breakLockMaxTime", "直接读取惯性保持最大时间。"),
            "inertial_drift_speed_mps": _source("datamine", "rocket.guidance.inertialGuidance.inertialNavigationDriftSpeed", "直接读取惯性导航漂移速度。"),
            "datalink": _source("datamine", "rocket.guidance.inertialGuidance.datalink", "直接读取数据链标志。"),
            "reconnect_datalink": _source("assumed", None, "原始字段缺失，按计划固定为 false。"),
            "radar_seeker": _source("datamine", seeker_path, "保留主动雷达原始映射值。"),
        },
    }


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        value = _number(mapping.get(key))
        if value is not None:
            return value, key
    return None, None


def _duration_value(value: Any) -> tuple[float | None, bool]:
    number = _number(value)
    if number is not None:
        return number, False
    if isinstance(value, list):
        for item in value:
            number = _number(item)
            if number is not None and number > 0:
                return number, True
    return None, False


def _stages(rocket: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_mass = _number(rocket.get("mass"))
    if initial_mass is None:
        raise ValueError("缺少 rocket.mass")
    stages: list[dict[str, Any]] = []
    stage_source: dict[str, Any] = {}
    top_level_specs = (
        ("timeFire", "force", "massEnd", "rocket.timeFire", "rocket.force", "rocket.massEnd", "rocket.mass"),
        ("timeFire1", "force1", "massEnd1", "rocket.timeFire1", "rocket.force1", "rocket.massEnd1", "rocket.massEnd"),
    )
    previous_mass = initial_mass
    for time_key, force_key, mass_end_key, time_field, force_field, mass_end_field, mass_start_field in top_level_specs:
        duration, duration_was_array = _duration_value(rocket.get(time_key))
        thrust = _number(rocket.get(force_key))
        mass_end = _number(rocket.get(mass_end_key))
        # Historical entries use a zero-duration placeholder for an absent
        # first stage.  Do not emit an invalid stage or silently reorder the
        # remaining stage; the next valid stage consumes the preceding mass.
        if duration is None or duration <= 0 or thrust is None or mass_end is None:
            continue
        stage_name = "boost" if not stages else "sustain"
        stages.append({
            "name": stage_name,
            "duration_s": duration,
            "thrust_n": thrust,
            "mass_lost_kg": previous_mass - mass_end,
            "isp_s": None,
            "fire_delay_s": _number(rocket.get("fireDelay")) if not stages else None,
            "parameter_sources": {
                "duration_s": _source(
                    "datamine",
                    time_field,
                    "直接读取顶层燃烧时间。" if not duration_was_array else "原始值为数组；取第一个正数作为该阶段持续时间，保留原始字段路径。",
                ),
                "thrust_n": _source("datamine", force_field, "直接读取顶层推力。"),
                "mass_lost_kg": _source("derived", f"{mass_start_field} - {mass_end_field}", "由阶段开始质量和阶段末质量相减。"),
                "isp_s": _source("derived", "thrust_n / (mass_flow_kg_s * g0)", "由推力、阶段失重和公共重力常数计算。"),
            },
        })
        previous_mass = mass_end
    else:
        propulsion_keys = sorted(
            (key for key in rocket if re.fullmatch(r"propulsion\d+", key)),
            key=lambda key: int(re.search(r"\d+", key).group()),
        )
        for propulsion_key in propulsion_keys:
            propulsion = rocket.get(propulsion_key)
            if not isinstance(propulsion, dict):
                continue
            impulse_keys = sorted(
                (key for key in propulsion if re.fullmatch(r"impulse\d+", key)),
                key=lambda key: int(re.search(r"\d+", key).group()),
            )
            for impulse_key in impulse_keys:
                impulse = propulsion.get(impulse_key)
                if not isinstance(impulse, dict):
                    continue
                duration = _number(impulse.get("time"))
                thrust = _number(impulse.get("force"))
                mass_lost = _number(impulse.get("massLost"))
                if duration is None or duration <= 0 or thrust is None or mass_lost is None:
                    continue
                stage_index = len(stages)
                stages.append({
                    "name": "boost" if stage_index == 0 else "sustain",
                    "duration_s": duration,
                    "thrust_n": thrust,
                    "mass_lost_kg": mass_lost,
                    "isp_s": None,
                    "fire_delay_s": _number(propulsion.get("fireDelay")),
                    "parameter_sources": {
                        "duration_s": _source("datamine", f"rocket.{propulsion_key}.{impulse_key}.time", "直接读取嵌套脉冲时间。"),
                        "thrust_n": _source("datamine", f"rocket.{propulsion_key}.{impulse_key}.force", "直接读取嵌套脉冲推力。"),
                        "mass_lost_kg": _source("datamine", f"rocket.{propulsion_key}.{impulse_key}.massLost", "直接读取嵌套脉冲失重。"),
                        "isp_s": _source("derived", "thrust_n / (mass_flow_kg_s * g0)", "由推力、阶段失重和公共重力常数计算。"),
                    },
                })
        if not stages:
            raise ValueError("没有找到可解析的发动机阶段")
    for stage in stages:
        mass_lost = max(0.0, float(stage["mass_lost_kg"]))
        stage["mass_lost_kg"] = mass_lost
        duration = float(stage["duration_s"])
        stage["isp_s"] = float(stage["thrust_n"]) / (mass_lost / duration * G0) if mass_lost > 0 else 0.0
    if "extPressureToThrustMult" in rocket:
        stage_source["mach_thrust_correction"] = _source(
            "datamine",
            "rocket.extPressureToThrustMult",
            "保留原始外部压力推力修正；未宣称为完整冲压模型。",
        )
    return stages, stage_source


def _profile(
    document: dict[str, Any],
    rocket: dict[str, Any],
    canonical: Path,
    variants: list[Path],
    repo: Path,
    commit: str,
    version: str,
    retrieved_date: str,
    bullet_name: str,
    raw_bullet_name: str,
    unsupported_features: list[str],
) -> dict[str, Any]:
    guidance = rocket.get("guidance") or {}
    autopilot, autopilot_path = _autopilot(guidance)
    seeker, seeker_path = _seeker(rocket)
    stages, propulsion_sources = _stages(rocket)
    source_files = [_relative(path, repo) for path in variants]
    source_hashes = {_relative(path, repo): _sha256(path) for path in variants}
    source_root = _relative(canonical, repo)
    loft_enabled = bool(autopilot.get("loftEnabled", False))
    model_family = _model_family(unsupported_features, loft_enabled)
    seeker_rate = _number(seeker.get("rateMax"))
    lock_range = _number(guidance.get("lockDistance"))
    lock_range_source = "rocket.guidance.lockDistance"
    lock_range_kind = "datamine"
    if lock_range is None:
        lock_range = _number(seeker.get("rangeMax")) or _number(seeker.get("range"))
        lock_range_source = f"{seeker_path}.rangeMax/range"
        lock_range_kind = "derived"
    proximity = _number((rocket.get("proximityFuse") or {}).get("radius"))
    if proximity is None:
        proximity = _number(rocket.get("shutterDamageRadius"))
        proximity_source = "rocket.shutterDamageRadius"
        proximity_kind = "derived"
    else:
        proximity_source = "rocket.proximityFuse.radius"
        proximity_kind = "datamine"
    initial_mass = _number(rocket.get("mass"))
    caliber = _number(rocket.get("caliber"))
    length = _number(rocket.get("length"))
    moment_arm = _number(rocket.get("distFromCmToStab"))
    if None in (initial_mass, caliber, length, moment_arm):
        raise ValueError(f"{bullet_name}: 几何字段不完整")
    cx_k = _number(rocket.get("CxK"))
    if cx_k is None:
        raise ValueError(f"{bullet_name}: 缺少 rocket.CxK")
    pid_values = {
        "p": _number(autopilot.get("accelControlProp")),
        "i": _number(autopilot.get("accelControlIntg")),
        "d": _number(autopilot.get("accelControlDiff")),
        "integral_limit": _number(autopilot.get("accelControlIntgLim")),
    }
    aerodynamics = {
        "cx_k": cx_k,
        "cy_k": None,
        "cx_vs_aoa": {"model": "not_declared", "coefficient_per_rad2": None},
        "max_cy_at_aoa": {"value": None, "aoa_rad": None},
        "fins_lateral_acceleration_g": _number(rocket.get("finsLatAccel")),
        "fin_moment_arm_m": moment_arm,
        "fin_aoa_limit_rad": {
            "horizontal": _number(rocket.get("finsAoaHor")) or 0.0,
            "vertical": _number(rocket.get("finsAoaVer")) or 0.0,
        },
        "mach_drag_correction": {"model": "not_declared", "parameters": None},
        "mach_lift_correction": {"model": "not_declared", "parameters": None},
        "drag_scale": 1.0,
        "lift_area_scale": _number(rocket.get("wingAreaMult")) or 1.0,
        "parameter_sources": {
            "cx_k": _source("datamine", "rocket.CxK", "直接读取轴向阻力系数标度。"),
            "cy_k": _source("assumed", None, "datamine 未提供 CyK；保留 null，不写入数值假设。"),
            "cx_vs_aoa": _source("assumed", None, "datamine 未提供 CxAoA；仅保留模型槽位，未填拟合系数。"),
            "max_cy_at_aoa": _source("assumed", None, "datamine 未提供 CyMaxAoA；保留 null。"),
            "fins_lateral_acceleration_g": _source("datamine", "rocket.finsLatAccel", "直接读取舵面横向加速度能力；部分条目缺失时为 null。"),
            "fin_moment_arm_m": _source("datamine", "rocket.distFromCmToStab", "直接读取质心到稳定面力臂。"),
            "fin_aoa_limit_rad": _source("datamine", "rocket.finsAoaHor/finsAoaVer", "直接读取水平/垂直舵面 AoA 限制，原始值按弧度保留。"),
            "mach_drag_correction": _source("assumed", None, "datamine 未提供可直接映射的 Mach 阻力修正表。"),
            "mach_lift_correction": _source("assumed", None, "datamine 未提供可直接映射的 Mach 升力修正表。"),
            "drag_scale": _source("assumed", None, "中性候选标度 1.0；不是 datamine 识别结果。"),
            "lift_area_scale": _source("datamine", "rocket.wingAreaMult", "直接读取翼面积标度。"),
        },
    }
    guidance_profile = {
        "guidance_type": model_family["guidance"],
        "maximum_lateral_acceleration_g": _number(autopilot.get("reqAccelMax")) or 0.0,
        "maximum_angular_rate_deg_s": seeker_rate,
        "pn_gain": _number(autopilot.get("propNavMult")) or 0.0,
        "lock_range_m": lock_range,
        "lofting_enabled": loft_enabled,
        "lofting_elevation_deg": _number(autopilot.get("loftElevation")) or 0.0,
        "loft_exit_distance_m": None,
        "loft_exit_time_to_go_s": None,
        "proximity_radius_m": proximity or 0.1,
        "seeker_type": str(rocket.get("guidanceType", "unknown")),
        "parameter_sources": {
            "guidance_type": _source("derived", "model_family.guidance", "由原始制导结构和 loft 标志映射到显式 v1 模型族；不支持类型保持原值，不做 PN 退化。"),
            "maximum_lateral_acceleration_g": _source("datamine", f"{autopilot_path}.reqAccelMax", "直接读取制导横向指令上限，不等同于实战达到过载。"),
            "maximum_angular_rate_deg_s": _source("datamine", f"{seeker_path}.rateMax" if seeker_rate is not None else None, "直接读取 seeker rateMax；语义是 seeker 角速率上限。"),
            "pn_gain": _source("datamine", f"{autopilot_path}.propNavMult", "直接读取比例导引增益；非 PN 制导 profile 仅保留原始可用字段。"),
            "lock_range_m": _source(lock_range_kind, lock_range_source, "缺少 lockDistance 时使用 seeker rangeMax/range 作为锁定距离代理。" if lock_range_kind == "derived" else "直接读取锁定距离。"),
            "lofting_enabled": _source("derived", "rocket.guidance.guidanceAutopilot.loftEnabled", "存在 true 时映射为 pn_loft，否则按未启用处理。"),
            "lofting_elevation_deg": _source("datamine" if "loftElevation" in autopilot else "assumed", "rocket.guidance.guidanceAutopilot.loftElevation" if "loftElevation" in autopilot else None, "直接读取 loft 仰角；未声明时使用 0 度占位。"),
            "loft_exit_distance_m": _source("assumed", None, "datamine 未提供与此字段语义一致的退出距离。"),
            "loft_exit_time_to_go_s": _source("assumed", None, "datamine 未提供与此字段语义一致的剩余时间门槛。"),
            "proximity_radius_m": _source(proximity_kind, proximity_source, "优先读取 proximityFuse.radius；缺失时使用 shutterDamageRadius 作为代理。"),
            "seeker_type": _source("datamine", "rocket.guidanceType", "直接读取 seeker 制导类型标签。"),
        },
    }
    sensor_model = _sensor_model(guidance, seeker, seeker_path)
    if sensor_model is not None:
        guidance_profile["sensor_model"] = sensor_model
    propulsion = {
        "stages": stages,
        "mach_thrust_correction": (
            {"model": "external_pressure_multiplier", "values": rocket["extPressureToThrustMult"]}
            if isinstance(rocket.get("extPressureToThrustMult"), list)
            else None
        ),
        "parameter_sources": {
            "stages": _source("datamine", source_root, "阶段由顶层 timeFire/force 或嵌套 propulsion impulse 批量解析。"),
            "mach_thrust_correction": propulsion_sources.get(
                "mach_thrust_correction",
                _source("assumed", None, "该条目未声明可直接映射的 Mach 推力修正。"),
            ),
        },
    }
    maximum_speed_mps = _number(rocket.get("endSpeed"))
    speed_source = _source("datamine", "rocket.endSpeed", "直接读取速度上限字段。")
    speed_semantics = "hard_limit"
    if maximum_speed_mps is None or maximum_speed_mps <= 0:
        maximum_speed_mps = (_number(rocket.get("machMax")) or 0.0) * 340.3
        speed_source = _source(
            "derived",
            "rocket.machMax * 340.3 m/s",
            "datamine 未提供 endSpeed；用标准海平面音速将 Mach 上限转换为显示参考值，不作为硬限制。",
        )
        speed_semantics = "display_reference"
    performance = {
        "maximum_speed_mps": maximum_speed_mps,
        "maximum_distance_m": _number(rocket.get("maxDistance")) or 0.0,
        "lifetime_s": _number(rocket.get("timeLife")) or 0.0,
        "maximum_mach": _number(rocket.get("machMax")),
        "minimum_distance_m": _number(rocket.get("minDistance")),
        "range_reference_m": _number(rocket.get("rangeMax")),
        "limit_semantics": {
            "maximum_speed_mps": speed_semantics,
            "maximum_distance_m": "event_threshold",
            "lifetime_s": "hard_limit",
        },
        "parameter_sources": {
            "maximum_speed_mps": speed_source,
            "maximum_distance_m": _source("datamine", "rocket.maxDistance", "直接读取最大距离事件字段。"),
            "lifetime_s": _source("datamine", "rocket.timeLife", "直接读取寿命字段。"),
            "maximum_mach": _source("datamine", "rocket.machMax", "直接读取 Mach 上限字段。"),
            "minimum_distance_m": _source("datamine", "rocket.minDistance", "直接读取最小距离字段。"),
            "range_reference_m": _source("datamine", "rocket.rangeMax", "直接读取显示/性能参考距离。"),
        },
    }
    control = {
        "pid": pid_values,
        "fin_command_limit": 1.0,
        "actuator_time_constant_s": None,
        "max_pitch_yaw_rate_deg_s": None,
        "identified_controller": None,
        "parameter_sources": {
            "pid.p": _source("datamine", f"{autopilot_path}.accelControlProp", "直接读取比例控制项。"),
            "pid.i": _source("datamine", f"{autopilot_path}.accelControlIntg", "直接读取积分控制项。"),
            "pid.d": _source("datamine", f"{autopilot_path}.accelControlDiff", "直接读取微分控制项。"),
            "pid.integral_limit": _source("datamine", f"{autopilot_path}.accelControlIntgLim", "直接读取积分限幅。"),
            "fin_command_limit": _source("assumed", None, "标准化舵面命令中性上限 1.0；datamine 未声明该接口量。"),
            "actuator_time_constant_s": _source("assumed", None, "datamine 未提供独立舵机时间常数。"),
            "max_pitch_yaw_rate_deg_s": _source("assumed", None, "datamine 未提供导弹俯仰/偏航速率上限。"),
            "identified_controller": _source("assumed", None, "没有外部识别控制器结果；不得把 PID 与独立识别层混合。"),
        },
    }
    runtime_implemented = not unsupported_features
    unsupported_note = ""
    if unsupported_features:
        unsupported_note = (
            f"识别到未接入的模型类型：{', '.join(unsupported_features)}；仅保留只读 profile，GUI 必须显示不支持，"
            "不得退化为普通 PN/气动舵模型。"
        )
        if not isinstance(rocket.get("guidance"), dict):
            unsupported_note += " 原始 rocket.guidance 缺失；这是保守的不可运行分类，不是 datamine 直接给出的 PN 参数。"
    return {
        "schema_version": 1,
        "missile_id": bullet_name,
        "display_name": _display_name(bullet_name),
        "country": _country_for(canonical, bullet_name),
        "category": "air_to_air",
        "model_status": "experimental",
        "model_family": model_family,
        "geometry": {
            "initial_mass_kg": initial_mass,
            "caliber_m": caliber,
            "length_m": length,
            "wing_area_multiplier": _number(rocket.get("wingAreaMult")) or 1.0,
            "reference_area_mode": "caliber_times_wing_multiplier",
            "fin_moment_arm_m": moment_arm,
        },
        "propulsion": propulsion,
        "aerodynamics": aerodynamics,
        "performance": performance,
        "guidance": guidance_profile,
        "control": control,
        "provenance": {
            "source_name": "War Thunder public datamine",
            "source_version": version,
            "source_commit": commit,
            "retrieved_date": retrieved_date,
            "source_files": source_files,
            "source_sha256": source_hashes,
            "notes": (
                f"批处理导入；canonical source={source_root}；raw bulletName={raw_bullet_name!r}。"
                "按 bulletName 去重，保留所有 default、平台和吊舱变体路径。"
                "本 profile 是实验性数据层，不代表端到端轨迹已验证；缺失字段保留 null 或中性占位。"
                + (" " + unsupported_note if unsupported_note else "")
            ),
        },
        "validation": {
            "status": "experimental",
            "validated_cases": [],
            "known_limitations": [
                "仅完成 profile contract smoke，不等同于 War Thunder/StatShark 求解器复现。",
                "datamine 未声明的 CxAoA、CyK、Mach 气动修正和舵机动态未被虚构。",
                "reqAccelMax 是制导指令上限，不是时间域实际达到过载。",
            ],
        },
        "runtime": {
            "implemented": runtime_implemented,
            "config_path": None,
            "notes": (
                "使用公共 Python profile_h2_universal_v2 模型层并叠加本 JSON 的导弹参数；仍保持 experimental 状态。"
                if runtime_implemented
                else (
                    unsupported_note
                    if unsupported_note
                    else "当前公共 H2 runtime 不支持该 profile 的物理类型；GUI 必须显示不可运行。"
                )
            ),
        },
    }


def import_profiles(project_dir: Path, repo: Path) -> dict[str, Any]:
    rocket_dir = repo / "aces.vromfs.bin_u" / "gamedata" / "weapons" / "rocketguns"
    if not rocket_dir.is_dir():
        raise FileNotFoundError(f"找不到 rocketguns 目录：{rocket_dir}")
    commit, version = _git_metadata(repo)
    retrieved_date = datetime.now(timezone.utc).date().isoformat()
    grouped: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any], str, bool]]] = {}
    parse_errors: list[str] = []
    candidate_file_count = 0
    bullet_name_fallback_file_count = 0
    for path in sorted(rocket_dir.glob("*.blkx")):
        try:
            document = _read_json(path)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
            continue
        rocket = document.get("rocket")
        if not isinstance(rocket, dict):
            continue
        if not _is_air_to_air(document, rocket):
            continue
        raw_bullet_name = rocket.get("bulletName")
        bullet_name_fallback = not isinstance(raw_bullet_name, str) or not raw_bullet_name.strip()
        if bullet_name_fallback:
            raw_bullet_name = path.stem
            bullet_name_fallback_file_count += 1
        raw_bullet_name = str(raw_bullet_name).strip()
        candidate_file_count += 1
        grouped.setdefault(raw_bullet_name.lower(), []).append((path, rocket, document, raw_bullet_name, bullet_name_fallback))
    missiles_dir = project_dir / "missiles"
    missiles_dir.mkdir(parents=True, exist_ok=True)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    entity_ids: dict[str, str] = {}
    for raw_key, records in sorted(grouped.items()):
        raw_bullet_name = records[0][3]
        bullet_name = _safe_id(raw_bullet_name)
        prior_raw = entity_ids.get(bullet_name)
        if prior_raw is not None and prior_raw != raw_key:
            parse_errors.append(
                f"{raw_bullet_name}: bulletName 经规范化后与 {prior_raw} 冲突，无法生成唯一 missile_id"
            )
            continue
        entity_ids[bullet_name] = raw_key
        paths = [record[0] for record in records]
        canonical_record = min(
            records,
            key=lambda record: _variant_score(record[0], raw_bullet_name),
        )
        canonical_path, canonical_rocket, canonical_document = canonical_record[:3]
        canonical_raw_bullet_name = canonical_record[3]
        features = _unsupported_features(canonical_rocket)
        if "thrust_vector" in features:
            excluded.append({
                "missile_id": bullet_name,
                "display_name": _display_name(bullet_name),
                "reason": "tvc_excluded",
                "unsupported_features": features,
                "source_files": [_relative(path, repo) for path in paths],
                "bullet_name_fallback": any(record[4] for record in records),
            })
            continue
        try:
            profile = _profile(
                canonical_document,
                canonical_rocket,
                canonical_path,
                sorted(paths),
                repo,
                commit,
                version,
                retrieved_date,
                bullet_name,
                canonical_raw_bullet_name,
                features,
            )
        except Exception as exc:
            parse_errors.append(f"{bullet_name}: profile 生成失败：{exc}")
            continue
        profile_path = missiles_dir / f"{bullet_name}.json"
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        included.append({
            "missile_id": bullet_name,
            "display_name": profile["display_name"],
            "canonical_source": _relative(canonical_path, repo),
            "source_files": [_relative(path, repo) for path in sorted(paths)],
            "guidance_type": profile["guidance"]["seeker_type"],
            "model_family": profile["model_family"],
            "unsupported_features": features,
            "bullet_name_fallback": any(record[4] for record in records),
        })
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "War Thunder public datamine",
            "version": version,
            "commit": commit,
            "repository": "https://github.com/gszabi99/War-Thunder-Datamine",
            "relative_path": "aces.vromfs.bin_u/gamedata/weapons/rocketguns",
        },
        "selection": {
            "icon_type_prefix": "missile_air_to_air* or missile_type_[a-z]_air_to_air*",
            "fallback_selector": "rocket.bulletType=aam with a guidance object",
            "dedupe_key": "rocket.bulletName; missing values use filename stem",
            "canonical_variant_policy": "prefer exact bulletName filename, then non-default/non-pod/non-platform variant",
            "tvc_policy": "exclude profiles with thrust-vectoring fields; retain them in excluded list",
        },
        "candidate_file_count": candidate_file_count,
        "unique_entity_count": len(grouped),
        "bullet_name_fallback_file_count": bullet_name_fallback_file_count,
        "included_profile_count": len(included),
        "excluded_profile_count": len(excluded),
        "unsupported_read_only_profile_count": sum(bool(item["unsupported_features"]) for item in included),
        "included": included,
        "excluded": excluded,
        "parse_errors": parse_errors,
        "notes": [
            "Included profiles are experimental data profiles; they are not all validated flight models.",
            "The canonical profile stores unit-explicit fields; original BLK files remain available in the sparse checkout for raw provenance.",
        ],
    }
    manifest_path = project_dir / "data" / "aam_non_tvc_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量导入 datamine 空对空非 TVC 导弹")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args(argv)
    project_dir = args.project_dir.resolve()
    repo = (args.repo or project_dir / "data" / "datamine").resolve()
    manifest = import_profiles(project_dir, repo)
    print(json.dumps({
        "candidate_file_count": manifest["candidate_file_count"],
        "unique_entity_count": manifest["unique_entity_count"],
        "included_profile_count": manifest["included_profile_count"],
        "excluded_profile_count": manifest["excluded_profile_count"],
        "excluded": manifest["excluded"],
        "parse_errors": manifest["parse_errors"],
    }, ensure_ascii=False, indent=2))
    return 0 if not manifest["parse_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
