#!/usr/bin/env python3
"""Run the fixed S0/S1/S2 legacy-vs-body-Cm offline audit matrix."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sweep_pid_error_scale as sweep  # noqa: E402
from aim120_model.h2_simulator import H2Simulator  # noqa: E402
from aim120_model.metrics import continuous_closest_approach  # noqa: E402
from aim120_model.profile_adapter import (  # noqa: E402
    BODY_CM_TAIL_FORCE_PLANT,
    LEGACY_CRITICAL_DAMPED_PLANT,
    build_h2_candidate_config,
)
from aim120_model.public_api import _case_from_scenario  # noqa: E402


MISSILES = ("cn_pl12", "jp_aam4", "su_r_77", "us_aim_120a")
PLANTS = (LEGACY_CRITICAL_DAMPED_PLANT, BODY_CM_TAIL_FORCE_PLANT)
SCENARIOS = ("S0", "S1", "S2")
RUN_COMMAND = "py -3 scripts\\run_body_cm_off_axis_matrix.py"
FIXED_LIFTING_SURFACE_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
# Model C from the attached design note: keep the diagnostic k_W=2 surface,
# move only its signed CG-relative station, and inspect force/moment/energy
# sensitivity.  These are unsupported candidate geometry values.
FIXED_LIFTING_SURFACE_STATIONS_M = (-0.5, -0.25, 0.0, 0.25, 0.5)


def _git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=not binary
    ).strip() if not binary else subprocess.check_output(["git", *args], cwd=ROOT)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worktree_provenance() -> dict[str, Any]:
    status_lines = str(_git("status", "--short")).splitlines()
    untracked_hashes: dict[str, str] = {}
    for line in status_lines:
        if line.startswith("?? "):
            relative = line[3:]
            path = ROOT / relative
            if path.is_file():
                untracked_hashes[relative.replace("\\", "/")] = _sha256_file(path)
    tracked_diff = _git("diff", "--binary", "--no-ext-diff", "HEAD", "--", binary=True)
    assert isinstance(tracked_diff, bytes)
    return {
        "head": str(_git("rev-parse", "HEAD")),
        "dirty": bool(status_lines),
        "status_short": status_lines,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "untracked_file_sha256": untracked_hashes,
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _maximum(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return max(values) if values else None


def _left_sample_duration(samples: list[dict[str, Any]], predicate) -> float:
    return sum(
        max(float(right["time_s"]) - float(left["time_s"]), 0.0)
        for left, right in zip(samples, samples[1:])
        if predicate(left)
    )


def _trapezoid_integral(samples: list[dict[str, Any]], key: str) -> float:
    return sum(
        0.5
        * (float(left[key]) + float(right[key]))
        * max(float(right["time_s"]) - float(left["time_s"]), 0.0)
        for left, right in zip(samples, samples[1:])
    )


def _trajectory_path_length(samples: list[dict[str, Any]]) -> float:
    return sum(
        math.sqrt(
            sum(
                (float(right["position_m"][axis]) - float(left["position_m"][axis])) ** 2
                for axis in range(3)
            )
        )
        for left, right in zip(samples, samples[1:])
    )


def _specific_mechanical_energy(sample: dict[str, Any], gravity_mps2: float) -> float:
    speed_sq = sum(float(value) ** 2 for value in sample["velocity_mps"])
    return 0.5 * speed_sq + gravity_mps2 * float(sample["position_m"][1])


def _summarize(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    samples = result.get("samples") or []
    if not samples:
        return {
            "event_type": result.get("event_type"),
            "terminal_time_s": None,
            "hit_or_fuse": False,
            "proximity_fuse": False,
            "minimum_distance_m": None,
            "maximum_actual_g": None,
            "maximum_lateral_g": None,
            "maximum_body_alpha_deg": None,
            "maximum_abs_body_rate_deg_s": None,
            "maximum_actual_fin_angle_deg": None,
            "fin_limits_deg": None,
            "maximum_actual_fin_fraction": None,
            "actual_fin_saturated": None,
            "requested_fin_saturated": None,
            "maximum_body_force_contribution_n": None,
            "maximum_abs_tail_force_contribution_n": None,
            "maximum_abs_tail_moment_contribution_nm": None,
            "maximum_abs_total_moment_contribution_nm": None,
            "g_to_total_moment_ratio_g_per_nm": None,
            "trajectory_path_length_m": None,
            "initial_specific_mechanical_energy_m2_s2": None,
            "terminal_specific_mechanical_energy_m2_s2": None,
            "specific_mechanical_energy_retention_ratio": None,
            "integrated_drag_work_j": None,
            "integrated_drag_energy_loss_j": None,
            "drag_energy_loss_per_path_length_j_per_m": None,
            "integrated_lift_work_j": None,
            "sample_count": 0,
            "finite": False,
        }
    pitch_limit_rad = math.radians(
        float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
    )
    yaw_limit_rad = math.radians(
        float(config["aerodynamics"]["vertical_fin_aoa_limit_deg"])
    )
    fin_fractions = [
        max(
            abs(float(sample["actual_pitch_fin_angle_rad"])) / pitch_limit_rad,
            abs(float(sample["actual_yaw_fin_angle_rad"])) / yaw_limit_rad,
        )
        for sample in samples
    ]
    requested_fractions = [
        max(
            abs(float(sample["pitch_requested_fin_command"])),
            abs(float(sample["yaw_requested_fin_command"])),
        )
        for sample in samples
    ]
    event_type = str(result.get("event_type"))
    closest = continuous_closest_approach(samples)
    snapshot_keys = (
        "time_s", "mass_kg", "dynamic_pressure_pa", "indicated_speed_kmh", "mach",
        "angle_of_attack_rad", "actual_pitch_fin_angle_rad", "actual_yaw_fin_angle_rad",
        "pitch_tail_local_alpha_rad", "yaw_tail_local_alpha_rad",
        "pitch_tail_effective_incidence_rad", "yaw_tail_effective_incidence_rad",
        "pitch_tail_authority_fraction", "yaw_tail_authority_fraction",
        "tail_alpha_force_multiplier", "tail_delta_force_multiplier",
        "pitch_tail_alpha_force_slope_n_per_rad", "yaw_tail_alpha_force_slope_n_per_rad",
        "pitch_tail_delta_force_slope_n_per_rad", "yaw_tail_delta_force_slope_n_per_rad",
        "pitch_tail_alpha_force_n", "yaw_tail_alpha_force_n",
        "pitch_tail_delta_force_n", "yaw_tail_delta_force_n",
        "pitch_tail_net_force_pre_cap_n", "yaw_tail_net_force_pre_cap_n",
        "tail_force_cap_n", "tail_force_cap_scale", "tail_force_cap_active",
        "pitch_tail_alpha_moment_nm", "yaw_tail_alpha_moment_nm",
        "pitch_tail_delta_moment_nm", "yaw_tail_delta_moment_nm",
        "fixed_lifting_surface_multiplier",
        "body_normal_force_area_slope_m2_per_rad", "fixed_lifting_surface_area_slope_m2_per_rad",
        "fixed_lifting_surface_station_x_m",
        "pitch_fixed_lifting_surface_alpha_rad", "yaw_fixed_lifting_surface_alpha_rad",
        "pitch_fixed_lifting_surface_force_n", "yaw_fixed_lifting_surface_force_n",
        "pitch_fixed_lifting_surface_moment_nm", "yaw_fixed_lifting_surface_moment_nm",
        "pitch_total_body_wing_tail_normal_force_n", "yaw_total_body_wing_tail_normal_force_n",
        "pitch_body_normal_force_n", "yaw_body_normal_force_n", "pitch_tail_force_n", "yaw_tail_force_n",
        "pitch_body_total_moment_nm", "yaw_body_total_moment_nm", "pitch_tail_moment_nm", "yaw_tail_moment_nm",
        "pitch_total_moment_nm", "yaw_total_moment_nm",
        "commanded_acceleration_mps2", "commanded_acceleration_g", "controller_specific_force_command_g",
        "body_axis_pitch_specific_force_g", "body_axis_yaw_specific_force_g",
        "wind_normal_pitch_specific_force_g", "wind_normal_yaw_specific_force_g",
        "commanded_pitch_rate_rad_s", "commanded_yaw_rate_rad_s",
        "pitch_pid_integral", "yaw_pid_integral", "time_to_go_s", "current_gain", "closing_speed_mps",
        "cda0_m2", "cda_alpha_m2", "drag_power_w", "lift_power_w",
        "body_tail_force_power_at_cg_w", "velocity_mps",
    )
    def snapshot(sample: dict[str, Any]) -> dict[str, Any]:
        output = {key: sample.get(key) for key in snapshot_keys}
        velocity = sample.get("velocity_mps") or (0.0, 0.0, 0.0)
        output["speed_mps"] = math.sqrt(sum(float(value) ** 2 for value in velocity))
        command = sample.get("controller_specific_force_command_g") or (0.0, 0.0)
        output["pitch_outer_error_g"] = float(command[0]) - float(sample["wind_normal_pitch_specific_force_g"])
        output["yaw_outer_error_g"] = float(command[1]) - float(sample["wind_normal_yaw_specific_force_g"])
        return output
    peak_alpha_sample = max(samples, key=lambda sample: float(sample["angle_of_attack_rad"]))
    closest_time = float(closest["time_at_minimum_distance_s"])
    closest_sample = min(samples, key=lambda sample: abs(float(sample["time_s"]) - closest_time))
    saturation_exit_times = [
        float(samples[index]["time_s"])
        for index in range(1, len(samples))
        if fin_fractions[index - 1] >= 0.999 and fin_fractions[index] < 0.999
    ]
    summary = {
        "event_type": event_type,
        "terminal_time_s": float(result["terminal_time_s"]),
        "hit_or_fuse": event_type in {"impact", "fuse"},
        "proximity_fuse": event_type == "fuse",
        "minimum_distance_m": min(float(sample["distance_to_target_m"]) for sample in samples),
        "maximum_actual_g": _maximum(samples, "actual_overload_g"),
        "maximum_lateral_g": _maximum(samples, "lateral_load_g"),
        "maximum_body_alpha_deg": max(
            math.degrees(float(sample["angle_of_attack_rad"])) for sample in samples
        ),
        "maximum_abs_body_rate_deg_s": max(
            math.degrees(math.hypot(
                float(sample["pitch_rate_rad_s"]),
                float(sample["yaw_rate_rad_s"]),
            ))
            for sample in samples
        ),
        "maximum_actual_fin_angle_deg": max(
            math.degrees(math.hypot(
                float(sample["actual_pitch_fin_angle_rad"]),
                float(sample["actual_yaw_fin_angle_rad"]),
            ))
            for sample in samples
        ),
        "fin_limits_deg": {
            "pitch": math.degrees(pitch_limit_rad),
            "yaw": math.degrees(yaw_limit_rad),
        },
        "maximum_actual_fin_fraction": max(fin_fractions),
        "actual_fin_saturated": max(fin_fractions) >= 0.999,
        "requested_fin_saturated": max(requested_fractions) >= 0.999,
        "mechanical_fin_saturation_duration_s": _left_sample_duration(
            samples,
            lambda sample: max(
                abs(float(sample["actual_pitch_fin_angle_rad"])) / pitch_limit_rad,
                abs(float(sample["actual_yaw_fin_angle_rad"])) / yaw_limit_rad,
            ) >= 0.999,
        ),
        "first_mechanical_saturation_exit_time_s": saturation_exit_times[0] if saturation_exit_times else None,
        "last_mechanical_saturation_exit_time_s": saturation_exit_times[-1] if saturation_exit_times else None,
        "mechanical_saturation_exit_count": len(saturation_exit_times),
        "aerodynamic_radial_cap_duration_s": _left_sample_duration(
            samples, lambda sample: bool(sample.get("tail_force_cap_active", False))
        ),
        "initial_speed_mps": math.sqrt(sum(float(value) ** 2 for value in samples[0]["velocity_mps"])),
        "terminal_speed_mps": math.sqrt(sum(float(value) ** 2 for value in samples[-1]["velocity_mps"])),
        "initial_dynamic_pressure_pa": float(samples[0]["dynamic_pressure_pa"]),
        "terminal_dynamic_pressure_pa": float(samples[-1]["dynamic_pressure_pa"]),
        "maximum_dynamic_pressure_pa": max(float(sample["dynamic_pressure_pa"]) for sample in samples),
        "terminal_to_initial_speed_ratio": (
            math.sqrt(sum(float(value) ** 2 for value in samples[-1]["velocity_mps"]))
            / max(math.sqrt(sum(float(value) ** 2 for value in samples[0]["velocity_mps"])), 1e-12)
        ),
        "terminal_to_initial_dynamic_pressure_ratio": (
            float(samples[-1]["dynamic_pressure_pa"])
            / max(float(samples[0]["dynamic_pressure_pa"]), 1e-12)
        ),
        "maximum_body_force_contribution_n": max(
            math.hypot(
                float(sample["pitch_body_normal_force_n"]),
                float(sample["yaw_body_normal_force_n"]),
            )
            for sample in samples
        ),
        "maximum_abs_tail_force_contribution_n": max(
            math.hypot(
                float(sample["pitch_tail_force_n"]),
                float(sample["yaw_tail_force_n"]),
            )
            for sample in samples
        ),
        "maximum_abs_tail_moment_contribution_nm": max(
            math.hypot(
                float(sample["pitch_tail_moment_nm"]),
                float(sample["yaw_tail_moment_nm"]),
            )
            for sample in samples
        ),
        "maximum_abs_total_moment_contribution_nm": max(
            math.hypot(
                float(sample["pitch_total_moment_nm"]),
                float(sample["yaw_total_moment_nm"]),
            )
            for sample in samples
        ),
        # These are audit metrics, not claims of a calibrated range model:
        # path length is a trajectory-range proxy and work is integrated from
        # the candidate force telemetry over the actual sampled time base.
        "trajectory_path_length_m": _trajectory_path_length(samples),
        "initial_specific_mechanical_energy_m2_s2": _specific_mechanical_energy(
            samples[0], float(config["atmosphere"]["gravity_mps2"])
        ),
        "terminal_specific_mechanical_energy_m2_s2": _specific_mechanical_energy(
            samples[-1], float(config["atmosphere"]["gravity_mps2"])
        ),
        "specific_mechanical_energy_retention_ratio": (
            _specific_mechanical_energy(samples[-1], float(config["atmosphere"]["gravity_mps2"]))
            / max(
                _specific_mechanical_energy(samples[0], float(config["atmosphere"]["gravity_mps2"])),
                1e-12,
            )
        ),
        "integrated_drag_work_j": _trapezoid_integral(samples, "drag_power_w"),
        "integrated_drag_energy_loss_j": -_trapezoid_integral(samples, "drag_power_w"),
        "integrated_lift_work_j": _trapezoid_integral(samples, "lift_power_w"),
        "sample_count": len(samples),
        "finite": _finite_tree(samples),
        "diagnostic_snapshots": {
            "peak_body_alpha": snapshot(peak_alpha_sample),
            "nearest_sample_to_continuous_closest_approach": snapshot(closest_sample),
        },
    }
    summary["g_to_total_moment_ratio_g_per_nm"] = (
        float(summary["maximum_actual_g"])
        / max(float(summary["maximum_abs_total_moment_contribution_nm"]), 1e-12)
    )
    summary["drag_energy_loss_per_path_length_j_per_m"] = (
        float(summary["integrated_drag_energy_loss_j"])
        / max(float(summary["trajectory_path_length_m"]), 1e-12)
    )
    summary.update(closest)
    return summary


def _pl12_equilibrium_prediction(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    sample = summary["diagnostic_snapshots"]["peak_body_alpha"]
    pitch_alpha = float(sample["pitch_tail_local_alpha_rad"])
    yaw_alpha = float(sample["yaw_tail_local_alpha_rad"])
    axis = "pitch" if abs(pitch_alpha) >= abs(yaw_alpha) else "yaw"
    q_inf = float(sample["dynamic_pressure_pa"])
    diameter = float(config["geometry"]["caliber_m"])
    body = config["aerodynamics"]["body_cp_force_candidate"]
    body_arm = float(body["cp_cg_arm_over_diameter"]) * diameter
    body_slope = q_inf * math.pi * diameter ** 2 / 4.0 * float(body["cn_alpha_body_per_rad"]) * body_arm
    tail_arm = float(config["aerodynamics"]["split_tail_candidate"]["tail_alpha_moment_arm_m"])
    alpha_force_slope = float(sample[f"{axis}_tail_alpha_force_slope_n_per_rad"])
    delta_force_slope = float(sample[f"{axis}_tail_delta_force_slope_n_per_rad"])
    alpha_total_moment_slope = body_slope + tail_arm * alpha_force_slope
    delta_moment_slope = tail_arm * delta_force_slope
    delta = float(sample[f"actual_{axis}_fin_angle_rad"])
    predicted = (
        -delta_moment_slope * delta / alpha_total_moment_slope
        if abs(alpha_total_moment_slope) > 1e-12
        else None
    )
    return {
        "axis": axis,
        "peak_sample_mass_kg": sample["mass_kg"],
        "peak_sample_dynamic_pressure_pa": q_inf,
        "fin_angle_rad": delta,
        "body_static_moment_slope_nm_per_rad": body_slope,
        "tail_alpha_moment_slope_nm_per_rad": tail_arm * alpha_force_slope,
        "tail_delta_moment_slope_nm_per_rad": delta_moment_slope,
        "predicted_signed_alpha_rad": predicted,
        "predicted_abs_alpha_deg": None if predicted is None else abs(math.degrees(predicted)),
        "observed_peak_body_alpha_deg": summary["maximum_body_alpha_deg"],
        "approximation": "instantaneous single-axis small-angle balance at peak-AoA sample; ignores dynamics, radial-cap coupling, and changing q/mass",
    }


def main() -> int:
    started = time.perf_counter()
    defaults_path = ROOT / "config" / "profile_h2_runtime_defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    if defaults.get("plant_model") != LEGACY_CRITICAL_DAMPED_PLANT:
        raise ValueError("profile_h2_runtime_defaults.json must remain on legacy critical_damped_v12")
    manifest = json.loads((ROOT / "data" / "aam_non_tvc_manifest.json").read_text(encoding="utf-8"))
    profiles = {
        missile_id: json.loads(
            (ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8")
        )
        for missile_id in MISSILES
    }
    raw_pids = {
        missile_id: copy.deepcopy(profile["control"]["pid"])
        for missile_id, profile in profiles.items()
    }
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for plant in PLANTS:
        selected_defaults = copy.deepcopy(defaults)
        selected_defaults["plant_model"] = plant
        for missile_id, profile in profiles.items():
            config, _assumptions = build_h2_candidate_config(profile, selected_defaults)
            config["control"]["pid_error_units"] = "g"
            config["control"]["pid_error_scale"] = 1.0
            for term in ("p", "i", "d", "integral_limit"):
                if config["control"]["pid"][term] != raw_pids[missile_id][term]:
                    raise AssertionError(
                        f"raw PID term {term} changed while building {missile_id}/{plant}"
                    )
            configs[(plant, missile_id)] = config

    scenario_inputs = {name: sweep._scenario(name) for name in SCENARIOS}
    rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIOS:
        scenario = scenario_inputs[scenario_name]
        case = _case_from_scenario(scenario)
        for plant in PLANTS:
            for missile_id in MISSILES:
                config = configs[(plant, missile_id)]
                run_started = time.perf_counter()
                row: dict[str, Any] = {
                    "scenario": scenario_name,
                    "scenario_input": copy.deepcopy(scenario),
                    "missile_id": missile_id,
                    "plant_model": plant,
                    "control_model_version": config["control_model_version"],
                    "force_geometry_version": config["force_geometry_version"],
                    "model_label": config["model_label"],
                    "pid_error_scale": 1.0,
                    "raw_pid": copy.deepcopy(raw_pids[missile_id]),
                    "status": "completed",
                    "error": None,
                }
                try:
                    result = H2Simulator(config).run(copy.deepcopy(case))
                    row.update(_summarize(result, config))
                    if not row["finite"] or row["event_type"] == "numerical_failure":
                        row["status"] = "numerical_failure"
                except Exception as exc:  # independent runs continue without retry
                    row.update({
                        "status": "exception",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "event_type": None,
                        "terminal_time_s": None,
                        "hit_or_fuse": False,
                        "proximity_fuse": False,
                        "minimum_distance_m": None,
                        "maximum_actual_g": None,
                        "maximum_lateral_g": None,
                        "maximum_body_alpha_deg": None,
                        "maximum_abs_body_rate_deg_s": None,
                        "maximum_actual_fin_angle_deg": None,
                        "fin_limits_deg": None,
                        "maximum_actual_fin_fraction": None,
                        "actual_fin_saturated": None,
                        "requested_fin_saturated": None,
                        "maximum_body_force_contribution_n": None,
                        "maximum_abs_tail_force_contribution_n": None,
                        "maximum_abs_tail_moment_contribution_nm": None,
                        "sample_count": 0,
                        "finite": False,
                    })
                row["elapsed_s"] = time.perf_counter() - run_started
                rows.append(row)

    fixed_lifting_sweep_rows: list[dict[str, Any]] = []
    for scenario_name in ("S0", "S1"):
        scenario = scenario_inputs[scenario_name]
        for k_w in FIXED_LIFTING_SURFACE_MULTIPLIERS:
            config = copy.deepcopy(configs[(BODY_CM_TAIL_FORCE_PLANT, "cn_pl12")])
            config["aerodynamics"]["split_tail_candidate"]["tail_alpha_force_multiplier"] = 1.4
            fixed_lift = config["aerodynamics"]["fixed_lifting_surface_candidate"]
            fixed_lift["fixed_lifting_surface_multiplier"] = k_w
            fixed_lift["fixed_lifting_surface_area_slope_m2_per_rad"] = (
                k_w * fixed_lift["body_normal_force_area_slope_m2_per_rad"]
            )
            case = _case_from_scenario(scenario)
            result = H2Simulator(config).run(copy.deepcopy(case))
            summary = _summarize(result, config)
            fixed_lifting_sweep_rows.append({
                "scenario": scenario_name,
                "scenario_input": copy.deepcopy(scenario),
                "missile_id": "cn_pl12",
                "plant_model": BODY_CM_TAIL_FORCE_PLANT,
                "guidance_variant": "formal_profile_guidance",
                "tail_alpha_force_multiplier": 1.4,
                "tail_delta_force_multiplier": 1.0,
                "fixed_lifting_surface_multiplier": k_w,
                "fixed_lifting_surface_station_x_m": 0.0,
                "unsupported_parameter_boundary": "k_W is an unsupported near-CG lifting-surface multiplier, not derived from wingAreaMult or a real PL-12 coefficient",
                **summary,
                "analytic_peak_sample_equilibrium_prediction": _pl12_equilibrium_prediction(summary, config),
            })

    fixed_lifting_station_sweep_rows: list[dict[str, Any]] = []
    for scenario_name in ("S0", "S1"):
        scenario = scenario_inputs[scenario_name]
        for station_x_m in FIXED_LIFTING_SURFACE_STATIONS_M:
            config = copy.deepcopy(configs[(BODY_CM_TAIL_FORCE_PLANT, "cn_pl12")])
            config["aerodynamics"]["split_tail_candidate"]["tail_alpha_force_multiplier"] = 1.4
            fixed_lift = config["aerodynamics"]["fixed_lifting_surface_candidate"]
            fixed_lift["fixed_lifting_surface_multiplier"] = 2.0
            fixed_lift["fixed_lifting_surface_area_slope_m2_per_rad"] = (
                2.0 * fixed_lift["body_normal_force_area_slope_m2_per_rad"]
            )
            fixed_lift["station_x_m"] = station_x_m
            result = H2Simulator(config).run(
                copy.deepcopy(_case_from_scenario(scenario))
            )
            summary = _summarize(result, config)
            fixed_lifting_station_sweep_rows.append({
                "scenario": scenario_name,
                "scenario_input": copy.deepcopy(scenario),
                "missile_id": "cn_pl12",
                "plant_model": BODY_CM_TAIL_FORCE_PLANT,
                "guidance_variant": "formal_profile_guidance",
                "tail_alpha_force_multiplier": 1.4,
                "tail_delta_force_multiplier": 1.0,
                "fixed_lifting_surface_multiplier": 2.0,
                "fixed_lifting_surface_station_x_m": station_x_m,
                "unsupported_parameter_boundary": (
                    "k_W=2 and signed x_W are unsupported near-CG lifting-surface "
                    "candidate values; neither is derived from wingAreaMult or a "
                    "real PL-12 coefficient"
                ),
                **summary,
            })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "outputs" / f"body_cm_three_source_s0_s2_pl12_fixed_lift_station_{timestamp}.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    artifact = {
        "schema_version": 1,
        "artifact_type": "body_cm_three_source_off_axis_s0_s2_and_pl12_fixed_lift_station_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_command": RUN_COMMAND,
        "scope": {
            "network_used": False,
            "calculate_count": 0,
            "war_thunder_game_files_modified": False,
            "pid_error_scale": 1.0,
            "per_missile_pid_tuning": False,
            "retry_count": 0,
            "formal_guidance_unchanged": True,
            "formal_matrix_tail_alpha_force_multiplier": 1.0,
            "formal_matrix_fixed_lifting_surface_multiplier": 0.0,
            "diagnostic_tail_alpha_force_multiplier": 1.4,
            "tail_delta_force_multiplier": 1.0,
            "diagnostic_fixed_lifting_surface_multiplier_values": list(FIXED_LIFTING_SURFACE_MULTIPLIERS),
            "diagnostic_fixed_lifting_surface_station_x_m_values": list(FIXED_LIFTING_SURFACE_STATIONS_M),
            "frozen": [
                "tail moment arms", "finsLatAccel/current-mass scaling", "CNalpha/lambda/inertia",
                "K_Dalpha", "raw per-missile PID", "pid_error_scale=1.0", "tau_q/tau_a",
                "mechanical fin limits", "formal Ktgo/loft tables",
            ],
            "run_count": len(rows),
            "fixed_lifting_surface_sweep_run_count": len(fixed_lifting_sweep_rows),
            "fixed_lifting_surface_station_sweep_run_count": len(fixed_lifting_station_sweep_rows),
            "continuous_closest_approach_definition": "minimum norm of linearly interpolated relative-position segments; closing speed is positive when range is decreasing",
            "g_to_moment_definition": "maximum lateral-load G divided by maximum norm of recorded pitch/yaw total moment, units G per N m; aggregate diagnostic, not an aerodynamic coefficient",
            "energy_range_definition": "integrated positive drag-energy loss divided by trajectory path length, units J per m; path length is a trajectory proxy, not a validated maximum range",
        },
        "workspace": _worktree_provenance(),
        "provenance": {
            "manifest_path": "data/aam_non_tvc_manifest.json",
            "datamine": manifest["source"],
            "profiles": {
                missile_id: {
                    "path": f"missiles/{missile_id}.json",
                    "sha256": _sha256_file(ROOT / "missiles" / f"{missile_id}.json"),
                    "provenance": profile["provenance"],
                }
                for missile_id, profile in profiles.items()
            },
        },
        "plants": {
            plant: {
                "control_model_version": configs[(plant, "us_aim_120a")]["control_model_version"],
                "force_geometry_version": configs[(plant, "us_aim_120a")]["force_geometry_version"],
                "runtime_adapter": configs[(plant, "us_aim_120a")]["runtime_adapter"],
                "split_tail_candidate": (
                    copy.deepcopy(configs[(plant, "us_aim_120a")]["aerodynamics"].get("split_tail_candidate"))
                ),
                "fixed_lifting_surface_candidate": (
                    copy.deepcopy(configs[(plant, "us_aim_120a")]["aerodynamics"].get("fixed_lifting_surface_candidate"))
                ),
            }
            for plant in PLANTS
        },
        "scenario_source": "scripts/sweep_pid_error_scale.py:SCENARIOS and _scenario",
        "scenarios": scenario_inputs,
        "raw_pids": raw_pids,
        "runs": rows,
        "pl12_fixed_lifting_surface_sweep_runs": fixed_lifting_sweep_rows,
        "pl12_fixed_lifting_surface_station_sweep_runs": fixed_lifting_station_sweep_rows,
        "aggregate": {
            "completed": sum(row["status"] == "completed" for row in rows),
            "numerical_failure": sum(row["status"] == "numerical_failure" for row in rows),
            "exception": sum(row["status"] == "exception" for row in rows),
            "hit_or_fuse": sum(bool(row["hit_or_fuse"]) for row in rows),
            "elapsed_s": time.perf_counter() - started,
        },
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    print(json.dumps(artifact["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
