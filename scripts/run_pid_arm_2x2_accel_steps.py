#!/usr/bin/env python3
"""Offline old/new PID x effective-arm acceleration-step diagnostic.

The runner uses a copied PL-12 profile and explicit PID/arm overrides.  It
does not rewrite the profile, change the default runtime, or claim to execute
the War Thunder native solver.  The effective arm is applied only to the
existing unsupported body-Cm split-tail candidate's tail station, so the
artifact explicitly records that this is a local candidate separation test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.control import update_control_feedback  # noqa: E402
from aim120_model.dynamics import SimState, state_is_finite  # noqa: E402
from aim120_model.h2_dynamics import forces_for_state_h2, rk4_step_h2  # noqa: E402
from aim120_model.profile_adapter import (  # noqa: E402
    BODY_CM_TAIL_FORCE_PLANT,
    build_h2_candidate_config,
)
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402
from aim120_model.math3d import norm  # noqa: E402


MISSILE_ID = "cn_pl12"
EFFECTIVE_ARMS_M = (0.30, 0.12)
PID_OVERRIDES = {
    "old": {"p": 0.0046, "i": 0.0375, "d": 0.00015},
    "new": {"p": 0.0181, "i": 0.013, "d": 0.00025},
}
STEP_COMMANDS_G = (10.0, 15.0, 20.0, 25.0, 30.0)
DT_S = 0.002
DURATION_S = 2.0
SAMPLE_STRIDE = 5
INITIAL_SPEED_KMH = 1200.0
INITIAL_ALTITUDE_M = 6500.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _profile() -> dict[str, Any]:
    return json.loads(
        (ROOT / "missiles" / f"{MISSILE_ID}.json").read_text(encoding="utf-8")
    )


def _base_config() -> tuple[dict[str, Any], dict[str, Any]]:
    profile = _profile()
    defaults = json.loads(
        (ROOT / "config" / "profile_h2_runtime_defaults.json").read_text(
            encoding="utf-8"
        )
    )
    defaults["plant_model"] = BODY_CM_TAIL_FORCE_PLANT
    config, assumptions = build_h2_candidate_config(profile, defaults)
    return config, profile


def _experiment_config(base: dict[str, Any], arm_m: float, pid: dict[str, float]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["aerodynamics"]["tail_station_x_m"] = -float(arm_m)
    # These fields are audit labels in the existing split-tail candidate; the
    # runtime station above is the sole force/moment arm used by dynamics.
    config["aerodynamics"]["split_tail_candidate"]["tail_alpha_moment_arm_m"] = -float(arm_m)
    config["aerodynamics"]["split_tail_candidate"]["tail_delta_moment_arm_m"] = -float(arm_m)
    config["control"]["pid"].update({key: float(value) for key, value in pid.items()})
    return config


def _initial_state(config: dict[str, Any]) -> SimState:
    return SimState(
        position=(0.0, INITIAL_ALTITUDE_M, 0.0),
        velocity=(INITIAL_SPEED_KMH / 3.6, 0.0, 0.0),
        pitch=0.0,
        yaw=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        mass=float(config["geometry"]["initial_mass_kg"]),
    )


def _sample_row(time_s: float, diagnostics: Any, state: SimState) -> dict[str, float | bool]:
    return {
        "time_s": float(time_s),
        "alpha_rad": float(diagnostics.aero.pitch_alpha_rad),
        "alpha_deg": math.degrees(float(diagnostics.aero.pitch_alpha_rad)),
        "q_rad_s": float(state.pitch_rate),
        "q_deg_s": math.degrees(float(state.pitch_rate)),
        "delta_rad": float(state.actual_pitch_fin_angle_rad),
        "delta_deg": math.degrees(float(state.actual_pitch_fin_angle_rad)),
        "a_n_g": float(diagnostics.pitch_normal_acceleration_g),
        "wind_normal_a_n_g": float(diagnostics.wind_normal_pitch_acceleration_g),
        "V_mps": float(norm(diagnostics.aero.air_velocity_mps)),
        "dynamic_pressure_pa": float(diagnostics.aero.dynamic_pressure_pa),
    }


def _rise_time(samples: list[dict[str, float | bool]], command_g: float, fraction: float) -> float | None:
    threshold = command_g * fraction
    previous: dict[str, float | bool] | None = None
    for sample in samples:
        value = float(sample["a_n_g"])
        if value >= threshold:
            if previous is None:
                return float(sample["time_s"])
            y0 = float(previous["a_n_g"])
            y1 = value
            t0 = float(previous["time_s"])
            t1 = float(sample["time_s"])
            if y1 == y0:
                return t1
            return t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
        previous = sample
    return None


def _trapezoid(samples: list[dict[str, float | bool]], key: str) -> float:
    return sum(
        0.5
        * (float(left[key]) ** 2 + float(right[key]) ** 2)
        * max(float(right["time_s"]) - float(left["time_s"]), 0.0)
        for left, right in zip(samples, samples[1:])
    )


def _sign_reversals(samples: list[dict[str, float | bool]], key: str) -> int:
    signs: list[int] = []
    for sample in samples:
        value = float(sample[key])
        if abs(value) < 1.0e-9:
            continue
        sign = 1 if value > 0.0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return max(len(signs) - 1, 0)


def _run_step(config: dict[str, Any], command_g: float) -> dict[str, Any]:
    propulsion = PiecewisePropulsion.from_config(config)
    state = _initial_state(config)
    full_samples: list[dict[str, float | bool]] = []
    finite = True
    step_count = int(round(DURATION_S / DT_S))
    for index in range(step_count):
        time_s = index * DT_S
        before = forces_for_state_h2(state, time_s, config, propulsion, powered=False)
        state_feedback = replace(
            state,
            measured_pitch_normal_g=before.wind_normal_pitch_acceleration_g,
            measured_yaw_normal_g=before.wind_normal_yaw_acceleration_g,
        )
        updates = update_control_feedback(
            state_feedback,
            (float(command_g), 0.0),
            config,
            DT_S,
            enabled=True,
            plant_diagnostics=before,
        )
        controlled_state = replace(state_feedback, **updates)
        diagnostics = forces_for_state_h2(
            controlled_state, time_s, config, propulsion, powered=False
        )
        row = _sample_row(time_s, diagnostics, controlled_state)
        row["requested_delta_deg"] = math.degrees(
            float(controlled_state.pitch_requested_fin_command)
            * float(config["control"]["fin_actuator_travel"]["pitch_limit_rad"])
        )
        row["fin_limit_deg"] = math.degrees(
            float(config["control"]["fin_actuator_travel"]["pitch_limit_rad"])
        )
        row["actual_fin_fraction"] = abs(float(controlled_state.pitch_fin_command))
        full_samples.append(row)
        state = rk4_step_h2(
            controlled_state, time_s, DT_S, config, propulsion, powered=False
        )
        if not state_is_finite(state):
            finite = False
            break

    if not full_samples:
        raise RuntimeError("step produced no samples")
    sampled = [
        row
        for index, row in enumerate(full_samples)
        if index % SAMPLE_STRIDE == 0 or index == len(full_samples) - 1
    ]
    a_values = [float(row["a_n_g"]) for row in full_samples]
    alpha_values = [abs(float(row["alpha_deg"])) for row in full_samples]
    q_values = [abs(float(row["q_deg_s"])) for row in full_samples]
    delta_values = [abs(float(row["delta_deg"])) for row in full_samples]
    peak_g = max(a_values)
    rise_10 = _rise_time(full_samples, command_g, 0.10)
    rise_90 = _rise_time(full_samples, command_g, 0.90)
    return {
        "command_g": float(command_g),
        "finite": finite,
        "sample_count_full": len(full_samples),
        "sample_stride": SAMPLE_STRIDE,
        "samples": sampled,
        "metrics": {
            "alpha_max_abs_deg": max(alpha_values),
            "alpha_final_deg": float(full_samples[-1]["alpha_deg"]),
            "q_max_abs_deg_s": max(q_values),
            "q_final_deg_s": float(full_samples[-1]["q_deg_s"]),
            "delta_max_abs_deg": max(delta_values),
            "delta_final_deg": float(full_samples[-1]["delta_deg"]),
            "a_n_peak_g": peak_g,
            "a_n_final_g": float(full_samples[-1]["a_n_g"]),
            "V_initial_mps": float(full_samples[0]["V_mps"]),
            "V_final_mps": float(full_samples[-1]["V_mps"]),
            "rise_10_percent_s": rise_10,
            "rise_90_percent_s": rise_90,
            "rise_10_90_s": (
                None if rise_10 is None or rise_90 is None else rise_90 - rise_10
            ),
            "target_reached": peak_g >= command_g,
            "overshoot_status": (
                "defined_after_target_reached"
                if peak_g >= command_g
                else "not_defined_target_not_reached"
            ),
            "overshoot_g": max(peak_g - command_g, 0.0),
            "overshoot_percent": max((peak_g - command_g) / command_g * 100.0, 0.0),
            "integral_delta_squared_rad2_s": _trapezoid(full_samples, "delta_rad"),
            "a_n_sign_reversals": _sign_reversals(full_samples, "a_n_g"),
            "delta_saturated_sample_count": sum(
                1 for row in full_samples if float(row["actual_fin_fraction"]) >= 0.999
            ),
        },
    }


def _worktree_provenance() -> dict[str, Any]:
    status = _git("status", "--short").splitlines()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], cwd=ROOT
    )
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_short": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["pid_label"], float(row["effective_arm_m"]), float(row["command_g"])): row
        for row in rows
    }
    arm_effects: list[dict[str, Any]] = []
    pid_effects: list[dict[str, Any]] = []
    for pid_label in PID_OVERRIDES:
        for command_g in STEP_COMMANDS_G:
            short = by_key[(pid_label, 0.12, command_g)]
            long = by_key[(pid_label, 0.30, command_g)]
            arm_effects.append(
                {
                    "pid_label": pid_label,
                    "command_g": command_g,
                    "peak_g_arm_0.30_minus_0.12": (
                        long["metrics"]["a_n_peak_g"] - short["metrics"]["a_n_peak_g"]
                    ),
                    "rise_10_90_arm_0.30_minus_0.12_s": (
                        None
                        if long["metrics"]["rise_10_90_s"] is None
                        or short["metrics"]["rise_10_90_s"] is None
                        else long["metrics"]["rise_10_90_s"]
                        - short["metrics"]["rise_10_90_s"]
                    ),
                    "overshoot_percent_arm_0.30_minus_0.12": (
                        long["metrics"]["overshoot_percent"]
                        - short["metrics"]["overshoot_percent"]
                    ),
                }
            )
    for arm_m in EFFECTIVE_ARMS_M:
        for command_g in STEP_COMMANDS_G:
            old = by_key[("old", 0.12, command_g)]
            new = by_key[("new", 0.12, command_g)]
            if arm_m != 0.12:
                old = by_key[("old", float(arm_m), command_g)]
                new = by_key[("new", float(arm_m), command_g)]
            pid_effects.append(
                {
                    "effective_arm_m": float(arm_m),
                    "command_g": command_g,
                    "peak_g_new_minus_old": (
                        new["metrics"]["a_n_peak_g"] - old["metrics"]["a_n_peak_g"]
                    ),
                    "rise_10_90_new_minus_old_s": (
                        None
                        if old["metrics"]["rise_10_90_s"] is None
                        or new["metrics"]["rise_10_90_s"] is None
                        else new["metrics"]["rise_10_90_s"]
                        - old["metrics"]["rise_10_90_s"]
                    ),
                    "overshoot_percent_new_minus_old": (
                        new["metrics"]["overshoot_percent"]
                        - old["metrics"]["overshoot_percent"]
                    ),
                }
            )
    return {
        "arm_effects": arm_effects,
        "pid_effects": pid_effects,
        "pid_effects_at_arm_0.12": [
            row for row in pid_effects if row["effective_arm_m"] == 0.12
        ],
        "conclusion": {
            "supports_native_claim_pid_controls_oscillation_arm_controls_max_g": False,
            "status": "candidate_only_not_native_identification",
            "reason": (
                "The runner applies explicit arm/PID overrides to the local unsupported "
                "body-Cm split-tail candidate. Its first-order rate inner loop, empirical "
                "tail-force law, provisional inertia, and arm semantics are not the native "
                "War Thunder controller/plant. Results are therefore a bounded local 2x2 "
                "separation record, not evidence for native causality."
            ),
            "within_candidate_observation": (
                "Compare the complete arm_effects and pid_effects tables; no thresholded "
                "causal label is assigned."
            ),
        },
    }


def main() -> int:
    base, profile = _base_config()
    rows: list[dict[str, Any]] = []
    for pid_label, pid in PID_OVERRIDES.items():
        for arm_m in EFFECTIVE_ARMS_M:
            config = _experiment_config(base, arm_m, pid)
            for command_g in STEP_COMMANDS_G:
                result = _run_step(config, command_g)
                rows.append(
                    {
                        "pid_label": pid_label,
                        "pid_override": pid,
                        "effective_arm_m": arm_m,
                        "command_g": command_g,
                        "plant_model": BODY_CM_TAIL_FORCE_PLANT,
                        "result": result,
                        "metrics": result["metrics"],
                    }
                )
    artifact = {
        "schema_version": 1,
        "artifact_type": "offline_pid_effective_arm_2x2_acceleration_step_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "py -3 scripts\\run_pid_arm_2x2_accel_steps.py",
        "offline_only": True,
        "calculate_runs": 0,
        "network_accessed": False,
        "game_files_modified": False,
        "missile_id": MISSILE_ID,
        "plant_model": BODY_CM_TAIL_FORCE_PLANT,
        "experiment_conditions": {
            "initial_speed_kmh": INITIAL_SPEED_KMH,
            "initial_altitude_m": INITIAL_ALTITUDE_M,
            "powered": False,
            "dt_s": DT_S,
            "duration_s": DURATION_S,
            "step_commands_g": list(STEP_COMMANDS_G),
            "effective_arms_m": list(EFFECTIVE_ARMS_M),
            "response_output": "pitch_normal_acceleration_g",
            "alpha_output": "CG pitch flow alpha",
            "q_output": "body pitch rate",
            "delta_output": "actual pitch fin angle",
            "integral_delta_squared_units": "rad^2*s",
        },
        "pid_overrides": PID_OVERRIDES,
        "raw_profile_pid_before_override": profile["control"]["pid"],
        "model_boundary": (
            "Unsupported local body-Cm split-tail candidate; effective arm is an explicit "
            "diagnostic override of tail_station_x_m, not a sourced native distFromCmToStab mapping."
        ),
        "provenance": _worktree_provenance(),
        "source_sha256": {
            "missiles/cn_pl12.json": _sha256_file(ROOT / "missiles" / "cn_pl12.json"),
            "config/profile_h2_runtime_defaults.json": _sha256_file(
                ROOT / "config" / "profile_h2_runtime_defaults.json"
            ),
            "src/aim120_model/profile_adapter.py": _sha256_file(
                ROOT / "src" / "aim120_model" / "profile_adapter.py"
            ),
            "src/aim120_model/h2_dynamics.py": _sha256_file(
                ROOT / "src" / "aim120_model" / "h2_dynamics.py"
            ),
            "src/aim120_model/control.py": _sha256_file(
                ROOT / "src" / "aim120_model" / "control.py"
            ),
            "src/aim120_model/h2_simulator.py": _sha256_file(
                ROOT / "src" / "aim120_model" / "h2_simulator.py"
            ),
            "scripts/run_pid_arm_2x2_accel_steps.py": _sha256_file(Path(__file__)),
        },
        "runs": rows,
        "comparison": _comparison(rows),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs" / f"pid_arm_2x2_accel_steps_{timestamp}.json"
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    digest = _sha256_file(output)
    hash_path = output.with_suffix(output.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"artifact={output}")
    print(f"sha256={digest}")
    print(f"sha256_file={hash_path}")
    print(f"runs={len(rows)}")
    for row in rows:
        metrics = row["metrics"]
        print(
            "{pid} arm={arm:.2f} cmd={cmd:.0f} peak={peak:.6g} final={final:.6g} "
            "rise10_90={rise} overshoot={over:.6g}% delta2={delta:.6g}".format(
                pid=row["pid_label"],
                arm=row["effective_arm_m"],
                cmd=row["command_g"],
                peak=metrics["a_n_peak_g"],
                final=metrics["a_n_final_g"],
                rise=metrics["rise_10_90_s"],
                over=metrics["overshoot_percent"],
                delta=metrics["integral_delta_squared_rad2_s"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
