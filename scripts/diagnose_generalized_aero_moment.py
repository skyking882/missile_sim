#!/usr/bin/env python3
"""Offline local Jacobian audit for the unsupported generalized CN/Cm candidate.

This is a bounded plant diagnostic, not a combat-case tuning sweep.  It
perturbs one shared state input at a time and records the independent normal
force and moment derivatives, then performs two coefficient-isolation
counterfactuals.  No network, browser, Calculate, or War Thunder file is
accessed.
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
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.dynamics import SimState, state_is_finite  # noqa: E402
from aim120_model.h2_dynamics import forces_for_state_h2, rk4_step_h2  # noqa: E402
from aim120_model.profile_adapter import (  # noqa: E402
    GENERALIZED_AERO_MOMENT_PLANT,
    build_h2_candidate_config,
)
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402


MISSILE_ID = "us_aim_120a"
EPSILON = 1.0e-5
RATE_EPSILON = 1.0e-4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _config(overrides: dict[str, float] | None = None) -> dict[str, Any]:
    profile = json.loads(
        (ROOT / "missiles" / f"{MISSILE_ID}.json").read_text(encoding="utf-8")
    )
    defaults = json.loads(
        (ROOT / "config" / "profile_h2_runtime_defaults.json").read_text(
            encoding="utf-8"
        )
    )
    defaults["plant_model"] = GENERALIZED_AERO_MOMENT_PLANT
    if overrides:
        defaults["generalized_aero_moment_candidate"].update(overrides)
    config, _assumptions = build_h2_candidate_config(profile, defaults)
    return config


def _state(config: dict[str, Any], **kwargs: float) -> SimState:
    return SimState(
        position=(0.0, 3000.0, 0.0),
        velocity=(300.0, 0.0, 0.0),
        pitch=float(kwargs.get("pitch", 0.0)),
        yaw=float(kwargs.get("yaw", 0.0)),
        pitch_rate=float(kwargs.get("pitch_rate", 0.0)),
        yaw_rate=float(kwargs.get("yaw_rate", 0.0)),
        mass=float(config["geometry"]["initial_mass_kg"]),
        actual_pitch_fin_angle_rad=float(kwargs.get("pitch_fin", 0.0)),
        actual_yaw_fin_angle_rad=float(kwargs.get("yaw_fin", 0.0)),
    )


def _diag(config: dict[str, Any], state: SimState):
    return forces_for_state_h2(
        state,
        0.0,
        config,
        PiecewisePropulsion.from_config(config),
        powered=False,
    )


def _central(
    plus: Any,
    minus: Any,
    field: str,
    step: float,
) -> float:
    return (float(getattr(plus, field)) - float(getattr(minus, field))) / (2.0 * step)


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1.0e-12)


def _worktree_provenance() -> dict[str, Any]:
    status = _git("status", "--short").splitlines()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=ROOT,
    )
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_short": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _derivatives(config: dict[str, Any]) -> dict[str, Any]:
    prop = PiecewisePropulsion.from_config(config)
    zero = _diag(config, _state(config))
    alpha_plus = _diag(config, _state(config, pitch=EPSILON))
    alpha_minus = _diag(config, _state(config, pitch=-EPSILON))
    delta_plus = _diag(config, _state(config, pitch_fin=EPSILON))
    delta_minus = _diag(config, _state(config, pitch_fin=-EPSILON))
    rate_plus = _diag(config, _state(config, pitch_rate=RATE_EPSILON))
    rate_minus = _diag(config, _state(config, pitch_rate=-RATE_EPSILON))

    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    q_dyn = float(zero.aero.normal_force_dynamic_pressure_pa)
    area = float(zero.body_reference_area_m2)
    length = float(zero.body_reference_length_m)
    speed = float(zero.aero.speed_mps)
    q_s = q_dyn * area
    expected = {
        "dN_dalpha_N_per_rad": q_s * float(candidate["cn_alpha_per_rad"]),
        "dM_dalpha_Nm_per_rad": q_s * length * float(candidate["cm_alpha_per_rad"]),
        "dN_ddelta_N_per_rad": q_s * float(candidate["cn_delta_per_rad"]),
        "dM_ddelta_Nm_per_rad": q_s * length * float(candidate["cm_delta_per_rad"]),
        "dM_drate_Nm_per_rad_s": (
            q_s
            * length
            * float(candidate["cm_q"])
            * length
            / (2.0 * speed)
        ),
    }
    observed = {
        "dN_dalpha_N_per_rad": _central(
            alpha_plus, alpha_minus, "pitch_body_normal_force_n", EPSILON
        ),
        "dM_dalpha_Nm_per_rad": _central(
            alpha_plus, alpha_minus, "pitch_body_static_moment_nm", EPSILON
        ),
        "dN_ddelta_N_per_rad": _central(
            delta_plus, delta_minus, "pitch_tail_force_n", EPSILON
        ),
        "dM_ddelta_Nm_per_rad": _central(
            delta_plus, delta_minus, "pitch_tail_moment_nm", EPSILON
        ),
        "dM_drate_Nm_per_rad_s": _central(
            rate_plus, rate_minus, "pitch_body_rate_moment_nm", RATE_EPSILON
        ),
    }
    errors = {
        key: _relative_error(observed[key], expected[key]) for key in expected
    }

    # A short fixed-fin integration is only a numerical finite/boundedness
    # check.  It is intentionally not used to fit a coefficient.
    state = _state(config, pitch_fin=0.02)
    max_abs_rate = 0.0
    for index in range(100):
        state = rk4_step_h2(state, index * 0.001, 0.001, config, prop, False)
        if not state_is_finite(state):
            raise AssertionError("generalized candidate integration became non-finite")
        max_abs_rate = max(max_abs_rate, abs(state.pitch_rate), abs(state.yaw_rate))

    return {
        "fixed_state": {
            "velocity_mps": list(_state(config).velocity),
            "mass_kg": float(_state(config).mass),
            "dynamic_pressure_pa": q_dyn,
            "reference_area_m2": area,
            "reference_length_m": length,
            "epsilon_alpha_delta_rad": EPSILON,
            "epsilon_rate_rad_s": RATE_EPSILON,
        },
        "expected": expected,
        "observed": observed,
        "relative_error": errors,
        "maximum_relative_error": max(errors.values()),
        "zero_state": {
            "pitch_total_moment_nm": float(zero.pitch_total_moment_nm),
            "yaw_total_moment_nm": float(zero.yaw_total_moment_nm),
            "lateral_load_g": float(zero.lateral_load_g),
        },
        "short_fixed_fin_integration": {
            "duration_s": 0.1,
            "maximum_abs_body_rate_rad_s": max_abs_rate,
            "finite": True,
        },
        "alpha_dot_identifiability": {
            "runtime_cm_alpha_dot_per_rad": float(
                zero.generalized_cm_alpha_dot_per_rad
            ),
            "runtime_alpha_dot_hat_pitch": float(
                zero.generalized_pitch_alpha_dot_hat
            ),
            "runtime_alpha_dot_hat_yaw": float(
                zero.generalized_yaw_alpha_dot_hat
            ),
            "runtime_enabled": bool(
                zero.generalized_cm_alpha_dot_runtime_enabled
            ),
            "status": (
                "not_independently_identifiable; alpha_dot state unavailable and "
                "runtime Cm_alpha_dot is frozen at zero; rate probe identifies Cm_q only"
            ),
        },
    }


def _intervention(config: dict[str, Any], coefficient: str, factor: float) -> dict[str, Any]:
    changed = copy.deepcopy(config)
    changed["aerodynamics"]["generalized_aero_moment_candidate"][coefficient] *= factor
    plus = _diag(changed, _state(changed, pitch_fin=EPSILON))
    minus = _diag(changed, _state(changed, pitch_fin=-EPSILON))
    return {
        "coefficient_changed": coefficient,
        "factor": factor,
        "dN_ddelta_N_per_rad": _central(plus, minus, "pitch_tail_force_n", EPSILON),
        "dM_ddelta_Nm_per_rad": _central(plus, minus, "pitch_tail_moment_nm", EPSILON),
    }


def main() -> int:
    config = _config()
    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    baseline = _derivatives(config)
    interventions = [
        _intervention(config, "cm_delta_per_rad", 2.0),
        _intervention(config, "cn_delta_per_rad", 2.0),
    ]
    artifact = {
        "schema_version": 1,
        "artifact_type": "offline_generalized_aero_moment_identifiability_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "py -3 scripts\\diagnose_generalized_aero_moment.py",
        "offline_only": True,
        "calculate_runs": 0,
        "game_files_modified": False,
        "missile_id": MISSILE_ID,
        "runtime_adapter": config["runtime_adapter"],
        "plant_semantics": config["control"]["plant_semantics"],
        "model_boundary": config["reference"]["runtime_boundary"],
        "equations": candidate["equations"],
        "parameter_boundary": candidate["parameter_boundary"],
        "nominal_parameters": {
            key: candidate[key]
            for key in (
                "cn_alpha_per_rad",
                "cn_delta_per_rad",
                "cm_alpha_per_rad",
                "cm_delta_per_rad",
                "cm_q",
                "cm_alpha_dot",
                "cm_alpha_dot_requested",
            )
        },
        "alpha_dot_identifiability": {
            "runtime_status": candidate["cm_alpha_dot_runtime_status"],
            "requested_value": candidate["cm_alpha_dot_requested"],
            "runtime_value": candidate["cm_alpha_dot"],
            "independent_state_available": False,
            "identified_rate_term": "Cm_q only",
        },
        "provenance": _worktree_provenance(),
        "source_sha256": {
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
            "config/profile_h2_runtime_defaults.json": _sha256_file(
                ROOT / "config" / "profile_h2_runtime_defaults.json"
            ),
            f"missiles/{MISSILE_ID}.json": _sha256_file(
                ROOT / "missiles" / f"{MISSILE_ID}.json"
            ),
        },
        "derivative_audit": baseline,
        "coefficient_isolation_interventions": interventions,
        "interpretation": {
            "force_moment_station_constraint_removed": True,
            "cm_delta_change_keeps_delta_force_derivative": (
                interventions[0]["dN_ddelta_N_per_rad"]
                == baseline["observed"]["dN_ddelta_N_per_rad"]
            ),
            "cn_delta_change_keeps_delta_moment_derivative": (
                interventions[1]["dM_ddelta_Nm_per_rad"]
                == baseline["observed"]["dM_ddelta_Nm_per_rad"]
            ),
            "identifiability_scope": (
                "local structural independence only; coefficients remain unsupported "
                "and are not identified as War Thunder physics; Cm_alpha_dot is not "
                "independently identifiable in this runtime"
            ),
        },
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs" / f"generalized_aero_moment_identifiability_{timestamp}.json"
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = _sha256_file(output)
    hash_path = output.with_suffix(output.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"artifact={output}")
    print(f"sha256={digest}")
    print(f"sha256_file={hash_path}")
    print(f"maximum_relative_error={baseline['maximum_relative_error']:.3e}")
    print(
        "cm_delta_x2=(dN={:.9g}, dM={:.9g})".format(
            interventions[0]["dN_ddelta_N_per_rad"],
            interventions[0]["dM_ddelta_Nm_per_rad"],
        )
    )
    print(
        "cn_delta_x2=(dN={:.9g}, dM={:.9g})".format(
            interventions[1]["dN_ddelta_N_per_rad"],
            interventions[1]["dM_ddelta_Nm_per_rad"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
