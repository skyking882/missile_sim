#!/usr/bin/env python3
"""Bounded JARI-only calibration and launch-only prediction for a game unit.

The reference exports are commercial-video-game simulator observations. This
script is an opt-in experiment; it never reads native replay observations,
historical model parameters or historical forecast results. It separates
display-force fitting, source-conditioned angle response and free prediction.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.effective_surface_core import EffectiveSurfaceGame, forward_from_files, read_raw
import prepare_r771_jari_only as ds

OUT = ROOT/"outputs/jari_only/calibration"
CONFIG = ROOT/"research/legacy/config/r771_jari_only.json"
RAW = ROOT/"data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
G = 9.80665
PN = ("pure_velocity_normal", "true_los_normal", "ideal_relative_velocity_normal")
UNITS = ("G", "m/s^2")
CHANNELS = ("speed_mps", "body_pitch_deg", "body_yaw_deg", "geometric_incidence_unsigned_deg", "normal_specific_load_g")
METRIC_SOURCE = {"normal_specific_load_g": "normal_specific_load_5point_g"}
FIT_TOLERANCES = {"early": dict(zip(CHANNELS, (10., 3., 3., 3., 2.))),
                  "full": dict(zip(CHANNELS, (20., 5., 5., 3., 2.)))}


def jsonable(value: Any) -> Any:
    return ds.jsonable(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def protected() -> dict[str, tuple[int, int]]:
    owned = {ROOT/"src/aim120_model/effective_surface_core.py", CONFIG}
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in ("src", "config", "configs", "schemas", "docs", "data", "outputs/jari_offaxis", "outputs/jari_only/dataset")
            for p in (ROOT/folder).rglob("*") if p.is_file() and p not in owned
            and "__pycache__" not in p.parts and ".git" not in p.parts}


def atmospheric(a: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [StandardAtmosphere().sample(float(h)) for h in a["asl_source_m"]]
    rho = np.asarray([s.density_kg_m3 for s in samples])
    mach = a["speed_mps"]/np.asarray([s.speed_of_sound_mps for s in samples])
    dynamic = .5*rho*a["speed_mps"]**2
    return rho, mach, dynamic


def rmse(delta: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(delta)**2)))


def fit_force_parameters(fit: dict[str, dict[str, np.ndarray]], raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Four drag coefficients and three lift coefficients; site fit only."""
    blocks = [fit[c] for c in ds.FIT_CASES]
    atm = [atmospheric(a) for a in blocks]
    mach = np.concatenate([x[1] for x in atm])
    dynamic = np.concatenate([x[2] for x in atm])
    mass = np.concatenate([a["mass_source_kg"] for a in blocks])
    alpha = np.radians(np.concatenate([a["geometric_incidence_unsigned_deg"] for a in blocks]))
    cd0 = np.concatenate([a["cd_base_source"] for a in blocks])
    cda = np.concatenate([a["cd_aoa_source"] for a in blocks])
    matrix = np.column_stack((np.ones(len(mach)), 1/mach, 1/mach**2))
    base = np.linalg.lstsq(matrix, cd0, rcond=None)[0]
    incidence_basis = cd0*np.sin(alpha)**2
    incidence = float(incidence_basis@cda/(incidence_basis@incidence_basis))
    area = math.pi*raw["caliber"]**2/4
    lift = np.concatenate([a["acc_lift_source_mps2"] for a in blocks])
    usable = alpha > math.radians(.8)
    basis0 = dynamic*area*np.sin(alpha)/mass
    # Two finite grids, specified before validation. The slope is solved exactly
    # at each grid point. No optimizer, new package or extra forward run.
    best = None
    evaluations = 0
    for stall in np.arange(.30, .800001, .02):
        for width in np.arange(.10, .900001, .02):
            basis = basis0*np.exp(-(np.maximum(alpha-stall, 0.)/width)**2)
            k = float(basis[usable]@lift[usable]/(basis[usable]@basis[usable]))
            error = rmse((basis*k-lift)[usable])
            evaluations += 1
            if best is None or error < best[0]:
                best = (error, k, float(stall), float(width))
    _, _, center, width0 = best
    for stall in np.linspace(center-.02, center+.02, 17):
        for width in np.linspace(width0-.02, width0+.02, 17):
            basis = basis0*np.exp(-(np.maximum(alpha-stall, 0.)/width)**2)
            k = float(basis[usable]@lift[usable]/(basis[usable]@basis[usable]))
            error = rmse((basis*k-lift)[usable])
            evaluations += 1
            if error < best[0]:
                best = (error, k, float(stall), float(width))
    parameters = {"cd_base_inverse_mach": base.tolist(), "cd_incidence_ratio": incidence,
                  "lift_sine_slope": best[1], "lift_stall_rad": best[2], "lift_rolloff_width_rad": best[3]}
    diagnostics: dict[str, Any] = {"coefficients_count": 7, "linear_solves": 2,
        "lift_parameter_grid_evaluations": evaluations,
        "base_drag_cd_rmse": rmse(matrix@base-cd0), "base_drag_cd_max_abs": float(np.max(np.abs(matrix@base-cd0))),
        "incidence_drag_cd_rmse": rmse(incidence*incidence_basis-cda),
        "lift_display_accel_rmse_mps2": best[0], "lift_stall_degrees": math.degrees(best[2]),
        "lift_width_degrees": math.degrees(best[3]), "source_closure": {},
        "identified_scope": "Effective scalar force laws over site fit support; export update phase and physical surface area are not independently identified"}
    for a, (rho, _, q), case in zip(blocks, atm, ds.FIT_CASES):
        tangent = a["velocity_tangent"]
        forward = a["body_forward"]
        normal = a["specific_acceleration_5point_xyz_mps2"]-np.sum(a["specific_acceleration_5point_xyz_mps2"]*tangent, axis=1)[:, None]*tangent
        thrust_acc = a["thrust_source_n"][:, None]*forward/a["mass_source_kg"][:, None]
        thrust_normal = thrust_acc-np.sum(thrust_acc*tangent, axis=1)[:, None]*tangent
        lift_vector = normal-thrust_normal
        direction = forward-np.sum(forward*tangent, axis=1)[:, None]*tangent
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-12)
        lift_scalar = np.sum(lift_vector*direction, axis=1)
        mask = a["derivative_5point_valid_mask"] & (a["t"] > .34) & (a["t"] < 6.9) & (a["geometric_incidence_unsigned_deg"] > 1.)
        diagnostics["source_closure"][case] = {
            "inferred_drag_area_ratio_to_raw_cross_section_min_median_max": np.quantile(a["drag_base_source_n"]/(q*a["cd_base_source"]*area), [0., .5, 1.]),
            "lift_display_minus_kinematic_rmse_mps2": rmse(lift_scalar[mask]-a["acc_lift_source_mps2"][mask]),
            "lift_direction_nonparallel_vector_rms_mps2": float(np.sqrt(np.mean(np.sum((lift_vector[mask]-lift_scalar[mask, None]*direction[mask])**2, axis=1)))),
            "positive_projection_rows_fraction": float(np.mean(lift_scalar[mask] > 0.)),
            "mass_raw_formula_max_abs_kg": float(np.max(np.abs(a["mass_source_kg"]-(raw["mass"]+(raw["massEnd"]-raw["mass"])*np.clip(a["t"]/raw["timeFire"], 0., 1.))))),
            "thrust_raw_formula_max_abs_n": float(np.max(np.abs(a["thrust_source_n"]-np.where(a["t"] < raw["timeFire"], raw["force"], 0.)))),
            "thrust_display_phase_note": "Creation row t=0 displays zero thrust before the first powered update; all subsequent exported thrust rows match raw 23835 N until 7 s burnout. Runtime applies power from its first integration tick"}
    return parameters, diagnostics


def fit_angle_parameters(fit: dict[str, dict[str, np.ndarray]], raw: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    matrices, derivatives, case_blocks = [], [], {}
    matrices_lift = []
    for case in ds.FIT_CASES:
        a = fit[case]
        _, _, q = atmospheric(a)
        t, v, m = a["t"], a["speed_mps"], a["mass_source_kg"]
        omega = -a["omega_export_raw"]
        omega_dot = ds.derivative(omega, t, 5)
        body_v = np.column_stack([np.sum(a["velocity_xyz_mps"]*a[k], axis=1) for k in ("body_forward", "body_up", "body_right")])
        local_v = body_v+np.column_stack((np.zeros(len(t)), -raw["distFromCmToStab"]*omega[:, 2], raw["distFromCmToStab"]*omega[:, 1]))
        ap = np.arctan2(-local_v[:, 1], local_v[:, 0])
        ay = np.arctan2(-local_v[:, 2], local_v[:, 0])
        restoring = np.column_stack((np.zeros(len(t)), ay, -ap))
        control = np.column_stack((np.zeros(len(t)), -a["controls_export_raw"][:, 0], a["controls_export_raw"][:, 1]))
        A = (q/m)/(q[0]/m[0])
        B = A*v[0]/v
        X = np.stack((A[:, None]*control, A[:, None]*restoring, -B[:, None]*omega), axis=-1)
        valid = a["derivative_5point_valid_mask"] & (t >= .375) & (t <= 3.5)
        xx, yy = X[valid, 1:3, :].reshape(-1, 3), omega_dot[valid, 1:3].reshape(-1)
        matrices.append(xx)
        derivatives.append(yy)
        case_blocks[case] = (xx, yy)
        f, tangent = a["body_forward"], a["velocity_tangent"]
        lift_direction = f-np.sum(f*tangent, axis=1)[:, None]*tangent
        lift_direction /= np.maximum(np.linalg.norm(lift_direction, axis=1)[:, None], 1e-12)
        lift_vector = lift_direction*a["acc_lift_source_mps2"][:, None]*m[:, None]
        torque = np.cross(-raw["distFromCmToStab"]*f, lift_vector)
        torque_body = np.column_stack([np.sum(torque*a[k], axis=1)/m for k in ("body_forward", "body_up", "body_right")])
        X_lift = np.stack((A[:, None]*control, torque_body, -B[:, None]*omega), axis=-1)
        matrices_lift.append(X_lift[valid, 1:3, :].reshape(-1, 3))
    matrix, derivative = np.concatenate(matrices), np.concatenate(derivatives)
    coefficients = np.linalg.lstsq(matrix, derivative, rcond=None)[0]
    if np.any(coefficients <= 0.):
        raise ValueError("Angle response identification requires positive control/restoring/damping coefficients")
    lift_matrix = np.concatenate(matrices_lift)
    lift_coefficients = np.linalg.lstsq(lift_matrix, derivative, rcond=None)[0]
    parameters = dict(zip(("control_response_rad_s2", "restoring_response_rad_s2_per_rad", "damping_response_per_s"), coefficients.tolist()))
    return parameters, {"coefficients_count": 3, "linear_solves": 2,
        "effective_local_incidence_acceleration_rmse_rad_s2": rmse(matrix@coefficients-derivative),
        "observed_acceleration_rms_rad_s2": rmse(derivative),
        "literal_lift_lever_only_acceleration_rmse_rad_s2": rmse(lift_matrix@lift_coefficients-derivative),
        "per_case_acceleration_rmse_rad_s2": {c: rmse(x@coefficients-y) for c, (x, y) in case_blocks.items()},
        "control_units": "effective JARI export channel, unknown physical scale/hold phase",
        "omega_interpretation": "negative exported omega, body frame, consistent approximately with XYZW Hamilton differences",
        "single_surface": {"station_m": -.4, "force_surface_count": 1, "independent_same_station_couple": True,
                           "reason": "Observed lift rolls off before observed angular restoration; lift lever alone leaves larger known-input residual"}}


def arrays_from_rows(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
    result["body_yaw_deg"] = np.degrees(np.unwrap(np.radians(result["body_yaw_deg"])))
    return result


def interpolate(values: np.ndarray, t: np.ndarray, query: np.ndarray | float) -> np.ndarray:
    query = np.asarray(query)
    if np.any(query < t[0]-1e-10) or np.any(query > t[-1]+1e-10):
        raise ValueError("Interpolation outside support is prohibited")
    if values.ndim == 1:
        return np.interp(query, t, values)
    return np.stack([np.interp(query, t, values[:, i]) for i in range(values.shape[1])], axis=-1)


def source_conditioned_pid(config: dict[str, Any], raw: dict[str, Any], creation: dict[str, Any], a: dict[str, np.ndarray]) -> dict[str, Any]:
    """Offline controller diagnostic: observed states drive every tick.

    This intentionally uses interpolation of future display rows. It is never
    used by a free prediction and is not evidence of known guidance requests.
    """
    model = EffectiveSurfaceGame(raw, config, creation)
    last = min(3.5, a["t"][-1])
    controls, times = [], []
    first_active = None
    for tick in range(math.floor(last/model.dt)+1):
        t = tick*model.dt
        pos = interpolate(a["position_xyz_m"], a["t"], t)
        vel = interpolate(a["velocity_xyz_mps"], a["t"], t)
        quat = interpolate(a["quaternion_xyzw_normalized"], a["t"], t)
        quat /= np.linalg.norm(quat)
        omega = -interpolate(a["omega_export_raw"], a["t"], t)
        state = tuple(np.concatenate((pos, vel, quat, omega)))
        if tick:
            previous_v = interpolate(a["velocity_xyz_mps"], a["t"], t-model.dt)
            model.previous_acceleration = tuple((vel-previous_v)/model.dt)
        u = model.update_control(t, state)
        times.append(t+model.dt)
        controls.append(u)
        if first_active is None and max(abs(x) for x in u) > .001:
            first_active = {"tick_time_s": t, "control": u, "request_mps2": model.last_request,
                            "pid_error": model.last_error}
    controls, times = np.asarray(controls), np.asarray(times)
    mask = (a["t"] >= times[0]) & (a["t"] <= times[-1]) & (a["t"] <= 3.5)
    predicted = interpolate(controls, times, a["t"][mask])
    delta = predicted-a["controls_export_raw"][mask]
    initial_mask = (a["t"][mask] >= .333-1e-6) & (a["t"][mask] <= .42)
    return {"horizontal_control_rmse": rmse(delta[:, 0]), "vertical_control_rmse": rmse(delta[:, 1]),
            "initial_control_max_abs_error": float(np.max(np.abs(delta[initial_mask]))),
            "first_active": first_active, "source_forcing": True,
            "unidentified_phase": "Export covers every second physics tick; intermediate controls/feedback phase not observed"}


def measure(source: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    t = source["t"]
    support = (t >= prediction["t"][0]) & (t <= prediction["t"][-1])
    result: dict[str, Any] = {"source_rows": len(t), "prediction_support_s": [float(prediction["t"][0]), float(prediction["t"][-1])],
        "source_support_s": [float(t[0]), float(t[-1])], "uncovered_source_rows": int((~support).sum()), "windows": {}}
    for window, window_mask in (("early", t <= 3.5), ("full", np.ones(len(t), dtype=bool))):
        result["windows"][window] = {}
        for channel in CHANNELS:
            actual_key = METRIC_SOURCE.get(channel, channel)
            valid = support & window_mask
            if channel == "normal_specific_load_g":
                valid &= source["derivative_5point_valid_mask"]
            observed = source[actual_key][valid]
            predicted = interpolate(prediction[channel], prediction["t"], t[valid])
            delta = predicted-observed
            if channel == "body_yaw_deg":
                delta = (delta+180.) % 360.-180.
            result["windows"][window][channel] = {"units": "m/s" if channel == "speed_mps" else "G" if channel.endswith("_g") else "deg",
                "rows": int(valid.sum()), "rmse": rmse(delta), "mean_signed_error": float(np.mean(delta)),
                "max_abs_error": float(np.max(np.abs(delta))), "predeclared_rmse_tolerance": FIT_TOLERANCES[window][channel]}
        # Keep the source AoA diagnostic separate from geometry.
        valid = support & window_mask
        predicted = interpolate(prediction["geometric_incidence_unsigned_deg"], prediction["t"], t[valid])
        result["windows"][window]["source_unsigned_aoa_deg"] = {"rows": int(valid.sum()), "rmse": rmse(predicted-source["aoa_source_unsigned_deg"][valid]), "units": "deg"}
    mask = (t >= .30) & (t <= 1.2) & support
    result["first_peak_window"] = {}
    for channel in ("body_pitch_deg", "body_yaw_deg", "geometric_incidence_unsigned_deg"):
        pred = interpolate(prediction[channel], prediction["t"], t[mask])
        truth = source[channel][mask]
        # Max absolute yaw handles the mirrored site check.
        i, j = int(np.argmax(abs(pred))), int(np.argmax(abs(truth)))
        result["first_peak_window"][channel] = {"prediction": float(pred[i]), "source": float(truth[j]),
            "prediction_time_s": float(t[mask][i]), "source_time_s": float(t[mask][j])}
    return result


def known_control_angle_response(raw: dict[str, Any], config: dict[str, Any], creation: dict[str, Any], a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Known exported-channel input, source-velocity-forced angle response only."""
    model = EffectiveSurfaceGame(raw, config, creation)
    state = tuple(creation["position_xyz_m"])+tuple(creation["velocity_xyz_mps"])+tuple(creation["quaternion_xyzw_raw"])+tuple(-x for x in creation["omega_export_raw"])
    rows = []
    ticks = int(math.floor(3.5/model.dt))
    for tick in range(ticks+1):
        t = tick*model.dt
        pos = tuple(interpolate(a["position_xyz_m"], a["t"], t))
        vel = tuple(interpolate(a["velocity_xyz_mps"], a["t"], t))
        state = pos+vel+state[6:]
        control = tuple(interpolate(a["controls_export_raw"], a["t"], t))
        _, diagnostic = model.plant(t, state, control)
        rows.append({"t": t, "body_pitch_deg": diagnostic["body_pitch_deg"], "body_yaw_deg": diagnostic["body_yaw_deg"],
                     "body_omega_xyz_rad_s": state[10:], "geometric_incidence_unsigned_deg": diagnostic["geometric_incidence_unsigned_deg"]})
        h = model.dt/model.integration_substeps
        for j in range(model.integration_substeps):
            ts = t+j*h
            # Export control denotes the last held tick. Interpolating through
            # its endpoint is only an explicitly source-conditioned diagnostic.
            cs = tuple(interpolate(a["controls_export_raw"], a["t"], ts+model.dt))
            pos = tuple(interpolate(a["position_xyz_m"], a["t"], ts))
            vel = tuple(interpolate(a["velocity_xyz_mps"], a["t"], ts))
            state = model.step(ts, pos+vel+state[6:], cs, h)
    return arrays_from_rows(rows)


def plot_all(source: dict[str, dict[str, np.ndarray]], forecasts: dict[str, dict[str, np.ndarray]], path: Path, early: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ds.FIT_CASES+ds.VALIDATION_CASES
    labels = ("Speed [m/s]", "Pitch [deg]", "Yaw [deg]", "Unsigned incidence [deg]", "Normal specific load [G]")
    fig, axes_grid = plt.subplots(10, 5, figsize=(19, 25), squeeze=False)
    for row, case in enumerate(order):
        actual, pred = source[case], forecasts[case]
        for col, (channel, label) in enumerate(zip(CHANNELS, labels)):
            ax = axes_grid[row, col]
            valid = actual["t"] <= 3.5 if early else np.ones(len(actual["t"]), dtype=bool)
            if channel == "normal_specific_load_g":
                valid &= actual["derivative_5point_valid_mask"]
            ax.plot(actual["t"][valid], actual[METRIC_SOURCE.get(channel, channel)][valid], color="#111111", lw=1.8, label="JARI")
            last = 3.5 if early else actual["t"][-1]
            mask = pred["t"] <= last
            ax.plot(pred["t"][mask], pred[channel][mask], color="#dc5032", lw=1.3, label="Launch-only prediction")
            if channel == "geometric_incidence_unsigned_deg":
                source_mask = actual["t"] <= last
                ax.plot(actual["t"][source_mask], actual["aoa_source_unsigned_deg"][source_mask], color="#777777", ls=":", lw=.8, label="JARI exported AoA")
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(case.replace("angle_", "")+"\n"+("fit" if case in ds.FIT_CASES else "site check"), fontsize=10)
            ax.set_xlim(0, last)
            if row == 9:
                ax.set_xlabel("Time [s]")
    axes_grid[0, 0].legend(fontsize=8)
    fig.suptitle("JARI-only commercial-game fit: "+("early 0–3.5 s" if early else "original full source supports")+"\nThree fit cases; seven historically seen site checks; no replay input", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=145)
    plt.close(fig)


def calibrate() -> dict[str, Any]:
    before = protected()
    OUT.mkdir(parents=True, exist_ok=True)
    # Only the three declared fit cases are loaded until parameter freezing.
    fit = {case: ds.load_case(case)[1] for case in ds.FIT_CASES}
    creation = {case: ds.creation_input(case) for case in ds.FIT_CASES}
    raw = read_raw(RAW)
    force_parameters, force_diagnostics = fit_force_parameters(fit, raw)
    angle_parameters, angle_diagnostics = fit_angle_parameters(fit, raw)
    base = {"schema": 1, "model_family": "r771_jari_only_effective_single_surface",
        "status": "experimental_jari_only_frozen_for_site_validation", "source": "existing_public_JARI_compiled_c_2.59.0.28",
        "game_replay_observations_used": False, "old_game_coefficients_used": False,
        "raw_parameter_version": "2.59.0.27", "raw_to_site_resource_equivalence_verified": False,
        "forward_duration_s": 16., "integration_substeps": 4,
        "pn_candidate": None, "pid_error_units": None,
        "effective_parameters": {**force_parameters, **angle_parameters, "loft_common_scale": 1.},
        "controller_convention": {"clock_hz": 48, "raw_gains_unchanged": True,
            "form": "parallel PID, I output += ki*e*dt, clipped to raw I_limit; D=kd*(e-e_prev)/dt; output=clip(age_gain*IAS_gain*(P+I+D),-1,1)",
            "initial_I_and_previous_error": [0., 0.], "inactive_updates": "zero command; zero initial memory retained before age gain activates",
            "feedback": "previous completed plant acceleration including gravity, projected on current body up/right",
            "IAS_gain": "(raw baseIndSpeed / (speed_mps*3.6*sqrt(ISA_density/1.225)))^2",
            "gain_and_unit_semantics": "assumed mapping, selected on three site fit cases only", "high_angle_mode": False,
            "integral_transport": False, "request_slew": False},
        "loft_field_function": {"loftElevation": "24: bias aims nose 24 degrees above current world LOS elevation",
            "loftTargetElevation": "-3.5: finish bias when world LOS elevation minus flight-path elevation drops below this angle",
            "loftAngleToAccelMult": "17.5: scale nose-angle error in degrees to acceleration in m/s^2, times one common fitted scale",
            "loftTargetOmegaMax": "0.5: eligible only while LOS angular speed in deg/s is at most this value",
            "phase": "initial climb eligible at first active control tick; end once either raw termination threshold fails; no re-arming",
            "semantics_evidence": "game-effect hypotheses using raw fields; internal JARI bias and field units are not exported"},
        "single_equivalent_surface": {"count": 1, "distFromCmToStab_m": .4, "restoring_couple_at_same_station": True},
        "request_cap_interpretation": {"raw_field": "reqAccelMax", "value": 50., "assumed_units": "G",
            "assumed_operation": "Euclidean norm clip after complete PN and loft request",
            "evidence_level": "assumed raw-field mapping; not established by JARI exports",
            "terminal_requested_g": "The site's large terminal requested_g is a pre-native diagnostic and neither confirms nor rejects this assumed cap"},
        "coefficient_identifiability": {"nominal_scope": "Effective fit in JARI exported control and cross-section coefficient coordinates",
            "physical_control_gain_identification_interval": None,
            "reason": "Unknown exported control scale confounds any physical control gain; these coefficients do not identify fin angle or a physical actuator",
            "loft_scale_tested_interval": [.25, 1.5], "loft_scale_classification": "bounded calibration, not an independently identified field conversion",
            "lift_grid_tested_intervals_rad": {"stall": [.30, .80], "rolloff_width": [.10, .90]}},
        "runtime_source_boundary": {"required_files": [str(ds.OUT/"creation_inputs.json"), str(CONFIG), str(RAW)],
            "observation_archive": False, "terminal_state": False, "previous_model_or_forecast": False,
            "duration": "uniform 16 seconds from config; no source terminal time; analysis continues after hypothetical intercept"}}
    candidate_records = []
    candidate_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    pid_trials = 0
    free_selection_runs = 0
    for units in UNITS:
        for pn in PN:
            config = copy.deepcopy(base)
            config["pid_error_units"], config["pn_candidate"] = units, pn
            # Only the loft-common scale is calibrated here. This source-driven
            # diagnostic is separate from every free forward evaluation.
            best_scale = None
            for amp in np.linspace(.25, 1.5, 26):
                config["effective_parameters"]["loft_common_scale"] = float(amp)
                pid0 = source_conditioned_pid(config, raw, creation["angle_p0"], fit["angle_p0"])
                pid_trials += 1
                e = pid0["vertical_control_rmse"]
                if best_scale is None or e < best_scale[0]:
                    best_scale = (e, float(amp))
            config["effective_parameters"]["loft_common_scale"] = best_scale[1]
            controller = {c: source_conditioned_pid(config, raw, creation[c], fit[c]) for c in ds.FIT_CASES}
            pid_trials += 3
            predictions = {}
            metrics = {}
            for case in ds.FIT_CASES:
                predictions[case] = arrays_from_rows(EffectiveSurfaceGame(raw, config, creation[case]).run())
                free_selection_runs += 1
                metrics[case] = measure(fit[case], predictions[case])
            key = pn+"__"+units.replace("/", "_")
            candidate_predictions[key] = predictions
            record = {"candidate_id": key, "config": config, "source_conditioned_controller": controller,
                      "fit_launch_only_predictions": metrics}
            candidate_records.append(record)
            print(json.dumps({"candidate": key, "loft_scale": best_scale[1],
                  "controller_initial_max": [controller[c]["initial_control_max_abs_error"] for c in ds.FIT_CASES],
                  "early_channels": {c: {k: metrics[c]["windows"]["early"][k]["rmse"] for k in CHANNELS} for c in ds.FIT_CASES}}), flush=True)
    # Selection is explicit and channel based, with no combined score. First
    # choose the error units using initial exported control reproduction. Then
    # choose the complete PN candidate by fit heading RMS; speed, AoA and load
    # remain reported even if they disagree with that heading choice.
    unit_error = {u: max(r["source_conditioned_controller"][c]["initial_control_max_abs_error"]
                        for r in candidate_records if r["config"]["pid_error_units"] == u for c in ds.FIT_CASES) for u in UNITS}
    selected_units = min(UNITS, key=unit_error.__getitem__)
    admitted = [r for r in candidate_records if r["config"]["pid_error_units"] == selected_units]
    heading = {r["candidate_id"]: max(r["fit_launch_only_predictions"][c]["windows"]["early"]["body_yaw_deg"]["rmse"] for c in ("angle_p30", "angle_p60")) for r in admitted}
    selected = min(admitted, key=lambda r: heading[r["candidate_id"]])
    config = selected["config"]
    config["selection"] = {"PID_units": {"rule": "minimum worst initial exported control component error on fit cases, shared across PN candidates", "errors": unit_error},
        "PN": {"rule": "minimum worst early yaw RMSE across 30/60 fit cases; other channels retained separately", "yaw_rmse_by_candidate": heading},
        "validation_used_for_selection": False, "total_free_coefficients": 11,
        "fit_cases": list(ds.FIT_CASES), "site_validation_cases": list(ds.VALIDATION_CASES), "site_validation_is_blind": False}
    write_json(CONFIG, config)  # Freeze before loading any validation observations.
    frozen_before = CONFIG.read_bytes()
    sources = dict(fit)
    forecasts = {}
    final_metrics = {}
    for case in ds.FIT_CASES+ds.VALIDATION_CASES:
        # This call reopens only the three runtime files, with no fit arrays.
        forecasts[case] = arrays_from_rows(forward_from_files(ds.OUT/"creation_inputs.json", CONFIG, RAW, case))
        if case not in sources:
            sources[case] = ds.load_case(case)[1]
        final_metrics[case] = measure(sources[case], forecasts[case])
    known = {}
    for case in ds.FIT_CASES:
        known[case] = known_control_angle_response(raw, config, creation[case], fit[case])
    known_metrics = {}
    for case, response in known.items():
        a = fit[case]
        valid = a["t"] <= response["t"][-1]
        known_metrics[case] = {"source_velocity_and_control_forcing": True, "free_prediction": False,
                              "pitch_rmse_deg": rmse(interpolate(response["body_pitch_deg"], response["t"], a["t"][valid])-a["body_pitch_deg"][valid]),
                              "yaw_rmse_deg": rmse(interpolate(response["body_yaw_deg"], response["t"], a["t"][valid])-a["body_yaw_deg"][valid])}
    np.savez_compressed(OUT/"free_predictions.npz", **{case+"__"+key: value for case, a in forecasts.items() for key, value in a.items()})
    np.savez_compressed(OUT/"known_control_angle_response.npz", **{case+"__"+key: value for case, a in known.items() for key, value in a.items()})
    plot_all(sources, forecasts, OUT/"early_overlay.png", True)
    plot_all(sources, forecasts, OUT/"full_overlay.png", False)
    failed = []
    for case, m in final_metrics.items():
        for window in ("early", "full"):
            for channel in CHANNELS:
                channel_metric = m["windows"][window][channel]
                if channel_metric["rmse"] > channel_metric["predeclared_rmse_tolerance"]:
                    failed.append({"case": case, "window": window, "channel": channel, "rmse": channel_metric["rmse"], "tolerance": channel_metric["predeclared_rmse_tolerance"]})
    # One focused verification pass: fixed inputs/clock/support, no protected
    # changes, frozen config unchanged, and exact creation-only deterministic run.
    after = protected()
    if before != after:
        changes = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
        raise AssertionError(f"Protected file size/mtime changed: {changes[:5]}")
    if CONFIG.read_bytes() != frozen_before:
        raise AssertionError("Frozen parameters changed during site validation")
    for case, forecast in forecasts.items():
        if final_metrics[case]["uncovered_source_rows"]:
            raise AssertionError(f"Uniform runtime did not cover full original source support: {case}")
        if any(not np.isfinite(v).all() for v in forecast.values() if v.dtype.kind in "fiu"):
            raise AssertionError(f"Nonfinite output: {case}")
        if abs(np.linalg.norm(forecast["quaternion_xyzw"], axis=1)-1.).max() > 1e-12:
            raise AssertionError(f"Quaternion normalization failed: {case}")
        if np.abs(forecast["controls_raw"]).max() > 1.:
            raise AssertionError(f"Ordinary controller saturation failed: {case}")
        if np.abs(forecast["pid_integral_output"]).max() > raw["guidance"]["guidanceAutopilot"]["accelControlIntgLim"]:
            raise AssertionError(f"Ordinary controller integral limit failed: {case}")
    again = arrays_from_rows(forward_from_files(ds.OUT/"creation_inputs.json", CONFIG, RAW, "angle_p0"))
    deterministic = all(np.array_equal(again[k], forecasts["angle_p0"][k]) for k in again)
    if not deterministic:
        raise AssertionError("Creation-only repeat differs")
    p, m = forecasts["angle_p30"], forecasts["angle_m30"]
    mirror_errors = {"speed_mps": float(np.max(abs(p["speed_mps"]-m["speed_mps"]))),
        "body_pitch_deg": float(np.max(abs(p["body_pitch_deg"]-m["body_pitch_deg"]))),
        "body_yaw_deg": float(np.max(abs(p["body_yaw_deg"]+m["body_yaw_deg"])))}
    result = {"status": "accepted_jari_only_site_fit" if not failed else "unaccepted_jari_only_candidate",
        "config_path": str(CONFIG), "selected_candidate": selected["candidate_id"],
        "force_identification": force_diagnostics, "angle_identification": angle_diagnostics,
        "selection": config["selection"], "candidates": candidate_records,
        "final_launch_only_metrics": final_metrics, "known_control_angle_response": known_metrics,
        "acceptance": {"per_channel_tolerances_preregistered": FIT_TOLERANCES, "failed_conditions": failed, "overall_score": None,
                       "status": "accepted" if not failed else "not accepted", "game_model_identified": False, "JARI_internals_recovered": False},
        "execution_counts": {"complete_formula_unit_candidates": 6, "force_linear_solves": force_diagnostics["linear_solves"],
            "lift_grid_evaluations": force_diagnostics["lift_parameter_grid_evaluations"], "angle_linear_solves": angle_diagnostics["linear_solves"],
            "source_conditioned_controller_trials": pid_trials, "candidate_selection_free_runs": free_selection_runs,
            "frozen_final_free_runs": 10, "known_control_angle_runs": 3, "deterministic_verification_free_runs": 1,
            "optimizer_runs": 0, "new_network_or_site_runs": 0, "game_replay_test_runs": 0},
        "input_audit": {"fit_observations": str(ds.OUT/"observations.npz"), "fit_source_cases": list(ds.FIT_CASES),
            "validation_observations_loaded_after_freezing": True, "runtime_inputs": config["runtime_source_boundary"],
            "old_game_coefficients_or_old_forecasts_loaded": False, "replay_observations_loaded": False,
            "source_conditioned_fit_is_not_free_prediction": True, "old_production_defaults_changed": False},
        "verification": {"protected_size_mtime_unchanged": True, "protected_file_count": len(before),
            "frozen_config_unchanged_after_validation": True, "all_original_source_rows_covered": True,
            "no_time_warp_or_extrapolation": True, "all_outputs_finite": True, "unit_quaternions": True,
            "ordinary_PID_output_and_integral_bounded": True, "creation_only_repeat_identical": deterministic,
            "mirror_plus_minus_30_max_abs": mirror_errors},
        "limitations": ["Raw .27 parameters and public .28 outputs have not been proved resource-identical in this branch",
            "Raw controls have unknown physical units and unobserved intermediate hold phase; effective response gains do not identify fin angle",
            "Displayed lift and finite-cadence kinematics leave force/angle residuals; simple effective laws do not recover site internals",
            "Loft thresholds and acceleration mapping are hypotheses from original fields, with one calibrated common scale",
            "Selected loft scale is at the upper tested bound 1.5; it is not an interior identified optimum",
            "The 50 G request norm clip is an assumed reqAccelMax mapping, not established by JARI's pre-native requested_g diagnostic",
            "Creation row t=0 displays zero thrust before first update; runtime applies raw power from its first integration tick",
            "Seven site checks were seen historically; this split is fixed now but is not blind validation",
            "No game measurements were fitted, compared or tested"]}
    write_json(OUT/"results.json", result)
    print(json.dumps({"status": result["status"], "selected": result["selected_candidate"], "parameters": config["effective_parameters"],
                      "failed_conditions": len(failed), "counts": result["execution_counts"], "known_input": known_metrics}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-case", choices=ds.FIT_CASES+ds.VALIDATION_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.predict_case:
        if args.output is None:
            parser.error("--predict-case requires --output")
        rows = forward_from_files(ds.OUT/"creation_inputs.json", CONFIG, RAW, args.predict_case)
        np.savez_compressed(args.output, **arrays_from_rows(rows))
    else:
        calibrate()


if __name__ == "__main__":
    main()
