#!/usr/bin/env python3
"""Correct two fixed interfaces of the opt-in JARI commercial-game branch.

Only the three declared site fit cases participate in selection. Plant and
ordinary PID coefficients stay fixed. Export-forced diagnostics are separate
from launch-only predictions and cannot enter the runtime model.
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
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from aim120_model.effective_surface_core import EffectiveSurfaceGame, axes, cross, norm, read_raw, sub, add, scale, dot, unit
import fit_r771_jari_only as base
import prepare_r771_jari_only as ds

OUT = ROOT/"outputs/jari_only/correction"
CONFIG = ROOT/"research/legacy/config/r771_jari_only_corrected.json"
BASE_CONFIG = ROOT/"research/legacy/config/r771_jari_only.json"
CHANNELS = base.CHANNELS
PEAK_WINDOWS = {"first": [.30, .75], "second": [.75, 1.15]}


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(ds.jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def protected() -> dict[str, tuple[int, int]]:
    owned = {ROOT/"src/aim120_model/effective_surface_core.py", CONFIG}
    folders = ("src", "config", "configs", "schemas", "docs", "data", "outputs/jari_offaxis",
               "outputs/jari_only/dataset", "outputs/jari_only/calibration")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in folders for p in (ROOT/folder).rglob("*")
            if p.is_file() and p not in owned and "__pycache__" not in p.parts and ".git" not in p.parts}


def peaks(a: dict[str, np.ndarray], p: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for window, (lo, hi) in PEAK_WINDOWS.items():
        result[window] = {}
        for channel in ("body_pitch_deg", "geometric_incidence_unsigned_deg", "normal_specific_load_g"):
            mask = (a["t"] >= lo) & (a["t"] <= hi)
            if channel == "normal_specific_load_g":
                mask &= a["derivative_5point_valid_mask"]
            times = a["t"][mask]
            truth = a[base.METRIC_SOURCE.get(channel, channel)][mask]
            pred = base.interpolate(p[channel], p["t"], times)
            i, j = int(np.argmax(pred)), int(np.argmax(truth))
            # The maximum can be a window endpoint; that is explicitly marked.
            result[window][channel] = {"source": float(truth[j]), "prediction": float(pred[i]),
                "source_time_s": float(times[j]), "prediction_time_s": float(times[i]),
                "amplitude_error": float(pred[i]-truth[j]), "time_error_s": float(times[i]-times[j]),
                "prediction_peak_is_window_boundary": i in (0, len(times)-1)}
    return result


def state_at(a: dict[str, np.ndarray], t: float) -> tuple[float, ...]:
    q = base.interpolate(a["quaternion_xyzw_normalized"], a["t"], t)
    q /= np.linalg.norm(q)
    return tuple(np.concatenate((base.interpolate(a["position_xyz_m"], a["t"], t),
        base.interpolate(a["velocity_xyz_mps"], a["t"], t), q,
        -base.interpolate(a["omega_export_raw"], a["t"], t))))


def gate_values(model: EffectiveSurfaceGame, t: float, state: tuple[float, ...]) -> dict[str, float]:
    relative = sub(add(model.target0, scale(model.target_velocity, t)), state[:3])
    rv = sub(model.target_velocity, state[3:6])
    los_elevation = math.degrees(math.atan2(relative[1], math.hypot(relative[0], relative[2])))
    vel = state[3:6]
    flight = math.degrees(math.atan2(vel[1], math.hypot(vel[0], vel[2])))
    return {"los_omega_deg_s": math.degrees(norm(cross(relative, rv))/norm(relative)**2),
            "absolute_los_elevation_deg": los_elevation,
            "los_minus_flight_elevation_deg": los_elevation-flight}


def source_controller(config: dict[str, Any], raw: dict[str, Any], creation: dict[str, Any],
                      a: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Offline source-state PID; future interpolation is declared and isolated."""
    model = EffectiveSurfaceGame(raw, config, creation)
    rows, events, previous_phase = [], [], "uninitialized"
    count = math.floor(min(3.5, a["t"][-1])/model.dt)
    for tick in range(count+1):
        t = tick*model.dt
        state = state_at(a, t)
        if tick:
            old_v = base.interpolate(a["velocity_xyz_mps"], a["t"], t-model.dt)
            model.previous_acceleration = tuple((np.asarray(state[3:6])-old_v)/model.dt)
        control = model.update_control(t, state)
        gates = gate_values(model, t, state)
        if model.loft_phase != previous_phase:
            events.append({"control_tick_time_s": t, "phase": model.loft_phase, **gates})
        previous_phase = model.loft_phase
        rows.append({"t": t+model.dt, "controls_raw": control, "request_xyz_mps2": model.last_request,
                     "pid_error": model.last_error, "pid_integral_output": tuple(model.integral),
                     "loft_phase": model.loft_phase, **gates})
    p = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    mask = (a["t"] >= p["t"][0]) & (a["t"] <= p["t"][-1]) & (a["t"] <= 3.5)
    delta = base.interpolate(p["controls_raw"], p["t"], a["t"][mask])-a["controls_export_raw"][mask]
    active = a["t"][mask] >= .30
    result = {"horizontal_control_rmse": base.rmse(delta[:, 0]),
              "vertical_control_rmse": base.rmse(delta[:, 1]),
              "active_horizontal_control_rmse": base.rmse(delta[active, 0]),
              "active_vertical_control_rmse": base.rmse(delta[active, 1]),
              "loft_gate_events": events, "source_forcing": True, "free_prediction": False,
              "feedback": "backward velocity interval including gravity; intermediate site tick is not exported",
              "control_phase": "control computed at t is assigned to t+dt, matching completed held-tick export hypothesis"}
    return result, p


def metrics(a: dict[str, np.ndarray], p: dict[str, np.ndarray]) -> dict[str, Any]:
    m = base.measure(a, p)
    m["two_peak_windows"] = peaks(a, p)
    return m


def launch(config: dict[str, Any], raw: dict[str, Any], creation: dict[str, Any]) -> dict[str, np.ndarray]:
    return base.arrays_from_rows(EffectiveSurfaceGame(raw, config, creation).run())


def no_material_regression(old: dict[str, Any], new: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    violations = []
    for case in ds.FIT_CASES:
        for window in ("early", "full"):
            for channel in CHANNELS:
                before = old[case]["windows"][window][channel]["rmse"]
                after = new[case]["windows"][window][channel]["rmse"]
                tolerance = base.FIT_TOLERANCES[window][channel]
                bound = max(1.15*before, tolerance)
                if after > bound+1e-10:
                    violations.append({"case": case, "window": window, "channel": channel,
                                       "before": before, "after": after, "maximum_admitted": bound})
    return not violations, violations


def plot(sources: dict[str, dict[str, np.ndarray]], forecasts: dict[str, dict[str, np.ndarray]],
         baselines: dict[str, dict[str, np.ndarray]], early: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ds.FIT_CASES+ds.VALIDATION_CASES
    labels = ("Speed [m/s]", "Pitch [deg]", "Yaw [deg]", "Incidence [deg]", "Normal specific load [G]")
    fig, grid = plt.subplots(10, 5, figsize=(19, 25), squeeze=False)
    for row, case in enumerate(order):
        actual = sources[case]
        last = 3.5 if early else actual["t"][-1]
        for col, (channel, label) in enumerate(zip(CHANNELS, labels)):
            ax = grid[row, col]
            valid = actual["t"] <= last
            if channel == "normal_specific_load_g":
                valid &= actual["derivative_5point_valid_mask"]
            ax.plot(actual["t"][valid], actual[base.METRIC_SOURCE.get(channel, channel)][valid],
                    color="#111111", lw=1.8, label="JARI")
            for p, color, name in ((baselines[case], "#9298a0", "Previous JARI-only baseline"),
                                   (forecasts[case], "#d64b2c", "Corrected launch-only prediction")):
                mask = p["t"] <= last
                ax.plot(p["t"][mask], p[channel][mask], color=color, lw=1.15, label=name)
            ax.set_xlim(0, last)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(case.replace("angle_", "")+"\n"+("fit" if case in ds.FIT_CASES else "site check"), fontsize=10)
            if row == 9:
                ax.set_xlabel("Time [s]")
    grid[0, 0].legend(fontsize=7)
    fig.suptitle("JARI-only correction: "+("0–3.5 s" if early else "original full source support")+
                 "\nFixed plant and ordinary PID; three fit cases and seven historically seen site checks", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT/("early_overlay.png" if early else "full_overlay.png"), dpi=145)
    plt.close(fig)


def closing_candidate(cfg: dict[str, Any], chosen: str, raw: dict[str, Any],
                      creation: dict[str, Any], fit: dict[str, dict[str, np.ndarray]],
                      candidates: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    candidate = copy.deepcopy(cfg)
    candidate["time_to_hit_convention"] = "linear_closing"
    residuals = {c: source_controller(candidate, raw, creation[c], fit[c])[0] for c in ds.FIT_CASES}
    previous = candidates[chosen]
    source_admitted = all(residuals[c]["horizontal_control_rmse"] <
                          previous["source_conditioned_controller"][c]["horizontal_control_rmse"]
                          for c in ds.FIT_CASES[1:])
    if not source_admitted:
        return cfg, chosen, {"source_conditioned_controller": residuals,
            "source_admitted": False, "admitted": False, "free_runs": 0}
    pp = {c: launch(candidate, raw, creation[c]) for c in ds.FIT_CASES}
    mm = {c: metrics(fit[c], pp[c]) for c in ds.FIT_CASES}
    admitted, violations = no_material_regression(previous["fit_metrics"], mm)
    yaw_improves = all(mm[c]["windows"]["early"]["body_yaw_deg"]["rmse"] <
                       previous["fit_metrics"][c]["windows"]["early"]["body_yaw_deg"]["rmse"]
                       for c in ds.FIT_CASES[1:])
    record = {"source_conditioned_controller": residuals, "fit_metrics": mm,
              "source_admitted": True, "admitted": admitted and yaw_improves,
              "material_regressions": violations, "both_offaxis_yaw_improved": yaw_improves,
              "free_runs": 3, "time_to_hit_convention": "linear_closing",
              "nonclosing_definition": "tgo=infinity, hit gain=raw last table ordinate"}
    if admitted and yaw_improves:
        return candidate, "absolute_los_linear_closing", record
    return cfg, chosen, record


def correct(*, closing_followup: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected()
    original = json.loads(BASE_CONFIG.read_text())
    raw = read_raw(base.RAW)
    creation = json.loads((ds.OUT/"creation_inputs.json").read_text())["cases"]
    fit = {case: ds.load_case(case)[1] for case in ds.FIT_CASES}
    preregistration = {
        "fit": list(ds.FIT_CASES), "validation": list(ds.VALIDATION_CASES),
        "validation_used_for_selection": False, "validation_is_blind": False,
        "fixed": "all ten plant coefficients; complete PN; gains, 48 Hz, zero PID memory, saturation, gravity feedback phase",
        "structures": ["relative_flight_path/body baseline", "absolute_los/body",
                       "absolute_los/velocity_normal_projected_body"],
        "source_only_PID_conventions": ["scheduled_terms", "conditional_integration"],
        "source_only_PID_admission": "before free runs, neither off-axis horizontal control residual may worsen; no combination of conventions",
        "loft_scale": {"coarse": [.75, .875, 1., 1.125, 1.25, 1.375, 1.5],
                       "refinement": [1.20, 1.225, 1.25, 1.275, 1.30],
                       "rule": "minimum 0 degree early pitch RMSE; speed/incidence/load and first/second peaks reported separately",
                       "scale_is_not_an_internal_unit_identification": True},
        "frame_admission": "all fit early/full channels <= max(1.15*body candidate RMSE, existing channel tolerance); both off-axis early yaw RMSE must improve",
        "second_peak_condition": "0 degree pitch peak in .75–1.15 s must be interior, within .125 s and 4 degrees of source peak",
        "no_overall_score": True}
    if closing_followup:
        cached = json.loads((OUT/"results.json").read_text())
        candidates = cached["candidates"]
        scale_trials = cached["loft_scale_trials"]
        pid_convention_trials = cached["source_only_PID_convention_trials"]
        selected_scale = cached["selected_loft_scale"]
        cfg = json.loads(CONFIG.read_text())
        chosen = cached["selected_candidate"]
        preregistration = cached["preregistration"]
    else:
        candidates, candidate_predictions = {}, {}
        for name, gate in (("baseline", "relative_flight_path"), ("absolute_los_fixed_scale", "absolute_los")):
            cfg = copy.deepcopy(original)
            if name != "baseline":
                cfg["loft_gate_convention"] = gate
            pred = {c: launch(cfg, raw, creation[c]) for c in ds.FIT_CASES}
            mm = {c: metrics(fit[c], pred[c]) for c in ds.FIT_CASES}
            pid = {c: source_controller(cfg, raw, creation[c], fit[c])[0] for c in ds.FIT_CASES}
            candidates[name] = {"fit_metrics": mm, "source_conditioned_controller": pid,
                                "loft_gate_convention": gate, "pid_error_frame": "body", "loft_common_scale": 1.5}
            candidate_predictions[name] = pred
        cfg = copy.deepcopy(original)
        cfg["loft_gate_convention"] = "absolute_los"
        scale_trials = []
        scale_forecasts = {}
        scales = preregistration["loft_scale"]["coarse"]+preregistration["loft_scale"]["refinement"]
        for loft_scale in dict.fromkeys(scales):
            cfg["effective_parameters"]["loft_common_scale"] = loft_scale
            p = launch(cfg, raw, creation["angle_p0"])
            mm = metrics(fit["angle_p0"], p)
            scale_trials.append({"scale": loft_scale, "fit_case": "angle_p0", "metrics": mm,
                "source_conditioned_controller": source_controller(cfg, raw, creation["angle_p0"], fit["angle_p0"])[0]})
            scale_forecasts[loft_scale] = p
        selected = min(scale_trials, key=lambda r: r["metrics"]["windows"]["early"]["body_pitch_deg"]["rmse"])
        selected_scale = selected["scale"]
        cfg["effective_parameters"]["loft_common_scale"] = selected_scale
        body_predictions = {"angle_p0": scale_forecasts[selected_scale],
                            **{c: candidate_predictions["absolute_los_fixed_scale"][c] for c in ds.FIT_CASES if c != "angle_p0"}}
        body_metrics = {c: metrics(fit[c], body_predictions[c]) for c in ds.FIT_CASES}
        candidates["absolute_los_body_selected_scale"] = {"fit_metrics": body_metrics,
            "source_conditioned_controller": {c: source_controller(cfg, raw, creation[c], fit[c])[0] for c in ds.FIT_CASES},
            "loft_common_scale": selected_scale, "pid_error_frame": "body", "loft_gate_convention": "absolute_los"}
        projected = copy.deepcopy(cfg)
        projected["pid_error_frame"] = "velocity_normal_projected_body"
        pp = {c: launch(projected, raw, creation[c]) for c in ds.FIT_CASES}
        pm = {c: metrics(fit[c], pp[c]) for c in ds.FIT_CASES}
        admitted, regressions = no_material_regression(body_metrics, pm)
        yaw_improves = all(pm[c]["windows"]["early"]["body_yaw_deg"]["rmse"] <
                           body_metrics[c]["windows"]["early"]["body_yaw_deg"]["rmse"] for c in ds.FIT_CASES[1:])
        candidates["absolute_los_projected_error"] = {"fit_metrics": pm,
            "source_conditioned_controller": {c: source_controller(projected, raw, creation[c], fit[c])[0] for c in ds.FIT_CASES},
            "loft_common_scale": selected_scale, "pid_error_frame": "velocity_normal_projected_body",
            "loft_gate_convention": "absolute_los", "admitted": admitted and yaw_improves,
            "material_regressions": regressions, "both_offaxis_yaw_improved": yaw_improves}
        pid_convention_trials = {}
        for convention in preregistration["source_only_PID_conventions"]:
            diagnostic_config = copy.deepcopy(cfg)
            diagnostic_config["pid_gain_convention"] = convention
            residuals = {c: source_controller(diagnostic_config, raw, creation[c], fit[c])[0]
                         for c in ds.FIT_CASES}
            baseline_residuals = candidates["absolute_los_body_selected_scale"]["source_conditioned_controller"]
            source_admitted = all(residuals[c]["horizontal_control_rmse"] <= baseline_residuals[c]["horizontal_control_rmse"]
                                  for c in ds.FIT_CASES[1:])
            pid_convention_trials[convention] = {"source_conditioned_controller": residuals,
                                               "source_admitted": source_admitted, "free_runs": 0}
            if source_admitted:
                raise AssertionError("A previously rejected PID convention changed its source diagnostic; review before new free runs")
        if admitted and yaw_improves:
            cfg, chosen = projected, "absolute_los_projected_error"
        else:
            chosen = "absolute_los_body_selected_scale"
    preregistration["time_to_hit_candidate"] = {
        "formula": "R/positive closing speed; infinite when closing<=0, using raw final gain",
        "admission": "source horizontal RMSE improves at 30 and 60 before three free runs; no material channel regressions and both early yaw curves improve",
        "JARI_internal_mapping_identified": False}
    cfg, chosen, closing_trial = closing_candidate(cfg, chosen, raw, creation, fit, candidates)
    candidates["absolute_los_linear_closing"] = closing_trial
    cfg["status"] = "experimental_jari_only_corrected_frozen_for_site_validation"
    cfg["controller_convention"]["feedback"] = (
        "previous completed world acceleration including gravity; fixed velocity-normal projection then current body up/right"
        if cfg.get("pid_error_frame", "body") != "body" else original["controller_convention"]["feedback"])
    cfg["loft_field_function"]["loftTargetElevation"] = "-3.5: world LOS elevation lower threshold; used together with the LOS angular rate gate"
    cfg["loft_field_function"]["phase"] = "climb eligible at first active tick; finish once world LOS elevation or LOS angular rate gate fails; no re-arming"
    cfg["coefficient_identifiability"]["loft_scale_tested_interval"] = [.75, 1.5]
    cfg["coefficient_identifiability"]["loft_scale_classification"] = "bounded JARI-only calibration in corrected world-LOS gate; selected by 0 degree early pitch curve, internal conversion remains unidentified"
    cfg["selection"]["correction"] = {"selected_candidate": chosen, "loft_scale": selected_scale,
        "rule": preregistration["loft_scale"]["rule"], "fixed_plant_coefficients": True,
        "ordinary_PID_unchanged": True, "validation_used_for_selection": False}
    cfg["runtime_source_boundary"]["required_files"][1] = str(CONFIG)
    write(CONFIG, cfg)
    # Only now open the seven validation observations and freeze all choices.
    sources = {**fit, **{c: ds.load_case(c)[1] for c in ds.VALIDATION_CASES}}
    forecasts = {c: launch(cfg, raw, creation[c]) for c in ds.FIT_CASES+ds.VALIDATION_CASES}
    z = np.load(base.OUT/"free_predictions.npz", allow_pickle=False)
    baselines = {c: {key.split("__", 1)[1]: z[key] for key in z.files if key.startswith(c+"__")}
                 for c in ds.FIT_CASES+ds.VALIDATION_CASES}
    final_metrics = {c: metrics(sources[c], forecasts[c]) for c in forecasts}
    baseline_metrics = {c: metrics(sources[c], baselines[c]) for c in forecasts}
    np.savez_compressed(OUT/"free_predictions.npz", **{c+"__"+k: v for c, p in forecasts.items() for k, v in p.items()})
    pid_results, pid_curves = {}, {}
    for name, config in (("baseline", original), ("corrected", cfg)):
        pid_results[name], pid_curves[name] = {}, {}
        for case in ds.FIT_CASES:
            pid_results[name][case], pid_curves[name][case] = source_controller(config, raw, creation[case], fit[case])
    np.savez_compressed(OUT/"source_conditioned_pid.npz", **{name+"__"+c+"__"+k: v
        for name, cases in pid_curves.items() for c, p in cases.items() for k, v in p.items()})
    plot(sources, forecasts, baselines, True)
    plot(sources, forecasts, baselines, False)
    failed = []
    for case, mm in final_metrics.items():
        for window in ("early", "full"):
            for channel in CHANNELS:
                m = mm["windows"][window][channel]
                if m["rmse"] > m["predeclared_rmse_tolerance"]:
                    failed.append({"case": case, "window": window, "channel": channel,
                                   "rmse": m["rmse"], "tolerance": m["predeclared_rmse_tolerance"]})
    # One final focused verification pass. Do not fit or inspect new candidates.
    compatibility = {}
    for case in ds.FIT_CASES:
        rerun = launch(original, raw, creation[case])
        keys = [k for k in rerun if np.issubdtype(rerun[k].dtype, np.number)]
        compatibility[case] = all(np.array_equal(rerun[k], baselines[case][k]) for k in keys)
        if not compatibility[case]:
            raise AssertionError(f"Baseline compatibility changed: {case}")
    plant_keys = [k for k in original["effective_parameters"] if k != "loft_common_scale"]
    fixed_plant = all(cfg["effective_parameters"][k] == original["effective_parameters"][k] for k in plant_keys)
    if not fixed_plant:
        raise AssertionError("Plant changed during interface correction")
    max_norm_error, min_dt = 0., float("inf")
    for case, p in forecasts.items():
        if final_metrics[case]["uncovered_source_rows"]:
            raise AssertionError(f"Original source support uncovered: {case}")
        if np.abs(p["controls_raw"]).max() > 1. or np.abs(p["pid_integral_output"]).max() > 1.:
            raise AssertionError(f"PID saturation or integral limit failed: {case}")
        if np.any(p["pid_integral_output"][0] != 0.):
            raise AssertionError("PID initialization changed")
        for key, values in p.items():
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise AssertionError(f"Nonfinite forecast: {case}/{key}")
        max_norm_error = max(max_norm_error, float(np.max(np.abs(np.linalg.norm(p["quaternion_xyzw"], axis=1)-1.))))
        min_dt = min(min_dt, float(np.diff(p["t"]).min()))
    after = protected()
    if before != after:
        raise AssertionError("Protected files changed")
    second = final_metrics["angle_p0"]["two_peak_windows"]["second"]["body_pitch_deg"]
    second_ok = (not second["prediction_peak_is_window_boundary"] and abs(second["amplitude_error"]) <= 4.
                 and abs(second["time_error_s"]) <= .125)
    result = {"status": "accepted_jari_only_correction" if not failed and second_ok else "partial_jari_only_correction",
        "config_path": str(CONFIG), "selected_candidate": chosen, "selected_loft_scale": selected_scale,
        "preregistration": preregistration, "candidates": candidates, "loft_scale_trials": scale_trials,
        "source_only_PID_convention_trials": pid_convention_trials,
        "baseline_metrics": baseline_metrics, "final_launch_only_metrics": final_metrics,
        "source_conditioned_controller": pid_results,
        "acceptance": {"status": "accepted" if not failed and second_ok else "not accepted",
            "failed_conditions": failed, "second_peak_condition_passed": second_ok, "overall_score": None,
            "game_model_identified": False, "JARI_internals_recovered": False},
        "verification": {"baseline_fit_cases_bitwise_compatible": compatibility,
            "all_ten_plant_coefficients_unchanged": fixed_plant, "protected_file_count": len(before),
            "protected_files_unchanged": before == after, "source_rows_covered": sum(len(a["t"]) for a in sources.values()),
            "free_prediction_count": len(forecasts), "quaternion_norm_max_abs_error": max_norm_error,
            "control_interval_s": min_dt, "PID_gains_raw_unchanged": True, "PID_memory_transport_or_modes": False,
            "runtime_reads_observation_archive": False, "game_replay_observations_used": False},
        "execution_counts": {"structure_fit_free_runs": 12, "single_case_scale_free_runs": len(scale_trials),
            "frozen_final_free_runs": 10, "compatibility_free_runs": 3,
            "exploratory_pre_script_fit_free_runs": 13,
            "cumulative_frozen_final_free_runs": 20 if closing_followup else 10,
            "cumulative_compatibility_free_runs": 6 if closing_followup else 3,
            "actual_cumulative_fit_case_free_runs_including_exploration_and_verification":
                13+12+len(scale_trials)+(6 if closing_followup else 3)+(6 if closing_followup else 3),
            "new_source_requests": 0},
        "limitations": ["World LOS gate is a tested field-mapping hypothesis, not recovered JARI internal code",
            "Exported controls and intermediate hold phase remain unidentified; offline source-forced PID is not a free prediction",
            "Plant coefficients are deliberately fixed; remaining speed/attitude/load mismatch is reported separately",
            "Seven site checks were historically seen and are not blind; no game replay test performed",
            "Analysis continues uniformly to 16 s and compares original source support without timewarp or extrapolation"]}
    write(OUT/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--closing-followup", action="store_true", help="Reuse the completed loft fit and evaluate only the additional closing-time candidate")
    args = parser.parse_args()
    result = correct(closing_followup=args.closing_followup)
    print(json.dumps({"status": result["status"], "selected_candidate": result["selected_candidate"],
        "loft_scale": result["selected_loft_scale"], "failed_conditions": len(result["acceptance"]["failed_conditions"]),
        "verification": result["verification"], "early_channels": {c: {k: m["windows"]["early"][k]["rmse"]
        for k in CHANNELS} for c, m in result["final_launch_only_metrics"].items()}}, indent=2))
