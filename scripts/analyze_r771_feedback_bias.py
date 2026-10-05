#!/usr/bin/env python3
"""Bounded cached-feedback/finite-force diagnostics for a War Thunder game unit.

No runtime candidate or source-feedback controller is implemented. Independent
controller histories below are conditional replays on the same reference state,
not free trajectories and not identified native-game rules.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_pn_pid_decoupling as dec
from aim120_model.control import scheduled_pid_values
from aim120_model.r771_game_control import R771GameControl
from aim120_model.units import G0_MPS2

OUT = ROOT / "outputs/jari_offaxis/feedback_bias"
INTERFACES = ROOT / "outputs/jari_offaxis/feedback_interfaces"
CASES = ("jari_259_0", "jari_259_30", "jari_259_60")
AXES = ("pitch", "yaw")
METHODS = ("plant", "source3", "source5")


def snapshot():
    folders = ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for folder in folders for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts and OUT not in p.parents and p != Path(__file__).resolve()}


def interval_weights(t, lo, hi, allowed):
    """Original 48Hz zero-order-held interval, clipped to declared endpoints."""
    return np.maximum(0., np.minimum(t + dec.DT, hi) - np.maximum(t, lo)) * allowed


def intervals(t, mask, lo, hi):
    result = []; start = None; end = None
    for ts, flag in zip(t, mask):
        left = max(float(ts), lo); right = min(float(ts) + dec.DT, hi)
        if flag and right > left + 1e-12:
            if start is None:
                start = left
            end = right
        elif start is not None:
            result.append([start, end]); start = None
    if start is not None:
        result.append([start, end])
    return result


def sign_summary(t, error, plant_error, other_source_error, allowed, lo, hi):
    weights = interval_weights(t, lo, hi, allowed)
    opposite = error * plant_error < -1e-14
    strong = opposite & (error * other_source_error > 1e-14)
    mask = weights > 1e-12
    return {"error_G": dec.scalar_stats(error, mask),
        "negative_error_duration_s": float(weights[error < -1e-12].sum()),
        "positive_error_duration_s": float(weights[error > 1e-12].sum()),
        "sign_disagreement_with_plant_duration_s": float(weights[opposite].sum()),
        "sign_disagreement_intervals_s": intervals(t, opposite & allowed, lo, hi),
        "both_sources_agree_against_plant_duration_s": float(weights[strong].sum()),
        "both_sources_agree_against_plant_intervals_s": intervals(t, strong & allowed, lo, hi)}


def radial_force_jacobian(local, alpha_s, width):
    """J/(q*S*Ca/m) for F_lift_normal = (q*S*Ca/m) R(s) local.

    Wind basis, pressure, mass and thrust direction are held fixed here. This is
    an incidence-to-lift Jacobian, not the full dynamic input-to-total-force map.
    """
    s = np.linalg.norm(local, axis=1); high = s > alpha_s
    R = np.ones(len(s)); derivative = np.zeros(len(s))
    x = s[high] - alpha_s; den = width**2 + x**2
    R[high] = width**2 / den
    derivative[high] = -2 * width**2 * x / den**2
    outer = local[:, :, None] * local[:, None, :]
    J = R[:, None, None] * np.eye(2) + np.divide(derivative, s, out=np.zeros_like(s), where=s > 0)[:, None, None] * outer
    radial = R + s * derivative
    return s, R, derivative, radial, R.copy(), J


def replay(errors, t, gains, k, nonlinear, cfg, limits, one_step_error=None):
    controller = R771GameControl(); memory = [(0., 0., 0.), (0., 0., 0.)]
    keys = ("P_rad", "I_provisional_rad", "D_rad", "held_rad", "next_integral_rad", "transported_integral_rad",
        "error_increment_rad", "antiwindup_delta_rad", "antiwindup_blocked", "effective_limit_rad", "high_mode")
    a = {key: np.zeros((len(t), 2)) for key in keys}
    a["memory"] = np.zeros((len(t), 2, 3)); a["previous_memory"] = np.zeros((len(t), 2, 3))
    a["one_step_held_rad"] = np.zeros((len(t), 2))
    a["one_step_next_memory"] = np.zeros((len(t), 2, 3))
    for i, ts in enumerate(t):
        a["previous_memory"][i] = memory
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + dec.DT)
        if gains[i] > 1e-12:
            for axis in range(2):
                if one_step_error is not None:
                    trial = copy.deepcopy(controller)
                    a["one_step_held_rad"][i, axis], a["one_step_next_memory"][i, axis] = trial.update_axis(axis, float(ts),
                        float(one_step_error[i, axis]), memory[axis], pid, limits[axis], gains[i], k[i], nonlinear[i])
                term = dec.pid_terms(controller, axis, memory[axis], float(errors[i, axis]), pid, gains[i], k[i], nonlinear[i], limits[axis])
                for key in keys:
                    a[key][i, axis] = term[key]
                actual_u, memory[axis] = controller.update_axis(axis, float(ts), float(errors[i, axis]), memory[axis],
                    pid, limits[axis], gains[i], k[i], nonlinear[i])
                if abs(actual_u - a["held_rad"][i, axis]) > 1e-12 or abs(memory[axis][0] - a["next_integral_rad"][i, axis]) > 1e-12:
                    raise AssertionError("Same-tick PID output/next memory does not close")
        a["memory"][i] = memory
    return a


def diagnose_case(cid, item, interface_cache, pn_cache, coefficients):
    a = {key[len(cid) + 2:]: interface_cache[key] for key in interface_cache.files if key.startswith(cid + "__")}
    cfg = item["config"]; t = a["t"]; gains = a["flight_gain"]; k = a["K"]; limits = a["old_held_limits_rad"]
    request = a["wind_specific__request_G"]
    feedback = {"plant": a["wind_specific__feedback_G"],
        "source3": a["3__source_wind_specific_feedback_G_diagnostic"],
        "source5": a["5__source_wind_specific_feedback_G_diagnostic"]}
    errors = {name: request - feedback[name] for name in METHODS}
    arrays = {"t": t, "request_G": request, "flight_gain": gains, "K": k, "limits_rad": limits,
        "gain_transition_derivative_halo": a["gain_transition_derivative_halo"], "nonlinear_force_active": a["nonlinear_force_active"]}
    histories = {name: replay(errors[name], t, gains, k, a["nonlinear_force_active"], cfg, limits) for name in METHODS}
    checks = {"baseline_held_exact_cache_max_abs_rad": float(np.max(abs(histories["plant"]["held_rad"] - a["wind_specific__own_history_held_rad"]))),
        "baseline_memory_exact_cache_max_abs": float(np.max(abs(histories["plant"]["memory"] - a["wind_specific__own_history_memory"])))}
    for name in METHODS:
        arrays[name + "__feedback_G"] = feedback[name]; arrays[name + "__error_G"] = errors[name]
        for key, value in histories[name].items():
            arrays[name + "__" + key] = value
    # Existing single-step cache used component-interpolated source feedback.
    # Validate against exactly those inputs before separately using the more
    # explicit same-current-basis world-vector projection in the own histories.
    if cid != CASES[0]:
        for width in (3, 5):
            saved_fb = pn_cache[cid + f"__{width}__source_feedback_G"]
            original = replay(errors["plant"], t, gains, k, a["nonlinear_force_active"], cfg, limits, request - saved_fb)
            arrays[f"source{width}__existing_cache_one_step_held_rad"] = pn_cache[cid + f"__cf__source{width}__held_rad"]
            arrays[f"source{width}__existing_cache_feedback_G"] = saved_fb
            checks[f"source{width}_existing_one_step_replay_closure_rad"] = float(np.max(abs(original["one_step_held_rad"] - arrays[f"source{width}__existing_cache_one_step_held_rad"])))
            checks[f"source{width}_existing_one_step_nextI_closure_rad"] = float(np.max(abs(original["one_step_next_memory"] - pn_cache[cid + f"__cf__source{width}__discarded_next_memory"])))
            arrays[f"source{width}__same_basis_minus_component_interpolation_G"] = feedback[f"source{width}"] - saved_fb
    model = dec.audit.fix.make_model(cfg, coefficients); model.initialize(dataclasses.replace(item["initial"]))
    local = []; force_scale = []; current_feedback = []
    for ts in t:
        evaluation = model.evaluate(dec.audit.fix.source_state_at(item["ref"], float(ts)), float(ts), (0., 0.))
        local.append(evaluation.local_incidence); current_feedback.append(evaluation.feedback_g)
        force_scale.append(evaluation.qbar * model.area * model.coefficients.ca / (evaluation.mass * G0_MPS2))
    local = np.asarray(local); force_scale = np.asarray(force_scale)
    checks["current_plant_feedback_exact_cache_max_abs_G"] = float(np.max(abs(np.asarray(current_feedback) - feedback["plant"])))
    s, R, Rp, radial, tangential, J = radial_force_jacobian(local, dec.audit.fix.finite.ALPHA_S, dec.audit.fix.finite.WIDTH)
    eigen = np.linalg.eigvalsh(J); analytic_eigen = np.sort(np.column_stack((radial, tangential)), axis=1)
    checks["analytic_J_eigen_closure_dimensionless"] = float(np.max(abs(eigen - analytic_eigen)))
    arrays.update(local_incidence_rad=local, local_norm_rad=s, lift_rolloff=R, lift_rolloff_derivative_per_rad=Rp,
        lift_J_dimensionless=J, lift_J_G_per_rad=J * force_scale[:, None, None],
        lift_radial_slope_dimensionless=radial, lift_tangential_slope_dimensionless=tangential,
        lift_radial_slope_G_per_rad=radial * force_scale, lift_tangential_slope_G_per_rad=tangential * force_scale)
    active = gains > 1e-12; allowed = active & ~a["gain_transition_derivative_halo"]
    onset = {axis: next((float(t[i]) for i in range(len(t)) if active[i] and errors["plant"][i, j] < -1e-12), None) for j, axis in enumerate(AXES)}
    report = {"first_plant_r_below_f_s": onset, "windows": {}, "force_Jacobian_windows": {}, "checks": checks,
        "raw_PID_entire_dict": cfg["control"]["pid"], "source_interpolation_difference": {}, "sign_events": {}}
    for name in METHODS:
        report["sign_events"][name] = {axis: {field: next((float(t[i]) for i in range(len(t)) if active[i] and histories[name][field][i, j] < -1e-12), None)
            for field in ("P_rad", "I_provisional_rad", "D_rad", "held_rad")} for j, axis in enumerate(AXES)}
    for width in (3, 5):
        arrays[f"{width}__needed_held_rad"] = a[f"{width}__needed_held_rad"]
        arrays[f"{width}__impossible_mask"] = a[f"{width}__impossible_mask"]
        if cid != CASES[0]:
            report["source_interpolation_difference"][str(width)] = {axis: dec.scalar_stats(arrays[f"source{width}__same_basis_minus_component_interpolation_G"][:, j], allowed)
                for j, axis in enumerate(AXES)}
    for window, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
        weights = interval_weights(t, lo, hi, allowed); mask = weights > 1e-12
        comparison_mask = allowed & dec.audit.inverse.window_mask(t, lo, hi)
        report["windows"][window] = {}
        report["force_Jacobian_windows"][window] = {
            "local_norm_range_deg": np.degrees([s[mask].min(), s[mask].max()]).tolist(),
            "radial_negative_duration_s": float(weights[radial < 0].sum()),
            "radial_negative_intervals_s": intervals(t, (radial < 0) & allowed, lo, hi),
            "radial_slope_dimensionless": dec.scalar_stats(radial, mask), "tangential_slope_dimensionless": dec.scalar_stats(tangential, mask),
            "axis_diagonal_lift_J_G_per_rad": {axis: dec.scalar_stats(arrays["lift_J_G_per_rad"][:, j, j], mask) for j, axis in enumerate(AXES)},
            "axis_diagonal_negative_duration_s": {axis: float(weights[J[:, j, j] < 0].sum()) for j, axis in enumerate(AXES)}}
        for j, axis in enumerate(AXES):
            integral_gains = np.asarray([scheduled_pid_values(cfg["control"]["pid"], float(ts) + dec.DT)["i"] for ts in t])
            branch = {"full_window": {}, "from_first_plant_r_below_f": {}, "control_comparison": {}}
            for label, start in (("full_window", lo), ("from_first_plant_r_below_f", max(lo, onset[axis]) if onset[axis] is not None else hi)):
                w = interval_weights(t, start, hi, allowed)
                for name in METHODS:
                    other = errors["source5" if name == "source3" else "source3"][:, j]
                    entry = sign_summary(t, errors[name][:, j], errors["plant"][:, j], other, allowed, start, hi)
                    entry.update(unconstrained_Ki_integral_error_drive_rad=float(np.sum(integral_gains * errors[name][:, j] * w)),
                        Ki_drive_minus_plant_rad=float(np.sum(integral_gains * (errors[name][:, j] - errors["plant"][:, j]) * w)),
                        P_difference_from_plant_rad=dec.scalar_stats(histories[name]["P_rad"][:, j] - histories["plant"]["P_rad"][:, j], w > 1e-12),
                        D_own_history_difference_from_plant_rad=dec.scalar_stats(histories[name]["D_rad"][:, j] - histories["plant"]["D_rad"][:, j], w > 1e-12))
                    branch[label][name] = entry
            for name in METHODS:
                u = histories[name]["held_rad"][:, j]; target_I = k * histories[name]["next_integral_rad"][:, j] / limits[j]
                v = np.flatnonzero(comparison_mask)
                entry = {"target_I_normalized_at_first_and_last_tick": target_I[v[[0, -1]]].tolist(),
                    "own_history_antiwindup_duration_s": float(weights[histories[name]["antiwindup_blocked"][:, j] > 0].sum()),
                    "own_history_negative_u_duration_s": float(weights[u < -1e-12].sum()),
                    "negative_u_with_positive_needed_and_negative_radial_duration_s": {str(width): float(weights[(u < -1e-12) & (a[f"{width}__needed_held_rad"][:, j] > 1e-12) & (radial < 0)].sum()) for width in (3, 5)},
                    "conditional_frozen_plant_needed_comparison": {}}
                for width in (3, 5):
                    valid = comparison_mask & ~a[f"{width}__impossible_mask"][:, j]
                    needed = a[f"{width}__needed_held_rad"][:, j]; delta = k * (u - needed) / limits[j]
                    entry["conditional_frozen_plant_needed_comparison"][str(width)] = {
                        "feasible_ticks": int(valid.sum()), "impossible_ticks": int(np.sum(comparison_mask & a[f"{width}__impossible_mask"][:, j])),
                        "RMS_normalized_target_error": dec.audit.inverse.rms(delta[valid]) if valid.any() else None,
                        "negative_u_when_needed_positive_ticks": int(np.sum(valid & (u < -1e-12) & (needed > 1e-12)))}
                branch["control_comparison"][name] = entry
            report["windows"][window][axis] = branch
    for name in METHODS:
        arrays[name + "__cumulative_unconstrained_Ki_error_drive_rad"] = np.cumsum(histories[name]["error_increment_rad"], axis=0)
        arrays[name + "__target_I_normalized"] = k[:, None] * histories[name]["next_integral_rad"] / limits
    report["force_Jacobian_summary"] = {"peak_local_norm_deg": float(np.degrees(np.hypot(dec.audit.fix.finite.ALPHA_S, dec.audit.fix.finite.WIDTH))),
        "max_local_norm_deg": float(np.degrees(s.max())), "radial_negative_ticks_active_nonhalo": int(np.sum(allowed & (radial < 0))),
        "radial_negative_any": bool(np.any(allowed & (radial < 0))), "pitch_diagonal_negative_any": bool(np.any(allowed & (J[:, 0, 0] < 0))),
        "yaw_diagonal_negative_any": bool(np.any(allowed & (J[:, 1, 1] < 0))),
        "frozen_control_to_angular_acceleration_Ku_positive": model.coefficients.control_gain > 0}
    return arrays, report


def plot_case(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    colors = {"plant": "#78337b", "source3": "#6b82ba", "source5": "#ce772e"}
    for j, axis in enumerate(AXES):
        for name in METHODS:
            axes[0, j].plot(t, a[name + "__error_G"][:, j], color=colors[name], ls="--" if name == "source3" else "-", label=name + " error")
            axes[1, j].plot(t, a[name + "__cumulative_unconstrained_Ki_error_drive_rad"][:, j] * 1000, color=colors[name], ls="--" if name == "source3" else "-", label=name + ": Ki sum e dt")
            axes[2, j].plot(t, a[name + "__held_rad"][:, j] * a["K"] / a["limits_rad"][j], color=colors[name], ls="--" if name == "source3" else "-", label=name + ": independent conditional history")
        valid = (a["flight_gain"] > 0) & ~a["gain_transition_derivative_halo"]
        axes[2, j].plot(t, np.where(valid, a["5__needed_held_rad"][:, j] * a["K"] / a["limits_rad"][j], np.nan), color="#177277", label="Needed by frozen angular plant, 5pt")
        axes[2, j].fill_between(t, 0, 1, where=a["5__impossible_mask"][:, j], transform=axes[2, j].get_xaxis_transform(), color="#e6b5b5", alpha=.18, label="Above unchanged authority")
        axes[3, j].plot(t, a["lift_radial_slope_dimensionless"], color="#444", label="Lift radial slope / linear slope")
        axes[3, j].plot(t, a["lift_tangential_slope_dimensionless"], color="#6b82ba", label="Lift tangential slope / linear slope")
        axes[3, j].plot(t, a["lift_J_dimensionless"][:, j, j], color="#ce772e", label=axis + " diagonal slope / linear slope")
        axes[0, j].set_ylabel(axis + " r − f (G)"); axes[1, j].set_ylabel("Unclipped integral drive (mrad)")
        axes[2, j].set_ylabel("Equivalent target / raw finsAoa"); axes[3, j].set_ylabel("Fixed-basis incidence→lift slope"); axes[3, j].set_xlabel("Original clock (s)")
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.axhline(0, color="#aaa", lw=.6); ax.axvline(.7, color="#aaa", lw=.6); ax.axvline(2.3, color="#aaa", lw=.6)
        ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    fig.suptitle(cid + ": feedback bias versus request, and the frozen finite-force Jacobian\nSource 3/5pt feedback is diagnostic only; integral drive is not retained controller memory; no free trajectory", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT / (cid + "_feedback_bias.png"), dpi=160); plt.close(fig)


def main():
    before = snapshot(); inputs = dec.audit.inverse.source_inputs()
    interface_cache = np.load(INTERFACES / "axial_completed_interfaces.npz")
    pn_cache = np.load(dec.OUTPUT / "pn_pid_timeseries.npz")
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    OUT.mkdir(parents=True, exist_ok=True); arrays = {}; reports = {}
    for cid in CASES:
        a, r = diagnose_case(cid, inputs[cid], interface_cache, pn_cache, coefficients)
        arrays.update({cid + "__" + key: value for key, value in a.items()}); reports[cid] = r; plot_case(cid, a)
    verification = {"all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()), "protected_size_mtime_unchanged": before == snapshot(),
        "case_checks": {cid: r["checks"] for cid, r in reports.items()}, "controller_hz": 48, "plant_substeps_unchanged": 4,
        "plant_integrations": 0, "new_derivative_calculations": 0, "new_fits_scans_coefficients": 0, "source_feedback_in_runtime": False,
        "raw_PID_and_schedules_unchanged": True, "defaults_core_config_raw_docs_changed": False}
    if not verification["all_arrays_finite"] or not verification["protected_size_mtime_unchanged"] or any(v > 1e-12 for r in reports.values() for v in r["checks"].values()):
        raise AssertionError(verification)
    np.savez_compressed(OUT / "feedback_bias.npz", **arrays)
    doc = {"task": "War Thunder cached feedback-bias and frozen-force allocation diagnostic", "cases": reports, "verification": verification,
        "force_formula": {"s": "hypot(local_pitch, local_yaw)", "R": "1 below alpha_s; width^2/[width^2+(s-alpha_s)^2] above",
            "J_normalized": "R*I + (R_prime/s)*local*local_transpose", "lambda_tangential": "R", "lambda_radial": "R+s*R_prime",
            "peak_s": "sqrt(alpha_s^2+width^2)", "parameters_rad": {"alpha_s": dec.audit.fix.finite.ALPHA_S, "width": dec.audit.fix.finite.WIDTH}},
        "conditions_and_limits": [
            "All source feedback is cached derivative-derived total specific acceleration projected to the same current wind up/right basis; unchanged outer r is held fixed. No source vector or future derivative enters runtime.",
            "Existing PN/PID source3/5 one-step counterfactuals are replay-validated from exactly their saved component-interpolated feedback. Independent histories use previously cached world-vector interpolation/current-basis projection; the interpolation difference is reported separately.",
            "Each own history starts at original zero/inactive memory, with original raw p/i/d, 48Hz and gain schedule, K-transport, target limit and conditional antiwindup. Fixed-state histories are not free-state or native-game validation.",
            "Ki integral drive is the unconstrained error increment sum, not output I or retained memory. Coordinate transport and clipping/antiwindup are preserved; target K*I/rawlimit and antiwindup are separately shown.",
            "Duration/integral intervals use original48Hz zero-order hold clipped to window bounds. Control-error RMS and tick counts preserve the prior inclusive original sample-time masks. Both remove the existing conservative activation derivative halo. Needed3/5 inputs remain conditional frozen-plant inverses; impossible cap masks remain excluded from attainable control comparisons.",
            "Source3/5 derivatives are noncausal estimates on a24Hz export. Interpolation to48Hz creates no new information; body quaternion, velocity and world vector interpolations can introduce small differences. Only agreement between both methods supports a stronger diagnostic inference.",
            "Lift Jacobian fixes wind basis, qbar, mass and thrust direction, and differentiates only incidence→aerodynamic lift. It is not the total specific-force derivative or a closed-loop body actuator allocation. Positive Ku angular authority remains unchanged.",
            "Negative radial lift slope means a positive radial incidence change reduces aerodynamic lift within this candidate law; it cannot by itself establish that both pitch and yaw actuator inputs need reflection. Axis diagonals and cross-coupling are reported.",
            "No reflection, allocation candidate, extra coefficient, fit, trajectory integration or source-feedback runtime controller is implemented."]}
    dec.audit.slew.write_json(OUT / "results.json", doc)
    print(json.dumps({"verification": verification, "force_Jacobian_summary": {cid: r["force_Jacobian_summary"] for cid, r in reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
