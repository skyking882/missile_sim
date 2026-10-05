#!/usr/bin/env python3
"""Game-only conditional request audit; no plant integration or model edits.

Predeclared comparisons: legacy raw PID inverse from zero memory, and one-step
current-controller inverse given its conditional available-request memory.
The latter is explicitly infeasible above its existing target authority. The
saved 3/5-point inputs are noncausal diagnostic derivatives, not runtime data.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import invert_r771_jari_control as inverse
import fix_r771_jari_control as fix
from fit_r771_jari_finite_force import comparison, drag, slew
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/control_request_audit"
CASES = ("jari_259_30", "jari_259_60")
WINDOWS = (("startup", 14 / 48, .7), ("platform", .7, 2.3), ("unloading", 2.3, 3.5))
DT = 1 / 48


def interp(t, source_t, value):
    if t[0] < source_t[0] or t[-1] > source_t[-1]:
        raise ValueError("Interpolation outside original reference support")
    value = np.asarray(value)
    if value.ndim == 1:
        return np.interp(t, source_t, value)
    return np.column_stack([np.interp(t, source_t, value[:, j]) for j in range(value.shape[1])])


def clip(value, limit):
    return max(-limit, min(limit, value))


def natural_pid(error, memory, pid):
    """Unclamped output in exactly the existing legacy coordinates."""
    integral, previous_error, previous_derivative = memory
    provisional = clip(integral + pid["i"] * error * DT, pid["integral_limit"])
    derivative = previous_derivative + DT / (1e-9 + DT) * ((error - previous_error) / DT - previous_derivative)
    return pid["p"] * error + provisional + pid["d"] * derivative


def error_for_output(required, memory, pid, gain, limit):
    """Interior unique inverse; at-cap returns a boundary, above-cap impossible.

    The boundary error is not an exact saturated inverse: all errors further in
    the saturation direction produce that output, with different next memories.
    """
    if gain <= 1e-12:
        return 0., 0  # Inactive gain gives no identifiable request.
    desired = required / gain
    if abs(desired) > limit + 1e-12:
        return 0., -1
    lower, upper = -1., 1.
    while natural_pid(lower, memory, pid) > desired:
        lower *= 2
    while natural_pid(upper, memory, pid) < desired:
        upper *= 2
    for _ in range(80):
        middle = (lower + upper) / 2
        if natural_pid(middle, memory, pid) < desired:
            lower = middle
        else:
            upper = middle
    status = 2 if abs(abs(desired) - limit) <= 1e-12 else 1
    return (lower + upper) / 2, status


def legacy_inverse(required, gains, pids, limits):
    """Recursive inverse only while output is interior; no invented sat memory."""
    memory = [(0., 0., 0.), (0., 0., 0.)]
    errors = np.zeros_like(required); closed = np.zeros_like(required)
    histories = np.zeros((len(required), 2, 3)); status = np.zeros_like(required, dtype=int)
    identified = [True, True]
    for i, (need, gain, pid) in enumerate(zip(required, gains, pids)):
        for axis in range(2):
            if not identified[axis]:
                status[i, axis] = -2  # Prior saturated output leaves memory nonunique.
                continue
            error, code = error_for_output(need[axis], memory[axis], pid, gain, limits[axis])
            status[i, axis] = code; errors[i, axis] = error
            if code == 1:
                output, memory[axis] = SingleSurfaceGame._pid(
                    _LEGACY_CONTEXT, error, memory[axis], pid, limits[axis], gain)
                closed[i, axis] = output
            elif code in (-1, 2):
                identified[axis] = False
            histories[i, axis] = memory[axis]
    return errors, closed, histories, status


class _LegacyContext:
    controller_hz = 48


_LEGACY_CONTEXT = _LegacyContext()


def ranges(value, mask):
    return [[float(value[mask, j].min()), float(value[mask, j].max())] for j in range(2)] if mask.any() else None


def series_stats(value, t, mask, rate_valid=None):
    rate_mask = mask.copy()
    if rate_valid is not None:
        # Central differences adjacent to an impossible placeholder are not rates.
        rate_mask &= rate_valid & np.roll(rate_valid, 1) & np.roll(rate_valid, -1)
        rate_mask[0] = False; rate_mask[-1] = False
    return {"range_pitch_yaw_G": ranges(value, mask),
        "max_joint_norm_G": float(np.max(np.linalg.norm(value[mask], axis=1))) if mask.any() else None,
        "RMS_rate_pitch_yaw_G_s": [inverse.rms(np.gradient(value[:, j], t)[rate_mask]) for j in range(2)] if rate_mask.any() else None,
        "rate_rows": int(rate_mask.sum()),
        "negative_fraction_pitch_yaw": [float(np.mean(value[mask, j] < 0)) for j in range(2)]}


def compare_stats(needed, available, mask):
    residual = available - needed
    return {"available_minus_needed_bias_pitch_yaw_G": np.mean(residual[mask], axis=0).tolist(),
        "available_minus_needed_RMS_pitch_yaw_G": [inverse.rms(residual[mask, j]) for j in range(2)],
        "sign_disagreement_fraction_pitch_yaw": [float(np.mean(needed[mask, j] * available[mask, j] < 0)) for j in range(2)]}


def conditional_case(cid, item, z, coefficients):
    cfg = item["config"]; ref = item["ref"]; source_t = z[cid + "__t"]
    t = np.arange(math.ceil((3.5 + 1e-6) * 48) + 1) / 48
    model = fix.make_model(cfg, coefficients)
    model.initialize(dataclasses.replace(item["initial"]))
    raw_model = drag.FittedRateAngleGame(cfg, drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
    raw_model.initialize(dataclasses.replace(item["initial"]))
    limits = np.radians([cfg["aerodynamics"][key] for key in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    gains = np.array([interpolate_table(float(ts) + DT, cfg["guidance"]["flight_time_gain_table"]) for ts in t])
    pids = [scheduled_pid_values(cfg["control"]["pid"], float(ts) + DT) for ts in t]
    maximum = float(cfg["guidance"]["maximum_lateral_acceleration_g"]) * model.gravity / G0_MPS2
    if np.max(gains) > 1 or np.min(gains) < 0:
        raise AssertionError("Original flight gain outside zero to one")
    available = []; half = []; raw = []; plant_feedback = []; k = []; nonlinear = []
    for ts in t:
        state = fix.source_state_at(ref, float(ts)); f = model.evaluate(state, float(ts), (0., 0.))
        request, details = model.request(state, item["target_at"](float(ts)), float(ts), f)
        original, _ = raw_model.request(state, item["target_at"](float(ts)), float(ts), f)
        available.append(request); half.append(details["slew_original_request_g"]); raw.append(original)
        plant_feedback.append(f.feedback_g); k.append(model.current_control_k)
        nonlinear.append(model.current_force_nonlinear)
    available = np.asarray(available); half = np.asarray(half); raw = np.asarray(raw)
    plant_feedback = np.asarray(plant_feedback); k = np.asarray(k); nonlinear = np.asarray(nonlinear)
    current_high = np.maximum.accumulate(nonlinear & (gains > 1e-12))
    effective_limits = np.where(current_high[:, None], np.minimum(limits, limits / k[:, None]), limits)
    arrays = {"t": t, "source_t_original": source_t, "flight_gain": gains,
        "available_S110_G": available, "available_half_unslewed_G": half,
        "available_raw_full_RateAngle_G": raw, "plant_feedback_G": plant_feedback,
        "K_target_per_old_held": k, "current_high_mode": current_high,
        "current_effective_old_held_limit_rad": effective_limits,
        "source_controls_pitch_yaw": interp(t, source_t, z[cid + "__source_controls_pitch_yaw"])}
    report = {"source_original_rows": len(source_t), "diagnostic_grid_rows": len(t),
        "grid_support_s": [float(t[0]), float(t[-1])], "first_active_s": float(t[gains > 0][0]),
        "source_state_interpolation": "linear pos/velocity/omega/mass and normalized linear quaternion interpolation; no clock shift",
        "source_state_high_mode_first_s": float(t[current_high][0]) if current_high.any() else None,
        "request_cap_joint_norm_G": maximum, "original_PID_entire_dict": cfg["control"]["pid"],
        "original_old_held_axis_limits_deg": np.degrees(limits).tolist(), "methods": {}}
    active = gains > 1e-12
    for width in (3, 5):
        need = interp(t, source_t, z[cid + f"__{width}__old_held_req_rad"])
        norm_req = k[:, None] * need / limits
        source_feedback = interp(t, source_t, z[cid + f"__{width}__source_feedback_pitch_yaw_G"])
        errors, closed, memory, status = legacy_inverse(need, gains, pids, limits)
        prefix = str(width) + "__"
        arrays.update({prefix + "old_held_required_rad": need, prefix + "target_normalized_required": norm_req,
            prefix + "source_feedback_G": source_feedback, prefix + "legacy_required_error_G": errors,
            prefix + "legacy_forward_output_rad": closed, prefix + "legacy_inverse_memory": memory,
            prefix + "legacy_inverse_status": status})
        legacy_valid = np.all(status == 1, axis=1)
        record = {"legacy_recursive_forward_closure_max_abs_rad": float(np.max(abs(closed[status == 1] - need[status == 1]))),
            "legacy_interior_inverse_axis_ticks": int(np.sum(status == 1)),
            "legacy_noninterior_active_axis_ticks": int(np.sum(active[:, None] & (status != 1))),
            "inactive_nonzero_required_axis_ticks_above_1e_10_rad": int(np.sum(~active[:, None] & (abs(need) > 1e-10))),
            "inactive_inverse_required_max_abs_old_held_deg": np.degrees(np.max(abs(need[~active]), axis=0)).tolist(),
            "current_target_cap_impossible_axis_ticks": int(np.sum(active[:, None] & (abs(need) > gains[:, None] * effective_limits + 1e-12))),
            "windows": {}, "current_given_memory": {}}
        for feedback_name, feedback in (("plant", plant_feedback), ("source", source_feedback)):
            req_needed = errors + feedback
            arrays[prefix + "legacy_required_request_" + feedback_name + "_G"] = req_needed
            # Forward conditional controller on available S110, with two independent feedback assumptions.
            controller = R771GameControl(); forward_memory = [(0., 0., 0.), (0., 0., 0.)]
            held = np.zeros_like(need); histories = np.zeros((len(t), 2, 3))
            one_step_error = np.zeros_like(need); one_step_status = np.zeros_like(need, dtype=int)
            one_step_closed = np.zeros_like(need); transports = np.ones_like(need); blocked = np.zeros_like(need, dtype=bool)
            for i, ts in enumerate(t):
                if gains[i] <= 1e-12:
                    continue
                for axis in range(2):
                    mem = forward_memory[axis]
                    if current_high[i]:
                        prev_k = controller.previous_high_k[axis]
                        ratio = 1. if prev_k is None else prev_k / k[i]
                        mem = (clip(mem[0] * ratio, pids[i]["integral_limit"]), mem[1], mem[2])
                        transports[i, axis] = ratio
                    error, code = error_for_output(need[i, axis], mem, pids[i], gains[i], effective_limits[i, axis])
                    one_step_error[i, axis] = error; one_step_status[i, axis] = code
                    if code in (1, 2):
                        trial = copy.deepcopy(controller)
                        output, _ = trial.update_axis(axis, float(ts), error, forward_memory[axis], pids[i], limits[axis], gains[i], float(k[i]), bool(nonlinear[i]))
                        one_step_closed[i, axis] = output
                    output, forward_memory[axis] = controller.update_axis(axis, float(ts), float(available[i, axis] - feedback[i, axis]),
                        forward_memory[axis], pids[i], limits[axis], gains[i], float(k[i]), bool(nonlinear[i]))
                    held[i, axis] = output; histories[i, axis] = forward_memory[axis]
                    blocked[i, axis] = controller.diagnostics[axis]["antiwindup_blocked"]
            feasible = np.all(one_step_status == 1, axis=1)
            one_req = one_step_error + feedback
            tag = prefix + "current_" + feedback_name + "__"
            arrays.update({tag + "forward_held_rad": held, tag + "forward_memory": histories,
                tag + "forward_normalized_target": k[:, None] * held / limits,
                tag + "given_memory_required_error_G": one_step_error,
                tag + "given_memory_required_request_G": one_req,
                tag + "given_memory_status": one_step_status, tag + "forward_antiwindup_blocked": blocked,
                tag + "given_memory_closed_output_rad": one_step_closed, tag + "integral_transport_ratio": transports})
            unique = one_step_status == 1
            record["current_given_memory"][feedback_name] = {
                "forward_check_max_abs_rad_on_unique_axes": float(np.max(abs(one_step_closed[unique] - need[unique]))),
                "interpretation": "one-step request given available-S110 conditional prior memory; not a recursively exact desired-output history",
                "unique_axis_ticks": int(unique.sum()), "impossible_axis_ticks": int(np.sum(one_step_status == -1)),
                "saturated_boundary_axis_ticks": int(np.sum(one_step_status == 2)),
                "saturated_boundary_rule": "positive cap requires error >= boundary; negative cap error <= boundary; next memory not unique",
                "window_details": {}}
            for name, lo, hi in WINDOWS:
                mask = inverse.window_mask(t, lo, hi) & active
                both = mask & feasible
                record["current_given_memory"][feedback_name]["window_details"][name] = {
                    "joint_unique_ticks": int(both.sum()), "all_active_ticks": int(mask.sum()),
                    "required_given_memory_request_G": series_stats(one_req, t, both, feasible) if both.any() else None,
                    "required_joint_norm_over_50G_fraction_unique": float(np.mean(np.linalg.norm(one_req[both], axis=1) > maximum)) if both.any() else None,
                    "current_forward_target_cap_fraction_pitch_yaw": np.mean((abs(k[:, None] * held / limits) >= .99)[mask], axis=0).tolist(),
                    "antiwindup_fraction_pitch_yaw": np.mean(blocked[mask], axis=0).tolist(),
                    "forward_integral_old_rad_range_pitch_yaw": ranges(histories[:, :, 0], mask),
                    "forward_normalized_target_range_pitch_yaw": ranges(k[:, None] * held / limits, mask),
                    "forward_normalized_target_negative_fraction_pitch_yaw": np.mean((held < 0)[mask], axis=0).tolist(),
                    "inverse_vs_available": compare_stats(one_req, available, both) if both.any() else None}
            interior = inverse.window_mask(t, .5, 3.5)
            original_mask = inverse.window_mask(source_t)
            source_clock_target = interp(source_t[original_mask], t, k[:, None] * held / limits)
            record["current_given_memory"][feedback_name]["interior_0p5_3p5"] = {
                "conditional_48Hz_rows": int(interior.sum()),
                "negative_output_ticks_pitch_yaw": np.sum((held < 0)[interior], axis=0).tolist(),
                "original_source_rows": int(original_mask.sum()),
                "negative_sampled_output_source_rows_pitch_yaw": np.sum(source_clock_target < 0, axis=0).tolist(),
                "normalized_target_minus_source_control_RMS_pitch_yaw": [inverse.rms(source_clock_target[:, j] - z[cid + "__source_controls_pitch_yaw"][original_mask, j]) for j in range(2)]}
        for name, lo, hi in WINDOWS:
            mask = inverse.window_mask(t, lo, hi) & active
            valid = mask & legacy_valid
            high_mask = mask & current_high
            entry = {"window_s": [lo, hi], "active_ticks": int(mask.sum()), "unique_legacy_ticks": int(valid.sum()),
                "old_held_required_range_pitch_yaw_deg": ranges(np.degrees(need), mask),
                "target_normalized_required_range_pitch_yaw": ranges(norm_req, mask),
                "current_impossible_fraction_pitch_yaw": np.mean((abs(need) > gains[:, None] * effective_limits + 1e-12)[mask], axis=0).tolist(),
                "current_high_mode_ticks": int(high_mask.sum()),
                "legacy_inverse_integral_old_rad_range_pitch_yaw": ranges(memory[:, :, 0], valid),
                "source_controls_range_pitch_yaw": ranges(arrays["source_controls_pitch_yaw"], mask),
                "feedback_plant_minus_source_RMS_pitch_yaw_G": [inverse.rms((plant_feedback - source_feedback)[mask, j]) for j in range(2)],
                "feedback_plant_minus_source_bias_pitch_yaw_G": np.mean((plant_feedback - source_feedback)[mask], axis=0).tolist(),
                "feedback_source_range_pitch_yaw_G": ranges(source_feedback, mask),
                "feedback_plant_range_pitch_yaw_G": ranges(plant_feedback, mask),
                "available": {}, "legacy_required": {}}
            for label, values in (("S110", available), ("half_unslewed", half), ("raw_full_RateAngle", raw)):
                entry["available"][label] = series_stats(values, t, mask)
                entry["available"][label]["joint_norm_over_50G_fraction"] = float(np.mean(np.linalg.norm(values[mask], axis=1) > maximum + 1e-10))
            for feedback_name, feedback in (("plant", plant_feedback), ("source", source_feedback)):
                req = errors + feedback
                entry["legacy_required"][feedback_name] = {**series_stats(req, t, valid),
                    "joint_norm_over_50G_fraction": float(np.mean(np.linalg.norm(req[valid], axis=1) > maximum)),
                    "comparisons": {label: compare_stats(req, v, valid) for label, v in (("S110", available), ("half_unslewed", half), ("raw_full_RateAngle", raw))}}
            record["windows"][name] = entry
        impossible = active[:, None] & (abs(need) > gains[:, None] * effective_limits + 1e-12)
        record["current_impossible_intervals_s_pitch_yaw"] = [fix.finite.intervals(t, impossible[:, j]) for j in range(2)]
        duration = np.minimum(DT, np.maximum(3.5 - t, 0.))
        record["current_impossible_ZOH_duration_until_3p5_s_pitch_yaw"] = np.sum(impossible * duration[:, None], axis=0).tolist()
        halo = (t >= 14 / 48 - 2 / 24) & (t <= 14 / 48 + 2 / 24)
        record["gain_transition_stencil_halo"] = {
            "conservative_support_s": [float(t[halo][0]), float(t[halo][-1])],
            "required_old_held_range_pitch_yaw_deg": ranges(np.degrees(need), halo),
            "first_active_required_old_held_deg": np.degrees(need[np.flatnonzero(active)[0]]).tolist(),
            "classification": "noncausal derivative halo, not evidence for gain anticipation"}
        report["methods"][str(width) + "_point"] = record
    report["derivative_stencil_sensitivity"] = {
        name: {"legacy_request_source_feedback_3_minus_5_RMS_pitch_yaw_G": [inverse.rms((arrays["3__legacy_required_request_source_G"] - arrays["5__legacy_required_request_source_G"])[inverse.window_mask(t, lo, hi), j]) for j in range(2)]}
        for name, lo, hi in WINDOWS}
    # Saved free closed-loop history is only a same-time descriptive comparison.
    cached = json.loads((fix.OUTPUT / "full_runs" / (cid + ".json")).read_text())
    ct = np.asarray([row["time_s"] for row in cached["ticks"]])
    for field in ("request_g", "feedback_g", "held_fin_rad", "pid_memory"):
        value = np.asarray([row[field] for row in cached["ticks"]])
        arrays["cached_free__" + field] = value[ct <= t[-1] + 1e-8]
    arrays["cached_free__t"] = ct[ct <= t[-1] + 1e-8]
    report["cached_free_comparison"] = "Actual ControlFix closed-loop states differ from source states; same-time differences cannot establish PID insufficiency."
    return arrays, report


def plot(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; active = a["flight_gain"] > 0
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis, label in enumerate(("Pitch", "Yaw")):
        ax = axes[0, axis]
        for width, color in ((3, "#8aabab"), (5, "#086a6a")):
            req = a[f"{width}__legacy_required_request_source_G"][:, axis]
            ax.plot(t[active], req[active], color=color, label=f"Required, source feedback {width}pt")
        ax.plot(t[active], a["5__legacy_required_request_plant_G"][active, axis], color="#e48a4b", ls=":", label="Required, plant feedback 5pt")
        ax.plot(t, a["available_S110_G"][:, axis], color="#8d2d70", label="Available S110")
        ax.plot(t, a["available_half_unslewed_G"][:, axis], color="#555555", ls="--", label="Available half, unslewed")
        ax.plot(t, a["available_raw_full_RateAngle_G"][:, axis], color="#7781b2", ls="-.", label="Existing raw PN/full loft")
        ax.set_ylabel(label + " request (G)")
        ax = axes[1, axis]
        ax.plot(t, a["5__source_feedback_G"][:, axis], color="#086a6a", label="Source velocity feedback 5pt")
        ax.plot(t, a["plant_feedback_G"][:, axis], color="#e48a4b", label="Plant feedback")
        current = a["5__current_plant__given_memory_status"][:, axis] == 1
        ax.plot(t, np.where(current, a["5__current_plant__given_memory_required_request_G"][:, axis], np.nan), color="#bd4368", ls="--", label="One-step req / known current memory")
        bad = a["5__current_plant__given_memory_status"][:, axis] == -1
        ax.scatter(t[bad], a["5__source_feedback_G"][bad, axis], color="#c32626", s=5, label="Required output beyond current cap")
        ax.set_ylabel(label + " feedback / one-step request (G)")
        ax = axes[2, axis]
        for width, color in ((3, "#8aabab"), (5, "#086a6a")):
            ax.plot(t[active], a[f"{width}__target_normalized_required"][active, axis], color=color, label=f"Required target {width}pt")
        ax.plot(t, a["source_controls_pitch_yaw"][:, axis], color="#171717", ls=":", label="Jari exported control")
        ax.plot(t, a["5__current_plant__forward_normalized_target"][:, axis], color="#8d2d70", label="Conditional current PID / plant feedback")
        ax.plot(t, a["5__current_source__forward_normalized_target"][:, axis], color="#df9548", ls="--", label="Same requests / source feedback")
        ax.axhline(1, color="#c32626", ls="--", lw=.8, label="High-mode target authority")
        ax.set_ylabel(label + " target / raw finsAoa")
        iax = ax.twinx()
        iax.plot(t, a["5__legacy_inverse_memory"][:, axis, 0], color="#8a9535", alpha=.7, lw=.8, label="Inverse legacy I (old rad)")
        iax.plot(t, a["5__current_plant__forward_memory"][:, axis, 0], color="#cfa932", alpha=.7, lw=.8, ls="--", label="Current conditional I (old rad)")
        iax.set_ylabel("Integral contribution (old rad)", color="#7f7835", fontsize=8)
        iax.legend(loc="upper right", fontsize=6)
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.grid(alpha=.18)
        ax.axvspan(14 / 48, .7, color="#a5c9e4", alpha=.08)
        ax.axvspan(.7, 2.3, color="#d5ddc3", alpha=.08)
        ax.axvline(.7, color="#aaa", lw=.7); ax.axvline(2.3, color="#aaa", lw=.7)
        ax.legend(loc="best", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("Original clock (s)")
    fig.suptitle(cid + ": conditional request inverse, zero free integrations\nLegacy recursive inverse and current given-memory inverse are different questions", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUTPUT / (cid + "_request_audit.png"), dpi=160); plt.close(fig)


def protected_snapshot():
    files = [p for folder in (ROOT / "src", ROOT / "config", ROOT / "docs", ROOT / "scripts", ROOT / "outputs/jari_offaxis", ROOT / "data/raw")
        if folder.exists() for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and OUTPUT not in p.parents and p != Path(__file__).resolve()]
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}


def main():
    before = protected_snapshot()
    inputs = inverse.source_inputs(); z = np.load(inverse.OUTPUT / "inverse_inputs.npz")
    coefficients = drag.DragCoefficients(**json.loads((drag.OUTPUT / "fit.json").read_text())["coefficients"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = {}; cases = {}
    for cid in CASES:
        case_arrays, report = conditional_case(cid, inputs[cid], z, coefficients)
        arrays.update({cid + "__" + key: value for key, value in case_arrays.items()}); cases[cid] = report
        plot(cid, case_arrays)
        print(json.dumps({"case": cid, "legacy_forward_closure": report["methods"]["5_point"]["legacy_recursive_forward_closure_max_abs_rad"],
            "current_impossible": report["methods"]["5_point"]["current_impossible_intervals_s_pitch_yaw"],
            "startup_source_required": report["methods"]["5_point"]["windows"]["startup"]["legacy_required"]["source"]}), flush=True)
    np.savez_compressed(OUTPUT / "request_audit.npz", **arrays)
    finite = all(np.all(np.isfinite(v)) for v in arrays.values())
    checks = {"zero_free_integrations": True, "zero_API_calls": True, "zero_fits": True,
        "all_saved_arrays_finite": bool(finite), "protected_files_size_mtime_unchanged": before == protected_snapshot(),
        "controller_hz": 48, "plant_substeps_config_unchanged": 4, "clock_shift_s": 0,
        "no_source_extrapolation": True, "raw_PID_entire_dict_recorded": True,
        "legacy_unique_forward_max_abs_rad": max(r["methods"][m]["legacy_recursive_forward_closure_max_abs_rad"] for r in cases.values() for m in ("3_point", "5_point")),
        "current_given_memory_forward_max_abs_rad": max(v["forward_check_max_abs_rad_on_unique_axes"] for r in cases.values() for m in ("3_point", "5_point") for v in r["methods"][m]["current_given_memory"].values())}
    if not finite or not checks["protected_files_size_mtime_unchanged"] or checks["legacy_unique_forward_max_abs_rad"] > 1e-12 or checks["current_given_memory_forward_max_abs_rad"] > 1e-12:
        raise AssertionError(checks)
    document = {"task": "War Thunder/Jari game conditional request diagnosis", "cases": cases, "verification": checks,
        "conditions_and_limits": [
            "Original source at24Hz is interpolated to48Hz; source PID memory and intersample native commands are unobserved.",
            "Recursive legacy inverse assumes zero initial PID memory and preserves original inactive gain; it is conditional, not unique game guidance truth.",
            "3/5-point source derivatives use future observations and boundary stencils; their difference is numerical sensitivity, not a confidence interval.",
            "First active tick and two native source rows either side of gain transition are derivative-halo sensitive; no anticipation or time shift is introduced.",
            "Current one-step inverse uses the available-request conditional controller memory, not a fictional recursively exact history after impossible output.",
            "At saturation only a one-sided error bound exists; above effective cap no request can achieve the required output.",
            "Legacy old-held radians and normalized target radians/rawfinsAoa are different input semantics and different caps.",
            "Both request axes share a50G vector norm; axiswise bounds alone are insufficient.",
            "Source-feedback reconstruction assumes raw gravity and thrust/force interpretation already recorded in inverse-control evidence.",
            "Source-state comparisons are conditional diagnostics; cached free runtime histories are only same-time descriptions.",
            "All runtime/core/config/raw/docs and earlier outputs remain unchanged."]}
    slew.write_json(OUTPUT / "results.json", document)
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
