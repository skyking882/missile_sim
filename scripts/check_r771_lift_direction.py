#!/usr/bin/env python3
"""One opt-in, norm-preserving lift-direction experiment for a game unit.

Geometry is causal and script-local. Its force-direction rule is a D-grade
hypothesis, not recovered game source. No force magnitude or coefficient fit.
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
import analyze_r771_pn_pid_decoupling as dec
from aim120_model.aerodynamics import BodyAxes, body_axes_for_state
from aim120_model.math3d import add, dot, norm, scale, sub
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUT = ROOT / "outputs/jari_offaxis/lift_direction"
CASES = ("jari_259_0", "jari_259_30", "jari_259_60")
AXES = ("pitch", "yaw")
COM_GEOMETRY_FORMULA = {
    "forward_c": "1/sqrt(1+tan(alpha_pitch)^2+tan(alpha_yaw)^2), on the forward branch",
    "pitch_projection": "c^2*tan(alpha_pitch)/sqrt(1-c^2*tan(alpha_pitch)^2)",
    "yaw_projection": "c*tan(alpha_yaw)/sqrt(1-c^2*tan(alpha_pitch)^2)",
    "geometric_pitch_over_yaw": "c*tan(alpha_pitch)/tan(alpha_yaw), when yaw is nonzero",
    "old_direction_pitch_over_yaw": "alpha_pitch/alpha_yaw",
    "local_flow_note": "These formulas use COM angles and zero rotational-flow correction. The candidate reconstructs the original local angles and then explicitly reprojects into the COM plane instead of substituting these COM formulas."
}


def snapshot(*, native=False):
    other_script = ROOT / "scripts/check_r771_envelope_control.py"
    other_output = ROOT / "outputs/jari_offaxis/envelope_control"
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis")
        for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts and OUT not in p.parents and p != Path(__file__).resolve()
        and not (native and (p == other_script or other_output in p.parents))}


def geometric_lift(body, wb, local, existing_lift):
    """Equivalent local-flow forward branch, then COM-normal reprojection.

    The domain |local_pitch/yaw|<pi/2 is the branch on which the two original
    atan2 angle ratios uniquely reconstruct a forward flow. A free run leaving
    it is rejected, never rescued by an invented gain, angle cap or fallback.
    """
    magnitude = norm(existing_lift)
    if magnitude <= 1e-12:
        return existing_lift
    if any(abs(x) >= math.pi / 2 for x in local):
        raise ValueError("Local-flow direction left the declared forward reconstruction branch")
    flow = sub(sub(body.forward, scale(body.up, math.tan(local[0]))), scale(body.right, math.tan(local[1])))
    flow = scale(flow, 1 / norm(flow))
    direction = sub(body.forward, scale(flow, dot(body.forward, flow)))
    direction = sub(direction, scale(wb.forward, dot(direction, wb.forward)))
    length = norm(direction)
    if length <= 1e-12:
        raise ValueError("Nonzero lift has no reconstructable COM-normal geometric direction")
    return scale(direction, magnitude / length)


class LiftDirectionGame(dec.audit.fix.ControlFixGame):
    """Only rotate the current lift resultant; retain the same angular law."""
    def evaluate(self, state, time_s, held):
        original = super().evaluate(state, time_s, held)
        body = body_axes_for_state(state)
        lift = geometric_lift(body, original.wind_basis, original.local_incidence, original.wind_lift_force)
        aero = add(lift, original.wind_drag_force)
        specific = scale(add(aero, scale(body.forward, original.thrust)), 1 / original.mass)
        lever = (-self.arm * dot(aero, body.up), -self.arm * dot(aero, body.right))
        couple = tuple(original.inertia * q - l for q, l in zip(original.angular_acceleration, lever))
        return dataclasses.replace(original, acceleration=add(specific, (0., -self.gravity, 0.)), specific_acceleration=specific,
            aerodynamic_force=aero, wind_lift_force=lift, feedback_g=dec.projected(specific, original.wind_basis), lever_moment=lever, reference_couple=couple)


def make_model(cfg, coefficients):
    return LiftDirectionGame(cfg, request_slew_g_s=110., drag_coefficients=coefficients, coefficients=SurfaceCoefficients(),
        force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def angle_between(a, b):
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    cos = np.divide(np.sum(a * b, axis=1), na * nb, out=np.ones(len(a)), where=valid)
    return np.degrees(np.arccos(np.clip(cos, -1., 1.))), valid


def conditional_case(cid, item, inverse, world_cache, coefficients):
    cfg = item["config"]; t = world_cache[cid + "__t"]
    baseline = dec.audit.fix.make_model(cfg, coefficients); candidate = make_model(cfg, coefficients)
    baseline.initialize(dataclasses.replace(item["initial"])); candidate.initialize(dataclasses.replace(item["initial"]))
    rows = []; closure = []
    for ts in t:
        state = dec.audit.fix.source_state_at(item["ref"], float(ts)); body = body_axes_for_state(state)
        old = baseline.evaluate(state, float(ts), (0., 0.)); new = candidate.evaluate(state, float(ts), (0., 0.))
        req_old, _ = baseline.request(state, item["target_at"](float(ts)), float(ts), old)
        req_new, _ = candidate.request(state, item["target_at"](float(ts)), float(ts), new)
        wb = old.wind_basis; thrust = scale(body.forward, old.thrust / old.mass)
        com_geom = sub(body.forward, scale(wb.forward, dot(body.forward, wb.forward)))
        rows.append({"request_G": req_old, "baseline_feedback_G": old.feedback_g, "candidate_feedback_G": new.feedback_g,
            "baseline_lift_G": dec.projected(scale(old.wind_lift_force, 1 / old.mass), wb),
            "candidate_lift_G": dec.projected(scale(new.wind_lift_force, 1 / new.mass), wb), "plant_thrust_G": dec.projected(thrust, wb),
            "lift_norm_N": norm(old.wind_lift_force), "local_incidence_rad": old.local_incidence,
            "COM_geometric_incidence_rad": (old.alpha_pitch, old.alpha_yaw), "body_forward_COMnormal_projection": dec.projected(com_geom, wb),
            "candidate_lift_world_N": new.wind_lift_force, "baseline_lift_world_N": old.wind_lift_force,
            "K": baseline.current_control_k, "nonlinear_force_active": baseline.current_force_nonlinear})
        closure.append({"request_max_abs_G": max(abs(a - b) for a, b in zip(req_old, req_new)),
            "lift_norm_relative": abs(norm(new.wind_lift_force) - norm(old.wind_lift_force)) / max(norm(old.wind_lift_force), 1.),
            "drag_max_abs_N": max(abs(a - b) for a, b in zip(old.wind_drag_force, new.wind_drag_force)),
            "thrust_abs_N": abs(old.thrust - new.thrust), "angular_law_max_abs_rad_s2": max(abs(a - b) for a, b in zip(old.angular_acceleration, new.angular_acceleration)),
            "same_station_torque_closure_Nm": max(abs(new.inertia * q - l - c) for q, l, c in zip(new.angular_acceleration, new.lever_moment, new.reference_couple)),
            "lift_COMnormal_relative": abs(dot(new.wind_lift_force, wb.forward)) / max(norm(new.wind_lift_force), 1.)})
    a = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}; a["t"] = t
    a["halo"] = world_cache[cid + "__gain_transition_derivative_halo"]
    a["flight_gain"] = world_cache[cid + "__flight_gain"]
    report = {"checks": {key: max(row[key] for row in closure) for key in closure[0]}, "methods": {}, "geometric_mapping": {}}
    report["checks"]["baseline_feedback_exact_cache_max_abs_G"] = float(np.max(abs(a["baseline_feedback_G"] - world_cache[cid + "__wind_specific__feedback_G"])))
    for key, first, second in (("COM_atan_vector_vs_body_projection_angle_deg", "COM_geometric_incidence_rad", "body_forward_COMnormal_projection"),
            ("local_atan_vector_vs_candidate_direction_angle_deg", "local_incidence_rad", "candidate_lift_G")):
        a[key], a[key + "__valid"] = angle_between(a[first], a[second])
    active = (a["flight_gain"] > 1e-12) & ~a["halo"]
    for window, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
        mask = active & dec.audit.inverse.window_mask(t, lo, hi)
        report["geometric_mapping"][window] = {key: dec.scalar_stats(a[key], mask & a[key + "__valid"]) for key in (
            "COM_atan_vector_vs_body_projection_angle_deg", "local_atan_vector_vs_candidate_direction_angle_deg")}
    ref = item["ref"]; source_t = inverse[cid + "__t"]
    source_forward = np.asarray([body_axes_for_state(dec.audit.inverse.source_state(ref, i)).forward for i in range(len(source_t))])
    for width in (3, 5):
        aero_world = dec.audit.interp(t, source_t, inverse[cid + f"__{width}__source_aerodynamic_force_N"] / ref["mass"][:, None])
        thrust_world = dec.audit.interp(t, source_t, ref["thrust"][:, None] * source_forward / ref["mass"][:, None])
        source_aero = []; source_thrust = []
        for i, ts in enumerate(t):
            state = dec.audit.fix.source_state_at(ref, float(ts)); old = baseline.evaluate(state, float(ts), (0., 0.))
            source_aero.append(dec.projected(aero_world[i], old.wind_basis)); source_thrust.append(dec.projected(thrust_world[i], old.wind_basis))
        a[f"{width}__source_aero_normal_G"] = np.asarray(source_aero)
        a[f"{width}__source_thrust_normal_G"] = np.asarray(source_thrust)
        a[f"{width}__source_feedback_G"] = np.asarray(source_aero) + source_thrust
        report["checks"][f"{width}_source_total_specific_feedback_closure_G"] = float(np.max(abs(a[f"{width}__source_feedback_G"] - world_cache[cid + f"__{width}__source_wind_specific_feedback_G_diagnostic"])))
        report["methods"][str(width) + "_point"] = {}
        for window, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
            mask = active & dec.audit.inverse.window_mask(t, lo, hi)
            w = {"axes": {}, "lift_normal_magnitude_residual_G": {}, "lift_direction_residual_deg": {}}
            for j, axis in enumerate(AXES):
                w["axes"][axis] = {label: dec.scalar_stats(a[name][:, j] - a[f"{width}__source_feedback_G"][:, j], mask)
                    for label, name in (("baseline_total_feedback_residual_G", "baseline_feedback_G"), ("candidate_total_feedback_residual_G", "candidate_feedback_G"))}
                w["axes"][axis].update({label: dec.scalar_stats(a[name][:, j] - a[f"{width}__source_aero_normal_G"][:, j], mask)
                    for label, name in (("baseline_lift_vs_source_aero_normal_residual_G", "baseline_lift_G"), ("candidate_lift_vs_source_aero_normal_residual_G", "candidate_lift_G"))})
                w["axes"][axis]["plant_vs_source_normal_thrust_residual_G"] = dec.scalar_stats(a["plant_thrust_G"][:, j] - a[f"{width}__source_thrust_normal_G"][:, j], mask)
            source_norm = np.linalg.norm(a[f"{width}__source_aero_normal_G"], axis=1)
            for name in ("baseline", "candidate"):
                w["lift_normal_magnitude_residual_G"][name] = dec.scalar_stats(np.linalg.norm(a[name + "_lift_G"], axis=1) - source_norm, mask)
                angles, valid = angle_between(a[name + "_lift_G"], a[f"{width}__source_aero_normal_G"])
                a[f"{width}__{name}_vs_source_aero_direction_deg"] = angles
                a[f"{width}__source_aero_direction_valid"] = valid
                w["lift_direction_residual_deg"][name] = dec.scalar_stats(angles, mask & valid)
            report["methods"][str(width) + "_point"][window] = w
    return a, report


def plot_conditional(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for j, axis in enumerate(AXES):
        for name, color in (("baseline", "#78337b"), ("candidate", "#ce772e")):
            axes[0, j].plot(t, a[name + "_feedback_G"][:, j], color=color, label=name + " total normal specific feedback")
            axes[1, j].plot(t, a[name + "_lift_G"][:, j] - a["5__source_aero_normal_G"][:, j], color=color, label=name + " lift − source5 aero normal")
        axes[0, j].plot(t, a["3__source_feedback_G"][:, j], color="#6b82ba", ls="--", label="Source3 diagnostic")
        axes[0, j].plot(t, a["5__source_feedback_G"][:, j], color="#177277", label="Source5 diagnostic")
        axes[1, j].plot(t, a["plant_thrust_G"][:, j] - a["5__source_thrust_normal_G"][:, j], color="#999", label="Unchanged thrust − source5 thrust")
        axes[0, j].set_ylabel(axis + " feedback (G)"); axes[1, j].set_ylabel("Component residual (G)")
    axes[2, 0].plot(t, a["COM_atan_vector_vs_body_projection_angle_deg"], label="COM atan vector vs bodyForward projection", color="#6b82ba")
    axes[2, 0].plot(t, a["local_atan_vector_vs_candidate_direction_angle_deg"], label="Original local-angle direction vs candidate", color="#ce772e")
    axes[2, 0].set_ylabel("Direction change (deg)")
    for name, color in (("baseline", "#78337b"), ("candidate", "#ce772e")):
        valid = (a["flight_gain"] > 1e-12) & ~a["halo"] & a["5__source_aero_direction_valid"]
        axes[2, 1].plot(t, np.where(valid, a["5__" + name + "_vs_source_aero_direction_deg"], np.nan), color=color, label=name + " vs source5 aero normal")
    axes[2, 1].set_ylabel("Direction residual (deg)")
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.axhline(0, color="#aaa", lw=.6); ax.axvline(.7, color="#aaa", lw=.6); ax.axvline(2.3, color="#aaa", lw=.6)
        ax.axvspan(14 / 48 - 2 / 24, 14 / 48 + 2 / 24, color="#aaa", alpha=.10)
        ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    for ax in axes[2]: ax.set_xlabel("Original clock (s)")
    fig.suptitle(cid + ": one geometric lift-direction candidate, exactly unchanged current lift norm\nEquivalent single-station local flow projected into COM wind normal; source derivatives are diagnostic only", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT / (cid + "_conditional.png"), dpi=160); plt.close(fig)


def plot_free(cid, observed, baseline, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (field, units) in zip(axes.flat, (("speed_mps", "Speed (m/s)"), ("body_pitch_deg", "Body pitch (deg)"),
            ("body_yaw_deg", "Body yaw (deg)"), ("alpha_deg", "AoA (deg)"), ("normal_specific_load_g", "Normal specific load (G)"),
            ("body_yaw_rate_deg_s", "Body yaw rate (deg/s)"))):
        for name, p, color in (("Jari interim reference", observed, "#177277"), ("Cached original ControlFixGame", baseline, "#78337b"),
                ("Norm-preserving geometric direction", candidate, "#ce772e")):
            ax.plot(p["t"], p[field], color=color, label=name)
        ax.set_xlabel("Original clock (s)"); ax.set_ylabel(units)
        ax.set_xlim(0, min(observed["t"][-1], baseline["t"][-1], candidate["t"][-1]) if field == "speed_mps" else 3.5)
        ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    fig.suptitle(cid + ": actual free integration, original controller and all scalar coefficients fixed\nOnly current lift direction changes; no reference state/vector after creation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT / (cid + "_free.png"), dpi=160); plt.close(fig)


def run_jari_free(inputs, coefficients):
    """Exactly three free integrations; preserve/recover their own receipts."""
    full = OUT / "full_runs"; full.mkdir(exist_ok=True)
    cached = np.load(dec.audit.fix.OUTPUT / "trajectories.npz")
    reports = {}; arrays = {}; new = 0; recovered = 0
    fields = dec.audit.fix.finite.fields; drag = dec.audit.drag
    units = {**dec.audit.fix.finite.UNITS, "body_pitch_deg": "deg"}
    windows = (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5), ("full_common", 0., math.inf))
    for cid in CASES:
        x = inputs[cid]; model = make_model(x["config"], coefficients); path = full / (cid + ".json")
        if path.exists():
            result = json.loads(path.read_text()); model.initialize(dataclasses.replace(x["initial"])); recovered += 1
        else:
            result = model.run(dataclasses.replace(x["initial"]), x["target_at"], end_time_s=x["config"]["performance"]["lifetime_s"])
            result["model"].update(lift_direction_rule="single-local-flow geometry reprojected into COM windnormal; current finite-lift norm unchanged",
                lift_direction_evidence_grade="D opt-in script-local hypothesis", native_game_formula_recovered=False, reference_data_in_runtime=False)
            dec.audit.slew.write_json(path, result, True); new += 1
        prediction = fields.prediction_fields(result, model, x["run"]["scenario"])
        baseline = drag.selected_arrays(cached, cid + "__control_fix__")
        observed = dict(x["observed"])
        q = fields.gap.continuous_quaternions(x["ref"]["quaternion_xyzw"][:, [3, 0, 1, 2]])
        observed["body_pitch_deg"] = fields.gap.trajectory_fields(x["ref"]["t"], x["ref"]["position_xyz_m"], x["ref"]["velocity_xyz_mps"],
            x["run"]["scenario"]["target_position"], x["run"]["scenario"]["target_velocity"], q)["body_pitch_deg"]
        metrics = drag.pair_metrics(observed, baseline, prediction, tuple(units), windows)
        for w in metrics.values():
            for field, metric in w["metrics"].items(): metric["units"] = units[field]
        extrema = {name: dec.audit.fix.trim.response_extrema(p) for name, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction))}
        checks = {"original_raw_PID_and_config_equal": model.config == x["config"], "surface_coefficients_equal": model.coefficients == SurfaceCoefficients(),
            "drag_coefficients_equal": model.drag_coefficients == coefficients, "half_PN_loft_S110_equal": model.relative_blend == .5 and model.request_slew_g_s == 110. and [model.amplitude_coefficient, model.transition_coefficient] == [.75, 1.3],
            "controller_hz": model.controller_hz, "plant_substeps": model.plant_substeps, "same_station_arm_m": model.arm,
            "original_clock_max_abs_tick": float(np.max(abs(np.asarray([v["time_s"] for v in result["ticks"]]) * 48 - np.arange(len(result["ticks"])))))}
        if not all(checks[k] for k in ("original_raw_PID_and_config_equal", "surface_coefficients_equal", "drag_coefficients_equal", "half_PN_loft_S110_equal")) or checks["original_clock_max_abs_tick"] > 1e-8:
            raise AssertionError(checks)
        reports[cid] = {"metrics_separate_units": metrics, "summary": result["summary"], "full_result_path": str(path), "response_extrema": extrema,
            "events": drag.event_summary(x["ref"], prediction, result, x["run"]["summary"]), "checks": checks,
            "controller_checks": dec.audit.fix.verify_controls(model, result), "force_torque_record_closure": dec.audit.inverse.saved_closure(model, result)}
        # Evaluate the old force on the candidate's saved states only. This is
        # an algebraic frozen-law check, not a second baseline integration.
        old_model = dec.audit.fix.make_model(x["config"], coefficients); old_model.initialize(dataclasses.replace(x["initial"]))
        maxima = {"lift_norm_relative": 0., "drag_N": 0., "thrust_N": 0., "angular_law_rad_s2": 0.}
        for s in result["samples"]:
            state = dec.audit.fix.finite.sample_state(s); held = tuple(s["evaluated_old_held_rad"])
            a = old_model.evaluate(state, s["time_s"], held); b = model.evaluate(state, s["time_s"], held)
            for key, value in {"lift_norm_relative": abs(norm(a.wind_lift_force) - norm(b.wind_lift_force)) / max(norm(a.wind_lift_force), 1.),
                    "drag_N": norm(sub(a.wind_drag_force, b.wind_drag_force)), "thrust_N": abs(a.thrust - b.thrust),
                    "angular_law_rad_s2": max(abs(u - v) for u, v in zip(a.angular_acceleration, b.angular_acceleration))}.items():
                maxima[key] = max(maxima[key], value)
        if max(maxima.values()) > 1e-12: raise AssertionError(maxima)
        reports[cid]["frozen_scalar_law_checks_on_candidate_states"] = maxima
        fields.save_arrays(arrays, cid + "__reference", observed); fields.save_arrays(arrays, cid + "__baseline", baseline)
        fields.save_arrays(arrays, cid + "__candidate", prediction); plot_free(cid, observed, baseline, prediction)
        print(json.dumps({"free_case": cid, "summary": result["summary"]}), flush=True)
    np.savez_compressed(OUT / "free_trajectories.npz", **arrays)
    return reports, {"new_integrations": new, "recovered_without_reintegration": recovered, "unique_Jari_free_cases": 3,
        "baseline_integrations": 0, "all_free_arrays_finite": all(np.isfinite(v).all() for v in arrays.values())}


def main():
    before = snapshot(); inputs = dec.audit.inverse.source_inputs()
    inverse = np.load(dec.audit.inverse.OUTPUT / "inverse_inputs.npz")
    world_cache = np.load(ROOT / "outputs/jari_offaxis/feedback_interfaces/axial_completed_interfaces.npz")
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    OUT.mkdir(parents=True, exist_ok=True); arrays = {}; reports = {}; gate = {}
    for cid in CASES:
        a, r = conditional_case(cid, inputs[cid], inverse, world_cache, coefficients)
        arrays.update({cid + "__" + key: value for key, value in a.items()}); reports[cid] = r; plot_conditional(cid, a)
    for cid in CASES[1:]:
        methods = [reports[cid]["methods"][str(width) + "_point"]["full_0p5_3p5"]["axes"] for width in (3, 5)]
        pitch = all(m["pitch"]["candidate_total_feedback_residual_G"]["RMS"] <= .75 * m["pitch"]["baseline_total_feedback_residual_G"]["RMS"] for m in methods)
        yaw = all(m["yaw"]["candidate_total_feedback_residual_G"]["RMS"] <= 1.10 * m["yaw"]["baseline_total_feedback_residual_G"]["RMS"] for m in methods)
        gate[cid] = {"pitch_both_derivatives_RMS_reduced_at_least25percent": pitch, "yaw_both_derivatives_RMS_no_more_than10percent_worse": yaw, "passes": pitch and yaw}
    gate[CASES[0]] = {"near_axis_feedback_roundoff_unchanged": bool(np.max(abs(arrays[CASES[0] + "__candidate_feedback_G"] - arrays[CASES[0] + "__baseline_feedback_G"])) < 1e-12)}
    free_reports = {}; free_verification = {"new_integrations": 0, "unique_Jari_free_cases": 0, "baseline_integrations": 0, "all_free_arrays_finite": True}
    if all(gate[cid]["passes"] for cid in CASES[1:]) and gate[CASES[0]]["near_axis_feedback_roundoff_unchanged"]:
        free_reports, free_verification = run_jari_free(inputs, coefficients)
    verification = {"all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()), "protected_size_mtime_unchanged": before == snapshot(),
        "case_checks": {cid: r["checks"] for cid, r in reports.items()}, "free_runs": free_verification, "fits_scans_new_coefficients": 0,
        "raw_PID_originalControlFixGame": True, "controller_hz": 48, "plant_substeps": 4, "source_data_in_runtime": False,
        "new_geometry_rule_evidence_grade": "D", "frozen_lift_magnitude_drag_thrust_angularlaw_preserved": True}
    if not verification["all_arrays_finite"] or not verification["protected_size_mtime_unchanged"] or not free_verification["all_free_arrays_finite"] or any(v > (1e-8 if key.endswith("Nm") else 1e-12) for r in reports.values() for key, v in r["checks"].items()):
        raise AssertionError(verification)
    np.savez_compressed(OUT / "conditional.npz", **arrays)
    dec.audit.slew.write_json(OUT / "results.json", {"task": "Game-only one norm-preserving lift-direction experiment", "cases": reports,
        "conditional_gate": gate, "free_cases": free_reports, "verification": verification, "COM_zero_rate_geometric_projection": COM_GEOMETRY_FORMULA,
        "candidate_definition": "Local flow unit proportional to bodyForward-tan(localPitch)*bodyUp-tan(localYaw)*bodyRight; bodyForward projected perpendicular to this local flow, reprojected perpendicular to COM windForward, normalized, multiplied by the original current finite-lift norm.",
        "conditions_and_limits": [
            "Current R(hypot(local)) and all five surface coefficients remain fixed. This direction-only change alters the vector force formula, even though its scalar lift norm is exactly preserved.",
            "Same distFromCmToStab station and original angular law remain; lever moment and same-station reference couple are recomputed for algebraic torque closure. No extra surface or fitted component is introduced.",
            "No source state/vector is read by the candidate geometry. Equivalent local flow uses original current local_incidence, including the existing arm*rate/speed terms. Its declared forward branch requires each local axis angle below90deg; no fitted regularization/fallback.",
            "Old2.57 body-normal/body-drag mappings and body_linear incidence force project a different amplitude. They did not test this frozen2.59 norm-preserving finite-force direction; no old experiment is rerun.",
            "Source aerodynamic normal is independently derivative-derived total specific force minus exported thrust/body direction, projected into the same current wind basis. It is not an asserted native lift vector. World-vector interpolation may leave small source drag-direction interpolation residuals.",
            "Source3/5 are cached noncausal24Hz derivatives, interpolated to48Hz. They enter diagnostics only; agreement supports a conditional mechanism, not native game-source recovery.",
            "Fixed conditional gate:30/60 pitch total-feedback RMS versus each source3/5 at least25percent lower, yaw RMS no morethan10percent worse,0 near-axis roundoff unchanged. Passing permits exactly3 Jari actualfree runs with originalControlFixGame. No independent gain or combinedscore is introduced."]})
    print(json.dumps({"conditional_gate": gate, "verification": verification}), flush=True)


def native_plot(cid, observed, baseline, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    t = observed["t"]
    support = (t >= max(baseline["t"][0], candidate["t"][0])) & (t <= min(baseline["t"][-1], candidate["t"][-1]))
    for row, (key, unit) in enumerate((("speed_kmh", "Speed (km/h)"), ("alpha_deg", "Pitch AoA (deg)"), ("load_g", "Trajectory normal specific load (G)"))):
        for label, p, color in (("Cached original ControlFixGame", baseline, "#78337b"), ("Geometric direction; original controller", candidate, "#ce772e")):
            axes[row, 0].plot(p["t"], p[key], color=color, label=label)
        if key in observed:
            axes[row, 0].scatter(t, observed[key], s=10, facecolors="none", edgecolors="#177277", lw=.7, label="Original native observations", zorder=3)
        else:
            axes[row, 0].text(.03, .95, "No independent observed load in this payload", transform=axes[row, 0].transAxes, va="top", fontsize=9)
        delta = np.interp(t[support], candidate["t"], candidate[key])-np.interp(t[support], baseline["t"], baseline[key])
        axes[row, 1].plot(t[support], delta, color="#ce772e", label="Candidate − cached baseline, same native clock")
        axes[row, 0].set_ylabel(unit); axes[row, 1].set_ylabel("Prediction difference (" + ("km/h" if row == 0 else "deg" if row == 1 else "G") + ")")
    for ax in axes.flat:
        ax.set_xlim(0., float(t[-1])); ax.axhline(0., color="#aaa", lw=.6)
        for edge in (.7, 2.3, 3.5): ax.axvline(edge, color="#aaa", lw=.6)
        ax.grid(alpha=.16); ax.legend(fontsize=8, loc="best")
    for ax in axes[-1]: ax.set_xlabel("Original native age (s)")
    fig.suptitle(cid + ": actual free integration from original creation; fixed scalar plant and original ControlFixGame\nOnly lift direction changes; no missile observations enter runtime after creation", fontsize=11)
    fig.tight_layout(rect=(0., 0., 1., .94)); fig.savefig(OUT / (cid + "_overlay.png"), dpi=160); plt.close(fig)


def native_saved_checks(model, result, cfg, initial, target_at, coefficients):
    """One algebraic pass on candidate states, without another forward run."""
    baseline = dec.audit.fix.make_model(cfg, coefficients)
    baseline.initialize(dataclasses.replace(initial))
    bytime = {s["time_s"]: s for s in result["samples"]}
    maxima = {"lift_norm_relative": 0., "drag_N": 0., "thrust_N": 0., "angular_law_rad_s2": 0.,
        "lift_COMnormal_relative": 0., "pure_planar_limit_lift_relative": 0., "outer_request_same_state_G": 0.}
    geometry = {"max_abs_local_yaw_deg": 0., "max_lift_direction_change_deg": 0., "max_abs_local_minus_COM_pitch_deg": 0.}
    planar_body = BodyAxes((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    for s in result["samples"]:
        state = dec.audit.fix.finite.sample_state(s); held = tuple(s["evaluated_old_held_rad"])
        a = baseline.evaluate(state, s["time_s"], held); b = model.evaluate(state, s["time_s"], held)
        com, lp = a.alpha_pitch, a.local_incidence[0]
        planar_wind = BodyAxes((math.cos(com), -math.sin(com), 0.), (math.sin(com), math.cos(com), 0.), (0., 0., 1.))
        old_planar = scale(planar_wind.up, math.copysign(norm(a.wind_lift_force), lp)) if lp != 0. else (0., 0., 0.)
        planar = geometric_lift(planar_body, planar_wind, (lp, 0.), old_planar)
        errors = {"lift_norm_relative": abs(norm(a.wind_lift_force)-norm(b.wind_lift_force))/max(norm(a.wind_lift_force), 1.),
            "drag_N": norm(sub(a.wind_drag_force, b.wind_drag_force)), "thrust_N": abs(a.thrust-b.thrust),
            "angular_law_rad_s2": max(abs(u-v) for u, v in zip(a.angular_acceleration, b.angular_acceleration)),
            "lift_COMnormal_relative": abs(dot(b.wind_lift_force, b.wind_basis.forward))/max(norm(b.wind_lift_force), 1.),
            "pure_planar_limit_lift_relative": norm(sub(planar, old_planar))/max(norm(old_planar), 1.)}
        for key, value in errors.items(): maxima[key] = max(maxima[key], value)
        geometry["max_abs_local_yaw_deg"] = max(geometry["max_abs_local_yaw_deg"], math.degrees(abs(a.local_incidence[1])))
        geometry["max_abs_local_minus_COM_pitch_deg"] = max(geometry["max_abs_local_minus_COM_pitch_deg"], math.degrees(abs(lp-com)))
        na, nb = norm(a.wind_lift_force), norm(b.wind_lift_force)
        if na > 1e-12 and nb > 1e-12:
            # atan2 is stable for the small angles relevant to native near-axis cases.
            va, vb = np.asarray(a.wind_lift_force)/na, np.asarray(b.wind_lift_force)/nb
            angle = math.degrees(math.atan2(float(np.linalg.norm(np.cross(va, vb))), float(np.dot(va, vb))))
            geometry["max_lift_direction_change_deg"] = max(geometry["max_lift_direction_change_deg"], angle)
    for tick in result["ticks"]:
        s = bytime[tick["time_s"]]; state = dec.audit.fix.finite.sample_state(s)
        old = baseline.evaluate(state, tick["time_s"], tuple(s["evaluated_old_held_rad"]))
        request, details = baseline.request(state, target_at(tick["time_s"]), tick["time_s"], old)
        # The S110 state is replayed from creation, on candidate states, exactly
        # as in the original outer controller. No source missile state is read.
        maxima["outer_request_same_state_G"] = max(maxima["outer_request_same_state_G"], max(abs(a-b) for a, b in zip(request, tick["request_g"])))
        before = (baseline.slew_horizontal_g, baseline.slew_last_tick, len(baseline.slew_commit_history))
        repeated, again = baseline.request(state, target_at(tick["time_s"]), tick["time_s"], old)
        if repeated != request or again != details or before != (baseline.slew_horizontal_g, baseline.slew_last_tick, len(baseline.slew_commit_history)):
            raise AssertionError("Original outer request lost same-tick idempotence")
    if max(maxima.values()) > 1e-12: raise AssertionError(maxima)
    return {"algebraic_maxima": maxima, "actual_3D_geometry": geometry,
        "pure_planar_limit_definition": "Existing candidate helper at the saved COM/local pitch angles with yaw and bank set to zero algebraically; not another trajectory.",
        "same_state_request_replay": "Original S110 history on candidate saved states; unchanged request mapping, not equality between two diverging free trajectories.",
        "saved_force_torque_closure": dec.audit.inverse.saved_closure(model, result),
        "original_control_limits": dec.audit.fix.verify_controls(model, result)}


def native_main():
    """Only the three actual current native caches, no JARI rerun or baseline run."""
    before = snapshot(native=True)
    old_outputs = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in OUT.rglob("*") if p.is_file() and not p.relative_to(OUT).parts[0].startswith("native_")}
    harness = dec.audit.fix.finite.harness; fields = dec.audit.fix.finite.fields; drag = dec.audit.drag
    cases = {c["id"]: c for c in harness.load_cases(dec.audit.fix.finite.comparison.SNAPSHOT)}
    coefficients = drag.DragCoefficients(**json.loads((drag.OUTPUT / "fit.json").read_text())["coefficients"])
    cached = np.load(dec.audit.fix.OUTPUT / "trajectories.npz")
    arrays = {}; reports = {}; new = 0; recovered = 0
    windows = (*dec.WINDOWS, *harness.WINDOWS)
    for cid in ("native_H110", "native_H27.5", "native_H55native"):
        case = cases[cid.removeprefix("native_")]
        cfg = copy.deepcopy(case["profile"]["_model_config"]); initial = dataclasses.replace(case["_creation_state"])
        qx, qy, qz, qw = case["creation"]["u16_1"]; initial.orientation_quaternion = (qw, qx, qy, qz)
        target_at = case["_target_model"].state_at; end = case["scenario"]["max_simulation_time_s"]
        baseline = drag.selected_arrays(cached, cid + "__control_fix__")
        cached_full_path = dec.audit.fix.OUTPUT / "full_runs" / (cid + ".json")
        cached_full = json.loads(cached_full_path.read_text())
        reproduced = harness.sample_series(cached_full["samples"], case)
        baseline_exact = all(np.array_equal(value, baseline[key]) for key, value in reproduced.items())
        if not baseline_exact: raise AssertionError("Cached native baseline receipt does not recover its arrays: " + cid)
        model = make_model(cfg, coefficients); path = OUT / (cid + "_run.json")
        if path.exists():
            result = json.loads(path.read_text()); model.initialize(dataclasses.replace(initial)); recovered += 1
        else:
            result = model.run(dataclasses.replace(initial), target_at, end_time_s=end); new += 1
            result["model"].update(lift_direction_rule="single-local-flow geometry reprojected into COM windnormal; current finite-lift norm unchanged",
                lift_direction_evidence_grade="D opt-in script-local hypothesis", native_game_formula_recovered=False, reference_data_in_runtime=False)
            dec.audit.slew.write_json(path, result, True)
        prediction = harness.sample_series(result["samples"], case); observed = case["observed"]
        metrics = drag.pair_metrics(observed, baseline, prediction, ("speed_kmh", "alpha_deg", "pitch_deg", "omega_rad_s", "gamma_deg", "y_m"), windows)
        for w in metrics.values():
            for key, metric in w["metrics"].items(): metric["units"] = harness.UNITS[key]
        support = (observed["t"] >= max(baseline["t"][0], prediction["t"][0])) & (observed["t"] <= min(baseline["t"][-1], prediction["t"][-1]))
        load = {}
        for name, lo, hi in windows:
            mask = support & (observed["t"] >= lo-1e-9) & (observed["t"] <= hi+1e-9)
            ot = observed["t"][mask]
            load[name] = {"independent_observed_G_available": False, "same_clock_prediction_delta_G": drag.stats(
                np.interp(ot, prediction["t"], prediction["load_g"])-np.interp(ot, baseline["t"], baseline["load_g"]))}
        times = np.asarray([tick["time_s"] for tick in result["ticks"]])
        frozen = {"all_raw_PID_and_config_equal": model.config == case["profile"]["_model_config"],
            "five_surface_coefficients_equal": model.coefficients == SurfaceCoefficients(), "drag_coefficients_equal": model.drag_coefficients == coefficients,
            "half_PN_loft_S110_equal": model.relative_blend == .5 and model.request_slew_g_s == 110. and [model.amplitude_coefficient, model.transition_coefficient] == [.75, 1.3],
            "controller_hz": model.controller_hz, "plant_substeps": model.plant_substeps, "same_station_arm_m": model.arm,
            "original_clock_max_abs_tick": float(np.max(abs(times*48-np.arange(len(times))))), "cached_baseline_arrays_exact": baseline_exact}
        if not all(frozen[k] for k in ("all_raw_PID_and_config_equal", "five_surface_coefficients_equal", "drag_coefficients_equal", "half_PN_loft_S110_equal", "cached_baseline_arrays_exact")) or frozen["original_clock_max_abs_tick"] > 1e-8 or model.controller_hz != 48 or model.plant_substeps != 4:
            raise AssertionError(frozen)
        reports[cid] = {"harness_id": case["id"], "dataset": case["dataset"], "version": case["version"], "source_file": case["source_file"],
            "source_count": case["source_count"], "scenario": case["scenario"], "creation": case["creation"], "initial_state_used": dataclasses.asdict(initial),
            "creation_quaternion_full_roll_preserved": True, "source_parameter_equality": case["source_parameter_equality"],
            "native_custom_fins_lat_accel": case["native_custom_fins_lat_accel"], "raw_loft": case["raw_loft"], "clock_diagnostics": case["clock_diagnostics"],
            "alpha_semantics": "Pitch wind incidence in degrees, matching this native harness; not total vector AoA.",
            "summary": result["summary"], "cached_baseline_summary": cached_full["summary"], "full_result_path": str(path),
            "cached_baseline_full_result_path": str(cached_full_path), "separate_metrics": metrics, "load_prediction_change_windows": load,
            "existing_near_axis_acceptance": fields.native_pair(case, baseline, prediction), "events": drag.near_events(observed, baseline, prediction),
            "extrema": {name: {key: harness.extremum(p["t"], p[key], "max", 0., 3.5) for key in ("alpha_deg", "omega_rad_s", "pitch_deg")}
                for name, p in (("original_observed", observed), ("cached_baseline", baseline), ("candidate", prediction))},
            "frozen_checks": frozen, "saved_checks": native_saved_checks(model, result, cfg, initial, target_at, coefficients)}
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction)): fields.save_arrays(arrays, cid + "__" + label, p)
        native_plot(cid, observed, baseline, prediction)
        print(json.dumps({"native_free_case": cid, "near_axis_gate": reports[cid]["existing_near_axis_acceptance"]["full_gate_passed"], "summary": result["summary"]}), flush=True)
    verification = {"new_native_integrations": new, "recovered_without_reintegration": recovered, "unique_native_free_cases": 3,
        "baseline_integrations": 0, "JARI_integrations": 0, "all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()),
        "protected_size_mtime_unchanged": before == snapshot(native=True),
        "previous_lift_direction_outputs_size_mtime_unchanged": all((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == sig for p, sig in old_outputs.items()),
        "other_authorized_worker_exclusions": ["scripts/check_r771_envelope_control.py", "outputs/jari_offaxis/envelope_control/"],
        "source_missile_data_after_creation_in_runtime": False, "fits_scans_new_coefficients": 0,
        "original_ControlFixGame_and_48Hz_four_substeps": True}
    if not all(verification[k] for k in ("all_arrays_finite", "protected_size_mtime_unchanged", "previous_lift_direction_outputs_size_mtime_unchanged")):
        raise AssertionError(verification)
    np.savez_compressed(OUT / "native_trajectories.npz", **arrays)
    dec.audit.slew.write_json(OUT / "native_results.json", {"task": "Native near-axis validation of the already tested lift-direction candidate, original ControlFixGame only",
        "cases": reports, "verification": verification, "all_existing_near_axis_gates_passed": all(r["existing_near_axis_acceptance"]["full_gate_passed"] for r in reports.values()),
        "limits": ["Only H110/H27.5/H55native have the current accepted ControlFixGame caches. These actual IDs and original creation/target metadata define the three tests.",
            "No independent native load observation exists; load comparisons measure preservation of cached predictions, not absolute game agreement.",
            "Native b5 ties and target-clock rules are retained; replay-decoded pitch/rate channels remain candidate observations.",
            "Native creation roll and yaw are small but nonzero. Pure planar invariance is an algebraic limit, not asserted exact equality of these 3D free trajectories.",
            "Lift direction is a D-grade opt-in geometry hypothesis; exact lift norm, drag, thrust and angular law preservation does not make the vector force rule identical to the old rule or recovered native code."]})
    print(json.dumps({"native_verification": verification, "all_existing_near_axis_gates_passed": all(r["existing_near_axis_acceptance"]["full_gate_passed"] for r in reports.values())}), flush=True)


if __name__ == "__main__":
    native_main() if "--native" in sys.argv else main()
