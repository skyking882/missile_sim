#!/usr/bin/env python3
"""Bounded opt-in game capture hypotheses on the unchanged free inner model.

This experiment concerns War Thunder units, not real weapon guidance. Only
creation state and the externally specified game target drive each trajectory.
The old screenshot setup is approximate; its points are calibration targets,
not inputs to the plant. No core, production setting, or frozen fit is edited.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.aerodynamics import body_axes_from_quaternion, quaternion_from_pitch_yaw
from aim120_model.dynamics import SimState
from aim120_model.guidance import solve_pip_time_to_go_s
from aim120_model.math3d import add, clamp, clamp_norm, dot, norm, normalize, scale, sub
from aim120_model.public_api import _case_from_scenario, validate_scenario
from aim120_model.single_surface_game import SingleSurfaceGame
from aim120_model.target import TargetModel, TargetState
from aim120_model.units import G0_MPS2
import compare_unified_game_replays as harness

OUT = ROOT / "outputs/unified_game_replays/free_capture"
SNAPSHOT = ROOT / "outputs/unified_game_replays/corrected_input/runtime_profiles.json"
POLICIES = ("feasible_pn", "feasible_point", "geometric_point")
RATIOS = (.6, .8, 1.0, 1.1)
THRESHOLDS_DEG = (5., 10., 15.)


class CaptureGame(SingleSurfaceGame):
    """A game-only outer-request experiment; unchanged raw inner limits/PID."""

    def __init__(self, *args, policy="feasible_pn", alpha_ratio=.8, threshold_deg=10., **kwargs):
        super().__init__(*args, **kwargs)
        if policy not in (*POLICIES, "baseline"):
            raise ValueError(policy)
        self.policy = policy
        self.alpha_ratio = alpha_ratio
        self.threshold = math.radians(threshold_deg)

    def request(self, state, target, time_s, evaluation):
        original, details = super().request(state, target, time_s, evaluation)
        if self.policy == "baseline":
            return original, details
        rel = sub(target.position, state.position)
        speed = norm(state.velocity)
        point_time = solve_pip_time_to_go_s(rel, target.velocity, speed)
        point = add(rel, scale(target.velocity, point_time))
        tangent = evaluation.wind_basis.forward
        direction = normalize(point, fallback=tangent)
        cosine = clamp(dot(direction, tangent), -1., 1.)
        epsilon = math.acos(cosine)
        weight = clamp(epsilon / self.threshold - 1., 0., 1.)
        # Exact PN/loft below the geometric threshold. No version/case trigger.
        if weight == 0.:
            return original, {**details, "capture_weight": 0.,
                "capture_error_deg": math.degrees(epsilon), "capture_envelope_g": 0.}
        transverse = sub(direction, scale(tangent, cosine))
        point_normal = normalize(transverse, fallback=evaluation.wind_basis.up)
        horizontal = math.radians(self.config["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
        vertical = math.radians(self.config["aerodynamics"]["vertical_fin_aoa_limit_deg"])
        # Hypothesis, not a source-defined AoA cap: use the existing game fin
        # angular limit as the reference for a candidate attainable surface load.
        alpha_reference = self.alpha_ratio * min(horizontal, vertical)
        alpha_reference = min(alpha_reference, math.pi / 3)
        lift = evaluation.qbar * self.area * self.coefficients.ca * alpha_reference
        if self.force_basis == "body_linear":
            lift *= math.cos(alpha_reference)
        envelope = (lift + evaluation.thrust * math.sin(alpha_reference)) / evaluation.mass
        maximum = float(self.config["guidance"]["maximum_lateral_acceleration_g"]) * self.gravity
        envelope = min(maximum, max(0., envelope))
        pn = details["pn_mps2"]
        if self.policy == "feasible_pn":
            alternative = clamp_norm(pn, envelope)
        elif self.policy == "feasible_point":
            alternative = scale(point_normal, envelope)
        else:
            # Fixed geometric reference, no additional tuned gain or time.
            geometric = 2 * speed * speed * math.sin(epsilon) / max(norm(point), 1.)
            alternative = scale(point_normal, min(envelope, geometric))
        turning = add(scale(pn, 1. - weight), scale(alternative, weight))
        commanded = add(turning, (0., details["loft_g"] * self.gravity, 0.))
        commanded = clamp_norm(sub(commanded, scale(tangent, dot(commanded, tangent))), maximum)
        required = clamp_norm(add(commanded, (0., self.gravity, 0.)), maximum)
        request = (dot(required, evaluation.wind_basis.up) / G0_MPS2,
                   dot(required, evaluation.wind_basis.right) / G0_MPS2)
        return request, {**details, "capture_weight": weight,
            "capture_error_deg": math.degrees(epsilon),
            "capture_envelope_g": envelope / G0_MPS2,
            "capture_alpha_reference_deg": math.degrees(alpha_reference),
            "capture_point_time_s": point_time}


def runner_for(policy, ratio, threshold, *, force="body_linear", substeps=4, horizon=None):
    def runner(case, profile):
        config = copy.deepcopy(profile["_model_config"])
        scenario = validate_scenario(case["scenario"])
        config["guidance"]["lofting_enabled"] = scenario["loft_enabled"]
        initial = _case_from_scenario(scenario)["initial_conditions"]
        if "_creation_state" in case:
            state = dataclasses.replace(case["_creation_state"])
            x, y, z, w = case["creation"]["u16_1"]
            state.orientation_quaternion = (w, x, y, z)
            target_at = case["_target_model"].state_at
        else:
            pitch, yaw = map(math.radians, (initial["launch_angle_deg"], initial["launch_yaw_deg"]))
            quaternion = quaternion_from_pitch_yaw(pitch, yaw)
            forward = body_axes_from_quaternion(quaternion).forward
            state = SimState((0., initial["launch_altitude_m"], 0.),
                tuple(x * initial["start_speed_kmh"] / 3.6 for x in forward),
                pitch, yaw, 0., 0., config["geometry"]["initial_mass_kg"],
                orientation_quaternion=quaternion)
            target_model = TargetModel(initial, config["atmosphere"]["gravity_mps2"])
            if target_model.turn_g:
                raise ValueError("No invented screenshot target maneuver")
            target0 = target_model.initial_state
            target_at = lambda t: TargetState(tuple(x + t * v for x, v in
                zip(target0.position, target0.velocity)), target0.velocity)
        model = CaptureGame(config, force_basis=force, plant_substeps=substeps,
            loft_cap_unit="rad_s", policy=policy, alpha_ratio=ratio, threshold_deg=threshold)
        result = model.run(state, target_at,
            end_time_s=horizon if horizon is not None else scenario["max_simulation_time_s"])
        result["model"].update(capture_policy=policy, alpha_ratio=ratio,
            geometry_threshold_deg=threshold, raw_PID_preserved=True)
        return result
    return runner


def score(case, result):
    prediction = harness.sample_series(result["samples"], case)
    observed = case["observed"]
    t = observed["t"]
    valid = (t >= .7) & (t <= 7.) & (t <= prediction["t"][-1])
    metrics = {}
    for channel in ("speed_kmh", "alpha_deg", "load_g"):
        modeled = np.interp(t[valid], prediction["t"], prediction[channel])
        error = modeled - observed[channel][valid]
        metrics[channel] = float(np.sqrt(np.mean(error**2)))
    # AoA shape and early peak matter more than scalar speed error.
    cost = metrics["alpha_deg"] / 3. + metrics["load_g"] / 3. + metrics["speed_kmh"] / 150.
    early = prediction["t"] <= 2.3
    ix = np.flatnonzero(early)[np.argmax(prediction["alpha_deg"][early])]
    metrics.update(alpha_peak_deg=float(prediction["alpha_deg"][ix]),
        alpha_peak_time_s=float(prediction["t"][ix]), cost=float(cost),
        represented_burn_rows=int(valid.sum()), summary=result["summary"])
    return metrics, prediction


def plot_history(case, baseline, selected, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="row")
    channels = ("speed_kmh", "alpha_deg", "load_g")
    labels = ("Speed (km/h)", "AoA (deg)", "Normal load (g)")
    for row, bound in enumerate((2.5, 7.1)):
        for ax, channel, label in zip(axes[row], channels, labels):
            for curve, name, color in ((baseline, "Unchanged free candidate", "#888888"),
                                       (selected, "Capture hypothesis", "#ba503a")):
                mask = curve["t"] <= bound
                ax.plot(curve["t"][mask], curve[channel][mask], label=name, color=color)
            mask = case["observed"]["t"] <= bound
            ax.scatter(case["observed"]["t"][mask], case["observed"][channel][mask],
                s=20, facecolors="white", edgecolors="#20384b", label="Game screenshots", zorder=3)
            ax.set_xlim(0., bound); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[1]: ax.set_xlabel("Missile flight time (s)")
    fig.suptitle("2.57 R-77-1 game unit: free capture experiment; raw PID and inner coefficients fixed")
    fig.tight_layout()
    fig.savefig(folder / "historical_capture.png", dpi=150)
    plt.close(fig)


def append_existing_diagnostics(document, cases):
    """Direct final comparisons, with no additional trajectory evaluations."""
    selected = np.load(OUT / "selected/trajectories.npz")
    original = np.load(ROOT / "outputs/unified_game_replays/single_surface_candidate/body_linear_rad_s/trajectories.npz")
    unchanged = {}
    for cid in ("L110", "L27.5", "L55"):
        keys = [key for key in selected.files if key.startswith(cid + "__model__")]
        unchanged[cid] = {"model_array_count": len(keys),
            "all_arrays_exactly_equal_to_baseline": all(np.array_equal(selected[key], original[key]) for key in keys)}
    document["noLoft_validation"] = unchanged
    old = next(case for case in cases if case["id"] == "r77-1")
    observations = old["observed"]
    predicted_t = selected["r77-1__model__t"]
    baseline_t = original["r77-1__model__t"]
    common = (observations["t"] >= max(predicted_t[0], baseline_t[0])) & (
        observations["t"] <= min(predicted_t[-1], baseline_t[-1]))
    common_metrics = {"source_rows": int(common.sum()),
        "support_end_s": float(min(predicted_t[-1], baseline_t[-1]))}
    for name, archive, grid in (("baseline", original, baseline_t), ("selected", selected, predicted_t)):
        common_metrics[name] = {channel: float(np.sqrt(np.mean((np.interp(
            observations["t"][common], grid, archive["r77-1__model__" + channel]) -
            observations[channel][common])**2))) for channel in ("speed_kmh", "alpha_deg", "load_g")}
    document["historical_same_rows_comparison"] = common_metrics
    timeline = []
    for instant in (.2, .4, .5, .9, 1.5, 1.9, 2.1, 4.9, 6.9):
        timeline.append({"time_s": instant,
            "predicted_alpha_deg": float(np.interp(instant, predicted_t, selected["r77-1__model__alpha_deg"])),
            "predicted_load_g": float(np.interp(instant, predicted_t, selected["r77-1__model__load_g"])),
            "observed_alpha_deg": float(np.interp(instant, observations["t"], observations["alpha_deg"])),
            "observed_load_g": float(np.interp(instant, observations["t"], observations["load_g"]))})
    document["shape_acceptance"] = {"accepted": False,
        "reason": "The lower burn-window RMSE does not reproduce the main peak time or initial response.",
        "observed_alpha_main_peak_deg": 23.7, "observed_main_peak_first_time_s": 1.5,
        "selected_alpha_peak_deg": document["selected"]["metrics"]["alpha_peak_deg"],
        "selected_alpha_peak_time_s": document["selected"]["metrics"]["alpha_peak_time_s"],
        "initial_response_gap": "Observed alpha is 11.3deg at 0.2s; the current placement of raw flightTimeGainTable [.3,0]/[.31,1], with zero creation angular velocity, gives no guided control response before 0.291667s. Unknown creation state and gain semantics remain possible explanations.",
        "timeline": timeline,
        "selection_objective_limit": "Selection scores0.7..7s, while the0.2..0.6s initial response is assessed separately and fails."}
    # Conditional axial check is kept completely outside the forward model.
    from compare_rebuild_replays import axial_balance
    from aim120_model.aerodynamics import reference_area
    from aim120_model.atmosphere import StandardAtmosphere
    from aim120_model.propulsion import PiecewisePropulsion
    time_grid = observations["t"]
    obs = np.column_stack([observations[key] for key in ("speed_kmh", "alpha_deg", "load_g")])
    config = old["profile"]["_model_config"]
    conditional, impulses = axial_balance(config, old["scenario"], time_grid, obs)
    rho = StandardAtmosphere().sample(old["scenario"]["launch_altitude_m"]).density_kg_m3
    area = reference_area(config)
    propulsion = PiecewisePropulsion.from_config(config)
    retained = [obs[0, 0]]
    for index, (lo, hi) in enumerate(zip(time_grid[:-1], time_grid[1:])):
        count = math.ceil((hi - lo) / .005)
        dt = (hi - lo) / count
        midpoint = lo + (np.arange(count) + .5) * dt
        velocities = np.interp(midpoint, time_grid, obs[:, 0] / 3.6)
        alphas = np.radians(np.interp(midpoint, time_grid, obs[:, 1]))
        extra = sum(.5 * rho * v * v * area * 14.606915444766543 * a * math.sin(a) /
            propulsion.mass_at(float(t)) for t, v, a in zip(midpoint, velocities, alphas)) * dt * 3.6
        retained.append(retained[-1] + impulses[index][0] - impulses[index][1] - impulses[index][2] - extra)
    retained = np.asarray(retained)
    document["conditional_axial_counterevidence"] = {
        "input": "Measured speed/AoA and nominal altitude; diagnostic only, never forward inputs.",
        "existing_wind_drag": {"speed_rmse_kmh": float(np.sqrt(np.mean((conditional - obs[:, 0])**2))),
            "final_speed_error_kmh": float(conditional[-1] - obs[-1, 0])},
        "full_body_normal_tangent_added": {"speed_rmse_kmh": float(np.sqrt(np.mean((retained - obs[:, 0])**2))),
            "final_speed_error_kmh": float(retained[-1] - obs[-1, 0])},
        "interpretation": "The unchanged axial law remains deficient even at measured AoA. Simply adding the whole body-normal tangent overcorrects the final speed, so no force-basis repair is accepted.",
        "additional_coefficients_fitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=70)
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = harness.load_cases(SNAPSHOT)
    if args.audit_existing:
        document = json.loads((OUT / "results.json").read_text())
        append_existing_diagnostics(document, cases)
        (OUT / "results.json").write_text(json.dumps(document, indent=2, allow_nan=False))
        print(json.dumps({"noLoft_validation": document["noLoft_validation"],
            "shape_acceptance": document["shape_acceptance"],
            "conditional_axial_counterevidence": document["conditional_axial_counterevidence"]}, indent=2))
        return
    old = next(case for case in cases if case["id"] == "r77-1")
    baseline_result = runner_for("baseline", .8, 10., horizon=7.1)(old, old["profile"])
    baseline_metrics, baseline_curve = score(old, baseline_result)
    evaluations = 1
    scanned = []
    best = None
    for policy in POLICIES:
        for ratio in RATIOS:
            for threshold in THRESHOLDS_DEG:
                if evaluations >= args.budget - 17 or time.monotonic() - started > 570:
                    break
                result = runner_for(policy, ratio, threshold, horizon=7.1)(old, old["profile"])
                metrics, curve = score(old, result)
                item = {"policy": policy, "alpha_ratio": ratio,
                    "threshold_deg": threshold, "metrics": metrics}
                scanned.append(item); evaluations += 1
                if best is None or metrics["cost"] < best["metrics"]["cost"]:
                    best = item; best_curve = curve
    if best is None:
        raise RuntimeError("No candidate evaluation")
    runner = runner_for(best["policy"], best["alpha_ratio"], best["threshold_deg"])
    records, arrays = harness.run_suite(label="Bounded free game capture hypothesis",
        output_dir=OUT / "selected", profile_snapshot=SNAPSHOT, runner=runner)
    evaluations += len(cases)
    validation = []
    base_file = ROOT / "outputs/unified_game_replays/single_surface_candidate/body_linear_rad_s/results.json"
    base_records = {r["id"]: r for r in json.loads(base_file.read_text())["cases"]}
    for record in records:
        rid = record["id"]
        before = base_records[rid]
        validation.append({"id": rid, "split": "calibration" if rid == "r77-1" else "validation",
            "metrics": record.get("windows", {}).get("full", {}).get("metrics", {}),
            "baseline_metrics": before.get("windows", {}).get("full", {}).get("metrics", {}),
            "summary": record["model_summary"], "baseline_summary": before["model_summary"],
            "unrepresented_rows": record.get("unrepresented_rows"),
            "input_limitations": record.get("input_limitations", [])})
    # Necessary final checks: numerical refinement and observation independence.
    refinement = runner_for(best["policy"], best["alpha_ratio"], best["threshold_deg"],
        substeps=8, horizon=7.1)(old, old["profile"])
    _, refined_curve = score(old, refinement); evaluations += 1
    fine = {channel: float(np.max(np.abs(np.interp(best_curve["t"], refined_curve["t"],
        refined_curve[channel]) - best_curve[channel]))) for channel in
        ("speed_kmh", "alpha_deg", "load_g", "omega_rad_s")}
    poisoned = copy.deepcopy(old)
    poisoned["observed"] = {key: np.full_like(value, 987654.) for key, value in old["observed"].items()}
    poisoned["source_rows"] = []
    poisoned_result = runner_for(best["policy"], best["alpha_ratio"], best["threshold_deg"],
        horizon=7.1)(poisoned, poisoned["profile"]); evaluations += 1
    poison_same = poisoned_result == runner_for(best["policy"], best["alpha_ratio"],
        best["threshold_deg"], horizon=7.1)(old, old["profile"]); evaluations += 1
    plot_history(old, baseline_curve, best_curve, OUT)
    document = {"purpose": "War Thunder game reverse reproduction only",
        "predeclared_hypotheses": list(POLICIES), "force_basis": "body_linear",
        "loft_baseline": "unchanged simple pitch-error law, raw cap interpreted rad/s",
        "formulas": {"point": "r+target_velocity*positive_constant_speed_intercept_time",
            "weight": "clamp(epsilon/threshold-1,0,1)",
            "envelope": "(q*S*Ca*alpha_ref*cos(alpha_ref)+T*sin(alpha_ref))/mass; alpha_ref=ratio*min(raw fin limits)",
            "feasible_pn": "blend unchanged PN with PN norm-clamped to envelope",
            "feasible_point": "blend unchanged PN with envelope along point's wind-normal direction",
            "geometric_point": "blend unchanged PN with min(envelope,2*V^2*sin(epsilon)/point_distance) in point direction"},
        "classification": "calibration, not derived native engine semantics",
        "calibration_ranges": {"alpha_ratio": list(RATIOS), "threshold_deg": list(THRESHOLDS_DEG)},
        "calibration_window_s": [.7, 7.], "parameter_count": 2,
        "raw_inner_PID_and_fin_limits_preserved": True,
        "outer_envelope_is_an_additional_hypothesis_not_source_identified": True,
        "no_timed_request_or_state_reset": True, "missile_observation_input_after_creation": False,
        "observed_poison_forward_equality": poison_same,
        "baseline_metrics": baseline_metrics, "selected": best,
        "scanned": scanned, "validation": validation,
        "fixed_controller_hz": 48, "plant_substeps": 4,
        "selected_4_to_8_substep_max_difference": fine,
        "candidate_evaluations": evaluations, "wall_time_s": time.monotonic() - started,
        "production_changed": False, "accepted": False,
        "limitations": ["The old creation speed and target track are approximate; no clock shift is fitted.",
            "Raw fin angular limit is already imposed on PID output. Converting it to outer body-AoA is a separate D-level hypothesis.",
            "The two coefficients are shared across cases and versions; no case identifiers enter the policy.",
            "This experiment does not identify Jari maximum-AoA semantics or maximum-maneuver envelope."]}
    append_existing_diagnostics(document, cases)
    (OUT / "results.json").write_text(json.dumps(document, indent=2, allow_nan=False))
    print(json.dumps({"selected": best, "evaluations": evaluations, "refinement": fine,
        "observation_poison_equality": poison_same}, indent=2))


if __name__ == "__main__":
    main()
