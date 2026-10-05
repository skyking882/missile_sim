#!/usr/bin/env python3
"""Bounded War Thunder/Jari capture diagnosis; no new guidance or fitting.

Predeclared distinctions: onset-window error, command-duration/shape error,
body-versus-velocity turning error, and conditional frozen-inner fin feasibility.
Only the already-defined half 30/60 free trajectories are reproduced to recover
missing attitude/held/tick records. Jari states are used in a separate inverse
diagnostic and never enter those free predictions after creation.
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
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")

import numpy as np

from aim120_model.aerodynamics import body_axes_from_quaternion, quaternion_multiply
from aim120_model.dynamics import SimState
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients
import compare_r771_jari_offaxis as prior

OUT = ROOT / "outputs/jari_offaxis/capture_gap"
HALF = "half_horizontal_PN_and_loft_gate"


def derivative(values, t, width=3):
    """Local finite differences on original timestamps; not a state fit."""
    values = np.asarray(values)
    if width == 3:
        return np.gradient(values, t, axis=0, edge_order=2)
    answer = np.empty_like(values, dtype=float)
    for i in range(len(t)):
        start = min(max(i-width//2, 0), len(t)-width)
        indices = np.arange(start, start+width)
        delta = t[indices]-t[i]
        matrix = np.stack([delta**j for j in range(width)])
        rhs = np.zeros(width); rhs[1] = 1.
        weights = np.linalg.solve(matrix, rhs)
        answer[i] = weights @ values[indices]
    return answer


def continuous_quaternions(quaternions):
    q = np.asarray(quaternions, dtype=float).copy()
    q /= np.linalg.norm(q, axis=1)[:, None]
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0.:
            q[i] *= -1.
    return q


def body_rates(q, t, width):
    qdot = derivative(q, t, width)
    products = np.asarray([quaternion_multiply((a[0], -a[1], -a[2], -a[3]), b)
                           for a, b in zip(q, qdot)])
    # Local basis (forward, up, right): model pitch=omega_z, yaw=-omega_y.
    return 2.*products[:, 1:]


def geometry(position, velocity, target0, target_velocity):
    speed = np.linalg.norm(velocity, axis=1)
    return {"speed_mps": speed,
        "velocity_heading_deg": np.degrees(np.unwrap(np.arctan2(velocity[:, 2], velocity[:, 0]))),
        "velocity_gamma_deg": np.degrees(np.arctan2(velocity[:, 1], np.hypot(velocity[:, 0], velocity[:, 2])))}


def trajectory_fields(t, position, velocity, target0, target_velocity, q=None):
    answer = {"t": t, "position_xyz_m": position, "velocity_xyz_mps": velocity,
              **geometry(position, velocity, target0, target_velocity)}
    rel = np.asarray(target0)[None, :]+t[:, None]*np.asarray(target_velocity)[None, :]-position
    rv = np.asarray(target_velocity)[None, :]-velocity
    bearing = np.unwrap(np.arctan2(rel[:, 2], rel[:, 0]))
    heading = np.radians(answer["velocity_heading_deg"])
    horizontal_range2 = rel[:, 0]**2+rel[:, 2]**2
    answer.update(los_bearing_deg=np.degrees(bearing),
        los_rate_xyz_rad_s=np.cross(rel, rv)/np.sum(rel**2, axis=1)[:, None],
        los_bearing_rate_rad_s=(rel[:, 0]*rv[:, 2]-rel[:, 2]*rv[:, 0])/horizontal_range2,
        horizontal_velocity_to_los_error_deg=np.degrees(np.arctan2(np.sin(bearing-heading), np.cos(bearing-heading))))
    if q is None:
        return answer
    axes = [body_axes_from_quaternion(tuple(a)) for a in q]
    forward = np.asarray([a.forward for a in axes])
    up = np.asarray([a.up for a in axes]); right = np.asarray([a.right for a in axes])
    tangent = velocity/answer["speed_mps"][:, None]
    pitch = np.arcsin(np.clip(forward[:, 1], -1., 1.))
    yaw = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
    nominal_up = np.column_stack((-np.sin(pitch)*np.cos(yaw), np.cos(pitch), -np.sin(pitch)*np.sin(yaw)))
    nominal_right = np.column_stack((-np.sin(yaw), np.zeros(len(t)), np.cos(yaw)))
    dotf = np.sum(tangent*forward, axis=1)
    alpha = np.arctan2(np.linalg.norm(np.cross(tangent, forward), axis=1), dotf)
    answer.update(quaternion_wxyz=q, body_forward=forward, body_up=up, body_right=right,
        body_pitch_deg=np.degrees(pitch), body_yaw_deg=np.degrees(yaw),
        body_bank_deg=np.degrees(np.arctan2(np.sum(up*nominal_right, axis=1), np.sum(up*nominal_up, axis=1))),
        alpha_geometric_deg=np.degrees(alpha),
        alpha_pitch_rad=np.arctan2(-np.sum(tangent*up, axis=1), dotf),
        alpha_yaw_rad=np.arctan2(-np.sum(tangent*right, axis=1), dotf))
    # Total-incidence derivative splits into forward-axis and velocity turning.
    # Undefined near zero incidence is represented by an explicit valid mask.
    mask = alpha > math.radians(1.)
    fdot = derivative(forward, t)
    tdot = derivative(tangent, t)
    fdot -= np.sum(fdot*forward, axis=1)[:, None]*forward
    tdot -= np.sum(tdot*tangent, axis=1)[:, None]*tangent
    divisor = np.where(mask, np.sin(alpha), 1.)
    body_contribution = -np.sum(fdot*tangent, axis=1)/divisor
    velocity_contribution = -np.sum(forward*tdot, axis=1)/divisor
    answer.update(alpha_derivative_valid=mask,
        alpha_dot_body_deg_s=np.where(mask, np.degrees(body_contribution), 0.),
        alpha_dot_velocity_deg_s=np.where(mask, np.degrees(velocity_contribution), 0.))
    return answer


def onset_interval(t, values, threshold):
    values = np.asarray(values)
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)
    indices = np.flatnonzero(np.abs(values) > threshold)
    if not len(indices):
        return None
    i = int(indices[0])
    return {"last_below_s": float(t[max(0, i-1)]), "first_above_s": float(t[i]), "threshold": threshold}


def intervals(t, mask):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []
    return [{"first_sample_s": float(t[g[0]]), "last_sample_s": float(t[g[-1]]),
             "sample_count": len(g), "sample_span_s": float(t[g[-1]]-t[g[0]])} for g in groups]


def conditional_inverse(run, reference, fields, config):
    """Observed-state inverse only; frozen model and no native-control mapping."""
    t, q = fields["t"], fields["quaternion_wxyz"]
    state, _ = prior.free_inputs(run, reference, config)
    model = SingleSurfaceGame(config, force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
    model.initialize(state)
    c = model.coefficients
    methods = {}
    for width in (3, 5):
        omega = body_rates(q, t, width)
        rates = np.column_stack((omega[:, 2], -omega[:, 1]))
        accel = derivative(rates, t, width)
        active, ku, damping, local, required = [], [], [], [], []
        for i, time_s in enumerate(t):
            s = SimState(tuple(fields["position_xyz_m"][i]), tuple(fields["velocity_xyz_mps"][i]),
                math.radians(fields["body_pitch_deg"][i]), math.radians(fields["body_yaw_deg"][i]),
                *rates[i], config["geometry"]["initial_mass_kg"], orientation_quaternion=tuple(q[i]))
            evaluation = model.evaluate(s, float(time_s), (0., 0.))
            a = (evaluation.qbar/evaluation.mass)/model.q_over_mass0
            gain = c.control_gain*(1.+c.speed_slope*(fields["speed_mps"][i]/model.speed0-1.))
            d = a*model.speed0/fields["speed_mps"][i]*c.rate_damping
            active.append(a); ku.append(gain); damping.append(d); local.append(evaluation.local_incidence)
            required.append((accel[i]-np.asarray(evaluation.angular_acceleration))/gain)
        methods[str(width)] = {"omega_body_xyz_rad_s": omega, "rates_pitch_yaw_rad_s": rates,
            "rate_acceleration_pitch_yaw_rad_s2": accel, "Ku_rad_s2_per_rad": np.asarray(ku),
            "active": np.asarray(active), "damping_Kw_scaled": np.asarray(damping),
            "alpha_local_pitch_yaw_rad": np.asarray(local), "required_held_pitch_yaw_rad": np.asarray(required)}
    valid = (t >= t[2]) & (t <= min(3.5, t[-3]))
    controls = np.asarray([s["controls"] for s in run["samples"]])
    onset = onset_interval(t, controls, 1e-6)
    startup_halo = None
    if onset:
        dt = float(np.median(np.diff(t)))
        startup_halo = [onset["last_below_s"]-2.*dt, onset["first_above_s"]+2.*dt]
        valid &= (t < startup_halo[0]) | (t > startup_halo[1])
    # Three consecutive display intervals are required for a sustained signal.
    limits = np.radians([config["aerodynamics"]["horizontal_fin_aoa_limit_deg"],
                         config["aerodynamics"]["vertical_fin_aoa_limit_deg"]])
    a, b = methods["3"]["required_held_pitch_yaw_rad"], methods["5"]["required_held_pitch_yaw_rad"]
    robust_over = (np.abs(a) > limits) & (np.abs(b) > limits) & (np.sign(a)==np.sign(b))
    summary = {"formula": "u=(rate_dot+active*Ka*(alpha+arm*rate/V)+active*(V0/V)*Kw*rate)/Ku(V)",
        "conditional_on_frozen_inner_and_reference_state": True, "used_in_free_forward": False,
        "native_controls_units_or_channel_map_assumed": False,
        "fin_limit_pitch_yaw_rad": limits, "primary_interior_window_s": [float(t[2]), min(3.5, float(t[-3]))],
        "startup_stencil_halo_excluded_from_feasibility_s": startup_halo,
        "required_fin_ratio_peak_abs_3point_pitch_yaw": np.max(np.abs(a[valid])/limits, axis=0),
        "required_fin_ratio_peak_abs_5point_pitch_yaw": np.max(np.abs(b[valid])/limits, axis=0),
        "fin_3_vs_5_max_difference_rad_pitch_yaw": np.max(np.abs(a[valid]-b[valid]), axis=0),
        "roll_rate_3point_peak_abs_rad_s": float(np.max(np.abs(methods["3"]["omega_body_xyz_rad_s"][valid, 0]))),
        "robust_out_of_limit_intervals_pitch_yaw": [intervals(t, valid & robust_over[:, i]) for i in range(2)],
        "sustained_out_of_limit_pitch_yaw": [any(x["sample_span_s"] >= .125-1e-7 for x in intervals(t, valid & robust_over[:, i])) for i in range(2)],
        "derivative_limitation": "24Hz exported display samples; 3/5-point finite differences bound stencil sensitivity, not a statistical confidence interval. Endpoint and second-derivative transients are not native 48Hz states. Body roll is retained; yaw heading derivative is not substituted for body yaw rate."}
    return methods, valid, summary


def derive_findings(document, arrays):
    """Measured discriminants and bounded suggestions, with contrary evidence."""
    cases = {c["case_id"]: c for c in document["cases"]}
    rate_peaks = {}
    for deg in (30, 60):
        cid = f"jari_259_{deg}"
        peaks = {}
        for source in ("reference", "half"):
            t = arrays[cid+"__"+source+"__t"]
            rates = (arrays[cid+"__inverse3__rates_pitch_yaw_rad_s"][:, 1] if source == "reference"
                     else arrays[cid+"__half__exact_model_pitch_yaw_rates_rad_s"][:, 1])
            selected = np.flatnonzero(t<=3.5)
            ix = selected[np.argmax(rates[selected])]
            peaks[source] = {"peak_body_yaw_rate_deg_s": float(np.degrees(rates[ix])), "original_time_s": float(t[ix])}
        rate_peaks[cid] = peaks
    c60 = cases["jari_259_60"]
    at_half_peak = min(c60["matched_display_time_table"], key=lambda row: abs(row["reference_original_time_s"]-.625))
    part_body = at_half_peak["alpha_dot_body_deg_s"]
    part_vel = at_half_peak["alpha_dot_velocity_deg_s"]
    return {"measured_body_yaw_rate_peaks": rate_peaks,
        "60deg_at_half_peak_body_and_velocity_differences": {
            "reference_original_time_s": at_half_peak["reference_original_time_s"],
            "body_yaw_excess_deg": at_half_peak["body_yaw_deg"]["half"]-at_half_peak["body_yaw_deg"]["reference"],
            "velocity_heading_excess_deg": at_half_peak["velocity_heading_deg"]["half"]-at_half_peak["velocity_heading_deg"]["reference"],
            "body_incidence_rate_contribution_change_deg_s": part_body["half"]-part_body["reference"],
            "velocity_incidence_rate_contribution_change_deg_s": part_vel["half"]-part_vel["reference"],
            "finite_difference_caveat": "Display-grid derivative decomposition explains direction and scale; it does not locate an exact sub-sample zero derivative"},
        "supported_findings": [
            "The displayed onset interval only permits a small onset difference, not a 0.208s clock correction",
            "At t=.5s the 60deg half body heading is already about22deg ahead while half speed is lower than Jari: the initial attitude error is not caused solely by excess axial speed",
            "By t=.625s half body yaw rate has declined sharply while Jari is still turning rapidly; faster half velocity-direction rotation contributes but is smaller in the incidence-rate difference",
            "Jari30/60 share almost identical initial body turn and nearly equal body-yaw-rate peaks; current PN gives angle-dependent initial command/held amplitude and larger earlier body-rate peak at60",
            "Conditional required fins stay within local raw limits for both derivative stencils; there is no demonstrated frozen-inner feasibility failure, and no proof that its effective native control mapping is correct",
            "Exported control[0] stays at1 much longer at60 than30, but its units/channel mapping are not identified, and that plateau alone cannot identify a capture mode"],
        "contrary_evidence_and_nonidentifiability": [
            "Another worker reports missing same-state AoA-drag speed dependence; changing capture first could compensate an axial plant error",
            "Jari display samples do not provide every48Hz state/PID memory; normalized exportcontrol1 is not local held fin radians",
            "Only half30/60 body trajectories were recovered; baseline orientation is absent, so no baseline body-turn reconstruction is claimed",
            "Required fin inversion supplies one conditional input history, not a native fin mapping or a guidance identification"],
        "next_experiment_priority": "Reconcile documented same-state axial drag mapping, then reassess the fixed half trajectory before choosing a capture change; do not tune frozen inner/PID",
        "at_most_two_conditional_outer_suggestions": [
            {"name": "Shared bounded body-to-LOS angle capture with continuous geometric handover",
             "state": "beta=wrapped horizontal LOS bearing minus body heading; eta=min(previous_eta,beta^2/(beta^2+beta_c^2))",
             "horizontal_request": "(1-eta)*existing_horizontal_request + eta*A_c*tanh(beta/beta_c)",
             "shared_internal_coefficients_at_most": ["A_c positive", "beta_c positive"],
             "reason": "Large30/60 angles give a shared bounded initial command; larger body-to-LOS angle automatically lasts longer before geometric handover; no elapsed-time knots",
             "limits": "Hypothesis only, not run. Exact beta=0 keeps horizontal baseline unchanged; native near-axis quality would still require its existing acceptance check. Plateau ending cannot be equated with handover."},
            {"name": "Horizontal virtual target with bounded angular separation",
             "state": "virtual horizontal LOS bearing=body heading+beta_c*tanh(beta/beta_c); preserve actual horizontal range and target altitude",
             "horizontal_request": "Use a kinematically consistent derivative of that moving virtual point for horizontal guidance; preserve original vertical request, gravity/projection/caps",
             "shared_internal_coefficients_at_most": ["beta_c positive"],
             "reason": "Limits early geometric steering demand and brings the virtual point toward the real target as body angle closes; the original0deg plane is unchanged",
             "limits": "Hypothesis only, not run. Virtual-point motion must be differentiated consistently; retaining the real target velocity for a state-dependent point would give a false LOS-rate model."}],
        "preserved_requirements_for_any_later_candidate": "Original PID/frozen inner/.75,1.3 and all four raw loft fields; no new user fields or reference-state feedback; near-axis nonregression not asserted without a future authorized check"}


def plot_case(cid, reference, fields, half, baseline, controls, inverse, local_ticks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t = fields["t"]; h = half["t"]; keep = t <= 3.5; hk = h <= 3.5; bk = baseline["t"] <= 3.5
    ax = axes[0, 0]
    for values, color, label, style in ((fields["body_yaw_deg"], "#24566a", "Jari body yaw", "-"),
            (fields["velocity_heading_deg"], "#24566a", "Jari velocity heading", "--")):
        ax.plot(t[keep], values[keep], style, color=color, label=label)
    ax.plot(h[hk], half["body_yaw_deg"][hk], color="#a05080", label="half body yaw")
    ax.plot(h[hk], half["velocity_heading_deg"][hk], "--", color="#a05080", label="half velocity heading")
    ax.plot(baseline["t"][bk], baseline["velocity_heading_deg"][bk], ":", color="#66686b", label="baseline velocity heading")
    ax.set(ylabel="World heading (deg)")
    ax = axes[0, 1]
    ax.plot(t[keep], reference["alpha_deg"][keep], "o-", ms=2, color="#24566a", label="Jari AoA")
    ax.plot(h[hk], half["alpha_geometric_deg"][hk], color="#a05080", label="half AoA")
    ax.plot(baseline["t"][bk], baseline["alpha_deg"][bk], color="#66686b", label="baseline AoA")
    ax.set(ylabel="Unsigned incidence (deg)")
    ax = axes[0, 2]
    for source, time, data, color in (("Jari", t, fields, "#24566a"), ("half", h, half, "#a05080")):
        k = (time <= 1.5)&data["alpha_derivative_valid"]
        ax.plot(time[k], data["alpha_dot_body_deg_s"][k], color=color, label=source+" body contribution")
        ax.plot(time[k], data["alpha_dot_velocity_deg_s"][k], "--", color=color, label=source+" velocity contribution")
    ax.set(ylabel="Contribution to incidence rate (deg/s)")
    ax = axes[1, 0]
    a = np.degrees(inverse["3"]["rates_pitch_yaw_rad_s"][:, 1])
    b = np.degrees(inverse["5"]["rates_pitch_yaw_rad_s"][:, 1])
    ax.plot(t[keep], a[keep], color="#24566a", label="Jari body yaw rate, 3-point")
    ax.fill_between(t[keep], np.minimum(a, b)[keep], np.maximum(a, b)[keep], color="#24566a", alpha=.15, label="3/5 stencil span")
    ax.plot(h[hk], np.degrees(half["exact_model_pitch_yaw_rates_rad_s"][hk, 1]), color="#a05080", label="half body yaw rate")
    ax.set(ylabel="Body yaw rate (deg/s)")
    ax = axes[1, 1]
    limit = local_ticks["fin_limit_pitch_yaw_rad"][1]
    ax.plot(t[keep], controls[keep, 0], color="#24566a", label="Jari exported control[0], unit unknown")
    ax.plot(t[keep], inverse["3"]["required_held_pitch_yaw_rad"][keep, 1]/limit, "--", color="#398977", label="conditional required yaw fin / local limit")
    ax.plot(local_ticks["time_s"], local_ticks["held_fin_pitch_yaw_rad"][:, 1]/limit, color="#a05080", label="half held yaw fin / limit")
    ax.set(ylabel="Separately normalized diagnostics")
    ax = axes[1, 2]
    for label, data, color in (("Jari", fields, "#24566a"), ("half", half, "#a05080"), ("baseline", baseline, "#66686b")):
        k = data["t"] <= 3.5
        ax.plot(data["t"][k], data["horizontal_velocity_to_los_error_deg"][k], color=color, label=label)
    ax.set(ylabel="LOS bearing minus velocity heading (deg)")
    for ax in axes.flat:
        ax.set(xlabel="Original time (s)"); ax.grid(alpha=.18); ax.legend(fontsize=6.5)
    fig.suptitle(cid+": fixed half reproduction and observed-state conditional diagnosis; native control scale unknown", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(output / f"{cid}_capture_gap.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(prior.DEFAULT_MANIFEST.read_text())
    half_document = json.loads((prior.DEFAULT_OUTPUT / "half/results.json").read_text())
    half_cases = {c["case_id"]: c for c in half_document["cases"]}
    half_arrays = np.load(prior.DEFAULT_OUTPUT / "half/trajectories.npz")
    base_arrays = np.load(prior.DEFAULT_OUTPUT / "trajectories.npz")
    profiles = json.loads(prior.SNAPSHOT.read_text())
    read_only_paths = [prior.FROZEN, prior.SNAPSHOT, Path(prior.__file__),
        ROOT / "src/aim120_model/single_surface_game.py"]
    read_only_paths += [p for folder in (prior.DEFAULT_OUTPUT, prior.DEFAULT_OUTPUT / "ab", prior.DEFAULT_OUTPUT / "half") for p in folder.iterdir() if p.is_file()]
    read_only_paths += [Path(e["export_path"]) for e in manifest["references"]]
    before = {p: prior.sha256(p) for p in read_only_paths}
    arrays, records, reproduced = {}, [], {}
    for entry in manifest["references"]:
        cid = entry["id"]
        run = prior.split_runs(entry["export_path"])[0]
        reference, diagnostic = prior.parse_reference(run)
        profile, source = prior.profile_for(run, entry, profiles)
        config = copy.deepcopy(profile["_model_config"])
        target0, target_v = run["scenario"]["target_position"], run["scenario"]["target_velocity"]
        q = continuous_quaternions(reference["quaternion_xyzw"][:, [3, 0, 1, 2]])
        fields = trajectory_fields(reference["t"], reference["position_xyz_m"], reference["velocity_xyz_mps"], target0, target_v, q)
        controls = np.asarray([s["controls"] for s in run["samples"]])
        inverse, derivative_valid, conditional = conditional_inverse(run, reference, fields, config)
        record = {"case_id": cid, "source": source, "source_export_sha256": prior.sha256(entry["export_path"]),
            "reference_diagnostics": diagnostic, "conditional_inner_inverse": conditional,
            "source_control_first_nonzero_interval_s": onset_interval(fields["t"], controls, 1e-6),
            "source_body_heading_turn_interval_s": onset_interval(fields["t"], fields["body_yaw_deg"], .01),
            "source_velocity_heading_turn_interval_s": onset_interval(fields["t"], fields["velocity_heading_deg"], .01),
            "source_control_component_plateaus_abs_ge_0p999": [intervals(fields["t"], np.abs(controls[:, i])>=.999) for i in range(2)],
            "source_body_pitch_bank_peak_abs_0_3p5_deg": [float(np.max(np.abs(fields[key][fields["t"]<=3.5]))) for key in ("body_pitch_deg", "body_bank_deg")],
            "source_yaw_heading_rate_minus_body_yaw_rate_peak_abs_deg_s": float(np.max(np.abs(derivative(fields["body_yaw_deg"], fields["t"])[derivative_valid]-np.degrees(inverse["3"]["rates_pitch_yaw_rad_s"][derivative_valid, 1]))))}
        for key, value in fields.items(): arrays[cid+"__reference__"+key] = value
        arrays[cid+"__reference__controls"] = controls
        arrays[cid+"__reference__source_aoa_deg"] = reference["alpha_deg"]
        arrays[cid+"__reference__inverse_valid_mask"] = derivative_valid
        for width, data in inverse.items():
            for key, value in data.items(): arrays[cid+"__inverse"+width+"__"+key] = value
        if entry["offaxis_deg"] in (30, 60):
            state, target_at = prior.free_inputs(run, reference, config)
            model = prior.HalfHorizontalRelativePNGatedLoftGame(config, force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
            result = model.run(dataclasses.replace(state), target_at, end_time_s=config["performance"]["lifetime_s"])
            prediction = prior.prediction_arrays(result, model)
            for key, value in prediction.items():
                if not np.array_equal(value, half_arrays[cid+"__"+HALF+"__"+key]):
                    raise AssertionError("Existing half free trajectory did not reproduce exactly: "+cid+" / "+key)
            if result["summary"] != half_cases[cid]["variants"][HALF]["summary"]:
                raise AssertionError("Existing half summary differs")
            reproduced[cid] = {"all_prior_prediction_arrays_exact": True, "summary_exact": True}
            (OUT / f"{cid}_half_free_result.json").write_text(json.dumps(prior.jsonable(result), indent=2, allow_nan=False)+"\n")
            hq = continuous_quaternions(np.asarray([s["orientation_quaternion"] for s in result["samples"]]))
            half = trajectory_fields(prediction["t"], prediction["position_xyz_m"], prediction["velocity_xyz_mps"], target0, target_v, hq)
            half["exact_model_pitch_yaw_rates_rad_s"] = np.asarray([[s["pitch_rate_rad_s"], s["yaw_rate_rad_s"]] for s in result["samples"]])
            half["held_fin_pitch_yaw_rad"] = np.asarray([s["held_fin_rad"] for s in result["samples"]])
            baseline = {k.rsplit("__", 1)[-1]: base_arrays[k] for k in base_arrays.files if k.startswith(cid+"__wind_rate_angle__")}
            baseline.update(trajectory_fields(baseline["t"], baseline["position_xyz_m"], baseline["velocity_xyz_mps"], target0, target_v))
            ticks = prior.jsonable(prior.control_diagnostic(result, config)["ticks"])
            ticks = {key: np.asarray(value) for key, value in ticks.items()}
            ticks["fin_limit_pitch_yaw_rad"] = conditional["fin_limit_pitch_yaw_rad"]
            record.update(half_free_summary=result["summary"], half_early_control_diagnostic=prior.control_diagnostic(result, config),
                half_body_heading_turn_interval_s=onset_interval(half["t"], half["body_yaw_deg"], .01),
                half_velocity_heading_turn_interval_s=onset_interval(half["t"], half["velocity_heading_deg"], .01),
                half_body_pitch_bank_peak_abs_0_3p5_deg=[float(np.max(np.abs(half[key][half["t"]<=3.5]))) for key in ("body_pitch_deg", "body_bank_deg")],
                matched_display_time_table=[])
            for wanted in (.333333333, .5, .625, .833333333, 1., 1.5, 2., 3.):
                index = int(np.argmin(np.abs(fields["t"]-wanted)))
                t = fields["t"][index]
                if t > 3.5: continue
                row = {"reference_original_time_s": float(t), "half_values_interpolated_for_comparison_only": True}
                for key in ("speed_mps", "body_yaw_deg", "body_pitch_deg", "body_bank_deg", "velocity_heading_deg", "velocity_gamma_deg", "alpha_geometric_deg", "alpha_dot_body_deg_s", "alpha_dot_velocity_deg_s", "horizontal_velocity_to_los_error_deg"):
                    row[key] = {"reference": float(fields[key][index]), "half": float(np.interp(t, half["t"], half[key]))}
                row["body_yaw_rate_deg_s"] = {"reference_3point": float(np.degrees(inverse["3"]["rates_pitch_yaw_rad_s"][index, 1])),
                    "half": float(np.degrees(np.interp(t, half["t"], half["exact_model_pitch_yaw_rates_rad_s"][:, 1])))}
                row["conditional_reference_required_fin_pitch_yaw_rad"] = inverse["3"]["required_held_pitch_yaw_rad"][index]
                record["matched_display_time_table"].append(row)
            for key, value in half.items(): arrays[cid+"__half__"+key] = value
            for key, value in ticks.items(): arrays[cid+"__half_ticks__"+key] = value
            plot_case(cid, reference, fields, half, baseline, controls, inverse, ticks, OUT)
        records.append(record)
    assert len(reproduced)==2
    if any(prior.sha256(p)!=digest for p, digest in before.items()):
        raise AssertionError("Read-only comparison/input/kernel file changed")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise AssertionError("Nonfinite analysis arrays")
    document = {"reference_label": "Jari 2.59.0.28 interim videogame reference; true game revalidation pending",
        "local_profile_version": "2.59.0.27; entire rocket bridge equality previously verified; global .28 gameparams unverified",
        "predeclared_hypotheses": ["Observed onset interval versus existing local gain onset; no clock shift",
            "Command/held duration versus observed quaternion turn duration; exported control normalization unknown",
            "Incidence-rate decomposition into body-axis and velocity-direction turning",
            "Frozen Ku(speed)/active Ka/scaled Kw conditional required-fin feasibility, stencil sensitivity"],
        "new_free_predictions": 2, "reproduced_existing_half_only": reproduced,
        "baseline_AB_zero_native_not_rerun": True, "guidance_or_parameter_experiment_performed": False,
        "common_inner_coefficients": dataclasses.asdict(SurfaceCoefficients()), "accepted_loft_coefficients": [.75, 1.3],
        "local_baseline_orientation_not_saved_and_not_reconstructed": True,
        "native_48Hz_PID_reproduction_claimed": False, "source_omega_frame_units_assumed": False,
        "inverse_reference_state_used_in_free_forward": False, "cases": records,
        "verification": {"read_only_artifacts_unchanged": True, "all_arrays_finite": True,
            "all_original_reference_rows_retained": True, "two_free_predictions_exactly_reproduce_saved_half": True}}
    document["findings"] = derive_findings(document, arrays)
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    (OUT / "results.json").write_text(json.dumps(prior.jsonable(document), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(OUT), "reproduced": reproduced,
                      "conditional_fin": {c["case_id"]: c["conditional_inner_inverse"] for c in records}}, default=prior.jsonable))


if __name__ == "__main__":
    main()
