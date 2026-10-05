"""Bounded inverse reference-couple audit on the three no-loft replays.

This uses native velocity and angular increments, never a PID reconstruction.
One fixed passive law and one common reference-couple/force-residual relation
are tested. All states remain measured external inputs: this is not forward
validation of a plant, controller, or private game mechanism.
"""
from __future__ import annotations

import json
import math

import numpy as np

import analyze_r771_batch_loft as batch
import check_r771_body_impulse as impulse
import try_r771_effective_moment as model

OUT = batch.BASE / "no_loft_control" / "reference_couple_inverse"
NAMES = ("L110", "L27.5", "L55")
FIT = (15 / 48, 29 / 48)
END = 2.5
ARM = .4
K2 = 3.71**2 / 12
PASSIVE_SLOPE = 14.
POSE_FLAG_RAD = .001  # diagnostic sensitivity, never changes primary fit
MAX_NORMALIZED_INCREMENT_RMSE = .25
NODES, WEIGHTS = impulse.GL_NODES, impulse.GL_WEIGHTS


def passive_integrals(raw, i, lo, hi):
    """Fixed existing 14*alpha wind-normal law at one rotating aft surface.

    Local velocity is v + omega cross r = v - l*omega*n. Dividing by
    cos(alpha_CG) preserves the previous wind-normal/body-normal conversion.
    No separately fitted rotational damping enters this calculation.
    """
    r, p = raw["r"], raw["p"]
    area = math.pi * p["caliber_m"]**2 / 4 * p["wing_area_mult"]
    dt = hi - lo
    totals = np.zeros(6)
    for node, weight in zip(NODES, WEIGHTS):
        f = (node + 1) / 2
        t = lo + f * dt
        theta = float(r["theta"][i] + f * (r["theta"][i + 1] - r["theta"][i]))
        omega = float(r["omega"][i] + f * (r["omega"][i + 1] - r["omega"][i]))
        v = r["velocity3"][i] + f * (r["velocity3"][i + 1] - r["velocity3"][i])
        y = float(r["y"][i] + f * (r["y"][i + 1] - r["y"][i]))
        n = impulse.normal(theta)
        forward = np.array((math.cos(theta), math.sin(theta), 0.))
        local_v = v - ARM * omega * n
        alpha = theta - math.atan2(float(v[1]), float(v[0]))
        local_alpha = math.atan2(float(-local_v @ n), float(local_v @ forward))
        mass, _ = model.mass_thrust(t, p)
        rho = model.isa_density(y)
        qs_over_m = .5 * rho * float(v @ v) * area / mass
        local_qs_over_m = .5 * rho * float(local_v @ local_v) * area / mass
        passive = local_qs_over_m * PASSIVE_SLOPE * local_alpha / math.cos(alpha)
        # The rate-free comparison only quantifies already included geometry.
        rate_free = qs_over_m * PASSIVE_SLOPE * alpha / math.cos(alpha)
        totals += weight * dt / 2 * np.array(
            (passive, qs_over_m, alpha, omega, local_alpha, passive - rate_free))
    return dict(zip(("passive_J_mps", "qS_over_mass_integral_mps",
                     "alpha_integral_rad_s", "omega_integral_rad",
                     "local_alpha_integral_rad_s", "geometry_passive_J_mps"),
                    map(float, totals)))


def rows_for(raw):
    r = raw["r"]
    t = np.array([impulse.canonical_time(float(x)) for x in raw["b5"]])
    packet = np.asarray(r["t"])
    angles = np.array([[float(row[k]) for k in
                        ("raw_euler_x", "raw_euler_y", "raw_euler_z")]
                       for row in raw["case"]["missile_rows"]])
    assert np.all(np.diff(t) > 0) and np.all(np.diff(packet) > 0)
    rows = []
    for i in range(len(t) - 1):
        if t[i + 1] > END + 1e-10:
            break
        lo, hi = map(float, t[i:i + 2])
        dt = hi - lo
        th0, th1 = map(float, r["theta"][i:i + 2])
        om0, om1 = map(float, r["omega"][i:i + 2])
        dw = om1 - om0
        v0, v1 = r["velocity3"][i:i + 2]
        a = impulse.planar_impulse(v0, v1, th0, th1, dt)
        packet_dt = float(packet[i + 1] - packet[i])
        j_packet = impulse.planar_impulse(v0, v1, th0, th1, packet_dt)["primary"]
        j_3d = impulse.full_3d_candidate(v0, v1, angles[i], angles[i + 1], dt)
        mean_3d = np.zeros(3)
        for node, weight in zip(NODES, WEIGHTS):
            f = (node + 1) / 2
            bank, heading, pitch = angles[i] + f * (angles[i + 1] - angles[i])
            mean_3d += weight / 2 * batch.pose_rotation(float(bank), float(heading), float(pitch))[:, 1]
        passive = passive_integrals(raw, i, lo, hi)
        packet_passive = passive_integrals(raw, i, float(packet[i]), float(packet[i + 1]))
        pose_error = th1 - th0 - (om0 + om1) / 2 * dt
        j = a["primary"]
        ref = K2 * dw + ARM * j
        bound_j = impulse.VELOCITY_LSB_MPS * float(np.sum(abs(a["mean_normal"])))
        rows.append({"index": i, "start_s": lo, "end_s": hi,
                     "midpoint_s": (lo + hi) / 2, "duration_s": dt,
                     "packet_start_s": float(packet[i]), "packet_end_s": float(packet[i + 1]),
                     "packet_duration_s": packet_dt,
                     "body_normal_impulse_mps": j,
                     "observed_delta_omega_rad_s": dw,
                     "reference_couple_impulse_over_mass_m2ps": ref,
                     "mean_reference_couple_over_mass_m2ps2": ref / dt,
                     "weighted_reference_couple_coefficient": ref / (ARM * passive["qS_over_mass_integral_mps"]),
                     "force_residual_impulse_mps": j - passive["passive_J_mps"],
                     "mean_alpha_deg": math.degrees(passive["alpha_integral_rad_s"] / dt),
                     "mean_omega_rad_s": passive["omega_integral_rad"] / dt,
                     "pose_minus_rate_integral_rad": pose_error,
                     "pose_rate_flag": abs(pose_error) > POSE_FLAG_RAD,
                     "mean_normal": a["mean_normal"].tolist(),
                     "mean_normal_3d": mean_3d.tolist(),
                     "gain_start": model.interpolate_table(lo, raw["p"]["time_gain"]),
                     "gain_end": model.interpolate_table(hi, raw["p"]["time_gain"]),
                     "data_only_reference_impulse_quantization_bound_m2ps":
                         K2 * impulse.RATE_LSB_RAD_S + ARM * bound_j,
                     "passive": passive,
                     "sensitivity": {
                         "packet_body_J_mps": j_packet,
                         "packet_reference_R_m2ps": K2 * dw + ARM * j_packet,
                         "packet_force_residual_J_mps": j_packet - packet_passive["passive_J_mps"],
                         "packet_passive": packet_passive,
                         "full_3d_body_J_mps": j_3d,
                         "full_3d_reference_R_m2ps": K2 * dw + ARM * j_3d,
                         "full_3d_force_residual_J_mps": j_3d - passive["passive_J_mps"],
                         "max_planar_projection_change_J_mps": max(
                             abs(a[method] - j) for method in ("midpoint", "trapezoid", "end_direction"))}})
    assert rows
    return rows


def select(rows, lo, hi, *, pose_consistent=False):
    return [r for r in rows if r["start_s"] >= lo - 1e-10
            and r["end_s"] <= hi + 1e-10
            and (not pose_consistent or not r["pose_rate_flag"])]


def shared_endpoint_bound(rows, route="b5"):
    """A decoded endpoint has one error, shared by adjacent intervals.

    This avoids treating the same velocity endpoint as independent errors in
    every native interval. Pose, time, inertia and force-law error are excluded.
    """
    vc, wc = {}, {}
    for r in rows:
        i = r["index"]
        n = np.array(r["mean_normal_3d"] if route == "full_3d" else r["mean_normal"])
        vc[i] = vc.get(i, np.zeros(3)) - n
        vc[i + 1] = vc.get(i + 1, np.zeros(3)) + n
        wc[i] = wc.get(i, 0.) - 1.
        wc[i + 1] = wc.get(i + 1, 0.) + 1.
    jb = impulse.VELOCITY_LSB_MPS / 2 * sum(float(np.sum(abs(v))) for v in vc.values())
    wb = impulse.RATE_LSB_RAD_S / 2 * sum(abs(v) for v in wc.values())
    return jb, K2 * wb + ARM * jb


def evaluate(rows, beta, route="b5"):
    dt = np.array([r["duration_s"] if route != "packet" else r["packet_duration_s"] for r in rows])
    dw = np.array([r["observed_delta_omega_rad_s"] for r in rows])
    if route == "b5":
        j = np.array([r["body_normal_impulse_mps"] for r in rows])
        c = np.array([r["force_residual_impulse_mps"] for r in rows])
        ref = np.array([r["reference_couple_impulse_over_mass_m2ps"] for r in rows])
        denom = sum(r["passive"]["qS_over_mass_integral_mps"] for r in rows)
    else:
        prefix = "packet" if route == "packet" else "full_3d"
        j = np.array([r["sensitivity"][prefix + "_body_J_mps"] for r in rows])
        c = np.array([r["sensitivity"][prefix + "_force_residual_J_mps"] for r in rows])
        ref = np.array([r["sensitivity"][prefix + "_reference_R_m2ps"] for r in rows])
        denom = sum((r["sensitivity"]["packet_passive"] if route == "packet" else r["passive"])
                    ["qS_over_mass_integral_mps"] for r in rows)
    pred = (-ARM * j + beta * c) / K2
    residual = beta * c - ref
    jb, rb = shared_endpoint_bound(rows, route)
    observed_rms = float(np.sqrt(np.mean(dw**2)))
    rmse = float(np.sqrt(np.mean((pred - dw)**2)))
    result = {"native_interval_count": len(rows), "start_s": rows[0]["start_s"],
              "end_s": rows[-1]["end_s"], "summed_duration_s": float(sum(dt)),
              "body_normal_impulse_mps": float(sum(j)),
              "passive_body_normal_impulse_mps": float(sum(j - c)),
              "force_residual_impulse_mps": float(sum(c)),
              "observed_delta_omega_rad_s": float(sum(dw)),
              "reference_couple_impulse_over_mass_m2ps": float(sum(ref)),
              "mean_reference_couple_over_mass_m2ps2": float(sum(ref) / sum(dt)),
              "weighted_reference_couple_coefficient": float(sum(ref) / (ARM * denom)),
              "predicted_reference_couple_impulse_m2ps": float(beta * sum(c)),
              "predicted_delta_omega_rad_s": float(sum(pred)),
              "reference_relation_impulse_error_m2ps": float(sum(residual)),
              "angular_increment_rmse_rad_s": rmse,
              "angular_increment_rmse_normalized_by_observed_rms": rmse / max(observed_rms, 1e-12),
              "summed_absolute_reference_impulse_error_m2ps": float(sum(abs(residual))),
              "angular_increment_sign_disagreement_count": int(sum(pred * dw < 0)),
              "pose_rate_flagged_interval_count": sum(r["pose_rate_flag"] for r in rows),
              "data_only_shared_endpoint_J_quantization_bound_mps": jb,
              "data_only_shared_endpoint_R_quantization_bound_m2ps": rb,
              "reference_impulse_sign_resolved_at_fixed_pose_time_inertia": abs(float(sum(ref))) > rb,
              "geometry_passive_impulse_mps": sum(
                  (r["sensitivity"]["packet_passive"] if route == "packet" else r["passive"])
                  ["geometry_passive_J_mps"] for r in rows)}
    result["passes_predeclared_increment_accuracy"] = (
        result["angular_increment_rmse_normalized_by_observed_rms"] <= MAX_NORMALIZED_INCREMENT_RMSE)
    return result


def alignment_sensitivity(raw, rows):
    """Hypothetical one-record rate/velocity skew, not an asserted error bar."""
    i0, i1 = rows[0]["index"], rows[-1]["index"] + 1
    omega = raw["r"]["omega"]
    j = sum(r["body_normal_impulse_mps"] for r in rows)
    variants = {}
    for shift in (-1, 0, 1):
        if i0 + shift < 0 or i1 + shift >= len(omega):
            continue
        dw = float(omega[i1 + shift] - omega[i0 + shift])
        variants[str(shift)] = {"delta_omega_rad_s": dw,
                                "reference_impulse_m2ps": K2 * dw + ARM * j,
                                "positive_J_and_positive_delta_omega": j > 0 and dw > 0}
    return {"meaning": "Rate endpoints moved by one recorded row relative to fixed velocity/pose window. Unknown internal skew is not bounded by this test.",
            "shifts_in_recorded_rows": variants}


def plot(all_rows, beta):
    plt = batch.plt
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex="col", layout="constrained")
    for col, name in enumerate(NAMES):
        rows = all_rows[name]
        t = np.array([r["midpoint_s"] for r in rows])
        dt = np.array([r["duration_s"] for r in rows])
        ref = np.array([r["reference_couple_impulse_over_mass_m2ps"] for r in rows])
        c = np.array([r["force_residual_impulse_mps"] for r in rows])
        j = np.array([r["body_normal_impulse_mps"] for r in rows])
        dw = np.array([r["observed_delta_omega_rad_s"] for r in rows])
        axes[0, col].plot(t, [r["mean_alpha_deg"] for r in rows], color="#3577a8", label="Measured-state mean alpha (deg)")
        twin = axes[0, col].twinx()
        twin.plot(t, [r["mean_omega_rad_s"] for r in rows], color="#bf7f26", alpha=.8)
        twin.set_ylabel("Pitch rate (rad/s)", color="#bf7f26", fontsize=9)
        axes[0, col].set_ylabel("AoA (deg)")
        axes[0, col].set_title(name + (" | calibration + continuation" if name == "L110" else " | frozen cross-case check"))
        ax = axes[1, col]
        ax.plot(t, ref / dt, "ko-", ms=3, lw=1.2, label="Required reference couple / mass")
        ax.plot(t, beta * c / dt, color="#ae4f3d", label="Frozen beta × force residual")
        ax.plot(t, [r["sensitivity"]["packet_reference_R_m2ps"] / r["packet_duration_s"] for r in rows],
                color=".55", ls=":", label="Required, packet clock")
        ax.set_ylabel("Interval mean Mref/m (m²/s²)")
        ax = axes[2, col]
        ax.plot(t, dw, "ko-", ms=3, label="Decoded angular increment")
        ax.plot(t, (-ARM * j + beta * c) / K2, color="#ae4f3d", label="One frozen reference-couple relation")
        ax.set_ylabel("Pitch-rate increment (rad/s)")
        ax = axes[3, col]
        ax.plot([r["end_s"] for r in rows], np.cumsum(ref), color="k", lw=1.6, label="Integrated required Mref/m")
        ax.plot([r["end_s"] for r in rows], beta * np.cumsum(c), color="#ae4f3d", lw=1.6, label="Integrated frozen relation")
        ax.set_ylabel("Reference angular impulse / mass (m²/s)")
        for row in range(4):
            ax = axes[row, col]
            ax.axhline(0, color=".6", lw=.7)
            ax.axvspan(0, .30, color=".85", alpha=.4)
            if name == "L110":
                ax.axvspan(*FIT, color="#d2e5ef", alpha=.55)
            for r in rows:
                if r["pose_rate_flag"]:
                    ax.axvspan(r["start_s"], r["end_s"], color="#f1bd9f", alpha=.23)
            ax.set_xlim(0, END)
            ax.grid(alpha=.14)
        axes[3, col].set_xlabel("b5 native flight age (s); shaded orange = pose/rate inconsistency")
    axes[1, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    axes[3, 0].legend(fontsize=8)
    fig.suptitle("Inverse single-surface reference-couple audit | one force residual law, one frozen beta\n"
                 "Measured states drive every interval; shaded grey = gain inactive, blue = L110 calibration", fontsize=13)
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    raw_shots, parameters = batch.load_all()
    assert parameters["source_version"] == "2.59.0.28"
    assert abs(parameters["tail_arm_m"] - ARM) < 1e-12
    all_rows = {name: rows_for(raw_shots[name]) for name in NAMES}
    train = select(all_rows["L110"], *FIT)
    assert train[0]["start_s"] == FIT[0] and train[-1]["end_s"] == FIT[1]
    x = np.array([r["force_residual_impulse_mps"] for r in train])
    y = np.array([r["reference_couple_impulse_over_mass_m2ps"] for r in train])
    assert x @ x > 1e-12
    beta = float(x @ y / (x @ x))
    packet_x = np.array([r["sensitivity"]["packet_force_residual_J_mps"] for r in train])
    packet_y = np.array([r["sensitivity"]["packet_reference_R_m2ps"] for r in train])
    packet_beta = float(packet_x @ packet_y / (packet_x @ packet_x))
    cases = {}
    for name, rows in all_rows.items():
        windows = {"gain_inactive": select(rows, 0, .30),
                   "activation_transition": select(rows, 13 / 48, 15 / 48),
                   "early_active": select(rows, *FIT),
                   "continuation": select(rows, FIT[1], END),
                   "downstroke": select(rows, 29 / 48, 37 / 48),
                   "early_recovery": select(rows, 37 / 48, 44 / 48),
                   "positive_peak": select(rows, 44 / 48, 52 / 48),
                   "all_active": select(rows, FIT[0], END)}
        summaries = {w: evaluate(rs, beta) for w, rs in windows.items() if rs}
        active = windows["all_active"]
        sensitivity = {"packet_clock_same_frozen_beta": evaluate(active, beta, "packet"),
                       "full_3d_force_projection_same_frozen_beta": evaluate(active, beta, "full_3d"),
                       "remove_pose_rate_flagged_same_frozen_beta": evaluate(
                           select(rows, FIT[0], END, pose_consistent=True), beta),
                       "recovery_remove_pose_rate_flagged": evaluate(
                           select(rows, 37 / 48, 44 / 48, pose_consistent=True), beta)}
        err = np.array([r["pose_minus_rate_integral_rad"] for r in rows])
        r = raw_shots[name]["r"]
        required = np.array([v["reference_couple_impulse_over_mass_m2ps"] / v["duration_s"] for v in active])
        description = {}
        for key in ("mean_alpha_deg", "mean_omega_rad_s"):
            state = np.array([v[key] for v in active])
            description[key] = {"range": [float(min(state)), float(max(state))],
                                "correlation_with_interval_mean_Mref_over_mass": float(np.corrcoef(state, required)[0, 1])}
        cases[name] = {"case": raw_shots[name]["case"]["case"],
                       "finsLatAccel_raw": raw_shots[name]["case"]["finsLatAccel"],
                       "loft_enabled": raw_shots[name]["p"]["loft_enabled"],
                       "native_intervals": len(rows), "windows": summaries,
                       "sensitivities": sensitivity,
                       "pose_rate": {"median_abs_integral_error_rad": float(np.median(abs(err))),
                                     "p95_abs_integral_error_rad": float(np.quantile(abs(err), .95)),
                                     "max_abs_integral_error_rad": float(max(abs(err))),
                                     "flagged_interval_indices": [v["index"] for v in rows if v["pose_rate_flag"]]},
                       "state_relation_description": {"values": description,
                                                      "meaning": "Descriptive correlation on measured-state interval means; no coefficient fit and no causal or constitutive identification."},
                       "early_recovery_rate_alignment_sensitivity": alignment_sensitivity(
                           raw_shots[name], windows["early_recovery"]),
                       "packet_minus_b5_range_s": [float(min(r["t"] - raw_shots[name]["b5"])),
                                                   float(max(r["t"] - raw_shots[name]["b5"]))],
                       "intervals": rows}
    # Reuse the prior primary result as a comparison, not another rerun.
    previous_path = impulse.OUT / "results.json"
    previous = json.loads(previous_path.read_text())
    prior_difference = {stage: cases["L110"]["windows"][stage]
                        ["reference_couple_impulse_over_mass_m2ps"] - values["residual_m2ps"]
                        for stage, values in previous["stages"].items()}
    assert max(abs(v) for v in prior_difference.values()) < 1e-12
    ref_110 = cases["L110"]["windows"]["all_active"]["reference_couple_impulse_over_mass_m2ps"]
    intervention = {"raw_finsLatAccel_ratio_L110_to_L27p5": 4.,
                    "all_active_required_reference_impulse_relative_difference_L27p5_vs_L110":
                    (cases["L27.5"]["windows"]["all_active"]["reference_couple_impulse_over_mass_m2ps"] - ref_110) / ref_110,
                    "interpretation": "The fourfold raw-field intervention produces nearly equal inferred integrated reference impulse. This does not identify control force or prove a particular field mapping or saturation law."}
    result = {
        "scope": "Conditional inverse/identification experiment; no full-forward propagation or production/PID changes.",
        "sources": [str(batch.BASE / "source_payload.json"), str(batch.BASE / "batch_compact.json"),
                    str(previous_path)],
        "fixed_geometry": {"single_equivalent_surface_aft_arm_m": ARM, "length_m": 3.71,
                           "provisional_k_squared_m2": K2,
                           "inertia_assumption": "I=m*L²/12, instantaneous angular equation I*omega_dot=M"},
        "equations": {"primary": "R=integral(Mref/m dt)=k²*Delta(omega)+l*Jbody; M=-l*Nbody+Mref",
                      "force": "Jbody=integral n(theta) dot (dv-g dt); decoded inertial velocities and linearly interpolated pose",
                      "force_residual": "Jc=Jbody-integral(Npassive/m dt)",
                      "tested_relation": "Mref=beta*Nc at the same equivalent surface/reference, constant beta; no second force location",
                      "passive_assumption": "Npassive=q_local*S*14*alpha_local/cos(alpha_CG), v_local=v-l*omega*n; no extra damping term",
                      "qS_normalization": "weighted CMref=R/(l*integral(q_CG*S/m dt)); q/m and mass law are assumed inputs"},
        "telemetry_vs_inference": {
            "decoded_inputs": "b7 velocity, -b8_z pitch rate, Euler Z pitch, Euler bank/heading, b5/packet stamps, positions",
            "inferred": "body-normal specific impulse, reference couple, force residual and qS-normalized coefficients",
            "not_telemetry": "fin input, PID output, passive/control force split, inertia, aerodynamic coefficients, reference couple"},
        "predeclared_test": {"calibration_case": "L110", "calibration_window_s": list(FIT),
                             "fit": "one scalar least squares on native integrated R versus Jc, no intercept",
                             "frozen_evaluations": "L110 continuation and all L27.5/L55 active native intervals",
                             "acceptance_max_increment_rmse_over_observed_rms": MAX_NORMALIZED_INCREMENT_RMSE,
                             "pose_flag_threshold_rad": POSE_FLAG_RAD,
                             "pose_flag_use": "sensitivity exclusion only, primary fit remains unchanged"},
        "calibration": {"beta_m": beta, "native_interval_count": len(train),
                        "force_residual_predictor_l2_mps": float(np.linalg.norm(x)),
                        "required_reference_predictand_l2_m2ps": float(np.linalg.norm(y)),
                        "unconstrained_same_relation_packet_clock_beta_m": packet_beta,
                        "packet_beta_use": "timing sensitivity of identification only; all reported holdouts retain primary beta",
                        "primary": evaluate(train, beta)},
        "candidate_accepted": bool(evaluate(train, beta)["passes_predeclared_increment_accuracy"] and all(
            cases[n]["windows"]["continuation" if n == "L110" else "all_active"]
            ["passes_predeclared_increment_accuracy"] for n in NAMES)),
        "raw_field_intervention": intervention,
        "cases": cases,
        "limitations": [
            "The 14/rad passive slope is an existing analysis law, absent from the raw parent rocket; its force residual is not decoded control force.",
            "Body-normal impulse includes all non-gravity force projections; assuming axial thrust gives zero transverse component. Drag/passive-law errors can enter Jc.",
            "Pose and velocity are linearly interpolated within native intervals; internal synchronization and the decoded velocity reference as COM remain unverified.",
            "Rod inertia is provisional. Residual moments scale with its value, although positive J and positive Delta(omega) cannot obey a purely aft-force moment for any positive inertia.",
            "qS normalization adds the previous ISA density, raw caliber/wingAreaMult area and linear mass-burn assumptions; Mref/m requires none of these.",
            "Quantization bounds cover only decoded velocity/rate at fixed pose/time/inertia, with shared endpoint errors; they exclude passive law, pose and timestamp error.",
            "A scalar couple/force relation failure does not identify the private game law or prove the correct constitutive Mref(alpha,omega,input)."],
        "verification": {"matches_prior_L110_stage_R_max_abs_m2ps": max(abs(v) for v in prior_difference.values()),
                         "all_sources_keep_same_arm_and_length": all(
                             s["p"]["tail_arm_m"] == ARM and s["p"]["length_m"] == 3.71 for s in raw_shots.values()),
                         "no_pid_or_previous_fitted_coefficients_read": True,
                         "all_reported_values_finite": True}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    arrays = {}
    for name, rows in all_rows.items():
        for key in ("start_s", "end_s", "duration_s", "body_normal_impulse_mps", "observed_delta_omega_rad_s",
                    "reference_couple_impulse_over_mass_m2ps", "force_residual_impulse_mps", "weighted_reference_couple_coefficient"):
            arrays[name + "__" + key] = np.array([r[key] for r in rows])
        arrays[name + "__predicted_delta_omega_rad_s"] = np.array([
            (-ARM * r["body_normal_impulse_mps"] + beta * r["force_residual_impulse_mps"]) / K2 for r in rows])
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(OUT / "intervals.npz", **arrays)
    plot(all_rows, beta)
    print(json.dumps({"calibration": result["calibration"], "candidate_accepted": result["candidate_accepted"],
                      "cases": {n: {"windows": c["windows"], "sensitivities": c["sensitivities"],
                                    "pose_rate": c["pose_rate"]} for n, c in cases.items()},
                      "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
