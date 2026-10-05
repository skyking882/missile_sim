"""Direct velocity/body-normal impulse audit of L110 replay intervals.

The primary calculation contains no guidance, PID, aerodynamic coefficient,
thrust magnitude, mass, or calibrated moment gain. It tests only a conditional
identity: all aerodynamic body-normal force at one fixed aft point, with no
other pitch couple, would require k² Δω + l J_body = 0.
"""
from __future__ import annotations

import json
import math

import numpy as np

import analyze_r771_batch_loft as batch

OUT = batch.BASE / "no_loft_control" / "l110_body_impulse"
PREVIOUS = batch.BASE / "no_loft_control" / "l110_local_rotation" / "results.json"
START = 29 / 48
END_LIMIT = 1.4
LENGTH_M = 3.71
ARM_M = .4
K2_M2 = LENGTH_M**2 / 12
VELOCITY_LSB_MPS = 4000 / 65535
RATE_LSB_RAD_S = 30 / 65535
G_WORLD = np.array((0., -batch.G, 0.))
STAGES = (("downstroke", 29 / 48, 37 / 48),
          ("early_recovery", 37 / 48, 44 / 48),
          ("positive_peak", 44 / 48, 52 / 48))
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


def canonical_time(raw):
    tick = round(raw * 48)
    t = tick / 48
    return t if abs(raw - t) < 1e-7 else raw


def normal(theta):
    return np.array((-math.sin(theta), math.cos(theta), 0.))


def planar_impulse(v0, v1, theta0, theta1, dt):
    delta_theta = theta1 - theta0
    theta_mid = (theta0 + theta1) / 2
    mean_normal = np.sinc(delta_theta / (2 * math.pi)) * normal(theta_mid)
    dv_minus_gdt = v1 - v0 - G_WORLD * dt
    primary = float(mean_normal @ dv_minus_gdt)
    midpoint = float(normal(theta_mid) @ dv_minus_gdt)
    trapezoid = float((normal(theta0) + normal(theta1)) @ dv_minus_gdt / 2)
    end_direction = float(normal(theta1) @ dv_minus_gdt)
    # Independent quadrature of the stated linear theta and velocity paths.
    integral = 0.
    for node, weight in zip(GL_NODES, GL_WEIGHTS):
        fraction = (node + 1) / 2
        theta = theta0 + fraction * delta_theta
        integral += weight * float(normal(theta) @ dv_minus_gdt) / 2
    return {"primary": primary, "midpoint": midpoint,
            "trapezoid": trapezoid, "end_direction": end_direction,
            "quadrature": integral, "mean_normal": mean_normal}


def full_3d_candidate(v0, v1, angles0, angles1, dt):
    dv_minus_gdt = v1 - v0 - G_WORLD * dt
    result = 0.
    for node, weight in zip(GL_NODES, GL_WEIGHTS):
        fraction = (node + 1) / 2
        bank, heading, pitch = angles0 + fraction * (angles1 - angles0)
        body_up = batch.pose_rotation(float(bank), float(heading), float(pitch))[:, 1]
        result += weight * float(body_up @ dv_minus_gdt) / 2
    return result


def stage(rows, name, lo, hi):
    selected = [r for r in rows if r["canonical_start_s"] >= lo - 1e-10
                and r["canonical_end_s"] <= hi + 1e-10]
    assert selected and abs(selected[0]["canonical_start_s"] - lo) < 1e-10
    assert abs(selected[-1]["canonical_end_s"] - hi) < 1e-10
    summed_dw = sum(r["observed_delta_omega_rad_s"] for r in selected)
    summed_j = sum(r["body_normal_impulse_mps"] for r in selected)
    summed_j_bound = sum(r["quantization"]["body_impulse_abs_bound_mps"]
                         for r in selected)
    stage_r_bound = K2_M2 * RATE_LSB_RAD_S + ARM_M * summed_j_bound
    sums = {"name": name, "start_s": lo, "end_s": hi,
            "native_interval_count": len(selected),
            "observed_delta_omega_rad_s": summed_dw,
            "observed_mean_omega_dot_rad_s2": summed_dw / (hi - lo),
            "body_normal_impulse_mps": summed_j,
            "mean_body_normal_specific_accel_mps2": summed_j / (hi - lo),
            "fixed_aft_point_predicted_delta_omega_rad_s": -ARM_M * summed_j / K2_M2,
            "fixed_aft_point_predicted_mean_omega_dot_rad_s2":
                -ARM_M * summed_j / (K2_M2 * (hi - lo)),
            "residual_m2ps": K2_M2 * summed_dw + ARM_M * summed_j,
            "conditional_quantization_abs_bound_residual_m2ps": stage_r_bound,
            "conditional_quantization_abs_bound_J_mps": summed_j_bound,
            "opposite_sign_interval_count": int(sum(
                r["observed_delta_omega_rad_s"] * r["body_normal_impulse_mps"] > 0
                for r in selected)),
            "J_sign_robust_to_velocity_quantization": bool(abs(summed_j) > summed_j_bound),
            "R_sign_robust_to_velocity_rate_quantization": bool(
                abs(K2_M2 * summed_dw + ARM_M * summed_j) > stage_r_bound)}
    sums["inferred_effective_arm_m"] = (
        -K2_M2 * summed_dw / summed_j if abs(summed_j) > summed_j_bound else None)
    for method in ("midpoint", "trapezoid", "end_direction", "full_3d_candidate"):
        sums[method + "_body_impulse_mps"] = sum(r["sensitivity"][method + "_J_mps"]
                                                     for r in selected)
    return sums


def plot(rows):
    plt = batch.plt
    mid = np.array([r["midpoint_s"] for r in rows])
    dt = np.array([r["duration_s"] for r in rows])
    dw = np.array([r["observed_delta_omega_rad_s"] for r in rows])
    j = np.array([r["body_normal_impulse_mps"] for r in rows])
    j_bound = np.array([r["quantization"]["body_impulse_abs_bound_mps"]
                        for r in rows])
    residual = np.array([r["identity_residual_m2ps"] for r in rows])
    residual_bound = np.array([r["quantization"]["identity_residual_abs_bound_m2ps"]
                               for r in rows])
    arms = np.array([np.nan if r["inferred_effective_arm_m"] is None else
                     r["inferred_effective_arm_m"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                             layout="constrained")
    ax = axes[0, 0]
    ax.plot(mid, dw, "ko-", ms=4, lw=1.2, label="Decoded pitch-rate increment")
    ax.plot(mid, -ARM_M * j / K2_M2, color="#873b9e", marker="d", ms=3.5,
            lw=1.3, label="Fixed aft point prediction")
    ax.set_ylabel("Change in pitch rate per interval (rad/s)")
    ax = axes[0, 1]
    ax.errorbar(mid, j, yerr=j_bound, color="#2779b5", marker="o", ms=3.5,
                lw=1.2, capsize=2, label="Body-normal impulse")
    ax.set_ylabel("Body-normal impulse per mass (m/s)")
    ax = axes[1, 0]
    ax.errorbar(mid, residual, yerr=residual_bound, color="#c0533b",
                marker="o", ms=3.5, lw=1.2, capsize=2,
                label="k² Δω + l J")
    ax.set_ylabel("Identity residual (m²/s)")
    ax = axes[1, 1]
    ax.plot(mid, arms, color="#287c69", marker="o", ms=3.5, lw=1.2,
            label="Implied arm from each interval")
    ax.axhline(ARM_M, color="#873b9e", ls="--", lw=1.2,
               label="Fixed +0.4 m aft arm")
    ax.set_ylabel("Effective aft arm (m)")
    for ax in axes.flat:
        ax.axhline(0, color=".6", lw=.7)
        for _, _, boundary in STAGES:
            ax.axvline(boundary, color=".78", lw=.7)
        ax.set(xlim=(START, rows[-1]["canonical_end_s"]),
               xlabel="Time from launch (s)")
        ax.grid(alpha=.16)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("L110: decoded velocity impulse vs angular impulse, native intervals\n"
                 "Error bars cover only known velocity/rate quantization at fixed pose and time")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shots, parameters = batch.load_all()
    shot = shots["L110"]
    assert parameters["source_version"] == "2.59.0.28"
    assert abs(parameters["tail_arm_m"] - ARM_M) < 1e-12
    assert abs(parameters["length_m"] - LENGTH_M) < 1e-12
    r = shot["r"]
    raw_t = np.asarray(shot["b5"], float)
    canonical = np.array([canonical_time(float(t)) for t in raw_t])
    packet_t = np.asarray(r["t"], float)
    theta = np.asarray(r["theta"], float)
    omega = np.asarray(r["omega"], float)
    velocity = np.asarray(r["velocity3"], float)
    angles = np.array([[float(row[k]) for k in
                        ("raw_euler_x", "raw_euler_y", "raw_euler_z")]
                       for row in shot["case"]["missile_rows"]])
    assert len(raw_t) == len(theta) == len(omega) == len(velocity) == len(angles)
    indices = [i for i in range(len(raw_t) - 1)
               if canonical[i] >= START - 1e-10 and canonical[i] < END_LIMIT]
    assert indices and indices[0] == 14 and indices[-1] == 32
    rows = []
    for i in indices:
        lo, hi = float(canonical[i]), float(canonical[i + 1])
        dt = hi - lo
        v0, v1 = velocity[i], velocity[i + 1]
        th0, th1 = float(theta[i]), float(theta[i + 1])
        dw = float(omega[i + 1] - omega[i])
        projected = planar_impulse(v0, v1, th0, th1, dt)
        j = projected["primary"]
        j_3d = full_3d_candidate(v0, v1, angles[i], angles[i + 1], dt)
        j_velocity_bound = VELOCITY_LSB_MPS * (
            abs(projected["mean_normal"][0]) + abs(projected["mean_normal"][1]))
        residual = K2_M2 * dw + ARM_M * j
        r_bound = K2_M2 * RATE_LSB_RAD_S + ARM_M * j_velocity_bound
        denom_robust = abs(j) > j_velocity_bound
        angle_integral = (omega[i] + omega[i + 1]) * dt / 2
        row = {
            "index": i, "raw_b5_start_s": float(raw_t[i]),
            "raw_b5_end_s": float(raw_t[i + 1]),
            "packet_start_s": float(packet_t[i]),
            "packet_end_s": float(packet_t[i + 1]),
            "canonical_start_s": lo, "canonical_end_s": hi,
            "midpoint_s": (lo + hi) / 2, "duration_s": dt,
            "theta_start_rad": th0, "theta_end_rad": th1,
            "observed_delta_theta_rad": th1 - th0,
            "trapezoid_omega_integral_rad": angle_integral,
            "theta_minus_integrated_omega_rad": th1 - th0 - angle_integral,
            "observed_delta_omega_rad_s": dw,
            "observed_mean_omega_dot_rad_s2": dw / dt,
            "velocity_start_xyz_mps": v0.tolist(),
            "velocity_end_xyz_mps": v1.tolist(),
            "body_normal_impulse_mps": j,
            "mean_body_normal_specific_accel_mps2": j / dt,
            "fixed_aft_point_predicted_delta_omega_rad_s": -ARM_M * j / K2_M2,
            "fixed_aft_point_predicted_mean_omega_dot_rad_s2":
                -ARM_M * j / (K2_M2 * dt),
            "identity_residual_m2ps": residual,
            "inferred_effective_arm_m": -K2_M2 * dw / j if denom_robust else None,
            "sensitivity": {
                "midpoint_J_mps": projected["midpoint"],
                "trapezoid_J_mps": projected["trapezoid"],
                "end_direction_J_mps": projected["end_direction"],
                "full_3d_candidate_J_mps": j_3d,
                "exact_linear_planar_quadrature_J_mps": projected["quadrature"]},
            "quantization": {
                "body_impulse_abs_bound_mps": j_velocity_bound,
                "pitch_rate_delta_abs_bound_rad_s": RATE_LSB_RAD_S,
                "identity_residual_abs_bound_m2ps": r_bound,
                "impulse_denominator_resolved": bool(denom_robust),
                "impulse_sign_robust": bool(abs(j) > j_velocity_bound),
                "residual_sign_robust": bool(abs(residual) > r_bound)}}
        rows.append(row)
    assert len(rows) == 19
    max_quadrature_difference = max(abs(r["body_normal_impulse_mps"] -
                                        r["sensitivity"]["exact_linear_planar_quadrature_J_mps"])
                                    for r in rows)
    assert max_quadrature_difference < 1e-12
    timing = {
        "b5_to_canonical_max_abs_s": float(np.max(abs(canonical[indices[0]:indices[-1] + 2] -
                                                      raw_t[indices[0]:indices[-1] + 2]))),
        "packet_minus_b5_min_s": float(np.min(packet_t[indices[0]:indices[-1] + 2] -
                                             raw_t[indices[0]:indices[-1] + 2])),
        "packet_minus_b5_max_s": float(np.max(packet_t[indices[0]:indices[-1] + 2] -
                                             raw_t[indices[0]:indices[-1] + 2])),
        "first_raw_b5_s": rows[0]["raw_b5_start_s"],
        "last_raw_b5_s": rows[-1]["raw_b5_end_s"],
        "first_canonical_s": rows[0]["canonical_start_s"],
        "last_canonical_s": rows[-1]["canonical_end_s"]}
    angle_errors = np.array([r["theta_minus_integrated_omega_rad"] for r in rows])
    local_angles = angles[indices[0]:indices[-1] + 2]
    local_velocity = velocity[indices[0]:indices[-1] + 2]
    full_speed = np.linalg.norm(local_velocity, axis=1)
    stage_result = {name: stage(rows, name, lo, hi) for name, lo, hi in STAGES}
    for name, lo, hi in STAGES:
        selected = [r for r in rows if r["canonical_start_s"] >= lo - 1e-10
                    and r["canonical_end_s"] <= hi + 1e-10]
        assert abs(stage_result[name]["observed_delta_omega_rad_s"] -
                   sum(r["observed_delta_omega_rad_s"] for r in selected)) < 1e-14
        assert abs(stage_result[name]["body_normal_impulse_mps"] -
                   sum(r["body_normal_impulse_mps"] for r in selected)) < 1e-14
        assert abs(stage_result[name]["residual_m2ps"] -
                   sum(r["identity_residual_m2ps"] for r in selected)) < 1e-14
    previous_comparison = None
    if PREVIOUS.exists():
        old = json.loads(PREVIOUS.read_text())
        previous_comparison = {
            name: {"previous_approx_implied_mean_omega_dot_rad_s2":
                       old["windows"][name]["single_point_mean_implied_omega_dot_rad_s2"],
                   "direct_body_impulse_implied_mean_omega_dot_rad_s2":
                       stage_result[name]["fixed_aft_point_predicted_mean_omega_dot_rad_s2"]}
            for name, _, _ in STAGES}
    result = {
        "case": "L110", "source_version": parameters["source_version"],
        "primary_identity": "R_i=k²*Δω_i+l*J_b,i; k²=L²/12; J_b=mean[-sin(theta),cos(theta),0]·(Δv-[0,-g,0]*Δt). Strict fixed aft point with zero residual couple requires R_i=0.",
        "primary_inputs": "Decoded b7 full inertial velocity3, decoded Euler Z theta, decoded -b8_z omega, raw b5 native clock. No game control input, CL, drag, thrust magnitude, mass, PN, PID, or fitted coefficients enter the primary identity.",
        "interpolation_assumption": "Within a native interval, inertial velocity and theta vary linearly; sinc(Δtheta/2) gives the exact mean planar body-normal direction under that assumption. This does not reconstruct hidden subinterval game states.",
        "3d_candidate_convention": "Existing pose_rotation=R_y(heading)@R_z(pitch)@R_x(bank); full-body-normal candidate is R[:,1]. Orientation convention is an existing analysis candidate, not verified private game truth.",
        "time_provenance": timing,
        "fixed_geometry": {"length_m": LENGTH_M, "k_squared_m2": K2_M2,
                           "aft_arm_m": ARM_M,
                           "all_existing_batch_cases_have_arm_0p4m": bool(all(
                               abs(s["p"]["tail_arm_m"] - ARM_M) < 1e-12
                               for s in shots.values())),
                           "existing_batch_arm_0p2m_case_count": int(sum(
                               abs(s["p"]["tail_arm_m"] - .2) < 1e-12
                               for s in shots.values()))},
        "quantization": {
            "decoded_velocity_component_lsb_mps": VELOCITY_LSB_MPS,
            "decoded_rate_component_lsb_rad_s": RATE_LSB_RAD_S,
            "bound_interpretation": "Per interval, each velocity endpoint and rate endpoint is allowed ±half an LSB. Worst-case Δv component error is one full velocity LSB and Δomega error one full rate LSB. Pose and time held fixed. These are conditional perturbation bounds, not total replay error bars."},
        "pose_rate_check": {
            "median_abs_theta_minus_trapezoid_omega_rad": float(np.median(abs(angle_errors))),
            "max_abs_theta_minus_trapezoid_omega_rad": float(np.max(abs(angle_errors))),
            "signed_cumulative_bias_rad": float(np.sum(angle_errors)),
            "worst_interval_index": rows[int(np.argmax(abs(angle_errors)))]["index"],
            "worst_interval_raw_b5_s": [rows[int(np.argmax(abs(angle_errors)))]["raw_b5_start_s"],
                                        rows[int(np.argmax(abs(angle_errors)))]["raw_b5_end_s"]]},
        "planar_validity": {
            "max_abs_bank_deg": float(np.max(abs(np.degrees(local_angles[:, 0])))),
            "max_abs_heading_deg": float(np.max(abs(np.degrees(local_angles[:, 1])))),
            "max_abs_vz_over_v": float(np.max(abs(local_velocity[:, 2]) / full_speed)),
            "max_abs_full3d_minus_planar_J_mps": float(max(abs(
                r["sensitivity"]["full_3d_candidate_J_mps"] - r["body_normal_impulse_mps"])
                for r in rows))},
        "resolution_and_signs": {
            "interval_count": len(rows),
            "positive_J_count": int(sum(r["body_normal_impulse_mps"] > 0 for r in rows)),
            "positive_dw_count": int(sum(r["observed_delta_omega_rad_s"] > 0 for r in rows)),
            "J_and_dw_same_sign_count": int(sum(
                r["body_normal_impulse_mps"] * r["observed_delta_omega_rad_s"] > 0
                for r in rows)),
            "resolved_J_count": int(sum(r["quantization"]["impulse_denominator_resolved"]
                                        for r in rows)),
            "residual_sign_robust_count": int(sum(r["quantization"]["residual_sign_robust"]
                                                  for r in rows)),
            "max_abs_midpoint_minus_exact_J_mps": float(max(abs(
                r["sensitivity"]["midpoint_J_mps"] - r["body_normal_impulse_mps"])
                for r in rows)),
            "max_abs_trapezoid_minus_exact_J_mps": float(max(abs(
                r["sensitivity"]["trapezoid_J_mps"] - r["body_normal_impulse_mps"])
                for r in rows)),
            "max_abs_end_minus_exact_J_mps": float(max(abs(
                r["sensitivity"]["end_direction_J_mps"] - r["body_normal_impulse_mps"])
                for r in rows))},
        "stages": stage_result, "prior_approximation_comparison": previous_comparison,
        "intervals": rows,
        "limitations": ["b5 and packet time differ; exact internal sample synchronization remains unverified.",
                        "Replay state/velocity reference point is not independently confirmed as the missile center of mass.",
                        "The k²=L²/12 inertia is a provisional rod assumption, not decoded game inertia.",
                        "Velocity/rate quantization bounds exclude pose, timestamp, and decoding/model assumptions.",
                        "A failed identity audits only fixed application point plus zero residual couple; it does not identify the game's internal force or moment mechanism."],
        "verification": {"exact_formula_vs_8point_quadrature_max_abs_mps":
                             max_quadrature_difference,
                         "stage_sums_consistent": True,
                         "all_primary_values_finite": bool(np.isfinite([
                             r["body_normal_impulse_mps"] for r in rows] +
                             [r["identity_residual_m2ps"] for r in rows]).all()),
                         "input_time_range_unchanged": True}}
    assert result["verification"]["all_primary_values_finite"]
    (OUT / "results.json").write_text(json.dumps(result, indent=2,
                                                 allow_nan=False) + "\n")
    arrays = {"raw_b5_start_s": np.array([r["raw_b5_start_s"] for r in rows]),
              "raw_b5_end_s": np.array([r["raw_b5_end_s"] for r in rows]),
              "canonical_midpoint_s": np.array([r["midpoint_s"] for r in rows]),
              "observed_delta_omega_rad_s": np.array([r["observed_delta_omega_rad_s"]
                                                      for r in rows]),
              "body_normal_impulse_mps": np.array([r["body_normal_impulse_mps"] for r in rows]),
              "identity_residual_m2ps": np.array([r["identity_residual_m2ps"] for r in rows]),
              "predicted_delta_omega_rad_s": np.array([
                  r["fixed_aft_point_predicted_delta_omega_rad_s"] for r in rows]),
              "inferred_effective_arm_m": np.array([
                  np.nan if r["inferred_effective_arm_m"] is None else
                  r["inferred_effective_arm_m"] for r in rows])}
    np.savez_compressed(OUT / "intervals.npz", **arrays)
    plot(rows)
    print(json.dumps({"stages": stage_result,
                      "pose_rate_check": result["pose_rate_check"],
                      "planar_validity": result["planar_validity"],
                      "resolution_and_signs": result["resolution_and_signs"],
                      "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
