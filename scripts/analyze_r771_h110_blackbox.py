"""Bounded, single-replay R-77-1 h110 kinematic and conditional identification.

Reads the unmodified transferred source payload. Writes only analysis artifacts
alongside it. This is a conditional analysis, not a free-flight or PID model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r771_replay_20260927_h110"
SOURCE = OUT / "source_payload.json"
G = 9.80665


def signed16(raw):
    return np.where(raw >= 32768, raw - 65536, raw)


def rotations(euler):
    bank, heading, pitch = euler.T
    cb, sb = np.cos(bank), np.sin(bank)
    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # Columns of R_y(heading) @ R_z(pitch) @ R_x(bank); world Y is up.
    r = np.empty((len(euler), 3, 3))
    r[:, :, 0] = np.stack((ch * cp, sp, -sh * cp), axis=1)
    r[:, :, 1] = np.stack((-ch * sp * cb + sh * sb,
                           cp * cb,
                           sh * sp * cb + ch * sb), axis=1)
    r[:, :, 2] = np.stack((ch * sp * sb + sh * cb,
                           -cp * sb,
                           -sh * sp * sb + ch * cb), axis=1)
    return r


def rotation_vector(delta):
    """SO(3) logarithm; result is axis angle in the first frame."""
    trace = np.trace(delta, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    skew = np.stack((delta[:, 2, 1] - delta[:, 1, 2],
                     delta[:, 0, 2] - delta[:, 2, 0],
                     delta[:, 1, 0] - delta[:, 0, 1]), axis=1)
    factor = np.empty_like(angle)
    small = angle < 1e-5
    factor[small] = 0.5
    factor[~small] = angle[~small] / (2 * np.sin(angle[~small]))
    return skew * factor[:, None]


def metrics(actual, predicted):
    residual = np.asarray(actual) - np.asarray(predicted)
    return {"rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "correlation": float(np.corrcoef(actual, predicted)[0, 1])
            if np.std(actual) > 0 and np.std(predicted) > 0 else None}


def isa_density(height):
    # Transparent standard-atmosphere approximation, not replay telemetry.
    temp = 288.15 - 0.0065 * height
    pressure = 101325.0 * (temp / 288.15) ** 5.2558797
    return pressure / (287.05 * temp)


def condition(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[0] / singular[-1]), [float(v) for v in singular]


def fit_integral(t, other_t, alpha_drive, aero_n, q, fit_bounds, hold_bounds, steps=4):
    # Each row is an observed finite change in q versus integrals over four
    # original sample intervals. Adjacent windows overlap; holdout is a
    # separate, contiguous time block, not an independent missile shot.
    rows, response, middle = [], [], []
    for i in range(len(t) - steps):
        j = i + steps
        dt = np.diff(t[i:j + 1])
        features = np.stack((alpha_drive[i:j + 1], aero_n[i:j + 1], q[i:j + 1]), axis=1)
        integrated = np.sum((features[:-1] + features[1:]) * dt[:, None] / 2, axis=0)
        assert np.all(np.isfinite(integrated))
        rows.append(integrated)
        response.append(q[j] - q[i])
        middle.append((t[i] + t[j]) / 2)
    X, y, middle = np.array(rows), np.array(response), np.array(middle)
    # The *entire* native four-interval window must lie in its block on both
    # clocks. This also fixes identical source rows for the clock comparison.
    starts, ends = t[:-steps], t[steps:]
    other_starts, other_ends = other_t[:-steps], other_t[steps:]
    fit = ((starts >= fit_bounds[0]) & (ends <= fit_bounds[1]) &
           (other_starts >= fit_bounds[0]) & (other_ends <= fit_bounds[1]))
    hold = ((starts >= hold_bounds[0]) & (ends <= hold_bounds[1]) &
            (other_starts >= hold_bounds[0]) & (other_ends <= hold_bounds[1]))
    if fit.sum() < 8 or hold.sum() < 4:
        return None
    fit_ids = np.flatnonzero(fit)
    hold_ids = np.flatnonzero(hold)
    fit_samples = set().union(*(set(range(int(i), int(i) + steps + 1)) for i in fit_ids))
    hold_samples = set().union(*(set(range(int(i), int(i) + steps + 1)) for i in hold_ids))
    sample_intersection = fit_samples & hold_samples
    assert not sample_intersection
    scale = np.linalg.norm(X[fit], axis=0)
    x_scaled = X / scale
    coef_scaled, _, rank, _ = np.linalg.lstsq(x_scaled[fit], y[fit], rcond=None)
    beta = coef_scaled / scale
    pred = X @ beta
    return {"beta": beta, "midtime": middle, "observed_dq": y,
            "predicted_dq": pred, "fit_mask": fit, "hold_mask": hold,
            "fit": metrics(y[fit], pred[fit]), "holdout": metrics(y[hold], pred[hold]),
            "fit_zero_baseline_rmse": float(np.sqrt(np.mean(y[fit]**2))),
            "hold_zero_baseline_rmse": float(np.sqrt(np.mean(y[hold]**2))),
            "scaled_feature_condition": condition(x_scaled[fit])[0],
            "scaled_feature_singular_values": condition(x_scaled[fit])[1],
            "rank": int(rank), "fit_windows": int(fit.sum()),
            "hold_windows": int(hold.sum()), "integration_native_intervals": steps,
            "fit_window_ids": [int(v) for v in fit_ids],
            "hold_window_ids": [int(v) for v in hold_ids],
            "fit_native_sample_ids": sorted(fit_samples),
            "hold_native_sample_ids": sorted(hold_samples),
            "fit_hold_sample_intersection_count": len(sample_intersection),
            "fit_actual_start_end_s": [float(np.min(starts[fit])), float(np.max(ends[fit]))],
            "hold_actual_start_end_s": [float(np.min(starts[hold])), float(np.max(ends[hold]))]}


def main():
    payload = json.loads(SOURCE.read_text())
    mc = payload["missile_columns"]
    tc = payload["target_columns"]
    a = np.array(payload["missile_rows"], dtype=float)
    target = np.array(payload["target_rows"], dtype=float)
    assert a.shape == (165, 18) and target.shape == (51, 7)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(target))
    assert np.all(np.diff(a[:, mc.index("replay_time_ms")]) > 0)
    assert np.all(np.diff(a[:, mc.index("raw_b5")]) > 0)
    assert np.all(np.diff(a[:, mc.index("raw_sf1")]) > 0)
    assert np.all(np.diff(target[:, tc.index("time_ms")]) > 0)
    col = lambda name: a[:, mc.index(name)]

    creation = payload["creation"]
    t_b5 = col("raw_b5") / 48 - creation["creation_time"]
    t_sf1 = col("raw_sf1")
    t_packet = (col("replay_time_ms") - col("replay_time_ms")[0]) / 1000
    pos = np.stack((col("x_m"), col("y_m"), col("z_m")), axis=1)
    angles = np.stack((col("raw_euler_x"), col("raw_euler_y"), col("raw_euler_z")), axis=1)
    R = rotations(angles)
    ortho_error = float(np.max(np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3))))
    b7_raw = np.stack([col(f"raw_b7_{axis}_raw_u16") for axis in "xyz"], axis=1)
    velocity = signed16(b7_raw) * (4000 / 65535)
    speed = np.linalg.norm(velocity, axis=1)
    fwd, body_up = R[:, :, 0], R[:, :, 1]
    # Full 3D incidence from forward and velocity; sign set by body +Y.
    vhat = velocity / speed[:, None]
    alpha = np.arctan2(-np.sum(vhat * body_up, axis=1), np.sum(vhat * fwd, axis=1))
    theta = np.arctan2(fwd[:, 1], np.hypot(fwd[:, 0], fwd[:, 2]))
    gamma = np.arctan2(velocity[:, 1], np.hypot(velocity[:, 0], velocity[:, 2]))
    alpha_planar = theta - gamma

    clock_checks = {}
    for name, time in (("b5", t_b5), ("sf1", t_sf1), ("packet", t_packet)):
        dt = np.diff(time)
        integrated = np.vstack((np.zeros(3), np.cumsum((velocity[:-1] + velocity[1:]) * dt[:, None] / 2, axis=0)))
        displacement = pos - pos[0]
        residual = displacement - integrated
        # Position and b7 velocity can have sub-frame phase jitter; compare
        # cumulative displacement, including the vector terminal residual.
        clock_checks[name] = {"elapsed_s": float(time[-1] - time[0]),
            "interval_s_min_median_max": [float(v) for v in np.quantile(dt, [0, .5, 1])],
            "position_integration_rms_m": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            "terminal_position_residual_xyz_m": [float(v) for v in residual[-1]],
            "terminal_residual_norm_m": float(np.linalg.norm(residual[-1]))}

    b8_raw = np.stack([col(f"raw_b8_{axis}_raw_u16") for axis in "xyz"], axis=1)
    b8 = signed16(b8_raw) * (30 / 65535)
    relative_body = np.einsum("nji,njk->nik", R[:-1], R[1:])
    relative_world = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    pose_body_rate = rotation_vector(relative_body) / np.diff(t_b5)[:, None]
    pose_world_rate = rotation_vector(relative_world) / np.diff(t_b5)[:, None]
    b8_interval = (b8[:-1] + b8[1:]) / 2
    first2 = ((t_b5[:-1] + t_b5[1:]) / 2) <= 2
    rate_checks = {}
    for axis, k in zip("xyz", range(3)):
        rate_checks[axis] = {"raw_nonzero": int(np.count_nonzero(b8[:, k])),
            "body_first2_negated_b8": metrics(pose_body_rate[first2, k], -b8_interval[first2, k]),
            "world_first2_negated_b8": metrics(pose_world_rate[first2, k], -b8_interval[first2, k]),
            "body_full_negated_b8": metrics(pose_body_rate[:, k], -b8_interval[:, k]),
            "unnegated_body_first2_correlation": metrics(pose_body_rate[first2, k], b8_interval[first2, k])["correlation"]}
    # Test only a local one-sample shift. Different conventions are reported,
    # not silently selected as fitting degrees of freedom.
    pitch_lags = {}
    for lag in (-1, 0, 1):
        idx = np.arange(len(pose_body_rate)) + lag
        mask = first2 & (idx >= 0) & (idx < len(b8_interval))
        pitch_lags[str(lag)] = metrics(pose_body_rate[mask, 2], -b8_interval[idx[mask], 2])
    pitch_clock_checks = {}
    for name, time in (("b5", t_b5), ("sf1", t_sf1), ("packet", t_packet)):
        derived = rotation_vector(relative_body) / np.diff(time)[:, None]
        pitch_clock_checks[name] = metrics(derived[first2, 2], -b8_interval[first2, 2])

    # Native-interval observed acceleration from b7. Force terms are resolved
    # in the upward trajectory-normal direction; this retains thrust-normal.
    vhat_mid = (vhat[:-1] + vhat[1:]) / 2
    vhat_mid /= np.linalg.norm(vhat_mid, axis=1)[:, None]
    up_mid = np.array([0.0, 1.0, 0.0]) - vhat_mid[:, 1, None] * vhat_mid
    up_mid /= np.linalg.norm(up_mid, axis=1)[:, None]
    fwd_mid = (fwd[:-1] + fwd[1:]) / 2
    fwd_mid /= np.linalg.norm(fwd_mid, axis=1)[:, None]
    mass0 = float(payload["optional_existing_parent_parameters"]["rocket"]["mass"])
    mass_end = float(payload["optional_existing_parent_parameters"]["rocket"]["massEnd"])
    fire_time = float(payload["optional_existing_parent_parameters"]["rocket"]["timeFire"])
    thrust_n = float(payload["optional_existing_parent_parameters"]["rocket"]["force"])
    gravity = np.array([0.0, -G, 0.0])
    rho = isa_density((pos[:-1, 1] + pos[1:, 1]) / 2)
    qdyn = .5 * rho * ((speed[:-1] + speed[1:]) / 2) ** 2
    rocket = payload["optional_existing_parent_parameters"]["rocket"]
    area = math.pi * (float(rocket["caliber"]) / 2) ** 2 * float(rocket["wingAreaMult"])
    alpha_mid = (alpha[:-1] + alpha[1:]) / 2
    q_node = -b8[:, 2]  # Directly decoded native state, not pose differentiated.

    def clock_dynamics(time, other_time):
        tmid = (time[:-1] + time[1:]) / 2
        accel = np.diff(velocity, axis=0) / np.diff(time)[:, None]
        mass = mass0 - (mass0 - mass_end) * np.clip(tmid / fire_time, 0, 1)
        thrust_vector = (thrust_n / mass)[:, None] * fwd_mid
        measured_up = np.sum((accel - gravity) * up_mid, axis=1)
        thrust_up = np.sum(thrust_vector * up_mid, axis=1)
        aero_up = measured_up - thrust_up
        closure14 = qdyn * area / mass * 14 * alpha_mid
        # Propulsion-clock sensitivity, with all other source values fixed.
        mass_shift = mass0 - (mass0 - mass_end) * np.clip((tmid + 1/48) / fire_time, 0, 1)
        aero_shift = measured_up - np.sum((thrust_n / mass_shift)[:, None] * fwd_mid * up_mid, axis=1)
        aero_node = np.r_[aero_up[0], (aero_up[:-1] + aero_up[1:]) / 2, aero_up[-1]]
        drive_interval = qdyn * area / mass
        drive_node = np.r_[drive_interval[0],
                           (drive_interval[:-1] + drive_interval[1:]) / 2,
                           drive_interval[-1]]
        alpha_drive = drive_node * alpha
        fit = fit_integral(time, other_time, alpha_drive, aero_node, q_node,
                           (.30, 1.35), (1.45, 2.45))
        later_fit = fit_integral(time, other_time, alpha_drive, aero_node, q_node,
                                 (1.45, 2.45), (2.55, 3.55))
        return {"time": time, "tmid": tmid, "mass": mass,
                "measured_up": measured_up, "thrust_up": thrust_up,
                "aero_up": aero_up, "aero_shift": aero_shift,
                "closure14": closure14, "fit": fit, "later_fit": later_fit}

    clock_data = {"b5": clock_dynamics(t_b5, t_packet),
                  "packet": clock_dynamics(t_packet, t_b5)}
    b5_data, packet_data = clock_data["b5"], clock_data["packet"]
    assert b5_data["fit"] is not None and packet_data["fit"] is not None
    assert b5_data["later_fit"] is not None and packet_data["later_fit"] is not None
    for phase in ("fit", "later_fit"):
        assert b5_data[phase]["fit_window_ids"] == packet_data[phase]["fit_window_ids"]
        assert b5_data[phase]["hold_window_ids"] == packet_data[phase]["hold_window_ids"]
    tmid = b5_data["tmid"]
    measured_up, thrust_up = b5_data["measured_up"], b5_data["thrust_up"]
    aero_up, aero_shift, closure14 = b5_data["aero_up"], b5_data["aero_shift"], b5_data["closure14"]
    clock_force_difference = aero_up - packet_data["aero_up"]
    first2_ids = np.flatnonzero(first2)
    largest_clock_difference_id = int(first2_ids[np.argmax(np.abs(clock_force_difference[first2]))])

    target_pos0 = target[0, 1:4]
    target_time_ms = target[:, tc.index("time_ms")]
    target_pos_at_creation = np.array([np.interp(col("replay_time_ms")[0], target_time_ms, target[:, j])
                                       for j in (1, 2, 3)])
    target_pos_at_last = np.array([np.interp(col("replay_time_ms")[-1], target_time_ms, target[:, j])
                                   for j in (1, 2, 3)])

    def fit_summary(fit, fit_bounds, hold_bounds):
        return {"fit_bounds_s": list(fit_bounds), "holdout_bounds_s": list(hold_bounds),
                "coefficient_Calpha_Cn_Cq": [float(v) for v in fit["beta"]],
                "fit": fit["fit"], "holdout": fit["holdout"],
                "fit_zero_baseline_rmse": fit["fit_zero_baseline_rmse"],
                "hold_zero_baseline_rmse": fit["hold_zero_baseline_rmse"],
                "rank": fit["rank"], "scaled_feature_condition": fit["scaled_feature_condition"],
                "scaled_feature_singular_values": fit["scaled_feature_singular_values"],
                "fit_windows": fit["fit_windows"], "hold_windows": fit["hold_windows"],
                "integration_native_intervals": fit["integration_native_intervals"],
                "fit_actual_start_end_s": fit["fit_actual_start_end_s"],
                "hold_actual_start_end_s": fit["hold_actual_start_end_s"],
                "fit_window_ids": fit["fit_window_ids"],
                "hold_window_ids": fit["hold_window_ids"],
                "fit_native_sample_ids": fit["fit_native_sample_ids"],
                "hold_native_sample_ids": fit["hold_native_sample_ids"],
                "fit_hold_sample_intersection_count": fit["fit_hold_sample_intersection_count"]}

    def clock_force_summary(d):
        mask = d["tmid"] <= 2
        spike_mask = (d["tmid"] >= 1.7) & (d["tmid"] <= 1.9)
        spike_ids = np.flatnonzero(spike_mask)
        spike_id = int(spike_ids[np.argmin(d["aero_up"][spike_mask])])
        return {"engine_time_zero_plus_one_count_aero_difference_rmse_mps2":
                    float(np.sqrt(np.mean((d["aero_up"]-d["aero_shift"])**2))),
                "aero_up_mps2_min_max_first2": [float(np.min(d["aero_up"][mask])),
                                                  float(np.max(d["aero_up"][mask]))],
                "measured_up_mps2_min_max_first2": [float(np.min(d["measured_up"][mask])),
                                                      float(np.max(d["measured_up"][mask]))],
                "thrust_up_mps2_min_max_first2": [float(np.min(d["thrust_up"][mask])),
                                                    float(np.max(d["thrust_up"][mask]))],
                "interval_1p7_1p9_lowest_aero": {"interval_id": spike_id,
                    "midtime_s": float(d["tmid"][spike_id]),
                    "aero_up_mps2": float(d["aero_up"][spike_id])},
                "closure_14_first2": metrics(d["aero_up"][mask], d["closure14"][mask])}

    result = {
        "source": "source_payload.json; original numeric cells read without modification",
        "validation": {"missile_rows": int(len(a)), "missile_columns": int(a.shape[1]),
            "target_rows": int(len(target)), "target_columns": int(target.shape[1]),
            "all_finite": True, "time_columns_strictly_increasing": True,
            "rotation_orthogonality_max_abs": ortho_error},
        "geometry": {"missile_creation_to_target_t0_m": float(np.linalg.norm(target_pos0 - np.array(creation["starting_pos"]))),
            "target_t0_minus_missile_creation_xyz_m": [float(v) for v in target_pos0 - np.array(creation["starting_pos"])],
            "target_t0_to_missile_first_sample_m": float(np.linalg.norm(target_pos0 - pos[0])),
            "target_interpolated_at_missile_creation_packet_xyz_m": [float(v) for v in target_pos_at_creation],
            "actual_initial_range_interpolated_m": float(np.linalg.norm(target_pos_at_creation - np.array(creation["starting_pos"]))),
            "actual_initial_relative_xyz_m": [float(v) for v in target_pos_at_creation - np.array(creation["starting_pos"])],
            "target_interpolated_at_last_missile_packet_xyz_m": [float(v) for v in target_pos_at_last],
            "last_sample_same_packet_time_separation_m_not_miss_distance": float(np.linalg.norm(target_pos_at_last - pos[-1])),
            "missile_first_last_altitude_m": [float(pos[0, 1]), float(pos[-1, 1])],
            "missile_max_altitude_m": float(np.max(pos[:, 1])),
            "max_climb_from_first_m": float(np.max(pos[:, 1])-pos[0, 1]),
            "speed_mps_first_last": [float(speed[0]), float(speed[-1])],
            "pitch_deg_max_first2": float(np.max(np.degrees(theta[t_b5 <= 2]))),
            "alpha_deg_min_max_first2": [float(v) for v in (np.min(np.degrees(alpha[t_b5 <= 2])), np.max(np.degrees(alpha[t_b5 <= 2])))],
            "candidate_speed_rmse_mps": metrics(speed, col("speed_mps_candidate"))["rmse"],
            "candidate_aoa_rmse_deg": metrics(np.degrees(alpha), col("aoa_deg_estimate_zero_wind"))["rmse"],
            "candidate_gamma_rmse_deg": metrics(np.degrees(gamma), col("flight_path_angle_deg_estimate"))["rmse"]},
        "clock": {"b5_minus_sf1_s_min_max": [float(np.min(t_b5-t_sf1)), float(np.max(t_b5-t_sf1))],
            "packet_minus_b5_s_min_max": [float(np.min(col("replay_time_ms")/1000-col("raw_b5")/48)),
                                             float(np.max(col("replay_time_ms")/1000-col("raw_b5")/48))],
            "integration": clock_checks,
            "interpretation": "b5/48-creation_time selected as candidate simulation clock; its count does not establish controller update rate"},
        "rate": {"candidate_decode": "signed16(raw_b8)*30/65535 rad/s; measured physical rate agrees after sign reversal",
            "pose_derivation": "SO(3) log(R_i^T R_{i+1}) / delta t, body interval rate",
            "axis_comparisons": rate_checks,
            "pitch_one_interval_lag_comparisons": pitch_lags,
            "pitch_clock_comparisons": pitch_clock_checks,
            "native_sample_interval_count": int(len(pose_body_rate))},
        "forces": {"definition": "N_aero/m = (delta_v/delta_t - gravity - T/m*body_forward) dot upward wind-normal; native intervals",
            "engine_assumption": "source force constant for timeFire=7s, mass linear from mass to massEnd, time zero at missile creation",
            "packet_instead_of_b5_clock_aero_difference_rmse_first2_mps2": float(np.sqrt(np.mean((aero_up[first2]-packet_data["aero_up"][first2])**2))),
            "largest_first2_clock_force_difference": {
                "interval_id": largest_clock_difference_id,
                "b5_midtime_s": float(tmid[largest_clock_difference_id]),
                "b5_interval_s": float(np.diff(t_b5)[largest_clock_difference_id]),
                "packet_interval_s": float(np.diff(t_packet)[largest_clock_difference_id]),
                "b5_aero_mps2": float(aero_up[largest_clock_difference_id]),
                "packet_aero_mps2": float(packet_data["aero_up"][largest_clock_difference_id])},
            "reference_area_m2": area,
            "density": "ISA troposphere at replay altitude; atmospheric assumption",
            "clock_comparisons": {name: clock_force_summary(d) for name, d in clock_data.items()},
            "closure_14_meaning": "force closure diagnostic of current empirical baseline, not retuned coefficient or engine truth"},
        "conditional_identification": {
            "equation": "Delta q = C_alpha integral((q_dynamic*S/m)*alpha dt) + C_N integral((N_aero/m) dt) + C_q integral(q dt)",
            "q_source": "negative decoded b8_z at native state nodes; pose-derived angular rate is verification only",
            "interpretation": "At most coefficient combinations after eliminating unknown fin input; no individual b_delta, b_alpha, b_q, a_delta are identified",
            "same_native_window_ids_across_clocks": True,
            "clock_comparisons": {name: {
                "first_fit": fit_summary(d["fit"], (.30, 1.35), (1.45, 2.45)),
                "later_fit": fit_summary(d["later_fit"], (1.45, 2.45), (2.55, 3.55))}
                for name, d in clock_data.items()},
            "limitation": "Observed N is fed to the relation, so holdout is a conditional time block, not autonomous or cross-shot validation."}}

    fig, axs = plt.subplots(2, 3, figsize=(16, 9.3), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, .055, 1, .94))
    ax = axs[0, 0]
    ax.plot(t_b5, np.degrees(theta), label="body pitch θ", lw=1.6)
    ax.plot(t_b5, np.degrees(gamma), label="path angle γ", lw=1.5)
    ax.plot(t_b5, np.degrees(alpha), label="signed incidence α", lw=1.5)
    ax.set(xlim=(0, 2.5), ylabel="degrees", title="Native pose and velocity (initial 2.5 s)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axs[0, 1]
    ax.plot(tmid, pose_body_rate[:, 2], label="pose-derived body z", lw=1.6)
    ax.plot(tmid, -b8_interval[:, 2], label="negative b8 z decode, endpoint mean", lw=1.1, alpha=.8)
    ax.set(xlim=(0, 2.5), ylabel="rad/s", title="Pitch rate: SO(3) versus raw b8")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axs[0, 2]
    ax.plot(tmid, measured_up, label="observed non-gravity +Y", lw=1.3)
    ax.plot(tmid, thrust_up, label="assumed thrust +Y", lw=1.3)
    ax.plot(tmid, aero_up, label="derived aero +Y", lw=1.5)
    ax.plot(tmid, closure14, label="14/rad baseline lift", lw=1.2, ls="--")
    ax.set(xlim=(0, 2.5), ylabel="m/s²", title="Force decomposition, b5 clock (native intervals)")
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axs[1, 0]
    ax.plot(t_b5, speed, label="decoded b7 speed")
    ax2 = ax.twinx()
    ax2.plot(t_b5, pos[:, 1]-pos[0, 1], color="tab:orange", label="height above first sample")
    ax.set(xlabel="time since creation (s)", ylabel="speed (m/s)", title="Full trajectory context")
    ax2.set(ylabel="height gain (m)")
    ax.grid(alpha=.25)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    ax = axs[1, 1]
    b5_fit, packet_fit = b5_data["fit"], packet_data["fit"]
    ax.plot(b5_fit["midtime"], b5_fit["observed_dq"], label="observed −b8_z Δq", lw=1.5, color="black")
    ax.plot(b5_fit["midtime"], b5_fit["predicted_dq"], label="conditional b5", lw=1.25)
    ax.plot(packet_fit["midtime"], packet_fit["predicted_dq"], label="conditional packet", lw=1.25)
    ax.axvspan(.30, 1.35, color="tab:blue", alpha=.08, label="fit bounds")
    ax.axvspan(1.45, 2.45, color="tab:orange", alpha=.10, label="holdout bounds")
    ax.set(xlim=(0, 2.55), xlabel="time since creation/first packet (s)",
           ylabel="Δq over native four intervals (rad/s)",
           title="Conditional rotation: same windows, two clocks")
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axs[1, 2]
    ax.plot(tmid, aero_up, label="derived aero +Y", lw=1.3)
    ax.plot(tmid, closure14, label="14/rad baseline", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="time since creation (s)", ylabel="m/s²", title="Aero force, b5 clock, full flight")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("War Thunder R-77-1 custom finsLatAccel=110: one replay, conditional diagnostics", fontsize=14)
    fig.text(.5, .015, "Native samples; no smoothing or deleted spike. Thrust/mass timing and ISA density are assumptions; both fits receive observed force.",
             ha="center", fontsize=9)
    plot_path = OUT / "r771_h110_blackbox.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    json_path = OUT / "r771_h110_blackbox_results.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"FIGURE {plot_path}")
    print(f"RESULTS {json_path}")


if __name__ == "__main__":
    main()
