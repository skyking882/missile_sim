"""Bounded read-only observation-frame/clock audit of seven decoded game shots.

All source rows, including duplicate b5 ticks, remain in the saved arrays.
Position-derived interval durations and Hermite rate quadrature are diagnostics
only. They must not be substituted into forward-model clocks or input states.
No coefficients, axis permutations, clock shifts, or game fields are fitted.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/r771_replay_batch_20260928/source_payload.json"
OUT = ROOT / "outputs/native_observation_frames"
HZ = 48.0
RATE_LSB = 30.0 / 65535.0
NAMES = {"pi28_r771_h_110": "H110", "pi28_r771_h_27p5": "H27.5",
         "pi28_r771_l_110": "L110", "pi28_r771_l_27p5": "L27.5",
         "pi28_r771_l_55": "L55", "su_r_77_1": "H55native",
         "pi28_r771_h_clone": "H55clone"}


def signed16(a):
    return np.where(a >= 32768, a - 65536, a)


def rotations(euler):
    """Existing local decoder convention: R_y(heading) R_z(pitch) R_x(bank)."""
    bank, heading, pitch = euler.T
    cb, sb, ch, sh, cp, sp = (f(a) for a, f in
                              ((bank, np.cos), (bank, np.sin),
                               (heading, np.cos), (heading, np.sin),
                               (pitch, np.cos), (pitch, np.sin)))
    r = np.empty((len(euler), 3, 3))
    r[:, :, 0] = np.stack((ch * cp, sp, -sh * cp), axis=1)
    r[:, :, 1] = np.stack((-ch * sp * cb + sh * sb, cp * cb,
                           sh * sp * cb + ch * sb), axis=1)
    r[:, :, 2] = np.stack((ch * sp * sb + sh * cb, -cp * sb,
                           -sh * sp * sb + ch * cb), axis=1)
    return r


def quaternion_rotation(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                     (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
                     (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))))


def log_so3(r):
    angle = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2)-1)/2, -1, 1))
    skew = np.stack((r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0],
                     r[:, 1, 0]-r[:, 0, 1]), axis=1)
    factor = np.full(len(r), .5)
    regular = angle > 1e-5
    factor[regular] = angle[regular]/(2*np.sin(angle[regular]))
    return factor[:, None] * skew


def scalar_metrics(a):
    a = np.asarray(a)
    return {"n": len(a), "rmse": float(np.sqrt(np.mean(a*a))),
            "maximum_absolute": float(np.max(abs(a))),
            "median_absolute": float(np.median(abs(a)))} if len(a) else {"n": 0}


def target_mapping(packet, b5, target_age):
    positive = packet > 0
    xp, fp = np.r_[0., packet[positive]], np.r_[0., b5[positive]]
    assert np.all(np.diff(xp) > 0) and np.all(np.diff(fp) >= 0)
    mapped = np.interp(target_age, xp, fp)
    mapped[target_age < xp[0]] = target_age[target_age < xp[0]]
    mapped[target_age > xp[-1]] = target_age[target_age > xp[-1]]+fp[-1]-xp[-1]
    return mapped


def case_audit(case, meta, arrays):
    name = NAMES[case["case"]]
    rows, creation = case["missile_rows"], case["creation"]
    col = lambda key: np.asarray([float(row[key]) for row in rows])
    euler = np.column_stack([col("raw_euler_"+a) for a in "xyz"])
    poses = rotations(euler)
    b7 = np.column_stack([col("raw_b7_"+a+"_raw_u16") for a in "xyz"])
    b8 = np.column_stack([col("raw_b8_"+a+"_raw_u16") for a in "xyz"])
    velocity = signed16(b7)*4000/65535
    omega = -signed16(b8)*RATE_LSB
    position = np.column_stack([col(a+"_m") for a in "xyz"])
    times = {"b5": col("raw_b5")/HZ-float(creation["creation_time"]),
             "sf1": col("raw_sf1"),
             "packet": (col("replay_time_ms")-float(meta["creation_time_ms"]))/1000}
    body_relative = poses[:-1].transpose(0, 2, 1) @ poses[1:]
    body_increment = log_so3(body_relative)
    world_increment = log_so3(poses[1:] @ poses[:-1].transpose(0, 2, 1))
    # Express both endpoint body vectors in the interval's first body frame.
    body_mid = .5*(omega[:-1]+np.einsum("nij,nj->ni", body_relative, omega[1:]))
    world_mid = .5*(omega[:-1]+omega[1:])
    vel_mid = .5*(velocity[:-1]+velocity[1:])
    dpos = np.diff(position, axis=0)
    scalar_increment = np.diff(np.unwrap(euler[:, 2]))
    # Euler pitch derivative for the two explicit rate-frame hypotheses.
    body_pitch_dot = np.sin(euler[:, 0])*omega[:, 1]+np.cos(euler[:, 0])*omega[:, 2]
    world_pitch_dot = np.sin(euler[:, 1])*omega[:, 0]+np.cos(euler[:, 1])*omega[:, 2]
    reference_mask = (np.diff(times["b5"]) > 0) & (times["b5"][1:] <= 3.5)
    clocks = {}
    for clock, t in times.items():
        dt = np.diff(t)
        valid = dt > 0
        mask = reference_mask & valid
        scalar_error = scalar_increment-world_mid[:, 2]*dt
        body_error = body_increment-body_mid*dt[:, None]
        world_error = world_increment-world_mid*dt[:, None]
        pos_error = dpos-vel_mid*dt[:, None]
        cumulative_pos = np.vstack((np.zeros(3), np.cumsum(pos_error, axis=0)))
        clocks[clock] = {
            "nonpositive_interval_count": int(np.sum(~valid)),
            "scalar_pitch_increment_error_deg_0_3p5": scalar_metrics(np.degrees(scalar_error[mask])),
            "body_log_pitch_increment_error_deg_0_3p5": scalar_metrics(np.degrees(body_error[mask, 2])),
            "body_log_vector_rate_rmse_rad_s_xyz_0_3p5": np.sqrt(np.mean((body_error[mask]/dt[mask, None])**2, axis=0)).tolist(),
            "world_log_vector_rate_rmse_rad_s_xyz_0_3p5": np.sqrt(np.mean((world_error[mask]/dt[mask, None])**2, axis=0)).tolist(),
            "positive_b8_body_log_vector_rate_rmse_rad_s_xyz_0_3p5": np.sqrt(np.mean((body_increment[mask]/dt[mask, None]+body_mid[mask])**2, axis=0)).tolist(),
            "position_interval_residual_norm_m_0_3p5": scalar_metrics(np.linalg.norm(pos_error[mask], axis=1)),
            "position_interval_residual_norm_m_full": scalar_metrics(np.linalg.norm(pos_error[valid], axis=1)),
            "cumulative_position_residual_terminal_xyz_m": cumulative_pos[-1].tolist(),
        }
        for key, a in (("age_s", t), ("scalar_pitch_increment_error_rad", scalar_error),
                       ("body_log_increment_error_rad_xyz", body_error),
                       ("world_log_increment_error_rad_xyz", world_error),
                       ("position_increment_error_m_xyz", pos_error)):
            arrays[f"{name}__{clock}__{key}"] = a
    dt = np.diff(times["b5"])
    scalar_error = scalar_increment-world_mid[:, 2]*dt
    worst = int(np.argmax(np.where(reference_mask, abs(scalar_error), -1)))
    effective_dt = np.sum(dpos*vel_mid, axis=1)/np.sum(vel_mid*vel_mid, axis=1)
    effective_ticks = np.rint(effective_dt*HZ)
    clock_offset = effective_ticks-np.rint(dt*HZ)
    fixed_tick_pos_error = dpos-vel_mid*(effective_ticks/HZ)[:, None]
    gaps = {}
    for gap in sorted(set(np.rint(dt[reference_mask]*HZ).astype(int))):
        m = reference_mask & (abs(dt*HZ-gap) < 1e-5)
        gaps[str(gap)] = scalar_metrics(np.degrees(scalar_error[m]))
    # Hermite quadrature is a predeclared diagnostic, not a corrected rate feed.
    # Duplicate ticks stay in all originals; only last-of-tie stencil is used here.
    unique = np.r_[np.diff(times["b5"]) > 0, True]
    tu, qu, theta_u = times["b5"][unique], omega[unique, 2], euler[unique, 2]
    derivative = np.gradient(qu, tu, edge_order=2)
    du = np.diff(tu)
    hermite_integral = .5*du*(qu[:-1]+qu[1:])+du**2*(derivative[:-1]-derivative[1:])/12
    hermite_error = np.diff(theta_u)-hermite_integral
    hermite_mask = tu[1:] <= 3.5
    q0 = quaternion_rotation(creation["u16_1"])
    qc_pitch = math.asin(float(np.clip(q0[1, 0], -1, 1)))
    source_up = np.asarray(creation["u12_5"], float)
    creation_pos = np.asarray(creation["starting_pos"], float)
    creation_vel = np.asarray(creation["u12_2"], float)
    first_mid_vel = .5*(creation_vel+velocity[0])
    creation_displacement = position[0]-creation_pos
    creation_effective_dt = float(creation_displacement@first_mid_vel/(first_mid_vel@first_mid_vel))
    ttarget = np.array([float(row["time_ms"])/1000 for row in case["target_rows"]])
    ptarget = np.array([[float(row[a+"_m"]) for a in "xyz"] for row in case["target_rows"]])
    mapped = target_mapping(times["packet"], times["b5"], ttarget-float(meta["creation_time_ms"])/1000)
    mt = np.diff(mapped)
    target_valid = mt > 0
    original_target_velocity = np.diff(ptarget, axis=0)/np.diff(ttarget)[:, None]
    mapped_target_velocity = np.diff(ptarget, axis=0)[target_valid]/mt[target_valid, None]
    for key, a in (("euler_rad_xyz", euler), ("velocity_mps_xyz", velocity),
                   ("omega_candidate_body_rad_s_xyz", omega), ("position_m_xyz", position),
                   ("pose_body_log_increment_rad_xyz", body_increment),
                   ("pose_world_log_increment_rad_xyz", world_increment),
                   ("diagnostic_position_effective_dt_s", effective_dt),
                   ("diagnostic_position_nearest_tick_count", effective_ticks),
                   ("target_packet_absolute_time_s", ttarget),
                   ("target_mapped_b5_age_s", mapped)):
        arrays[f"{name}__{key}"] = a
    ties = np.flatnonzero(dt <= 0)
    return {
        "case": case["case"], "samples": len(rows), "intervals": len(rows)-1,
        "creation_quaternion_norm": float(np.linalg.norm(creation["u16_1"])),
        "creation_quaternion_body_up_minus_u12_5_max_abs": float(np.max(abs(q0[:, 1]-source_up))),
        "creation_quaternion_pitch_rad": qc_pitch,
        "first_euler_pitch_minus_creation_deg": math.degrees(float(euler[0, 2]-qc_pitch)),
        "creation_to_first_observation": {
            "displacement_m_xyz": creation_displacement.tolist(),
            "position_effective_dt_s_diagnostic": creation_effective_dt,
            "first_clock_age_s": {clock: float(t[0]) for clock, t in times.items()},
            "position_residual_m_xyz_by_clock": {clock: (creation_displacement-first_mid_vel*t[0]).tolist() for clock, t in times.items()},
        },
        "pose_orthogonality_max_abs": float(np.max(abs(poses.transpose(0, 2, 1)@poses-np.eye(3)))),
        "bank_heading_absolute_max_deg": np.max(abs(np.degrees(euler[:, :2])), axis=0).tolist(),
        "bank_heading_range_deg": np.ptp(np.degrees(euler[:, :2]), axis=0).tolist(),
        "decoded_b8_xyz_max_abs_rad_s": np.max(abs(omega), axis=0).tolist(),
        "b8_x_nonzero_count": int(np.count_nonzero(b8[:, 0])),
        "body_log_minus_scalar_pitch_increment_deg_0_3p5": scalar_metrics(np.degrees((body_increment[:, 2]-scalar_increment)[reference_mask])),
        "body_pitch_dot_minus_scalar_q_max_abs_rad_s": float(np.max(abs(body_pitch_dot-omega[:, 2]))),
        "world_pitch_dot_minus_scalar_q_max_abs_rad_s": float(np.max(abs(world_pitch_dot-omega[:, 2]))),
        "b5_minus_sf1_age_max_abs_s": float(np.max(abs(times["b5"]-times["sf1"]))),
        "packet_minus_b5_age_range_s": [float(np.min(times["packet"]-times["b5"])), float(np.max(times["packet"]-times["b5"]))],
        "clocks": clocks, "b5_gap_tick_counts_0_3p5": gaps,
        "diagnostic_hermite_pitch_increment_error_deg_0_3p5": scalar_metrics(np.degrees(hermite_error[hermite_mask])),
        "diagnostic_nearest_position_tick_minus_b5_tick_counts_full": {str(int(k)): int(np.sum(clock_offset == k)) for k in sorted(set(clock_offset))},
        "diagnostic_nearest_tick_position_residual_norm_m_full": scalar_metrics(np.linalg.norm(fixed_tick_pos_error, axis=1)),
        "diagnostic_nearest_position_tick_pitch_increment_error_deg_0_3p5": scalar_metrics(np.degrees((scalar_increment-world_mid[:, 2]*effective_ticks/HZ)[reference_mask])),
        "target_clock": {"samples": len(ttarget), "nonpositive_mapped_intervals": int(np.sum(~target_valid)),
                         "packet_dt_over_mapped_age_dt_range": [float(np.min(np.diff(ttarget)[target_valid]/mt[target_valid])), float(np.max(np.diff(ttarget)[target_valid]/mt[target_valid]))],
                         "secant_velocity_difference_norm_mps": scalar_metrics(np.linalg.norm(mapped_target_velocity-original_target_velocity[target_valid], axis=1))},
        "duplicate_b5_intervals": [{"source_indices": [int(i), int(i+1)], "b5_age_s": float(times["b5"][i]),
                                    "packet_dt_s": float(np.diff(times["packet"])[i]),
                                    "position_change_m": dpos[i].tolist(), "euler_change_deg": np.degrees(euler[i+1]-euler[i]).tolist()} for i in ties],
        "worst_b5_scalar_interval": {
            "source_indices": [worst, worst+1], "age_s": times["b5"][[worst, worst+1]].tolist(),
            "raw_rows": rows[worst:worst+2],
            "position_effective_dt_s_diagnostic": float(effective_dt[worst]),
            "clocks": {clock: {"dt_s": float(np.diff(t)[worst]),
                                "pitch_increment_error_deg": float(np.degrees(scalar_increment[worst]-world_mid[worst, 2]*np.diff(t)[worst])),
                                "position_increment_residual_m_xyz": (dpos[worst]-vel_mid[worst]*np.diff(t)[worst]).tolist()}
                       for clock, t in times.items()},
            "body_log_minus_scalar_increment_deg": float(np.degrees(body_increment[worst, 2]-scalar_increment[worst])),
            "rate_quantization_conservative_increment_bound_deg": float(math.degrees(RATE_LSB*dt[worst])),
            "65535_to_65536_scale_change_deg": float(math.degrees(abs(world_mid[worst, 2]*dt[worst])/65536)),
        },
    }


def main():
    payload = json.loads(SOURCE.read_text())
    compact = json.loads((SOURCE.parent/"batch_compact.json").read_text())
    assert len(payload["cases"]) == len(compact["shots"]) == 7
    arrays, records = {}, {}
    for case, meta in zip(payload["cases"], compact["shots"]):
        assert case["case"] == meta["actual_case"] and len(case["missile_rows"]) == meta["sample_count"]
        records[NAMES[case["case"]]] = case_audit(case, meta, arrays)
    assert sum(r["samples"] for r in records.values()) == 1153
    assert all(np.isfinite(a).all() for a in arrays.values())
    result = {
        "source": str(SOURCE.relative_to(ROOT)), "samples_retained": 1153,
        "predeclared_tests": ["Existing Euler convention Ry(heading) Rz(pitch) Rx(bank), creation quaternion xyzw cross-check against u12_5 body up",
                              "signed16 b7*4000/65535; signed16 b8*30/65535 with global positive and negative signs",
                              "Finite SO3 body and world logs; endpoint body vectors resolved into the first interval frame",
                              "b5/48-creation_time, raw_sf1 and packet-release clocks, without shifts or scale fits",
                              "Native gap groups; fixed Hermite rate quadrature diagnostic with numpy gradient slopes",
                              "Nearest 48 Hz displacement-effective interval diagnostic, never an accepted clock",
                              "Target packet secants versus same position samples on existing mapped b5 age"],
        "records": records,
        "conclusions": ["For these almost planar shots, full 3D conversion does not materially repair the pitch increment residual.",
                        "The body-rate hypothesis matches bank/heading increments substantially better than the world-rate hypothesis; both keep the same pitch outlier.",
                        "Worst pitch residual intervals coincide with a one-tick discrepancy between displacement and the native b5 interval. Some are repaired by sf1 or packet time, but no alternative clock repairs all intervals.",
                        "Endpoint trapezoid nonclosure alone is not proof of inconsistent continuous observations. Sparse rate curvature and mixed snapshot/clock phase remain observational possibilities.",
                        "Raw parser/engine conventions are not established by this diagnostic. Available source paths point to Windows capture files; no local replay parser source was found in the project or targeted Documents search.",
                        "Keep original rows and clocks. Do not feed position-derived times, pose-derived angular rates or Hermite interpolation into a forward model."],
        "model_or_decoder_changed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"native_arrays.npz", **arrays)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/r771_observation_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex="col")
    colors = {"b5": "#7c5599", "sf1": "#278384", "packet": "#d88a25"}
    for j, name in enumerate(("H110", "L110")):
        t = arrays[name+"__b5__age_s"]
        mid = .5*(t[:-1]+t[1:])
        for clock, color in colors.items():
            axs[0, j].plot(mid, np.degrees(arrays[f"{name}__{clock}__scalar_pitch_increment_error_rad"]), ".-", ms=3, lw=.8, color=color, label=clock)
            axs[1, j].plot(mid, arrays[f"{name}__{clock}__position_increment_error_m_xyz"][:, 0], ".-", ms=3, lw=.8, color=color)
        axs[0, j].set_title(name+": same decoded states, three unshifted clocks")
        for ax in axs[:, j]:
            ax.set_xlim(0, 3.5)
            ax.axhline(0, color="black", lw=.6)
            ax.grid(alpha=.2)
        axs[1, j].set_xlabel("Original b5 flight age of interval midpoint (s)")
    axs[0, 0].set_ylabel("Pitch increment minus endpoint rate integral (deg)")
    axs[1, 0].set_ylabel("X displacement minus endpoint velocity integral (m)")
    axs[0, 0].legend()
    fig.suptitle("Observation closure audit: local time phase cannot be repaired by 3D pitch conversion", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"clock_closure.png", dpi=160)
    plt.close(fig)
    print(json.dumps({n: {"samples": r["samples"], "scalar_max_deg": r["clocks"]["b5"]["scalar_pitch_increment_error_deg_0_3p5"]["maximum_absolute"],
                          "body_3d_correction_max_deg": r["body_log_minus_scalar_pitch_increment_deg_0_3p5"]["maximum_absolute"],
                          "position_tick_minus_b5_counts": r["diagnostic_nearest_position_tick_minus_b5_tick_counts_full"]}
                      for n, r in records.items()}, indent=2))


if __name__ == "__main__":
    main()
