"""Native-interval, measured-state L110 rotation audit.

Each game interval starts independently from its decoded state. The only
control input is a previously reconstructed, held command sequence, not a
decoded game fin angle. No PID or PN runs in these local propagations.
"""
from __future__ import annotations

import json
import math

import numpy as np

import check_r771_measured_feedback as prior

mid, fit, model = prior.mid, prior.fit, prior.model
OUT = fit.OUT / "l110_local_rotation"
PREVIOUS = prior.OUT / "trajectories.npz"
END_LIMIT_S = 1.4
STEP = prior.STEP
STATE_KEYS = mid.STATE_KEYS
ARMS = ("model", "replay")


def align_native_time(raw_t):
    tick = round(raw_t * mid.HZ)
    aligned = tick / mid.HZ
    return aligned if abs(raw_t - aligned) < 1e-7 else raw_t


def controls(saved, arm):
    """Read one value per saved 48 Hz hold interval, with no interpolation."""
    t, u = saved[arm + "__t"], saved[arm + "__u"]
    first, last = prior.CUT, round(mid.HORIZON * mid.HZ) - 1
    result = {}
    for tick in range(first, last + 1):
        k = int(round((tick / mid.HZ - prior.START) / STEP))
        assert abs(t[k] - tick / mid.HZ) < 1e-11
        result[tick] = float(u[k])
        # Check saved trajectories implement the same zero-order hold.
        assert np.max(abs(u[k:k + round(1 / (mid.HZ * STEP))] - result[tick])) < 1e-12
    return result


def partition(lo, hi):
    boundaries = [lo]
    for k in range(math.floor(lo * mid.HZ) + 1, math.ceil(hi * mid.HZ)):
        t = k / mid.HZ
        if lo + 1e-12 < t < hi - 1e-12:
            boundaries.append(t)
    boundaries.append(hi)
    return list(zip(boundaries[:-1], boundaries[1:]))


def rhs(t, y, u, shot, coeff, reference):
    s = y[:5]
    control, restoring, damping, load_g, v = prior.acceleration(
        t, s, u, shot, coeff, reference)
    gamma_dot = model.G * (load_g - math.cos(s[2])) / v
    return np.array((s[1], control + restoring + damping, gamma_dot,
                     v * math.cos(s[2]), v * math.sin(s[2]),
                     control, restoring, damping))


def integrate_interval(shot, coeff, reference, s0, lo, hi, command, *, step=STEP):
    y = np.r_[s0, 0., 0., 0.].astype(float)
    held = []
    for a, b in partition(lo, hi):
        tick = math.floor((a + 1e-10) * mid.HZ)
        u = command[tick]
        held.append((a, b, tick, u))
        count = math.ceil((b - a) / step - 1e-9)
        h = (b - a) / count
        for j in range(count):
            t = a + j * h
            k1 = rhs(t, y, u, shot, coeff, reference)
            k2 = rhs(t + h / 2, y + h * k1 / 2, u, shot, coeff, reference)
            k3 = rhs(t + h / 2, y + h * k2 / 2, u, shot, coeff, reference)
            k4 = rhs(t + h, y + h * k3, u, shot, coeff, reference)
            y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y, held


def measured_state_moment(shot, coeff, reference, s0, s1, lo, hi, command):
    """Quadrature of old moment law on linearly interpolated observed states."""
    contributions = np.zeros(3)
    for a, b in partition(lo, hi):
        tick = math.floor((a + 1e-10) * mid.HZ)
        u = command[tick]
        count = math.ceil((b - a) / STEP - 1e-9)
        h = (b - a) / count
        for j in range(count):
            t = a + j * h

            def parts(at):
                fraction = (at - lo) / (hi - lo)
                state = s0 + fraction * (s1 - s0)
                return np.array(prior.acceleration(at, state, u, shot, coeff,
                                                   reference)[:3])

            contributions += h * (parts(t) + 4 * parts(t + h / 2) +
                                  parts(t + h)) / 6
    return contributions


def closure(shot, s0, s1, lo, hi):
    """Assumption audit: one fixed aft application point, no separate couple."""
    p = shot["p"]
    mid_t = (lo + hi) / 2
    middle = (s0 + s1) / 2
    gamma0, gamma1 = s0[2], s1[2]
    v0 = float(np.interp(lo, shot["replay"]["t"], shot["replay"]["speed2"]))
    v1 = float(np.interp(hi, shot["replay"]["t"], shot["replay"]["speed2"]))
    n_g = ((v0 + v1) / 2 * (gamma1 - gamma0) /
           (model.G * (hi - lo)) + (math.cos(gamma0) + math.cos(gamma1)) / 2)
    mass, thrust = model.mass_thrust(mid_t, p)
    alpha = middle[0] - middle[2]
    body_normal = (mass * model.G * n_g - thrust * math.sin(alpha)) / math.cos(alpha)
    inertia = mass * p["length_m"]**2 / 12
    implied_dot = -p["tail_arm_m"] * body_normal / inertia
    return {"kinematic_specific_force_g": n_g,
            "midpoint_mass_kg": mass, "midpoint_thrust_n": thrust,
            "midpoint_alpha_rad": alpha, "inferred_normal_body_n": body_normal,
            "provisional_inertia_kg_m2": inertia,
            "single_aft_point_implied_omega_dot_rad_s2": implied_dot}


def windows(rows):
    definitions = (("downstroke", 29 / 48, 37 / 48),
                   ("early_recovery", 37 / 48, 44 / 48),
                   ("positive_peak", 44 / 48, 52 / 48))
    out = {}
    for name, a, b in definitions:
        selected = [r for r in rows if r["aligned_start_s"] >= a - 1e-10
                    and r["aligned_end_s"] <= b + 1e-10]
        assert selected and abs(selected[0]["aligned_start_s"] - a) < 1e-10
        assert abs(selected[-1]["aligned_end_s"] - b) < 1e-10
        duration = b - a
        observed = sum(r["observed_delta_omega_rad_s"] for r in selected)
        entry = {"start_s": a, "end_s": b, "interval_count": len(selected),
                 "observed_delta_omega_rad_s": observed,
                 "observed_mean_omega_dot_rad_s2": observed / duration,
                 "mean_kinematic_specific_force_g": sum(
                     r["closure"]["kinematic_specific_force_g"] * r["duration_s"]
                     for r in selected) / duration,
                 "mean_inferred_body_normal_n": sum(
                     r["closure"]["inferred_normal_body_n"] * r["duration_s"]
                     for r in selected) / duration,
                 "opposite_sign_intervals_for_one_aft_point": int(sum(
                     r["observed_mean_omega_dot_rad_s2"] *
                     r["closure"]["single_aft_point_implied_omega_dot_rad_s2"] < 0
                     for r in selected)),
                 "single_point_mean_implied_omega_dot_rad_s2":
                     sum(r["closure"]["single_aft_point_implied_omega_dot_rad_s2"] *
                         r["duration_s"] for r in selected) / duration}
        for arm in ARMS:
            entry[arm] = {
                "local_delta_omega_rad_s": sum(r[arm]["local_delta_omega_rad_s"]
                                               for r in selected),
                "local_mean_omega_dot_rad_s2": sum(
                    r[arm]["local_delta_omega_rad_s"] for r in selected) / duration,
                "observed_state_law_mean_omega_dot_rad_s2": sum(
                    r[arm]["moment_on_interpolated_measured_state"]["total_delta_omega_rad_s"]
                    for r in selected) / duration}
        out[name] = entry
    return out


def plot(rows):
    plt = fit.plt
    x = np.array([r["midpoint_s"] for r in rows])
    observed_dot = np.array([r["observed_mean_omega_dot_rad_s2"] for r in rows])
    observed_delta = np.array([r["observed_delta_omega_rad_s"] for r in rows])
    implied = np.array([r["closure"]["single_aft_point_implied_omega_dot_rad_s2"]
                        for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                             layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, observed_dot, "ko-", ms=4, lw=1.1, label="Game interval")
    ax.set_ylabel("Mean pitch angular acceleration (rad/s²)")
    ax = axes[0, 1]
    ax.plot(x, observed_delta, "ko-", ms=4, lw=1.1, label="Game interval")
    ax.set_ylabel("Change in pitch rate per interval (rad/s)")
    ax = axes[1, 0]
    ax.axhline(0, color=".5", lw=.8)
    ax.set_ylabel("AoA endpoint error (deg)")
    ax = axes[1, 1]
    ax.plot(x, observed_dot, "ko-", ms=4, lw=1.1, label="Game interval")
    ax.plot(x, implied, color="#803a96", marker="d", ms=3, lw=1.2,
            label="One aft force, rod inertia")
    ax.set_ylabel("Pitch angular acceleration (rad/s²)")
    for arm, color, label in (("model", "#d88330", "orange command"),
                              ("replay", "#2779b5", "blue command")):
        local_dot = np.array([r[arm]["local_mean_omega_dot_rad_s2"] for r in rows])
        local_delta = np.array([r[arm]["local_delta_omega_rad_s"] for r in rows])
        measured_dot = np.array([
            r[arm]["moment_on_interpolated_measured_state"]["total_delta_omega_rad_s"] /
            r["duration_s"] for r in rows])
        alpha_error = np.array([r[arm]["endpoint_error"]["alpha_deg"] for r in rows])
        axes[0, 0].plot(x, local_dot, color=color, marker="o", ms=3, lw=1.5,
                        label=f"Local reset, {label}")
        axes[0, 0].plot(x, measured_dot, color=color, ls="--", lw=1.2,
                        label=f"On observed states, {label}")
        axes[0, 1].plot(x, local_delta, color=color, marker="o", ms=3,
                        lw=1.5, label=f"Local reset, {label}")
        axes[1, 0].plot(x, alpha_error, color=color, marker="o", ms=3,
                        lw=1.5, label=f"Local reset, {label}")
    for ax in axes.flat:
        for boundary in (37 / 48, 44 / 48, 52 / 48):
            ax.axvline(boundary, color=".78", lw=.7)
        ax.axhline(0, color=".6", lw=.6)
        ax.set_xlim(prior.START, rows[-1]["aligned_end_s"])
        ax.set_xlabel("Time from launch (s)")
        ax.grid(alpha=.16)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("L110 native-interval rotation: reset to game state each interval\n"
                 "Inputs are reconstructed 48 Hz commands, not decoded fin angles")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, raw, parent = prior.surface.previous.load()
    shot = fit.make_shot(raw["L110"])
    assert parent["source_version"] == "2.59.0.28"
    coeff = np.array([float(old[0]), float(old[1]),
                      float(old[1]) * shot["p"]["tail_arm_m"] /
                      shot["initial_speed_mps"]])
    assert abs(coeff[2] - .039577498843208075) < 1e-12
    v0 = shot["initial_speed_mps"]
    reference = .5 * model.isa_density(shot["initial"][4]) * v0**2 / shot["p"]["mass0_kg"]
    saved = np.load(PREVIOUS)
    command = {arm: controls(saved, arm) for arm in ARMS}
    t_raw = np.asarray(shot["replay"]["t"])
    t_aligned = np.array([align_native_time(float(t)) for t in t_raw])
    relevant = [i for i in range(len(t_raw) - 1)
                if t_aligned[i] >= prior.START - 1e-10 and
                   t_aligned[i] < END_LIMIT_S]
    assert relevant and relevant[0] == 15 and relevant[-1] == 33
    rows = []
    max_component_residual = 0.
    max_reset_delta = 0.
    max_zoh_delta = 0.
    for i in relevant:
        lo, hi = map(float, t_aligned[i:i + 2])
        s0 = mid.measured_state(shot, float(t_raw[i]))
        s1 = mid.measured_state(shot, float(t_raw[i + 1]))
        exact = np.array([shot["replay"][k][i] for k in STATE_KEYS])
        max_reset_delta = max(max_reset_delta, float(np.max(abs(s0 - exact))))
        observed_delta = float(s1[1] - s0[1])
        record = {"native_start_s": float(t_raw[i]),
                  "native_end_s": float(t_raw[i + 1]),
                  "aligned_start_s": lo, "aligned_end_s": hi,
                  "midpoint_s": (lo + hi) / 2, "duration_s": hi - lo,
                  "observed_delta_omega_rad_s": observed_delta,
                  "observed_mean_omega_dot_rad_s2": observed_delta / (hi - lo),
                  "observed_delta_alpha_deg": math.degrees(
                      (s1[0] - s1[2]) - (s0[0] - s0[2])),
                  "closure": closure(shot, s0, s1, lo, hi)}
        for arm in ARMS:
            y, held = integrate_interval(shot, coeff, reference, s0, lo, hi,
                                         command[arm])
            predicted_delta = float(y[1] - s0[1])
            contribution = y[5:8]
            residual = abs(predicted_delta - float(np.sum(contribution)))
            max_component_residual = max(max_component_residual, residual)
            for a, b, tick, u in held:
                max_zoh_delta = max(max_zoh_delta, abs(u - command[arm][tick]))
            prescribed = measured_state_moment(shot, coeff, reference, s0, s1,
                                               lo, hi, command[arm])
            record[arm] = {
                "held_command_segments": [{"start_s": a, "end_s": b,
                                            "tick": tick, "u_rad": u}
                                           for a, b, tick, u in held],
                "local_delta_omega_rad_s": predicted_delta,
                "local_mean_omega_dot_rad_s2": predicted_delta / (hi - lo),
                "local_omega_dot_bias_rad_s2": (predicted_delta - observed_delta) / (hi - lo),
                "component_delta_omega_rad_s": dict(zip(
                    ("control", "restoring", "damping"), map(float, contribution))),
                "endpoint_error": {
                    "theta_deg": math.degrees(float(y[0] - s1[0])),
                    "omega_rad_s": float(y[1] - s1[1]),
                    "gamma_deg": math.degrees(float(y[2] - s1[2])),
                    "alpha_deg": math.degrees(float((y[0] - y[2]) - (s1[0] - s1[2]))),
                    "x_m": float(y[3] - s1[3]), "y_m": float(y[4] - s1[4])},
                "moment_on_interpolated_measured_state": {
                    "component_delta_omega_rad_s": dict(zip(
                        ("control", "restoring", "damping"), map(float, prescribed))),
                    "total_delta_omega_rad_s": float(np.sum(prescribed)),
                    "mean_omega_dot_rad_s2": float(np.sum(prescribed) / (hi - lo))}}
            assert np.isfinite(y).all() and np.isfinite(prescribed).all()
        rows.append(record)
    assert max_reset_delta == 0 and max_component_residual < 1e-12
    assert max_zoh_delta == 0
    timing_offsets = t_aligned[relevant[0]:relevant[-1] + 2] - t_raw[relevant[0]:relevant[-1] + 2]
    summary = windows(rows)
    result = {
        "case": "L110", "source_version": parent["source_version"],
        "scope": "Independent native-interval rollouts of the existing orange candidate; no PN/PID update and no coefficient fit.",
        "control_provenance": "The orange and blue 48 Hz commands are previously reconstructed model outputs, not decoded game fin commands. ZOH values are taken from saved trajectories, not interpolated.",
        "time_alignment": {"rule": "Raw native stamp within 1e-7 s of 48 Hz tick aligned to exact tick for propagation; raw values retained per interval.",
                           "max_abs_alignment_s": float(np.max(abs(timing_offsets))),
                           "first_raw_s": float(t_raw[relevant[0]]),
                           "first_aligned_s": float(t_aligned[relevant[0]]),
                           "last_raw_s": float(t_raw[relevant[-1] + 1]),
                           "last_aligned_s": float(t_aligned[relevant[-1] + 1])},
        "fixed_parameters": {"ku": float(coeff[0]), "ka": float(coeff[1]),
                             "kw": float(coeff[2]),
                             "tail_arm_m": shot["p"]["tail_arm_m"],
                             "length_m": shot["p"]["length_m"],
                             "normal_force_slope_per_rad": 14.},
        "force_moment_assumption": "If all aerodynamic normal force acts at one aft point x=-l and no independent couple exists, M=-l*N_body. The inferred N_body=(m*g*n-T*sin(alpha))/cos(alpha), with n from replay V*delta_gamma/g/dt+cos(gamma), assumes axial aerodynamic force contributes negligibly to wind-normal acceleration. Axial force along body at an on-axis point has zero planar torque. I=m*L^2/12 is provisional, not decoded game inertia.",
        "interpolated_measured_state_assumption": "Theta, omega, gamma, x, y vary linearly within each native interval only for the moment-law quadrature; their subinterval values are not decoded replay data.",
        "windows": summary, "intervals": rows,
        "verification": {"native_interval_count": len(rows),
                         "exact_measured_reset_max_delta": max_reset_delta,
                         "saved_ZOH_command_max_delta": max_zoh_delta,
                         "max_component_sum_residual_rad_s": max_component_residual,
                         "all_intervals_finite": True}}
    (OUT / "results.json").write_text(json.dumps(result, indent=2,
                                                 allow_nan=False) + "\n")
    arrays = {"native_start_s": np.array([r["native_start_s"] for r in rows]),
              "native_end_s": np.array([r["native_end_s"] for r in rows]),
              "aligned_start_s": np.array([r["aligned_start_s"] for r in rows]),
              "aligned_end_s": np.array([r["aligned_end_s"] for r in rows]),
              "midpoint_s": np.array([r["midpoint_s"] for r in rows]),
              "observed_omega_dot": np.array([r["observed_mean_omega_dot_rad_s2"]
                                              for r in rows]),
              "one_point_implied_omega_dot": np.array([
                  r["closure"]["single_aft_point_implied_omega_dot_rad_s2"]
                  for r in rows])}
    for arm in ARMS:
        arrays[arm + "_local_omega_dot"] = np.array([
            r[arm]["local_mean_omega_dot_rad_s2"] for r in rows])
        arrays[arm + "_observed_state_law_omega_dot"] = np.array([
            r[arm]["moment_on_interpolated_measured_state"]["mean_omega_dot_rad_s2"]
            for r in rows])
        arrays[arm + "_endpoint_alpha_error_deg"] = np.array([
            r[arm]["endpoint_error"]["alpha_deg"] for r in rows])
    np.savez_compressed(OUT / "intervals.npz", **arrays)
    plot(rows)
    print(json.dumps({"windows": summary, "verification": result["verification"],
                      "time_alignment": result["time_alignment"]}, indent=2))


if __name__ == "__main__":
    main()
