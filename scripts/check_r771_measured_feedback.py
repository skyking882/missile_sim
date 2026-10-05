"""Fixed-request L110 diagnostic with replay-kinematic PID feedback.

This reuses the previously saved orange (geometry damping) candidate. Its
independently calibrated moment law is retained only to isolate feedback; this
is not an implementation of the intended one-surface force/moment plant.
"""
from __future__ import annotations

import json
import math

import numpy as np

import replay_r771_single_surface as surface

mid = surface.mid
fit = surface.fit
model = surface.model
OUT = fit.OUT / "l110_measured_feedback"
CUT = 29
START = CUT / mid.HZ
STEP = mid.rate.STEP
PREFIX = "restart_29__geometry_only__"
SAVED = surface.OUT / "trajectories.npz"
STATE = mid.STATE_KEYS
FIELDS = (*mid.KEYS, "alpha", "a_req_g", "plant_feedback_g",
          "controller_feedback_g", "pid_error_g", "omega_dot",
          "omega_dot_control", "omega_dot_restoring", "omega_dot_damping")


def replay_feedback(shot):
    """Average normal specific force over each native interval, at its center."""
    r = shot["replay"]
    t, gamma, speed = (np.asarray(r[k], float) for k in ("t", "gamma", "speed2"))
    dt = np.diff(t)
    assert np.all(dt > 0) and np.all(speed > 0)
    centered_t = (t[:-1] + t[1:]) / 2
    average_g = ((speed[:-1] + speed[1:]) * np.diff(gamma) /
                 (2 * model.G * dt) +
                 (np.cos(gamma[:-1]) + np.cos(gamma[1:])) / 2)
    assert np.isfinite(average_g).all()
    assert centered_t[0] < START and centered_t[-1] > mid.HORIZON
    return centered_t, average_g, dt


def fixed_requests(shot, saved):
    """PN/request evaluated once on saved orange states at all PID ticks."""
    tick = np.arange(CUT, round(mid.HORIZON * mid.HZ))
    times = tick / mid.HZ
    saved_t = saved[PREFIX + "t"]
    values = []
    for t in times:
        j = int(round((t - START) / STEP))
        assert abs(saved_t[j] - t) < 1e-11
        state = np.array([saved[PREFIX + k][j] for k in STATE])
        values.append(model.request(t, state, shot["replay"], shot["p"])["a_req_g"])
    return tick, np.asarray(values)


def acceleration(t, state, u, shot, coeff, reference):
    theta, omega, gamma, _, y = state
    p = shot["p"]
    terms = model.force_terms(t, state, shot["replay"], p)
    v = terms["v"]
    a = (.5 * model.isa_density(y) * v * v / terms["mass"]) / reference
    b = a * shot["initial_speed_mps"] / v
    ku, ka, kw = coeff
    return (a * ku * u, -a * ka * (theta - gamma), -b * kw * omega,
            terms["actual_specific_force_g"], v)


def simulate(shot, coeff, initial, pid_initial, tick, requests, feedback_t,
             feedback_g, *, mode):
    assert mode in ("model", "replay")
    p = shot["p"]
    r = shot["replay"]
    state = initial.copy()
    pid = tuple(pid_initial)
    v0 = shot["initial_speed_mps"]
    reference = .5 * model.isa_density(shot["initial"][4]) * v0 * v0 / p["mass0_kg"]
    rows = []
    tick_trace = []

    def rhs(t, s, u):
        control, restoring, damping, plant_g, v = acceleration(t, s, u, shot, coeff, reference)
        gamma_dot = model.G * (plant_g - math.cos(s[2])) / v
        return np.array([s[1], control + restoring + damping, gamma_dot,
                         v * math.cos(s[2]), v * math.sin(s[2])])

    def append_row(t, s, u, request_g, sensed_g, error_g):
        control, restoring, damping, plant_g, _ = acceleration(t, s, u, shot, coeff, reference)
        rows.append([t, *s, u, pid[0], s[0] - s[2], request_g, plant_g,
                     sensed_g, error_g, control + restoring + damping,
                     control, restoring, damping])

    for i, k in enumerate(tick):
        t = k / mid.HZ
        end = (k + 1) / mid.HZ
        plant_g = model.force_terms(t, state, r, p)["actual_specific_force_g"]
        sensed_g = (plant_g if mode == "model" else
                    float(np.interp(t, feedback_t, feedback_g)))
        error_g = float(requests[i] - sensed_g)
        gain = model.interpolate_table(end, p["time_gain"])
        assert gain > 0
        raw_u, pid = fit.pid_step_with_dt(error_g, pid, p, 1 / mid.HZ)
        u = gain * raw_u
        tick_trace.append({"t": t, "request_g": float(requests[i]),
                           "plant_g": plant_g, "sensed_g": sensed_g,
                           "error_g": error_g, "raw_u": raw_u, "u": u,
                           "I": pid[0], "D_memory": pid[2]})
        append_row(t, state, u, requests[i], sensed_g, error_g)
        count = math.ceil((end - t) / STEP - 1e-9)
        h = (end - t) / count
        for j in range(count):
            now = t + j * h
            d1 = rhs(now, state, u)
            d2 = rhs(now + h / 2, state + h * d1 / 2, u)
            d3 = rhs(now + h / 2, state + h * d2 / 2, u)
            d4 = rhs(now + h, state + h * d3, u)
            state += h * (d1 + 2 * d2 + 2 * d3 + d4) / 6
            if j < count - 1:
                append_row(now + h, state, u, requests[i], sensed_g, error_g)
    append_row(mid.HORIZON, state, u, requests[-1], sensed_g, error_g)
    a = np.asarray(rows)
    assert np.isfinite(a).all()
    return {key: a[:, j] for j, key in enumerate(FIELDS)}, tick_trace


def extremum(t, v, lo, hi, kind):
    ids = np.flatnonzero((t >= lo - 1e-10) & (t <= hi + 1e-10))
    assert len(ids) > 2
    j = ids[np.argmin(v[ids]) if kind == "min" else np.argmax(v[ids])]
    return {"time_s": float(t[j]), "value": float(v[j]),
            "at_window_edge": bool(j == ids[0] or j == ids[-1])}


def zero_cross_after_valley(t, omega, valley_t, hi=1.25):
    for i in range(len(t) - 1):
        if t[i] < valley_t - 1e-10 or t[i + 1] > hi:
            continue
        if omega[i] <= 0 < omega[i + 1]:
            return float(t[i] - omega[i] * (t[i + 1] - t[i]) /
                         (omega[i + 1] - omega[i]))
    return None


def events(curve):
    t, omega, alpha = (curve[k] for k in ("t", "omega", "alpha"))
    valley = extremum(t, omega, .6, .84, "min")
    return {"omega_valley": valley,
            "omega_recovery_zero_s": zero_cross_after_valley(t, omega, valley["time_s"]),
            "omega_next_peak": extremum(t, omega, .85, 1.18, "max"),
            "alpha_valley": extremum(t, alpha * 180 / math.pi, .8, 1.02, "min")}


def observed_mean_normal(shot, lo, hi):
    r = shot["replay"]
    t = np.r_[lo, r["t"][(r["t"] > lo) & (r["t"] < hi)], hi]
    gamma = np.interp(t, r["t"], r["gamma"])
    speed = np.interp(t, r["t"], r["speed2"])
    turning = np.sum(.5 * (speed[:-1] + speed[1:]) * np.diff(gamma))
    gravity = np.sum(.5 * (np.cos(gamma[:-1]) + np.cos(gamma[1:])) * np.diff(t))
    return float(turning / (model.G * (hi - lo)) + gravity / (hi - lo))


def means(curve, lo=.8, hi=.875):
    t = curve["t"]
    grid = np.linspace(lo, hi, 301)
    result = {"omega_dot_rad_s2": float((np.interp(hi, t, curve["omega"]) -
                                         np.interp(lo, t, curve["omega"])) / (hi - lo))}
    for key in ("a_req_g", "plant_feedback_g", "controller_feedback_g", "u",
                "omega_dot_control", "omega_dot_restoring", "omega_dot_damping"):
        result[key] = float(np.trapezoid(np.interp(grid, t, curve[key]), grid) / (hi - lo))
    return result


def plot(shot, curves, feedback_t, feedback_g, tick, requests):
    plt = fit.plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, layout="constrained")
    obs = shot["observed"]
    specs = (("omega", 1., "Pitch rate (rad/s)"),
             ("alpha", 180 / math.pi, "Angle of attack (deg)"),
             ("controller_feedback_g", 1., "Specific load (g)"),
             ("u", 180 / math.pi, "Control output (deg)"),
             ("omega_dot", 1., "Angular acceleration (rad/s²)"),
             ("omega_dot_control", 1., "Angular acceleration components (rad/s²)"))
    styles = (("model", "#d88330", "Saved orange model feedback"),
              ("replay", "#2779b5", "Replay-motion feedback"))
    for ax, (key, scale, label) in zip(axes.flat, specs):
        for tag, color, legend in styles:
            c = curves[tag]
            ax.plot(c["t"], c[key] * scale, color=color, lw=1.8,
                    label=legend if key != "omega_dot_control" else f"{tag} control")
        if key in ("omega", "alpha"):
            keep = (obs["t"] >= START) & (obs["t"] <= 1.4)
            ax.plot(obs["t"][keep], obs[key][keep] * scale, "ko", ms=3.5,
                    zorder=5, label="Game replay")
        if key == "controller_feedback_g":
            mask = (feedback_t >= START) & (feedback_t <= 1.4)
            ax.plot(feedback_t[mask], feedback_g[mask], "ko", ms=2.3,
                    label="Native interval estimates")
            ax.plot(tick / mid.HZ, requests, color=".35", ls="--", lw=1.2,
                    label="Frozen request")
        if key == "omega_dot_control":
            for tag, color, _ in styles:
                c = curves[tag]
                ax.plot(c["t"], c["omega_dot_restoring"], color=color,
                        ls="--", lw=1., label=f"{tag} restoring")
                ax.plot(c["t"], c["omega_dot_damping"], color=color,
                        ls=":", lw=1., label=f"{tag} damping")
        ax.set(xlim=(.6, 1.4), ylabel=label, xlabel="Time from launch (s)")
        ax.axvline(START, color=".5", lw=.7)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("L110: fixed request and PID; feedback substitution at 0.6042 s\n"
                 "Kinematic load is offline, centered on native replay intervals")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for ax, (key, scale, label) in zip(axes, specs[:2]):
        ax.plot(obs["t"], obs[key] * scale, "ko", ms=2.4, label="Game replay")
        for tag, color, legend in styles:
            c = curves[tag]
            ax.plot(c["t"], c[key] * scale, color=color, lw=1.6, label=legend)
        ax.set(ylabel=label, xlim=(START, mid.HORIZON))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Time from launch (s)")
    fig.suptitle("L110 continuation of the same fixed-request diagnostic")
    fig.savefig(OUT / "continuation.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coeff_old, raw, parent = surface.previous.load()
    shot = fit.make_shot(raw["L110"])
    assert parent["source_version"] == "2.59.0.28"
    assert not shot["p"]["loft_enabled"]
    coeff = np.array([float(coeff_old[0]), float(coeff_old[1]),
                      float(coeff_old[1]) * shot["p"]["tail_arm_m"] /
                      shot["initial_speed_mps"]])
    assert abs(coeff[2] - .039577498843208075) < 1e-12
    saved = np.load(SAVED)
    tick, requests = fixed_requests(shot, saved)
    feedback_t, feedback_g, native_dt = replay_feedback(shot)
    initial = mid.measured_state(shot, START)
    pid_initial = mid.replay_pid_before(shot, CUT)
    curves = {}
    trace = {}
    for mode in ("model", "replay"):
        curves[mode], trace[mode] = simulate(
            shot, coeff, initial, pid_initial, tick, requests,
            feedback_t, feedback_g, mode=mode)

    saved_error = max(float(np.max(abs(curves["model"][key] -
                                       saved[PREFIX + key])))
                      for key in (*mid.KEYS, "alpha"))
    assert saved_error < 1e-10
    assert np.array_equal(curves["model"]["t"], curves["replay"]["t"])
    assert np.array_equal(curves["model"]["a_req_g"], curves["replay"]["a_req_g"])
    assert np.array_equal(np.array([curves[m][k][0] for k in STATE for m in ("model", "replay")]).reshape(-1, 2)[:, 0], initial)
    assert np.array_equal(np.array([curves[m][k][0] for k in STATE for m in ("model", "replay")]).reshape(-1, 2)[:, 1], initial)
    component_error = max(float(np.max(abs(c["omega_dot"] - c["omega_dot_control"] -
                                           c["omega_dot_restoring"] - c["omega_dot_damping"])))
                          for c in curves.values())
    assert component_error < 1e-12

    obs = shot["observed"]
    observed = events(obs)
    observed["mean_0p800_0p875"] = {
        "omega_dot_rad_s2": float((np.interp(.875, obs["t"], obs["omega"]) -
                                   np.interp(.8, obs["t"], obs["omega"])) / .075),
        "kinematic_specific_force_g": observed_mean_normal(shot, .8, .875)}
    # The identical PID memory gives a measurable D change at the first swap.
    baseline_first, injected_first = trace["model"][0], trace["replay"][0]
    delta_error = injected_first["error_g"] - baseline_first["error_g"]
    derivative_weight = min(1., (1 / mid.HZ) / (1e-9 + 1 / mid.HZ))
    d_first = shot["p"]["kd"] * derivative_weight * mid.HZ * delta_error
    p_first = shot["p"]["kp"] * delta_error
    i_first = shot["p"]["ki"] * delta_error / mid.HZ
    result = {
        "case": "L110", "source_version": parent["source_version"],
        "purpose": "Diagnostic intervention on saved orange candidate, not a new game plant or one-surface model validation.",
        "fixed": {"PID": {k: shot["p"][k] for k in ("kp", "ki", "kd", "integral_limit")},
                  "moment_coefficients": dict(zip(("ku", "ka", "geometry_kw"), coeff.tolist())),
                  "start_time_s": START, "initial_state": dict(zip(STATE, initial.tolist())),
                  "initial_PID_memory": list(pid_initial), "request_at_48hz": "Evaluated once from saved orange states; identical for both arms.",
                  "speed_and_target": "Replay inputs; original absolute launch age and normalization."},
        "replay_feedback": {
            "definition": "Interval mean (V_i+V_i+1)*(gamma_i+1-gamma_i)/(2*g*dt) + (cos(gamma_i)+cos(gamma_i+1))/2, assigned to interval midpoint then linearly interpolated to PID ticks.",
            "provenance": "Offline from decoded replay velocity direction and planar speed; not decoded internal game acceleration feedback.",
            "native_intervals": int(len(native_dt)),
            "native_dt_min_s": float(np.min(native_dt)),
            "native_dt_max_s": float(np.max(native_dt)),
            "native_dt_median_s": float(np.median(native_dt)),
            "early_window_interval_g_min_max": [float(np.min(feedback_g[(feedback_t >= .6) & (feedback_t <= 1.4)])),
                                                float(np.max(feedback_g[(feedback_t >= .6) & (feedback_t <= 1.4)]))],
            "supported_tick_range_s": [float(feedback_t[0]), float(feedback_t[-1])]},
        "first_tick_substitution": {
            "model_feedback_g": baseline_first["sensed_g"],
            "replay_feedback_g": injected_first["sensed_g"],
            "error_change_g": delta_error,
            "predicted_raw_PID_change_rad": p_first + i_first + d_first,
            "PID_change_components_rad": {"P": p_first, "I": i_first, "D": d_first},
            "actual_raw_PID_change_rad": injected_first["raw_u"] - baseline_first["raw_u"],
            "held_command_change_rad": injected_first["u"] - baseline_first["u"]},
        "observed": observed,
        "arms": {mode: {"events": events(c), "mean_0p800_0p875": means(c),
                        "first_tick": trace[mode][0], "last_tick": trace[mode][-1],
                        "max_abs_alpha_deg": float(np.max(abs(c["alpha"])) * 180 / math.pi)}
                 for mode, c in curves.items()},
        "verification": {"saved_orange_max_abs_difference": saved_error,
                         "request_sequence_identical": True,
                         "initial_state_and_PID_identical": True,
                         "omega_accel_component_max_residual": component_error,
                         "finite_to_2p5_s": True}}
    assert abs(result["first_tick_substitution"]["predicted_raw_PID_change_rad"] -
               result["first_tick_substitution"]["actual_raw_PID_change_rad"]) < 1e-12
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "trajectories.npz",
                        **{mode + "__" + k: v for mode, c in curves.items() for k, v in c.items()},
                        replay_feedback_midpoint_t=feedback_t,
                        replay_feedback_interval_g=feedback_g,
                        request_t=tick / mid.HZ, request_g=requests)
    plot(shot, curves, feedback_t, feedback_g, tick, requests)
    print(json.dumps({"observed": observed, "arms": result["arms"],
                      "first_tick_substitution": result["first_tick_substitution"],
                      "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
