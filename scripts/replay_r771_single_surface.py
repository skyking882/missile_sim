"""Bounded L110 replay with one equivalent aerodynamic surface at the stab arm.

The strict candidate uses one total transverse force for translation and
moment. It is conditional on replay speed/target, provisional rod inertia,
and the previously calibrated command-to-moment gain; no coefficient is fit.
"""
from __future__ import annotations

import json
import math

import numpy as np

import replay_r771_l110_from_midflight as mid

previous = mid.previous
fit = mid.fit
model = mid.model
OUT = fit.OUT / "l110_single_surface"
KEYS = (*mid.KEYS, "alpha", "normal_body_n", "normal_wind_n", "moment_nm",
        "feedback_g", "gamma_force_g", "pid_measured_g", "pid_error_g",
        "passive_body_n", "control_body_n", "alpha_local_rad")
LIMIT_RAD = math.radians(30.)
STEP = mid.rate.STEP


def surface(t, state, held_u, shot, ku, reference):
    """One body-normal force, applied at x=-l in the small-angle planar model."""
    p = shot["p"]
    th, omega, gamma, _, y = state
    alpha = th - gamma
    terms = model.force_terms(t, state, shot["replay"], p)
    speed, mass, thrust = terms["v"], terms["mass"], terms["thrust"]
    qbar = .5 * model.isa_density(y) * speed**2
    area = math.pi * p["caliber_m"]**2 / 4 * p["wing_area_mult"]
    arm = p["tail_arm_m"]
    inertia = mass * p["length_m"]**2 / 12
    active = (qbar / mass) / reference
    local_alpha = alpha + arm * omega / speed
    # The earlier 14*alpha law was defined as wind-normal. Dividing by cos
    # preserves that law exactly at zero rate/input before body projection.
    passive_body = qbar * area * 14. * local_alpha / math.cos(alpha)
    control_body = -(inertia / arm) * active * ku * held_u
    normal_body = passive_body + control_body
    normal_wind = normal_body * math.cos(alpha)
    moment = -arm * normal_body
    feedback_g = (normal_wind + thrust * math.sin(alpha)) / (mass * model.G)
    return {"v": speed, "mass": mass, "thrust": thrust,
            "qbar": qbar, "area": area, "inertia": inertia,
            "alpha_local_rad": local_alpha, "passive_body_n": passive_body,
            "control_body_n": control_body, "normal_body_n": normal_body,
            "normal_wind_n": normal_wind, "moment_nm": moment,
            "feedback_g": feedback_g}


def request_and_pid(tick, state, held_u, pid, shot, ku, reference):
    """Sample feedback under the previous held command, then update the PID."""
    start = tick / mid.HZ
    measured = surface(start, state, held_u, shot, ku, reference)
    command = model.request(start, state, shot["replay"], shot["p"])
    error = command["a_req_g"] - measured["feedback_g"]
    gain = model.interpolate_table((tick + 1) / mid.HZ, shot["p"]["time_gain"])
    if gain <= 1e-12:
        assert pid == (0., 0., 0.)
        return 0., pid, error, measured["feedback_g"]
    raw, next_pid = fit.pid_step_with_dt(error, pid, shot["p"], 1 / mid.HZ)
    return gain * raw, next_pid, error, measured["feedback_g"]


def replay_pid_before(shot, cut, ku, reference):
    pid = (0., 0., 0.)
    held_u = 0.
    for tick in range(cut):
        state = mid.measured_state(shot, tick / mid.HZ)
        held_u, pid, _, _ = request_and_pid(tick, state, held_u, pid,
                                             shot, ku, reference)
    return pid, held_u


def simulate_strict(shot, ku, *, start_tick=0, initial=None,
                    pid_initial=(0., 0., 0.), held_initial=0., step=STEP):
    state = (shot["initial"] if initial is None else initial).copy()
    pid = tuple(pid_initial)
    held_u = float(held_initial)
    q0 = .5 * model.isa_density(shot["initial"][4]) * shot["initial_speed_mps"]**2
    reference = q0 / shot["p"]["mass0_kg"]
    rows = []
    termination = {"reason": "horizon", "time_s": mid.HORIZON}

    def rhs(t, s, u):
        f = surface(t, s, u, shot, ku, reference)
        theta, omega, gamma, _, _ = s
        return np.array([omega, f["moment_nm"] / f["inertia"],
                         model.G * (f["feedback_g"] - math.cos(gamma)) / f["v"],
                         f["v"] * math.cos(gamma), f["v"] * math.sin(gamma)])

    def append_row(t, s, u, measured_g, error):
        f = surface(t, s, u, shot, ku, reference)
        rows.append([t, *s, u, pid[0], s[0] - s[2], f["normal_body_n"],
                     f["normal_wind_n"], f["moment_nm"], f["feedback_g"],
                     f["feedback_g"], measured_g, error, f["passive_body_n"],
                     f["control_body_n"], f["alpha_local_rad"]])

    for tick in range(start_tick, round(mid.HORIZON * mid.HZ)):
        start, end = tick / mid.HZ, (tick + 1) / mid.HZ
        if abs(state[0] - state[2]) > LIMIT_RAD:
            termination = {"reason": "alpha_domain_exit", "time_s": start,
                           "alpha_deg": math.degrees(state[0] - state[2])}
            break
        held_u, pid, error, measured_g = request_and_pid(
            tick, state, held_u, pid, shot, ku, reference)
        append_row(start, state, held_u, measured_g, error)
        steps = math.ceil((end - start) / step - 1e-9)
        h = (end - start) / steps
        terminated = False
        for k in range(steps):
            t = start + k * h
            d1 = rhs(t, state, held_u)
            d2 = rhs(t + h / 2, state + h * d1 / 2, held_u)
            d3 = rhs(t + h / 2, state + h * d2 / 2, held_u)
            d4 = rhs(t + h, state + h * d3, held_u)
            state += h * (d1 + 2 * d2 + 2 * d3 + d4) / 6
            if not np.isfinite(state).all() or abs(state[0] - state[2]) > LIMIT_RAD:
                termination = {"reason": "alpha_domain_exit", "time_s": t + h,
                               "alpha_deg": math.degrees(state[0] - state[2])}
                terminated = True
                break
            if k < steps - 1:
                append_row(t + h, state, held_u, measured_g, error)
        if terminated:
            break
    else:
        append_row(mid.HORIZON, state, held_u, measured_g, error)
    a = np.asarray(rows)
    return {key: a[:, j] for j, key in enumerate(KEYS)}, termination


def events(shot, pred, start):
    obs = shot["observed"]
    out = {}
    windows = (("q_peak1", "omega", .35, .58, "max"),
               ("q_valley1", "omega", .60, .84, "min"),
               ("q_peak2", "omega", .85, 1.18, "max"),
               ("alpha_peak1", "alpha", .38, .76, "max"),
               ("alpha_valley1", "alpha", .80, 1.02, "min"))
    for name, key, lo, hi, kind in windows:
        if hi <= start or pred["t"][-1] < max(lo, start):
            continue
        real = fit.event(obs["t"], obs[key], lo, hi, kind)
        ids = np.flatnonzero((pred["t"] >= max(lo, start)) & (pred["t"] <= hi))
        if not len(ids):
            continue
        j = ids[np.argmax(pred[key][ids]) if kind == "max" else np.argmin(pred[key][ids])]
        out[name] = {"observed": real,
                     "predicted": {"time_s": float(pred["t"][j]),
                                   "value": float(pred[key][j])},
                     "predicted_extremum_at_window_edge": bool(j in (ids[0], ids[-1])),
                     "window_complete": bool(pred["t"][-1] >= hi)}
    return out


def zero_crossings(pred, lo=.3, hi=1.4):
    t, q = pred["t"], pred["omega"]
    result = []
    for i in range(len(t) - 1):
        if q[i] * q[i + 1] >= 0:
            continue
        crossing = float(t[i] - q[i] * (t[i + 1] - t[i]) / (q[i + 1] - q[i]))
        if lo <= crossing <= hi:
            result.append(crossing)
    return result


def plot(shot, curves):
    plt = fit.plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col",
                             layout="constrained")
    obs = shot["observed"]
    for col, tag in enumerate(("launch", "restart_29")):
        start = 0. if col == 0 else 29 / mid.HZ
        for row, (key, scale, ylabel) in enumerate(
                (("omega", 1., "Pitch rate (rad/s)"),
                 ("alpha", 180 / math.pi, "AoA (deg)"))):
            ax = axes[row, col]
            keep = (obs["t"] >= .3) & (obs["t"] <= 1.5)
            ax.plot(obs["t"][keep], obs[key][keep] * scale, "ko",
                    ms=3.6, zorder=5, label="Game replay")
            for name, color, style, title in (
                    ("baseline", ".6", "--", "Original"),
                    ("geometry_only", "#d88330", "-", "Geometry damping only"),
                    ("strict", "#2779b5", "-", "Single surface force + moment")):
                pred = curves[tag][name]
                keep = (pred["t"] >= start) & (pred["t"] <= 1.5)
                ax.plot(pred["t"][keep], pred[key][keep] * scale,
                        color=color, ls=style, lw=1.8, label=title)
            if col:
                ax.axvline(start, color=".35", lw=.8)
            ax.set(xlim=(.3, 1.5), xlabel="Time from launch (s)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
        axes[0, col].set_title("Launch" if col == 0 else
                                f"Measured-state restart at {start:.4f} s")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("L110 | fixed PID and guidance | one-surface geometry test")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)

    # The strict candidate sets a much larger scale in the overview. Show the
    # isolated damping comparison at the observed scale without hiding that
    # this panel retains the former independently calibrated moment terms.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             layout="constrained")
    for ax, (key, scale, ylabel) in zip(axes,
            (("omega", 1., "Pitch rate (rad/s)"),
             ("alpha", 180 / math.pi, "AoA (deg)"))):
        keep = (obs["t"] >= .55) & (obs["t"] <= 1.45)
        ax.plot(obs["t"][keep], obs[key][keep] * scale, "ko", ms=4,
                label="Game replay", zorder=5)
        for name, color, style, title in (
                ("baseline", ".55", "--", "Original damping"),
                ("geometry_only", "#d88330", "-", "Geometry damping only")):
            pred = curves["restart_29"][name]
            keep = pred["t"] <= 1.45
            ax.plot(pred["t"][keep], pred[key][keep] * scale,
                    color=color, ls=style, lw=2, label=title)
        ax.axvline(29 / mid.HZ, color=".4", lw=.8)
        ax.set(ylabel=ylabel, xlim=(.55, 1.45))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Time from launch (s)")
    fig.suptitle("L110 | restart at 0.6042 s | isolated damping comparison\n"
                 "Diagnostic only: original control and restoring terms retained")
    fig.savefig(OUT / "damping_detail.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coeff, raw, parent = previous.load()
    shot = fit.make_shot(raw["L110"])
    assert not shot["p"]["loft_enabled"]
    assert [shot["p"][k] for k in ("kp", "ki", "kd")] == [.002125, .006481, .000023]
    ku, ka, old_kw = map(float, coeff)
    arm = shot["p"]["tail_arm_m"]
    v0 = shot["initial_speed_mps"]
    geometric_kw = ka * arm / v0
    diagnostic_coeff = np.array([ku, ka, geometric_kw])
    cut = 29
    start = cut / mid.HZ
    initial = mid.measured_state(shot, start)
    baseline_launch, _ = mid.simulate(shot, coeff)
    geometry_launch, _ = mid.simulate(shot, diagnostic_coeff)
    strict_launch, strict_launch_end = simulate_strict(shot, ku)
    original_pid = mid.replay_pid_before(shot, cut)
    baseline_restart, _ = mid.simulate(shot, coeff, cut, initial, original_pid)
    geometry_restart, _ = mid.simulate(shot, diagnostic_coeff, cut, initial, original_pid)
    q0 = .5 * model.isa_density(shot["initial"][4]) * v0**2
    reference = q0 / shot["p"]["mass0_kg"]
    strict_pid, prior_u = replay_pid_before(shot, cut, ku, reference)
    restart_force_before_pid = surface(start, initial, prior_u, shot, ku, reference)
    strict_restart, strict_restart_end = simulate_strict(
        shot, ku, start_tick=cut, initial=initial,
        pid_initial=strict_pid, held_initial=prior_u)
    curves = {
        "launch": {"baseline": baseline_launch, "geometry_only": geometry_launch,
                   "strict": strict_launch},
        "restart_29": {"baseline": baseline_restart,
                       "geometry_only": geometry_restart, "strict": strict_restart}}

    # One focused verification pass, including the stored original baseline.
    saved = np.load(fit.OUT / "l110_midflight" / "trajectories.npz")
    baseline_delta = max(float(np.max(abs(baseline_launch[k] - saved["baseline__" + k])))
                         for k in (*mid.KEYS, "alpha"))
    assert baseline_delta < 1e-12
    geometry_ratio = geometric_kw * v0 / ka
    assert abs(geometry_ratio - arm) < 1e-12
    identity = max(float(np.max(abs(pred["moment_nm"] + arm * pred["normal_body_n"])))
                   for pred in (strict_launch, strict_restart))
    assert identity < 1e-8
    feedback_delta = max(float(np.max(abs(pred["feedback_g"] - pred["gamma_force_g"])))
                         for pred in (strict_launch, strict_restart))
    assert feedback_delta < 1e-12
    assert strict_restart["t"][0] == start
    assert np.array_equal(np.array([strict_restart[k][0] for k in mid.STATE_KEYS]), initial)
    fine, fine_end = simulate_strict(shot, ku, start_tick=cut, initial=initial,
                                     pid_initial=strict_pid, held_initial=prior_u,
                                     step=STEP / 2)
    common = strict_restart["t"] <= fine["t"][-1]
    step_delta = max(float(np.max(abs(strict_restart[k][common] -
                                     np.interp(strict_restart["t"][common], fine["t"], fine[k]))))
                     for k in ("theta", "omega", "gamma", "x", "y"))
    assert step_delta < 1e-4
    # At alpha=0 the local-flow derivative is the usual q*S*a*l^2/V.
    sample = initial.copy()
    sample[0] = sample[2]
    sample[1] = 0.
    f0 = surface(start, sample, prior_u, shot, ku, reference)
    delta = 1e-5
    perturbed = sample.copy(); perturbed[1] = delta
    f1 = surface(start, perturbed, prior_u, shot, ku, reference)
    geometric_derivative = -f0["qbar"] * f0["area"] * 14. * arm**2 / f0["v"]
    observed_derivative = (f1["moment_nm"] - f0["moment_nm"]) / delta
    derivative_delta = abs(observed_derivative - geometric_derivative)
    assert derivative_delta < 1e-7

    result = {
        "case": "L110", "source_version": parent["source_version"],
        "scope": "Conditional 2.5 s replay: speed and target motion are external; position, attitude, flight path, pitch rate and PID memory evolve. No fitting.",
        "source_parameters": {k: shot["p"][k] for k in
                              ("caliber_m", "wing_area_mult", "tail_arm_m",
                               "mass0_kg", "length_m", "fin_aoa_rad", "kp", "ki",
                               "kd", "integral_limit", "loft_enabled", "time_gain")},
        "fixed_moment_coefficients": {"ku": ku, "ka": ka, "baseline_kw": old_kw,
                                      "diagnostic_kw": geometric_kw},
        "formulas": {
            "baseline": "omega_dot=A*(ku*u-ka*alpha)-B*kw*omega; original feedback and wind-normal qbar*S*14*alpha",
            "geometry_only": "same as baseline, except kw=ka*l/V0 so damping/restoring ratio=l/V; independently calibrated moment terms remain",
            "strict": "alpha_local=alpha+l*omega/V; N_passive_body=qbar*S*14*alpha_local/cos(alpha); I=m*L^2/12; N_control_body=-(I/l)*A*ku*u; N_total_body=N_passive_body+N_control_body; M=-l*N_total_body; gamma_dot=[(N_total_body*cos(alpha)+T*sin(alpha))/m-g*cos(gamma)]/V",
            "scaling": "A=(qbar/m)/(qbar0/m0); B=A*V0/V",
        },
        "assumptions": [
            "One equivalent normal aerodynamic force at x=-distFromCmToStab; no additional control couple or damping moment.",
            "Small-angle local rotational inflow alpha_local=alpha+l*omega/V; wind-normal projection uses cos(alpha).",
            "Provisional rod inertia I=m*length^2/12, not decoded missile inertia.",
            "The old calibrated ku is retained as a command torque and converted to same-point force; no actuator authority or finsLatAccel semantics added.",
            "Prior held control determines sampled acceleration at each 48 Hz tick; new PID control is held over the next interval. Midflight PID histories are reconstructed from measured states, not decoded game telemetry.",
        ],
        "initialization": {"restart_time_s": start,
                           "measured_state": dict(zip(mid.STATE_KEYS, initial.tolist())),
                           "baseline_pid": list(original_pid),
                           "strict_pid": list(strict_pid),
                           "strict_previous_held_u_rad": prior_u,
                           "strict_force_before_next_pid": {
                               k: restart_force_before_pid[k] for k in
                               ("passive_body_n", "control_body_n", "normal_body_n",
                                "moment_nm", "feedback_g", "alpha_local_rad")}},
        "runs": {tag: {name: {"termination": (strict_launch_end if tag == "launch" else strict_restart_end)
                                             if name == "strict" else {"reason": "horizon", "time_s": mid.HORIZON},
                             "events": events(shot, pred, 0. if tag == "launch" else start),
                             "omega_zero_crossings_s": zero_crossings(pred,
                                 lo=max(.3, 0. if tag == "launch" else start))}
                       for name, pred in group.items()}
                 for tag, group in curves.items()},
        "verification": {"stored_baseline_max_delta": baseline_delta,
                         "strict_max_abs_M_plus_lN_nm": identity,
                         "feedback_vs_gamma_force_max_delta_g": feedback_delta,
                         "restart_state_and_time_exact": True,
                         "step_halving_common_time_max_state_delta": step_delta,
                         "strict_fine_termination": fine_end,
                         "small_angle_dM_domega_derived_nm_s": geometric_derivative,
                         "small_angle_dM_domega_numeric_nm_s": observed_derivative,
                         "small_angle_derivative_abs_delta": derivative_delta,
                         "geometry_ratio_l_m": geometry_ratio}}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / "trajectories.npz",
                        **{tag + "__" + name + "__" + key: val
                           for tag, group in curves.items()
                           for name, pred in group.items()
                           for key, val in pred.items()})
    plot(shot, curves)
    print(json.dumps({"runs": result["runs"], "verification": result["verification"]},
                     indent=2))


if __name__ == "__main__":
    main()
