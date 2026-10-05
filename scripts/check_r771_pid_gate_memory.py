"""Bounded PID-gate lifecycle check on seven game replays; no production edits."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import analyze_r771_batch_loft as batch
import try_r771_effective_moment as model

OUT = batch.BASE / "gate_memory_check"
BODY_FEEDBACK = "--body-feedback" in sys.argv
if BODY_FEEDBACK:
    OUT = OUT / "body_feedback"


def controller_request(t, state, r, p):
    req = model.request(t, state, r, p)
    if BODY_FEEDBACK:
        theta, omega, gamma, x, y = state
        alpha = theta-gamma
        # A second explicit feedback hypothesis. Tangential specific force
        # comes from prescribed speed; rotation into body axes cancels the
        # thrust's wind-normal component, without inventing a drag coefficient.
        tangent_g = np.interp(t, r["t"], np.gradient(r["speed2"], r["t"]))/model.G + math.sin(gamma)
        req["a_req_g"] = req["a_req_g"]*math.cos(alpha)-math.sin(gamma)*math.sin(alpha)
        req["actual_g"] = req["actual_g"]*math.cos(alpha)-tangent_g*math.sin(alpha)
        req["error_g"] = req["a_req_g"]-req["actual_g"]
    return req


def clock_shots(original, clock):
    shots = deepcopy(original)
    for shot in shots.values():
        r = shot["r"]
        if clock == "packet":
            continue
        packet = r["t"].copy()
        age = shot["b5"].copy()
        # Associate target observations by packet time; preserve the independently
        # estimated target velocity instead of differentiating a warped clock.
        target_packet = r["target_t"] - r["missile_packet0_s"]
        mapped = np.interp(target_packet, packet, age)
        mapped[target_packet < packet[0]] = target_packet[target_packet < packet[0]] + age[0] - packet[0]
        mapped[target_packet > packet[-1]] = target_packet[target_packet > packet[-1]] + age[-1] - packet[-1]
        assert np.all(np.diff(mapped) > 0)
        r["target_t"] = mapped
        r["missile_packet0_s"] = 0.0
        # A repeated counter gives no physical interval. Keep its final state
        # only in this candidate interpolation, without editing source rows.
        keep = np.r_[np.diff(age) > 0, True]
        for key, value in list(r.items()):
            if isinstance(value, np.ndarray) and len(value) == len(age) and not key.startswith("target_"):
                r[key] = value[keep]
        r["t"] = age[keep]
        assert np.all(np.diff(r["t"]) > 0)
    return shots


def replay_diagnostic(shot, warm, exclusive=False):
    r, p = shot["r"], dict(shot["p"])
    if exclusive and p["loft_enabled"]:
        p["pn_gain"] = 0.0
    state = (0.0, 0.0, 0.0)
    was_active = False
    rows = []
    for j, t in enumerate(np.arange(0, 2.500001, model.CONTROL_DT)):
        measured = np.array([np.interp(t, r["t"], r[k]) for k in ("theta", "omega", "gamma", "x", "y")])
        req = controller_request(float(t), measured, r, p)
        if warm:
            if j == 0:
                state = (0.0, req["error_g"], 0.0)
            raw, state, diag = model.pid_step(req["error_g"], state, p)
            u = req["flight_gain"] * raw
        else:
            u, state, diag, was_active = model.scheduled_pid_step(float(t), req, state, p, was_active)
        rows.append(dict(t=float(t), u=float(u), alpha=float(measured[0]-measured[2]),
                         omega=float(measured[1]), integral=float(state[0]),
                         a_req_g=req["a_req_g"], actual_g=req["actual_g"],
                         loft_projected_g=req["loft_projected_g"], pn_g=req["pn_g"]))
    return rows


def get_all_diagnostics(shots, warm):
    return {name: {mode: replay_diagnostic(s, warm, exclusive=mode == "loft_exclusive_vertical")
                   for mode in ("loft_additive", "loft_exclusive_vertical")}
            for name, s in shots.items()}


def forward(shot, coeff, warm, exclusive=False, dt=.005):
    """Conditional closed-loop: only speed and target are prescribed after start."""
    r, p = shot["r"], dict(shot["p"])
    if exclusive and p["loft_enabled"]:
        p["pn_gain"] = 0.0
    # Start from the first native state, including its real positive time age.
    start = float(r["t"][0])
    values = np.array([r[k][0] for k in ("theta", "omega", "gamma", "x", "y")])
    pid = (0.0, 0.0, 0.0)
    active = False
    nsub = round(model.CONTROL_DT/dt)
    assert abs(nsub*dt-model.CONTROL_DT) < 1e-10
    rows = []
    first_tick = math.ceil((start-1e-10)/model.CONTROL_DT)
    # For b5 positive initial age, the unavailable prefix is explicitly omitted.
    # The first integration includes only elapsed physical time to that tick.
    def rhs(t, v, u):
        th, om, ga, x, y = v
        f = model.force_terms(t, v, r, p)
        return np.array([om, coeff[0]*u-coeff[1]*(th-ga)-coeff[2]*om,
                         model.G*(f["actual_specific_force_g"]-math.cos(ga))/f["v"],
                         f["v"]*math.cos(ga), f["v"]*math.sin(ga)])

    def integrate(t, h, u):
        nonlocal values
        a = rhs(t, values, u)
        b = rhs(t+h/2, values+h*a/2, u)
        c = rhs(t+h/2, values+h*b/2, u)
        d = rhs(t+h, values+h*c, u)
        values = values+h*(a+2*b+2*c+d)/6

    gap = first_tick*model.CONTROL_DT-start
    if gap > 1e-12:
        integrate(start, gap, 0.0)
    for j, tick in enumerate(range(first_tick, 126)):
        t = tick*model.CONTROL_DT
        if not np.isfinite(values).all() or abs(values[0]-values[2]) > 1.25:
            return None
        req = controller_request(t, values, r, p)
        if warm:
            if j == 0:
                pid = (0.0, req["error_g"], 0.0)
            raw, pid, _ = model.pid_step(req["error_g"], pid, p)
            u = req["flight_gain"]*raw
        else:
            u, pid, _, active = model.scheduled_pid_step(t, req, pid, p, active)
        rows.append(dict(t=t, theta=float(values[0]), omega=float(values[1]),
                         gamma=float(values[2]), alpha=float(values[0]-values[2]),
                         x=float(values[3]), y=float(values[4]), u=float(u)))
        for sub in range(nsub):
            integrate(t+sub*dt, dt, u)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original, p = batch.load_all()
    variants, artifacts = {}, {}
    for clock in ("packet", "b5"):
        shots = clock_shots(original, clock)
        for warm in (False, True):
            key = clock + ("_updating" if warm else "_frozen")
            diags = get_all_diagnostics(shots, warm)
            fit = batch.fit_l_chain(shots, diags)
            coeff = list(fit["coefficients"].values())
            conflicts = {name: {mode: model.full_interval_sign_conflicts(s["r"], d)
                                 for mode, d in diags[name].items()} for name, s in shots.items()}
            forwards, scores = {}, {}
            for name in ("L55", "H110", "H55clone"):
                forwards[name] = forward(shots[name], coeff, warm)
                pred = forwards[name]
                if pred is None:
                    scores[name] = {"diverged": True}
                    continue
                r = shots[name]["r"]
                ts = np.array([v["t"] for v in pred])
                mask = (r["t"] >= .3) & (r["t"] <= 2.45)
                scores[name] = {k+"_rmse": float(np.sqrt(np.mean((np.interp(r["t"][mask], ts, [v[k] for v in pred])-r[k][mask])**2)))
                                for k in ("theta", "omega", "alpha", "y")}
                for k, unit in (("alpha",180/math.pi),("omega",1)):
                    vals = np.array([v[k] for v in pred])*unit
                    scores[name]["first_"+k+"_peak"] = batch.extrema(ts, vals, .35, .75, "max")
            variants[key] = {"conditional_fit": fit, "conflicts": conflicts,
                             "I_at_arming": {n: min(ds["loft_additive"], key=lambda v: abs(v["t"]-.30))["integral"] for n, ds in diags.items()},
                             "forward": scores}
            artifacts[key] = (shots, diags, forwards)
    # One timestep acceptance check on the warm lifecycle, each clock, no refit.
    step_check = {}
    for clock in ("packet", "b5"):
        key = clock+"_updating"
        shots, diags, preds = artifacts[key]
        coeff = list(variants[key]["conditional_fit"]["coefficients"].values())
        for name in ("L55", "H110"):
            fine = forward(shots[name], coeff, True, dt=.0025)
            coarse = preds[name]
            step_check[key+"_"+name] = ({k: max(abs(a[k]-b[k]) for a,b in zip(coarse,fine)) for k in ("theta","omega","y")}
                                        if coarse and fine else {"diverged":True})
    fig, axs = plt.subplots(2, 2, figsize=(14,9), layout="constrained", sharex=True)
    for ax, name in zip(axs[0], ("L55", "H110")):
        r = original[name]["r"]
        ax.plot(r["t"],np.degrees(r["alpha"]),"k.-",label="Replay",lw=1.6)
        for key,color,style in (("packet_frozen","#be7820",":"),("packet_updating","#b02050","-"),("b5_updating","#2069b0","--")):
            pred = artifacts[key][2][name]
            if pred:
                ax.plot([v["t"] for v in pred],np.degrees([v["alpha"] for v in pred]),color=color,ls=style,label=key)
        ax.set(title=name+" conditional closed-loop forward",ylabel="Alpha (deg)",xlim=(0,2.5))
        ax.grid(alpha=.2)
    for ax, mode in zip(axs[1], ("loft_additive", "loft_exclusive_vertical")):
        for key,color,style in (("packet_frozen","#be7820",":"),("packet_updating","#b02050","-"),("b5_updating","#2069b0","--")):
            rows = artifacts[key][1]["H110"][mode]
            ax.plot([v["t"] for v in rows],[v["u"] for v in rows],color=color,ls=style,label=key)
        ax.axhline(0,color="0.3",lw=.7)
        ax.set(title="H110 diagnostic u: "+mode,ylabel="Candidate u (rad)",xlabel="Age on stated clock (s)")
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    frame = "body lateral" if BODY_FEEDBACK else "wind normal"
    fig.suptitle("PID output gating does not determine memory lifecycle\nRaw P/I/D fixed; one L110/L27.5 moment fit per candidate; " + frame + " feedback")
    fig.savefig(OUT/"gate_memory_comparison.png",dpi=170)
    plt.close(fig)
    result = {"feedback_frame":frame,"fixed_pid":{k:p[k] for k in ("kp","ki","kd","integral_limit")},
              "variants":variants,"step_halving":step_check,
              "limits":["u is reconstructed, not decoded fin; the updating lifecycle is a candidate.",
                        "b5 duplicate interpolation retains final state; source unchanged. Target positions are associated through packet-to-b5 interpolation, while target velocity keeps packet estimate.",
                        "Unknown release prefix uses first native measurement for diagnostic interpolation; forward begins at first observed age. No observed state resets thereafter.",
                        "Forward uses measured speed and target trajectory, not a free-flight simulation." ]}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:{"coeff":v["conditional_fit"]["coefficients"],
                          "conflicts":{n:{m:len(c) for m,c in vv.items()} for n,vv in v["conflicts"].items()},
                          "forward":v["forward"]} for k,v in variants.items()},indent=2))


if __name__ == "__main__":
    main()
