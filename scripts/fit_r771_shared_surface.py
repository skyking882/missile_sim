"""Shared empirical one-surface R-77-1 fit, continuous from decoded creation.

Only this experiment is changed. V(t) and target motion are replay inputs;
missile attitude, flight path, position, and fixed PID memory are integrated.
The aerodynamic resultant and reference free couple share the .4 m surface.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import fit_r771_no_loft_control as old
import fit_r771_single_shot_damping as damping

model = old.model
OUT = old.OUT / "shared_surface_forward"
NAMES = ("L110", "L27.5", "L55")
TRAIN_NAMES = NAMES[:2]
FIT_END = 2.45
HORIZON = 3.5
HZ = 48
FIT_SUBSTEPS = 4
SHAPE_REFERENCE = None
COEFF_NAMES = ("lift_slope_per_rad", "control_speed_power", "control_gain_s_minus2",
               "restoring_gain_s_minus2", "reference_damping_s_minus1")
COEFF_UNITS = ("rad^-1", "dimensionless", "s^-2", "s^-2", "s^-1")
BOUNDS = np.array([[3., 30.], [-2., 2.], [100., 1800.], [8., 100.], [.05, 15.]])
SCALE = np.array([14., 1., 650., 35., 3.5])
STATE_SCALES = {"theta": .02, "omega": .004, "alpha": .00075, "gamma": .003}
EVENTS = (("q_peak1", "omega", .35, .58, "max", .025),
          ("q_valley1", "omega", .60, .84, "min", .025),
          ("q_peak2", "omega", .85, 1.18, "max", .025),
          ("alpha_peak1", "alpha", .38, .76, "max", math.radians(.20)),
          ("alpha_valley1", "alpha", .80, 1.02, "min", math.radians(.20)))


def load():
    raw, p = old.batch.load_all()
    assert p["source_version"] == "2.59.0.28"
    assert [p[k] for k in ("kp", "ki", "kd", "integral_limit")] == [.002125, .006481, .000023, 1.]
    shots = {n: old.make_shot(raw[n], "b5") for n in NAMES}
    assert all(not s["p"]["loft_enabled"] and s["p"]["tail_arm_m"] == .4 for s in shots.values())
    return shots, p


def surface(t, state, held_u, shot, coeff, family="new"):
    """One resultant plus reference couple; rate incidence enters N once."""
    ca, power, ku, ka, kw = map(float, coeff)
    th, omega, gamma, _, y = state
    alpha = th - gamma
    p, r = shot["p"], shot["replay"]
    v = float(np.interp(t, r["t"], r["speed2"]))
    mass, thrust = model.mass_thrust(t, p)
    qbar = .5 * model.isa_density(y) * v * v
    area = math.pi * p["caliber_m"]**2 / 4 * p["wing_area_mult"]
    arm = p["tail_arm_m"]
    inertia = mass * p["length_m"]**2 / 12
    qref = .5 * model.isa_density(shot["initial"][4]) * shot["initial_speed_mps"]**2
    active = (qbar / mass) / (qref / p["mass0_kg"])
    damp_scale = active * shot["initial_speed_mps"] / v
    alpha_local = alpha + arm * omega / v if family == "new" else alpha
    # Preserve the old wind-normal interpretation at zero body rate. The body
    # resultant below projects back to this exact wind-normal force.
    normal_wind = qbar * area * ca * alpha_local
    normal_body = normal_wind / math.cos(alpha)
    feedback = (normal_wind + thrust * math.sin(alpha)) / (mass * model.G)
    gamma_dot = model.G * (feedback - math.cos(gamma)) / v
    if family == "new":
        qdot = ku * (v / shot["initial_speed_mps"])**power * held_u - active * ka * alpha_local - damp_scale * kw * omega
    elif family in ("archive_constant", "archive_qbar"):
        qdot = (ku * held_u if family == "archive_constant" else active*ku*held_u) - active*ka*alpha - active*shot["initial_speed_mps"]/v*kw*omega
    else:
        raise ValueError(family)
    reference_couple = inertia * qdot + arm * normal_body
    moment = -arm * normal_body + reference_couple
    return {"v": v, "mass": mass, "thrust": thrust, "alpha_local": alpha_local,
            "normal_wind": normal_wind, "normal_body": normal_body,
            "feedback_g": feedback, "gamma_dot": gamma_dot, "qdot": moment / inertia,
            "reference_couple": reference_couple, "moment": moment, "inertia": inertia}


def simulate(shot, coeff, *, substeps=FIT_SUBSTEPS, horizon=HORIZON, family="new"):
    p, r = shot["p"], shot["replay"]
    state = shot["initial"].copy()
    pid = (0., 0., 0.)
    held = 0.
    rows = []
    step = 1 / (HZ * substeps)
    ticks = round(horizon * HZ)
    first = None

    def rhs(t, s, u):
        f = surface(t, s, u, shot, coeff, family)
        return np.array([s[1], f["qdot"], f["gamma_dot"],
                         f["v"] * math.cos(s[2]), f["v"] * math.sin(s[2])])

    def record(t, s, u, req):
        f = surface(t, s, u, shot, coeff, family)
        rows.append([t, *s, s[0]-s[2], u, pid[0], f["feedback_g"], req,
                     f["normal_body"], f["normal_wind"], f["moment"], f["reference_couple"],
                     f["alpha_local"], f["qdot"], f["gamma_dot"]])

    for tick in range(ticks+1):
        start = tick / HZ
        if not np.isfinite(state).all() or abs(state[0]-state[2]) > .6:
            return None
        req = model.request(start, state, r, p)["a_req_g"]
        prior_feedback = surface(start, state, held, shot, coeff, family)["feedback_g"]
        gain = model.interpolate_table((tick+1)/HZ, p["time_gain"])
        if gain > 1e-12:
            raw, pid = old.pid_step_with_dt(req-prior_feedback, pid, p, 1/HZ)
            held = gain*raw
            if first is None: first = {"start_s": start, "gain": gain, "raw_output_rad": raw}
        else:
            assert pid == (0., 0., 0.)
            held = 0.
        record(start, state, held, req)
        if tick == ticks: break
        for k in range(substeps):
            t = start + k*step
            d1 = rhs(t, state, held)
            d2 = rhs(t+step/2, state+step*d1/2, held)
            d3 = rhs(t+step/2, state+step*d2/2, held)
            d4 = rhs(t+step, state+step*d3, held)
            state += step*(d1+2*d2+2*d3+d4)/6
            if k < substeps-1: record(t+step, state, held, req)
    a = np.array(rows)
    keys = ("t", "theta", "omega", "gamma", "x", "y", "alpha", "u", "I", "feedback_g", "a_req_g",
            "normal_body_n", "normal_wind_n", "moment_nm", "reference_couple_nm", "alpha_local", "qdot", "gamma_dot")
    result = {key: a[:, j] for j, key in enumerate(keys)}
    result["first_active"] = first
    return result


def residual(shot, pred, coeff):
    obs = shot["observed"]
    mask = obs["t"] <= FIT_END+1e-9
    ot = obs["t"][mask]
    n = len(ot)
    res = []
    for key, scale in STATE_SCALES.items():
        err = (np.interp(ot, pred["t"], pred[key])-obs[key][mask])/scale
        res.extend(err / math.sqrt(n*4))
    for _, key, lo, hi, kind, scale in EVENTS:
        ot0, ov = damping.extremum(obs["t"], obs[key], lo, hi, kind)
        pt0, pv = damping.extremum(pred["t"], pred[key], lo, hi, kind)
        res.extend([.25*(pt0-ot0)/.03, .25*(pv-ov)/scale])
    # Native angular/flight-path increments are independent constraints on
    # force and moment, evaluated at measured state with the forward-held u.
    ids = np.flatnonzero((obs["t"][:-1] >= .03) & (obs["t"][1:] <= FIT_END+1e-9))
    ares, gres = [], []
    for i in ids:
        lo, hi = obs["t"][i:i+2]
        tm = (lo+hi)/2
        state = np.array([.5*(obs[k][i]+obs[k][i+1]) for k in ("theta", "omega", "gamma", "x", "y")])
        j = min(len(pred["t"])-1, np.searchsorted(pred["t"], tm, side="right")-1)
        f = surface(tm, state, pred["u"][j], shot, coeff)
        ares.append((f["qdot"]*(hi-lo)-(obs["omega"][i+1]-obs["omega"][i]))/.03)
        gres.append((f["gamma_dot"]*(hi-lo)-(obs["gamma"][i+1]-obs["gamma"][i]))/.0005)
    res.extend(.15*np.asarray(ares)/math.sqrt(len(ids)))
    res.extend(.15*np.asarray(gres)/math.sqrt(len(ids)))
    if SHAPE_REFERENCE is not None:
        baseline = SHAPE_REFERENCE[shot["name"]]
        current = metrics(shot, pred)
        # Preserve the ordinary sample/event definitions for acceptance. This
        # bounded penalty addresses the specific recovery regression found in
        # review; it adds no plant/controller parameters or excluded samples.
        violations = []
        for window, key, limit in (("first_recovery", "alpha", 1.0),
                                    ("full", "omega", .999),
                                    ("full", "alpha", .999),
                                    ("full", "gamma", .999)):
            base = baseline["windows"][window][key+"_rmse"]
            value = current["windows"][window][key+"_rmse"]
            violations.append((10 if window == "full" else 1)*max(0., value/base-limit))
        for label, *_ in EVENTS:
            for field in ("value_error", "parabolic_time_error_s"):
                base = abs(baseline["events"][label][field])
                value = abs(current["events"][label][field])
                limit = (.60 if label == "q_valley1" and field == "value_error" else
                         .98 if label == "alpha_peak1" and field == "value_error" else
                         1.03 if field == "value_error" else 1.0)
                violations.append(max(0., value/max(base, 1e-6)-limit))
        res.extend(100*np.asarray(violations))
    return np.asarray(res)


def optimize(shots, seeds):
    """Bounded finite-difference least squares; uses installed NumPy only."""
    lo, hi = BOUNDS[:, 0]/SCALE, BOUNDS[:, 1]/SCALE
    evaluations = 0
    start_time = time.monotonic()
    best = None
    traces = []

    def fun(z):
        nonlocal evaluations
        evaluations += 1
        coeff = np.clip(z, lo, hi)*SCALE
        values = []
        for shot in shots.values():
            pred = simulate(shot, coeff, horizon=max(1.5, math.ceil((FIT_END+.1)*HZ)/HZ))
            if pred is None: return None
            values.append(residual(shot, pred, coeff))
        return np.concatenate(values) / math.sqrt(len(shots))

    for seed_index, seed in enumerate(seeds):
        z = np.clip(np.array(seed)/SCALE, lo, hi)
        r = fun(z)
        loss = float(r@r)
        lm = .03
        for iteration in range(65):
            jac = []
            for j in range(len(z)):
                delta = .0005*max(1., abs(z[j]))
                zz = z.copy(); zz[j] = min(hi[j], z[j]+delta)
                if zz[j] == z[j]: zz[j] = max(lo[j], z[j]-delta)
                rr = fun(zz)
                if rr is None: jac.append(np.zeros_like(r))
                else: jac.append((rr-r)/(zz[j]-z[j]))
            jac = np.array(jac).T
            a = jac.T@jac
            gradient = jac.T@r
            improved = False
            for trial in range(7):
                dz = np.linalg.solve(a+lm*np.diag(np.maximum(np.diag(a), .01)), -gradient)
                zz = np.clip(z+dz, lo, hi)
                rr = fun(zz)
                ll = 1e9 if rr is None else float(rr@rr)
                if ll < loss:
                    improvement = loss-ll
                    z, r, loss = zz, rr, ll
                    lm = max(1e-7, lm*.4)
                    improved = True
                    break
                lm *= 5
            if iteration % 10 == 0:
                print(json.dumps({"seed": seed_index, "iteration": iteration, "loss": loss,
                                  "coeff": (z*SCALE).tolist(), "evaluations": evaluations}), flush=True)
            if not improved or (improvement < 1e-7 and np.max(np.abs(dz)) < 1e-4): break
            if time.monotonic()-start_time > 120: break
        traces.append({"seed": list(seed), "coefficients": (z*SCALE).tolist(), "objective": loss,
                       "iterations": iteration+1})
        if best is None or loss < best[0]: best = loss, z.copy()
        if time.monotonic()-start_time > 120: break
    return best[1]*SCALE, {"objective": best[0], "evaluations": evaluations,
                           "elapsed_s": time.monotonic()-start_time, "starts": traces}


def metrics(shot, pred):
    obs = shot["observed"]
    result = {"windows": {}, "events": {}}
    for label, lo, hi in (("fit", 0., FIT_END), ("late_calibration", 1.45, 2.45),
                           ("frozen_continuation", 2.55, 3.5),
                           ("full", 0., 2.45), ("passive", 0., .30),
                           ("first_recovery", .6, 1.15)):
        mask = (obs["t"] >= lo-1e-9) & (obs["t"] <= hi+1e-9)
        result["windows"][label] = {k+"_rmse": float(np.sqrt(np.mean((
            np.interp(obs["t"][mask], pred["t"], pred[k])-obs[k][mask])**2)))
            for k in (*STATE_SCALES, "y")}
    for label, key, lo, hi, kind, _ in EVENTS:
        actual = old.event(obs["t"], obs[key], lo, hi, kind)
        predicted = old.event(pred["t"], pred[key], lo, hi, kind)
        interpolated_actual = damping.extremum(obs["t"], obs[key], lo, hi, kind)
        interpolated_predicted = damping.extremum(pred["t"], pred[key], lo, hi, kind)
        result["events"][label] = {"observed": actual, "predicted": predicted,
            "time_error_s": predicted["time_s"]-actual["time_s"],
            "value_error": predicted["value"]-actual["value"],
            "parabolic_native_observed": {"time_s":interpolated_actual[0], "value":interpolated_actual[1]},
            "parabolic_continuous_prediction": {"time_s":interpolated_predicted[0], "value":interpolated_predicted[1]},
            "parabolic_time_error_s":interpolated_predicted[0]-interpolated_actual[0]}
    result["omega_crossings_s"] = old.crossings(pred["t"], pred["omega"], .3, 2.45)
    result["observed_omega_crossings_s"] = old.crossings(obs["t"], obs["omega"], .3, 2.45)
    return result


def shape_acceptance(baseline, current):
    full = {key: current["windows"]["full"][key+"_rmse"] /
                 baseline["windows"]["full"][key+"_rmse"] for key in ("omega", "alpha", "gamma")}
    recovery = current["windows"]["first_recovery"]["alpha_rmse"] / baseline["windows"]["first_recovery"]["alpha_rmse"]
    events = {label: {field: abs(current["events"][label][field]) /
                     max(abs(baseline["events"][label][field]), 1e-6)
                     for field in ("value_error", "parabolic_time_error_s")}
              for label, *_ in EVENTS}
    failures = []
    for key in ("omega", "alpha"):
        if full[key] >= 1: failures.append("full_"+key+"_rmse")
    if recovery > 1.05: failures.append("first_recovery_alpha_rmse")
    for label, record in events.items():
        for field, ratio in record.items():
            if ratio > 1.05: failures.append(label+"_"+field)
    return {"full_rmse_ratios":full, "first_recovery_alpha_rmse_ratio":recovery,
            "event_error_ratios":events, "failed_conditions":failures,
            "aggregate_omega_alpha_pass":all(full[k] < 1 for k in ("omega", "alpha")),
            "stronger_shape_pass":not failures}


def finalize_refinement():
    """Keep the original candidate if the stronger acceptance is infeasible."""
    frozen = json.loads((OUT/"fit.json").read_text())
    archived_result = json.loads((OUT/"results.json").read_text())
    original = archived_result["fitted_coefficients"]
    candidate = frozen["shape_refinement"]["refined_coefficients"]
    shots, _ = load()
    constants = archived_result["old_best"]["base_coefficients"]
    records, acceptance = {}, {}
    for name, shot in shots.items():
        base = metrics(shot, simulate(shot, np.r_[14., 0., constants],
                       substeps=10, family="archive_constant"))
        refined = metrics(shot, simulate(shot, candidate, substeps=10))
        records[name] = refined
        acceptance[name] = shape_acceptance(base, refined)
    accepted = all(a["stronger_shape_pass"] for a in acceptance.values())
    refinement = frozen["shape_refinement"]
    refinement["starting_coefficients"] = refinement.pop("prior_aggregate_candidate_coefficients")
    refinement["original_aggregate_candidate_coefficients"] = original
    refinement["candidate_records"] = records
    refinement["candidate_acceptance"] = acceptance
    refinement["accepted"] = accepted
    refinement["selection"] = "refined candidate" if accepted else "original aggregate candidate preserved; stronger shape acceptance failed"
    refinement["penalty_targets"] = {"recovery_alpha_ratio":1., "q_valley1_amplitude_error_ratio":.60,
        "alpha_peak1_amplitude_error_ratio":.98, "other_amplitude_error_ratio":1.03,
        "continuous_event_time_error_ratio":1., "full_omega_alpha_gamma_ratio":.999,
        "full_penalty_multiplier":10., "other_penalty_multiplier":1.}
    refinement["bounded_work"] = "One same-family refinement with three successive penalty settings; total optimizer time under40s, below120s budget. No L55 residuals in objective."
    frozen["fitted_coefficients"] = candidate if accepted else original
    frozen["fit_substeps_per_48Hz_tick"] = (8 if accepted else
        archived_result.get("fit_substeps_per_48Hz_tick", 4))
    frozen["acceptance"] += " Full-window aggregate pass is separate from stronger first-recovery/five-event shape pass; shape failures prevent a general better-fit claim."
    frozen["crosscase_provenance"] = "L55 excluded from every coefficient objective, but previously inspected and its reviewed results informed the acceptance margin; not a blind holdout."
    (OUT/"fit.json").write_text(json.dumps(frozen, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"shape_refinement_accepted":accepted, "candidate_acceptance":acceptance,
                      "preserved_coefficients":frozen["fitted_coefficients"]}, indent=2), flush=True)


def train():
    global SHAPE_REFERENCE
    OUT.mkdir(parents=True, exist_ok=True)
    shots, p = load()
    predeclared = {"fit_cases": list(TRAIN_NAMES), "fit_window_s": [0., FIT_END],
        "frozen_continuation_window_s": [2.55, 3.5], "crosscase": "L55",
        "crosscase_provenance": "Previously inspected; excluded from all current coefficient optimization.",
        "acceptance": "Across all cases 0-2.45s ordinary omega and alpha RMSE must improve versus archived best rotation model. Frozen 2.55-3.5s continuation and gamma/path tradeoffs checked separately against both archived models. No state resets or time shifts; native extrema and common continuous parabolic event algorithm reported.",
        "coefficients": dict(zip(COEFF_NAMES, COEFF_UNITS)), "bounds": BOUNDS.tolist(),
        "conditional_inputs": "Measured V and target motion; decoded creation state only; integrated theta/omega/gamma/x/y and fixed PID.",
        "source_version": p["source_version"], "fixed_PID": {k:p[k] for k in ("kp", "ki", "kd", "integral_limit")},
        "schedule": "48Hz interval-END raw gain, zero PID memory before activation; no fitted timing or per-case constants.",
        "single_surface": "Nw=qS*Ca*alpha_local; Nb=Nw/cos(alpha); Mcm=-l*Nb+Mref; alpha_local=alpha+l*omega/V, l=.4m.",
        "angular_equation": "omega_dot=Ku*(V/V0)^p*u-A*Ka*alpha_local-B*Kw*omega; A=(qbar/m)/(qbar0/m0), B=A*V0/V; Mref=I*omega_dot+l*Nb; I=mL^2/12.",
        "force_projection": "Preserves previous wind-normal slope at zero rate; same resultant is body-projected for .4m arm. No separate wing/body/fin surfaces.",
        "fit_objective": "Native theta/omega/alpha/gamma residual plus five early peak/valley times/amplitudes and native angular/flight-path interval balances. All native samples retained; ordinary RMSE reported. No replay state resets.",
        "state_scales": STATE_SCALES,
        "data_consistency_limitation": "L110 .770833-.833333s has native delta_theta minus trapezoid omega integral +.003411303rad; Euler projection does not remove it. Integrated theta and native omega cannot both reproduce this interval exactly. No correction injected.",
        "calibration_provenance": "Early-only trials did not simultaneously improve archived best full-launch rotation and continuation. Practical calibration expanded to2.45s on L110/L27.5; weights were tuned on those cases. L55 excluded from all coefficient fits but inspected historically and during crosscase comparison; not a blind validation.",
        "fit_substeps_per_48Hz_tick": FIT_SUBSTEPS}
    (OUT/"fit.json").write_text(json.dumps(predeclared, indent=2)+"\n")
    seeds = ([14., 0., 881.374875, 33.588067, 3.542585],
             [14.4623, .26418, 860.161, 31.447, 3.5553])
    coeff, details = optimize({n:shots[n] for n in TRAIN_NAMES}, seeds)
    record = {**predeclared, "fitted_coefficients": coeff.tolist(), "optimization": details}
    (OUT/"fit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"fit": record["fitted_coefficients"], "optimization": details}), flush=True)


def refine_shape():
    """One bounded same-family refinement after concrete shape regression."""
    global SHAPE_REFERENCE, FIT_SUBSTEPS
    prior = json.loads((OUT/"fit.json").read_text())
    shots, _ = load()
    constant_record = json.loads((old.OUT/"schedule_48hz"/"zero_memory_standard_D_constant_control"/"results.json").read_text())
    constants = np.r_[14., 0., np.asarray(constant_record["refit_coefficients"], float)]
    SHAPE_REFERENCE = {n: metrics(shots[n], simulate(shots[n], constants, substeps=10,
                       family="archive_constant")) for n in TRAIN_NAMES}
    FIT_SUBSTEPS = 8
    prior["shape_refinement"] = {"time_limit_s":120, "fit_cases":list(TRAIN_NAMES),
        "requirement":"Refinement objective targets no training-case recovery/peak regression to reserve crosscase margin; final shape acceptance allows at most5percent recovery/peak amplitude/time regression. Full omega/alpha/gamma RMSE below archive_constant. All samples retained.",
        "provenance":"Added after root acceptance found early recovery regression. L55 and frozen continuation excluded from refinement objective; prior case comparisons informed the reviewed requirement.",
        "prior_aggregate_candidate_coefficients":prior["fitted_coefficients"],
        "penalty_weight":100, "substeps_per_tick":FIT_SUBSTEPS}
    (OUT/"fit.json").write_text(json.dumps(prior, indent=2)+"\n")
    coeff, details = optimize({n:shots[n] for n in TRAIN_NAMES},
        [prior["fitted_coefficients"], constants.tolist()])
    prior["shape_refinement"]["optimization"] = details
    prior["shape_refinement"]["refined_coefficients"] = coeff.tolist()
    prior["fitted_coefficients"] = coeff.tolist()
    prior["fit_substeps_per_48Hz_tick"] = FIT_SUBSTEPS
    (OUT/"fit.json").write_text(json.dumps(prior, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"shape_refined_coefficients":coeff.tolist(), "optimization":details}), flush=True)
    finalize_refinement()


def validate():
    frozen = json.loads((OUT/"fit.json").read_text())
    shots, _ = load()
    coeff = np.array(frozen["fitted_coefficients"])
    oldcoeff, _, _ = damping.load()
    constant_record = json.loads((old.OUT/"schedule_48hz"/"zero_memory_standard_D_constant_control"/"results.json").read_text())
    constant_coeff = np.asarray(constant_record["refit_coefficients"], float)
    curves, records, verification, acceptance = {}, {}, {}, {}
    arrays = {}
    for name, shot in shots.items():
        curves[name] = {"old_best": simulate(shot, np.r_[14., 0., constant_coeff], substeps=10, family="archive_constant"),
                        "old_balanced": simulate(shot, np.r_[14., 0., oldcoeff], substeps=10, family="archive_qbar"),
                        "new": simulate(shot, coeff, substeps=10)}
        if "shape_refinement" in frozen:
            curves[name]["shape_candidate"] = simulate(shot,
                frozen["shape_refinement"]["refined_coefficients"], substeps=10)
        assert curves[name]["new"] is not None
        records[name] = {tag:metrics(shot, pred) for tag, pred in curves[name].items()}
        acceptance[name] = {"full_rmse_improvement_percent_vs_old_best": {
            key:100*(1-records[name]["new"]["windows"]["full"][key+"_rmse"] /
                       records[name]["old_best"]["windows"]["full"][key+"_rmse"])
            for key in ("omega", "alpha", "theta", "gamma", "y")}}
        acceptance[name]["omega_alpha_improve"] = all(acceptance[name]["full_rmse_improvement_percent_vs_old_best"][k] > 0
                                                    for k in ("omega", "alpha"))
        acceptance[name]["shape"] = shape_acceptance(records[name]["old_best"], records[name]["new"])
        if "shape_candidate" in curves[name]:
            acceptance[name]["shape_candidate"] = shape_acceptance(records[name]["old_best"], records[name]["shape_candidate"])
        coarse = simulate(shot, coeff)
        fine = simulate(shot, coeff, substeps=20)
        main = curves[name]["new"]
        verification[name] = {"step_halving_max_abs": {key:float(np.max(np.abs(main[key]-np.interp(main["t"], fine["t"], fine[key]))))
                               for key in ("omega", "alpha", "gamma", "y")},
            "fit_vs_final_max_abs": {key:float(np.max(np.abs(coarse[key]-np.interp(coarse["t"], main["t"], main[key]))))
                                      for key in ("omega", "alpha", "gamma")},
            "one_surface_identity_max_nm": float(np.max(np.abs(main["moment_nm"]+
               shot["p"]["tail_arm_m"]*main["normal_body_n"]-main["reference_couple_nm"]))),
            "first_active_s": main["first_active"]["start_s"]}
        legacy = old.simulate(shot, constant_coeff, "zero_memory_standard_D", "constant_control",
                              schedule="48hz_end_hold", plant_dt=1/960, horizon=HORIZON)
        lt = [row["t"] for row in legacy]
        verification[name]["archived_implementation_reproduction_max_abs"] = {
            key:float(np.max(np.abs(np.interp(lt, curves[name]["old_best"]["t"], curves[name]["old_best"][key])-
                                      np.array([row[key] for row in legacy]))))
            for key in ("omega", "alpha", "gamma", "y")}
        assert verification[name]["step_halving_max_abs"]["omega"] < 2e-6
        assert verification[name]["one_surface_identity_max_nm"] < 1e-8
        assert verification[name]["first_active_s"] == 14/48
        assert verification[name]["archived_implementation_reproduction_max_abs"]["omega"] < 2e-6
        if "shape_candidate" in curves[name]:
            candidate = curves[name]["shape_candidate"]
            candidate_fine = simulate(shot, frozen["shape_refinement"]["refined_coefficients"], substeps=20)
            verification[name]["shape_candidate_step_halving_max_abs"] = {
                key:float(np.max(np.abs(candidate[key]-np.interp(candidate["t"], candidate_fine["t"], candidate_fine[key]))))
                for key in ("omega", "alpha", "gamma", "y")}
            verification[name]["shape_candidate_one_surface_identity_max_nm"] = float(np.max(np.abs(
                candidate["moment_nm"]+shot["p"]["tail_arm_m"]*candidate["normal_body_n"]-candidate["reference_couple_nm"])))
            assert verification[name]["shape_candidate_step_halving_max_abs"]["omega"] < 2e-6
            assert verification[name]["shape_candidate_one_surface_identity_max_nm"] < 1e-8
        for tag, pred in curves[name].items():
            for key, value in pred.items():
                if isinstance(value, np.ndarray): arrays[name+"__"+tag+"__"+key] = value
        for key, value in shot["observed"].items(): arrays[name+"__observed__"+key] = value
    result = {**frozen, "old_best": {"family": "zero_memory_standard_D_constant_control",
               "base_coefficients": constant_coeff.tolist(), "archive": str(old.OUT/"schedule_48hz"/"zero_memory_standard_D_constant_control"/"results.json")},
              "old_balanced": {"family": "zero_memory_standard_D_qbar_control", "base_coefficients": oldcoeff.tolist()},
              "records": records, "verification": verification, "acceptance": acceptance,
              "accepted_full_window_omega_alpha": all(a["omega_alpha_improve"] for a in acceptance.values()),
              "accepted_stronger_shape": all(a["shape"]["stronger_shape_pass"] for a in acceptance.values()),
              "native_game_truth": False}
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(OUT/"trajectories.npz", **arrays)
    plot(shots, curves)
    print(json.dumps({"coefficients": coeff.tolist(), "full_forward_rmse": {n:{tag: records[n][tag]["windows"]["full"] for tag in curves[n]} for n in shots},
                      "verification": verification}, indent=2), flush=True)


def plot(shots, curves):
    fig, axes = old.plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey="row", layout="constrained")
    for col, name in enumerate(NAMES):
        obs = shots[name]["observed"]
        for row, (key, scale, label) in enumerate((("omega", 1., "Body pitch rate (rad/s)"),
                ("alpha", 180/math.pi, "Angle of attack (deg)"),
                ("theta", 180/math.pi, "Pitch angle (deg)"),
                ("gamma", 180/math.pi, "Flight-path angle (deg)"))):
            ax = axes[row, col]
            keep = obs["t"] <= HORIZON
            ax.plot(obs["t"][keep], obs[key][keep]*scale, "k.", ms=4.5, label="Native replay", zorder=5)
            for tag, color, label in (("old_best", "#d48732", "Archived best rotation fit"),
                                       ("old_balanced", "#999999", "Archived balanced baseline"),
                                       ("new", "#2677ad", "Aggregate candidate"),
                                       ("shape_candidate", "#8449a4", "Recovery candidate; residual tradeoffs")):
                if tag not in curves[name]: continue
                pred = curves[name][tag]
                keep = pred["t"] <= HORIZON
                ax.plot(pred["t"][keep], pred[key][keep]*scale, color=color,
                        lw=(1.7 if tag=="shape_candidate" else 1.3),
                        ls=("--" if tag=="new" else "-"), label=label)
            ax.axvline(FIT_END, color=".6", ls=":", lw=.7)
            ax.axvspan(2.55, HORIZON, color="#e5eef6", alpha=.3)
            ax.grid(alpha=.2)
            ax.set(xlim=(0., HORIZON), xlabel="Physical age from creation (s)")
            if col == 0: ax.set_ylabel(label if row == -1 else ("Body pitch rate (rad/s)", "Angle of attack (deg)", "Pitch angle (deg)", "Flight-path angle (deg)")[row])
            if row == 0: ax.set_title(name+(" — crosscase" if name == "L55" else " — fit to 2.45 s / frozen late"))
    axes[0,0].legend(fontsize=8)
    fig.suptitle("R-77-1 loft OFF | continuous from creation | fixed PID / 48 Hz / .4 m equivalent surface\nMeasured speed and target are inputs; five shared empirical coefficients; two retained candidates", fontsize=13)
    fig.savefig(OUT/"comparison.png", dpi=180)
    old.plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "validate", "refine_shape"))
    args = parser.parse_args()
    if args.action == "train": train()
    elif args.action == "refine_shape": refine_shape()
    else: validate()
