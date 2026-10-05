"""Bounded joint calibration with three or four shared positive coefficients.

The constant and affine speed gains are nested.  Only the aerodynamic slope,
control gain and restoring gain are calibrated; the existing reference damping
coefficient, PID, creation states and one-surface geometry remain fixed.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import fit_r771_shared_surface as shared

OUT = shared.old.OUT / "simple_joint_calibration"
KW = 3.534664411413437
COEFF_NAMES = ("lift_slope_per_rad", "affine_control_speed_slope",
               "control_gain_s_minus2", "restoring_gain_s_minus2")
UNITS = ("rad^-1", "dimensionless", "s^-2", "s^-2")
BOUNDS = np.array([[10., 20.], [0., 1.], [600., 1100.], [20., 50.]])
SCALE = np.array([14., 1., 850., 32.])
FIT_SUBSTEPS = 8
FINAL_SUBSTEPS = 10
TIME_BUDGET_S = 120.
PENALTY_WEIGHT = 100.
SHAPE_LIMIT = 1.05
ORIGINAL_SURFACE = shared.surface


def surface(t, state, held_u, shot, coeff, family="simple"):
    if family != "simple":
        return ORIGINAL_SURFACE(t, state, held_u, shot, coeff, family)
    ca, b, ku, ka = map(float, coeff)
    theta, omega, gamma, _, y = state
    alpha = theta - gamma
    p, replay = shot["p"], shot["replay"]
    speed = float(np.interp(t, replay["t"], replay["speed2"]))
    speed0 = shot["initial_speed_mps"]
    mass, thrust = shared.model.mass_thrust(t, p)
    qbar = .5 * shared.model.isa_density(y) * speed**2
    area = math.pi * p["caliber_m"]**2 / 4 * p["wing_area_mult"]
    arm = p["tail_arm_m"]
    inertia = mass * p["length_m"]**2 / 12
    qref = .5 * shared.model.isa_density(shot["initial"][4]) * speed0**2
    active = (qbar / mass) / (qref / p["mass0_kg"])
    damping_scale = active * speed0 / speed
    alpha_local = alpha + arm * omega / speed
    normal_wind = qbar * area * ca * alpha_local
    normal_body = normal_wind / math.cos(alpha)
    feedback = (normal_wind + thrust * math.sin(alpha)) / (mass * shared.model.G)
    gamma_dot = shared.model.G * (feedback - math.cos(gamma)) / speed
    control_gain_ratio = 1. + b * (speed / speed0 - 1.)
    qdot = ku * control_gain_ratio * held_u - active * ka * alpha_local - damping_scale * KW * omega
    reference_couple = inertia * qdot + arm * normal_body
    moment = -arm * normal_body + reference_couple
    return {"v": speed, "mass": mass, "thrust": thrust,
            "alpha_local": alpha_local, "normal_wind": normal_wind,
            "normal_body": normal_body, "feedback_g": feedback,
            "gamma_dot": gamma_dot, "qdot": moment / inertia,
            "reference_couple": reference_couple, "moment": moment,
            "inertia": inertia, "active": active,
            "damping_scale": damping_scale, "control_gain_ratio": control_gain_ratio}


# Reuse exactly the existing creation-state RK4/PID execution and archived laws.
# Its family argument selects this function for simple fits and the saved
# original function for every archived comparison.
shared.surface = surface


def simulate(shot, coeff, *, substeps=None, horizon=shared.HORIZON):
    return shared.simulate(shot, coeff, substeps=FIT_SUBSTEPS if substeps is None else substeps,
                           horizon=horizon, family="simple")


def residual(shot, pred, coeff, reference):
    obs = shot["observed"]
    mask = obs["t"] <= shared.FIT_END + 1e-9
    ot = obs["t"][mask]
    values = []
    for key, scale in shared.STATE_SCALES.items():
        values.extend((np.interp(ot, pred["t"], pred[key]) - obs[key][mask]) /
                      (scale * math.sqrt(len(ot) * len(shared.STATE_SCALES))))
    for _, key, lo, hi, kind, scale in shared.EVENTS:
        actual_t, actual_v = shared.damping.extremum(obs["t"], obs[key], lo, hi, kind)
        pred_t, pred_v = shared.damping.extremum(pred["t"], pred[key], lo, hi, kind)
        values.extend([.25 * (pred_t - actual_t) / .03, .25 * (pred_v - actual_v) / scale])
    ids = np.flatnonzero((obs["t"][:-1] >= .03) &
                         (obs["t"][1:] <= shared.FIT_END + 1e-9))
    qbalances, gbalances = [], []
    for i in ids:
        lo, hi = obs["t"][i:i+2]
        middle = (lo + hi) / 2
        state = np.array([.5 * (obs[key][i] + obs[key][i+1])
                          for key in ("theta", "omega", "gamma", "x", "y")])
        j = min(len(pred["t"]) - 1, np.searchsorted(pred["t"], middle, side="right") - 1)
        terms = surface(middle, state, pred["u"][j], shot, coeff)
        qbalances.append((terms["qdot"] * (hi-lo) - obs["omega"][i+1] + obs["omega"][i]) / .03)
        gbalances.append((terms["gamma_dot"] * (hi-lo) - obs["gamma"][i+1] + obs["gamma"][i]) / .0005)
    values.extend(.15 * np.asarray(qbalances) / math.sqrt(len(ids)))
    values.extend(.15 * np.asarray(gbalances) / math.sqrt(len(ids)))
    current = shared.metrics(shot, pred)
    violations = []
    for window, key in (("first_recovery", "alpha"), ("full", "omega"), ("full", "alpha")):
        ratio = current["windows"][window][key+"_rmse"] / reference["windows"][window][key+"_rmse"]
        limit = SHAPE_LIMIT if window == "first_recovery" else 1.
        violations.append(max(0., ratio-limit))
    for label, *_ in shared.EVENTS:
        for field in ("value_error", "parabolic_time_error_s"):
            base = max(abs(reference["events"][label][field]), 1e-6)
            ratio = abs(current["events"][label][field]) / base
            violations.append(max(0., ratio-SHAPE_LIMIT))
    values.extend(PENALTY_WEIGHT * np.asarray(violations))
    return np.asarray(values)


def optimize(shots, reference, seed, free, deadline, max_iterations=60):
    """One local bounded least-squares run; no random starts or family sweep."""
    started = time.monotonic()
    low, high = BOUNDS[:, 0] / SCALE, BOUNDS[:, 1] / SCALE
    z = np.clip(np.asarray(seed) / SCALE, low, high)
    if 1 not in free:
        z[1] = 0.
    count = 0
    trace = []

    def fun(parameters):
        nonlocal count
        count += 1
        coeff = parameters * SCALE
        values = []
        for name in shared.TRAIN_NAMES:
            pred = simulate(shots[name], coeff, horizon=2.5625)
            if pred is None:
                return None
            values.append(residual(shots[name], pred, coeff, reference[name]))
        return np.concatenate(values) / math.sqrt(len(values))

    r = fun(z)
    if r is None:
        raise RuntimeError("Invalid initial simulation")
    loss = float(r @ r)
    lm = .03
    reason = "iteration_limit"
    for iteration in range(max_iterations):
        if time.monotonic() >= deadline:
            reason = "time_budget"
            break
        columns = []
        for j in free:
            delta = .0005 * max(1., abs(z[j]))
            trial_z = z.copy()
            trial_z[j] = min(high[j], z[j] + delta)
            if trial_z[j] == z[j]:
                trial_z[j] = max(low[j], z[j] - delta)
            rr = fun(trial_z)
            columns.append(np.zeros_like(r) if rr is None else (rr-r)/(trial_z[j]-z[j]))
        jac = np.asarray(columns).T
        normal, gradient = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(7):
            step = np.linalg.solve(normal + lm*np.diag(np.maximum(np.diag(normal), .01)), -gradient)
            trial_z = z.copy()
            trial_z[list(free)] += step
            trial_z = np.clip(trial_z, low, high)
            rr = fun(trial_z)
            trial_loss = math.inf if rr is None else float(rr @ rr)
            if trial_loss < loss:
                change = loss - trial_loss
                z, r, loss = trial_z, rr, trial_loss
                lm = max(1e-7, lm*.4)
                improved = True
                break
            lm *= 5
            if time.monotonic() >= deadline:
                break
        trace.append({"iteration": iteration+1, "loss": loss, "coefficients": (z*SCALE).tolist()})
        if iteration % 10 == 0:
            print(json.dumps({"free": list(free), **trace[-1]}), flush=True)
        if not improved:
            reason = "no_accepted_local_step"
            break
        if change < 1e-7 and np.max(np.abs(step)) < 1e-4:
            reason = "local_step_tolerance"
            break
    return z*SCALE, {"objective": loss, "evaluations": count,
                     "elapsed_s": time.monotonic()-started, "stop": reason,
                     "free_indices": list(free), "start": list(seed), "trace": trace}


def archives():
    saved = json.loads((shared.OUT / "results.json").read_text())
    return {"old_constant": np.r_[14., 0., saved["old_best"]["base_coefficients"]],
            "old_balanced": np.r_[14., 0., saved["old_balanced"]["base_coefficients"]],
            "prior_purple": np.asarray(saved["shape_refinement"]["refined_coefficients"]),
            "prior_blue": np.asarray(saved["fitted_coefficients"])}, saved


def train():
    shots, p = shared.load()
    baselines, saved = archives()
    OUT.mkdir(parents=True, exist_ok=True)
    reference = {name: shared.metrics(shots[name], shared.simulate(shots[name], baselines["old_constant"],
                  substeps=FINAL_SUBSTEPS, family="archive_constant")) for name in shared.TRAIN_NAMES}
    record = {"fit_cases": list(shared.TRAIN_NAMES), "fit_window_s": [0., shared.FIT_END],
        "continuation_window_s": [2.55, 3.5], "crosscase": "L55",
        "crosscase_provenance": "Excluded from every current optimization; historically inspected, not a blind holdout.",
        "coefficient_names": list(COEFF_NAMES), "coefficient_units": list(UNITS), "bounds": BOUNDS.tolist(),
        "bounds_provenance": "Local calibration bounds around old Ca14, Ku661-881, Ka31-35 and recent purple Ca14.618/Ku864/Ka31.845. These are experiment bounds, not universal aerodynamic ranges.",
        "fixed_reference_damping_s_minus1": KW,
        "model": "Nw=qbar*S*Ca*(alpha+l*omega/V); Nb=Nw/cos(alpha); qdot=Ku*[1+b*(V/V0-1)]*u-A*Ka*(alpha+l*omega/V)-B*Kw*omega; A=(qbar/m)/(qbar0/m0); B=A*V0/V; Mcm=-l*Nb+Mref; Mref=I*qdot+l*Nb; l=.4m; I=m*L^2/12.",
        "nested_constant": "b=0; free Ca,Ku,Ka only",
        "loss": {"ordinary_state_scales": shared.STATE_SCALES,
                 "ordinary_state_normalization": "1/sqrt(sample_count*4); two training cases equal weight",
                 "five_event_weight": .25, "event_time_scale_s": .03,
                 "native_interval_weight": .15, "angular_interval_scale_rad_s": .03,
                 "flight_path_interval_scale_rad": .0005,
                 "shape_penalty_weight": PENALTY_WEIGHT, "max_recovery_and_peak_regression_ratio": SHAPE_LIMIT,
                 "shape_reference": "old_constant on training cases only; full omega/alpha target <=1, early recovery and five peak amplitude/parabolic time error target <=1.05"},
        "acceptance_rule": "All cases full omega and alpha RMSE improve over oldconstant; recovery and five event amplitude/time error ratios <=1.05. Path and frozen continuation tradeoffs explicit. No boundary-hitting fitted values. Fourth coefficient retained only with >=1% training objective and >=2% training-case frozen continuation mean omega/alpha improvement over nested constant, and no >5% recovery/peak degradation over constant.",
        "budget": {"total_optimizer_limit_s": TIME_BUDGET_S, "nested_cases": 2,
                   "constant_limit_s":45, "affine_limit_s":45, "necessary_refinement_reserve_s":30},
        "fixed_PID": {k:p[k] for k in ("kp", "ki", "kd", "integral_limit")},
        "source_version": p["source_version"], "simulation": "Continuous creation forward integration; measured speed and target motion external; all missile states and fixed PID memory integrated. Same48Hz end gain schedule, no state forcing/reset/time shift.",
        "fit_substeps": FIT_SUBSTEPS, "final_substeps": FINAL_SUBSTEPS,
        "data_consistency_limitation": saved["data_consistency_limitation"],
        "production_changed": False, "native_game_truth": False,
        "archives": {k:v.tolist() for k,v in baselines.items()}}
    (OUT / "fit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    start = time.monotonic()
    purple = baselines["prior_purple"]
    seed = [float(purple[0]), 0., float(purple[2]), float(purple[3])]
    constant, constant_details = optimize(shots, reference, seed, (0, 2, 3), start+45.)
    affine_seed = constant.copy()
    affine_seed[1] = .15
    affine, affine_details = optimize(shots, reference, affine_seed, (0, 1, 2, 3),
                                      min(start+90., time.monotonic()+45.))
    record["candidates"] = {"constant": {"coefficients": constant.tolist(), "optimization": constant_details},
                             "affine": {"coefficients": affine.tolist(), "optimization": affine_details}}
    record["total_optimizer_elapsed_s"] = constant_details["elapsed_s"]+affine_details["elapsed_s"]
    (OUT / "fit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"candidate_coefficients": {k:v["coefficients"] for k,v in record["candidates"].items()},
                      "optimizer_s": record["total_optimizer_elapsed_s"]}), flush=True)


def refine_failed_affine():
    """The single reserved same-loss refinement after concrete event failures."""
    record = json.loads((OUT / "fit.json").read_text())
    prior_result = json.loads((OUT / "results.json").read_text())
    assert any(prior_result["comparisons"][n]["affine"]["old_constant"]["failed_conditions"]
               for n in shared.TRAIN_NAMES)
    assert "necessary_affine_refinement" not in record
    shots, _ = shared.load()
    baselines, _ = archives()
    reference = {name:shared.metrics(shots[name],shared.simulate(shots[name],baselines["old_constant"],
                 substeps=FINAL_SUBSTEPS,family="archive_constant")) for name in shared.TRAIN_NAMES}
    purple = baselines["prior_purple"]
    seed = [float(purple[0]),.15,float(purple[2]),float(purple[3])]
    coeff, details = optimize(shots,reference,seed,(0,1,2,3),time.monotonic()+30.,max_iterations=120)
    record["necessary_affine_refinement"] = {
        "trigger":"Initial affine fails oldconstant qpeak2/alphapeak1 amplitude or timing5% shape limits; initial affine stopped on60 iterations while loss still falling.",
        "unchanged_loss_and_bounds":True,"no_L55_residuals":True,"limit_s":30,
        "starting_coefficients":seed,"substeps":FIT_SUBSTEPS,"optimization":details,
        "initial_affine":record["candidates"]["affine"]}
    if details["objective"] < record["candidates"]["affine"]["optimization"]["objective"]:
        record["candidates"]["affine"] = {"coefficients":coeff.tolist(),"optimization":details,"fit_substeps":FIT_SUBSTEPS}
    record["total_optimizer_elapsed_s"] += details["elapsed_s"]
    (OUT / "fit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"refined_affine":coeff.tolist(),"loss":details["objective"],"optimizer_total_s":record["total_optimizer_elapsed_s"]}),flush=True)


def comparison(base, candidate):
    out = shared.shape_acceptance(base, candidate)
    out["continuation_rmse_ratios"] = {k:candidate["windows"]["frozen_continuation"][k+"_rmse"] /
                                         base["windows"]["frozen_continuation"][k+"_rmse"]
                                         for k in ("omega", "alpha", "gamma", "y")}
    out["full_path_rmse_ratios"] = {k:candidate["windows"]["full"][k+"_rmse"] /
                                    base["windows"]["full"][k+"_rmse"] for k in ("theta", "gamma", "y")}
    return out


def coefficient_scales(shot, pred, coeff):
    ca, b, ku, ka = map(float, coeff)
    terms = [surface(t, np.array([pred[k][i] for k in ("theta", "omega", "gamma", "x", "y")]),
                     pred["u"][i], shot, coeff) for i, t in enumerate(pred["t"])]
    speed = np.array([f["v"] for f in terms])
    gain = np.array([f["control_gain_ratio"] for f in terms])
    active = np.array([f["active"] for f in terms])
    dscale = np.array([f["damping_scale"] for f in terms])
    control = ku*gain*pred["u"]
    restoring = -active*ka*pred["alpha_local"]
    damping = -dscale*KW*pred["omega"]
    total = control+restoring+damping
    magnitude = np.abs(control)+np.abs(restoring)+np.abs(damping)
    coupled = magnitude > .1
    p = shot["p"]
    area = math.pi*p["caliber_m"]**2/4*p["wing_area_mult"]
    qref = .5*shared.model.isa_density(shot["initial"][4])*shot["initial_speed_mps"]**2
    inertia0 = p["mass0_kg"]*p["length_m"]**2/12
    aft_stiffness0 = p["tail_arm_m"]*qref*area*ca/inertia0
    effective_damping = dscale*KW + active*ka*p["tail_arm_m"]/speed
    frozen_gamma_zeta = effective_damping/(2*np.sqrt(active*ka))
    return {"initial_speed_mps": shot["initial_speed_mps"],
            "sampled_speed_range_mps": [float(speed.min()), float(speed.max())],
            "control_gain_ratio_range": [float(gain.min()), float(gain.max())],
            "control_qdot_per_degree_u_minmax_rad_s2": [float((ku*gain*math.pi/180).min()), float((ku*gain*math.pi/180).max())],
            "held_control_max_abs_rad": float(np.max(np.abs(pred["u"]))),
            "held_control_max_abs_deg": float(np.max(np.abs(pred["u"]))*180/math.pi),
            "reference_area_m2": area,
            "reference_area_to_frontal_ratio": p["wing_area_mult"],
            "frontal_area_equivalent_force_slope_per_rad": ca*p["wing_area_mult"],
            "reference_restoring_natural_frequency_rad_s": math.sqrt(ka),
            "restoring_natural_frequency_range_rad_s": [float(np.sqrt(active*ka).min()), float(np.sqrt(active*ka).max())],
            "reference_restoring_frequency_hz": math.sqrt(ka)/(2*math.pi),
            "frozen_gamma_damping_ratio_range": [float(frozen_gamma_zeta.min()),float(frozen_gamma_zeta.max())],
            "aft_force_reference_stiffness_s_minus2": aft_stiffness0,
            "net_restoring_to_aft_reference_stiffness_ratio": ka/aft_stiffness0,
            "force_slope_ratio_to_old_14": ca/14.,
            "angular_term_peak_abs_rad_s2": {"control":float(np.max(np.abs(control))),
                "restoring":float(np.max(np.abs(restoring))), "fixed_damping":float(np.max(np.abs(damping))),
                "net":float(np.max(np.abs(total)))},
            "angular_term_rms_rad_s2": {"control":float(np.sqrt(np.mean(control**2))),
                "restoring":float(np.sqrt(np.mean(restoring**2))), "fixed_damping":float(np.sqrt(np.mean(damping**2))),
                "net":float(np.sqrt(np.mean(total**2)))},
            "median_net_to_sum_abs_active_terms": float(np.median(np.abs(total[coupled])/magnitude[coupled])),
            "warning": "Large SI Ku is interpreted per degree command; this does not establish physical realism. Opposing control/restoring/damping terms are a closed-loop property; their cancellation is reported, not used as proof."}


def validate():
    frozen = json.loads((OUT / "fit.json").read_text())
    shots, _ = shared.load()
    baselines, saved = archives()
    curves, records, comparisons, checks, arrays = {}, {}, {}, {}, {}
    for name, shot in shots.items():
        curves[name] = {"old_constant": shared.simulate(shot, baselines["old_constant"], substeps=FINAL_SUBSTEPS, family="archive_constant"),
                        "old_balanced": shared.simulate(shot, baselines["old_balanced"], substeps=FINAL_SUBSTEPS, family="archive_qbar"),
                        "prior_purple": shared.simulate(shot, baselines["prior_purple"], substeps=FINAL_SUBSTEPS, family="new")}
        curves[name].update({tag:simulate(shot, item["coefficients"], substeps=FINAL_SUBSTEPS)
                             for tag,item in frozen["candidates"].items()})
        assert all(pred is not None for pred in curves[name].values())
        records[name] = {tag:shared.metrics(shot, pred) for tag,pred in curves[name].items()}
        comparisons[name] = {tag:{base:comparison(records[name][base], records[name][tag])
                                  for base in ("old_constant", "old_balanced", "prior_purple")}
                             for tag in frozen["candidates"]}
        checks[name] = {}
        for tag,item in frozen["candidates"].items():
            main = curves[name][tag]
            fine = simulate(shot, item["coefficients"], substeps=2*FINAL_SUBSTEPS)
            coarse = simulate(shot, item["coefficients"])
            stepcheck = {k:float(np.max(np.abs(main[k]-np.interp(main["t"], fine["t"], fine[k]))))
                         for k in ("theta", "omega", "alpha", "gamma", "y")}
            identity = float(np.max(np.abs(main["moment_nm"] + shot["p"]["tail_arm_m"]*main["normal_body_n"] - main["reference_couple_nm"])))
            assert stepcheck["omega"] < 2e-6 and identity < 1e-8
            assert main["first_active"]["start_s"] == 14/48
            assert all(np.isfinite(v).all() for v in main.values() if isinstance(v,np.ndarray))
            checks[name][tag] = {"step_halving_max_abs":stepcheck, "surface_identity_max_nm":identity,
                 "first_active_s":main["first_active"]["start_s"], "all_arrays_finite": True,
                 "fit_to_final_max_abs":{k:float(np.max(np.abs(coarse[k]-np.interp(coarse["t"],main["t"],main[k])))) for k in ("omega","alpha","gamma")}}
        legacy = shared.old.simulate(shot, baselines["old_constant"][2:], "zero_memory_standard_D", "constant_control",
                           schedule="48hz_end_hold", plant_dt=1/960, horizon=shared.HORIZON)
        lt = np.array([r["t"] for r in legacy])
        legacy_err = {k:float(np.max(np.abs(np.interp(lt,curves[name]["old_constant"]["t"],curves[name]["old_constant"][k])-np.array([r[k] for r in legacy]))))
                      for k in ("omega","alpha","gamma","y")}
        assert legacy_err["omega"] < 2e-6
        checks[name]["old_constant_reproduction_max_abs"] = legacy_err
        checks[name]["prior_purple_saved_metrics_max_abs"] = max(abs(records[name]["prior_purple"]["windows"][window][field]-saved["records"][name]["shape_candidate"]["windows"][window][field])
                      for window in records[name]["prior_purple"]["windows"] for field in records[name]["prior_purple"]["windows"][window])
        assert checks[name]["prior_purple_saved_metrics_max_abs"] < 1e-10
        for tag,pred in curves[name].items():
            for key,value in pred.items():
                if isinstance(value,np.ndarray): arrays[f"{name}__{tag}__{key}"] = value
        for key,value in shot["observed"].items(): arrays[f"{name}__observed__{key}"] = value
    affine_vs_constant = {name:comparison(records[name]["constant"],records[name]["affine"]) for name in shots}
    objective_improvement = 1 - frozen["candidates"]["affine"]["optimization"]["objective"] / frozen["candidates"]["constant"]["optimization"]["objective"]
    continuation_ratio = float(np.mean([affine_vs_constant[n]["continuation_rmse_ratios"][k] for n in shared.TRAIN_NAMES for k in ("omega","alpha")]))
    training_shape_ok = all(affine_vs_constant[n]["first_recovery_alpha_rmse_ratio"] <= SHAPE_LIMIT and
        all(ratio <= SHAPE_LIMIT for event in affine_vs_constant[n]["event_error_ratios"].values() for ratio in event.values())
        for n in shared.TRAIN_NAMES)
    b = frozen["candidates"]["affine"]["coefficients"][1]
    fourth_warranted = objective_improvement >= .01 and continuation_ratio <= .98 and training_shape_ok and .01 < b < .99
    automatic_chosen = "affine" if fourth_warranted else "constant"
    chosen = frozen.get("reviewed_selection", {}).get("chosen_candidate", automatic_chosen)
    failures = {n:comparisons[n][chosen]["old_constant"]["failed_conditions"] for n in shots}
    boundary_hits = {tag:[COEFF_NAMES[j] for j,value in enumerate(item["coefficients"])
        if j != 1 or tag != "constant" if min(value-BOUNDS[j,0],BOUNDS[j,1]-value) < .005*(BOUNDS[j,1]-BOUNDS[j,0])]
        for tag,item in frozen["candidates"].items()}
    candidate_scales = {tag:{name:coefficient_scales(shots[name],curves[name][tag],item["coefficients"])
                            for name in shots} for tag,item in frozen["candidates"].items()}
    scales = candidate_scales[chosen]
    for name in shots:
        assert scales[name]["control_gain_ratio_range"][0] > 0.
        assert not boundary_hits[chosen]
        checks[name][chosen].update({"positive_control_gain_full_horizon":True,
             "positive_interiored_coefficients":True,"fixed_reference_damping_s_minus1":KW})
    result = {**frozen, "chosen_candidate":chosen,"predeclared_selected_candidate":automatic_chosen,
        "fourth_coefficient_decision":{"warranted":fourth_warranted,"training_objective_improvement_fraction":objective_improvement,
             "training_case_frozen_continuation_mean_omega_alpha_ratio":continuation_ratio,"training_shape_no_5pct_regression_vs_constant":training_shape_ok,
             "affine_vs_constant":affine_vs_constant},
        "accepted_stronger_fit":not any(failures.values()) and not boundary_hits[chosen],
        "predeclared_fit_acceptance":not any(failures.values()) and not boundary_hits[chosen],
        "failed_conditions_vs_oldconstant":failures,"boundary_hits":boundary_hits,
        "records":records,"comparisons":comparisons,"verification":checks,
        "coefficient_interpretation":scales,"candidate_coefficient_interpretation":candidate_scales,
        "final_coefficients":frozen["candidates"][chosen]["coefficients"]}
    (OUT / "results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(OUT / "trajectories.npz",**arrays)
    plot(shots,curves,chosen)
    print(json.dumps({"chosen_candidate":chosen,"fourth_coefficient_decision":{k:v for k,v in result["fourth_coefficient_decision"].items() if k!="affine_vs_constant"},
                      "coefficients":result["final_coefficients"],"failures":failures,"boundary_hits":boundary_hits}),flush=True)


def plot(shots,curves,chosen):
    fig,axes = shared.old.plt.subplots(4,3,figsize=(15,12),sharex=True,sharey="row",layout="constrained")
    styles = (("old_constant","#d48732","Old best rotation","-"),
              ("old_balanced","#a2a2a2","Old balanced path","-"),
              ("prior_purple","#8449a4","Prior five-coefficient recovery","--"),
              ("constant","#2677ad","Three-coefficient constant control","--"),
              ("affine","#14825e","Four-coefficient affine control","-"))
    for col,name in enumerate(shared.NAMES):
        obs = shots[name]["observed"]
        for row,(key,scale,label) in enumerate((("omega",1.,"Pitch rate (rad/s)"),("alpha",180/math.pi,"Angle of attack (deg)"),
                             ("theta",180/math.pi,"Pitch angle (deg)"),("gamma",180/math.pi,"Flight-path angle (deg)"))):
            ax = axes[row,col]
            keep = obs["t"] <= shared.HORIZON
            ax.plot(obs["t"][keep],obs[key][keep]*scale,"k.",ms=4.5,label="Native replay",zorder=6)
            for tag,color,legend,ls in styles:
                pred = curves[name][tag]
                ax.plot(pred["t"],pred[key]*scale,color=color,ls=ls,lw=1.8 if tag==chosen else 1.15,label=legend)
            ax.axvline(shared.FIT_END,color=".6",ls=":",lw=.7)
            ax.axvspan(2.55,3.5,color="#e5eef6",alpha=.3)
            ax.grid(alpha=.2)
            ax.set(xlim=(0.,shared.HORIZON),xlabel="Physical age from creation (s)")
            if col==0: ax.set_ylabel(label)
            if row==0: ax.set_title(name+(" — excluded crosscase" if name=="L55" else " — calibration to2.45s"))
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Simple joint calibration | one .4 m equivalent surface | fixed PID and damping\nContinuous from creation; measured speed/target inputs; constant and affine control gains",fontsize=13)
    fig.savefig(OUT / "comparison.png",dpi=180)
    shared.old.plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("train","validate","run","refine_failed_affine"))
    args = parser.parse_args()
    if args.action in ("train","run"): train()
    if args.action == "refine_failed_affine": refine_failed_affine()
    if args.action in ("validate","run"): validate()
