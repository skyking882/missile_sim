"""Bounded empirical loft-function identification with the frozen replay plant.

The four original numbers receive explicit empirical control roles here. Their
native units and native execution law are not asserted. No production file is
changed, and no-loft returns the original request without modification.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import matplotlib.pyplot as plt

import analyze_r771_frozen_loft as loft

frozen, shared, old, model = loft.frozen, loft.shared, loft.old, loft.model
OUT = old.batch.BASE / "empirical_loft_function"
TRAIN = ("H110", "H27.5")
WITHHELD = ("H55native", "H55clone")
METRICS = ("omega", "alpha", "gamma", "y")
FAMILIES = ("exponential", "rational")
FOLLOWUP_FAMILY = "rational_range_taper"
FOLLOWUP_WINDOWS = ("full", "frozen_continuation")
MODES = ("current_pn", "empirical_loft")
FIT_SUBSTEPS, FINAL_SUBSTEPS = 4, 10
BOUNDS = (.25, 4.)
MAX_EVALUATIONS = 100
TOTAL_FIT_SECONDS = 480.
EXPECTED_COEFF = np.array([14.606915444766543, .17017244388915495,
                          865.9202017326287, 31.833800106342316])


def empirical_bias(p, theta, omega, los, los_rate, coeff, family):
    c_scale, c_shape = coeff
    eta = theta - los
    eta_dot = omega - los_rate
    d = max(eta, 0.)
    z = c_shape * p["loft_angle_to_accel_mult"] * d
    w = math.exp(-z) if family == "exponential" else 1. / (1. + z)
    lift = math.degrees(p["loft_elevation_rad"])
    recovery = p["loft_target_elevation_deg"]
    endpoint = c_scale * (lift*w + recovery*(1.-w))
    rate = -c_scale * p["loft_omega_max_deg_s"] * eta_dot
    return endpoint + rate, eta, eta_dot, w, endpoint, rate


def make_request(original, coeff, family, initial_ranges=None):
    if family == FOLLOWUP_FAMILY:
        assert initial_ranges is not None
    def request(t, state, replay, p, loft_armed=True, force_geometric_active=None):
        prior = original(t, state, replay, p, loft_armed, force_geometric_active)
        if not p["loft_enabled"]:
            return prior
        theta, omega, gamma = map(float, state[:3])
        los = math.radians(prior["los_elevation_deg"])
        bias, eta, eta_dot, w, endpoint, rate = empirical_bias(
            p, theta, omega, los, prior["los_rate_z_rad_s"], coeff, family)
        xi = 1.
        if family == FOLLOWUP_FAMILY:
            xi = float(np.clip(prior["range_m"]/initial_ranges[id(replay)], 0., 1.))
            bias *= xi**2
            endpoint *= xi**2
            rate *= xi**2
        normal = np.array([-math.sin(gamma), math.cos(gamma)])
        loft_normal = bias * model.G * normal[1]
        limit = p["req_accel_max_g"] * model.G
        command = model.clamp(prior["pn_for_command_g"]*model.G + loft_normal, -limit, limit)
        vector = command*normal + np.array([0., model.G])
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > limit:
            vector *= limit/vector_norm
        required = model.clamp(float(np.dot(vector, normal)), -limit, limit)
        return dict(prior, a_req_g=required/model.G,
                    error_g=required/model.G-prior["actual_g"],
                    loft_raw_g=bias, loft_limited_g=bias,
                    loft_projected_g=loft_normal/model.G,
                    loft_reference_rad=eta, loft_target_rad=0.,
                    empirical_relative_rate=eta_dot, empirical_transition_weight=w,
                    empirical_range_fraction=xi,
                    empirical_endpoint_bias_g=endpoint, empirical_rate_bias_g=rate,
                    empirical_vector_limit_active=vector_norm > limit)
    return request


def run(shot, plant_coeff, request, substeps=FINAL_SUBSTEPS, horizon=3.5):
    previous = model.request
    try:
        model.request = request
        return frozen.simulate(shot, plant_coeff, substeps=substeps, horizon=horizon)
    finally:
        model.request = previous


def normalized_residual(shots, preds, reference, fit_windows=("full",)):
    parts = []
    for name in TRAIN:
        obs, pr = shots[name]["observed"], preds[name]
        for window in fit_windows:
            _, lo, hi = next(row for row in loft.WINDOWS if row[0] == window)
            mask = (obs["t"] >= lo-1e-9) & (obs["t"] <= hi+1e-9)
            n = int(np.sum(mask))
            for key in METRICS:
                baseline = reference[name]["windows"][window][key+"_rmse"]
                delta = np.interp(obs["t"][mask], pr["t"], pr[key])-obs[key][mask]
                parts.append(delta/(baseline*math.sqrt(n*len(METRICS)*len(TRAIN)*len(fit_windows))))
    return np.concatenate(parts)


def train_family(shots, plant_coeff, original, reference, family, deadline,
                 fit_windows=("full",), initial_ranges=None):
    started = time.monotonic()
    count, trace = 0, []
    def evaluate(z):
        nonlocal count
        if count >= MAX_EVALUATIONS or time.monotonic() >= deadline:
            return None
        count += 1
        request = make_request(original, z, family, initial_ranges)
        horizon = 3.5 if "frozen_continuation" in fit_windows else 2.5
        preds = {name: run(shots[name], plant_coeff, request, FIT_SUBSTEPS, horizon)
                 for name in TRAIN}
        if any(pr is None for pr in preds.values()):
            return None
        return normalized_residual(shots, preds, reference, fit_windows)
    z = np.array([1., 1.])
    r = evaluate(z)
    if r is None:
        raise RuntimeError(f"{family} initial point invalid or budget unavailable")
    loss, lm = float(r@r), .03
    stop = "iteration_limit"
    for iteration in range(30):
        if count+3 > MAX_EVALUATIONS or time.monotonic() >= deadline:
            stop = "evaluation_or_time_budget"
            break
        jac = []
        for j in range(2):
            zz = z.copy()
            delta = .0005*max(1., abs(z[j]))
            zz[j] = min(BOUNDS[1], z[j]+delta)
            if zz[j] == z[j]: zz[j] = max(BOUNDS[0], z[j]-delta)
            rr = evaluate(zz)
            if rr is None:
                stop = "invalid_difference_or_budget"
                break
            jac.append((rr-r)/(zz[j]-z[j]))
        if len(jac) != 2: break
        jac = np.asarray(jac).T
        normal, gradient = jac.T@jac, jac.T@r
        improved, change, accepted_step = False, 0., np.zeros(2)
        for _ in range(7):
            step = np.linalg.solve(normal+lm*np.diag(np.maximum(np.diag(normal), .01)), -gradient)
            zz = np.clip(z+step, *BOUNDS)
            rr = evaluate(zz)
            trial = math.inf if rr is None else float(rr@rr)
            if trial < loss:
                change, accepted_step = loss-trial, zz-z
                z, r, loss = zz, rr, trial
                lm = max(1e-7, lm*.4)
                improved = True
                break
            lm *= 5.
            if count >= MAX_EVALUATIONS or time.monotonic() >= deadline: break
        trace.append({"iteration": iteration+1, "loss": loss, "coefficients": z.tolist()})
        if iteration % 5 == 0:
            print(json.dumps({"family": family, "evaluations": count, **trace[-1]}), flush=True)
        if not improved:
            stop = "no_accepted_local_step"
            break
        if change < 1e-7 and np.max(np.abs(accepted_step)) < 1e-4:
            stop = "local_step_tolerance"
            break
    return z, {"coefficients": z.tolist(), "seed": [1., 1.], "bounds": list(BOUNDS),
               "coefficient_names": ["c_scale", "c_shape"],
               "bound_active": [bool(abs(v-BOUNDS[0]) < 1e-7 or abs(v-BOUNDS[1]) < 1e-7) for v in z],
               "objective_mean_squared_baseline_normalized_residual": loss,
               "evaluations": count, "elapsed_s": time.monotonic()-started,
               "stop": stop, "trace": trace}


def ratio_records(records):
    return {name: {window: {key: rec[MODES[1]]["windows"][window][key+"_rmse"] /
                              rec[MODES[0]]["windows"][window][key+"_rmse"]
                            for key in loft.UNITS}
                   for window in [row[0] for row in loft.WINDOWS]+["whole_trace"]}
            for name, rec in records.items()}


def metrics(shot, prediction):
    rec = loft.metrics(shot, prediction)
    obs = shot["observed"]
    mask = (obs["t"] >= -1e-9) & (obs["t"] <= 3.5+1e-9)
    whole = {"native_sample_count":int(np.sum(mask))}
    for key in loft.UNITS:
        delta = np.interp(obs["t"][mask], prediction["t"], prediction[key])-obs[key][mask]
        whole[key+"_rmse"] = float(np.sqrt(np.mean(delta**2)))
        whole[key+"_mean_error"] = float(np.mean(delta))
    rec["windows"]["whole_trace"] = whole
    return rec


def selection_score(train_metrics, reference, windows):
    return float(np.mean([train_metrics[name]["windows"][window][key+"_rmse"] /
        reference[name]["windows"][window][key+"_rmse"]
        for name in TRAIN for window in windows for key in METRICS]))


def request_trace(shot, pred, request):
    rows = []
    for tick in range(round(3.5*48)+1):
        t = tick/48
        state = np.array([np.interp(t, pred["t"], pred[k]) for k in ("theta", "omega", "gamma", "x", "y")])
        req = request(t, state, shot["replay"], shot["p"])
        rows.append([t, req["a_req_g"], req["pn_for_command_g"], req["loft_raw_g"],
                     req["loft_projected_g"], req.get("empirical_transition_weight", np.nan),
                     req.get("empirical_relative_rate", 0.), req.get("empirical_endpoint_bias_g", 0.),
                     req.get("empirical_rate_bias_g", 0.), req["los_elevation_deg"],
                     req["los_rate_z_rad_s"], req["gravity_compensation_g"],
                     float(req.get("empirical_vector_limit_active", False)),
                     req.get("empirical_range_fraction", 0.)])
    a = np.asarray(rows)
    # Baseline has no empirical transition; its absent weight is represented by 0.
    a[:, 5] = np.nan_to_num(a[:, 5])
    keys = ("t", "a_req_g", "pn_g", "bias_g", "projected_bias_g", "weight",
            "relative_rate", "endpoint_g", "rate_g", "los_deg", "los_rate", "gravity_g", "vector_limit_active", "xi")
    return {key: a[:, i] for i, key in enumerate(keys)}


def draw(shots, preds, lo, hi, continuation=False):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22})
    rows = (("omega", 1., "Pitch rate [rad/s]"), ("alpha", 180/math.pi, "Incidence [deg]"),
            ("gamma", 180/math.pi, "Flight path [deg]"), ("y", 1., "Altitude [m]")) if continuation else (
            ("omega", 1., "Pitch rate [rad/s]"), ("alpha", 180/math.pi, "Incidence [deg]"),
            ("theta", 180/math.pi, "Pitch [deg]"), ("gamma", 180/math.pi, "Flight path [deg]"))
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for col, name in enumerate(loft.H_NAMES):
        obs = shots[name]["observed"]
        for row, (key, scale, label) in enumerate(rows):
            ax = axes[row, col]
            ax.plot(obs["t"], obs[key]*scale, ".", color="#202020", ms=3, label="Decoded replay")
            for mode, color, label_curve in ((MODES[0], "#15856f", "Previous PN + loft"),
                    (MODES[1], "#8b48ad", "Empirical loft function")):
                pr = preds[name][mode]
                ax.plot(pr["t"], pr[key]*scale, color=color, lw=1.7, label=label_curve)
            ax.set_xlim(lo, hi)
            if col == 0: ax.set_ylabel(label)
            if row == 0: ax.set_title(name + (" (fit)" if name in TRAIN else " (withheld)"))
            if row == 3: ax.set_xlabel("Native b5 age [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Empirical loft control comparison; frozen plant/PID; replay speed and target remain inputs" +
                 ("; this late window now participates in fit" if continuation else "; early and late windows fit with four balanced metrics"))
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT/("continuation.png" if continuation else "comparison.png"), dpi=180)
    plt.close(fig)


def draw_guidance(traces):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    oldtr, newtr = traces["H110"][MODES[0]], traces["H110"][MODES[1]]
    axes[0].plot(oldtr["t"], oldtr["bias_g"], color="#15856f", label="Previous loft bias")
    axes[0].plot(newtr["t"], newtr["bias_g"], color="#8b48ad", label="New total bias")
    axes[0].plot(newtr["t"], newtr["endpoint_g"], "--", color="#bc7dcc", label="24 / -3.5 blended component")
    axes[0].plot(newtr["t"], newtr["rate_g"], color="#b17423", label="0.5 relative-rate component")
    axes[0].set_ylabel("Control bias [g interface]")
    axes[0].legend(fontsize=8)
    axes[1].plot(newtr["t"], newtr["weight"], color="#8b48ad", label="w: 17.5 controls transition")
    axes[1].plot(newtr["t"], newtr["xi"]**2, "--", color="#b17423", label="(R/R0)^2: geometric fade")
    axes[1].set_ylabel("Transition weight w")
    axes[1].set_ylim(-.05, 1.05)
    axes[1].legend(fontsize=8)
    for tr, color, label in ((oldtr, "#15856f", "Previous total request"), (newtr, "#8b48ad", "New total request")):
        axes[2].plot(tr["t"], tr["a_req_g"], color=color, label=label)
    axes[2].plot(newtr["t"], newtr["pn_g"], "--", color="#777777", label="PN contribution (new trajectory)")
    axes[2].set_ylabel("Normal request [g]")
    axes[2].set_xlabel("Native b5 age [s]")
    axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlim(0., 3.5); ax.grid(alpha=.22)
    fig.suptitle("H110: explicit empirical roles of all four unchanged raw loft numbers")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT/"guidance_components.png", dpi=180)
    plt.close(fig)


def main():
    snapshot_bytes = loft.SNAPSHOT.read_bytes()
    cfg, coeff, _ = loft.read_snapshot()
    assert np.array_equal(coeff, EXPECTED_COEFF)
    raw, p0 = old.batch.load_all()
    shots = {n: loft.make_loft_shot(raw[n]) for n in loft.H_NAMES}
    shots.update({n: old.make_shot(raw[n], "b5") for n in shared.NAMES})
    original = model.request
    for shot in shots.values():
        assert shot["p"]["tail_arm_m"] == .4
        assert [shot["p"][k] for k in ("kp", "ki", "kd", "integral_limit")] == [.002125, .006481, .000023, 1.]
    baseline = {name: run(shot, coeff, original) for name, shot in shots.items()}
    assert all(pr is not None for pr in baseline.values())
    reference = {name: metrics(shots[name], baseline[name]) for name in shots}
    initial_ranges = {id(shot["replay"]): float(np.linalg.norm(
        model.target_at(shot["replay"], 0.)[0]-shot["initial"][3:5])) for shot in shots.values()}
    # Preserve the completed, early-only comparison without repeating its fit.
    previous_result = json.loads((OUT/"results.json").read_text())
    previous_candidates = previous_result["optimization"]["candidates"]
    early_only = previous_result.get("early_only_experiment", {
        "selected_family": previous_result["selected_family"],
        "selected_coefficients": previous_result["selected_coefficients"],
        "selected_formula": previous_result["selected_formula"],
        "selection_windows": ["full"], "records": previous_result["records"],
        "candidate_over_baseline_rmse_ratios": previous_result["candidate_over_baseline_rmse_ratios"],
        "candidates": {family: previous_candidates[family] for family in FAMILIES},
        "late_window_was_excluded_from_fit": True})
    start = time.monotonic()
    candidates = {family: dict(previous_candidates[family]) for family in FAMILIES}
    candidate_predictions = {}
    # Old final-resolution metrics already contain both windows. Re-score them
    # on the explicitly changed objective; keep their original selection score.
    for family in FAMILIES:
        candidates[family]["balanced_early_late_selection_score"] = selection_score(
            candidates[family]["final_train_metrics"], reference, FOLLOWUP_WINDOWS)
        candidates[family]["fit_windows"] = ["full"]
    z, detail = train_family(shots, coeff, original, reference, FOLLOWUP_FAMILY,
                            start+300., FOLLOWUP_WINDOWS, initial_ranges)
    req = make_request(original, z, FOLLOWUP_FAMILY, initial_ranges)
    final_train = {name: run(shots[name], coeff, req) for name in TRAIN}
    if any(pr is None for pr in final_train.values()):
        raise RuntimeError("Range-taper final simulation invalid")
    detail["final_train_metrics"] = {name: metrics(shots[name], final_train[name]) for name in TRAIN}
    detail["fit_windows"] = list(FOLLOWUP_WINDOWS)
    detail["balanced_early_late_selection_score"] = selection_score(detail["final_train_metrics"], reference, FOLLOWUP_WINDOWS)
    candidates[FOLLOWUP_FAMILY], candidate_predictions[FOLLOWUP_FAMILY] = detail, final_train
    print(json.dumps({"family": FOLLOWUP_FAMILY, "final_coefficients": z.tolist(),
        "selection_score": detail["balanced_early_late_selection_score"], "elapsed_s":detail["elapsed_s"],
        "stop":detail["stop"]}), flush=True)
    selected = min((*FAMILIES, FOLLOWUP_FAMILY), key=lambda fam: candidates[fam]["balanced_early_late_selection_score"])
    chosen = np.asarray(candidates[selected]["coefficients"])
    selected_request = make_request(original, chosen, selected, initial_ranges)
    if selected not in candidate_predictions:
        candidate_predictions[selected] = {name: run(shots[name], coeff, selected_request) for name in TRAIN}
    predictions, records, arrays, traces = {}, {}, {}, {}
    for name, shot in shots.items():
        if name in TRAIN: pr = candidate_predictions[selected][name]
        else: pr = run(shot, coeff, selected_request)
        if pr is None: raise RuntimeError(f"Selected function violates .6rad guard in {name}")
        predictions[name] = {MODES[0]: baseline[name], MODES[1]: pr}
        records[name] = {mode: metrics(shot, pred) for mode, pred in predictions[name].items()}
        traces[name] = {}
        for mode, request in ((MODES[0], original), (MODES[1], selected_request)):
            pred = predictions[name][mode]
            for key, value in pred.items():
                if isinstance(value, np.ndarray): arrays[name+"__"+mode+"__"+key] = value
            tr = request_trace(shot, pred, request)
            traces[name][mode] = tr
            for key, value in tr.items(): arrays[name+"__"+mode+"__request_"+key] = value
            records[name][mode]["forward_guidance"] = {
                "bias_range_g_interface": [float(np.min(tr["bias_g"])), float(np.max(tr["bias_g"]))],
                "request_range_g": [float(np.min(tr["a_req_g"])), float(np.max(tr["a_req_g"]))],
                "relative_rate_component_range": [float(np.min(tr["rate_g"])), float(np.max(tr["rate_g"]))],
                "command_vector_limit_tick_count": int(np.sum(tr["vector_limit_active"]))}
        for key, value in shot["observed"].items(): arrays[name+"__observed__"+key] = value
    print(json.dumps({"selected_family": selected, "verification": "starting focused final pass"}), flush=True)
    # One focused final verification pass, including parameter participation.
    archives = np.load(loft.OUT/"trajectories.npz")
    baseline_diff, no_loft_diff, sample_counts, first_active = {}, {}, {}, {}
    for name, shot in shots.items():
        archive_mode = "pn_plus_loft" if name in loft.H_NAMES else "no_loft"
        baseline_diff[name] = {k: float(np.max(np.abs(v-archives[name+"__"+archive_mode+"__"+k])))
            for k, v in baseline[name].items() if isinstance(v, np.ndarray)}
        assert max(baseline_diff[name].values()) < 1e-12
        assert len(shot["observed"]["t"]) == len(raw[name]["b5"]) == shot["sample_count"]
        sample_counts[name] = shot["sample_count"]
        first_active[name] = {mode: pr["first_active"]["start_s"] for mode, pr in predictions[name].items()}
        assert all(v == 14/48 for v in first_active[name].values())
        if name in shared.NAMES:
            no_loft_diff[name] = {k: float(np.max(np.abs(v-predictions[name][MODES[1]][k])))
                for k, v in baseline[name].items() if isinstance(v, np.ndarray)}
            assert max(no_loft_diff[name].values()) == 0.
            assert selected_request(0., shot["initial"], shot["replay"], shot["p"]) == original(
                0., shot["initial"], shot["replay"], shot["p"])
    hshot, coarse = shots["H110"], predictions["H110"][MODES[1]]
    fine = run(hshot, coeff, selected_request, 2*FINAL_SUBSTEPS)
    assert fine is not None
    halving = {key: float(np.max(np.abs(coarse[key]-np.interp(coarse["t"], fine["t"], fine[key]))))
               for key in ("theta", "omega", "alpha", "gamma", "y")}
    assert halving["omega"] < 1e-5 and halving["alpha"] < 1e-6
    sensitivity = {}
    # Each parameter is changed individually; all others and fitted coefficients stay fixed.
    fields = {"loftElevation": "loft_elevation_rad", "loftTargetElevation": "loft_target_elevation_deg",
              "loftAngleToAccelMult": "loft_angle_to_accel_mult", "loftTargetOmegaMax": "loft_omega_max_deg_s"}
    base_trace = traces["H110"][MODES[1]]
    for native, key in fields.items():
        sensitivity[native] = {}
        for fraction in (.9, 1.1):
            varied = dict(hshot, p=dict(hshot["p"]))
            varied["p"][key] *= fraction
            perturbed = run(varied, coeff, selected_request)
            assert perturbed is not None
            assert all(np.isfinite(v).all() for v in perturbed.values() if isinstance(v, np.ndarray))
            same_state = request_trace(varied, coarse, selected_request)
            changed_state = request_trace(varied, perturbed, selected_request)
            delta_command = same_state["a_req_g"]-base_trace["a_req_g"]
            differences = {k: float(np.max(np.abs(perturbed[k]-coarse[k]))) for k in ("omega", "alpha", "gamma", "y")}
            assert float(np.max(np.abs(delta_command))) > 1e-12
            assert differences["omega"] > 1e-12
            sensitivity[native][str(fraction)] = {
                "raw_multiplier": fraction, "same_state_command_delta_range_g": [float(np.min(delta_command)), float(np.max(delta_command))],
                "same_state_command_max_abs_delta_g": float(np.max(np.abs(delta_command))),
                "forward_command_max_abs_delta_g": float(np.max(np.abs(changed_state["a_req_g"]-base_trace["a_req_g"]))),
                "forward_state_max_abs_delta": differences}
    assert all(np.isfinite(v).all() for v in arrays.values())
    assert loft.SNAPSHOT.read_bytes() == snapshot_bytes
    livecfg, livecoeff, _ = loft.read_snapshot()
    assert cfg == livecfg and np.array_equal(livecoeff, EXPECTED_COEFF)
    assert model.request is original
    ratios = ratio_records(records)
    all_h_full_improve = all(ratios[name]["full"][key] < 1. for name in loft.H_NAMES for key in METRICS)
    result = {
        "status": "all_four_full_metrics_improve" if all_h_full_improve else "best_bounded_candidate_with_tradeoffs",
        "selected_family": selected, "selected_coefficients": chosen.tolist(),
        "selected_formula": ("b=c_scale*xi^2*[24*w-3.5*(1-w)-0.5*(omega-lambda_dot)]; xi=clip(R/R0,0,1); "
            if selected == FOLLOWUP_FAMILY else "b=c_scale*[24*w-3.5*(1-w)-0.5*(omega-lambda_dot)]; ") +
            "eta=theta-lambda; d=max(eta,0); " +
            ("w=exp(-c_shape*17.5*d)" if selected == "exponential" else "w=1/(1+c_shape*17.5*d)"),
        "parameter_roles": {"loftElevation=24": "positive loft control-bias endpoint",
            "loftTargetElevation=-3.5": "recovery control-bias endpoint reached progressively as body moves above target LOS",
            "loftAngleToAccelMult=17.5": "transition steepness in positive body-relative target angle",
            "loftTargetOmegaMax=0.5": "relative body-to-target-LOS rate feedback weight; no prototype loft-only cap"},
        "empirical_semantics": "Raw values unchanged. Angles/rates are evaluated in rad/rad_s in this function and its output enters the existing g command interface. These are chosen empirical control roles, not identification of native game units or code.",
        "frozen_snapshot": str(loft.SNAPSHOT), "frozen_coefficients": coeff.tolist(),
        "fixed_reference_damping_s_minus1": frozen.KW, "source_version": p0["source_version"],
        "raw_loft_fields": {"loftElevation":24., "loftTargetElevation":-3.5, "loftAngleToAccelMult":17.5, "loftTargetOmegaMax":.5, "propNavMult":6.},
        "production_changed": False, "native_game_law_identified": False,
        "fit_cases": list(TRAIN), "fit_windows_s": [[0., 2.45], [2.55, 3.5]], "withheld_cases": list(WITHHELD),
        "withheld_provenance": "Excluded from this fit; historically inspected, not blind holdout",
        "late_window_s": [2.55, 3.5], "late_window_now_participates_in_fit":True,
        "legacy_window_key_note": "frozen_continuation retained as a metric key for artifact compatibility; this window now participates in range-taper fit and final selection",
        "geometric_fade": {"exponent":2, "exponent_fitted":False,
            "R0": "Range from creation-state missile position to target_at(t=0), separately percase",
            "R": "Current forward-state missile position to external target; no observed missile-state feedback",
            "initial_ranges_m": {name:initial_ranges[id(shot["replay"])] for name,shot in shots.items()}},
        "early_only_experiment":early_only,
        "optimization": {"families": list((*FAMILIES,FOLLOWUP_FAMILY)), "coefficient_count":2, "positive_coefficient_bounds":list(BOUNDS),
            "followup_families_fitted": [FOLLOWUP_FAMILY], "original_two_family_results_reused":True,
            "max_evaluations_per_family":MAX_EVALUATIONS, "followup_fit_budget_s":300.,
            "fit_substeps":FIT_SUBSTEPS, "final_substeps":FINAL_SUBSTEPS,
            "objective": "Equal-weight mean squared residual across omega/alpha/gamma/y, both training cases, two windows0..2.45 and2.55..3.5s; each metric/window divided by its previous baseline RMSE",
            "selection_predeclared": "Lowest arithmetic mean of16 baseline-normalized ordinary RMSE values (4metrics x2train cases x2windows), at final integration resolution; both windows now fit-informed; no selection on H55 cases",
            "baseline_selection_score":1.,
            "all_features_claim_gate": "Every four-metric full-window RMSE improves in all four loft cases; peak/recovery and late tradeoffs still reported separately",
            "candidates":candidates, "total_elapsed_s":time.monotonic()-start},
        "simulation": "Continuous creation-state integration; one equivalent surface at0.4m; frozen plant/PID;48Hz/end-gain zero-memory lifecycle; measured speed and packet-derived target are external inputs",
        "common_guidance": "PN6, timeToHit/timeToGain, gravity compensation,50g normal/vector limits unchanged; new function replaces only loft bias and removes old loft-only cap; smooth normalized-range fade, no discrete range/time exit, latch, state forcing, reset, or measured missile feedback",
        "metric_definition": "Ordinary RMSE at all retained native b5 observations, including ties; native events use observed sample extrema",
        "metric_units":loft.UNITS, "records":records, "candidate_over_baseline_rmse_ratios":ratios,
        "verification": {"baseline_saved_array_reproduction_max_abs":baseline_diff,
            "no_loft_all_array_difference_max_abs":no_loft_diff, "native_sample_counts_all_retained":sample_counts,
            "all_arrays_finite":True, "H110_step_halving_max_abs":halving,
            "first_active_start_s":first_active, "frozen_config_bytes_and_coefficients_unchanged":True,
            "request_patch_restored":model.request is original, "individual_raw_parameter_sensitivity":sensitivity},
        "scope_limitations": ["Replay-conditioned fit does not establish hit probability, range, drag/energy efficiency, or full-flight advantage",
            "Frozen linear plant was calibrated near4deg incidence and is extrapolated toward27deg loft responses",
            "Native finsLatAccel field is not dynamically mapped in frozen plant; cross-case agreement does not identify that game mapping",
            "All four loft numbers are shared between captures; perturbation proves this function uses them, not that native semantics are identified",
            "Three total family local fits with bounded positive conversion coefficients do not establish a global optimum; only range-taper family fitted in this followup",
            "Late window now fit-informed; it is no longer an independent continuation test"]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"trajectories.npz", **arrays)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    draw(shots, predictions, 0., 2.45)
    draw(shots, predictions, 2.55, 3.5, continuation=True)
    draw_guidance(traces)
    print(json.dumps({"output":str(OUT), "status":result["status"], "selected":selected,
        "coefficients":chosen.tolist(), "full_metric_ratios": {name:ratios[name]["full"] for name in loft.H_NAMES},
        "baseline_max_difference":max(max(v.values()) for v in baseline_diff.values()),
        "no_loft_max_difference":max(max(v.values()) for v in no_loft_diff.values()),
        "step_halving":halving}, indent=2), flush=True)


if __name__ == "__main__":
    main()
