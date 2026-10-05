"""One bounded virtual-target PN handover trial with the frozen replay plant.

The virtual target has a constant creation-derived position offset and exactly
the real target's velocity. Guidance commands, rather than target positions,
are blended by a nonincreasing forward-LOS-rate strategy weight. This is an
empirical use of the four raw loft numbers, not a claim about native game code.
"""
from __future__ import annotations

import json
import math
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import analyze_r771_frozen_loft as loft
import fit_r771_empirical_loft as empirical
import analyze_r771_h110_l110_delta as paired

frozen, shared, old, model = loft.frozen, loft.shared, loft.old, loft.model
OUT = old.batch.BASE / "virtual_target_strategy"
BASE = old.batch.BASE / "empirical_loft_function"
TRAIN = ("H110", "H27.5")
METRICS = ("omega", "alpha", "gamma", "y")
FIT_WINDOWS = ("full", "frozen_continuation")
MODES = ("empirical_loft", "virtual_target")
BOUNDS = np.array([[.25, 1.], [.5, 2.]])
FIT_SUBSTEPS, FINAL_SUBSTEPS = 4, 10
MAX_EVALUATIONS, FIT_SECONDS = 120, 300.
EXPECTED_COEFF = np.array([14.606915444766543, .17017244388915495,
                          865.9202017326287, 31.833800106342316])
TRACE_KEYS = ("t", "weight", "raw_weight", "pn_real_g", "pn_virtual_g",
              "pn_blend_g", "recovery_raw_g", "recovery_projected_g",
              "a_req_g", "true_los_rate_deg_s", "true_los_deg", "xi",
              "real_range_m", "virtual_range_m", "real_closing_mps",
              "virtual_closing_mps", "virtual_los_rate_deg_s", "real_x",
              "real_y", "virtual_x", "virtual_y", "real_vx", "real_vy",
              "virtual_vx", "virtual_vy", "missile_x", "missile_y",
              "theta", "omega", "gamma", "alpha",
              "normal_limit_active", "vector_limit_active", "hit_gain")


class StrategyRequest:
    """Fresh per-shot/per-trial memory, called exactly once per guidance tick."""

    def __init__(self, original, shot, parameters):
        self.original, self.shot = original, shot
        self.f, self.c_rate = map(float, parameters)
        self.target0 = model.target_at(shot["replay"], 0.)[0]
        self.r0 = self.target0-shot["initial"][3:5]
        self.R0 = float(np.linalg.norm(self.r0))
        h0 = self.f*self.R0*math.tan(shot["p"]["loft_elevation_rad"])
        self.virtual0 = shot["initial"][3:5]+self.f*self.r0+np.array([0., h0])
        self.offset = self.virtual0-self.target0
        self.weight, self.last_t, self.rows = 1., None, []

    def __call__(self, t, state, replay, p, loft_armed=True,
                 force_geometric_active=None):
        prior = self.original(t, state, replay, p, loft_armed,
                              force_geometric_active)
        if not p["loft_enabled"]:
            return prior
        assert replay is self.shot["replay"]
        if self.last_t is None:
            assert t == 0.
        else:
            assert abs(t-self.last_t-1/48) < 1e-10
        self.last_t = t
        gamma = float(state[2])
        normal = np.array([-math.sin(gamma), math.cos(gamma)])
        real_xy, real_vxy = model.target_at(replay, t)
        virtual_xy, virtual_vxy = real_xy+self.offset, real_vxy.copy()
        speed = float(np.interp(t, replay["t"], replay["speed2"]))
        missile_vxy = speed*np.array([math.cos(gamma), math.sin(gamma)])
        virtual_r = virtual_xy-state[3:5]
        virtual_rel_v = virtual_vxy-missile_vxy
        pn_vec, closing, los_rate = model.pn_acceleration(
            (*virtual_r, 0.), (*virtual_rel_v, 0.), (*missile_vxy, 0.), p["pn_gain"])
        virtual_pn = float(np.dot(pn_vec[:2], normal))*prior["hit_gain"]
        if p.get("schedule_gate") != "total_pid_output":
            virtual_pn *= prior["flight_gain"]
        real_pn = prior["pn_for_command_g"]*model.G
        true_rate = math.degrees(prior["los_rate_z_rad_s"])
        threshold = self.c_rate*p["loft_omega_max_deg_s"]
        z = p["loft_angle_to_accel_mult"]*(abs(true_rate)/threshold-1.)
        raw_weight = 1./(1.+math.exp(float(np.clip(z, -700., 700.))))
        self.weight = min(self.weight, raw_weight)
        xi = float(np.clip(prior["range_m"]/self.R0, 0., 1.))
        recovery_g = (1.-self.weight)*p["loft_target_elevation_deg"]*xi**2
        recovery_normal = recovery_g*model.G*normal[1]
        blend = self.weight*virtual_pn+(1.-self.weight)*real_pn
        limit = p["req_accel_max_g"]*model.G
        unconstrained = blend+recovery_normal
        command = model.clamp(unconstrained, -limit, limit)
        vector = command*normal+np.array([0., model.G])
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > limit:
            vector *= limit/vector_norm
        required = model.clamp(float(np.dot(vector, normal)), -limit, limit)
        self.rows.append([t, self.weight, raw_weight, real_pn/model.G,
            virtual_pn/model.G, blend/model.G, recovery_g,
            recovery_normal/model.G, required/model.G, true_rate,
            prior["los_elevation_deg"], xi, prior["range_m"],
            float(np.linalg.norm(virtual_r)), prior["closing_mps"], closing,
            math.degrees(los_rate[2]), *real_xy, *virtual_xy, *real_vxy,
            *virtual_vxy, *state[3:5], *state[:3], state[0]-state[2], float(abs(unconstrained) > limit),
            float(vector_norm > limit), prior["hit_gain"]])
        return dict(prior, a_req_g=required/model.G,
                    error_g=required/model.G-prior["actual_g"],
                    pn_g=blend/model.G, pn_for_command_g=blend/model.G,
                    loft_raw_g=recovery_g, loft_limited_g=recovery_g,
                    loft_projected_g=recovery_normal/model.G,
                    strategy_weight=self.weight)

    def trace(self):
        a = np.asarray(self.rows, float)
        return {key: a[:, i] for i, key in enumerate(TRACE_KEYS)}


def run(shot, coeff, original, parameters, substeps=FINAL_SUBSTEPS):
    request = StrategyRequest(original, shot, parameters)
    previous = model.request
    try:
        model.request = request
        pred = frozen.simulate(shot, coeff, substeps=substeps, horizon=3.5)
    finally:
        model.request = previous
    return pred, request


def score(records, reference, squared=False):
    ratios = [records[name]["windows"][window][key+"_rmse"] /
              reference[name]["windows"][window][key+"_rmse"]
              for name in TRAIN for window in FIT_WINDOWS for key in METRICS]
    return float(np.mean(np.square(ratios) if squared else ratios))


def optimize(shots, coeff, original, reference):
    started, history, cache = time.monotonic(), [], {}
    def evaluate(parameters):
        z = np.clip(parameters, BOUNDS[:, 0], BOUNDS[:, 1])
        key = tuple(np.round(z, 9))
        if key in cache:
            return cache[key]
        if len(history) >= MAX_EVALUATIONS or time.monotonic()-started >= FIT_SECONDS:
            return None
        predictions = {n: run(shots[n], coeff, original, z, FIT_SUBSTEPS)[0] for n in TRAIN}
        valid = all(p is not None for p in predictions.values())
        loss = score({n: empirical.metrics(shots[n], p) for n, p in predictions.items()},
                     reference, squared=True) if valid else 1e12
        row = {"parameters": z.tolist(), "objective": loss, "valid": valid,
               "evaluation": len(history)+1}
        cache[key] = row
        history.append(row)
        if len(history) % 6 == 0:
            print(json.dumps({"search_evaluations": len(history),
                "best": min(history, key=lambda r:r["objective"]),
                "elapsed_s": time.monotonic()-started}), flush=True)
        return row
    for f in np.linspace(.25, 1., 6):
        for c in np.linspace(.5, 2., 6):
            if evaluate([f, c]) is None:
                break
    valid = [r for r in history if r["valid"]]
    if not valid:
        raise RuntimeError("Every bounded virtual-target trial violated the existing .6rad guard")
    best = min(valid, key=lambda r:r["objective"])
    steps = np.array([.075, .15])
    stop = "bounded_pattern_step_tolerance"
    while steps.max() > .001:
        previous = best
        center = np.array(best["parameters"])
        for df, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            row = evaluate(center+steps*np.array([df, dc]))
            if row is None:
                stop = "evaluation_or_time_budget"
                break
            if row["objective"] < best["objective"]:
                best = row
        if stop == "evaluation_or_time_budget":
            break
        if best is previous:
            steps *= .5
    # Final selection uses ordinary normalized RMSE, rather than the squared
    # search objective, at final integration resolution for the top three.
    finalists = []
    for row in sorted((r for r in history if r["valid"]), key=lambda r:r["objective"])[:3]:
        predictions = {n: run(shots[n], coeff, original, row["parameters"])[0] for n in TRAIN}
        if any(p is None for p in predictions.values()):
            continue
        records = {n: empirical.metrics(shots[n], p) for n, p in predictions.items()}
        finalists.append({"parameters":row["parameters"],
            "final_selection_score":score(records, reference), "final_train_metrics":records})
    if not finalists:
        raise RuntimeError("All final-resolution candidates invalid")
    chosen = min(finalists, key=lambda r:r["final_selection_score"])
    return np.array(chosen["parameters"]), {"evaluations":len(history),
        "invalid_trials":sum(not r["valid"] for r in history),
        "elapsed_s":time.monotonic()-started, "stop":stop,
        "history":history, "finalists":finalists,
        "selected_final_score":chosen["final_selection_score"]}


def paired_deltas(shots, references, candidates, arrays):
    common_lo = max(shots[n]["observed"]["t"][0] for n in ("H110", "L110"))
    common_hi = min(3.5, *(shots[n]["observed"]["t"][-1] for n in ("H110", "L110")))
    grid = np.arange(math.ceil(common_lo*48-1e-10), math.floor(common_hi*48+1e-10)+1)/48
    assert grid[0] >= common_lo and grid[-1] <= common_hi+1e-12
    deltas = {}
    for label, curves in (("observed", {n:shots[n]["observed"] for n in shots}),
                          (MODES[0], references), (MODES[1], candidates)):
        deltas[label] = {}
        for key in paired.KEYS:
            def values(n):
                curve = curves[n]
                if key == "speed2" and key not in curve:
                    return np.interp(grid, shots[n]["replay"]["t"], shots[n]["replay"]["speed2"])
                return np.interp(grid, curve["t"], curve[key])
            delta = values("H110")-values("L110")
            if key in ("x", "y"):
                index = 3 if key == "x" else 4
                delta -= shots["H110"]["initial"][index]-shots["L110"]["initial"][index]
            deltas[label][key] = delta
            arrays["paired__"+label+"__"+key] = delta
        np.testing.assert_allclose(deltas[label]["alpha"],
            deltas[label]["theta"]-deltas[label]["gamma"], atol=2e-16)
    arrays["paired__common_age_grid_s"] = grid
    events = {}
    for label, key, lo, hi, kind in (("delta_omega_peak1","omega",.35,.60,"max"),
            ("delta_omega_valley1","omega",.60,.84,"min"),
            ("delta_omega_peak2","omega",.85,1.18,"max"),
            ("delta_alpha_peak1","alpha",.38,.76,"max"),
            ("delta_alpha_negative_extreme","alpha",1.,2.45,"min")):
        events[label] = {mode:paired.event(grid, signals[key]*paired.SCALES[key],lo,hi,kind)
                         for mode, signals in deltas.items()}
    stats = {mode:paired.window_stats(grid, deltas["observed"], deltas[mode]) for mode in MODES}
    crossing = {mode:paired.crossings(grid, d["alpha"],.8) for mode,d in deltas.items()}
    return grid, deltas, {"events":events,"alpha_positive_to_negative_crossings":crossing,
        "missing_response_windows":stats, "common_grid_bounds_s":[float(grid[0]),float(grid[-1])],
        "common_grid_count":len(grid), "controlled_loft_only_pair":False,
        "geometry_caveat":"Creation states and target tracks are kept separately; H110/L110 initial target relative heights differ by about150m, so the observed subtraction is not a pure causal loft response."}


def draw(shots, reference, candidates, grid, deltas, traces):
    colors = {MODES[0]:"#8b48ad", MODES[1]:"#2371ad"}
    labels = {MODES[0]:"Current empirical loft", MODES[1]:"Virtual-target strategy"}
    rows = (("omega","Pitch rate [rad/s]"),("alpha","Incidence [deg]"),
            ("theta","Pitch [deg]"),("gamma","Flight-path angle [deg]"))
    fig, axs = plt.subplots(4,4,figsize=(17,12),sharex=True)
    for col,name in enumerate(loft.H_NAMES):
        for row,(key,label) in enumerate(rows):
            scale = paired.SCALES[key]
            obs=shots[name]["observed"]
            ax=axs[row,col]
            ax.plot(obs["t"],obs[key]*scale,".",color="#111111",ms=3,label="Decoded replay")
            for mode,curves in ((MODES[0],reference),(MODES[1],candidates)):
                ax.plot(curves[name]["t"],curves[name][key]*scale,color=colors[mode],lw=1.5,label=labels[mode])
            ax.axvspan(2.55,3.5,color="#e8e8e8",alpha=.45)
            ax.set_xlim(0,3.5);ax.grid(alpha=.22)
            if col==0:ax.set_ylabel(label)
            if row==0:ax.set_title(name)
            if row==3:ax.set_xlabel("Native b5 age [s]")
    axs[0,0].legend(fontsize=8)
    fig.suptitle("Frozen plant/PID: one virtual-target PN handover trial; early and late windows fit-informed")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/"comparison.png",dpi=160);plt.close(fig)
    rows=(("omega","Pitch-rate difference [rad/s]"),("alpha","Incidence difference [deg]"),
          ("gamma","Flight-path difference [deg]"),("y","Vertical-displacement difference [m]"))
    fig,axs=plt.subplots(4,2,figsize=(13,12),sharex=True)
    for row,(key,label) in enumerate(rows):
        scale=paired.SCALES[key]
        axs[row,0].plot(grid,deltas["observed"][key]*scale,color="#111111",lw=1.7,label="Decoded H110−L110")
        for mode in MODES:
            axs[row,0].plot(grid,deltas[mode][key]*scale,color=colors[mode],lw=1.5,label=labels[mode])
            axs[row,1].plot(grid,(deltas["observed"][key]-deltas[mode][key])*scale,
                           color=colors[mode],lw=1.5,label=labels[mode]+" residual")
        for ax in axs[row]:ax.axhline(0,color="#666",lw=.6);ax.grid(alpha=.22);ax.set_xlim(0,3.5)
        axs[row,0].set_ylabel(label)
    axs[0,0].set_title("H110−L110 paired response");axs[0,1].set_title("Observed pair minus modeled pair")
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    for ax in axs[-1]:ax.set_xlabel("Native b5 age [s]")
    fig.suptitle("Own creation/V/target inputs preserved; pair has different target geometry; no time shift")
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/"paired_delta.png",dpi=170);plt.close(fig)
    tr=traces["H110"]
    fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True)
    axs[0].plot(tr["t"],tr["weight"],label="Virtual PN weight (running minimum)")
    axs[0].plot(tr["t"],tr["raw_weight"],"--",label="Current LOS-based target weight")
    axs[0].set_ylabel("Strategy weight");axs[0].legend()
    for key,label in (("pn_real_g","Real target PN6"),("pn_virtual_g","Fixed-offset virtual target PN6"),
                      ("pn_blend_g","Blended PN6"),("recovery_projected_g","Recovery bias")):
        axs[1].plot(tr["t"],tr[key],label=label)
    axs[1].set_ylabel("Normal component [g]");axs[1].legend(fontsize=8)
    axs[2].plot(tr["t"],tr["a_req_g"],color=colors[MODES[1]],label="Virtual strategy total request")
    axs[2].plot(reference["H110"]["t"],reference["H110"]["a_req_g"],color=colors[MODES[0]],label="Current empirical total request")
    axs[2].set_ylabel("Normal request [g]");axs[2].legend()
    axs[3].plot(tr["t"],np.abs(tr["true_los_rate_deg_s"]),label="Absolute real-target LOS rate")
    axs[3].axhline(traces["threshold_deg_s"],color="#666",ls="--",label="Fitted conversion × raw 0.5")
    axs[3].set_ylabel("Internal LOS-rate scale");axs[3].legend();axs[3].set_xlabel("Native b5 age [s]")
    for ax in axs:ax.set_xlim(0,3.5);ax.grid(alpha=.22)
    fig.suptitle("H110: recorded forward guidance calls; real and virtual target velocities identical")
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"guidance_components.png",dpi=170);plt.close(fig)


def main(final_parameters=None):
    snapshot_bytes = loft.SNAPSHOT.read_bytes()
    cfg, coeff, _ = loft.read_snapshot()
    assert np.array_equal(coeff, EXPECTED_COEFF)
    raw, p0 = old.batch.load_all()
    shots = {n:loft.make_loft_shot(raw[n]) for n in loft.H_NAMES}
    shots.update({n:old.make_shot(raw[n],"b5") for n in shared.NAMES})
    original = model.request
    saved = np.load(BASE/"trajectories.npz")
    saved_meta = json.loads((BASE/"results.json").read_text())
    assert np.array_equal(saved_meta["frozen_coefficients"], coeff)
    reference = {n:{k:saved[n+"__empirical_loft__"+k].copy() for k in
                   ("t","theta","omega","gamma","x","y","alpha","u","I","feedback_g","a_req_g",
                    "normal_body_n","normal_wind_n","moment_nm","reference_couple_nm","alpha_local","qdot","gamma_dot")}
                 for n in shots}
    reference_records = {n:empirical.metrics(shots[n],reference[n]) for n in shots}
    reference_metric_difference=max(abs(reference_records[n]["windows"][w][k+"_rmse"]-
        saved_meta["records"][n]["empirical_loft"]["windows"][w][k+"_rmse"])
        for n in shots for w in reference_records[n]["windows"] for k in loft.UNITS)
    assert reference_metric_difference==0.
    objective = {"formula":"Mean squared baseline-normalized ordinary RMSE over omega/alpha/gamma/y × H110/H27.5 × 0..2.45/2.55..3.5s; equal16 entries; no event weights",
        "final_selection":"Lowest arithmetic mean of the16 normalized ordinary RMSE values at10substeps among three best valid search points",
        "parameter_names":["virtual_downrange_fraction_f","LOS_rate_scale_conversion_c_rate"],
        "bounds":BOUNDS.tolist(),"coarse_grid_shape":[6,6],"max_objective_evaluations":MAX_EVALUATIONS,
        "fit_time_budget_s":FIT_SECONDS,"fit_substeps":FIT_SUBSTEPS,"final_substeps":FINAL_SUBSTEPS,
        "extra_event_residuals":False,"predeclared_before_search":True}
    print(json.dumps({"predeclared_search":objective}),flush=True)
    if final_parameters is None:
        parameters, optimization = optimize(shots,coeff,original,reference_records)
    else:
        # Retry only the final artifact/verification pass after an invalid
        # counterfactual probe. The completed search is not repeated.
        parameters = np.asarray(final_parameters,float)
        assert np.array_equal(parameters,np.array([1.,1.255859375]))
        optimization = {"evaluations":None,"evaluation_count_bounds_from_progress":[78,83],
            "max_evaluations":MAX_EVALUATIONS,"stop":"bounded_pattern_step_tolerance",
            "selected_final_score":.5537547342293293,
            "search_was_completed_before_final_probe_failure":True,
            "search_repeated":False,"final_only_retry":True}
    print(json.dumps({"chosen":parameters.tolist(),"score":optimization["selected_final_score"],
                      "verification":"starting one focused final pass"}),flush=True)
    candidates,requests,traces,records,arrays={},{},{},{},{}
    for name,shot in shots.items():
        assert shot["p"]["tail_arm_m"]==.4
        assert [shot["p"][k] for k in ("kp","ki","kd","integral_limit")]==[.002125,.006481,.000023,1.]
        pred,request=run(shot,coeff,original,parameters)
        if pred is None:raise RuntimeError(f"Final candidate violated existing .6rad guard: {name}")
        candidates[name],requests[name]=pred,request
        records[name]={MODES[0]:reference_records[name],MODES[1]:empirical.metrics(shot,pred)}
        for mode,pr in ((MODES[0],reference[name]),(MODES[1],pred)):
            for key,value in pr.items():
                if isinstance(value,np.ndarray):arrays[name+"__"+mode+"__"+key]=value
        for key,value in shot["observed"].items():
            assert np.array_equal(value,saved[name+"__observed__"+key])
            arrays[name+"__observed__"+key]=value.copy()
        if name in loft.H_NAMES:
            traces[name]=request.trace()
            for key,value in traces[name].items():arrays[name+"__strategy_trace__"+key]=value
    traces["threshold_deg_s"]=float(parameters[1]*shots["H110"]["p"]["loft_omega_max_deg_s"])
    grid,deltas,paired_records=paired_deltas(shots,reference,candidates,arrays)
    accepted=np.load(old.OUT/"simple_joint_calibration/trajectories.npz")
    baseline_diff,sample_counts,first_active,geometry={},{},{},{}
    for name,shot in shots.items():
        assert len(shot["observed"]["t"])==len(raw[name]["b5"])==shot["sample_count"]
        sample_counts[name]=shot["sample_count"]
        first_active[name]=candidates[name]["first_active"]["start_s"]
        assert first_active[name]==14/48
        if name in shared.NAMES:
            baseline_diff[name]={k:float(np.max(np.abs(v-accepted[name+"__affine__"+k])))
                for k,v in candidates[name].items() if isinstance(v,np.ndarray)}
            assert max(baseline_diff[name].values())==0.
            assert requests[name].rows==[] and requests[name].weight==1.
        else:
            tr,request=traces[name],requests[name]
            assert len(tr["t"])==169 and np.all(np.diff(tr["weight"])<=0.)
            assert np.all((tr["weight"]>=0.)&(tr["weight"]<=1.))
            np.testing.assert_allclose(tr["virtual_x"]-tr["real_x"],request.offset[0],atol=2e-12)
            np.testing.assert_allclose(tr["virtual_y"]-tr["real_y"],request.offset[1],atol=2e-12)
            assert np.array_equal(tr["real_vx"],tr["virtual_vx"])
            assert np.array_equal(tr["real_vy"],tr["virtual_vy"])
            geometry[name]={"initial_real_target_xy":request.target0.tolist(),
                "initial_virtual_target_xy":request.virtual0.tolist(),"fixed_offset_xy":request.offset.tolist(),
                "initial_range_m":request.R0,"internal_LOS_rate_threshold":traces["threshold_deg_s"],
                "handover_weight_below":{"0.9":next((float(t) for t,w in zip(tr["t"],tr["weight"]) if w<=.9),None),
                    "0.5":next((float(t) for t,w in zip(tr["t"],tr["weight"]) if w<=.5),None),
                    "0.1":next((float(t) for t,w in zip(tr["t"],tr["weight"]) if w<=.1),None)},
                "normal_limit_ticks":int(tr["normal_limit_active"].sum()),
                "vector_limit_ticks":int(tr["vector_limit_active"].sum()),
                "max_abs_alpha_rad":float(np.max(np.abs(candidates[name]["alpha"]))),
                "request_range_g":[float(tr["a_req_g"].min()),float(tr["a_req_g"].max())]}
    fine,fine_request=run(shots["H110"],coeff,original,parameters,2*FINAL_SUBSTEPS)
    assert fine is not None
    h=candidates["H110"]
    halving={k:float(np.max(np.abs(h[k]-np.interp(h["t"],fine["t"],fine[k])))) for k in ("theta","omega","alpha","gamma","y")}
    assert halving["omega"]<1e-5 and halving["alpha"]<1e-6
    # Same fresh factory gives an independent repeated trajectory; this checks
    # stateful strategy isolation across the preceding trials and seven cases.
    repeated,repeat_request=run(shots["H110"],coeff,original,parameters)
    repeat_diff=max(float(np.max(np.abs(v-repeated[k]))) for k,v in h.items() if isinstance(v,np.ndarray))
    assert repeat_diff==0.
    assert all(np.array_equal(v,repeat_request.trace()[k]) for k,v in traces["H110"].items())
    sensitivity={}
    fields={"loftElevation":"loft_elevation_rad","loftTargetElevation":"loft_target_elevation_deg",
            "loftAngleToAccelMult":"loft_angle_to_accel_mult","loftTargetOmegaMax":"loft_omega_max_deg_s"}
    for native,key in fields.items():
        sensitivity[native]={}
        for fraction in (.9,1.1):
            varied=dict(shots["H110"],p=dict(shots["H110"]["p"]))
            varied["p"][key]*=fraction
            pp,rr=run(varied,coeff,original,parameters)
            if pp is None:
                trace=rr.trace()
                command_difference=float(np.max(np.abs(trace["a_req_g"]-
                    np.interp(trace["t"],traces["H110"]["t"],traces["H110"]["a_req_g"]))))
                assert command_difference>1e-12 and all(np.isfinite(v).all() for v in trace.values())
                sensitivity[native][str(fraction)]={"raw_multiplier":fraction,
                    "full_forward_valid":False,"failure":"Existing0.6rad incidence validity guard; no reset or plant clamp introduced",
                    "recorded_valid_guidance_support_s":[float(trace["t"][0]),float(trace["t"][-1])],
                    "partial_forward_command_max_abs_delta_g":command_difference,
                    "partial_forward_omega_max_abs_delta":float(np.max(np.abs(trace["omega"]-
                        np.interp(trace["t"],traces["H110"]["t"],traces["H110"]["omega"])))),
                    "initial_virtual_xy_delta_m":(rr.virtual0-requests["H110"].virtual0).tolist()}
                continue
            state_difference={k:float(np.max(np.abs(pp[k]-h[k]))) for k in ("omega","alpha","gamma","y")}
            command_difference=float(np.max(np.abs(rr.trace()["a_req_g"]-traces["H110"]["a_req_g"])))
            assert command_difference>1e-12 and state_difference["omega"]>1e-12
            sensitivity[native][str(fraction)]={"raw_multiplier":fraction,
                "full_forward_valid":True,
                "forward_command_max_abs_delta_g":command_difference,"forward_state_max_abs_delta":state_difference,
                "initial_virtual_xy_delta_m":(rr.virtual0-requests["H110"].virtual0).tolist()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    assert loft.SNAPSHOT.read_bytes()==snapshot_bytes
    assert np.array_equal(loft.read_snapshot()[1],EXPECTED_COEFF)
    assert model.request is original
    for mode in MODES:
        for key in paired.KEYS:
            missing=deltas["observed"][key]-deltas[mode][key]
            arrays["paired__missing_"+mode+"__"+key]=missing
            mask=(grid<=3.5+1e-10)
            expected=float(np.sqrt(np.mean((missing[mask]*paired.SCALES[key])**2)))
            actual=paired_records["missing_response_windows"][mode]["whole_0_to_3p5"]["signals"][key]["missing_response_rmse"]
            assert abs(expected-actual)<1e-12
    ratios={n:{window:{key:records[n][MODES[1]]["windows"][window][key+"_rmse"] /
                        records[n][MODES[0]]["windows"][window][key+"_rmse"] for key in loft.UNITS}
                 for window in records[n][MODES[1]]["windows"]} for n in shots}
    result={"status":"one_bounded_virtual_target_strategy_completed",
        "parameters":parameters.tolist(),"raw_loft_fields":{"loftElevation":24.,"loftTargetElevation":-3.5,
            "loftAngleToAccelMult":17.5,"loftTargetOmegaMax":.5,"propNavMult":6.},
        "formula":{"creation_virtual_target":"V0=M0+f*(T0-M0)+f*R0*tan(24*pi/180)*ey",
            "moving_virtual_target":"V(t)=T(t)+(V0-T0); virtual velocity = real target velocity exactly",
            "weight":"s=logistic[-17.5*(abs(real_LOS_rate_in_internal_deg_s)/(c_rate*0.5)-1)]; w[n]=min(w[n-1],s[n]); w[-1]=1",
            "command":"w*PN6(virtual)+(1-w)*PN6(real)+(1-w)*(-3.5)*clip(R/R0,0,1)^2*g*normal_y; unchanged gravity and50g limits"},
        "empirical_semantics":"The internal angle/rate and g interfaces are chosen conversions for this one trial; raw native units and game law are not asserted. PN commands are blended; a moving blended point with omitted motion is not used.",
        "frozen_snapshot":str(loft.SNAPSHOT),"frozen_coefficients":coeff.tolist(),"fixed_reference_damping_s_minus1":frozen.KW,
        "production_changed":False,"native_game_law_identified":False,
        "fit_cases":list(TRAIN),"fit_windows_s":[[0.,2.45],[2.55,3.5]],"withheld_cases":["H55native","H55clone"],
        "withheld_provenance":"Not optimized in this trial; historically inspected, not blind holdout",
        "predeclared_search":objective,"optimization":optimization,"reference_family":saved_meta["selected_family"],
        "reference_parameters":saved_meta["selected_coefficients"],"records":records,"candidate_over_current_rmse_ratios":ratios,
        "paired_H110_L110":paired_records,"forward_strategy_geometry":geometry,
        "verification":{"frozen_config_bytes_and_coefficients_unchanged":True,"all_arrays_finite":True,
            "all_native_observations_equal_existing_archive":True,"native_sample_counts_all_retained":sample_counts,
            "saved_current_reference_metric_reconstruction_max_abs":reference_metric_difference,
            "no_loft_all_array_difference_from_accepted_max_abs":baseline_diff,"first_active_start_s":first_active,
            "H110_step_halving_max_abs":halving,"fresh_factory_repeat_all_array_max_abs":repeat_diff,
            "fresh_factory_repeat_strategy_traces_exact":True,"request_patch_restored":model.request is original,
            "virtual_position_offset_constant":True,"virtual_and_target_velocities_exactly_equal":True,
            "weights_nonincreasing_in_unit_interval":True,"paired_delta_and_missing_rmse_recomputed":True,
            "individual_raw_parameter_sensitivity":sensitivity},
        "limitations":["Replay prescribed speed and external target tracks preclude energy/range/hit probability claims",
            "Frozen plant extrapolates from near4deg incidence into large-angle loft motion",
            "One bounded two-parameter family does not identify native loft semantics or establish a global optimum",
            "Both fit windows now fit-informed; H55 cases historically inspected",
            "Pair difference is confounded by creation/target geometry differences; no fitted clock or observed missile-state feeding"]}
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    draw(shots,reference,candidates,grid,deltas,traces)
    print(json.dumps({"out":str(OUT),"parameters":parameters.tolist(),"optimization":{
        k:v for k,v in optimization.items() if k not in ("history","finalists")},
        "all_H_whole_trace_ratios":{n:ratios[n]["whole_trace"] for n in loft.H_NAMES},
        "paired_events":paired_records["events"],"paired_alpha_crossings":paired_records["alpha_positive_to_negative_crossings"],
        "geometry":geometry,"verification":{"finite":True,"frozen_unchanged":True,"no_loft_exact":True,"halving":halving}},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--final-parameters",nargs=2,type=float,
        help="Retry final verification for this completed fit only; accepts1.0,1.255859375")
    args=parser.parse_args()
    main(args.final_parameters)
