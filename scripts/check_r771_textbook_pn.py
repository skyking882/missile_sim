#!/usr/bin/env python3
"""Fixed textbook-inspired PN hypotheses for War Thunder game references.

Predeclared: baseline half/S110; C closing-speed wind-normal; P velocity
PPN; T LOS-normal TPN; B finite-AoA body-normal PPN extension. No fitting,
web/API, plant integration, runtime edits, or new field values. Same-prior
one-step comparisons and separate recursive controller histories are distinct.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_game_request_inverse as audit
import analyze_r771_pn_pid_decoupling as split
from aim120_model.aerodynamics import body_axes_for_state
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import pn_acceleration
from aim120_model.math3d import cross, dot, norm, scale, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/textbook_hypotheses/pn"
NAMES = ("baseline", "C", "P", "T", "B")
FORMULAS = {
    "baseline": "Existing half horizontal relative-PN, gated loft and S110",
    "C": "N*Vc*(omegaLOS cross vhatM), with existing gated loft/S110",
    "P": "N*(omegaLOS cross VM): textbook planar PPN extended to 3D",
    "T": "N*Vc*(omegaLOS cross rhat): textbook planar TPN extended to 3D",
    "B": "N*|VM|*(omegaLOS cross bodyForward): finite-AoA directional hypothesis",
}
COLORS = {"baseline": "#72777d", "C": "#1877ad", "P": "#7a47a5", "T": "#cf4747", "B": "#d28a18"}


def prior_controller(previous_high, previous_k):
    """Baseline pre-tick latch/K; supplied memory is independently cached."""
    c = R771GameControl()
    c.nonlinear_latched = bool(previous_high)
    c.previous_high_k = [float(previous_k), float(previous_k)] if previous_high else [None, None]
    return c


def control_tick(controller, time, request, feedback, memories, pid, gain, k, nonlinear, limits):
    if gain <= 1e-12:
        return np.zeros(2), list(memories), np.zeros(2, dtype=bool), 0.
    outputs = []; next_mem = []; blocked = []; closure = 0.
    for axis in range(2):
        error = float(request[axis] - feedback[axis])
        terms = split.pid_terms(controller, axis, memories[axis], error, pid, gain, k, nonlinear, limits[axis])
        output, mem = controller.update_axis(axis, float(time), error, tuple(memories[axis]), pid,
            float(limits[axis]), float(gain), float(k), bool(nonlinear))
        closure = max(closure, abs(output - terms["held_rad"]), abs(mem[0] - terms["next_integral_rad"]))
        outputs.append(output); next_mem.append(mem); blocked.append(terms["antiwindup_blocked"])
    return np.asarray(outputs), next_mem, np.asarray(blocked), closure


def stats(value, mask):
    value = np.asarray(value)[mask]
    if not len(value):
        return None
    return {"n": int(len(value)), "bias": float(value.mean()),
        "RMS": float(np.sqrt(np.mean(value**2))), "range": [float(value.min()), float(value.max())]}


def protected_snapshot():
    paths = list((ROOT / "src/aim120_model").glob("*.py"))
    paths += [Path(audit.__file__), Path(split.__file__), Path(audit.inverse.__file__), Path(audit.fix.__file__),
        Path(audit.fix.finite.__file__), Path(audit.slew.__file__), Path(audit.drag.__file__),
        Path(audit.comparison.__file__), audit.comparison.SNAPSHOT, audit.comparison.FROZEN,
        audit.comparison.DEFAULT_MANIFEST,
        ROOT / "outputs/jari_offaxis/textbook_hypotheses/sources.json",
        audit.OUTPUT / "request_audit.npz", audit.OUTPUT / "results.json",
        split.OUTPUT / "pn_pid_timeseries.npz", split.OUTPUT / "results.json",
        audit.inverse.OUTPUT / "inverse_inputs.npz",
        audit.fix.finite.OUTPUT / "fit.json", audit.drag.OUTPUT / "fit.json"]
    paths += list((ROOT / "outputs/jari_offaxis/reference").glob("*.json"))
    # Exact protected paths exclude the independent PID worker's new script/output.
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(set(paths)) if p.is_file()}


def case_arrays(cid, item, cache, old):
    ref = item["ref"]; cfg = item["config"]
    a = {key[len(cid)+2:]: cache[key] for key in cache.files if key.startswith(cid+"__")}
    z = {key[len(cid)+2:]: old[key] for key in old.files if key.startswith(cid+"__")}
    t = a["t"]; gains = a["flight_gain"]; limits = a["old_held_limits_rad"]
    k = a["K"]; nonlinear = a["nonlinear_force_active"]
    maximum = float(cfg["guidance"]["maximum_lateral_acceleration_g"]) * float(cfg["atmosphere"]["gravity_mps2"])
    gravity = float(cfg["atmosphere"]["gravity_mps2"])
    pn_gain = float(cfg["guidance"]["pn_gain"])
    if pn_gain != 6.:
        raise AssertionError("Predeclared original PN field changed")
    raw = {name: [] for name in ("C", "P", "T", "B")}
    worlds = {name: [] for name in raw}; pure_projected = {name: [] for name in raw}
    unslewed = {name: [] for name in raw}; single_requests = {name: [] for name in raw}
    single_held = {name: [] for name in raw}; discarded_memory = {name: [] for name in raw}
    own_requests = {name: [] for name in NAMES}; own_held = {name: [] for name in NAMES}
    own_memory = {name: [] for name in NAMES}; own_previous_slew = {name: [] for name in NAMES}
    own_blocked = {name: [] for name in NAMES}; cap_counts = {name: [0, 0, 0] for name in raw}
    own_slew = {name: 0. for name in NAMES}
    controllers = {name: R771GameControl() for name in NAMES}
    memories = {name: [(0.,0.,0.), (0.,0.,0.)] for name in NAMES}
    baseline_held = []; baseline_memory = []
    checks = {"request_to_held_max_abs_rad": 0., "C_same_prior_request_cache_max_abs_G": 0.,
        "C_same_prior_held_cache_max_abs_rad": 0., "C_unslewed_cache_max_abs_G": 0.,
        "baseline_own_request_cache_max_abs_G": 0., "baseline_own_held_cache_max_abs_rad": 0.,
        "baseline_own_memory_cache_max_abs": 0., "same_prior_memory_mutation_max_abs": 0.}
    creation = {}
    for i, ts in enumerate(t):
        state = audit.fix.source_state_at(ref, float(ts)); target = item["target_at"](float(ts))
        rel = sub(target.position, state.position); rel_v = sub(target.velocity, state.velocity)
        pure, closing, los = pn_acceleration(rel, rel_v, state.velocity, pn_gain)
        basis = audit.fix.finite.body_axes_for_state(state)
        # Identical wind basis used by the existing reference-state audit.
        from aim120_model.aerodynamics import wind_normal_basis
        wind_basis = wind_normal_basis(basis, sub(state.velocity, tuple(cfg["atmosphere"].get("wind_mps", (0.,0.,0.)))))
        speed = norm(state.velocity); rhat = scale(rel, 1./norm(rel))
        vectors = {"C": pure, "P": scale(cross(los, state.velocity), pn_gain),
            "T": scale(cross(los, rhat), pn_gain*closing),
            "B": scale(cross(los, basis.forward), pn_gain*speed)}
        if closing <= 1e-9 or speed <= 1e-6:
            vectors = {name: (0.,0.,0.) for name in vectors}
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts)+audit.DT)
        previous_high = bool(a["pid__high_mode"][i-1,0]) if i else False
        previous_k = k[i-1] if i else k[i]
        prior_memory = a["previous_current_memory"][i]
        previous_slew = float(a["previous_slew_yaw_G"][i])
        for name, vector in vectors.items():
            raw[name].append(vector); pure_projected[name].append(split.projected(vector, wind_basis))
            weighted = scale(vector, float(a["hitGain"][i])); worlds[name].append(weighted)
            bounds = split.outer_boundaries(weighted, float(a["loft_actual_G"][i]), wind_basis, gravity, maximum)
            req_unslewed = bounds["unslewed_request_G"]
            if name == "C":
                checks["C_unslewed_cache_max_abs_G"] = max(checks["C_unslewed_cache_max_abs_G"],
                    float(np.max(abs(np.asarray(req_unslewed)-a["cf_original_horizontal_PN_unslewed_G"][i]))))
                # Reuse exact saved C counterfactual; no duplicate guidance replay.
                req_unslewed = tuple(a["cf_original_horizontal_PN_unslewed_G"][i])
            unslewed[name].append(req_unslewed)
            request, _, cap3 = split.same_previous_slew(req_unslewed, previous_slew, gains[i], abs(los[1]), maximum/G0_MPS2)
            cap_counts[name][0] += int(bounds["cap1_active"]); cap_counts[name][1] += int(bounds["cap2_active"]); cap_counts[name][2] += int(cap3)
            c = prior_controller(previous_high, previous_k)
            before = prior_memory.copy()
            held, next_mem, _, error = control_tick(c, ts, request, a["plant_feedback_G"][i],
                [tuple(x) for x in prior_memory], pid, gains[i], k[i], nonlinear[i], limits)
            checks["request_to_held_max_abs_rad"] = max(checks["request_to_held_max_abs_rad"], error)
            checks["same_prior_memory_mutation_max_abs"] = max(checks["same_prior_memory_mutation_max_abs"], float(np.max(abs(prior_memory-before))))
            if name == "C":
                checks["C_same_prior_request_cache_max_abs_G"] = max(checks["C_same_prior_request_cache_max_abs_G"], float(np.max(abs(np.asarray(request)-a["cf_original_horizontal_PN_request_G"][i]))))
                checks["C_same_prior_held_cache_max_abs_rad"] = max(checks["C_same_prior_held_cache_max_abs_rad"], float(np.max(abs(held-a["cf__original_horizontal_PN__held_rad"][i]))))
                request = tuple(a["cf_original_horizontal_PN_request_G"][i]); held = a["cf__original_horizontal_PN__held_rad"][i]
                next_mem = a["cf__original_horizontal_PN__discarded_next_memory"][i]
            single_requests[name].append(request); single_held[name].append(held); discarded_memory[name].append(next_mem)
        for name in NAMES:
            target_request = tuple(a["unslewed_request_G"][i]) if name == "baseline" else unslewed[name][-1]
            own_previous_slew[name].append(own_slew[name])
            request, before_cap, _ = split.same_previous_slew(target_request, own_slew[name], gains[i], abs(los[1]), maximum/G0_MPS2)
            if abs(los[1]) != 0.:
                own_slew[name] = before_cap[1]
            held, memories[name], blocked, error = control_tick(controllers[name], ts, request,
                a["plant_feedback_G"][i], memories[name], pid, gains[i], k[i], nonlinear[i], limits)
            checks["request_to_held_max_abs_rad"] = max(checks["request_to_held_max_abs_rad"], error)
            own_requests[name].append(request); own_held[name].append(held)
            own_memory[name].append(copy.deepcopy(memories[name])); own_blocked[name].append(blocked)
            if name == "baseline":
                baseline_held.append(held); baseline_memory.append(copy.deepcopy(memories[name]))
        if i == 0:
            cosine = dot(rhat, scale(state.velocity, 1/speed)); cy = split.projected(vectors["C"], wind_basis)[1]
            # Exported creation VM is float32 while scenario VT is decimal.
            # Use the actual two speeds for the analytic head-on-axis identity.
            exact_velocity_ratio = speed / ((speed-target.velocity[0])*cosine)
            expected = {"P_over_C":exact_velocity_ratio, "B_over_C":exact_velocity_ratio, "T_projected_over_C":cosine}
            measured = {"P_over_C":pure_projected["P"][-1][1]/cy,
                "B_over_C":pure_projected["B"][-1][1]/cy, "T_projected_over_C":pure_projected["T"][-1][1]/cy}
            creation = {"offaxis_deg":math.degrees(math.acos(cosine)), "expected":expected, "measured":measured,
                "nominal_equal_speed_P_over_C":1/(2*cosine), "creation_VM_mps":speed,
                "creation_target_Vx_mps":target.velocity[0],
                "pure_yaw_G":{name:pure_projected[name][-1][1] for name in raw},
                "hand_ratio_max_abs_error":max(abs(expected[x]-measured[x]) for x in expected),
                "C_dot_velocity":dot(vectors["C"], state.velocity), "P_dot_velocity":dot(vectors["P"], state.velocity),
                "T_dot_LOS":dot(vectors["T"], rhat), "B_dot_bodyForward":dot(vectors["B"], basis.forward)}
    out = {"t":t, "source_t":z["source_t_original"], "K":k, "flight_gain":gains, "limits_rad":limits,
        "feedback_G":a["plant_feedback_G"], "baseline_previous_memory":a["previous_current_memory"],
        "baseline_previous_slew_G":a["previous_slew_yaw_G"], "baseline_high_mode":a["pid__high_mode"],
        "baseline__single_request_G":a["final_request_G"], "baseline__single_held_rad":a["pid__held_rad"]}
    for name in raw:
        for label, values in (("pure_world_mps2",raw), ("hitGain_world_mps2",worlds), ("pure_projected_G",pure_projected),
            ("unslewed_G",unslewed), ("single_request_G",single_requests), ("single_held_rad",single_held),
            ("single_discarded_memory",discarded_memory)):
            out[name+"__"+label] = np.asarray(values[name])
    for name in NAMES:
        for label, values in (("recursive_request_G",own_requests),("recursive_held_rad",own_held),
            ("recursive_memory",own_memory),("recursive_previous_slew_G",own_previous_slew),("recursive_antiwindup",own_blocked)):
            out[name+"__"+label] = np.asarray(values[name])
        out[name+"__single_target_normalized"] = k[:,None]*out[name+"__single_held_rad"]/limits
        out[name+"__recursive_target_normalized"] = k[:,None]*out[name+"__recursive_held_rad"]/limits
    checks["baseline_own_request_cache_max_abs_G"] = float(np.max(abs(out["baseline__recursive_request_G"]-a["final_request_G"])))
    checks["baseline_own_held_cache_max_abs_rad"] = float(np.max(abs(out["baseline__recursive_held_rad"]-a["pid__held_rad"])))
    checks["baseline_own_memory_cache_max_abs"] = float(np.max(abs(out["baseline__recursive_memory"]-a["next_current_memory"])))
    checks["baseline_request_bitwise_equal"] = bool(np.array_equal(out["baseline__recursive_request_G"],a["final_request_G"]))
    checks["baseline_held_bitwise_equal"] = bool(np.array_equal(out["baseline__recursive_held_rad"],a["pid__held_rad"]))
    checks["baseline_memory_bitwise_equal"] = bool(np.array_equal(out["baseline__recursive_memory"],a["next_current_memory"]))
    checks["C_single_request_bitwise_equal_reused"] = bool(np.array_equal(out["C__single_request_G"],a["cf_original_horizontal_PN_request_G"]))
    checks["C_single_held_bitwise_equal_reused"] = bool(np.array_equal(out["C__single_held_rad"],a["cf__original_horizontal_PN__held_rad"]))
    report = {"formulas":FORMULAS,"original_N":pn_gain,"original_PID":cfg["control"]["pid"],
        "frozen_surface":dataclasses.asdict(SurfaceCoefficients()), "original_finsAoa_rad":limits.tolist(),
        "original_loft_fields":{key:cfg["guidance"][key] for key in ("lofting_elevation_deg","target_elevation_deg","angle_to_acceleration_multiplier","loft_omega_max_deg_s")},
        "loft_coefficients_unchanged":[.75,1.3],"clock_hz":48,"plant_substeps_unchanged":4,
        "creation_hand_check":creation,"caps_active_ticks":cap_counts,"methods":{},"checks":checks}
    for width in (3,5):
        prefix = str(width)+"__"
        need = z[prefix+"old_held_required_rad"]; need_norm = z[prefix+"target_normalized_required"]
        one_status = z[prefix+"current_plant__given_memory_status"]
        legacy_status = z[prefix+"legacy_inverse_status"]
        required_r = z[prefix+"current_plant__given_memory_required_request_G"]
        legacy_r = z[prefix+"legacy_required_request_plant_G"]
        half_width = width//2
        halo = (t>=14/48-half_width/24-audit.inverse.BOUNDARY_TOL)&(t<=14/48+half_width/24+audit.inverse.BOUNDARY_TOL)
        source_t = z["source_t_original"]
        derivative_valid = (~halo)&(t>=source_t[half_width])&(t<=source_t[-half_width-1])
        impossible = one_status == -1
        out[prefix+"derivative_halo"] = halo; out[prefix+"derivative_valid"] = derivative_valid
        out[prefix+"required_held_rad"] = need; out[prefix+"required_target_normalized"] = need_norm
        out[prefix+"required_output_impossible"] = impossible
        out[prefix+"baseline_given_memory_required_request_G"] = required_r
        out[prefix+"legacy_required_total_request_plant_G"] = legacy_r
        windows = {}
        for wname, lo, hi in audit.WINDOWS:
            window = audit.inverse.window_mask(t,lo,hi)&(gains>1e-12)
            entry = {"active_ticks":int(window.sum()),"excluded_derivative_halo_ticks":int((window&~derivative_valid).sum()),
                "impossible_pitch_yaw_ticks":np.sum(window[:,None]&impossible,axis=0).tolist(),"candidates":{}}
            for name in NAMES:
                modes = {}
                for mode in ("single","recursive"):
                    modes[mode] = {}
                    for axis, label in enumerate(("pitch","yaw")):
                        valid = window&derivative_valid&(one_status[:,axis]==1)
                        legacy_valid = window&derivative_valid&(legacy_status[:,axis]==1)
                        req = out[name+"__"+mode+"_request_G"][:,axis]
                        target = out[name+"__"+mode+"_target_normalized"][:,axis]
                        modes[mode][label] = {"request_minus_baseline_G":stats(req-out["baseline__"+mode+"_request_G"][:,axis],window),
                            "request_minus_legacy_required_total_G_old_PID_memory":stats(req-legacy_r[:,axis],legacy_valid),
                            "request_minus_given_baseline_memory_required_G":stats(req-required_r[:,axis],valid) if mode=="single" else None,
                            "held_target_minus_required_normalized_feasible_only":stats(target-need_norm[:,axis],valid),
                            "negative_held_ticks":int(np.sum(window&(target<0))),
                            "target_range_all_active":stats(target,window)["range"]}
                entry["candidates"][name] = modes
            windows[wname] = entry
        report["methods"][str(width)] = windows
    report["point_0p5"] = {}
    i = int(np.argmin(abs(t-.5)))
    for name in NAMES:
        report["point_0p5"][name] = {mode:{"request_pitch_yaw_G":out[name+"__"+mode+"_request_G"][i].tolist(),
            "held_pitch_yaw_deg":np.degrees(out[name+"__"+mode+"_held_rad"][i]).tolist(),
            "target_pitch_yaw_normalized":out[name+"__"+mode+"_target_normalized"][i].tolist()} for mode in ("single","recursive")}
    if cid.endswith("60"):
        for width in (3,5):
            if report["methods"][str(width)]["platform"]["impossible_pitch_yaw_ticks"][1] != 77:
                raise AssertionError("Preexisting 60-degree platform impossibility mask changed")
    if not all(np.isfinite(v).all() for v in out.values()):
        raise AssertionError("Nonfinite saved arrays")
    if not all(checks[x] for x in ("baseline_request_bitwise_equal","baseline_held_bitwise_equal","baseline_memory_bitwise_equal")):
        raise AssertionError(checks)
    if max(v for x,v in checks.items() if "max_abs" in x) > 1e-10 or creation["hand_ratio_max_abs_error"] > 1e-12:
        raise AssertionError(checks)
    return out, report


def plot_case(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(14,11),sharex=True)
    t = a["t"]; active = a["flight_gain"]>1e-12
    for axis,label in enumerate(("Pitch","Yaw")):
        for name in NAMES:
            for row,key in ((0,"single_request_G"),(1,"single_target_normalized"),(2,"recursive_target_normalized")):
                axes[row,axis].plot(t[active],a[name+"__"+key][active,axis],color=COLORS[name],lw=1.55,
                    ls="--" if name=="baseline" else "-",label=name)
        valid = a["5__derivative_valid"]&active
        axes[0,axis].plot(t[valid],a["5__legacy_required_total_request_plant_G"][valid,axis],color="#181818",ls=":",lw=1.8,label="Required / legacy memory")
        for row in (1,2):
            axes[row,axis].plot(t[valid],a["5__required_target_normalized"][valid,axis],color="#181818",ls=":",lw=1.8,label="Required / frozen plant")
            axes[row,axis].axhline(1.,color="#aaaaaa",lw=.8)
            mask = a["5__required_output_impossible"][:,axis]&active
            axes[row,axis].fill_between(t,0,1,where=mask,transform=axes[row,axis].get_xaxis_transform(),color="#f2d9d9",alpha=.45,zorder=-2)
        axes[0,axis].set_title(label+": same previous S110 / PID state")
        axes[0,axis].set_ylabel("Total request (G)")
        axes[1,axis].set_ylabel("Same-prior held target / raw finsAoa")
        axes[2,axis].set_ylabel("Own-history held target / raw finsAoa")
        axes[2,axis].set_xlabel("Time (s)")
        for row in range(3):
            axes[row,axis].set_xlim(.25,3.5);axes[row,axis].grid(alpha=.18)
            axes[row,axis].axvline(.7,color="#bcbcbc",lw=.6);axes[row,axis].axvline(2.3,color="#bcbcbc",lw=.6)
    handles,labels = axes[1,1].get_legend_handles_labels()
    labels = ["Required (legacy request / plant held)" if label=="Required / frozen plant" else label for label in labels]
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.945),ncol=6,frameon=False,fontsize=9)
    fig.suptitle(cid+": fixed reference states; PN hypotheses C / P / T / B\nRows separate single-step same-prior from each candidate's recursive memory; pink = impossible required output",fontsize=12,y=.993)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.subplots_adjust(left=.10,right=.985,bottom=.065,top=.83,wspace=.24,hspace=.10)
    fig.savefig(OUTPUT/(cid+"_pn_hypotheses.png"),dpi=160);plt.close(fig)


def main():
    before = protected_snapshot()
    sources = json.loads((ROOT/"outputs/jari_offaxis/textbook_hypotheses/sources.json").read_text())
    inputs = audit.inverse.source_inputs()
    cache = np.load(split.OUTPUT/"pn_pid_timeseries.npz")
    old = np.load(audit.OUTPUT/"request_audit.npz")
    arrays = {}; reports = {}
    for cid in audit.CASES:
        a,r = case_arrays(cid,inputs[cid],cache,old); reports[cid] = r
        arrays.update({cid+"__"+key:value for key,value in a.items()})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTPUT/"timeseries.npz",**arrays)
    for cid in audit.CASES:
        a = {key[len(cid)+2:]:v for key,v in arrays.items() if key.startswith(cid+"__")}
        plot_case(cid,a)
    after = protected_snapshot()
    if before != after:
        raise AssertionError("Protected input size/mtime changed")
    document = {"task":"War Thunder game textbook PN conditional discrimination","predeclared_formulas":FORMULAS,
        "cases":reports,"verification":{"zero_free_integrations":True,"zero_API_or_web_calls":True,"zero_fits_or_scans":True,
            "protected_size_mtime_unchanged":True,"independent_PID_worker_owned_outputs_excluded":True,
            "all_arrays_finite":True,"original_N_and_PID_unchanged":True,"single_aerodynamic_station_unchanged":True,
            "same_prior_comparison":"baseline prior PID memory/K/latch and prior S110, copied; counterfactual next memory discarded",
            "recursive_comparison":"each PN candidate owns its S110/PID history on identical reference states; histories are not called same memory"},
        "sources":{"NPTEL_url":sources["source_entries"][0]["url"],"section":"9.1/9.2/9.5; PPN zero-AoA assumption and finite-AoA directional caveat"},
        "limits":["All field mappings and 3D extensions are D-level game hypotheses, not native-game identification.",
            "TPN N is an effective ratio; reusing raw6 across conventions is a tested assumption, not a textbook identity.",
            "B changes only request direction before the original wind-normal projection; plant/one-surface architecture stays fixed.",
            "24Hz source states interpolated to48Hz; inverse inputs use noncausal3/5point derivatives with launch and source-boundary halos excluded.",
            "Required legacy total request assumes the old PID's recursively inverted memory; it is not native purePN.",
            "All77 60deg platform yaw ticks remain infeasible under the current target cap; feasible RMS is null there.",
            "No new free trajectory: speed, attitude and load improvements in closed loop remain unverified."]}
    audit.slew.write_json(OUTPUT/"results.json",document)
    print(json.dumps({"output":str(OUTPUT),"cases":list(reports),"verification":document["verification"]}))


if __name__=="__main__":
    main()
