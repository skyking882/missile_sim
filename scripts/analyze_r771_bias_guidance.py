#!/usr/bin/env python3
"""Read-only causal loft diagnostics on saved War Thunder/JARI trajectories.

No forward integration, fit, source-forced controller replay, or runtime change.
Source-state weights below are diagnostic samples, not inferred game internals.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
sys.dont_write_bytecode = True
import numpy as np
from aim120_model.aerodynamics import body_axes_from_quaternion, wind_normal_basis
from aim120_model.math3d import dot

OUT = ROOT / "outputs/jari_offaxis/bias_discussion/guidance"
BASE = ROOT / "outputs/jari_offaxis"
WINDOWS = (("startup", .3, .7), ("platform", .7, 2.3), ("unloading", 2.3, 3.5))


def weights(rate, horizontal):
    z = max(-60., min(60., 17.5 * (rate / math.radians(.65) - 1.)))
    weight = 1. / (1. + math.exp(z))
    z = max(-60., min(60., 17.5 * (1. - horizontal / math.radians(.5))))
    gate = min(1., (1. / (1. + math.exp(-z))) / (1. / (1. + math.exp(-17.5))))
    return weight, gate


def stats(x):
    x = np.asarray(x)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "rms": float(np.sqrt(np.mean(x*x)))}


def first(t, mask):
    index = np.flatnonzero(mask)
    return float(t[index[0]]) if len(index) else None


def analyse(cid, source_path, run_path):
    source = json.loads(source_path.read_text())
    result = json.loads(run_path.read_text())
    source_rows, local = source["samples"], result["ticks"]
    rt = np.array([x["t"] for x in source_rows])
    t = np.array([x["time_s"] for x in local])
    sample_by_time = {x["time_s"]: x for x in result["samples"]}
    observed_pitch = []
    source_diagnostic = []
    minw = 1.
    rv_target = np.array(source["scenario"]["target_velocity"])
    for s in source_rows:
        qxy = s["quaternion"]
        q = (qxy[3], *qxy[:3])
        axes = body_axes_from_quaternion(q)
        observed_pitch.append(math.degrees(math.atan2(axes.forward[1], math.hypot(axes.forward[0], axes.forward[2]))))
        rel = np.asarray(s["target"]) - np.asarray(s["position"])
        rv = rv_target - np.asarray(s["velocity"])
        los = np.cross(rel, rv) / (rel @ rel)
        cw, gate = weights(float(np.linalg.norm(los)), abs(float(los[1])))
        minw = min(minw, cw)
        source_diagnostic.append((cw, gate, minw))
    observed_pitch = np.asarray(observed_pitch)
    source_diagnostic = np.asarray(source_diagnostic)
    arrays = {k: np.asarray([x[k] for x in local]) for k in ("loft_g", "loft_weight", "loft_current_weight", "request_g", "feedback_g")}
    arrays["horizontal_gate"] = np.array([x.get("horizontal_loft_gate", 1.) for x in local])
    arrays["horizontal_rate_deg_s"] = np.array([math.degrees(abs(x["los_rate_rad_s"][1])) for x in local])
    arrays["total_rate_deg_s"] = np.array([math.degrees(x["loft_los_rate_magnitude_rad_s"]) for x in local])
    arrays["pitch_deg"] = np.array([math.degrees(sample_by_time[x["time_s"]]["pitch_rad"]) for x in local])
    arrays["pitch_error_deg"] = arrays["pitch_deg"] - np.interp(t, rt, observed_pitch)
    arrays["pn_pitch_G"] = np.array([dot(x["pn_mps2"], wind_normal_basis(
        body_axes_from_quaternion(sample_by_time[x["time_s"]]["orientation_quaternion"]),
        tuple(sample_by_time[x["time_s"]]["velocity_mps"])).up) / 9.80665 for x in local])
    gate = arrays["horizontal_gate"]
    firstlow = first(t, gate < .5)
    out = {"source_path": str(source_path), "saved_actual_free_path": str(run_path),
        "source_profile": source["profile_source"],
        "events_actual_free": {
            "first_gate_below_half_s": firstlow,
            "first_gate_above_half_after_prior_low_s": first(t, (gate >= .5) & (t > firstlow)) if firstlow is not None else None,
            "first_minimum_weight_below_half_s": first(t, arrays["loft_weight"] < .5),
            "first_current_weight_recovered_above_half_after_latched_low_s": first(t, (arrays["loft_current_weight"] > .5) & (arrays["loft_weight"] < .5)),
            "first_loft_below_minus_point1_G_s": first(t, arrays["loft_g"] < -.1),
            "first_shared_high_latch_s": first(t, np.array([x["nonlinear_control_latched"] for x in local])),
        }, "windows_actual_free": {}, "sampled_actual_free": []}
    for label, lo, hi in WINDOWS:
        mask = (t >= lo-1e-8) & (t <= hi+1e-8)
        out["windows_actual_free"][label] = {k: stats(v[mask]) for k,v in arrays.items() if v.ndim == 1}
    for nominal in (0., 14/48, .5, .7, 1., 1.5, 2.3, 3.5, 5.):
        i = int(np.argmin(abs(t-nominal)))
        out["sampled_actual_free"].append({"t": float(t[i]), **{k: v[i].tolist() for k,v in arrays.items()},
            "reference_pitch_deg_interpolated": float(np.interp(t[i], rt, observed_pitch))})
    out["source_state_diagnostic_only"] = {
        "description": "Candidate formulas on existing source snapshots with 24Hz sampled minimum; not actual game loft or48Hz runtime memory",
        "first_sampled_weight_below_half_s": first(rt, source_diagnostic[:,2] < .5),
        "first_gate_below_half_s": first(rt, source_diagnostic[:,1] < .5)}
    out["independent_formula_verification"] = {
        "current_weight_max_abs": max(abs(weights(x["loft_los_rate_magnitude_rad_s"], abs(x["los_rate_rad_s"][1]))[0]-x["loft_current_weight"]) for x in local),
        "horizontal_gate_max_abs": max(abs(weights(x["loft_los_rate_magnitude_rad_s"], abs(x["los_rate_rad_s"][1]))[1]-x.get("horizontal_loft_gate",1.)) for x in local),
        "minimum_weight_max_abs": float(np.max(abs(np.minimum.accumulate(arrays["loft_current_weight"])-arrays["loft_weight"]))) }
    assert max(out["independent_formula_verification"].values()) < 1e-12
    return out


def analyse_completed_grid():
    """Inspect completed independent captures, keeping all-state replays diagnostic."""
    metadata = json.loads((BASE/"angle_grid/results.json").read_text())
    assert metadata["status"] == "completed_declared_angle_transfer"
    assert metadata["new_successful_source_cases"] == metadata["new_actual_free_forwards"] == 6
    z = np.load(BASE/"angle_grid/trajectories.npz")
    cases = {}
    for cid, case in metadata["cases"].items():
        rt = z[cid+"__reference__t"]
        lt = z[cid+"__candidate__t"]
        los = z[cid+"__source_diagnostic__los_rate_xyz_rad_s"]
        total = np.degrees(np.linalg.norm(los,axis=1))
        horizontal = np.degrees(abs(los[:,1]))
        early = rt <= 3.5+1e-8
        events = {"source_total_LOS_max_before3p5_deg_s": float(total[early].max()),
                  "source_horizontal_LOS_max_before3p5_deg_s": float(horizontal[early].max())}
        for rate in (.5,.65):
            events[f"source_first_total_LOS_cross_{rate}_deg_s_before3p5"] = first(rt,early&(total>=rate))
        peaks = {}
        for label in ("reference","candidate"):
            t = z[cid+"__"+label+"__t"]
            value = z[cid+"__"+label+"__body_pitch_deg"]
            ids = [i for i in range(1,len(t)-1) if .3 <= t[i]<1.5 and value[i]>value[i-1] and value[i]>=value[i+1]]
            peaks[label] = [{"t":float(t[i]),"pitch_deg":float(value[i])} for i in ids[:2]]
        samples = []
        for nominal in (.5,1.,1.5,2.3,3.5):
            i = int(np.argmin(abs(rt-nominal)))
            actual = float(rt[i])
            row = {"source_time_s":actual, "source_pitch_deg":float(z[cid+"__reference__body_pitch_deg"][i]),
                   "local_pitch_deg_interpolated":float(np.interp(actual,lt,z[cid+"__candidate__body_pitch_deg"])),
                   "source_total_LOS_deg_s":float(total[i]),"source_horizontal_LOS_deg_s":float(horizontal[i])}
            for key in ("candidate_loft_G","candidate_loft_horizontal_gate","candidate_loft_current_weight","candidate_loft_display_minimum_weight"):
                row["source_state_formula_only__"+key] = z[cid+"__source_diagnostic__"+key][i].tolist()
            row["observed_source_controls_raw_units_unidentified"] = z[cid+"__source_diagnostic__source_controls_raw"][i].tolist()
            samples.append(row)
        local = analyse(cid,Path(case["source_path"]),Path(case["candidate_full_path"]))
        cases[cid] = {"signed_angle_deg":case["signed_offaxis_deg"],"source_path":case["source_path"],
            "actual_free_path":case["candidate_full_path"],"events_observed_source_geometry":events,
            "first_two_pitch_peaks_observed_and_actual_free":peaks,"source_samples_and_candidate_formula_diagnostic":samples,
            "actual_free_loft_events":local["events_actual_free"],
            "independent_formula_verification":local["independent_formula_verification"],
            "metrics_on_source_timestamps":case["free_vs_source_errors"]}
    return {"source":str(BASE/"angle_grid/results.json"),"status":metadata["status"],
        "new_source_cases":metadata["new_successful_source_cases"],"new_free_cases_by_worker":metadata["new_actual_free_forwards"],
        "this_analysis_new_forwards":0,"source_loft_fields_unobserved":True,
        "source_internal_gate_identified":False,"cases":cases,
        "mirror_check":metadata["positive_negative_30_mirror"],
        "predeclared_prediction_comparison":{
            "small_angle_branch": "Source5 firstpitch peak28.822deg and source10 firstpeak1.591deg independently bracket a response branch between5 and10deg; source data do not identify exact6.011/7.824deg thresholds or sigmoid",
            "mirror": "Source and free plus/minus30 primary physical traces agree exactly under expected parity",
            "large_angle": "45/60/75 all suppress local earlyloft; their unequal source peak amplitudes and phase residuals cannot be attributed to that initial loft switch",
            "new_counterevidence": "Source5 totalLOS remains below0.65deg/s through3.5s and source-state current loft stays positive despite observed early pitch unloading; local-state feedback crosses the latch differently, exposing control/guidance compensation"},
        "cross_review_with_bias_control_review": {
            "low_angle": "Other worker confirms5/10/15 have no outputclip,AW,highlatch,Itransport; source5 second raw control becomes negative0.625s while local pitchheld first negative1.145833s. Raw export units remain unidentified.",
            "high_angle": "Other worker confirms75 first raw horizontal control leaves1 at0.583333s while45/60 remain1; candidate saturation persists. This correlates with the lower75 source firstAoA peak and cannot be explained by earlyloft, which is suppressed in all three."},
        "next_falsifiable_directions_not_executed": [
            "For0/5/collision compare source-state conditional request and stored-I contributions during actual first reversal; distinguish loft command-shape from inner-control memory instead of retuning both together.",
            "For45/60/75 locate release of source raw control plateau versus candidate cap, then audit ordering of existing PN/hitGain/limits at identical states; do not add offaxis-dependent coefficients.",
            "Use intermediate6/7/8degree independent source cases to locate observed branch and matched-LOS/different-closing target motion to separate rate gating fromtime-to-go dependence; exact current sigmoid remains D-grade."
        ]}


def main():
    protected = [ROOT/p for p in ("src/aim120_model/single_surface_game.py", "scripts/check_r771_horizontal_sinc_cap.py", "scripts/compare_r771_jari_offaxis.py", "config/r771_simple_joint_frozen.json")]
    receipt = [(p.stat().st_size,p.stat().st_mtime_ns) for p in protected]
    initial = []
    for angle in (0,5,10,15,30,45,60,75,-30):
        rate = (2*333.333333/8000)*abs(math.sin(math.radians(angle)))
        w,g = weights(rate,rate)
        initial.append({"angle_deg": angle, "rate_deg_s": math.degrees(rate), "weight": w, "gate": g,
            "initial_formula_loft_G": .75*(24*w-3.5*(1-w))*g})
    cases = {}
    for angle in (0,30,60):
        cid = f"jari_259_{angle}"
        cases[cid] = analyse(cid, BASE/f"reference/{cid}_response.json", BASE/f"horizontal_sinc_cap/full_runs/{cid}.json")
    cases["collision"] = analyse("collision", BASE/"collision_pn_probe/collision_config_response.json", BASE/"collision_pn_probe/collision_best_free.json")
    assert receipt == [(p.stat().st_size,p.stat().st_mtime_ns) for p in protected]
    out = {"scope": "War Thunder game-only saved-result analysis; no forward/fit/network/default change", 
        "classification_predeclared": ["abs horizontal LOS gate changes sharply near6deg", "total LOS min memory changes near8deg and cannot recover", "same angle different target velocity may select different loft branch", "plus/minus30 mirror expected", "largeangle capture may differ with loft identically suppressed"],
        "initial_half_threshold_angles_deg": {"gate": math.degrees(math.asin(math.radians(.5)*8000/(2*333.333333))),
                                             "weight": math.degrees(math.asin(math.radians(.65)*8000/(2*333.333333)))},
        "initial_analytic_formula_only": initial, "cases": cases,
        "protected_critical_files_size_mtime_unchanged": True}
    if "--angle-grid" in sys.argv:
        out["completed_angle_grid_second_stage"] = analyse_completed_grid()
        assert receipt == [(p.stat().st_size,p.stat().st_mtime_ns) for p in protected]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v["events_actual_free"] for k,v in cases.items()},indent=2))


if __name__ == "__main__":
    main()
