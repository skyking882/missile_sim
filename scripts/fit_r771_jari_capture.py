#!/usr/bin/env python3
"""One bounded War Thunder/Jari capture-family calibration; script-local only.

Predeclared 30-degree grid: Ac={10,14,18}G, tau={.04,.08,.12}s.
Loss=sum of five normalized MSEs: body yaw/5deg, body yaw rate/30deg_s,
unsigned AoA/5deg at .3..1.5s; speed/50mps and derived normal G/3 at
.7..3.5s. All selected reference rows must have prediction support.
The 60-degree reference has already been examined: transfer is not blind.
No existing baseline is rerun; new drag, inner, PID and loft remain fixed.
"""
from __future__ import annotations

import copy
import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np

import analyze_r771_jari_capture_gap as gap
import compare_r771_jari_offaxis as comparison
import compare_unified_game_replays as harness
import fit_r771_jari_drag as drag
from aim120_model.aerodynamics import body_axes_for_state, pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.math3d import sub
from aim120_model.guidance import interpolate_table
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/capture_fit"
GRID = tuple((a, tau) for a in (10., 14., 18.) for tau in (.04, .08, .12))
LOSS_CHANNELS = (("body_yaw_deg", .3, 1.5, 5.),
                 ("body_yaw_rate_deg_s", .3, 1.5, 30.),
                 ("alpha_deg", .3, 1.5, 5.),
                 ("speed_mps", .7, 3.5, 50.),
                 ("normal_specific_load_g", .7, 3.5, 3.))
EDGE_TOLERANCE_S = 1e-6  # window membership only; timestamps are untouched


class AngleCaptureGame(drag.FittedHalfPNGame):
    """Continuous angle capture through the existing acceleration PID."""
    def __init__(self, config, *, capture_amplitude_g, capture_tau_s, **kwargs):
        if not all(math.isfinite(x) and x > 0. for x in (capture_amplitude_g, capture_tau_s)):
            raise ValueError("Capture coefficients must be finite and positive")
        self.capture_amplitude_g = float(capture_amplitude_g)
        self.capture_tau_s = float(capture_tau_s)
        super().__init__(config, **kwargs)

    def initialize(self, state):
        self.capture_weight = 1.
        self.capture_filtered_g = 0.
        self.capture_last_tick = None
        self.capture_update_count = 0
        self.capture_commit_history = []
        return super().initialize(state)

    def request(self, state, target, time_s, evaluation):
        original, details = super().request(state, target, time_s, evaluation)
        # Exact planar pass-through, including all diagnostic dictionaries.
        if abs(details["los_rate_rad_s"][1]) == 0.:
            return original, details
        cfg = self.config["guidance"]
        pn_gain = float(cfg["pn_gain"])
        if pn_gain <= 0.:
            raise ValueError("Existing PN gain must be positive")
        relative = sub(target.position, state.position)
        forward = body_axes_for_state(state).forward
        beta = math.atan2(relative[2], relative[0])-math.atan2(forward[2], forward[0])
        beta = math.atan2(math.sin(beta), math.cos(beta))
        raw = self.capture_amplitude_g*math.tanh(pn_gain*beta)
        current_weight = beta*beta/(beta*beta+(1./pn_gain)**2)
        tick = int(round(time_s*self.controller_hz))
        is_tick = abs(time_s-tick/self.controller_hz) < 1e-9
        gain = interpolate_table((tick+1)/self.controller_hz, cfg["flight_time_gain_table"])
        if is_tick and tick != self.capture_last_tick:
            self.capture_weight = min(self.capture_weight, current_weight)
            if gain > 1e-12:
                fraction = -math.expm1(-1./(self.controller_hz*self.capture_tau_s))
                self.capture_filtered_g += fraction*(raw-self.capture_filtered_g)
                self.capture_update_count += 1
            else:
                # No anticipation or hidden inactive settling.
                self.capture_filtered_g = 0.
            self.capture_last_tick = tick
            self.capture_commit_history.append({"tick": tick, "time_s": float(time_s),
                "gain": gain, "weight": self.capture_weight, "raw_g": raw,
                "filtered_g": self.capture_filtered_g, "update_count": self.capture_update_count})
        requested = (original[0], (1.-self.capture_weight)*original[1]+self.capture_weight*self.capture_filtered_g)
        maximum = float(cfg["maximum_lateral_acceleration_g"])*self.gravity/G0_MPS2
        length = math.hypot(*requested)
        request = tuple(x*maximum/length for x in requested) if length > maximum else requested
        details = {**details, "capture_beta_rad": beta, "capture_current_weight": current_weight,
            "capture_latched_weight": self.capture_weight, "capture_raw_target_g": raw,
            "capture_filtered_g": self.capture_filtered_g, "capture_original_request_g": original,
            "capture_combined_request_before_cap_g": requested, "capture_controller_tick": is_tick,
            "capture_last_committed_tick": self.capture_last_tick,
            "capture_filter_update_count": self.capture_update_count,
            "capture_scheduled_flight_gain": gain, "capture_amplitude_g": self.capture_amplitude_g,
            "capture_tau_s": self.capture_tau_s}
        self.request_records[time_s] = {"request_g": request, **details}
        return request, details


def prediction_fields(result, model, scenario):
    prediction = comparison.prediction_arrays(result, model)
    q = gap.continuous_quaternions(np.asarray([s["orientation_quaternion"] for s in result["samples"]]))
    geometric = gap.trajectory_fields(prediction["t"], prediction["position_xyz_m"], prediction["velocity_xyz_mps"],
        scenario["target_position"], scenario["target_velocity"], q)
    prediction.update({key: geometric[key] for key in ("body_yaw_deg", "body_pitch_deg", "body_bank_deg", "velocity_heading_deg", "quaternion_wxyz")})
    prediction["body_yaw_rate_deg_s"] = np.degrees([s["yaw_rate_rad_s"] for s in result["samples"]])
    return prediction


def observed_fields(run, reference):
    q = gap.continuous_quaternions(reference["quaternion_xyzw"][:, [3, 0, 1, 2]])
    geometric = gap.trajectory_fields(reference["t"], reference["position_xyz_m"], reference["velocity_xyz_mps"],
        run["scenario"]["target_position"], run["scenario"]["target_velocity"], q)
    output = {**reference, "body_yaw_deg": geometric["body_yaw_deg"],
        "body_yaw_rate_deg_s": np.degrees(-gap.body_rates(q, reference["t"], 3)[:, 1])}
    return output


def loss(reference, predicted):
    channels = {}
    for field, lo, hi, normalization in LOSS_CHANNELS:
        keep = (reference["t"] >= lo-EDGE_TOLERANCE_S)&(reference["t"] <= hi+EDGE_TOLERANCE_S)
        times = reference["t"][keep]
        if not len(times) or np.any(times < predicted["t"][0]) or np.any(times > predicted["t"][-1]):
            raise ValueError("Incomplete support for a predeclared loss window")
        delta = np.interp(times, predicted["t"], predicted[field])-reference[field][keep]
        channels[field] = {"window_s": [lo, hi], "reference_rows": int(keep.sum()),
            "normalization": normalization, "rmse": float(np.sqrt(np.mean(delta**2))),
            "normalized_mse": float(np.mean((delta/normalization)**2))}
    values = [v["normalized_mse"] for v in channels.values()]
    return {"total": sum(values), "early_shape": sum(values[:3]), "turn_speed_load": sum(values[3:]), "channels": channels}


def cached_prediction(cid, run, reference, arrays, profiles, coefficients):
    cached = json.loads((drag.OUTPUT / "full_runs" / f"{cid}__half_PN_gate.json").read_text())
    profile, _ = comparison.profile_for(run, {}, profiles)
    config = copy.deepcopy(profile["_model_config"])
    state, _ = comparison.free_inputs(run, reference, config)
    model = drag.FittedHalfPNGame(config, drag_coefficients=coefficients, force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
    model.initialize(dataclasses.replace(state))
    prediction = prediction_fields(cached, model, run["scenario"])
    for key in ("t", "position_xyz_m", "velocity_xyz_mps", "speed_mps", "alpha_deg", "normal_specific_load_g"):
        if not np.array_equal(prediction[key], arrays[f"{cid}__new_drag_half_PN_gate__{key}"]):
            raise AssertionError("Cached new-drag baseline mismatch")
    return cached, prediction


def make_model(config, coefficients, selected):
    return AngleCaptureGame(config, drag_coefficients=coefficients,
        capture_amplitude_g=selected[0], capture_tau_s=selected[1], coefficients=SurfaceCoefficients(),
        force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def record_model(result, model):
    result["model"].update(drag_coefficients=dataclasses.asdict(model.drag_coefficients),
        capture_family="bounded_body_to_LOS_angle_with_active_only_first_order_filter",
        capture_coefficients={"amplitude_g": model.capture_amplitude_g, "tau_s": model.capture_tau_s},
        capture_angle_scale_rad=1./model.config["guidance"]["pn_gain"],
        capture_one_way_latch=True, capture_commit_hz=48, native_game_law_identified=False)


def save_arrays(arrays, prefix, prediction):
    for key, value in prediction.items(): arrays[prefix+"__"+key] = value


def recover_full(model, initial, path):
    """Recover a delivered free run after an output error, never integrate."""
    result=json.loads(path.read_text())
    model.initialize(dataclasses.replace(initial))
    model.capture_commit_history=[{"tick":int(round(r["time_s"]*48)),"time_s":r["time_s"],
        "gain":r["gain"],"weight":r["capture_latched_weight"],"raw_g":r["capture_raw_target_g"],
        "filtered_g":r["capture_filtered_g"],"update_count":r["capture_filter_update_count"]}
        for r in result["ticks"] if "capture_filtered_g" in r]
    return result


def keep_recovered_arrays(predicted, cached):
    # JSON restores scalar types as Python floats. Re-evaluated force-only
    # diagnostics can differ by roundoff; all saved kinematic arrays are exact.
    for key,value in predicted.items():
        if key in ("acc_drag","acc_lift","acc_thrust"):
            if not np.allclose(value,cached[key],rtol=0.,atol=1e-11):
                raise AssertionError("Recovered force diagnostics differ: "+key)
        elif not np.array_equal(value,cached[key]):
            raise AssertionError("Recovered kinematic arrays differ: "+key)
    return cached


def native_pair(case, baseline, new):
    limits = {"alpha_deg": .01, "omega_rad_s": .001, "gamma_deg": .01, "y_m": .1}
    fields = ("speed_kmh", *limits)
    windows = drag.pair_metrics(case["observed"], baseline, new, fields, harness.WINDOWS)
    for window in windows.values():
        for key, metric in window["metrics"].items():
            if key == "speed_kmh":
                allowed = .1*3.6
                metric["units"] = "km/h"
            else:
                allowed = max(.01*metric["old"]["rmse"], limits[key])
                metric["units"] = harness.UNITS[key]
            metric.update(allowed_worsening=allowed, passed=metric["rmse_change"] <= allowed)
    t = case["observed"]["t"]
    common = (t >= max(baseline["t"][0], new["t"][0]))&(t <= min(baseline["t"][-1], new["t"][-1]))
    delta = np.interp(t[common], new["t"], new["load_g"])-np.interp(t[common], baseline["t"], baseline["load_g"])
    load = {"independent_observed_G_available": False, "prediction_delta_rms_g": float(np.sqrt(np.mean(delta**2))),
        "prediction_delta_max_abs_g": float(np.max(np.abs(delta))), "allowed_prediction_rms_change_g": .01,
        "passed": float(np.sqrt(np.mean(delta**2))) <= .01}
    return {"windows": windows, "load_prediction_change": load,
        "full_gate_passed": all(v["passed"] for v in windows["full"]["metrics"].values()) and load["passed"]}


def request_checks(config, initial, target_at, selected, coefficients, result, trained_model):
    """Request-only checks at saved free states, no new forward integration."""
    validation = make_model(config, coefficients, selected)
    validation.initialize(dataclasses.replace(initial))
    by_time = {s["time_s"]: s for s in result["samples"]}
    largest_request_difference = 0.; repeats = 0; non_tick_checks = 0; up_delta = 0.
    for tick in result["ticks"]:
        time_s = tick["time_s"]; s = by_time[time_s]
        q = tuple(s["orientation_quaternion"]); pitch, yaw = pitch_yaw_from_quaternion(q)
        state = SimState(tuple(s["position_m"]), tuple(s["velocity_mps"]), pitch, yaw,
            s["pitch_rate_rad_s"], s["yaw_rate_rad_s"], s["mass_kg"], orientation_quaternion=q)
        f = validation.evaluate(state, time_s, tuple(s["held_fin_rad"]))
        first, details = validation.request(state, target_at(time_s), time_s, f)
        before = (validation.capture_weight, validation.capture_filtered_g, validation.capture_update_count, len(validation.capture_commit_history))
        second, repeated = validation.request(state, target_at(time_s), time_s, f)
        after = (validation.capture_weight, validation.capture_filtered_g, validation.capture_update_count, len(validation.capture_commit_history))
        if first != second or details != repeated or before != after:
            raise AssertionError("Same-tick request is not idempotent")
        repeats += 1
        largest_request_difference = max(largest_request_difference, max(abs(x-y) for x,y in zip(first,tick["request_g"])))
        if "capture_filtered_g" in details:
            if tick["gain"] <= 1e-12 and details["capture_filtered_g"] != 0.:
                raise AssertionError("Inactive capture filter settled before gain")
            if math.hypot(*first) > config["guidance"]["maximum_lateral_acceleration_g"]*validation.gravity/G0_MPS2+1e-10:
                raise AssertionError("Combined request exceeds original cap")
            up_delta = max(up_delta, abs(first[0]-details["capture_original_request_g"][0]))
        # A non-controller call may inspect geometry, never advance either state.
        other_time = time_s+.5/validation.controller_hz
        before = (validation.capture_weight, validation.capture_filtered_g, validation.capture_update_count, len(validation.capture_commit_history))
        validation.request(state, target_at(other_time), other_time, f)
        after = (validation.capture_weight, validation.capture_filtered_g, validation.capture_update_count, len(validation.capture_commit_history))
        if before != after: raise AssertionError("Non-tick request committed capture memory")
        non_tick_checks += 1
    history = trained_model.capture_commit_history
    previous_weight, previous_filter, previous_count = 1., 0., 0
    fraction = -math.expm1(-1./(48.*selected[1]))
    maximum_recurrence_error = 0.
    for commit in history:
        if abs(commit["time_s"]-commit["tick"]/48.) > 1e-9 or commit["weight"] > previous_weight:
            raise AssertionError("Capture latch clock/monotonicity failed")
        expected = previous_filter+fraction*(commit["raw_g"]-previous_filter) if commit["gain"]>1e-12 else 0.
        maximum_recurrence_error = max(maximum_recurrence_error, abs(expected-commit["filtered_g"]))
        expected_count = previous_count+int(commit["gain"]>1e-12)
        if expected_count != commit["update_count"]: raise AssertionError("Capture filter update count failed")
        previous_weight, previous_filter, previous_count = commit["weight"], commit["filtered_g"], expected_count
    if largest_request_difference > 1e-10 or maximum_recurrence_error > 1e-12:
        raise AssertionError("Request-only replay differs from saved free ticks")
    return {"same_tick_idempotence_checks": repeats, "non_tick_no_commit_checks": non_tick_checks,
        "saved_free_request_max_difference_g": largest_request_difference,
        "filter_recurrence_max_difference_g": maximum_recurrence_error,
        "inactive_filter_exact_zero": True, "latch_nonincreasing": True,
        "capture_commit_hz": 48, "up_request_max_change_g": up_delta,
        "request_only_replay_used_saved_free_states_not_reference": True}


def plot_case(cid, reference, observed, baseline, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = ("body_yaw_deg", "body_yaw_rate_deg_s", "alpha_deg", "speed_mps", "normal_specific_load_g")
    labels = ("Body yaw (deg)", "Body yaw rate (deg/s), Jari derivative", "Unsigned incidence (deg)", "Speed (m/s)", "Normal specific load (G)")
    for ax, field, label in zip(axes.flat, fields, labels):
        ax.plot(observed["t"], observed[field], "o", ms=2, mfc="none", color="#24566a", label="Jari interim reference")
        for name, prediction, color in (("fixed new-drag half", baseline, "#686b70"), ("selected angle capture", selected, "#a05080")):
            ax.plot(prediction["t"], prediction[field], color=color, lw=1.2, label=name)
        ax.set(xlabel="Original time (s)", ylabel=label, xlim=(0, 3.5 if field in fields[:3] else reference["t"][-1])); ax.grid(alpha=.18)
    ax=axes[1,2];offset=reference["position_xyz_m"][0,1]
    ax.plot(reference["t"], reference["position_xyz_m"][:,1]-offset,"o",ms=2,mfc="none",color="#24566a")
    for prediction,color in ((baseline,"#686b70"),(selected,"#a05080")):
        ax.plot(prediction["t"],prediction["position_xyz_m"][:,1]-offset,color=color,lw=1.2)
    ax.set(xlabel="Original time (s)",ylabel="Y displacement (m)",xlim=(0,reference["t"][-1]));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": capture fit uses30deg only; frozen inner/PID/drag/loft, no clock shift",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/f"{cid}_overlay.png",dpi=150);plt.close(fig)


def main(recover_saved=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = [p for folder in (comparison.DEFAULT_OUTPUT, ROOT/"outputs/jari_offaxis/reference", drag.OUTPUT, gap.OUT)
                 for p in folder.rglob("*") if p.is_file()]
    protected += [comparison.FROZEN, comparison.SNAPSHOT, Path(comparison.__file__), Path(drag.__file__), Path(gap.__file__),
                  Path(harness.__file__), ROOT/"src/aim120_model/single_surface_game.py", ROOT/"src/aim120_model/drag_models.py"]
    protected_hashes = {str(p):comparison.sha256(p) for p in protected}
    drag_fit = json.loads((drag.OUTPUT/"fit.json").read_text())
    coefficients = drag.DragCoefficients(**drag_fit["coefficients"])
    drag_arrays = np.load(drag.OUTPUT/"trajectories.npz")
    drag_document = json.loads((drag.OUTPUT/"results.json").read_text())
    profiles = json.loads(comparison.SNAPSHOT.read_text())
    manifest = json.loads(comparison.DEFAULT_MANIFEST.read_text())
    inputs = {}
    for entry in manifest["references"]:
        cid=entry["id"];run=comparison.split_runs(entry["export_path"])[0]
        ref,diagnostic=comparison.parse_reference(run);profile,source=comparison.profile_for(run,entry,profiles)
        config=copy.deepcopy(profile["_model_config"]);observed=observed_fields(run,ref)
        cached,baseline=cached_prediction(cid,run,ref,drag_arrays,profiles,coefficients)
        inputs[cid]={"run":run,"reference":ref,"diagnostic":diagnostic,"source":source,"config":config,
                     "observed":observed,"cached_full":cached,"baseline":baseline}
    recovery_fit=json.loads((OUTPUT/"fit.json").read_text()) if recover_saved else None
    recovery_arrays=np.load(OUTPUT/"trajectories.npz") if recover_saved else None
    predeclared={"family":"bounded body-to-real-LOS angle capture; one-way48Hz latch; active-only exponential filter",
        "grid":[{"amplitude_g":a,"tau_s":tau} for a,tau in GRID],
        "fit_case":"jari_259_30","loss":"sum of five equally weighted normalized mean-square errors",
        "loss_channels":[{"field":f,"window_s":[lo,hi],"normalization":s} for f,lo,hi,s in LOSS_CHANNELS],
        "window_membership_tolerance_s":EDGE_TOLERANCE_S,"original_timestamps_unchanged":True,
        "source_rate_method":"3-point quaternion finite differences in full body basis; native omega unused; display cadence uncertainty remains",
        "selection":"lowest total loss; deterministic grid order breaks ties; require total and early-shape loss below cached new-drag half baseline",
        "transfer_60_is_blind":False,"new_drag_coefficients_fixed":dataclasses.asdict(coefficients),
        "inner_coefficients_fixed":dataclasses.asdict(SurfaceCoefficients()),"loft_coefficients_fixed":[.75,1.3],
        "new_free_integration_budget":14,"baseline_integrations":0,"clock_shift_s":0.,"missile_reference_after_creation":False}
    # This file is written before the first new free integration.
    if not recover_saved:
        (OUTPUT/"fit.json").write_text(json.dumps(predeclared,indent=2,allow_nan=False)+"\n")
    elif recovery_fit["grid"] != predeclared["grid"] or recovery_fit["new_drag_coefficients_fixed"] != predeclared["new_drag_coefficients_fixed"]:
        raise AssertionError("Recovery coefficients/grid differ from the saved experiment")
    arrays={key:recovery_arrays[key] for key in recovery_arrays.files} if recover_saved else {}
    grid_records,grid_results,models=[],{},{}
    for cid,x in inputs.items():
        save_arrays(arrays,cid+"__reference",x["observed"])
        save_arrays(arrays,cid+"__baseline_new_drag_half",x["baseline"])
    train=inputs["jari_259_30"];baseline_loss=loss(train["observed"],train["baseline"])
    for index,selected in enumerate(GRID):
        config=copy.deepcopy(train["config"]);state,target_at=comparison.free_inputs(train["run"],train["reference"],config)
        model=make_model(config,coefficients,selected)
        grid_id=f"Ac{selected[0]:g}_tau{selected[1]:g}"
        if recover_saved:
            cached_record=recovery_fit["candidates"][index]
            if cached_record["grid_id"] != grid_id:
                raise AssertionError("Recovery grid order differs")
            saved_prediction=drag.selected_arrays(recovery_arrays,"fit30__"+grid_id+"__")
            if loss(train["observed"],saved_prediction)!=cached_record["loss"]:
                raise AssertionError("Cached candidate loss differs")
            grid_records.append(cached_record)
            if grid_id==recovery_fit["selected"]["grid_id"]:
                result=recover_full(model,state,OUTPUT/"full_runs/jari_259_30_selected.json")
                predicted=prediction_fields(result,model,train["run"]["scenario"])
                predicted=keep_recovered_arrays(predicted,saved_prediction)
                grid_results[grid_id]=(result,predicted,config,state,target_at);models[grid_id]=model
            continue
        result=model.run(dataclasses.replace(state),target_at,end_time_s=config["performance"]["lifetime_s"])
        predicted=prediction_fields(result,model,train["run"]["scenario"])
        scored=loss(train["observed"],predicted)
        grid_records.append({"grid_id":grid_id,"grid_index":index,"amplitude_g":selected[0],"tau_s":selected[1],
            "loss":scored,"events":drag.event_summary(train["reference"],predicted,result,train["run"]["summary"]),
            "summary":result["summary"]})
        grid_results[grid_id]=(result,predicted,config,state,target_at);models[grid_id]=model
        save_arrays(arrays,"fit30__"+grid_id,predicted)
        print(json.dumps({"completed_grid_id":grid_id,"loss":scored["total"]}),flush=True)
    selected_record=min(grid_records,key=lambda r:(r["loss"]["total"],r["grid_index"]))
    improved=selected_record["loss"]["total"]<baseline_loss["total"] and selected_record["loss"]["early_shape"]<baseline_loss["early_shape"]
    predeclared.update(baseline_loss=baseline_loss,candidates=grid_records,selected=selected_record,
                       selected_training_shape_improved=improved,parameter_fit_performed=True)
    (OUTPUT/"fit.json").write_text(json.dumps(comparison.jsonable(predeclared),indent=2,allow_nan=False)+"\n")
    records,native,force_checks,request_verification={}, {}, {}, {}
    free_count=9
    selected=(selected_record["amplitude_g"],selected_record["tau_s"])
    selected_runs={"jari_259_30":grid_results[selected_record["grid_id"]]}
    selected_models={"jari_259_30":models[selected_record["grid_id"]]}
    if improved:
        for cid in ("jari_259_60","jari_259_0"):
            x=inputs[cid];config=copy.deepcopy(x["config"]);state,target_at=comparison.free_inputs(x["run"],x["reference"],config)
            model=make_model(config,coefficients,selected)
            if recover_saved:
                result=recover_full(model,state,OUTPUT/"full_runs"/f"{cid}_selected.json")
            else:
                result=model.run(dataclasses.replace(state),target_at,end_time_s=config["performance"]["lifetime_s"])
            free_count+=1
            predicted=prediction_fields(result,model,x["run"]["scenario"])
            if recover_saved:
                predicted=keep_recovered_arrays(predicted,drag.selected_arrays(recovery_arrays,cid+"__selected__"))
            selected_runs[cid]=(result,predicted,config,state,target_at);selected_models[cid]=model
    zero_invariance=None
    full_folder=OUTPUT/"full_runs";full_folder.mkdir(exist_ok=True)
    for cid,(result,predicted,config,state,target_at) in selected_runs.items():
        x=inputs[cid];model=selected_models[cid]
        if cid=="jari_259_0":
            zero_invariance={key:comparison.jsonable(result[key])==x["cached_full"][key] for key in ("samples","ticks","summary")}
            if not all(zero_invariance.values()):raise AssertionError("0deg pass-through differs from new-drag half")
        request_verification[cid]=request_checks(config,state,target_at,selected,coefficients,result,model)
        force_checks[cid]=drag.force_bookkeeping(model,result)
        record_model(result,model)
        (full_folder/f"{cid}_selected.json").write_text(json.dumps(comparison.jsonable(result),separators=(",",":"),allow_nan=False)+"\n")
        records[cid]={"source":x["source"],"original_scenario":x["run"]["scenario"],"raw_PID":config["control"]["pid"],
            "original_loft_fields":{k:config["guidance"][k] for k in ("lofting_elevation_deg","target_elevation_deg","angle_to_acceleration_multiplier","loft_omega_max_deg_s")},
            "matched_new_drag_half_vs_capture":drag.pair_metrics(x["reference"],x["baseline"],predicted,("speed_mps","alpha_deg","normal_specific_load_g"),comparison.WINDOWS),
            "selected_vs_reference":comparison.errors(x["reference"],predicted),"loss_channels":loss(x["observed"],predicted),
            "events":drag.event_summary(x["reference"],predicted,result,x["run"]["summary"]),
            "baseline_events":drag_document["cases"][cid]["variants"]["half_PN_gate"]["events"],
            "capture_commits":model.capture_commit_history,"summary":result["summary"]}
        save_arrays(arrays,cid+"__selected",predicted);save_arrays(arrays,cid+"__baseline_new_drag_half",x["baseline"])
        save_arrays(arrays,cid+"__reference",x["observed"])
        if cid!="jari_259_0":plot_case(cid,x["reference"],x["observed"],x["baseline"],predicted)
    if improved:
        cases={c["id"]:c for c in harness.load_cases(comparison.SNAPSHOT)}
        for name in ("H110","H27.5","H55native"):
            case=cases[name];config=copy.deepcopy(case["profile"]["_model_config"]);state=dataclasses.replace(case["_creation_state"])
            qx,qy,qz,qw=case["creation"]["u16_1"];state.orientation_quaternion=(qw,qx,qy,qz)
            target_at=case["_target_model"].state_at;model=make_model(config,coefficients,selected)
            if recover_saved:
                result=recover_full(model,state,OUTPUT/"full_runs"/f"native_{name}_selected.json")
            else:
                result=model.run(dataclasses.replace(state),target_at,end_time_s=case["scenario"]["max_simulation_time_s"])
            free_count+=1
            predicted=harness.sample_series(result["samples"],case)
            new_drag=drag.selected_arrays(drag_arrays,f"native_{name}__new_drag_half_PN_gate__")
            old_accepted=drag.selected_arrays(drag_arrays,f"native_{name}__old_rate_angle__")
            native[name]={"same_original_creation_and_target":True,"fixture_projection":False,
                "capture_relative_new_drag_half":native_pair(case,new_drag,predicted),
                "overall_relative_old_accepted_rate_angle":native_pair(case,old_accepted,predicted),
                "raw_PID":config["control"]["pid"],"summary":result["summary"]}
            force_checks["native_"+name]=drag.force_bookkeeping(model,result)
            request_verification["native_"+name]=request_checks(config,state,target_at,selected,coefficients,result,model)
            record_model(result,model)
            (full_folder/f"native_{name}_selected.json").write_text(json.dumps(comparison.jsonable(result),separators=(",",":"),allow_nan=False)+"\n")
            for label,prediction in (("selected",predicted),("baseline_new_drag_half",new_drag),("baseline_old_accepted",old_accepted),("observed",case["observed"])):
                save_arrays(arrays,f"native_{name}__{label}",prediction)
    # Final focused acceptance; no additional free integrations.
    unchanged=protected_hashes=={str(p):comparison.sha256(p) for p in protected}
    frozen=json.loads(comparison.FROZEN.read_text());s=SurfaceCoefficients()
    frozen_equal=frozen["coefficients"]==[s.ca,s.speed_slope,s.control_gain,s.restoring_gain] and frozen["fixed_reference_damping_s_minus1"]==s.rate_damping
    fuse={}
    for cid,(result,prediction,config,state,target_at) in selected_runs.items():
        actual=np.linalg.norm(np.asarray(target_at(prediction["t"][-1]).position)-prediction["position_xyz_m"][-1])
        fuse[cid]={"end_separation_m":float(actual),"boundary_error_m":float(abs(actual-10.)),
                   "passed":bool(result["summary"]["termination_event"]=="proximity_fuse" and abs(actual-10.)<.001)}
    all_finite=all(np.isfinite(value).all() for value in arrays.values())
    rows=sum(len(x["reference"]["t"]) for x in inputs.values())
    if not unchanged or not frozen_equal or not all_finite or free_count>14 or rows!=631 or not all(v["passed"] for v in fuse.values()):
        raise AssertionError("Capture experiment preservation/finite/count/fuse acceptance failed")
    document={"status":"bounded_capture_fit_completed" if improved else "bounded_grid_no_training_structure_improvement",
        "reference_label":"Jari .28 interim videogame reference; real-game validation pending","global_28_gameparams_verified":False,
        "local_profile_version":"2.59.0.27; whole-rocket equality bridge preserved",
        "fit":predeclared,"cases":records,"native_near_axis":native,
        "capture_near_incremental_gate_all_passed":all(c["capture_relative_new_drag_half"]["full_gate_passed"] for c in native.values()) if native else None,
        "overall_old_accepted_near_gate_all_passed":all(c["overall_relative_old_accepted_rate_angle"]["full_gate_passed"] for c in native.values()) if native else None,
        "zero_planar_exact_relative_new_drag_half":zero_invariance,"force_bookkeeping":force_checks,"request_state_verification":request_verification,
        "verification":{"protected_inputs_and_previous_outputs_unchanged":unchanged,"frozen_angular_law_equal":frozen_equal,
            "all_arrays_finite":all_finite,"original_reference_rows_retained":rows,"new_free_integrations":free_count,
            "new_free_integrations_this_invocation":0 if recover_saved else free_count,"output_error_recovery_without_forward_integration":recover_saved,
            "existing_baselines_rerun":0,"first_fuse_boundary_checks":fuse,"capture_commit_hz":48,
            "raw_PID_changed":False,"new_user_fields":False,"clock_shift_s":0.,"source_missile_after_creation":False,
            "single_equivalent_surface_and_same_station_couple":True},
        "production_or_GUI_changed":False,"limitations":["Outer coefficients calibrated only on30deg;60 transfer is not blind",
            "New drag already worsens several near-axis angular/height metrics versus old accepted model; incremental capture success is not overall acceptance",
            "Jari display-derived body rates and normal G have cadence/endpoint uncertainty; native48Hz PID states are unavailable",
            "No second capture family, expanded grid, or loft/inner refit is performed"]}
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    (OUTPUT/"results.json").write_text(json.dumps(comparison.jsonable(document),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":document["status"],"selected":selected,"new_free_integrations":free_count,
        "training_loss":selected_record["loss"]["total"],"baseline_loss":baseline_loss["total"],
        "incremental_near_gate":document["capture_near_incremental_gate_all_passed"],
        "overall_old_near_gate":document["overall_old_accepted_near_gate_all_passed"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-saved",action="store_true",help="Recover saved complete grid/selected runs after an output failure; no forward integration")
    main(parser.parse_args().recover_saved)
