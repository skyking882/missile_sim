#!/usr/bin/env python3
"""Bounded game-only horizontal-request slew calibration, no core changes.

Fit only Jari30 with S={50,75,110,165,250} G/s and the previously declared
five-channel loss. Predict60,0 and the three original near-axis fixtures.
Fixed fitted drag, half PN, loft, frozen inner and native PID stay unchanged.
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
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np

import compare_r771_jari_offaxis as comparison
import compare_unified_game_replays as harness
import fit_r771_jari_capture as previous
import fit_r771_jari_drag as drag
from aim120_model.aerodynamics import pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.guidance import interpolate_table
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/request_slew"
GRID = (50., 75., 110., 165., 250.)


class RequestSlewGame(drag.FittedHalfPNGame):
    """One state limits horizontal acceleration request through the same PID."""
    def __init__(self, config, *, request_slew_g_s, **kwargs):
        if not math.isfinite(request_slew_g_s) or request_slew_g_s <= 0.:
            raise ValueError("Request slew must be finite and positive")
        self.request_slew_g_s = float(request_slew_g_s)
        super().__init__(config, **kwargs)

    def initialize(self, state):
        self.slew_horizontal_g = 0.
        self.slew_last_tick = None
        self.slew_last_step_g = 0.
        self.slew_active_updates = 0
        self.slew_commit_history = []
        return super().initialize(state)

    def request(self, state, target, time_s, evaluation):
        original, details = super().request(state, target, time_s, evaluation)
        if abs(details["los_rate_rad_s"][1]) == 0.:
            return original, details
        cfg = self.config["guidance"]
        tick = int(round(time_s*self.controller_hz))
        is_tick = abs(time_s-tick/self.controller_hz) < 1e-9
        gain = interpolate_table((tick+1)/self.controller_hz, cfg["flight_time_gain_table"])
        if is_tick and tick != self.slew_last_tick:
            before = self.slew_horizontal_g
            if gain > 1e-12:
                limit = self.request_slew_g_s/self.controller_hz
                self.slew_horizontal_g += max(-limit, min(limit, original[1]-before))
                self.slew_active_updates += 1
            else:
                self.slew_horizontal_g = 0.
            self.slew_last_step_g = self.slew_horizontal_g-before
            self.slew_last_tick = tick
            self.slew_commit_history.append({"tick":tick, "time_s":float(time_s),
                "gain":gain, "target_g":original[1], "horizontal_state_g":self.slew_horizontal_g,
                "step_g":self.slew_last_step_g, "active_updates":self.slew_active_updates})
        combined = (original[0], self.slew_horizontal_g)
        maximum = float(cfg["maximum_lateral_acceleration_g"])*self.gravity/G0_MPS2
        length = math.hypot(*combined)
        requested = tuple(x*maximum/length for x in combined) if length > maximum else combined
        details = {**details, "slew_original_request_g":original,
            "slew_horizontal_state_g":self.slew_horizontal_g,
            "slew_last_step_g":self.slew_last_step_g,
            "slew_request_before_cap_g":combined,
            "slew_limit_g_s":self.request_slew_g_s,
            "slew_controller_tick":is_tick, "slew_last_committed_tick":self.slew_last_tick,
            "slew_active_updates":self.slew_active_updates, "slew_scheduled_flight_gain":gain}
        self.request_records[time_s] = {"request_g":requested, **details}
        return requested, details


def make_model(config, coefficients, selected):
    return RequestSlewGame(config, request_slew_g_s=selected, drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def write_json(path, value, compact=False):
    def scalar(x):
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(comparison.jsonable(value), allow_nan=False,
        default=scalar, indent=None if compact else 2,
        separators=(",", ":") if compact else None)+"\n")


def record_model(result, model):
    result["model"].update(drag_coefficients=dataclasses.asdict(model.drag_coefficients),
        request_family="active_48Hz_horizontal_request_slew",
        request_slew_g_s=model.request_slew_g_s, native_game_law_identified=False)


def early_events(prediction):
    keep = prediction["t"] <= 1.5+1e-6
    ids = np.flatnonzero(keep)
    out = {}
    for field in ("body_yaw_deg", "body_yaw_rate_deg_s", "alpha_deg"):
        i = ids[np.argmax(prediction[field][keep])]
        out[field] = {"peak":float(prediction[field][i]), "time_s":float(prediction["t"][i])}
    return out


def request_checks(config, initial, target_at, selected, coefficients, result, trained_model):
    """Inspect saved free states only; no extra forward integrations."""
    validation = make_model(config, coefficients, selected)
    validation.initialize(dataclasses.replace(initial))
    by_time = {s["time_s"]:s for s in result["samples"]}
    request_delta = 0.; up_delta = 0.; repeats = 0; non_tick_checks = 0
    def memory(m):
        return (m.slew_horizontal_g, m.slew_last_tick, m.slew_last_step_g,
                m.slew_active_updates, len(m.slew_commit_history))
    for tick in result["ticks"]:
        t = tick["time_s"]; s = by_time[t]; q = tuple(s["orientation_quaternion"])
        pitch, yaw = pitch_yaw_from_quaternion(q)
        state = SimState(tuple(s["position_m"]), tuple(s["velocity_mps"]), pitch, yaw,
            s["pitch_rate_rad_s"], s["yaw_rate_rad_s"], s["mass_kg"], orientation_quaternion=q)
        evaluation = validation.evaluate(state, t, tuple(s["held_fin_rad"]))
        first, details = validation.request(state, target_at(t), t, evaluation)
        before = memory(validation)
        second, again = validation.request(state, target_at(t), t, evaluation)
        if first != second or details != again or before != memory(validation):
            raise AssertionError("Same-tick request was not idempotent")
        repeats += 1
        request_delta = max(request_delta, max(abs(x-y) for x,y in zip(first, tick["request_g"])))
        if "slew_horizontal_state_g" in details:
            if tick["gain"] <= 1e-12 and details["slew_horizontal_state_g"] != 0.:
                raise AssertionError("Inactive slew state was not exactly zero")
            up_delta = max(up_delta, abs(first[0]-details["slew_original_request_g"][0]))
            if math.hypot(*first) > config["guidance"]["maximum_lateral_acceleration_g"]*validation.gravity/G0_MPS2+1e-10:
                raise AssertionError("Original final request cap exceeded")
        before = memory(validation)
        other = t+.5/validation.controller_hz
        validation.request(state, target_at(other), other, evaluation)
        if before != memory(validation): raise AssertionError("Non-tick request committed slew memory")
        non_tick_checks += 1
    before = 0.; count = 0; max_step = 0.; recurrence_error = 0.; seen = set()
    for commit in trained_model.slew_commit_history:
        if commit["tick"] in seen or abs(commit["time_s"]-commit["tick"]/48.) > 1e-9:
            raise AssertionError("Slew commit was duplicated or off the native clock")
        seen.add(commit["tick"])
        active = commit["gain"] > 1e-12
        limit = selected/48.
        expected = before+max(-limit, min(limit, commit["target_g"]-before)) if active else 0.
        count += int(active)
        recurrence_error = max(recurrence_error, abs(expected-commit["horizontal_state_g"]))
        if commit["active_updates"] != count: raise AssertionError("Slew update counter differs")
        if active: max_step = max(max_step, abs(commit["step_g"]))
        elif commit["horizontal_state_g"] != 0.: raise AssertionError("Inactive state anticipated activation")
        before = commit["horizontal_state_g"]
    if request_delta > 1e-10 or recurrence_error > 1e-12 or max_step > selected/48.+1e-12 or up_delta > 1e-12:
        raise AssertionError({"request_delta":request_delta, "recurrence_error":recurrence_error,
            "largest_active_step":max_step, "up_request_change":up_delta})
    return {"same_tick_idempotence_checks":repeats, "non_tick_no_commit_checks":non_tick_checks,
        "saved_free_request_max_difference_g":request_delta, "recurrence_max_difference_g":recurrence_error,
        "maximum_active_step_g":max_step, "allowed_active_step_g":selected/48.,
        "inactive_state_exact_zero":True, "up_request_max_change_g":up_delta,
        "commit_hz":48, "used_saved_free_states_not_reference":True}


def plot_case(cid, reference, observed, baseline, rejected, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = ("body_yaw_deg", "body_yaw_rate_deg_s", "alpha_deg", "speed_mps", "normal_specific_load_g")
    labels = ("Body yaw (deg)", "Body yaw rate (deg/s), Jari derivative", "Unsigned incidence (deg)", "Speed (m/s)", "Normal specific load (G)")
    series = (("fixed new-drag half", baseline, "#686b70"),
              ("rejected angle capture", rejected, "#bb869f"),
              ("selected request slew", selected, "#1b8f76"))
    for ax, field, label in zip(axes.flat, fields, labels):
        ax.plot(observed["t"], observed[field], "o", ms=2, mfc="none", color="#24566a", label="Jari interim reference")
        for name, prediction, color in series:
            ax.plot(prediction["t"], prediction[field], color=color, lw=1.2, label=name)
        ax.set(xlabel="Original time (s)", ylabel=label,
            xlim=(0, 3.5 if field in fields[:3] else reference["t"][-1])); ax.grid(alpha=.18)
    ax=axes[1,2];offset=reference["position_xyz_m"][0,1]
    ax.plot(reference["t"], reference["position_xyz_m"][:,1]-offset,"o",ms=2,mfc="none",color="#24566a")
    for _, prediction,color in series:
        ax.plot(prediction["t"],prediction["position_xyz_m"][:,1]-offset,color=color,lw=1.2)
    ax.set(xlabel="Original time (s)",ylabel="Y displacement (m)",xlim=(0,reference["t"][-1]));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(cid+": horizontal slew calibrated on30deg only; fixed inner/PID/drag/loft",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUTPUT/f"{cid}_overlay.png",dpi=150);plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_folder=OUTPUT/"full_runs";full_folder.mkdir(exist_ok=True)
    protected = [p for folder in (comparison.DEFAULT_OUTPUT, ROOT/"outputs/jari_offaxis/reference",
                 drag.OUTPUT, previous.OUTPUT, previous.gap.OUT) for p in folder.rglob("*") if p.is_file()]
    protected += [comparison.FROZEN, comparison.SNAPSHOT, Path(comparison.__file__), Path(drag.__file__),
        Path(previous.__file__), Path(previous.gap.__file__), Path(harness.__file__),
        ROOT/"src/aim120_model/single_surface_game.py", ROOT/"src/aim120_model/drag_models.py"]
    hashes = {str(p):comparison.sha256(p) for p in protected}
    fit_drag=json.loads((drag.OUTPUT/"fit.json").read_text())
    coefficients=drag.DragCoefficients(**fit_drag["coefficients"])
    drag_arrays=np.load(drag.OUTPUT/"trajectories.npz")
    rejected_arrays=np.load(previous.OUTPUT/"trajectories.npz")
    drag_document=json.loads((drag.OUTPUT/"results.json").read_text())
    profiles=json.loads(comparison.SNAPSHOT.read_text())
    manifest=json.loads(comparison.DEFAULT_MANIFEST.read_text())
    inputs={};arrays={}
    for entry in manifest["references"]:
        cid=entry["id"];run=comparison.split_runs(entry["export_path"])[0]
        ref, diagnostic=comparison.parse_reference(run)
        profile, source=comparison.profile_for(run, entry, profiles)
        config=copy.deepcopy(profile["_model_config"])
        cached, baseline=previous.cached_prediction(cid,run,ref,drag_arrays,profiles,coefficients)
        observed=previous.observed_fields(run,ref)
        rejected=drag.selected_arrays(rejected_arrays,cid+"__selected__")
        inputs[cid]={"run":run,"reference":ref,"observed":observed,"diagnostic":diagnostic,
            "config":config,"source":source,"baseline":baseline,"cached_full":cached,"rejected":rejected}
        for label,prediction in (("reference",observed),("baseline_new_drag_half",baseline),("rejected_angle_capture",rejected)):
            previous.save_arrays(arrays,cid+"__"+label,prediction)
    declaration={"family":"active48Hz horizontal acceleration request slew through unchanged PID",
        "grid_g_s":list(GRID),"fit_case":"jari_259_30","loss":"sum of five equally weighted normalized mean-square errors",
        "loss_channels":[{"field":f,"window_s":[lo,hi],"normalization":s} for f,lo,hi,s in previous.LOSS_CHANNELS],
        "window_membership_tolerance_s":previous.EDGE_TOLERANCE_S,"original_timestamps_unchanged":True,
        "selection":"single minimum total loss; deterministic grid order breaks ties; no posthoc gate",
        "rate_method":"3-point full-body quaternion derivative; source omega unused; display cadence uncertainty remains",
        "transfer_60_is_blind":False,"new_drag_coefficients_fixed":dataclasses.asdict(coefficients),
        "inner_coefficients_fixed":dataclasses.asdict(SurfaceCoefficients()),"half_PN_blend_fixed":.5,
        "loft_coefficients_fixed":[.75,1.3],"new_free_integration_budget":10,"baseline_integrations":0,
        "clock_shift_s":0.,"missile_reference_after_creation":False}
    write_json(OUTPUT/"fit.json",declaration)
    train=inputs["jari_259_30"];grid_records=[];grid_runs={};free_count=0
    for index,slew in enumerate(GRID):
        config=copy.deepcopy(train["config"])
        state,target_at=comparison.free_inputs(train["run"],train["reference"],config)
        model=make_model(config,coefficients,slew)
        result=model.run(dataclasses.replace(state),target_at,end_time_s=config["performance"]["lifetime_s"])
        free_count+=1
        prediction=previous.prediction_fields(result,model,train["run"]["scenario"])
        scored=previous.loss(train["observed"],prediction)
        grid_records.append({"grid_index":index,"request_slew_g_s":slew,"loss":scored,
            "events":drag.event_summary(train["reference"],prediction,result,train["run"]["summary"]),
            "early_events":early_events(prediction),"summary":result["summary"]})
        grid_runs[slew]=(result,prediction,model,config,state,target_at)
        previous.save_arrays(arrays,f"fit30__S{slew:g}",prediction)
        print(json.dumps({"completed_slew_g_s":slew,"loss":scored["total"]}),flush=True)
    selected_record=min(grid_records,key=lambda r:(r["loss"]["total"],r["grid_index"]))
    selected=selected_record["request_slew_g_s"]
    declaration.update(candidates=grid_records,selected=selected_record,
        selected_on_grid_boundary=bool(selected in (GRID[0],GRID[-1])),
        baseline_loss=previous.loss(train["observed"],train["baseline"]),parameter_fit_performed=True)
    write_json(OUTPUT/"fit.json",declaration)
    selected_runs={"jari_259_30":grid_runs[selected]}
    for cid in ("jari_259_60","jari_259_0"):
        x=inputs[cid];config=copy.deepcopy(x["config"])
        state,target_at=comparison.free_inputs(x["run"],x["reference"],config)
        model=make_model(config,coefficients,selected)
        result=model.run(dataclasses.replace(state),target_at,end_time_s=config["performance"]["lifetime_s"])
        free_count+=1
        prediction=previous.prediction_fields(result,model,x["run"]["scenario"])
        selected_runs[cid]=(result,prediction,model,config,state,target_at)
    records={};native={};force_checks={};request_verification={};zero=None;fuse={}
    for cid,(result,prediction,model,config,state,target_at) in selected_runs.items():
        x=inputs[cid]
        if cid=="jari_259_0":
            zero={key:comparison.jsonable(result[key])==x["cached_full"][key] for key in ("samples","ticks","summary")}
            if not all(zero.values()):raise AssertionError("0deg differs from cached new-drag half")
        record_model(result,model)
        write_json(full_folder/f"{cid}_selected.json",result,True)
        previous.save_arrays(arrays,cid+"__selected",prediction)
        records[cid]={"source":x["source"],"original_scenario":x["run"]["scenario"],
            "raw_PID":config["control"]["pid"],"original_loft_fields":{k:config["guidance"][k] for k in
                ("lofting_elevation_deg","target_elevation_deg","angle_to_acceleration_multiplier","loft_omega_max_deg_s")},
            "loss_channels":previous.loss(x["observed"],prediction),
            "baseline_loss":previous.loss(x["observed"],x["baseline"]),
            "rejected_angle_capture_loss":previous.loss(x["observed"],x["rejected"]),
            "matched_new_drag_half_vs_slew":drag.pair_metrics(x["reference"],x["baseline"],prediction,
                ("speed_mps","alpha_deg","normal_specific_load_g"),comparison.WINDOWS),
            "matched_rejected_angle_capture_vs_slew":drag.pair_metrics(x["reference"],x["rejected"],prediction,
                ("speed_mps","alpha_deg","normal_specific_load_g"),comparison.WINDOWS),
            "selected_vs_reference":comparison.errors(x["reference"],prediction),
            "events":drag.event_summary(x["reference"],prediction,result,x["run"]["summary"]),
            "baseline_events":drag_document["cases"][cid]["variants"]["half_PN_gate"]["events"],
            "early_events":{"reference":early_events(x["observed"]),"baseline":early_events(x["baseline"]),
                "rejected_angle_capture":early_events(x["rejected"]),"selected":early_events(prediction)},
            "slew_commits":model.slew_commit_history,"summary":result["summary"]}
    cases={c["id"]:c for c in harness.load_cases(comparison.SNAPSHOT)}
    native_runs={}
    for name in ("H110","H27.5","H55native"):
        case=cases[name];config=copy.deepcopy(case["profile"]["_model_config"])
        state=dataclasses.replace(case["_creation_state"])
        qx,qy,qz,qw=case["creation"]["u16_1"];state.orientation_quaternion=(qw,qx,qy,qz)
        target_at=case["_target_model"].state_at;model=make_model(config,coefficients,selected)
        result=model.run(dataclasses.replace(state),target_at,end_time_s=case["scenario"]["max_simulation_time_s"])
        free_count+=1
        prediction=harness.sample_series(result["samples"],case)
        new_drag=drag.selected_arrays(drag_arrays,f"native_{name}__new_drag_half_PN_gate__")
        old_accepted=drag.selected_arrays(drag_arrays,f"native_{name}__old_rate_angle__")
        native[name]={"same_original_creation_and_target":True,"fixture_projection":False,
            "slew_relative_new_drag_half":previous.native_pair(case,new_drag,prediction),
            "overall_relative_old_accepted_rate_angle":previous.native_pair(case,old_accepted,prediction),
            "raw_PID":config["control"]["pid"],"summary":result["summary"]}
        native_runs["native_"+name]=(result,prediction,model,config,state,target_at)
        record_model(result,model)
        write_json(full_folder/f"native_{name}_selected.json",result,True)
        for label,p in (("selected",prediction),("baseline_new_drag_half",new_drag),
                        ("baseline_old_accepted",old_accepted),("observed",case["observed"])):
            previous.save_arrays(arrays,f"native_{name}__{label}",p)
    np.savez_compressed(OUTPUT/"trajectories.npz",**arrays)
    # One focused final acceptance pass, without further free integrations.
    for cid,(result,prediction,model,config,state,target_at) in {**selected_runs,**native_runs}.items():
        request_verification[cid]=request_checks(config,state,target_at,selected,coefficients,result,model)
        force_checks[cid]=drag.force_bookkeeping(model,result)
    for cid,(result,prediction,model,config,state,target_at) in selected_runs.items():
        separation=np.linalg.norm(np.asarray(target_at(prediction["t"][-1]).position)-prediction["position_xyz_m"][-1])
        fuse[cid]={"end_separation_m":float(separation),"boundary_error_m":float(abs(separation-10.)),
            "passed":bool(result["summary"]["termination_event"]=="proximity_fuse" and abs(separation-10.)<.001)}
    unchanged=hashes=={str(p):comparison.sha256(p) for p in protected}
    frozen=json.loads(comparison.FROZEN.read_text());s=SurfaceCoefficients()
    frozen_equal=frozen["coefficients"]==[s.ca,s.speed_slope,s.control_gain,s.restoring_gain] and frozen["fixed_reference_damping_s_minus1"]==s.rate_damping
    finite=all(np.isfinite(value).all() for value in arrays.values())
    rows=sum(len(x["reference"]["t"]) for x in inputs.values())
    clock_unchanged=all(np.array_equal(arrays[cid+"__reference__t"],x["reference"]["t"]) for cid,x in inputs.items())
    raw_unchanged=True
    for cid,(result,prediction,model,config,state,target_at) in selected_runs.items():
        raw_unchanged=raw_unchanged and config==inputs[cid]["config"]
    for cid,(result,prediction,model,config,state,target_at) in native_runs.items():
        raw_unchanged=raw_unchanged and config==cases[cid.removeprefix("native_")]["profile"]["_model_config"]
    if not unchanged or not frozen_equal or not finite or free_count!=10 or rows!=631 or not clock_unchanged or not raw_unchanged or not all(v["passed"] for v in fuse.values()):
        raise AssertionError("Request-slew preservation/clock/finite/count/fuse acceptance failed")
    for cid in ("jari_259_30","jari_259_60"):
        x=inputs[cid];plot_case(cid,x["reference"],x["observed"],x["baseline"],x["rejected"],selected_runs[cid][1])
    document={"status":"bounded_request_slew_fit_completed","fit":declaration,"cases":records,"native_near_axis":native,
        "reference_label":"Jari2.59.0.28 interim videogame reference; real-game validation pending",
        "local_profile_version":"2.59.0.27; whole-rocket equality bridge preserved","global_28_gameparams_verified":False,
        "incremental_near_gate_all_passed":all(v["slew_relative_new_drag_half"]["full_gate_passed"] for v in native.values()),
        "overall_old_accepted_near_gate_all_passed":all(v["overall_relative_old_accepted_rate_angle"]["full_gate_passed"] for v in native.values()),
        "zero_planar_exact_relative_new_drag_half":zero,"request_state_verification":request_verification,"force_bookkeeping":force_checks,
        "verification":{"protected_inputs_and_previous_outputs_unchanged":unchanged,"frozen_angular_law_equal":frozen_equal,
            "all_arrays_finite":finite,"original_reference_rows_retained":rows,"original_reference_clock_unchanged":clock_unchanged,
            "raw_config_PID_and_loft_fields_equal":raw_unchanged,"new_free_integrations":free_count,"existing_baselines_rerun":0,
            "first_fuse_boundary_checks":fuse,"commit_hz":48,"plant_substeps":4,"new_user_fields":False,
            "clock_shift_s":0.,"source_missile_after_creation":False,"single_equivalent_surface_and_same_station_couple":True},
        "production_or_GUI_changed":False,
        "limitations":["Only30 calibrated;60 transfer is not blind","Body rate and derived G retain display-cadence uncertainty",
            "Fixed new drag already worsens some native angular/height metrics; incremental success cannot imply overall acceptance",
            "No other family, grid extension, gain/inner/loft/drag change, or extra free run"]}
    write_json(OUTPUT/"results.json",document)
    print(json.dumps({"status":document["status"],"selected_slew_g_s":selected,
        "selected_on_grid_boundary":declaration["selected_on_grid_boundary"],"training_loss":selected_record["loss"]["total"],
        "baseline_loss":declaration["baseline_loss"]["total"],"new_free_integrations":free_count,
        "incremental_near_gate":document["incremental_near_gate_all_passed"],
        "overall_old_near_gate":document["overall_old_accepted_near_gate_all_passed"]}),flush=True)


if __name__ == "__main__": main()
