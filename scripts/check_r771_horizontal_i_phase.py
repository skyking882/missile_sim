#!/usr/bin/env python3
"""One yaw integral-output phase hypothesis for the War Thunder game.

Reuse the tested horizontal-only sinc model and diagnostic helpers. Pitch is
unchanged; yaw uses transported previous I in output, then updates I using its
actual-natural original-cap antiwindup decision. No runtime observation input.
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
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import check_r771_horizontal_sinc_cap as base
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.control import scheduled_pid_values
from aim120_model.math3d import add, cross, dot, norm, scale, sub

OUT = ROOT / "outputs/jari_offaxis/horizontal_i_phase"
dec, geometry = base.dec, base.geometry
drag, fields, harness = dec.audit.drag, dec.audit.fix.finite.fields, dec.audit.fix.finite.harness


class HorizontalIPhaseControl(base.HorizontalSincCapControl):
    """Keep the existing lifecycle and caps; change only yaw I-output ordering."""
    def update_axis(self, axis, time_s, error_g, memory, pid, old_held_limit_rad, gain,
                    target_per_old_held, nonlinear_force_active):
        if axis in (0, 1) and self.last_time[axis] == time_s:
            return self.cached[axis]
        result = super().update_axis(axis, time_s, error_g, memory, pid, old_held_limit_rad,
            gain, target_per_old_held, nonlinear_force_active)
        if gain <= 1e-12 or axis == 0:
            return result
        d = self.diagnostics[axis]
        transported_i = d["transported_integral_old_rad"]
        natural = d["P_old_rad"] + transported_i + d["D_old_rad"]
        saturated = dec.audit.clip(natural, d["effective_old_held_limit_rad"])
        counterfactual = dec.audit.clip(natural, d["original_AW_limit_old_rad"])
        blocked = abs(counterfactual) >= .99 * d["original_AW_limit_old_rad"] and error_g * counterfactual > 0
        next_i = transported_i if blocked else d["I_provisional_old_rad"]
        result = gain * saturated, (next_i, error_g, result[1][2])
        self.cached[axis] = result
        d.update(I_output_old_rad=transported_i, yaw_previous_integral_in_output=True,
            phase_removed_provisional_increment_old_rad=d["I_provisional_old_rad"] - transported_i,
            natural_old_rad=natural, original_counterfactual_saturated_old_rad=counterfactual,
            antiwindup_blocked=blocked, next_integral_old_rad=next_i,
            integral_target_rad=target_per_old_held * next_i, held_old_rad=result[0],
            effective_target_rad=target_per_old_held * result[0],
            normalized_target_at_tick=target_per_old_held * result[0] / old_held_limit_rad)
        return result


class HorizontalIPhaseGame(base.HorizontalSincCapGame):
    def initialize(self, state):
        initial = super().initialize(state)
        self.game_control = HorizontalIPhaseControl()
        return initial

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(runtime_controller="check_r771_horizontal_i_phase.HorizontalIPhaseControl",
            controller_mechanism="yaw output uses transported previous I; actual-natural original-cap AW; pitch unchanged",
            yaw_natural_PID="P*error + transported_previous_I + D*filtered_derivative",
            yaw_next_I="original Ki*error/48 with raw Ilim and original high-only K transport; actual-natural original-cap .99 AW",
            evidence_grade="D horizontal discrete PID interface hypothesis", source_data_in_runtime=False,
            new_fitted_coefficients=0)
        return result


def make_model(cfg, coefficients):
    return HorizontalIPhaseGame(cfg, request_slew_g_s=110., drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def protected_snapshot():
    result = {}
    for folder in ("src", "config", "configs", "docs", "data", "scripts", "outputs"):
        for p in (ROOT / folder).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or ".git" in p.parts or OUT in p.parents:
                continue
            relative = str(p.relative_to(ROOT))
            if relative == "scripts/analyze_r771_speed_residual.py" or relative.startswith("outputs/jari_offaxis/speed_residual/"):
                continue
            st = p.stat()
            result[str(p)] = [st.st_size, st.st_mtime_ns]
    return result


def verify(model, result, cfg, coefficients, initial, target_at):
    """Independent tick equations on candidate states; no additional forward run."""
    outer = base.make_model(cfg, coefficients)
    outer.initialize(dataclasses.replace(initial))
    original_pitch = R771GameControl()
    memory, previous_k, high = [(0., 0., 0.), (0., 0., 0.)], [None, None], False
    samples = {s["time_s"]: s for s in result["samples"]}
    limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    residuals = {k: 0. for k in ("request_same_state_G", "full_relative_PN_formula_mps2",
        "full_relative_outer_S110_formula_G", "force_same_state_N", "drag_same_state_N",
        "angular_same_state_rad_s2", "K_same_state_equal", "beta_current_state_equal_rad",
        "sinc_independent_equal", "output_cap_formula_rad", "original_AW_cap_formula_rad",
        "natural_PID_formula_rad", "counterfactual_actual_natural_rad", "held_gain_last_rad",
        "full_memory_formula", "I_transport_ratio", "pitch_original_held_rad", "pitch_original_memory",
        "raw_output_cap_excess_rad", "raw_I_limit_excess", "yaw_sinc_bound_excess")}
    effect = {"yaw_AW_differs_from_previous_provisional_natural_ticks": 0,
        "yaw_same_state_previous_phase_output_changes": 0, "first_yaw_output_phase_change_s": None,
        "max_same_state_yaw_output_phase_change_rad": 0., "max_yaw_removed_provisional_increment_rad": 0.}
    arrays = {k: [] for k in ("t", "gain", "error", "held", "P", "I_transported", "I_provisional", "I_output",
        "D", "natural", "effective_limit", "original_AW_limit", "counterfactual_output", "AW", "I_next",
        "memory", "K", "high", "beta", "sinc_factor", "removed_increment")}
    for tick in result["ticks"]:
        ts, gain = tick["time_s"], tick["gain"]
        state = dec.audit.fix.finite.sample_state(samples[ts])
        held_before = tuple(samples[ts]["evaluated_old_held_rad"])
        prior_eval = outer.evaluate(state, ts, held_before)
        prior_slew = outer.slew_horizontal_g
        request, details = outer.request(state, target_at(ts), ts, prior_eval)
        residuals["request_same_state_G"] = max(residuals["request_same_state_G"], max(abs(a-b) for a,b in zip(request, tick["request_g"])))
        if abs(details["los_rate_rad_s"][1]) != 0.:
            pn0 = details["original_pn_mps2"]
            alt = scale(cross(details["los_rate_rad_s"], scale(sub(target_at(ts).velocity, state.velocity), -1.)), cfg["guidance"]["pn_gain"] * details["hit_gain"])
            full_pn = add(pn0, scale(prior_eval.wind_basis.right, dot(alt, prior_eval.wind_basis.right)-dot(pn0, prior_eval.wind_basis.right)))
            residuals["full_relative_PN_formula_mps2"] = max(residuals["full_relative_PN_formula_mps2"], norm(sub(full_pn, details["pn_mps2"])))
            maximum = cfg["guidance"]["maximum_lateral_acceleration_g"] * model.gravity
            boundaries = dec.outer_boundaries(full_pn, details["loft_g"], prior_eval.wind_basis, model.gravity, maximum)
            expected, _, _ = dec.same_previous_slew(boundaries["unslewed_request_G"], prior_slew, gain, abs(details["los_rate_rad_s"][1]), maximum/dec.G0_MPS2)
            residuals["full_relative_outer_S110_formula_G"] = max(residuals["full_relative_outer_S110_formula_G"], max(abs(a-b) for a,b in zip(expected, request)))
        evaluated = model.evaluate(state, ts, held_before)
        for key, value in (("force_same_state_N", norm(sub(evaluated.wind_lift_force, prior_eval.wind_lift_force))),
            ("drag_same_state_N", norm(sub(evaluated.wind_drag_force, prior_eval.wind_drag_force))),
            ("angular_same_state_rad_s2", max(abs(a-b) for a,b in zip(evaluated.angular_acceleration, prior_eval.angular_acceleration)))):
            residuals[key] = max(residuals[key], value)
        if gain <= 1e-12:
            if np.any(tick["pid_memory"]) or np.any(tick["held_fin_rad"]):
                raise AssertionError("inactive memory and held must be zero")
            continue
        high = high or bool(outer.current_force_nonlinear)
        k = outer.current_control_k
        beta = math.hypot(*prior_eval.local_incidence)
        factor = 1. / float(np.sinc(beta/math.pi))
        pid = scheduled_pid_values(cfg["control"]["pid"], (round(ts*48)+1)/48)
        values = {key: [] for key in arrays if key not in ("t", "gain")}
        for axis, d in enumerate(tick["control_runtime_diagnostics"]):
            error = tick["request_g"][axis] - tick["feedback_g"][axis]
            ratio = 1. if not high or previous_k[axis] is None else previous_k[axis]/k
            transported = dec.audit.clip(memory[axis][0]*ratio, pid["integral_limit"]) if high else memory[axis][0]
            provisional = dec.audit.clip(transported + pid["i"]*error/48, pid["integral_limit"])
            derivative = memory[axis][2] + (1/48)/(1e-9+1/48) * ((error-memory[axis][1])*48-memory[axis][2])
            p, derivative_term = pid["p"]*error, pid["d"]*derivative
            i_output = transported if axis == 1 else provisional
            natural = p + i_output + derivative_term
            aw_limit = min(limits[axis], limits[axis]/k) if high else limits[axis]
            output_limit = min(limits[axis], limits[axis]/k*factor) if axis == 1 else aw_limit
            counterfactual = dec.audit.clip(natural, aw_limit)
            blocked = abs(counterfactual) >= .99*aw_limit and error*counterfactual > 0
            next_i = transported if blocked else provisional
            expected_held = gain*dec.audit.clip(natural, output_limit)
            expected_memory = (next_i, error, derivative)
            for key, value in (("K_same_state_equal", abs(k-d["K_target_per_old_held"])),
                ("beta_current_state_equal_rad", abs(beta-d["beta_rad"])), ("sinc_independent_equal", abs(factor-d["inverse_sinc_factor"])),
                ("output_cap_formula_rad", abs(output_limit-d["effective_old_held_limit_rad"])),
                ("original_AW_cap_formula_rad", abs(aw_limit-d["original_AW_limit_old_rad"])),
                ("natural_PID_formula_rad", abs(natural-d["natural_old_rad"])),
                ("counterfactual_actual_natural_rad", abs(counterfactual-d["original_counterfactual_saturated_old_rad"])),
                ("held_gain_last_rad", abs(expected_held-tick["held_fin_rad"][axis])),
                ("full_memory_formula", max(abs(a-b) for a,b in zip(expected_memory, tick["pid_memory"][axis]))),
                ("I_transport_ratio", abs(ratio-d["integral_transport_ratio"])),
                ("raw_output_cap_excess_rad", abs(expected_held)-limits[axis]),
                ("raw_I_limit_excess", abs(next_i)-pid["integral_limit"])):
                residuals[key] = max(residuals[key], value)
            if blocked != d["antiwindup_blocked"] or high != d["nonlinear_latched"]:
                raise AssertionError("actual-natural AW or original high lifecycle mismatch")
            if axis == 0:
                original = original_pitch.update_axis(axis, ts, error, memory[axis], pid, limits[axis], gain, k, outer.current_force_nonlinear)
                residuals["pitch_original_held_rad"] = max(residuals["pitch_original_held_rad"], abs(original[0]-expected_held))
                residuals["pitch_original_memory"] = max(residuals["pitch_original_memory"], max(abs(a-b) for a,b in zip(original[1], expected_memory)))
                removed = 0.
            else:
                residuals["yaw_sinc_bound_excess"] = max(residuals["yaw_sinc_bound_excess"], abs(k*expected_held/limits[axis])-factor)
                old_natural = p + provisional + derivative_term
                old_counter = dec.audit.clip(old_natural, aw_limit)
                old_blocked = abs(old_counter) >= .99*aw_limit and error*old_counter > 0
                old_held = gain*dec.audit.clip(old_natural, output_limit)
                delta = abs(old_held-expected_held)
                removed = provisional-transported
                effect["yaw_AW_differs_from_previous_provisional_natural_ticks"] += int(old_blocked != blocked)
                effect["yaw_same_state_previous_phase_output_changes"] += int(delta > 1e-12)
                effect["max_same_state_yaw_output_phase_change_rad"] = max(effect["max_same_state_yaw_output_phase_change_rad"], delta)
                effect["max_yaw_removed_provisional_increment_rad"] = max(effect["max_yaw_removed_provisional_increment_rad"], abs(removed))
                if delta > 1e-12 and effect["first_yaw_output_phase_change_s"] is None:
                    effect["first_yaw_output_phase_change_s"] = ts
            memory[axis] = expected_memory
            if high:
                previous_k[axis] = k
            for key, value in (("error", error), ("held", expected_held), ("P", p), ("I_transported", transported),
                ("I_provisional", provisional), ("I_output", i_output), ("D", derivative_term), ("natural", natural),
                ("effective_limit", output_limit), ("original_AW_limit", aw_limit), ("counterfactual_output", counterfactual),
                ("AW", blocked), ("I_next", next_i), ("memory", expected_memory), ("K", k), ("high", high),
                ("beta", beta), ("sinc_factor", factor), ("removed_increment", removed)):
                values[key].append(value)
        arrays["t"].append(ts)
        arrays["gain"].append(gain)
        for key, value in values.items():
            arrays[key].append(value)
    fixed = {"raw_PID_config_equal": model.config == cfg, "five_coefficients_equal": model.coefficients == SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coefficients, "full_horizontal_PN": model.relative_blend == 1.,
        "raw_N6_S110_loft_unchanged": cfg["guidance"]["pn_gain"] == 6. and model.request_slew_g_s == 110. and [model.amplitude_coefficient, model.transition_coefficient] == [.75,1.3],
        "clock48_substeps4_surface0p4": model.controller_hz == 48 and model.plant_substeps == 4 and model.arm == .4,
        "tick_clock_exact": np.max(abs(np.asarray([t["time_s"] for t in result["ticks"]])*48-np.arange(len(result["ticks"])))) < 1e-8,
        "source_creation_only": result["summary"]["creation_state_only"] and result["model"]["source_data_in_runtime"] is False}
    if max(residuals.values()) > 1e-10 or not all(fixed.values()):
        raise AssertionError({"residuals": residuals, "fixed": fixed})
    return {"max_residuals": residuals, "fixed": fixed, "phase_effect_on_same_error_memory": effect,
        "force_couple_record_closure": dec.audit.inverse.saved_closure(model, result)}, {k: np.asarray(v) for k,v in arrays.items()}


def pair(observed, baseline, prediction, channels, windows, units):
    metrics = drag.pair_metrics(observed, baseline, prediction, channels, windows)
    for item in metrics.values():
        for key, metric in item["metrics"].items():
            metric["units"] = units[key]
    return metrics


def run_jari(inputs, coefficients):
    cached = np.load(geometry.OUT/"free_trajectories.npz")
    prior_cached = np.load(base.OUT/"trajectories.npz")
    arrays, reports, trajectories = {}, {}, {}
    new, recovered = 0, 0
    (OUT/"full_runs").mkdir(exist_ok=True)
    for cid in base.CASES:
        x = inputs[cid]
        model = make_model(x["config"], coefficients)
        path = OUT/"full_runs"/(cid+".json")
        if path.exists():
            result = json.loads(path.read_text()); model.initialize(dataclasses.replace(x["initial"])); recovered += 1
        else:
            result = model.run(dataclasses.replace(x["initial"]), x["target_at"], end_time_s=x["config"]["performance"]["lifetime_s"])
            dec.audit.slew.write_json(path, result, True); new += 1
        observed = drag.selected_arrays(cached, cid+"__reference__")
        baseline = drag.selected_arrays(cached, cid+"__candidate__")
        prior = drag.selected_arrays(prior_cached, cid+"__candidate__")
        prediction = fields.prediction_fields(result, model, x["run"]["scenario"])
        baseline_full = json.loads((geometry.OUT/"full_runs"/(cid+".json")).read_text())
        baseline_model = geometry.make_model(x["config"], coefficients); baseline_model.initialize(dataclasses.replace(x["initial"]))
        restored = fields.prediction_fields(baseline_full, baseline_model, x["run"]["scenario"])
        if not all(np.array_equal(restored[k], baseline[k]) for k in ("t", *base.UNITS)):
            raise AssertionError("geometry cached arrays do not restore")
        prior_full = json.loads((base.OUT/"full_runs"/(cid+".json")).read_text())
        focused, ticks = verify(model, result, x["config"], coefficients, x["initial"], x["target_at"])
        reports[cid] = {"separate_metrics": pair(observed, baseline, prediction, tuple(base.UNITS), base.WINDOWS, base.UNITS),
            "first_positive_peaks": base.peaks(observed, baseline, prediction),
            "cached_horizontal_sinc_comparison": pair(observed, prior, prediction, tuple(base.UNITS), base.WINDOWS, base.UNITS),
            "cached_horizontal_sinc_peaks": {k:base.first_peak(prior,k) for k in ("body_yaw_rate_deg_s", "alpha_deg", "normal_specific_load_g")},
            "summary": result["summary"], "focused_checks": focused, "full_result_path": str(path),
            "cached_geometry_receipt_arrays_exact": True, "physical_geometry_exact": base.physical_exact(result, baseline_full),
            "physical_previous_horizontal_sinc_exact": base.physical_exact(result, prior_full)}
        trajectories[cid] = observed, baseline, prior, prediction
        for label, p in (("reference",observed),("geometry",baseline),("horizontal_sinc",prior),("candidate",prediction)):
            fields.save_arrays(arrays,cid+"__"+label,p)
        arrays.update({cid+"__tick__"+k:v for k,v in ticks.items()})
        print(json.dumps({"completed_Jari":cid,"summary":result["summary"],"phase_effect":focused["phase_effect_on_same_error_memory"]}),flush=True)
    np.savez_compressed(OUT/"trajectories.npz",**arrays)
    return reports, trajectories, {"new_Jari_integrations":new,"recovered_Jari":recovered,"baseline_integrations":0,
        "all_arrays_finite":all(np.isfinite(v).all() for v in arrays.values())}


def run_native(coefficients):
    cases = {c["id"]:c for c in harness.load_cases(dec.audit.fix.finite.comparison.SNAPSHOT)}
    cached, prior_cached = np.load(geometry.OUT/"native_trajectories.npz"), np.load(base.OUT/"native_trajectories.npz")
    windows, channels = (*dec.WINDOWS,*harness.WINDOWS), ("speed_kmh","alpha_deg","pitch_deg","omega_rad_s","gamma_deg","y_m")
    reports, arrays, new, recovered = {}, {}, 0, 0
    for cid in base.NATIVE:
        x=cases[cid.removeprefix("native_")]; cfg=copy.deepcopy(x["profile"]["_model_config"])
        initial=dataclasses.replace(x["_creation_state"]);qx,qy,qz,qw=x["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
        target_at=x["_target_model"].state_at;model=make_model(cfg,coefficients);path=OUT/"full_runs"/(cid+".json")
        if path.exists():
            result=json.loads(path.read_text());model.initialize(dataclasses.replace(initial));recovered+=1
        else:
            result=model.run(dataclasses.replace(initial),target_at,end_time_s=x["scenario"]["max_simulation_time_s"])
            dec.audit.slew.write_json(path,result,True);new+=1
        observed=drag.selected_arrays(cached,cid+"__reference__");baseline=drag.selected_arrays(cached,cid+"__candidate__")
        prior=drag.selected_arrays(prior_cached,cid+"__candidate__");prediction=harness.sample_series(result["samples"],x)
        for cache_root,p in ((geometry.OUT,baseline),(base.OUT,prior)):
            full=json.loads((cache_root/(cid+"_run.json")).read_text());restored=harness.sample_series(full["samples"],x)
            if not all(np.array_equal(v,p[k]) for k,v in restored.items()):raise AssertionError("native cache receipt")
        metrics=pair(observed,baseline,prediction,channels,windows,harness.UNITS)
        rules={w+"__"+k:m["new"]["rmse"]<=1.1*m["old"]["rmse"]+1e-12 for w,item in metrics.items() for k,m in item["metrics"].items() if k in ("speed_kmh","alpha_deg","pitch_deg")}
        loads={}
        for name,lo,hi in windows:
            mask=(observed["t"]>=lo-1e-9)&(observed["t"]<=hi+1e-9)&(observed["t"]>=max(baseline["t"][0],prediction["t"][0]))&(observed["t"]<=min(baseline["t"][-1],prediction["t"][-1]))
            t=observed["t"][mask]
            loads[name]={"independent_observed_G_available":False,"prediction_change_G":drag.stats(np.interp(t,prediction["t"],prediction["load_g"])-np.interp(t,baseline["t"],baseline["load_g"]))}
        focused,ticks=verify(model,result,cfg,coefficients,initial,target_at)
        reports[cid]={"separate_metrics":metrics,"primary_72_rules":rules,"passes":all(rules.values()),
            "cached_horizontal_sinc_comparison":pair(observed,prior,prediction,channels,windows,harness.UNITS),
            "load_prediction_changes":loads,"focused_checks":focused,"summary":result["summary"],"full_result_path":str(path),
            "signed_alpha_pitch_observation_semantics_preserved":True,"creation_quaternion_full_roll_preserved":True}
        for label,p in (("reference",observed),("geometry",baseline),("horizontal_sinc",prior),("candidate",prediction)):fields.save_arrays(arrays,cid+"__"+label,p)
        arrays.update({cid+"__tick__"+k:v for k,v in ticks.items()})
        print(json.dumps({"completed_native":cid,"passes":reports[cid]["passes"]}),flush=True)
    np.savez_compressed(OUT/"native_trajectories.npz",**arrays)
    return reports,{"new_native_integrations":new,"recovered_native":recovered,"baseline_integrations":0,
        "all_arrays_finite":all(np.isfinite(v).all() for v in arrays.values())}


def plot_overlay(trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,cid in enumerate(("jari_259_30","jari_259_60")):
        for ax,key in zip(axes[row],base.PRIMARY):
            for p,label,color in zip(trajectories[cid],("JARI reference","Cached geometry","Cached horizontal sinc","Yaw previous-I output"),("#147378","#813a8a","#ce7b2f","#2465b3")):
                ax.plot(p["t"],p[key],label=label,color=color)
            ax.set(title=cid,xlabel="Original time (s)",ylabel=key+" ("+base.UNITS[key]+")",xlim=(0,trajectories[cid][0]["t"][-1] if key=="speed_mps" else 3.5))
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Actual free yaw integral-output phase test; unchanged horizontal sinc cap and original pitch/plant/AW limits",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"overlay_30_60.png",dpi=160);plt.close(fig)


def main():
    before=protected_snapshot();OUT.mkdir(parents=True,exist_ok=True)
    protocol={"candidate":"yaw natural=P+transported_previous_I+D; nextI raw Ki*error/48, original Ilim/high-only K transport, actual-natural original-cap .99 AW; pitch unchanged",
        "unchanged":"yaw sinc output cap, full horizontal relative PN, vertical PN/loft/S110/50G, raw PID, frozen five plant coefficients/drag/geometry/force/angular law,one surface0.4m,48Hz+4substeps",
        "Jari_gate":"Identical base.jari_gate:0/30 early+full speed/alpha/load no>10percent worse;60 each early+full improve, pitch no>10percent worse and first yaw/AoA amplitudes closer",
        "native_trigger":"Only if every strict Jari item passes;3 current H110/H27.5/H55native actualfree, same72 speed/signed-alpha/signed-pitch existing-window rules",
        "load":"Native has no independent load observation; only prediction differences",
        "runtime_inputs":"Creation own state/current external target only; source missile states/derivatives only in diagnostic metrics",
        "scans_fits_new_runtime_constants":0,"default_update_authorized":False}
    dec.audit.slew.write_json(OUT/"protocol.json",protocol)
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    inputs=dec.audit.inverse.source_inputs();reports,trajectories,verification=run_jari(inputs,coefficients)
    gate=base.jari_gate(reports);plot_overlay(trajectories)
    native,native_verification={}, {"new_native_integrations":0,"not_run_reason":"Original strict Jari gate failed"}
    if gate["passes"]:native,native_verification=run_native(coefficients)
    protected=before==protected_snapshot()
    verification.update(native=native_verification,protected_size_mtime_unchanged=protected,
        old_horizontal_sinc_and_all_other_artifacts_unchanged=protected,fits_scans_new_coefficients=0,source_data_in_runtime=False,
        parallel_authorized_exclusions=["scripts/analyze_r771_speed_residual.py","outputs/jari_offaxis/speed_residual/"])
    if not protected or not verification["all_arrays_finite"] or not native_verification.get("all_arrays_finite",True):raise AssertionError(verification)
    accepted=gate["passes"] and bool(native) and all(c["passes"] for c in native.values())
    dec.audit.slew.write_json(OUT/"results.json",{"task":"War Thunder single yaw integral-output phase experiment", "protocol":protocol,
        "cases":reports,"Jari_gate":gate,"failed_Jari_rules":[k for k,v in gate["rules"].items() if not v],
        "native_cases":native,"verification":verification,"candidate_accepted_by_original_Jari_gate":gate["passes"],
        "candidate_accepted_by_predeclared_gate":accepted,"evidence_grade":"D discrete PID interface hypothesis","default_model_changed":False})
    print(json.dumps({"Jari_gate":gate,"accepted":accepted,"verification":verification,"output":str(OUT)}),flush=True)


if __name__=="__main__":main()
