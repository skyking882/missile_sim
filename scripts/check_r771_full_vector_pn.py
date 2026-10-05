#!/usr/bin/env python3
"""One full relative-vector PN experiment for the War Thunder game.

Only the wind-up PN component changes. Frozen loft, plant, drag, S110 and
the current horizontal-sinc/original-pitch controller are reused unchanged.
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
import check_r771_horizontal_sinc_cap as best
import check_r771_collision_pn_probe as probe
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table, pn_acceleration, estimate_time_to_go_s
from aim120_model.math3d import add, clamp, cross, dot, norm, scale, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import RateAngleLoftGame, SurfaceCoefficients
from aim120_model.aerodynamics import body_axes_from_quaternion

OUT = ROOT / "outputs/jari_offaxis/full_vector_pn"
dec, geometry = best.dec, best.geometry
drag, fields, harness = dec.audit.drag, dec.audit.fix.finite.fields, dec.audit.fix.finite.harness
CHANNELS = tuple(best.UNITS)
WINDOWS = best.WINDOWS


class FullVectorPNGame(best.HorizontalSincCapGame):
    """Existing horizontal PN plus the relative-vector wind-up replacement."""
    def request(self, state, target, time_s, evaluation):
        # This preserves the original loft state and equations before the
        # horizontal/vertical PN replacement and existing outer boundaries.
        _, details = RateAngleLoftGame.request(self, state, target, time_s, evaluation)
        cfg = self.config["guidance"]
        horizontal_rate = abs(details["los_rate_rad_s"][1])
        pn0 = details["pn_mps2"]
        relative = scale(cross(details["los_rate_rad_s"], scale(sub(target.velocity, state.velocity), -1.)),
            float(cfg["pn_gain"]) * details["hit_gain"])
        basis = evaluation.wind_basis
        mixed = pn0 if horizontal_rate == 0. else add(pn0,
            scale(basis.right, dot(relative, basis.right) - dot(pn0, basis.right)))
        up_delta = dot(relative, basis.up) - dot(mixed, basis.up)
        pn = add(mixed, scale(basis.up, up_delta))
        gate = 1.
        if horizontal_rate != 0. and self.gate_loft and cfg.get("lofting_enabled", False):
            steepness = float(cfg["angle_to_acceleration_multiplier"])
            threshold = math.radians(abs(float(cfg["loft_omega_max_deg_s"])))
            z = clamp(steepness * (1. - horizontal_rate / max(threshold, 1e-12)), -60., 60.)
            gate = clamp((1. / (1. + math.exp(-z))) /
                (1. / (1. + math.exp(-clamp(steepness, -60., 60.)))), 0., 1.)
        loft_g = details["loft_g"] * gate
        maximum = float(cfg["maximum_lateral_acceleration_g"]) * self.gravity
        boundaries = dec.outer_boundaries(pn, loft_g, basis, self.gravity, maximum)
        original = boundaries["unslewed_request_G"]
        request = original
        if horizontal_rate != 0.:
            tick = int(round(time_s * self.controller_hz))
            is_tick = abs(time_s - tick / self.controller_hz) < 1e-9
            gain = interpolate_table((tick + 1) / self.controller_hz, cfg["flight_time_gain_table"])
            if is_tick and tick != self.slew_last_tick:
                before = self.slew_horizontal_g
                if gain > 1e-12:
                    limit = self.request_slew_g_s / self.controller_hz
                    self.slew_horizontal_g += max(-limit, min(limit, original[1] - before))
                    self.slew_active_updates += 1
                else:
                    self.slew_horizontal_g = 0.
                self.slew_last_step_g = self.slew_horizontal_g - before
                self.slew_last_tick = tick
                self.slew_commit_history.append({"tick": tick, "time_s": float(time_s),
                    "gain": gain, "target_g": original[1], "horizontal_state_g": self.slew_horizontal_g,
                    "step_g": self.slew_last_step_g, "active_updates": self.slew_active_updates})
            combined = (original[0], self.slew_horizontal_g)
            maximum_g = maximum / dec.G0_MPS2
            length = math.hypot(*combined)
            request = tuple(x * maximum_g / length for x in combined) if length > maximum_g else combined
            details.update(slew_original_request_g=original, slew_horizontal_state_g=self.slew_horizontal_g,
                slew_last_step_g=self.slew_last_step_g, slew_request_before_cap_g=combined,
                slew_limit_g_s=self.request_slew_g_s, slew_controller_tick=is_tick,
                slew_last_committed_tick=self.slew_last_tick, slew_active_updates=self.slew_active_updates,
                slew_scheduled_flight_gain=gain)
        details.update(pn_mps2=pn, original_pn_mps2=pn0,
            existing_full_horizontal_PN_mps2=mixed, full_relative_vector_PN_mps2=relative,
            wind_up_PN_replacement_mps2=up_delta, original_loft_g=details["loft_g"], loft_g=loft_g,
            horizontal_loft_gate=gate, horizontal_los_rate_rad_s=horizontal_rate,
            outer_AB_variant="full_relative_vector_PN_windnormal")
        self.request_records[time_s] = {"request_g": request, **details}
        speed = norm(sub(state.velocity, self.wind))
        a = (evaluation.qbar / evaluation.mass) / self.q_over_mass0
        c = self.coefficients
        self.current_control_k = c.control_gain * (1 + c.speed_slope * (speed / self.speed0 - 1)) / (a * c.restoring_gain)
        self.current_force_nonlinear = math.hypot(*evaluation.local_incidence) > dec.audit.fix.finite.ALPHA_S
        self.current_control_time, self.current_control_axis = time_s, 0
        beta = math.hypot(*evaluation.local_incidence)
        best.inverse_sinc(beta)
        self.game_control.current_beta_rad = beta
        return request, details

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(outer="N*hitGain cross(LOSrate,-Vrel), projected on original wind normal; frozen loft and S110",
            PN_vector_formula="N*hitGain*cross(LOSrate,-(VT-VM))",
            evidence_grade="D full relative-vector PN placement hypothesis",
            axial_before_projection="Existing PN axial component retained; final wind-normal projection removes it",
            source_data_in_runtime=False, new_fitted_coefficients=0)
        return result


def make_model(cfg, coefficients):
    return FullVectorPNGame(cfg, request_slew_g_s=110., drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def protected_snapshot():
    result = {}
    for folder in ("src", "config", "configs", "schemas", "docs", "data", "scripts", "outputs"):
        for p in (ROOT / folder).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or ".git" in p.parts or OUT in p.parents:
                continue
            rel = str(p.relative_to(ROOT))
            if rel == "scripts/analyze_r771_speed_residual.py" or rel.startswith("outputs/jari_offaxis/speed_residual/"):
                continue
            st = p.stat()
            result[str(p)] = [st.st_size, st.st_mtime_ns]
    return result


def focused_verify(model, result, cfg, coefficients, initial, target_at):
    replay, old = make_model(cfg, coefficients), best.make_model(cfg, coefficients)
    replay.initialize(dataclasses.replace(initial)); old.initialize(dataclasses.replace(initial))
    controller, pitch = best.HorizontalSincCapControl(), R771GameControl()
    memory = [(0., 0., 0.), (0., 0., 0.)]
    samples = {s["time_s"]: s for s in result["samples"]}
    limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    keys = ("request_same_state_G", "LOS_vector_formula_rad_s", "hitGain_formula", "PN_full_vector_formula_mps2",
        "same_state_horizontal_PN_mps2", "same_state_axial_PN_mps2", "normal_projected_full_vector_mps2",
        "outer_boundaries_S110_G", "loft_original_same_state_G", "loft_gate_same_state", "loft_minimum_weight_same_state",
        "loft_original_formula_G", "force_same_state_N", "drag_same_state_N", "angular_same_state_rad_s2",
        "K_same_state", "PID_held_rad", "PID_memory", "pitch_original_held_rad", "pitch_original_memory",
        "AW_original_same_state_next_I", "raw_I_excess", "clock48_tick")
    maxima = dict.fromkeys(keys, 0.)
    trace = {k: [] for k in ("t", "old_PN_windup_G", "new_PN_windup_G", "horizontal_PN_G", "loft_G",
        "old_request_pitch_G", "new_request_pitch_G", "wind_up_delta_G", "K", "gain", "held", "memory")}
    weight = 1.; initial_range = None
    for tick in result["ticks"]:
        ts = tick["time_s"]; s = samples[ts]; state = dec.audit.fix.finite.sample_state(s)
        held = tuple(s["evaluated_old_held_rad"]); target = target_at(ts)
        f = replay.evaluate(state, ts, held); before = replay.slew_horizontal_g
        request, d = replay.request(state, target, ts, f)
        fo = old.evaluate(state, ts, held); old_request, od = old.request(state, target, ts, fo)
        r = np.asarray(target.position) - np.asarray(state.position)
        rv = np.asarray(target.velocity) - np.asarray(state.velocity)
        los = np.cross(r, rv) / (r @ r)
        pure, closing, _ = pn_acceleration(tuple(r), tuple(rv), state.velocity, cfg["guidance"]["pn_gain"])
        hit = interpolate_table(estimate_time_to_go_s(float(np.linalg.norm(r)), closing, tuple(rv), cfg), cfg["guidance"]["time_to_hit_gain_table"])
        alt = cfg["guidance"]["pn_gain"] * hit * np.cross(los, -rv)
        basis = f.wind_basis; pn = np.asarray(d["pn_mps2"])
        projected = pn - np.asarray(basis.forward) * float(pn @ basis.forward)
        alt_projected = alt - np.asarray(basis.forward) * float(alt @ basis.forward)
        maximum = cfg["guidance"]["maximum_lateral_acceleration_g"] * model.gravity
        boundary = dec.outer_boundaries(tuple(alt), d["loft_g"], basis, model.gravity, maximum)
        expected, _, _ = dec.same_previous_slew(boundary["unslewed_request_G"], before, tick["gain"], abs(los[1]), maximum / dec.G0_MPS2)
        original_loft = 0.
        if cfg["guidance"]["lofting_enabled"]:
            distance = float(np.linalg.norm(r))
            if initial_range is None: initial_range = distance
            gc = cfg["guidance"]; multiplier = gc["angle_to_acceleration_multiplier"]
            threshold = model.transition_coefficient * math.radians(abs(gc["loft_omega_max_deg_s"]))
            z = max(-60., min(60., multiplier * (np.linalg.norm(los) / max(threshold, 1e-12) - 1.)))
            weight = min(weight, 1. / (1. + math.exp(z)))
            axes = body_axes_from_quaternion(state.orientation_quaternion)
            eta = math.atan2(-dot(tuple(r), axes.up), dot(tuple(r), axes.forward))
            xi = max(0., min(1., distance / initial_range))
            original_loft = .75 * xi**2 * ((gc["lofting_elevation_deg"] - multiplier * max(eta, 0.)) * weight + gc["target_elevation_deg"] * (1. - weight))
        values = {"request_same_state_G": max(abs(np.asarray(request) - tick["request_g"])),
            "LOS_vector_formula_rad_s": max(abs(los - d["los_rate_rad_s"])), "hitGain_formula": abs(hit - d["hit_gain"]),
            "PN_full_vector_formula_mps2": max(abs(alt - d["full_relative_vector_PN_mps2"])),
            "same_state_horizontal_PN_mps2": abs(dot(d["pn_mps2"], basis.right) - dot(od["pn_mps2"], basis.right)),
            "same_state_axial_PN_mps2": abs(dot(d["pn_mps2"], basis.forward) - dot(od["pn_mps2"], basis.forward)),
            "normal_projected_full_vector_mps2": max(abs(projected - alt_projected)),
            "outer_boundaries_S110_G": max(abs(np.asarray(expected) - request)),
            "loft_original_same_state_G": abs(d["original_loft_g"] - od.get("original_loft_g", od["loft_g"])),
            "loft_gate_same_state": abs(d["horizontal_loft_gate"] - od.get("horizontal_loft_gate", 1.)),
            "loft_minimum_weight_same_state": abs(replay.minimum_weight - old.minimum_weight),
            "loft_original_formula_G": abs(d["original_loft_g"] - original_loft),
            "force_same_state_N": norm(sub(f.wind_lift_force, fo.wind_lift_force)),
            "drag_same_state_N": norm(sub(f.wind_drag_force, fo.wind_drag_force)),
            "angular_same_state_rad_s2": max(abs(np.asarray(f.angular_acceleration) - fo.angular_acceleration)),
            "K_same_state": abs(replay.current_control_k - old.current_control_k),
            "clock48_tick": abs(ts * 48 - round(ts * 48))}
        for key, value in values.items(): maxima[key] = max(maxima[key], float(value))
        if tick["gain"] > 1e-12:
            controller.current_beta_rad = math.hypot(*f.local_incidence)
            pid = scheduled_pid_values(cfg["control"]["pid"], (round(ts * 48) + 1) / 48)
            for axis in (0, 1):
                error = tick["request_g"][axis] - tick["feedback_g"][axis]
                got = controller.update_axis(axis, ts, error, memory[axis], pid, limits[axis], tick["gain"], replay.current_control_k, replay.current_force_nonlinear)
                maxima["PID_held_rad"] = max(maxima["PID_held_rad"], abs(got[0] - tick["held_fin_rad"][axis]))
                maxima["PID_memory"] = max(maxima["PID_memory"], max(abs(np.asarray(got[1]) - tick["pid_memory"][axis])))
                original = pitch.update_axis(axis, ts, error, memory[axis], pid, limits[axis], tick["gain"], replay.current_control_k, replay.current_force_nonlinear)
                if axis == 0:
                    maxima["pitch_original_held_rad"] = max(maxima["pitch_original_held_rad"], abs(got[0] - original[0]))
                    maxima["pitch_original_memory"] = max(maxima["pitch_original_memory"], max(abs(np.asarray(got[1]) - original[1])))
                maxima["AW_original_same_state_next_I"] = max(maxima["AW_original_same_state_next_I"], abs(got[1][0] - original[1][0]))
                if controller.diagnostics[axis]["antiwindup_blocked"] != pitch.diagnostics[axis]["antiwindup_blocked"]:
                    raise AssertionError("original AW condition changed")
                maxima["raw_I_excess"] = max(maxima["raw_I_excess"], abs(got[1][0]) - pid["integral_limit"])
                memory[axis] = got[1]
            if controller.previous_high_k != pitch.previous_high_k or controller.nonlinear_latched != pitch.nonlinear_latched:
                raise AssertionError("original high-only I/K lifecycle")
        elif np.any(tick["held_fin_rad"]) or np.any(tick["pid_memory"]):
            raise AssertionError("inactive held/memory")
        for key, value in {"t": ts, "old_PN_windup_G": dot(od["pn_mps2"], basis.up)/dec.G0_MPS2,
            "new_PN_windup_G": dot(d["pn_mps2"], basis.up)/dec.G0_MPS2,
            "horizontal_PN_G": dot(d["pn_mps2"], basis.right)/dec.G0_MPS2,
            "loft_G": d["loft_g"], "old_request_pitch_G": old_request[0], "new_request_pitch_G": request[0],
            "wind_up_delta_G": d["wind_up_PN_replacement_mps2"]/dec.G0_MPS2,
            "K": replay.current_control_k, "gain": tick["gain"], "held": tick["held_fin_rad"], "memory": tick["pid_memory"]}.items(): trace[key].append(value)
    fixed = {"whole_config_equal": model.config == cfg, "five_coefficients_equal": model.coefficients == SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coefficients,
        "N6_loft_S110": cfg["guidance"]["pn_gain"] == 6. and model.request_slew_g_s == 110. and (model.amplitude_coefficient, model.transition_coefficient) == (.75, 1.3),
        "clock_surface": model.controller_hz == 48 and model.plant_substeps == 4 and model.arm == .4,
        "current_PID_class": type(model.game_control) is best.HorizontalSincCapControl,
        "creation_only": result["summary"]["creation_state_only"] and not result["model"]["source_data_in_runtime"]}
    if max(maxima.values()) > 1e-10 or not all(fixed.values()): raise AssertionError({"maxima": maxima, "fixed": fixed})
    return {"max_residuals": maxima, "fixed": fixed,
        "force_couple_closure": dec.audit.inverse.saved_closure(model, result)}, {k: np.asarray(v) for k,v in trace.items()}


def pair(o, b, c, windows=WINDOWS, channels=CHANNELS, units=best.UNITS):
    m = drag.pair_metrics(o, b, c, channels, windows)
    for w in m.values():
        for k,v in w["metrics"].items(): v["units"] = units[k]
    return m


def source_rates(ref):
    q = fields.gap.continuous_quaternions(ref["quaternion_xyzw"][:, [3, 0, 1, 2]])
    return {str(n): {"body_yaw_rate_deg_s": np.degrees(-fields.gap.body_rates(q, ref["t"], n)[:, 1]),
        "body_pitch_rate_deg_s": np.degrees(fields.gap.body_rates(q, ref["t"], n)[:, 2])} for n in (3,5)}


def run_case(cid, x, coefficients, geometry_cache, best_cache, arrays):
    model = make_model(x["config"], coefficients)
    path = OUT / "full_runs" / (cid + ".json")
    if path.exists(): raise FileExistsError("This experiment permits one actual forward per requested case")
    result = model.run(dataclasses.replace(x["initial"]), x["target_at"], end_time_s=x["config"]["performance"]["lifetime_s"])
    dec.audit.slew.write_json(path, result, True)
    prediction = fields.prediction_fields(result, model, x["run"]["scenario"])
    if cid == "collision":
        prior = drag.selected_arrays(best_cache, cid + "__local_free__")
        observed = drag.selected_arrays(best_cache, cid + "__reference__")
        baseline = prior
        best_full = json.loads((probe.OUT / "collision_best_free.json").read_text())
    else:
        observed = drag.selected_arrays(geometry_cache, cid + "__reference__")
        baseline = drag.selected_arrays(geometry_cache, cid + "__candidate__")
        prior = drag.selected_arrays(best_cache, cid + "__candidate__")
        best_full = json.loads((best.OUT / "full_runs" / (cid + ".json")).read_text())
    prior_model = best.make_model(x["config"], coefficients); prior_model.initialize(dataclasses.replace(x["initial"]))
    restored = fields.prediction_fields(best_full, prior_model, x["run"]["scenario"])
    if not all(np.array_equal(restored[k], prior[k]) for k in ("t",*CHANNELS)):
        raise AssertionError("current best saved free receipt does not restore arrays")
    focused, trace = focused_verify(model, result, x["config"], coefficients, x["initial"], x["target_at"])
    rates = source_rates(x["ref"])
    report = {"separate_metrics": pair(observed, baseline, prediction), "currentbest_separate_metrics": pair(observed, prior, prediction),
        "first_positive_peaks": best.peaks(observed, baseline, prediction),
        "first_positive_pitch_peaks": {label:best.first_peak(p,"body_pitch_deg") for label,p in (("reference",observed),("currentbest",prior),("candidate",prediction))},
        "source_rate_peaks_separate_deg_s": {n:{k:best.first_peak({"t":x["ref"]["t"],k:v},k) for k,v in values.items()}for n,values in rates.items()},
        "rate_semantics": "Quaternion3 and quaternion5 body rates are separately in deg/s; yaw=-bodyY,pitch=+bodyZ matching quaternion_derivative; exported raw omega units/frame unverified and not converted",
        "summary": result["summary"], "currentbest_summary": best_full["summary"], "focused_checks": focused,
        "full_result_path": str(path), "currentbest_cache_restores_exactly": True}
    if cid == "collision":
        report["paired_activation_0p3_0p7_metrics"] = pair(observed, prior, prediction, (("activation_0p3_0p7", .3, .7),))
        report["actual_pitch_at_0p5_s"] = {label:float(np.interp(.5,p["t"],p["body_pitch_deg"]))
            for label,p in (("reference",observed),("currentbest",prior),("candidate",prediction))}
    for label,p in (("reference",observed),("geometry",baseline),("currentbest",prior),("candidate",prediction)):
        fields.save_arrays(arrays,cid+"__"+label,p)
    arrays.update({cid+"__tick__"+k:v for k,v in trace.items()})
    arrays[cid+"__source_raw_omega_unverified"] = x["ref"]["omega_xyz_export_units"]
    for n,values in rates.items():
        arrays.update({cid+"__source_quaternion"+n+"__"+k:v for k,v in values.items()})
    print(json.dumps({"actual_forward_completed":cid,"summary":result["summary"],
        "currentbest_startup_RMSE":{k:[report["currentbest_separate_metrics"]["startup"]["metrics"][k][v]["rmse"]for v in("old","new")]for k in CHANNELS}}),flush=True)
    return report, (observed,baseline,prior,prediction)


def plot(trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(6,3,figsize=(16,17),sharex="col")
    order=("jari_259_30","jari_259_60","collision")
    for j,cid in enumerate(order):
        observed,baseline,prior,prediction=trajectories[cid]
        for i,k in enumerate(CHANNELS):
            ax=axes[i,j]
            for label,p,color in (("Source",observed,"#147378"),("Cached geometry",baseline,"#813a8a"),("Current best",prior,"#ce7b2f"),("Full relative-vector PN",prediction,"#2465b3")):
                mask=p["t"]<=3.5;ax.plot(p["t"][mask],p[k][mask],label=label,color=color)
            if k=="body_yaw_rate_deg_s":
                rates=observed.get("body_yaw_rate_5point_deg_s")
                if rates is not None:
                    mask=observed["t"]<=3.5;ax.plot(observed["t"][mask],rates[mask],"--",color="#528baa",label="Source quaternion5")
            ax.set_ylabel(k+" ("+best.UNITS[k]+")");ax.grid(alpha=.2);ax.axvspan(14/48,.7,color="#cbd9e5",alpha=.15)
            if i==0:ax.set_title(cid)
            if i==5:ax.set_xlabel("Original time (s)")
            ax.legend(fontsize=6)
    fig.suptitle("War Thunder game: full relative-vector PN, original loft and current best PID\nFour actual free cases; source states used only for diagnostics",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(OUT/"overlay_30_60_collision.png",dpi=145);plt.close(fig)


def native_stage(coefficients):
    cases={c["id"]:c for c in harness.load_cases(dec.audit.fix.finite.comparison.SNAPSHOT)}
    gcache=np.load(geometry.OUT/"native_trajectories.npz"); bcache=np.load(best.OUT/"native_trajectories.npz")
    channels=("speed_kmh","alpha_deg","pitch_deg","omega_rad_s","gamma_deg","y_m")
    windows=(*dec.WINDOWS,*harness.WINDOWS); reports,arrays={},{}
    for cid in best.NATIVE:
        x=cases[cid.removeprefix("native_")];cfg=copy.deepcopy(x["profile"]["_model_config"])
        initial=dataclasses.replace(x["_creation_state"]);qx,qy,qz,qw=x["creation"]["u16_1"];initial.orientation_quaternion=(qw,qx,qy,qz)
        target=x["_target_model"].state_at;model=make_model(cfg,coefficients);path=OUT/"full_runs"/(cid+".json")
        if path.exists():raise FileExistsError("native actual run exists")
        result=model.run(initial,target,end_time_s=x["scenario"]["max_simulation_time_s"]);dec.audit.slew.write_json(path,result,True)
        obs=drag.selected_arrays(gcache,cid+"__reference__");baseline=drag.selected_arrays(gcache,cid+"__candidate__")
        prior=drag.selected_arrays(bcache,cid+"__candidate__");pred=harness.sample_series(result["samples"],x)
        metrics=pair(obs,baseline,pred,windows,channels,harness.UNITS)
        rules={w+"__"+k:m["new"]["rmse"]<=1.1*m["old"]["rmse"]+1e-12 for w,item in metrics.items()for k,m in item["metrics"].items()if k in("speed_kmh","alpha_deg","pitch_deg")}
        load={}
        for name,lo,hi in windows:
            mask=(obs["t"]>=lo-1e-9)&(obs["t"]<=hi+1e-9)&(obs["t"]<=min(baseline["t"][-1],pred["t"][-1]))
            ts=obs["t"][mask];load[name]={"independent_observation_available":False,"prediction_delta_G":drag.stats(np.interp(ts,pred["t"],pred["load_g"])-np.interp(ts,baseline["t"],baseline["load_g"]))}
        focused,trace=focused_verify(model,result,cfg,coefficients,initial,target)
        reports[cid]={"separate_metrics":metrics,"currentbest_separate_metrics":pair(obs,prior,pred,windows,channels,harness.UNITS),
            "rules":rules,"passes":all(rules.values()),"load_prediction_changes":load,"focused_checks":focused,"summary":result["summary"]}
        for label,p in(("reference",obs),("geometry",baseline),("currentbest",prior),("candidate",pred)):fields.save_arrays(arrays,cid+"__"+label,p)
        arrays.update({cid+"__tick__"+k:v for k,v in trace.items()})
    np.savez_compressed(OUT/"native_trajectories.npz",**arrays)
    return reports


def main():
    protected=protected_snapshot();OUT.mkdir(parents=True,exist_ok=True);(OUT/"full_runs").mkdir(exist_ok=True)
    protocol={"candidate":"Replace only current full-horizontal PN wind-up component with N*hitGain cross(LOSrate,-Vrel); retain old axial until original projection",
        "frozen":"five coefficients,drag,oneface0.4,raw PID,original AW/I high lifecycle,yaw-only sinc,pitchoriginal,N6,loft0.75/1.3,S110,50G,48Hz4",
        "old_Jari_gate":"Exactly current best.jari_gate against cached geometry:0/30 early/full primary no>10%;60 early/full primary all improve,pitch no>10%,first yaw/AoA amplitude closer",
        "collision_gate":"startup14/48..0.7 pitch/AoA/load RMSE each improves against cached currentbest; speed RMSE no>10% worse in every original window",
        "native_trigger":"Only if original strict old-Jari gate AND collision gate pass; then3currentnative, original72 no>10% speed/signedalpha/pitch rules",
        "forward_budget":{"Jari":3,"collision":1,"conditional_native":3,"new_baselines":0,"headon":0},
        "grade":"D structural placement, not recovered game guidance", "runtime_source_states":False,"fitting_scans":0}
    dec.audit.slew.write_json(OUT/"protocol.json",protocol)
    inputs=dec.audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    head=json.loads((probe.OUT/"headon_config_response.json").read_text()); collision=json.loads((probe.OUT/"collision_config_response.json").read_text())
    old=inputs["jari_259_30"]["run"]
    if not head["samples"]==old["samples"]:raise AssertionError("paired head-on source is not identical to old30")
    if not head["profile_source"]==collision["profile_source"]==old["profile_source"] or not head["backend"]==collision["backend"]==old["backend"]:
        raise AssertionError("source profiles/backends mismatch")
    differences={k:[head["scenario"].get(k),old["scenario"].get(k)]for k in head["scenario"].keys()|old["scenario"].keys()if head["scenario"].get(k)!=old["scenario"].get(k)}
    ref,source_diag=probe.comparison.parse_reference(collision)
    profiles=json.loads(probe.comparison.SNAPSHOT.read_text());profile,bridge=probe.comparison.profile_for(collision,{},profiles)
    cfg=copy.deepcopy(profile["_model_config"]);initial,target=probe.comparison.free_inputs(collision,ref,cfg)
    xcollision={"config":cfg,"initial":initial,"target_at":target,"ref":ref,"run":collision}
    gc=np.load(geometry.OUT/"free_trajectories.npz");bc=np.load(best.OUT/"trajectories.npz");cc=np.load(probe.OUT/"trajectories.npz")
    reports,trajectories,arrays={},{},{}
    for cid in best.CASES:
        reports[cid],trajectories[cid]=run_case(cid,inputs[cid],coefficients,gc,bc,arrays)
    reports["collision"],trajectories["collision"]=run_case("collision",xcollision,coefficients,gc,cc,arrays)
    jgate=best.jari_gate(reports)
    cm=reports["collision"]["currentbest_separate_metrics"]
    crules={"startup__"+k:cm["startup"]["metrics"][k]["new"]["rmse"]<cm["startup"]["metrics"][k]["old"]["rmse"]for k in("body_pitch_deg","alpha_deg","normal_specific_load_g")}
    crules.update({w+"__speed_no10percent_worse":m["metrics"]["speed_mps"]["new"]["rmse"]<=1.1*m["metrics"]["speed_mps"]["old"]["rmse"]+1e-12 for w,m in cm.items()})
    cgate={"rules":crules,"passes":all(crules.values())}
    native=native_stage(coefficients)if jgate["passes"]and cgate["passes"]else {}
    np.savez_compressed(OUT/"trajectories.npz",**arrays);plot(trajectories)
    unchanged=protected==protected_snapshot()
    if not unchanged or not all(np.isfinite(v).all()for v in arrays.values()):raise AssertionError("protected receipt or finite arrays")
    report={"protocol":protocol,"cases":reports,"original_strict_Jari_gate":jgate,"collision_gate":cgate,"native_cases":native,
        "candidate_accepted":jgate["passes"]and cgate["passes"]and bool(native)and all(v["passes"]for v in native.values()),
        "source_pair":{"headon_samples_identical_to_old30":True,"headon_metadata_scenario_differences":differences,
            "profiles_and_compiled_backend_identical":True,"collision_source_bridge":bridge,"collision_parse_diagnostics":source_diag,
            "headon_extra_run_needed":False,"source_version":"2.59.0.28","local_profile_version":"2.59.0.27"},
        "verification":{"actual_Jari_forward":3,"actual_collision_forward":1,"actual_native_forward":len(native),"baseline_forward":0,"API_requests":0,
            "protected_size_mtime_unchanged":unchanged,"source_states_runtime":False,"fitting_scans_new_coefficients":0,"default_core_config_data_docs_changed":False}}
    dec.audit.slew.write_json(OUT/"results.json",report)
    print(json.dumps({"original_strict_Jari_gate":jgate,"collision_gate":cgate,"candidate_accepted":report["candidate_accepted"],"verification":report["verification"]}),flush=True)


if __name__=="__main__":main()
