#!/usr/bin/env python3
"""Bounded request/control-coordinate experiments for the War Thunder game.

Saved references and needed inputs are diagnostic only. Every proposed flight
starts from the creation state and prescribed target motion, with frozen plant.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import check_r771_angle_grid as grid
import check_r771_horizontal_sinc_cap as best
import analyze_r771_pitch_reversal as pitch_audit
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm, sub
from aim120_model.single_surface_game import SurfaceCoefficients

OUT = ROOT / "outputs/jari_offaxis/command_coordinates"
SCREEN = (0, 5, 30, 75)
TRANSFER = (10, 15, 45, 60, "collision")
REQUIRED = (*best.PRIMARY, "body_pitch_deg")
CANDIDATE = "full_active_K_transport_actual_output_tracking"


class CommandCoordinateControl(best.HorizontalSincCapControl):
    """One causal candidate; old-radian tuples encode a normalized coordinate."""
    def __init__(self):
        super().__init__()
        self.previous_active_k = [None,None]

    def update_axis(self,axis,time_s,error_g,memory,pid,old_held_limit_rad,gain,
                    target_per_old_held,nonlinear_force_active):
        if axis not in (0,1): raise ValueError("axis must be pitch0 or yaw1")
        k = float(target_per_old_held)
        if not all(math.isfinite(v) for v in (time_s,error_g,k,old_held_limit_rad,gain)) or k<=0 or old_held_limit_rad<=0 or not 0<=gain<=1:
            raise ValueError("finite positive normalization/limit and flight gain in [0,1] required")
        if self.last_time[axis] is not None:
            if time_s==self.last_time[axis]: return self.cached[axis]
            if time_s<self.last_time[axis]: raise ValueError("control clock cannot move backwards")
        if abs(time_s*self.controller_hz-round(time_s*self.controller_hz))>1e-8:
            raise ValueError("control memory may update only at controller ticks")
        if gain<=1e-12:
            if memory!=(0.,0.,0.): raise ValueError("inactive PID memory must be zero")
            return 0.,memory
        self.nonlinear_latched = self.nonlinear_latched or bool(nonlinear_force_active)
        integral,previous_error,previous_derivative = memory
        previous_k = self.previous_active_k[axis]
        ratio = 1. if previous_k is None else previous_k/k
        ilim = float(pid["integral_limit"])
        transported = best.dec.audit.clip(ratio*integral,ilim)
        dt = 1./self.controller_hz
        increment = float(pid["i"])*error_g*dt
        provisional = best.dec.audit.clip(transported+increment,ilim)
        derivative = previous_derivative+dt/(1e-9+dt)*((error_g-previous_error)/dt-previous_derivative)
        p = float(pid["p"])*error_g
        d = float(pid["d"])*derivative
        natural = p+provisional+d
        beta = float(self.current_beta_rad)
        factor = best.inverse_sinc(beta)
        cap = min(old_held_limit_rad,old_held_limit_rad/k*factor)
        saturated = best.dec.audit.clip(natural,cap)
        tracking_delta = saturated-natural
        next_i = best.dec.audit.clip(provisional+tracking_delta,ilim)
        result = gain*saturated,(next_i,error_g,derivative)
        self.last_time[axis] = time_s
        self.cached[axis] = result
        self.previous_active_k[axis] = k
        self.diagnostics[axis] = {"time_s":time_s,"axis":axis,"error_g":error_g,
            "K_target_per_old_held":k,"previous_active_K":previous_k,
            "nonlinear_latched":self.nonlinear_latched,"nonlinear_latch_affects_control":False,
            "integral_transport_ratio":ratio,"transported_integral_old_rad":transported,
            "transport_unclipped_delta_old_rad":(ratio-1)*integral,
            "transport_clip_delta_old_rad":transported-ratio*integral,
            "integral_target_rad":k*next_i,"P_target_rad":k*p,"D_target_rad":k*d,
            "target_candidate_limit_rad":old_held_limit_rad,"effective_old_held_limit_rad":cap,
            "original_old_held_limit_rad":old_held_limit_rad,"antiwindup_blocked":False,
            "antiwindup_tracking_applied":tracking_delta!=0.,"tracking_delta_old_rad":tracking_delta,
            "tracking_integral_clip_delta_old_rad":next_i-provisional-tracking_delta,
            "held_old_rad":result[0],"effective_target_rad":k*result[0],
            "normalized_target_at_tick":k*result[0]/old_held_limit_rad,"gain":gain,
            "P_old_rad":p,"I_provisional_old_rad":provisional,"D_old_rad":d,
            "I_increment_old_rad":increment,"natural_old_rad":natural,
            "beta_rad":beta,"inverse_sinc_factor":factor,"next_integral_old_rad":next_i,
            "original_AW_limit_old_rad":cap,"AW_limit_is_actual_output_cap":True,
            "normalized_integral_ZI":k*next_i/old_held_limit_rad,
            "all_active_I_transport":True,"same_cap_all_axes_all_active":True}
        return result


class CommandCoordinateGame(best.HorizontalSincCapGame):
    def initialize(self,state):
        initial = super().initialize(state)
        self.game_control = CommandCoordinateControl()
        return initial

    def run(self,initial_state,target_at,*,end_time_s):
        result = super().run(initial_state,target_at,end_time_s=end_time_s)
        result["model"].update(runtime_controller="check_r771_command_coordinates.CommandCoordinateControl",
            candidate=CANDIDATE,evidence_grade="D unverified shared control-interface mapping",
            controller_mechanism="All-active K/L integral coordinate; both-axis raw and target sinc cap; exact one-tick actual-output tracking",
            integral_transport="Every active tick I_old <- clip(K_previous/K_current*I_old,rawIlimit), first ratio1",
            experimental_target_cap="both axes min(raw,raw/K*inverse_sinc(beta)); every active tick",
            saturation_memory="I_next=clip(I_star+sat-natural,rawIlimit); flight gain remains last",
            nonlinear_trigger="reported shared latch only; it does not gate coordinate/cap/AW",
            new_fitted_coefficients=0,source_data_in_runtime=False)
        return result


def make_candidate(cfg,coeff):
    return CommandCoordinateGame(cfg,request_slew_g_s=110.,drag_coefficients=coeff,
        coefficients=SurfaceCoefficients(),force_basis="wind_linear",plant_substeps=4,loft_cap_unit="rad_s")


def dump(path, value):
    path.write_text(json.dumps(grid.probe.serializable(value), indent=2,
        ensure_ascii=False, allow_nan=False) + "\n")


def protected():
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
        for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs")
        for p in (ROOT / folder).rglob("*") if p.is_file()
        and "__pycache__" not in p.parts and ".git" not in p.parts
        and p != Path(__file__).resolve() and OUT not in p.parents}


def inputs():
    receipt = json.loads((grid.OUT / "results.json").read_text())
    profiles = json.loads(grid.comparison.SNAPSHOT.read_text())
    coeff = best.dec.audit.drag.DragCoefficients(**json.loads(
        (best.dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    answer = {}
    for angle in (*SCREEN, *TRANSFER):
        name = grid.cid(angle) if angle != "collision" else "collision"
        if angle == "collision":
            source = grid.probe.OUT / "collision_config_response.json"
            baseline = grid.probe.OUT / "collision_best_free.json"
        else:
            source = Path(receipt["cases"][name]["source_path"])
            baseline = Path(receipt["cases"][name]["candidate_full_path"])
        run = json.loads(source.read_text())
        ref, source_checks = grid.comparison.parse_reference(run)
        profile, bridge = grid.comparison.profile_for(run, {}, profiles)
        cfg = copy.deepcopy(profile["_model_config"])
        initial, target_at = grid.comparison.free_inputs(run, ref, cfg)
        old_model = best.make_model(cfg, coeff)
        old_model.initialize(dataclasses.replace(initial))
        old_full = json.loads(baseline.read_text())
        observed, _, _ = grid.probe.source_geometry(run, ref, cfg)
        old_prediction = grid.fields.prediction_fields(old_full, old_model, run["scenario"])
        grid.add_rates(observed, old_prediction, ref, old_full)
        answer[name] = {"angle": angle, "run": run, "ref": ref, "config": cfg,
            "initial": initial, "target_at": target_at, "source_path": source,
            "source_checks": source_checks, "bridge": bridge,
            "baseline_path": baseline, "baseline_full": old_full,
            "observed": observed, "baseline_prediction": old_prediction}
    return coeff, answer


def pair_metrics(observed, baseline, candidate):
    # Baseline and candidate use exactly the same source support for ratios.
    low = max(baseline["t"][0], candidate["t"][0])
    high = min(baseline["t"][-1], candidate["t"][-1])
    mask = (observed["t"] >= low - 1e-9) & (observed["t"] <= high + 1e-9)
    selected = {key: np.asarray(value)[mask] for key, value in observed.items()}
    return {"baseline": grid.metrics(selected, baseline),
        "candidate": grid.metrics(selected, candidate),
        "source_rows_outside_common_support": np.flatnonzero(~mask).tolist(),
        "original_source_rows": len(mask), "common_source_rows": int(mask.sum()),
        "all_supported_unfavorable_rows_retained": True}


def compatibility(pair):
    rules = {}
    for window in ("early", "full_common"):
        old = pair["baseline"]["windows"][window]["metrics"]
        new = pair["candidate"]["windows"][window]["metrics"]
        for field in REQUIRED:
            rules[window + "__" + field] = new[field]["rmse"] <= 1.1 * old[field]["rmse"] + 1e-12
    return {"rules": rules, "passes": all(rules.values()),
        "failed_rules": [key for key, value in rules.items() if not value]}


def first_peaks(trajectory):
    return {key: grid.first_peak(trajectory, key) for key in grid.CHANNELS[1:]}


def events(result):
    active = [row for row in result["ticks"] if row["gain"] > 1e-12]
    t = np.asarray([row["time_s"] for row in active])
    answer = {}
    for axis, label in enumerate(("pitch", "yaw")):
        ds = [row["control_runtime_diagnostics"][axis] for row in active]
        clipped = np.asarray([abs(d["natural_old_rad"]) > d["effective_old_held_limit_rad"] for d in ds])
        below = np.flatnonzero((t >= .5 - 1e-9) & ~clipped)
        answer[label] = {"held": grid.events(t, [row["held_fin_rad"][axis] for row in active]),
            "request": grid.events(t, [row["request_g"][axis] for row in active]),
            "first_unclipped_at_or_after_0p5_s": float(t[below[0]]) if len(below) else None,
            "clipped_active_ticks": int(clipped.sum()),
            "AW_blocked_active_ticks": sum(bool(d["antiwindup_blocked"]) for d in ds),
            "selected_ticks": [{"time_s": row["time_s"], "request_g": row["request_g"][axis],
                "feedback_g": row["feedback_g"][axis], "held_rad": row["held_fin_rad"][axis],
                "diagnostics": d} for row, d in zip(active, ds)
                if any(abs(row["time_s"] - ts) < 1e-8 for ts in (.5, 28 / 48, .625, 34 / 48, .75, 1., 55 / 48))]}
    return answer


def trace_arrays(result):
    active = [row for row in result["ticks"] if row["gain"] > 1e-12]
    fields = ("K_target_per_old_held", "P_old_rad", "I_provisional_old_rad", "D_old_rad",
        "next_integral_old_rad", "natural_old_rad", "effective_old_held_limit_rad",
        "original_AW_limit_old_rad", "antiwindup_blocked", "integral_transport_ratio")
    answer = {"t": np.asarray([row["time_s"] for row in active])}
    for field in fields:
        answer[field] = np.asarray([[d[field] for d in row["control_runtime_diagnostics"]] for row in active])
    for field in ("request_g", "feedback_g", "held_fin_rad", "pid_memory"):
        answer[field] = np.asarray([row[field] for row in active])
    return answer


def verify_saved(candidate_factory, model, result, item, coeff):
    """Reconstruct causal runtime from saved free states, without another flight."""
    cfg = item["config"]
    rebuilt = candidate_factory(cfg, coeff)
    rebuilt.initialize(dataclasses.replace(item["initial"]))
    frozen = best.make_model(cfg, coeff)
    frozen.initialize(dataclasses.replace(item["initial"]))
    limits = np.radians([cfg["aerodynamics"][key] for key in
        ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    samples = {sample["time_s"]: sample for sample in result["samples"]}
    memory = [(0., 0., 0.), (0., 0., 0.)]
    residuals = {key: 0. for key in ("request_G", "feedback_G", "held_rad", "PID_memory",
        "frozen_acceleration_mps2", "frozen_qdot_rad_s2", "drag_N", "lift_N",
        "same_station_torque_Nm", "K_formula", "held_bound_excess_rad", "inactive_memory",
        "independent_normalized_ZI_memory_rad", "independent_normalized_Z_output_rad",
        "transport_delta_equation_rad", "tracking_delta_equation_rad")}
    independent_previous_k = [None,None]
    for tick in result["ticks"]:
        ts = tick["time_s"]
        sample = samples[ts]
        state = best.dec.audit.fix.finite.sample_state(sample)
        previous_held = tuple(tick["evaluated_old_held_rad"])
        f = rebuilt.evaluate(state, ts, previous_held)
        old_f = frozen.evaluate(state, ts, previous_held)
        request, _ = rebuilt.request(state, item["target_at"](ts), ts, f)
        residuals["request_G"] = max(residuals["request_G"], max(abs(a-b) for a,b in zip(request,tick["request_g"])))
        residuals["feedback_G"] = max(residuals["feedback_G"], max(abs(a-b) for a,b in zip(f.feedback_g,tick["feedback_g"])))
        for key, field in (("frozen_acceleration_mps2", "acceleration"),
            ("frozen_qdot_rad_s2", "angular_acceleration"), ("drag_N", "wind_drag_force"),
            ("lift_N", "wind_lift_force")):
            residuals[key] = max(residuals[key], max(abs(a-b) for a,b in zip(getattr(f,field),getattr(old_f,field))))
        residuals["same_station_torque_Nm"] = max(residuals["same_station_torque_Nm"],
            max(abs(f.inertia*q-l-c) for q,l,c in zip(f.angular_acceleration,f.lever_moment,f.reference_couple)))
        c = rebuilt.coefficients
        a = (f.qbar / f.mass) / rebuilt.q_over_mass0
        expected_k = c.control_gain * (1 + c.speed_slope * (norm(sub(state.velocity,rebuilt.wind))/rebuilt.speed0-1)) / (a*c.restoring_gain)
        residuals["K_formula"] = max(residuals["K_formula"], abs(rebuilt.current_control_k-expected_k))
        pid = scheduled_pid_values(cfg["control"]["pid"], (round(ts*48)+1)/48)
        if tick["gain"] <= 1e-12:
            residuals["inactive_memory"] = max(residuals["inactive_memory"],
                float(np.max(np.abs(tick["pid_memory"]))), max(abs(v) for v in tick["held_fin_rad"]))
            continue
        for axis in (0,1):
            error = request[axis]-f.feedback_g[axis]
            previous_i,previous_e,previous_d = memory[axis]
            current_k = rebuilt.current_control_k
            limit = float(limits[axis])
            prior_k = current_k if independent_previous_k[axis] is None else independent_previous_k[axis]
            zmax = current_k/limit*pid["integral_limit"]
            ztransported = float(np.clip(prior_k/limit*previous_i,-zmax,zmax))
            zstar = float(np.clip(ztransported+current_k/limit*pid["i"]*error/48,-zmax,zmax))
            derivative = previous_d+(1/48)/(1e-9+1/48)*((error-previous_e)*48-previous_d)
            znatural = current_k/limit*(pid["p"]*error+pid["d"]*derivative)+zstar
            zcap = min(current_k,1./float(np.sinc(math.hypot(*f.local_incidence)/math.pi)))
            zsaturated = float(np.clip(znatural,-zcap,zcap))
            znext = float(np.clip(zstar+zsaturated-znatural,-zmax,zmax))
            expected_output = tick["gain"]*limit/current_k*zsaturated
            expected_i = limit/current_k*znext
            output, new_memory = rebuilt._pid(error, memory[axis], pid,
                float(limits[axis]), tick["gain"])
            residuals["held_rad"] = max(residuals["held_rad"], abs(output-tick["held_fin_rad"][axis]))
            residuals["PID_memory"] = max(residuals["PID_memory"],
                max(abs(x-y) for x,y in zip(new_memory,tick["pid_memory"][axis])))
            residuals["held_bound_excess_rad"] = max(residuals["held_bound_excess_rad"], abs(output)-limits[axis])
            residuals["independent_normalized_ZI_memory_rad"] = max(residuals["independent_normalized_ZI_memory_rad"],abs(expected_i-new_memory[0]))
            residuals["independent_normalized_Z_output_rad"] = max(residuals["independent_normalized_Z_output_rad"],abs(expected_output-output))
            d = rebuilt.game_control.diagnostics[axis]
            residuals["transport_delta_equation_rad"] = max(residuals["transport_delta_equation_rad"],
                abs(d["transport_unclipped_delta_old_rad"]-(prior_k/current_k-1)*previous_i))
            residuals["tracking_delta_equation_rad"] = max(residuals["tracking_delta_equation_rad"],
                abs(d["tracking_delta_old_rad"]-limit/current_k*(zsaturated-znatural)))
            independent_previous_k[axis] = current_k
            memory[axis] = new_memory
    fixed = {"raw_config_PID_loft_equal": model.config == cfg,
        "five_frozen_coefficients_equal": model.coefficients == SurfaceCoefficients(),
        "frozen_drag_equal": model.drag_coefficients == coeff,
        "controller48_plant_substeps4": model.controller_hz == 48 and model.plant_substeps == 4,
        "single_equivalent_station0p4m": model.arm == .4,
        "loft_amplitude_transition_frozen": [model.amplitude_coefficient,model.transition_coefficient] == [.75,1.3],
        "initial_state_equals_cached_creation": result["samples"][0]["position_m"] == list(item["initial"].position) or tuple(result["samples"][0]["position_m"]) == item["initial"].position,
        "source_feedback_in_runtime": False}
    if max(residuals.values()) > 1e-10 or not all(v for k,v in fixed.items() if k != "source_feedback_in_runtime"):
        raise AssertionError({"residuals": residuals, "fixed": fixed})
    saved = best.dec.audit.inverse.saved_closure(model, result)
    return {"causal_runtime_and_frozen_plant_max_residuals": residuals, "fixed": fixed,
        "saved_force_torque_record_closure": saved, "verification_forward_integrations": 0}


def same_state_diagnostic(candidate_factory, item, coeff):
    """Reference-state, own-memory diagnostics; never an input to free runs."""
    kin_input = {"ref": item["ref"]}
    kinetics = pitch_audit.source_kinematics(kin_input)
    result = {}
    for width in (3,5):
        kin = kinetics[f"quaternion{width}"]
        end = min(3.5,float(item["ref"]["t"][-1]))
        times = np.arange(math.floor(end*48)+1)/48
        rates = pitch_audit.interp(times,item["ref"]["t"],kin["rates"])
        qdot = pitch_audit.interp(times,item["ref"]["t"],kin["qdot"])
        models = {"baseline": best.make_model(item["config"],coeff),
            "candidate": candidate_factory(item["config"],coeff)}
        for model in models.values(): model.initialize(dataclasses.replace(item["initial"]))
        memories = {label:[(0.,0.,0.),(0.,0.,0.)] for label in models}
        limits = np.radians([item["config"]["aerodynamics"][key] for key in
            ("horizontal_fin_aoa_limit_deg","vertical_fin_aoa_limit_deg")])
        rows = []
        for i,ts in enumerate(times):
            state = dataclasses.replace(best.dec.audit.fix.source_state_at(item["ref"],float(ts)),
                pitch_rate=float(rates[i,0]),yaw_rate=float(rates[i,1]))
            stored = {"time_s":float(ts)}
            for label,model in models.items():
                f = model.evaluate(state,float(ts),(0.,0.))
                request,_ = model.request(state,item["target_at"](float(ts)),float(ts),f)
                gain = interpolate_table(float(ts)+1/48,item["config"]["guidance"]["flight_time_gain_table"])
                pid = scheduled_pid_values(item["config"]["control"]["pid"],float(ts)+1/48)
                held = []
                if gain > 1e-12:
                    for axis in (0,1):
                        u,memories[label][axis] = model._pid(request[axis]-f.feedback_g[axis],memories[label][axis],
                            pid,float(limits[axis]),gain)
                        held.append(u)
                else: held = [0.,0.]
                stored[label] = {"held_rad":held,"request_G":request,"feedback_G":f.feedback_g,
                    "PID_memory":copy.deepcopy(memories[label]),
                    "diagnostics":copy.deepcopy(model.game_control.diagnostics) if gain>1e-12 else None}
                if label == "baseline":
                    c = model.coefficients
                    ku = c.control_gain*(1+c.speed_slope*(norm(sub(state.velocity,model.wind))/model.speed0-1))
                    needed = (qdot[i]-np.asarray(f.angular_acceleration))/ku
                    closed = model.evaluate(state,float(ts),tuple(needed)).angular_acceleration
                    if np.max(np.abs(np.asarray(closed)-qdot[i])) > 1e-10:
                        raise AssertionError("Conditional needed-input angular inverse does not close")
                    stored["needed_old_held_rad"] = needed
            rows.append(stored)
        need = np.asarray([r["needed_old_held_rad"] for r in rows])
        tables = {}
        for label in models:
            held = np.asarray([r[label]["held_rad"] for r in rows])
            tables[label] = {}
            for window,low,high in grid.WINDOWS:
                # Noncausal derivative startup halo is explicit and excluded.
                mask = (times >= max(low,14/48+kin["derivative_half_support_s"])+1e-9) & (times <= high+1e-9)
                if not mask.any(): continue
                tables[label][window] = {axis:grid.stats((held-need)[mask,j]) for j,axis in enumerate(("pitch","yaw"))}
        points = [rows[int(round(ts*48))] for ts in (.5,28/48,.625,34/48,.75,1.,55/48) if ts <= times[-1]]
        result[f"quaternion{width}"] = {"input_old_radians_residuals":tables,"points":points,
            "source_48Hz_interpolation_and_own_memory_conditional":True,
            "derivative_noncausal_half_support_s":kin["derivative_half_support_s"],
            "first_candidate_negative_pitch_s":pitch_audit.first_negative(times,
                np.asarray([r["candidate"]["held_rad"][0] for r in rows])),
            "source_needed_first_negative_pitch_after_halo_s":pitch_audit.first_negative(times,
                np.where(times>14/48+kin["derivative_half_support_s"],need[:,0],0.)),
            "interpretation":"Conditional diagnostics do not identify hidden game PID memory, actuator units, or native plant law"}
    return result


def overlay(all_trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(SCREEN)
    channels = (*best.PRIMARY, "body_pitch_deg", "body_yaw_rate_deg_s")
    fig, axes = plt.subplots(len(channels), len(order), figsize=(15, 13), sharex=True)
    colors = {"reference": "#157b7b", "baseline": "#8c8c8c", "candidate0": "#bf661d", "candidate1": "#8053a0"}
    for j, angle in enumerate(order):
        name = grid.cid(angle)
        for i, field in enumerate(channels):
            ax = axes[i, j]
            for label, p in all_trajectories[name].items():
                ax.plot(p["t"], p[field], color=colors[label], label=label, lw=1.2)
            if i == 0: ax.set_title(f"{angle}°")
            if j == 0: ax.set_ylabel(field.replace("body_", "").replace("_", " ") + " (" + grid.UNITS[field] + ")", fontsize=8)
            ax.set_xlim(0, 3.5); ax.grid(alpha=.22)
            if i == len(channels) - 1: ax.set_xlabel("Original time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("War Thunder: frozen plant, request/control-coordinate screen\nActual free prediction; no source trajectory or needed-input feedback", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "screen_responses.png", dpi=145)
    plt.close(fig)


def direction_checks(name,pair,peaks,baseline_events,candidate_events):
    checks = {}
    if name in ("angle_p0","angle_p5"):
        for window in ("early","unloading"):
            checks[window+"__pitch_RMSE_improves"] = pair["candidate"]["windows"][window]["metrics"]["body_pitch_deg"]["rmse"] < pair["baseline"]["windows"][window]["metrics"]["body_pitch_deg"]["rmse"]
        key = "body_pitch_deg"
    elif name == "angle_p75": key = "alpha_deg"
    else: key = None
    if key is not None:
        r,b,c = (peaks[label][key] for label in ("reference","baseline","candidate"))
        checks["first_"+key+"_amplitude_closer"] = bool(r and b and c and
            abs(c["signed_value"]-r["signed_value"]) < abs(b["signed_value"]-r["signed_value"]))
        checks["first_"+key+"_timing_no_worse"] = bool(r and b and c and
            abs(c["time_s"]-r["time_s"]) <= abs(b["time_s"]-r["time_s"])+1e-9)
    if name == "angle_p75":
        old = baseline_events["yaw"]["first_unclipped_at_or_after_0p5_s"]
        new = candidate_events["yaw"]["first_unclipped_at_or_after_0p5_s"]
        # Source's original raw controls first release at7/12; units stay unknown.
        target_release = 7/12
        checks["75_release_time_closer_to_source_original_control_shape"] = bool(old is not None and new is not None and abs(new-target_release)<abs(old-target_release))
    if name == "angle_p30":
        old = pair["baseline"]["windows"]["full_common"]["metrics"]["speed_mps"]["rmse"]
        new = pair["candidate"]["windows"]["full_common"]["metrics"]["speed_mps"]["rmse"]
        checks["30_full_speed_original_failure_does_not_worsen"] = new <= old+1e-12
    return {"rules":checks,"passes":all(checks.values()),"failed_rules":[k for k,v in checks.items() if not v]}


def run_case(name,item,coeff,arrays,trajectories):
    model = make_candidate(item["config"],coeff)
    full_path = OUT/"full_runs"/(name+".json")
    if full_path.exists(): raise FileExistsError("An actual free screen cannot be silently repeated")
    result = model.run(dataclasses.replace(item["initial"]),item["target_at"],
        end_time_s=item["config"]["performance"]["lifetime_s"])
    dump(full_path,result)
    prediction = grid.fields.prediction_fields(result,model,item["run"]["scenario"])
    observed = copy.deepcopy(item["observed"])
    grid.add_rates(observed,prediction,item["ref"],result)
    pair = pair_metrics(observed,item["baseline_prediction"],prediction)
    verification = verify_saved(make_candidate,model,result,item,coeff)
    peaks = {label:first_peaks(p) for label,p in (("reference",observed),
        ("baseline",item["baseline_prediction"]),("candidate",prediction))}
    old_events = events(item["baseline_full"])
    new_events = events(result)
    primary_gate = compatibility(pair)
    direction_gate = direction_checks(name,pair,peaks,old_events,new_events)
    original_30_gate = None
    if name == "angle_p30":
        geom_path = best.geometry.OUT/"full_runs/jari_259_30.json"
        geom_full = json.loads(geom_path.read_text())
        geom_model = best.geometry.make_model(item["config"],coeff)
        geom_model.initialize(dataclasses.replace(item["initial"]))
        geom_prediction = grid.fields.prediction_fields(geom_full,geom_model,item["run"]["scenario"])
        grid.add_rates(observed,geom_prediction,item["ref"],geom_full)
        geo_pair = pair_metrics(observed,geom_prediction,prediction)
        original_rules = {window+"__"+field:
            geo_pair["candidate"]["windows"][window]["metrics"][field]["rmse"] <=
            1.1*geo_pair["baseline"]["windows"][window]["metrics"][field]["rmse"]+1e-12
            for window in ("early","full_common") for field in best.PRIMARY}
        original_30_gate = {"rules":original_rules,"passes":all(original_rules.values()),
            "failed_rules":[key for key,value in original_rules.items() if not value],
            "scope":"Original30-degree subset only: speed/AoA/load early/full; original0/60 gate not claimed"}
        original_30_gate["fullspeed_vs_original_geometry"] = {
            label:geo_pair[label]["windows"]["full_common"]["metrics"]["speed_mps"] for label in ("baseline","candidate")}
        original_30_gate["original_rule_not_relaxed"] = True
    for label,p in (("reference",observed),("baseline",item["baseline_prediction"]),("candidate0",prediction)):
        for key,value in p.items(): arrays[name+"__"+label+"__"+key] = np.asarray(value)
    for key,value in trace_arrays(result).items(): arrays[name+"__tick__"+key] = value
    trajectories[name] = {"reference":observed,"baseline":item["baseline_prediction"],"candidate0":prediction}
    return {"case_id":name,"signed_angle_deg":item["angle"],"source_path":str(item["source_path"]),
        "baseline_full_path":str(item["baseline_path"]),"candidate_full_path":str(full_path),
        "source_metadata_version_bridge":item["bridge"],"raw_PID":item["config"]["control"]["pid"],
        "same_support_separate_errors":pair,"first_peaks":peaks,
        "actual_free_events":{"baseline":old_events,"candidate":new_events},
        "primary_early_full_and_pitch_no10percent_worse":primary_gate,
        "predeclared_target_directions":direction_gate,"original30_geometry_gate":original_30_gate,
        "candidate_summary":result["summary"],"baseline_summary":item["baseline_full"]["summary"],
        "focused_verification":verification,"terminal_numerical_failure":result["summary"]["termination_event"]=="numerical_failure",
        "passes":primary_gate["passes"] and direction_gate["passes"]}


def main():
    before = protected()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"results.json").exists(): raise FileExistsError("No silent repeat of a declared candidate")
    (OUT/"full_runs").mkdir(exist_ok=True)
    protocol = {"candidate":CANDIDATE,"evidence_grade":"D shared control-interface hypothesis; not game law identified",
        "new_fitted_coefficients":0,"candidate_count":1,"default_changed":False,
        "equations":{"h":"1/48","K":"Ku(V)/(A*Ka)","beta":"hypot(current local incidence)",
            "r":"first_active1; subsequent previous_active_K/current_K", "I0":"clip(r*Iprevious,+-rawIlimit)",
            "Istar":"clip(I0+raw_ki*e*h,+-rawIlimit)","derivative":"dprev+h/(1e-9+h)*((e-eprev)/h-dprev)",
            "natural":"raw_kp*e+Istar+raw_kd*derivative","cap":"min(raw_L,raw_L/K*inverse_sinc(beta)), both axes all-active",
            "held":"gain*clip(natural,+-cap)","I_next":"clip(Istar+sat-natural,+-rawIlimit)",
            "normalized_coordinate":"ZI=(K/L)*I; tracking residual coefficient1 is algebraic projection, not fitted"},
        "necessary_coupled_assumptions":["Integral transport starts at first active tick on each axis, rather than shared high-angle latch",
            "Both axes use the same target/old actuator cap throughout active lifetime",
            "Actual output residual updates provisional I in one tick; it replaces counterfactual-bound freeze",
            "Raw finsAoa still defines L in radians; raw PID acts on G error; these units remain D mappings"],
        "unchanged":"PN N6/full horizontal relative-vector interface; original hit gain/time-to-go, loft and horizontal gate/latch; S110 stage/order; raw PID/four loft fields; frozen five coefficients/drag/finiteforce; one0.4m equivalent station;48Hz4substeps;gain last",
        "screen_order":list(SCREEN),"transfer_only_if_screen_passes":list(TRANSFER),
        "native_only_if_screen_passes":list(best.NATIVE),"maximum_new_free_integrations":12,
        "screen_rejection":"Any0/5/30/75 early/full speed/AoA/load/pitch >10% worse than same-support currentbest; target0/5 earlypitch/unload/firstpeak closer;75peak and output release closer;30fullspeed no worsening",
        "failure_handling":"Full four-case screen is saved once; reject without transfer, retuning or extra candidate",
        "source_in_runtime":False,"needed_input_in_runtime":False,"source_requests":0,
        "source_rate_diagnostics":"quaternion3 and quaternion5 remain separate; raw controls shape only; startup derivative halo excluded only from inverse diagnostics"}
    output = {"status":"predeclared_before_candidate_computation","protocol":protocol,
        "new_actual_free_integrations":0,"source_or_API_requests":0,"fits_or_parameter_scans":0,"cases":{}}
    dump(OUT/"results.json",output)
    coeff,items = inputs()
    arrays,trajectories = {},{}
    for angle in SCREEN:
        name = grid.cid(angle)
        conditional = same_state_diagnostic(make_candidate,items[name],coeff)
        output["new_actual_free_integrations"] += 1
        dump(OUT/"results.json",output)
        try:
            case = run_case(name,items[name],coeff,arrays,trajectories)
        except ValueError as exc:
            if "forward domain" not in str(exc) and "forward reconstruction branch" not in str(exc): raise
            case = {"case_id":name,"passes":False,"status":"declared_forward_domain_exit",
                "actual_error":str(exc),"verified_free_prediction":False}
        case["same_state_conditional_only"] = conditional
        output["cases"][name] = case
        output["status"] = "screen_in_progress"
        np.savez_compressed(OUT/"trajectories.npz",**arrays)
        dump(OUT/"results.json",output)
        print(json.dumps({"actual_free_screen_completed":name,"passes":case["passes"],
            "failed_primary_rules":case.get("primary_early_full_and_pitch_no10percent_worse",{}).get("failed_rules"),
            "failed_direction_rules":case.get("predeclared_target_directions",{}).get("failed_rules")}),flush=True)
    output["screen_passed"] = all(case["passes"] for case in output["cases"].values())
    if output["screen_passed"]:
        raise RuntimeError("Screen passed; the predeclared holdout implementation must now be completed without changing candidate equations")
    if len(trajectories)==4: overlay(trajectories)
    output.update(status="rejected_complete_four_case_screen",candidate_accepted=False,
        new_transfer_or_native_integrations=0,new_baseline_integrations=0,
        protected_size_mtime_unchanged=before==protected(),protected_file_count=len(before),
        all_numeric_arrays_finite=all(np.isfinite(a).all() for a in arrays.values() if np.issubdtype(a.dtype,np.number)),
        model_default_changed=False,maximum_candidate_budget_used=1,
        limitations=["JARI is interim public game-simulator reference, not recovered private game source",
            "Source quaternion derivative/inverse and conditional PID memories are noncausal diagnostics, not runtime or actual game memory",
            "Common original timestamp support retained; any candidate unsupported rows separately listed",
            "No native/transfer claim follows a failed screen; old candidate remains unchanged"])
    if not output["protected_size_mtime_unchanged"] or not output["all_numeric_arrays_finite"]:
        raise AssertionError("Final protected/finite verification failed")
    dump(OUT/"results.json",output)
    print(json.dumps({"status":output["status"],"actual_free":output["new_actual_free_integrations"],
        "candidate_accepted":False,"protected_unchanged":output["protected_size_mtime_unchanged"],"results":str(OUT/"results.json")}),flush=True)


if __name__=="__main__": main()
