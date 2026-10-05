#!/usr/bin/env python3
"""Saved-state control-chain diagnosis for War Thunder/JARI angle transfers.

No forward simulation, fit, API, or runtime candidate. Exactly two diagnostic
signal interpretations are declared below; each is evaluated on fixed states.
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
import check_r771_horizontal_sinc_cap as best
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import add, dot, norm, scale, sub

OUT = ROOT / "outputs/jari_offaxis/bias_discussion/large_angle_release"
GRID = ROOT / "outputs/jari_offaxis/angle_grid"
DEC = best.dec
AUDIT = DEC.audit
IV = AUDIT.inverse
COMPARISON = AUDIT.fix.finite.comparison
ANGLES = (0, 10, 15, 30, 45, 60, 75)
PROBES = ("closing_horizontal_PN", "relative_PN_closing_tgo_hitGain")
POINTS = (.5, 7 / 12, .625, 17 / 24, 5 / 6, 1.)
WINDOWS = (("release_0p5_1", .5, 1.), ("platform", .7, 2.3))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def protected():
    excluded = {Path(__file__).resolve(), ROOT / "scripts/analyze_r771_pitch_reversal.py"}
    other = ROOT / "outputs/jari_offaxis/bias_discussion/pitch_reversal"
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for folder in ("src", "config", "configs", "schemas", "docs", "data", "scripts", "outputs/jari_offaxis")
        for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and ".git" not in p.parts and p not in excluded and OUT not in p.parents and other not in p.parents}


def statistics(v, mask):
    z = np.asarray(v)[mask]
    return {"n": len(z), "RMS": np.sqrt(np.mean(z * z, axis=0)), "mean": np.mean(z, axis=0),
        "minimum": np.min(z, axis=0), "maximum": np.max(z, axis=0)} if len(z) else {"n": 0}


def first_time(t, condition):
    i = np.flatnonzero(condition)
    return float(t[i[0]]) if len(i) else None


def plateau_release(t, controls):
    """A release exists only after this channel actually attained +1."""
    start = first_time(t, controls >= 1 - 1e-7)
    release = None if start is None else first_time(t,
        (t >= max(.5, start) - 1e-8) & (controls < 1 - 1e-7))
    return {"raw_index0_first_plus1_plateau_s": start,
        "raw_index0_first_below1_after_plateau_s": release}


def unchanged_model(model, config, coeff):
    checks = {"config_and_raw_PID_equal": model.config == config,
        "five_coefficients_equal": model.coefficients == best.SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coeff,
        "controller48_substeps4": model.controller_hz == 48 and model.plant_substeps == 4,
        "single_station_0p4m": model.arm == .4,
        "N6_full_relativePN_S110": config["guidance"]["pn_gain"] == 6. and model.relative_blend == 1. and model.request_slew_g_s == 110.,
        "loft_frozen": [model.amplitude_coefficient, model.transition_coefficient] == [.75, 1.3],
        "raw_hitGain_table_equal": config["guidance"]["time_to_hit_gain_table"] == [[10.,1.],[25.,.8],[50.,.5]],
        "H2_mapping_frozen": config["guidance"]["time_to_go_mode"] == "closing_or_relative_speed" and config["guidance"]["time_to_go_relative_speed_weight"] == 1.}
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def inputs():
    grid = json.loads((GRID / "results.json").read_text())
    profiles = json.loads(COMPARISON.SNAPSHOT.read_text())
    coeff = AUDIT.drag.DragCoefficients(**json.loads((AUDIT.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    result = {}
    for angle in ANGLES:
        cid = f"angle_p{angle}"
        receipt = grid["cases"][cid]
        run = json.loads(Path(receipt["source_path"]).read_text())
        ref, _ = COMPARISON.parse_reference(run)
        profile, _ = COMPARISON.profile_for(run, {}, profiles)
        cfg = copy.deepcopy(profile["_model_config"])
        initial, target_at = COMPARISON.free_inputs(run, ref, cfg)
        result[cid] = {"angle": angle, "receipt": receipt, "run": run, "ref": ref,
            "config": cfg, "initial": initial, "target_at": target_at}
    return grid, coeff, result


def signals(state, target, model, f, details):
    """Two alternatives; old half PN is descriptive only, never a third probe."""
    full = tuple(details["pn_mps2"])
    closing = tuple(details.get("original_pn_mps2", full))
    half = add(closing, scale(f.wind_basis.right, .5 * dot(sub(full, closing), f.wind_basis.right)))
    relative = sub(target.position, state.position)
    rel_v = sub(target.velocity, state.velocity)
    distance = norm(relative)
    vc = -dot(relative, rel_v) / distance
    if vc <= 0.:
        raise ValueError("R/Vc hitGain interpretation is undefined outside the declared positive-closing domain")
    tgo_closing = distance / vc
    old_hit = float(details["hit_gain"])
    closing_hit = interpolate_table(tgo_closing, model.config["guidance"]["time_to_hit_gain_table"])
    if old_hit <= 0:
        raise ValueError("Declared positive-gain comparison domain exited")
    changed_hit = scale(full, closing_hit / old_hit)
    vectors = {"current": full, "half_descriptive": half,
        "closing_horizontal_PN": closing, "relative_PN_closing_tgo_hitGain": changed_hit}
    maximum = model.config["guidance"]["maximum_lateral_acceleration_g"] * model.gravity
    bounds = {k: DEC.outer_boundaries(pn, details["loft_g"], f.wind_basis, model.gravity, maximum)
        for k, pn in vectors.items()}
    return vectors, bounds, {"closing_mps": vc, "relative_speed_mps": norm(rel_v),
        "current_tgo_minus_relative_tgo_s": details["tgo_s"] - distance / norm(rel_v),
        "PN_full_minus_closing_nonhorizontal_mps2": norm(sub(sub(full, closing), scale(f.wind_basis.right, dot(sub(full, closing), f.wind_basis.right)))),
        "current_tgo_s": details["tgo_s"], "closing_tgo_s": tgo_closing,
        "current_hitGain": old_hit, "closing_tgo_hitGain": closing_hit,
        "PN_current_horizontal_G": dot(full, f.wind_basis.right) / DEC.G0_MPS2,
        "PN_closing_horizontal_G": dot(closing, f.wind_basis.right) / DEC.G0_MPS2,
        "PN_half_horizontal_G": dot(half, f.wind_basis.right) / DEC.G0_MPS2}


def saved_free(cid, x, coeff):
    receipt = json.loads(Path(x["receipt"]["candidate_full_path"]).read_text())
    model = best.make_model(x["config"], coeff)
    model.initialize(dataclasses.replace(x["initial"]))
    invariants = unchanged_model(model, x["config"], coeff)
    controller = best.HorizontalSincCapControl()
    memory = [(0., 0., 0.), (0., 0., 0.)]
    samples = {round(s["time_s"] * 48): s for s in receipt["samples"]}
    limits = np.radians([x["config"]["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    rows = []
    maxima = {k: 0. for k in ("request_G", "held_rad", "full_memory", "PN_world_mps2", "K", "beta_rad",
        "angular_law_rad_s2", "transport_sum_closure_rad", "P_I_D_closure_rad")}
    for tick in receipt["ticks"]:
        ts = float(tick["time_s"])
        if ts > 3.5 + 1e-8:
            break
        s = samples[round(ts * 48)]
        state = AUDIT.fix.finite.sample_state(s)
        f = model.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        previous_slew = model.slew_horizontal_g
        request, details = model.request(state, x["target_at"](ts), ts, f)
        pn, bounds, stages = signals(state, x["target_at"](ts), model, f, details)
        maxima["request_G"] = max(maxima["request_G"], np.max(abs(np.asarray(request) - tick["request_g"])))
        maxima["PN_world_mps2"] = max(maxima["PN_world_mps2"], np.max(abs(np.asarray(details["pn_mps2"]) - tick["pn_mps2"])))
        maxima["angular_law_rad_s2"] = max(maxima["angular_law_rad_s2"], np.max(abs(np.asarray(f.angular_acceleration) - s["angular_acceleration_rad_s2"])))
        row = {"t": ts, "gain": tick["gain"], **stages, "PN_world_mps2": pn["current"],
            "request_G": request, "feedback_G": f.feedback_g, "loft_G": details["loft_g"],
            "unslewed_request_G": bounds["current"]["unslewed_request_G"],
            "projected_PN_G": DEC.projected(bounds["current"]["after_projection_world_mps2"], f.wind_basis),
            "cap1_active": bounds["current"]["cap1_active"], "cap2_active": bounds["current"]["cap2_active"],
            "previous_S110_yaw_G": previous_slew, "S110_delta_yaw_G": request[1] - previous_slew,
            "held_rad": tick["held_fin_rad"], "memory": tick["pid_memory"]}
        if tick["gain"] > 1e-12:
            controller.current_beta_rad = math.hypot(*f.local_incidence)
            pid = scheduled_pid_values(x["config"]["control"]["pid"], (round(ts * 48) + 1) / 48)
            ds = []
            for axis in (0, 1):
                old = memory[axis]
                result = controller.update_axis(axis, ts, request[axis] - f.feedback_g[axis], old, pid,
                    limits[axis], tick["gain"], model.current_control_k, model.current_force_nonlinear)
                d = copy.deepcopy(controller.diagnostics[axis])
                memory[axis] = result[1]
                maxima["held_rad"] = max(maxima["held_rad"], abs(result[0] - tick["held_fin_rad"][axis]))
                maxima["full_memory"] = max(maxima["full_memory"], max(abs(a-b) for a, b in zip(result[1], tick["pid_memory"][axis])))
                saved = tick["control_runtime_diagnostics"][axis]
                maxima["K"] = max(maxima["K"], abs(d["K_target_per_old_held"] - saved["K_target_per_old_held"]))
                maxima["beta_rad"] = max(maxima["beta_rad"], abs(d["beta_rad"] - saved["beta_rad"]))
                transport = d["transported_integral_old_rad"] - old[0]
                aw = d["next_integral_old_rad"] - d["I_provisional_old_rad"]
                maxima["transport_sum_closure_rad"] = max(maxima["transport_sum_closure_rad"],
                    abs(d["next_integral_old_rad"] - old[0] - transport - d["I_increment_old_rad"] - aw))
                maxima["P_I_D_closure_rad"] = max(maxima["P_I_D_closure_rad"], abs(d["natural_old_rad"] -
                    d["P_old_rad"] - d["I_provisional_old_rad"] - d["D_old_rad"]))
                d["transport_delta_rad"] = transport
                d["AW_retention_delta_rad"] = aw
                ds.append(d)
            row.update({key: [d[key] for d in ds] for key in ("P_old_rad", "I_provisional_old_rad", "D_old_rad", "natural_old_rad",
                "effective_old_held_limit_rad", "original_AW_limit_old_rad", "antiwindup_blocked", "next_integral_old_rad",
                "transport_delta_rad", "AW_retention_delta_rad", "integral_transport_ratio", "K_target_per_old_held", "beta_rad", "nonlinear_latched")})
        rows.append(row)
    full = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    active = [r for r in rows if r["gain"] > 1e-12]
    full.update({"active__" + k: np.asarray([r[k] for r in active]) for k in active[0] if k not in full})
    full["active__t"] = np.asarray([r["t"] for r in active])
    # Reconstructed request agreement also closes all nested PN/slew stages;
    # each two-vector 50G boundary is reported independently below.
    if max(maxima.values()) > 1e-10:
        raise AssertionError((cid, maxima))
    report = {"checks": maxima, "unchanged_model": invariants, "points": [], "first_yaw_clip_s": first_time(full["active__t"],
        abs(full["active__natural_old_rad"][:, 1]) > full["active__effective_old_held_limit_rad"][:, 1] + 1e-13),
        "first_shared_high_latch_s": first_time(full["active__t"], full["active__nonlinear_latched"][:, 1]),
        "first_yaw_AW_s": first_time(full["active__t"], full["active__antiwindup_blocked"][:, 1]),
        "vector_caps_active_ticks": {k: int(np.sum(full[k])) for k in ("cap1_active", "cap2_active")}}
    for ts in POINTS:
        i = int(np.argmin(abs(full["t"]-ts))); ai = int(np.argmin(abs(full["active__t"]-ts)))
        report["points"].append({**{k: full[k][i] for k in ("t", "current_tgo_s", "closing_tgo_s", "current_hitGain", "closing_tgo_hitGain",
            "PN_current_horizontal_G", "PN_closing_horizontal_G", "PN_half_horizontal_G", "request_G", "unslewed_request_G", "feedback_G", "held_rad", "S110_delta_yaw_G")},
            **{k: full["active__"+k][ai] for k in ("P_old_rad", "I_provisional_old_rad", "D_old_rad", "effective_old_held_limit_rad", "original_AW_limit_old_rad", "antiwindup_blocked")}})
    return full, report


def source_conditional(cid, x, coeff):
    """Fixed quaternion states with explicit zero initial conditional PID memory.

    Body rates from quaternion5 define the common state. Quaternion3/5 angular
    acceleration estimates are compared at that same state. Raw-omega inverses
    use their own exported-rate state and are separate sensitivity diagnostics.
    """
    ref = x["ref"]; st = ref["t"]
    q = IV.gap.continuous_quaternions(ref["quaternion_xyzw"][:, [3, 0, 1, 2]])
    body = {w: IV.gap.body_rates(q, st, w) for w in (3, 5)}
    rates = {w: np.column_stack((body[w][:, 2], -body[w][:, 1])) for w in (3, 5)}
    qdot = {w: IV.gap.derivative(rates[w], st, w) for w in (3, 5)}
    raw_rates = np.column_stack((-ref["omega_xyz_export_units"][:, 2], ref["omega_xyz_export_units"][:, 1]))
    raw_qdot = {w: IV.gap.derivative(raw_rates, st, w) for w in (3, 5)}
    t = np.arange(169) / 48
    model = best.make_model(x["config"], coeff); model.initialize(dataclasses.replace(x["initial"]))
    invariants = unchanged_model(model, x["config"], coeff)
    raw_model = best.make_model(x["config"], coeff); raw_model.initialize(dataclasses.replace(x["initial"]))
    ctrl = {name: best.HorizontalSincCapControl() for name in ("current", *PROBES)}
    memories = {name: [(0.,0.,0.), (0.,0.,0.)] for name in ctrl}
    alt_slew = {name: 0. for name in PROBES}
    limits = np.radians([x["config"]["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    rows = []; max_inverse = 0.; max_qdot = 0.; max_current_request = 0.
    maximum_G = x["config"]["guidance"]["maximum_lateral_acceleration_g"] * model.gravity / DEC.G0_MPS2
    for ts in t:
        state_raw = AUDIT.fix.source_state_at(ref, float(ts))
        r5 = np.asarray([np.interp(ts, st, rates[5][:, j]) for j in (0, 1)])
        state = dataclasses.replace(state_raw, pitch_rate=float(r5[0]), yaw_rate=float(r5[1]))
        f = model.evaluate(state, float(ts), (0.,0.))
        previous_slew = model.slew_horizontal_g
        request, details = model.request(state, x["target_at"](float(ts)), float(ts), f)
        vectors, boundaries, stages = signals(state, x["target_at"](float(ts)), model, f, details)
        gain = interpolate_table(float(ts + 1/48), x["config"]["guidance"]["flight_time_gain_table"])
        pid = scheduled_pid_values(x["config"]["control"]["pid"], float(ts + 1/48))
        beta = math.hypot(*f.local_incidence)
        ku = model.coefficients.control_gain * (1 + model.coefficients.speed_slope * (norm(sub(state.velocity, model.wind)) / model.speed0 - 1))
        current_reconstructed, _, _ = DEC.same_previous_slew(boundaries["current"]["unslewed_request_G"], previous_slew, gain,
            abs(details["los_rate_rad_s"][1]), maximum_G)
        max_current_request = max(max_current_request, np.max(abs(np.asarray(current_reconstructed) - request)))
        row = {"t": ts, "gain": gain, **stages, "beta": beta, "K": model.current_control_k,
            "request_G": request, "feedback_G": f.feedback_g, "loft_G": details["loft_g"],
            "source_controls_raw": [np.interp(ts, st, np.asarray([s["controls"] for s in x["run"]["samples"]])[:, j]) for j in (0,1)],
            "cap1_active": boundaries["current"]["cap1_active"], "cap2_active": boundaries["current"]["cap2_active"],
            "rates_quat5_rad_s": r5}
        needed = {}
        raw_f = raw_model.evaluate(state_raw, float(ts), (0.,0.))
        for width in (3,5):
            observed = np.asarray([np.interp(ts, st, qdot[width][:, j]) for j in (0,1)])
            need = (observed - np.asarray(f.angular_acceleration)) / ku
            needed[width] = need
            got = model.evaluate(state, float(ts), tuple(need))
            max_qdot = max(max_qdot, np.max(abs(np.asarray(got.angular_acceleration) - observed)))
            raw_observed = np.asarray([np.interp(ts, st, raw_qdot[width][:, j]) for j in (0,1)])
            row[f"needed_quat{width}_held_rad"] = need
            row[f"needed_export{width}_held_rad"] = (raw_observed - np.asarray(raw_f.angular_acceleration)) / ku
        if gain > 1e-12:
            baseline_prior = copy.deepcopy(ctrl["current"])
            baseline_memory = copy.deepcopy(memories["current"])
            baseline_prior.current_beta_rad = beta
            for name in ctrl:
                ctrl[name].current_beta_rad = beta
                if name == "current":
                    req = request
                else:
                    req, _, _ = DEC.same_previous_slew(boundaries[name]["unslewed_request_G"], alt_slew[name], gain,
                        abs(details["los_rate_rad_s"][1]), maximum_G)
                    alt_slew[name] = req[1]
                output = []; diagnostics = []
                for axis in (0,1):
                    value, memories[name][axis] = ctrl[name].update_axis(axis,float(ts),req[axis]-f.feedback_g[axis],memories[name][axis],pid,
                        limits[axis],gain,model.current_control_k,model.current_force_nonlinear)
                    output.append(value); diagnostics.append(copy.deepcopy(ctrl[name].diagnostics[axis]))
                row[name+"__request_G"] = req; row[name+"__held_rad"] = output
                row[name+"__P_I_D_rad"] = [[d[k] for k in ("P_old_rad","I_provisional_old_rad","D_old_rad")] for d in diagnostics]
                row[name+"__natural_rad"] = [d["natural_old_rad"] for d in diagnostics]
                row[name+"__memory"] = memories[name]
                row[name+"__output_clipped"] = [abs(d["natural_old_rad"]) > d["effective_old_held_limit_rad"]+1e-13 for d in diagnostics]
                row[name+"__AW"] = [d["antiwindup_blocked"] for d in diagnostics]
                row[name+"__cap_rad"] = [d["effective_old_held_limit_rad"] for d in diagnostics]
                row[name+"__AW_limit_rad"] = [d["original_AW_limit_old_rad"] for d in diagnostics]
                if name != "current":
                    same_request, _, _ = DEC.same_previous_slew(boundaries[name]["unslewed_request_G"], previous_slew, gain,
                        abs(details["los_rate_rad_s"][1]), maximum_G)
                    one = copy.deepcopy(baseline_prior); same_output=[]
                    for axis in (0,1):
                        value,_=one.update_axis(axis,float(ts),same_request[axis]-f.feedback_g[axis],baseline_memory[axis],pid,
                            limits[axis],gain,model.current_control_k,model.current_force_nonlinear)
                        same_output.append(value)
                    row[name+"__same_memory_request_G"] = same_request
                    row[name+"__same_memory_held_rad"] = same_output
            ds=ctrl["current"].diagnostics
            for width,need in needed.items():
                inverse_req=[]; status=[]
                for axis in (0,1):
                    old=baseline_memory[axis]
                    transported=(ds[axis]["transported_integral_old_rad"],old[1],old[2])
                    error,code=AUDIT.error_for_output(need[axis],transported,pid,gain,ds[axis]["effective_old_held_limit_rad"])
                    status.append(code)
                    inverse_req.append(float(error+f.feedback_g[axis]) if code>0 else None)
                    if code>0:
                        got=gain*AUDIT.clip(AUDIT.natural_pid(error,transported,pid),ds[axis]["effective_old_held_limit_rad"])
                        max_inverse=max(max_inverse,abs(got-need[axis]))
                row[f"quat{width}_needed_request_given_current_memory_G"] = inverse_req
                row[f"quat{width}_inverse_status"] = status
        rows.append(row)
    a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    active=[r for r in rows if r["gain"]>1e-12]
    a.update({k:np.asarray([r[k] for r in active], dtype=float if "needed_request_given" in k else None) for k in active[0] if k not in a})
    a["active_t"]=np.asarray([r["t"] for r in active])
    active_mask=a["gain"]>1e-12
    # Quat5 rate + quat5 derivative has a four-source-row support halo.
    halo=(t>=14/48-4/24-1e-10)&(t<=14/48+4/24+1e-10)
    a["two_derivative_gain_halo"]=halo
    ar={k:np.asarray([r[k] for r in active]) for k in active[0]}
    report={"unchanged_model":invariants,"checks":{"inverse_PID_closure_rad":max_inverse,"needed_angular_closure_rad_s2":max_qdot,
        "current_request_stage_closure_G":max_current_request},"points":[],"windows":{}}
    report["creation_tgo_hitGain"]={k:a[k][0] for k in ("current_tgo_s","closing_tgo_s","current_hitGain","closing_tgo_hitGain","closing_mps","relative_speed_mps")}
    report["geometry_mapping_checks"]={"minimum_relative_speed_minus_closing_mps":float(np.min(a["relative_speed_mps"]-a["closing_mps"])),
        "maximum_current_tgo_minus_R_over_relative_abs_s":float(np.max(abs(a["current_tgo_minus_relative_tgo_s"]))),
        "maximum_full_vs_original_nonhorizontal_difference_mps2":float(np.max(a["PN_full_minus_closing_nonhorizontal_mps2"]))}
    for ts in POINTS:
        i=int(round(ts*48)); ai=int(np.argmin(abs(a["active_t"]-ts)))
        row={k:a[k][i] for k in ("t","closing_mps","relative_speed_mps","current_tgo_s","closing_tgo_s","current_hitGain","closing_tgo_hitGain",
            "PN_current_horizontal_G","PN_closing_horizontal_G","PN_half_horizontal_G","request_G","feedback_G","beta","K","loft_G","source_controls_raw",
            "needed_quat3_held_rad","needed_quat5_held_rad","needed_export3_held_rad","needed_export5_held_rad")}
        row.update({k:ar[k][ai] for k in ar if k not in row and k not in ("t","gain","rates_quat5_rad_s","cap1_active","cap2_active")})
        report["points"].append(row)
    for w,lo,hi in WINDOWS:
        mask=(a["active_t"]>=lo-1e-8)&(a["active_t"]<=hi+1e-8)
        target_need=np.column_stack([np.interp(a["active_t"],t,a["needed_quat5_held_rad"][:,j]) for j in (0,1)])
        report["windows"][w]={"n":int(mask.sum()),"source_state_output_residual_rad":{name:statistics(ar[name+"__held_rad"]-target_need,mask)
            for name in ctrl},"yaw_clip_ticks":{name:int(np.sum(ar[name+"__output_clipped"][mask,1])) for name in ctrl},
            "yaw_AW_ticks":{name:int(np.sum(ar[name+"__AW"][mask,1])) for name in ctrl},
            "inverse_infeasible_ticks_quat3_5":[int(np.sum(ar[f"quat{n}_inverse_status"][mask,1]==-1)) for n in (3,5)]}
    report["source_release_events"]=plateau_release(st, np.asarray([s["controls"][0] for s in x["run"]["samples"]]))
    for name in ctrl:
        c=ar[name+"__output_clipped"][:,1]
        after=a["active_t"]>=.5-1e-8
        report["source_release_events"][name+"__first_unclipped_after0p5_s"]=first_time(a["active_t"],after&~c)
    if max(max_inverse,max_qdot,max_current_request)>1e-10:
        raise AssertionError((cid,report["checks"]))
    return a, report


def plot(sources):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(4,2,figsize=(13,12),sharex=True)
    colors={"current":"#7a397e","closing_horizontal_PN":"#228b73","relative_PN_closing_tgo_hitGain":"#d08b20"}
    for i,angle in enumerate((30,45,60,75)):
        a=sources[f"angle_p{angle}"];t=a["active_t"]
        for name,color in colors.items():
            ax[i,0].plot(t,a[name+"__request_G"][:,1],color=color,label=name)
            cap=a["current__cap_rad"][:,1]
            ax[i,1].plot(t,a[name+"__held_rad"][:,1]/cap,color=color,label=name+" own conditional history")
        for width in (3,5):
            need=np.interp(t,a["t"],a[f"needed_quat{width}_held_rad"][:,1])
            ax[i,1].plot(t,need/cap,color="#456e9b",ls="--" if width==3 else ":",label=f"source quaternion{width} needed / current cap")
        ax[i,1].axhline(1,color="#777",lw=.7)
        for j in (0,1):
            ax[i,j].axvspan(.291667,.458333,color="#999",alpha=.12)
            ax[i,j].set_xlim(.29,1.05);ax[i,j].grid(alpha=.15)
            ax[i,j].set_ylabel(f"{angle}deg " + ("yaw request G" if j==0 else "held / current cap"))
        if i==0:
            ax[i,0].legend(fontsize=7);ax[i,1].legend(fontsize=6)
    for a in ax[-1]:a.set_xlabel("Original time (s)")
    fig.suptitle("Fixed source states: two declared signal interpretations, unchanged S110/PID/cap\nConditional zero-initial-memory histories only; no free trajectory or native-controller identification",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.955));fig.savefig(OUT/"source_release_signals.png",dpi=150);plt.close(fig)


def main():
    before=protected();grid,coeff,xs=inputs();OUT.mkdir(parents=True,exist_ok=True)
    saved={};actuals={};sources={};reports={}
    for cid,x in xs.items():
        actual,ar=saved_free(cid,x,coeff);source,sr=source_conditional(cid,x,coeff)
        actuals[cid]=actual;sources[cid]=source;reports[cid]={"actual_saved_free_chain":ar,"fixed_source_conditional":sr}
        saved.update({cid+"__actual__"+k:v for k,v in actual.items()})
        saved.update({cid+"__source__"+k:v for k,v in source.items()})
        print(json.dumps(serial({"case":cid,"actual_checks":ar["checks"],"source_checks":sr["checks"],"release":sr["source_release_events"]})),flush=True)
    after=protected();finite=all(np.isfinite(v).all() for k,v in saved.items() if "needed_request_given" not in k)
    if before!=after or not finite:raise AssertionError((before==after,finite))
    np.savez_compressed(OUT/"diagnostics.npz",**saved)
    plot(sources)
    limits=["War Thunder public-game reference only; no private source recovered.",
        "Exactly two coefficient-free signal interpretations, not free-flight candidates. Half PN is descriptive only.",
        "Current and alternatives use identical source states and plant feedback; own recursive conditional histories and current-memory one-step probes are labeled separately.",
        "Common conditional state uses quaternion5 body rates; quaternion3/5 angular acceleration estimates vary at that same state. Raw export omega inverse is separate sensitivity.",
        "Source quaternion estimates use future samples. Double-derivative gain halo through .458333s is explicit; endpoints are not evidence for runtime anticipation.",
        "Conditional PID memory starts at zero and is not observed native memory. At/above cap inverse request is nonunique/infeasible, not a missing finite command.",
        "Source controls index0 is yaw-associated by mirror; raw units and meaning remain unidentified, never converted into fin radians.",
        "Alternative R/Vc hitGain uses existing raw table; field-time argument is a D-grade mapping hypothesis, no new parameter.",
        "No change to S110 ordering, AW, D, high latch, loft function, plant, five coefficients, drag, raw PID, one .4m surface, default, 48Hz, or four substeps.",
        "PN and controller signal differences do not establish improved free trajectories or identify native game control."]
    same_yaw=np.stack([actuals[f"angle_p{a}"]["request_G"][:,1] for a in (45,60,75)])
    same_t=actuals["angle_p45"]["t"]
    collapsed=(np.ptp(same_yaw,axis=0)<1e-12)&(same_t>=14/48-1e-8)&(same_t<=.7)
    result={"task":"Large-angle War Thunder request/cap release diagnosis","predeclared_interpretations":PROBES,
        "field_mapping_provenance":{"raw_table_path":str(ROOT/"data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"),
            "raw_table_lines":[170,181],"raw_table":[[10.,1.],[25.,.8],[50.,.5]],
            "H2_default_path":str(ROOT/"src/aim120_model/profile_adapter.py"),"H2_default_lines":[88,89],"H2_assumption_line":516,
            "time_function_path":str(ROOT/"src/aim120_model/guidance.py"),"max_divisor_line":146,
            "interpretation_grade":"D: neither time argument nor native control memory is recovered source"},
        "actual_45_60_75_identical_horizontal_S110_requests_s":same_t[collapsed],
        "cases":reports,"verification":{"protected_previous_size_mtime_equal":before==after,"numeric_arrays_finite_except_explicit_unidentified_requests":finite,
        "new_API_calls":0,"new_free_integrations":0,"new_fits_or_coefficients":0,"runtime_changed":False},"limits":limits}
    (OUT/"results.json").write_text(json.dumps(serial(result),indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":main()
