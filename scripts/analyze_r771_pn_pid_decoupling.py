#!/usr/bin/env python3
"""Conditional PN/PID separation for cached War Thunder game references.

Predeclared questions: outer request boundaries; PID provisional output versus
saved memory; transport/error/clipping/antiwindup memory increments; one-step
feedback and horizontal-PN counterfactuals from identical baseline memories.
No plant integration, fit, network request, runtime change, or new coefficient.
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
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table, pn_acceleration
from aim120_model.math3d import add, clamp_norm, cross, dot, norm, scale, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/pn_pid_decoupling"
CASES = audit.CASES
WINDOWS = audit.WINDOWS
DT = 1 / 48
MEMORY_TERMS = ("transport_delta_rad", "transport_clip_delta_rad",
    "error_increment_rad", "update_clip_delta_rad", "antiwindup_delta_rad")


def projected(vector, basis):
    return (dot(vector, basis.up) / G0_MPS2, dot(vector, basis.right) / G0_MPS2)


def outer_boundaries(pn, loft_g, basis, gravity, maximum):
    """Preserve stage order: projection, vector cap, gravity, vector cap."""
    loft = (0., loft_g * gravity, 0.)
    before_projection = add(pn, loft)
    after_projection = sub(before_projection, scale(basis.forward, dot(before_projection, basis.forward)))
    cap1 = clamp_norm(after_projection, maximum)
    before_cap2 = add(cap1, (0., gravity, 0.))
    cap2 = clamp_norm(before_cap2, maximum)
    return {"loft_world_mps2": loft, "before_projection_world_mps2": before_projection,
        "after_projection_world_mps2": after_projection, "after_cap1_world_mps2": cap1,
        "before_cap2_world_mps2": before_cap2, "after_cap2_world_mps2": cap2,
        "after_projection_G": projected(after_projection, basis),
        "after_cap1_G": projected(cap1, basis), "before_cap2_G": projected(before_cap2, basis),
        "unslewed_request_G": projected(cap2, basis),
        "cap1_active": norm(after_projection) > maximum,
        "cap2_active": norm(before_cap2) > maximum}


def same_previous_slew(request, previous_yaw, gain, horizontal_rate, maximum_G):
    """Exactly one existing S110 update, without a new recursive slew history."""
    if horizontal_rate == 0.:
        return request, request, False
    yaw = previous_yaw + audit.clip(request[1] - previous_yaw, 110. * DT) if gain > 1e-12 else 0.
    before_cap = (request[0], yaw)
    length = math.hypot(*before_cap)
    final = tuple(x * maximum_G / length for x in before_cap) if length > maximum_G else before_cap
    return final, before_cap, length > maximum_G


def pid_terms(controller, axis, memory, error, pid, gain, k, nonlinear, limit):
    """Output I is provisional; saved I is after antiwindup retention."""
    old_i, old_error, old_d = memory
    high = controller.nonlinear_latched or nonlinear
    previous_k = controller.previous_high_k[axis]
    ratio = 1. if not high or previous_k is None else previous_k / k
    transport_raw = old_i * ratio
    transported = audit.clip(transport_raw, pid["integral_limit"]) if high else old_i
    increment = pid["i"] * error * DT
    provisional = audit.clip(transported + increment, pid["integral_limit"])
    derivative = old_d + DT / (1e-9 + DT) * ((error - old_error) / DT - old_d)
    p = pid["p"] * error; d = pid["d"] * derivative
    prelimit = p + provisional + d
    effective_limit = min(limit, limit / k) if high else limit
    postlimit = audit.clip(prelimit, effective_limit)
    blocked = abs(postlimit) >= .99 * effective_limit and error * postlimit > 0
    next_i = transported if blocked else provisional
    return {"P_rad": p, "I_provisional_rad": provisional, "D_rad": d,
        "prelimit_rad": prelimit, "postlimit_rad": postlimit, "held_rad": gain * postlimit,
        "previous_integral_rad": old_i, "transport_ratio": ratio,
        "transported_integral_rad": transported, "next_integral_rad": next_i,
        "transport_delta_rad": transport_raw - old_i,
        "transport_clip_delta_rad": transported - transport_raw,
        "error_increment_rad": increment, "update_clip_delta_rad": provisional - transported - increment,
        "antiwindup_delta_rad": next_i - provisional, "antiwindup_blocked": blocked,
        "effective_limit_rad": effective_limit, "derivative_G_s": derivative,
        "high_mode": high}


def scalar_stats(values, mask):
    v = np.asarray(values)[mask]
    if not len(v):
        return None
    return {"ticks": int(len(v)), "mean": float(v.mean()),
        "RMS": audit.inverse.rms(v), "range": [float(v.min()), float(v.max())]}


def masked_error(output, needed, k, limits, mask):
    delta = k[:, None] * (output - needed) / limits
    return {axis: scalar_stats(delta[:, j], mask[:, j]) for j, axis in enumerate(("pitch", "yaw"))}


def replay_case(cid, item, cache, coefficients):
    cfg = item["config"]; ref = item["ref"]
    cached = {key[len(cid) + 2:]: cache[key] for key in cache.files if key.startswith(cid + "__")}
    t = cached["t"]; gains = cached["flight_gain"]
    limits = np.radians([cfg["aerodynamics"][key] for key in (
        "horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    model = audit.fix.make_model(cfg, coefficients)
    model.initialize(dataclasses.replace(item["initial"]))
    controller = R771GameControl(); memories = [(0., 0., 0.), (0., 0., 0.)]
    records = []; terms = []; counterfactuals = {"source3": [], "source5": [], "original_horizontal_PN": []}
    cf_memories = {key: [] for key in counterfactuals}
    next_memories = []; prev_memories = []
    for i, ts in enumerate(t):
        state = audit.fix.source_state_at(ref, float(ts)); target = item["target_at"](float(ts))
        f = model.evaluate(state, float(ts), (0., 0.))
        previous_slew = model.slew_horizontal_g
        request, details = model.request(state, target, float(ts), f)
        pure, _, los = pn_acceleration(sub(target.position, state.position),
            sub(target.velocity, state.velocity), state.velocity, float(cfg["guidance"]["pn_gain"]))
        original_pn = scale(pure, details["hit_gain"])
        alternative = scale(cross(details["los_rate_rad_s"], scale(sub(target.velocity, state.velocity), -1.)),
            float(cfg["guidance"]["pn_gain"]) * details["hit_gain"])
        half_delta = scale(f.wind_basis.right, .5 * (dot(alternative, f.wind_basis.right) - dot(original_pn, f.wind_basis.right)))
        mixed = add(original_pn, half_delta)
        maximum = float(cfg["guidance"]["maximum_lateral_acceleration_g"]) * model.gravity
        boundaries = outer_boundaries(mixed, details["loft_g"], f.wind_basis, model.gravity, maximum)
        reconstructed, before_slew_cap, cap3 = same_previous_slew(boundaries["unslewed_request_G"],
            previous_slew, gains[i], abs(los[1]), maximum / G0_MPS2)
        # Only the wind-right half increment changes; loft gate and history stay identical.
        cf_boundaries = outer_boundaries(original_pn, details["loft_g"], f.wind_basis, model.gravity, maximum)
        cf_request, cf_slew_pre_cap, cf_cap3 = same_previous_slew(cf_boundaries["unslewed_request_G"],
            previous_slew, gains[i], abs(los[1]), maximum / G0_MPS2)
        row = {**boundaries, "pure_PN_world_mps2": pure, "hitGain_PN_world_mps2": original_pn,
            "half_increment_world_mps2": half_delta, "mixed_PN_world_mps2": mixed,
            "pure_PN_G": projected(pure, f.wind_basis), "hitGain_PN_G": projected(original_pn, f.wind_basis),
            "half_increment_G": projected(half_delta, f.wind_basis), "mixed_PN_G": projected(mixed, f.wind_basis),
            "loft_projected_G": projected(boundaries["loft_world_mps2"], f.wind_basis),
            "gravity_projected_G": projected((0., model.gravity, 0.), f.wind_basis),
            "loft_original_G": details.get("original_loft_g", details["loft_g"]),
            "loft_gate": details.get("horizontal_loft_gate", 1.), "loft_actual_G": details["loft_g"],
            "hitGain": details["hit_gain"], "previous_slew_yaw_G": previous_slew,
            "slew_before_joint_cap_G": before_slew_cap, "final_request_G": request,
            "reconstructed_request_G": reconstructed, "slew_delta_G": np.asarray(before_slew_cap) - boundaries["unslewed_request_G"],
            "slew_binding": abs(before_slew_cap[1] - boundaries["unslewed_request_G"][1]) > 1e-10,
            "cap3_active": cap3, "plant_feedback_G": f.feedback_g, "error_G": np.asarray(request) - f.feedback_g,
            "K": model.current_control_k, "nonlinear_force_active": model.current_force_nonlinear,
            "cf_original_horizontal_PN_unslewed_G": cf_boundaries["unslewed_request_G"],
            "cf_original_horizontal_PN_request_G": cf_request,
            "cf_original_horizontal_PN_cap1": cf_boundaries["cap1_active"],
            "cf_original_horizontal_PN_cap2": cf_boundaries["cap2_active"], "cf_original_horizontal_PN_cap3": cf_cap3,
            "PN_detail_closure_mps2": np.asarray(details["pn_mps2"]) - mixed,
            "original_PN_detail_closure_mps2": np.asarray(details.get("original_pn_mps2", original_pn)) - original_pn,
            "unslewed_detail_closure_G": np.asarray(details["slew_original_request_g"]) - boundaries["unslewed_request_G"]}
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + DT)
        tick_terms = []; tick_cf = {key: [] for key in counterfactuals}; tick_cf_mem = {key: [] for key in counterfactuals}
        previous = copy.deepcopy(memories)
        for axis in range(2):
            if gains[i] <= 1e-12:
                tick_terms.append({key: False if key in ("antiwindup_blocked", "high_mode") else
                    (1. if key == "transport_ratio" else 0.) for key in pid_terms(controller, axis, memories[axis], 0., pid,
                        0., model.current_control_k, False, limits[axis])})
                for key in counterfactuals:
                    tick_cf[key].append(0.); tick_cf_mem[key].append(memories[axis])
                continue
            term = pid_terms(controller, axis, memories[axis], float(row["error_G"][axis]), pid,
                gains[i], model.current_control_k, model.current_force_nonlinear, limits[axis])
            # Every counterfactual starts from the baseline's same previous memory,
            # including previous K/latch. Its returned memory is discarded.
            for key in counterfactuals:
                fb = cached[("3" if key == "source3" else "5") + "__source_feedback_G"][i] if key.startswith("source") else f.feedback_g
                req = cf_request if key == "original_horizontal_PN" else request
                trial = copy.deepcopy(controller)
                held, next_memory = trial.update_axis(axis, float(ts), float(req[axis] - fb[axis]), memories[axis],
                    pid, limits[axis], gains[i], model.current_control_k, model.current_force_nonlinear)
                tick_cf[key].append(held); tick_cf_mem[key].append(next_memory)
            held, memories[axis] = controller.update_axis(axis, float(ts), float(row["error_G"][axis]), memories[axis],
                pid, limits[axis], gains[i], model.current_control_k, model.current_force_nonlinear)
            term["runtime_held_rad"] = held; term["runtime_next_integral_rad"] = memories[axis][0]
            tick_terms.append(term)
        for term in tick_terms:
            term.setdefault("runtime_held_rad", 0.); term.setdefault("runtime_next_integral_rad", 0.)
        records.append(row); terms.append(tick_terms); prev_memories.append(previous); next_memories.append(copy.deepcopy(memories))
        for key in counterfactuals:
            counterfactuals[key].append(tick_cf[key]); cf_memories[key].append(tick_cf_mem[key])
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    arrays.update({"pid__" + key: np.asarray([[row[axis][key] for axis in range(2)] for row in terms]) for key in terms[0][0]})
    arrays.update({"t": t, "flight_gain": gains, "old_held_limits_rad": limits,
        "previous_current_memory": np.asarray(prev_memories), "next_current_memory": np.asarray(next_memories)})
    for key, values in counterfactuals.items():
        arrays["cf__" + key + "__held_rad"] = np.asarray(values)
        arrays["cf__" + key + "__discarded_next_memory"] = np.asarray(cf_memories[key])
    halo = (t >= 14 / 48 - 2 / 24) & (t <= 14 / 48 + 2 / 24)
    arrays["gain_transition_derivative_halo"] = halo
    active = gains > 1e-12
    checks = {
        "pure_hitGain_PN_detail_max_abs_mps2": float(np.max(abs(arrays["original_PN_detail_closure_mps2"]))),
        "mixed_PN_detail_max_abs_mps2": float(np.max(abs(arrays["PN_detail_closure_mps2"]))),
        "unslewed_request_closure_max_abs_G": float(np.max(abs(arrays["unslewed_detail_closure_G"]))),
        "final_request_closure_max_abs_G": float(np.max(abs(arrays["final_request_G"] - arrays["reconstructed_request_G"]))),
        "saved_request_max_abs_G": float(np.max(abs(arrays["final_request_G"] - cached["available_S110_G"]))),
        "saved_feedback_max_abs_G": float(np.max(abs(arrays["plant_feedback_G"] - cached["plant_feedback_G"]))),
        "PID_output_closure_max_abs_rad": float(np.max(abs(arrays["pid__held_rad"] - arrays["pid__runtime_held_rad"]))),
        "PID_nextI_closure_max_abs_rad": float(np.max(abs(arrays["pid__next_integral_rad"] - arrays["pid__runtime_next_integral_rad"]))),
        "saved_held_max_abs_rad": float(np.max(abs(arrays["pid__held_rad"] - cached["5__current_plant__forward_held_rad"]))),
        "saved_memory_max_abs": float(np.max(abs(arrays["next_current_memory"] - cached["5__current_plant__forward_memory"]))) }
    increments = sum(arrays["pid__" + key] for key in MEMORY_TERMS)
    checks["I_increment_tick_closure_max_abs_rad"] = float(np.max(abs(increments - (arrays["pid__next_integral_rad"] - arrays["pid__previous_integral_rad"]))))
    checks["I_increment_cumulative_closure_max_abs_rad"] = float(np.max(abs(np.cumsum(increments, axis=0) - arrays["pid__next_integral_rad"])))
    arrays["pid__total_memory_delta_rad"] = increments
    # K*I is the equivalent target coordinate. Old-coordinate transport is not
    # a physical memory injection: after entry its change cancels the K change.
    k_now = arrays["K"][:, None]
    k_prev = np.concatenate((arrays["K"][:1], arrays["K"][:-1]))[:, None]
    prev_i = arrays["pid__previous_integral_rad"]
    transported_raw = prev_i + arrays["pid__transport_delta_rad"]
    arrays["target_I_before_normalized"] = k_prev * prev_i / limits
    arrays["target_I_after_normalized"] = k_now * arrays["pid__next_integral_rad"] / limits
    arrays["target_I_K_coordinate_delta_normalized"] = (k_now - k_prev) * prev_i / limits
    for term in MEMORY_TERMS:
        arrays["target_I__" + term.replace("_rad", "_normalized")] = k_now * arrays["pid__" + term] / limits
    target_delta = arrays["target_I_K_coordinate_delta_normalized"] + sum(
        arrays["target_I__" + term.replace("_rad", "_normalized")] for term in MEMORY_TERMS)
    arrays["target_I_total_delta_normalized"] = target_delta
    arrays["target_I_K_and_transport_delta_normalized"] = (k_now * transported_raw - k_prev * prev_i) / limits
    # Reference K is current K on entry; thereafter it is the stored previous K.
    transport_reference_k = k_now * arrays["pid__transport_ratio"]
    arrays["target_I_pure_transport_closure_rad"] = k_now * transported_raw - transport_reference_k * prev_i
    checks["target_I_increment_tick_closure_max_abs_normalized"] = float(np.max(abs(target_delta -
        (arrays["target_I_after_normalized"] - arrays["target_I_before_normalized"]))))
    checks["target_I_increment_cumulative_closure_max_abs_normalized"] = float(np.max(abs(
        np.cumsum(target_delta, axis=0) - arrays["target_I_after_normalized"])))
    checks["target_I_pure_transport_preservation_max_abs_rad"] = float(np.max(abs(arrays["target_I_pure_transport_closure_rad"])))
    report = {"axis_order": ["pitch", "yaw"], "original_PID_entire_dict": cfg["control"]["pid"],
        "original_flight_time_gain_table": cfg["guidance"]["flight_time_gain_table"],
        "original_PN_gain": cfg["guidance"]["pn_gain"], "checks": checks, "windows": {}, "methods": {}}
    for name, lo, hi in WINDOWS:
        mask = audit.inverse.window_mask(t, lo, hi) & active
        report["windows"][name] = {"support_s": [lo, hi], "active_ticks": int(mask.sum()),
            "caps_active_ticks_1_2_3": [int(np.sum(arrays[key][mask])) for key in ("cap1_active", "cap2_active", "cap3_active")],
            "S110_binding_ticks": int(arrays["slew_binding"][mask].sum()), "axes": {}}
        for axis, label in enumerate(("pitch", "yaw")):
            entry = {"outer_stage_G": {key: scalar_stats(arrays[key][:, axis], mask) for key in (
                "pure_PN_G", "hitGain_PN_G", "half_increment_G", "loft_projected_G", "gravity_projected_G",
                "after_projection_G", "after_cap1_G", "before_cap2_G", "unslewed_request_G", "slew_delta_G", "final_request_G")},
                "feedback_G": scalar_stats(arrays["plant_feedback_G"][:, axis], mask),
                "error_G": scalar_stats(arrays["error_G"][:, axis], mask),
                "PID_output_terms_deg": {key: scalar_stats(np.degrees(arrays["pid__" + key][:, axis]), mask) for key in (
                    "P_rad", "I_provisional_rad", "D_rad", "prelimit_rad", "postlimit_rad", "held_rad")},
                "I_memory_increment_sum_deg": {key: float(np.degrees(arrays["pid__" + key][mask, axis].sum())) for key in MEMORY_TERMS},
                "I_memory_net_change_deg": float(np.degrees(increments[mask, axis].sum())),
                "target_I_normalized": {"window_start_before_tick": float(arrays["target_I_before_normalized"][mask, axis][0]),
                    "window_end_after_tick": float(arrays["target_I_after_normalized"][mask, axis][-1]),
                    "net_change": float(target_delta[mask, axis].sum()),
                    "K_coordinate_plus_transport_change": float(arrays["target_I_K_and_transport_delta_normalized"][mask, axis].sum()),
                    "error_increment": float(arrays["target_I__error_increment_normalized"][mask, axis].sum()),
                    "antiwindup_retention_delta": float(arrays["target_I__antiwindup_delta_normalized"][mask, axis].sum())},
                "antiwindup_ticks": int(arrays["pid__antiwindup_blocked"][mask, axis].sum()),
                "output_negative_ticks": int(np.sum(arrays["pid__held_rad"][mask, axis] < 0)),
                "negative_error_positive_output_ticks": int(np.sum((arrays["error_G"][mask, axis] < 0) & (arrays["pid__held_rad"][mask, axis] > 0))),
                "I_opposes_current_error_ticks": int(np.sum(arrays["pid__I_provisional_rad"][mask, axis] * arrays["error_G"][mask, axis] < 0))}
            report["windows"][name]["axes"][label] = entry
    for width in (3, 5):
        prefix = str(width) + "__"; need = cached[prefix + "old_held_required_rad"]
        status = cached[prefix + "current_plant__given_memory_status"]
        arrays[prefix + "needed_held_rad"] = need
        arrays[prefix + "needed_normalized_target"] = cached[prefix + "target_normalized_required"]
        arrays[prefix + "source_feedback_G"] = cached[prefix + "source_feedback_G"]
        arrays[prefix + "given_memory_inverse_status"] = status
        arrays[prefix + "legacy_inverse_status"] = cached[prefix + "legacy_inverse_status"]
        arrays[prefix + "legacy_required_request_source_G"] = cached[prefix + "legacy_required_request_source_G"]
        arrays[prefix + "current_given_memory_required_request_plant_G"] = cached[prefix + "current_plant__given_memory_required_request_G"]
        feasible = active[:, None] & (status == 1)
        arrays[prefix + "needed_feasible_mask"] = feasible
        report["methods"][str(width) + "_point"] = {"windows": {}}
        for name, lo, hi in WINDOWS:
            w = audit.inverse.window_mask(t, lo, hi) & active
            valid = feasible & w[:, None] & ~halo[:, None]
            outputs = {"current": arrays["pid__held_rad"],
                "source_feedback_one_step": arrays["cf__source" + str(width) + "__held_rad"],
                "original_horizontal_PN_one_step": arrays["cf__original_horizontal_PN__held_rad"]}
            stat = {"derivative_halo_excluded_ticks": int(np.sum(w & halo)),
                "impossible_axis_ticks_pitch_yaw": np.sum(w[:, None] & (status == -1), axis=0).tolist(),
                "unique_nonhalo_axis_ticks_pitch_yaw": valid.sum(axis=0).tolist(),
                "normalized_output_error_on_feasible_nonhalo": {key: masked_error(value, need, arrays["K"], limits, valid) for key, value in outputs.items()},
                "axes": {}}
            for axis, label in enumerate(("pitch", "yaw")):
                vm = valid[:, axis]
                base_delta = arrays["pid__held_rad"][:, axis] - need[:, axis]
                stat["axes"][label] = {key: {"closer_to_needed_ticks": int(np.sum(vm & (abs(value[:, axis] - need[:, axis]) < abs(base_delta) - 1e-12))),
                    "farther_from_needed_ticks": int(np.sum(vm & (abs(value[:, axis] - need[:, axis]) > abs(base_delta) + 1e-12))),
                    "output_negative_ticks_all_window": int(np.sum(w & (value[:, axis] < 0))),
                    "output_change_old_held_deg": scalar_stats(np.degrees(value[:, axis] - arrays["pid__held_rad"][:, axis]), w)}
                    for key, value in outputs.items() if key != "current"}
            report["methods"][str(width) + "_point"]["windows"][name] = stat
    # Known points are a direct check of the request/feedback/output meanings.
    report["point_checks"] = []
    if cid == "jari_259_60":
        for ts, axis in ((.5, 1), (1.5, 0), (3.5, 1)):
            i = int(round(ts * 48))
            report["point_checks"].append({"t_s": ts, "axis": ("pitch", "yaw")[axis],
                "r_G": float(arrays["final_request_G"][i, axis]), "f_G": float(arrays["plant_feedback_G"][i, axis]),
                "e_G": float(arrays["error_G"][i, axis]), "u_deg": float(np.degrees(arrays["pid__held_rad"][i, axis])),
                "needed_u_5pt_deg": float(np.degrees(arrays["5__needed_held_rad"][i, axis])),
                "P_Iprovisional_D_deg": np.degrees([arrays["pid__" + key][i, axis] for key in ("P_rad", "I_provisional_rad", "D_rad")]).tolist()})
    return arrays, report


def plot_case(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; active = a["flight_gain"] > 0
    fig, axes = plt.subplots(4, 2, figsize=(15, 13.5), sharex=True)
    colors = {"current": "#7c317b", "required": "#116d71", "feedback": "#c3692d", "raw": "#787f95"}
    for axis, label in enumerate(("Pitch", "Yaw")):
        ax = axes[0, axis]
        ax.plot(t, a["pure_PN_G"][:, axis], color=colors["raw"], ls=":", label="Pure geometric PN")
        ax.plot(t, a["hitGain_PN_G"][:, axis], color=colors["raw"], label="Original PN × hitGain")
        ax.plot(t, a["unslewed_request_G"][:, axis], color="#353535", ls="--", label="Mixed PN + gated loft + gravity, capped")
        ax.plot(t, a["final_request_G"][:, axis], color=colors["current"], lw=1.8, label="Final request, S110")
        valid = (a["5__legacy_inverse_status"][:, axis] == 1) & active & ~a["gain_transition_derivative_halo"]
        ax.plot(t, np.where(valid, a["5__legacy_required_request_source_G"][:, axis], np.nan), color=colors["required"], label="Needed request: legacy inverse + source feedback")
        ax.set_ylabel(label + " request (G)")
        ax = axes[1, axis]
        ax.plot(t, a["plant_feedback_G"][:, axis], color=colors["feedback"], label="Plant feedback")
        ax.plot(t, a["5__source_feedback_G"][:, axis], color=colors["required"], ls="--", label="Velocity-derived feedback, 5pt")
        ax.plot(t, a["error_G"][:, axis], color=colors["current"], label="Current PID error = request − feedback")
        ax.axhline(0, color="#aaa", lw=.7); ax.set_ylabel("Feedback / error (G)")
        ax = axes[2, axis]
        for key, color, style, name in (("P_rad", "#497bad", "-", "P"), ("I_provisional_rad", "#b59a2e", "-", "Provisional I used in output"),
            ("D_rad", "#64a38d", "-", "D"), ("held_rad", colors["current"], "-", "Final held output")):
            ax.plot(t, np.degrees(a["pid__" + key][:, axis]), color=color, ls=style, label=name)
        ax.plot(t, np.degrees(a["pid__next_integral_rad"][:, axis]), color="#b59a2e", ls="--", label="Saved I after antiwindup")
        ax.axhline(0, color="#aaa", lw=.7); ax.set_ylabel("Equivalent old-input contribution (deg)")
        ax = axes[3, axis]
        normalization = a["K"] / a["old_held_limits_rad"][axis]
        valid = active & ~a["gain_transition_derivative_halo"]
        for width, style, alpha in ((3, ":", .7), (5, "-", 1.)):
            ax.plot(t, np.where(valid, a[str(width) + "__needed_normalized_target"][:, axis], np.nan), color=colors["required"], ls=style, alpha=alpha, label=f"Needed equivalent target, {width}pt")
        ax.plot(t, normalization * a["pid__held_rad"][:, axis], color=colors["current"], lw=1.8, label="Current")
        ax.plot(t, normalization * a["cf__source5__held_rad"][:, axis], color="#cb7841", ls="--", label="One step: source feedback")
        ax.plot(t, normalization * a["cf__original_horizontal_PN__held_rad"][:, axis], color="#5b76bd", ls="-.", label="One step: original horizontal PN")
        cap = normalization * a["pid__effective_limit_rad"][:, axis] * a["flight_gain"]
        high = active & a["pid__high_mode"][:, axis]
        # The original old-held cap corresponds to >20 normalized units before
        # the latch. It would hide all relevant target curves on this panel.
        ax.plot(t[high], cap[high], color="#696969", ls=":", lw=1., label="High-mode positive authority = 1")
        shown = np.concatenate([a[str(w) + "__needed_normalized_target"][valid, axis] for w in (3, 5)] +
            [normalization[valid] * a[key][valid, axis] for key in (
                "pid__held_rad", "cf__source5__held_rad", "cf__original_horizontal_PN__held_rad")])
        lower, upper = float(shown.min()), float(shown.max())
        margin = max(.03, .08 * (upper - lower))
        ax.set_ylim(lower - margin, upper + margin)
        if axis == 0 and high.any() and np.min(cap[high]) > upper + margin:
            ax.text(.02, .97, "High-mode authority = ±1 (outside pitch range)", transform=ax.transAxes,
                va="top", fontsize=8, color="#555")
        impossible = active & (a["5__given_memory_inverse_status"][:, axis] == -1)
        if impossible.any():
            ax.fill_between(t, 0, 1, where=impossible, transform=ax.get_xaxis_transform(), color="#e2a4a4", alpha=.16, label="Needed output above authority")
        ax.set_ylabel("Equivalent target / raw finsAoa"); ax.set_xlabel("Original clock (s)")
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.grid(alpha=.16)
        ax.axvspan(14 / 48, 14 / 48 + 2 / 24, color="#d8b477", alpha=.12)
        ax.axvline(.7, color="#999", lw=.6); ax.axvline(2.3, color="#999", lw=.6)
        ax.legend(loc="best", fontsize=7, framealpha=.85)
    fig.suptitle(cid + ": PN → request → PID, on identical reference states\nCounterfactuals are one tick from the same current previous memory; no free trajectory validation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .955)); fig.savefig(OUTPUT / (cid + "_pn_pid.png"), dpi=160); plt.close(fig)


def protected_snapshot():
    folders = (ROOT / "src", ROOT / "config", ROOT / "docs", ROOT / "scripts", ROOT / "data/raw", ROOT / "outputs/jari_offaxis")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for folder in folders if folder.exists()
        for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and OUTPUT not in p.parents and p != Path(__file__).resolve()}


def main():
    before = protected_snapshot()
    inputs = audit.inverse.source_inputs()
    cache = np.load(audit.OUTPUT / "request_audit.npz")
    coefficients = audit.drag.DragCoefficients(**json.loads((audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    saved = {}; reports = {}
    for cid in CASES:
        a, r = replay_case(cid, inputs[cid], cache, coefficients)
        saved.update({cid + "__" + key: value for key, value in a.items()}); reports[cid] = r
        plot_case(cid, a)
    finite = all(np.isfinite(value).all() for value in saved.values())
    checks = {"all_saved_arrays_finite": bool(finite), "protected_files_size_mtime_unchanged": before == protected_snapshot(),
        "zero_free_integrations": True, "zero_fits": True, "zero_API_or_web_calls": True,
        "controller_hz": 48, "plant_substeps_config_unchanged": 4, "clock_shift_s": 0.,
        "counterfactual_previous_PID_memory": "identical current baseline memory and previous K at each tick, no independent recursive CF history",
        "counterfactual_previous_S110_state": "same baseline previous slew state; one existing S110 step only",
        "closure_tolerance_rad_or_G": 1e-12,
        "case_max_closure": {cid: r["checks"] for cid, r in reports.items()}}
    if not finite or not checks["protected_files_size_mtime_unchanged"] or any(v > 1e-12 for r in reports.values() for v in r["checks"].values()):
        raise AssertionError(checks)
    np.savez_compressed(OUTPUT / "pn_pid_timeseries.npz", **saved)
    document = {"task": "War Thunder/Jari conditional PN-PID separation", "cases": reports, "verification": checks,
        "conditions_and_limits": [
            "Interpolated 24Hz source states on original 48Hz clock; native intermediate state/request/PID memory unobserved.",
            "PN vectors in request details already contain hitGain and horizontal mixing; pure PN is independently recomputed from geometry.",
            "Caps and slew are nonlinear ordered boundaries; projected stage differences are recorded, not globally additive component allocations.",
            "Current and source-feedback recursive histories in earlier cache are distinct. This analysis uses only current baseline history for every one-step counterfactual.",
            "Provisional integral contributes to this tick's output; antiwindup next integral is separately saved.",
            "Transport increment, transport clipping, error increment, update clipping and antiwindup retention sum exactly to saved I change.",
            "Old-coordinate transport is a coordinate conversion: K_current*I_transported = K_reference*I_previous before clipping. Target-coordinate memory changes are separately recorded.",
            "3/5-point derivative inputs are noncausal diagnostics; gain-transition halo of two original-source rows on either side remains explicitly masked.",
            "Above current authority no needed request is asserted; saturated boundary is not a unique inverse.",
            "Source controls are not treated as known actuator truth; output comparison uses frozen-plant equivalent required input.",
            "Counterfactuals are fixed-state one-tick tests. No closed-loop improvement or real-game equivalence is verified.",
            "Runtime/core/config/raw fields and previously frozen plant, drag, loft and PID parameters remain unchanged."]}
    audit.slew.write_json(OUTPUT / "results.json", document)
    print(json.dumps({"verification": checks, "point_checks_60": reports["jari_259_60"]["point_checks"]}), flush=True)


if __name__ == "__main__":
    main()
