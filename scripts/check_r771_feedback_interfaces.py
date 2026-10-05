#!/usr/bin/env python3
"""Fixed game-only request/feedback coordinate hypotheses, no retuning.

Specific/net conversions keep the original request caps and S110 ordering.
Body requests keep the axial command implied by the original gravity pipeline,
not an observed axial acceleration. Plant pitch/yaw output mappings stay fixed.
"""
from __future__ import annotations
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_pn_pid_decoupling as dec
from aim120_model.aerodynamics import body_axes_for_state, wind_normal_basis
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import add, dot, scale
from aim120_model.r771_game_control import R771GameControl
from aim120_model.units import G0_MPS2

OUT = ROOT / "outputs/jari_offaxis/feedback_interfaces"
FRAMES = ("wind_specific", "wind_net", "body_specific", "body_net")
CASES = ("jari_259_0", "jari_259_30", "jari_259_60")
AXES = ("pitch", "yaw")


def interface_vectors(state, evaluation, request, details, gravity, max_accel):
    wb = evaluation.wind_basis; body = body_axes_for_state(state)
    outer = dec.outer_boundaries(details["pn_mps2"], details["loft_g"], wb, gravity, max_accel)
    # Retain the axial component actually constructed by original cap2. The
    # post-S110 stage changes only wind-normal components; no source force used.
    original_axial = dot(outer["after_cap2_world_mps2"], wb.forward)
    command = add(add(scale(wb.up, request[0] * G0_MPS2), scale(wb.right, request[1] * G0_MPS2)),
        scale(wb.forward, original_axial))
    gravity_world = (0., -gravity, 0.)
    net_command = add(command, gravity_world)
    body_r = dec.projected(command, body); body_f = dec.projected(evaluation.specific_acceleration, body)
    wind_g = dec.projected(gravity_world, wb); body_g = dec.projected(gravity_world, body)
    r = {"wind_specific": request,
        "wind_net": tuple(a + b for a, b in zip(request, wind_g)),
        "body_specific": body_r, "body_net": dec.projected(net_command, body)}
    f = {"wind_specific": evaluation.feedback_g,
        "wind_net": dec.projected(evaluation.acceleration, wb),
        "body_specific": body_f, "body_net": dec.projected(evaluation.acceleration, body)}
    return r, f, {"command_specific_world_mps2": command, "command_net_world_mps2": net_command,
        "original_command_axial_mps2": original_axial, "plant_specific_world_mps2": evaluation.specific_acceleration,
        "plant_net_world_mps2": evaluation.acceleration, "gravity_wind_G": wind_g, "gravity_body_G": body_g,
        "cap1_active": outer["cap1_active"], "cap2_active": outer["cap2_active"],
        "world_command_wind_projection_closure_G": np.asarray(dec.projected(command, wb)) - request}


def first_time(t, mask):
    return float(t[np.flatnonzero(mask)[0]]) if mask.any() else None


def outcome(output, needed, base, k, limit, mask):
    d = k * (output - needed) / limit; b = k * (base - needed) / limit
    return {"ticks": int(mask.sum()), "RMS_normalized_target_error": dec.audit.inverse.rms(d[mask]) if mask.any() else None,
        "bias_normalized_target_error": float(d[mask].mean()) if mask.any() else None,
        "opposite_sign_ticks": int(np.sum(mask & (output * needed < 0))),
        "negative_output_with_positive_needed_ticks": int(np.sum(mask & (output < 0) & (needed > 0))),
        "closer_ticks": int(np.sum(mask & (abs(d) < abs(b) - 1e-12))),
        "farther_ticks": int(np.sum(mask & (abs(d) > abs(b) + 1e-12)))}


def conditional(cid, item, inverse_cache, previous_cache, drag_coefficients):
    cfg = item["config"]; ref = item["ref"]
    t = np.arange(170) / 48
    model = dec.audit.fix.make_model(cfg, drag_coefficients)
    model.initialize(dataclasses.replace(item["initial"]))
    limits = np.radians([cfg["aerodynamics"][key] for key in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    gains = np.asarray([interpolate_table(float(ts) + dec.DT, cfg["guidance"]["flight_time_gain_table"]) for ts in t])
    base_controller = R771GameControl(); base_memory = [(0., 0., 0.), (0., 0., 0.)]
    controllers = {name: R771GameControl() for name in FRAMES}
    memories = {name: [(0., 0., 0.), (0., 0., 0.)] for name in FRAMES}
    rows = []; history = []; one_step = []; term_history = []; memory_history = []
    for i, ts in enumerate(t):
        state = dec.audit.fix.source_state_at(ref, float(ts)); target = item["target_at"](float(ts))
        evaluation = model.evaluate(state, float(ts), (0., 0.))
        request, details = model.request(state, target, float(ts), evaluation)
        requests, feedbacks, vector_data = interface_vectors(state, evaluation, request, details,
            model.gravity, float(cfg["guidance"]["maximum_lateral_acceleration_g"]) * model.gravity)
        row = {**vector_data, "K": model.current_control_k, "nonlinear_force_active": model.current_force_nonlinear}
        for name in FRAMES:
            row[name + "__request_G"] = requests[name]; row[name + "__feedback_G"] = feedbacks[name]
            row[name + "__error_G"] = np.asarray(requests[name]) - feedbacks[name]
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + dec.DT)
        tick_history = {name: [0., 0.] for name in FRAMES}; tick_one = copy.deepcopy(tick_history)
        tick_terms = {name: [{"P_rad": 0., "I_provisional_rad": 0., "D_rad": 0., "held_rad": 0.,
            "next_integral_rad": 0.} for _ in range(2)] for name in FRAMES}
        if gains[i] > 1e-12:
            for axis in range(2):
                for name in FRAMES:
                    trial = copy.deepcopy(base_controller)
                    tick_one[name][axis], _ = trial.update_axis(axis, float(ts), float(row[name + "__error_G"][axis]),
                        base_memory[axis], pid, limits[axis], gains[i], model.current_control_k, model.current_force_nonlinear)
                    terms = dec.pid_terms(controllers[name], axis, memories[name][axis], float(row[name + "__error_G"][axis]),
                        pid, gains[i], model.current_control_k, model.current_force_nonlinear, limits[axis])
                    tick_history[name][axis], memories[name][axis] = controllers[name].update_axis(axis, float(ts), float(row[name + "__error_G"][axis]),
                        memories[name][axis], pid, limits[axis], gains[i], model.current_control_k, model.current_force_nonlinear)
                    tick_terms[name][axis] = {key: terms[key] for key in tick_terms[name][axis]}
                _, base_memory[axis] = base_controller.update_axis(axis, float(ts), float(row["wind_specific__error_G"][axis]),
                    base_memory[axis], pid, limits[axis], gains[i], model.current_control_k, model.current_force_nonlinear)
        rows.append(row); history.append(tick_history); one_step.append(tick_one); term_history.append(tick_terms)
        memory_history.append(copy.deepcopy(memories))
    a = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    a.update(t=t, flight_gain=gains, old_held_limits_rad=limits,
        gain_transition_derivative_halo=(t >= 14 / 48 - 2 / 24) & (t <= 14 / 48 + 2 / 24))
    for name in FRAMES:
        a[name + "__own_history_held_rad"] = np.asarray([row[name] for row in history])
        a[name + "__one_step_held_rad"] = np.asarray([row[name] for row in one_step])
        a[name + "__own_history_memory"] = np.asarray([row[name] for row in memory_history])
        for term in term_history[0][name][0]:
            a[name + "__" + term] = np.asarray([[row[name][axis][term] for axis in range(2)] for row in term_history])
    active = gains > 1e-12; high = np.maximum.accumulate(a["nonlinear_force_active"] & active)
    effective_limits = np.where(high[:, None], np.minimum(limits, limits / a["K"][:, None]), limits)
    report = {"raw_PID_entire_dict": cfg["control"]["pid"], "original_flight_gain_table": cfg["guidance"]["flight_time_gain_table"],
        "equivalence": {}, "methods": {}, "sign_event_sequence": {}, "checks": {}}
    for first, second in (("wind_specific", "wind_net"), ("body_specific", "body_net")):
        report["equivalence"][first + "_vs_" + second] = {key: float(np.max(abs(a[first + "__" + key] - a[second + "__" + key])))
            for key in ("error_G", "own_history_held_rad", "one_step_held_rad", "own_history_memory")}
    report["checks"]["command_projection_closure_max_abs_G"] = float(np.max(abs(a["world_command_wind_projection_closure_G"])))
    for name in FRAMES:
        report["checks"][name + "__PID_forward_held_closure_rad"] = float(np.max(abs(a[name + "__held_rad"] - a[name + "__own_history_held_rad"])))
        report["checks"][name + "__PID_savedI_closure_rad"] = float(np.max(abs(a[name + "__next_integral_rad"] - a[name + "__own_history_memory"][:, :, 0])))
    if cid != "jari_259_0":
        for key, source in (("wind_specific__request_G", "final_request_G"), ("wind_specific__feedback_G", "plant_feedback_G"),
            ("wind_specific__own_history_held_rad", "pid__held_rad"), ("wind_specific__own_history_memory", "next_current_memory")):
            report["checks"]["cached_baseline_" + key + "_max_abs"] = float(np.max(abs(a[key] - previous_cache[cid + "__" + source])))
    for width in (3, 5):
        source_t = inverse_cache[cid + "__t"]
        need = dec.audit.interp(t, source_t, inverse_cache[cid + f"__{width}__old_held_req_rad"])
        a[str(width) + "__needed_held_rad"] = need
        a[str(width) + "__needed_normalized_target"] = a["K"][:, None] * need / limits
        impossible = active[:, None] & (abs(need) > gains[:, None] * effective_limits + 1e-12)
        a[str(width) + "__impossible_mask"] = impossible
        report["methods"][str(width) + "_point"] = {}
        for name, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
            window = active & dec.audit.inverse.window_mask(t, lo, hi) & ~a["gain_transition_derivative_halo"]
            w = {"window_s": [lo, hi], "axes": {}}
            for axis, label in enumerate(AXES):
                feasible = window & ~impossible[:, axis]
                baseline = a["wind_specific__own_history_held_rad"][:, axis]
                w["axes"][label] = {"impossible_ticks": int(np.sum(window & impossible[:, axis])), "candidates": {}}
                for frame in FRAMES:
                    entry = {"request_G": dec.scalar_stats(a[frame + "__request_G"][:, axis], window),
                        "feedback_G": dec.scalar_stats(a[frame + "__feedback_G"][:, axis], window),
                        "error_G": dec.scalar_stats(a[frame + "__error_G"][:, axis], window)}
                    for mode in ("one_step", "own_history"):
                        u = a[frame + "__" + mode + "_held_rad"][:, axis]
                        entry[mode] = {"feasible": outcome(u, need[:, axis], baseline, a["K"], limits[axis], feasible),
                            "all_including_unattainable": outcome(u, need[:, axis], baseline, a["K"], limits[axis], window)}
                    w["axes"][label]["candidates"][frame] = entry
            report["methods"][str(width) + "_point"][name] = w
    for frame in ("wind_specific", "body_specific"):
        event = {}
        for axis, label in enumerate(AXES):
            candidate = {"first_negative_" + field: first_time(t, active & (a[frame + "__" + field][:, axis] < -1e-12)) for field in (
                "request_G", "feedback_G", "error_G", "P_rad", "I_provisional_rad", "D_rad", "own_history_held_rad")}
            mask = active & (a[frame + "__own_history_held_rad"][:, axis] < -1e-12) & (a["5__needed_held_rad"][:, axis] > 1e-12)
            candidate["first_negative_u_when_needed_positive_s"] = first_time(t, mask)
            if mask.any():
                i = int(np.flatnonzero(mask)[0]); candidate["at_first_negative_u_when_needed_positive"] = {
                    field: float(a[frame + "__" + field][i, axis]) for field in ("request_G", "feedback_G", "error_G", "P_rad", "I_provisional_rad", "D_rad", "own_history_held_rad")}
            event[label] = candidate
        report["sign_event_sequence"][frame] = event
    return a, report


def plot_case(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; active = a["flight_gain"] > 0
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis, label in enumerate(AXES):
        ax = axes[0, axis]
        for frame, color in (("wind_specific", "#78337b"), ("body_specific", "#ce772e")):
            ax.plot(t, a[frame + "__request_G"][:, axis], color=color, label=frame + " request")
            ax.plot(t, a[frame + "__feedback_G"][:, axis], color=color, ls="--", label=frame + " feedback")
        ax.set_ylabel(label + " r / f (G)")
        ax = axes[1, axis]
        for frame, color in (("wind_specific", "#78337b"), ("body_specific", "#ce772e")):
            ax.plot(t, a[frame + "__error_G"][:, axis], color=color, label=frame + " error")
        ax.axhline(0, color="#aaa", lw=.6); ax.set_ylabel("Error r − f (G)")
        ax = axes[2, axis]; factor = a["K"] / a["old_held_limits_rad"][axis]
        valid = active & ~a["gain_transition_derivative_halo"]
        ax.plot(t, np.where(valid, a["5__needed_normalized_target"][:, axis], np.nan), color="#177277", label="Frozen-plant needed, 5pt")
        ax.plot(t, factor * a["wind_specific__own_history_held_rad"][:, axis], color="#78337b", label="Current baseline")
        ax.plot(t, factor * a["body_specific__one_step_held_rad"][:, axis], color="#6b82ba", ls="--", label="Body: one tick, same previous baseline memory")
        ax.plot(t, factor * a["body_specific__own_history_held_rad"][:, axis], color="#ce772e", label="Body: own history, fixed reference states")
        impossible = a["5__impossible_mask"][:, axis]
        if impossible.any():
            ax.fill_between(t, 0, 1, where=impossible, transform=ax.get_xaxis_transform(), color="#e6b5b5", alpha=.18, label="Above unchanged authority")
        ax.axhline(0, color="#aaa", lw=.6); ax.set_ylabel("Equivalent target / raw finsAoa"); ax.set_xlabel("Original clock (s)")
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.axvline(.7, color="#aaa", lw=.6); ax.axvline(2.3, color="#aaa", lw=.6)
        ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    fig.suptitle(cid + ": paired request/feedback interface transforms\nSpecific↔net is algebraically equivalent with unchanged cap order; body command axial part comes from the original outer pipeline", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT / (cid + "_interfaces.png"), dpi=160); plt.close(fig)


def snapshot():
    folders = (ROOT / "src", ROOT / "config", ROOT / "docs", ROOT / "scripts", ROOT / "data/raw", ROOT / "outputs/jari_offaxis")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for folder in folders if folder.exists() for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and OUT not in p.parents and p != Path(__file__).resolve()}


def axial_completed_request(body, wind_basis, current_specific_force, final_wind_request):
    """Current-state-only interface: leave the unregulated axial residual zero.

    This is a candidate body input to the existing body pitch/yaw actuator law,
    not a change to PN, force, output coordinates, or the controller gains.
    """
    axial = dot(current_specific_force, wind_basis.forward)
    command = add(add(scale(wind_basis.up, final_wind_request[0] * G0_MPS2),
                      scale(wind_basis.right, final_wind_request[1] * G0_MPS2)),
                  scale(wind_basis.forward, axial))
    return dec.projected(command, body), dec.projected(current_specific_force, body), command


def axial_completed_case(cid, item, old_cache, inverse_cache):
    """Reuse saved current force/request; replay only controller memories."""
    a = {key[len(cid) + 2:]: old_cache[key] for key in old_cache.files if key.startswith(cid + "__")}
    cfg = item["config"]; ref = item["ref"]; t = a["t"]; gains = a["flight_gain"]
    limits = a["old_held_limits_rad"]; active = gains > 1e-12
    geometries = []
    for ts in t:
        state = dec.audit.fix.source_state_at(ref, float(ts))
        body = body_axes_for_state(state)
        wind = tuple(cfg["atmosphere"].get("wind_mps", (0., 0., 0.)))
        wb = wind_normal_basis(body, tuple(x - y for x, y in zip(state.velocity, wind)))
        geometries.append((body, wb))
    r = []; f = []; commands = []; old_normal = []; old_axial = []; normal_world = []; axial_world = []
    for i, (body, wb) in enumerate(geometries):
        force = a["plant_specific_world_mps2"][i]
        request, feedback, command = axial_completed_request(body, wb, force, a["wind_specific__request_G"][i])
        residual = a["command_specific_world_mps2"][i] - force
        axial = np.asarray(wb.forward) * dot(residual, wb.forward); normal = residual - axial
        r.append(request); f.append(feedback); commands.append(command)
        old_normal.append(dec.projected(normal, body)); old_axial.append(dec.projected(axial, body))
        normal_world.append(normal); axial_world.append(axial)
    a.update(body_axial_completed__request_G=np.asarray(r), body_axial_completed__feedback_G=np.asarray(f),
        body_axial_completed__error_G=np.asarray(r) - f, axial_completed_command_world_mps2=np.asarray(commands),
        old_body_error_normal_contribution_G=np.asarray(old_normal), old_body_error_axial_contribution_G=np.asarray(old_axial),
        old_normal_residual_world_mps2=np.asarray(normal_world), old_axial_residual_world_mps2=np.asarray(axial_world))
    names = ("wind_specific", "body_axial_completed")
    controllers = {name: R771GameControl() for name in names}
    memories = {name: [(0., 0., 0.), (0., 0., 0.)] for name in names}
    held = {name: np.zeros((len(t), 2)) for name in names}; saved = {name: np.zeros((len(t), 2, 3)) for name in names}
    one = np.zeros((len(t), 2)); terms = {key: np.zeros((len(t), 2)) for key in (
        "P_rad", "I_provisional_rad", "D_rad", "held_rad", "next_integral_rad")}
    for i, ts in enumerate(t):
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + dec.DT)
        if active[i]:
            for axis in range(2):
                trial = copy.deepcopy(controllers["wind_specific"])
                one[i, axis], _ = trial.update_axis(axis, float(ts), float(a["body_axial_completed__error_G"][i, axis]),
                    memories["wind_specific"][axis], pid, limits[axis], gains[i], a["K"][i], a["nonlinear_force_active"][i])
                part = dec.pid_terms(controllers["body_axial_completed"], axis, memories["body_axial_completed"][axis],
                    float(a["body_axial_completed__error_G"][i, axis]), pid, gains[i], a["K"][i], a["nonlinear_force_active"][i], limits[axis])
                for key in terms:
                    terms[key][i, axis] = part[key]
                for name in names:
                    held[name][i, axis], memories[name][axis] = controllers[name].update_axis(axis, float(ts),
                        float(a[name + "__error_G"][i, axis]), memories[name][axis], pid, limits[axis], gains[i],
                        a["K"][i], a["nonlinear_force_active"][i])
        for name in names:
            saved[name][i] = memories[name]
    a["body_axial_completed__own_history_held_rad"] = held["body_axial_completed"]
    a["body_axial_completed__own_history_memory"] = saved["body_axial_completed"]
    a["body_axial_completed__one_step_held_rad"] = one
    for key, value in terms.items():
        a["body_axial_completed__" + key] = value
    checks = {
        "old_body_error_normal_plus_axial_closure_G": float(np.max(abs(a["body_specific__error_G"] - np.asarray(old_normal) - old_axial))),
        "completed_error_equals_old_normal_residual_G": float(np.max(abs(a["body_axial_completed__error_G"] - old_normal))),
        "completed_feedback_equals_old_body_specific_G": float(np.max(abs(a["body_axial_completed__feedback_G"] - a["body_specific__feedback_G"]))),
        "cached_baseline_held_max_abs_rad": float(np.max(abs(held["wind_specific"] - a["wind_specific__own_history_held_rad"]))),
        "cached_baseline_memory_max_abs": float(np.max(abs(saved["wind_specific"] - a["wind_specific__own_history_memory"]))),
        "PID_forward_held_closure_rad": float(np.max(abs(terms["held_rad"] - held["body_axial_completed"]))),
        "PID_nextI_closure_rad": float(np.max(abs(terms["next_integral_rad"] - saved["body_axial_completed"][:, :, 0]))),
        "original_clock_max_abs_tick": float(np.max(abs(t * 48 - np.arange(len(t))))),
    }
    for i, (body, wb) in enumerate(geometries):
        command = a["axial_completed_command_world_mps2"][i]; force = a["plant_specific_world_mps2"][i]
        checks["normal_request_unchanged_max_abs_G"] = max(checks.get("normal_request_unchanged_max_abs_G", 0.),
            float(np.max(abs(np.asarray(dec.projected(command, wb)) - a["wind_specific__request_G"][i]))))
        checks["completed_axial_residual_max_abs_G"] = max(checks.get("completed_axial_residual_max_abs_G", 0.),
            abs(dot(np.asarray(command) - force, wb.forward)) / G0_MPS2)
    report = {"checks": checks, "error_decomposition_windows": {}, "methods": {}, "sign_event_sequence": {},
        "raw_PID_entire_dict": cfg["control"]["pid"], "original_flight_gain_table": cfg["guidance"]["flight_time_gain_table"]}
    for name, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
        mask = active & dec.audit.inverse.window_mask(t, lo, hi) & ~a["gain_transition_derivative_halo"]
        report["error_decomposition_windows"][name] = {label: {
            "old_total_body_error_G": dec.scalar_stats(a["body_specific__error_G"][:, axis], mask),
            "wind_normal_residual_body_contribution_G": dec.scalar_stats(a["old_body_error_normal_contribution_G"][:, axis], mask),
            "wind_forward_residual_body_contribution_G": dec.scalar_stats(a["old_body_error_axial_contribution_G"][:, axis], mask)
        } for axis, label in enumerate(AXES)}
    for width in (3, 5):
        report["methods"][str(width) + "_point"] = {}
        for name, lo, hi in (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5)):
            window = active & dec.audit.inverse.window_mask(t, lo, hi) & ~a["gain_transition_derivative_halo"]
            w = {"window_s": [lo, hi], "axes": {}}
            for axis, label in enumerate(AXES):
                mask = window & ~a[str(width) + "__impossible_mask"][:, axis]
                baseline = a["wind_specific__own_history_held_rad"][:, axis]; need = a[str(width) + "__needed_held_rad"][:, axis]
                w["axes"][label] = {"impossible_ticks": int(np.sum(window & a[str(width) + "__impossible_mask"][:, axis])), "candidates": {}}
                for frame in ("wind_specific", "body_specific", "body_axial_completed"):
                    entry = {field: dec.scalar_stats(a[frame + "__" + field][:, axis], window) for field in ("request_G", "feedback_G", "error_G")}
                    for mode in ("one_step", "own_history"):
                        u = a[frame + "__" + mode + "_held_rad"][:, axis]
                        entry[mode] = {"feasible": outcome(u, need, baseline, a["K"], limits[axis], mask),
                            "all_including_unattainable": outcome(u, need, baseline, a["K"], limits[axis], window)}
                    w["axes"][label]["candidates"][frame] = entry
            report["methods"][str(width) + "_point"][name] = w
    for frame in ("wind_specific", "body_specific", "body_axial_completed"):
        report["sign_event_sequence"][frame] = {}
        for axis, label in enumerate(AXES):
            event = {"first_negative_" + field: first_time(t, active & (a[frame + "__" + field][:, axis] < -1e-12)) for field in (
                "request_G", "feedback_G", "error_G", "P_rad", "I_provisional_rad", "D_rad", "own_history_held_rad")}
            mask = active & (a[frame + "__own_history_held_rad"][:, axis] < -1e-12) & (a["5__needed_held_rad"][:, axis] > 1e-12)
            event["first_negative_u_when_needed_positive_s"] = first_time(t, mask)
            report["sign_event_sequence"][frame][label] = event
    # Source derivatives remain diagnostic. Recover their cached world specific
    # vector before interpolation, then project into each SAME current wind basis.
    source_t = inverse_cache[cid + "__t"]
    source_forward = np.asarray([body_axes_for_state(dec.audit.inverse.source_state(ref, i)).forward for i in range(len(source_t))])
    source_mass = ref["mass"]; source_thrust = ref["thrust"]
    first_negative = first_time(t, active & (a["wind_specific__error_G"][:, 0] < -1e-12))
    report["source_feedback_at_first_baseline_r_below_f"] = {"first_negative_pitch_error_s": first_negative, "methods": {}}
    for width in (3, 5):
        full_source = (inverse_cache[cid + f"__{width}__source_aerodynamic_force_N"] + source_thrust[:, None] * source_forward) / source_mass[:, None]
        full_source_tick = dec.audit.interp(t, source_t, full_source)
        feedback = np.asarray([dec.projected(full_source_tick[i], wb) for i, (_, wb) in enumerate(geometries)])
        a[str(width) + "__source_specific_world_mps2_diagnostic"] = full_source_tick
        a[str(width) + "__source_wind_specific_feedback_G_diagnostic"] = feedback
        source_basis = [wind_normal_basis(body_axes_for_state(dec.audit.inverse.source_state(ref, i)),
            tuple(x - y for x, y in zip(ref["velocity_xyz_mps"][i], cfg["atmosphere"].get("wind_mps", (0., 0., 0.))))) for i in range(len(source_t))]
        original_feedback = np.asarray([dec.projected(full_source[i], wb) for i, wb in enumerate(source_basis)])
        checks[str(width) + "_source_specific_feedback_reconstruction_closure_G"] = float(np.max(abs(original_feedback - inverse_cache[cid + f"__{width}__source_feedback_pitch_yaw_G"])))
        if first_negative is not None:
            around = abs(t - first_negative) <= 2 / 24 + 1e-12
            indices = np.flatnonzero(around); center = int(np.argmin(abs(t - first_negative)))
            report["source_feedback_at_first_baseline_r_below_f"]["methods"][str(width) + "_point"] = {
                "window_s": [float(t[indices[0]]), float(t[indices[-1]])],
                "at_first_negative_pitch_error": {"request_G": a["wind_specific__request_G"][center].tolist(),
                    "plant_feedback_G": a["wind_specific__feedback_G"][center].tolist(), "source_feedback_G": feedback[center].tolist(),
                    "plant_minus_source_G": (a["wind_specific__feedback_G"][center] - feedback[center]).tolist(),
                    "request_minus_source_G": (a["wind_specific__request_G"][center] - feedback[center]).tolist()},
                "window": {label: {"plant_minus_source_G": dec.scalar_stats(a["wind_specific__feedback_G"][:, axis] - feedback[:, axis], around),
                    "request_minus_source_G": dec.scalar_stats(a["wind_specific__request_G"][:, axis] - feedback[:, axis], around)} for axis, label in enumerate(AXES)}}
    return a, report


def plot_axial_completed(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis, label in enumerate(AXES):
        ax = axes[0, axis]
        ax.plot(t, a["body_specific__error_G"][:, axis], color="#999", label="Previous body error: total")
        ax.plot(t, a["old_body_error_normal_contribution_G"][:, axis], color="#ce772e", label="Wind-normal residual → body")
        ax.plot(t, a["old_body_error_axial_contribution_G"][:, axis], color="#6b82ba", label="Wind-forward residual → body")
        ax.set_ylabel(label + " error contributions (G)")
        ax = axes[1, axis]
        ax.plot(t, a["wind_specific__error_G"][:, axis], color="#78337b", label="Current wind error")
        ax.plot(t, a["body_axial_completed__error_G"][:, axis], color="#ce772e", label="Axial-completed body error")
        ax.set_ylabel("Error r − f (G)")
        ax = axes[2, axis]; factor = a["K"] / a["old_held_limits_rad"][axis]
        valid = (a["flight_gain"] > 0) & ~a["gain_transition_derivative_halo"]
        ax.plot(t, np.where(valid, a["5__needed_normalized_target"][:, axis], np.nan), color="#177277", label="Frozen-plant needed, 5pt")
        for frame, color, caption in (("wind_specific", "#78337b", "Current baseline"), ("body_specific", "#999", "Previous body interface"),
                ("body_axial_completed", "#ce772e", "Axial-completed: own history")):
            ax.plot(t, factor * a[frame + "__own_history_held_rad"][:, axis], color=color, label=caption)
        ax.plot(t, factor * a["body_axial_completed__one_step_held_rad"][:, axis], color="#6b82ba", ls="--", label="Axial-completed: same baseline previous memory")
        ax.fill_between(t, 0, 1, where=a["5__impossible_mask"][:, axis], transform=ax.get_xaxis_transform(), color="#e6b5b5", alpha=.18, label="Above unchanged authority")
        ax.set_ylabel("Equivalent target / raw finsAoa"); ax.set_xlabel("Original clock (s)")
    for ax in axes.flat:
        ax.set_xlim(14 / 48, 3.5); ax.axhline(0, color="#aaa", lw=.6); ax.axvline(.7, color="#aaa", lw=.6); ax.axvline(2.3, color="#aaa", lw=.6)
        ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    fig.suptitle(cid + ": axial completion using the current plant force only\nBody error becomes the wind-normal residual projected to body; body actuator output and original PID remain fixed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT / (cid + "_axial_completed.png"), dpi=160); plt.close(fig)


def axial_completed_main():
    before = snapshot()
    protected_old = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in OUT.iterdir() if p.is_file() and "axial_completed" not in p.name}
    inputs = dec.audit.inverse.source_inputs(); old = np.load(OUT / "conditional_interfaces.npz")
    inverse_cache = np.load(dec.audit.inverse.OUTPUT / "inverse_inputs.npz")
    arrays = {}; reports = {}; gate = {}
    for cid in CASES:
        a, r = axial_completed_case(cid, inputs[cid], old, inverse_cache)
        arrays.update({cid + "__" + key: value for key, value in a.items()}); reports[cid] = r
        plot_axial_completed(cid, a)
    for cid in CASES[1:]:
        r = reports[cid]["methods"]["5_point"]
        pitch = r["full_0p5_3p5"]["axes"]["pitch"]["candidates"]
        bp = pitch["wind_specific"]["own_history"]["feasible"]; cp = pitch["body_axial_completed"]["own_history"]["feasible"]
        yaw = {w: r[w]["axes"]["yaw"]["candidates"] for w, _, _ in dec.WINDOWS}
        yaw_ok = all(v["body_axial_completed"]["own_history"]["feasible"]["RMS_normalized_target_error"] is None or
            v["body_axial_completed"]["own_history"]["feasible"]["RMS_normalized_target_error"] <= 1.05 * v["wind_specific"]["own_history"]["feasible"]["RMS_normalized_target_error"] for v in yaw.values())
        gate[cid] = {"negative_pitch_when_needed_positive_ticks": cp["negative_output_with_positive_needed_ticks"],
            "pitch_RMS_not_worse": cp["RMS_normalized_target_error"] <= bp["RMS_normalized_target_error"],
            "yaw_each_feasible_window_no_more_than_5percent_worse": yaw_ok,
            "passes": cp["negative_output_with_positive_needed_ticks"] == 0 and cp["RMS_normalized_target_error"] <= bp["RMS_normalized_target_error"] and yaw_ok}
    if all(v["passes"] for v in gate.values()):
        raise RuntimeError("Conditional gate passed: implement exactly one six-scenario free validation before final delivery.")
    unchanged_old = all(p.exists() and (p.stat().st_size, p.stat().st_mtime_ns) == pair for name, pair in protected_old.items() for p in [Path(name)])
    verification = {"all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()), "protected_size_mtime_unchanged": before == snapshot(),
        "previous_interface_outputs_size_mtime_unchanged": unchanged_old, "case_checks": {cid: r["checks"] for cid, r in reports.items()},
        "controller_hz": 48, "plant_substeps_unchanged": 4, "clock_shift_s": 0., "fits_scans_API_web_calls": 0,
        "plant_integrations": 0, "controller_only_fixed_state_replays": 3, "raw_fields_parameters_defaults_changed": False}
    if not all(verification[k] for k in ("all_arrays_finite", "protected_size_mtime_unchanged", "previous_interface_outputs_size_mtime_unchanged")) or any(v > 1e-12 for r in reports.values() for v in r["checks"].values()):
        raise AssertionError(verification)
    np.savez_compressed(OUT / "axial_completed_interfaces.npz", **arrays)
    doc = {"task": "War Thunder game-only axial-completed feedback interface", "cases": reports, "free_validation_gate": gate,
        "free_validation_candidate": None, "verification": verification, "candidate_definition":
            "body_axial_completed: keep original final S110 wind-normal request; complete wind-forward command with current plant specific-force forward component; project both command and specific force to body up/right; keep old-held body pitch/yaw output law.",
        "conditions_and_limits": [
            "Previous body's retained axial command came from the gravity/cap outer construction, not a source axial acceleration. The new interface cancels its unregulated forward residual exactly.",
            "Completed body error is purely the current wind-normal residual projected into body axes. This is a zero-new-coefficient input-interface hypothesis; it is not whole-controller coordinate equivalence because independent scalar PID histories and body output mappings are retained.",
            "Original PN, half-yaw, loft, gravity/cap ordering, S110, raw PID, target limit/latch, gain schedule, frozen single-surface plant and output mapping are unchanged.",
            "One-step uses identical previous baseline memory/K. Candidate own history starts at the same zero/inactive initialization and then recurses separately on fixed reference states. These histories are not interchangeable.",
            "Source 3/5-point vectors are reconstructed solely from previously cached derivative-derived aerodynamic force, source thrust/body direction/mass for diagnostic projection. No derivative, source vector, control, or future sample enters the candidate function.",
            "Source and plant diagnostics are both total specific acceleration in the same current wind-normal basis; no net/specific mixture or independent new derivative calculation.",
            "Gate is unchanged: clear negative pitch while needed pitch is positive, no feasible pitch RMS worsening, and each feasible yaw window at most 5 percent worse. Derivative halo and impossible cap masks remain intact.",
            "Rejected conditional candidate: no free trajectory and no gain/mixing/coefficient scan. Diagnostic comparison is not native game validation."]}
    dec.audit.slew.write_json(OUT / "axial_completed_results.json", doc)
    print(json.dumps({"free_validation_gate": gate, "verification": verification}), flush=True)


def main():
    before = snapshot(); inputs = dec.audit.inverse.source_inputs()
    ic = np.load(dec.audit.inverse.OUTPUT / "inverse_inputs.npz"); old = np.load(dec.OUTPUT / "pn_pid_timeseries.npz")
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    OUT.mkdir(parents=True, exist_ok=True); all_arrays = {}; reports = {}
    for cid in CASES:
        a, r = conditional(cid, inputs[cid], ic, old, coefficients)
        reports[cid] = r; all_arrays.update({cid + "__" + key: value for key, value in a.items()}); plot_case(cid, a)
    gate = {}
    for cid in ("jari_259_30", "jari_259_60"):
        r = reports[cid]["methods"]["5_point"]
        pitch = r["full_0p5_3p5"]["axes"]["pitch"]["candidates"]
        bp = pitch["wind_specific"]["own_history"]["feasible"]; cp = pitch["body_specific"]["own_history"]["feasible"]
        yaw = {w: r[w]["axes"]["yaw"]["candidates"] for w, _, _ in dec.WINDOWS}
        yaw_ok = all(v["body_specific"]["own_history"]["feasible"]["RMS_normalized_target_error"] is None or
            v["body_specific"]["own_history"]["feasible"]["RMS_normalized_target_error"] <= 1.05 * v["wind_specific"]["own_history"]["feasible"]["RMS_normalized_target_error"] for v in yaw.values())
        gate[cid] = {"negative_pitch_when_needed_positive_ticks": cp["negative_output_with_positive_needed_ticks"],
            "pitch_RMS_not_worse": cp["RMS_normalized_target_error"] <= bp["RMS_normalized_target_error"],
            "yaw_each_feasible_window_no_more_than_5percent_worse": yaw_ok,
            "passes": cp["negative_output_with_positive_needed_ticks"] == 0 and cp["RMS_normalized_target_error"] <= bp["RMS_normalized_target_error"] and yaw_ok}
    verification = {"all_arrays_finite": all(np.isfinite(v).all() for v in all_arrays.values()),
        "protected_size_mtime_unchanged": before == snapshot(), "zero_fits": True, "free_integrations": 0,
        "zero_API_or_web_calls": True, "controller_hz": 48, "plant_substeps_unchanged": 4, "clock_shift_s": 0.,
        "case_checks": {cid: r["checks"] for cid, r in reports.items()},
        "specific_net_equivalent_error_max_abs_G": max(v["error_G"] for r in reports.values() for v in r["equivalence"].values()),
        "specific_net_equivalent_held_max_abs_rad": max(max(v["one_step_held_rad"], v["own_history_held_rad"]) for r in reports.values() for v in r["equivalence"].values())}
    if not verification["all_arrays_finite"] or not verification["protected_size_mtime_unchanged"] or any(v > 1e-12 for r in reports.values() for v in r["checks"].values()) or verification["specific_net_equivalent_held_max_abs_rad"] > 1e-12:
        raise AssertionError(verification)
    np.savez_compressed(OUT / "conditional_interfaces.npz", **all_arrays)
    doc = {"task": "War Thunder consistent request/feedback interface diagnostics", "cases": reports, "free_validation_gate": gate,
        "verification": verification, "candidate_definitions": {
            "wind_specific": "Current wind-normal specific-force request and feedback; original scalar pitch/yaw actuator mapping.",
            "wind_net": "After originalcap/S110 add rawgravity projection to both request and specificfeedback; no recapping; mathematically same error.",
            "body_specific": "Project original totalworldspecificcommand and currentplant specificforce into body up/right; retain originalbodypitch/yaw angularlaw outputmapping.",
            "body_net": "Project netcommand and currentnetacceleration into body up/right; samegravity removed on both sides; no recapping."},
        "conditions_and_limits": [
            "World command retains wind-forward component of originalgravity-compensatedcap2, then replaces two wind-normal components with finalS110/caprequest. This axialcommand is implied by outerconstruction, not sourceaxialacceleration.",
            "No cap is moved or reapplied after net/body conversion. A cap on a net/body request would be a separate ordering hypothesis, not an equivalent coordinate change.",
            "Controller output still means old-held pitch/yaw input to originalfrozenbodyangularlaw. No hidden outputrotation, newunitgain, bodyforce remapping or independentstation is introduced.",
            "One-step uses identicalbaseline previousmemory/previousK; own-history eachframe evolves separately on fixedreference states. Bodyone-step previouserror/D are baselinewindhistory by design, not a claimednative bodymemory.",
            "24Hz states interpolated on original48Hz clock; 3/5pointneededinputs and activationhalo retained. Conditionalstates enter diagnostics only; no sourcecontrols/feedback/futurederivative is used in runtimecandidate transform.",
            "Early specific/net roundoff doesnot alter the physical error; permissions/impossiblemasks are inherited from unchangedcurrentcontroller.",
            "Earlier2.57 measured-feedback/orangeplant and forceframe tests do not establish the current2.59 interface; no duplicate oldexperiment run.",
            "Conditional gate: clearpitchnegativeoutput wherepositive needed and no pitchRMSworsening; yawfeasible windowRMS no morethan5percent worse. No overallscore or gain scan."],
        "free_validation_candidate": "body_specific" if all(v["passes"] for v in gate.values()) else None}
    dec.audit.slew.write_json(OUT / "results.json", doc)
    print(json.dumps({"verification": verification, "free_validation_gate": gate}), flush=True)


if __name__ == "__main__":
    axial_completed_main() if "--axial-completed" in sys.argv else main()
