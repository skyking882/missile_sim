#!/usr/bin/env python3
"""Fixed textbook execution hypotheses for the War Thunder game controller.

Predeclared candidates: I output before update and D on measurement. Each is
evaluated from baseline previous memory for one tick, and with its own history
on the identical reference-state grid. No fit, network or plant integration.
"""
from __future__ import annotations
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import analyze_r771_pn_pid_decoupling as decoupling
from aim120_model.control import scheduled_pid_values
from aim120_model.r771_game_control import R771GameControl

OUT = ROOT / "outputs/jari_offaxis/textbook_hypotheses/pid"
OTHER_WORKER_SCRIPT = ROOT / "scripts/check_r771_textbook_pn.py"
OTHER_WORKER_OUT = ROOT / "outputs/jari_offaxis/textbook_hypotheses/pn"
CANDIDATES = ("I_phase", "D_measurement")
FIELDS = ("P_rad", "I_used_rad", "I_provisional_rad", "D_rad", "prelimit_rad", "postlimit_rad", "held_rad",
    "next_integral_rad", "transported_integral_rad", "error_increment_rad", "antiwindup_blocked",
    "derivative_G_s", "output_closure_rad", "nextI_closure_rad")


def candidate_step(kind, controller, axis, ts, error, feedback, previous_feedback, memory, pid, gain, k, nonlinear, limit):
    terms = decoupling.pid_terms(controller, axis, memory, error, pid, gain, k, nonlinear, limit)
    derivative = terms["derivative_G_s"]
    if kind == "D_measurement":
        raw = -(feedback - previous_feedback) / decoupling.DT
        derivative = memory[2] + decoupling.DT / (1e-9 + decoupling.DT) * (raw - memory[2])
    used_i = terms["transported_integral_rad"] if kind == "I_phase" else terms["I_provisional_rad"]
    p = pid["p"] * error; d = pid["d"] * derivative
    natural = p + used_i + d
    post = decoupling.audit.clip(natural, terms["effective_limit_rad"])
    blocked = abs(post) >= .99 * terms["effective_limit_rad"] and error * post > 0
    next_i = terms["transported_integral_rad"] if blocked else terms["I_provisional_rad"]
    held = gain * post; next_memory = (next_i, error, derivative)
    # Candidate metadata follows the unchanged latch and K coordinate rule.
    controller.nonlinear_latched = terms["high_mode"]
    if controller.nonlinear_latched:
        controller.previous_high_k[axis] = k
    controller.last_time[axis] = ts; controller.cached[axis] = (held, next_memory)
    direct_i = decoupling.audit.clip(terms["transported_integral_rad"] + pid["i"] * error * decoupling.DT, pid["integral_limit"])
    expected_next_i = terms["transported_integral_rad"] if blocked else direct_i
    result = {"P_rad": p, "I_used_rad": used_i, "I_provisional_rad": terms["I_provisional_rad"], "D_rad": d,
        "prelimit_rad": natural, "postlimit_rad": post, "held_rad": held, "next_integral_rad": next_i,
        "transported_integral_rad": terms["transported_integral_rad"], "error_increment_rad": terms["error_increment_rad"],
        "antiwindup_blocked": blocked, "derivative_G_s": derivative,
        "output_closure_rad": held - gain * decoupling.audit.clip(p + used_i + d, terms["effective_limit_rad"]),
        "nextI_closure_rad": next_i - expected_next_i}
    return held, next_memory, result


def errors_stats(output, needed, baseline, factor, mask):
    diff = factor * (output - needed); previous = factor * (baseline - needed)
    return {"ticks": int(mask.sum()), "RMS_normalized_target_error": decoupling.audit.inverse.rms(diff[mask]) if mask.any() else None,
        "bias_normalized_target_error": float(diff[mask].mean()) if mask.any() else None,
        "closer_ticks": int(np.sum(mask & (abs(diff) < abs(previous) - 1e-12))),
        "farther_ticks": int(np.sum(mask & (abs(diff) > abs(previous) + 1e-12)))}


def case_analysis(cid, cached, cfg):
    t = cached["t"]; gains = cached["flight_gain"]; limits = cached["old_held_limits_rad"]
    req = cached["final_request_G"]; fb = cached["plant_feedback_G"]; k = cached["K"]
    nonlinear = cached["nonlinear_force_active"]; error = req - fb
    active = gains > 1e-12
    controllers = {kind: R771GameControl() for kind in CANDIDATES}
    memories = {kind: [(0., 0., 0.), (0., 0., 0.)] for kind in CANDIDATES}
    baseline_controller = R771GameControl(); baseline_memory = [(0., 0., 0.), (0., 0., 0.)]
    arrays = {key: cached[key] for key in ("t", "flight_gain", "K", "old_held_limits_rad", "final_request_G", "plant_feedback_G",
        "gain_transition_derivative_halo", "pid__held_rad", "next_current_memory", "3__needed_held_rad", "5__needed_held_rad",
        "3__needed_normalized_target", "5__needed_normalized_target", "3__given_memory_inverse_status", "5__given_memory_inverse_status")}
    for kind in CANDIDATES:
        for mode in ("one_step", "own_history"):
            for field in FIELDS:
                arrays[kind + "__" + mode + "__" + field] = np.zeros((len(t), 2))
            arrays[kind + "__" + mode + "__previous_memory"] = np.zeros((len(t), 2, 3))
            arrays[kind + "__" + mode + "__next_memory"] = np.zeros((len(t), 2, 3))
    replayed_held = np.zeros_like(req); replayed_memory = np.zeros((len(t), 2, 3))
    for i, ts in enumerate(t):
        if gains[i] <= 1e-12:
            continue
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + decoupling.DT)
        for axis in range(2):
            previous_fb = float(fb[i - 1, axis]) if i else float(fb[i, axis])
            for kind in CANDIDATES:
                for mode in ("one_step", "own_history"):
                    c = copy.deepcopy(baseline_controller) if mode == "one_step" else controllers[kind]
                    mem = baseline_memory[axis] if mode == "one_step" else memories[kind][axis]
                    tag = kind + "__" + mode + "__"
                    arrays[tag + "previous_memory"][i, axis] = mem
                    held, next_mem, terms = candidate_step(kind, c, axis, float(ts), float(error[i, axis]), float(fb[i, axis]),
                        previous_fb, mem, pid, float(gains[i]), float(k[i]), bool(nonlinear[i]), float(limits[axis]))
                    for field in FIELDS:
                        arrays[tag + field][i, axis] = terms[field]
                    arrays[tag + "next_memory"][i, axis] = next_mem
                    if mode == "own_history":
                        memories[kind][axis] = next_mem
            replayed_held[i, axis], baseline_memory[axis] = baseline_controller.update_axis(axis, float(ts), float(error[i, axis]),
                baseline_memory[axis], pid, limits[axis], gains[i], float(k[i]), bool(nonlinear[i]))
            replayed_memory[i, axis] = baseline_memory[axis]
    checks = {"baseline_held_exact_max_abs_rad": float(np.max(abs(replayed_held - cached["pid__held_rad"]))),
        "baseline_memory_exact_max_abs": float(np.max(abs(replayed_memory - cached["next_current_memory"]))) }
    for kind in CANDIDATES:
        for mode in ("one_step", "own_history"):
            tag = kind + "__" + mode + "__"
            checks[tag + "output_closure_max_abs_rad"] = float(np.max(abs(arrays[tag + "output_closure_rad"])))
            checks[tag + "nextI_closure_max_abs_rad"] = float(np.max(abs(arrays[tag + "nextI_closure_rad"])))
            if mode == "one_step":
                checks[tag + "same_previous_memory_max_abs"] = float(np.max(abs(arrays[tag + "previous_memory"][active] - cached["previous_current_memory"][active])))
    report = {"original_PID_entire_dict": cfg["control"]["pid"], "original_flight_gain_table": cfg["guidance"]["flight_time_gain_table"],
        "checks": checks, "methods": {}}
    windows = (*decoupling.WINDOWS, ("full_active_0p2917_3p5", 14 / 48, 3.5))
    for width in (3, 5):
        status = cached[str(width) + "__given_memory_inverse_status"]
        need = cached[str(width) + "__needed_held_rad"]
        report["methods"][str(width) + "_point"] = {}
        for name, lo, hi in windows:
            window = active & decoupling.audit.inverse.window_mask(t, lo, hi)
            record = {"window_s": [lo, hi], "ticks": int(window.sum()), "axes": {}}
            for axis, label in enumerate(("pitch", "yaw")):
                nonhalo = window & ~cached["gain_transition_derivative_halo"]
                feasible = nonhalo & (status[:, axis] == 1)
                factor = k / limits[axis]; baseline = cached["pid__held_rad"][:, axis]
                entry = {"impossible_nonhalo_ticks": int(np.sum(nonhalo & (status[:, axis] == -1))),
                    "baseline_feasible_nonhalo": errors_stats(baseline, need[:, axis], baseline, factor, feasible),
                    "baseline_all_nonhalo": errors_stats(baseline, need[:, axis], baseline, factor, nonhalo), "candidates": {}}
                for kind in CANDIDATES:
                    for mode in ("one_step", "own_history"):
                        tag = kind + "__" + mode + "__"; output = arrays[tag + "held_rad"][:, axis]
                        entry["candidates"][kind + "__" + mode] = {
                            "feasible_nonhalo": errors_stats(output, need[:, axis], baseline, factor, feasible),
                            "all_nonhalo_including_unattainable": errors_stats(output, need[:, axis], baseline, factor, nonhalo),
                            "output_change_RMS_normalized": decoupling.audit.inverse.rms((factor * (output - baseline))[window]),
                            "negative_output_ticks": int(np.sum(window & (output < 0))),
                            "antiwindup_ticks": int(np.sum(arrays[tag + "antiwindup_blocked"][window, axis])),
                            "end_target_integral_normalized": float(k[window][-1] * arrays[tag + "next_integral_rad"][window, axis][-1] / limits[axis])}
                record["axes"][label] = entry
            report["methods"][str(width) + "_point"][name] = record
    return arrays, report


def plot_case(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]; active = a["flight_gain"] > 0
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for row, mode in enumerate(("one_step", "own_history")):
        for axis, label in enumerate(("Pitch", "Yaw")):
            ax = axes[row, axis]; factor = a["K"] / a["old_held_limits_rad"][axis]
            valid = active & ~a["gain_transition_derivative_halo"]
            for width, style in ((3, ":"), (5, "-")):
                ax.plot(t, np.where(valid, a[str(width) + "__needed_normalized_target"][:, axis], np.nan), color="#136d72", ls=style, label=f"Frozen-plant needed input, {width}pt")
            ax.plot(t, factor * a["pid__held_rad"][:, axis], color="#7d307c", lw=1.7, label="Current baseline")
            for kind, color, style in (("I_phase", "#ca792f", "--"), ("D_measurement", "#537cac", "-.")):
                ax.plot(t, factor * a[kind + "__" + mode + "__held_rad"][:, axis], color=color, ls=style, label=kind)
            impossible = active & (a["5__given_memory_inverse_status"][:, axis] == -1)
            if impossible.any():
                ax.fill_between(t, 0, 1, where=impossible, transform=ax.get_xaxis_transform(), color="#e8b4b4", alpha=.2, label="Above unchanged current authority")
            ax.axhline(0, color="#aaa", lw=.6); ax.axvline(.7, color="#aaa", lw=.6); ax.axvline(2.3, color="#aaa", lw=.6)
            ax.axvspan(14 / 48, 14 / 48 + 2 / 24, color="#d8b477", alpha=.1)
            ax.set_title(label + ": " + ("one tick, identical previous baseline memory" if mode == "one_step" else "candidate's own history, fixed reference states"), fontsize=10)
            ax.set_ylabel("Equivalent target / raw finsAoa"); ax.set_xlim(14 / 48, 3.5); ax.grid(alpha=.16); ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Original clock (s)")
    fig.suptitle(cid + ": fixed raw PID, textbook I phase / derivative-signal hypotheses\nConditional diagnostics; no free trajectory, no field semantics identified", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .935)); fig.savefig(OUT / (cid + "_pid_hypotheses.png"), dpi=160); plt.close(fig)


def snapshot():
    folders = (ROOT / "src", ROOT / "config", ROOT / "docs", ROOT / "scripts", ROOT / "data/raw", ROOT / "outputs/jari_offaxis")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for folder in folders if folder.exists()
        for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and OUT not in p.parents
        and OTHER_WORKER_OUT not in p.parents and p not in (Path(__file__).resolve(), OTHER_WORKER_SCRIPT)}


def main():
    before = snapshot(); sources = json.loads((OUT.parent / "sources.json").read_text())
    inputs = decoupling.audit.inverse.source_inputs(); z = np.load(decoupling.OUTPUT / "pn_pid_timeseries.npz")
    reports = {}; saved = {}; OUT.mkdir(parents=True, exist_ok=True)
    for cid in decoupling.CASES:
        cached = {key[len(cid) + 2:]: z[key] for key in z.files if key.startswith(cid + "__")}
        a, r = case_analysis(cid, cached, inputs[cid]["config"])
        reports[cid] = r; saved.update({cid + "__" + key: value for key, value in a.items()}); plot_case(cid, a)
    profiles = json.loads(decoupling.audit.comparison.SNAPSHOT.read_text())
    profile = profiles["2.59.0.27/su_r_77_1"]
    metadata = {"raw_PID": profile["control"]["pid"], "control_parameter_sources": profile["control"]["parameter_sources"],
        "raw_baseIndSpeed": profile["control"]["base_indicated_speed_kmh"], "raw_finsAoa_rad": profile["aerodynamics"]["fin_aoa_limit_rad"],
        "finsAoa_source": profile["aerodynamics"]["parameter_sources"]["fin_aoa_limit_rad"], "baseIndSpeed_mapping": [
            {"role": "raw numeric field", "units": "not identified; local schema labels km/h", "placement": "guidanceAutopilot.baseIndSpeed", "evidence": "numeric source datamine; solver meaning D"},
            {"role": "reference dynamic pressure", "units": "km/h candidate ->1800/3.6=500m/s; qref=.5*1.225000018*500^2=153125.00225Pa", "placement": "only a candidate normalization", "evidence": "D; IAS approximately EAS is unproven at current compressible speeds"},
            {"role": "q output multiplier", "units": "qref/q dimensionless", "placement": "natural PID sum before existing clamp", "evidence": "D; not executed because existing transport preservesK*I, external multiplier changes output-integral coordinate toK*(qref/q)*I"},
            {"role": "other scheduling/reference role", "units": "depends on formula", "placement": "gain-table variable or feedback crossover etc", "evidence": "D, nonunique; no original extra gains to implement"}],
        "interface_ambiguity": "For unsaturated old-held dynamics, scaling error bys_e and output bys_u makes effective gains Ku*s_u*s_e*(Kp,Ki,Kd). Raw numbers alone do not identify s_e/s_u/Ku or saturation placement. FreezingKu fixes a conditional test, not native units.",
        "prior_experiments_not_repeated": [
            {"path": str(ROOT / "outputs/jari_offaxis/trim_target/results.json"), "rejected_specific_mapping": "SI G0 error -> normalized[-1,1] -> rawfinsAoa target; removes frozenKu/slope", "H110_full_speed_RMSE_old_new_kmh": [7.046760538783545, 12.919015170627457], "H110_full_alpha_RMSE_old_new_deg": [1.6471037792042018, 3.4259254905472774]},
            {"path": str(ROOT / "outputs/r77_1_pid_angle_output.json"), "boundary": "2.57FocusPlant mapping-only/AW angle-output comparison; not current2.59single-surface"},
            {"path": str(ROOT / "scripts/compare_base_ind_speed_candidates.py"), "boundary": "H2Simulator B0–B3 script; declared MISSILES excludesR77-1; B1 changesplant, B2 matched authority andcommand, B3 multipliesq/qref notinverse; no currentfrozen-interface cache found"}]}
    finite = all(np.isfinite(v).all() for v in saved.values())
    verification = {"all_arrays_finite": bool(finite), "protected_size_mtime_unchanged": before == snapshot(),
        "zero_fits": True, "zero_free_integrations": True, "zero_API_or_web_calls": True,
        "controller_hz": 48, "plant_substeps_unchanged": 4, "clock_shift_s": 0.,
        "checks": {cid: r["checks"] for cid, r in reports.items()}}
    if not finite or not verification["protected_size_mtime_unchanged"] or any(v > 1e-12 for r in reports.values() for v in r["checks"].values()):
        raise AssertionError(verification)
    np.savez_compressed(OUT / "pid_candidates.npz", **saved)
    doc = {"task": "War Thunder conditional fixed textbook PID/interface hypotheses", "source_ids_used": ["ASTROM_MURRAY_PID", "CORNELL_DYNAMICS", "IITB_AIRSPEED", "DIONNE_SCHEDULING"],
        "cases": reports, "metadata_and_deduplication": metadata, "verification": verification, "conditions_and_limits": [
            "I_phase uses transportedpreviousI in current output; nextI updates afterward with sameKi/error/conditionalAW/rawlimit/flightGain.",
            "D_measurement uses-feedbackdifference/DT with originalKd and1e-9 derivativefilter, no new coefficient. Previous measurement is preceding reference tick, including inactive tick before activation.",
            "One-stepD uses baselinepreviousfilteredD, meaning a signal switch at this tick, not historicalD-measurement memory. Own-historyD startszero derivative at activation with precedingreference measurement; histories are distinct.",
            "All candidates receive identicalreference states/request/plantfeedback and originalclock; one-stepclones previouscurrent memory, recursivehistorybelongs only to each candidate.",
            "Unchangedhigh-angleauthority makes60platformneededoutput impossible. Feasiblemask separates this from candidate residuals; no neededrequest is fabricated.",
            "3/5point source derivatives and activationhalo remain; same-state outputcomparison doesnot validate nativegame controller or freeclosedloop performance.",
            "Rawcoefficients preserved doesnot mean effectivegain preserved if units/outputscale/execution order changes; only specific mappedcandidates can be excluded.",
            "No pressurecandidate, PIDcombination, feedbackmix, extra rateinnerloop, fit, or plant integration was executed."]}
    decoupling.audit.slew.write_json(OUT / "results.json", doc)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
