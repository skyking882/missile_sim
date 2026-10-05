#!/usr/bin/env python3
"""Read-only prepeak PID lifecycle audit for the War Thunder game candidate.

Uses existing geometry-corrected free runs; performs no forward integration,
fit or field change. Source derivatives enter diagnostics only.
"""
from __future__ import annotations

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
import check_r771_lift_direction as geometry
import analyze_r771_pn_pid_decoupling as dec
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm, sub

OUT = ROOT / "outputs/jari_offaxis/prepeak_pid_interface"
CASES = geometry.CASES
DT = 1 / 48
RAW = ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"


def snapshot():
    paths = [RAW, ROOT / "src/aim120_model/r771_game_control.py", ROOT / "src/aim120_model/single_surface_game.py",
        ROOT / "src/aim120_model/control.py", ROOT / "scripts/fix_r771_jari_control.py",
        ROOT / "scripts/check_r771_lift_direction.py", ROOT / "scripts/check_r771_control_authority.py",
        ROOT / "scripts/check_r771_textbook_pid.py", ROOT / "scripts/check_r771_control_response.py"]
    for folder in (geometry.OUT, dec.OUTPUT, dec.audit.inverse.OUTPUT,
                   ROOT / "outputs/jari_offaxis/textbook_hypotheses/pid", ROOT / "config"):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def case(cid, item, coefficients, inv):
    cfg = item["config"]
    result = json.loads((geometry.OUT / "full_runs" / (cid + ".json")).read_text())
    model = geometry.make_model(cfg, coefficients)
    model.initialize(dataclasses.replace(item["initial"]))
    ticks = result["ticks"]
    t = np.asarray([v["time_s"] for v in ticks])
    samples = {s["time_s"]: s for s in result["samples"]}
    limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    a = {"t": t, "gain": np.asarray([v["gain"] for v in ticks]),
        "start_gain": np.asarray([interpolate_table(float(ts), cfg["guidance"]["flight_time_gain_table"]) for ts in t]),
        "held_rad": np.asarray([v["held_fin_rad"] for v in ticks]),
        "request_G": np.asarray([v["request_g"] for v in ticks]),
        "feedback_G": np.asarray([v["feedback_g"] for v in ticks]),
        "rates_deg_s": np.degrees([[samples[ts]["pitch_rate_rad_s"], samples[ts]["yaw_rate_rad_s"]] for ts in t]),
        "K": np.zeros(len(t)), "local_deg": np.zeros(len(t)), "q_over_qref": np.zeros(len(t)),
        "high": np.asarray([v["nonlinear_control_latched"] for v in ticks], dtype=bool)}
    for k in ("P_rad", "I_provisional_rad", "D_rad", "increment_rad", "normalized_target", "prelimit_normalized"):
        a[k] = np.zeros((len(t), 2))
    checks = {"PID_output_closure_rad": 0., "PID_memory_closure": 0., "flight_gain_vs_request_gate": 0.,
        "binary_gain_error": float(np.min(abs(a["gain"][:, None] - np.array([0., 1.])), axis=1).max()),
        "original_scalar_PID_unchanged": True}
    previous = np.zeros((2, 3))
    active = a["gain"] > 1e-12
    for i, ts in enumerate(t):
        s = samples[ts]
        f = model.evaluate(dec.audit.fix.finite.sample_state(s), float(ts), tuple(s["evaluated_old_held_rad"]))
        speed = norm(sub(tuple(s["velocity_mps"]), model.wind))
        A = (f.qbar / f.mass) / model.q_over_mass0
        c = model.coefficients
        K = c.control_gain * (1 + c.speed_slope * (speed / model.speed0 - 1)) / (A * c.restoring_gain)
        a["K"][i] = K
        a["local_deg"][i] = math.degrees(math.hypot(*f.local_incidence))
        # A labeled D-grade normalization, never applied to the plant/PID.
        qref = .5 * 1.225000018 * (cfg["control"]["base_indicated_speed_kmh"] / 3.6) ** 2
        a["q_over_qref"][i] = f.qbar / qref
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + DT)
        checks["original_scalar_PID_unchanged"] &= not cfg["control"]["pid"].get("schedule") and pid == {
            key: float(cfg["control"]["pid"][key]) for key in ("p", "i", "d", "integral_limit")}
        if not active[i]:
            if ticks[i]["pid_memory"] != [[0., 0., 0.], [0., 0., 0.]] or np.any(a["held_rad"][i]):
                raise AssertionError("Inactive memory/held is not zero")
            continue
        if "slew_scheduled_flight_gain" in ticks[i]:
            checks["flight_gain_vs_request_gate"] = max(checks["flight_gain_vs_request_gate"], abs(a["gain"][i] - ticks[i]["slew_scheduled_flight_gain"]))
        for axis in range(2):
            d = ticks[i]["control_runtime_diagnostics"][axis]
            e = float(a["request_G"][i, axis] - a["feedback_G"][i, axis])
            transported = d["transported_integral_old_rad"]
            increment = pid["i"] * e * DT
            provisional = dec.audit.clip(transported + increment, pid["integral_limit"])
            derivative = previous[axis, 2] + DT / (1e-9 + DT) * ((e - previous[axis, 1]) / DT - previous[axis, 2])
            p, dd = pid["p"] * e, pid["d"] * derivative
            natural = p + provisional + dd
            cap = d["effective_old_held_limit_rad"]
            expected = a["gain"][i] * dec.audit.clip(natural, cap)
            next_i = transported if d["antiwindup_blocked"] else provisional
            next_memory = np.asarray((next_i, e, derivative))
            checks["PID_output_closure_rad"] = max(checks["PID_output_closure_rad"], abs(expected - a["held_rad"][i, axis]))
            checks["PID_memory_closure"] = max(checks["PID_memory_closure"], float(np.max(abs(next_memory - np.asarray(ticks[i]["pid_memory"][axis])))))
            for key, value in (("P_rad", p), ("I_provisional_rad", provisional), ("D_rad", dd), ("increment_rad", increment),
                ("normalized_target", K * a["held_rad"][i, axis] / limits[axis]), ("prelimit_normalized", K * natural / limits[axis])):
                a[key][i, axis] = value
            previous[axis] = next_memory
    first_active = int(np.flatnonzero(active)[0])
    early = active & (t <= .7 + 1e-9)
    yaw_peak = int(np.flatnonzero(early)[np.argmax(abs(a["rates_deg_s"][early, 1]))])
    prepeak = active & (t <= t[yaw_peak] + 1e-9)
    before_latch = active & ~a["high"]
    source_t = inv[cid + "__t"]
    a["source_t"] = source_t
    a["source_control_normalized"] = inv[cid + "__source_controls_pitch_yaw"]
    a["source_rates_deg_s"] = np.degrees(inv[cid + "__rates_rad_s"])
    for width in (3, 5):
        a[str(width) + "__source_inverse_normalized"] = inv[cid + f"__{width}__u_norm_req"]
    rows = []
    for when in (t[first_active], .3958333333333333, .4166666666666667, .5, t[yaw_peak], .5833333333333333):
        i = int(np.argmin(abs(t - when)))
        rows.append({"time_s": float(t[i]), "gain": float(a["gain"][i]), "high_latch": bool(a["high"][i]),
            "local_incidence_deg": float(a["local_deg"][i]), "yaw_rate_deg_s": float(a["rates_deg_s"][i, 1]),
            "q_over_qref_D_hypothesis": float(a["q_over_qref"][i]), "K": float(a["K"][i]),
            "pitch_yaw": {key: a[key][i].tolist() for key in ("request_G", "feedback_G", "P_rad", "I_provisional_rad", "D_rad", "held_rad", "normalized_target")}})
    exceeding = before_latch & (abs(a["normalized_target"][:, 1]) > 1 + 1e-12)
    gain_different = abs(a["gain"] - a["start_gain"]) > 1e-12
    report = {"checks": checks, "raw_PID": cfg["control"]["pid"], "raw_flight_gain_table": cfg["guidance"]["flight_time_gain_table"],
        "first_active_tick_s": float(t[first_active]), "first_active_start_gain_tick_s": float(t[np.flatnonzero(a["start_gain"] > 1e-12)[0]]),
        "gain_end_vs_start_different_ticks": t[gain_different].tolist(),
        "yaw_first_peak_in_0p2917_0p7": {"time_s": float(t[yaw_peak]), "rate_deg_s": float(a["rates_deg_s"][yaw_peak, 1])},
        "first_high_latch_s": float(t[np.flatnonzero(a["high"])[0]]) if a["high"].any() else None,
        "pre_latch_yaw_target_above_one": {"first_s": float(t[np.flatnonzero(exceeding)[0]]) if exceeding.any() else None,
            "ticks": int(exceeding.sum()), "max_normalized": float(abs(a["normalized_target"][before_latch, 1]).max())},
        "prepeak_unclipped_yaw_PID_term_tick_sums_rad": {key: float(a[key][prepeak, 1].sum()) for key in ("P_rad", "I_provisional_rad", "D_rad", "held_rad")},
        "representative_ticks": rows}
    plot(cid, a)
    return a, report


def plot(cid, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.3))
    axes[0, 0].plot(t, a["rates_deg_s"][:, 1], label="Geometry free yaw rate")
    axes[0, 0].plot(a["source_t"], a["source_rates_deg_s"][:, 1], "o", ms=2, mfc="none", label="JARI exported rate")
    axes[0, 0].set_ylabel("Yaw rate (deg/s)")
    for key, label in (("P_rad", "P"), ("I_provisional_rad", "I provisional"), ("D_rad", "D"), ("held_rad", "Held")):
        axes[0, 1].plot(t, a[key][:, 1] * a["K"] / .698132, label=label)
    axes[0, 1].set_ylabel("Yaw PID term × K / raw finsAoa")
    axes[0, 2].plot(t, a["normalized_target"][:, 1], label="Free normalized target")
    for width, style in ((3, ":"), (5, "--")):
        # Noncausal and evaluated on reference states, not free runtime.
        valid = a["source_t"] > 14 / 48 + (width // 2) / 24 + 1e-12
        axes[0, 2].plot(a["source_t"], np.where(valid, a[str(width) + "__source_inverse_normalized"][:, 1], np.nan), style, label=f"Source-state inverse {width}pt")
    axes[0, 2].plot(a["source_t"], a["source_control_normalized"][:, 1], "o", ms=2, label="Export controls (units unknown)")
    axes[0, 2].axhline(1, color="black", lw=.8)
    axes[0, 2].set_ylabel("Yaw target / raw finsAoa")
    for axis, label in ((0, "Pitch"), (1, "Yaw")):
        axes[1, 0].plot(t, a["request_G"][:, axis] - a["feedback_G"][:, axis], label=label)
    axes[1, 0].set_ylabel("Raw PID error (G)")
    axes[1, 1].step(t, a["gain"], where="post", label="End sampled gain")
    axes[1, 1].step(t, a["start_gain"], where="post", ls="--", label="Start sampled gain")
    axes[1, 1].set_ylabel("Flight gain")
    axes[1, 2].plot(t, a["local_deg"], label="Current local norm (deg)")
    axes[1, 2].axhline(math.degrees(dec.audit.fix.finite.ALPHA_S), color="black", lw=.8, label="Existing rolloff onset")
    axes[1, 2].set_ylabel("Local incidence (deg)")
    for ax in axes.flat:
        ax.set(xlabel="Original time (s)", xlim=(.25, .75))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(cid + ": existing geometry free run, frozen raw fields; no new free integration\nPID audit and source-state inverse are diagnostic; exported control units remain unidentified", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / (cid + "_prepeak.png"), dpi=160)
    plt.close(fig)


def main():
    before = snapshot()
    raw = json.loads(RAW.read_text())["rocket"]["guidance"]["guidanceAutopilot"]
    inputs = dec.audit.inverse.source_inputs()
    coefficients = dec.audit.fix.drag.DragCoefficients(**json.loads((dec.audit.fix.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    inv = np.load(dec.audit.inverse.OUTPUT / "inverse_inputs.npz")
    prior = json.loads((ROOT / "outputs/jari_offaxis/textbook_hypotheses/pid/results.json").read_text())
    old_response = json.loads((ROOT / "outputs/r771_replay_batch_20260928/no_loft_control/control_response/results.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    reports, arrays = {}, {}
    for cid in CASES:
        a, r = case(cid, inputs[cid], coefficients, inv)
        reports[cid] = r
        arrays.update({cid + "__" + k: v for k, v in a.items()})
    dedup = {"authority_2x2": "Every target1 cell was conditional on high latch; no prior always-active target1 test.",
        "textbook_PID": {cid: {"baseline_startup_yaw_RMS": c["methods"]["5_point"]["startup"]["axes"]["yaw"]["baseline_all_nonhalo"]["RMS_normalized_target_error"],
            "candidate_startup_yaw_RMS": {k: v["all_nonhalo_including_unattainable"]["RMS_normalized_target_error"] for k, v in c["methods"]["5_point"]["startup"]["axes"]["yaw"]["candidates"].items()}}
            for cid, c in prior["cases"].items()},
        "control_response": {"source_version": old_response["source_version"], "selected_tau_s": old_response["selected_tau_s"],
            "scope": old_response["scope"], "distinct_from_current_free_candidate": True},
        "baseIndSpeed": "Existing B0-B3 H2 script excludes R-77-1; qref/q textbook mapping was not executed. Current single-surface uses neither q/qref nor qref/q."}
    checks = {"all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values()),
        "protected_size_mtime_unchanged": before == snapshot(), "new_free_integrations": 0, "fits": 0,
        "web_API_calls": 0, "new_coefficients": 0, "controller_hz": 48, "plant_substeps": 4}
    if not checks["all_arrays_finite"] or not checks["protected_size_mtime_unchanged"]:
        raise AssertionError(checks)
    for r in reports.values():
        if any(v > 1e-11 for k, v in r["checks"].items() if not isinstance(v, bool)) or not r["checks"]["original_scalar_PID_unchanged"]:
            raise AssertionError(r["checks"])
    doc = {"task": "War Thunder game-only prepeak raw PID/gain lifecycle audit", "raw_guidanceAutopilot": raw,
        "chain": ["timeToHitToGain scales PN once", "Horizontal half-relative PN, gated loft and S110 create current request",
            "timeToGain is queried in S110 only for an active/inactive decision", "The same end-sampled timeToGain multiplies total clipped PID output once",
            "PID memory is zero before activation; provisional I enters output before saved I conditional antiwindup",
            "The scalar raw PID dictionary has no schedule; end-vs-start PID sampling does not change coefficients",
            "baseIndSpeed is preserved but not used by the current single-surface plant/controller"],
        "cases": reports, "deduplication": dedup, "verification": checks,
        "recommended_single_candidate": {"hypothesis": "Apply existing target1 output cap to every active tick; retain original high-only I coordinate transport",
            "reason": "Free 60 target exceeds one from 0.416667 s before first rate peak; original high-only cap starts 0.583333 s after peak",
            "new_numeric_constants": 0, "exact_implementation_bug_identified": False, "evidence_grade": "D", "root_selection_required_before_free": True},
        "limits": ["Source 3/5 derivatives are noncausal conditional diagnostics, not runtime inputs.",
            "Tick sums describe PID terms, not exact angular impulses through changing plant coefficients.",
            "A normalized target bound of one is an existing model hypothesis, not independently recovered native-game cap.",
            "Gain placement with binary gain and inactive-zero PID is algebraically indistinguishable on this clock; changing start/end sampling is an actual activation-time hypothesis."]}
    np.savez_compressed(OUT / "audit.npz", **arrays)
    dec.audit.fix.slew.write_json(OUT / "results.json", doc)
    print(json.dumps({"output": str(OUT), "verification": checks, "pre_latch_yaw_target_above_one": {cid: r["pre_latch_yaw_target_above_one"] for cid, r in reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
