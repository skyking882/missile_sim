#!/usr/bin/env python3
"""Fixed 2x2 output-limit / integral-coordinate game hypotheses; no fitting.

Legacy/raw and transport/target1 are existing algorithms. Only the other two
cells are new hypotheses. Reference-state calculations are conditional.
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
import analyze_r771_pn_pid_decoupling as split
from fit_r771_jari_finite_force import comparison, drag, slew, harness
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients

OUT = ROOT / "outputs/jari_offaxis/control_authority"
CASES = audit.inverse.CASES
CELLS = {"legacy_raw": (False, False), "transport_target1": (True, True),
         "transport_raw": (True, False), "legacy_target1": (False, True)}
DT = 1 / 48
TERM_KEYS = ("P_rad", "I_provisional_rad", "D_rad", "prelimit_rad", "postlimit_rad",
    "held_rad", "previous_integral_rad", "transported_integral_rad", "next_integral_rad",
    "transport_ratio", "transport_delta_rad", "error_increment_rad", "update_clip_delta_rad",
    "antiwindup_delta_rad", "antiwindup_blocked", "effective_limit_rad", "derivative_G_s", "high_mode")
WINDOWS = (*audit.WINDOWS, ("active", 14 / 48, 3.5))


class AuthorityControl(R771GameControl):
    """Script-local 2x2 mechanism; raw PID/plant interface remain unchanged."""
    def __init__(self, transport, target_cap):
        super().__init__()
        self.transport = transport; self.target_cap = target_cap; self.last_terms = [None, None]

    def update_axis(self, axis, time_s, error_g, memory, pid, old_held_limit_rad, gain,
                    target_per_old_held, nonlinear_force_active):
        if abs(time_s * 48 - round(time_s * 48)) > 1e-8:
            raise ValueError("48Hz update only")
        if self.last_time[axis] is not None and time_s <= self.last_time[axis] + 1e-12:
            if abs(time_s - self.last_time[axis]) < 1e-12:
                return self.cached[axis]
            raise ValueError("backwards controller clock")
        if gain <= 1e-12:
            return 0., memory
        k = float(target_per_old_held); limit = float(old_held_limit_rad)
        if not (math.isfinite(k) and k > 0 and 0 < gain <= 1):
            raise ValueError("invalid frozen interface input")
        high = self.nonlinear_latched or bool(nonlinear_force_active)
        old_i, prev_error, prev_d = memory
        prev_k = self.previous_high_k[axis]
        ratio = prev_k / k if self.transport and high and prev_k is not None else 1.
        raw_transport = old_i * ratio
        integral = audit.clip(raw_transport, pid["integral_limit"]) if self.transport and high else old_i
        effective = min(limit, limit / k) if self.target_cap and high else limit
        increment = pid["i"] * error_g * DT
        provisional = audit.clip(integral + increment, pid["integral_limit"])
        derivative = prev_d + DT / (1e-9 + DT) * ((error_g - prev_error) / DT - prev_d)
        p = pid["p"] * error_g; d = pid["d"] * derivative
        natural = p + provisional + d; post = audit.clip(natural, effective)
        blocked = abs(post) >= .99 * effective and error_g * post > 0
        next_i = integral if blocked else provisional
        held = gain * post; result = held, (next_i, error_g, derivative)
        self.nonlinear_latched = high
        if high: self.previous_high_k[axis] = k
        self.last_time[axis] = time_s; self.cached[axis] = result
        self.last_terms[axis] = {"P_rad": p, "I_provisional_rad": provisional, "D_rad": d,
            "prelimit_rad": natural, "postlimit_rad": post, "held_rad": held,
            "previous_integral_rad": old_i, "transported_integral_rad": integral,
            "next_integral_rad": next_i, "transport_ratio": ratio,
            "transport_delta_rad": raw_transport - old_i, "error_increment_rad": increment,
            "update_clip_delta_rad": provisional - integral - increment,
            "antiwindup_delta_rad": next_i - provisional, "antiwindup_blocked": blocked,
            "effective_limit_rad": effective, "derivative_G_s": derivative, "high_mode": high}
        self.diagnostics[axis] = {"time_s": time_s, "axis": axis, "error_g": error_g,
            "K_target_per_old_held": k, "nonlinear_latched": high,
            "integral_transport_ratio": ratio, "transported_integral_old_rad": integral,
            "integral_target_rad": k * next_i, "P_target_rad": k * p, "D_target_rad": k * d,
            "effective_old_held_limit_rad": effective, "target_candidate_limit_rad": limit,
            "antiwindup_blocked": blocked, "held_old_rad": held,
            "effective_target_rad": k * held, "normalized_target_at_tick": k * held / limit,
            "gain": gain, "extra_target_cap": self.target_cap, "transport_integral": self.transport}
        return result


def snapshot():
    paths = list((ROOT / "src/aim120_model").glob("*.py"))
    paths += [Path(m.__file__) for m in (audit, split, audit.fix, audit.inverse, audit.fix.finite, drag, slew, harness, comparison)]
    paths += [comparison.SNAPSHOT, comparison.FROZEN, comparison.DEFAULT_MANIFEST,
        drag.OUTPUT/"fit.json", drag.OUTPUT/"trajectories.npz", slew.OUTPUT/"trajectories.npz",
        ROOT/"data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx",
        harness.NATIVE_BASE/"source_payload.json"]
    for folder in (ROOT / "config", ROOT / "configs", ROOT / "data/raw", ROOT / "outputs/jari_offaxis/reference",
        audit.fix.OUTPUT, audit.fix.finite.OUTPUT, audit.OUTPUT, split.OUTPUT, audit.inverse.OUTPUT,
        ROOT / "outputs/jari_offaxis/textbook_hypotheses"):
        if folder.exists(): paths.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(set(paths)) if p.is_file()}


def conditional(cid, item, z, cache, legacy_cache, coefficients):
    cfg = item["config"]; ref = item["ref"]
    t = np.arange(math.ceil((3.5 + 1e-6) * 48) + 1) / 48
    model = audit.fix.make_model(cfg, coefficients); model.initialize(dataclasses.replace(item["initial"]))
    fixed = (model.coefficients == SurfaceCoefficients() and model.drag_coefficients == coefficients
        and model.relative_blend == .5 and model.request_slew_g_s == 110.
        and [model.amplitude_coefficient,model.transition_coefficient] == [.75,1.3]
        and model.plant_substeps == 4 and model.controller_hz == 48 and cfg["guidance"]["pn_gain"] == 6.)
    if not fixed: raise AssertionError("Frozen plant/outer/clock fields changed")
    limits = np.radians([cfg["aerodynamics"][key] for key in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    controllers = {name: AuthorityControl(*flags) for name, flags in CELLS.items()}
    memories = {name: [(0., 0., 0.), (0., 0., 0.)] for name in CELLS}
    original = R771GameControl(); original_memory = [(0., 0., 0.), (0., 0., 0.)]
    a = {"t": t, "limits_rad": limits}
    for key in ("request_G", "feedback_G", "error_G"): a[key] = np.zeros((len(t), 2))
    for key in ("K", "flight_gain", "nonlinear_force_active"): a[key] = np.zeros(len(t))
    for name in CELLS:
        for key in TERM_KEYS: a[name + "__" + key] = np.zeros((len(t), 2))
        a[name + "__previous_memory"] = np.zeros((len(t), 2, 3))
        a[name + "__next_memory"] = np.zeros((len(t), 2, 3))
    checks = {"legacy_raw_algorithm_max_abs_rad": 0., "transport_target1_algorithm_max_abs": 0.}
    for i, ts in enumerate(t):
        state = audit.fix.source_state_at(ref, float(ts)); f = model.evaluate(state, float(ts), (0., 0.))
        request, _ = model.request(state, item["target_at"](float(ts)), float(ts), f)
        k = model.current_control_k; nonlinear = model.current_force_nonlinear
        gain = interpolate_table(float(ts) + DT, cfg["guidance"]["flight_time_gain_table"])
        pid = scheduled_pid_values(cfg["control"]["pid"], float(ts) + DT)
        error = np.asarray(request) - f.feedback_g
        a["request_G"][i] = request; a["feedback_G"][i] = f.feedback_g; a["error_G"][i] = error
        a["K"][i] = k; a["flight_gain"][i] = gain; a["nonlinear_force_active"][i] = nonlinear
        for axis in range(2):
            for name, c in controllers.items():
                previous = memories[name][axis]; a[name + "__previous_memory"][i, axis] = previous
                if gain > 1e-12:
                    held, memory = c.update_axis(axis, float(ts), float(error[axis]), previous, pid, limits[axis], gain, k, nonlinear)
                    terms = c.last_terms[axis]
                    for key in TERM_KEYS: a[name + "__" + key][i, axis] = terms[key]
                    memories[name][axis] = memory
                    if name == "legacy_raw":
                        direct = SingleSurfaceGame._pid(audit._LEGACY_CONTEXT, float(error[axis]), previous, pid, limits[axis], gain)
                        checks["legacy_raw_algorithm_max_abs_rad"] = max(checks["legacy_raw_algorithm_max_abs_rad"],
                            abs(held - direct[0]), max(abs(x-y) for x,y in zip(memory,direct[1])))
                a[name + "__next_memory"][i, axis] = memories[name][axis]
            if gain > 1e-12:
                held, original_memory[axis] = original.update_axis(axis, float(ts), float(error[axis]), original_memory[axis], pid, limits[axis], gain, k, nonlinear)
                checks["transport_target1_algorithm_max_abs"] = max(checks["transport_target1_algorithm_max_abs"],
                    abs(held-a["transport_target1__held_rad"][i,axis]),
                    float(np.max(abs(np.asarray(original_memory[axis])-a["transport_target1__next_memory"][i,axis]))))
    if cid in split.CASES:
        for new, old in (("request_G", "final_request_G"), ("feedback_G", "plant_feedback_G"),
            ("transport_target1__held_rad", "pid__held_rad"), ("transport_target1__next_memory", "next_current_memory")):
            checks["cached_" + old + "_max_abs"] = float(np.max(abs(a[new] - cache[cid + "__" + old])))
    if cid == "jari_259_30":
        checks["cached_legacy30_max_abs_rad"] = float(np.max(abs(a["legacy_raw__held_rad"] - legacy_cache["legacy_held_old_rad"])))
    high = a["transport_target1__high_mode"][:,0].astype(bool); active = a["flight_gain"] > 1e-12
    low = active & ~high
    checks["all_four_before_latch_exact"] = bool(all(np.array_equal(a[name+"__held_rad"][low],a["legacy_raw__held_rad"][low]) for name in CELLS))
    for name in CELLS:
        natural = a[name + "__P_rad"] + a[name + "__I_provisional_rad"] + a[name + "__D_rad"]
        post = np.maximum(-a[name+"__effective_limit_rad"],np.minimum(a[name+"__effective_limit_rad"],natural))
        checks[name + "__output_closure_max_abs_rad"] = float(np.max(abs(a[name+"__held_rad"]-a["flight_gain"][:,None]*post)))
        checks[name + "__raw_cap_max_excess_rad"] = float(np.maximum(abs(a[name+"__held_rad"])-limits,0).max())
        checks[name + "__raw_I_limit_max_excess_rad"] = float(np.maximum(abs(a[name+"__next_integral_rad"])-cfg["control"]["pid"]["integral_limit"],0).max())
    report = {"original_PID": cfg["control"]["pid"], "checks": checks,
        "raw_old_held_limits_rad": limits.tolist(),
        "first_latch_s": float(t[np.flatnonzero(high)[0]]) if high.any() else None, "windows": {}}
    source_t = z[cid+"__t"]
    for width in (3, 5):
        halo = (abs(t-14/48) <= (width//2)/24 + 1e-12)
        valid_edge = (t >= source_t[width//2]) & (t <= source_t[-1-width//2])
        a[str(width)+"__diagnostic_mask"] = active & ~halo & valid_edge
        need = audit.interp(t, source_t, z[cid+f"__{width}__old_held_req_rad"])
        a[str(width)+"__needed_old_rad"] = need
        a[str(width)+"__needed_normalized"] = a["K"][:,None]*need/limits
        for name, lo, hi in WINDOWS:
            valid = a[str(width)+"__diagnostic_mask"] & audit.inverse.window_mask(t,lo,hi)
            rec = {"ticks": int(valid.sum()), "axes": {}}
            for axis, label in enumerate(("pitch","yaw")):
                data = {"needed_normalized_range": [float(a[str(width)+"__needed_normalized"][valid,axis].min()),
                    float(a[str(width)+"__needed_normalized"][valid,axis].max())] if valid.any() else None, "cells": {}}
                for cell in CELLS:
                    held = a[cell+"__held_rad"][:,axis]; factor = a["K"]/limits[axis]
                    diff = factor*(held-need[:,axis]); effective = a[cell+"__effective_limit_rad"][:,axis]*a["flight_gain"]
                    feasible = valid & (abs(need[:,axis]) <= effective+1e-12)
                    data["cells"][cell] = {"RMS_normalized_input_error_all_diagnostic": audit.inverse.rms(diff[valid]) if valid.any() else None,
                        "RMS_normalized_input_error_feasible": audit.inverse.rms(diff[feasible]) if feasible.any() else None,
                        "unattainable_ticks_under_cell_cap": int(np.sum(valid & ~feasible)),
                        "negative_held_ticks": int(np.sum(valid & (held<0))),
                        "wrong_sign_ticks": int(np.sum(valid & (held*need[:,axis]<0))),
                        "antiwindup_ticks": int(np.sum(a[cell+"__antiwindup_blocked"][valid,axis])),
                        "instantaneous_output_clipping_ticks": int(np.sum(valid & (abs(a[cell+"__prelimit_rad"][:,axis]-a[cell+"__postlimit_rad"][:,axis])>1e-12))),
                        "end_I_normalized": float(factor[valid][-1]*a[cell+"__next_integral_rad"][valid,axis][-1]) if valid.any() else None,
                        "held_normalized_range": [float((factor*held)[valid].min()),float((factor*held)[valid].max())] if valid.any() else None}
                rec["axes"][label] = data
            report["windows"][f"{width}pt__{name}"] = rec
    report["representative_ticks"] = []
    for when in (.5, 1.5, 2.3, 3.5):
        i = int(np.argmin(abs(t-when)))
        report["representative_ticks"].append({"time_s": float(t[i]), "request_G": a["request_G"][i].tolist(),
            "feedback_G": a["feedback_G"][i].tolist(), "K": float(a["K"][i]),
            "needed_5pt_normalized": a["5__needed_normalized"][i].tolist(),
            "needed_5pt_old_held_rad": a["5__needed_old_rad"][i].tolist(), "cells": {cell:
                {key.replace("_rad","_normalized"): (a[cell+"__"+key][i]* (a["K"][i]/limits if key.endswith("_rad") else 1)).tolist()
                 for key in ("held_rad","P_rad","I_provisional_rad","D_rad","next_integral_rad","antiwindup_blocked")}
                for cell in CELLS}})
    if cid in split.CASES:
        old_requests=np.load(audit.OUTPUT/"request_audit.npz")
        for row in report["representative_ticks"]:
            i=int(round(row["time_s"]*48))
            row["current_memory_inverse_required_pitch_request_G"] = float(old_requests[cid+"__5__current_plant__given_memory_required_request_G"][i,0])
            row["current_memory_inverse_pitch_status"] = int(old_requests[cid+"__5__current_plant__given_memory_status"][i,0])
    return a, report


def plot_conditional(cid,a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True)
    colors = {"legacy_raw":"#879198","transport_target1":"#883b8f","transport_raw":"#147c71","legacy_target1":"#d18a34"}
    for axis,label in enumerate(("Pitch","Yaw")):
        factor = a["K"]/a["limits_rad"][axis]; ax = axes[0,axis]
        for width,style in ((3,":"),(5,"-")):
            ax.plot(a["t"],np.where(a[f"{width}__diagnostic_mask"],a[f"{width}__needed_normalized"][:,axis],np.nan),color="#253f56",ls=style,label=f"Needed, {width}pt")
        for cell,color in colors.items():
            ax.plot(a["t"],factor*a[cell+"__held_rad"][:,axis],color=color,label=cell,lw=1.3)
            axes[1,axis].plot(a["t"],factor*a[cell+"__next_integral_rad"][:,axis],color=color,label=cell,lw=1.3)
        ax.set_title(label); ax.set_ylabel("Held input: K old-held / raw finsAoa")
        axes[1,axis].set_ylabel("Saved integral: K I_old / raw finsAoa")
        for row in range(2):
            x=axes[row,axis]; x.grid(alpha=.18); x.axhline(0,color="#aaa",lw=.5)
            x.axvline(.7,color="#aaa",lw=.5);x.axvline(2.3,color="#aaa",lw=.5);x.set_xlim(14/48,3.5)
        axes[1,axis].set_xlabel("Original clock (s)")
    axes[0,0].legend(fontsize=7); axes[1,0].legend(fontsize=7)
    fig.suptitle(cid+": output-cap / integral-coordinate 2x2; each controller has its own history\nSame reference state, request and frozen-plant feedback; conditional diagnostic")
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/(cid+"_conditional.png"),dpi=150);plt.close(fig)


def support_gate(reports):
    checks = {}
    for width in (3,5):
        def rms(cid,window,axis,cell):
            return reports[cid]["windows"][f"{width}pt__{window}"]["axes"][axis]["cells"][cell]["RMS_normalized_input_error_all_diagnostic"]
        checks[f"{width}pt_60_platform_yaw_improves"] = rms("jari_259_60","platform","yaw","transport_raw") < rms("jari_259_60","platform","yaw","transport_target1")
        for axis in ("pitch","yaw"):
            base=rms("jari_259_30","active",axis,"transport_target1");candidate=rms("jari_259_30","active",axis,"transport_raw")
            checks[f"{width}pt_30_active_{axis}_not_materially_worse"] = candidate <= base*1.10 + .01
    checks["all_low_regime_exact"] = all(r["checks"]["all_four_before_latch_exact"] for r in reports.values())
    return {"predeclared_rule":"60 platform yaw 3pt AND 5pt improve; 30 active each axis <=baseline*1.10+.01 normalized; all before-latch exact",
        "threshold_role":"bounded free-run screening only; no composite score or coefficient selection", "checks": checks,"supported_for_one_fixed_six_scene_check":all(checks.values())}


def main():
    if sys.argv[1:]: raise ValueError("This completed diagnostic has no free-run option")
    before=snapshot(); inputs=audit.inverse.source_inputs()
    coefficients=drag.DragCoefficients(**json.loads((drag.OUTPUT/"fit.json").read_text())["coefficients"])
    z=np.load(audit.inverse.OUTPUT/"inverse_inputs.npz"); cache=np.load(split.OUTPUT/"pn_pid_timeseries.npz")
    legacy=np.load(audit.fix.OUTPUT/"conditional_30.npz"); OUT.mkdir(parents=True,exist_ok=True)
    reports={}; saved={}
    for cid in CASES:
        a,r=conditional(cid,inputs[cid],z,cache,legacy,coefficients)
        reports[cid]=r;saved.update({cid+"__"+key:value for key,value in a.items()});plot_conditional(cid,a)
    finite=all(np.isfinite(v).all() for v in saved.values())
    numerical=all(v <= 1e-12 for r in reports.values() for k,v in r["checks"].items() if k!="all_four_before_latch_exact")
    if not (finite and numerical and before==snapshot()):raise AssertionError("Focused conditional verification failed")
    gate=support_gate(reports)
    doc={"task":"War Thunder fixed control authority / memory coordinate 2x2", "cells":CELLS,"cases":reports,"support_gate":gate,
        "deduplication":["legacy/raw is existing finite_force; transport/target1 is existing control_fix",
            "30 conditional baseline cache reused and closed; 30/60 transport baseline cached request/feedback/held/memory closed",
            "textbook I_phase/D_measurement and trim_target change other interfaces, not these two fixed cells"],
        "limits":["Raw finsAoa as old effective actuator limit is D hypothesis, not observed fin angle.",
            "Normalized target1 is an experimental extra bound, not a known native blkx cap.",
            "Output uses provisional I; saved I uses conditional antiwindup after output clipping.",
            "Each cell owns I/previous error/filtered derivative/K/latch history; shared request/feedback are fixed reference-state diagnostics.",
            "3/5-point required inputs are noncausal frozen-plant inverses; 24-to-48 interpolation is conditional evidence only.",
            "Removing a cap is not increasing Ku or proving larger physical fin authority; none of the field roles is identified."],
        "representative_value_units":"All cell terms ending normalized are K*(old-coordinate rad)/raw finsAoa. Output uses provisional I; next_integral is saved after conditionalAW.",
        "free_validation_decision":"Not run: both 3pt and 5pt fixed-reference 60-platform support checks failed. Speed/attitude/load effects of either new cell in free trajectories remain untested.",
        "verification":{"all_arrays_finite":finite,"protected_old_size_mtime_unchanged":before==snapshot(),
            "all_algorithm_cache_closure_le_1e12":numerical,"zero_fits":True,"new_fixed_cells":2,"free_integrations":0,
            "controller_hz":48,"plant_substeps":4,"surface":dataclasses.asdict(SurfaceCoefficients())}}
    np.savez_compressed(OUT/"conditional.npz",**saved);slew.write_json(OUT/"results.json",doc)
    print(json.dumps({"conditional_complete":True,"gate":gate,"checks":{cid:r["checks"] for cid,r in reports.items()}}),flush=True)


if __name__=="__main__":main()
