#!/usr/bin/env python3
"""One predeclared all-active target-cap experiment for a War Thunder unit.

Only move the existing target1 cap before the existing nonlinear-force latch.
Integral transport retains its original high-angle lifecycle. No fits, no new
coefficients, no observed missile state after runtime creation.
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
import check_r771_lift_direction as geometry
import analyze_r771_pn_pid_decoupling as dec
from aim120_model.control import scheduled_pid_values
from aim120_model.guidance import interpolate_table
from aim120_model.math3d import norm, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients

OUT = ROOT / "outputs/jari_offaxis/all_active_cap"
CASES = geometry.CASES
NATIVE = ("native_H110", "native_H27.5", "native_H55native")
WINDOWS = (*dec.WINDOWS, ("early_0_3p5", 0., 3.5), ("full", 0., math.inf))
UNITS = {**dec.audit.fix.finite.UNITS, "body_pitch_deg": "deg"}
PRIMARY = ("speed_mps", "alpha_deg", "normal_specific_load_g")


class AllActiveCapControl(R771GameControl):
    """Original controller with just one changed effective-limit condition."""
    def update_axis(self, axis, time_s, error_g, memory, pid, old_held_limit_rad, gain,
                    target_per_old_held, nonlinear_force_active):
        if axis not in (0, 1):
            raise ValueError("axis must be pitch0 or yaw1")
        k = float(target_per_old_held)
        if not all(math.isfinite(v) for v in (time_s, error_g, k, old_held_limit_rad, gain)) or k <= 0 or old_held_limit_rad <= 0 or not 0 <= gain <= 1:
            raise ValueError("finite positive normalization/limit and flight gain in [0,1] required")
        if self.last_time[axis] is not None:
            if time_s == self.last_time[axis]:
                return self.cached[axis]
            if time_s < self.last_time[axis]:
                raise ValueError("control clock cannot move backwards")
        if abs(time_s * self.controller_hz - round(time_s * self.controller_hz)) > 1e-8:
            raise ValueError("control memory may update only at controller ticks")
        if gain <= 1e-12:
            if memory != (0., 0., 0.):
                raise ValueError("inactive PID memory must be zero")
            return 0., memory
        self.nonlinear_latched = self.nonlinear_latched or bool(nonlinear_force_active)
        integral, previous_error, previous_derivative = memory
        integral_limit = float(pid["integral_limit"])
        ratio = 1.
        if self.nonlinear_latched:
            previous_k = self.previous_high_k[axis]
            ratio = 1. if previous_k is None else previous_k / k
            integral = dec.audit.clip(integral * ratio, integral_limit)
        # The only candidate change: target1 is active before the high latch.
        effective_limit = min(old_held_limit_rad, old_held_limit_rad / k)
        dt = 1. / self.controller_hz
        increment = float(pid["i"]) * error_g * dt
        provisional = dec.audit.clip(integral + increment, integral_limit)
        raw_d = (error_g - previous_error) / dt
        derivative = previous_derivative + dt / (1e-9 + dt) * (raw_d - previous_derivative)
        p = float(pid["p"]) * error_g
        d = float(pid["d"]) * derivative
        output = p + provisional + d
        saturated = dec.audit.clip(output, effective_limit)
        blocked = abs(saturated) >= .99 * effective_limit and error_g * saturated > 0
        next_i = integral if blocked else provisional
        result = gain * saturated, (next_i, error_g, derivative)
        self.last_time[axis] = time_s
        self.cached[axis] = result
        if self.nonlinear_latched:
            self.previous_high_k[axis] = k
        self.diagnostics[axis] = {"time_s": time_s, "axis": axis, "error_g": error_g,
            "K_target_per_old_held": k, "nonlinear_latched": self.nonlinear_latched,
            "integral_transport_ratio": ratio, "transported_integral_old_rad": integral,
            "integral_target_rad": k * next_i, "P_target_rad": k * p, "D_target_rad": k * d,
            "target_candidate_limit_rad": old_held_limit_rad, "effective_old_held_limit_rad": effective_limit,
            "original_old_held_limit_rad": old_held_limit_rad, "antiwindup_blocked": blocked,
            "held_old_rad": result[0], "effective_target_rad": k * result[0],
            "normalized_target_at_tick": k * result[0] / old_held_limit_rad, "gain": gain,
            "P_old_rad": p, "I_provisional_old_rad": provisional, "D_old_rad": d,
            "I_increment_old_rad": increment, "natural_old_rad": output,
            "all_active_target_cap": True, "I_transport_high_latch_only": True}
        return result


class AllActiveCapGame(geometry.LiftDirectionGame):
    def initialize(self, state):
        initial = super().initialize(state)
        self.game_control = AllActiveCapControl()
        return initial

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(controller_mechanism="existing target1 cap on all active ticks; original high-only I transport",
            experimental_target_cap="min(raw old-held limit,raw/K) at every active tick; native semantics D",
            new_fitted_coefficients=0, source_data_in_runtime=False,
            lift_direction_rule="same six-case-validated current local-flow geometry")
        return result


def make_model(cfg, coefficients):
    return AllActiveCapGame(cfg, request_slew_g_s=110., drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def snapshot():
    folders = (ROOT / "src", ROOT / "config", ROOT / "configs", ROOT / "docs",
        ROOT / "outputs/jari_offaxis/reference", geometry.OUT,
        dec.audit.fix.OUTPUT, dec.audit.inverse.OUTPUT,
        ROOT / "outputs/jari_offaxis/prepeak_pid_interface")
    paths = [p for folder in folders for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths.extend([Path(m.__file__) for m in (geometry, dec, dec.audit, dec.audit.fix, dec.audit.fix.finite,
        dec.audit.drag, dec.audit.slew, dec.audit.fix.finite.comparison, dec.audit.fix.finite.harness)])
    paths.extend([dec.audit.fix.finite.comparison.SNAPSHOT, dec.audit.fix.finite.comparison.FROZEN,
        dec.audit.fix.finite.comparison.DEFAULT_MANIFEST, dec.audit.drag.OUTPUT / "fit.json",
        ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx",
        ROOT / "scripts/check_r771_prepeak_pid_interface.py"])
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in set(paths)}


def physical_exact(a, b):
    # Diagnostics necessarily report a new cap even when natural output is unchanged.
    return json.loads(json.dumps(a["samples"])) == b["samples"] and a["summary"] == b["summary"] and all(
        all(np.array_equal(new[k], old[k]) for k in ("time_s", "gain", "held_fin_rad", "feedback_g", "request_g", "pid_memory"))
        for new, old in zip(a["ticks"], b["ticks"])) and len(a["ticks"]) == len(b["ticks"])


def verify(model, result, cfg, coefficients, initial, target_at, baseline_full):
    original = geometry.make_model(cfg, coefficients)
    original.initialize(dataclasses.replace(initial))
    direct = AllActiveCapControl()
    original_controller = R771GameControl()
    memory = [(0., 0., 0.), (0., 0., 0.)]
    samples = {s["time_s"]: s for s in result["samples"]}
    limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    maxima = {"held_PID_closure_rad": 0., "memory_PID_closure": 0., "request_same_state_G": 0.,
        "force_same_state_N": 0., "drag_same_state_N": 0., "angular_same_state_rad_s2": 0.,
        "raw_cap_excess_rad": 0., "target_cap_excess": 0., "raw_I_limit_excess": 0.,
        "transport_ratio_original_equal": 0., "P_I_D_natural_original_equal_rad": 0.}
    effect = {"pre_high_axis_ticks_new_clip": 0, "pre_high_axis_ticks_new_AW": 0,
        "old_same_error_memory_output_change_max_rad": 0., "first_changed_tick_s": None,
        "before_high_changed_axes": {"pitch": 0, "yaw": 0}}
    tick_arrays = {"t": [], "gain": [], "held": [], "P": [], "I_provisional": [], "D": [],
        "effective_limit": [], "latch": [], "AW": [], "K": [], "normalized_target": [], "error_G": []}
    for tick in result["ticks"]:
        ts = tick["time_s"]
        s = samples[ts]
        state = dec.audit.fix.finite.sample_state(s)
        f = original.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        request, _ = original.request(state, target_at(ts), ts, f)
        maxima["request_same_state_G"] = max(maxima["request_same_state_G"], max(abs(x-y) for x, y in zip(request, tick["request_g"])))
        current = model.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        maxima["force_same_state_N"] = max(maxima["force_same_state_N"], norm(sub(f.wind_lift_force, current.wind_lift_force)))
        maxima["drag_same_state_N"] = max(maxima["drag_same_state_N"], norm(sub(f.wind_drag_force, current.wind_drag_force)))
        maxima["angular_same_state_rad_s2"] = max(maxima["angular_same_state_rad_s2"], max(abs(x-y) for x, y in zip(f.angular_acceleration, current.angular_acceleration)))
        if tick["gain"] <= 1e-12:
            if np.any(np.asarray(tick["pid_memory"])) or np.any(np.asarray(tick["held_fin_rad"])):
                raise AssertionError("inactive memory/held")
            continue
        pid = scheduled_pid_values(cfg["control"]["pid"], (round(ts * 48) + 1) / 48)
        ds = tick["control_runtime_diagnostics"]
        values = {k: [] for k in tick_arrays if k not in ("t", "gain")}
        for axis, d in enumerate(ds):
            e = tick["request_g"][axis] - tick["feedback_g"][axis]
            k = d["K_target_per_old_held"]
            direct_result = direct.update_axis(axis, ts, e, memory[axis], pid, limits[axis], tick["gain"], k, original.current_force_nonlinear)
            maxima["held_PID_closure_rad"] = max(maxima["held_PID_closure_rad"], abs(direct_result[0] - tick["held_fin_rad"][axis]))
            maxima["memory_PID_closure"] = max(maxima["memory_PID_closure"], max(abs(x-y) for x, y in zip(direct_result[1], tick["pid_memory"][axis])))
            # Original one-tick counterfactual gets the candidate's identical previous memory and original K history.
            old_result = original_controller.update_axis(axis, ts, e, memory[axis], pid, limits[axis], tick["gain"], k, original.current_force_nonlinear)
            old_d = original_controller.diagnostics[axis]
            maxima["transport_ratio_original_equal"] = max(maxima["transport_ratio_original_equal"], abs(old_d["integral_transport_ratio"] - d["integral_transport_ratio"]))
            # Calculate natural from the transported I recorded by the original update.
            old_provisional = dec.audit.clip(old_d["transported_integral_old_rad"] + pid["i"] * e / 48, pid["integral_limit"])
            old_natural = old_d["P_target_rad"] / k + old_provisional + old_d["D_target_rad"] / k
            maxima["P_I_D_natural_original_equal_rad"] = max(maxima["P_I_D_natural_original_equal_rad"], abs(old_natural - d["natural_old_rad"]))
            delta = abs(old_result[0] - direct_result[0])
            effect["old_same_error_memory_output_change_max_rad"] = max(effect["old_same_error_memory_output_change_max_rad"], delta)
            if not d["nonlinear_latched"]:
                clipped = delta > 1e-12
                aw_changed = bool(old_d["antiwindup_blocked"]) != bool(d["antiwindup_blocked"])
                effect["pre_high_axis_ticks_new_clip"] += int(clipped)
                effect["pre_high_axis_ticks_new_AW"] += int(aw_changed)
                effect["before_high_changed_axes"][("pitch", "yaw")[axis]] += int(clipped or aw_changed)
                if (clipped or aw_changed) and effect["first_changed_tick_s"] is None:
                    effect["first_changed_tick_s"] = ts
            elif delta > 1e-12 or old_d["antiwindup_blocked"] != d["antiwindup_blocked"]:
                raise AssertionError("high-latch algorithm differs beyond the moved cap")
            maxima["raw_cap_excess_rad"] = max(maxima["raw_cap_excess_rad"], abs(direct_result[0]) - limits[axis])
            maxima["target_cap_excess"] = max(maxima["target_cap_excess"], abs(k * direct_result[0] / limits[axis]) - 1.)
            maxima["raw_I_limit_excess"] = max(maxima["raw_I_limit_excess"], abs(direct_result[1][0]) - pid["integral_limit"])
            memory[axis] = direct_result[1]
            for key, value in (("held", direct_result[0]), ("P", d["P_old_rad"]), ("I_provisional", d["I_provisional_old_rad"]),
                ("D", d["D_old_rad"]), ("effective_limit", d["effective_old_held_limit_rad"]), ("latch", d["nonlinear_latched"]),
                ("AW", d["antiwindup_blocked"]), ("K", k), ("normalized_target", d["normalized_target_at_tick"]), ("error_G", e)):
                values[key].append(value)
        tick_arrays["t"].append(ts)
        tick_arrays["gain"].append(tick["gain"])
        for key, value in values.items():
            tick_arrays[key].append(value)
        if direct.previous_high_k != original_controller.previous_high_k or direct.nonlinear_latched != original_controller.nonlinear_latched:
            raise AssertionError("original high-only K lifecycle changed")
    fixed = {"config_and_raw_PID_equal": model.config == cfg, "five_coefficients_equal": model.coefficients == SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coefficients, "PN_loft_S110_equal": model.relative_blend == .5 and model.request_slew_g_s == 110. and [model.amplitude_coefficient,model.transition_coefficient] == [.75,1.3],
        "clock48": model.controller_hz == 48 and np.max(abs(np.asarray([t["time_s"] for t in result["ticks"]]) * 48 - np.arange(len(result["ticks"])))) < 1e-8,
        "plant_substeps4": model.plant_substeps == 4, "one_surface_0p4m": model.arm == .4}
    if max(maxima.values()) > 1e-10 or not all(fixed.values()):
        raise AssertionError({"maxima": maxima, "fixed": fixed})
    return {"max_residuals": maxima, "fixed": fixed, "cap_only_same_memory_effect": effect,
        "physical_baseline_exact": physical_exact(result, baseline_full),
        "force_torque_record_closure": dec.audit.inverse.saved_closure(model, result)}, {k: np.asarray(v) for k, v in tick_arrays.items()}


def first_peak(prediction, field):
    t, v = prediction["t"], prediction[field]
    mask = (t >= .3 - 1e-6) & (t <= 3.5 + 1e-6)
    ids = [i for i in range(1, len(t)-1) if mask[i] and v[i] > 0 and v[i] > v[i-1] and v[i] >= v[i+1]]
    if not ids:
        return None
    i = ids[0]
    return {"time_s": float(t[i]), "value": float(v[i])}


def peaks(observed, baseline, candidate):
    return {label: {field: first_peak(p, field) for field in ("body_yaw_rate_deg_s", "alpha_deg", "normal_specific_load_g")}
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", candidate))}


def plot_jari(cid, observed, baseline, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, field in zip(axes.flat, UNITS):
        for label, p, color in (("JARI reference", observed, "#147378"), ("Cached geometry baseline", baseline, "#813a8a"),
                ("All-active target1", candidate, "#ce7b2f")):
            ax.plot(p["t"], p[field], color=color, label=label)
        ax.set(xlabel="Original time (s)", ylabel=field + " (" + UNITS[field] + ")", xlim=(0, 3.5 if field != "speed_mps" else observed["t"][-1]))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(cid + ": actual free trajectory; only existing target cap starts before high-angle latch\nFrozen plant, geometry, raw PID, PN/loft/S110 and 48 Hz + four substeps", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / (cid + "_free.png"), dpi=160)
    plt.close(fig)


def jari_gate(reports):
    check = {}
    for cid in CASES:
        for window in ("early_0_3p5", "full"):
            m = reports[cid]["separate_metrics"][window]["metrics"]
            for field in PRIMARY:
                base, new = m[field]["old"]["rmse"], m[field]["new"]["rmse"]
                check[cid + "__" + window + "__" + field] = new < base if cid == "jari_259_60" else new <= 1.1 * base + 1e-12
            if cid == "jari_259_60":
                check[cid + "__" + window + "__pitch_no10percent_worse"] = m["body_pitch_deg"]["new"]["rmse"] <= 1.1 * m["body_pitch_deg"]["old"]["rmse"] + 1e-12
    extrema = reports["jari_259_60"]["first_positive_peaks"]
    for field in ("body_yaw_rate_deg_s", "alpha_deg"):
        r, b, c = (extrema[k][field] for k in ("reference", "baseline", "candidate"))
        check["jari_259_60__first_peak_amplitude_closer__" + field] = bool(r and b and c and abs(c["value"] - r["value"]) < abs(b["value"] - r["value"]))
    return {"rules": check, "passes": all(check.values())}


def run_jari(inputs, coefficients):
    cached = np.load(geometry.OUT / "free_trajectories.npz")
    fields, drag = dec.audit.fix.finite.fields, dec.audit.drag
    reports, arrays = {}, {}
    new = recovered = 0
    full = OUT / "full_runs"
    full.mkdir(exist_ok=True)
    for cid in CASES:
        x = inputs[cid]
        model = make_model(x["config"], coefficients)
        path = full / (cid + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            model.initialize(dataclasses.replace(x["initial"]))
            recovered += 1
        else:
            result = model.run(dataclasses.replace(x["initial"]), x["target_at"], end_time_s=x["config"]["performance"]["lifetime_s"])
            dec.audit.slew.write_json(path, result, True)
            new += 1
        baseline = drag.selected_arrays(cached, cid + "__candidate__")
        observed = drag.selected_arrays(cached, cid + "__reference__")
        prediction = fields.prediction_fields(result, model, x["run"]["scenario"])
        baseline_path = geometry.OUT / "full_runs" / (cid + ".json")
        baseline_full = json.loads(baseline_path.read_text())
        baseline_model = geometry.make_model(x["config"], coefficients)
        baseline_model.initialize(dataclasses.replace(x["initial"]))
        restored = fields.prediction_fields(baseline_full, baseline_model, x["run"]["scenario"])
        restored_exact = all(np.array_equal(restored[k], baseline[k]) for k in ("t", "speed_mps", "alpha_deg", "normal_specific_load_g", "body_pitch_deg", "body_yaw_deg", "body_yaw_rate_deg_s"))
        if not restored_exact:
            raise AssertionError("geometry baseline receipt cannot restore primary arrays")
        metrics = drag.pair_metrics(observed, baseline, prediction, tuple(UNITS), WINDOWS)
        for w in metrics.values():
            for field, metric in w["metrics"].items():
                metric["units"] = UNITS[field]
        check, ta = verify(model, result, x["config"], coefficients, x["initial"], x["target_at"], baseline_full)
        reports[cid] = {"separate_metrics": metrics, "first_positive_peaks": peaks(observed, baseline, prediction),
            "summary": result["summary"], "baseline_summary": baseline_full["summary"], "full_result_path": str(path),
            "baseline_receipt_restores_primary_arrays_exactly": restored_exact, "focused_checks": check}
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction)):
            fields.save_arrays(arrays, cid + "__" + label, p)
        arrays.update({cid + "__tick__" + k: v for k, v in ta.items()})
        plot_jari(cid, observed, baseline, prediction)
        print(json.dumps({"completed_free_case": cid, "summary": result["summary"], "first_peaks": reports[cid]["first_positive_peaks"],
            "early_primary_RMSE": {k: [metrics["early_0_3p5"]["metrics"][k][label]["rmse"] for label in ("old", "new")] for k in PRIMARY}}), flush=True)
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    return reports, {"new_Jari_integrations": new, "recovered_Jari_without_reintegration": recovered,
        "unique_Jari_cases": 3, "baseline_integrations": 0, "all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values())}


def run_native(coefficients):
    harness, fields, drag = dec.audit.fix.finite.harness, dec.audit.fix.finite.fields, dec.audit.drag
    cases = {c["id"]: c for c in harness.load_cases(dec.audit.fix.finite.comparison.SNAPSHOT)}
    cached = np.load(geometry.OUT / "native_trajectories.npz")
    windows = (*dec.WINDOWS, *harness.WINDOWS)
    arrays, reports = {}, {}
    new = recovered = 0
    for cid in NATIVE:
        x = cases[cid.removeprefix("native_")]
        cfg = copy.deepcopy(x["profile"]["_model_config"])
        initial = dataclasses.replace(x["_creation_state"])
        qx, qy, qz, qw = x["creation"]["u16_1"]
        initial.orientation_quaternion = (qw, qx, qy, qz)
        target_at = x["_target_model"].state_at
        model = make_model(cfg, coefficients)
        path = OUT / (cid + "_run.json")
        if path.exists():
            result = json.loads(path.read_text())
            model.initialize(dataclasses.replace(initial))
            recovered += 1
        else:
            result = model.run(dataclasses.replace(initial), target_at, end_time_s=x["scenario"]["max_simulation_time_s"])
            dec.audit.slew.write_json(path, result, True)
            new += 1
        baseline = drag.selected_arrays(cached, cid + "__candidate__")
        observed = drag.selected_arrays(cached, cid + "__reference__")
        prediction = harness.sample_series(result["samples"], x)
        baseline_full = json.loads((geometry.OUT / (cid + "_run.json")).read_text())
        restored = harness.sample_series(baseline_full["samples"], x)
        restored_exact = all(np.array_equal(v, baseline[k]) for k, v in restored.items())
        if not restored_exact:
            raise AssertionError("native geometry cache receipt")
        metrics = drag.pair_metrics(observed, baseline, prediction, ("speed_kmh", "alpha_deg", "pitch_deg", "omega_rad_s", "gamma_deg", "y_m"), windows)
        gate = {w + "__" + k: m["new"]["rmse"] <= 1.1 * m["old"]["rmse"] + 1e-12
            for w, ws in metrics.items() for k, m in ws["metrics"].items() if k in ("speed_kmh", "alpha_deg")}
        load = {}
        for name, lo, hi in windows:
            mask = (observed["t"] >= lo - 1e-9) & (observed["t"] <= hi + 1e-9)
            mask &= (observed["t"] >= max(baseline["t"][0], prediction["t"][0])) & (observed["t"] <= min(baseline["t"][-1], prediction["t"][-1]))
            ts = observed["t"][mask]
            load[name] = {"independent_observed_G_available": False, "prediction_change_G": drag.stats(np.interp(ts, prediction["t"], prediction["load_g"]) - np.interp(ts, baseline["t"], baseline["load_g"]))}
        check, ta = verify(model, result, cfg, coefficients, initial, target_at, baseline_full)
        reports[cid] = {"separate_metrics": metrics, "load_prediction_changes": load, "gate_primary_no10percent_worse": gate,
            "passes": all(gate.values()), "summary": result["summary"], "baseline_receipt_restores_arrays_exactly": restored_exact,
            "focused_checks": check, "full_result_path": str(path)}
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction)):
            fields.save_arrays(arrays, cid + "__" + label, p)
        arrays.update({cid + "__tick__" + k: v for k, v in ta.items()})
        native_plot(cid, observed, baseline, prediction)
        print(json.dumps({"completed_native_case": cid, "passes": reports[cid]["passes"], "summary": result["summary"]}), flush=True)
    np.savez_compressed(OUT / "native_trajectories.npz", **arrays)
    return reports, {"new_native_integrations": new, "recovered_native_without_reintegration": recovered,
        "unique_native_cases": 3, "baseline_integrations": 0, "all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values())}


def native_plot(cid, observed, baseline, candidate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for ax, field in zip(axes.flat, ("speed_kmh", "alpha_deg", "pitch_deg", "omega_rad_s", "gamma_deg", "y_m")):
        for label, p, color in (("Native replay", observed, "#147378"), ("Cached geometry baseline", baseline, "#813a8a"), ("All-active cap", candidate, "#ce7b2f")):
            ax.plot(p["t"], p[field], color=color, label=label)
        ax.set(xlabel="Original time (s)", ylabel=field)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(cid + ": actual free all-active target-cap validation, frozen current native fixture", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / (cid + "_free.png"), dpi=150)
    plt.close(fig)


def main():
    before = snapshot()
    OUT.mkdir(parents=True, exist_ok=True)
    # This protocol is written before loading trajectory observations or running.
    protocol = {"candidate": "min(raw_oldheld_limit,raw_limit/K) at all active ticks; original high-only I transport",
        "unchanged": "raw PID/Ilim/.99AW/gain-last, five plant coefficients, geometry, force norm, angular law, drag, loft, PN/S110, 48Hz+4substep, one 0.4m surface",
        "Jari_gate": "60 first yaw-rate and alpha amplitudes closer; 60 speed/alpha/load each improve early 0-3.5 and full and pitch no>10% worse;0/30 primary early/full no>10% worse",
        "native_trigger": "Only if every Jari gate passes, run current H110/H27.5/H55native once each; each speed/alpha full and existing windows no>10% worse; load prediction change only",
        "report": "Separate first peaks/timing and every variable in startup/platform/unloading/early/full; no combined score; unfavorable windows remain"}
    dec.audit.slew.write_json(OUT / "protocol.json", protocol)
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    inputs = dec.audit.inverse.source_inputs()
    reports, verification = run_jari(inputs, coefficients)
    gate = jari_gate(reports)
    native, native_verification = {}, {"new_native_integrations": 0, "unique_native_cases": 0, "not_run_reason": "Jari gate did not pass"}
    if gate["passes"]:
        native, native_verification = run_native(coefficients)
    protected = before == snapshot()
    verification.update(native=native_verification, protected_size_mtime_unchanged=protected,
        fits_scans_new_coefficients=0, baseline_integrations=0, source_data_in_runtime=False,
        core_config_data_docs_changes=0, controller_hz=48, plant_substeps=4)
    if not protected or not verification["all_arrays_finite"] or not native_verification.get("all_arrays_finite", True):
        raise AssertionError(verification)
    accepted = gate["passes"] and bool(native) and all(r["passes"] for r in native.values())
    dec.audit.slew.write_json(OUT / "results.json", {"task": "War Thunder all-active existing target cap, actual free test",
        "protocol": protocol, "cases": reports, "Jari_gate": gate, "native_cases": native,
        "verification": verification, "candidate_accepted_by_predeclared_gate": accepted,
        "native_game_cap_identified": False, "evidence_grade": "D", "default_model_changed": False})
    print(json.dumps({"output": str(OUT), "Jari_gate": gate, "accepted": accepted, "verification": verification}), flush=True)


if __name__ == "__main__":
    main()
