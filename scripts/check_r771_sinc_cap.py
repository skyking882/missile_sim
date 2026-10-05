#!/usr/bin/env python3
"""One fixed beta/sin(beta) output-bound experiment for the War Thunder game.

Full horizontal relative PN and original AW/plant remain unchanged. Only the
upper output bound changes; unsaturated PID is not multiplied by the factor.
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
from aim120_model.math3d import add, cross, dot, norm, scale, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients

OUT = ROOT / "outputs/jari_offaxis/sinc_cap"
CASES = geometry.CASES
NATIVE = ("native_H110", "native_H27.5", "native_H55native")
WINDOWS = (*dec.WINDOWS, ("early_0_3p5", 0., 3.5), ("full", 0., math.inf))
UNITS = {**dec.audit.fix.finite.UNITS, "body_pitch_deg": "deg"}
PRIMARY = ("speed_mps", "alpha_deg", "normal_specific_load_g")


def inverse_sinc(beta):
    if not math.isfinite(beta) or not 0 <= beta < math.pi / 2:
        raise ValueError("Declared sinc-cap forward domain requires0<=beta<pi/2")
    return 1. if beta == 0. else beta / math.sin(beta)


class SincCapControl(R771GameControl):
    """Only the output bound uses beta/sin(beta); original-cap AW retained."""
    current_beta_rad = 0.
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
        beta = float(self.current_beta_rad)
        factor = inverse_sinc(beta)
        effective_limit = min(old_held_limit_rad, old_held_limit_rad / k * factor)
        dt = 1. / self.controller_hz
        increment = float(pid["i"]) * error_g * dt
        provisional = dec.audit.clip(integral + increment, integral_limit)
        raw_d = (error_g - previous_error) / dt
        derivative = previous_derivative + dt / (1e-9 + dt) * (raw_d - previous_derivative)
        p = float(pid["p"]) * error_g
        d = float(pid["d"]) * derivative
        output = p + provisional + d
        saturated = dec.audit.clip(output, effective_limit)
        aw_limit = min(old_held_limit_rad, old_held_limit_rad / k) if self.nonlinear_latched else old_held_limit_rad
        old_saturated = dec.audit.clip(output, aw_limit)
        blocked = abs(old_saturated) >= .99 * aw_limit and error_g * old_saturated > 0
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
            "sinc_output_cap_only": True, "I_transport_high_latch_only": True,
            "beta_rad": beta, "inverse_sinc_factor": factor,
            "original_AW_limit_old_rad": aw_limit, "original_counterfactual_saturated_old_rad": old_saturated,
            "next_integral_old_rad": next_i}
        return result


class SincCapGame(geometry.LiftDirectionGame):
    relative_blend = 1.
    variant_label = "full_horizontal_relative_PN_original_loft_gate"
    def initialize(self, state):
        initial = super().initialize(state)
        self.game_control = SincCapControl()
        return initial

    def request(self, state, target, time_s, evaluation):
        request, details = super().request(state, target, time_s, evaluation)
        beta = math.hypot(*evaluation.local_incidence)
        inverse_sinc(beta)  # Reject branch exits; no fallback or artificial cap.
        self.game_control.current_beta_rad = beta
        return request, details

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(controller_mechanism="sinc only-output bound; original cap-based AW and high-only I transport",
            runtime_controller="check_r771_sinc_cap.SincCapControl",
            outer="full horizontal existing relativePN, gated loft and unchanged S110", relative_blend=1.,
            experimental_target_cap="output=min(raw,raw/K*beta/sin(beta)); original cap-based AW; native semantics D",
            new_fitted_coefficients=0, source_data_in_runtime=False,
            lift_direction_rule="same six-case-validated current local-flow geometry")
        return result


def make_model(cfg, coefficients):
    return SincCapGame(cfg, request_slew_g_s=110., drag_coefficients=coefficients,
        coefficients=SurfaceCoefficients(), force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def snapshot():
    folders = (ROOT / "src", ROOT / "config", ROOT / "configs", ROOT / "docs",
        ROOT / "outputs/jari_offaxis/reference", geometry.OUT,
        dec.audit.fix.OUTPUT, dec.audit.inverse.OUTPUT,
        ROOT / "outputs/jari_offaxis/prepeak_pid_interface", ROOT / "outputs/jari_offaxis/all_active_cap",
        ROOT / "outputs/jari_offaxis/early_clip_only", ROOT / "outputs/jari_offaxis/full_relative_clip")
    paths = [p for folder in folders for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths.extend([Path(m.__file__) for m in (geometry, dec, dec.audit, dec.audit.fix, dec.audit.fix.finite,
        dec.audit.drag, dec.audit.slew, dec.audit.fix.finite.comparison, dec.audit.fix.finite.harness)])
    paths.extend([dec.audit.fix.finite.comparison.SNAPSHOT, dec.audit.fix.finite.comparison.FROZEN,
        dec.audit.fix.finite.comparison.DEFAULT_MANIFEST, dec.audit.drag.OUTPUT / "fit.json",
        ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx",
        ROOT / "scripts/check_r771_prepeak_pid_interface.py", ROOT / "scripts/check_r771_all_active_cap.py",
        ROOT / "scripts/check_r771_early_clip_only.py", ROOT / "scripts/check_r771_full_relative_clip.py"])
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in set(paths)}


def physical_exact(a, b):
    # Diagnostics necessarily report a new cap even when natural output is unchanged.
    return json.loads(json.dumps(a["samples"])) == b["samples"] and a["summary"] == b["summary"] and all(
        all(np.array_equal(new[k], old[k]) for k in ("time_s", "gain", "held_fin_rad", "feedback_g", "request_g", "pid_memory"))
        for new, old in zip(a["ticks"], b["ticks"])) and len(a["ticks"]) == len(b["ticks"])


def verify(model, result, cfg, coefficients, initial, target_at, baseline_full):
    original = make_model(cfg, coefficients)
    original.initialize(dataclasses.replace(initial))
    original_force = geometry.make_model(cfg, coefficients)
    original_force.initialize(dataclasses.replace(initial))
    direct = SincCapControl()
    original_controller = R771GameControl()
    memory = [(0., 0., 0.), (0., 0., 0.)]
    samples = {s["time_s"]: s for s in result["samples"]}
    limits = np.radians([cfg["aerodynamics"][k] for k in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg")])
    maxima = {"held_PID_closure_rad": 0., "memory_PID_closure": 0., "request_same_state_G": 0.,
        "force_same_state_N": 0., "drag_same_state_N": 0., "angular_same_state_rad_s2": 0.,
        "raw_cap_excess_rad": 0., "target_cap_excess": 0., "raw_I_limit_excess": 0.,
        "transport_ratio_original_equal": 0., "P_I_D_natural_original_equal_rad": 0.,
        "original_AW_limit_equal_rad": 0., "original_counterfactual_output_equal_rad": 0.,
        "next_I_original_same_memory_equal_rad": 0., "full_relative_PN_3D_formula_mps2": 0.,
        "full_minus_half_PN_vertical_mps2": 0., "full_minus_half_PN_axial_mps2": 0.,
        "full_minus_half_PN_horizontal_delta_mps2": 0., "full_unslewed_request_formula_G": 0.,
        "projected_command_windnormal_mps2": 0., "K_same_state_equal": 0.,
        "beta_current_local_equal_rad": 0., "factor_independent_sinc_equal": 0.,
        "newcap_independent_formula_equal_rad": 0., "unsaturated_PID_not_scaled_rad": 0.}
    effect = {"pre_high_axis_ticks_new_clip": 0, "pre_high_axis_ticks_new_AW": 0,
        "old_same_error_memory_output_change_max_rad": 0., "first_changed_tick_s": None,
        "before_high_changed_axes": {"pitch": 0, "yaw": 0}, "high_latch_extra_output_axis_ticks": 0}
    tick_arrays = {"t": [], "gain": [], "held": [], "P": [], "I_provisional": [], "D": [],
        "effective_limit": [], "original_AW_limit": [], "original_counterfactual_output": [],
        "I_next": [], "latch": [], "AW": [], "K": [], "normalized_target": [], "error_G": [],
        "beta_rad": [], "inverse_sinc_factor": [], "natural_PID_old_rad": []}
    request_arrays = {"PN_t_all": [], "full_PN_world_mps2": [], "half_PN_world_same_state_mps2": [],
        "half_to_full_horizontal_delta_mps2": []}
    for tick in result["ticks"]:
        ts = tick["time_s"]
        s = samples[ts]
        state = dec.audit.fix.finite.sample_state(s)
        f = original.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        target = target_at(ts)
        previous_slew = original.slew_horizontal_g
        request, details = original.request(state, target, ts, f)
        maxima["request_same_state_G"] = max(maxima["request_same_state_G"], max(abs(x-y) for x, y in zip(request, tick["request_g"])))
        # Current target/state only: independently reconstruct the existing full PN formula.
        if abs(details["los_rate_rad_s"][1]) != 0.:
            pn0 = details["original_pn_mps2"]
            alternative = scale(cross(details["los_rate_rad_s"], scale(sub(target.velocity, state.velocity), -1.)),
                float(cfg["guidance"]["pn_gain"]) * details["hit_gain"])
            delta = dot(alternative, f.wind_basis.right) - dot(pn0, f.wind_basis.right)
            full_pn = add(pn0, scale(f.wind_basis.right, delta))
            half_pn = add(pn0, scale(f.wind_basis.right, .5 * delta))
            difference = sub(full_pn, half_pn)
            maxima["full_relative_PN_3D_formula_mps2"] = max(maxima["full_relative_PN_3D_formula_mps2"], norm(sub(full_pn, details["pn_mps2"])))
            maxima["full_minus_half_PN_vertical_mps2"] = max(maxima["full_minus_half_PN_vertical_mps2"], abs(dot(difference, f.wind_basis.up)))
            maxima["full_minus_half_PN_axial_mps2"] = max(maxima["full_minus_half_PN_axial_mps2"], abs(dot(difference, f.wind_basis.forward)))
            maxima["full_minus_half_PN_horizontal_delta_mps2"] = max(maxima["full_minus_half_PN_horizontal_delta_mps2"], abs(dot(difference, f.wind_basis.right) - .5 * delta))
            maximum = float(cfg["guidance"]["maximum_lateral_acceleration_g"]) * model.gravity
            boundaries = dec.outer_boundaries(full_pn, details["loft_g"], f.wind_basis, model.gravity, maximum)
            expected, _, _ = dec.same_previous_slew(boundaries["unslewed_request_G"], previous_slew,
                tick["gain"], abs(details["los_rate_rad_s"][1]), maximum / dec.G0_MPS2)
            maxima["full_unslewed_request_formula_G"] = max(maxima["full_unslewed_request_formula_G"],
                max(abs(x-y) for x, y in zip(boundaries["unslewed_request_G"], details["slew_original_request_g"])),
                max(abs(x-y) for x, y in zip(expected, request)))
            maxima["projected_command_windnormal_mps2"] = max(maxima["projected_command_windnormal_mps2"], abs(dot(boundaries["after_projection_world_mps2"], f.wind_basis.forward)))
        else:
            full_pn = half_pn = details["pn_mps2"]
        request_arrays["PN_t_all"].append(ts)
        request_arrays["full_PN_world_mps2"].append(full_pn)
        request_arrays["half_PN_world_same_state_mps2"].append(half_pn)
        request_arrays["half_to_full_horizontal_delta_mps2"].append(dot(sub(full_pn, half_pn), f.wind_basis.right))
        current = model.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        old_force = original_force.evaluate(state, ts, tuple(s["evaluated_old_held_rad"]))
        maxima["force_same_state_N"] = max(maxima["force_same_state_N"], norm(sub(old_force.wind_lift_force, current.wind_lift_force)))
        maxima["drag_same_state_N"] = max(maxima["drag_same_state_N"], norm(sub(old_force.wind_drag_force, current.wind_drag_force)))
        maxima["angular_same_state_rad_s2"] = max(maxima["angular_same_state_rad_s2"], max(abs(x-y) for x, y in zip(old_force.angular_acceleration, current.angular_acceleration)))
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
            maxima["K_same_state_equal"] = max(maxima["K_same_state_equal"], abs(k - original.current_control_k))
            beta = math.hypot(*f.local_incidence)
            factor_independent = 1. / float(np.sinc(beta / math.pi))
            direct.current_beta_rad = beta
            maxima["beta_current_local_equal_rad"] = max(maxima["beta_current_local_equal_rad"], abs(beta - d["beta_rad"]))
            maxima["factor_independent_sinc_equal"] = max(maxima["factor_independent_sinc_equal"], abs(factor_independent - d["inverse_sinc_factor"]))
            expected_cap = min(limits[axis], limits[axis] / k * factor_independent)
            maxima["newcap_independent_formula_equal_rad"] = max(maxima["newcap_independent_formula_equal_rad"], abs(expected_cap - d["effective_old_held_limit_rad"]))
            direct_result = direct.update_axis(axis, ts, e, memory[axis], pid, limits[axis], tick["gain"], k, original.current_force_nonlinear)
            maxima["held_PID_closure_rad"] = max(maxima["held_PID_closure_rad"], abs(direct_result[0] - tick["held_fin_rad"][axis]))
            maxima["memory_PID_closure"] = max(maxima["memory_PID_closure"], max(abs(x-y) for x, y in zip(direct_result[1], tick["pid_memory"][axis])))
            # Original one-tick counterfactual gets the candidate's identical previous memory and original K history.
            old_result = original_controller.update_axis(axis, ts, e, memory[axis], pid, limits[axis], tick["gain"], k, original.current_force_nonlinear)
            old_d = original_controller.diagnostics[axis]
            maxima["transport_ratio_original_equal"] = max(maxima["transport_ratio_original_equal"], abs(old_d["integral_transport_ratio"] - d["integral_transport_ratio"]))
            maxima["original_AW_limit_equal_rad"] = max(maxima["original_AW_limit_equal_rad"], abs(old_d["effective_old_held_limit_rad"] - d["original_AW_limit_old_rad"]))
            maxima["original_counterfactual_output_equal_rad"] = max(maxima["original_counterfactual_output_equal_rad"], abs(old_result[0] / tick["gain"] - d["original_counterfactual_saturated_old_rad"]))
            maxima["next_I_original_same_memory_equal_rad"] = max(maxima["next_I_original_same_memory_equal_rad"], abs(old_result[1][0] - direct_result[1][0]))
            if old_d["antiwindup_blocked"] != d["antiwindup_blocked"]:
                raise AssertionError("Original-cap antiwindup changed")
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
            else:
                effect["high_latch_extra_output_axis_ticks"] += int(delta > 1e-12)
            if abs(d["natural_old_rad"]) <= d["effective_old_held_limit_rad"]:
                maxima["unsaturated_PID_not_scaled_rad"] = max(maxima["unsaturated_PID_not_scaled_rad"], abs(direct_result[0] / tick["gain"] - d["natural_old_rad"]))
            maxima["raw_cap_excess_rad"] = max(maxima["raw_cap_excess_rad"], abs(direct_result[0]) - limits[axis])
            maxima["target_cap_excess"] = max(maxima["target_cap_excess"], abs(k * direct_result[0] / limits[axis]) - factor_independent)
            maxima["raw_I_limit_excess"] = max(maxima["raw_I_limit_excess"], abs(direct_result[1][0]) - pid["integral_limit"])
            memory[axis] = direct_result[1]
            for key, value in (("held", direct_result[0]), ("P", d["P_old_rad"]), ("I_provisional", d["I_provisional_old_rad"]),
                ("D", d["D_old_rad"]), ("effective_limit", d["effective_old_held_limit_rad"]),
                ("original_AW_limit", d["original_AW_limit_old_rad"]), ("original_counterfactual_output", d["original_counterfactual_saturated_old_rad"]),
                ("I_next", d["next_integral_old_rad"]), ("latch", d["nonlinear_latched"]),
                ("beta_rad", d["beta_rad"]), ("inverse_sinc_factor", d["inverse_sinc_factor"]), ("natural_PID_old_rad", d["natural_old_rad"]),
                ("AW", d["antiwindup_blocked"]), ("K", k), ("normalized_target", d["normalized_target_at_tick"]), ("error_G", e)):
                values[key].append(value)
        tick_arrays["t"].append(ts)
        tick_arrays["gain"].append(tick["gain"])
        for key, value in values.items():
            tick_arrays[key].append(value)
        if direct.previous_high_k != original_controller.previous_high_k or direct.nonlinear_latched != original_controller.nonlinear_latched:
            raise AssertionError("original high-only K lifecycle changed")
    fixed = {"config_and_raw_PID_equal": model.config == cfg, "five_coefficients_equal": model.coefficients == SurfaceCoefficients(),
        "drag_equal": model.drag_coefficients == coefficients, "PN_full_loft_S110_equal": model.relative_blend == 1. and float(cfg["guidance"]["pn_gain"]) == 6. and model.request_slew_g_s == 110. and [model.amplitude_coefficient,model.transition_coefficient] == [.75,1.3],
        "clock48": model.controller_hz == 48 and np.max(abs(np.asarray([t["time_s"] for t in result["ticks"]]) * 48 - np.arange(len(result["ticks"])))) < 1e-8,
        "plant_substeps4": model.plant_substeps == 4, "one_surface_0p4m": model.arm == .4}
    if max(maxima.values()) > 1e-10 or not all(fixed.values()):
        raise AssertionError({"maxima": maxima, "fixed": fixed})
    representatives = []
    for when in (0., .5, 1.5, 2.3, 3.5):
        j = int(np.argmin(abs(np.asarray(request_arrays["PN_t_all"]) - when)))
        representatives.append({k: np.asarray(v)[j].tolist() for k, v in request_arrays.items()})
    return {"max_residuals": maxima, "fixed": fixed, "cap_only_same_memory_effect": effect,
        "same_state_half_to_full_representative_PN": representatives,
        "physical_baseline_exact": physical_exact(result, baseline_full),
        "force_torque_record_closure": dec.audit.inverse.saved_closure(model, result)}, {k: np.asarray(v) for k, v in {**tick_arrays, **request_arrays}.items()}


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
    prior_cache = np.load(ROOT / "outputs/jari_offaxis/full_relative_clip/trajectories.npz")
    prior = dec.audit.drag.selected_arrays(prior_cache, cid + "__candidate__")
    for ax, field in zip(axes.flat, UNITS):
        for label, p, color in (("JARI reference", observed, "#147378"), ("Cached geometry baseline", baseline, "#813a8a"),
                ("Cached full-relative target1 clip", prior, "#939393"), ("Sinc output cap", candidate, "#ce7b2f")):
            ax.plot(p["t"], p[field], color=color, label=label)
        ax.set(xlabel="Original time (s)", ylabel=field + " (" + UNITS[field] + ")", xlim=(0, 3.5 if field != "speed_mps" else observed["t"][-1]))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(cid + ": actual free trajectory; only current beta/sin(beta) output bound; full horizontal relative PN unchanged\nOriginal cap-based AW/I lifecycle; frozen plant, geometry, raw PID and outer", fontsize=11)
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
    prior_cache = np.load(ROOT / "outputs/jari_offaxis/full_relative_clip/trajectories.npz")
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
        prior = drag.selected_arrays(prior_cache, cid + "__candidate__")
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
        prior_full = json.loads((ROOT / "outputs/jari_offaxis/full_relative_clip/full_runs" / (cid + ".json")).read_text())
        zero_offaxis_exact = physical_exact(result, prior_full) if cid == "jari_259_0" else None
        reports[cid] = {"separate_metrics": metrics, "first_positive_peaks": peaks(observed, baseline, prediction),
            "cached_full_relative_target1_comparison": drag.pair_metrics(observed, prior, prediction, tuple(UNITS), WINDOWS),
            "cached_full_relative_target1_first_peaks": {field: first_peak(prior, field) for field in ("body_yaw_rate_deg_s", "alpha_deg", "normal_specific_load_g")},
            "summary": result["summary"], "baseline_summary": baseline_full["summary"], "full_result_path": str(path),
            "baseline_receipt_restores_primary_arrays_exactly": restored_exact, "focused_checks": check,
            "zero_offaxis_cached_full_relative_physical_exact": zero_offaxis_exact}
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction)):
            fields.save_arrays(arrays, cid + "__" + label, p)
        fields.save_arrays(arrays, cid + "__cached_full_relative_target1", prior)
        arrays.update({cid + "__tick__" + k: v for k, v in ta.items()})
        plot_jari(cid, observed, baseline, prediction)
        print(json.dumps({"completed_free_case": cid, "summary": result["summary"], "first_peaks": reports[cid]["first_positive_peaks"],
            "early_primary_RMSE": {k: [metrics["early_0_3p5"]["metrics"][k][label]["rmse"] for label in ("old", "new")] for k in PRIMARY}}), flush=True)
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    return reports, {"new_Jari_integrations": new, "recovered_Jari_without_reintegration": recovered,
        "unique_Jari_cases": 3, "baseline_integrations": 0, "all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values())}


def run_native(coefficients, *, diagnostic=False):
    harness, fields, drag = dec.audit.fix.finite.harness, dec.audit.fix.finite.fields, dec.audit.drag
    cases = {c["id"]: c for c in harness.load_cases(dec.audit.fix.finite.comparison.SNAPSHOT)}
    cached = np.load(geometry.OUT / "native_trajectories.npz")
    control_cached = np.load(dec.audit.fix.OUTPUT / "trajectories.npz") if diagnostic else None
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
        for w in metrics.values():
            for field, metric in w["metrics"].items():
                metric["units"] = harness.UNITS[field]
        primary = ("speed_kmh", "alpha_deg", "pitch_deg") if diagnostic else ("speed_kmh", "alpha_deg")
        gate = {w + "__" + k: m["new"]["rmse"] <= 1.1 * m["old"]["rmse"] + 1e-12
            for w, ws in metrics.items() for k, m in ws["metrics"].items() if k in primary}
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
        if diagnostic:
            control_baseline = drag.selected_arrays(control_cached, cid + "__control_fix__")
            control_full = json.loads((dec.audit.fix.OUTPUT / "full_runs" / (cid + ".json")).read_text())
            control_restored = harness.sample_series(control_full["samples"], x)
            control_exact = all(np.array_equal(v, control_baseline[k]) for k, v in control_restored.items())
            if not control_exact:
                raise AssertionError("native ControlFix cache receipt")
            control_metrics = drag.pair_metrics(observed, control_baseline, prediction,
                ("speed_kmh", "alpha_deg", "pitch_deg", "omega_rad_s", "gamma_deg", "y_m"), windows)
            for w in control_metrics.values():
                for field, metric in w["metrics"].items():
                    metric["units"] = harness.UNITS[field]
            reports[cid].update(harness_id=x["id"], dataset=x["dataset"], version=x["version"],
                source_file=x["source_file"], source_count=x["source_count"], scenario=x["scenario"],
                creation=x["creation"], initial_state_used=dataclasses.asdict(initial),
                creation_quaternion_full_roll_preserved=True,
                alpha_semantics="Signed pitch wind incidence in degrees, matching the current native harness; total vector AoA is not substituted.",
                pitch_semantics="Signed body pitch in degrees, matching the current native harness.",
                primary_compatibility_fields=list(primary),
                cached_ControlFix_receipt_restores_arrays_exactly=control_exact,
                cached_ControlFix_separate_metrics=control_metrics,
                cached_geometry_summary=baseline_full["summary"], cached_ControlFix_summary=control_full["summary"])
            fields.save_arrays(arrays, cid + "__cached_ControlFix", control_baseline)
        for label, p in (("reference", observed), ("baseline", baseline), ("candidate", prediction)):
            fields.save_arrays(arrays, cid + "__" + label, p)
        arrays.update({cid + "__tick__" + k: v for k, v in ta.items()})
        native_plot(cid, observed, baseline, prediction, diagnostic=diagnostic,
            control_baseline=control_baseline if diagnostic else None)
        print(json.dumps({"completed_native_case": cid, "passes": reports[cid]["passes"], "summary": result["summary"]}), flush=True)
    np.savez_compressed(OUT / "native_trajectories.npz", **arrays)
    return reports, {"new_native_integrations": new, "recovered_native_without_reintegration": recovered,
        "unique_native_cases": 3, "baseline_integrations": 0, "all_arrays_finite": all(np.isfinite(v).all() for v in arrays.values())}


def native_plot(cid, observed, baseline, candidate, *, diagnostic=False, control_baseline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for ax, field in zip(axes.flat, ("speed_kmh", "alpha_deg", "pitch_deg", "omega_rad_s", "gamma_deg", "y_m")):
        series = [("Native replay", observed, "#147378"), ("Cached geometry baseline", baseline, "#813a8a")]
        if control_baseline is not None:
            series.append(("Cached ControlFix", control_baseline, "#939393"))
        series.append(("Sinc output cap", candidate, "#ce7b2f"))
        for label, p, color in series:
            ax.plot(p["t"], p[field], color=color, label=label)
        ax.set(xlabel="Original time (s)", ylabel=field + " (" + dec.audit.fix.finite.harness.UNITS[field] + ")")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(cid + (": independent native compatibility diagnostic; original JARI strict gate remains failed" if diagnostic else
        ": actual free sinc-only-output-bound validation, frozen current native fixture"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / (cid + "_free.png"), dpi=150)
    plt.close(fig)


def native_diagnostic_snapshot():
    """Protect previous artifacts; exclude only this run's native files and the authorized parallel worker."""
    folders = ("src", "config", "configs", "docs", "data", "scripts", "outputs")
    protected = {}
    for folder in folders:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            if str(relative) == "scripts/analyze_r771_speed_residual.py" or str(relative).startswith("outputs/jari_offaxis/speed_residual/"):
                continue
            if path.parent == OUT and path.name.startswith("native_"):
                continue
            stat = path.stat()
            protected[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return protected


def native_diagnostic_main():
    """Root-authorized three free compatibility runs, independent of the unchanged failed JARI gate."""
    before = native_diagnostic_snapshot()
    original_jari = json.loads((OUT / "results.json").read_text())
    if original_jari["Jari_gate"]["passes"] or original_jari["candidate_accepted_by_predeclared_gate"]:
        raise AssertionError("This diagnostic requires the existing unaccepted JARI candidate classification")
    protocol = {"mode": "native-diagnostic", "candidate": "unchanged SincCapGame: full horizontal relative PN and only-output sinc cap",
        "cases": list(NATIVE), "forward_count": 3, "baseline_integrations": 0, "JARI_integrations": 0,
        "windows": [[n, lo, None if math.isinf(hi) else hi] for n, lo, hi in (*dec.WINDOWS, *dec.audit.fix.finite.harness.WINDOWS)],
        "compatibility_rule": "Separately speed_kmh, signed pitch-incidence alpha_deg, and signed body pitch_deg RMSE in every existing window/full no more than10percent worse than cached geometry baseline",
        "load_rule": "No independent load observation; report prediction differences only",
        "original_Jari_gate_unchanged": True, "candidate_accepted_by_original_Jari_gate": False,
        "runtime_inputs": "creation own state and current external target only; no source missile trajectory or derivatives",
        "unchanged": "raw PID, caps/AW and high-only I/K lifecycle, five plant coefficients, force/drag/angular law, geometry, loft/vertical PN/S110/50G,48Hz+4substeps,one0.4m surface"}
    # Predeclare before native trajectories are observed or integrated.
    dec.audit.slew.write_json(OUT / "native_results.json", {"status": "running", "protocol": protocol,
        "candidate_accepted_by_original_Jari_gate": False})
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    reports, verification = run_native(coefficients, diagnostic=True)
    protected = before == native_diagnostic_snapshot()
    verification.update(JARI_integrations=0, protected_previous_artifacts_size_mtime_unchanged=protected,
        parallel_authorized_exclusions=["scripts/analyze_r771_speed_residual.py", "outputs/jari_offaxis/speed_residual/"],
        original_Jari_results_and_trajectories_unchanged=protected, baseline_integrations=0,
        fits_scans_new_coefficients=0, source_missile_data_after_creation_in_runtime=False,
        core_config_data_docs_changes=0, controller_hz=48, plant_substeps=4)
    if not protected or not verification["all_arrays_finite"] or verification["new_native_integrations"] + verification["recovered_native_without_reintegration"] != 3:
        raise AssertionError(verification)
    passed = all(r["passes"] for r in reports.values())
    report = {"task": "Independent native compatibility diagnostic of the current sinc-only-output-cap candidate",
        "protocol": protocol, "cases": reports, "verification": verification,
        "compatibility_label": "passes_predeclared_native_compatibility" if passed else "fails_predeclared_native_compatibility",
        "all_native_compatibility_rules_passed": passed, "candidate_accepted_by_original_Jari_gate": False,
        "original_Jari_failed_rules": [k for k, v in original_jari["Jari_gate"]["rules"].items() if not v],
        "native_game_cap_identified": False, "evidence_grade": "D", "default_model_changed": False}
    dec.audit.slew.write_json(OUT / "native_results.json", report)
    print(json.dumps({"output": str(OUT / "native_results.json"), "compatibility_label": report["compatibility_label"],
        "candidate_accepted_by_original_Jari_gate": False, "verification": verification}), flush=True)


def main():
    before = snapshot()
    OUT.mkdir(parents=True, exist_ok=True)
    # This protocol is written before loading trajectory observations or running.
    protocol = {"candidate": "Only output cap=min(raw,raw/K*beta/sin(beta)); beta=current local norm; original AW limit high-before raw/high-after raw/K",
        "unchanged": "raw PID/Ilim, original-cap .99 AW/counterfactual output and high-only I transport, gain-last, five plant coefficients, geometry, force norm, angular law, drag, loft, PN/S110, 48Hz+4substep, one 0.4m surface",
        "Jari_gate": "60 first yaw-rate and alpha amplitudes closer; 60 speed/alpha/load each improve early 0-3.5 and full and pitch no>10% worse;0/30 primary early/full no>10% worse",
        "native_trigger": "Only if every Jari gate passes, run current H110/H27.5/H55native once each; each speed/alpha full and existing windows no>10% worse; load prediction change only",
        "report": "Separate first peaks/timing and every variable in startup/platform/unloading/early/full; no combined score; unfavorable windows remain"}
    dec.audit.slew.write_json(OUT / "protocol.json", protocol)
    coefficients = dec.audit.drag.DragCoefficients(**json.loads((dec.audit.drag.OUTPUT / "fit.json").read_text())["coefficients"])
    numeric_beta = np.array([0., math.nextafter(0., 1.), np.finfo(float).eps,
        math.sqrt(np.finfo(float).eps), 1e-6, math.pi / 6, math.pi / 4, math.nextafter(math.pi / 2, 0.)])
    numeric_factor = np.array([inverse_sinc(float(v)) for v in numeric_beta])
    numeric_reference = 1. / np.sinc(numeric_beta / math.pi)
    numeric_check = {"beta_rad": numeric_beta.tolist(), "runtime_factor": numeric_factor.tolist(),
        "independent_numpy_inverse_sinc": numeric_reference.tolist(),
        "max_abs_difference": float(abs(numeric_factor - numeric_reference).max()),
        "zero_analytic_limit_exact": bool(numeric_factor[0] == 1.)}
    if numeric_check["max_abs_difference"] > 1e-12 or not numeric_check["zero_analytic_limit_exact"]:
        raise AssertionError(numeric_check)
    inputs = dec.audit.inverse.source_inputs()
    reports, verification = run_jari(inputs, coefficients)
    gate = jari_gate(reports)
    native, native_verification = {}, {"new_native_integrations": 0, "unique_native_cases": 0, "not_run_reason": "Jari gate did not pass"}
    if gate["passes"]:
        native, native_verification = run_native(coefficients)
    protected = before == snapshot()
    verification.update(native=native_verification, protected_size_mtime_unchanged=protected,
        fits_scans_new_coefficients=0, baseline_integrations=0, source_data_in_runtime=False,
        core_config_data_docs_changes=0, controller_hz=48, plant_substeps=4,
        independent_sinc_numerical_check=numeric_check)
    if not protected or not verification["all_arrays_finite"] or not native_verification.get("all_arrays_finite", True):
        raise AssertionError(verification)
    accepted = gate["passes"] and bool(native) and all(r["passes"] for r in native.values())
    dec.audit.slew.write_json(OUT / "results.json", {"task": "War Thunder sinc output cap with full relativePN/original angular law and original-cap AW, actual free test",
        "protocol": protocol, "cases": reports, "Jari_gate": gate, "native_cases": native,
        "verification": verification, "candidate_accepted_by_predeclared_gate": accepted,
        "native_game_cap_identified": False, "evidence_grade": "D", "default_model_changed": False})
    print(json.dumps({"output": str(OUT), "Jari_gate": gate, "accepted": accepted, "verification": verification}), flush=True)


if __name__ == "__main__":
    native_diagnostic_main() if "--native-diagnostic" in sys.argv else main()
