#!/usr/bin/env python3
"""One script-local polar direction allocation for a War Thunder game unit.

The finite-force geometry, magnitude, angular law, requests and raw PID fields
are read-only. No singular-value gain inversion or data-derived coefficients.
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
import check_r771_lift_direction as geometry
from aim120_model.aerodynamics import BodyAxes, body_axes_for_state
from aim120_model.control import scheduled_pid_values
from aim120_model.math3d import add, dot, norm, scale, sub
from aim120_model.r771_game_control import R771GameControl
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

dec = geometry.dec
fix = dec.audit.fix
finite = fix.finite
fields = finite.fields
drag = dec.audit.drag
slew = dec.audit.slew
OUT = ROOT / "outputs/jari_offaxis/force_allocation"
CASES = geometry.CASES
WINDOWS = (*dec.WINDOWS, ("full_0p5_3p5", .5, 3.5), ("full_common", 0., math.inf))
UNITS = {**finite.UNITS, "body_pitch_deg": "deg", "body_pitch_rate_deg_s": "deg/s"}
EPS = np.finfo(float).eps


def snapshot():
    other_script = ROOT / "scripts/check_r771_nonlinear_moment.py"
    other_output = ROOT / "outputs/jari_offaxis/nonlinear_moment"
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis")
        for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
        and OUT not in p.parents and p != Path(__file__).resolve() and p != other_script and other_output not in p.parents}


def finite_lift_normal_G(body, wind, local, pressure_area_ca_over_mass):
    """The exact existing finite lift geometry at fixed body/wind/q/m.

    Independent local incidence variations here define a *partial* Jacobian.
    Thrust, drag, attitude dynamics and movement of the bases are not inverted.
    """
    s = math.hypot(*local)
    ratio = 1. if s <= finite.ALPHA_S else 1./(1.+((s-finite.ALPHA_S)/finite.WIDTH)**2)
    original = add(scale(wind.up, local[0]*pressure_area_ca_over_mass*ratio),
        scale(wind.right, local[1]*pressure_area_ca_over_mass*ratio))
    lift = geometry.geometric_lift(body, wind, local, original)
    return np.asarray((dot(lift, wind.up), dot(lift, wind.right))) / G0_MPS2


class AllocationUndefined(ValueError):
    pass


def force_allocation(body, wind, local, pressure_area_ca_over_mass):
    """C=V U.T; two central-difference resolutions reject unresolved rank.

    h=cuberoot(machine epsilon)*max(1,|local|) is numerical, not calibrated.
    No previous-C hold, smoothing, pseudoinverse or arbitrary rank extension.
    """
    local = np.asarray(local, dtype=float)
    h = float(np.cbrt(EPS)*max(1., np.linalg.norm(local)))
    if np.max(abs(local))+h >= math.pi/2:
        raise AllocationUndefined("Jacobian stencil leaves the declared forward local-flow branch")
    def derivative(step):
        columns = []
        for axis in range(2):
            delta = np.zeros(2); delta[axis] = step
            columns.append((finite_lift_normal_G(body, wind, local+delta, pressure_area_ca_over_mass)
                -finite_lift_normal_G(body, wind, local-delta, pressure_area_ca_over_mass))/(2*step))
        return np.column_stack(columns)
    coarse = derivative(h); jacobian = derivative(h/2)
    u, sigma, vt = np.linalg.svd(jacobian)
    uncertainty = max(128*EPS*float(sigma[0]), 4*float(np.linalg.norm(jacobian-coarse, ord=2)))
    if not np.isfinite(jacobian).all() or not np.isfinite(sigma).all() or sigma[1] <= uncertainty:
        raise AllocationUndefined("Polar direction is not unique/resolved near rank loss: " + json.dumps({
            "sigma_G_per_rad": sigma.tolist(), "numerical_uncertainty_G_per_rad": uncertainty, "local_rad": local.tolist()}))
    allocation = vt.T @ u.T
    return allocation, {"J_G_per_rad": jacobian.tolist(), "C": allocation.tolist(), "sigma_G_per_rad": sigma.tolist(),
        "det_J": float(np.linalg.det(jacobian)), "det_C": float(np.linalg.det(allocation)), "stencil_h_rad": h,
        "numerical_uncertainty_G_per_rad": uncertainty,
        "orthogonality_residual": float(np.max(abs(allocation.T @ allocation-np.eye(2))))}


class AllocationControl(R771GameControl):
    """Raw wind error/D memory, integral in unchanged body old-held coordinates."""
    def update_vector(self, time_s, errors, memories, pid, limits, gain, k, nonlinear, allocation, jacobian_details):
        if gain <= 1e-12: raise ValueError("Calling integrator must preserve original inactive initialization")
        if any(t is not None and time_s <= t for t in self.last_time): raise ValueError("Exactly one vector update per original tick")
        if abs(time_s*self.controller_hz-round(time_s*self.controller_hz)) > 1e-8: raise ValueError("Non-native controller clock")
        self.nonlinear_latched = self.nonlinear_latched or bool(nonlinear)
        dt = 1./self.controller_hz
        errors = tuple(float(x) for x in errors)
        raw_derivatives = tuple(memory[2]+dt/(1e-9+dt)*((e-memory[1])/dt-memory[2]) for e, memory in zip(errors, memories))
        allocated_errors = allocation @ np.asarray(errors)
        allocated_derivatives = allocation @ np.asarray(raw_derivatives)
        results = []
        for axis, (error, memory, limit) in enumerate(zip(errors, memories, limits)):
            old_i = memory[0]; ratio = 1.
            if self.nonlinear_latched:
                previous_k = self.previous_high_k[axis]
                ratio = 1. if previous_k is None else previous_k/k
                transported = dec.audit.clip(old_i*ratio, pid["integral_limit"])
                cap = min(limit, limit/k)
            else:
                transported = old_i; cap = limit
            increment = float(pid["i"])*float(allocated_errors[axis])*dt
            provisional = dec.audit.clip(transported+increment, pid["integral_limit"])
            p = float(pid["p"])*float(allocated_errors[axis]); d = float(pid["d"])*float(allocated_derivatives[axis])
            before = p+provisional+d; post = dec.audit.clip(before, cap)
            blocked = abs(post) >= .99*cap and increment*post > 0
            next_i = transported if blocked else provisional
            result = (gain*post, (next_i, error, raw_derivatives[axis])); results.append(result)
            self.last_time[axis] = time_s; self.cached[axis] = result
            if self.nonlinear_latched: self.previous_high_k[axis] = k
            self.diagnostics[axis] = {"time_s": time_s, "axis": axis, "error_g": error, "K_target_per_old_held": k,
                "nonlinear_latched": self.nonlinear_latched, "integral_transport_ratio": ratio,
                "transported_integral_old_rad": transported, "integral_target_rad": k*next_i,
                "P_target_rad": k*p, "D_target_rad": k*d, "target_candidate_limit_rad": limit,
                "effective_old_held_limit_rad": cap, "original_old_held_limit_rad": limit,
                "antiwindup_blocked": blocked, "held_old_rad": result[0], "effective_target_rad": k*result[0],
                "normalized_target_at_tick": k*result[0]/limit, "gain": gain,
                "allocated_error_G": float(allocated_errors[axis]), "raw_filtered_derivative_G_s": raw_derivatives[axis],
                "allocated_derivative_G_s": float(allocated_derivatives[axis]), "P_old_rad": p, "I_provisional_old_rad": provisional,
                "D_old_rad": d, "allocated_integral_increment_rad": increment, "prelimit_old_rad": before, "postlimit_old_rad": post,
                "allocation": jacobian_details}
        return tuple(results)


class ForceAllocationGame(geometry.LiftDirectionGame):
    """Original force/angular plant and request; one vector controller adapter."""
    def initialize(self, state):
        state = super().initialize(state)
        self.game_control = AllocationControl()
        self.allocation_memories = ((0., 0., 0.), (0., 0., 0.))
        self.allocation_previous_memories = self.allocation_memories
        self.allocation_results = None
        return state

    def request(self, state, target, time_s, evaluation):
        request, details = super().request(state, target, time_s, evaluation)
        self.allocation_errors = tuple(r-f for r, f in zip(request, evaluation.feedback_g))
        self.allocation_state = state; self.allocation_evaluation = evaluation
        return request, details

    def _pid(self, error, memory, pid, fin_limit, gain):
        axis = self.current_control_axis; self.current_control_axis += 1
        if axis == 0:
            self.allocation_previous_memories = self.allocation_memories
            f = self.allocation_evaluation
            c, details = force_allocation(body_axes_for_state(self.allocation_state), f.wind_basis, f.local_incidence,
                f.qbar*self.area*self.coefficients.ca/f.mass)
            limits = tuple(math.radians(self.config["aerodynamics"][key]) for key in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg"))
            self.allocation_results = self.game_control.update_vector(self.current_control_time, self.allocation_errors,
                self.allocation_previous_memories, pid, limits, gain, self.current_control_k, self.current_force_nonlinear, c, details)
            self.allocation_memories = tuple(result[1] for result in self.allocation_results)
        if error != self.allocation_errors[axis] or tuple(memory) != self.allocation_previous_memories[axis]:
            raise AssertionError("Vector allocation does not match the original current errors/previous memories")
        if axis == 1: self.control_history[self.current_control_time] = copy.deepcopy(self.game_control.diagnostics)
        return self.allocation_results[axis]

    def run(self, initial_state, target_at, *, end_time_s):
        result = super().run(initial_state, target_at, end_time_s=end_time_s)
        result["model"].update(runtime_controller="script-local AllocationControl", controller_mechanism="inverse polar orthogonal direction of the current partial finite-lift Jacobian",
            raw_PID_input_units="raw wind-normal G error; orthogonal mapping into body old-held command coordinates",
            memory_error_derivative_units="previous raw wind-normal G error and its original filtered G/s derivative",
            P_D_target_terms="K*Kp*(C e) and K*Kd*(C raw_filtered_d); C switching is not differentiated",
            integral_increment="original Ki*(C e)/48 in original body old-held coordinates; original K transport and raw I clipping",
            antiwindup="original .99 postlimit guard using allocated Ki*C e increment direction",
            request_projection_added=False, target1_authority_relaxed=False, singular_value_inversion=False,
            allocation_evidence_grade="D opt-in partial-force direction hypothesis", reference_data_in_runtime=False)
        return result


def make_model(config, coefficients):
    return ForceAllocationGame(config, request_slew_g_s=110., drag_coefficients=coefficients, coefficients=SurfaceCoefficients(),
        force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")


def observed_fields(item):
    observed = dict(item["observed"])
    q = fields.gap.continuous_quaternions(item["ref"]["quaternion_xyzw"][:, [3, 0, 1, 2]])
    observed["body_pitch_deg"] = fields.gap.trajectory_fields(item["ref"]["t"], item["ref"]["position_xyz_m"], item["ref"]["velocity_xyz_mps"],
        item["run"]["scenario"]["target_position"], item["run"]["scenario"]["target_velocity"], q)["body_pitch_deg"]
    observed["body_pitch_rate_deg_s"] = -np.degrees(item["ref"]["omega_xyz_export_units"][:, 2])
    return observed


def tick_arrays(result):
    rows = [tick for tick in result["ticks"] if tick["control_runtime_diagnostics"]]
    a = {key: np.asarray([tick[key] for tick in rows]) for key in ("time_s", "request_g", "feedback_g", "held_fin_rad", "pid_memory")}
    a["t"] = a.pop("time_s")
    for key in ("allocated_error_G", "raw_filtered_derivative_G_s", "allocated_derivative_G_s", "P_old_rad", "I_provisional_old_rad", "D_old_rad",
            "allocated_integral_increment_rad", "transported_integral_old_rad", "integral_transport_ratio", "prelimit_old_rad", "postlimit_old_rad",
            "effective_old_held_limit_rad", "antiwindup_blocked", "K_target_per_old_held"):
        a[key] = np.asarray([[d[key] for d in tick["control_runtime_diagnostics"]] for tick in rows])
    for key in ("J_G_per_rad", "C", "sigma_G_per_rad", "det_C", "numerical_uncertainty_G_per_rad", "orthogonality_residual"):
        a[key] = np.asarray([tick["control_runtime_diagnostics"][0]["allocation"][key] for tick in rows])
    return a


def allocation_windows(a):
    report = {}
    for name, lo, hi in WINDOWS:
        mask = (a["t"] >= lo-1e-9) & (a["t"] <= hi+1e-9)
        report[name] = {"ticks": int(mask.sum()), "reflection_ticks": int(np.sum(mask & (a["det_C"] < 0))),
            "minimum_sigma_G_per_rad": float(a["sigma_G_per_rad"][mask, 1].min()) if mask.any() else None,
            "minimum_sigma_over_uncertainty": float(np.min(a["sigma_G_per_rad"][mask, 1]/a["numerical_uncertainty_G_per_rad"][mask])) if mask.any() else None,
            "max_abs_C_minus_I": float(np.max(abs(a["C"][mask]-np.eye(2)))) if mask.any() else None}
    signs = a["det_C"] < 0
    report["reflection_switch_times_s"] = a["t"][np.r_[False, signs[1:] != signs[:-1]]].tolist()
    return report


def verify_result(model, result, config, initial, target_at, coefficients):
    limits = tuple(math.radians(config["aerodynamics"][key]) for key in ("horizontal_fin_aoa_limit_deg", "vertical_fin_aoa_limit_deg"))
    memories = ((0., 0., 0.), (0., 0., 0.)); previous_k = (None, None); high = False
    maximum = {"held_rad": 0., "next_memory": 0., "transport_ratio": 0., "allocation_orthogonality": 0.,
        "J_recompute_G_per_rad": 0., "C_recompute": 0., "lift_map_current_G": 0., "frozen_force_N": 0., "frozen_angular_law_rad_s2": 0., "request_same_state_G": 0.}
    baseline = geometry.make_model(config, coefficients); baseline.initialize(dataclasses.replace(initial))
    bytime = {s["time_s"]: s for s in result["samples"]}
    for tick in result["ticks"]:
        state = finite.sample_state(bytime[tick["time_s"]]); held = tuple(tick["evaluated_old_held_rad"])
        f = model.evaluate(state, tick["time_s"], held)
        request, _ = baseline.request(state, target_at(tick["time_s"]), tick["time_s"], f)
        maximum["request_same_state_G"] = max(maximum["request_same_state_G"], max(abs(x-y) for x, y in zip(request, tick["request_g"])))
        if tick["gain"] <= 1e-12:
            if tick["pid_memory"] != ((0., 0., 0.), (0., 0., 0.)) and tick["pid_memory"] != [[0., 0., 0.], [0., 0., 0.]]:
                raise AssertionError("Inactive initialization changed")
            continue
        diagnostics = tick["control_runtime_diagnostics"]; record = diagnostics[0]["allocation"]
        c = np.asarray(record["C"])
        rebuilt_c, rebuilt = force_allocation(body_axes_for_state(state), f.wind_basis, f.local_incidence, f.qbar*model.area*model.coefficients.ca/f.mass)
        maximum["J_recompute_G_per_rad"] = max(maximum["J_recompute_G_per_rad"], float(np.max(abs(np.asarray(rebuilt["J_G_per_rad"])-record["J_G_per_rad"]))))
        maximum["C_recompute"] = max(maximum["C_recompute"], float(np.max(abs(rebuilt_c-c))))
        maximum["allocation_orthogonality"] = max(maximum["allocation_orthogonality"], float(np.max(abs(c.T @ c-np.eye(2)))))
        mapped = finite_lift_normal_G(body_axes_for_state(state), f.wind_basis, f.local_incidence, f.qbar*model.area*model.coefficients.ca/f.mass)
        maximum["lift_map_current_G"] = max(maximum["lift_map_current_G"], float(np.max(abs(mapped-np.asarray(dec.projected(scale(f.wind_lift_force, 1/f.mass), f.wind_basis))))))
        errors = np.asarray(tick["request_g"])-tick["feedback_g"]
        pid = scheduled_pid_values(config["control"]["pid"], (round(tick["time_s"]*48)+1)/48)
        raw_d = np.asarray([m[2]+dec.DT/(1e-9+dec.DT)*((e-m[1])/dec.DT-m[2]) for e, m in zip(errors, memories)])
        allocated_e, allocated_d = c @ errors, c @ raw_d
        k = diagnostics[0]["K_target_per_old_held"]
        high = high or math.hypot(*f.local_incidence) > finite.ALPHA_S
        updated = []
        for axis, (e, m, limit, d) in enumerate(zip(errors, memories, limits, diagnostics)):
            ratio = previous_k[axis]/k if high and previous_k[axis] is not None else 1.
            transported = dec.audit.clip(m[0]*ratio, pid["integral_limit"]) if high else m[0]
            increment = pid["i"]*allocated_e[axis]/48
            provisional = dec.audit.clip(transported+increment, pid["integral_limit"])
            cap = min(limit, limit/k) if high else limit
            output = pid["p"]*allocated_e[axis]+provisional+pid["d"]*allocated_d[axis]
            post = dec.audit.clip(output, cap)
            blocked = abs(post) >= .99*cap and increment*post > 0
            next_memory = (transported if blocked else provisional, e, raw_d[axis])
            maximum["held_rad"] = max(maximum["held_rad"], abs(tick["gain"]*post-tick["held_fin_rad"][axis]))
            maximum["next_memory"] = max(maximum["next_memory"], max(abs(x-y) for x, y in zip(next_memory, tick["pid_memory"][axis])))
            maximum["transport_ratio"] = max(maximum["transport_ratio"], abs(ratio-d["integral_transport_ratio"]))
            if blocked != d["antiwindup_blocked"] or high != d["nonlinear_latched"] or abs(cap-d["effective_old_held_limit_rad"]) > 1e-12:
                raise AssertionError("Allocated AW/target limit/transport did not retain the registered order")
            updated.append(next_memory)
        memories = tuple(updated)
        if high: previous_k = (k, k)
    for s in result["samples"]:
        state = finite.sample_state(s); held = tuple(s["evaluated_old_held_rad"])
        new = model.evaluate(state, s["time_s"], held); old = baseline.evaluate(state, s["time_s"], held)
        maximum["frozen_force_N"] = max(maximum["frozen_force_N"], norm(sub(old.aerodynamic_force, new.aerodynamic_force)))
        maximum["frozen_angular_law_rad_s2"] = max(maximum["frozen_angular_law_rad_s2"], max(abs(x-y) for x, y in zip(old.angular_acceleration, new.angular_acceleration)))
    clock = np.asarray([tick["time_s"] for tick in result["ticks"]])
    fixed = model.config == config and model.coefficients == SurfaceCoefficients() and model.drag_coefficients == coefficients and model.relative_blend == .5 and model.request_slew_g_s == 110. and [model.amplitude_coefficient, model.transition_coefficient] == [.75, 1.3]
    if max(maximum.values()) > 1e-10 or not fixed or model.controller_hz != 48 or model.plant_substeps != 4 or np.max(abs(clock*48-np.arange(len(clock)))) > 1e-8:
        raise AssertionError((maximum, fixed))
    return {"maximum_closure_residuals": maximum, "all_raw_config_coefficients_PN_loft_S110_equal": fixed,
        "controller_hz": model.controller_hz, "plant_substeps": model.plant_substeps, "same_station_arm_m": model.arm,
        "raw_and_target_bounds": fix.verify_controls(model, result), "force_moment_closure": dec.audit.inverse.saved_closure(model, result)}


def plot_case(cid, observed, baseline, candidate, ticks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    specs = (("speed_mps", "Speed (m/s)"), ("alpha_deg", "AoA (deg)"), ("normal_specific_load_g", "Normal specific load (G)"),
        ("body_pitch_deg", "Body pitch (deg)"), ("body_yaw_deg", "Body yaw (deg)"), ("body_yaw_rate_deg_s", "Body yaw rate (deg/s)"))
    for ax, (key, unit) in zip(axes.flat, specs):
        for label, p, color in (("JARI interim reference", observed, "#177277"), ("Cached geometry / original controller", baseline, "#78337b"),
                ("Geometry / polar direction allocation", candidate, "#ce772e")):
            ax.plot(p["t"], p[key], label=label, color=color)
        ax.set(xlabel="Original clock (s)", ylabel=unit, xlim=(0., min(observed["t"][-1], candidate["t"][-1]) if key == "speed_mps" else 3.5))
        ax.grid(alpha=.17); ax.legend(fontsize=7, loc="best")
    fig.suptitle(cid + ": actual free integration; only bounded orthogonal allocation of raw PID error/derivative\nOriginal finite-force geometry, moment law, PN/loft/S110 and all coefficients retained", fontsize=11)
    fig.tight_layout(rect=(0., 0., 1., .94)); fig.savefig(OUT / (cid + "_free.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    t = ticks["t"]
    for axis, label in enumerate(("pitch", "yaw")):
        axes[0, axis].plot(t, ticks["request_g"][:, axis]-ticks["feedback_g"][:, axis], label="raw wind G error", color="#177277")
        axes[0, axis].plot(t, ticks["allocated_error_G"][:, axis], label="C e in body command coordinates", color="#ce772e")
        axes[0, axis].set_ylabel(label + " error (G)")
        for key, text, color in (("P_old_rad", "P", "#177277"), ("I_provisional_old_rad", "I provisional", "#78337b"), ("D_old_rad", "D from raw wind derivative", "#ce772e")):
            axes[1, axis].plot(t, np.degrees(ticks[key][:, axis]), label=text, color=color)
        axes[1, axis].plot(t, np.degrees(ticks["held_fin_rad"][:, axis]), label="postlimit × flight gain", color="#222", lw=1.)
        axes[1, axis].set_ylabel(label + " old-held terms (deg)")
        for col in range(2): axes[2, axis].plot(t, ticks["C"][:, axis, col], label="C["+str(axis)+","+str(col)+"]")
        axes[2, axis].set_ylabel(label + " allocation row"); axes[2, axis].set_xlabel("Original clock (s)")
    for ax in axes.flat:
        ax.set_xlim(14/48, 3.5); ax.grid(alpha=.17); ax.axhline(0., color="#aaa", lw=.6)
        for edge in (.7, 2.3): ax.axvline(edge, color="#aaa", lw=.6)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(cid + ": allocation direction and raw-memory PID decomposition; same candidate free trajectory", fontsize=11)
    fig.tight_layout(rect=(0., 0., 1., .95)); fig.savefig(OUT / (cid + "_allocation.png"), dpi=160); plt.close(fig)


def run_jari(inputs, coefficients):
    cache = np.load(geometry.OUT / "free_trajectories.npz")
    arrays = {}; reports = {}; runs = {}; new = 0; recovered = 0
    full = OUT / "full_runs"; full.mkdir(exist_ok=True)
    for cid in CASES:
        item = inputs[cid]; cfg = copy.deepcopy(item["config"]); model = make_model(cfg, coefficients)
        path = full / (cid + ".json")
        if path.exists():
            result = json.loads(path.read_text()); model.initialize(dataclasses.replace(item["initial"])); recovered += 1
        else:
            try:
                result = model.run(dataclasses.replace(item["initial"]), item["target_at"], end_time_s=cfg["performance"]["lifetime_s"])
            except AllocationUndefined as error:
                return reports, arrays, runs, {"new_integrations_completed": new, "recovered_without_reintegration": recovered, "rejected_case": cid, "reason": str(error)}
            slew.write_json(path, result, True); new += 1
        p = fields.prediction_fields(result, model, item["run"]["scenario"])
        p["body_pitch_rate_deg_s"] = np.degrees([s["pitch_rate_rad_s"] for s in result["samples"]])
        baseline = drag.selected_arrays(cache, cid + "__candidate__")
        old_full = json.loads((geometry.OUT / "full_runs" / (cid + ".json")).read_text())
        baseline["body_pitch_rate_deg_s"] = np.degrees([s["pitch_rate_rad_s"] for s in old_full["samples"]])
        observed = observed_fields(item); ticks = tick_arrays(result)
        metrics = drag.pair_metrics(observed, baseline, p, tuple(UNITS), WINDOWS)
        for window in metrics.values():
            for key, metric in window["metrics"].items(): metric["units"] = UNITS[key]
        reports[cid] = {"metrics_separate_units": metrics, "summary": result["summary"], "cached_geometry_summary": old_full["summary"],
            "full_result_path": str(path), "allocation_windows": allocation_windows(ticks),
            "extrema": {name: fix.trim.response_extrema(prediction) for name, prediction in (("reference", observed), ("cached_geometry", baseline), ("candidate", p))},
            "events": drag.event_summary(item["ref"], p, result, item["run"]["summary"])}
        for label, prediction in (("reference", observed), ("baseline_geometry", baseline), ("candidate", p)):
            fields.save_arrays(arrays, cid + "__" + label, prediction)
        arrays.update({cid + "__controller__" + key: value for key, value in ticks.items()})
        runs[cid] = (model, result, cfg, item["initial"], item["target_at"])
        plot_case(cid, observed, baseline, p, ticks)
        print(json.dumps({"free_case": cid, "summary": result["summary"]}), flush=True)
    return reports, arrays, runs, {"new_integrations_completed": new, "recovered_without_reintegration": recovered, "baseline_integrations": 0}


def next_native_gate(reports):
    if len(reports) != 3: return {"passed": False, "reason": "Candidate not defined for all three free cases"}
    zero = reports[CASES[0]]["metrics_separate_units"]["full_common"]["metrics"]
    zero_ok = all(metric["prediction_delta"]["max_abs"] <= 1e-10 for metric in zero.values())
    axes = {}
    for cid in CASES[1:]:
        metrics = reports[cid]["metrics_separate_units"]["full_0p5_3p5"]["metrics"]
        pitch = metrics["body_pitch_deg"]["new"]["rmse"] < metrics["body_pitch_deg"]["old"]["rmse"]
        other = all(metrics[key]["new"]["rmse"] <= 1.1*metrics[key]["old"]["rmse"] for key in ("speed_mps", "alpha_deg", "normal_specific_load_g", "body_yaw_deg", "body_yaw_rate_deg_s"))
        axes[cid] = {"pitch_RMSE_improves": pitch, "speed_alpha_load_yaw_yawrate_each_no_more_than10percent_worse": other, "passes": pitch and other}
    return {"zero_prediction_roundoff_preserved": zero_ok, "offaxis_cases": axes, "passed": zero_ok and all(case["passes"] for case in axes.values())}


def run_native(coefficients):
    cases = {c["id"]: c for c in finite.harness.load_cases(finite.comparison.SNAPSHOT)}
    cache = np.load(geometry.OUT / "native_trajectories.npz"); arrays = {}; reports = {}; runs = {}
    for cid in ("native_H110", "native_H27.5", "native_H55native"):
        case = cases[cid.removeprefix("native_")]; cfg = copy.deepcopy(case["profile"]["_model_config"])
        initial = dataclasses.replace(case["_creation_state"]); qx, qy, qz, qw = case["creation"]["u16_1"]
        initial.orientation_quaternion = (qw, qx, qy, qz); target_at = case["_target_model"].state_at
        model = make_model(cfg, coefficients); result = model.run(dataclasses.replace(initial), target_at, end_time_s=case["scenario"]["max_simulation_time_s"])
        path = OUT / "full_runs" / (cid + ".json"); slew.write_json(path, result, True)
        p = finite.harness.sample_series(result["samples"], case); baseline = drag.selected_arrays(cache, cid + "__candidate__")
        reports[cid] = {"harness_id": case["id"], "source_file": case["source_file"], "scenario": case["scenario"], "creation": case["creation"],
            "clock_diagnostics": case["clock_diagnostics"], "summary": result["summary"], "full_result_path": str(path),
            "near_axis_acceptance": fields.native_pair(case, baseline, p), "allocation_windows": allocation_windows(tick_arrays(result))}
        for label, prediction in (("reference", case["observed"]), ("baseline_geometry", baseline), ("candidate", p)): fields.save_arrays(arrays, cid + "__" + label, prediction)
        arrays.update({cid + "__controller__" + key: value for key, value in tick_arrays(result).items()})
        runs[cid] = (model, result, cfg, initial, target_at)
        print(json.dumps({"native_case": cid, "passes": reports[cid]["near_axis_acceptance"]["full_gate_passed"]}), flush=True)
    return reports, arrays, runs


def main():
    before = snapshot(); OUT.mkdir(parents=True, exist_ok=True)
    inputs = dec.audit.inverse.source_inputs()
    coefficients = drag.DragCoefficients(**json.loads((drag.OUTPUT / "fit.json").read_text())["coefficients"])
    # Independently verify the low-angle aligned plane limit before a free run.
    axes = BodyAxes((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    plane_c, plane_j = force_allocation(axes, axes, (.1, 0.), 1.)
    if np.max(abs(plane_c-np.eye(2))) > 1e-12: raise AssertionError("Positive low-angle planar C is not identity")
    reports, arrays, runs, execution = run_jari(inputs, coefficients)
    gate = next_native_gate(reports); native = {}
    if gate["passed"]:
        native, native_arrays, native_runs = run_native(coefficients); arrays.update(native_arrays); runs.update(native_runs)
    # One final focused pass; no baseline integration or additional candidate.
    checks = {cid: verify_result(model, result, config, initial, target_at, coefficients)
        for cid, (model, result, config, initial, target_at) in runs.items()}
    verification = {"cases": checks, "finite_arrays": all(np.isfinite(value).all() for value in arrays.values()),
        "protected_size_mtime_unchanged": before == snapshot(), "positive_low_angle_planar_C_identity_max_abs": float(np.max(abs(plane_c-np.eye(2)))),
        "original_control_clock_hz": 48, "original_plant_substeps": 4, "new_native_integrations": len(native), "baseline_integrations": 0,
        "reference_in_runtime": False, "fits_scans_new_coefficients": 0, "direction_only_no_inverse_sigma_gain": True,
        "other_authorized_worker_exclusions": ["scripts/check_r771_nonlinear_moment.py", "outputs/jari_offaxis/nonlinear_moment/"]}
    if not verification["finite_arrays"] or not verification["protected_size_mtime_unchanged"]: raise AssertionError(verification)
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    slew.write_json(OUT / "results.json", {"task": "Game-only one partial-force polar direction allocation, frozen geometry/finite lift/angular law",
        "candidate": "J(current local body pitch/yaw -> windnormal lift G); J=U Sigma V.T, C=V U.T; P=Kp C e, Iincrement=Ki C e/48, D=Kd C original raw-filtered e_dot. Previous e/e_dot remain raw wind coordinates. I and K transport stay body old-held.",
        "numerical_rule": "Central h=cuberoot(machine epsilon)*max(1,|local|), final h/2. Reject sigma_min <= max(128eps sigma_max,4norm(J_h/2-J_h)); reject stencils beyond local forward branch. No singular-value inversion, blending, previous-C hold, smoothing or fallback.",
        "predeclared_native_gate": "Both30/60 body-pitch RMSE improves over .5-3.5s; speed/alpha/load/bodyyaw/yawrate each at most10percent worse there; 0deg all reported full-window prediction deltas <=1e-10. No combined score.",
        "execution": execution, "JARI_cases": reports, "native_gate": gate, "native_cases": native, "verification": verification,
        "limitations": ["The Jacobian is a current partial lift map at fixed body/wind/q/m, not a full dynamic inverse from held input to total force. Thrust direction, base rotation and angular lag remain in the frozen plant.",
            "Polar reflection can switch discontinuously on crossing a force peak; no extra regularizer is introduced. Truly unresolved rank loss is rejected.",
            "Integral memory remains in body command coordinates; it is not rotated by subsequent C changes. D uses C times the current original filtered raw-error derivative, excluding dC/dt.",
            "Raw field equality does not establish native units or native allocation. This is one D-grade causal hypothesis; JARI is only interim reference.",
            "No envelope request projection, target1 relaxation, new gain, fit, new station, source feedback or trajectory injection is used."]})
    print(json.dumps({"native_gate": gate, "execution": execution, "verification_passed": True}), flush=True)


if __name__ == "__main__":
    main()
