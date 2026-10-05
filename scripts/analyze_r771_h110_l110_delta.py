"""Overlay and subtract the native H110/L110 responses and saved forward runs.

No simulation, optimization, clock shift, or change to the frozen model is made.
The subtraction is a paired response diagnostic, not an identified loft input.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_r771_frozen_loft as loft

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/r771_replay_batch_20260928"
OUT = BASE / "h110_l110_delta"
NAMES = ("H110", "L110")
KEYS = ("omega", "alpha", "theta", "gamma", "x", "y", "speed2")
WINDOWS = {"early_0_to_2p45": (0., 2.45),
           "late_2p55_to_3p5": (2.55, 3.5), "whole_0_to_3p5": (0., 3.5)}
SCALES = {key: (180/math.pi if key in ("alpha", "theta", "gamma") else 1.) for key in KEYS}
UNITS = {"omega": "rad/s", "alpha": "deg", "theta": "deg", "gamma": "deg",
         "x": "m", "y": "m", "speed2": "m/s"}


def event(t, z, lo, hi, kind):
    ids = np.flatnonzero((t >= lo-1e-10) & (t <= hi+1e-10))
    i = ids[np.argmax(z[ids]) if kind == "max" else np.argmin(z[ids])]
    return {"time_s": float(t[i]), "value": float(z[i]), "grid_index": int(i)}


def crossings(t, z, lo=.8, direction="positive_to_negative"):
    if direction == "positive_to_negative":
        hits = np.flatnonzero((t[:-1] >= lo) & (z[:-1] > 0) & (z[1:] <= 0))
    else:
        hits = np.flatnonzero((t[:-1] >= lo) & (z[:-1] < 0) & (z[1:] >= 0))
    return [{"time_s": float(t[i] - z[i]*(t[i+1]-t[i])/(z[i+1]-z[i])),
             "bracket_s": [float(t[i]), float(t[i+1])],
             "direction": direction} for i in hits]


def window_stats(t, observed, predicted):
    result = {}
    for label, (lo, hi) in WINDOWS.items():
        mask = (t >= lo-1e-10) & (t <= hi+1e-10)
        rec = {"grid_sample_count": int(mask.sum()), "signals": {}}
        for key in KEYS:
            d, p = observed[key][mask]*SCALES[key], predicted[key][mask]*SCALES[key]
            r = d-p
            rec["signals"][key] = {
                "display_unit": UNITS[key], "missing_response_rmse": float(np.sqrt(np.mean(r*r))),
                "missing_response_mean": float(np.mean(r)),
                "missing_response_abs_quantiles_50_90_100": np.quantile(np.abs(r), [.5, .9, 1.]).tolist(),
                "observed_delta_min_max": [float(d.min()), float(d.max())],
                "model_delta_min_max": [float(p.min()), float(p.max())]}
        result[label] = rec
    return result


def plot_response(shots, forward, grid, paired, deltas):
    rows = (("omega", "Pitch rate [rad/s]"), ("alpha", "Incidence [deg]"),
            ("theta", "Pitch [deg]"), ("gamma", "Flight-path angle [deg]"),
            ("y", "Vertical displacement [m]"), ("speed2", "Speed [m/s]; prescribed in model"))
    colors = {"H110": "#111111", "L110": "#2371ad"}
    mcolors = {"H110": "#27854f", "L110": "#db8a20"}
    fig, axs = plt.subplots(6, 2, figsize=(15, 17), sharex=True)
    for row, (key, label) in enumerate(rows):
        scale = SCALES[key]
        for name in NAMES:
            o, s, f = shots[name]["observed"], shots[name], forward[name]
            z = o[key] - (s["initial"][4] if key == "y" else 0.)
            axs[row, 0].plot(o["t"], z*scale, ".-", color=colors[name], lw=.85, ms=3.2,
                             label=name+" decoded" if row == 0 else None)
            zf = f[key] - (s["initial"][4] if key == "y" else 0.)
            axs[row, 0].plot(f["t"], zf*scale, color=mcolors[name], lw=1.6,
                             label=name+" saved model" if row == 0 else None)
        axs[row, 1].plot(grid, deltas["observed"][key]*scale, color="#111111", lw=1.4,
                         label="Decoded H−L" if row == 0 else None)
        axs[row, 1].plot(grid, deltas["model"][key]*scale, color="#27854f", lw=1.5,
                         label="Model H−L" if row == 0 else None)
        axs[row, 1].plot(grid, deltas["missing"][key]*scale, color="#9853b5", lw=1.3,
                         label="Missing: decoded Δ − model Δ" if row == 0 else None)
        axs[row, 0].set_ylabel(label)
        axs[row, 1].set_ylabel("Difference ["+UNITS[key]+"]")
        for ax in axs[row]:
            ax.set_xlim(0., 3.5)
            ax.axvline(14/48, color=".7", lw=.7, ls=":")
            ax.axhline(0, color=".7", lw=.6)
            ax.grid(alpha=.2)
    axs[0, 0].set_title("Same 110: with loft / without loft")
    axs[0, 1].set_title("Paired response and the part missing from current model")
    for ax in axs[0]: ax.legend(loc="upper right", fontsize=9)
    for ax in axs[-1]: ax.set_xlabel("Native b5 flight age [s]")
    fig.suptitle("H110 − L110: replay responses versus saved empirical loft model\n"
                 "Subtraction uses a 48 Hz common-age grid with linear interpolation; no clock shift. "
                 "Initial/target states differ.", fontsize=14)
    fig.text(.5, .015, "Vertical positions are relative to each decoded creation height. "
             "Speed is a replay input, so its model difference is not an energy prediction. "
             "Dotted line: frozen controller activation.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .955))
    fig.savefig(OUT / "paired_response.png", dpi=170)
    plt.close(fig)


def plot_path(shots, forward, grid, geometry):
    fig, axs0 = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs0.ravel()
    pn_ax = axs[3].twinx()
    for name, color in (("H110", "#111111"), ("L110", "#2371ad")):
        o, init, f = shots[name]["observed"], shots[name]["initial"], forward[name]
        mask = o["t"] <= 3.5+1e-10
        axs[0].plot(o["x"][mask]-init[3], o["y"][mask]-init[4], ".-", color=color,
                    lw=1., ms=3., label=name+" decoded")
        mc = "#27854f" if name == "H110" else "#db8a20"
        axs[0].plot(f["x"]-init[3], f["y"]-init[4], color=mc, lw=1.5, label=name+" saved model")
        r = shots[name]["replay"]
        tt = np.arange(169)/48
        target = np.array([loft.model.target_at(r, t)[0]-init[3:5] for t in tt])
        axs[1].plot(target[:, 0], target[:, 1], color=color, label=name+" actual target")
        axs[1].plot(target[0, 0], target[0, 1], "o", color=color)
        axs[2].plot(grid, geometry[name]["los_elevation_deg"], color=color, label=name+" LOS angle")
        axs[3].plot(grid, geometry[name]["los_rate_deg_s"], color=color, label=name+" LOS rate")
        pn_ax.plot(grid, geometry[name]["pn_g"], "--", color=color, label=name+" PN6 component")
    axs[0].set(title="Missile path; each creation point is (0, 0)",
               xlabel="Forward displacement [m]", ylabel="Vertical displacement [m]")
    axs[1].set(title="Target path relative to each missile creation point",
               xlabel="Target relative x [m]", ylabel="Target relative y [m]")
    axs[2].set(title="True-target LOS from decoded missile states", xlabel="Flight age [s]", ylabel="LOS elevation [deg]")
    axs[3].set(title="Calculated true-target LOS rate and normal PN6 component", xlabel="Flight age [s]", ylabel="LOS rate [deg/s]")
    pn_ax.set_ylabel("Normal PN component [g]; excludes gravity and loft")
    for ax in axs[2:]: ax.set_xlim(0, 3.5); ax.axhline(0, color=".7", lw=.6)
    for ax in axs: ax.grid(alpha=.2); ax.legend(fontsize=9, loc="upper left")
    pn_ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("The same 110 weapon setting is paired, but target geometry is not identical", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_overlay.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _ = loft.old.batch.load_all()
    shots = {"H110": loft.make_loft_shot(raw["H110"]), "L110": loft.old.make_shot(raw["L110"], "b5")}
    cfg, coeff, _ = loft.read_snapshot()
    current_meta = json.loads((BASE/"empirical_loft_function/results.json").read_text())
    assert np.array_equal(coeff, current_meta["frozen_coefficients"])
    current = np.load(BASE/"empirical_loft_function/trajectories.npz")
    accepted = np.load(BASE/"no_loft_control/simple_joint_calibration/trajectories.npz")
    old_archive = np.load(BASE/"frozen_loft/trajectories.npz")
    arrays, forward = {}, {}
    baseline_no_loft_max = 0.
    for name, s in shots.items():
        f = {key: current[name+"__empirical_loft__"+key].copy() for key in ("t", *KEYS[:-1])}
        f["speed2"] = np.interp(f["t"], s["replay"]["t"], s["replay"]["speed2"])
        forward[name] = f
        for key, value in s["observed"].items():
            assert np.array_equal(value, current[name+"__observed__"+key])
            arrays[name+"__native_decoded__"+key] = value.copy()
        for key, value in f.items(): arrays[name+"__saved_forward__"+key] = value
        arrays[name+"__creation__theta_omega_gamma_x_y"] = s["initial"].copy()
        arrays[name+"__raw_missile_rows"] = loft.old.batch.rows_array(raw[name]["case"], "missile", raw[name]["case"]["missile_columns"])
        arrays[name+"__raw_target_rows"] = loft.old.batch.rows_array(raw[name]["case"], "target", raw[name]["case"]["target_columns"])
        for key in ("target_t", "target_xy", "target_vxy"):
            arrays[name+"__external_input__"+key] = np.asarray(s["replay"][key]).copy()
    for key in ("t", "theta", "omega", "gamma", "x", "y", "alpha", "u", "I",
                "feedback_g", "a_req_g", "normal_body_n", "normal_wind_n", "moment_nm",
                "reference_couple_nm", "alpha_local", "qdot", "gamma_dot"):
        gap = np.max(np.abs(current["L110__empirical_loft__"+key]-accepted["L110__affine__"+key]))
        baseline_no_loft_max = max(baseline_no_loft_max, float(gap))
    assert baseline_no_loft_max == 0.
    common_lo = max(s["observed"]["t"][0] for s in shots.values())
    common_hi = min(3.5, *(s["observed"]["t"][-1] for s in shots.values()))
    grid = np.arange(math.ceil(common_lo*48-1e-10), math.floor(common_hi*48+1e-10)+1)/48
    assert grid[0] >= common_lo and grid[-1] <= common_hi+1e-12
    arrays["common_age_grid_s"] = grid
    paired = {"observed": {}, "model": {}}
    for kind in paired:
        for name, s in shots.items():
            src = s["observed"] if kind == "observed" else forward[name]
            paired[kind][name] = {key: np.interp(grid, src["t"], src[key]) for key in KEYS}
            for key, value in paired[kind][name].items():
                arrays[name+"__common_grid__"+kind+"__"+key] = value
    deltas = {kind: {key: paired[kind]["H110"][key]-paired[kind]["L110"][key] for key in KEYS}
              for kind in paired}
    for kind in paired:
        for key, index in (("x", 3), ("y", 4)):
            arrays["delta_"+kind+"__"+key+"_absolute"] = deltas[kind][key].copy()
            deltas[kind][key] -= shots["H110"]["initial"][index]-shots["L110"]["initial"][index]
    deltas["missing"] = {key: deltas["observed"][key]-deltas["model"][key] for key in KEYS}
    for kind, signals in deltas.items():
        for key, value in signals.items(): arrays["delta_"+kind+"__"+key] = value
    old_delta = {}
    for key in KEYS[:-1]:
        old_delta[key] = np.interp(grid, old_archive["H110__pn_plus_loft__t"], old_archive["H110__pn_plus_loft__"+key])-np.interp(grid, old_archive["L110__no_loft__t"], old_archive["L110__no_loft__"+key])
        if key in ("x", "y"):
            index = 3 if key == "x" else 4
            old_delta[key] -= shots["H110"]["initial"][index]-shots["L110"]["initial"][index]
        arrays["delta_old_degrees_based_model__"+key] = old_delta[key]
    old_delta["speed2"] = deltas["model"]["speed2"].copy()
    initial, target0, sampling = {}, {}, {}
    for name, s in shots.items():
        c = raw[name]["case"]["creation"]
        initial[name] = {"state_theta_omega_gamma_x_y": s["initial"].tolist(),
                         "pitch_deg": float(np.rad2deg(s["initial"][0])),
                         "flight_path_deg": float(np.rad2deg(s["initial"][2])),
                         "alpha_deg": float(np.rad2deg(s["initial"][0]-s["initial"][2])),
                         "speed_mps": s["initial_speed_mps"], "position_xyz_m": c["starting_pos"],
                         "velocity_xyz_mps": c["u12_2"], "quaternion_xyzw": c["u16_1"]}
        xy, vel = loft.model.target_at(s["replay"], 0.)
        rel = xy-s["initial"][3:5]
        target0[name] = {"xy_m": xy.tolist(), "vxy_mps": vel.tolist(),
                         "relative_xy_m": rel.tolist(), "range_m": float(np.linalg.norm(rel)),
                         "los_elevation_deg": float(np.rad2deg(math.atan2(rel[1], rel[0]))),
                         "mapped_target_age_support_s": [float(s["replay"]["target_t"][0]), float(s["replay"]["target_t"][-1])]}
        t = s["observed"]["t"]
        sampling[name] = {"raw_observation_count": len(t), "age_bounds_s": [float(t[0]), float(t[-1])],
                          "unique_dt_s": np.unique(np.round(np.diff(t), 9)).tolist(),
                          "missile_columns": raw[name]["case"]["missile_columns"],
                          "target_columns": raw[name]["case"]["target_columns"],
                          "creation_fields_are_separate_from_observations": True}
    target_grid = {}
    for name, s in shots.items():
        target_grid[name] = np.asarray([(*loft.model.target_at(s["replay"], t)[0],
                                        *loft.model.target_at(s["replay"], t)[1]) for t in grid])
        arrays[name+"__target_common_grid__xy_vxy"] = target_grid[name]
    target_difference = target_grid["H110"]-target_grid["L110"]
    # Evaluate the existing true-target PN6 kinematics on decoded missile states.
    # This calculation does not reconstruct the game's guidance input.
    geometry, native_geometry = {}, {}
    geo_keys = ("los_elevation_deg", "los_rate_deg_s", "pn_g", "closing_mps", "range_m", "hit_gain")
    for name, s in shots.items():
        pn_only = dict(s["p"], loft_enabled=False)
        geometry[name], native_geometry[name] = {}, {}
        for tag, tt, states, destination in (("common_grid", grid, paired["observed"][name], geometry[name]),
                ("native_rows", s["observed"]["t"], s["observed"], native_geometry[name])):
            values = [loft.model.request(float(t), np.array([states[key][i] for key in ("theta", "omega", "gamma", "x", "y")]), s["replay"], pn_only)
                      for i, t in enumerate(tt)]
            for key in geo_keys:
                destination[key] = np.asarray([v[key] for v in values])
                arrays[name+"__calculated_true_target_"+tag+"__"+key] = destination[key]
    geometry_summary = {}
    for name in NAMES:
        g = geometry[name]
        geometry_summary[name] = {
            "method": "Existing PN6 function evaluated on decoded/interpolated missile positions, measured speed and direction, and mapped packet-derived target position/velocity. pn_g includes existing hit_gain, excludes loft/gravity; not a game command.",
            "samples": {str(t): {key: float(np.interp(t, grid, g[key])) for key in geo_keys} for t in (.6, 1., 1.25, 1.7)},
            "pn_positive_to_negative_crossings": crossings(grid, g["pn_g"], 0.),
            "los_rate_positive_to_negative_crossings": crossings(grid, g["los_rate_deg_s"], 0.),
            "heuristic_abs_los_rate_crosses_0p5_deg_s": {direction: crossings(grid, np.abs(g["los_rate_deg_s"])-.5, 0., direction) for direction in ("positive_to_negative", "negative_to_positive")},
            "heuristic_los_crosses_minus_3p5_deg": {direction: crossings(grid, g["los_elevation_deg"]+3.5, 0., direction) for direction in ("positive_to_negative", "negative_to_positive")},
            "heuristic_thresholds_are_unverified_field_interpretations": True}
    weapon_diff = {key: {"H110": raw["H110"]["p"][key], "L110": raw["L110"]["p"][key]}
                   for key in raw["H110"]["p"] if raw["H110"]["p"][key] != raw["L110"]["p"][key]}
    events = {}
    for label, key, lo, hi, kind in (("delta_omega_peak1", "omega", .35, .60, "max"),
            ("delta_omega_valley1", "omega", .60, .84, "min"),
            ("delta_omega_peak2", "omega", .85, 1.18, "max"),
            ("delta_alpha_peak1", "alpha", .38, .76, "max"),
            ("delta_alpha_negative_extreme", "alpha", 1., 2.45, "min")):
        o = event(grid, deltas["observed"][key]*SCALES[key], lo, hi, kind)
        p = event(grid, deltas["model"][key]*SCALES[key], lo, hi, kind)
        events[label] = {"signal": key, "unit": UNITS[key], "window_s": [lo, hi],
                         "observed_delta": o, "model_delta": p,
                         "model_minus_observed_event_time_s": p["time_s"]-o["time_s"],
                         "missing_at_observed_event": float(deltas["missing"][key][o["grid_index"]]*SCALES[key])}
    crossing = {key: {kind: {direction: crossings(grid, deltas[kind][key], .8, direction)
                            for direction in ("positive_to_negative", "negative_to_positive")}
                     for kind in ("observed", "model")} for key in ("omega", "alpha", "theta", "gamma")}
    closure = {}
    for name, s in shots.items():
        o = s["observed"]
        residual = np.diff(o["theta"])-.5*(o["omega"][:-1]+o["omega"][1:])*np.diff(o["t"])
        i = int(np.argmax(np.abs(residual)))
        closure[name] = {"largest_native_interval_theta_minus_trapezoid_omega_deg": float(np.rad2deg(residual[i])),
                         "interval_s": [float(o["t"][i]), float(o["t"][i+1])],
                         "interpretation": "Decoded angle/rate closure diagnostic; no correction applied."}
    result = {
        "status": "paired_response_diagnostic_completed_no_fit_or_production_change",
        "source_files": [str(BASE/"source_payload.json"), str(BASE/"batch_compact.json"),
                         str(BASE/"empirical_loft_function/trajectories.npz"), str(BASE/"frozen_loft/trajectories.npz")],
        "frozen_snapshot": str(loft.SNAPSHOT), "frozen_coefficients": coeff.tolist(),
        "current_model_family": current_meta["selected_family"],
        "comparability": {"weapon_parameter_differences": weapon_diff,
            "creation": initial, "creation_H_minus_L_state_theta_omega_gamma_x_y": (shots["H110"]["initial"]-shots["L110"]["initial"]).tolist(),
            "target_at_creation": target0,
            "target_common_grid_H_minus_L_xy_vxy_min": target_difference.min(axis=0).tolist(),
            "target_common_grid_H_minus_L_xy_vxy_max": target_difference.max(axis=0).tolist(),
            "controlled_loft_only_pair": False,
            "reason": "Weapon parameters differ only by loft_enabled, but creation states and target geometry differ; target relative height differs by about150m."},
        "sampling": sampling,
        "common_grid": {"frequency_hz": 48, "sample_count": len(grid), "age_bounds_s": [float(grid[0]), float(grid[-1])],
            "method": "Linear interpolation of each native curve on common48Hz flight-age ticks within both observed supports; no extrapolation and no fitted time offset.",
            "native_rows_retained_in_npz": True, "events_are_on_interpolated_pair_grid": True},
        "delta_definition": "H110 minus L110; x/y differences use displacement from each decoded creation position. Absolute position differences are also saved.",
        "missing_response_definition": "decoded paired response minus saved-model paired response; not a decoded control request or a linear transfer function",
        "speed_definition": "Decoded planar speed; model speed is the same prescribed external replay input, not a forward energy state.",
        "windows": window_stats(grid, deltas["observed"], deltas["model"]),
        "old_degrees_based_model_windows": window_stats(grid, deltas["observed"], old_delta),
        "events": events, "crossings_after_0p8": crossing, "native_angle_rate_closure": closure,
        "calculated_true_target_geometry": geometry_summary,
        "limitations": ["Initial state/target differences confound a pure loft-only subtraction; model subtraction accounts for their modeled effect but not unknown game effects.",
            "Native seeker guidance/PID/fin inputs are unobserved. A virtual target and an equivalent acceleration bias can produce the same observed motion.",
            "Early strong positive response followed by negative relative incidence supports a time/state-varying loft-to-intercept strategy; it does not uniquely identify an engine algorithm.",
            "No aerodynamic/PID retuning, strategy implementation, inverse input fit, state feeding, smoothing, clock fitting, or new simulation was performed."],
        "verification": {"H110_rows": len(shots["H110"]["observed"]["t"]),
            "L110_rows": len(shots["L110"]["observed"]["t"]), "no_loft_saved_forward_minus_accepted_max_abs": baseline_no_loft_max,
            "current_saved_frozen_coefficients_match": True}}
    for kind in deltas:
        np.testing.assert_allclose(deltas[kind]["alpha"], deltas[kind]["theta"]-deltas[kind]["gamma"], atol=2e-16)
    assert (len(shots["H110"]["observed"]["t"]), len(shots["L110"]["observed"]["t"])) == (165, 164)
    assert all(np.all(np.isfinite(value)) for value in arrays.values())
    result["verification"].update({"all_arrays_finite": True, "delta_alpha_equals_delta_theta_minus_delta_gamma": True,
        "common_grid_within_both_observed_supports": True, "all_native_observations_equal_saved_archive": True})
    plot_response(shots, forward, grid, paired, deltas)
    plot_path(shots, forward, grid, geometry)
    np.savez_compressed(OUT/"paired_arrays.npz", **arrays)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(OUT), "events": events,
          "alpha_positive_to_negative": crossing["alpha"], "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
