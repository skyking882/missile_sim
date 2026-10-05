#!/usr/bin/env python3
"""Fixed-coefficient restoring-vector experiment on existing JARI game exports.

The three-phase control-forced diagnostic and launch-only forecasts are kept
separate. All historically seen site checks are nonblind; no coefficient,
controller, guidance, phase or other model selection is performed here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aim120_model.effective_surface_core import EffectiveSurfaceGame, read_raw
import fit_r771_jari_only as base
import prepare_r771_jari_only as ds
import diagnose_r771_jari_control_forcing as forcing

CONFIG = ROOT / "research/legacy/config/r771_jari_only_vector_restoring.json"
BASE_CONFIG = ROOT / "research/legacy/config/r771_jari_only_corrected.json"
OUT = ROOT / "outputs/jari_only/vector_restoring"
CASES = ds.FIT_CASES + ds.VALIDATION_CASES
CASE = "angle_p75"
LABELS = ("Speed [m/s]", "Body pitch [deg]", "Body yaw [deg]",
          "Geometric incidence [deg]", "Normal specific load [G]")


def receipt(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def load_predictions(path: Path, names: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        return {name: {k.split("__", 1)[1]: z[k].copy() for k in z.files if k.startswith(name + "__")}
                for name in names}


def launch(config: dict[str, Any], raw: dict[str, Any], case: str) -> dict[str, np.ndarray]:
    return base.arrays_from_rows(EffectiveSurfaceGame(raw, config, ds.creation_input(case)).run())


def geometry_checks(raw: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    creation = ds.creation_input("angle_p0")
    old = EffectiveSurfaceGame(raw, baseline, creation)
    new = EffectiveSurfaceGame(raw, candidate, creation)
    errors = {}
    # Production-plant comparisons, with exactly one transverse velocity and
    # no rotation: both mappings must produce the same pure-axis response.
    vectors = {"zero_incidence": (400., 0., 0.),
               "pitch_positive": (400., -150., 0.), "pitch_negative": (400., 150., 0.),
               "yaw_positive": (400., 0., -150.), "yaw_negative": (400., 0., 150.)}
    for name, velocity in vectors.items():
        state = (0., creation["position_xyz_m"][1], 0.) + velocity + (0., 0., 0., 1.) + (0., 0., 0.)
        first, first_diag = old.plant(1., state, (.2, -.1))
        second, second_diag = new.plant(1., state, (.2, -.1))
        error = float(np.max(np.abs(np.asarray(first) - second)))
        assert error < 1e-12 and first_diag == second_diag
        errors[name] = error
    return {"production_derivative_max_abs_difference": errors, "pure_axis_tolerance": 1e-12,
            "force_diagnostics_exactly_equal": True, "zero_incidence_finite": True}


def prediction_checks(prediction: dict[str, np.ndarray], creation: dict[str, Any]) -> dict[str, Any]:
    initial = np.concatenate((prediction["position_xyz_m"][0], prediction["velocity_xyz_mps"][0],
                              prediction["quaternion_xyzw"][0], prediction["body_omega_xyz_rad_s"][0]))
    numeric = [v for v in prediction.values() if np.issubdtype(v.dtype, np.number)]
    applied = prediction.get("next_applied_controls_raw", prediction["controls_raw"])
    check = {"same_initial_state": bool(np.array_equal(initial, forcing.initial_state(creation))),
             "finite_numeric_arrays": all(bool(np.isfinite(v).all()) for v in numeric),
             "quaternion_norm_max_error": float(np.abs(np.linalg.norm(prediction["quaternion_xyzw"], axis=1)-1.).max()),
             "max_abs_applied_control": float(np.abs(applied).max())}
    assert check["same_initial_state"] and check["finite_numeric_arrays"]
    assert check["quaternion_norm_max_error"] < 1e-12 and check["max_abs_applied_control"] <= 1.+1e-12
    return check


def comparisons(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for window in ("early", "full"):
        result[window] = {}
        for channel in base.CHANNELS:
            a, b = before["windows"][window][channel], after["windows"][window][channel]
            result[window][channel] = {"baseline_rmse": a["rmse"], "candidate_rmse": b["rmse"],
                                       "rmse_change": b["rmse"]-a["rmse"],
                                       "baseline_mean_signed_error": a["mean_signed_error"],
                                       "candidate_mean_signed_error": b["mean_signed_error"],
                                       "rmse_increased": bool(b["rmse"] > a["rmse"]+1e-10)}
    return result


def pitch_peaks(source: dict[str, np.ndarray], forecasts: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    result = {}
    for name, lo, hi in (("first", .333, .75), ("second", .875, 1.333), ("third", 1.4, 2.)):
        ids = np.flatnonzero((source["t"] >= lo) & (source["t"] <= hi))
        i = ids[int(np.argmax(source["body_pitch_deg"][ids]))]
        t, pitch = float(source["t"][i]), float(source["body_pitch_deg"][i])
        row = {"source_peak_time_s": t, "source_peak_pitch_deg": pitch, "forecasts": {}}
        for label, p in forecasts.items():
            at_source = float(base.interpolate(p["body_pitch_deg"], p["t"], t))
            predicted = base.interpolate(p["body_pitch_deg"], p["t"], source["t"][ids])
            j = int(np.argmax(predicted))
            row["forecasts"][label] = {"pitch_at_source_peak_time_deg": at_source,
                                       "same_time_signed_error_deg": at_source-pitch,
                                       "separate_window_peak_pitch_deg": float(predicted[j]),
                                       "separate_window_peak_time_s": float(source["t"][ids[j]]),
                                       "window_peak_is_boundary": j in (0, len(ids)-1)}
        result[name] = row
    return result


def plot_case(source: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"baseline_closed_loop": ("#90969d", "--", "Current closed loop"),
             "candidate_closed_loop": ("#cf4c35", "-", "Vector restoring closed loop"),
             "baseline_forced_plus_dt": ("#277aab", "--", "Current plant, JARI control +dt"),
             "candidate_forced_plus_dt": ("#28854a", "-", "Vector restoring, JARI control +dt")}
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for ax, channel, label in zip(axes.flat, base.CHANNELS, LABELS):
        valid = source["t"] <= 3.5
        if channel == "normal_specific_load_g":
            valid &= source["derivative_5point_valid_mask"]
        ax.plot(source["t"][valid], source[base.METRIC_SOURCE.get(channel, channel)][valid],
                color="#151515", lw=2., label="JARI")
        for name, p in predictions.items():
            mask = p["t"] <= 3.5+1e-6
            color, line, legend = style[name]
            ax.plot(p["t"][mask], p[channel][mask], color=color, ls=line, lw=1.6, label=legend)
        ax.axvspan(1.5, 2., color="#899299", alpha=.08)
        ax.set_title(label)
        ax.set_xlim(0., 3.5)
        ax.grid(alpha=.2)
    ax = axes.flat[-1]
    mask = source["t"] <= 3.5
    ax.plot(source["t"][mask], source["controls_export_raw"][mask, 1], color="#151515", lw=2., label="JARI")
    for name, p in predictions.items():
        if "forced" in name:
            continue
        valid = p["t"] <= 3.5+1e-6
        color, line, legend = style[name]
        # A launch row carries the command held during its completed tick.
        ax.plot(p["t"][valid], p["controls_raw"][valid, 1], color=color, ls=line, lw=1.6, label=legend)
    ax.set_title("Vertical control recorded at completed tick")
    ax.set_xlim(0., 3.5)
    ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("JARI 75 degree: only restoring-angle geometry changes; all coefficients fixed", y=.988, fontsize=14)
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .957), ncol=3, fontsize=10)
    fig.tight_layout(rect=(0., 0., 1., .905))
    fig.savefig(OUT/"angle75_overlay.png", dpi=150)
    plt.close(fig)


def plot_all(sources: dict[str, dict[str, np.ndarray]], baseline: dict[str, dict[str, np.ndarray]],
             candidate: dict[str, dict[str, np.ndarray]], early: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots(10, 5, figsize=(19, 25), squeeze=False)
    for row, case in enumerate(CASES):
        s = sources[case]
        end = 3.5 if early else float(s["t"][-1])
        for col, (channel, label) in enumerate(zip(base.CHANNELS, LABELS)):
            ax = grid[row, col]
            mask = s["t"] <= end
            if channel == "normal_specific_load_g":
                mask &= s["derivative_5point_valid_mask"]
            ax.plot(s["t"][mask], s[base.METRIC_SOURCE.get(channel, channel)][mask], color="#141414", lw=1.7, label="JARI")
            for p, color, name in ((baseline[case], "#9298a0", "Current closed loop"),
                                   (candidate[case], "#cc4b32", "Vector restoring closed loop")):
                valid = p["t"] <= end
                ax.plot(p["t"][valid], p[channel][valid], color=color, lw=1.2, label=name)
            ax.set_xlim(0., end)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(case.replace("angle_", "") + "\n" + ("fit" if case in ds.FIT_CASES else "nonblind site check"), fontsize=9)
            if row == 9:
                ax.set_xlabel("Time [s]")
    grid[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed-coefficient total-incidence restoring candidate: " +
                 ("0–3.5 s" if early else "original full source support"), fontsize=14)
    fig.tight_layout(rect=(0., 0., 1., .975))
    fig.savefig(OUT/("early_overlay.png" if early else "full_overlay.png"), dpi=145)
    plt.close(fig)


def main() -> None:
    baseline_cfg = json.loads(BASE_CONFIG.read_text())
    candidate_cfg = json.loads(CONFIG.read_text())
    expected = copy.deepcopy(baseline_cfg)
    expected["status"] = candidate_cfg["status"]
    expected["restoring_angle_convention"] = "total_incidence_vector"
    expected["runtime_source_boundary"]["required_files"][1] = str(CONFIG)
    assert candidate_cfg == expected
    assert "restoring_angle_convention" not in baseline_cfg
    assert candidate_cfg["effective_parameters"] == baseline_cfg["effective_parameters"]
    manifest = json.loads((ds.OUT/"manifest.json").read_text())
    dependencies = [BASE_CONFIG, base.CONFIG, CONFIG, base.RAW, ROOT/"config/r771_simple_joint_frozen.json",
                    ROOT/"src/aim120_model/effective_surface_core.py", Path(__file__),
                    ROOT/"scripts/diagnose_r771_jari_control_forcing.py",
                    ROOT/"scripts/fit_r771_jari_only.py", ROOT/"scripts/prepare_r771_jari_only.py",
                    ds.OUT/"manifest.json", ds.OUT/"observations.npz", ds.OUT/"creation_inputs.json",
                    ROOT/"outputs/jari_only/correction/free_predictions.npz",
                    ROOT/"outputs/jari_only/calibration/free_predictions.npz",
                    ROOT/"outputs/jari_only/control_forcing/predictions.npz"]
    dependencies += [ROOT/manifest["cases"][case]["source_path"] for case in CASES]
    before = receipt(dependencies)
    raw = read_raw(base.RAW)
    geometry = geometry_checks(raw, baseline_cfg, candidate_cfg)
    baseline = load_predictions(ROOT/"outputs/jari_only/correction/free_predictions.npz", CASES)
    old_forced = load_predictions(ROOT/"outputs/jari_only/control_forcing/predictions.npz", tuple(forcing.PHASES))
    source75 = ds.load_case(CASE)[1]
    creation75 = ds.creation_input(CASE)
    # This interface accepts source controls only, never source force/state rows.
    forced = {name: forcing.forced_run(raw, candidate_cfg, creation75, source75["t"],
                                     source75["controls_export_raw"], phase)
              for name, phase in forcing.PHASES.items()}
    forced_metrics = {name: {"baseline": forcing.evaluate(source75, old_forced[name]),
                             "candidate": forcing.evaluate(source75, p)} for name, p in forced.items()}
    print(json.dumps({"stage": "three_phase_control_forced_complete",
                      "early_RMSE": {name: {k: m["candidate"]["early"][k]["rmse"] for k in base.CHANNELS}
                                     for name, m in forced_metrics.items()}}, ensure_ascii=False), flush=True)
    sources = {case: ds.load_case(case)[1] for case in CASES}
    candidate, metrics, checks = {}, {}, {}
    for case in CASES:
        candidate[case] = launch(candidate_cfg, raw, case)
        checks[case] = prediction_checks(candidate[case], ds.creation_input(case))
        a, b = base.measure(sources[case], baseline[case]), base.measure(sources[case], candidate[case])
        assert b["uncovered_source_rows"] == 0
        metrics[case] = {"baseline": a, "candidate": b, "comparison": comparisons(a, b)}
        print(json.dumps({"stage": "launch_only_complete", "case": case,
                          "early_RMSE": {k: b["windows"]["early"][k]["rmse"] for k in base.CHANNELS}}, ensure_ascii=False), flush=True)
    # One focused final pass: old configurations must match their existing
    # caches exactly, and the forcing baseline must match its fixed protocol.
    legacy_cfg = json.loads(base.CONFIG.read_text())
    legacy_cache = load_predictions(base.OUT/"free_predictions.npz", ds.FIT_CASES)
    compatibility = {}
    for label, cfg, cache in (("legacy_calibration", legacy_cfg, legacy_cache),
                              ("current_corrected", baseline_cfg, baseline)):
        compatibility[label] = {}
        for case in ds.FIT_CASES:
            rerun = launch(cfg, raw, case)
            match = set(rerun) == set(cache[case]) and all(np.array_equal(rerun[k], cache[case][k]) for k in rerun)
            assert match, (label, case)
            compatibility[label][case] = match
    forcing_compatibility, forced_checks = {}, {}
    for name, phase in forcing.PHASES.items():
        rerun = forcing.forced_run(raw, baseline_cfg, creation75, source75["t"], source75["controls_export_raw"], phase)
        assert all(np.array_equal(rerun[k], old_forced[name][k]) for k in rerun)
        forcing_compatibility[name] = True
        forced_checks[name] = prediction_checks(forced[name], creation75)
        query = forced[name]["control_source_query_time_s"]
        assert query.max() <= source75["t"][-1]
        forced_checks[name]["source_control_support_covers_queries_after_creation"] = True
    failed = []
    regressions = []
    for case, m in metrics.items():
        for window in ("early", "full"):
            for channel in base.CHANNELS:
                stat = m["candidate"]["windows"][window][channel]
                if stat["rmse"] > stat["predeclared_rmse_tolerance"]:
                    failed.append({"case": case, "window": window, "channel": channel,
                                   "rmse": stat["rmse"], "tolerance": stat["predeclared_rmse_tolerance"]})
                row = m["comparison"][window][channel]
                if row["rmse_increased"]:
                    regressions.append({"case": case, "window": window, "channel": channel, **row})
    pitch_forecasts = {"baseline_closed_loop": baseline[CASE], "candidate_closed_loop": candidate[CASE],
                       "baseline_forced_plus_dt": old_forced["advance_one_tick"],
                       "candidate_forced_plus_dt": forced["advance_one_tick"]}
    peaks = pitch_peaks(source75, pitch_forecasts)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"free_predictions.npz", **{c+"__"+k: v for c, p in candidate.items() for k, v in p.items()})
    np.savez_compressed(OUT/"control_forced_predictions.npz", **{c+"__"+k: v for c, p in forced.items() for k, v in p.items()})
    plot_case(source75, pitch_forecasts)
    plot_all(sources, baseline, candidate, True)
    plot_all(sources, baseline, candidate, False)
    assert receipt(dependencies) == before
    result = {"status": "experimental_candidate_not_accepted", "config": str(CONFIG.relative_to(ROOT)),
              "change": "single-station local total-incidence vector components replace independent plane restoring angles only",
              "formula": "rho=hypot(local_v.y,local_v.z); alpha=atan2(rho,local_v.x); beta_p=-alpha*local_v.y/rho; beta_y=-alpha*local_v.z/rho; rho=0 gives both zero",
              "fixed_effective_parameters": candidate_cfg["effective_parameters"],
              "coefficient_or_controller_or_guidance_selection": False,
              "network_requests": 0, "game_replay_observations_used": False,
              "case_roles": {"fit": list(ds.FIT_CASES), "historically_seen_nonblind_site_checks": list(ds.VALIDATION_CASES)},
              "control_forced": {"case": CASE, "free_launch_prediction": False, "duration_s": 3.5,
                                 "protocol": {"only_source_input_during_integration": "exported controls",
                                              "interpolation": "existing linear interpolation at tick start, held for all four integration substeps",
                                              "phases_ticks": forcing.PHASES,
                                              "phase_definition": "u_apply(t)=u_export(t+delta); +dt corresponds to exported completed held tick",
                                              "same_creation_state": True, "state_reset": False},
                                 "results": forced_metrics},
              "launch_only": {"creation_only_runtime_inputs": True, "duration_s": candidate_cfg["forward_duration_s"],
                              "metrics": metrics},
              "angle75_pitch_peaks": peaks, "all_rmse_increases": regressions,
              "failed_existing_per_channel_tolerances": failed,
              "limits": ["Exported control is not an independently identified fin angle; intermediate 48Hz commands are absent from 24Hz records.",
                         "Same-state derivative improvement does not establish full trajectory agreement or internal JARI model recovery.",
                         "Existing site checks have been seen historically and are not blind tests.",
                         "The inherited PN/timeToHit candidate and effective angular coefficients remain unchanged and may retain separate errors."],
              "verification": {"geometry": geometry, "legacy_missing_field_configs_bitwise_compatible": compatibility,
                               "forcing_baseline_bitwise_compatible": forcing_compatibility,
                               "candidate_launch_arrays": checks, "candidate_forced_arrays": forced_checks,
                               "direct_dependencies_metadata_unchanged_during_execution": before,
                               "all_coefficients_and_other_runtime_config_fields_unchanged": True},
              "execution_counts": {"candidate_launch_only": 10, "candidate_control_forced": 3,
                                   "legacy_compatibility_launch_only": 6, "baseline_forcing_compatibility": 3}}
    (OUT/"results.json").write_text(json.dumps(ds.jsonable(result), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "failed_existing_tolerances": failed,
                      "angle75_pitch_peaks": peaks, "output": str(OUT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
