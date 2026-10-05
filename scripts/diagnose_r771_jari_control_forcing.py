#!/usr/bin/env python3
"""JARI 75-degree control-forced plant diagnostic for a commercial game unit.

All integrated states evolve from creation. Only the exported control record
is supplied during integration. This is neither launch-only prediction nor a
reconstruction of unexported JARI actuator or guidance internals.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aim120_model.effective_surface_core import EffectiveSurfaceGame, normalized, read_raw
import fit_r771_jari_only as base
import prepare_r771_jari_only as ds

CASE = "angle_p75"
CONFIG = ROOT / "research/legacy/config/r771_jari_only_corrected.json"
OUT = ROOT / "outputs/jari_only/control_forcing"
EARLY_END = 3.5
CHECKPOINTS = (1.5, 2., 2.25, 2.5, 3., 3.5)
PHASES = {"lag_one_tick": -1, "zero_shift": 0, "advance_one_tick": 1}


def protected() -> dict[str, tuple[int, int]]:
    folders = ("src", "scripts", "config", "configs", "data",
               "outputs/jari_offaxis", "outputs/jari_only/dataset",
               "outputs/jari_only/calibration", "outputs/jari_only/correction")
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in folders for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_state(creation: dict[str, Any]) -> tuple[float, ...]:
    return (tuple(creation["position_xyz_m"]) + tuple(creation["velocity_xyz_mps"])
            + normalized(tuple(creation["quaternion_xyzw_raw"]))
            + tuple(-x for x in creation["omega_export_raw"]))


def forced_run(raw: dict[str, Any], config: dict[str, Any], creation: dict[str, Any],
               source_t: np.ndarray, source_controls: np.ndarray, ticks_shift: int
               ) -> dict[str, np.ndarray]:
    # Source state and source force arrays are deliberately absent from this
    # integration interface. Future controls are offline diagnostic inputs.
    model = EffectiveSurfaceGame(raw, config, creation)
    state = initial_state(creation)
    held = tuple(creation["controls_export_raw"])
    rows = []
    count = math.ceil(EARLY_END / model.dt)
    for tick in range(count + 1):
        t = tick * model.dt
        _, diagnostic = model.plant(t, state, held)
        query = t + ticks_shift * model.dt
        # Before creation, hold the actual creation control. No extrapolation
        # of other fields, padding of force histories or state reset is used.
        if query < source_t[0]:
            command = tuple(creation["controls_export_raw"])
        else:
            command = tuple(base.interpolate(source_controls, source_t, query))
        rows.append({"t": t, "position_xyz_m": state[:3], "velocity_xyz_mps": state[3:6],
                     "quaternion_xyzw": state[6:10], "body_omega_xyz_rad_s": state[10:],
                     "controls_raw": held, "next_applied_controls_raw": command,
                     "control_source_query_time_s": query, **diagnostic})
        if tick == count:
            break
        held = command
        h = model.dt / model.integration_substeps
        for substep in range(model.integration_substeps):
            state = model.step(t + substep * h, state, held, h)
    return base.arrays_from_rows(rows)


def delta_for(source: dict[str, np.ndarray], prediction: dict[str, np.ndarray],
              channel: str, mask: np.ndarray) -> np.ndarray:
    predicted = base.interpolate(prediction[channel], prediction["t"], source["t"][mask])
    delta = predicted - source[base.METRIC_SOURCE.get(channel, channel)][mask]
    return (delta + 180.) % 360. - 180. if channel == "body_yaw_deg" else delta


def window(source: dict[str, np.ndarray], prediction: dict[str, np.ndarray],
           mask: np.ndarray) -> dict[str, Any]:
    result = {}
    for channel in base.CHANNELS:
        valid = mask.copy()
        if channel == "normal_specific_load_g":
            valid &= source["derivative_5point_valid_mask"]
        error = delta_for(source, prediction, channel, valid)
        result[channel] = {"rmse": base.rmse(error), "mean_signed_error": float(error.mean()),
                           "max_abs_error": float(np.abs(error).max()), "rows": int(valid.sum())}
    return result


def evaluate(source: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    early = source["t"] <= EARLY_END
    plateau = early & (source["t"] >= 1.5) & (source["t"] <= 2.)
    saturated = early & (source["controls_export_raw"][:, 0] >= 1. - 1e-7)
    snapshots = {}
    for t in CHECKPOINTS:
        right = int(np.searchsorted(source["t"], t))
        assert source["derivative_5point_valid_mask"][max(0, right-1):right+1].all()
        row = {}
        for channel in base.CHANNELS:
            observed = float(base.interpolate(source[base.METRIC_SOURCE.get(channel, channel)], source["t"], t))
            predicted = float(base.interpolate(prediction[channel], prediction["t"], t))
            difference = predicted - observed
            if channel == "body_yaw_deg":
                difference = (difference + 180.) % 360. - 180.
            row[channel] = {"source": observed, "prediction": predicted, "signed_error": difference}
        row["controls_export_raw"] = base.interpolate(source["controls_export_raw"], source["t"], t).tolist()
        row["last_held_controls_raw"] = base.interpolate(prediction["controls_raw"], prediction["t"], t).tolist()
        snapshots[str(t)] = row
    return {"early": window(source, prediction, early),
            "shared_saturation_1p5_to_2s": window(source, prediction, plateau),
            "source_horizontal_saturation_rows": window(source, prediction, saturated),
            "snapshots": snapshots}


def plot(source: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"current_launch_only": "#b8bbc1", "lag_one_tick": "#2378a9",
              "zero_shift": "#d66523", "advance_one_tick": "#338554"}
    names = {"current_launch_only": "Current launch-only", "lag_one_tick": "Forced delta = -dt",
             "zero_shift": "Forced delta = 0", "advance_one_tick": "Forced delta = +dt (completed tick)"}
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    labels = ("Speed [m/s]", "Body pitch [deg]", "Body yaw [deg]",
              "Geometric incidence [deg]", "Normal specific load [G]")
    for ax, channel, label in zip(axes.flat, base.CHANNELS, labels):
        mask = source["t"] <= EARLY_END
        if channel == "normal_specific_load_g":
            mask &= source["derivative_5point_valid_mask"]
        ax.plot(source["t"][mask], source[base.METRIC_SOURCE.get(channel, channel)][mask],
                color="#171717", lw=2., label="JARI")
        for name, prediction in predictions.items():
            valid = prediction["t"] <= EARLY_END + 1e-6
            ax.plot(prediction["t"][valid], prediction[channel][valid], color=colors[name],
                    lw=1.6, ls="--" if name == "current_launch_only" else "-", label=names[name])
        ax.axvspan(1.5, 2., color="#8b9399", alpha=.09)
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.set_xlim(0., EARLY_END)
    ax = axes.flat[-1]
    mask = source["t"] <= EARLY_END
    ax.plot(source["t"][mask], source["controls_export_raw"][mask, 0], color="#171717", lw=2., label="JARI horizontal")
    for name, prediction in predictions.items():
        valid = prediction["t"] <= EARLY_END + 1e-6
        ax.step(prediction["t"][valid], prediction["next_applied_controls_raw"][valid, 0], where="post", color=colors[name],
                lw=1.5, ls="--" if name == "current_launch_only" else "-", label=names[name])
    ax.set_title("Horizontal command applied on [t, t+dt)")
    ax.set_ylim(-.05, 1.05)
    ax.set_xlim(0., EARLY_END)
    ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("JARI 75 degree: control-forced frozen plant from identical creation state", y=.988, fontsize=14)
    fig.text(.5, .955, "u_apply(t) = linear interpolation of u_export(t + delta), held for one 48 Hz tick",
             ha="center", fontsize=11)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .938), ncol=5, fontsize=9)
    fig.tight_layout(rect=(0., 0., 1., .905))
    fig.savefig(OUT / "early_overlay.png", dpi=150)
    plt.close(fig)


def main() -> None:
    before = protected()
    metadata, source = ds.load_case(CASE)
    creation = ds.creation_input(CASE)
    raw = read_raw(base.RAW)
    config = json.loads(CONFIG.read_text())
    dependencies = [CONFIG, base.RAW, ROOT / "src/aim120_model/effective_surface_core.py",
                    ds.OUT / "manifest.json", ds.OUT / "observations.npz", ds.OUT / "creation_inputs.json",
                    ROOT / "outputs/jari_only/correction/free_predictions.npz",
                    ROOT / "config/r771_simple_joint_frozen.json"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in dependencies}
    with np.load(ROOT / "outputs/jari_only/correction/free_predictions.npz", allow_pickle=False) as archive:
        prefix = CASE + "__"
        cached = {key[len(prefix):]: archive[key].copy() for key in archive.files if key.startswith(prefix)}
    # Launch-only rows carry the command held in the preceding completed tick;
    # the next row's command is what was applied on [t,t+dt).
    cached["next_applied_controls_raw"] = np.vstack((cached["controls_raw"][1:], cached["controls_raw"][-1:]))
    predictions = {"current_launch_only": cached}
    for name, phase in PHASES.items():
        predictions[name] = forced_run(raw, config, creation, source["t"], source["controls_export_raw"], phase)
    results = {name: evaluate(source, prediction) for name, prediction in predictions.items()}
    checks = {}
    expected = np.asarray(initial_state(creation))
    for name in PHASES:
        p = predictions[name]
        actual = np.concatenate((p["position_xyz_m"][0], p["velocity_xyz_mps"][0],
                                 p["quaternion_xyzw"][0], p["body_omega_xyz_rad_s"][0]))
        numeric = [v for v in p.values() if np.issubdtype(v.dtype, np.number)]
        check = {"same_initial_state": bool(np.array_equal(actual, expected)),
                 "finite_numeric_arrays": all(bool(np.isfinite(v).all()) for v in numeric),
                 "quaternion_norm_max_error": float(np.abs(np.linalg.norm(p["quaternion_xyzw"], axis=1) - 1.).max()),
                 "max_abs_applied_control": float(np.abs(p["next_applied_controls_raw"]).max()),
                 "ticks": len(p["t"])-1, "source_control_support_covers_all_queries_after_creation":
                    bool(p["control_source_query_time_s"].max() <= source["t"][-1])}
        assert check["same_initial_state"] and check["finite_numeric_arrays"]
        assert check["quaternion_norm_max_error"] < 1e-12 and check["max_abs_applied_control"] <= 1. + 1e-12
        assert check["source_control_support_covers_all_queries_after_creation"]
        checks[name] = check
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", **{name+"__"+key: value
                        for name, p in predictions.items() for key, value in p.items()})
    plot(source, predictions)
    unchanged = before == protected()
    assert unchanged
    assert all(digest(ROOT / path) == saved for path, saved in hashes.items())
    result = {"case": CASE, "diagnostic": "control-forced full plant; no integrated source state reset",
              "source": metadata["source_path"], "free_launch_prediction": False,
              "coefficient_selection_or_changes": False, "network_requests": 0,
              "game_replay_observations_used": False,
              "protocol": {"dt_s": creation["scenario_exact"]["dt"], "integration_substeps": config["integration_substeps"],
                           "integration": "original midpoint step and per-substep quaternion normalization",
                           "duration_s": EARLY_END, "control_interpolation": "linear at tick start, then held over all four substeps",
                           "phases": {name: {"delta_ticks": value, "delta_s": value*creation["scenario_exact"]["dt"]}
                                      for name, value in PHASES.items()},
                           "phase_definition": "u_apply(t)=u_export(t+delta) on [t,t+dt); +dt corresponds to exported completed held tick",
                           "before_creation": "hold actual creation controls; negative query occurs only for lag at first tick",
                           "omega_convention": "negative source-exported creation omega, Hamilton body convention",
                           "metric_support": "original source rows t<=3.5, exact existing early convention",
                           "load_reference": "5point kinematic normal specific force with source validity mask",
                           "snapshots": "linear interpolation at nominal seconds; all load brackets use valid source stencil rows",
                           "unknowns": "intermediate 48Hz commands are absent from 24Hz exports; exported control is not a physical fin angle, scale/hold phase unverified"},
              "results": results, "verification": {"predictions": checks, "protected_files_unchanged": unchanged,
                    "protected_file_count": len(before), "direct_dependency_sha256_unchanged": hashes,
                    "fixed_effective_parameters": config["effective_parameters"]}}
    (OUT / "results.json").write_text(json.dumps(ds.jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"early_RMSE": {name: {k: v["rmse"] for k, v in r["early"].items()} for name, r in results.items()},
                      "plateau_mean_error": {name: {k: v["mean_signed_error"] for k, v in r["shared_saturation_1p5_to_2s"].items()}
                                             for name, r in results.items()},
                      "protected_files_unchanged": unchanged, "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
