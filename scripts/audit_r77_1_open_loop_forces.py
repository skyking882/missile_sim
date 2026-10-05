"""Controller-independent R-77-1 aggregate force audit at measured states.

This diagnostic conditions the current frozen 2.57.1.107 translational force
law on the replay's measured speed and total angle of attack.  It does not use
target geometry, guidance, a controller, an inferred fin command, or a freely
integrated attitude state.  Aggregate wind-normal load is checked at each
native replay timestamp.  Axial force is checked by integrating the current
thrust/drag acceleration across each native measurement interval.

The resulting agreement is conditional on fixed 6300 m altitude, the current
nominal thrust/mass history, level-flight component alignment, and rounded
replay readouts.  It cannot separate body and fin lift or validate attitude
dynamics.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import area_basis, effective_cda0, effective_cda_alpha
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2 as G0
from check_r77_1_257_plant_constraints import COMMIT, VERSION
from check_r77_1_capture_strategy import digest
from run_r77_1_focus_moment import inputs


START_S = 0.2
END_S = 3.5
ALTITUDE_M = 6300.0
LIFT_SLOPE_PER_RAD = 14.0
EXPECTED_K = 0.0767
PRIMARY_QUADRATURE_SUBDIVISIONS = 64
REFINED_QUADRATURE_SUBDIVISIONS = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_paths() -> dict[str, Path]:
    return {
        "aggregate_force_model": ROOT / "src/aim120_model/alpha_servo.py",
        "plant_constructor": ROOT / "scripts/run_r77_1_focus_moment.py",
        "runtime_config": ROOT / "config/profile_h2_rebuild_defaults.json",
        "missile_profile": ROOT / "missiles/su_r_77_1.json",
        "raw_datamine": (
            ROOT
            / "data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
        ),
        "replay": ROOT / "data/replays/r77_1_level_20260824.tsv",
    }


def load_replay(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream, delimiter="\t"))
    rows: list[dict[str, float | str]] = []
    for source in raw:
        time_s = float(source["missile_flight_time_s"])
        if START_S - 1e-12 <= time_s <= END_S + 1e-12:
            rows.append(
                {
                    "sample_id": source["sample_id"],
                    "time_s": time_s,
                    "speed_mps": float(source["speed_kmh"]) / 3.6,
                    "alpha_rad": math.radians(float(source["angle_of_attack_deg"])),
                    "observed_load_g": float(source["overload_g"]),
                }
            )
    if len(rows) != 19 or rows[0]["time_s"] != START_S or rows[-1]["time_s"] != END_S:
        raise ValueError("Expected the 19 native replay rows from 0.2 through 3.5 s")
    times = [float(row["time_s"]) for row in rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Replay timestamps must be strictly increasing")
    return rows


def sinc(angle_rad: float) -> float:
    return math.sin(angle_rad) / angle_rad if angle_rad else 1.0


def force_state(config, parameters, atmosphere, propulsion, time_s, speed_mps, alpha_rad):
    prop = propulsion.sample(time_s)
    qbar = 0.5 * atmosphere.density_kg_m3 * speed_mps**2
    mach = speed_mps / atmosphere.speed_of_sound_mps
    normal_gain = qbar * parameters.lift_area_per_rad + prop.thrust_n * sinc(alpha_rad)
    vertical_alpha = prop.mass_kg * G0 / normal_gain
    valid = alpha_rad + 1e-14 >= vertical_alpha
    horizontal_alpha = math.sqrt(max(alpha_rad**2 - vertical_alpha**2, 0.0)) if valid else math.nan

    predicted_aero_normal_n = qbar * parameters.lift_area_per_rad * alpha_rad
    predicted_thrust_normal_n = prop.thrust_n * math.sin(alpha_rad)
    predicted_total_normal_n = predicted_aero_normal_n + predicted_thrust_normal_n
    predicted_total_load_g = predicted_total_normal_n / (prop.mass_kg * G0)
    decomposed_load_g = (
        math.hypot(normal_gain * horizontal_alpha / (prop.mass_kg * G0), 1.0)
        if valid
        else math.nan
    )

    drag_zero_n = qbar * effective_cda0(mach, config)
    drag_alpha_n = qbar * effective_cda_alpha(alpha_rad, 0.0, mach, config)
    axial_thrust_n = prop.thrust_n * math.cos(alpha_rad)
    predicted_axial_acceleration_mps2 = (axial_thrust_n - drag_zero_n - drag_alpha_n) / prop.mass_kg
    return {
        "mass_kg": prop.mass_kg,
        "thrust_n": prop.thrust_n,
        "dynamic_pressure_pa": qbar,
        "mach_model": mach,
        "vertical_trim_alpha_rad": vertical_alpha,
        "horizontal_alpha_rad": horizontal_alpha,
        "level_decomposition_valid": valid,
        "predicted_aero_normal_n": predicted_aero_normal_n,
        "predicted_thrust_normal_n": predicted_thrust_normal_n,
        "predicted_total_normal_n": predicted_total_normal_n,
        "predicted_total_load_g": predicted_total_load_g,
        "decomposed_total_load_g": decomposed_load_g,
        "drag_zero_aoa_n": drag_zero_n,
        "drag_alpha_n": drag_alpha_n,
        "drag_total_n": drag_zero_n + drag_alpha_n,
        "axial_thrust_n": axial_thrust_n,
        "predicted_axial_acceleration_mps2": predicted_axial_acceleration_mps2,
    }


def integrate_interval(config, parameters, atmosphere, propulsion, left, right, subdivisions):
    duration = float(right["time_s"]) - float(left["time_s"])
    fractions = (np.arange(subdivisions, dtype=float) + 0.5) / subdivisions
    times = float(left["time_s"]) + fractions * duration
    speeds = float(left["speed_mps"]) + fractions * (
        float(right["speed_mps"]) - float(left["speed_mps"])
    )
    alphas = float(left["alpha_rad"]) + fractions * (
        float(right["alpha_rad"]) - float(left["alpha_rad"])
    )
    components = []
    for time_s, speed_mps, alpha_rad in zip(times, speeds, alphas):
        prop = propulsion.sample(float(time_s))
        qbar = 0.5 * atmosphere.density_kg_m3 * float(speed_mps) ** 2
        mach = float(speed_mps) / atmosphere.speed_of_sound_mps
        components.append(
            (
                prop.thrust_n * math.cos(float(alpha_rad)) / prop.mass_kg,
                -qbar * effective_cda0(mach, config) / prop.mass_kg,
                -qbar
                * effective_cda_alpha(float(alpha_rad), 0.0, mach, config)
                / prop.mass_kg,
            )
        )
    impulses = np.mean(np.asarray(components), axis=0) * duration
    return {
        "thrust_delta_v_mps": float(impulses[0]),
        "zero_aoa_drag_delta_v_mps": float(impulses[1]),
        "alpha_drag_delta_v_mps": float(impulses[2]),
        "predicted_delta_v_mps": float(np.sum(impulses)),
    }


def metrics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "maximum_absolute": float(np.max(np.abs(values))),
    }


def time_windows(frame_rows, interval_rows):
    load_windows = []
    for start, end in ((0.2, 0.5), (0.7, 1.5), (1.6, 2.3), (2.5, 3.5)):
        selected = [row for row in frame_rows if start - 1e-12 <= row["time_s"] <= end + 1e-12]
        residual = np.asarray([row["load_residual_g"] for row in selected])
        load_windows.append(
            {
                "start_s": start,
                "end_s": end,
                "frame_count": len(selected),
                "mean_predicted_minus_observed_g": float(np.mean(residual)),
                "maximum_absolute_residual_g": float(np.max(np.abs(residual))),
            }
        )

    speed_windows = []
    for start, end in ((0.2, 0.7), (0.7, 1.5), (1.5, 2.3), (2.3, 3.5)):
        selected = [
            row
            for row in interval_rows
            if row["start_s"] >= start - 1e-12 and row["end_s"] <= end + 1e-12
        ]
        observed = sum(row["observed_delta_v_mps"] for row in selected)
        predicted = sum(row["predicted_delta_v_mps"] for row in selected)
        without_alpha_drag = sum(
            row["thrust_delta_v_mps"] + row["zero_aoa_drag_delta_v_mps"]
            for row in selected
        )
        alpha_sensitivity = sum(row["alpha_drag_delta_v_mps"] / EXPECTED_K for row in selected)
        k_required = (observed - without_alpha_drag) / alpha_sensitivity
        speed_windows.append(
            {
                "start_s": start,
                "end_s": end,
                "interval_count": len(selected),
                "observed_delta_v_kmh": observed * 3.6,
                "predicted_delta_v_kmh": predicted * 3.6,
                "predicted_minus_observed_delta_v_kmh": (predicted - observed) * 3.6,
                "conditional_cumulative_k_required": k_required,
            }
        )
    return load_windows, speed_windows


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(frame_rows, interval_rows, prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([row["time_s"] for row in frame_rows])
    observed_load = np.asarray([row["observed_load_g"] for row in frame_rows])
    predicted_load = np.asarray([row["predicted_total_load_g"] for row in frame_rows])
    midpoints = np.asarray([row["midpoint_s"] for row in interval_rows])
    observed_interval = np.asarray([row["observed_delta_v_kmh"] for row in interval_rows])
    predicted_interval = np.asarray([row["predicted_delta_v_kmh"] for row in interval_rows])
    cumulative_observed = np.r_[0.0, np.cumsum(observed_interval)]
    cumulative_predicted = np.r_[0.0, np.cumsum(predicted_interval)]
    cumulative_without_alpha_drag = np.r_[
        0.0,
        np.cumsum(
            [
                3.6 * (row["thrust_delta_v_mps"] + row["zero_aoa_drag_delta_v_mps"])
                for row in interval_rows
            ]
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.2))
    axes[0].plot(times, observed_load, "o-", color="#222222", lw=1.2, ms=4.5, label="Replay total load")
    axes[0].plot(times, predicted_load, "o-", color="#167d9a", lw=1.8, ms=4,
                 label="Current aggregate force at measured V, AoA")
    axes[0].set(ylabel="Total wind-normal load [g]", title="Normal-force closure")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(midpoints, observed_interval, "o-", color="#222222", lw=1.2, ms=4.5,
                 label="Observed endpoint speed change")
    axes[1].plot(midpoints, predicted_interval, "o-", color="#b87925", lw=1.8, ms=4,
                 label="Integrated current axial force")
    axes[1].axhline(0.0, color="#888888", lw=0.8)
    axes[1].set(ylabel="Native-interval delta V [km/h]", title="Interval axial impulse balance")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(times, cumulative_observed, "o-", color="#222222", lw=1.2, ms=4.5,
                 label="Observed cumulative speed change")
    axes[2].plot(times, cumulative_predicted, "o-", color="#b87925", lw=1.8, ms=4,
                 label="Current drag, K = 0.0767")
    axes[2].plot(times, cumulative_without_alpha_drag, "--", color="#8f6aa8", lw=1.6,
                 label="Diagnostic: zero AoA-drag term")
    axes[2].set(xlabel="Missile flight time [s]", ylabel="Cumulative delta V [km/h]",
                title="Cumulative axial balance from 0.2 s")
    axes[2].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.set_xlim(0.15, 3.55)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("R-77-1 | controller-independent aggregate force audit | 2.57.1.107", y=0.987,
                 fontsize=15)
    fig.text(
        0.08,
        0.018,
        "Predictions are conditioned on measured V(t) and AoA(t), linearly interpolated only within each native interval.\n"
        "Fixed 6300 m ISA atmosphere and nominal thrust/mass history; no target geometry, PN, PID, fin command, or time shift.\n"
        "This tests aggregate translation only; it does not separate body/fin lift or validate causal attitude dynamics.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.12, hspace=0.31)
    fig.savefig(prefix.with_suffix(".png"), dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    baseline_path = ROOT / "outputs/r77_1_target_course_corrected.json"
    baseline = json.loads(baseline_path.read_text())
    paths = protected_paths()
    before = {name: sha256(path) for name, path in paths.items()}

    if (baseline["source_version"], baseline["source_commit"]) != (VERSION, COMMIT):
        raise ValueError("Baseline source version/commit is not frozen 2.57.1.107")
    config, parameters, raw_limit = inputs()
    parameters = replace(parameters, fin_force_area_per_rad=2.73404068, guidance_delay_s=0.0)
    if asdict(parameters) != baseline["fixed_parameters"]:
        raise ValueError("Current force-plant parameters do not match the selected baseline report")
    if digest(config) != baseline["configuration_sha256"]:
        raise ValueError("Current constructed configuration hash does not match the baseline report")
    if before["plant_constructor"] != baseline["plant_sha256"]:
        raise ValueError("Current plant constructor hash does not match the baseline report")
    if before["replay"] != baseline["replay_sha256"]:
        raise ValueError("Current replay hash does not match the baseline report")
    if abs(raw_limit - parameters.horizontal_fin_limit_rad) > 1e-15:
        raise ValueError("Raw and current horizontal travel limits differ")

    atmosphere = StandardAtmosphere().sample(ALTITUDE_M)
    propulsion = PiecewisePropulsion.from_config(config)
    replay = load_replay(paths["replay"])
    body_area = math.pi * config["geometry"]["caliber_m"] ** 2 / 4.0
    effective_area = area_basis(config)
    equivalent_k = (
        body_area
        * config["aerodynamics"]["cx_vs_aoa"]
        / (effective_area * LIFT_SLOPE_PER_RAD**2)
    )
    if abs(parameters.lift_area_per_rad - LIFT_SLOPE_PER_RAD * effective_area) > 1e-12:
        raise ValueError("Lift-area convention is inconsistent with the frozen 14/rad baseline")
    if abs(equivalent_k - EXPECTED_K) > 1e-12:
        raise ValueError("AoA-drag convention is inconsistent with K=0.0767")

    frame_rows = []
    max_decomposition_residual = 0.0
    for row in replay:
        state = force_state(
            config,
            parameters,
            atmosphere,
            propulsion,
            float(row["time_s"]),
            float(row["speed_mps"]),
            float(row["alpha_rad"]),
        )
        observed_total_normal_n = state["mass_kg"] * G0 * float(row["observed_load_g"])
        inferred_aero_normal_n = observed_total_normal_n - state["predicted_thrust_normal_n"]
        inferred_lift_slope = inferred_aero_normal_n / (
            state["dynamic_pressure_pa"] * effective_area * float(row["alpha_rad"])
        )
        if state["level_decomposition_valid"]:
            max_decomposition_residual = max(
                max_decomposition_residual,
                abs(state["decomposed_total_load_g"] - state["predicted_total_load_g"]),
            )
        frame_rows.append(
            {
                "sample_id": row["sample_id"],
                "time_s": row["time_s"],
                "speed_kmh": float(row["speed_mps"]) * 3.6,
                "alpha_deg": math.degrees(float(row["alpha_rad"])),
                "mass_kg": state["mass_kg"],
                "thrust_n": state["thrust_n"],
                "dynamic_pressure_pa": state["dynamic_pressure_pa"],
                "mach_model": state["mach_model"],
                "vertical_trim_alpha_deg": math.degrees(state["vertical_trim_alpha_rad"]),
                "horizontal_alpha_deg": math.degrees(state["horizontal_alpha_rad"]),
                "level_decomposition_valid": state["level_decomposition_valid"],
                "observed_load_g": row["observed_load_g"],
                "predicted_total_load_g": state["predicted_total_load_g"],
                "load_residual_g": state["predicted_total_load_g"] - float(row["observed_load_g"]),
                "observed_total_normal_n": observed_total_normal_n,
                "inferred_aero_normal_n": inferred_aero_normal_n,
                "predicted_aero_normal_n": state["predicted_aero_normal_n"],
                "predicted_thrust_normal_n": state["predicted_thrust_normal_n"],
                "inferred_lift_slope_per_rad": inferred_lift_slope,
                "drag_zero_aoa_n": state["drag_zero_aoa_n"],
                "drag_alpha_n": state["drag_alpha_n"],
                "drag_total_n": state["drag_total_n"],
                "axial_thrust_n": state["axial_thrust_n"],
                "predicted_axial_acceleration_mps2": state["predicted_axial_acceleration_mps2"],
            }
        )

    interval_rows = []
    max_refinement_difference = 0.0
    cumulative_observed = 0.0
    cumulative_predicted = 0.0
    for left, right in zip(replay, replay[1:]):
        primary = integrate_interval(
            config,
            parameters,
            atmosphere,
            propulsion,
            left,
            right,
            PRIMARY_QUADRATURE_SUBDIVISIONS,
        )
        refined = integrate_interval(
            config,
            parameters,
            atmosphere,
            propulsion,
            left,
            right,
            REFINED_QUADRATURE_SUBDIVISIONS,
        )
        max_refinement_difference = max(
            max_refinement_difference,
            abs(primary["predicted_delta_v_mps"] - refined["predicted_delta_v_mps"]),
        )
        observed = float(right["speed_mps"]) - float(left["speed_mps"])
        without_alpha_drag = primary["thrust_delta_v_mps"] + primary["zero_aoa_drag_delta_v_mps"]
        k_sensitivity = primary["alpha_drag_delta_v_mps"] / equivalent_k
        k_required = (observed - without_alpha_drag) / k_sensitivity
        cumulative_observed += observed
        cumulative_predicted += primary["predicted_delta_v_mps"]
        interval_rows.append(
            {
                "start_s": left["time_s"],
                "end_s": right["time_s"],
                "midpoint_s": 0.5 * (float(left["time_s"]) + float(right["time_s"])),
                "duration_s": float(right["time_s"]) - float(left["time_s"]),
                "observed_delta_v_mps": observed,
                "predicted_delta_v_mps": primary["predicted_delta_v_mps"],
                "residual_delta_v_mps": primary["predicted_delta_v_mps"] - observed,
                "observed_delta_v_kmh": observed * 3.6,
                "predicted_delta_v_kmh": primary["predicted_delta_v_mps"] * 3.6,
                "residual_delta_v_kmh": (primary["predicted_delta_v_mps"] - observed) * 3.6,
                "thrust_delta_v_mps": primary["thrust_delta_v_mps"],
                "zero_aoa_drag_delta_v_mps": primary["zero_aoa_drag_delta_v_mps"],
                "alpha_drag_delta_v_mps": primary["alpha_drag_delta_v_mps"],
                "conditional_interval_k_required": k_required,
                "cumulative_observed_delta_v_kmh": cumulative_observed * 3.6,
                "cumulative_predicted_delta_v_kmh": cumulative_predicted * 3.6,
                "cumulative_residual_delta_v_kmh": (cumulative_predicted - cumulative_observed) * 3.6,
            }
        )

    load_residual = np.asarray([row["load_residual_g"] for row in frame_rows])
    interval_residual = np.asarray([row["residual_delta_v_mps"] for row in interval_rows])
    observed_load = np.asarray([row["observed_load_g"] for row in frame_rows])
    predicted_load = np.asarray([row["predicted_total_load_g"] for row in frame_rows])
    inferred_slopes = np.asarray([row["inferred_lift_slope_per_rad"] for row in frame_rows])
    load_windows, speed_windows = time_windows(frame_rows, interval_rows)

    observed_total_delta_v = sum(row["observed_delta_v_mps"] for row in interval_rows)
    thrust_delta_v = sum(row["thrust_delta_v_mps"] for row in interval_rows)
    zero_drag_delta_v = sum(row["zero_aoa_drag_delta_v_mps"] for row in interval_rows)
    alpha_drag_delta_v = sum(row["alpha_drag_delta_v_mps"] for row in interval_rows)
    predicted_total_delta_v = thrust_delta_v + zero_drag_delta_v + alpha_drag_delta_v
    without_alpha_drag_delta_v = thrust_delta_v + zero_drag_delta_v
    cumulative_k_required = (
        (observed_total_delta_v - without_alpha_drag_delta_v)
        / (alpha_drag_delta_v / equivalent_k)
    )

    after = {name: sha256(path) for name, path in paths.items()}
    if after != before:
        changed = [name for name in before if before[name] != after[name]]
        raise RuntimeError(f"Protected inputs changed during audit: {changed}")

    largest_load_index = int(np.argmax(np.abs(load_residual)))
    largest_interval_index = int(np.argmax(np.abs(interval_residual)))
    report = {
        "verdict": {
            "aggregate_normal_force": (
                "Supported over the high-load part of 0.7-3.5 s, with a systematic positive "
                "bias that shrinks from about 0.42 g at 0.7-1.5 s to 0.10 g at 2.5-3.5 s; "
                "the 0.2-0.5 s low-load transient does not close as well (0.82 g mean bias)."
            ),
            "axial_force_and_drag": (
                "The full-window impulse closes to -13.76 km/h on an observed +1080 km/h "
                "speed change, but time-local residuals alternate in sign and reach 43 km/h "
                "over grouped windows, so the aggregate axial magnitude is supported more "
                "strongly than the interval shape."
            ),
            "alpha_drag_k": (
                "K=0.0767 is supported conditionally by the full-window measured-input impulse: "
                "removing AoA drag overshoots the speed gain by 369.54 km/h, while the inverse "
                "cumulative diagnostic requires K=0.07395. This is not an independent game "
                "coefficient identification."
            ),
            "identifiability_boundary": (
                "This validates only the aggregate empirical translation relation at measured "
                "states; it does not separate body and fin forces and does not validate the "
                "causal attitude or controller model."
            ),
        },
        "scope": {
            "estimand": "current aggregate translational force at measured V(t) and total AoA(t)",
            "window_s": [START_S, END_S],
            "native_timestamps_no_time_shift": True,
            "measured_inputs": ["speed_kmh", "angle_of_attack_deg", "overload_g"],
            "excluded": [
                "target geometry",
                "PN",
                "PID",
                "synthetic or inferred fin command",
                "free attitude integration",
            ],
            "altitude_m": ALTITUDE_M,
        },
        "source": {
            "version": VERSION,
            "commit": COMMIT,
            "baseline_report": str(baseline_path),
            "configuration_sha256": digest(config),
            "baseline_configuration_sha256": baseline["configuration_sha256"],
            "baseline_plant_sha256": baseline["plant_sha256"],
            "baseline_replay_sha256": baseline["replay_sha256"],
            "protected_hashes_before": before,
            "protected_hashes_after": after,
            "protected_inputs_unchanged": True,
        },
        "fixed_force_parameters": {
            "lift_area_per_rad_m2": parameters.lift_area_per_rad,
            "effective_reference_area_m2": effective_area,
            "body_reference_area_m2": body_area,
            "lift_slope_per_rad_on_effective_area": LIFT_SLOPE_PER_RAD,
            "alpha_drag_k_on_effective_area": equivalent_k,
            "configured_body_area_alpha_drag_coefficient": config["aerodynamics"]["cx_vs_aoa"],
            "thrust_n_during_window": propulsion.sample(START_S).thrust_n,
            "mass_model": "190 kg minus 71 kg linearly over 7 s",
            "atmosphere_density_kg_m3": atmosphere.density_kg_m3,
            "speed_of_sound_mps": atmosphere.speed_of_sound_mps,
            "fin_translation_force_added": False,
        },
        "normal_force": {
            "frame_count": len(frame_rows),
            "invalid_level_decomposition_frames": sum(
                not row["level_decomposition_valid"] for row in frame_rows
            ),
            "predicted_minus_observed_g": metrics(load_residual),
            "correlation": float(np.corrcoef(observed_load, predicted_load)[0, 1]),
            "largest_absolute_residual": {
                "time_s": frame_rows[largest_load_index]["time_s"],
                "residual_g": frame_rows[largest_load_index]["load_residual_g"],
            },
            "inferred_aggregate_lift_slope_per_rad": {
                "mean": float(np.mean(inferred_slopes)),
                "median": float(np.median(inferred_slopes)),
                "minimum": float(np.min(inferred_slopes)),
                "maximum": float(np.max(inferred_slopes)),
            },
            "time_local_windows": load_windows,
        },
        "axial_force": {
            "interval_count": len(interval_rows),
            "interval_predicted_minus_observed_mps": metrics(interval_residual),
            "largest_absolute_interval_residual": {
                "start_s": interval_rows[largest_interval_index]["start_s"],
                "end_s": interval_rows[largest_interval_index]["end_s"],
                "residual_mps": interval_rows[largest_interval_index]["residual_delta_v_mps"],
            },
            "window_balance": {
                "observed_delta_v_mps": observed_total_delta_v,
                "observed_delta_v_kmh": observed_total_delta_v * 3.6,
                "thrust_delta_v_mps": thrust_delta_v,
                "zero_aoa_drag_delta_v_mps": zero_drag_delta_v,
                "alpha_drag_delta_v_mps": alpha_drag_delta_v,
                "predicted_delta_v_mps": predicted_total_delta_v,
                "predicted_delta_v_kmh": predicted_total_delta_v * 3.6,
                "predicted_minus_observed_mps": predicted_total_delta_v - observed_total_delta_v,
                "predicted_minus_observed_kmh": (predicted_total_delta_v - observed_total_delta_v) * 3.6,
                "without_alpha_drag_predicted_delta_v_kmh": without_alpha_drag_delta_v * 3.6,
                "without_alpha_drag_minus_observed_kmh": (
                    without_alpha_drag_delta_v - observed_total_delta_v
                )
                * 3.6,
            },
            "time_local_windows": speed_windows,
        },
        "conditional_k_diagnostic": {
            "status": "inverse data diagnostic only; no coefficient was fitted or changed",
            "current_k": equivalent_k,
            "cumulative_window_k_required": cumulative_k_required,
            "required_minus_current": cumulative_k_required - equivalent_k,
            "required_over_current": cumulative_k_required / equivalent_k,
            "interpretation_boundary": (
                "Conditioned on current Mach drag, AoA-drag formula, force direction, atmosphere, "
                "thrust/mass history, measured V/AoA interpolation, and replay readout precision."
            ),
        },
        "verification": {
            "normal_scalar_vs_level_decomposition_max_abs_g": max_decomposition_residual,
            "quadrature_primary_subdivisions_per_interval": PRIMARY_QUADRATURE_SUBDIVISIONS,
            "quadrature_refined_subdivisions_per_interval": REFINED_QUADRATURE_SUBDIVISIONS,
            "quadrature_refinement_max_abs_interval_delta_v_mps": max_refinement_difference,
        },
        "limitations": [
            "Overload is treated as total wind-normal load magnitude and includes thrust-normal force.",
            "The level-flight decomposition assumes one vertical g and aligned horizontal/vertical normal-force components.",
            "Fixed 6300 m ISA density and nominal thrust/mass are model inputs, not replay measurements.",
            "Sparse screenshot timestamps, speeds, AoA and overload are rounded readouts; native-interval residuals are correspondingly noisy.",
            "The replay columns are encoded at 0.1 s, 1 km/h, 0.1 degree and 0.1 g precision; the underlying display/screenshot rounding rule is not independently known.",
            "Measured-state agreement validates only the aggregate empirical force relation; it cannot separate body and fin force or validate causal attitude dynamics.",
        ],
    }

    prefix = ROOT / "outputs/r77_1_open_loop_forces"
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_table(prefix.with_name(prefix.name + "_frames.tsv"), frame_rows)
    write_table(prefix.with_name(prefix.name + "_intervals.tsv"), interval_rows)
    save_figure(frame_rows, interval_rows, prefix)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
