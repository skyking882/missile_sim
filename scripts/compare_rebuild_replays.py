#!/usr/bin/env python3
"""Compare the unchanged rebuild with all five screenshot replays.

Scenario inputs follow tests/test_replay_anchors.py (A2/A3/A4/A7). A2 uses
straight target motion although its recorded target maneuvered. Metrics use
missile flight time, linear model interpolation, and no extrapolation. Game
hits are user-confirmed; last screenshot times are not impact times.

The separate axial balance evaluates fixed propulsion/drag at measured V and
AoA and nominal launch altitude, with zero flight-path slope. It is a
conditional component comparison, not a second closed-loop simulation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from run_rebuild_scenarios import SCENARIO, load_rebuild_profile
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import effective_cda0, effective_cda_alpha
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.public_api import simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CASES = (
    ("30degpl12", "cn_pl12", "pl12_30deg_20260823.tsv", 30,
     dict(launch_speed_kmh=1508, launch_altitude_m=6783,
          target_speed_kmh=1549, target_altitude_m=6783, max_simulation_time_s=40)),
    ("30degr77", "su_r_77", "r77_30deg_20260823.tsv", 41,
     dict(launch_speed_kmh=1500)),
    ("30degpl12slow", "cn_pl12", "pl12_30deg_slow_20260823.tsv", 50, {}),
    ("pl12level", "cn_pl12", "pl12_level_20260824.tsv", 12,
     dict(launch_altitude_m=6300, target_speed_kmh=1200,
          target_altitude_m=6300, max_simulation_time_s=40)),
    ("r77-1", "su_r_77_1", "r77_1_level_20260824.tsv", 46,
     dict(launch_altitude_m=6300, target_speed_kmh=1200,
          target_heading_deg=0, target_course_reference="sensorwhale_launch_axis",
          target_altitude_m=6300, max_simulation_time_s=40)),
)
OBS_FIELDS = ("speed_kmh", "angle_of_attack_deg", "overload_g")


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def axial_balance(config, scenario, time, observed):
    """Midpoint integration avoids derivatives of rounded screenshot speeds."""
    atmosphere = StandardAtmosphere().sample(scenario["launch_altitude_m"])
    propulsion = PiecewisePropulsion.from_config(config)
    speed = observed[:, 0] / 3.6
    alpha = np.radians(observed[:, 1])
    predicted = [observed[0, 0]]
    components = []
    for lo, hi in zip(time[:-1], time[1:]):
        count = math.ceil((hi - lo) / 0.005)
        step = (hi - lo) / count
        midpoint = lo + (np.arange(count) + 0.5) * step
        velocities = np.interp(midpoint, time, speed)
        alphas = np.interp(midpoint, time, alpha)
        terms = []
        for t, v, a in zip(midpoint, velocities, alphas):
            p = propulsion.sample(float(t))
            q = 0.5 * atmosphere.density_kg_m3 * v * v
            mach = v / atmosphere.speed_of_sound_mps
            terms.append((p.thrust_n * math.cos(a) / p.mass_kg,
                          q * effective_cda0(mach, config) / p.mass_kg,
                          q * effective_cda_alpha(a, 0.0, mach, config) / p.mass_kg))
        impulse = np.sum(terms, axis=0) * step * 3.6
        components.append(impulse)
        predicted.append(predicted[-1] + impulse[0] - impulse[1] - impulse[2])
    return np.array(predicted), np.array(components)


def run_case(folder, missile, filename, image_count, overrides):
    scenario = dict(SCENARIO, **overrides)
    profile = load_rebuild_profile(missile)
    config = profile["_model_config"]
    assert config["control"]["plant_semantics"] == "rebuild_first_principles"
    with (ROOT / "data/replays" / filename).open(encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(all_rows) == image_count
    rows = [r for r in all_rows if all(r[k] != "NA" for k in
            ("missile_flight_time_s", *OBS_FIELDS))]
    time = np.array([float(r["missile_flight_time_s"]) for r in rows])
    observed = np.array([[float(r[k]) for k in OBS_FIELDS] for r in rows])
    assert np.all(np.diff(time) > 0) and np.all(np.isfinite(observed))

    result = simulate(profile, scenario)
    samples = result["samples"]
    model_time = np.array([s["time_s"] for s in samples])
    model = np.array([[math.hypot(*s["velocity_mps"]) * 3.6,
                       math.degrees(s["angle_of_attack_rad"]),
                       s["trajectory_lateral_load_g"]] for s in samples])
    assert np.all(np.diff(model_time) > 0) and np.all(np.isfinite(model))
    matched = (time >= model_time[0]) & (time <= model_time[-1])
    interpolated = np.full_like(observed, np.nan)
    for column in range(3):
        interpolated[matched, column] = np.interp(time[matched], model_time, model[:, column])
    summary = result["summary"]
    metrics = dict(group=folder, missile=missile, images=len(all_rows),
                   telemetry_rows=len(rows), compared_rows=int(matched.sum()),
                   game_outcome="hit_user_confirmed", model_event=summary["termination_event"],
                   model_end_time_s=summary["flight_time_s"],
                   model_minimum_distance_m=summary["minimum_distance_m"])
    for column, key in enumerate(OBS_FIELDS):
        error = (interpolated - observed)[matched, column]
        metrics[f"{key}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(error * error)))
        metrics[f"{key}_bias"] = float(np.mean(error))

    axial, impulses = axial_balance(config, scenario, time, observed)
    metrics["axial_conditional_speed_mae_kmh"] = float(np.mean(np.abs(axial - observed[:, 0])))
    metrics["axial_conditional_final_error_kmh"] = float(axial[-1] - observed[-1, 0])
    interval_rows = []
    for index, (lo, hi) in enumerate(zip(time[:-1], time[1:])):
        impulse = impulses[index]
        actual = observed[index + 1, 0] - observed[index, 0]
        predicted = impulse[0] - impulse[1] - impulse[2]
        interval_rows.append(dict(group=folder, start_s=lo, end_s=hi,
            observed_delta_v_kmh=actual, predicted_delta_v_kmh=predicted,
            thrust_delta_v_kmh=impulse[0], zero_aoa_drag_delta_v_kmh=impulse[1],
            aoa_drag_delta_v_kmh=impulse[2],
            missing_acceleration_mps2=(actual - predicted) / (3.6 * (hi - lo))))

    diagnostics = {}
    diagnostic_series = {
        "fin_magnitude_deg": [math.degrees(math.hypot(s["actual_pitch_fin_angle_rad"],
                                                     s["actual_yaw_fin_angle_rad"])) for s in samples],
        "command_g": [math.hypot(*s["wind_normal_specific_force_command_g"]) for s in samples],
        "pitch_alpha_deg": [math.degrees(s["pitch_alpha_rad"]) for s in samples],
        "yaw_alpha_deg": [math.degrees(s["yaw_alpha_rad"]) for s in samples],
        "altitude_m": [s["position_m"][1] for s in samples],
    }
    for key, series in diagnostic_series.items():
        diagnostics[key] = np.interp(time, model_time, series, left=np.nan, right=np.nan)
    aligned_rows = []
    for index, row in enumerate(rows):
        aligned = dict(group=folder, source_file=row["source_file"], time_s=time[index],
                       compared=bool(matched[index]))
        for column, key in enumerate(OBS_FIELDS):
            aligned[f"game_{key}"] = observed[index, column]
            aligned[f"model_{key}"] = interpolated[index, column]
        aligned["axial_conditional_speed_kmh"] = axial[index]
        for key, values in diagnostics.items():
            aligned[f"model_{key}"] = values[index]
        aligned_rows.append(aligned)
    print(f"{folder}: {matched.sum()}/{len(rows)} rows; "
          f"MAE V={metrics['speed_kmh_mae']:.1f} km/h, "
          f"alpha={metrics['angle_of_attack_deg_mae']:.2f} deg, "
          f"G={metrics['overload_g_mae']:.2f}; "
          f"event={summary['termination_event']} at {summary['flight_time_s']:.3f} s, "
          f"min range={summary['minimum_distance_m']:.1f} m; "
          f"conditional axial final error={axial[-1]-observed[-1, 0]:+.1f} km/h", flush=True)
    return dict(folder=folder, time=time, observed=observed, model_time=model_time,
                model=model, axial=axial, metrics=metrics, aligned=aligned_rows,
                intervals=interval_rows)


def plot(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(5, 3, figsize=(14, 14), sharex=True, sharey="col")
    for row, case in enumerate(cases):
        for col, label in enumerate(("Speed (km/h)", "AoA (deg)", "Normal load (g)")):
            ax = axes[row, col]
            within = case["model_time"] <= 8.1
            ax.plot(case["model_time"][within], case["model"][within, col],
                    color="#c45137", lw=1.4, label="Rebuild (unchanged)")
            ax.scatter(case["time"], case["observed"][:, col], s=19,
                       facecolors="white", edgecolors="#1b3448", lw=1.1, zorder=3,
                       label="Game screenshots")
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
            ax.set_xlim(0, 8.1)
            if row == 4:
                ax.set_xlabel("Missile flight time (s)")
        metric = case["metrics"]
        axes[row, 0].set_title(case["folder"], loc="left", fontweight="bold")
        axes[row, 1].set_title(f"MAE {metric['angle_of_attack_deg_mae']:.2f} deg", loc="left")
        axes[row, 2].set_title(f"MAE {metric['overload_g_mae']:.2f} g", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Five game replays vs the current rebuild", x=0.07, ha="left", fontsize=20)
    fig.text(0.07, 0.951, "All five game shots hit. Same rebuild coefficients and raw PN/PID in every model run.", fontsize=11)
    fig.text(0.07, 0.028, "Native model curves and original screenshot points. Metrics interpolate only the model, with no time shift or extrapolation.\n"
             "30degpl12: historical straight-target approximation; actual target maneuvered. Model G is unsigned; signed screenshot G is preserved.",
             fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.915, bottom=0.09, hspace=0.45, wspace=0.25)
    fig.savefig(OUT / "rebuild_replay_comparison.png", dpi=155, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)
    for ax, case in zip(axes, cases):
        ax.plot(case["time"], case["axial"], color="#2c7199", lw=2,
                label="Fixed axial model at measured V/AoA")
        ax.scatter(case["time"], case["observed"][:, 0], s=20, facecolors="white",
                   edgecolors="#1b3448", zorder=3, label="Game screenshots")
        ax.set(title=case["folder"], xlabel="Missile flight time (s)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Speed (km/h)")
    fig.suptitle("Axial balance with the observed AoA history", x=0.06, ha="left", fontsize=19)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.89), ncol=2, frameon=False)
    fig.text(0.06, 0.03, "Conditional integral: V(first frame) + integral[(T cos(alpha) - D0 - Dalpha) / m] dt, evaluated at measured V and alpha.\n"
             "Fixed launch altitude, zero flight-path slope, nominal propulsion, unchanged drag. No guidance input or coefficient fit; no trajectory validation.",
             fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.73, bottom=0.22, wspace=0.2)
    fig.savefig(OUT / "rebuild_replay_axial_balance.png", dpi=155, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    cases = [run_case(*case) for case in CASES]
    write_tsv(OUT / "rebuild_replay_metrics.tsv", [c["metrics"] for c in cases])
    write_tsv(OUT / "rebuild_replay_comparison.tsv", [r for c in cases for r in c["aligned"]])
    write_tsv(OUT / "rebuild_replay_axial_balance.tsv", [r for c in cases for r in c["intervals"]])
    plot(cases)
    print(f"Figures and numerical comparisons: {OUT}")


if __name__ == "__main__":
    main()
