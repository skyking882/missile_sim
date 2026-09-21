#!/usr/bin/env python3
"""Run the shared-lift fin-moment candidate freely against 2.57 replays."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from aim120_model.alpha_servo import AlphaServoParameters, run_alpha_servo
from aim120_model.aerodynamics import reference_area
from gen0_alpha_servo_model import CASES, observed
from run_rebuild_scenarios import load_rebuild_profile


SHARED_LIFT_SLOPE_PER_RAD = 14.0
R77_1_INDUCED_DRAG_K = 0.0767
# Frozen numerical starting points from the previous candidate, not identified
# aerodynamic coefficients. Neither finsLatAccel nor fin travel changes them.
PROVISIONAL_FIN_FORCE_AREAS_PER_RAD = {
    "su_r_77_1": 0.4151096915927259,
    "su_r_77": 0.38670744953638153,
    "cn_pl12": 0.33423966313841424,
}


def candidate_inputs(missile_id: str, *, fin_force_area_per_rad: float | None = None,
                     restoring_force_area_per_rad: float | None = None,
                     rate_damping_derivative_m4: float = 0.0):
    if not math.isfinite(rate_damping_derivative_m4) or rate_damping_derivative_m4 < 0.0:
        raise ValueError("rate_damping_derivative_m4 must be nonnegative and finite")
    if restoring_force_area_per_rad is not None and (
            not math.isfinite(restoring_force_area_per_rad) or restoring_force_area_per_rad <= 0):
        raise ValueError("restoring_force_area_per_rad must be positive and finite")
    profile = load_rebuild_profile(missile_id)
    provenance = profile["provenance"]
    if not provenance["source_version"].startswith("2.57."):
        raise ValueError("The frozen lift/response identification applies to 2.57 only")
    sources = provenance["source_files"]
    source = next((path for path in sources if Path(path).stem == missile_id), sources[0])
    raw_path = ROOT / "data/datamine" / source
    import hashlib
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != provenance["source_sha256"][source]:
        raise ValueError(f"raw source and profile version differ: {missile_id}")
    rocket = json.loads(raw_path.read_text())["rocket"]
    autopilot = rocket["guidance"]["guidanceAutopilot"]
    fuse = rocket["proximityFuse"]
    for field in ("finsAoaHor", "finsAoaVer"):
        if not 0.0 < rocket[field] <= math.pi / 2.0:
            raise ValueError(f"{field} must be a control-angle limit in (0, pi/2] radians")
    config = profile["_model_config"]
    fin_force_area = (PROVISIONAL_FIN_FORCE_AREAS_PER_RAD.get(missile_id)
                      if fin_force_area_per_rad is None else fin_force_area_per_rad)
    if fin_force_area is None or not math.isfinite(fin_force_area) or fin_force_area <= 0.0:
        raise ValueError("Supply an independent positive fin_force_area_per_rad for this missile")
    if missile_id == "su_r_77_1":
        # Retain the preceding speed-fit estimate as an input, not a refit of
        # the new attitude model. This is an internal alpha² coefficient.
        config["aerodynamics"]["cx_vs_aoa"] = (
            R77_1_INDUCED_DRAG_K * rocket["wingAreaMult"] * SHARED_LIFT_SLOPE_PER_RAD ** 2
        )
    parameters = AlphaServoParameters(
        lift_area_per_rad=(SHARED_LIFT_SLOPE_PER_RAD
                           * reference_area(config)),
        # The normalized PID output multiplies this raw radian limit once.
        # Its interpretation as a physical fin angle remains a hypothesis.
        horizontal_fin_limit_rad=rocket["finsAoaHor"],
        vertical_fin_limit_rad=rocket["finsAoaVer"],
        stability_distance_m=rocket["distFromCmToStab"],
        fin_force_area_per_rad=fin_force_area,
        restoring_force_area_per_rad=restoring_force_area_per_rad,
        fin_lateral_acceleration_limit_g=rocket["finsLatAccel"],
        length_m=rocket["length"],
        navigation_gain=autopilot["propNavMult"],
        command_limit_g=autopilot["reqAccelMax"],
        fuse_radius_m=fuse["radius"],
        fuse_arming_time_s=fuse["timeOut"],
        capture_time_constant_s=0.70 if missile_id == "cn_pl12" else 0.30,
        acceleration_pid_kp=autopilot["accelControlProp"],
        acceleration_pid_ki=autopilot["accelControlIntg"],
        acceleration_pid_kd=autopilot["accelControlDiff"],
        acceleration_pid_integral_limit=autopilot["accelControlIntgLim"],
        # R-77-1 AoA-only calibration over 1.5--3.5 s; not a datamine value.
        release_time_constant_s=2.5 if missile_id == "su_r_77_1" else 0.95,
        rate_damping_derivative_m4=rate_damping_derivative_m4,
    )
    return config, parameters


def compare_samples(samples, observations):
    times = np.array([sample["time_s"] for sample in samples])
    fields = ("speed_kmh", "alpha_deg", "wind_normal_specific_force_g")
    mask = (observations[:, 0] >= times[0]) & (observations[:, 0] <= times[-1])
    rows = []
    errors = []
    for index, obs in enumerate(observations):
        row = dict(time_s=obs[0], compared=bool(mask[index]))
        error = []
        for column, field in enumerate(fields, 1):
            predicted = (float(np.interp(obs[0], times, [s[field] for s in samples]))
                         if mask[index] else math.nan)
            row[f"observed_{field}"] = obs[column]
            row[f"predicted_{field}"] = predicted
            row[f"error_{field}"] = predicted - obs[column]
            error.append(predicted - obs[column])
        rows.append(row)
        if mask[index]:
            errors.append(error)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2, axis=0)) if errors else np.full(3, np.nan)
    return rows, rmse, int(mask.sum())


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--case", choices=["all"] + [case[0] for case in CASES], default="all")
    parser.add_argument("--max-time-s", type=float, default=15.0)
    parser.add_argument("--fin-force-area-per-rad", type=float,
                        help="Independent control-force slope [m^2/rad]")
    parser.add_argument("--restoring-force-area-per-rad", type=float,
                        help="Independent restoring-force slope [m^2/rad]; omitted uses control slope")
    parser.add_argument("--rate-damping-derivative-m4", type=float, default=0.0,
                        help="Independent D_omega in M=-(q/V)*D_omega*omega; not datamine WdK")
    args = parser.parse_args()
    cases = [case for case in CASES if args.case in ("all", case[0])]
    results = {}
    case_parameters = {}
    # All requested integrations finish before the replay reader is called.
    for case, missile_id, replay, scenario in cases:
        config, parameters = candidate_inputs(
            missile_id, fin_force_area_per_rad=args.fin_force_area_per_rad,
            restoring_force_area_per_rad=args.restoring_force_area_per_rad,
            rate_damping_derivative_m4=args.rate_damping_derivative_m4)
        results[case] = run_alpha_servo(config, parameters, scenario, max_time_s=args.max_time_s)
        case_parameters[case] = parameters

    all_samples, comparisons, metrics = [], [], []
    observations = {}
    for case, missile_id, replay, scenario in cases:
        result = results[case]
        obs = observed(replay)
        obs = obs[obs[:, 0] <= args.max_time_s]
        observations[case] = obs
        rows, rmse, count = compare_samples(result.samples, obs)
        comparisons.extend(dict(case=case, **row) for row in rows)
        all_samples.extend(dict(case=case, **row) for row in result.samples)
        metrics.append(dict(case=case, event=result.event,
                            fin_force_area_per_rad=case_parameters[case].fin_force_area_per_rad,
                            restoring_force_area_per_rad=case_parameters[case].restoring_area_per_rad,
                            rate_damping_derivative_m4=args.rate_damping_derivative_m4,
                            terminal_time_s=result.samples[-1]["time_s"],
                            minimum_distance_m=result.minimum_distance_m,
                            compared_frames=count, total_frames=len(obs),
                            speed_rmse_kmh=rmse[0], alpha_rmse_deg=rmse[1], g_rmse=rmse[2],
                            max_alpha_deg=max(row["alpha_deg"] for row in result.samples),
                            max_g=max(row["wind_normal_specific_force_g"] for row in result.samples)))
        print(f"{case}: {result.event} t={result.samples[-1]['time_s']:.3f}s "
              f"distance={result.minimum_distance_m:.2f}m frames={count}/{len(obs)} "
              f"RMSE V/alpha/G={rmse[0]:.2f}/{rmse[1]:.2f}/{rmse[2]:.2f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("trajectory", all_samples), ("comparison", comparisons), ("metrics", metrics)):
        write_table(args.output_dir / f"fin_moment_{name}.tsv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(cases), figsize=(max(9, 4 * len(cases)), 10), sharex="col", squeeze=False)
    for column, (case, _, _, _) in enumerate(cases):
        result, obs, metric = results[case], observations[case], metrics[column]
        for row, (field, ylabel) in enumerate((("speed_kmh", "Speed [km/h]"),
                                             ("alpha_deg", "AoA [deg]"),
                                             ("wind_normal_specific_force_g", "Wind-normal load [g]"))):
            ax = axes[row, column]
            ax.plot([s["time_s"] for s in result.samples], [s[field] for s in result.samples],
                    color="#236a94", label="Free integration", lw=1.5)
            ax.scatter(obs[:, 0], obs[:, row + 1], color="#bb592f", s=16, label="2.57 replay", zorder=3)
            ax.axvline(result.samples[-1]["time_s"], color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, max(obs[-1, 0], result.samples[-1]["time_s"]) + 0.15)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{case}\nIntegration through {metric['terminal_time_s']:.2f}s; "
                             f"{metric['compared_frames']}/{metric['total_frames']} frames", fontsize=10)
            if row == 2:
                ax.set_xlabel("Flight time [s]")
            error = (metric["speed_rmse_kmh"], metric["alpha_rmse_deg"], metric["g_rmse"])[row]
            ax.text(0.04, 0.93, f"RMSE {error:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Fin-moment candidate | independent free flight | 2.57 parameters", fontsize=17, y=0.98)
    fig.text(0.05, 0.055,
             f"Shared lift slope={SHARED_LIFT_SLOPE_PER_RAD:g} per rad; "
             "Sref=body cross-section x wingAreaMult.\n"
             f"Additional rate derivative D_omega={args.rate_damping_derivative_m4:g} m^4; not datamine WdK.\n"
             "Fixed altitude with vertical 1g trim; 11° initial horizontal AoA; 0.5s guidance delay. "
             "R-77-1 uses K=0.0767 from the preceding speed fit.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.15, top=0.90, hspace=0.16, wspace=0.25)
    path = args.output_dir / "fin_moment_comparison.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(path)

    fig, axes = plt.subplots(3, len(cases), figsize=(max(9, 4 * len(cases)), 9),
                             sharex="col", squeeze=False)
    for column, (case, _, _, _) in enumerate(cases):
        samples = results[case].samples
        times = [row["time_s"] for row in samples]
        for row, (field, factor, label) in enumerate((
                ("horizontal_fin_deflection_rad", 180.0 / math.pi, "Horizontal fin deflection [deg]"),
                ("body_rate_rad_s", 180.0 / math.pi, "Body angular rate [deg/s]"),
                ("total_aerodynamic_moment_nm", 1.0, "Aerodynamic moment [N m]"))):
            ax = axes[row, column]
            ax.plot(times, [sample[field] * factor for sample in samples],
                    color="#236a94", lw=1.8, label="Model-computed fin deflection")
            ax.axhline(0.0, color="0.6", lw=0.7)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.plot(times, [math.degrees(sample["horizontal_alpha_rad"]) for sample in samples],
                        color="#bb592f", lw=1.2, ls="--", label="Geometric horizontal AoA")
                ax.set_title(case)
                ax.legend(fontsize=9)
            if column == 0:
                ax.set_ylabel(label)
            if row == 2:
                ax.set_xlabel("Flight time [s]")
    fig.suptitle("Fin deflection, moment and body rotation", fontsize=17)
    fig.text(0.08, 0.035, "Computed internal states; the replay supplies no fin-deflection or moment measurements.",
             fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.11, top=0.91, hspace=0.17)
    path = args.output_dir / "fin_moment_controls.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
