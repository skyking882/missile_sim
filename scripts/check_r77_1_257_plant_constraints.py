"""Conditional R-77-1 force and moment identifiability check, frozen at 2.57.

This reads replay states, not a free-flight prediction. No controller or plant
coefficient is changed. The force check retains the 14/rad baseline. The
separate moment check examines the constant-coefficient candidate family

    N_a,h = q (A alpha_h + B delta)
    M = q (C alpha_h + D delta + E omega / V).

For B != 0, eliminate the unknown fin angle:

    omega_dot = (q/I) P alpha_h + (N_a,h/I) Q + (q/(I V)) E omega,
    P = C - D A/B, Q = D/B.

Angles and forces here use the candidate's single-plane wind-normal
approximation. Level flight, parallel normal-force/AoA components, positive
horizontal turn, fixed altitude, nominal propulsion and I=m L^2/12 are
conditions, not additional observations. This does not validate B != 0 or
identify separate aerodynamic derivatives. No second derivative of replay
AoA is used: integrate the moment equation twice against the body-angle
change reconstructed from AoA and trajectory curvature. Piecewise-linear
reconstruction still depends on the sparse, rounded screenshot timestamps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2 as G0
from run_alpha_servo_candidate import candidate_inputs

VERSION = "2.57.1.107"
COMMIT = "3f20645862c538b46c94e3f60ee30da9dfacd81a"


def load_observations():
    profile = json.loads((ROOT / "missiles/su_r_77_1.json").read_text())
    provenance = profile["provenance"]
    if (provenance["source_version"], provenance["source_commit"]) != (VERSION, COMMIT):
        raise ValueError("This analysis requires the frozen 2.57.1.107 snapshot")
    for name, expected in provenance["source_sha256"].items():
        raw = (ROOT / "data/datamine" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen source checksum mismatch: {name}")
    config, parameters = candidate_inputs("su_r_77_1")
    with (ROOT / "data/replays/r77_1_level_20260824.tsv").open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    fields = ("missile_flight_time_s", "speed_kmh", "angle_of_attack_deg", "overload_g")
    data = np.array([[float(row[key]) for key in fields] for row in records])
    data = data[(data[:, 0] >= 0.2) & (data[:, 0] <= 3.5)]
    time, speed, alpha, load = data.T
    speed = speed / 3.6
    alpha = np.radians(alpha)
    propulsion = PiecewisePropulsion.from_config(config)
    mass = np.array([propulsion.sample(float(t)).mass_kg for t in time])
    thrust = np.array([propulsion.sample(float(t)).thrust_n for t in time])
    rho = StandardAtmosphere().sample(6300.0).density_kg_m3
    pressure = 0.5 * rho * speed**2
    sref = parameters.lift_area_per_rad / 14.0
    normal_aero = mass * G0 * load - thrust * np.sin(alpha)
    effective_slope = normal_aero / (pressure * sref * alpha)
    baseline_load = (pressure * parameters.lift_area_per_rad * alpha
                     + thrust * np.sin(alpha)) / (mass * G0)
    observations = dict(time=time, speed=speed, alpha=alpha, load=load,
                        mass=mass, thrust=thrust, pressure=pressure, rho=rho,
                        sref=sref, normal_aero=normal_aero,
                        effective_slope=effective_slope, baseline_load=baseline_load)
    return parameters, observations


def weak_moment_system(observations, parameters, start_s, subdivisions=16):
    o = observations
    mask = o["time"] >= start_s
    time, speed, mass, thrust, alpha, load = (
        o[name][mask] for name in ("time", "speed", "mass", "thrust", "alpha", "load"))
    if np.any(load <= 1.0):
        raise ValueError("Selected frames cannot satisfy the level-support decomposition")
    horizontal_fraction = np.sqrt(1.0 - 1.0 / load**2)
    alpha_h = alpha * horizontal_fraction
    accel_h = G0 * np.sqrt(load**2 - 1.0)
    normal_aero_h = (mass * G0 * load - thrust * np.sin(alpha)) * horizontal_fraction
    intervals = np.diff(time)
    heading_change = np.r_[0.0, np.cumsum(
        0.5 * (accel_h[:-1] / speed[:-1] + accel_h[1:] / speed[1:]) * intervals)]
    body_change = alpha_h - alpha_h[0] + heading_change

    # Include every observed time as a quadrature boundary. Body angle is
    # piecewise linear; omega enters only its weighted integral below.
    edges = np.concatenate([np.linspace(a, b, subdivisions, endpoint=False)
                            for a, b in zip(time[:-1], time[1:])] + [time[-1:]])
    widths = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    v = np.interp(mid, time, speed)
    m = np.interp(mid, time, mass)
    q = 0.5 * o["rho"] * v**2
    inertia = m * parameters.length_m**2 / 12.0
    segment = np.clip(np.searchsorted(time, mid, side="right") - 1, 0, len(time) - 2)
    omega_piecewise = np.diff(body_change)[segment] / intervals[segment]
    force_terms = np.column_stack((
        q / inertia * np.interp(mid, time, alpha_h),
        np.interp(mid, time, normal_aero_h) / inertia,
        q / (inertia * v) * omega_piecewise,
    ))
    kernel = np.maximum(time[:, None] - mid[None, :], 0.0) * widths[None, :]
    # theta-theta0 = omega0*dt + integral((t-u)*omega_dot(u), du).
    matrix = np.column_stack((time - time[0], kernel @ force_terms))
    return time, matrix, body_change


def solve_scaled(matrix, target):
    scales = np.linalg.norm(matrix, axis=0)
    solution, _, _, singular = np.linalg.lstsq(matrix / scales, target, rcond=None)
    return solution / scales, float(singular[0] / singular[-1])


def identify_combinations(observations, parameters):
    results = []
    # Changing the start discards different transient information; this is
    # an identifiability check, not selection of the best-looking fit.
    for start in (0.5, 0.7, 0.9, 1.1):
        time, matrix, target = weak_moment_system(observations, parameters, start)
        free, condition = solve_scaled(matrix, target)
        fixed_q = -parameters.stability_distance_m
        fixed, fixed_condition = solve_scaled(
            matrix[:, [0, 1, 3]], target - fixed_q * matrix[:, 2])
        results.append(dict(
            start_s=start, end_s=float(time[-1]),
            omega0_rad_s=float(free[0]), alpha_combo_m3_per_rad=float(free[1]),
            force_moment_ratio_m=float(free[2]), rate_combo_m4=float(free[3]),
            scaled_condition_number=condition,
            max_body_angle_residual_deg=float(np.degrees(abs(matrix @ free - target)).max()),
            fixed_force_moment_ratio_m=fixed_q,
            fixed_ratio_omega0_rad_s=float(fixed[0]),
            fixed_ratio_alpha_combo_m3_per_rad=float(fixed[1]),
            fixed_ratio_rate_combo_m4=float(fixed[2]),
            fixed_ratio_scaled_condition_number=fixed_condition,
        ))
    return results


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    parameters, o = load_observations()
    fits = identify_combinations(o, parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "r77_1_257_plant_constraints"
    force_rows = [dict(time_s=t, observed_load_g=n, alpha14_load_g=b,
                       observed_alpha_deg=float(np.degrees(a)), effective_lift_slope_per_rad=k)
                  for t, n, b, a, k in zip(o["time"], o["load"], o["baseline_load"],
                                          o["alpha"], o["effective_slope"])]
    write_table(prefix.with_name(prefix.name + "_force.tsv"), force_rows)
    write_table(prefix.with_name(prefix.name + "_moment.tsv"), fits)
    print(f"Frozen version={VERSION}, commit={COMMIT}; all source checksums match")
    print("PID unchanged:", parameters.acceleration_pid_kp, parameters.acceleration_pid_ki,
          parameters.acceleration_pid_kd, parameters.acceleration_pid_integral_limit)
    print("FORCE: time, replay G, alpha14 G, effective slope")
    for row in force_rows:
        print(" ".join(f"{row[key]:.4f}" for key in
                       ("time_s", "observed_load_g", "alpha14_load_g", "effective_lift_slope_per_rad")))
    print("MOMENT: start, P, Q, E, scaled condition, E with Q=-ell")
    for fit in fits:
        print(" ".join(f"{fit[key]:.5g}" for key in
                       ("start_s", "alpha_combo_m3_per_rad", "force_moment_ratio_m",
                        "rate_combo_m4", "scaled_condition_number", "fixed_ratio_rate_combo_m4")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    axes[0].plot(o["time"], o["baseline_load"], color="#246a92", label="14/rad at observed V and AoA")
    axes[0].scatter(o["time"], o["load"], color="#b45b32", s=23, label="Replay load", zorder=3)
    axes[0].set(xlim=(0.15, 1.55), xlabel="Flight time [s]", ylabel="Total normal load [g]",
                title="Early force check: observed states, no controller")
    axes[0].legend(fontsize=9)
    axes[1].plot(o["time"], o["effective_slope"], "o-", color="#b45b32", markersize=4)
    axes[1].axhline(14, color="#246a92", ls="--", label="Retained baseline: 14/rad")
    axes[1].set(xlim=(.15, 3.55), xlabel="Flight time [s]", ylabel="Effective lift slope [1/rad]",
                title="Force-derived slope along the observed trajectory")
    axes[1].legend(fontsize=9)
    axes[2].plot([f["start_s"] for f in fits], [f["force_moment_ratio_m"] for f in fits],
                 "o-", color="#754b91", label="Unconstrained weak-angle identification")
    axes[2].axhline(-parameters.stability_distance_m, color="#555", ls="--", label="Tail-force hypothesis: Q = -0.175 m")
    axes[2].axhline(0, color="#aaa", lw=.7)
    axes[2].set(xlabel="Identification window start [s]; common end = 3.5 s",
                ylabel="Fitted moment/force ratio Q [m]",
                title="Moment identification is sensitive to the included transient")
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("R-77-1 | frozen 2.57.1.107 | conditional plant constraints", fontsize=14)
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
