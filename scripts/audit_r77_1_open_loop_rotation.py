"""Controller-independent rotational audit of the frozen R-77-1 candidate.

The replay supplies only speed, unsigned total AoA and total wind-normal load.
Under the explicitly conditional level-flight, positive-horizontal-turn
reconstruction, this script asks whether *some* unobserved fin history can
satisfy the current rotational equation and its raw travel/force limits.

No PN, PID, target motion, coefficient fit or inferred fin input is fed back
into a forward simulation.  Passing an admissibility inequality is therefore
not empirical validation.  The actual fin angle and body rate are unobserved.
Sparse rounded measurements are handled with four fixed local-polynomial
stencils, display-rounding intervals (at fixed reported timestamps), and a
weak interval moment balance that avoids directly using a twice-differentiated
piecewise-linear AoA trace.
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

from aim120_model.drag_models import effective_cda0, effective_cda_alpha
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2 as G0
from check_r77_1_257_plant_constraints import COMMIT, VERSION, load_observations
from run_r77_1_focus_moment import FocusPlant, inputs


ALTITUDE_M = 6300.0
WINDOW_S = (0.2, 3.5)
SPEED_HALF_ROUND_MPS = 0.5 / 3.6
ALPHA_HALF_ROUND_RAD = math.radians(0.05)
LOAD_HALF_ROUND_G = 0.05
SCHEMES = {
    "q3": (3, 2, "3-point quadratic"),
    "q5": (5, 2, "5-point quadratic"),
    "c5": (5, 3, "5-point cubic"),
    "c7": (7, 3, "7-point cubic"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    paths = [
        ROOT / "outputs/r77_1_target_course_corrected.json",
        ROOT / "data/replays/r77_1_level_20260824.tsv",
        ROOT / "scripts/run_r77_1_focus_moment.py",
        ROOT / "src/aim120_model/alpha_servo.py",
        ROOT / "config/profile_h2_rebuild_defaults.json",
        ROOT / "missiles/su_r_77_1.json",
    ]
    return {str(path): digest(path) for path in paths}


def load_current():
    reference_path = ROOT / "outputs/r77_1_target_course_corrected.json"
    reference = json.loads(reference_path.read_text())
    config, p, raw_limit = inputs()
    p = replace(
        p,
        fin_force_area_per_rad=reference["fixed_parameters"]["fin_force_area_per_rad"],
        guidance_delay_s=0.0,
    )
    if asdict(p) != reference["fixed_parameters"]:
        raise AssertionError("Current inputs do not reproduce the frozen report parameters")
    if (VERSION, COMMIT) != (reference["source_version"], reference["source_commit"]):
        raise AssertionError("Frozen source provenance changed")
    if digest(ROOT / "scripts/run_r77_1_focus_moment.py") != reference["plant_sha256"]:
        raise AssertionError("FocusPlant source no longer matches the current report")
    if digest(ROOT / "data/replays/r77_1_level_20260824.tsv") != reference["replay_sha256"]:
        raise AssertionError("Replay source no longer matches the current report")
    _, obs = load_observations()
    mask = (obs["time"] >= WINDOW_S[0] - 1e-12) & (obs["time"] <= WINDOW_S[1] + 1e-12)
    obs = {key: np.asarray(value)[mask] if np.ndim(value) else value for key, value in obs.items()}
    if len(obs["time"]) != 19:
        raise AssertionError("Expected 19 native replay frames in 0.2--3.5 s")
    plant = FocusPlant(config, p, "legacy_control_scale", raw_limit)
    return reference, config, p, plant, obs


def alpha_components(total_alpha: float, speed: float, time_s: float, p, propulsion):
    ps = propulsion.sample(float(time_s))
    qbar = 0.5 * RHO * speed * speed
    sinc = math.sin(total_alpha) / total_alpha if total_alpha else 1.0
    vertical = ps.mass_kg * G0 / (qbar * p.lift_area_per_rad + ps.thrust_n * sinc)
    if total_alpha + 1e-14 < vertical:
        raise ValueError("Measured total AoA cannot supply current algebraic vertical support")
    horizontal = math.sqrt(max(total_alpha * total_alpha - vertical * vertical, 0.0))
    return horizontal, vertical


def stencil_indices(index: int, count: int, total: int) -> np.ndarray:
    left = min(max(index - count // 2, 0), total - count)
    return np.arange(left, left + count)


def derivative_weights(time: np.ndarray, index: int, count: int, degree: int, order: int):
    selected = stencil_indices(index, count, len(time))
    x = time[selected] - time[index]
    matrix = np.column_stack([x**power for power in range(degree + 1)])
    weights = math.factorial(order) * np.linalg.pinv(matrix)[order]
    return selected, weights


def differentiate(time: np.ndarray, values: np.ndarray, count: int, degree: int, order: int):
    result = np.empty_like(values, dtype=float)
    for index in range(len(time)):
        selected, weights = derivative_weights(time, index, count, degree, order)
        result[index] = weights @ values[selected]
    return result


def weighted_interval(weights: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    lo = np.sum(np.where(weights >= 0.0, weights * lower, weights * upper))
    hi = np.sum(np.where(weights >= 0.0, weights * upper, weights * lower))
    return float(lo), float(hi)


def product_interval(a_lo: float, a_hi: float, b_lo: float, b_hi: float):
    values = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    return min(values), max(values)


def add_intervals(*terms):
    return sum(term[0] for term in terms), sum(term[1] for term in terms)


def scaled_interval(scale: float, lo: float, hi: float):
    return (scale * lo, scale * hi) if scale >= 0.0 else (scale * hi, scale * lo)


def divide_interval(lo: float, hi: float, denominator_lo: float, denominator_hi: float):
    if denominator_lo <= 0.0:
        raise ValueError("Positive denominator required")
    values = (lo / denominator_lo, lo / denominator_hi, hi / denominator_lo, hi / denominator_hi)
    return min(values), max(values)


def kinematics_and_bounds(obs, p, propulsion):
    time = obs["time"]
    speed = obs["speed"]
    alpha = obs["alpha"]
    load = obs["load"]
    horizontal = np.empty_like(alpha)
    vertical = np.empty_like(alpha)
    alpha_lo = np.empty_like(alpha)
    alpha_hi = np.empty_like(alpha)
    chi_lo = np.empty_like(alpha)
    chi_hi = np.empty_like(alpha)
    body_normal_lo = np.empty_like(alpha)
    body_normal_hi = np.empty_like(alpha)
    damping_lo = np.empty_like(alpha)
    damping_hi = np.empty_like(alpha)
    force_slope_lo = np.empty_like(alpha)
    force_slope_hi = np.empty_like(alpha)

    for i, (t, v, a, g_load) in enumerate(zip(time, speed, alpha, load)):
        horizontal[i], vertical[i] = alpha_components(a, v, t, p, propulsion)
        alpha_corners = []
        normal_corners = []
        damping_corners = []
        slope_corners = []
        for vv in (v - SPEED_HALF_ROUND_MPS, v + SPEED_HALF_ROUND_MPS):
            for aa in (a - ALPHA_HALF_ROUND_RAD, a + ALPHA_HALF_ROUND_RAD):
                ah, _ = alpha_components(aa, vv, t, p, propulsion)
                alpha_corners.append(ah)
                qbar = 0.5 * RHO * vv * vv
                mach = vv / PLANT.atmo.speed_of_sound_mps
                lift = qbar * p.lift_area_per_rad * ah
                drag = qbar * (
                    effective_cda0(mach, PLANT.config)
                    + effective_cda_alpha(aa, 0.0, mach, PLANT.config)
                )
                normal_corners.append(lift * math.cos(ah) + drag * math.sin(ah))
                damping_corners.append(qbar / vv * PLANT.damping_m4)
                slope_corners.append(qbar * p.fin_force_area_per_rad)
        alpha_lo[i], alpha_hi[i] = min(alpha_corners), max(alpha_corners)
        normal_corners_nominal = PLANT.body_force(v, horizontal[i], a)
        # Confirm that the explicit formula above uses the exact current force implementation.
        if not min(normal_corners) <= normal_corners_nominal[0] <= max(normal_corners):
            raise AssertionError("Body-normal interval does not contain nominal FocusPlant value")
        body_normal_lo[i], body_normal_hi[i] = min(normal_corners), max(normal_corners)
        damping_lo[i], damping_hi[i] = min(damping_corners), max(damping_corners)
        force_slope_lo[i], force_slope_hi[i] = min(slope_corners), max(slope_corners)
        low_g = max(g_load - LOAD_HALF_ROUND_G, 1.0 + 1e-12)
        high_g = g_load + LOAD_HALF_ROUND_G
        chi_lo[i] = G0 * math.sqrt(low_g * low_g - 1.0) / (v + SPEED_HALF_ROUND_MPS)
        chi_hi[i] = G0 * math.sqrt(high_g * high_g - 1.0) / (v - SPEED_HALF_ROUND_MPS)

    chi_dot = G0 * np.sqrt(load * load - 1.0) / speed
    return dict(
        alpha_h=horizontal,
        alpha_v=vertical,
        alpha_h_lo=alpha_lo,
        alpha_h_hi=alpha_hi,
        chi_dot=chi_dot,
        chi_dot_lo=chi_lo,
        chi_dot_hi=chi_hi,
        normal_lo=body_normal_lo,
        normal_hi=body_normal_hi,
        damping_lo=damping_lo,
        damping_hi=damping_hi,
        force_slope_lo=force_slope_lo,
        force_slope_hi=force_slope_hi,
    )


def nominal_plant_arrays(obs, p, propulsion, motion):
    arrays = {name: [] for name in (
        "mass", "thrust", "qbar", "normal", "axial", "lift", "drag",
        "damping_coefficient", "force_slope", "travel_force", "force_cap", "effective_force_cap",
    )}
    for t, v, total_alpha, ah in zip(obs["time"], obs["speed"], obs["alpha"], motion["alpha_h"]):
        ps = propulsion.sample(float(t))
        qbar = 0.5 * RHO * v * v
        normal, axial, lift, drag, residual = PLANT.body_force(v, ah, total_alpha)
        if residual > 1e-8:
            raise AssertionError("FocusPlant force rotation identity failed")
        slope = qbar * p.fin_force_area_per_rad
        travel = slope * p.horizontal_fin_limit_rad
        force_cap = ps.mass_kg * G0 * p.fin_lateral_acceleration_limit_g
        values = (
            ps.mass_kg, ps.thrust_n, qbar, normal, axial, lift, drag,
            qbar / v * PLANT.damping_m4, slope, travel, force_cap, min(travel, force_cap),
        )
        for name, value in zip(arrays, values):
            arrays[name].append(value)
    return {name: np.asarray(value, dtype=float) for name, value in arrays.items()}


def scheme_result(name, count, degree, obs, motion, plant_arrays, p, inertia_scale=1.0):
    time = obs["time"]
    alpha_rate = differentiate(time, motion["alpha_h"], count, degree, 1)
    alpha_accel = differentiate(time, motion["alpha_h"], count, degree, 2)
    chi_accel = differentiate(time, motion["chi_dot"], count, degree, 1)
    omega = motion["chi_dot"] + alpha_rate
    omega_dot = chi_accel + alpha_accel
    inertia = inertia_scale * plant_arrays["mass"] * p.length_m**2 / 12.0
    required_moment = inertia * omega_dot
    damping_moment = -plant_arrays["damping_coefficient"] * omega
    required_force = (
        required_moment + p.stability_distance_m * plant_arrays["normal"]
        + plant_arrays["damping_coefficient"] * omega
    ) / p.stability_distance_m
    required_delta = required_force / plant_arrays["force_slope"]
    static_force = (
        p.stability_distance_m * plant_arrays["normal"]
        + plant_arrays["damping_coefficient"] * omega
    ) / p.stability_distance_m
    static_delta = static_force / plant_arrays["force_slope"]

    omega_lo = np.empty_like(omega)
    omega_hi = np.empty_like(omega)
    omega_dot_lo = np.empty_like(omega)
    omega_dot_hi = np.empty_like(omega)
    for i in range(len(time)):
        selected, w1 = derivative_weights(time, i, count, degree, 1)
        ar_lo, ar_hi = weighted_interval(
            w1, motion["alpha_h_lo"][selected], motion["alpha_h_hi"][selected]
        )
        selected, w2 = derivative_weights(time, i, count, degree, 2)
        aa_lo, aa_hi = weighted_interval(
            w2, motion["alpha_h_lo"][selected], motion["alpha_h_hi"][selected]
        )
        selected, wc = derivative_weights(time, i, count, degree, 1)
        ca_lo, ca_hi = weighted_interval(
            wc, motion["chi_dot_lo"][selected], motion["chi_dot_hi"][selected]
        )
        omega_lo[i] = motion["chi_dot_lo"][i] + ar_lo
        omega_hi[i] = motion["chi_dot_hi"][i] + ar_hi
        omega_dot_lo[i] = aa_lo + ca_lo
        omega_dot_hi[i] = aa_hi + ca_hi

    weak = []
    inertia_dot = inertia_scale * (-71.0 / 7.0) * p.length_m**2 / 12.0
    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        inertial_nominal = (
            inertia[i + 1] * omega[i + 1] - inertia[i] * omega[i]
            - 0.5 * dt * inertia_dot * (omega[i] + omega[i + 1])
        )
        passive_nominal = 0.5 * dt * p.stability_distance_m * (
            plant_arrays["normal"][i] + plant_arrays["normal"][i + 1]
        )
        damping_nominal = 0.5 * dt * (
            plant_arrays["damping_coefficient"][i] * omega[i]
            + plant_arrays["damping_coefficient"][i + 1] * omega[i + 1]
        )
        required_impulse = (inertial_nominal + passive_nominal + damping_nominal) / p.stability_distance_m
        static_impulse = (passive_nominal + damping_nominal) / p.stability_distance_m

        inertial_interval = add_intervals(
            scaled_interval(inertia[i + 1], omega_lo[i + 1], omega_hi[i + 1]),
            scaled_interval(-inertia[i], omega_lo[i], omega_hi[i]),
            scaled_interval(-0.5 * dt * inertia_dot, omega_lo[i], omega_hi[i]),
            scaled_interval(-0.5 * dt * inertia_dot, omega_lo[i + 1], omega_hi[i + 1]),
        )
        passive_interval = scaled_interval(
            0.5 * dt * p.stability_distance_m,
            motion["normal_lo"][i] + motion["normal_lo"][i + 1],
            motion["normal_hi"][i] + motion["normal_hi"][i + 1],
        )
        damping_terms = []
        for j in (i, i + 1):
            product = product_interval(
                motion["damping_lo"][j], motion["damping_hi"][j], omega_lo[j], omega_hi[j]
            )
            damping_terms.append(scaled_interval(0.5 * dt, *product))
        impulse_lo, impulse_hi = add_intervals(inertial_interval, passive_interval, *damping_terms)
        impulse_lo /= p.stability_distance_m
        impulse_hi /= p.stability_distance_m

        slope_impulse = 0.5 * dt * (
            plant_arrays["force_slope"][i] + plant_arrays["force_slope"][i + 1]
        )
        slope_impulse_lo = 0.5 * dt * (
            motion["force_slope_lo"][i] + motion["force_slope_lo"][i + 1]
        )
        slope_impulse_hi = 0.5 * dt * (
            motion["force_slope_hi"][i] + motion["force_slope_hi"][i + 1]
        )
        travel_cap_impulse = p.horizontal_fin_limit_rad * slope_impulse
        force_cap_impulse = 0.5 * dt * (
            plant_arrays["force_cap"][i] + plant_arrays["force_cap"][i + 1]
        )
        effective_cap_impulse = 0.5 * dt * (
            plant_arrays["effective_force_cap"][i] + plant_arrays["effective_force_cap"][i + 1]
        )
        # Use the largest capacity allowed by display rounding for a conservative rejection check.
        travel_cap_hi = p.horizontal_fin_limit_rad * slope_impulse_hi
        force_cap_hi = force_cap_impulse
        effective_cap_hi = min(travel_cap_hi, force_cap_hi)
        delta_equiv = required_impulse / slope_impulse
        static_delta_equiv = static_impulse / slope_impulse
        delta_lo, delta_hi = divide_interval(impulse_lo, impulse_hi, slope_impulse_lo, slope_impulse_hi)
        weak.append(dict(
            interval_index=i,
            start_s=float(time[i]),
            end_s=float(time[i + 1]),
            center_s=float(0.5 * (time[i] + time[i + 1])),
            required_control_impulse_ns=float(required_impulse),
            required_control_impulse_lo_ns=float(impulse_lo),
            required_control_impulse_hi_ns=float(impulse_hi),
            equivalent_delta_rad=float(delta_equiv),
            static_equivalent_delta_rad=float(static_delta_equiv),
            inertial_delta_increment_rad=float(delta_equiv - static_delta_equiv),
            equivalent_delta_lo_rad=float(delta_lo),
            equivalent_delta_hi_rad=float(delta_hi),
            travel_cap_impulse_ns=float(travel_cap_impulse),
            force_cap_impulse_ns=float(force_cap_impulse),
            effective_cap_impulse_ns=float(effective_cap_impulse),
            nominal_travel_violation=bool(abs(required_impulse) > travel_cap_impulse),
            nominal_force_violation=bool(abs(required_impulse) > force_cap_impulse),
            nominal_effective_violation=bool(abs(required_impulse) > effective_cap_impulse),
            effective_limit_utilization=float(abs(required_impulse) / effective_cap_impulse),
            display_rounding_robust_violation=bool(
                impulse_lo > effective_cap_hi or impulse_hi < -effective_cap_hi
            ),
        ))

    return dict(
        name=name,
        count=count,
        degree=degree,
        inertia_scale=inertia_scale,
        alpha_rate=alpha_rate,
        alpha_accel=alpha_accel,
        chi_accel=chi_accel,
        omega=omega,
        omega_lo=omega_lo,
        omega_hi=omega_hi,
        omega_dot=omega_dot,
        omega_dot_lo=omega_dot_lo,
        omega_dot_hi=omega_dot_hi,
        required_moment=required_moment,
        damping_moment=damping_moment,
        required_force=required_force,
        required_delta=required_delta,
        static_force=static_force,
        static_delta=static_delta,
        weak=weak,
    )


def summary_for_scheme(result, plant_arrays, p):
    weak = result["weak"]
    point_effective = np.abs(result["required_force"]) > plant_arrays["effective_force_cap"]
    point_travel = np.abs(result["required_delta"]) > p.horizontal_fin_limit_rad
    point_force = np.abs(result["required_force"]) > plant_arrays["force_cap"]
    static_effective = np.abs(result["static_force"]) > plant_arrays["effective_force_cap"]
    return dict(
        stencil_points=result["count"],
        polynomial_degree=result["degree"],
        inertia_scale=result["inertia_scale"],
        body_rate_min_rad_s=float(np.min(result["omega"])),
        body_rate_max_rad_s=float(np.max(result["omega"])),
        max_abs_required_net_moment_nm=float(np.max(np.abs(result["required_moment"]))),
        max_abs_pointwise_required_delta_rad=float(np.max(np.abs(result["required_delta"]))),
        max_abs_static_required_delta_rad=float(np.max(np.abs(result["static_delta"]))),
        pointwise_travel_violation_times_s=[
            float(t) for t, failed in zip(OBS["time"], point_travel) if failed
        ],
        pointwise_force_violation_times_s=[
            float(t) for t, failed in zip(OBS["time"], point_force) if failed
        ],
        pointwise_effective_violation_times_s=[
            float(t) for t, failed in zip(OBS["time"], point_effective) if failed
        ],
        static_effective_violation_times_s=[
            float(t) for t, failed in zip(OBS["time"], static_effective) if failed
        ],
        weak_nominal_effective_violation_intervals_s=[
            [row["start_s"], row["end_s"]] for row in weak if row["nominal_effective_violation"]
        ],
        weak_display_rounding_robust_violation_intervals_s=[
            [row["start_s"], row["end_s"]]
            for row in weak if row["display_rounding_robust_violation"]
        ],
        max_abs_weak_equivalent_delta_rad=float(max(abs(row["equivalent_delta_rad"]) for row in weak)),
        max_weak_effective_limit_utilization=float(max(row["effective_limit_utilization"] for row in weak)),
        max_abs_weak_dynamic_minus_static_delta_rad=float(
            max(abs(row["inertial_delta_increment_rad"]) for row in weak)
        ),
        weak_dynamic_static_sign_mismatch_intervals_s=[
            [row["start_s"], row["end_s"]]
            for row in weak
            if row["equivalent_delta_rad"] * row["static_equivalent_delta_rad"] < 0.0
        ],
    )


def write_tsv(path: Path, results, obs, motion, plant_arrays):
    rows = []
    for name, result in results.items():
        for i, t in enumerate(obs["time"]):
            rows.append(dict(
                record_type="point",
                scheme=name,
                time_s=float(t),
                speed_mps=float(obs["speed"][i]),
                total_alpha_deg=float(math.degrees(obs["alpha"][i])),
                horizontal_alpha_deg=float(math.degrees(motion["alpha_h"][i])),
                vertical_alpha_deg=float(math.degrees(motion["alpha_v"][i])),
                total_load_g=float(obs["load"][i]),
                heading_rate_rad_s=float(motion["chi_dot"][i]),
                body_rate_rad_s=float(result["omega"][i]),
                body_rate_lo_rad_s=float(result["omega_lo"][i]),
                body_rate_hi_rad_s=float(result["omega_hi"][i]),
                body_angular_accel_rad_s2=float(result["omega_dot"][i]),
                required_net_moment_nm=float(result["required_moment"][i]),
                body_normal_force_n=float(plant_arrays["normal"][i]),
                damping_moment_nm=float(result["damping_moment"][i]),
                required_control_force_n=float(result["required_force"][i]),
                required_delta_rad=float(result["required_delta"][i]),
                static_required_delta_rad=float(result["static_delta"][i]),
                effective_force_cap_n=float(plant_arrays["effective_force_cap"][i]),
            ))
        for row in result["weak"]:
            rows.append(dict(record_type="weak_interval", scheme=name, **row))
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, results, obs, motion, plant_arrays, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"q3": "#b85f45", "q5": "#277c91", "c5": "#76539b", "c7": "#3f8b62"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    ax.plot(obs["time"], np.degrees(obs["alpha"]), "o-", color="#222", label="Measured total AoA")
    ax.plot(obs["time"], np.degrees(motion["alpha_h"]), "o-", color="#ba6b2e", label="Reconstructed horizontal AoA")
    ax.plot(obs["time"], np.degrees(motion["alpha_v"]), "o-", color="#6f8791", label="Algebraic vertical support")
    ax.set(ylabel="Angle [deg]", title="Conditional AoA decomposition")
    ax.legend(frameon=False, fontsize=8.8)

    ax = axes[0, 1]
    ax.plot(obs["time"], motion["chi_dot"], "o-", color="#222", lw=1.2, label="Trajectory rate from G,V")
    for name, result in results.items():
        ax.plot(obs["time"], result["omega"], color=colors[name], lw=1.4, label=SCHEMES[name][2])
    ax.set(ylabel="Angular rate [rad/s]", title="Body rate = trajectory rate + horizontal-AoA rate")
    ax.legend(frameon=False, fontsize=8.2, ncol=2)

    ax = axes[1, 0]
    moments = np.vstack([result["required_moment"] for result in results.values()])
    ax.fill_between(obs["time"], moments.min(axis=0), moments.max(axis=0), color="#7a6c8d", alpha=0.22,
                    label="Fixed-stencil sensitivity envelope")
    for name, result in results.items():
        ax.plot(obs["time"], result["required_moment"], color=colors[name], lw=1.15)
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set(ylabel="Required net moment [N m]", xlabel="Missile flight time [s]",
           title="Conditional I omega-dot; I = mL²/12")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 1]
    for name, result in results.items():
        centers = np.array([row["center_s"] for row in result["weak"]])
        delta = np.array([row["equivalent_delta_rad"] for row in result["weak"]])
        ax.plot(centers, delta, "o-", ms=3.2, color=colors[name], lw=1.2, label=SCHEMES[name][2])
    effective_delta_cap = plant_arrays["effective_force_cap"] / plant_arrays["force_slope"]
    ax.plot(obs["time"], effective_delta_cap, color="#111", ls="--", lw=1.2, label="Travel/force effective limit")
    ax.plot(obs["time"], -effective_delta_cap, color="#111", ls="--", lw=1.2)
    ax.axhline(p.horizontal_fin_limit_rad, color="#777", ls=":", lw=1, label="Raw travel limit")
    ax.axhline(-p.horizontal_fin_limit_rad, color="#777", ls=":", lw=1)
    ax.set(ylabel="Weak equivalent fin demand [rad]", xlabel="Interval midpoint [s]",
           title="Native-interval moment balance; actual fin is unobserved")
    ax.legend(frameon=False, fontsize=8.0, ncol=2)

    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.15, 3.55)
    fig.suptitle("R-77-1 | controller-independent rotational admissibility audit", fontsize=15.5, y=0.985)
    fig.text(
        0.07, 0.018,
        "Measured V, total AoA and total G only; 6300 m, level positive horizontal turn. No PN/PID/target motion.\n"
        "Current fixed plant: A_delta=2.73404068 m²/rad, ell=0.175 m, D_omega,total=14.435425 m⁴, I=mL²/12.\n"
        "Curves are conditional derivative constructions, not measured body rate/fin angle. Limit admissibility cannot validate the plant.",
        fontsize=8.8, color="#444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.925, bottom=0.13, hspace=0.29, wspace=0.22)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def internal_checks(results, plant_arrays, p):
    # Algebraic moment closure using the conditionally inferred input. This is
    # an implementation identity, not comparison with an observed fin input.
    residual = 0.0
    for result in results.values():
        reconstructed = (
            p.stability_distance_m * result["required_force"]
            - p.stability_distance_m * plant_arrays["normal"]
            - plant_arrays["damping_coefficient"] * result["omega"]
        )
        residual = max(residual, float(np.max(np.abs(reconstructed - result["required_moment"]))))

    # Check the counterintuitive telemetry sign against FocusPlant itself.
    i = 8
    delta = 0.1
    sample = PLANT.sample(
        float(OBS["speed"][i]), float(MOTION["alpha_h"][i]), 0.0,
        float(plant_arrays["qbar"][i]), float(plant_arrays["mass"][i]),
        delta, p, float(OBS["alpha"][i]),
    )
    if not (sample.control_moment_nm > 0.0 and sample.control_force_n < 0.0):
        raise AssertionError("FocusPlant control sign convention changed")

    # Exact integration-by-parts identity on a synthetic linear I and omega.
    t0, t1 = 0.2, 0.9
    i0, i_dot = 200.0, -7.0
    w0, w_dot = 0.3, 0.8
    direct = i0 * w_dot * (t1 - t0) + 0.5 * i_dot * w_dot * (t1 - t0) ** 2
    ib = i0 + i_dot * (t1 - t0)
    wb = w0 + w_dot * (t1 - t0)
    weak = ib * wb - i0 * w0 - (
        i_dot * w0 * (t1 - t0) + 0.5 * i_dot * w_dot * (t1 - t0) ** 2
    )
    weak_residual = abs(direct - weak)
    if residual > 1e-9 or weak_residual > 1e-12:
        raise AssertionError((residual, weak_residual))
    return dict(
        max_algebraic_moment_identity_residual_nm=residual,
        synthetic_weak_integration_identity_residual_nms=weak_residual,
        focusplant_positive_delta_control_moment_nm=float(sample.control_moment_nm),
        focusplant_positive_delta_telemetry_force_n=float(sample.control_force_n),
        interpretation="implementation identities only; not empirical plant validation",
    )


def main():
    before = protected_hashes()
    global REFERENCE, CONFIG, PARAMETERS, PLANT, OBS, RHO, MOTION
    REFERENCE, CONFIG, PARAMETERS, PLANT, OBS = load_current()
    RHO = PLANT.atmo.density_kg_m3
    propulsion = PiecewisePropulsion.from_config(CONFIG)
    MOTION = kinematics_and_bounds(OBS, PARAMETERS, propulsion)
    arrays = nominal_plant_arrays(OBS, PARAMETERS, propulsion, MOTION)

    results = {
        name: scheme_result(name, count, degree, OBS, MOTION, arrays, PARAMETERS)
        for name, (count, degree, _) in SCHEMES.items()
    }
    scheme_summaries = {
        name: summary_for_scheme(result, arrays, PARAMETERS) for name, result in results.items()
    }
    sensitivity = {}
    for scale in (0.5, 1.0, 2.0):
        sensitivity[str(scale)] = {}
        for name, (count, degree, _) in SCHEMES.items():
            candidate = scheme_result(
                name, count, degree, OBS, MOTION, arrays, PARAMETERS, inertia_scale=scale
            )
            sensitivity[str(scale)][name] = dict(
                weak_nominal_effective_violation_count=sum(
                    row["nominal_effective_violation"] for row in candidate["weak"]
                ),
                weak_display_rounding_robust_violation_count=sum(
                    row["display_rounding_robust_violation"] for row in candidate["weak"]
                ),
                max_abs_weak_equivalent_delta_rad=max(
                    abs(row["equivalent_delta_rad"]) for row in candidate["weak"]
                ),
            )

    common_nominal = []
    common_robust = []
    for i in range(len(OBS["time"]) - 1):
        interval = [float(OBS["time"][i]), float(OBS["time"][i + 1])]
        if all(result["weak"][i]["nominal_effective_violation"] for result in results.values()):
            common_nominal.append(interval)
        if all(result["weak"][i]["display_rounding_robust_violation"] for result in results.values()):
            common_robust.append(interval)

    pointwise_delta = np.vstack([result["required_delta"] for result in results.values()])
    weak_delta = np.vstack([
        [row["equivalent_delta_rad"] for row in result["weak"]] for result in results.values()
    ])
    weak_delta_lo = np.vstack([
        [row["equivalent_delta_lo_rad"] for row in result["weak"]] for result in results.values()
    ])
    weak_delta_hi = np.vstack([
        [row["equivalent_delta_hi_rad"] for row in result["weak"]] for result in results.values()
    ])
    checks = internal_checks(results, arrays, PARAMETERS)
    after = protected_hashes()
    if after != before:
        raise AssertionError("Protected inputs changed during the audit")

    prefix = ROOT / "outputs/r77_1_open_loop_rotation"
    write_tsv(prefix.with_suffix(".tsv"), results, OBS, MOTION, arrays)
    plot(prefix.with_suffix(".png"), results, OBS, MOTION, arrays, PARAMETERS)
    output = dict(
        audit="controller-independent rotational admissibility; no PN/PID/target motion",
        source_version=VERSION,
        source_commit=COMMIT,
        current_report=str(ROOT / "outputs/r77_1_target_course_corrected.json"),
        current_report_sha256=before[str(ROOT / "outputs/r77_1_target_course_corrected.json")],
        replay_sha256=before[str(ROOT / "data/replays/r77_1_level_20260824.tsv")],
        fit_window_s=list(WINDOW_S),
        native_frame_count=len(OBS["time"]),
        assumptions=dict(
            altitude_m=ALTITUDE_M,
            level_flight=True,
            positive_horizontal_turn=True,
            total_load_semantics="wind-normal total load including thrust-normal force",
            horizontal_trajectory_rate="chi_dot = g*sqrt(G^2-1)/V",
            horizontal_alpha="sqrt(total_alpha^2 - algebraic_vertical_alpha^2)",
            body_rate="omega = chi_dot + alpha_h_dot",
            inertia="I = m*L^2/12",
            mass="190 - 71*t/7 kg through the audited interval",
            nominal_thrust_n=24350.0,
            actual_fin_deflection_observed=False,
            body_rate_observed=False,
            signed_3d_attitude_observed=False,
        ),
        fixed_plant=dict(
            **REFERENCE["fixed_parameters"],
            damping_m4_total=PLANT.damping_m4,
            moment_equation="I*omega_dot = ell*F_control - ell*N_body - (q/V)*Domega_total*omega",
            control_force="clip(q*A_delta*delta, +/-m*g*finsLatAccel)",
            body_normal_force="q*A_L*alpha_h*cos(alpha_h) + D*sin(alpha_h)",
            translation_has_extra_fin_force=False,
        ),
        provenance_caveat=dict(
            current_confirmed_target_heading_deg=REFERENCE["scenario_heading_deg"]["confirmed_180"],
            historical_fit_target_heading_deg=REFERENCE["scenario_heading_deg"]["old_210"],
            target_heading_absent_from_open_loop_rotation_equation=True,
            A_delta_selected_on_old_closed_loop_replay=True,
            Domega_14p41_selected_on_old_closed_loop_replay=True,
            damping_old_fit_horizontal_fin_limit_rad=0.7238417969430099,
            same_replay_is_not_a_holdout=True,
            static_and_step_in_focus_script_is_synthetic_not_game_response=True,
        ),
        derivative_treatment=dict(
            schemes={name: dict(points=count, degree=degree, label=label)
                     for name, (count, degree, label) in SCHEMES.items()},
            display_rounding_intervals=dict(
                speed_kmh="+/-0.5",
                total_alpha_deg="+/-0.05",
                total_load_g="+/-0.05",
                timestamps="held at reported values; no source establishes a tighter timestamp rounding model",
            ),
            weak_balance=(
                "native-interval integral of I*omega_dot; integration by parts uses endpoint omega "
                "and avoids pointwise second differentiation in the limit test"
            ),
            no_smoothing_parameter_selected_from_results=True,
            early_transients_retained=True,
        ),
        reconstructed_motion=dict(
            horizontal_alpha_deg_range=[
                float(np.degrees(MOTION["alpha_h"]).min()),
                float(np.degrees(MOTION["alpha_h"]).max()),
            ],
            vertical_support_alpha_deg_range=[
                float(np.degrees(MOTION["alpha_v"]).min()),
                float(np.degrees(MOTION["alpha_v"]).max()),
            ],
            trajectory_rate_rad_s_range=[
                float(MOTION["chi_dot"].min()), float(MOTION["chi_dot"].max())
            ],
            body_rate_across_stencils_rad_s_range=[
                float(min(np.min(result["omega"]) for result in results.values())),
                float(max(np.max(result["omega"]) for result in results.values())),
            ],
        ),
        actuator_limits=dict(
            raw_travel_rad=PARAMETERS.horizontal_fin_limit_rad,
            raw_travel_deg=math.degrees(PARAMETERS.horizontal_fin_limit_rad),
            fin_force_limit_g=PARAMETERS.fin_lateral_acceleration_limit_g,
            travel_limited_force_n_range=[
                float(arrays["travel_force"].min()), float(arrays["travel_force"].max())
            ],
            force_cap_n_range=[float(arrays["force_cap"].min()), float(arrays["force_cap"].max())],
            active_effective_limit=[
                "travel" if travel <= force else "force"
                for travel, force in zip(arrays["travel_force"], arrays["force_cap"])
            ],
        ),
        scheme_results=scheme_summaries,
        cross_scheme=dict(
            pointwise_required_delta_rad_envelope=[
                float(pointwise_delta.min()), float(pointwise_delta.max())
            ],
            weak_equivalent_delta_rad_envelope=[float(weak_delta.min()), float(weak_delta.max())],
            weak_display_rounding_delta_rad_envelope=[
                float(weak_delta_lo.min()), float(weak_delta_hi.max())
            ],
            weak_nominal_violation_intervals_common_to_all_schemes_s=common_nominal,
            weak_display_rounding_robust_violation_intervals_common_to_all_schemes_s=common_robust,
        ),
        inertia_scale_sensitivity=sensitivity,
        identifiability=dict(
            physically_necessary_constraints=[
                "measured total AoA must exceed the algebraic vertical-support angle",
                "there must exist a fin-force history within both travel and finsLatAccel limits",
                "its native-interval impulse must cover the weak required control impulse",
            ],
            conditionally_computable=[
                "alpha_h and chi_dot under the level positive-turn assumption",
                "omega under a stated derivative stencil",
                "required moment and fin demand after fixing I, A_delta, ell and Domega",
            ],
            not_separately_identified=[
                "actual fin deflection history",
                "fin effectiveness A_delta",
                "rate damping Domega",
                "inertia I",
                "signed 3-D attitude/body rate",
            ],
            interpretation=(
                "A robust limit violation rejects the fixed plant under the stated reconstruction. "
                "An admissible conditional input only means the fixed equation is not rejected; it is not a pass."
            ),
        ),
        implementation_checks=checks,
        conclusion=dict(
            status="not_identified",
            decisive_fixed_plant_rejection=False,
            reason=(
                "All four fixed stencil families admit a control-force impulse within both raw travel and "
                "finsLatAccel limits on every native interval; the display-rounding intervals also produce "
                "no robust violation. Because fin deflection and body rate were not observed, this is only "
                "failure to reject necessary constraints, not an empirical open-loop pass."
            ),
        ),
        protected_input_hashes_before=before,
        protected_input_hashes_after=after,
        protected_inputs_unchanged=before == after,
        artifacts=[str(prefix.with_suffix(suffix)) for suffix in (".json", ".tsv", ".png")],
    )
    prefix.with_suffix(".json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "common_nominal_weak_violations": common_nominal,
        "common_display_rounding_robust_weak_violations": common_robust,
        "pointwise_delta_envelope_rad": output["cross_scheme"]["pointwise_required_delta_rad_envelope"],
        "weak_delta_envelope_rad": output["cross_scheme"]["weak_equivalent_delta_rad_envelope"],
        "body_rate_range_rad_s": output["reconstructed_motion"]["body_rate_across_stencils_rad_s_range"],
        "protected_inputs_unchanged": before == after,
    }, indent=2))


if __name__ == "__main__":
    main()
