"""Conditional inner-plant consistency check for the rebuild model.

This script deliberately bypasses PN, PID, launch capture and fin history.  It
uses only the R-77-1 level replay's measured speed, overload and angle of
attack, together with the rebuild profile's fixed datamine-backed geometry and
propulsion.  The replay contains no signed normal-force direction, body angle,
body rate or fin angle, so the numerical conclusions are conditional:

* a signed early horizontal normal-force direction is evaluated both ways;
* ``overload_g`` is evaluated both as trajectory-normal non-gravity specific
  force and as a total load with one g of level-flight support removed;
* the hidden fin angle is inferred from the force equation only, never fitted;
* one nuisance initial body-rate value is estimated for the weak attitude
  compatibility check.  No aerodynamic coefficient is fitted.

For the shipped rebuild law, the single-plane equations are

    F_N = K_F(t) (A alpha + B delta)
    I omega_dot = K_M(t) (C alpha + D delta),  I = m(t) L^2 / 12.

Eliminating the unobserved delta gives

    I omega_dot = K_M[(C - D A/B) alpha + (D/B) F_N/K_F].

The script checks that this identity and the implemented q*S scales agree,
then asks whether the replay force proxy admits a finite hidden fin angle and
whether the same moment law is compatible with the weak attitude integral.
It cannot identify A, B, C and D separately from this replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "data" / "replays" / "r77_1_level_20260824.tsv"
ALTITUDE_M = 6300.0
WINDOW_S = (0.5, 2.3)
G0 = 9.80665

import sys

sys.path.insert(0, str(ROOT / "src"))

from aim120_model.atmosphere import StandardAtmosphere  # noqa: E402
from aim120_model.dynamics import SimState  # noqa: E402
from aim120_model.h2_dynamics import forces_for_state_h2  # noqa: E402
from aim120_model.profile_adapter import (  # noqa: E402
    build_h2_candidate_config,
    load_runtime_defaults,
)
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402


def load_replay_window() -> dict[str, np.ndarray]:
    with REPLAY_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    time = np.array([float(row["missile_flight_time_s"]) for row in rows], dtype=float)
    mask = (time >= WINDOW_S[0]) & (time <= WINDOW_S[1])
    if int(mask.sum()) < 3:
        raise AssertionError("R-77-1 replay does not contain enough early-window points")
    return {
        "time_s": time[mask],
        "speed_mps": np.array([float(row["speed_kmh"]) / 3.6 for row in rows], dtype=float)[mask],
        "overload_g": np.array([float(row["overload_g"]) for row in rows], dtype=float)[mask],
        "alpha_rad": np.radians(
            np.array([float(row["angle_of_attack_deg"]) for row in rows], dtype=float)[mask]
        ),
    }


def load_rebuild() -> tuple[dict[str, Any], PiecewisePropulsion]:
    profile = json.loads((ROOT / "missiles" / "su_r_77_1.json").read_text(encoding="utf-8"))
    defaults = load_runtime_defaults(str(ROOT / "config" / "profile_h2_rebuild_defaults.json"))
    config, _ = build_h2_candidate_config(profile, defaults)
    if config["control"]["plant_semantics"] != "rebuild_first_principles":
        raise AssertionError("the consistency check must use the rebuild profile")
    return config, PiecewisePropulsion.from_config(config)


def nominal_gains(config: dict[str, Any]) -> tuple[float, float, float, float]:
    normalized = config["aerodynamics"]["rebuild_force_moment_candidate"]
    gains = tuple(
        float(normalized[key])
        for key in ("alpha_force_gain", "delta_force_gain", "alpha_moment_gain", "delta_moment_gain")
    )
    expected = (2.0, -1.0, -1.0, 1.0)
    if not np.allclose(gains, expected, rtol=0.0, atol=1.0e-14):
        raise AssertionError(f"unexpected rebuild nominal gains: {gains}")
    return gains  # type: ignore[return-value]


def scales_at(
    config: dict[str, Any],
    altitude_m: float,
    speed_mps: float,
) -> tuple[float, float, float, float]:
    """Return K_F, K_M, q and the shared reference arm from resolved config."""

    candidate = config["aerodynamics"]["generalized_aero_moment_candidate"]
    normalization = candidate["normalization"]
    atmosphere = StandardAtmosphere().sample(altitude_m)
    q = 0.5 * atmosphere.density_kg_m3 * speed_mps * speed_mps
    area = float(normalization["reference_area_m2"])
    reference_angle = float(normalization["reference_angle_rad"])
    q_base = float(normalization["q_base_pa"])
    authority_force = float(normalization["authority_force_n_at_reference"])
    force_scale = authority_force / max(q_base * area, 1.0e-12)
    k_force = q * area * force_scale / reference_angle
    arm = float(config["aerodynamics"]["distance_cm_to_stabilizer_m"])
    k_moment = k_force * arm
    return k_force, k_moment, q, arm


def state_for(
    config: dict[str, Any], mass: float, speed_mps: float, alpha_rad: float, delta_rad: float
) -> SimState:
    return SimState(
        position=(0.0, ALTITUDE_M, 0.0),
        velocity=(float(speed_mps), 0.0, 0.0),
        pitch=float(alpha_rad),
        yaw=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        mass=float(mass),
        actual_pitch_fin_angle_rad=float(delta_rad),
    )


def assert_implementation_scales(
    config: dict[str, Any], propulsion: PiecewisePropulsion, replay: dict[str, np.ndarray]
) -> None:
    """Compare finite differences of the actual plant to its declared scales."""

    t = 1.5
    speed = float(np.interp(t, replay["time_s"], replay["speed_mps"]))
    prop = propulsion.sample(t, powered=True)
    h = 1.0e-7

    def diagnostic(alpha: float, delta: float):
        return forces_for_state_h2(
            state_for(config, prop.mass_kg, speed, alpha, delta),
            t,
            config,
            propulsion,
            powered=True,
        )

    alpha_plus = diagnostic(h, 0.0)
    alpha_minus = diagnostic(-h, 0.0)
    delta_plus = diagnostic(0.0, h)
    delta_minus = diagnostic(0.0, -h)
    slopes = np.array(
        [
            (alpha_plus.pitch_body_normal_force_n - alpha_minus.pitch_body_normal_force_n) / (2.0 * h),
            (delta_plus.pitch_tail_force_n - delta_minus.pitch_tail_force_n) / (2.0 * h),
            (alpha_plus.pitch_body_static_moment_nm - alpha_minus.pitch_body_static_moment_nm) / (2.0 * h),
            (delta_plus.pitch_tail_moment_nm - delta_minus.pitch_tail_moment_nm) / (2.0 * h),
        ]
    )
    k_force, k_moment, q, arm = scales_at(config, ALTITUDE_M, speed)
    a, b, c, d = nominal_gains(config)
    expected = np.array([k_force * a, k_force * b, k_moment * c, k_moment * d])
    relative_error = np.max(np.abs((slopes - expected) / np.maximum(np.abs(expected), 1.0)))
    if relative_error > 2.0e-6:
        raise AssertionError(
            f"rebuild finite-difference scale mismatch: slopes={slopes}, expected={expected}"
        )
    if not math.isclose(k_moment / k_force, arm, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AssertionError("implemented K_M/K_F is not the resolved distFromCmToStab arm")
    print(
        "IMPLEMENTATION"
        f" t={t:.1f}s speed={speed * 3.6:.1f}km/h q={q:.1f}Pa"
        f" KF={k_force:.3f}N/rad KM={k_moment:.3f}Nm/rad"
        f" KM_over_KF={k_moment / k_force:.6f}m"
        f" max_relative_error={relative_error:.3e}"
    )
    combo_alpha = c - d * a / b
    combo_force = d / b
    for alpha, delta in ((0.03, 0.02), (-0.04, 0.01), (0.01, -0.03)):
        force = k_force * (a * alpha + b * delta)
        direct_moment = k_moment * (c * alpha + d * delta)
        eliminated_moment = k_moment * (
            combo_alpha * alpha + combo_force * force / k_force
        )
        if not math.isclose(direct_moment, eliminated_moment, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise AssertionError("delta-eliminated moment identity disagrees with direct law")
    print(
        "ELIMINATED_MOMENT"
        f" C-D*A/B={combo_alpha:.6g} D/B={combo_force:.6g}"
        " (only these combinations enter after delta elimination)"
    )


def weak_attitude_metrics(
    time_s: np.ndarray,
    speed_mps: np.ndarray,
    alpha_rad: np.ndarray,
    normal_accel_mps2: np.ndarray,
    angular_accel_rad_s2: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Integrate flight-path and body-angle changes, estimating only omega_0."""

    dt = np.diff(time_s)
    normal_rate = normal_accel_mps2 / np.maximum(speed_mps, 1.0)
    gamma_delta = np.cumsum(0.5 * (normal_rate[:-1] + normal_rate[1:]) * dt)
    observed_body_angle_delta = gamma_delta + alpha_rad[1:] - alpha_rad[0]

    forced_body_angle_delta = np.zeros_like(observed_body_angle_delta)
    omega = 0.0
    angle = 0.0
    for index, step in enumerate(dt):
        acceleration = 0.5 * (angular_accel_rad_s2[index] + angular_accel_rad_s2[index + 1])
        angle += omega * step + 0.5 * acceleration * step * step
        omega += acceleration * step
        forced_body_angle_delta[index] = angle
    elapsed = time_s[1:] - time_s[0]
    nuisance_omega_0 = float(
        np.dot(elapsed, observed_body_angle_delta - forced_body_angle_delta)
        / max(np.dot(elapsed, elapsed), 1.0e-12)
    )
    residual = nuisance_omega_0 * elapsed + forced_body_angle_delta - observed_body_angle_delta
    return (
        nuisance_omega_0,
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(np.abs(residual))),
        residual,
    )


def evaluate_branch(
    config: dict[str, Any],
    propulsion: PiecewisePropulsion,
    replay: dict[str, np.ndarray],
    load_mode: str,
    sign: int,
) -> dict[str, Any]:
    a, b, c, d = nominal_gains(config)
    gravity = float(config["atmosphere"]["gravity_mps2"])
    fin_limit = math.radians(float(config["aerodynamics"]["horizontal_fin_aoa_limit_deg"]))
    times = replay["time_s"]
    speeds = replay["speed_mps"]
    alphas = replay["alpha_rad"]
    loads = replay["overload_g"]
    mass = np.array([propulsion.sample(float(t), powered=True).mass_kg for t in times])
    thrust = np.array([propulsion.sample(float(t), powered=True).thrust_n for t in times])
    if load_mode == "displayed_G":
        normal_g = loads.copy()
    elif load_mode == "support_removed_G":
        normal_g = np.sqrt(np.maximum(loads * loads - 1.0, 0.0))
    else:
        raise ValueError(load_mode)
    signed_normal_accel = float(sign) * normal_g * gravity
    thrust_normal = thrust * np.sin(alphas)
    aero_force = mass * signed_normal_accel - thrust_normal
    k_force = np.empty_like(times)
    k_moment = np.empty_like(times)
    for index, speed in enumerate(speeds):
        k_force[index], k_moment[index], _q, _arm = scales_at(config, ALTITUDE_M, float(speed))
    hidden_delta = (aero_force / k_force - a * alphas) / b
    moment = k_moment * (c * alphas + d * hidden_delta)
    inertia = mass * float(config["geometry"]["length_m"]) ** 2 / 12.0
    angular_accel = moment / inertia
    omega_0, attitude_rmse, attitude_max, attitude_residual = weak_attitude_metrics(
        times, speeds, alphas, signed_normal_accel, angular_accel
    )
    return {
        "load_mode": load_mode,
        "sign": sign,
        "normal_g_min": float(np.min(normal_g)),
        "normal_g_max": float(np.max(normal_g)),
        "delta_min_deg": float(np.degrees(np.min(hidden_delta))),
        "delta_max_deg": float(np.degrees(np.max(hidden_delta))),
        "delta_max_abs_deg": float(np.degrees(np.max(np.abs(hidden_delta)))),
        "fin_limit_deg": math.degrees(fin_limit),
        "fin_feasible": bool(np.max(np.abs(hidden_delta)) <= fin_limit + 1.0e-12),
        "moment_min_nm": float(np.min(moment)),
        "moment_max_nm": float(np.max(moment)),
        "omega0_nuisance_rad_s": omega_0,
        "attitude_rmse_deg": math.degrees(attitude_rmse),
        "attitude_max_abs_deg": math.degrees(attitude_max),
        "hidden_delta_rad": hidden_delta,
        "attitude_residual_rad": attitude_residual,
    }


def plot_results(
    path: Path,
    replay: dict[str, np.ndarray],
    branches: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"displayed_G": "#2f6dae", "support_removed_G": "#c57936"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    time = replay["time_s"]
    for branch in branches:
        label = f'{branch["load_mode"]}, sign={branch["sign"]:+d}'
        color = colors[branch["load_mode"]]
        style = "-" if branch["sign"] > 0 else "--"
        axes[0].plot(time, np.degrees(branch["hidden_delta_rad"]), color=color, ls=style, label=label)
    limit = branches[0]["fin_limit_deg"]
    axes[0].axhline(limit, color="#555555", lw=0.8, ls=":")
    axes[0].axhline(-limit, color="#555555", lw=0.8, ls=":")
    axes[0].set_ylabel("Inferred fin angle (deg)")
    axes[0].set_title("R-77-1 early inner-plant force inversion (conditional branches)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for branch in branches:
        label = f'{branch["load_mode"]}, sign={branch["sign"]:+d}'
        color = colors[branch["load_mode"]]
        style = "-" if branch["sign"] > 0 else "--"
        residual = branch["attitude_residual_rad"]
        axes[1].plot(time[1:], np.degrees(residual), color=color, ls=style, label=label)
    axes[1].axhline(0.0, color="#555555", lw=0.8)
    axes[1].set_xlabel("Missile flight time (s)")
    axes[1].set_ylabel("Angle residual after fitted $\\omega_0$ (deg)")
    for axis in axes:
        axis.grid(axis="y", color="#e7e7e7", lw=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.text(
        0.08,
        0.02,
        "No PN/PID/capture or fin history.  Direction, 1g support convention and vertical component are unresolved.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12, hspace=0.28)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="optional diagnostic PNG path")
    args = parser.parse_args()

    config, propulsion = load_rebuild()
    replay = load_replay_window()
    a, b, c, d = nominal_gains(config)
    assert_implementation_scales(config, propulsion, replay)
    print(
        "REPLAY"
        f" source={REPLAY_PATH.name} points={replay['time_s'].size}"
        f" window={replay['time_s'][0]:.1f}-{replay['time_s'][-1]:.1f}s"
        f" alpha={math.degrees(np.min(replay['alpha_rad'])):.1f}-"
        f"{math.degrees(np.max(replay['alpha_rad'])):.1f}deg"
        f" overload={np.min(replay['overload_g']):.1f}-{np.max(replay['overload_g']):.1f}g"
    )
    print(
        "ASSUMPTIONS"
        " early horizontal single-plane sign branches +/-;"
        " overload branches displayed_G and sqrt(G^2-1) support_removed_G;"
        " alpha is treated as signed positive pitch-plane AoA;"
        " only initial body rate is a nuisance fit"
    )

    branches: list[dict[str, Any]] = []
    for load_mode in ("displayed_G", "support_removed_G"):
        for sign in (1, -1):
            branch = evaluate_branch(config, propulsion, replay, load_mode, sign)
            branches.append(branch)
            print(
                "BRANCH"
                f" load={load_mode} sign={sign:+d}"
                f" hidden_delta={branch['delta_min_deg']:+.2f}..{branch['delta_max_deg']:+.2f}deg"
                f" max_abs={branch['delta_max_abs_deg']:.2f}deg"
                f" limit={branch['fin_limit_deg']:.2f}deg"
                f" force_feasible={branch['fin_feasible']}"
                f" moment={branch['moment_min_nm']:+.0f}..{branch['moment_max_nm']:+.0f}Nm"
                f" omega0={branch['omega0_nuisance_rad_s']:+.4f}rad/s"
                f" attitude_rmse={branch['attitude_rmse_deg']:.2f}deg"
                f" attitude_max={branch['attitude_max_abs_deg']:.2f}deg"
            )
    feasible = [
        f'{branch["load_mode"]}/sign={branch["sign"]:+d}'
        for branch in branches
        if branch["fin_feasible"]
    ]
    print(
        "FORCE_NECESSARY_CONDITION"
        f" {'PASS' if feasible else 'FAIL'}"
        f" branches={', '.join(feasible) if feasible else 'none'}"
        " (conditional on load convention and early turn sign)"
    )
    print(
        "IDENTIFIABILITY"
        f" A={a:g} B={b:g} C={c:g} D={d:g};"
        " force inversion constrains delta for fixed gains,"
        " while angle-only data constrain at most the eliminated combinations"
        " C-D*A/B and D/B plus the assumed inertia ratio."
    )
    if args.figure:
        plot_results(args.figure, replay, branches)
        print(f"FIGURE {args.figure}")


if __name__ == "__main__":
    main()
