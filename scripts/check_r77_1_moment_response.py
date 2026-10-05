"""Compare equal-slope and independent moment derivatives with the same PID.

The independent candidate was screened on this replay. It is a structural
diagnostic, not a held-out validation or identification of game derivatives.
Free trajectories never read replay states. Separate frozen small-angle
analyses use common observed speed/mass points and have no guidance dynamics.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from audit_r77_1_257_input_mapping import protected_hashes
from check_r77_1_257_plant_constraints import load_observations
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import candidate_inputs, run_alpha_servo, write_table

ROOT = Path(__file__).resolve().parents[1]


def variants():
    config, original = candidate_inputs("su_r_77_1")
    area = original.fin_force_area_per_rad
    candidates = {"original": original}
    for name, control, restoring, damping in (
            ("damping_only", area, area, 16.),
            ("independent_moments", 4 * area, 2 * area, 12.)):
        other_config, p = candidate_inputs("su_r_77_1", fin_force_area_per_rad=control,
            restoring_force_area_per_rad=restoring, rate_damping_derivative_m4=damping)
        assert other_config == config
        allowed = {"fin_force_area_per_rad", "restoring_force_area_per_rad", "rate_damping_derivative_m4"}
        assert all(value == asdict(original)[key] for key, value in asdict(p).items() if key not in allowed)
        candidates[name] = p
    return config, candidates


def frozen_matrices(p, speed, mass, thrust, rho):
    """Small-angle wind-normal aggregate force, unsaturated moment and PID."""
    pressure = .5 * rho * speed**2
    inertia = mass * p.length_m**2 / 12
    normal_per_rad = (pressure * p.lift_area_per_rad + thrust) / mass
    b = normal_per_rad / speed
    ka = pressure * p.restoring_area_per_rad * p.stability_distance_m / inertia
    ku = pressure * p.fin_force_area_per_rad * p.stability_distance_m / inertia
    c = pressure / (speed * inertia) * (
        p.restoring_area_per_rad * p.stability_distance_m**2 + p.rate_damping_derivative_m4)
    matrix = np.array([[-b, 1.], [-ka, -c]])
    input_vector = np.array([0., ku])
    # Exact held-input discretization; match the existing integral-before-output
    # update and unfiltered backward-difference derivative, period 20 ms.
    dt = .02
    values, vectors = np.linalg.eig(matrix)
    transition = np.real_if_close(vectors @ np.diag(np.exp(values * dt)) @ np.linalg.inv(vectors)).real
    held_input = np.linalg.solve(matrix, (transition - np.eye(2)) @ input_vector)
    h = normal_per_rad / 9.80665
    output = p.horizontal_fin_limit_rad * np.array([
        -h * (p.acceleration_pid_kp + p.acceleration_pid_ki * dt + p.acceleration_pid_kd / dt),
        0., 1., -p.acceleration_pid_kd / dt])
    # State = alpha, omega, accumulated integral output, previous g error.
    closed = np.zeros((4, 4))
    closed[:2, :2] = transition
    closed[:2, :] += np.outer(held_input, output)
    closed[2] = [-p.acceleration_pid_ki * dt * h, 0., 1., 0.]
    closed[3] = [-h, 0., 0., 0.]
    return matrix, input_vector, closed


def describe(rows):
    times = np.array([row["time_s"] for row in rows])
    alpha = np.array([row["alpha_deg"] for row in rows])
    peaks = [i for i in range(1, len(alpha)-1)
             if times[i] >= .7 and alpha[i] > alpha[i-1] and alpha[i] >= alpha[i+1]]
    swings = [min(alpha[left], alpha[right]) - np.min(alpha[left:right+1])
              for left, right in zip(peaks, peaks[1:])]
    return dict(alpha_0p9_deg=float(np.interp(.9, times, alpha)),
                alpha_1p5_deg=float(np.interp(1.5, times, alpha)),
                alpha_3p5_deg=float(alpha[-1]),
                repeated_swing_deg=float(max(swings, default=0.)),
                final_speed_kmh=rows[-1]["speed_kmh"],
                final_load_g=rows[-1]["wind_normal_specific_force_g"],
                control_force_saturated_samples=sum(row["fin_control_force_saturated"] for row in rows))


def main():
    before = protected_hashes()
    _, observations = load_observations()
    config, candidates = variants()
    results = {name: run_alpha_servo(config, p, CASES[0][3], max_time_s=3.5)
               for name, p in candidates.items()}
    summary = {}
    for name, result in results.items():
        assert math.isclose(result.samples[-1]["time_s"], 3.5)
        assert all(row["rate_damping_moment_nm"] * row["body_rate_rad_s"] <= 0 for row in result.samples)
        summary[name] = dict(parameters=asdict(candidates[name]), **describe(result.samples))
        pole_rows = []
        for time in [.5, .9, 1.5, 2.5, 3.5]:
            v, m, thrust = [float(np.interp(time, observations["time"], observations[field]))
                            for field in ("speed", "mass", "thrust")]
            matrix, _, closed = frozen_matrices(candidates[name], v, m, thrust, observations["rho"])
            poles = np.linalg.eigvals(matrix)
            pole_rows.append(dict(time_s=time, plant_poles=[[float(z.real), float(z.imag)] for z in poles],
                                  closed_loop_spectral_radius=float(max(abs(np.linalg.eigvals(closed))))))
        summary[name]["frozen_small_angle_checks"] = pole_rows
    # A single final timestep check on all three presented cases.
    for name, p in candidates.items():
        fine = run_alpha_servo(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
        rows = results[name].samples
        assert len(rows) == len(fine.samples)
        differences = {field: max(abs(a[field]-b[field]) for a, b in zip(rows, fine.samples))
                       for field in ("alpha_deg", "speed_kmh", "wind_normal_specific_force_g")}
        assert differences["alpha_deg"] < 1e-3 and differences["speed_kmh"] < .01
        summary[name]["half_step_max_absolute_difference"] = differences
    prefix = ROOT / "outputs/r77_1_independent_moments"
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(prefix.with_suffix(".tsv"), [dict(variant=name, **row)
                for name, result in results.items() for row in result.samples])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    styles = {"original": ("#b95765", "Original: equal moment slopes"),
              "damping_only": ("#b18a38", "Damping only"),
              "independent_moments": ("#17888e", "Independent moment slopes + damping")}
    index = int(np.argmin(abs(observations["time"] - 1.5)))
    step_times = np.linspace(0., 2.5, 501)
    for name, result in results.items():
        color, label = styles[name]
        for ax, field in zip(axes[:2], ("alpha_deg", "wind_normal_specific_force_g")):
            ax.plot([row["time_s"] for row in result.samples], [row[field] for row in result.samples],
                    color=color, lw=1.8, label=label)
        if name == "damping_only":
            continue
        matrix, inputs, _ = frozen_matrices(candidates[name], observations["speed"][index],
            observations["mass"][index], observations["thrust"][index], observations["rho"])
        steady = -np.linalg.solve(matrix, inputs * math.radians(2.))
        eigen, vectors = np.linalg.eig(matrix)
        transient = np.linalg.solve(vectors, -steady)
        step_alpha = [math.degrees(float(np.real(steady[0] + (vectors @ (np.exp(eigen*t) * transient))[0])))
                      for t in step_times]
        axes[2].plot(step_times, step_alpha, color=color, lw=1.8, label=label)
        axes[2].axhline(math.degrees(steady[0]), color=color, ls=":", lw=.9)
    axes[0].plot(observations["time"], np.degrees(observations["alpha"]), "o-", color="#222222", ms=4, lw=1, label="Game replay, 2.57")
    axes[1].plot(observations["time"], observations["load"], "o-", color="#222222", ms=4, lw=1)
    axes[0].set(ylabel="Total AoA [deg]", title="Free flight: unchanged PID, guidance, lift and drag")
    axes[1].set(ylabel="Total normal load [g]", xlabel="Flight time [s]")
    axes[2].set(ylabel="Horizontal AoA [deg]", xlabel="Time after fin step [s]",
                title="Plant alone: 2-degree fin step, fixed V/m/T at replay 1.5 s (linearized)")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[2].legend(fontsize=9, loc="lower right")
    for ax in axes: ax.grid(alpha=.2)
    for ax in axes[:2]: ax.set_xlim(.2, 3.55)
    axes[2].set_xlim(0., 2.5)
    fig.suptitle("R-77-1: decouple control effectiveness, static stability and damping", fontsize=14)
    fig.text(.11, .022, "Screened on the same replay; not independent validation or a recovered game formula.\n"
             "Candidate: control slope x4, restoring slope x2, D=12 m^4. Baseline PID and 2.57 source unchanged.\n"
             "Large repeated oscillations are suppressed; the capture handover and remaining trajectory mismatch persist.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=.11, right=.98, top=.92, bottom=.11, hspace=.4)
    fig.savefig(prefix.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)
    assert protected_hashes() == before
    for name, item in summary.items():
        print(name, json.dumps({key: value for key, value in item.items()
                                if key not in ("parameters", "frozen_small_angle_checks")}))
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
