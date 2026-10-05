"""One-parameter damping comparison for the frozen R-77-1 candidate.

Only D_omega in M_extra = -(q/V)*D_omega*omega is varied. These are trial
values, not a fitted coefficient or a claimed interpretation of WdK.
PID, angle mapping, outer guidance, lift and drag remain identical.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from audit_r77_1_257_input_mapping import protected_hashes, turning_points
from check_r77_1_257_plant_constraints import load_observations
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import candidate_inputs, run_alpha_servo, write_table

ROOT = Path(__file__).resolve().parents[1]
TRIALS_M4 = (0.0, 2.0, 4.0, 8.0, 16.0)


def run_trials(plant_dt_s=0.002):
    # The reader enforces the exact 2.57.1.107 commit and raw-source hashes.
    load_observations()
    base_config, base_parameters = candidate_inputs("su_r_77_1")
    results = {}
    for damping in TRIALS_M4:
        config, parameters = candidate_inputs("su_r_77_1", rate_damping_derivative_m4=damping)
        assert config == base_config
        assert {k: v for k, v in asdict(parameters).items() if k != "rate_damping_derivative_m4"} == {
            k: v for k, v in asdict(base_parameters).items() if k != "rate_damping_derivative_m4"}
        result = run_alpha_servo(config, parameters, CASES[0][3], max_time_s=3.5,
                                 plant_dt_s=plant_dt_s)
        assert math.isclose(result.samples[-1]["time_s"], 3.5)
        for row in result.samples:
            assert row["rate_damping_moment_nm"] * row["body_rate_rad_s"] <= 0.0
        results[damping] = result
    return results


def main():
    before = protected_hashes()
    results = run_trials()
    _, observations = load_observations()
    summary = []
    for damping, result in results.items():
        rows = result.samples
        points = turning_points(rows)
        peaks = [point for point in points if point["kind"] == "peak"]
        max_row = max((row for row in rows if row["time_s"] >= 0.8), key=lambda r: r["alpha_deg"])
        times = [row["time_s"] for row in rows]
        summary.append(dict(
            rate_damping_derivative_m4=damping,
            alpha_peaks_after_0p8s=len(peaks),
            largest_alpha_after_0p8s_deg=max_row["alpha_deg"],
            largest_alpha_time_s=max_row["time_s"],
            alpha_at_0p9s_deg=float(np.interp(.9, times, [row["alpha_deg"] for row in rows])),
            alpha_at_1p5s_deg=float(np.interp(1.5, times, [row["alpha_deg"] for row in rows])),
            speed_at_3p5s_kmh=rows[-1]["speed_kmh"],
        ))
        print(summary[-1])

    output = ROOT / "outputs/r77_1_rate_damping"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_table(output.with_name(output.name + "_trajectory.tsv"), [
        dict(rate_damping_derivative_m4=damping, **row)
        for damping, result in results.items() for row in result.samples])
    write_table(output.with_name(output.name + "_summary.tsv"), summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for damping, color in ((0.0, "#9d8da9"), (8.0, "#267d87"), (16.0, "#c57931")):
        rows = results[damping].samples
        times = [row["time_s"] for row in rows]
        for ax, field, scale in zip(axes,
                ("alpha_deg", "wind_normal_specific_force_g", "body_rate_rad_s"),
                (1.0, 1.0, 180 / math.pi)):
            ax.plot(times, [row[field] * scale for row in rows], lw=1.7, color=color,
                    label=f"D = {damping:g} m$^4$" + (" (previous candidate)" if damping == 0 else ""))
    axes[0].plot(observations["time"], np.degrees(observations["alpha"]), "o-",
                 color="#222222", ms=4, lw=1, label="2.57 replay")
    axes[1].plot(observations["time"], observations["load"], "o-", color="#222222", ms=4, lw=1)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set(ylabel="Total AoA [deg]", title="Stronger damping suppresses ringing, but delays the turn")
    axes[1].set(ylabel="Total normal load [g]")
    axes[2].set(ylabel="Body rate [deg/s]", xlabel="Flight time [s]",
                title="Internal model state: no measured body-rate channel")
    axes[2].axhline(0, color="0.5", lw=.7)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, 3.55)
    fig.suptitle("R-77-1 | frozen 2.57.1.107 | independent angular-rate damping", fontsize=15)
    fig.text(.10, .025,
             "Added moment: M = -(q / V) D omega. D is an unidentified model derivative, not game WdK.\n"
             "Only damping varies. Original PID, fin-angle mapping, guidance, CL-alpha = 14/rad and K = 0.0767 retained.\n"
             "Trials D = 0, 2, 4, 8, 16 m^4; representative curves shown. No coefficient selected as a calibrated default.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=.11, right=.98, top=.92, bottom=.13, hspace=.25)
    fig.savefig(output.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)
    assert protected_hashes() == before
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
