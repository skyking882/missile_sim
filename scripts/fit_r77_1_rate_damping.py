"""Calibrate only the added rate damping of the independent-moment candidate.

Use the previously selected 0.2--3.5 s replay window. The objective is the
largest absolute AoA error at its native timestamps, not an average that
can hide an incorrect peak or trough. Load and speed are comparison outputs,
not fitted inputs. All predictions are independent free-flight integrations.
The fitted coefficient remains a model assumption, not recovered game WdK.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_r77_1_257_input_mapping import protected_hashes
from check_r77_1_257_plant_constraints import load_observations, VERSION, COMMIT
from check_r77_1_moment_response import variants
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import run_alpha_servo, write_table

ROOT = Path(__file__).resolve().parents[1]


def main():
    before = protected_hashes()
    _, obs = load_observations()
    config, candidates = variants()
    baseline = candidates["independent_moments"]
    observed_alpha = np.degrees(obs["alpha"])
    observed_peak = float(max(observed_alpha))
    cache = {}

    def evaluate(damping):
        key = round(float(damping), 8)
        if key in cache:
            return cache[key]
        p = replace(baseline, rate_damping_derivative_m4=key)
        assert all(v == asdict(baseline)[k] for k, v in asdict(p).items()
                   if k != "rate_damping_derivative_m4")
        result = run_alpha_servo(config, p, CASES[0][3], max_time_s=3.5)
        rows = result.samples
        assert math.isclose(rows[-1]["time_s"], 3.5)
        assert all(row["rate_damping_moment_nm"] * row["body_rate_rad_s"] <= 0 for row in rows)
        time = np.array([row["time_s"] for row in rows])
        alpha = np.array([row["alpha_deg"] for row in rows])
        predicted = np.interp(obs["time"], time, alpha)
        residual = predicted - observed_alpha
        peak_mask = time >= .5
        peak_index = np.flatnonzero(peak_mask)[np.argmax(alpha[peak_mask])]
        summary = dict(damping_m4=key, worst_alpha_error_deg=float(max(abs(residual))),
            worst_alpha_error_time_s=float(obs["time"][np.argmax(abs(residual))]),
            peak_alpha_deg=float(alpha[peak_index]), peak_time_s=float(time[peak_index]),
            alpha_0p7_deg=float(np.interp(.7, time, alpha)),
            alpha_0p9_deg=float(np.interp(.9, time, alpha)),
            alpha_1p1_deg=float(np.interp(1.1, time, alpha)),
            alpha_1p5_deg=float(np.interp(1.5, time, alpha)),
            alpha_1p9_deg=float(np.interp(1.9, time, alpha)),
            alpha_3p5_deg=float(alpha[-1]),
            final_speed_kmh=float(rows[-1]["speed_kmh"]),
            final_load_g=float(rows[-1]["wind_normal_specific_force_g"]))
        cache[key] = (summary, rows)
        return cache[key]

    coarse = [evaluate(d)[0] for d in np.arange(0., 64.01, .5)]
    coarse_best = min(coarse, key=lambda row: row["worst_alpha_error_deg"])
    print("Coarse minimax:", coarse_best, flush=True)
    # The 20 ms capture switch makes the objective nonsmooth. Refine the two
    # best distinct grid neighborhoods explicitly, without a smooth optimizer.
    centers = []
    for row in sorted(coarse, key=lambda row: row["worst_alpha_error_deg"]):
        center = row["damping_m4"]
        if all(abs(center - prior) > 1. for prior in centers):
            centers.append(center)
        if len(centers) == 2:
            break
    for center in centers:
        for d in np.arange(max(0., center-.5), min(64., center+.5)+.005, .01):
            evaluate(d)
    best = min((value[0] for value in cache.values()), key=lambda row: row["worst_alpha_error_deg"])

    # Separate diagnostic: how much damping matches the observed *maximum*
    # angle alone? Bisect the first downward crossing on the sampled high-
    # damping branch; this is not a fit to the whole time history.
    peak_branch = [row for row in coarse if row["damping_m4"] >= 12.]
    brackets = [(a,b) for a,b in zip(peak_branch, peak_branch[1:])
                if a["peak_alpha_deg"] >= observed_peak >= b["peak_alpha_deg"]]
    peak_fit = None
    if brackets:
        lower, upper = brackets[0][0]["damping_m4"], brackets[0][1]["damping_m4"]
        for _ in range(12):
            midpoint = (lower+upper)/2
            if evaluate(midpoint)[0]["peak_alpha_deg"] > observed_peak:
                lower = midpoint
            else:
                upper = midpoint
        peak_fit = evaluate((lower+upper)/2)[0]
    selected = {"previous_D12": evaluate(12.)[0], "minimax_AoA": best}
    if peak_fit is not None:
        selected["maximum_angle_only"] = peak_fit

    verification = {}
    for name, summary in selected.items():
        d = summary["damping_m4"]
        fine = run_alpha_servo(config, replace(baseline, rate_damping_derivative_m4=d),
                               CASES[0][3], max_time_s=3.5, plant_dt_s=.001).samples
        rows = evaluate(d)[1]
        assert len(rows) == len(fine)
        error = {field: max(abs(a[field]-b[field]) for a,b in zip(rows,fine))
                 for field in ("alpha_deg", "speed_kmh", "wind_normal_specific_force_g")}
        assert error["alpha_deg"] < 1e-3 and error["speed_kmh"] < .01
        verification[name] = error

    prefix = ROOT / "outputs/r77_1_damping_calibration"
    comparison = []
    for name, summary in selected.items():
        rows = evaluate(summary["damping_m4"])[1]
        t = [row["time_s"] for row in rows]
        for i, time in enumerate(obs["time"]):
            comparison.append(dict(variant=name, damping_m4=summary["damping_m4"], time_s=float(time),
                observed_alpha_deg=float(observed_alpha[i]),
                predicted_alpha_deg=float(np.interp(time, t, [row["alpha_deg"] for row in rows])),
                observed_load_g=float(obs["load"][i]),
                predicted_load_g=float(np.interp(time, t, [row["wind_normal_specific_force_g"] for row in rows])),
                observed_speed_kmh=float(obs["speed"][i]*3.6),
                predicted_speed_kmh=float(np.interp(time, t, [row["speed_kmh"] for row in rows]))))
    write_table(prefix.with_suffix(".tsv"), comparison)
    replay = ROOT / "data/replays/r77_1_level_20260824.tsv"
    report = dict(source_version=VERSION, source_commit=COMMIT,
        replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
        fit_window_s=[.2,3.5], fitted_variable="rate_damping_derivative_m4 only",
        objective="minimize maximum absolute AoA error at all 19 native replay timestamps",
        search_interval_m4=[0.,64.], coarse_spacing_m4=.5, local_spacing_m4=.01,
        frozen_parameters=asdict(baseline), selected=selected,
        observations_are_calibration_data_not_holdout=True,
        damping_is_assumed_not_game_parameter=True, baseline_default_changed=False,
        coarse_scan=coarse, comparisons=comparison, verification=verification)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    styles = {"previous_D12": ("#83969b", "Previous"),
              "minimax_AoA": ("#167d9a", "Smallest worst AoA error"),
              "maximum_angle_only": ("#c38331", "Match maximum angle only")}
    for name, summary in selected.items():
        color, label = styles[name]
        rows = evaluate(summary["damping_m4"])[1]
        for ax, field in zip(axes[:2], ("alpha_deg", "wind_normal_specific_force_g")):
            ax.plot([row["time_s"] for row in rows], [row[field] for row in rows], color=color,
                    lw=1.8, label=f"{label}: D={summary['damping_m4']:.2f}")
    axes[0].plot(obs["time"], observed_alpha, "o-", color="#222222", ms=4, lw=1.3, label="2.57 game replay")
    axes[1].plot(obs["time"], obs["load"], "o-", color="#222222", ms=4, lw=1.3)
    axes[0].set(ylabel="Total AoA [deg]", title="Only the constant added damping is fitted")
    axes[1].set(ylabel="Total normal load [g]", xlabel="Flight time [s]")
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[2].plot([r["damping_m4"] for r in coarse], [r["worst_alpha_error_deg"] for r in coarse], color="#167d9a", lw=1.6)
    for name, summary in selected.items():
        color, _ = styles[name]
        axes[2].plot(summary["damping_m4"], summary["worst_alpha_error_deg"], "o", color=color)
    axes[2].set(xlabel="Added damping D [m^4]", ylabel="Worst sampled AoA error [deg]",
                title="Whole-window error: a peak match does not match the transient")
    for ax in axes: ax.grid(alpha=.2)
    for ax in axes[:2]: ax.set_xlim(.2,3.55)
    axes[2].set_xlim(0.,64.)
    fig.suptitle("R-77-1 | one-coefficient damping calibration | frozen 2.57.1.107", fontsize=14)
    fig.text(.11,.022,"Calibration window: 0.2-3.5 s, 19 native observations; no time shift or display scaling.\n"
             "Control/restoring moment slopes, raw PID, guidance, lift and drag all fixed. Speed is integrated freely.\n"
             "D is a self-defined model parameter. This replay was used for calibration, not independent validation.",
             fontsize=9,color="#444444")
    fig.subplots_adjust(left=.11,right=.98,top=.92,bottom=.11,hspace=.4)
    fig.savefig(prefix.with_suffix(".png"),dpi=160,facecolor="white")
    plt.close(fig)
    assert protected_hashes() == before
    print(json.dumps(selected,indent=2),flush=True)
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
