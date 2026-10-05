"""Isolate the fin-input scale in the frozen 2.57 R-77-1 candidate.

The historical AIM-54 value 0.436332 is approximately 25 degrees in
radians, matching Gaijin's stated trim angle. This motivates testing raw
radians against the historical raw-value-times-pi/2 mapping. It does NOT
establish that the parameter is a physical fin angle or identify the
game's PID output units. Both runs retain the existing assumed PID
interface, outer guidance, force law, moment closure and integration.

Only this diagnostic reconstructs the historical input mapping. No runtime
default is changed, and no parameters are fitted. Replay observations are
used after the free integrations, except for the independent force check.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.alpha_servo import run_alpha_servo
from check_r77_1_257_plant_constraints import COMMIT, VERSION, load_observations
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import candidate_inputs, write_table


def protected_hashes():
    paths = [ROOT / name for name in (
        "src/aim120_model/alpha_servo.py",
        "scripts/run_alpha_servo_candidate.py",
        "config/profile_h2_rebuild_defaults.json",
        "missiles/su_r_77_1.json",
        "data/replays/r77_1_level_20260824.tsv",
    )]
    paths += sorted((ROOT / "src/aim120_model").glob("*.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_comparison(plant_dt_s=0.002):
    # Validate the exact historical source, not merely the 2.57 prefix.
    profile = json.loads((ROOT / "missiles/su_r_77_1.json").read_text())
    provenance = profile["provenance"]
    assert (provenance["source_version"], provenance["source_commit"]) == (VERSION, COMMIT)
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((ROOT / "data/datamine" / name).read_bytes()).hexdigest() == digest
    config, alternate = candidate_inputs("su_r_77_1")
    raw = json.loads((ROOT / "data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx").read_text())["rocket"]
    original = replace(alternate, horizontal_fin_limit_rad=raw["finsAoaHor"] * math.pi / 2,
                       vertical_fin_limit_rad=raw["finsAoaVer"] * math.pi / 2)
    variants = {"historical_pi_over_2": original, "raw_radians_hypothesis": alternate}
    results = {}
    for name, parameters in variants.items():
        assert (parameters.acceleration_pid_kp, parameters.acceleration_pid_ki,
                parameters.acceleration_pid_kd, parameters.acceleration_pid_integral_limit) == (
                    0.0071, 0.005, 0.00015, 1.0)
        results[name] = run_alpha_servo(config, parameters, CASES[0][3],
                                        max_time_s=3.5, plant_dt_s=plant_dt_s)
        assert abs(results[name].samples[-1]["time_s"] - 3.5) < 1e-10
    return variants, results


def turning_points(samples):
    points = []
    # Ignore tiny displayed total-AoA turns around a horizontal sign change.
    # Keep all extrema after 0.8 s; no smoothing or replay fitting is used.
    for before, row, after in zip(samples, samples[1:], samples[2:]):
        if row["time_s"] < 0.8:
            continue
        a, b, c = (r["alpha_deg"] for r in (before, row, after))
        kind = "peak" if a < b > c else "valley" if a > b < c else None
        if kind:
            points.append(dict(kind=kind, time_s=row["time_s"], alpha_deg=b,
                               load_g=row["wind_normal_specific_force_g"]))
    return points


def main():
    before = protected_hashes()
    variants, results = run_comparison()
    _, observations = load_observations()
    output = ROOT / "outputs/r77_1_257_input_mapping"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(variant=name, **sample)
            for name, result in results.items() for sample in result.samples]
    write_table(output.with_suffix(".tsv"), rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = ("#7b6195", "#247b83")
    labels = ("Historical input scale: 41.47 deg", "Raw-radian input scale: 26.40 deg")
    for (name, result), color, label in zip(results.items(), colors, labels):
        times = [s["time_s"] for s in result.samples]
        axes[0].plot(times, [s["alpha_deg"] for s in result.samples],
                     color=color, lw=1.7, label=label)
        axes[1].plot(times, [s["wind_normal_specific_force_g"] for s in result.samples],
                     color=color, lw=1.7)
        print(name, "input scale [deg]", math.degrees(variants[name].horizontal_fin_limit_rad))
        print(json.dumps(turning_points(result.samples), indent=2))
        print("final speed [km/h]", result.samples[-1]["speed_kmh"])
    axes[0].plot(observations["time"], np.degrees(observations["alpha"]), "o-",
                 color="#222222", ms=4, lw=1, label="2.57 replay")
    axes[1].plot(observations["time"], observations["load"], "o-",
                 color="#222222", ms=4, lw=1)
    axes[0].set(ylabel="Total AoA [deg]", title="Free flight: same PID and force/moment equations; only input scale differs")
    axes[1].set(ylabel="Total normal load [g]")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[2].plot(observations["time"], observations["baseline_load"] - observations["load"],
                 "o-", color="#bc6536", ms=4, lw=1.5)
    axes[2].axhline(0, color="0.4", lw=0.8)
    axes[2].set(ylabel="Predicted minus replay [g]", xlabel="Flight time [s]",
                title="Separate force check at observed V and AoA: 14/rad + nominal thrust; no PID")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 3.55)
    fig.suptitle("R-77-1 | frozen 2.57.1.107 | input-mapping audit", fontsize=15)
    fig.text(0.10, 0.025,
             "No refit: CL-alpha = 14/rad, K = 0.0767, original PID coefficients and algorithm.\n"
             "The alternative changes actuator gain; its physical interpretation remains unverified.\n"
             "Shared assumptions: fixed altitude, 1 g vertical trim, 11 deg initial horizontal AoA, 0.5 s guidance delay.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.92, hspace=0.27)
    fig.savefig(output.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)
    assert protected_hashes() == before, "Protected model/input files changed"
    print("Exact source version, commit and checksums verified; protected files unchanged.")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
