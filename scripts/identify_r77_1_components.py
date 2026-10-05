"""Minimal shared-force identification from the existing game replays.

One fitted constant is shared by R-77-1 and PL-12. Missile-specific geometry,
propulsion, finsLatAccel and finsAoa come from their datamine-backed profiles.
Compare aerodynamic force normalized by current mass with force normalized by
launch mass. No per-missile eta law or fitted control history enters either fit.

The axial check keeps CxK, CxAoA and the current shared Mach curve fixed. It
integrates acceleration along measured V/alpha instead of differentiating noisy
frames. Positive residual means missing acceleration; negative means missing
retardation. The N*tan(alpha) column is a projection hypothesis, not a correction
silently added to the model. Both checks assume h=6300 m and nominal propulsion.

Run with the project environment; an optional first argument writes a PNG.
These are conditional component checks, not closed-loop trajectory validation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import effective_cda0, effective_cda_alpha
from aim120_model.propulsion import PiecewisePropulsion
from missile_gui.library import scan_library

# These entries select observations, not missile-specific model constants.
CASES = (
    ("su_r_77_1", "R-77-1", "r77_1_level_20260824.tsv", ((0.5, 1.9), (2.1, 3.7), (4.3, 6.9), (7.1, 7.8))),
    ("cn_pl12", "PL-12", "pl12_level_20260824.tsv", ((2.9, 4.9), (4.9, 7.5))),
)


def load_cases():
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    if errors:
        raise ValueError(errors)
    atmosphere = StandardAtmosphere().sample(6300.0)
    cases = []
    for ident, label, filename, intervals in CASES:
        cfg = next(p for p in profiles if p["missile_id"] == ident)["_model_config"]
        prop = PiecewisePropulsion.from_config(cfg)
        with (ROOT / "data/replays" / filename).open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        time = np.array([float(r["missile_flight_time_s"]) for r in rows])
        speed = np.array([float(r["speed_kmh"]) / 3.6 for r in rows])
        alpha = np.radians([float(r["angle_of_attack_deg"]) for r in rows])
        load = np.array([float(r["overload_g"]) for r in rows])
        samples = [prop.sample(float(t)) for t in time]
        mass = np.array([p.mass_kg for p in samples])
        thrust = np.array([p.thrust_n for p in samples])
        gravity = cfg["atmosphere"]["gravity_mps2"]
        q_base = 0.5 * 1.225 * (cfg["control"]["base_indicated_speed_kmh"] / 3.6) ** 2
        eta = 0.5 * atmosphere.density_kg_m3 * speed**2 / q_base
        fin_limit = math.radians(cfg["aerodynamics"]["horizontal_fin_aoa_limit_deg"])
        basis = cfg["aerodynamics"]["fins_lateral_acceleration_g"] * eta * np.minimum(alpha / fin_limit, 1.0)
        mask = (time >= 0.5) & (alpha >= math.radians(1.0))
        cases.append(dict(
            ident=ident, label=label, cfg=cfg, prop=prop, atmosphere=atmosphere,
            time=time, speed=speed, alpha=alpha, load=load, mask=mask,
            basis=basis, mass_ratio=cfg["geometry"]["initial_mass_kg"] / mass,
            thrust_g=thrust * np.sin(alpha) / (mass * gravity), intervals=intervals,
        ))
    return cases


def force_basis(case, reference):
    return case["basis"] * (case["mass_ratio"] if reference == "launch_mass" else 1.0)


def fit_shared_scale(cases, reference):
    # Equal weight per missile, so the 43 R-77-1 points do not overwhelm PL-12.
    numerator = denominator = 0.0
    for case in cases:
        selected = case["mask"]
        x = force_basis(case, reference)[selected]
        y = (case["load"] - case["thrust_g"])[selected]
        numerator += float(np.mean(x * y))
        denominator += float(np.mean(x * x))
    return numerator / denominator


def force_metrics(case, reference, coefficient):
    predicted = case["thrust_g"] + coefficient * force_basis(case, reference)
    error = (predicted - case["load"])[case["mask"]]
    return dict(n=int(case["mask"].sum()), rmse_g=float(np.sqrt(np.mean(error**2))),
                max_abs_g=float(np.max(np.abs(error))), bias_g=float(np.mean(error)))


def axial_intervals(case):
    cfg, prop, atmosphere = case["cfg"], case["prop"], case["atmosphere"]
    results = []
    for lo, hi in case["intervals"]:
        count = round((hi - lo) / 0.005)
        step = (hi - lo) / count
        time = lo + (np.arange(count) + 0.5) * step
        speed = np.interp(time, case["time"], case["speed"])
        alpha = np.interp(time, case["time"], case["alpha"])
        load = np.interp(time, case["time"], case["load"])
        terms = []
        for t, v, a, normal_g in zip(time, speed, alpha, load):
            p = prop.sample(float(t))
            mach = v / atmosphere.speed_of_sound_mps
            q = 0.5 * atmosphere.density_kg_m3 * v**2
            drag0 = q * effective_cda0(mach, cfg)
            drag_alpha = q * effective_cda_alpha(a, 0.0, mach, cfg)
            lift = max(0.0, p.mass_kg * cfg["atmosphere"]["gravity_mps2"] * normal_g - p.thrust_n * math.sin(a))
            terms.append((p.thrust_n * math.cos(a) / p.mass_kg, drag0 / p.mass_kg,
                          drag_alpha / p.mass_kg, lift * math.tan(a) / p.mass_kg))
        impulse = np.sum(terms, axis=0) * step
        observed = float(np.interp(hi, case["time"], case["speed"]) - np.interp(lo, case["time"], case["speed"]))
        predicted = impulse[0] - impulse[1] - impulse[2]
        results.append(dict(
            interval_s=[lo, hi], observed_delta_v_kmh=observed * 3.6,
            predicted_delta_v_kmh=float(predicted * 3.6),
            missing_axial_acceleration_mps2=float((observed - predicted) / (hi - lo)),
            projection_candidate_acceleration_mps2=float(-impulse[3] / (hi - lo)),
        ))
    return results


def plot_results(cases, fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    blue, gray, orange = "#2F6DAE", "#777777", "#C57936"
    for column, case in enumerate(cases):
        selected = case["mask"]
        ax = axes[0, column]
        for reference, color, style, label in (
            ("current_mass", gray, "--", "Current-mass normalization"),
            ("launch_mass", blue, "-", "Launch-mass normalization"),
        ):
            predicted = case["thrust_g"] + fits[reference] * force_basis(case, reference)
            ax.plot(case["time"], predicted, color=color, ls=style, label=label)
        ax.scatter(case["time"][selected], case["load"][selected], facecolors="white",
                   edgecolors="#222222", s=28, zorder=4, label="Observations used in fit")
        ax.set(title=f'{case["label"]}: shared-force prediction', xlabel="Missile flight time (s)", ylabel="Trajectory-normal load (g)")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, column]
        intervals = case["axial"]
        positions = np.arange(len(intervals))
        ax.bar(positions - 0.18, [r["missing_axial_acceleration_mps2"] for r in intervals],
               width=0.36, color=blue, label="Acceleration missing from current model")
        ax.bar(positions + 0.18, [r["projection_candidate_acceleration_mps2"] for r in intervals],
               width=0.36, color=orange, label="Extra body-normal projection candidate")
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_xticks(positions, [f'{r["interval_s"][0]:g}–{r["interval_s"][1]:g}' for r in intervals])
        ax.set(title=f'{case["label"]}: axial balance', xlabel="Flight-time interval (s)", ylabel="Mean acceleration (m/s²)")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e7e7e7", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("One shared coefficient; fixed datamine parameters", x=0.08, ha="left", fontsize=17)
    fig.text(0.08, 0.925, f'Force fits: k = {fits["current_mass"]:.4f} (current mass), {fits["launch_mass"]:.4f} (launch mass). Equal weight per missile.', fontsize=10)
    fig.text(0.08, 0.025, "Measured speed and AoA are inputs. h=6300 m; nominal propulsion; existing shared Mach curve.\nForce-fit points: flight time ≥0.5 s and AoA ≥1°. Conditional component checks; full closed-loop behavior remains unverified.", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, hspace=0.42, wspace=0.25)
    fig.savefig(path, dpi=170, facecolor="white")


def main():
    cases = load_cases()
    fits = {reference: fit_shared_scale(cases, reference) for reference in ("current_mass", "launch_mass")}
    for reference, coefficient in fits.items():
        print("FORCE", json.dumps(dict(reference=reference, shared_k=coefficient,
              missiles={c["ident"]: force_metrics(c, reference, coefficient) for c in cases})))
    coefficient = fit_shared_scale(cases[:1], "launch_mass")
    print("TRANSFER", json.dumps(dict(fit_missile=cases[0]["ident"], shared_k=coefficient,
          other_missile=cases[1]["ident"], **force_metrics(cases[1], "launch_mass", coefficient))))
    for case in cases:
        case["axial"] = axial_intervals(case)
        print("AXIAL", json.dumps(dict(missile=case["ident"], cx_k=case["cfg"]["aerodynamics"]["cx_k"],
              cx_aoa=case["cfg"]["aerodynamics"]["cx_vs_aoa"], intervals=case["axial"])))
    if len(sys.argv) > 1:
        plot_results(cases, fits, Path(sys.argv[1]))
        print("FIGURE", sys.argv[1])


if __name__ == "__main__":
    main()
