#!/usr/bin/env python3
"""P0: controller-independent trim-lift identification from screenshot replays.

Per-frame inversion using only the missile's own displayed state (no guidance,
no PID, no target motion):

    L = n * g0 * m(t) - T(t) * sin(alpha)          [trajectory-normal load,
                                                    thrust-inclusive, README/
                                                    UPDATE_V1.0.2 section 5]
    Cl = L / (qbar * S_ref)

Altitude (hence density) is recovered per frame from the displayed speed and
Mach columns: a = V/M -> ISA temperature -> altitude, using the project's own
StandardAtmosphere so the atmosphere assumption matches the rest of the model.

The identified object is the TRIM lift curve Cl_trim(alpha): total normal
load at the fin deflection the game's (unknown) inner loop chose. It contains
the trim fin share and cannot separate body from fin lift, nor any moment
quantity. Pre-registered candidate law: Cl = min(k * alpha, cap) (game
CyK/CyMaxAoA semantics). Known systematics: displayed-alpha bias for
alpha <~ 1.5 deg (frames flagged, excluded from fits); ISA-vs-game
atmosphere; boost frames carry the datamined thrust through T*sin(alpha).
S_ref = pi/4 * caliber^2 * wingAreaMult (profile reference_area_mode).

Outputs: outputs/lift_identification_frames.tsv, .../lift_identification.png
and a stdout summary with bootstrap intervals.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_rebuild_scenarios import load_rebuild_profile  # noqa: E402
from aim120_model.atmosphere import StandardAtmosphere  # noqa: E402
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402

G0 = 9.80665
SMALL_ALPHA_DEG = 2.0          # display-bias zone, excluded from fits
DN = 0.05                      # display rounding: overload_g, 1 decimal
DM = 0.005                     # display rounding: mach, 2 decimals
DALPHA_DEG = 0.05              # display rounding: alpha, 1 decimal
CASES = (
    ("30degpl12", "cn_pl12", "pl12_30deg_20260823.tsv"),
    ("30degr77", "su_r_77", "r77_30deg_20260823.tsv"),
    ("30degpl12slow", "cn_pl12", "pl12_30deg_slow_20260823.tsv"),
    ("pl12level", "cn_pl12", "pl12_level_20260824.tsv"),
    ("r77-1", "su_r_77_1", "r77_1_level_20260824.tsv"),
)

ATMO = StandardAtmosphere()


def altitude_from_speed_mach(speed_mps: float, mach: float) -> float:
    """Invert ISA troposphere from local speed of sound a = V/M."""
    sound = speed_mps / mach
    temperature = sound * sound / (ATMO._GAMMA * ATMO._R)
    if temperature <= ATMO._T11:
        return ATMO._H11  # isothermal band: altitude not identifiable from a
    return (ATMO._T0 - temperature) / ATMO._L


def load_frames() -> list[dict]:
    frames = []
    profiles = {}
    for case, missile_id, tsv in CASES:
        if missile_id not in profiles:
            profile = load_rebuild_profile(missile_id)
            geometry = profile["_model_config"]["geometry"]
            s_ref = (math.pi / 4.0) * geometry["caliber_m"] ** 2 \
                * geometry["wing_area_multiplier"]
            profiles[missile_id] = (
                PiecewisePropulsion.from_config(profile["_model_config"]), s_ref)
        propulsion, s_ref = profiles[missile_id]
        with (ROOT / "data" / "replays" / tsv).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                fields = (row["speed_kmh"], row["mach"], row["overload_g"],
                          row["angle_of_attack_deg"], row["missile_flight_time_s"])
                if any(value in ("", "NA") for value in fields):
                    continue
                speed = float(row["speed_kmh"]) / 3.6
                mach = float(row["mach"])
                load_g = float(row["overload_g"])
                alpha_deg = float(row["angle_of_attack_deg"])
                time_s = float(row["missile_flight_time_s"])
                if mach <= 0.0 or speed <= 0.0:
                    continue
                altitude = altitude_from_speed_mach(speed, mach)
                rho = ATMO.sample(altitude).density_kg_m3
                rho_lo = ATMO.sample(altitude_from_speed_mach(speed, mach + DM)).density_kg_m3
                rho_hi = ATMO.sample(altitude_from_speed_mach(speed, mach - DM)).density_kg_m3
                drho = 0.5 * abs(rho_hi - rho_lo)
                sample = propulsion.sample(time_s)
                alpha = math.radians(alpha_deg)
                qbar = 0.5 * rho * speed * speed
                lift = load_g * G0 * sample.mass_kg - sample.thrust_n * math.sin(alpha)
                cl = lift / (qbar * s_ref)
                dlift = math.hypot(DN * G0 * sample.mass_kg,
                                   sample.thrust_n * math.cos(alpha)
                                   * math.radians(DALPHA_DEG))
                dcl = math.hypot(dlift / (qbar * s_ref), cl * drho / rho)
                frames.append(dict(
                    case=case, missile=missile_id, t=time_s, speed_mps=speed,
                    mach=mach, altitude_m=altitude, rho=rho, thrust_n=sample.thrust_n,
                    mass_kg=sample.mass_kg, load_g=load_g, alpha_deg=alpha_deg,
                    qbar_pa=qbar, lift_area_m2=lift / qbar, cl=cl, dcl=dcl,
                    boost=int(time_s < propulsion.burn_time_s),
                    small_alpha=int(alpha_deg < SMALL_ALPHA_DEG)))
    return frames


def fit_min_law(alpha_rad, cl, weight):
    ks = np.linspace(0.5, 25.0, 491)
    caps = np.linspace(0.2, 6.0, 581)
    best = (np.inf, np.nan, np.nan)
    for cap in caps:
        pred = np.minimum(ks[:, None] * alpha_rad[None, :], cap)
        sse = (weight[None, :] * (pred - cl[None, :]) ** 2).sum(axis=1)
        index = int(np.argmin(sse))
        if sse[index] < best[0]:
            best = (float(sse[index]), float(ks[index]), cap)
    return best[1], best[2]


def linear_wls(alpha_rad, cl, weight):
    return float((weight * alpha_rad * cl).sum() / (weight * alpha_rad ** 2).sum())


def fit_missile(rows, rng):
    alpha = np.array([r["alpha_deg"] for r in rows])
    alpha_rad = np.radians(alpha)
    cl = np.array([r["cl"] for r in rows])
    dcl = np.maximum(np.array([r["dcl"] for r in rows]), 1e-3)
    weight = 1.0 / dcl ** 2
    coast = np.array([not r["boost"] for r in rows])

    k_hat, cap_hat = fit_min_law(alpha_rad, cl, weight)
    break_deg = math.degrees(cap_hat / k_hat)
    saturated = alpha > break_deg
    cap_identified = break_deg < 0.9 * alpha.max() and saturated.sum() >= 8
    if not cap_identified:
        k_hat = linear_wls(alpha_rad, cl, weight)
        cap_hat, break_deg = float("nan"), float("nan")

    boot_k, boot_cap = [], []
    n = len(rows)
    for _ in range(1000):
        pick = rng.integers(0, n, n)
        if cap_identified:
            ks = np.linspace(k_hat * 0.7, k_hat * 1.3, 61)
            caps = np.linspace(cap_hat * 0.7, cap_hat * 1.3, 61)
            best = (np.inf, np.nan, np.nan)
            for cap in caps:
                pred = np.minimum(ks[:, None] * alpha_rad[pick][None, :], cap)
                sse = (weight[pick][None, :] * (pred - cl[pick][None, :]) ** 2).sum(axis=1)
                index = int(np.argmin(sse))
                if sse[index] < best[0]:
                    best = (float(sse[index]), float(ks[index]), cap)
            boot_k.append(best[1])
            boot_cap.append(best[2])
        else:
            boot_k.append(linear_wls(alpha_rad[pick], cl[pick], weight[pick]))
    k_ci = np.percentile(boot_k, [2.5, 97.5])
    cap_ci = np.percentile(boot_cap, [2.5, 97.5]) if cap_identified else (np.nan, np.nan)

    pred = np.minimum(k_hat * alpha_rad, cap_hat) if cap_identified else k_hat * alpha_rad
    residual = cl - pred
    unsat = ~saturated if cap_identified else np.ones(n, bool)
    mach = np.array([r["mach"] for r in rows])
    mach_r = float(np.corrcoef(mach[unsat], residual[unsat])[0, 1]) if unsat.sum() > 2 else float("nan")
    k_coast = linear_wls(alpha_rad[coast & unsat], cl[coast & unsat],
                         weight[coast & unsat]) if (coast & unsat).sum() > 2 else float("nan")
    return dict(k=k_hat, k_ci=tuple(k_ci), cap=cap_hat, cap_ci=tuple(cap_ci),
                cap_identified=cap_identified, break_deg=break_deg,
                rms=float(np.sqrt(np.mean(residual ** 2))), mach_residual_r=mach_r,
                k_coast_only=k_coast, n=n, n_boost=int((~coast).sum()),
                alpha_max=float(alpha.max()))


def main() -> None:
    frames = load_frames()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "lift_identification_frames.tsv").open("w", encoding="utf-8",
                                                           newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frames[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(frames)

    rng = np.random.default_rng(20260911)
    fits = {}
    for missile_id in ("cn_pl12", "su_r_77", "su_r_77_1"):
        rows = [r for r in frames if r["missile"] == missile_id and not r["small_alpha"]]
        fits[missile_id] = fit_missile(rows, rng)
        f = fits[missile_id]
        cap_text = (f"cap={f['cap']:.2f} [{f['cap_ci'][0]:.2f},{f['cap_ci'][1]:.2f}]"
                    f" break={f['break_deg']:.1f}deg" if f["cap_identified"]
                    else f"cap NOT identified (alpha_max={f['alpha_max']:.1f}deg)")
        print(f"{missile_id}: n={f['n']} (boost {f['n_boost']}) "
              f"k={f['k']:.2f}/rad [{f['k_ci'][0]:.2f},{f['k_ci'][1]:.2f}] "
              f"{cap_text} rms={f['rms']:.3f} "
              f"coast-only k={f['k_coast_only']:.2f} "
              f"Mach-residual r={f['mach_residual_r']:+.2f}")

    print("\n--- diagnostics: residuals against the fitted line ---")
    for missile_id, f in fits.items():
        rows = [r for r in frames if r["missile"] == missile_id and not r["small_alpha"]]
        alpha_rad = np.radians([r["alpha_deg"] for r in rows])
        cl = np.array([r["cl"] for r in rows])
        pred = (np.minimum(f["k"] * alpha_rad, f["cap"]) if f["cap_identified"]
                else f["k"] * alpha_rad)
        residual = cl - pred
        sigma = np.std(residual)
        times = np.array([r["t"] for r in rows])
        machs = np.array([r["mach"] for r in rows])
        print(f"{missile_id}: resid-vs-t r={np.corrcoef(times, residual)[0,1]:+.2f}")
        for r_, res in sorted(zip(rows, residual), key=lambda p: p[1]):
            if abs(res) > 3.0 * sigma:
                print(f"  outlier {r_['case']} t={r_['t']:.1f}s M={r_['mach']:.2f} "
                      f"alpha={r_['alpha_deg']:.1f} n={r_['load_g']:.1f} resid={res:+.2f}")
        late = times >= 1.0
        if late.sum() > 4:
            weight = 1.0 / np.maximum(np.array([r["dcl"] for r in rows]), 1e-3)[late] ** 2
            k_late = linear_wls(alpha_rad[late], cl[late], weight)
            pred_l = k_late * alpha_rad[late]
            res_l = cl[late] - pred_l
            print(f"  excl t<1s: n={late.sum()} k={k_late:.2f} "
                  f"Mach-resid r={np.corrcoef(machs[late], res_l)[0,1]:+.2f} "
                  f"t-resid r={np.corrcoef(times[late], res_l)[0,1]:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_color = {"30degpl12": "tab:blue", "30degpl12slow": "tab:cyan",
                  "pl12level": "tab:green", "30degr77": "tab:orange",
                  "r77-1": "tab:red"}
    fit_color = {"cn_pl12": "tab:green", "su_r_77": "tab:orange",
                 "su_r_77_1": "tab:red"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    for frame in frames:
        ax1.errorbar(frame["alpha_deg"], frame["cl"], yerr=frame["dcl"],
                     fmt="o" if not frame["boost"] else "s",
                     mfc=case_color[frame["case"]] if not frame["boost"] else "none",
                     mec=case_color[frame["case"]], ecolor="0.8",
                     ms=5, elinewidth=0.8, zorder=3)
        if frame["t"] < 0.5:
            ax1.plot(frame["alpha_deg"], frame["cl"], "o", mfc="none", mec="0.2",
                     ms=11, mew=1.1, zorder=4)
    ax1.axvspan(0.0, SMALL_ALPHA_DEG, color="0.92", zorder=0)
    ax1.text(SMALL_ALPHA_DEG / 2, ax1.get_ylim()[0], "display\nbias", fontsize=7,
             ha="center", va="bottom", color="0.5")
    grid = np.linspace(0.0, 26.0, 200)
    fit_lines = []
    for missile_id, f in fits.items():
        if f["cap_identified"]:
            curve = np.minimum(f["k"] * np.radians(grid), f["cap"])
            label = (f"{missile_id}: min({f['k']:.1f}·α, {f['cap']:.2f}), "
                     f"break {f['break_deg']:.1f}°")
        else:
            curve = f["k"] * np.radians(grid)
            label = f"{missile_id}: {f['k']:.1f}·α (cap unseen ≤{f['alpha_max']:.0f}°)"
        fit_lines.append(ax1.plot(grid, curve, color=fit_color[missile_id],
                                  lw=1.6, label=label)[0])
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec=c, label=case)
               for case, c in case_color.items()]
    handles.append(plt.Line2D([], [], marker="s", ls="", mfc="none", mec="0.3",
                              label="boost (thrust-corrected)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.2",
                              ms=11, label="launch transient t<0.5 s"))
    handles.extend(fit_lines)
    ax1.legend(handles=handles, fontsize=7, loc="upper left")
    ax1.set_xlabel("displayed AoA [deg]")
    ax1.set_ylabel("Cl  (S_ref = π/4·d²·wingAreaMult)")
    ax1.set_title("Trim lift from replays: Cl = (n·g·m − T·sinα)/(q̄·S_ref)")
    ax1.grid(alpha=0.3)

    mach_all = [f_["mach"] for f_ in frames]
    scatter = ax2.scatter([f_["alpha_deg"] for f_ in frames],
                          [f_["cl"] for f_ in frames],
                          c=mach_all, cmap="viridis", s=26, zorder=3)
    fig.colorbar(scatter, ax=ax2, label="Mach")
    ax2.set_xlabel("displayed AoA [deg]")
    ax2.set_ylabel("Cl")
    machs = ", ".join(f"{m}: r={fits[m]['mach_residual_r']:+.2f}" for m in fits)
    ax2.set_title(f"Mach collapse check (residual corr {machs})", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "lift_identification.png", dpi=150)
    print(f"wrote {out_dir / 'lift_identification.png'}")


if __name__ == "__main__":
    main()
