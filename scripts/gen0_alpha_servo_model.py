#!/usr/bin/env python3
"""Standalone gen0 alpha-servo closed-loop model vs the five replays.

Model shape follows docs/BLKX_EVIDENCE_LEDGER.md (gen0 stratum):
    PN command -> trim-alpha command (<= finsAoa total-alpha cap)
    -> first-order alpha relaxation (time constant tau, three variants)
    -> lift from the P0-identified trim lift line (lift area, m^2/rad)
    -> planar kinematics; vertical channel assumed trimmed at 1 g.
Drag/propulsion/atmosphere reuse the project model. Non-stiff by
construction: no torque loop, no PID, no fin state.

Pre-registered tau variants (C fitted ONLY on the r77-1 level replay,
the other four scenarios are predictions without retuning):
    V1: tau = C                       (constant)
    V2: tau = C / (dist * qbar)       (stiffness ~ dist*qbar, 1st order)
    V3: tau = C / sqrt(dist * qbar)   (2nd order critically damped ~ 1/wn)

Fixed non-fitted inputs: lift areas from P0 (R-77-1 0.620, R-77 0.601,
PL-12 0.637 m^2/rad), finsAoa/dist/propNavMult and propulsion from
datamine, guidance start delay 0.5 s and alpha(0)=11 deg from the
observed launch transients (shared across scenarios, not per-case).
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
from aim120_model.alpha_servo import planar_target_motion  # noqa: E402
from aim120_model.drag_models import effective_cda0, effective_cda_alpha  # noqa: E402
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402

G0 = 9.80665
ATMO = StandardAtmosphere()
CONTROL_TICK_S = 0.02
PLANT_DT_S = 0.002
GUIDANCE_DELAY_S = 0.5
ALPHA0_DEG = 11.0
PN_GAIN = 4.0                      # propNavMult, 2.57 datamine
LIFT_AREA = {"cn_pl12": 0.637, "su_r_77": 0.601, "su_r_77_1": 0.620}  # m^2/rad, P0
FINS_AOA = {"cn_pl12": 0.375092, "su_r_77": 0.460812,
            "su_r_77_1": 0.460812}
DIST_CM_STAB = {"cn_pl12": 0.12, "su_r_77": 0.175, "su_r_77_1": 0.175}

CASES = (
    ("r77-1", "su_r_77_1", "r77_1_level_20260824.tsv",
     dict(launch_speed_kmh=1200, launch_altitude_m=6300, target_speed_kmh=1200,
          initial_distance_m=8000, target_azimuth_deg=30,
          target_heading_deg=180, target_course_reference="absolute_world")),
    ("30degpl12", "cn_pl12", "pl12_30deg_20260823.tsv",
     dict(launch_speed_kmh=1508, launch_altitude_m=6783, target_speed_kmh=1549)),
    ("30degr77", "su_r_77", "r77_30deg_20260823.tsv",
     dict(launch_speed_kmh=1500, launch_altitude_m=6200, target_speed_kmh=1550)),
    ("30degpl12slow", "cn_pl12", "pl12_30deg_slow_20260823.tsv",
     dict(launch_speed_kmh=1200, launch_altitude_m=6200, target_speed_kmh=1550)),
    ("pl12level", "cn_pl12", "pl12_level_20260824.tsv",
     dict(launch_speed_kmh=1200, launch_altitude_m=6300, target_speed_kmh=1200)),
)
TARGET_AZIMUTH_RAD = math.radians(30.0)
INITIAL_DISTANCE_M = 8000.0


def tau_of(variant: str, coeff: float, dist: float, qbar: float) -> float:
    if variant == "V1":
        return coeff
    if variant == "V2":
        return coeff / max(dist * qbar, 1.0)
    return coeff / math.sqrt(max(dist * qbar, 1.0))


def simulate_case(missile_id: str, scenario: dict, variant: str, coeff: float,
                  config: dict, propulsion: PiecewisePropulsion,
                  max_time_s: float = 15.0, forced_speed=None, tau_c: float = 0.30):
    atmo = ATMO.sample(scenario["launch_altitude_m"])
    rho, sound = atmo.density_kg_m3, atmo.speed_of_sound_mps
    lift_area = LIFT_AREA[missile_id]
    fins_aoa = FINS_AOA[missile_id]
    dist_stab = DIST_CM_STAB[missile_id]

    speed = scenario["launch_speed_kmh"] / 3.6
    heading = 0.0
    pos = np.zeros(2)
    target0, target_v = planar_target_motion(scenario)
    tgt = np.array(target0)
    tgt_vel = np.array(target_v)
    alpha_h = math.radians(ALPHA0_DEG)
    alpha_cmd = 0.0
    capturing = True
    release_t, release_a, release_sign = 0.0, 0.0, 1.0

    t = 0.0
    next_tick = 0.0
    log = []
    prev_miss = None
    while t < max_time_s:
        p = propulsion.sample(t)
        qbar = 0.5 * rho * speed * speed
        mach = speed / sound
        # Vertical channel: trimmed at 1 g (aero+thrust).
        norm_per_alpha = qbar * lift_area + p.thrust_n
        alpha_v = min(p.mass_kg * G0 / max(norm_per_alpha, 1.0), fins_aoa)

        if t >= next_tick - 1e-9:  # 50 Hz guidance/command, zero-order hold
            rel = tgt - pos
            vel = speed * np.array([math.cos(heading), math.sin(heading)])
            rel_v = tgt_vel - vel
            r2 = float(rel @ rel)
            los_rate = float(rel[0] * rel_v[1] - rel[1] * rel_v[0]) / max(r2, 1.0)
            closing = -float(rel @ rel_v) / max(math.sqrt(r2), 1.0)
            if t < GUIDANCE_DELAY_S:
                alpha_cmd = 0.0
            else:
                # PCC capture phase: steer to predicted collision course with
                # a_cap = V*sin(eps)/tau_c (identified tau_c=0.30, eps_exit=15
                # deg, one-way release) then PN tracking.
                r = math.sqrt(r2)
                u = rel / max(r, 1.0)
                vt_perp = float(u[0] * tgt_vel[1] - u[1] * tgt_vel[0])
                lead = math.asin(max(-1.0, min(1.0, vt_perp / max(speed, 1.0))))
                chi_c = math.atan2(u[1], u[0]) + lead
                eps = (chi_c - heading + math.pi) % (2 * math.pi) - math.pi
                a_cap = speed * math.sin(eps) / tau_c
                if capturing and abs(eps) < math.radians(15.0):
                    capturing = False
                    release_t, release_a = t, abs(a_cap)
                    release_sign = 1.0 if a_cap >= 0.0 else -1.0
                if capturing:
                    a_cmd = a_cap
                else:
                    a_pn = PN_GAIN * max(closing, 1.0) * los_rate
                    hold = release_a * math.exp(-(t - release_t) / 0.95)
                    a_cmd = a_pn if abs(a_pn) > hold else release_sign * hold
                alpha_cmd = p.mass_kg * a_cmd / max(norm_per_alpha, 1.0)
            cap = math.sqrt(max(fins_aoa ** 2 - alpha_v ** 2, 0.0))
            alpha_cmd = max(-cap, min(cap, alpha_cmd))
            next_tick += CONTROL_TICK_S

        tau = max(tau_of(variant, coeff, dist_stab, qbar), 0.02)
        alpha_total = math.hypot(alpha_h, alpha_v)
        a_perp = norm_per_alpha * alpha_h / p.mass_kg
        drag_n = qbar * (effective_cda0(mach, config)
                         + effective_cda_alpha(alpha_total, 0.0, mach, config))
        load_g = math.hypot(a_perp / G0, 1.0)

        log.append((t, speed, math.degrees(alpha_total), load_g))

        # plant substep integration (semi-implicit alpha, Euler kinematics)
        alpha_h += (alpha_cmd - alpha_h) * (1.0 - math.exp(-PLANT_DT_S / tau))
        if forced_speed is not None:
            speed = float(np.interp(t + PLANT_DT_S, forced_speed[:, 0],
                                    forced_speed[:, 1]))
        else:
            accel = (p.thrust_n * math.cos(alpha_total) - drag_n) / p.mass_kg
            speed = max(speed + accel * PLANT_DT_S, 50.0)
        heading += (a_perp / speed) * PLANT_DT_S
        pos += speed * np.array([math.cos(heading), math.sin(heading)]) * PLANT_DT_S
        tgt += tgt_vel * PLANT_DT_S
        t += PLANT_DT_S

        miss = float(np.linalg.norm(tgt - pos))
        if prev_miss is not None and miss > prev_miss and prev_miss < 300.0:
            break
        prev_miss = miss
    return np.array(log), prev_miss


def observed(tsv: str):
    rows = []
    with (ROOT / "data" / "replays" / tsv).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            f = (row["missile_flight_time_s"], row["speed_kmh"],
                 row["angle_of_attack_deg"], row["overload_g"])
            if any(v in ("", "NA") for v in f):
                continue
            rows.append(tuple(float(v) for v in f))
    arr = np.array(rows)
    return arr[arr[:, 0].argsort()]


def rmse_case(log: np.ndarray, obs: np.ndarray):
    tmax = log[-1, 0]
    mask = obs[:, 0] <= tmax
    if mask.sum() < 3:
        return dict(v=float("nan"), a=float("nan"), g=float("nan"), n=0)
    ot = obs[mask, 0]
    sim_v = np.interp(ot, log[:, 0], log[:, 1]) * 3.6
    sim_a = np.interp(ot, log[:, 0], log[:, 2])
    sim_g = np.interp(ot, log[:, 0], log[:, 3])
    return dict(v=float(np.sqrt(np.mean((sim_v - obs[mask, 1]) ** 2))),
                a=float(np.sqrt(np.mean((sim_a - obs[mask, 2]) ** 2))),
                g=float(np.sqrt(np.mean((sim_g - obs[mask, 3]) ** 2))),
                n=int(mask.sum()))


def main() -> None:
    profiles = {}
    for _case, mid, _tsv, _s in CASES:
        if mid not in profiles:
            prof = load_rebuild_profile(mid)
            cfg = prof["_model_config"]
            profiles[mid] = (cfg, PiecewisePropulsion.from_config(cfg))

    calib_case, calib_mid, calib_tsv, calib_s = CASES[0]
    cfg, prop = profiles[calib_mid]
    obs0 = observed(calib_tsv)
    sp0 = np.column_stack([obs0[:, 0], obs0[:, 1] / 3.6])
    fits = {"V0": 0.02}                      # quasi-static null: instant alpha
    for variant, grid in (("V1", np.linspace(0.05, 1.2, 116)),
                          ("V2", np.linspace(50, 6000, 120)),
                          ("V3", np.linspace(1, 80, 159))):
        best = (np.inf, None)
        for coeff in grid:
            log, _ = simulate_case(calib_mid, calib_s, variant, coeff, cfg, prop,
                                   forced_speed=sp0)
            m = rmse_case(log, obs0)
            score = m["a"] + m["g"]          # alpha + G, deg+g equal weight
            if score < best[0]:
                best = (score, float(coeff))
        fits[variant] = best[1]
        print(f"{variant}: C={best[1]:.4g} (calib score {best[0]:.2f} on {calib_case})")

    # Per-missile capture constant: PL-12 never rides its alpha cap, so its
    # tau_c is identifiable; calibrate on pl12level (same A7 geometry as the
    # r77-1 calibration case), the two 30-deg PL-12 cases stay predictions.
    tau_c_by_missile = {"su_r_77": 0.30, "su_r_77_1": 0.30}
    pl_case, pl_mid, pl_tsv, pl_scen = CASES[4]
    cfgp, propp = profiles[pl_mid]
    obsp = observed(pl_tsv)
    spp = np.column_stack([obsp[:, 0], obsp[:, 1] / 3.6])
    best = (np.inf, 0.30)
    for tau_c in np.linspace(0.3, 2.0, 35):
        log, _ = simulate_case(pl_mid, pl_scen, "V2", fits["V2"], cfgp, propp,
                               forced_speed=spp, tau_c=float(tau_c))
        m = rmse_case(log, obsp)
        if m["a"] + m["g"] < best[0]:
            best = (m["a"] + m["g"], float(tau_c))
    tau_c_by_missile["cn_pl12"] = best[1]
    print(f"PL-12 tau_c = {best[1]:.2f} s (calibrated on {pl_case}, "
          f"score {best[0]:.2f}); R-77 family keeps 0.30")

    print(f"\n{'case':14s} {'variant':7s} {'tau@calib':>9s} "
          f"{'RMSE_a':>7s} {'RMSE_g':>7s} {'miss':>6s} {'t_end':>6s}")
    results = {}
    for case, mid, tsv, scen in CASES:
        cfg, prop = profiles[mid]
        obs = observed(tsv)
        sp = np.column_stack([obs[:, 0], obs[:, 1] / 3.6])
        for variant in ("V0", "V1", "V2", "V3"):
            log, miss = simulate_case(mid, scen, variant, fits[variant], cfg, prop,
                                      forced_speed=sp,
                                      tau_c=tau_c_by_missile[mid])
            m = rmse_case(log, obs)
            q0 = 0.5 * ATMO.sample(scen["launch_altitude_m"]).density_kg_m3 \
                * (scen["launch_speed_kmh"] / 3.6) ** 2
            tau0 = tau_of(variant, fits[variant], DIST_CM_STAB[mid], q0)
            tag = " calib" if case == calib_case else ""
            print(f"{case:14s} {variant:7s} {tau0:9.3f} "
                  f"{m['a']:7.2f} {m['g']:7.2f} {miss:6.1f} {log[-1,0]:6.2f}{tag}")
            results[(case, variant)] = (log, obs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex="col")
    for col, (case, mid, tsv, scen) in enumerate(CASES):
        obs = results[(case, "V2")][1]
        for row, (idx, label, scale) in enumerate(
                ((1, "speed [km/h]", 3.6), (2, "AoA [deg]", 1.0), (3, "G", 1.0))):
            ax = axes[row][col]
            for variant, color in (("V0", "0.8"), ("V1", "0.5"),
                                   ("V2", "tab:red"), ("V3", "tab:blue")):
                log = results[(case, variant)][0]
                ax.plot(log[:, 0], log[:, idx] * (scale if idx == 1 else 1.0),
                        color=color, lw=1.3, label=variant if (row == 0 and col == 0) else None)
            ax.plot(obs[:, 0], obs[:, row + 1], "k.", ms=5,
                    label="replay" if (row == 0 and col == 0) else None)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(case + (" (calib)" if case == "r77-1" else " (pred)"),
                             fontsize=10)
            if row == 2:
                ax.set_xlabel("t [s]")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("gen0 alpha-servo model vs replays — C fitted on r77-1 only, "
                 "others pure prediction", fontsize=12)
    fig.tight_layout()
    out = ROOT / "outputs" / "gen0_alpha_servo_validation.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
