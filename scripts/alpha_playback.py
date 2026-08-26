#!/usr/bin/env python3
"""Alpha-playback open-loop plant exam for the R-77-1 level-launch replay.

This is an EXAM HARNESS, not a fitting script.  It feeds the game's own
measured alpha(t) (data/replays/r77_1_level_20260824.tsv) into the shipped H2
force model with guidance, control, and rotation fully bypassed, and compares
the plant's predicted G and integrated speed against the replay's measured
columns.  No parameter is adjusted based on the results printed here.

MODE 1 (state-matched, no integration): at each replay frame, evaluate three
path-normal-G force-law variants against the frame's own measured
speed/Mach/alpha/time, with no time integration at all.  A subset of frames
is additionally cross-checked through the real model code path
(aim120_model.h2_dynamics.forces_for_state_h2) to confirm the closed-form
analytic evaluation used for the full 46-frame table matches the shipped
plant.

MODE 2 (integrated open-loop energy exam): starting from the S001 replay
state, integrate dv/dt = (T cos(alpha) - D_total) / m forward in time at a
fixed 6300 m altitude, with alpha(t) linearly interpolated from the replay's
own (flight_time_s, angle_of_attack_deg) samples, under two drag-law
variants, and compare the resulting v(t) against the replay's measured speed
column.

Every physical constant used below (fins_lateral_acceleration_g,
horizontal_fin_aoa_limit_deg, packed_lift_slope_scale, the per-missile
packed_lift_force_eta_law, propulsion stage thrust/duration/mass, cx_k, wing
area multiplier, caliber, the shipped Cx(M) drag curve, cx_vs_aoa, and base
indicated speed) is read out of the built runtime config -- the same config
missiles/su_r_77_1.json + config/profile_h2_runtime_defaults.json produce via
aim120_model.profile_adapter.build_h2_candidate_config, exactly as
scripts/run_m0_scenarios.py and the test suite construct it.  Nothing here
is hardcoded independently of that config, and the drag terms are evaluated
by calling the shipped aim120_model.drag_models.effective_cda0 /
effective_cda_alpha functions directly rather than re-deriving the Cx(M)
interpolation by hand.

This script performs no filesystem writes; it only prints to stdout.
"""

from __future__ import annotations

import bisect
import csv
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aim120_model.atmosphere import StandardAtmosphere  # noqa: E402
from aim120_model.drag_models import effective_cda0, effective_cda_alpha  # noqa: E402
from aim120_model.dynamics import SimState  # noqa: E402
from aim120_model.h2_dynamics import forces_for_state_h2  # noqa: E402
from aim120_model.profile_adapter import (  # noqa: E402
    build_h2_candidate_config,
    load_runtime_defaults,
)
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402

MISSILE_PATH = PROJECT_DIR / "missiles" / "su_r_77_1.json"
DEFAULTS_PATH = PROJECT_DIR / "config" / "profile_h2_runtime_defaults.json"
M0_DEFAULTS_PATH = PROJECT_DIR / "config" / "profile_m0_strict.json"
REPLAY_PATH = PROJECT_DIR / "data" / "replays" / "r77_1_level_20260824.tsv"

# --- Atmosphere reconstruction constants (identical algebra to the shipped
# StandardAtmosphere troposphere formula; see the "atmosphere equivalence"
# note printed by main()). ---
SOUND_SPEED_DIVISOR = 20.0468  # sqrt(gamma_air * R_air), gamma=1.4, R=287.05287
T0_K = 288.15
RHO0_KG_M3 = 1.225
LAPSE_K_PER_M = 0.0065
RHO_EXPONENT = 4.2559

FIXED_ALTITUDE_M = 6300.0
MACH_BANDS = [
    ("M<1.7", lambda m: m < 1.7),
    ("1.7<=M<2.4", lambda m: 1.7 <= m < 2.4),
    ("2.4<=M<2.9", lambda m: 2.4 <= m < 2.9),
    ("M>=2.9", lambda m: m >= 2.9),
]
CROSS_CHECK_SAMPLE_IDS = ("S006", "S023", "S040")  # early / mid / late
EXPLICIT_DELTA_TIMES_S = (2.3, 4.3, 6.9, 7.8)


# --------------------------------------------------------------------------
# Config / data loading
# --------------------------------------------------------------------------

def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the shipped candidate config and the M0-strict candidate config.

    Wiring mirrors scripts/run_m0_scenarios.py's load_m0_profile(): the same
    profile JSON is adapted twice, once against each runtime-defaults file.
    """

    profile = __import__("json").loads(MISSILE_PATH.read_text(encoding="utf-8"))
    shipped_defaults = load_runtime_defaults(str(DEFAULTS_PATH))
    shipped_config, _assumptions = build_h2_candidate_config(profile, shipped_defaults)

    m0_defaults = load_runtime_defaults(str(M0_DEFAULTS_PATH))
    m0_config, _m0_assumptions = build_h2_candidate_config(profile, m0_defaults)
    overrides = m0_defaults.get("m0_overrides", {})
    for key in overrides.get("strip_profile_aerodynamics_keys", []):
        m0_config["aerodynamics"].pop(key, None)
    return shipped_config, m0_config


def load_replay_rows() -> list[dict[str, str]]:
    with REPLAY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) != 46:
        raise ValueError(f"expected 46 replay frames, found {len(rows)}")
    return rows


# --------------------------------------------------------------------------
# Shared physics helpers
# --------------------------------------------------------------------------

def reconstruct_atmosphere(speed_mps: float, mach: float) -> tuple[float, float, float]:
    """Per-frame atmosphere reconstruction from the replay's own speed/Mach.

    Returns (speed_of_sound_mps, temperature_K, density_kg_m3).
    """

    a_sound = speed_mps / mach
    t_k = (a_sound / SOUND_SPEED_DIVISOR) ** 2
    rho = RHO0_KG_M3 * (t_k / T0_K) ** RHO_EXPONENT
    return a_sound, t_k, rho


def equivalent_altitude_m(t_k: float) -> float:
    """Invert the ISA troposphere lapse rate to find the altitude whose
    StandardAtmosphere sample reproduces the reconstructed temperature.
    """

    return (T0_K - t_k) / LAPSE_K_PER_M


def eta_of(q_pa: float, q_base_pa: float, ratio_max: float | None) -> float:
    raw = q_pa / q_base_pa
    return min(raw, ratio_max) if ratio_max is not None else raw


def force_k_eta_law(eta: float, eta_law: dict[str, Any]) -> float:
    eta_lo, eta_hi = eta_law["eta_clamp"]
    clamped = min(max(eta, float(eta_lo)), float(eta_hi))
    return float(eta_law["coefficient"]) * clamped ** float(eta_law["eta_exponent"])


def g_pred_components(
    k: float,
    fins_g: float,
    eta: float,
    alpha_rad: float,
    alpha_max_rad: float,
    thrust_n: float,
    mass_kg: float,
    gravity_mps2: float,
) -> tuple[float, float, float]:
    """Return (aero_path_g, thrust_lateral_g, total_g) for the shared H2 spec
    packed-lift force law: G = k*A*eta*(alpha/alpha_max) + T*sin(alpha)/(m*g).
    """

    aero_g = k * fins_g * eta * (alpha_rad / alpha_max_rad)
    thrust_g = thrust_n * math.sin(alpha_rad) / (mass_kg * gravity_mps2)
    return aero_g, thrust_g, aero_g + thrust_g


def interp_linear(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    i = max(0, min(i, len(xs) - 2))
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    frac = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
    return y0 + frac * (y1 - y0)


def find_row_by_time(rows: list[dict[str, str]], target_t: float) -> dict[str, str]:
    best = min(rows, key=lambda r: abs(float(r["missile_flight_time_s"]) - target_t))
    if abs(float(best["missile_flight_time_s"]) - target_t) > 1e-6:
        raise ValueError(f"no exact replay frame at t={target_t}")
    return best


def find_row_by_id(rows: list[dict[str, str]], sample_id: str) -> dict[str, str]:
    for row in rows:
        if row["sample_id"] == sample_id:
            return row
    raise KeyError(sample_id)


# --------------------------------------------------------------------------
# MODE 1 -- state-matched force-law exam
# --------------------------------------------------------------------------

def run_mode1(config: dict[str, Any], config_m0: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    aero_cfg = config["aerodynamics"]
    fins_g = float(aero_cfg["fins_lateral_acceleration_g"])
    alpha_max_rad = math.radians(float(aero_cfg["horizontal_fin_aoa_limit_deg"]))
    k_a = float(aero_cfg["packed_lift_slope_scale"])
    eta_law = aero_cfg["packed_lift_force_eta_law"]
    k_c = float(config_m0["aerodynamics"]["packed_lift_slope_scale"])
    gravity = float(config["atmosphere"]["gravity_mps2"])
    base_kmh = float(config["control"]["base_indicated_speed_kmh"])
    ratio_max = config["control"].get("base_indicated_speed_ratio_max")
    ratio_max = float(ratio_max) if ratio_max is not None else None
    q_base = 0.5 * 1.225 * (base_kmh / 3.6) ** 2

    prop = PiecewisePropulsion.from_config(config)

    results: list[dict[str, Any]] = []
    for row in rows:
        t = float(row["missile_flight_time_s"])
        mach = float(row["mach"])
        speed_mps = float(row["speed_kmh"]) / 3.6
        alpha_deg = float(row["angle_of_attack_deg"])
        alpha_rad = math.radians(alpha_deg)
        g_game = float(row["overload_g"])

        a_sound, t_k, rho = reconstruct_atmosphere(speed_mps, mach)
        q = 0.5 * rho * speed_mps * speed_mps
        eta = eta_of(q, q_base, ratio_max)

        sample = prop.sample(t, powered=True)
        thrust_n = sample.thrust_n
        mass_kg = sample.mass_kg

        k_b = force_k_eta_law(eta, eta_law)
        _aero_a, _thr_a, g_a = g_pred_components(k_a, fins_g, eta, alpha_rad, alpha_max_rad, thrust_n, mass_kg, gravity)
        _aero_b, _thr_b, g_b = g_pred_components(k_b, fins_g, eta, alpha_rad, alpha_max_rad, thrust_n, mass_kg, gravity)
        _aero_c, _thr_c, g_c = g_pred_components(k_c, fins_g, eta, alpha_rad, alpha_max_rad, thrust_n, mass_kg, gravity)

        include = (t >= 0.5) and (alpha_deg >= 1.0)
        results.append(
            {
                "sample_id": row["sample_id"],
                "t": t,
                "mach": mach,
                "eta": eta,
                "alpha_deg": alpha_deg,
                "alpha_rad": alpha_rad,
                "t_k": t_k,
                "rho": rho,
                "q": q,
                "mass_kg": mass_kg,
                "thrust_n": thrust_n,
                "g_game": g_game,
                "g_a": g_a,
                "g_b": g_b,
                "g_c": g_c,
                "err_a": abs(g_a - g_game),
                "err_b": abs(g_b - g_game),
                "err_c": abs(g_c - g_game),
                "include": include,
                "k_b": k_b,
            }
        )
    return results


def print_mode1_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'id':5} {'t':>6} {'M':>5} {'eta':>6} {'a_deg':>6} "
        f"{'G_game':>7} {'G_A':>7} {'G_B':>7} {'G_C':>7} "
        f"{'|eA|':>6} {'|eB|':>6} {'|eC|':>6} {'incl':>5}"
    )
    for r in results:
        print(
            f"{r['sample_id']:5} {r['t']:6.3f} {r['mach']:5.2f} {r['eta']:6.3f} {r['alpha_deg']:6.1f} "
            f"{r['g_game']:7.2f} {r['g_a']:7.2f} {r['g_b']:7.2f} {r['g_c']:7.2f} "
            f"{r['err_a']:6.2f} {r['err_b']:6.2f} {r['err_c']:6.2f} "
            f"{'Y' if r['include'] else 'n':>5}"
        )


def print_mode1_band_summary(results: list[dict[str, Any]]) -> None:
    print(f"{'band':12} {'n':>4} {'mean|eA|':>9} {'mean|eB|':>9} {'mean|eC|':>9}")
    for band_name, predicate in MACH_BANDS:
        subset = [r for r in results if r["include"] and predicate(r["mach"])]
        n = len(subset)
        if n == 0:
            print(f"{band_name:12} {n:4d} {'n/a':>9} {'n/a':>9} {'n/a':>9}")
            continue
        mean_a = sum(r["err_a"] for r in subset) / n
        mean_b = sum(r["err_b"] for r in subset) / n
        mean_c = sum(r["err_c"] for r in subset) / n
        print(f"{band_name:12} {n:4d} {mean_a:9.3f} {mean_b:9.3f} {mean_c:9.3f}")
    subset_all = [r for r in results if r["include"]]
    n = len(subset_all)
    mean_a = sum(r["err_a"] for r in subset_all) / n
    mean_b = sum(r["err_b"] for r in subset_all) / n
    mean_c = sum(r["err_c"] for r in subset_all) / n
    print(f"{'ALL':12} {n:4d} {mean_a:9.3f} {mean_b:9.3f} {mean_c:9.3f}")


# --------------------------------------------------------------------------
# Cross-check: analytic V_B vs the real forces_for_state_h2 code path
# --------------------------------------------------------------------------

def run_cross_check(config: dict[str, Any], rows: list[dict[str, str]], mode1_by_id: dict[str, dict[str, Any]]) -> None:
    aero_cfg = config["aerodynamics"]
    alpha_max_rad = math.radians(float(aero_cfg["horizontal_fin_aoa_limit_deg"]))
    gravity = float(config["atmosphere"]["gravity_mps2"])
    prop = PiecewisePropulsion.from_config(config)

    print(
        f"{'id':5} {'t':>6} {'a_deg':>6} {'alt_eq_m':>9} "
        f"{'M_frame':>8} {'M_code':>8} {'G_B_analytic':>13} {'G_B_code':>10} {'rel_diff_%':>11}"
    )
    for sample_id in CROSS_CHECK_SAMPLE_IDS:
        row = find_row_by_id(rows, sample_id)
        mode1 = mode1_by_id[sample_id]
        t = mode1["t"]
        alpha_rad = mode1["alpha_rad"]
        speed_mps = float(row["speed_kmh"]) / 3.6
        alt_eq = equivalent_altitude_m(mode1["t_k"])

        sample = prop.sample(t, powered=True)
        mass_kg = sample.mass_kg

        state = SimState(
            (0.0, alt_eq, 0.0),
            (speed_mps, 0.0, 0.0),
            alpha_rad,
            0.0,
            0.0,
            0.0,
            mass_kg,
            actual_pitch_fin_angle_rad=alpha_rad,
        )
        diagnostics = forces_for_state_h2(state, t, config, prop, powered=True)
        g_code = diagnostics.trajectory_pitch_normal_acceleration_g
        g_analytic = mode1["g_b"]
        rel_diff_pct = 100.0 * (g_code - g_analytic) / g_analytic if abs(g_analytic) > 1e-9 else float("nan")

        print(
            f"{sample_id:5} {t:6.3f} {mode1['alpha_deg']:6.1f} {alt_eq:9.1f} "
            f"{mode1['mach']:8.4f} {diagnostics.aero.mach:8.4f} "
            f"{g_analytic:13.4f} {g_code:10.4f} {rel_diff_pct:11.4f}"
        )
        alpha_check = math.degrees(diagnostics.aero.pitch_alpha_rad)
        q_analytic = mode1["q"]
        q_code = diagnostics.aero.dynamic_pressure_pa
        print(
            f"      code-path AoA={alpha_check:.4f} deg (frame {mode1['alpha_deg']:.4f}); "
            f"q_analytic={q_analytic:.2f} Pa, q_code={q_code:.2f} Pa "
            f"(rel diff {100.0 * (q_code - q_analytic) / q_analytic:.5f}%)"
        )


# --------------------------------------------------------------------------
# MODE 2 -- integrated open-loop energy exam
# --------------------------------------------------------------------------

def run_mode2(config: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    aero_cfg = config["aerodynamics"]
    fins_g = float(aero_cfg["fins_lateral_acceleration_g"])
    alpha_max_rad = math.radians(float(aero_cfg["horizontal_fin_aoa_limit_deg"]))
    eta_law = aero_cfg["packed_lift_force_eta_law"]
    gravity = float(config["atmosphere"]["gravity_mps2"])
    base_kmh = float(config["control"]["base_indicated_speed_kmh"])
    ratio_max = config["control"].get("base_indicated_speed_ratio_max")
    ratio_max = float(ratio_max) if ratio_max is not None else None
    q_base = 0.5 * 1.225 * (base_kmh / 3.6) ** 2

    prop = PiecewisePropulsion.from_config(config)
    atm_fixed = StandardAtmosphere().sample(FIXED_ALTITUDE_M)
    rho_fixed = atm_fixed.density_kg_m3
    c_fixed = atm_fixed.speed_of_sound_mps

    frame_times = [float(r["missile_flight_time_s"]) for r in rows]
    frame_alpha = [float(r["angle_of_attack_deg"]) for r in rows]

    def alpha_deg_at(t: float) -> float:
        return interp_linear(t, frame_times, frame_alpha)

    t0 = frame_times[0]
    t_end = frame_times[-1]
    v0 = float(rows[0]["speed_kmh"]) / 3.6
    dt = 0.005

    t = t0
    v1 = v0
    v2 = v0
    hist_t = [t]
    hist_v1 = [v1]
    hist_v2 = [v2]

    while t < t_end - 1e-12:
        alpha_deg = alpha_deg_at(t)
        alpha_rad = math.radians(alpha_deg)
        sample = prop.sample(t, powered=True)
        thrust_n = sample.thrust_n
        mass_kg = sample.mass_kg

        mach1 = v1 / c_fixed
        mach2 = v2 / c_fixed
        q1 = 0.5 * rho_fixed * v1 * v1
        q2 = 0.5 * rho_fixed * v2 * v2

        cda0_1 = effective_cda0(mach1, config)
        cda_alpha_1 = effective_cda_alpha(alpha_rad, 0.0, mach1, config)
        drag1_n = q1 * (cda0_1 + cda_alpha_1)

        cda0_2 = effective_cda0(mach2, config)
        eta2 = eta_of(q2, q_base, ratio_max)
        k_b2 = force_k_eta_law(eta2, eta_law)
        lift_aero2_n = k_b2 * fins_g * eta2 * (alpha_rad / alpha_max_rad) * mass_kg * gravity
        drag2_n = q2 * cda0_2 + lift_aero2_n * math.tan(alpha_rad)

        dv1_dt = (thrust_n * math.cos(alpha_rad) - drag1_n) / mass_kg
        dv2_dt = (thrust_n * math.cos(alpha_rad) - drag2_n) / mass_kg

        step = min(dt, t_end - t)
        v1 = v1 + step * dv1_dt
        v2 = v2 + step * dv2_dt
        t = t + step

        hist_t.append(t)
        hist_v1.append(v1)
        hist_v2.append(v2)

    peak1_mps = max(hist_v1)
    peak2_mps = max(hist_v2)
    peak1_t = hist_t[hist_v1.index(peak1_mps)]
    peak2_t = hist_t[hist_v2.index(peak2_mps)]

    return {
        "hist_t": hist_t,
        "hist_v1": hist_v1,
        "hist_v2": hist_v2,
        "rho_fixed": rho_fixed,
        "c_fixed": c_fixed,
        "t_k_fixed": atm_fixed.temperature_k,
        "peak1_kmh": peak1_mps * 3.6,
        "peak2_kmh": peak2_mps * 3.6,
        "peak1_t": peak1_t,
        "peak2_t": peak2_t,
    }


def print_mode2_table(mode2: dict[str, Any], rows: list[dict[str, str]]) -> None:
    hist_t = mode2["hist_t"]
    hist_v1 = mode2["hist_v1"]
    hist_v2 = mode2["hist_v2"]
    print(
        f"{'id':5} {'t':>6} {'M_game':>7} {'v_game':>8} {'v1_shipped':>11} "
        f"{'v2_diag':>9} {'d1':>7} {'d2':>7}"
    )
    for row in rows:
        t = float(row["missile_flight_time_s"])
        v_game_kmh = float(row["speed_kmh"])
        v1_kmh = interp_linear(t, hist_t, hist_v1) * 3.6
        v2_kmh = interp_linear(t, hist_t, hist_v2) * 3.6
        print(
            f"{row['sample_id']:5} {t:6.3f} {row['mach']:>7} {v_game_kmh:8.1f} {v1_kmh:11.1f} "
            f"{v2_kmh:9.1f} {v1_kmh - v_game_kmh:7.1f} {v2_kmh - v_game_kmh:7.1f}"
        )


def print_mode2_explicit_deltas(mode2: dict[str, Any], rows: list[dict[str, str]]) -> None:
    hist_t = mode2["hist_t"]
    hist_v1 = mode2["hist_v1"]
    hist_v2 = mode2["hist_v2"]
    print(f"{'t_s':>6} {'v_game_kmh':>11} {'v1_kmh':>9} {'v2_kmh':>9} {'d1_kmh':>8} {'d2_kmh':>8}")
    for target_t in EXPLICIT_DELTA_TIMES_S:
        row = find_row_by_time(rows, target_t)
        v_game_kmh = float(row["speed_kmh"])
        v1_kmh = interp_linear(target_t, hist_t, hist_v1) * 3.6
        v2_kmh = interp_linear(target_t, hist_t, hist_v2) * 3.6
        print(
            f"{target_t:6.2f} {v_game_kmh:11.1f} {v1_kmh:9.1f} {v2_kmh:9.1f} "
            f"{v1_kmh - v_game_kmh:8.1f} {v2_kmh - v_game_kmh:8.1f}"
        )


def _sign_runs(labelled_deltas: list[tuple[str, float, float]]) -> list[tuple[str, str, float, str, float, int]]:
    """Group a (sample_id, t, delta) sequence into consecutive-same-sign runs.

    Returns a list of (sign, start_id, start_t, end_id, end_t, count) tuples,
    in frame order, sign in {"+", "-", "0"}.
    """

    runs: list[tuple[str, str, float, str, float, int]] = []
    current_sign = None
    current_start: tuple[str, float] | None = None
    current_end: tuple[str, float] | None = None
    current_count = 0
    for sample_id, t, delta in labelled_deltas:
        sign = "+" if delta > 0.0 else ("-" if delta < 0.0 else "0")
        if sign != current_sign:
            if current_sign is not None:
                runs.append((current_sign, current_start[0], current_start[1], current_end[0], current_end[1], current_count))
            current_sign = sign
            current_start = (sample_id, t)
            current_count = 0
        current_end = (sample_id, t)
        current_count += 1
    if current_sign is not None:
        runs.append((current_sign, current_start[0], current_start[1], current_end[0], current_end[1], current_count))
    return runs


def print_mode2_delta_summary(mode2: dict[str, Any], rows: list[dict[str, str]]) -> None:
    """Programmatic (not eyeballed) sign-run and min/max summary of the
    D_1/D_2 speed deltas across all 46 frame times, so the prose summary in
    docs/ALPHA_PLAYBACK_R77_1.md is transcribed from computed output rather
    than manually scanned off the table.
    """

    hist_t = mode2["hist_t"]
    hist_v1 = mode2["hist_v1"]
    hist_v2 = mode2["hist_v2"]
    d1_series: list[tuple[str, float, float]] = []
    d2_series: list[tuple[str, float, float]] = []
    for row in rows:
        t = float(row["missile_flight_time_s"])
        v_game_kmh = float(row["speed_kmh"])
        v1_kmh = interp_linear(t, hist_t, hist_v1) * 3.6
        v2_kmh = interp_linear(t, hist_t, hist_v2) * 3.6
        d1_series.append((row["sample_id"], t, v1_kmh - v_game_kmh))
        d2_series.append((row["sample_id"], t, v2_kmh - v_game_kmh))

    for label, series in (("D_1 shipped", d1_series), ("D_2 diagnostic", d2_series)):
        print(f"  {label} delta sign runs (sample_id@t -> sample_id@t, count):")
        for sign, start_id, start_t, end_id, end_t, count in _sign_runs(series):
            print(f"    {sign} : {start_id}@{start_t:.3f}s -> {end_id}@{end_t:.3f}s  (n={count})")
        min_id, min_t, min_d = min(series, key=lambda item: item[2])
        max_id, max_t, max_d = max(series, key=lambda item: item[2])
        print(f"    min delta = {min_d:+.1f} km/h at {min_id}@{min_t:.3f}s")
        print(f"    max delta = {max_d:+.1f} km/h at {max_id}@{max_t:.3f}s")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    config, config_m0 = load_configs()
    rows = load_replay_rows()

    print("=" * 78)
    print("ALPHA-PLAYBACK OPEN-LOOP PLANT EXAM -- R-77-1 level launch")
    print("Harness: open-loop alpha playback; guidance/control/rotation bypassed.")
    print("No parameter is adjusted based on the results below.")
    print("=" * 78)

    aero_cfg = config["aerodynamics"]
    eta_law = aero_cfg["packed_lift_force_eta_law"]
    print("\nConstants sourced from the built runtime config (not hardcoded):")
    print(f"  fins_lateral_acceleration_g (A)      = {aero_cfg['fins_lateral_acceleration_g']}")
    print(f"  horizontal_fin_aoa_limit_deg (a_max) = {aero_cfg['horizontal_fin_aoa_limit_deg']}")
    print(f"  packed_lift_slope_scale (V_A k)      = {aero_cfg['packed_lift_slope_scale']}")
    print(f"  packed_lift_force_eta_law (V_B)      = {eta_law}")
    print(f"  M0 packed_lift_slope_scale (V_C k)   = {config_m0['aerodynamics']['packed_lift_slope_scale']}")
    print(f"  cx_k                                  = {aero_cfg['cx_k']}")
    print(f"  wing_area_multiplier                  = {config['geometry']['wing_area_multiplier']}")
    print(f"  caliber_m                             = {config['geometry']['caliber_m']}")
    print(f"  cx_vs_aoa                             = {aero_cfg['cx_vs_aoa']}")
    print(f"  base_indicated_speed_kmh              = {config['control']['base_indicated_speed_kmh']}")
    print(f"  base_indicated_speed_ratio_max        = {config['control'].get('base_indicated_speed_ratio_max')}")
    print(f"  gravity_mps2                          = {config['atmosphere']['gravity_mps2']}")
    stage0 = config["propulsion"]["stages"][0]
    print(f"  propulsion stage0                     = {stage0}")
    print(f"  initial_mass_kg                       = {config['geometry']['initial_mass_kg']}")
    print(f"  drag_model cx_vs_mach knots            = {len(config['drag_model']['cx_vs_mach'])}")

    print("\n" + "-" * 78)
    print("MODE 1 -- state-matched force-law exam (per-frame, no integration)")
    print("-" * 78)
    mode1_results = run_mode1(config, config_m0, rows)
    print_mode1_table(mode1_results)
    mode1_by_id = {r["sample_id"]: r for r in mode1_results}

    print("\nMean |error| per Mach band (frames with t>=0.5s and alpha>=1.0deg only):")
    print_mode1_band_summary(mode1_results)

    print("\n" + "-" * 78)
    print("MODE 1 cross-check -- analytic V_B vs real forces_for_state_h2 code path")
    print("-" * 78)
    run_cross_check(config, rows, mode1_by_id)

    print("\n" + "-" * 78)
    print("MODE 2 -- integrated open-loop energy exam (dt=0.005s, fixed 6300 m)")
    print("-" * 78)
    mode2 = run_mode2(config, rows)
    print(
        f"Fixed-altitude atmosphere at {FIXED_ALTITUDE_M:.0f} m: "
        f"T={mode2['t_k_fixed']:.3f} K, rho={mode2['rho_fixed']:.5f} kg/m3, "
        f"a={mode2['c_fixed']:.3f} m/s"
    )
    print_mode2_table(mode2, rows)

    print("\nExplicit speed deltas at t = 2.3, 4.3, 6.9, 7.8 s:")
    print_mode2_explicit_deltas(mode2, rows)

    print("\nDelta sign-run / min / max summary across all 46 frame times:")
    print_mode2_delta_summary(mode2, rows)

    print("\nBurnout peak speed:")
    game_peak_kmh = max(float(r["speed_kmh"]) for r in rows)
    game_peak_row = max(rows, key=lambda r: float(r["speed_kmh"]))
    print(f"  game measured peak   = {game_peak_kmh:.0f} km/h at t={game_peak_row['missile_flight_time_s']}s ({game_peak_row['sample_id']})")
    print(f"  D_1 shipped predicted peak    = {mode2['peak1_kmh']:.1f} km/h at t={mode2['peak1_t']:.3f}s")
    print(f"  D_2 diagnostic predicted peak = {mode2['peak2_kmh']:.1f} km/h at t={mode2['peak2_t']:.3f}s")

    print("\n" + "=" * 78)
    print("End of exam output.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
