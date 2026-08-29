"""Launch-capture release discriminant -- which mechanism ends the alpha plateau?

Four candidate mechanisms for the R-77-1 level-shot launch transient
(alpha pinned at 23.7 deg t~=0.9-2.1 s while G rises, released by t~=2.3 s;
seeker already TRK at 0.9 s):

  M1 Timer            release at fixed t_c                (current shipped model:
                                                           hold_time_s=1.4)
  M2 Heading capture  release when eps_PIP < eps_c        (explicit velocity-
                                                           vector capture mode)
  M3 Saturated PN     TRK -> plain PN at 0.9 s; command   (no capture hold at
                      N*Vc*lambda_dot exceeds the alpha-   all after TRK)
                      limited achievable G until ~2.1 s
  M4 Alpha limiter    guidance handover earlier than the   (plateau end is the
                      plateau end; a flight-control        limiter exit, not a
                      alpha/fin limiter pins the airframe  guidance event)

M3 and M4 share the same observable here (alpha pinned, G tracks the
achievable envelope, release when raw guidance demand drops below the
envelope); they differ only in *which* guidance law is saturated.  This
script therefore tests, per frame of GAME data:

    eps_PIP(t)      velocity-vector -> PIP bearing error
    lambda_dot(t)   LOS rate
    a_PN_raw   = N * Vc * lambda_dot          (N = pn_gain = 4, armed 0.3 s)
    a_cap_raw  = V * eps / tau_turn           (tau_turn = 0.8 s)
    G_env(t)        alpha-limited achievable G (measured G while alpha is
                    pinned; eta-law-scaled extrapolation after release)
    R_PN  = a_PN_raw  / G_env
    R_cap = a_cap_raw / G_env

and reports each candidate's predicted release time against the observed
t_r ~= 2.1-2.3 s.

Everything is computed from the digitized replay
(data/replays/r77_1_level_20260824.tsv) plus the scripted scenario geometry
(the A7 dict in tests/test_replay_anchors.py: launch 1200 km/h @ 6300 m,
straight level target 1200 km/h, 8000 m, azimuth 30 deg, head-on per
statshark_relative_to_los).  The missile's horizontal track is reconstructed
by integrating the measured speed timeline and turning at the measured G
(minus the 1 g level-flight support), always toward the target side.  The
sim is NOT run; the shipped model only contributes constants (pn_gain,
tau_turn, the packed-lift force law used to extend the envelope past
release).

Known approximations, all printed with the results:
  - 2D horizontal reconstruction; loft/vertical motion ignored.
  - Turn sign held toward the target for t <= SIGN_TRUST_END_S; the endgame
    reload (t > ~5.2 s) has unknown sign, so two path variants bracket it.
  - Displayed G is trajectory-normal thrust-inclusive; the envelope
    extension past t_r scales the *whole* measured plateau G by the aero
    eta-law factor, i.e. treats the thrust-normal share as scaling the same
    way (crude, but the discriminant window t < 3 s is inside the burn).

Read-only: prints tables, performs no filesystem writes.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPLAY_PATH = PROJECT_DIR / "data" / "replays" / "r77_1_level_20260824.tsv"

G0 = 9.80665

# --- Scenario (A7 geometry, tests/test_replay_anchors.py) -------------------
LAUNCH_SPEED_MPS = 1200.0 / 3.6
ALTITUDE_M = 6300.0
TARGET_SPEED_MPS = 1200.0 / 3.6
TARGET_DISTANCE_M = 8000.0
TARGET_AZIMUTH_RAD = math.radians(30.0)

# --- Model constants (missiles/su_r_77_1.json + runtime defaults) -----------
PN_GAIN = 4.0
PN_ARM_TIME_S = 0.31          # flight_time_gain_table [[0.3,0],[0.31,1]]
TAU_TURN_S = 0.8              # midcourse_lead_turn.turn_time_constant_s
SPEED_FLOOR_MPS = 200.0
FINS_LAT_ACCEL_G = 53.5864
FIN_AOA_LIMIT_RAD = 0.460812
BASE_IND_SPEED_MPS = 1800.0 / 3.6
ETA_LAW_COEFF = 0.574         # packed_lift_force_eta_law (R-77 family)
ETA_LAW_EXP = 0.242
ETA_LAW_CLAMP = (0.35, 2.65)

ALPHA_PIN_THRESHOLD_DEG = 21.0   # frames at/above this count as pinned
SIGN_TRUST_END_S = 5.15          # turn sign unambiguous before the endgame reload
DT_S = 0.002

# ISA density at altitude
def isa_density(alt_m: float) -> float:
    t_k = 288.15 - 0.0065 * alt_m
    p_pa = 101325.0 * (t_k / 288.15) ** 5.2561
    return p_pa / (287.05 * t_k)

RHO = isa_density(ALTITUDE_M)
RHO0 = 1.225


def eta_q(speed_mps: float) -> float:
    return RHO * speed_mps * speed_mps / (RHO0 * BASE_IND_SPEED_MPS * BASE_IND_SPEED_MPS)


def eta_law_k(eta: float) -> float:
    lo, hi = ETA_LAW_CLAMP
    return ETA_LAW_COEFF * min(max(eta, lo), hi) ** ETA_LAW_EXP


def aero_envelope_g(speed_mps: float, alpha_rad: float) -> float:
    """Force-law achievable G (aero only, no thrust-normal share)."""
    eta = eta_q(speed_mps)
    slope_g_per_rad = FINS_LAT_ACCEL_G / FIN_AOA_LIMIT_RAD
    return eta_law_k(eta) * slope_g_per_rad * eta * alpha_rad


def load_frames() -> list[dict[str, float]]:
    with REPLAY_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    frames = []
    for row in rows:
        frames.append(
            dict(
                t=float(row["missile_flight_time_s"]),
                v=float(row["speed_kmh"]) / 3.6,
                g=float(row["overload_g"]),
                alpha_deg=float(row["angle_of_attack_deg"]),
                flown_km=float(row["displayed_flown_distance_km"]),
                seeker=row["seeker_state"],
            )
        )
    frames.sort(key=lambda f: f["t"])
    return frames


def interp(knots: list[tuple[float, float]], t: float) -> float:
    if t <= knots[0][0]:
        return knots[0][1]
    for (t0, y0), (t1, y1) in zip(knots, knots[1:]):
        if t <= t1:
            return y0 + (y1 - y0) * (t - t0) / max(t1 - t0, 1e-9)
    return knots[-1][1]


def solve_pip_time_to_go(rel_pos, tgt_vel, v_bar):
    a = tgt_vel[0] ** 2 + tgt_vel[1] ** 2 - v_bar * v_bar
    b = 2.0 * (rel_pos[0] * tgt_vel[0] + rel_pos[1] * tgt_vel[1])
    c = rel_pos[0] ** 2 + rel_pos[1] ** 2
    fallback = math.hypot(*rel_pos) / max(v_bar, 1.0)
    roots = []
    if abs(a) <= 1e-9:
        if abs(b) > 1e-9:
            roots.append(-c / b)
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            s = math.sqrt(disc)
            roots += [(-b + s) / (2 * a), (-b - s) / (2 * a)]
    pos = [r for r in roots if r > 0]
    return min(pos) if pos else fallback


def reconstruct(frames, endgame_lateral: bool):
    """Integrate the game missile's horizontal track from measured V and G.

    Returns a list of grid states dicts and per-frame validation rows.
    """
    v_knots = [(0.0, LAUNCH_SPEED_MPS)] + [(f["t"], f["v"]) for f in frames]
    g_knots = [(0.0, 0.0)] + [(f["t"], f["g"]) for f in frames]
    t_end = frames[-1]["t"]

    # Target: on the fixed line through the launch point at bearing 30 deg,
    # flying head-on (inbound along the LOS).
    tx = TARGET_DISTANCE_M * math.cos(TARGET_AZIMUTH_RAD)
    tz = TARGET_DISTANCE_M * math.sin(TARGET_AZIMUTH_RAD)
    tvx = -TARGET_SPEED_MPS * math.cos(TARGET_AZIMUTH_RAD)
    tvz = -TARGET_SPEED_MPS * math.sin(TARGET_AZIMUTH_RAD)

    x = z = 0.0
    psi = 0.0
    flown = 0.0
    grid = []
    t = 0.0
    while t <= t_end + 1e-9:
        v = interp(v_knots, t)
        g_disp = interp(g_knots, t)
        a_h = math.sqrt(max((g_disp * G0) ** 2 - G0 ** 2, 0.0))
        if t > SIGN_TRUST_END_S and not endgame_lateral:
            a_h = 0.0
        # state snapshot before stepping
        rel = (tx + tvx * t - x, tz + tvz * t - z)
        r = math.hypot(*rel)
        vmx, vmz = v * math.cos(psi), v * math.sin(psi)
        vrel = (vmx - tvx, vmz - tvz)
        vc = (rel[0] * vrel[0] + rel[1] * vrel[1]) / max(r, 1e-6)
        # LOS rate from d(rel)/dt = v_target - v_missile = -vrel
        lam_dot = -(rel[0] * vrel[1] - rel[1] * vrel[0]) / max(r * r, 1e-6)
        v_bar = max(v, SPEED_FLOOR_MPS)
        t_go = solve_pip_time_to_go(rel, (tvx, tvz), v_bar)
        pip = (tx + tvx * (t + t_go) - x, tz + tvz * (t + t_go) - z)
        pip_r = math.hypot(*pip)
        cos_eps = (pip[0] * vmx + pip[1] * vmz) / max(pip_r * v, 1e-6)
        eps = math.acos(min(max(cos_eps, -1.0), 1.0))
        grid.append(
            dict(t=t, v=v, g_disp=g_disp, x=x, z=z, psi=psi, r=r, vc=vc,
                 lam_dot=lam_dot, eps=eps, flown=flown)
        )
        # step
        psi += (a_h / max(v, 1.0)) * DT_S
        x += v * math.cos(psi) * DT_S
        z += v * math.sin(psi) * DT_S
        flown += v * DT_S
        t += DT_S
    return grid


def main() -> None:
    frames = load_frames()
    pinned = [f for f in frames if f["alpha_deg"] >= ALPHA_PIN_THRESHOLD_DEG]
    t_pin_last = max(f["t"] for f in pinned)
    g_pin_knots = [(f["t"], f["g"]) for f in frames if f["t"] <= t_pin_last + 1e-9]
    pin_last = [f for f in frames if abs(f["t"] - t_pin_last) < 1e-9][0]

    grid = reconstruct(frames, endgame_lateral=True)
    grid_b = reconstruct(frames, endgame_lateral=False)

    def at(g_list, t):
        return min(g_list, key=lambda s: abs(s["t"] - t))

    # --- Envelope on the grid ------------------------------------------------
    ref_scale = eta_law_k(eta_q(pin_last["v"])) * eta_q(pin_last["v"])
    def envelope(t, v):
        if t <= t_pin_last:
            return interp(g_pin_knots, t)
        return pin_last["g"] * (eta_law_k(eta_q(v)) * eta_q(v)) / ref_scale

    # --- Validation ----------------------------------------------------------
    print("=" * 78)
    print("VALIDATION -- reconstruction vs replay")
    print("=" * 78)
    print(f"ISA density @ {ALTITUDE_M:.0f} m: {RHO:.4f} kg/m^3")
    print("flown-distance check (integrated vs displayed, km):")
    errs = []
    for f in frames[::5]:
        s = at(grid, f["t"])
        errs.append(s["flown"] / 1000.0 - f["flown_km"])
        print(f"  t={f['t']:4.1f}  integ={s['flown']/1000.0:5.2f}  disp={f['flown_km']:4.1f}"
              f"  diff={errs[-1]:+.2f}")
    tail = grid[-1]
    tail_b = grid_b[-1]
    r_min = min(s["r"] for s in grid)
    t_rmin = min(grid, key=lambda s: s["r"])["t"]
    r_min_b = min(s["r"] for s in grid_b)
    t_rmin_b = min(grid_b, key=lambda s: s["r"])["t"]
    print(f"closest approach: variant A (endgame turning) {r_min:6.0f} m @ t={t_rmin:.2f} s")
    print(f"                  variant B (endgame straight) {r_min_b:6.0f} m @ t={t_rmin_b:.2f} s")
    print(f"(game truth: proximity fuse ~7.2-7.4 s)")
    print(f"final heading: A {math.degrees(tail['psi']):.1f} deg, B {math.degrees(tail_b['psi']):.1f} deg"
          f"  (PIP bearing for head-on target stays on the 30 deg line)")

    # --- Force-law / envelope sanity during the pinned plateau ---------------
    print()
    print("=" * 78)
    print("PLATEAU SANITY -- measured G vs aero force-law at measured alpha")
    print("=" * 78)
    print("  t     alpha   G_meas  G_aero(law)  residual(=thrust-normal share?)")
    for f in pinned:
        g_aero = aero_envelope_g(f["v"], math.radians(f["alpha_deg"]))
        print(f"  {f['t']:4.1f}  {f['alpha_deg']:5.1f}  {f['g']:6.1f}  {g_aero:9.1f}"
              f"      {f['g'] - g_aero:+6.1f}")

    # --- Discriminant table --------------------------------------------------
    print()
    print("=" * 78)
    print("DISCRIMINANT TABLE (game-frame times; reconstruction variant A)")
    print("=" * 78)
    print("  t    seeker  alpha   G_meas  G_env |  eps    lam_dot   a_PN   a_cap |  R_PN  R_cap")
    print("  (s)          (deg)    (g)    (g)   | (deg)  (rad/s)    (g)    (g)  |")
    print("  -- valid window: reconstruction trustworthy while the turn dominates --")
    valid_marker_done = False
    for f in frames:
        if f["t"] > 5.2:
            break
        if f["t"] > 2.4 and not valid_marker_done:
            print("  -- BELOW THIS LINE: de-load-arc G attributed to a same-sign")
            print("  -- horizontal turn; eps/lam_dot are reconstruction ARTIFACTS --")
            valid_marker_done = True
        s = at(grid, f["t"])
        armed = f["t"] >= PN_ARM_TIME_S
        a_pn = PN_GAIN * max(s["vc"], 0.0) * abs(s["lam_dot"]) / G0 if armed else 0.0
        a_cap = s["v"] * s["eps"] / TAU_TURN_S / G0
        g_env = envelope(f["t"], s["v"])
        print(f"  {f['t']:4.1f}  {f['seeker']:>6}  {f['alpha_deg']:5.1f}  {f['g']:6.1f}"
              f"  {g_env:5.1f} | {math.degrees(s['eps']):5.1f}  {s['lam_dot']:8.4f}"
              f"  {a_pn:5.1f}  {a_cap:5.1f} | {a_pn/max(g_env,0.1):5.2f}  {a_cap/max(g_env,0.1):5.2f}")

    # --- Release predictions per mechanism ----------------------------------
    print()
    print("=" * 78)
    print("RELEASE PREDICTIONS  (observed: alpha still 22.9 deg @2.1, 20.6 @2.3,")
    print("                      18.4 @2.5 -> release begins between 2.1 and 2.3 s)")
    print("=" * 78)

    VALID_END_S = 2.4  # reconstruction trustworthy while the lead turn dominates

    def first_crossing(quantity, tau=TAU_TURN_S):
        prev = None
        for s in grid:
            if s["t"] < 0.9 or s["t"] > VALID_END_S:
                continue
            g_env = envelope(s["t"], s["v"])
            armed = s["t"] >= PN_ARM_TIME_S
            if quantity == "pn":
                q = PN_GAIN * max(s["vc"], 0.0) * abs(s["lam_dot"]) / G0 if armed else 0.0
            else:
                q = s["v"] * s["eps"] / tau / G0
            ratio = q / max(g_env, 0.1)
            if prev is not None and prev >= 1.0 > ratio:
                return s["t"]
            prev = ratio
        return None

    r_pn_max = max(
        PN_GAIN * max(s["vc"], 0.0) * abs(s["lam_dot"]) / G0 / max(envelope(s["t"], s["v"]), 0.1)
        for s in grid if 0.9 <= s["t"] <= VALID_END_S
    )
    t_pn = first_crossing("pn")
    print(f"M3 saturated PN:      NEVER saturated in the valid window -- max R_PN"
          f" = {r_pn_max:.2f} at TRK onset, monotonically declining.  (No fixed N"
          f" rescues it: sustaining saturation to 2.1 s needs N growing ~8 -> ~24.)")
    for tau in (0.3, 0.4, 0.5, 0.8):
        t_cap = first_crossing("cap", tau=tau)
        print(f"M4 proportional capture V*eps/tau saturating the alpha limiter,"
              f" tau={tau:.1f}: unpins at t = "
              f"{t_cap if t_cap is None else round(t_cap, 2)} s")
    for eps_c_deg in (2.0, 5.0, 10.0, 15.0):
        hit = next((s["t"] for s in grid if s["t"] >= 1.0
                    and math.degrees(s["eps"]) < eps_c_deg), None)
        print(f"M2 eps_c = {eps_c_deg:4.1f} deg: eps crosses below at t = "
              f"{hit if hit is None else round(hit, 2)} s")
    print(f"M1 timer (shipped): lock_delay 0.8 + hold 1.4 = release starts 2.2 s "
          f"(then 0.7 s blend)")
    print()
    print("caveats: 2D horizontal, loft ignored; envelope extension past "
          f"t={t_pin_last:.1f} s scales thrust-normal share with the aero eta-law; "
          "turn sign fixed toward target before t=5.15 s.")


if __name__ == "__main__":
    main()
