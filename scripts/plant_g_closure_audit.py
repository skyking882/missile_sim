"""Plant alpha->G closure audit against the R-77-1 level replay.

Question (2026-08-26 review): the closed-loop model produced ~18 g at
alpha ~18.5 deg where the game needs ~23 deg for 17.8 g -- so before any
further guidance calibration against alpha points, the simulator's actual
force path must be shown to close against the replay at the GAME's own
operating points (M, q, m, T, alpha all taken from the replay/propulsion
timeline, not from a closed-loop run whose speed runs hot).

For six replay frames this script builds a SimState with the game's TAS at
6300 m ISA and pitch offset = displayed alpha (level velocity, so
pitch_alpha == alpha, total incidence == displayed magnitude), then calls
forces_for_state_h2 -- the same force path the simulator integrates -- and
prints the decomposition:

    aero-only lateral G   (body basis; packed-lift force channel + any body
                           lift the plant adds)
    thrust-normal G       (T*sin(alpha)/(m*g))
    trajectory-normal G   (thrust-inclusive magnitude the HUD comparison uses)
    standalone law G      (capture_alpha_envelope_g scaled to this alpha --
                           the formula the release discriminant closed with)

and solves the inverse alpha_inv(G_game) by bisection on the same path.

Closure gate (docs/CAPTURE_RELEASE_DISCRIMINANT.md showed the standalone
law + thrust closes to ~+-0.5 g): |G_model(alpha_game) - G_game| < 1 g.
Read-only; prints tables, writes nothing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from aim120_model.dynamics import SimState  # noqa: E402
from aim120_model.h2_dynamics import capture_alpha_envelope_g, forces_for_state_h2  # noqa: E402
from aim120_model.propulsion import PiecewisePropulsion  # noqa: E402
from missile_gui.library import scan_library  # noqa: E402

ALTITUDE_M = 6300.0
ALPHA_MAX_DEG = 23.7

# (flight_time_s, speed_kmh, alpha_deg, displayed_G) from
# data/replays/r77_1_level_20260824.tsv
FRAMES = [
    (0.9, 1529.0, 23.2, 13.4),
    (1.5, 1662.0, 23.7, 15.8),
    (2.1, 1802.0, 22.9, 17.8),
    (2.5, 1927.0, 18.4, 16.2),
    (2.9, 2064.0, 14.5, 14.6),
    (3.4, 2245.0, 10.5, 12.3),
]


def load_config() -> dict:
    profiles, errors = scan_library(PROJECT_DIR / "missiles", PROJECT_DIR)
    assert not errors, errors[:3]
    profile = next(p for p in profiles if p["missile_id"] == "su_r_77_1")
    return profile["_model_config"]


def model_g(config, propulsion, time_s, speed_mps, alpha_rad, powered=True):
    """Trajectory-normal G (and components) from the simulator's force path."""

    mass = propulsion.mass_at(time_s, powered=powered)
    state = SimState(
        (0.0, ALTITUDE_M, 0.0),
        (speed_mps, 0.0, 0.0),
        alpha_rad,  # pitch above a level velocity vector -> pitch_alpha = alpha
        0.0,
        0.0,
        0.0,
        mass,
        actual_pitch_fin_angle_rad=alpha_rad,  # trim: delta = alpha in this plant
        actual_yaw_fin_angle_rad=0.0,
    )
    d = forces_for_state_h2(state, time_s, config, propulsion, powered)
    traj = math.hypot(
        d.trajectory_pitch_normal_acceleration_g, d.trajectory_yaw_normal_acceleration_g
    )
    body_aero = math.hypot(d.pitch_normal_acceleration_g, d.yaw_normal_acceleration_g)
    gravity = float(config["atmosphere"]["gravity_mps2"])
    thrust_normal = d.propulsion.thrust_n * math.sin(alpha_rad) / (mass * gravity)
    return dict(
        traj_g=traj,
        body_aero_g=body_aero,
        thrust_normal_g=thrust_normal,
        thrust_n=d.propulsion.thrust_n,
        mass=mass,
        q_pa=d.aero.dynamic_pressure_pa,
        mach=d.aero.mach,
        alpha_check_deg=math.degrees(d.aero.angle_of_attack_rad),
    )


def invert_alpha_deg(config, propulsion, time_s, speed_mps, g_target):
    lo, hi = 0.0, math.radians(35.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if model_g(config, propulsion, time_s, speed_mps, mid)["traj_g"] < g_target:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


def main() -> None:
    config = load_config()
    propulsion = PiecewisePropulsion.from_config(config)
    print("=" * 100)
    print("PLANT alpha->G CLOSURE AUDIT -- su_r_77_1 force path at GAME operating points (6300 m ISA)")
    print("=" * 100)
    print(
        "  t     V_kmh  Mach   q_kPa  mass   T_kN | alpha  G_game | G_model  aero  T-norm | law_G | dG     alpha_inv"
    )
    worst = 0.0
    for t, v_kmh, a_deg, g_game in FRAMES:
        v = v_kmh / 3.6
        a = math.radians(a_deg)
        r = model_g(config, propulsion, t, v, a)
        law_aero = capture_alpha_envelope_g(r["q_pa"], r["mass"], 0.0, a, config)
        law_total = capture_alpha_envelope_g(r["q_pa"], r["mass"], r["thrust_n"], a, config)
        dg = r["traj_g"] - g_game
        worst = max(worst, abs(dg))
        a_inv = invert_alpha_deg(config, propulsion, t, v, g_game)
        print(
            f"  {t:4.1f}  {v_kmh:6.0f}  {r['mach']:4.2f}  {r['q_pa']/1e3:6.1f}  {r['mass']:5.1f}"
            f"  {r['thrust_n']/1e3:5.1f} | {a_deg:5.1f}  {g_game:6.1f} | {r['traj_g']:7.2f}"
            f"  {r['body_aero_g']:5.2f}  {r['thrust_normal_g']:6.2f} | {law_total:5.2f}"
            f" | {dg:+5.2f}  {a_inv:8.2f}"
        )
    print()
    print(f"closure gate |dG| < 1 g: worst |dG| = {worst:.2f} g -> {'PASS' if worst < 1.0 else 'FAIL'}")
    print(
        "columns: G_model = trajectory-normal (thrust-inclusive) magnitude from forces_for_state_h2;"
    )
    print(
        "  aero = body-basis aero-only; T-norm = T*sin(alpha)/(m*g); law_G = standalone"
    )
    print(
        "  capture_alpha_envelope_g at this alpha (the discriminant's closing formula);"
    )
    print("  alpha_inv = alpha the PLANT needs for G_game.  Game plateau alpha ~= 23.2-23.7 deg.")


if __name__ == "__main__":
    main()
