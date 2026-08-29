"""Frozen plant alpha->G closure gate (2026-08-26c review).

Mandated after the v0.1 washout episode: the closed-loop model's hot speed
(+200-330 km/h, the open drag-law thread) makes any alpha-vs-alpha comparison
against the replay double-count the dynamic-pressure error, so before any
guidance time constant is calibrated against replay alpha points, the FORCE
PATH itself must close at the game's own operating points.  This test freezes
that closure: six R-77-1 level-replay frames, game TAS at 6300 m ISA, game
alpha as pure pitch incidence, mass/thrust from the model propulsion
timeline, |G_model - G_game| < 1 g through forces_for_state_h2 (the audit run
2026-08-26 measured worst |dG| = 0.17 g and inverse alpha within 0.3 deg;
see scripts/plant_g_closure_audit.py for the full decomposition).

If this test fails after a force-side change, the closure is broken -- fix
the force change or re-run the audit and re-justify; do NOT absorb the gap
into guidance parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

from aim120_model.dynamics import SimState
from aim120_model.h2_dynamics import forces_for_state_h2
from aim120_model.propulsion import PiecewisePropulsion
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]
ALTITUDE_M = 6300.0

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


def _r77_1_config() -> dict:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    return next(p for p in profiles if p["missile_id"] == "su_r_77_1")["_model_config"]


def _trajectory_g(config, propulsion, time_s, speed_mps, alpha_rad) -> float:
    state = SimState(
        (0.0, ALTITUDE_M, 0.0),
        (speed_mps, 0.0, 0.0),
        alpha_rad,
        0.0,
        0.0,
        0.0,
        propulsion.mass_at(time_s, powered=True),
        actual_pitch_fin_angle_rad=alpha_rad,
        actual_yaw_fin_angle_rad=0.0,
    )
    d = forces_for_state_h2(state, time_s, config, propulsion, powered=True)
    return math.hypot(
        d.trajectory_pitch_normal_acceleration_g, d.trajectory_yaw_normal_acceleration_g
    )


def test_plant_g_closure_at_game_operating_points() -> None:
    config = _r77_1_config()
    propulsion = PiecewisePropulsion.from_config(config)
    for time_s, speed_kmh, alpha_deg, g_game in FRAMES:
        g_model = _trajectory_g(
            config, propulsion, time_s, speed_kmh / 3.6, math.radians(alpha_deg)
        )
        assert abs(g_model - g_game) < 1.0, (
            f"t={time_s}: plant G {g_model:.2f} vs game {g_game} "
            "(alpha->G closure broken; see scripts/plant_g_closure_audit.py)"
        )


def test_plant_alpha_inverse_matches_game_plateau() -> None:
    """The alpha the plant needs for the plateau G must be the game's alpha."""

    config = _r77_1_config()
    propulsion = PiecewisePropulsion.from_config(config)
    for time_s, speed_kmh, alpha_deg, g_game in FRAMES[:3]:  # plateau frames
        lo, hi = 0.0, math.radians(35.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if _trajectory_g(config, propulsion, time_s, speed_kmh / 3.6, mid) < g_game:
                lo = mid
            else:
                hi = mid
        alpha_inv_deg = math.degrees(0.5 * (lo + hi))
        assert abs(alpha_inv_deg - alpha_deg) < 0.6, (
            f"t={time_s}: inverse alpha {alpha_inv_deg:.2f} vs game {alpha_deg}"
        )
