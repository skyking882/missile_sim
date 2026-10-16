"""Loft target convention changes its bias, with identical gate inputs."""
import json
import math
from pathlib import Path

import pytest

from aim120_model.effective_surface_core import EffectiveSurfaceGame

ROOT = Path(__file__).resolve().parents[1]


def game(convention=None, *, target_height=1000.0, target_velocity=(350.0, 0.0, 0.0)):
    raw = json.loads((ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx").read_text())["rocket"]
    config = json.loads((ROOT / "config/effective_surface_defaults.json").read_text())
    config.pop("loft_pitch_target_convention")
    if convention is not None:
        config["loft_pitch_target_convention"] = convention
    raw["guidance"]["guidanceAutopilot"].update(propNavMult=0.0, reqAccelMax=1000.0)
    creation = dict(position_xyz_m=(0.0, 8000.0, 0.0), velocity_xyz_mps=(350.0, 0.0, 0.0),
                    mass_source_kg=raw["mass"], target_position_xyz_m=(20000.0, 8000.0+target_height, 0.0),
                    target_velocity_xyz_mps=target_velocity, scenario_exact=dict(dt=1/48, loft_state="config"))
    model = EffectiveSurfaceGame(raw, config, creation)
    state = (0.0, 8000.0, 0.0, 350.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    return model, state


@pytest.mark.parametrize("height", [-1000.0, 0.0, 1000.0])
def test_missing_and_explicit_legacy_convention_are_exactly_equal(height):
    missing, state = game(target_height=height)
    explicit, _ = game("plus_los", target_height=height)
    assert missing.guidance(0.0, state) == explicit.guidance(0.0, state)
    assert missing.loft_phase == explicit.loft_phase


@pytest.mark.parametrize("height", [-1000.0, 1000.0])
def test_absolute_removes_only_los_elevation_from_pitch_bias(height):
    legacy, state = game("plus_los", target_height=height)
    absolute, _ = game("absolute", target_height=height)
    old = legacy.guidance(0.0, state)
    new = absolute.guidance(0.0, state)
    multiplier = legacy.p["loft_common_scale"] * legacy.autopilot["loftAngleToAccelMult"]
    los = math.degrees(math.atan2(height, 20000.0))
    assert old[1] - new[1] == pytest.approx(multiplier * los)
    assert old[0] == new[0] == 0.0
    assert old[2] == new[2] == 0.0
    assert legacy.loft_phase == absolute.loft_phase == "climb"


def test_gate_failure_and_permanent_exit_are_identical_for_both_conventions():
    for convention in ("plus_los", "absolute"):
        model, state = game(convention)
        model.guidance(0.0, state)
        assert model.loft_phase == "climb"
        # This target-motion LOS rate exceeds the original 0.5 deg/s gate.
        model.target_velocity = (350.0, 1000.0, 0.0)
        model.guidance(0.0, state)
        assert model.loft_phase == "complete"
        model.target_velocity = (350.0, 0.0, 0.0)
        assert model.guidance(0.0, state) == (0.0, 0.0, 0.0)
        assert model.loft_phase == "complete"
        failed, state = game(convention, target_height=-5000.0)
        assert failed.guidance(0.0, state) == (0.0, 0.0, 0.0)
        assert failed.loft_phase == "complete"


def test_unknown_pitch_target_convention_is_rejected():
    with pytest.raises(ValueError, match="Unknown loft pitch target convention"):
        game("relative")
