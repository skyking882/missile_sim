from pathlib import Path

from aim120_model.config import load_model_config
from aim120_model.dynamics import SimState
from aim120_model.events import event_candidates
from aim120_model.target import TargetState


CONFIG = load_model_config(Path(__file__).parents[1] / "configs" / "aim120a_statshark.yaml")


def state(position):
    return SimState(tuple(position), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 100.0)


def test_fuse_is_interpolated_between_samples():
    previous = state((0.0, 1000.0, 0.0))
    current = state((0.0, 1000.0, 0.0))
    target0 = TargetState((20.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    target1 = TargetState((0.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target0, target1, 0.0, 1.0, CONFIG, (0.0, 1000.0, 0.0))
    fuse = next(candidate for candidate in candidates if candidate.event_type == "fuse")
    assert abs(fuse.fraction - 0.4) < 1e-12


def test_fuse_detects_a_complete_between_sample_pass():
    previous = state((0.0, 1000.0, 0.0))
    current = state((0.0, 1000.0, 0.0))
    target0 = TargetState((20.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    target1 = TargetState((-20.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target0, target1, 0.0, 1.0, CONFIG, (0.0, 1000.0, 0.0))
    fuse = next(candidate for candidate in candidates if candidate.event_type == "fuse")
    assert abs(fuse.fraction - 0.2) < 1e-12


def test_fuse_detects_a_tangent_between_sample_pass():
    previous = state((0.0, 1000.0, 0.0))
    current = state((0.0, 1000.0, 0.0))
    radius = CONFIG["guidance"]["proximity_radius_m"]
    target0 = TargetState((20.0, 1000.0 + radius, 0.0), (0.0, 0.0, 0.0))
    target1 = TargetState((-20.0, 1000.0 + radius, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target0, target1, 0.0, 1.0, CONFIG, (0.0, 1000.0, 0.0))
    fuse = next(candidate for candidate in candidates if candidate.event_type == "fuse")
    assert abs(fuse.fraction - 0.5) < 1e-12


def test_fuse_does_not_trigger_for_a_between_sample_near_miss():
    previous = state((0.0, 1000.0, 0.0))
    current = state((0.0, 1000.0, 0.0))
    radius = CONFIG["guidance"]["proximity_radius_m"]
    target0 = TargetState((20.0, 1000.0 + radius + 1e-4, 0.0), (0.0, 0.0, 0.0))
    target1 = TargetState((-20.0, 1000.0 + radius + 1e-4, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target0, target1, 0.0, 1.0, CONFIG, (0.0, 1000.0, 0.0))
    assert all(candidate.event_type != "fuse" for candidate in candidates)


def test_last_curve_sample_is_not_automatically_a_fuse():
    previous = state((0.0, 1000.0, 0.0))
    current = state((0.0, 1000.0, 0.0))
    target0 = TargetState((100.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    target1 = TargetState((100.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target0, target1, 0.0, 1.0, CONFIG, (0.0, 1000.0, 0.0))
    assert all(candidate.event_type != "fuse" for candidate in candidates)


def test_ground_crossing_is_detected():
    previous = state((0.0, 10.0, 0.0))
    current = state((0.0, -10.0, 0.0))
    target = TargetState((1000.0, 1000.0, 0.0), (0.0, 0.0, 0.0))
    candidates = event_candidates(previous, current, target, target, 0.0, 1.0, CONFIG, (0.0, 10.0, 0.0))
    ground = next(candidate for candidate in candidates if candidate.event_type == "ground")
    assert abs(ground.fraction - 0.5) < 1e-12
