import math

from aim120_model.target import TargetModel


def make_initial(course_deg, reference="absolute_world", azimuth_deg=38.0):
    return {
        "initial_target_distance_m": 15000.0,
        "target_azimuth_deg": azimuth_deg,
        "target_altitude_m": 6500.0,
        "target_speed_kmh": 1200.0,
        "target_course_deg": course_deg,
        "target_course_reference": reference,
        "target_vertical_course_deg": 0.0,
        "target_constant_g_turn": 0.0,
    }


def horizontal_radial_dot(model):
    position = model.initial_state.position
    velocity = model.initial_state.velocity
    return position[0] * velocity[0] + position[2] * velocity[2]


def test_absolute_world_course_remains_backward_compatible():
    model = TargetModel(make_initial(0.0), 9.80665)
    assert model.initial_state.velocity[0] > 0.0
    assert abs(model.initial_state.velocity[2]) < 1e-9


def test_statshark_zero_course_is_head_on_for_off_axis_target():
    model = TargetModel(make_initial(0.0, "statshark_relative_to_los"), 9.80665)
    assert horizontal_radial_dot(model) < 0.0
    position = model.initial_state.position
    velocity = model.initial_state.velocity
    horizontal_range = math.hypot(position[0], position[2])
    horizontal_speed = math.hypot(velocity[0], velocity[2])
    assert abs(velocity[0] / horizontal_speed + position[0] / horizontal_range) < 1e-12
    assert abs(velocity[2] / horizontal_speed + position[2] / horizontal_range) < 1e-12


def test_statshark_180_course_is_tail_away():
    model = TargetModel(make_initial(180.0, "statshark_relative_to_los"), 9.80665)
    assert horizontal_radial_dot(model) > 0.0


def test_unknown_course_reference_is_rejected():
    try:
        TargetModel(make_initial(0.0, "unknown"), 9.80665)
    except ValueError as exc:
        assert "unknown target_course_reference" in str(exc)
    else:
        raise AssertionError("unknown course reference was accepted")
