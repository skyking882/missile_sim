import math

import pytest

from aim120_model.thrust_vectoring import thrust_vectoring


@pytest.mark.parametrize("control", [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
def test_tail_force_turns_nose_towards_shared_command(control):
    result = thrust_vectoring(control, 100.0, 0.1, 0.5)
    _, fy, fz = result.thrust_body_n
    mx, my, mz = result.moment_body_nm
    yaw, pitch = control
    assert mx == 0.0
    if yaw:
        assert yaw * fz < 0.0
        assert yaw * my < 0.0
        assert fy == mz == 0.0
    else:
        assert pitch * fy < 0.0
        assert pitch * mz > 0.0
        assert fz == my == 0.0
    assert math.hypot(my, mz) == pytest.approx(0.5 * math.hypot(fy, fz))


@pytest.mark.parametrize("control", [(0.0, 0.0), (0.7, -0.4), (1.0, 1.0), (-1.0, 1.0)])
def test_force_magnitude_plane_angles_and_aero_command(control):
    result = thrust_vectoring(control, 100.0, 0.045, 0.5)
    fx, fy, fz = result.thrust_body_n
    assert math.hypot(fx, fy, fz) == pytest.approx(100.0)
    assert fx > 0.0
    assert math.atan2(-fz, fx) == pytest.approx(control[0] * 0.045)
    assert math.atan2(-fy, fx) == pytest.approx(control[1] * 0.045)
    assert result.steering_angles_rad == pytest.approx(tuple(u * 0.045 for u in control))
    assert result.aerodynamic_control == control
    if control == (1.0, 1.0):
        assert math.atan2(math.hypot(fy, fz), fx) > 0.045


@pytest.mark.parametrize("control,angle", [((0.0, 0.0), 0.1), ((0.7, -0.4), 0.0)])
def test_zero_command_or_zero_angle_is_axial(control, angle):
    result = thrust_vectoring(control, 100.0, angle, 0.5)
    assert result.thrust_body_n == (100.0, 0.0, 0.0)
    assert result.moment_body_nm == (0.0, 0.0, 0.0)
    assert result.aerodynamic_control == control


def test_burnout_removes_propulsion_but_preserves_aero_command():
    control = (0.7, -0.4)
    result = thrust_vectoring(control, 0.0, 0.045, 0.5)
    assert result.thrust_body_n == (0.0, 0.0, 0.0)
    assert result.moment_body_nm == (0.0, 0.0, 0.0)
    assert result.aerodynamic_control == control


def test_station_at_centre_of_mass_produces_no_moment():
    result = thrust_vectoring((1.0, 1.0), 100.0, 0.1, 0.0)
    assert result.thrust_body_n[1] < 0.0
    assert result.thrust_body_n[2] < 0.0
    assert result.moment_body_nm == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("changes", [
    {"control": (1.01, 0.0)},
    {"control": (0.0, -1.01)},
    {"control": (math.nan, 0.0)},
    {"control": (0.0, math.inf)},
    {"control": (0.0,)},
    {"thrust_n": -1.0},
    {"thrust_n": math.inf},
    {"thrust_n": math.nan},
    {"max_angle_rad": -0.01},
    {"max_angle_rad": math.pi / 2},
    {"max_angle_rad": math.inf},
    {"max_angle_rad": math.nan},
    {"aft_distance_m": -0.1},
    {"aft_distance_m": math.inf},
    {"aft_distance_m": math.nan},
])
def test_invalid_input_is_rejected(changes):
    arguments = dict(control=(0.0, 0.0), thrust_n=100.0, max_angle_rad=0.045, aft_distance_m=0.5)
    arguments.update(changes)
    with pytest.raises(ValueError):
        thrust_vectoring(**arguments)
