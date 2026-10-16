"""Controller-only time series; no flight integration or parameter fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aim120_model.effective_surface_core import EffectiveSurfaceGame


@pytest.mark.parametrize("convention", ["scheduled_terms", "post_gain"])
def test_integral_memory_when_speed_gain_falls(convention):
    model = EffectiveSurfaceGame.__new__(EffectiveSurfaceGame)
    model.config = dict(pid_error_units="m/s^2", pid_error_frame="velocity_normal_projected_body",
                        pid_pressure_gain_convention="floor_one", pid_gain_convention=convention)
    model.autopilot = dict(_age_gain=[[0., 1.]], baseIndSpeed=1800., accelControlProp=.002125,
                           accelControlIntg=.006481, accelControlDiff=.000023, accelControlIntgLim=1.)
    model.atmosphere = SimpleNamespace(sample=lambda altitude: SimpleNamespace(density_kg_m3=1.225))
    model.dt = 1 / 48
    model.integral = [0., 0.]
    model.previous_error = [0., 0.]
    model.previous_acceleration = (0., 0., 0.)
    expected_i = [0., 0.]
    previous = [0., 0.]
    memory = []
    for k, (speed, errors) in enumerate([(250., (8., -4.)), (400., (0., 0.)),
                                         (700., (0., 0.)), (700., (10000., -10000.))]):
        model.guidance = lambda t, state: (0., errors[1], errors[0])
        state = (0., 8000., 0., speed, 0., 0., 0., 0., 0., 1., 0., 0., 0.)
        gain = max(1., (500. / speed) ** 2)
        expected = []
        for j, error in enumerate(errors):
            multiplier = gain if convention == "scheduled_terms" else 1.
            expected_i[j] = max(-1., min(1., expected_i[j] + multiplier * .006481 * error * model.dt))
            pd = .002125 * error + .000023 * (error - previous[j]) / model.dt
            value = gain * pd + expected_i[j] if convention == "scheduled_terms" else gain * (pd + expected_i[j])
            expected.append(max(-1., min(1., value)))
        assert model.update_control(k * model.dt, state) == pytest.approx(expected)
        assert model.integral == pytest.approx(expected_i)
        memory.append(tuple(model.integral))
        previous = list(errors)
    assert memory[0] == memory[1] == memory[2]
    assert memory[-1] == (1., -1.)
    if convention == "scheduled_terms":
        # Gain falls 4 -> 1.5625 -> 1 with no error; output-unit I remains intact.
        assert memory[2][0] == pytest.approx(4. * .006481 * 8. / 48)


def test_shared_runtime_defaults_select_output_unit_integral():
    defaults = json.loads((Path(__file__).resolve().parents[1] / "config/effective_surface_defaults.json").read_text())
    assert defaults["pid_gain_convention"] == "scheduled_terms"
