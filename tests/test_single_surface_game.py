"""Focused invariants for the opt-in game candidate, independent of replay fit."""
from __future__ import annotations

import math

import pytest

from aim120_model.aerodynamics import quaternion_from_pitch_yaw
from aim120_model.dynamics import SimState
from aim120_model.math3d import dot
from aim120_model.single_surface_game import SingleSurfaceGame
from aim120_model.target import TargetState


def config():
    return {"geometry": {"initial_mass_kg":190., "caliber_m":.2, "length_m":3.71,
        "wing_area_multiplier":1.5},
        "aerodynamics": {"distance_cm_to_stabilizer_m":.4,"cx_k":1.4,"cx_vs_aoa":9.,
            "horizontal_fin_aoa_limit_deg":40.,"vertical_fin_aoa_limit_deg":40.},
        "atmosphere": {"gravity_mps2":9.80665,"wind_mps":(0.,0.,0.)},
        "drag_model": {"shape_mode":"interpolated_cx_1943_x1_10",
            "alpha_drag_area_basis_mode":"caliber_area","alpha_drag_mach_shape":False},
        "propulsion":{"stages":[{"name":"boost","duration_s":.75,"thrust_n":23835.,
            "mass_lost_kg":71.,"isp_s":239.626}]},
        "control":{"pid":{"p":.002125,"i":.006481,"d":.000023,"integral_limit":1.}},
        "guidance":{"pn_gain":6.,"lofting_enabled":False,"maximum_lateral_acceleration_g":50.,
            "flight_time_gain_table":[[.3,0.],[.31,1.]],"time_to_hit_gain_table":[[10.,1.],[25.,.8]],
            "time_to_go_mode":"closing_or_relative_speed","time_to_go_relative_speed_weight":1.,
            "proximity_radius_m":10.}}


def state(alpha=0.):
    return SimState((0.,6300.,0.),(350.,0.,0.),alpha,0.,0.,0.,190.,
        orientation_quaternion=quaternion_from_pitch_yaw(alpha,0.))


@pytest.mark.parametrize("basis", ("wind_linear","body_linear"))
@pytest.mark.parametrize("alpha", (math.radians(40),-math.radians(40)))
def test_force_basis_has_one_wind_drag_and_correct_thrust_projection(basis,alpha):
    model = SingleSurfaceGame(config(),force_basis=basis)
    s = model.initialize(state(alpha))
    f = model.evaluate(s,0.,(0.,0.))
    assert dot(f.wind_lift_force,f.wind_basis.forward) == pytest.approx(0.,abs=1e-9)
    assert dot(f.aerodynamic_force,f.wind_basis.forward) == pytest.approx(-f.drag_n,abs=1e-9)
    # Wind-normal specific force includes thrust but excludes gravity.
    expected = (dot(f.wind_lift_force,f.wind_basis.up)+f.thrust*math.sin(alpha))/(f.mass*9.80665)
    assert f.feedback_g[0] == pytest.approx(expected,abs=1e-10)
    expected_lift = f.qbar*model.area*model.coefficients.ca*alpha
    if basis == "body_linear":
        expected_lift *= math.cos(alpha)
    assert dot(f.wind_lift_force,f.wind_basis.up) == pytest.approx(expected_lift,rel=1e-12)


def test_burnout_does_not_inject_velocity_and_mass_is_continuous():
    model = SingleSurfaceGame(config(),plant_substeps=8)
    s = model.initialize(state())
    before = model.evaluate(s,math.nextafter(.75,0.),(0.,0.))
    after = model.evaluate(s,.75,(0.,0.))
    assert before.mass == pytest.approx(after.mass,abs=1e-10)
    assert after.mass == 119.
    assert before.thrust == 23835. and after.thrust == 0.
    # A step ending at burnout uses the left-side motor in all four RK stages.
    dt = .001
    nxt = model.step(s,.75-dt,dt,(0.,0.),left_end=True)
    assert nxt.mass == 119.
    assert nxt.velocity[0] > s.velocity[0]
    assert norm_difference(nxt.velocity,s.velocity) < 1.


def norm_difference(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def test_controller_clock_and_zero_memory_are_independent_of_plant_substeps():
    target_at = lambda t: TargetState((8000.-300*t,6450.,0.),(-300.,0.,0.))
    outputs = [SingleSurfaceGame(config(),plant_substeps=n).run(state(),target_at,end_time_s=1.25)
               for n in (4,8)]
    for result in outputs:
        assert result["summary"]["first_active_s"] == pytest.approx(14/48,abs=1e-12)
        ticks = result["ticks"]
        assert len(ticks) == 61
        for i,tick in enumerate(ticks):
            assert tick["time_s"] == pytest.approx(i/48,abs=1e-12)
            if i < 14:
                assert tick["pid_memory"] == ((0.,0.,0.),(0.,0.,0.))
        assert result["samples"][36]["mass_kg"] == 119.
    for key in ("velocity_mps","position_m"):
        assert norm_difference(outputs[0]["samples"][-1][key],outputs[1]["samples"][-1][key]) < 1e-5
    assert [t["gain"] for t in outputs[0]["ticks"]] == [t["gain"] for t in outputs[1]["ticks"]]


def test_minimum_range_stops_at_proximity_fuse_instead_of_full_substep():
    target_at = lambda t: TargetState((80.-300*t,6300.,0.),(-300.,0.,0.))
    result = SingleSurfaceGame(config()).run(state(),target_at,end_time_s=.25)
    final = result["samples"][-1]
    final_range = math.dist(target_at(final["time_s"]).position,final["position_m"])
    assert result["summary"]["termination_event"] == "proximity_fuse"
    assert final_range == pytest.approx(10.,abs=1e-3)
    assert result["summary"]["minimum_range_m"] == pytest.approx(10.,abs=1e-12)
    assert result["summary"]["minimum_range_m"] == pytest.approx(final_range,abs=1e-3)
