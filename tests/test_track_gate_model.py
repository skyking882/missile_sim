"""Opt-in WT track association, loss and acquisition contracts."""
import copy
import math
import pytest
from aim120_model.chaff import Reflector, ChaffSpec
from aim120_model.dynamics import SimState
from aim120_model.target import TargetState
from aim120_model.radar_seeker import RadarSeekerObserver, validate_track_gate_model


def missile():
    return SimState((0., 1000., 0.), (300., 0., 0.), 0., 0., 0., 0., 100.)


def target(vg=0., y=1000., z=0.):
    return TargetState((1000., y, z), (vg, 0., 0.))


def observer(**options):
    o = RadarSeekerObserver(dict(active_radar=True, radar_seeker=dict(active=True,
        angle_max_deg=60., lock_angle_max_deg=60., angle_gate_rate_deg_s=30.,
        rate_max_deg_s=180., receiver=dict(range_max_m=20000., antenna=dict(angle_half_sens_deg=15.)),
        doppler_speed=dict(width_mps=20., ref_width_mps=80.))), lock_range_m=20000.)
    o.enable_track_gate_model({'notch_track_hazard': [], **options})
    return o


def bundle(identity='a', position=(950., 990., 0.), rcs=1., release=0.):
    return Reflector('chaff', position, (0., 0., 0.), rcs, identity,
                     release_identity=identity, release_time_s=release)


def test_tracking_downview_notch_does_not_delete_but_search_does():
    o=observer(); m=missile()
    assert not o.update(0., .02, m, target(y=990.))[0].valid
    assert o.last_detection.reason == 'ground_clutter_notch'
    assert o.update(.02, .02, m, target())[0].valid
    assert o.update(.04, .02, m, target(y=990.))[0].valid
    assert o.last_reflector == 'target'


def test_chaff_competes_as_envelope_opens_without_added_delay():
    o=observer(); m=missile(); t=target()
    o.update(0., .02, m, t)
    spec=ChaffSpec(1.)
    assert o.update(.04, .04, m, t, [bundle(rcs=spec.envelope(.04))])[0].valid
    assert o.last_reflector == 'target'
    o.update(.1, .06, m, t, [bundle(position=(950., 1000., 0.), rcs=spec.envelope(.1))])
    assert o.last_reflector == 'chaff'
    assert o.track_gate_events[-1]['time_s'] == .1
    assert o.track_gate_identity == ('decoy', 'a')


def test_head_on_chaff_outside_velocity_gate_cannot_capture():
    o=observer(); m=missile(); t=target(-200.)
    assert o.update(0., .02, m, t)[0].valid
    o.update(.1, .1, m, t, [bundle(rcs=100.)])
    assert o.last_reflector == 'target'


def test_track_follows_bundle_and_rejects_stronger_aircraft_outside_angle_gate():
    o=observer(range_gate_m=1000.); m=missile(); t=target()
    o.update(0., .02, m, t)
    d=bundle(position=(950.,1000.,0.),rcs=10.)
    o.update(.1,.1,m,t,[d]); assert o.last_reflector=='chaff'
    # Still inside the hardware cone, but outside the track's 5-degree angle gate.
    turned=target(z=1000.*math.tan(math.radians(8.)))
    o.update(.2,.1,m,turned,[d],target_rcs_m2=10000.)
    assert o.last_reflector=='chaff' and o.last_solution.position==d.position


def test_lost_chaff_reacquires_only_near_current_memory_or_datalink_pointing():
    o=observer();m=missile();t=target()
    o.update(0.,.02,m,t);o.update(.1,.1,m,t,[bundle(position=(950.,1000.,0.),rcs=10.)])
    outside=target(z=1000.*math.tan(math.radians(6.)))
    assert not o.update(.2,.1,m,outside,[])[0].valid
    assert o.last_solution.mode.value=='radar_coast'
    assert not o.update(1.2,1.,m,outside,[])[0].valid  # hold expires before reacquisition
    assert o.last_detection.reason=='reacquire_window'
    o.track_gate_estimate=outside.position  # real available data-link estimate
    assert o.update(1.3,.1,m,outside,[])[0].valid
    assert o.last_reflector=='target'


def test_hazard_same_timestamp_once_and_deepcopy_restore():
    o=observer(notch_track_hazard=[[75.,1000.]],seed=5);m=missile()
    o.update(0.,.02,m,target())
    before=copy.deepcopy(o)
    a=o.update(.02,.02,m,target(y=990.))
    assert not a[0].valid and a[0].reason=='track_notch_hazard'
    assert o.update(.02,.02,m,target(y=990.)) is a
    b=before.update(.02,.02,m,target(y=990.))
    assert a==b and o.track_gate_summary()==before.track_gate_summary()


def test_seeded_fluctuation_restores_identity_and_outcome():
    o=observer(compete='stronger_fluct',seed=17);m=missile();t=target()
    o.update(0.,.02,m,t);clone=copy.deepcopy(o)
    for stamp in (.02,.04,.06):
        assert o.update(stamp,.02,m,t,[bundle()])==clone.update(stamp,.02,m,t,[bundle()])
    assert o.track_gate_summary()==clone.track_gate_summary()


def test_hardware_range_and_rate_gates_stay_in_effect():
    o=observer();m=missile();t=target()
    o.update(0.,.02,m,t)
    o.receiver_range_max_m=500.
    assert not o.update(.02,.02,m,t)[0].valid
    assert o.last_detection.reason=='range_gate'
    o=observer();o.rate_max_rad_s=0.
    assert not o.update(0.,.02,m,target(z=100.))[0].valid
    assert o.last_detection.reason=='rate_gate'


def test_old_capture_rejects_combination_in_both_orders():
    o=observer()
    with pytest.raises(ValueError,match='exclusive'):o.enable_chaff_capture({})
    o.track_gate_model=None;o.enable_chaff_capture({})
    with pytest.raises(ValueError,match='combine'):o.enable_track_gate_model({})


@pytest.mark.parametrize('bad',[{'x':1},{'range_gate_m':-1},{'seed':True},{'angle_gate_deg':181.},
                                {'notch_track_hazard':[[50.,.3],[25.,.4]]},{'compete':'dice'}])
def test_invalid_options_fail_explicitly(bad):
    with pytest.raises(ValueError):validate_track_gate_model(bad)


def test_public_simulation_surface_and_stepping_expose_same_new_model():
    from pathlib import Path
    from missile_gui.library import scan_library
    from aim120_model.public_api import simulate, create_surface_missile, SimulationInputError
    from aim120_model.surface_runtime import simulate_surface
    root=Path(__file__).resolve().parents[1]
    p=next(p for p in scan_library(root/'missiles',root)[0] if p['missile_id']=='cn_pl12')
    scene=dict(launch_speed_kmh=1000.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=300.,target_altitude_m=1000.,initial_distance_m=7000.,target_azimuth_deg=0.,
        target_heading_deg=180.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=.85,observation_mode='sensor_track',loft_enabled=False)
    a=simulate(p,scene,track_gate_model={}); b=simulate_surface(p,scene,track_gate_model={})
    assert a==b and a['model']['track_gate_model']['range_gate_m']==150.
    assert a['model']['parameter_sources']['gate_widths']['kind']=='assumed'
    assert a['samples'][-1]['tracked_reflector']=='target'
    class T:
        def state_at(self,t):return TargetState((7000.-100.*t,1000.,0.),(-100.,0.,0.))
    rt=create_surface_missile(p,launch_position_m=(0.,1000.,0.),launch_velocity_mps=(300.,0.,0.),
        launch_pitch_deg=0.,launch_heading_deg=0.,target=T(),end_time_s=.85,track_gate_model={})
    while not rt.done:rt.step()
    assert rt.result()['track_gate_model']['events'][0]['event']=='acquire'
    with pytest.raises(SimulationInputError,match='exclusive'):
        simulate(p,scene,track_gate_model={},chaff_capture={})
    with pytest.raises(SimulationInputError):
        simulate(p,dict(scene,observation_mode='ideal_truth'),track_gate_model={})


def test_tracking_hazard_honors_existing_clutter_depression_option():
    o=observer(notch_track_hazard=[[75.,1000.]])
    o.clutter_model='look_down_angle';o.clutter_min_depression_rad=math.radians(2.)
    m=missile();o.update(0.,.02,m,target())
    # -0.57 degrees belongs to the look-up/no-clutter sample group.
    assert o.update(.02,.02,m,target(y=990.))[0].valid
    assert not o.update(.04,.02,m,target(y=950.))[0].valid
    assert o.last_detection.reason=='track_notch_hazard'
