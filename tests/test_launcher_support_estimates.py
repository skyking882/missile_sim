"""Opt-in datalink estimates must reach every guidance consumer without truth leakage."""
import pytest
from test_observation_modes import _model, _missile
from aim120_model.observation import SensorTrackProvider
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode

TRUTH = TargetState((50000., 1000., 0.), (-300., 0., 0.))

def test_estimated_datalink_seeds_ins_and_survives_loss():
    model = _model(); model['inertial_drift_speed_mps'] = 0.
    provider = SensorTrackProvider(model)
    provider.enable_launcher_support(lambda t, truth: ('', (40000.-100*t, 1500., 2500.), (-100., 0., 0.)) if t < 1 else ('track_lost', None, None))
    first = provider.update(0., .1, _missile(), TRUTH)
    assert first.mode is TrackMode.DATALINK and first.position == (40000.,1500.,2500.)
    provider.update(.5, .5, _missile(), TRUTH)
    third = provider.update(1.5, 1., _missile(), TRUTH)
    assert third.position == (39850.,1500.,2500.) and third.velocity == (-100.,0.,0.)
    assert provider.datalink_lost_reason == 'track_lost'


def test_estimate_drives_seeker_search_even_when_truth_is_elsewhere():
    from test_seeker_search import sensor
    provider = SensorTrackProvider(sensor(), lock_range_m=100.)
    from aim120_model.radar_seeker import SeekerSearch
    provider.radar.search = SeekerSearch(provider.radar, {'pattern': 'spiral'})
    provider.enable_launcher_support(lambda t, truth: ('', (40000.,1500.,2500.), (-100.,0.,0.)))
    provider.update(0., .1, _missile(), TRUTH)
    assert provider.radar.search.estimate == (40000.,1500.,2500.)


def test_empty_string_legacy_and_truth_tuple_are_identical():
    legacy = SensorTrackProvider(_model()); estimated = SensorTrackProvider(_model())
    legacy.enable_launcher_support(lambda t, truth: '')
    estimated.enable_launcher_support(lambda t, truth: ('', truth.position, truth.velocity))
    for t in (0.,.1,.2):
        assert legacy.update(t,.1,_missile(),TRUTH) == estimated.update(t,.1,_missile(),TRUTH)


@pytest.mark.parametrize('reply', [('','bad',(0.,0.,0.)), ('',None,None), ('',(1.,2.,float('nan')),(0.,0.,0.)), (None,None,None)])
def test_invalid_estimate_surfaces_error(reply):
    provider=SensorTrackProvider(_model()); provider.enable_launcher_support(lambda t,truth:reply)
    with pytest.raises((TypeError,ValueError)): provider.update(0.,.1,_missile(),TRUTH)


def test_estimate_loss_does_not_reconnect_or_fall_back_to_truth():
    provider=SensorTrackProvider(_model(reconnect=True))
    provider.enable_launcher_support(lambda t,truth:('missing',None,None) if t==0 else ('',truth.position,truth.velocity))
    assert not provider.update(0.,.1,_missile(),TRUTH).valid
    assert not provider.update(1.,1.,_missile(),TRUTH).valid
    assert provider.datalink_lost_reason=='missing'

@pytest.mark.parametrize('api', ['simulate','simulate_surface','create_surface_missile'])
def test_public_apis_accept_estimates(api):
    from pathlib import Path
    from aim120_model.public_api import simulate
    from aim120_model.surface_runtime import simulate_surface, create_surface_missile
    from missile_gui.library import scan_library
    root=Path(__file__).resolve().parents[1]
    profiles,errors=scan_library(root/'data/profile_sets/2.59.0.27/missiles',root)
    if errors or not profiles:
        profiles,errors=scan_library(root/'missiles',root)
    profile=next(p for p in profiles if p.get('missile_id')=='us_aim_120c_5')
    callback=lambda t,truth:('',(50000.-100*t,9000.,2000.),(-100.,0.,0.))
    scenario=dict(launch_speed_kmh=1200.,launch_altitude_m=8000.,launch_pitch_deg=0.,launch_heading_deg=0.,
                  target_speed_kmh=1000.,target_altitude_m=8000.,initial_distance_m=50000.,target_azimuth_deg=0.,
                  target_heading_deg=180.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
                  max_simulation_time_s=.1,observation_mode='sensor_track',loft_enabled=True)
    if api=='simulate':
        result=simulate(profile,scenario,launcher_support=callback)
        assert result['samples'][0]['observed_target_position_m']==(50000.,9000.,2000.)
    elif api=='simulate_surface':
        result=simulate_surface(profile,scenario,launcher_support=callback)
        assert result['samples'][0]['observed_target_position_m']==(50000.,9000.,2000.)
    else:
        class Target:
            def state_at(self,t):return TRUTH
        runtime=create_surface_missile(profile,launch_position_m=(0.,8000.,0.),launch_velocity_mps=(300.,0.,0.),
                    launch_pitch_deg=0.,launch_heading_deg=0.,target=Target(),launcher_support=callback)
        runtime.step()
        assert runtime.provider.datalink_estimate[0]==pytest.approx((50000.-100*runtime.time_s,9000.,2000.))
        assert runtime.provider.datalink_estimate[1]==(-100.,0.,0.)
