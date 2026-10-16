"""Source-informed game IR mechanics, deterministic flare entities and live PN seam."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from aim120_model import (FlareSpec, FlareRelease, FlareState, FlaringTarget,
                          InfraredSeekerConfig, InfraredTrackProvider, flaring_factory)
from aim120_model.dynamics import SimState
from aim120_model.public_api import create_surface_missile, simulate, SimulationInputError
from aim120_model.target import TargetState
from aim120_model.tracking import TrackMode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns'

def weapon(name):
    return json.loads((RAW/(name+'.blkx')).read_text())

def config(**changes):
    return replace(InfraredSeekerConfig(5., 75., 45., 10000., 11000., 1.,
        lambda t, target, own: 1., 1., 1e-12, signal_rule='strongest'), **changes)

def own():
    return SimState((0., 1000., 0.), (300., 0., 0.), 0., 0., 0., 0., 80.)

def aircraft(t=0., z=0.):
    return TargetState((1000.+100.*t, 1000., z), (100., 0., 0.))

def flare(identity='flare:1', angle=2., power=10., release=0.):
    return FlareState(identity, release, (1000., 1000., 1000.*math.tan(math.radians(angle))),
                      (0., 0., 0.), power, (1000.,1000.,0.), (100.,0.,0.))

class Scene:
    def __init__(self):
        self.flares = []
    def flares_at(self, t):
        return self.flares


def test_real_game_configs_distinguish_gate_rejection_and_baseline():
    kwargs = dict(angles_are_full_width=True, target_emission=lambda *a: 1.,
                  flare_gain=1., minimum_signal=1e-12, source='2.59.0.27 local datamine; mappings D')
    r = InfraredSeekerConfig.from_game_weapon(weapon('su_r_73'), **kwargs)
    m = InfraredSeekerConfig.from_game_weapon(weapon('us_aim9m_sidewinder'), **kwargs)
    l = InfraredSeekerConfig.from_game_weapon(weapon('us_aim9l_sidewinder'), **kwargs)
    assert (r.fov_half_deg, r.gated_half_deg, r.reject_flare_fraction) == (2.25,.375,None)
    assert (m.fov_half_deg, m.gated_half_deg, m.reject_flare_fraction, m.rejection_reaction_s) == (1.8,None,.1,.02)
    assert (l.gated_half_deg, l.reject_flare_fraction) == (None,None)
    r_half = InfraredSeekerConfig.from_game_weapon(weapon('su_r_73'), **{**kwargs,'angles_are_full_width':False})
    assert r_half.gated_half_deg == .75  # Caller convention, never silently chosen.


def test_baseline_capture_and_gated_exclusion_same_scene():
    scene = Scene()
    plain = InfraredTrackProvider(config(), scene)
    gated = InfraredTrackProvider(config(gated_half_deg=.5), scene)
    for p in (plain, gated):
        assert p.update(0., 0., own(), aircraft()).mode is TrackMode.INFRARED_TRACK
    scene.flares = [flare()]
    a = plain.update(.1,.1,own(),aircraft(.1))
    b = gated.update(.1,.1,own(),aircraft(.1))
    assert a.position == scene.flares[0].position and plain.last_detection.selected_identity == 'flare:1'
    assert b.position == aircraft(.1).position and gated.last_detection.visible_identities == ('target',)
    assert gated.last_detection.fov_half_deg == .5
    # Same-axis flare still passes the gate and can capture it.
    scene.flares = [flare(angle=.1)]
    gated.update(.2,.1,own(),aircraft(.2))
    assert gated.last_detection.selected_identity == 'flare:1'


def test_centroid_is_between_sources_not_random_lock_loss():
    scene = Scene(); scene.flares = [flare(power=3.)]
    p = InfraredTrackProvider(config(signal_rule='centroid'), scene)
    result = p.update(0.,0.,own(),aircraft())
    assert result.valid and 0 < result.position[2] < scene.flares[0].position[2]
    assert result.source == 'ir_truth_assisted_centroid' and result.full_rate_rad_s is None
    assert 'no IR range measurement' in p.provenance['kinematic_closure']


def test_suspension_predicts_old_state_then_reacquires_without_current_truth_leak():
    scene = Scene()
    p = InfraredTrackProvider(config(target_emission=lambda *a: 20.,
        reject_flare_fraction=.1, rejection_reaction_s=.02), scene)
    p.update(0.,0.,own(),aircraft())
    scene.flares = [flare(power=10.)]
    p.update(.1,.1,own(),aircraft(.1))
    old = p._last_measurement
    turned = TargetState((9000.,2000.,500.),(-50.,20.,80.))
    p.update(.12,.02,own(),turned)
    assert p.status == 'suspended' and p._solution.mode is TrackMode.INERTIAL
    expected = tuple(x+v*(.12-old.sample_time_s) for x,v in zip(old.position, old.velocity))
    assert p._solution.position == pytest.approx(expected)
    assert p._solution.velocity == old.velocity and p._solution.sample_time_s == .1
    scene.flares = []
    assert p.update(.2,.08,own(),aircraft(.2)).mode is TrackMode.INFRARED_TRACK
    assert p.status == 'track'


def test_reaction_resets_outside_fov_and_repeated_time_is_idempotent():
    scene = Scene()
    p = InfraredTrackProvider(config(reject_flare_fraction=.1,rejection_reaction_s=.02), scene)
    p.update(0.,0.,own(),aircraft())
    scene.flares = [flare(angle=10.)]
    assert p.update(.1,.1,own(),aircraft(.1)).mode is TrackMode.INFRARED_TRACK
    scene.flares = [flare(angle=0.)]
    first = p.update(.2,.1,own(),aircraft(.2))
    assert p.update(.2,.1,own(),aircraft(.2)) is first
    assert p.status == 'track'
    p.update(.21,.01,own(),aircraft(.21)); assert p.status == 'track'
    p.update(.22,.01,own(),aircraft(.22)); assert p.status == 'suspended'
    with pytest.raises(ValueError, match='nondecreasing'):
        p.update(.2,.1,own(),aircraft(.2))


def test_memory_expires_and_reacquisition_respects_view_and_gate_policy():
    scene = Scene()
    p = InfraredTrackProvider(config(memory_s=.2,gated_half_deg=.5), scene)
    p.update(0.,0.,own(),aircraft())
    far = TargetState((50000.,1000.,0.),(0.,0.,0.))
    assert p.update(.1,.1,own(),far).valid and p.status == 'memory'
    assert not p.update(.21,.11,own(),far).valid and p.status == 'lost'
    # Restored wider search admits an off-axis aircraft after memory expiration.
    near = aircraft(z=20.)
    assert p.update(.22,.01,own(),near).valid and p.status == 'track'
    retained = InfraredTrackProvider(config(memory_s=.2,gated_half_deg=.5,gate_on_loss='retain_gate'),scene)
    retained.update(0.,0.,own(),aircraft());retained.update(.21,.21,own(),far)
    assert not retained.update(.22,.01,own(),near).valid


def test_preflare_can_capture_search_and_later_release_is_not_visible():
    scene = Scene();scene.flares=[flare(release=1.)]
    p = InfraredTrackProvider(config(reject_flare_fraction=.1),scene)
    p.update(0.,0.,own(),aircraft())
    assert p.last_detection.selected_identity == 'target'
    scene.flares=[flare(release=0.)]
    pre = InfraredTrackProvider(config(reject_flare_fraction=.1),scene)
    pre.update(0.,0.,own(),aircraft())
    assert pre.status == 'track' and pre.last_detection.selected_identity == 'flare:1'


def test_flare_release_inherits_carrier_then_moves_and_burns_out_deterministically():
    spec = FlareSpec.from_game_weapon(weapon('countermeasure_split_launcher_jet_only_flare_large'),
                                     density_kg_m3=1.225,source='2.59.0.27; physics D')
    assert (spec.eject_speed_mps,spec.emission_band2,spec.emission_profile_s) == (70.,4.5,(.4,4.,4.4))
    class Carrier:
        def __init__(self):self.queries=[]
        def state_at(self,t):self.queries.append(t);return aircraft(t)
        def decoys_at(self,t):return ['radar chaff untouched']
    c = Carrier(); wrapped=FlaringTarget(c,spec,[FlareRelease(.5,'left',(0.,-1.,0.)),FlareRelease(2.,'right',(0.,0.,1.))])
    assert wrapped.flares_at(.49) == [] and c.queries == []
    born=wrapped.flares_at(.5)[0]
    assert born.position == aircraft(.5).position
    assert born.velocity == (100.,-70.,0.) and born.emission_band2 == 0
    out=wrapped.flares_at(.9)[0]
    assert out.emission_band2 == pytest.approx(4.5)
    assert out.position[0] > born.position[0] and out.position[1] < born.position[1]
    assert wrapped.flares_at(.9)[0] == out and c.queries == [.5]
    assert wrapped.flares_at(.5+4.4)[0].emission_band2 == 0  # Alive but dark.
    assert all(f.identity != 'left' for f in wrapped.flares_at(5.))
    assert wrapped.decoys_at(1.) == ['radar chaff untouched']
    copy.deepcopy(wrapped)  # Delegation must not recurse while restoring inner.
    with pytest.raises(ValueError,match='unique'):
        FlaringTarget(c,spec,[FlareRelease(0.,'same',(1.,0.,0.)),FlareRelease(1.,'same',(1.,0.,0.))])


def test_signal_engine_aspect_callback_and_detection_floor():
    scene=Scene()
    p=InfraredTrackProvider(config(target_emission=lambda t,*a: 1. if t < .1 else 0.,memory_s=0.),scene)
    assert p.update(0.,0.,own(),aircraft()).valid
    assert not p.update(.1,.1,own(),aircraft(.1)).valid
    bad=InfraredTrackProvider(config(target_emission=lambda *a:float('nan')),scene)
    with pytest.raises(ValueError,match='emission'):
        bad.update(0.,0.,own(),aircraft())


def test_public_runtime_flare_enters_actual_guidance_and_none_is_identical():
    profile=json.loads((ROOT/'data/profile_sets/2.59.0.27/missiles/us_aim9m_sidewinder.json').read_text())
    original=copy.deepcopy(profile)
    spec=FlareSpec.from_game_weapon(weapon('countermeasure_split_launcher_jet_only_flare_large'),
        density_kg_m3=1.225,source='2.59.0.27; D mapping')
    releases=(FlareRelease(.1,'test-flare',(0.,0.,1.)),)
    factory=flaring_factory(None,spec,releases)
    s=dict(launch_speed_kmh=1080.,launch_altitude_m=1000.,launch_pitch_deg=0.,
        launch_heading_deg=0.,target_speed_kmh=360.,target_altitude_m=1000.,initial_distance_m=2000.,
        target_azimuth_deg=0.,target_heading_deg=0.,target_vertical_heading_deg=0.,
        target_constant_turn_g=0.,max_simulation_time_s=.5,observation_mode='sensor_track',loft_enabled=False)
    base=simulate(profile,s)
    explicit=simulate(profile,s,infrared_seeker=None)
    ignored=simulate(profile,s,target_factory=factory)
    assert base == explicit
    assert base['samples'] == ignored['samples']
    assert base['summary'] == ignored['summary']
    assert 'infrared_seeker' not in base['model'] and 'ir_state' not in base['samples'][0]
    cfg=config(flare_gain=100., signal_rule='centroid')
    clear=simulate(profile,s,infrared_seeker=cfg)
    decoy=simulate(profile,s,target_factory=factory,infrared_seeker=cfg)
    assert decoy['model']['infrared_seeker']['kind'] == 'assumed'
    rows=[x for x in decoy['samples'] if x['ir_selected_identity']=='test-flare']
    assert rows
    assert any(x['observed_target_position_m'][2] > 0. for x in rows)
    assert any(abs(x['commanded_acceleration_g'][2]) > 1e-6 for x in rows)
    assert clear['samples'][-1]['position_m'] != decoy['samples'][-1]['position_m']
    assert any(x['guidance_input_source']=='ir_truth_assisted_centroid' for x in rows)
    assert profile == original
    json.dumps(decoy,allow_nan=False)
    # The explicit launch/public stepping entry exercises the same provider.
    class Carrier:
        def state_at(self,t):return TargetState((2000.+100*t,1000.,0.),(100.,0.,0.))
    target=FlaringTarget(Carrier(),spec,releases)
    rt=create_surface_missile(profile,launch_position_m=(0.,1000.,0.),launch_velocity_mps=(300.,0.,0.),
        launch_pitch_deg=0.,launch_heading_deg=0.,target=target,loft=False,end_time_s=.5,infrared_seeker=cfg)
    while not rt.done:rt.step()
    assert any(x['ir_selected_identity']=='test-flare' for x in rt.result()['samples'])
    assert rt.last_request[2] != 0.
    with pytest.raises(SimulationInputError,match='optical sensor_track'):
        simulate(profile,{**s,'observation_mode':'ideal_truth'},infrared_seeker=cfg)
    with pytest.raises(SimulationInputError,match='cannot combine'):
        simulate(profile,s,infrared_seeker=cfg,guidance_input=lambda *a:None)
