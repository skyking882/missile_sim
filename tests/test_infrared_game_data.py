"""Frozen game heat/flare assembly requires behavior, never hand-entered heat scalars."""
import copy
import json
from pathlib import Path

import pytest

from aim120_model import (AircraftEngineState, AircraftInfraredSignature, FlareSpec,
    FlareRelease, FlaringTarget, flaring_factory, InfraredSeekerConfig, simulate)
from aim120_model.infrared_data import InfraredGameData
from aim120_model.public_api import SimulationInputError
from aim120_model.target import TargetState
from aim120_model.dynamics import SimState

ROOT=Path(__file__).resolve().parents[1]

def own(p=(0.,0.,0.)):
    return SimState(p,(300.,0.,0.),0.,0.,0.,0.,80.)

def stationary():
    return TargetState((1000.,0.,0.),(0.,0.,0.))

@pytest.mark.parametrize('model,count,base,coefficient',[
    ('f_16c_block_50',1,7030.,.99),('su_30mkk',2,7680.,.94)])
def test_real_aircraft_game_heat_fields(model,count,base,coefficient):
    heat=AircraftInfraredSignature(model,forward_world=(1.,0.,0.))
    raw=heat.components(0.,stationary(),own())
    assert heat.provenance['engine_count']==count
    assert raw['engine_brightness']==pytest.approx(count*base*coefficient)
    assert raw['airframe_brightness']==200.
    assert raw['afterburner_brightness']==0.
    assert heat.provenance['version']=='2.59.0.27'
    assert heat.provenance['commit']=='4ac255b829a27346b9300fccdc138362c2535fbe'
    assert heat.provenance['conversion_values']==dict(thrustKgsToInfraRedBrightness=1.,afterburnerThrustKgsToInfraRedBrightness=4.5,horsePowersToInfraRedBrightness=.42)
    assert heat.reference_engine_brightness==pytest.approx(6959.7)


def test_game_aspect_and_afterburner_constants_drive_real_heat_changes():
    target=stationary()
    heat=AircraftInfraredSignature('f_16c_block_50',forward_world=(1.,0.,0.))
    rear=heat.components(0.,target,own())
    front=heat.components(0.,target,own((2000.,0.,0.)))
    side=heat.components(0.,target,own((1000.,0.,1000.)))
    assert front['engine_brightness']/rear['engine_brightness']==pytest.approx(.0006)
    assert side['engine_brightness']/rear['engine_brightness']==pytest.approx(.06)
    after=AircraftInfraredSignature('f_16c_block_50',forward_world=(1.,0.,0.),
        engine_state=AircraftEngineState(afterburner=True))
    ar=after.components(0.,target,own());af=after.components(0.,target,own((2000.,0.,0.)))
    assert ar['afterburner_brightness']==pytest.approx((7030.*1.25-7030.*.99)*4.5)
    assert af['afterburner_brightness']/ar['afterburner_brightness']==pytest.approx(.05)
    assert ar['bands'][7]>0. and ar['bands'][0]==rear['bands'][0]
    idle=AircraftInfraredSignature('f_16c_block_50',forward_world=(1.,0.,0.),engine_state=AircraftEngineState(throttle=0.))
    assert idle.components(0.,target,own())['engine_brightness']==pytest.approx(rear['engine_brightness']*.02)
    off=AircraftInfraredSignature('f_16c_block_50',forward_world=(1.,0.,0.),engine_state=AircraftEngineState(running=False))
    assert off.components(0.,target,own())['engine_brightness']==0.
    assert off.components(0.,target,own())['airframe_brightness']==200.


def test_engine_command_callback_altitude_speed_and_aspect_behavior_inputs():
    heat=AircraftInfraredSignature('su_30mkk',engine_state=lambda t:AircraftEngineState(afterburner=t>=1.))
    t=TargetState((1000.,5000.,0.),(400.,0.,0.))
    a=heat.components(0.,t,own());b=heat.components(1.,t,own())
    assert a['afterburner_brightness']==0. and b['afterburner_brightness']>0.
    assert a['airframe_brightness']>200.*2.4
    with pytest.raises(ValueError,match='forward_world'):
        heat.components(0.,stationary(),own())
    with pytest.raises(FileNotFoundError,match='No other aircraft'):
        AircraftInfraredSignature('not_a_real_game_plane')
    with pytest.raises(ValueError,match='aircraft_model'):
        InfraredSeekerConfig.from_game_weapon('us_aim9l_sidewinder')


@pytest.mark.parametrize('name,mass,end,eject,cx,life,profile,peak',[
    ('regular',.09,.04,55.,2.58,4.4,(.4,4.,4.4),1.),
    ('bol',.054,.028,20.,12.,1.8,(.75,1.65,1.8),.6),
    ('large',1.,.27,70.,2.58,4.5,(.4,4.,4.4),4.5)])
def test_three_actual_flare_variants(name,mass,end,eject,cx,life,profile,peak):
    spec=FlareSpec.from_game_weapon(name)
    assert (spec.mass_kg,spec.mass_end_kg,spec.eject_speed_mps,spec.drag_coefficient,spec.lifetime_s,spec.emission_profile_s,spec.emission_band2)==(mass,end,eject,cx,life,profile,peak)
    assert spec.density_kg_m3 is None
    assert spec.parameter_sources['source']['version']=='2.59.0.27'
    assert spec.emission(profile[0])==pytest.approx(peak)
    assert spec.emission(profile[2])==0.
    assert spec.emission(.2)<peak
    assert spec.motion(.4,(0.,1000.,0.),(100.,-eject,0.),(0.,-1.,0.))[0][1]<1000.


def test_flare_motion_and_envelope_differ_by_real_fields_not_filename():
    specs=[FlareSpec.from_game_weapon(n) for n in ('regular','bol','large')]
    assert [s.emission(.4) for s in specs]==pytest.approx([1.,.32,4.5])
    assert [s.emission(2.) for s in specs]==pytest.approx([1.,0.,4.5])
    velocities=[s.motion(.4,(0.,1000.,0.),(100.,0.,s.eject_speed_mps),(0.,0.,1.))[1] for s in specs]
    assert len(set(velocities))==3
    high=specs[0].motion(.4,(0.,10000.,0.),(100.,0.,55.),(0.,0.,1.))[1]
    low=specs[0].motion(.4,(0.,0.,0.),(100.,0.,55.),(0.,0.,1.))[1]
    assert high[0]>low[0]  # Existing atmosphere, not a caller density guess.
    # These particular BOL-pod filenames contain an ordinary flare payload.
    for name in ('countermeasure_split_launcher_jet_bol_pod_aim9m','countermeasure_split_launcher_jet_bol_pod_aim9l'):
        spec=FlareSpec.from_game_weapon(name)
        assert spec.emission_band2==1. and spec.eject_speed_mps==55.


def test_only_real_flare_payloads_selected_and_zero_burn_has_no_epsilon():
    data=InfraredGameData()
    regular=data.weapon('countermeasure_split_launcher_jet')
    chaff=regular['countermeasures_launcher_chaff']
    with pytest.raises(ValueError,match='non-flare/chaff'):
        FlareSpec.from_game_weapon(chaff)
    # Actual gun AP/I ammunition also has emission2: it is not a flare.
    path='aces.vromfs.bin_u/gamedata/weapons/cannonns23.blkx'
    with pytest.raises(ValueError,match='non-flare/chaff'):
        FlareSpec.from_game_weapon(data.read(path))
    one=regular['bullet']
    spec=FlareSpec.from_game_weapon({'bullet':[chaff['bullet'],one]})
    assert spec.parameter_sources['payload_index']==1 and spec.emission_band2==1.
    with pytest.raises(ValueError,match='multiple flares'):
        FlareSpec.from_game_weapon({'bullet':[one,one]})
    assert FlareSpec.from_game_weapon({'bullet':[one,one]},bullet_index=1).emission_band2==1.
    # Explicit unpowered/no-start-speed payload: inherit carrier velocity exactly.
    raw=copy.deepcopy(one)
    raw['rocket'].update(timeFire=0.,force=0.,massEnd=.09,useStartSpeed=False)
    no_start=FlareSpec.from_game_weapon({'bullet':raw},density_kg_m3=0.)
    assert no_start.burn_s==0. and no_start.eject_speed_mps==0. and not no_start.use_start_speed
    p,v=no_start.motion(1.,(0.,1000.,0.),(100.,0.,0.),(0.,0.,1.))
    assert p==pytest.approx((100.,1000.-9.80665/2,0.)) and v==pytest.approx((100.,-9.80665,0.))


def scenario():
    return dict(launch_speed_kmh=1080.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=360.,target_altitude_m=1000.,initial_distance_m=2000.,target_azimuth_deg=0.,
        target_heading_deg=0.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=.9,observation_mode='sensor_track',loft_enabled=False)


def test_public_game_assembly_no_numeric_heat_inputs_three_flare_trajectories():
    profile=json.loads((ROOT/'data/profile_sets/2.59.0.27/missiles/us_aim9l_sidewinder.json').read_text())
    original=copy.deepcopy(profile)
    config=InfraredSeekerConfig.from_game_weapon('us_aim9l_sidewinder',aircraft_model='f_16c_block_50',
        engine_state=AircraftEngineState(throttle=0.))
    assert config.minimum_signal==config.flare_gain==1.
    assert dict(config.range_bands_m)[2]==11000.
    clear=simulate(profile,scenario(),infrared_seeker=config)
    outcomes=[]
    for name in ('regular','bol','large'):
        spec=FlareSpec.from_game_weapon(name)
        result=simulate(profile,scenario(),infrared_seeker=config,target_factory=flaring_factory(None,spec,
            [FlareRelease(.1,'flare:0',(0.,0.,1.))]))
        rows=[r for r in result['samples'] if r['ir_selected_identity']=='flare:0']
        assert rows and any(abs(r['commanded_acceleration_g'][2])>1e-6 for r in rows)
        assert rows[-1]['guidance_input_source']=='ir_truth_assisted_centroid'
        assert result['samples'][-1]['position_m']!=clear['samples'][-1]['position_m']
        assert result['model']['infrared_seeker']['thermal_data']['aircraft_model']=='f_16c_block_50'
        assert result['model']['target_model']['flares']['parameter_sources']['source']['version']=='2.59.0.27'
        json.dumps(result,allow_nan=False)
        outcomes.append(result['samples'][-1]['position_m'])
    assert len(set(outcomes))==3 and profile==original
    # Dataset mismatch must not become a mixed-version calibration.
    wrong=copy.deepcopy(profile);wrong['provenance']['source_version']='2.59.0.54'
    with pytest.raises(SimulationInputError,match='same game version'):
        simulate(wrong,scenario(),infrared_seeker=config)
