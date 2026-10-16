"""Frozen Air RB >=13.0 coverage: actual FM/mounts, all seeker types, packaged inputs."""
import json
import math
import shutil
from pathlib import Path

import pytest

from aim120_model import (AircraftEngineState, AircraftInfraredSignature, FlareSpec,
    FlareRelease, InfraredSeekerConfig, InfraredTrackProvider, flaring_factory,
    infrared_coverage, load_infrared_missile_profile, simulate)
from aim120_model.infrared_data import InfraredGameData
from aim120_model.target import TargetState
from aim120_model.dynamics import SimState

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = infrared_coverage()


def own():
    return SimState((0.,1000.,0.),(300.,0.,0.),0.,0.,0.,0.,80.)


def target():
    return TargetState((2000.,1000.,0.),(200.,0.,0.))


def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    if isinstance(value,(int,float)):return math.isfinite(value)
    return True


def test_exact_frozen_br_selection_and_known_shared_fm():
    assert COVERAGE['counts'] == dict(aircraft=91,fm_files=79,weapon_configs=40,missile_ids=21,containers=220,excluded=10,
        automatic_weapon_configs=38,unresolved_weapon_configs=2,runtime_profile_missile_ids=19)
    assert COVERAGE['selection']['mode']=='Air RB'
    assert COVERAGE['selection']['rank_field']=='wpcost.economicRankHistorical'
    assert all(v['economicRankHistorical']>=36 and v['br_air_rb']>=13. for v in COVERAGE['aircraft'].values())
    assert all('hideBrForVehicle=true' in v['reason'] for v in COVERAGE['excluded'].values())
    assert not (set(COVERAGE['aircraft']) & set(COVERAGE['excluded']))
    assert COVERAGE['aircraft']['f_4e_aup_greece']['fm_source'].endswith('/fm/f-4f_kws_lv.blkx')
    assert COVERAGE['aircraft']['f_16am_block_20_mlu_norway']['fm_source'].endswith('/fm/f_16a_block_20_mlu.blkx')
    assert COVERAGE['aircraft']['ef_2000_fgr4']['fm_source'].endswith('/fm/typhoon_fgr4.blkx')
    assert COVERAGE['aircraft']['f_16am_block_15_mlu_belgium']['fm_source'].endswith('.blkx')
    assert COVERAGE['aircraft']['tornado_f3_late']['fm_source'].endswith('/fm/tornado_f3_late.blkx')


@pytest.mark.parametrize('model',sorted(COVERAGE['aircraft']))
def test_every_aircraft_real_engine_instances_and_thermal_states(model):
    heat=AircraftInfraredSignature(model)
    fm=heat.fm
    assert heat.provenance['engine_count']==sum(k.startswith('Engine') and k[6:].isdigit() for k in fm)
    assert heat.provenance['fm_source']==COVERAGE['aircraft'][model]['fm_source']
    raw=heat.components(0.,target(),own())
    assert finite(raw) and raw['engine_brightness']>0. and raw['airframe_brightness']>0.
    idle=AircraftInfraredSignature(model,engine_state=AircraftEngineState(throttle=0.)).components(0.,target(),own())
    ab=AircraftInfraredSignature(model,engine_state=AircraftEngineState(afterburner=True)).components(0.,target(),own())
    off=AircraftInfraredSignature(model,engine_state=AircraftEngineState(running=False)).components(0.,target(),own())
    assert finite(idle) and finite(ab) and finite(off)
    assert idle['engine_brightness']<raw['engine_brightness']
    assert ab['afterburner_brightness']>=0. and off['engine_brightness']==0.
    assert len(COVERAGE['aircraft'][model]['infrared_weapons'])>0
    assert len(COVERAGE['aircraft'][model]['flare_payloads'])>0
    high=TargetState((2000.,25000.,0.),(900.,0.,0.))
    assert finite(heat.components(0.,high,own()))


def test_strength_not_normalized_per_aircraft_and_missing_j8f_velocity_is_explicit():
    strengths={AircraftInfraredSignature(n).components(0.,target(),own())['engine_brightness'] for n in COVERAGE['aircraft']}
    assert len(strengths)>30
    j8=AircraftInfraredSignature('j_8f')
    assert j8.provenance['table_velocity_assumptions'][0]['kind']=='assumed'
    assert j8.provenance['table_velocity_assumptions'][0]['value']=='TAS'
    assert 'VelocityType' not in j8.engines[0]['ThrustMax']  # Never fabricated in the original fields.


@pytest.mark.parametrize('weapon',sorted(COVERAGE['weapons']))
def test_every_mounted_ir_config_instantiates_and_observes(weapon):
    entry=COVERAGE['weapons'][weapon]
    if not entry['supported']:
        assert entry['mechanism']=='mask_only_unresolved'
        assert weapon in ('jp_aam3','jp_aam3_default')
        with pytest.raises(ValueError,match='Unresolved IRCCM'):
            InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='su_30mkk')
        explicit=InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='su_30mkk',reject_flare_fraction=.2)
        assert explicit.weapon_provenance['rejection_override']==dict(kind='assumed',evidence_level='D',value=.2)
        return
    assert not entry['unsupported']
    config=InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='su_30mkk')
    seeker=entry['optical_seeker']
    assert dict(config.range_bands_m)=={int(k[9:]):v for k,v in seeker.items() if k.startswith('rangeBand')}
    assert config.weapon_provenance['raw_optical_seeker']==seeker
    assert (config.gated_half_deg is not None)==('gateWidth' in seeker)
    assert (config.reject_flare_fraction is not None)==bool(seeker.get('bandMaskToReject'))
    class PlainTarget:
        pass
    provider=InfraredTrackProvider(config,PlainTarget())
    solution=provider.update(0.,.02,own(),target())
    assert provider.status=='track' and finite(solution.position) and finite(solution.velocity)


def test_aliases_mount_validation_profile_mapping_and_tvc_boundary():
    config=InfraredSeekerConfig.from_game_weapon('pl_5e2',aircraft_model='su_30mkk',launch_aircraft_model='j_10a')
    assert config.weapon_provenance['source_path'].endswith('/su_pl5e2.blkx')
    assert load_infrared_missile_profile('pl_5e2')['missile_id']=='pl_5e2'
    assert load_infrared_missile_profile('us_aim9l_sidewinder_bol_pod')['missile_id']=='us_aim9l_sidewinder'
    with pytest.raises(ValueError,match='not an IR AAM mounted'):
        InfraredSeekerConfig.from_game_weapon('su_r_73',aircraft_model='su_30mkk',launch_aircraft_model='f_16c_block_50')
    for weapon in ('su_r_73','su_r_73e'):
        with pytest.raises(ValueError,match='tvc_excluded'):
            load_infrared_missile_profile(weapon)
    for weapon,entry in COVERAGE['weapons'].items():
        if entry['runtime_profile']:
            assert load_infrared_missile_profile(weapon)['missile_id']==entry['missile_id']


def test_every_actual_aircraft_flare_payload_has_automatic_game_spec():
    values=set()
    for model,entry in COVERAGE['aircraft'].items():
        for option in entry['flare_payloads']:
            spec=FlareSpec.from_game_aircraft(model,launcher=option['weapon'],bullet_index=option['bullet_index'])
            assert finite(spec.motion(.2,(0.,1000.,0.),(200.,0.,spec.eject_speed_mps),(0.,0.,1.)))
            assert finite(spec.emission(.2)) and spec.emission(.2)>0.
            values.add(spec.emission_band2)
    assert values=={.6,1.,4.5}
    with pytest.raises(ValueError,match='actual flare launcher/payload'):
        FlareSpec.from_game_aircraft('ef_2000a',launcher='not_a_mounted_launcher')


def test_versioned_assets_work_without_sparse_clone_or_outputs_cache(tmp_path):
    output=tmp_path/'data/profile_sets/2.59.0.27/infrared'
    shutil.copytree(ROOT/'data/profile_sets/2.59.0.27/infrared',output)
    data=InfraredGameData(project_dir=tmp_path)
    for model in COVERAGE['aircraft']:
        heat=AircraftInfraredSignature(model,project_dir=tmp_path)
        assert finite(heat.components(0.,target(),own()))
    for weapon in COVERAGE['weapons']:
        if COVERAGE['weapons'][weapon]['supported']:
            assert InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='f_16c_block_50',project_dir=tmp_path).minimum_signal==1.
        else:
            with pytest.raises(ValueError,match='Unresolved IRCCM'):
                InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='f_16c_block_50',project_dir=tmp_path)
    for name in ('regular','bol','large'):
        assert FlareSpec.from_game_weapon(name,project_dir=tmp_path).emission(.2)>0.
    assert len(data.coverage()['aircraft'])==91


@pytest.mark.parametrize('launch,weapon,target_plane',[
    ('f_16c_block_50','us_aim9l_sidewinder','f_4e_aup_greece'),
    ('su_30mkk','cn_pl8b','ef_2000_fgr4'),
    ('fa_18c_late','us_aim9m_sidewinder','f_16am_block_20_mlu_norway')])
def test_public_runtime_for_three_seeker_mechanisms_uses_real_aircraft_inputs(launch,weapon,target_plane):
    profile=load_infrared_missile_profile(weapon,launch_aircraft_model=launch)
    config=InfraredSeekerConfig.from_game_weapon(weapon,launch_aircraft_model=launch,
        aircraft_model=target_plane,engine_state=AircraftEngineState(throttle=0.))
    scenario=dict(launch_speed_kmh=1080.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=360.,target_altitude_m=1000.,initial_distance_m=1500.,target_azimuth_deg=0.,
        target_heading_deg=0.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=.6,observation_mode='sensor_track',loft_enabled=False)
    option=COVERAGE['aircraft'][target_plane]['flare_payloads'][0]
    spec=FlareSpec.from_game_aircraft(target_plane,launcher=option['weapon'],bullet_index=option['bullet_index'])
    result=simulate(profile,scenario,infrared_seeker=config,target_factory=flaring_factory(None,spec,
        [FlareRelease(.1,'actual-plane-flare',(0.,0.,1.))]))
    assert result['samples'] and finite(result)
    assert result['model']['infrared_seeker']['thermal_data']['fm_source']==COVERAGE['aircraft'][target_plane]['fm_source']
    assert any('actual-plane-flare' in x['ir_visible_identities'] for x in result['samples'])
    assert any(x['guidance_input_source'].startswith('ir_truth_assisted') for x in result['samples'])
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('weapon',sorted({v['missile_id'] for v in COVERAGE['weapons'].values()
    if v['runtime_profile'] and v['supported']}))
def test_all_automatic_non_tvc_missile_profiles_enter_public_runtime(weapon):
    profile=load_infrared_missile_profile(weapon)
    config=InfraredSeekerConfig.from_game_weapon(weapon,aircraft_model='su_30mkk')
    scenario=dict(launch_speed_kmh=1080.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=360.,target_altitude_m=1000.,initial_distance_m=2000.,target_azimuth_deg=0.,
        target_heading_deg=180.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=.1,observation_mode='sensor_track',loft_enabled=False)
    result=simulate(profile,scenario,infrared_seeker=config)
    assert result['samples'] and finite(result)
    assert any(x['guidance_input_source'].startswith('ir_truth_assisted') for x in result['samples'])
