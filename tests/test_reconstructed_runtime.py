"""Focused candidate acceptance: default entry, real branches, causal inputs."""
import copy
import json
import unittest
from aim120_model.public_api import simulate,SimulationInputError,UnsupportedPhysicsError
from aim120_model.reconstructed import load_profile,library_profiles
from aim120_model.reconstructed.runtime import ReconstructedRuntime
from aim120_model.reconstructed.resources import read


def scenario():
    return dict(launch_speed_kmh=1080.,launch_altitude_m=1000.,launch_pitch_deg=0.,launch_heading_deg=0.,
        target_speed_kmh=360.,target_altitude_m=1000.,initial_distance_m=2000.,target_azimuth_deg=0.,
        target_heading_deg=0.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        max_simulation_time_s=1.5,observation_mode='sensor_track',loft_enabled=False,target_aircraft='su_30mkm',
        infrared_source='explicit_bands',target_ir={0:1.,1:1.,7:0.},
        target_ir_description='Authored demonstration signature/C0; not a measured aircraft source',
        target_engine=dict(throttle=1.,afterburner=False,running=True))


def illumination():
    return dict(position_m=[-1000.,1000.,0.],velocity_mps=[100.,0.,0.],track_scene_target=True,enabled=True,waveform='pd')


class CandidateAcceptance(unittest.TestCase):
    def test_default_public_three_branches_and_consumed_control_order(self):
        for mid,kind in [('us_aim9m_sidewinder','ir'),('su_r_77_1','arh'),('us_aim7f_sparrow','sarh')]:
            with self.subTest(mid=mid):
                s=scenario()
                if kind=='sarh':s['illuminator']=illumination()
                result=simulate(load_profile(mid),s)
                self.assertEqual(result['summary']['observation_provider'],'reconstructed_'+kind)
                self.assertTrue(any(row['measurement_fresh'] for row in result['samples']))
                self.assertTrue(any(any(abs(x)>1e-6 for x in row['control']) for row in result['samples']))
                for a,b in zip(result['samples'],result['samples'][1:]):self.assertEqual(a['control'],b['control_consumed'])
                self.assertEqual(result['missile']['parameter_source']['version'],'2.59.0.80')
                json.dumps(result,allow_nan=False)

    def test_ir_never_uses_truth_range_or_velocity_for_guidance(self):
        r=simulate(load_profile('us_aim9m_sidewinder'),json.loads(json.dumps(scenario())))
        self.assertIsNotNone(r['model']['thermal_source'])
        self.assertEqual(r['model']['thermal_source']['kind'],'explicit_scene_signature')
        for row in r['samples']:
            self.assertFalse(row['range_valid']);self.assertFalse(row['doppler_valid'])
            self.assertIsNone(row['observed_range_m']);self.assertEqual(row['guidance_velocity_source'],'missile_velocity')
            self.assertEqual(row['velocity_mps'],row['guidance_velocity_argument'])
        self.assertIsNone(r['summary']['maximum_track_error_m'])

    def test_ir_fm_is_explicit_approximation_with_all_engine_controls(self):
        s=scenario();s.pop('target_ir');s['max_simulation_time_s']=.1
        with self.assertRaisesRegex(SimulationInputError,'target_ir'):
            simulate(load_profile('us_aim9m_sidewinder'),s)
        s['infrared_source']='legacy_fm_approximation'
        for key in ['throttle','afterburner','running']:
            incomplete=copy.deepcopy(s);incomplete['target_engine'].pop(key)
            with self.assertRaisesRegex(SimulationInputError,'target_engine'):
                simulate(load_profile('us_aim9m_sidewinder'),incomplete)
        r=simulate(load_profile('us_aim9m_sidewinder'),s)
        p=r['model']['thermal_source']
        self.assertEqual(p['kind'],'legacy_fm_approximation')
        self.assertIn('unverified',p['optical_contract_mapping'])
        self.assertEqual(p['engine_input'],s['target_engine'])

    def test_ir_fm_aspect_uses_body_orientation(self):
        from aim120_model.reconstructed.aircraft_infrared import AircraftInfraredSignature,AircraftEngineState
        from aim120_model.reconstructed.types import BodyState
        producer=AircraftInfraredSignature('su_30mkm',engine_state=AircraftEngineState(1.,False,True))
        own=BodyState((0.,1000.,0.),(300.,0.,0.))
        front=BodyState((1000.,1000.,0.),(100.,0.,0.),(0.,0.,0.,1.))
        back=BodyState(front.position,front.velocity,(0.,1.,0.,0.))
        self.assertNotEqual(producer.components(0.,front,own)['engine_brightness'],
                            producer.components(0.,back,own)['engine_brightness'])
        s=scenario();s['target_quaternion_xyzw']=[0.,1.,0.,0.]
        runtime=ReconstructedRuntime(load_profile('us_aim9m_sidewinder'),s)
        self.assertEqual(tuple(runtime.scene.state_at(0.).q),(0.,1.,0.,0.))

    def test_sarh_signal_approximation_is_exposed_in_result(self):
        s=scenario();s['max_simulation_time_s']=.1;s['illuminator']=illumination()
        r=simulate(load_profile('us_aim7f_sparrow'),s)
        self.assertEqual(r['model']['SARH_signal_model']['status'],'derived_approximation')
        self.assertIn('not recovered',r['model']['SARH_signal_model']['native_closure'])
        for row in r['samples']:
            self.assertEqual(row['bistatic']['signal_model_status'],'derived_approximation')
            self.assertIn('not recovered',row['bistatic']['signal_model_limit'])

    def test_sarh_illumination_loss_reaches_hold_lost_and_reacquisition(self):
        s=scenario();s['max_simulation_time_s']=2.;s['illuminator']=illumination()
        s['illuminator']['off_intervals']=[[.2,1.5]]
        rows=simulate(load_profile('us_aim7f_sparrow'),s)['samples']
        self.assertTrue(any(r['measurement_held'] for r in rows))
        self.assertTrue(any(not r['track_valid'] and .2<r['time_s']<1.5 for r in rows))
        self.assertTrue(any(r['measurement_fresh'] and r['time_s']>1.5 for r in rows))
        self.assertTrue(all(not r['range_valid'] and not r['doppler_valid'] for r in rows))

    def test_active_search_hold_loss_reacquisition(self):
        s=scenario();s['max_simulation_time_s']=3.2;s['target_absent_intervals']=[[1.,2.5]]
        rows=simulate(load_profile('su_r_77_1'),s)['samples']
        self.assertTrue(any(r['measurement_fresh'] and r['time_s']<1. for r in rows))
        self.assertTrue(any(r['measurement_held'] for r in rows))
        self.assertTrue(any(not r['track_valid'] and 2.<r['time_s']<2.5 for r in rows))
        self.assertTrue(any(r['measurement_fresh'] and r['time_s']>2.5 for r in rows))

    def test_missing_sources_and_unsupported_are_explicit(self):
        with self.assertRaisesRegex(SimulationInputError,'illuminator'):simulate(load_profile('us_aim7f_sparrow'),scenario())
        s=scenario();del s['target_ir']
        with self.assertRaisesRegex(SimulationInputError,'target_ir'):simulate(load_profile('us_aim9m_sidewinder'),s)
        p=load_profile('us_aim9m_sidewinder');p['missile_id']='su_r_73'
        with self.assertRaisesRegex(UnsupportedPhysicsError,'thrust_vector'):simulate(p,scenario())
        s=scenario();s['illuminator']=illumination();s['illuminator']['waveform']='cw'
        with self.assertRaisesRegex(UnsupportedPhysicsError,'PD|CW|waveform'):simulate(load_profile('us_aim7f_sparrow'),s)

    def test_each_missile_keeps_own_stages_pid_and_response_scope(self):
        s=scenario();s['illuminator']=illumination()
        aim=ReconstructedRuntime(load_profile('us_aim_120a'),s)
        sparrow=ReconstructedRuntime(load_profile('us_aim7f_sparrow'),s)
        self.assertEqual(len(aim.physics.stages),2);self.assertEqual(len(sparrow.physics.stages),2)
        self.assertIsNotNone(aim.physics.responses);self.assertIsNotNone(sparrow.physics.responses)
        self.assertEqual(aim.physics.responses.shapes,sparrow.physics.responses.shapes)
        self.assertNotEqual(aim.params.rocket['guidance']['guidanceAutopilot'],sparrow.params.rocket['guidance']['guidanceAutopilot'])
        self.assertAlmostEqual(aim.physics.motor(100.)[1],aim.params.rocket['mass']-aim.params.rocket['massEnd1'],places=4)
        aim.run();self.assertTrue(any(r['measurement_fresh'] for r in aim.rows))
        er=simulate(load_profile('su_r_27er'),s)
        self.assertTrue(any(r['measurement_fresh'] for r in er['samples']))

    def test_packaged_data_and_gui_use_primary(self):
        from missile_gui.server import GuiHandler
        from missile_gui.library import public_profile
        profiles,errors=GuiHandler._profiles(None)
        self.assertFalse(errors);self.assertEqual(len(profiles),122)
        self.assertTrue(all(public_profile(p)['runtime_adapter']=='shared_response_1.2.0rc3' for p in profiles))
        entries=read('resources.json')['missiles']
        self.assertEqual(len(entries),126);self.assertEqual(sum(len(x['variants']) for x in entries.values()),233)

    def test_frozen_legacy_is_explicit(self):
        p=load_profile('su_r_77_1');s=scenario();s['max_simulation_time_s']=.1
        r=simulate(p,s,model_mode='legacy_surface')
        self.assertEqual(r['model']['runtime_adapter'],'effective_single_surface_v1')


if __name__=='__main__':unittest.main()
