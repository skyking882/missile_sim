"""Original resource scales remain live inputs to the common response laws."""
import copy
import math
import unittest
from aim120_model.reconstructed.resources import Parameters,RAW
from aim120_model.reconstructed.responses import SharedResponses,aircraft_inputs
from aim120_model.reconstructed.runtime import StagedPhysics
from aim120_model.reconstructed.python_physics118 import PhysicalState
from aim120_model.reconstructed.python_seeker118 import Seeker
from aim120_model.reconstructed.python_chaff148 import ChaffParameters,cloud_radius


class SharedResponseTests(unittest.TestCase):
    def test_motor_uses_each_original_force_stage_mass_and_common_shape(self):
        for mid in ['us_aim9m_sidewinder','us_aim_120a']:
            with self.subTest(mid=mid):
                params=Parameters(mid);state=PhysicalState([0.,1000.,0.],[300.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.],0.)
                response=SharedResponses();base=StagedPhysics(params,state,1/48,response)
                t=base.stages[0][0]+.2*base.stages[0][1]
                thrust,lost=base.motor(t)
                flat=SharedResponses(shape_overrides={'motor':{'tilt':0.,'bulge':0.}})
                self.assertNotEqual(thrust,flat.motor(base.stages,t)[0])
                changed=copy.deepcopy(params);changed.rocket['force']*=1.5
                altered=StagedPhysics(changed,state,1/48,response)
                self.assertAlmostEqual(altered.motor(t)[0]/thrust,1.5,places=5)
                self.assertEqual(altered.motor(t)[1],lost)
                shaped=SharedResponses(shape_overrides={'motor':{'tilt':.2}})
                self.assertNotEqual(thrust,shaped.motor(base.stages,t)[0])
                final_mass=params.rocket.get('massEnd1',params.rocket['massEnd'])
                self.assertAlmostEqual(base.motor(100.)[1],params.rocket['mass']-final_mass,places=4)

    def test_antenna_and_receiver_use_own_raw_scales_for_active_and_semiactive(self):
        angle=math.radians(10.);beam=[1.,0.,0.];ray=[math.cos(angle),math.sin(angle),0.]
        for mid in ['us_aim_120a','us_aim7f_sparrow']:
            with self.subTest(mid=mid):
                radar=Parameters(mid).rocket['guidance']['radarSeeker'];r=SharedResponses(radar)
                antenna=radar['receiver']['antenna'];gain=r.point_gain('receiver',beam,ray)
                reference=Seeker.antenna_gain(None,beam,ray,antenna['angleHalfSens'],antenna['sideLobesSensitivity'])
                self.assertNotAlmostEqual(gain,reference,places=6)
                wide=copy.deepcopy(radar);wide['receiver']['antenna']['angleHalfSens']*=1.5
                self.assertGreater(SharedResponses(wide).point_gain('receiver',beam,ray),gain)
                shape=SharedResponses(radar,shape_overrides={'antenna':{'main_power':1.5}})
                self.assertNotEqual(shape.point_gain('receiver',beam,ray),gain)
                legs=dict(tx=(1.,10000.,.5,11000.),rx=(1.,10000.,.5,11000.))
                power=r.receiver_power(2.,**legs)
                ranged=copy.deepcopy(radar);ranged['receiver']['range']*=2
                self.assertAlmostEqual(SharedResponses(ranged).receiver_power(2.,**legs)/power,16.)
                powered=copy.deepcopy(radar);powered['transmitter']['power']*=2
                self.assertGreater(SharedResponses(powered).receiver_power(2.,**legs),power)
                scaled=SharedResponses(radar,shape_overrides={'receiver':{'sensitivity':1.2}})
                self.assertAlmostEqual(scaled.receiver_power(2.,**legs)/power,1.2)

    def test_rcs_is_generic_over_original_aircraft_dimensions_and_aspects(self):
        response=SharedResponses();shaped=SharedResponses(shape_overrides={'aircraft':{'aspect_power':2.}})
        for name in ['su_30mkm','f_16c_block_50']:
            with self.subTest(aircraft=name):
                raw=aircraft_inputs(name);value=response.aircraft_sigma(raw,.5)
                bigger=copy.deepcopy(raw);bigger['span']*=2
                self.assertAlmostEqual(response.aircraft_sigma(bigger,.5)/value,2.)
                self.assertNotEqual(shaped.aircraft_sigma(raw,.5),value)
                front=copy.deepcopy(raw);front['aspects'][0]*=2
                self.assertAlmostEqual(response.aircraft_sigma(front,1.)/response.aircraft_sigma(raw,1.),2.)

    def test_cloud_uses_own_raw_signature_global_scales_and_common_shapes(self):
        response=SharedResponses();shaped=SharedResponses(shape_overrides={'chaff':{'density_power':2.2,'reflection_power':4.}})
        for name in ['countermeasure_large_split_launcher_jet','countermeasure_split_launcher_jet']:
            with self.subTest(cloud=name):
                p=ChaffParameters(RAW,name,responses=response);g=p.global_params
                radius=response.cloud_radius(p,0.,200.,1000.,1/48)
                reference=cloud_radius(0.,200.,1000.,1/48,from_speed=g['chaffDispersionFromSpeed'],
                    from_viscosity=g['chaffDispersionFromAirViscosity'],response=g['chaffDispersionSpeedFromAirViscosity'],
                    max_radius=g['chaffCloudMaxRadius'],radius_mult=p.radius_mult)
                self.assertNotEqual(radius,reference)
                self.assertNotEqual(shaped.cloud_radius(p,0.,200.,1000.,1/48),radius)
                bigger=copy.deepcopy(p);bigger.global_params['chaffCloudMaxRadius']*=2
                self.assertAlmostEqual(response.cloud_radius(bigger,0.,200.,1000.,1/48)/radius,2.,places=6)
                signature=response.cloud_intrinsic(p,1.,20.)
                stronger=copy.deepcopy(p);stronger.signature['radar']['multiplier0']*=2
                self.assertAlmostEqual(response.cloud_intrinsic(stronger,1.,20.)/signature,2.,places=6)
                self.assertNotEqual(shaped.cloud_intrinsic(p,1.,20.),signature)

    def test_stage_pressure_knots_are_original_and_all_stages_keep_raw_clocks(self):
        p=Parameters('us_aim7f_sparrow');state=PhysicalState([0.,1000.,0.],[300.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.],0.)
        response=SharedResponses();physics=StagedPhysics(p,state,1/48,response)
        self.assertEqual(physics.stages[0][0],p.rocket['fireDelay'])
        self.assertEqual(physics.stages[1][0],p.rocket['fireDelay']+p.rocket['timeFire'])
        self.assertEqual(physics.motor(p.rocket['fireDelay'])[0],0.)
        points=p.rocket['extPressureToThrustMult'];saved=copy.deepcopy(points)
        low=response.motor(physics.stages,1.,pressure_knots=points,height=0.)[0]
        high=response.motor(physics.stages,1.,pressure_knots=points,height=10000.)[0]
        self.assertGreater(high,low);self.assertEqual(points,saved)


if __name__=='__main__':unittest.main()
