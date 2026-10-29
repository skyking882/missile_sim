"""Causal forward driver: physical step -> observation -> INS -> PN/PID.

Only independent scene sources and a once-only launch state are inputs. No
historical missile samples, native image, emulator or research path is loaded.
"""
import copy
import math
from .python_math118 import *
from .python_physics118 import PhysicalState,release_velocity
from .python_physics148 import Physics148
from .python_guidance118 import Controller
from .responses import SharedResponses,aircraft_inputs
from .python_chaff148 import ChaffParameters,ChaffWorld
from .resources import Parameters,read,RAW
from .types import BodyState,IlluminatorState,SceneSnapshot,SignalSource
from .infrared import InfraredSeeker,emission_profile
from .sarh import SemiActiveSeeker
from .active import ActiveSeeker
from .aircraft_infrared import AircraftInfraredSignature,AircraftEngineState
from ..surface_runtime import ScenarioTarget
from ..aerodynamics import quaternion_from_pitch_yaw

MODEL='shared_response_1.2.0rc3'


class StagedPhysics(Physics148):
    def __init__(self,params,state,dt,responses):
        super().__init__(params.physical(),state,physical_dt=dt)
        self.raw=params.rocket;self.responses=responses
        self.burnout=0.;self.stages=[];previous=self.raw['mass']
        for index in range(3):
            suffix=str(index) if index else ''
            if 'timeFire'+suffix not in self.raw:break
            duration=self.raw['timeFire'+suffix];end=self.raw['massEnd'+suffix]
            delay=self.raw['fireDelay'] if index==0 else self.raw.get('fireDelay'+suffix,0.)
            start=self.burnout+delay;self.burnout=start+duration
            self.stages.append((start,duration,self.raw['force'+suffix],previous-end));previous=end

    def motor(self,elapsed):
        return self.responses.motor(self.stages,elapsed,
            pressure_knots=self.raw.get('extPressureToThrustMult'),height=self.state.p[1])


class ForwardScene:
    def __init__(self,target,config,params,responses=None):
        self.target=target;self.config=config;self.params=params
        self.responses=responses if responses is not None else SharedResponses(shape_overrides=config.get('response_shapes'))
        self.aircraft=config.get('target_aircraft');self.explicit_rcs=config.get('target_rcs_m2')
        self.airframe=config.get('target_airframe')
        if self.airframe is None and self.aircraft and self.explicit_rcs is None and params.rocket.get('guidanceType')!='optical':
            self.airframe=aircraft_inputs(self.aircraft)
        self.thermal=None;self.thermal_provenance=None;self.target_ir={}
        if params.rocket.get('guidanceType')=='optical':
            source=config.get('infrared_source','explicit_bands')
            if source=='legacy_fm_approximation':
                if 'target_ir' in config:raise ValueError('Choose target_ir or legacy_fm_approximation, not both')
                if not self.aircraft:raise ValueError('Legacy FM approximation needs target_aircraft')
                engine=config.get('target_engine')
                required=('throttle','afterburner','running')
                if not isinstance(engine,dict) or any(k not in engine for k in required):
                    raise ValueError('Legacy FM IR needs explicit target_engine.throttle, target_engine.afterburner and target_engine.running')
                self.thermal=AircraftInfraredSignature(self.aircraft,engine_state=AircraftEngineState(**engine))
                self.thermal_provenance=self.thermal.provenance
            elif source=='explicit_bands':
                if 'target_ir' not in config:
                    raise ValueError('IR needs explicit target_ir directional signatures/C0 (C0=24034.501953125); target_aircraft does not supply a recovered heat model')
                if not isinstance(config['target_ir'],dict):raise ValueError('target_ir must map band numbers to directional signatures/C0')
                # JSON object keys are strings; convert once at the public
                # scene boundary, keeping the seeker's internal band contract.
                self.target_ir={int(k):v for k,v in config['target_ir'].items()}
                self.thermal_provenance=dict(kind='explicit_scene_signature',contract='directional signature / C0 before propagation',
                    C0=24034.501953125,producer='caller supplied; not inferred from aircraft throttle or reference engine')
            else:raise ValueError('infrared_source must be explicit_bands or legacy_fm_approximation')
        self.flare_specs={}
        for row in config.get('flares',[]):
            name=row['weapon'];doc=read('raw/'+name+'.blk')
            bullets=[v['bullet']['rocket'] for v in doc.values() if isinstance(v,dict) and isinstance(v.get('bullet'),dict)]
            if doc.get('bullet',{}).get('rocket'):bullets.insert(0,doc['bullet']['rocket'])
            choices=[b for b in bullets if 'infraRed' in b.get('signature',{})]
            if len(choices)!=1:raise ValueError('Flare resource must select one infrared bullet')
            self.flare_specs[name]=choices[0]

    def state_at(self,t):
        value=self.target.state_at(t)
        q=getattr(value,'orientation',None)
        if q is None:q=getattr(value,'q',None)
        if q is None:q=self.config.get('target_quaternion_xyzw')
        if q is None:
            # This authored kinematic scene aligns its synthetic body with its
            # declared course; a supplied body attitude always takes priority.
            forward=unit(value.velocity)
            yaw=math.atan2(forward[2],forward[0]);pitch=math.asin(clip(forward[1],-1.,1.))
            q=quaternion_from_pitch_yaw(pitch,yaw);q=(q[1],q[2],q[3],q[0])
        hidden=any(a<=t<b for a,b in self.config.get('target_absent_intervals',[]))
        return BodyState(tuple(value.position),tuple(value.velocity),q,not hidden)

    def snapshot(self,t,missile):
        target=self.state_at(t);illum=None;cfg=self.config.get('illuminator')
        if cfg is not None:
            p=add(cfg['position_m'],mul(cfg['velocity_mps'],t))
            direction=unit(sub(target.p,p)) if cfg.get('track_scene_target',False) else cfg['direction']
            powered=bool(cfg['enabled']) and not any(a<=t<b for a,b in cfg.get('off_intervals',[]))
            illum=IlluminatorState(tuple(p),tuple(cfg['velocity_mps']),tuple(direction),powered,
                cfg.get('target_identity','target'),cfg.get('waveform','pd'),t)
        rcs=self.explicit_rcs
        if rcs is None and self.airframe is not None:
            n=unit(sub(missile.position,target.p));mu=tv(matrix(target.q),n)[0]
            rcs=self.responses.aircraft_sigma(self.airframe,mu)
        thermal=self.thermal(t,target,missile) if self.thermal else self.target_ir
        flares=[]
        for index,row in enumerate(self.config.get('flares',[])):
            age=t-row['time_s']
            if age<0:continue
            raw=self.flare_specs[row['weapon']];origin=self.state_at(row['time_s'])
            velocity=add(origin.v,row['eject_velocity_mps'])
            position=add(origin.p,mul(velocity,age));position[1]-=.5*9.8100004196167*age*age
            signature=raw['signature']['infraRed'];profile=signature['emissionProfile']
            envelope=emission_profile(age,(0.,*profile))
            emission={int(k[8:]):v*envelope for k,v in signature.items() if k.startswith('emission') and k[8:].isdigit()}
            flares.append(SignalSource('flare:'+str(index),tuple(position),tuple(velocity),emission))
        return SceneSnapshot(target,illum,thermal,tuple(flares),(),rcs)


class ReconstructedRuntime:
    def __init__(self,profile,scenario,*,scene=None):
        from ..public_api import validate_scenario,_case_from_scenario
        self.profile=profile;self.scenario=validate_scenario(scenario);self.input=copy.deepcopy(scenario)
        self.params=Parameters(profile['missile_id'],loft=self.scenario['loft_enabled'])
        p=self.params;raw=p.rocket;guidance=raw['guidance'];radar=guidance.get('radarSeeker')
        self.kind='ir' if raw.get('guidanceType')=='optical' else 'arh' if radar and radar['active'] else 'sarh'
        if self.kind=='sarh' and scene is None and 'illuminator' not in scenario:
            raise ValueError('SARH requires an explicit illuminator scene input; no automatic DL/truth substitution')
        if self.kind=='sarh' and scenario.get('chaff'):
            raise NotImplementedError('SARH chaff association is not reconstructed')
        if self.kind!='ir' and radar is None:raise NotImplementedError('Missing supported radar seeker branch')
        self.dt=self.scenario['simulation_dt_s'] or 1/48
        # Recovered wrapper and physical cadence are 48 Hz. Sub-stepping is
        # deliberately not invented for this candidate.
        if abs(self.dt-1/48)>1e-9:raise ValueError('Reconstructed caller requires simulation_dt_s=1/48')
        q=quaternion_from_pitch_yaw(math.radians(scenario['launch_pitch_deg']),math.radians(scenario['launch_heading_deg']))
        q=[q[1],q[2],q[3],q[0]];velocity=mul([x[0] for x in matrix(q)],scenario['launch_speed_kmh']/3.6)
        if scenario.get('launch_state'):
            birth=scenario['launch_state'];position=list(birth['position_m']);velocity=list(birth['velocity_mps']);q=list(birth['quaternion_xyzw']);omega=list(birth['omega_rads'])
        else:
            position=[0.,scenario['launch_altitude_m'],0.];omega=[0.,0.,0.]
            velocity=release_velocity(q,velocity,raw['startSpeed'] if raw.get('useStartSpeed',False) else 200.,.333 if raw['fireDelay']<=0 else 1.)
        state=PhysicalState(position,velocity,q,omega,0.)
        self.responses=SharedResponses(radar,shape_overrides=scenario.get('response_shapes'))
        self.physics=StagedPhysics(p,state,self.dt,self.responses);self.controller=Controller(p)
        self.target=ScenarioTarget(_case_from_scenario(self.scenario)['initial_conditions'])
        self.scene=ForwardScene(self.target,scenario,p,self.responses) if scene is None else scene
        self.wind=list(scenario.get('wind_mps',[0.,0.,0.]));self.noise=Noise(0x42c5ffb1)
        self.ins=None;self.tick=0;self.rows=[];self.event='lifetime';self.done=False
        self.birth_designation=scenario.get('birth_designation')
        if self.birth_designation is not None:
            self.ins=dict(p=list(self.birth_designation['position_m']),v=list(self.birth_designation['velocity_mps']),error=[0.,0.,0.])
        body=self.body();snap=self.snapshot(0.,body)
        if self.kind=='ir':self.seeker=InfraredSeeker(raw)
        elif self.kind=='sarh':self.seeker=SemiActiveSeeker(raw,defaults=p.radar_defaults(),response_model=self.responses)
        else:
            if snap.target_rcs_m2 is None:raise ValueError('Radar scene needs original target_airframe scales, a documented target_aircraft, or explicit target_rcs_m2')
            records=[]
            for index,row in enumerate(scenario.get('chaff',[])):
                origin=self.scene.state_at(row['time_s'])
                records.append(dict(eid=100+index,owner_eid=1,weapon_name=row['weapon'],creation_raw_fields=dict(
                    creation_time=row['time_s'],starting_pos=list(origin.p),u12_2=add(origin.v,row['eject_velocity_mps']),u16_1=list(origin.q))))
            world=ChaffWorld(records,ChaffParameters(RAW,responses=self.responses),self.wind)
            bounds=scenario.get('target_bounds_m')
            self.seeker=ActiveSeeker(p,q,world,responses=self.responses,rcs_m2=snap.target_rcs_m2,bounds=bounds)
        self.previous_target=snap.target

    def body(self):
        s=self.physics.state;return BodyState(tuple(s.p),tuple(s.v),tuple(s.q))

    def snapshot(self,t,body):
        return self.scene.snapshot(t,body) if hasattr(self.scene,'snapshot') else self.scene(t,body)

    def auxiliary(self):
        s=self.physics.state
        if self.ins is None:return dict(valid=False,n=[1.,0.,0.],w=[0.,0.,0.],R=0.,D=0.)
        r=sub(add(self.ins['p'],self.ins['error']),s.p);R=norm(r);n=unit(r)
        w=mul(cross(sub(self.ins['v'],s.v),r),1/(R*R)) if R else [0.,0.,0.]
        return dict(valid=True,n=n,w=w,R=R,D=dot(sub(s.v,self.ins['v']),n))

    def update_ins(self,measurement,snapshot,dt):
        s=self.physics.state;g=self.params.rocket['guidance'];ig=g.get('inertialGuidance',{})
        if measurement.held:return 'held_cache'
        if measurement.fresh and measurement.range_valid and measurement.doppler_valid:
            rd=mul(measurement.direction,measurement.range_m)
            # Recovered INS input is range rate, not Doppler. Preserve this
            # separate field for active-radar observer output.
            radial=measurement.metadata.get('radial_mps',measurement.closing_mps)
            self.ins=dict(p=add(s.p,rd),v=add(sub(s.v,cross(measurement.angular_rate,rd)),
                mul(measurement.direction,radial-dot(s.v,measurement.direction))),error=[0.,0.,0.])
            return 'seeker_measurement'
        virtual=self.input.get('virtual_datalink',False)
        cutoff=self.scenario['datalink_disconnect_time_s']
        if virtual and ig.get('datalink',False) and self.scenario['datalink_enabled'] and (cutoff is None or s.time<cutoff):
            # This is an explicitly declared independent scene radio service,
            # not a reconstruction of absent historical carrier packets.
            self.ins=dict(p=list(snapshot.target.p),v=list(snapshot.target.v),error=[0.,0.,0.]);return 'virtual_datalink'
        if self.ins is not None:
            old_direction=unit(sub(add(self.ins['p'],self.ins['error']),s.p))
            if not ig.get('gnss',False):
                drift=ig.get('inertialNavigationDriftSpeed',0.)
                self.ins['error']=[f(self.ins['error'][i]+f(20.293798446655273*dt*drift*self.noise.at(
                    f(f(s.time+f(60*old_direction[i]))*.30000001192092896)))) for i in range(3)]
            self.ins['p']=fv(add(self.ins['p'],mul(self.ins['v'],dt)));return 'cache_prediction'
        return 'invalid'

    def step(self):
        s=self.physics.state;self.tick+=1;end=f(self.tick*self.dt);oldtime=s.time
        if self.kind=='arh' and self.seeker.core.previous_time is None:
            c=self.seeker.core;c.previous_time=s.time;c.previous_p=list(s.p);c.previous_v=list(s.v);c.previous_q=list(s.q)
            c.previous_target=(self.previous_target.p,self.previous_target.v,self.previous_target.q)
        physical=self.physics.step(end,self.wind);dt=f(end-oldtime);body=self.body();scene=self.snapshot(end,body)
        aux=self.auxiliary()
        if self.kind=='arh':
            self.seeker.core.airframe['explicit_rcs']=scene.target_rcs_m2
            measurement=self.seeker.update(end,dt,s,(scene.target.p,scene.target.v,scene.target.q),aux,present=scene.target.present)
        else:measurement=self.seeker.update(end,dt,body,scene)
        ins_source=self.update_ins(measurement,scene,dt);aux_after=self.auxiliary()
        direction=list(measurement.direction) if measurement.valid else aux_after['n']
        omega=list(measurement.angular_rate) if measurement.valid else aux_after['w']
        velocity_argument=list(s.v);R=measurement.range_m if measurement.range_valid else 0.;D=measurement.closing_mps if measurement.doppler_valid else 0.
        argument_source='missile_velocity'
        use_target=self.params.rocket['guidance'].get('useTargetVel',False)
        if measurement.valid and use_target and measurement.range_valid and measurement.doppler_valid:
            velocity_argument=add(cross(omega,mul(direction,R)),mul(direction,D));argument_source='measured_relative_velocity'
        elif not measurement.valid and use_target and aux_after['valid']:
            R,D=aux_after['R'],aux_after['D'];velocity_argument=add(cross(omega,mul(direction,R)),mul(direction,D));argument_source='INS_relative_velocity'
        tgo=R/D if abs(D)>3.99999987306209e-19 else 0.
        autopilot=self.params.rocket['guidance']['guidanceAutopilot']
        pid=sorted((v for k,v in autopilot.items() if k.startswith('pid') and k[3:].isdigit()),key=lambda x:x['time'])
        if pid:
            for rawkey,key in [('accelControlProp','prop'),('accelControlIntg','intg'),('accelControlDiff','diff'),('accelControlIntgLim','intgLim')]:
                autopilot[rawkey]=curve([[p['time'],p[key]] for p in pid],end)
        allowed=(measurement.valid or aux_after['valid']) and end>=autopilot.get('timeOut',0.)
        control=self.controller.step(s,end,dt,direction,omega,velocity_argument,physical['feedback'],tgo,R*R,allowed)
        self.physics.control=control
        distance=norm(sub(scene.target.p,s.p));forward=[x[0] for x in matrix(s.q)]
        track=measurement.mode if measurement.valid else 'ins_search' if aux_after['valid'] else 'lost'
        request=self.controller.last.get('request',[0.,0.,0.]);accel=physical['acceleration']
        row=dict(time_s=end,position_m=list(s.p),velocity_mps=list(s.v),target_position_m=list(scene.target.p),target_velocity_mps=list(scene.target.v),
            distance_to_target_m=distance,speed_kmh=norm(s.v)*3.6,altitude_m=s.p[1],pitch_deg=math.degrees(math.asin(clip(forward[1],-1.,1.))),
            yaw_deg=math.degrees(math.atan2(forward[2],forward[0])),quaternion_xyzw=list(s.q),omega_rads=list(s.omega),
            mass_kg=physical['mass'],thrust_n=physical['thrust'],commanded_acceleration_g=mul(request,1/9.81),
            mach=norm(s.v)/atmosphere(s.p[1])[1],drag_n=-dot(physical['aerodynamic_force'],unit(s.v)),loft_active=bool(self.controller.loft_mode),
            actual_overload_g=norm(add(accel,[0.,9.81,0.]))/9.81,trajectory_lateral_load_g=norm(cross(unit(s.v),accel))/9.81,
            control=list(control),control_consumed=physical['control_consumed'],observation_mode='sensor_track',observation_provider='reconstructed_'+self.kind,
            track_mode=track,seeker_state=measurement.mode,seeker_display_state=measurement.mode.upper(),track_valid=measurement.valid,
            observed_direction=direction,observed_angular_rate=omega,observed_range_m=measurement.range_m,observed_closing_mps=measurement.closing_mps,
            range_valid=measurement.range_valid,doppler_valid=measurement.doppler_valid,measurement_fresh=measurement.fresh,measurement_held=measurement.held,
            ins_source=ins_source,guidance_velocity_argument=velocity_argument,guidance_velocity_source=argument_source,
            observation_reject_reason=measurement.metadata.get('reason',''),sensor_source=measurement.source)
        if self.kind=='sarh':row['bistatic']=measurement.metadata
        if self.kind=='arh':
            row['wrapper_mode']=measurement.metadata['wrapper_mode'];row['echo_contributions']=measurement.metadata['signal'].get('contributions',[])
        self.rows.append(row);self.previous_target=scene.target
        r=self.params.rocket
        if s.p[1]<=0:self.event='ground';self.done=True
        elif norm(sub(s.p,[0.,self.scenario['launch_altitude_m'],0.]))>=r['maxDistance']:self.event='max_distance';self.done=True
        elif measurement.valid and end>=r.get('proximityFuse',{}).get('timeOut',0.) and distance<=r.get('explodeTreshold',0.):self.event='fuse';self.done=True
        return row

    def run(self):
        requested=self.scenario['max_simulation_time_s'];end=min(self.params.rocket['timeLife'],requested) if requested is not None else self.params.rocket['timeLife']
        while not self.done and self.physics.state.time+1e-7<end:self.step()
        return self.result()

    def result(self):
        from ..public_api import _summarize
        rawresult=dict(samples=self.rows,event_type=self.event,observation_mode='sensor_track',observation_provider='reconstructed_'+self.kind)
        config=copy.deepcopy(self.profile);config['guidance']['lofting_enabled']=self.params.rocket['guidance']['guidanceAutopilot']['loftEnabled']
        summary=_summarize(rawresult,config,True,self.scenario['loft_enabled'],self.profile['guidance']['lofting_enabled'])
        summary['burnout_time_s']=self.physics.burnout
        summary['maximum_track_error_m']=None # An angular observation has no fabricated 3D track error.
        return dict(schema_version=1,missile=dict(id=self.profile['missile_id'],name=self.profile['display_name'],status='experimental',
            parameter_source=dict(version='2.59.0.80',representation='original JSON .blk',file=self.params.entry['file'])),
            scenario=self.input,summary=summary,markers=[dict(time_s=self.physics.burnout,label='燃尽',kind='burnout')],samples=self.rows,
            model=dict(label=MODEL,runtime_adapter=MODEL,observation_provider='reconstructed_'+self.kind,time_step_s=self.dt,
                dynamics='reconstructed nominal-inertia rigid body',control='reconstructed parallel acceleration PID',
                response_functions=dict(scope='same functions and common internal shape set for every supported model',
                    motor='own original force/duration/delay/mass endpoints/pressure knots * shared unit-integral stage shape',
                    radar='own original beam widths/sidelobes/reference rcs/range/power * shared dimensionless response',
                    aircraft='own original length/span/aspect factors * shared projected-area/log-aspect law; explicit measured RCS remains an input contract',
                    chaff='own original body/cloud/global scales and signature multiplier/envelope * shared expansion/reflection shapes'),
                response_shapes=self.responses.shapes,response_shape_sources=read('response_shapes.json')['parameter_sources'],
                thermal_source=getattr(self.scene,'thermal_provenance',None),
                SARH_signal_model=dict(status='derived_approximation',formula='shared raw-resource-scaled two-leg endpoint response; saturating original Tx power and receiver reference normalization',
                    native_closure='full bistatic echo normalization/association not recovered') if self.kind=='sarh' else None,
                support=dict(IR='point-source bands, angle observer, gating, IRCCM hold/loss/reacquire; no range/Doppler',
                    ARH='zero-sector relative-Doppler capture/hold/loss/reacquire and common raw-scaled chaff responses',
                    SARH='explicit sampled PD illuminator, geometric bistatic Doppler and measured-angle observer; echo power is a derived approximation, not a closed native reconstruction',
                    unsupported=['TVC/jet vanes','CW/absolute Doppler','radar nonzero-sector scans','radar clutter spectrum/notch intervals',
                        'SARH chaff association','IR resolved-target/world occlusion/sun/background']),
                boundary='common response-function candidate with unchanged original resource coefficients; recovered control/state ordering and explicit scene inputs, not full game equivalence'))


def simulate_reconstructed(profile,scenario,*,scene=None):
    return ReconstructedRuntime(profile,scenario,scene=scene).run()
