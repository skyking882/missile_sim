"""Shared effective single-surface runtime with profile-specific game inputs.

The reference coefficients are a common approximation, not a per-missile
accuracy claim. Geometry scales the same force/moment laws for every profile.
Guidance consumes copied track solutions; truth is used only for propagation,
observation generation, terminal events and plotted target telemetry.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .effective_surface_core import EffectiveSurfaceGame, G, add, scale, sub, norm, axes, body, dot, cross, table
from .aerodynamics import quaternion_from_pitch_yaw
from .control import scheduled_pid_values
from .dynamics import SimState
from .observation import IdealTruthTrackProvider, KinematicTrackProvider, SensorTrackProvider, validate_min_altitude
from .target import TargetModel, TargetState
from .events import event_candidates
from .effective_surface_core import normalized
from .angular_guidance import GuidanceInputProvider

MODE = "effective_single_surface_v1"


class ScenarioTarget:
    def __init__(self, initial):
        self.model = TargetModel(initial, G)
        self.initial = self.model.initial_state

    def state_at(self, t):
        p, v = self.initial.position, self.initial.velocity
        speed = math.hypot(v[0], v[2])
        if not self.model.turn_g or not speed:
            return TargetState(add(p, scale(v, t)), v)
        omega = self.model.turn_g*G/speed
        c, s = math.cos(omega*t), math.sin(omega*t)
        return TargetState((p[0]+(v[0]*s+v[2]*(c-1))/omega, p[1]+v[1]*t,
                            p[2]+(v[0]*(1-c)+v[2]*s)/omega),
                           (c*v[0]-s*v[2], v[1], s*v[0]+c*v[2]))


def _launch_quaternion(value):
    """Normalize a body-to-world quaternion in scalar-last xyzw order."""
    if isinstance(value,(str,bytes)):
        raise ValueError('launch_quaternion_xyzw must be four finite numbers with nonzero norm')
    try: q=tuple(float(x) for x in value)
    except (TypeError,ValueError) as exc:
        raise ValueError('launch_quaternion_xyzw must be four finite numbers with nonzero norm') from exc
    if len(q)!=4 or not all(math.isfinite(x) for x in q) or max(map(abs,q))==0:
        raise ValueError('launch_quaternion_xyzw must be four finite numbers with nonzero norm')
    size=max(map(abs,q));scaled=tuple(x/size for x in q);length=math.hypot(*scaled)
    return tuple(x/length for x in scaled)


def launch_quaternion_xyzw_from_dagor(value):
    """Native missile creation body-to-world xyzw to runtime xyzw.

    Dagor uses x east/y up/z north; runtime uses x east/y up/z south.
    Reflect both world and body bases: R_runtime = C R_native C with
    C=diag(1,1,-1). This retains the complete attitude, including roll,
    independently of launch velocity. ``value`` must belong to launch time.
    """
    x,y,z,w=_launch_quaternion(value)
    return (-x,-y,z,w)


def launch_quaternion_xyzw_from_velocity_normal(velocity,normal,aoa_deg):
    """Body-to-world xyzw frame from velocity, lift normal and signed AoA.

    Both vectors use world x/y-up/z. Body axes are forward/up/right. The
    normal defines roll about velocity; its component along velocity is
    removed. No sideslip or relative mount rotation is introduced.
    """
    def vector(value,label):
        try: out=tuple(float(x) for x in value)
        except (TypeError,ValueError) as exc:raise ValueError(label+' must be a finite xyz vector') from exc
        if len(out)!=3 or not all(math.isfinite(x) for x in out):raise ValueError(label+' must be a finite xyz vector')
        return out
    v=vector(velocity,'velocity');n=vector(normal,'normal')
    speed=math.hypot(*v)
    if speed==0. or not math.isfinite(speed):raise ValueError('velocity must have a finite nonzero norm')
    if isinstance(aoa_deg,bool) or not isinstance(aoa_deg,(int,float)) or not math.isfinite(aoa_deg):raise ValueError('aoa_deg must be finite')
    tangent=scale(v,1/speed);n=sub(n,scale(tangent,dot(n,tangent)));length=math.hypot(*n)
    if length<=1e-12 or not math.isfinite(length):raise ValueError('normal must define a nonzero transverse lift direction')
    n=scale(n,1/length);alpha=math.radians(aoa_deg);c,s=math.cos(alpha),math.sin(alpha)
    f=add(scale(tangent,c),scale(n,s));up=add(scale(tangent,-s),scale(n,c));right=cross(f,up)
    # Rows of the proper rotation whose columns are forward/up/right.
    m=tuple(zip(f,up,right));trace=sum(m[i][i] for i in range(3))
    if trace>0:
        z=2*math.sqrt(1+trace);q=((m[2][1]-m[1][2])/z,(m[0][2]-m[2][0])/z,(m[1][0]-m[0][1])/z,z/4)
    else:
        i=max(range(3),key=lambda k:m[k][k]);j=(i+1)%3;k=(i+2)%3
        z=2*math.sqrt(1+m[i][i]-m[j][j]-m[k][k]);xyz=[0.,0.,0.]
        xyz[i]=z/4;xyz[j]=(m[j][i]+m[i][j])/z;xyz[k]=(m[k][i]+m[i][k])/z
        q=tuple(xyz)+((m[k][j]-m[j][k])/z,)
    return _launch_quaternion(q)


class SurfaceRuntime(EffectiveSurfaceGame):
    def __init__(self, profile, scenario, case, defaults, target_factory=None, *,
                 launch_position_m=None, launch_velocity_mps=None, target=None, min_altitude_m=None,
                 launch_quaternion_xyzw=None, guidance_input=None, infrared_seeker=None):
        min_altitude_m=validate_min_altitude(min_altitude_m)
        if min_altitude_m is not None and scenario['observation_mode']!='sensor_track':
            raise ValueError('min_altitude_m needs the sensor_track observation mode')
        self.profile, self.scenario = profile, scenario
        geometry, aero, control, guidance = (profile[k] for k in ('geometry','aerodynamics','control','guidance'))
        self.assumptions = []
        ref = defaults['reference']
        def supplied(node, key, common):
            value = node.get(key)
            if value is None:
                self.assumptions.append(f'{key} not declared: shared value {common}')
                return common
            return float(value)
        fins = aero['fin_aoa_limit_rad']
        pitch_limit = supplied(fins,'horizontal',.698132)
        yaw_limit = supplied(fins,'vertical',.698132)
        lift_scale = supplied(aero,'fins_lateral_acceleration_g',ref['fins_lateral_acceleration_g'])/ref['fins_lateral_acceleration_g']
        lift_scale *= supplied(geometry,'wing_area_multiplier',ref['wing_area_multiplier'])/ref['wing_area_multiplier']
        lift_scale *= .698132 / ((pitch_limit+yaw_limit)/2)
        cfg = copy.deepcopy(defaults)
        cfg['forward_duration_s'] = profile['performance']['lifetime_s']
        p = cfg['effective_parameters']
        p['lift_sine_slope'] *= lift_scale
        drag_scale = float(aero['cx_k'])/ref['cx_k']
        p['cd_base_inverse_mach'] = [v*drag_scale for v in p['cd_base_inverse_mach']]
        area_scale = (geometry['caliber_m']/ref['caliber_m'])**2
        arm_scale = geometry['fin_moment_arm_m']/ref['arm_m']
        moment_scale = area_scale*arm_scale*(ref['length_m']/geometry['length_m'])**2
        p['control_response_rad_s2'] *= moment_scale*lift_scale
        p['restoring_response_rad_s2_per_rad'] *= moment_scale*lift_scale
        p['damping_response_per_s'] *= moment_scale*arm_scale*lift_scale
        pid = control.get('pid_schedule')
        self.pid_source = {'schedule':pid} if pid else control['pid']
        gains = scheduled_pid_values(self.pid_source,0)
        autopilot = dict(propNavMult=guidance['pn_gain'], reqAccelMax=guidance['maximum_lateral_acceleration_g'],
            baseIndSpeed=control['base_indicated_speed_kmh'], accelControlProp=gains['p'],
            accelControlIntg=gains['i'],accelControlDiff=gains['d'],accelControlIntgLim=gains['integral_limit'],
            loftEnabled=bool(guidance['lofting_enabled'] and scenario['loft_enabled']),
            loftElevation=guidance['lofting_elevation_deg'])
        for key,field in [('loftTargetElevation','loft_target_elevation_deg'),('loftAngleToAccelMult','loft_angle_to_acceleration_multiplier'),('loftTargetOmegaMax','loft_omega_max_deg_s')]:
            autopilot[key]=guidance.get(field)
            if autopilot['loftEnabled'] and autopilot[key] is None:
                raise ValueError(f'Enabled loft requires profile field {field}')
        autopilot['_age_gain'] = guidance.get('flight_time_gain_table') or [[0.,1.]]
        autopilot['_hit_gain'] = guidance.get('time_to_hit_gain_table') or [[0.,1.]]
        for field in ['flight_time_gain_table','time_to_hit_gain_table']:
            if not guidance.get(field):self.assumptions.append(f'{field} not declared: unit gain')
        raw = dict(mass=geometry['initial_mass_kg'],caliber=geometry['caliber_m'],
                   distFromCmToStab=geometry['fin_moment_arm_m'],guidance={'guidanceAutopilot':autopilot})
        pitch,yaw=map(math.radians,(scenario['launch_pitch_deg'],scenario['launch_heading_deg']))
        wxyz=quaternion_from_pitch_yaw(pitch,yaw)
        q=(wxyz[1],wxyz[2],wxyz[3],wxyz[0])
        velocity_direction=axes(q)[0]
        if launch_quaternion_xyzw is not None:q=_launch_quaternion(launch_quaternion_xyzw)
        target_initial=(TargetModel(case['initial_conditions'],G).initial_state if target is None
                        else target.state_at(0.))
        creation=dict(position_xyz_m=([0.,scenario['launch_altitude_m'],0.] if launch_position_m is None
                                     else tuple(launch_position_m)),
            velocity_xyz_mps=(scale(velocity_direction,scenario['launch_speed_kmh']/3.6) if launch_velocity_mps is None
                              else tuple(launch_velocity_mps)),
            mass_source_kg=geometry['initial_mass_kg'],quaternion_xyzw_raw=q,omega_export_raw=[0.,0.,0.],
            target_position_xyz_m=target_initial.position,target_velocity_xyz_mps=target_initial.velocity,
            scenario_exact={'dt':1/48,'loft_state':'config' if autopilot['loftEnabled'] else 'off'})
        super().__init__(raw,cfg,creation)
        self.v0=ref['speed_mps']
        self.qm0=.5*self.atmosphere.sample(ref['altitude_m']).density_kg_m3*self.v0**2/ref['mass_kg']
        self.stages=profile['propulsion']['stages']
        self.track=None
        self.require_seeker_lock=False
        self.target=ScenarioTarget(case['initial_conditions']) if target is None else target
        # Opt-in: wrap the scenario target (e.g. EvasiveTarget); default path unchanged.
        if target_factory is not None:self.target=target_factory(self.target)
        self.wind=(0.,0.,0.)
        if scenario['simulation_dt_s'] is not None and abs(scenario['simulation_dt_s']-self.dt)>1e-10:
            raise ValueError('Shared effective runtime uses a 48 Hz controller')
        if scenario['observation_mode']=='ideal_truth':
            self.provider=IdealTruthTrackProvider()
        else:
            sensor=guidance.get('sensor_model')
            if sensor and sensor.get('provider','radar_datalink_ins_v1')!='profile_kinematic_v1':
                self.provider=SensorTrackProvider(sensor,lock_range_m=guidance.get('lock_range_m'),
                    datalink_enabled=scenario['datalink_enabled'] and sensor.get('datalink',True),
                    datalink_disconnect_time_s=scenario['datalink_disconnect_time_s'],
                    inertial_drift_direction=tuple(scenario['inertial_drift_direction']),
                    min_altitude_m=min_altitude_m)
            else:
                self.provider=KinematicTrackProvider(lock_range_m=guidance.get('lock_range_m'),
                    maximum_angular_rate_deg_s=guidance.get('maximum_angular_rate_deg_s'),
                    seeker_type=guidance.get('seeker_type','unknown'),min_altitude_m=min_altitude_m)

        if guidance_input is not None:
            if min_altitude_m is not None:
                raise ValueError('min_altitude_m cannot alter external guidance inputs')
            self.provider=GuidanceInputProvider(guidance_input)

        if infrared_seeker is not None:
            from .infrared_seeker import InfraredTrackProvider
            if scenario['observation_mode'] != 'sensor_track' or guidance.get('seeker_type') != 'optical':
                raise ValueError('infrared_seeker needs an optical sensor_track profile')
            if guidance_input is not None or min_altitude_m is not None:
                raise ValueError('infrared_seeker cannot combine with guidance_input or min_altitude_m')
            self.provider=InfraredTrackProvider(infrared_seeker,self.target)
            thermal=infrared_seeker.thermal_provenance
            if thermal is not None:
                version=thermal['version']
                if profile.get('provenance',{}).get('source_version') != version:
                    raise ValueError('infrared_seeker aircraft data and missile profile must use the same game version')
                flare_sources=getattr(getattr(self.target,'spec',None),'parameter_sources',None)
                flare_source=None if flare_sources is None else flare_sources.get('source')
                if flare_source is not None and flare_source['version'] != version:
                    raise ValueError('infrared_seeker and flare sources must use the same game version')

        self._begun=False
        self.time_s=0.
        self.done=False
        self.event=None
        self.state=(tuple(self.creation['position_xyz_m'])+tuple(self.creation['velocity_xyz_mps'])
                    +tuple(self.creation['quaternion_xyzw_raw'])+(0.,0.,0.))

    def mass_thrust(self,t):
        mass=self.raw['mass']; clock=0.;thrust=0.
        for stage in self.stages:
            start=clock+float(stage.get('fire_delay_s') or 0.)
            end=start+stage['duration_s']
            fraction=max(0.,min(1.,(t-start)/stage['duration_s']))
            mass-=stage['mass_lost_kg']*fraction
            if start<=t<end:thrust+=stage['thrust_n']
            clock=end
        if mass<=0:raise ValueError('Profile propulsion consumes all initial mass')
        return mass,thrust

    def coefficients(self,mach,alpha):
        c0,c1,c2=self.p['cd_base_inverse_mach']
        join=-2*c2/c1
        # The inverse-Mach fit has a stationary positive maximum here. Its
        # constant subsonic continuation has matching value and zero slope.
        m=max(mach,join)
        cd=c0+c1/m+c2/m**2
        if cd<=0:raise ValueError('Shared drag coefficient must be positive')
        stall=max(alpha-self.p['lift_stall_rad'],0.)/self.p['lift_rolloff_width_rad']
        return cd,self.p['cd_incidence_ratio']*cd*math.sin(alpha)**2,self.p['lift_sine_slope']*math.sin(alpha)*math.exp(-stall*stall)

    def guidance(self,t,state):
        if self.track is None or not self.track.valid:return (0.,0.,0.)
        # Only the provider's copied solution enters the reference PN formula.
        self.target0=sub(self.track.position,scale(self.track.velocity,t))
        self.target_velocity=self.track.velocity
        if norm(sub(self.track.position,state[:3]))<=1e-12:return (0.,0.,0.)
        return super().guidance(t,state)

    def guidance_kinematics(self,t,state):
        if self.track.full_rate_rad_s is not None:
            # The source has already transferred the angular state to this
            # missile. Do not remove its independent LOS-axis component.
            return (sub(self.track.position,state[:3]), sub(self.track.velocity,state[3:6]),
                    scale(self.track.full_rate_rad_s,-1.))
        return super().guidance_kinematics(t,state)

    def update_control(self,t,state):
        if self.track is None or not self.track.valid:
            self.last_request=(0.,0.,0.);self.last_error=(0.,0.)
            return (0.,0.)
        gains=scheduled_pid_values(self.pid_source,t)
        for field,key in [('accelControlProp','p'),('accelControlIntg','i'),('accelControlDiff','d'),('accelControlIntgLim','integral_limit')]:self.autopilot[field]=gains[key]
        if norm(state[3:6])==0:
            self.last_request=(0.,0.,0.);self.last_error=(0.,0.)
            return (0.,0.)
        return super().update_control(t,state)

    def missile_state(self,state):
        f=axes(state[6:10])[0]
        return SimState(position=state[:3],velocity=state[3:6],pitch=math.asin(max(-1.,min(1.,f[1]))),
            yaw=math.atan2(f[2],f[0]),pitch_rate=state[12],yaw_rate=-state[11],mass=self.mass_thrust(0)[0],
            orientation_quaternion=(state[9],state[6],state[7],state[8]))

    def begin(self,end_time,early_miss_s=None):
        """Begin once and record the launch sample; return this runtime.

        Time is seconds since launch. The controller remains fixed at 1/48 s.
        ``early_miss_s`` is an explicit opt-in shortcut; omitted means full flight.
        A runtime is single-use because its controller and observer carry history.
        """
        if self._begun:
            raise RuntimeError('SurfaceRuntime has already begun; create a new instance')
        if isinstance(end_time,bool) or not isinstance(end_time,(int,float)) or not math.isfinite(end_time) or end_time<0:
            raise ValueError('end_time must be a finite nonnegative number')
        if early_miss_s is not None and (isinstance(early_miss_s,bool) or not isinstance(early_miss_s,(int,float))
                                        or not math.isfinite(early_miss_s) or early_miss_s<=0):
            raise ValueError('early_miss_s must be a finite positive number')
        self._end_time=end_time
        self._early_miss_s=early_miss_s
        self._rows=[]
        self._initial_position=self.state[:3]
        self._terminal_event=None
        self._miss_since=None
        self._burnout=sum(float(s.get('fire_delay_s') or 0.)+s['duration_s'] for s in self.stages)
        self._event_config=copy.deepcopy(self.profile)
        self._event_config['performance']['lifetime_s']=end_time
        self._observe=getattr(self.target,'observe_missile',None)
        self._decoys_at=getattr(self.target,'decoys_at',None) if getattr(self.provider,'supports_decoys',False) else None
        self._begun=True
        self.event=self._record_sample()
        self.done=self.event is not None
        return self

    def _record_sample(self):
        state,time,rows=self.state,self.time_s,self._rows
        observe,decoys_at=self._observe,self._decoys_at
        terminal_event=self._terminal_event
        initial_position=self._initial_position
        end_time,early_miss_s,burnout=self._end_time,self._early_miss_s,self._burnout
        performance=self.profile['performance'];fuse=self.profile['guidance']['proximity_radius_m']
        if observe is not None:observe(time,state[:3],state[3:6])
        truth=self.target.state_at(time)
        self._truth=truth
        if decoys_at is None and isinstance(self.provider,SensorTrackProvider):
            self.track=self.provider.update(time,0. if not rows else self.dt,self.missile_state(state),truth,
                target_rcs_m2=getattr(self.target,'rcs_m2',1.))
        elif decoys_at is None:
            self.track=self.provider.update(time,0. if not rows else self.dt,self.missile_state(state),truth)
        else:
            self.track=self.provider.update(time,0. if not rows else self.dt,self.missile_state(state),truth,
                decoys=decoys_at(time),target_rcs_m2=getattr(self.target,'rcs_m2',1.))
        _,d=self.plant(time,state,self.control)
        distance=norm(sub(truth.position,state[:3]))
        track_error=norm(sub(self.track.position,truth.position))
        detection=getattr(self.provider,'last_detection',None)
        row=dict(time_s=time,position_m=state[:3],velocity_mps=state[3:6],target_position_m=truth.position,target_velocity_mps=truth.velocity,
            orientation_quaternion=(state[9],state[6],state[7],state[8]),quaternion_xyzw=state[6:10],pitch_rad=math.radians(d['body_pitch_deg']),yaw_rad=math.radians(d['body_yaw_deg']),
            alpha_deg=d['geometric_incidence_unsigned_deg'],actual_overload_g=d['normal_specific_load_g'],
            trajectory_lateral_load_g=d['normal_trajectory_load_g'],commanded_acceleration_g=scale(self.last_request,1/G),
            distance_to_target_m=distance,mach=d['mach'],thrust_n=d['thrust_n'],drag_n=d['drag_base_n']+d['drag_aoa_n'],mass_kg=d['mass_kg'],
            loft_active=self.loft_phase=='climb',loft_weight=1. if self.loft_phase=='climb' else 0.,loft_bias_g=0.,
            track_mode=self.track.mode.value,observation_mode=self.scenario['observation_mode'],observation_provider=self.provider.provider_name,
            seeker_state=getattr(getattr(self.provider,'seeker_state',None),'value','not_applicable'),
            seeker_display_state=getattr(self.provider,'seeker_display_state','—'),track_position_error_m=track_error,
            radar_reject_reason=getattr(self.provider,'observation_reject_reason',''),observation_reject_reason=getattr(self.provider,'observation_reject_reason',''),control=self.control,
            observed_target_position_m=self.track.position,observed_target_velocity_mps=self.track.velocity,track_valid=self.track.valid)
        rows.append(row)
        from .infrared_seeker import InfraredTrackProvider
        if isinstance(self.provider,InfraredTrackProvider):
            row.update(self.provider.diagnostics())
        if isinstance(self.provider,GuidanceInputProvider) or getattr(self.provider,'angular_observer',None) is not None:
            row['guidance_input_source']=self.track.source
            row['guidance_full_rate_rad_s']=self.track.full_rate_rad_s
            row['guidance_input_sample_time_s']=self.track.sample_time_s
        if hasattr(self,'seeker_search_status'):
            search=getattr(getattr(self.provider,'radar',None),'search',None)
            row['seeker_search_status']=self.seeker_search_status['status']
            row['seeker_scan_offset_deg']=search.scan_offset_deg if search is not None else None
            row['seeker_beam_target_deg']=search.beam_target_deg if search is not None else None
        if getattr(getattr(self.provider,'radar',None),'track_gate_model',None) is not None:
            radar=self.provider.radar
            row['tracked_reflector']=radar.last_reflector if detection.valid else None
            row['tracked_reflector_identity']=radar.track_gate_identity if detection.valid else None
        if terminal_event is not None:return terminal_event
        # Opt-in: a seeker that never locked cannot fuse (user-reported game rule, D).
        if distance<=fuse and (not self.require_seeker_lock or getattr(self.provider,'radar_has_tracked_once',True)):
            return 'fuse'
        if state[1]<=0:return 'ground'
        if norm(sub(state[:3],initial_position))>=performance['maximum_distance_m']:return 'max_distance'
        if early_miss_s is not None and time>=burnout:
            # Opt-in sweep shortcut. An unpowered missile is declared a miss
            # after early_miss_s of: opening from a faster target; opening
            # with the target behind it; or closing too slowly to arrive
            # before end_time. Not a proof: it assumes the closing rate does
            # not recover (no target turn-back, no large dive gain); callers
            # validate against full runs.
            los=sub(truth.position,state[:3]);rate=dot(los,sub(truth.velocity,state[3:6]))/max(distance,1e-9)
            behind=dot(los,state[3:6])<0
            late=rate<0 and distance>-rate*(end_time-time)
            if rate>0 and (behind or norm(state[3:6])<norm(truth.velocity)) or late:
                self._miss_since=time if self._miss_since is None else self._miss_since
                if time-self._miss_since>=early_miss_s-1e-10:return 'early_miss'
            else:self._miss_since=None
        if time>=end_time-1e-10:return 'lifetime'
        return None

    def step(self):
        """Advance one held-control tick, including its endpoint/terminal sample.

        Advance aircraft first, then missiles on the same world clock. A target
        must implement ``state_at(launch_relative_time_s) -> TargetState`` and
        may implement ``observe_missile(time_s, position, velocity)``. Retain
        aircraft history over the tick: queries include both endpoints and any
        interpolated terminal event time inside it, not only saved tick times.
        Convert world time to launch-relative time in the target proxy/callback.
        After termination this method is a no-op; it returns the current state.
        State is an immutable 13-tuple: position xyz, velocity xyz, quaternion
        xyzw and body angular velocity xyz. Coordinates use x/y-up/z; heading
        zero points along +x, positive heading turns toward +z.
        """
        if not self._begun:
            raise RuntimeError('Call begin before step')
        if self.done:
            return self.state
        state,time=self.state,self.time_s
        end_time=self._end_time
        truth=self._truth
        event_config,initial_position=self._event_config,self._initial_position
        self.control=self.update_control(time,state)
        step=min(self.dt,end_time-time)
        old_state=state
        # Split engine discontinuities within the held controller command.
        cuts=[time,time+step];clock=0.
        for stage in self.stages:
            start=clock+float(stage.get('fire_delay_s') or 0.);clock=start+stage['duration_s']
            cuts.extend(x for x in [start,clock] if time<x<time+step)
        cuts=sorted(set(cuts))
        for left,right in zip(cuts,cuts[1:]):
            h=(right-left)/self.integration_substeps
            for j in range(self.integration_substeps):state=super().step(left+j*h,state,self.control,h)
        next_time=time+step
        candidates=event_candidates(self.missile_state(old_state),self.missile_state(state),truth,
            self.target.state_at(next_time),time,next_time,event_config,initial_position)
        if self.require_seeker_lock and getattr(getattr(self.provider,'radar',None),'search',None) is not None                 and not self.provider.radar_has_tracked_once:
            candidates=[candidate for candidate in candidates if candidate.event_type!='fuse']
        if candidates:
            first=candidates[0]
            fraction=first.fraction
            state=tuple(a+fraction*(b-a) for a,b in zip(old_state,state))
            state=state[:6]+normalized(state[6:10])+state[10:]
            next_time=time+fraction*step
            self._terminal_event=first.event_type
        time=next_time
        _,d=self.plant(time,state,self.control);self.previous_acceleration=d['acceleration_xyz_mps2']
        self.state,self.time_s=state,time
        self.event=self._record_sample()
        self.done=self.event is not None
        return self.state

    def result(self):
        """Return the same raw sample/event contract as simulate, after done."""
        if not self._begun or not self.done:
            raise RuntimeError('Result is available only after termination')
        result=dict(samples=self._rows,event_type=self.event,observation_mode=self.scenario['observation_mode'],
                    observation_provider=self.provider.provider_name)
        if hasattr(self,'seeker_search_status'):
            result['seeker_search']=copy.deepcopy(self.seeker_search_status)
            result['summary']=_search_summary(self)
        if getattr(getattr(self.provider,'radar',None),'track_gate_model',None) is not None:
            result['track_gate_model']=copy.deepcopy(self.provider.radar.track_gate_summary())
        if getattr(getattr(self.provider,'radar',None),'chaff_capture',None) is not None:
            result['chaff_capture']=copy.deepcopy(self.provider.radar.capture_summary())
        from .infrared_seeker import InfraredTrackProvider
        if isinstance(self.provider,InfraredTrackProvider):
            result['infrared_seeker']=copy.deepcopy(self.provider.provenance)
        observer=getattr(self.provider,'angular_observer',None)
        if observer is not None:
            result['angular_observer']=dict(observer.provenance, capture_count=observer.capture_count,
                correction_count=observer.correction_count)
        return result

    def simulate(self,end_time,early_miss_s=None):
        """Run begin/step/result to completion on a fresh runtime."""
        self.begin(end_time,early_miss_s)
        while not self.done:
            self.step()
        return self.result()


def simulate_surface(profile,scenario,target_factory=None,early_miss_s=None,clutter_model=None,clutter_min_depression_deg=None,cw_on_clear_beam=False,multipath_gain=None,
                     launcher_radar_gimbal_deg=None,require_seeker_lock=False,launcher_support=None,
                     seeker_search=None, *, min_altitude_m=None,chaff_capture=None,track_gate_model=None,target_observer=None,
                     launch_quaternion_xyzw=None,launcher_radar_geometry=None,guidance_input=None,angular_observer=None,infrared_seeker=None):
    """Run the surface model; min_altitude_m optionally floors INS y-up estimates.

    Pass ``min_altitude_m=50.0`` as a top-level keyword with sensor_track;
    None disables it. Direct radar/datalink observations remain unchanged.
    """
    from .public_api import validate_scenario,_case_from_scenario,_summarize
    from .profile_adapter import unsupported_model_types
    if unsupported_model_types(profile):raise ValueError('Unsupported guidance/physics family')
    normalized=validate_scenario(scenario)
    defaults=json.loads((Path(__file__).resolve().parents[2]/'config/effective_surface_defaults.json').read_text())
    runtime=SurfaceRuntime(profile,normalized,_case_from_scenario(normalized),defaults,target_factory,
                           min_altitude_m=min_altitude_m,launch_quaternion_xyzw=launch_quaternion_xyzw,
                           guidance_input=guidance_input,infrared_seeker=infrared_seeker)
    _configure_runtime(runtime,clutter_model,clutter_min_depression_deg,cw_on_clear_beam,multipath_gain,
                       launcher_radar_gimbal_deg,require_seeker_lock,launcher_support,seeker_search,chaff_capture,track_gate_model,target_observer,launcher_radar_geometry,angular_observer)
    lifetime=profile['performance']['lifetime_s'];requested=normalized['max_simulation_time_s'];end=min(lifetime,requested) if requested is not None else lifetime
    result=runtime.simulate(end,early_miss_s)
    config=copy.deepcopy(profile)
    config['guidance']['lofting_enabled']=runtime.autopilot['loftEnabled']
    summary=_summarize(result,config,requested is not None and requested<lifetime,normalized['loft_enabled'],profile['guidance']['lofting_enabled'])
    summary['burnout_time_s']=sum(float(s.get('fire_delay_s') or 0.)+s['duration_s'] for s in runtime.stages)
    markers=[];clock=0.
    for i,s in enumerate(runtime.stages):
        clock+=float(s.get('fire_delay_s') or 0.)+s['duration_s'];markers.append(dict(time_s=clock,label='燃尽' if i==len(runtime.stages)-1 else '发动机切段',kind='burnout' if i==len(runtime.stages)-1 else 'stage'))
    markers.append(dict(time_s=result['samples'][-1]['time_s'],label='终止事件',kind='termination'))
    model=dict(label=MODE,aerodynamics='shared_effective_single_surface',control='raw_parallel_pid',geometry='one_station',integrator='midpoint',time_step_s=runtime.dt,
        observation_mode=normalized['observation_mode'],observation_provider=runtime.provider.provider_name,runtime_adapter=MODE,runtime_assumptions=runtime.assumptions,
        shared_approximation=defaults['boundary'],coefficient_scaling='fixed reference pressure/mass; area*arm/length^2 moments; profile lift/angle and drag ratios',reference=defaults['reference'])
    model.update(pn_candidate=runtime.config['pn_candidate'],pn_gain=runtime.autopilot['propNavMult'],
                 pid_error_frame=runtime.config['pid_error_frame'],
                 pid_pressure_gain_convention=runtime.config['pid_pressure_gain_convention'])
    if angular_observer is not None:
        model['angular_observer']=copy.deepcopy(runtime.provider.angular_observer.provenance)
    if infrared_seeker is not None:
        model['infrared_seeker']=copy.deepcopy(runtime.provider.provenance)
    if guidance_input is not None:
        model['guidance_input']='external full angular state or Ps/Vs; no second observer filter'
    if seeker_search is not None:
        summary.update(_search_summary(runtime))
        model['seeker_search']=copy.deepcopy(runtime.seeker_search_status)
    if chaff_capture is not None:
        model['chaff_capture']=copy.deepcopy(runtime.provider.radar.chaff_capture)
        summary['chaff_capture']=copy.deepcopy(result['chaff_capture'])
    if track_gate_model is not None:
        model['track_gate_model']=copy.deepcopy(runtime.provider.radar.track_gate_model)
        model['parameter_sources']=copy.deepcopy(runtime.provider.radar.track_gate_summary()['parameter_sources'])
        summary['track_gate_model']=copy.deepcopy(result['track_gate_model'])
    if min_altitude_m is not None:
        model['min_altitude_m']=runtime.provider.inertial.min_altitude_m
    if target_observer is not None:
        model['target_observer']=dict(mode=target_observer,alpha=.8,beta=.05,
            measurement='ideal world position; timestamped external DL; no echo velocity',
            initialization='external estimated velocity prior or zero; identity transfer resets zero')
    if clutter_model is not None:
        model['clutter_model']=clutter_model
        if clutter_model=='look_down_angle':
            model['clutter_min_depression_deg']=math.degrees(runtime.provider.radar.clutter_min_depression_rad)
    if multipath_gain is not None:
        model['multipath_gain']=runtime.provider.radar.multipath_gain
        model['multipath_effect']=[list(n) for n in runtime.provider.radar.multipath_nodes]
    if launcher_support is not None:
        model['launcher_support']='callback'
    if launcher_radar_geometry is not None:model['launcher_radar_geometry']=copy.deepcopy(launcher_radar_geometry)
    if launcher_radar_gimbal_deg is not None or launcher_radar_geometry is not None or launcher_support is not None:
        if launcher_radar_gimbal_deg is not None:model['launcher_radar_gimbal_deg']=float(launcher_radar_gimbal_deg)
        summary['datalink_lost_time_s']=runtime.provider.datalink_lost_time_s
        summary['datalink_lost_reason']=runtime.provider.datalink_lost_reason
    if require_seeker_lock:
        model['require_seeker_lock']=True
        summary['seeker_locked']=bool(runtime.provider.radar_has_tracked_once)
    if target_factory is not None:
        describe=getattr(runtime.target,'describe',None)
        model['target_model']=describe() if describe is not None else type(runtime.target).__name__
        if hasattr(runtime.target,'decoys_at') and getattr(runtime.provider,'supports_decoys',False):
            summary['first_decoy_track_time_s']=runtime.provider.first_decoy_track_time_s
            summary['decoy_track_time_s']=runtime.provider.decoy_track_ticks*runtime.dt
            if cw_on_clear_beam:summary['cw_mode_time_s']=runtime.provider.radar.cw_ticks*runtime.dt
    return dict(schema_version=1,missile=dict(id=profile['missile_id'],name=profile['display_name'],status=profile['model_status'],parameter_source=dict(version=profile['provenance']['source_version'],commit=profile['provenance']['source_commit'])),
        scenario=normalized,summary=summary,markers=markers,samples=result['samples'],model=model)


def _search_summary(runtime):
    search=getattr(getattr(runtime.provider,'radar',None),'search',None)
    return search.summary() if search is not None else dict(first_lock_time_s=None,lock_loss_count=0,
                                                          reacquire_times_s=[],reacquire_delays_s=[])


def _configure_search(runtime,options):
    from .radar_seeker import SeekerSearch,validate_seeker_search
    options=validate_seeker_search(options)
    if options is None:return
    radar=getattr(runtime.provider,'radar',None)
    if radar is None or not radar.active:
        reason=('ideal_truth' if runtime.scenario['observation_mode']=='ideal_truth' else
                'non_radar_observation' if radar is None else 'inactive_radar_observation')
        runtime.seeker_search_status=dict(status='not_applicable',reason=reason,options=options)
        return
    radar.search=SeekerSearch(radar,options)
    runtime.seeker_search_status=dict(status='enabled',options=options,
        scan_half_angle_deg=math.degrees(radar.search.half_rad),dwell_s=radar.search.dwell_s,
        parameter_sources={
            'scan':dict(kind='assumed',grade='D',note='configurable scan pattern, angle and frame time'),
            'antenna_gain':dict(kind='assumed',grade='D',note='per-missile tx/rx Gaussian half-power and linear sidelobe floor'),
            'detection_range':dict(kind='assumed',grade='D',note='per-missile reference RCS/range with fourth-root scaling'),
            'dwell':dict(kind='assumed',grade='D',note='explicit dwell_s; defaults to zero, independent of per-missile lock_timeout_s launch startup'),
            'pointing':dict(kind='assumed',grade='D',note='current available datalink/INS estimate; otherwise body-forward')})


def _configure_runtime(runtime,clutter_model,clutter_min_depression_deg,cw_on_clear_beam,multipath_gain,
                       launcher_radar_gimbal_deg,require_seeker_lock,launcher_support,seeker_search=None,chaff_capture=None,track_gate_model=None,target_observer=None,launcher_radar_geometry=None,angular_observer=None):
    from .observation import validate_target_observer
    validate_target_observer(target_observer)
    from .q_angular_observer import validate_angular_observer
    validate_angular_observer(angular_observer)
    if angular_observer is not None:
        if target_observer is not None:
            raise ValueError('angular_observer and target_observer are mutually exclusive')
        if isinstance(runtime.provider,GuidanceInputProvider):
            raise ValueError('angular_observer cannot alter external guidance_input')
        if not isinstance(runtime.provider,SensorTrackProvider):
            raise ValueError('angular_observer needs active radar sensor_track')
        runtime.provider.enable_q_angular_observer()
    if target_observer is not None:
        if not hasattr(runtime.provider,'enable_unified_target_observer'):
            raise ValueError('target_observer needs the radar sensor_track provider')
        runtime.provider.enable_unified_target_observer()
    _configure_search(runtime,seeker_search)
    if track_gate_model is not None:
        if chaff_capture is not None:raise ValueError("track_gate_model and chaff_capture are mutually exclusive")
        radar=getattr(runtime.provider,"radar",None)
        if radar is None:raise ValueError("track_gate_model needs the radar sensor_track provider")
        radar.enable_track_gate_model(track_gate_model)
    from .radar_seeker import validate_chaff_capture
    options=validate_chaff_capture(chaff_capture)
    if options is not None:
        radar=getattr(runtime.provider,'radar',None)
        if radar is None:raise ValueError('chaff_capture needs the radar sensor_track provider')
        radar.enable_chaff_capture(options)
        if hasattr(runtime.target,'capture_mode'):runtime.target.capture_mode=True
    if cw_on_clear_beam:
        radar=getattr(runtime.provider,'radar',None)
        if radar is None:raise ValueError('cw_on_clear_beam needs the radar sensor_track provider')
        radar.cw_on_clear_beam=True
    if clutter_model is not None:
        radar=getattr(runtime.provider,'radar',None)
        if radar is None:raise ValueError('clutter_model needs the radar sensor_track provider')
        radar.clutter_model=clutter_model
        if clutter_min_depression_deg is not None:radar.clutter_min_depression_rad=math.radians(clutter_min_depression_deg)
    if multipath_gain is not None:
        radar=getattr(runtime.provider,'radar',None)
        if radar is None:raise ValueError('multipath_gain needs the radar sensor_track provider')
        radar.enable_multipath(multipath_gain)
    if launcher_radar_gimbal_deg is not None or launcher_radar_geometry is not None:
        if not hasattr(runtime.provider,'enable_launcher_radar'):raise ValueError('launcher_radar_gimbal_deg needs the radar sensor_track provider')
        runtime.provider.enable_launcher_radar(runtime.creation['position_xyz_m'],runtime.creation['velocity_xyz_mps'],launcher_radar_gimbal_deg,geometry=launcher_radar_geometry)
    if require_seeker_lock:
        if not hasattr(runtime.provider,'radar_has_tracked_once'):raise ValueError('require_seeker_lock needs the radar sensor_track provider')
        runtime.require_seeker_lock=True
    if launcher_support is not None:
        if not hasattr(runtime.provider,'enable_launcher_support'):
            raise ValueError('launcher_support needs the radar sensor_track provider')
        runtime.provider.enable_launcher_support(launcher_support)


def create_surface_missile(profile, *, launch_position_m, launch_velocity_mps,
                           launch_pitch_deg, launch_heading_deg, target,
                           observation_mode='sensor_track', loft=True,
                           clutter_model=None, clutter_min_depression_deg=None, cw_on_clear_beam=False,
                           multipath_gain=None, launcher_radar_gimbal_deg=None, launcher_support=None,
                           require_seeker_lock=False, end_time_s=None, early_miss_s=None,
                           datalink_enabled=True, datalink_disconnect_time_s=None,
                           inertial_drift_direction=(0.,0.,1.),seeker_search=None,min_altitude_m=None,chaff_capture=None,track_gate_model=None,target_observer=None,
                           launch_quaternion_xyzw=None,launcher_radar_geometry=None,guidance_input=None,angular_observer=None,infrared_seeker=None):
    """Create and begin a single-surface runtime from an explicit launch state.

    ``launch_position_m`` and ``launch_velocity_mps`` are finite xyz triples in
    the caller's shared x/y-up/z coordinate system. Velocity is retained exactly,
    independently of pitch/heading (degrees; zero heading is +x, positive is +z).
    ``launch_quaternion_xyzw`` optionally overrides body attitude alone: a
    normalized body-to-world quaternion, scalar-last, with forward/up/right
    body axes. Pitch/heading still define the default attitude when omitted.
    Time starts at launch zero; the caller translates world time for ``target``
    and ``launcher_support``. Advance aircraft first and retain/interpolate history
    for state_at at tick endpoints and within-tick terminal times; see step().
    ``launcher_support(t, truth)`` returns '' for legacy support, a loss reason,
    or (reason, native_y_up_position, native_y_up_velocity) for estimate support;
    first loss is permanent. It cannot be combined with the linear-launcher gimbal.
    ``angular_observer='q_capture'`` enables the empirical Q-at-acquisition
    complete-rate observer using received radar directions; no replay input is
    needed. Q is applied once with the previous body posture. Subsequent echo
    innovations use the shared 11.23 /s gain; missing echoes only predict.
    This approximation is active-radar sensor_track only, mutually exclusive
    with target_observer/guidance_input; None retains the existing observer.
    ``infrared_seeker=InfraredSeekerConfig(...)`` opts optical sensor_track
    into the source-informed IR/flare provider. Range and velocity use explicit
    truth-assisted closure, not IR measurements. The config game factory reads
    aircraft/global heat and flare/band inputs from the frozen datamine.
    None preserves the existing provider. This cannot combine with radar options
    or guidance_input. See infrared_seeker.py for evidence and limitations.
    Sensor options match public_api.simulate. ``end_time_s`` is capped by profile
    lifetime; omitted uses that lifetime. Early miss is disabled unless supplied.
    Result is the raw SurfaceRuntime sample/event result. Opt-in seeker_search
    adds scan diagnostics and a lock-metrics summary to result(); None retains
    the existing contract. Active radar only; target.rcs_m2 defaults to 1.
    ``min_altitude_m=50.0`` optionally floors only INS estimates at 50 m in y-up
    coordinates and zeros estimated vertical velocity when clipped. None is off;
    it requires sensor_track and leaves direct radar/datalink observations intact.
    ``guidance_input(t, own_position, own_velocity) -> TrackSolution`` replaces
    the observation provider. Its optional full_rate_rad_s uses runtime world
    axes and n_dot = n x rate, relative to the supplied own state. Use
    angular_guidance.AngularObservation.track_for to transfer recorded inputs,
    or pass RecordedSeekerHistory directly. Target truth is still used for
    events/telemetry; it is never passed into this guidance source.
    """
    from .public_api import validate_scenario,_case_from_scenario,_validate_surface_options
    from .profile_adapter import unsupported_model_types
    _validate_surface_options(early_miss_s,clutter_model,clutter_min_depression_deg,multipath_gain,
                              launcher_radar_gimbal_deg,launcher_support,launcher_radar_geometry)
    if unsupported_model_types(profile):raise ValueError('Unsupported guidance/physics family')
    def vector(value,label):
        if isinstance(value,(str,bytes)):raise ValueError(label+' must be a finite xyz triple')
        try: values=tuple(float(v) for v in value)
        except (TypeError,ValueError) as exc:raise ValueError(label+' must be a finite xyz triple') from exc
        if len(values)!=3 or not all(math.isfinite(v) for v in values):
            raise ValueError(label+' must be a finite xyz triple')
        return values
    position=vector(launch_position_m,'launch_position_m')
    velocity=vector(launch_velocity_mps,'launch_velocity_mps')
    if not callable(getattr(target,'state_at',None)):raise ValueError('target must implement state_at(time_s)')
    if end_time_s is not None and (isinstance(end_time_s,bool) or not isinstance(end_time_s,(int,float))
                                  or not math.isfinite(end_time_s) or end_time_s<0):
        raise ValueError('end_time_s must be a finite nonnegative number')
    scenario=validate_scenario(dict(launch_speed_kmh=0.,launch_altitude_m=0.,
        launch_pitch_deg=launch_pitch_deg,launch_heading_deg=launch_heading_deg,
        target_speed_kmh=0.,target_altitude_m=0.,initial_distance_m=1.,target_azimuth_deg=0.,
        target_heading_deg=0.,target_vertical_heading_deg=0.,target_constant_turn_g=0.,
        observation_mode=observation_mode,loft_enabled=loft,datalink_enabled=datalink_enabled,
        datalink_disconnect_time_s=datalink_disconnect_time_s,inertial_drift_direction=inertial_drift_direction))
    scenario['launch_altitude_m']=position[1]
    scenario['launch_speed_kmh']=norm(velocity)*3.6
    defaults=json.loads((Path(__file__).resolve().parents[2]/'config/effective_surface_defaults.json').read_text())
    runtime=SurfaceRuntime(profile,scenario,_case_from_scenario(scenario),defaults,
                           launch_position_m=position,launch_velocity_mps=velocity,target=target,
                           min_altitude_m=min_altitude_m,launch_quaternion_xyzw=launch_quaternion_xyzw,
                           guidance_input=guidance_input,infrared_seeker=infrared_seeker)
    _configure_runtime(runtime,clutter_model,clutter_min_depression_deg,cw_on_clear_beam,multipath_gain,
                       launcher_radar_gimbal_deg,require_seeker_lock,launcher_support,seeker_search,chaff_capture,track_gate_model,target_observer,launcher_radar_geometry,angular_observer)
    lifetime=profile['performance']['lifetime_s']
    return runtime.begin(lifetime if end_time_s is None else min(lifetime,end_time_s),early_miss_s)
