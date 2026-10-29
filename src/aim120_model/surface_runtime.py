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

from .effective_surface_core import EffectiveSurfaceGame, G, add, scale, sub, norm, axes, body, dot, table
from .aerodynamics import quaternion_from_pitch_yaw
from .control import scheduled_pid_values
from .dynamics import SimState
from .observation import IdealTruthTrackProvider, KinematicTrackProvider, SensorTrackProvider
from .target import TargetModel, TargetState
from .events import event_candidates
from .effective_surface_core import normalized

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


class SurfaceRuntime(EffectiveSurfaceGame):
    def __init__(self, profile, scenario, case, defaults):
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
        target=TargetModel(case['initial_conditions'],G).initial_state
        creation=dict(position_xyz_m=[0.,scenario['launch_altitude_m'],0.],
            velocity_xyz_mps=scale(axes(q)[0],scenario['launch_speed_kmh']/3.6),
            mass_source_kg=geometry['initial_mass_kg'],quaternion_xyzw_raw=q,omega_export_raw=[0.,0.,0.],
            target_position_xyz_m=target.position,target_velocity_xyz_mps=target.velocity,
            scenario_exact={'dt':1/48,'loft_state':'config' if autopilot['loftEnabled'] else 'off'})
        super().__init__(raw,cfg,creation)
        self.v0=ref['speed_mps']
        self.qm0=.5*self.atmosphere.sample(ref['altitude_m']).density_kg_m3*self.v0**2/ref['mass_kg']
        self.stages=profile['propulsion']['stages']
        self.track=None
        self.target=ScenarioTarget(case['initial_conditions'])
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
                    inertial_drift_direction=tuple(scenario['inertial_drift_direction']))
            else:
                self.provider=KinematicTrackProvider(lock_range_m=guidance.get('lock_range_m'),
                    maximum_angular_rate_deg_s=guidance.get('maximum_angular_rate_deg_s'),
                    seeker_type=guidance.get('seeker_type','unknown'))

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

    def simulate(self,end_time):
        state=tuple(self.creation['position_xyz_m'])+tuple(self.creation['velocity_xyz_mps'])+tuple(self.creation['quaternion_xyzw_raw'])+(0.,0.,0.)
        rows=[];event='lifetime';time=0.;initial_position=state[:3];terminal_event=None
        event_config=copy.deepcopy(self.profile)
        event_config['performance']['lifetime_s']=end_time
        performance=self.profile['performance'];fuse=self.profile['guidance']['proximity_radius_m']
        while True:
            truth=self.target.state_at(time)
            self.track=self.provider.update(time,0. if not rows else self.dt,self.missile_state(state),truth)
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
            if terminal_event is not None:event=terminal_event;break
            if distance<=fuse:event='fuse';break
            if state[1]<=0:event='ground';break
            if norm(sub(state[:3],initial_position))>=performance['maximum_distance_m']:event='max_distance';break
            if time>=end_time-1e-10:break
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
                for j in range(self.integration_substeps):state=self.step(left+j*h,state,self.control,h)
            next_time=time+step
            candidates=event_candidates(self.missile_state(old_state),self.missile_state(state),truth,
                self.target.state_at(next_time),time,next_time,event_config,initial_position)
            if candidates:
                first=candidates[0]
                fraction=first.fraction
                state=tuple(a+fraction*(b-a) for a,b in zip(old_state,state))
                state=state[:6]+normalized(state[6:10])+state[10:]
                next_time=time+fraction*step
                terminal_event=first.event_type
            time=next_time
            _,d=self.plant(time,state,self.control);self.previous_acceleration=d['acceleration_xyz_mps2']
        return dict(samples=rows,event_type=event,observation_mode=self.scenario['observation_mode'],observation_provider=self.provider.provider_name)


def simulate_surface(profile,scenario):
    from .public_api import validate_scenario,_case_from_scenario,_summarize
    from .profile_adapter import unsupported_model_types
    if unsupported_model_types(profile):raise ValueError('Unsupported guidance/physics family')
    normalized=validate_scenario(scenario)
    defaults_path=Path(__file__).resolve().parents[2]/'config/effective_surface_defaults.json'
    if not defaults_path.is_file():
        defaults_path=Path(__file__).with_name('reconstructed')/'data/legacy_effective_surface_defaults.json'
    defaults=json.loads(defaults_path.read_text())
    runtime=SurfaceRuntime(profile,normalized,_case_from_scenario(normalized),defaults)
    lifetime=profile['performance']['lifetime_s'];requested=normalized['max_simulation_time_s'];end=min(lifetime,requested) if requested is not None else lifetime
    result=runtime.simulate(end)
    config=copy.deepcopy(profile)
    config['guidance']['lofting_enabled']=runtime.autopilot['loftEnabled']
    summary=_summarize(result,config,requested is not None and requested<lifetime,normalized['loft_enabled'],profile['guidance']['lofting_enabled'])
    summary['burnout_time_s']=sum(float(s.get('fire_delay_s') or 0.)+s['duration_s'] for s in runtime.stages)
    markers=[];clock=0.
    for i,s in enumerate(runtime.stages):
        clock+=float(s.get('fire_delay_s') or 0.)+s['duration_s'];markers.append(dict(time_s=clock,label='燃尽' if i==len(runtime.stages)-1 else '发动机切段',kind='burnout' if i==len(runtime.stages)-1 else 'stage'))
    markers.append(dict(time_s=result['samples'][-1]['time_s'],label='终止事件',kind='termination'))
    return dict(schema_version=1,missile=dict(id=profile['missile_id'],name=profile['display_name'],status=profile['model_status'],parameter_source=dict(version=profile['provenance']['source_version'],commit=profile['provenance']['source_commit'])),
        scenario=normalized,summary=summary,markers=markers,samples=result['samples'],model=dict(label=MODE,aerodynamics='shared_effective_single_surface',control='raw_parallel_pid',geometry='one_station',integrator='midpoint',time_step_s=runtime.dt,
        observation_mode=normalized['observation_mode'],observation_provider=runtime.provider.provider_name,runtime_adapter=MODE,runtime_assumptions=runtime.assumptions,
        shared_approximation=defaults['boundary'],coefficient_scaling='fixed reference pressure/mass; area*arm/length^2 moments; profile lift/angle and drag ratios',reference=defaults['reference']))
