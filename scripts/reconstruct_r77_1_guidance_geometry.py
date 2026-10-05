"""Feed measured R-77-1 motion into the unchanged PN equation.

Reconstruct horizontal motion from V and displayed total normal load, under
the existing level-flight, positive-turn assumption. This is a kinematic
reconstruction, not a free plant run or a recovery of game PID/fin states.
The first measurement is at 0.20 s; its pose is seeded explicitly. Target
course is user-confirmed at 180 degrees; the old 210-degree course is retained
only to show the consequence of the previous scenario error.
"""
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_r77_1_capture_strategy import (
    ROOT, inputs, CASES, protected_hashes, load_observations,
    VERSION, COMMIT, digest, write_table,
)

G0 = 9.80665
SPLICE_TIME = 2.10


def cumulative_integral(values, time):
    return np.r_[0., np.cumsum(.5*(values[:-1]+values[1:])*np.diff(time))]


def reconstruct(obs, seed, step=.002):
    time = np.linspace(obs['time'][0], obs['time'][-1],
                       round((obs['time'][-1]-obs['time'][0])/step)+1)
    speed = np.interp(time, obs['time'], obs['speed'])
    load = np.interp(time, obs['time'], obs['load'])
    alpha = np.interp(time, obs['time'], obs['alpha'])
    assert np.all(load > 1.)
    gh = np.sqrt(load**2-1.)
    heading = seed['heading_rad']+cumulative_integral(G0*gh/speed, time)
    x = seed['x_m']+cumulative_integral(speed*np.cos(heading), time)
    z = seed['z_m']+cumulative_integral(speed*np.sin(heading), time)
    distance = seed['flown_m']+cumulative_integral(speed, time)
    # AoA is carried as an observation, not used to produce normal force.
    return [dict(time_s=float(t), speed_kmh=float(v*3.6),
                 heading_rad=float(h), x_m=float(px), z_m=float(pz),
                 flown_m=float(s), observed_total_load_g=float(g),
                 reconstructed_horizontal_load_g=float(ah), alpha_deg=float(math.degrees(a)))
            for t,v,h,px,pz,s,g,ah,a in zip(time,speed,heading,x,z,distance,load,gh,alpha)]


def guidance(row, target_v, scenario, navigation_gain):
    azimuth = math.radians(scenario.get('target_azimuth_deg', 30.))
    initial_range = scenario.get('initial_distance_m', 8000.)
    t = row['time_s']
    tx = initial_range*math.cos(azimuth)+target_v[0]*t
    tz = initial_range*math.sin(azimuth)+target_v[1]*t
    rx, rz = tx-row['x_m'], tz-row['z_m']
    v, h = row['speed_kmh']/3.6, row['heading_rad']
    rvx, rvz = target_v[0]-v*math.cos(h), target_v[1]-v*math.sin(h)
    distance = math.hypot(rx, rz)
    closing = -(rx*rvx+rz*rvz)/distance
    los_rate = (rx*rvz-rz*rvx)/distance**2
    return dict(target_x_m=tx, target_z_m=tz, relative_x_m=rx, relative_z_m=rz,
                target_range_m=distance, closing_speed_mps=closing,
                los_angle_deg=math.degrees(math.atan2(rz,rx)),
                los_rate_deg_s=math.degrees(los_rate),
                pn_horizontal_request_g=navigation_gain*max(closing,0.)*los_rate/G0)


def at_time(rows, time):
    row = min(rows, key=lambda r:abs(r['time_s']-time))
    assert abs(row['time_s']-time) < 1e-8
    return row


def plot(reconstructed, baseline, alternatives, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    ax = axes[0,0]
    for rows,color,style,label in (
        (baseline,'#ce7826','--','Previous simulated motion'),
        (reconstructed,'#168777','-','Motion reconstructed from measurements')):
        early = [r for r in rows if r['time_s']<=SPLICE_TIME+1e-9]
        ax.plot([r['x_m'] for r in early],[r['z_m'] for r in early],
                color=color,ls=style,lw=2,label=label)
        r=at_time(rows,SPLICE_TIME)
        ax.scatter(r['x_m'],r['z_m'],color=color,s=28)
    ax.set(xlabel='Missile x [m]',ylabel='Missile horizontal z [m]',
           title='Reconstructed flight path through 2.10 s')
    ax.set_aspect('equal',adjustable='datalim')
    ax.legend(frameon=False,fontsize=8.5,loc='upper left')
    ax=axes[0,1]
    for rows,color,style,label in (
        (baseline,'#ce7826','--','Previous simulation'),
        (reconstructed,'#168777','-','Measured-motion reconstruction')):
        ax.plot([r['time_s'] for r in rows],np.degrees([r['heading_rad'] for r in rows]),
                color=color,ls=style,lw=2,label=label)
    ax.set(xlim=(.2,3.5),xlabel='Missile flight time [s]',ylabel='Velocity heading [deg]',
           title='Turning inferred from measured load and speed')
    ax=axes[1,0]
    ax.plot(obs['time'],np.sqrt(obs['load']**2-1.),'o-',color='#252525',lw=1,ms=4,
            label='Measured actual horizontal load (not a command)')
    ax.plot([r['time_s'] for r in baseline],[r['raw_pn_request_g'] for r in baseline],
            color='#ce7826',ls='--',lw=1.8,label='PN on previous simulated geometry')
    ax.plot([r['time_s'] for r in reconstructed],
            [r['pn_horizontal_request_g'] for r in reconstructed],color='#168777',lw=2,
            label='PN on reconstructed motion; confirmed 180° course')
    ax.set(xlim=(.2,3.5),xlabel='Missile flight time [s]',ylabel='Horizontal load / request [g]',
           title='Same PN coefficient, measured missile motion')
    ax.legend(frameon=False,fontsize=8.1)
    ax=axes[1,1]
    for name,color,label in (
        ('along_initial_los','#168777','Previous incorrect course: 210°'),
        ('opposite_launch_heading','#8061a7','User-confirmed parallel opposite course: 180°')):
        rows=alternatives[name]
        ax.plot([r['time_s'] for r in rows],[r['pn_horizontal_request_g'] for r in rows],
                color=color,lw=2,label=label)
    ax.set(xlim=(.2,3.5),xlabel='Missile flight time [s]',ylabel='PN horizontal request [g]',
           title='Effect of correcting the target course')
    ax.legend(frameon=False,fontsize=8.3)
    for ax in axes.flat:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    for ax in (axes[0,1],*axes[1,:]):
        ax.axvline(SPLICE_TIME,color='#777',ls=':',lw=1)
    fig.suptitle('R-77-1 | measured-motion reconstruction into unchanged PN',fontsize=16,y=.98)
    fig.text(.08,.025,
        '2.57.1.107; N = 4. No plant, PID, aerodynamic or guidance coefficient is fitted.\n'
        'Level flight and positive horizontal turn assumed. 0–0.20 s pose uses the old simulation; later motion uses measured V and G.\n'
        'Target: 8 km, 30° azimuth, 1200 km/h, user-confirmed absolute heading 180°. Reconstructed missile position is conditional.\n'
        '2.10–3.50 s curves continue the measurement reconstruction; they are not a free plant continuation.',
        fontsize=9,color='#444')
    fig.subplots_adjust(left=.08,right=.98,top=.89,bottom=.165,hspace=.34,wspace=.25)
    fig.savefig(prefix.with_suffix('.png'),dpi=165,facecolor='white')
    plt.close(fig)


def main():
    protected=protected_hashes()
    paths=[ROOT/'outputs/r77_1_capture_fit.json',ROOT/'outputs/r77_1_capture_fit_trajectory.tsv',
           ROOT/'scripts/run_r77_1_focus_moment.py',
           ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx']
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report=json.loads(paths[0].read_text())
    config,p,raw_limit=inputs()
    p=replace(p,fin_force_area_per_rad=report['fixed_parameters']['fin_force_area_per_rad'],guidance_delay_s=0.)
    assert asdict(p)==report['fixed_parameters'] and digest(config)==report['configuration_sha256']
    assert (VERSION,COMMIT)==(report['source_version'],report['source_commit'])
    raw=json.loads(paths[3].read_text())['rocket']['guidance']['guidanceAutopilot']
    assert raw['propNavMult']==p.navigation_gain==4.
    _,obs=load_observations()
    with paths[1].open() as f:
        baseline=[{k:(v=='True' if v in ('True','False') else float(v))
                   for k,v in row.items() if k!='case'}
                  for row in csv.DictReader(f,delimiter='\t') if row['case']=='early_capture']
    initial=at_time(baseline,.2)
    before=[r for r in baseline if r['time_s']<=.2]
    seed=dict(heading_rad=initial['heading_rad'],x_m=initial['x_m'],z_m=initial['z_m'],
              flown_m=float(cumulative_integral(np.array([r['speed_kmh']/3.6 for r in before]),
                                                np.array([r['time_s'] for r in before]))[-1]))
    scenario=dict(CASES[0][3],initial_distance_m=8000.,target_azimuth_deg=30.)
    target_speed=scenario['target_speed_kmh']/3.6;azimuth=math.radians(scenario['target_azimuth_deg'])
    target_velocities=dict(along_initial_los=(-target_speed*math.cos(azimuth),-target_speed*math.sin(azimuth)),
                           opposite_launch_heading=(-target_speed,0.))
    motion=reconstruct(obs,seed)
    alternatives={name:[dict(r,**guidance(r,tv,scenario,p.navigation_gain)) for r in motion]
                  for name,tv in target_velocities.items()}
    reconstructed=alternatives['opposite_launch_heading']
    anchor={name:at_time(rows,SPLICE_TIME) for name,rows in alternatives.items()}
    comparison=[]
    for i,t in enumerate(obs['time']):
        b=at_time(baseline,t);r=at_time(reconstructed,t)
        comparison.append(dict(time_s=float(t),observed_total_load_g=float(obs['load'][i]),
            observed_horizontal_load_g=float(math.sqrt(obs['load'][i]**2-1.)),
            previous_pn_request_g=b['raw_pn_request_g'],
            reconstructed_pn_request_g=r['pn_horizontal_request_g'],
            old_target_course_on_measured_motion_pn_request_g=at_time(alternatives['along_initial_los'],t)['pn_horizontal_request_g'],
            previous_heading_deg=math.degrees(b['heading_rad']),reconstructed_heading_deg=math.degrees(r['heading_rad']),
            previous_x_m=b['x_m'],previous_z_m=b['z_m'],reconstructed_x_m=r['x_m'],reconstructed_z_m=r['z_m']))
    # The unobserved launch interval gets two explicit, unfitted alternatives.
    v0=scenario['launch_speed_kmh']/3.6;v1=obs['speed'][0];t0=obs['time'][0]
    ti=np.linspace(0.,t0,201);vi=np.linspace(v0,v1,201)
    sensitivity={}
    for name,gh in [('straight_unobserved_interval',0.),('first_load_held_backward',math.sqrt(obs['load'][0]**2-1.))]:
        hi=cumulative_integral(G0*gh/vi,ti)
        alt_seed=dict(heading_rad=float(hi[-1]),x_m=float(cumulative_integral(vi*np.cos(hi),ti)[-1]),
                      z_m=float(cumulative_integral(vi*np.sin(hi),ti)[-1]),flown_m=float(cumulative_integral(vi,ti)[-1]))
        rr=at_time(reconstruct(obs,alt_seed),SPLICE_TIME)
        sensitivity[name]=dict(seed=alt_seed,**guidance(rr,target_velocities['opposite_launch_heading'],scenario,p.navigation_gain))

    # One verification pass: native values, geometric PN, quadrature and source receipts.
    for i,t in enumerate(obs['time']):
        row=at_time(motion,t)
        assert abs(row['speed_kmh']-obs['speed'][i]*3.6)<1e-8
        assert abs(row['observed_total_load_g']-obs['load'][i])<1e-10
        assert abs(row['alpha_deg']-math.degrees(obs['alpha'][i]))<1e-9
    residual=max(abs(guidance(row,target_velocities['along_initial_los'],scenario,p.navigation_gain)['pn_horizontal_request_g']
                     -row['raw_pn_request_g']) for row in baseline)
    assert residual<1e-10,residual
    fine=reconstruct(obs,seed,.001)
    errors={key:max(abs(at_time(fine,t)[key]-at_time(motion,t)[key]) for t in obs['time'])
            for key in ('heading_rad','x_m','z_m','flown_m')}
    assert max(errors.values())<.005,errors
    fine_pn=guidance(at_time(fine,SPLICE_TIME),target_velocities['opposite_launch_heading'],scenario,p.navigation_gain)
    assert abs(fine_pn['pn_horizontal_request_g']-anchor['opposite_launch_heading']['pn_horizontal_request_g'])<1e-4
    with (ROOT/'data/replays/r77_1_level_20260824.tsv').open() as f:
        native=[r for r in csv.DictReader(f,delimiter='\t') if float(r['missile_flight_time_s'])<=3.5]
    flown_residuals=[at_time(motion,float(r['missile_flight_time_s']))['flown_m']-1000*float(r['displayed_flown_distance_km']) for r in native]
    assert protected_hashes()==protected
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in hashes.items())
    prefix=ROOT/'outputs/r77_1_reconstructed_geometry_confirmed_heading'
    result=dict(source_version=VERSION,source_commit=COMMIT,scenario=scenario,
        navigation_gain=p.navigation_gain,splice_time_s=SPLICE_TIME,reconstruction_window_s=[.2,3.5],
        reconstruction='V(t), G(t) linearly interpolated; chi_dot=g*sqrt(G^2-1)/V; position integrated from velocity',
        level_flight_assumed=True,positive_horizontal_turn_assumed=True,
        measured_quantities=['speed','total normal load','AoA'],
        seed_at_first_measurement=seed,seed_source='previous model pose at 0.20 s; not a measurement',
        target_velocity_alternatives_mps=target_velocities,target_course_verified_from_telemetry=False,
        target_course_source='User confirmed parallel opposite courses: launch 0 deg, target 180 deg',
        selected_target_course='opposite_launch_heading',
        reconstructed_position_is_conditional=True,pid_and_fin_state_reconstructed=False,
        free_plant_continuation_performed=False,parameters_fitted=[],
        splice_states=anchor,previous_simulation_at_splice=at_time(baseline,SPLICE_TIME),
        native_comparison=comparison,unobserved_launch_interval_sensitivity=sensitivity,
        old_pn_reproduction_max_residual_g=residual,half_step_max_difference=errors,
        flown_distance_vs_rounded_display_max_residual_m=max(map(abs,flown_residuals)),
        source_hashes=hashes,protected_inputs_unchanged=True)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),comparison)
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
                [dict(target_course=name,**row) for name,rows in alternatives.items() for row in rows])
    plot(reconstructed,baseline,alternatives,obs,prefix)
    print(json.dumps(dict(anchor=anchor,initial_interval_sensitivity=sensitivity,
                          half_step=errors,flown_distance_max_residual_m=max(map(abs,flown_residuals)))),flush=True)
    print(prefix.with_suffix('.png'))


if __name__=='__main__':
    main()
