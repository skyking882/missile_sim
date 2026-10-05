"""Measured-state release at 1.50 s into the unchanged PN/PID/focal plant.

Only measurements at or before release initialize the state. Later data are
comparison values. Unobserved body rates and PID memory are conditional model
initializations, not recovered game telemetry or new fitted coefficients.
"""
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math
import textwrap

import numpy as np

from check_r77_1_capture_strategy import (
    ROOT, FocusPlant, alpha_servo, inputs, CASES, protected_hashes,
    load_observations, VERSION, COMMIT, FIELDS, digest, build_runner, enrich, write_table,
)
from check_r77_1_target_course import read_trace
from reconstruct_r77_1_guidance_geometry import reconstruct, guidance, at_time

RELEASE = 1.5
END = 3.5
CONTROL_DT = .02
G0 = alpha_servo.G0


def frozen_force_function(config, p, scenario):
    """Evaluate the exact candidate force block for release-state bookkeeping."""
    source = inspect.getsource(alpha_servo.run_alpha_servo)
    block = source[source.index('    def forces(t, s):'):source.index('    def record(t, s):')]
    namespace = dict(vars(alpha_servo), p=p, config=config, lift_mach_table=(),
        atmo=alpha_servo.StandardAtmosphere().sample(scenario['launch_altitude_m']),
        prop=alpha_servo.PiecewisePropulsion.from_config(config),
        gravity=float(config.get('atmosphere',{}).get('gravity_mps2',G0)))
    exec(compile(textwrap.dedent(block),'<unchanged candidate forces>','exec'),namespace)
    return namespace['forces'], hashlib.sha256(block.encode()).hexdigest()


def measured_state(row, p, plant, prop):
    t, v = row['time_s'], row['speed_kmh']/3.6
    alpha = math.radians(row['alpha_deg'])
    ps = prop.sample(t)
    pressure = .5*plant.atmo.density_kg_m3*v*v
    gain = pressure*p.lift_area_per_rad+ps.thrust_n*math.sin(alpha)/alpha
    av = ps.mass_kg*G0/gain
    assert alpha > av
    ah = math.sqrt(alpha*alpha-av*av)
    return (v,row['heading_rad'],row['x_m'],row['z_m'],row['heading_rad']+ah,0.)


def initialize(obs, seed, config, p, scenario, plant, forces, degree=2):
    # Causal split: future observations never enter reconstruction or derivatives.
    history = {k:np.asarray(obs[k])[obs['time'] <= RELEASE+1e-10].copy()
               for k in ('time','speed','alpha','load')}
    assert history['time'][-1] == RELEASE
    motion = reconstruct(history,seed)
    anchor = at_time(motion,RELEASE)
    prop = alpha_servo.PiecewisePropulsion.from_config(config)
    state = measured_state(anchor,p,plant,prop)
    count = degree+1
    tt = history['time'][-count:]
    angles = [measured_state(at_time(motion,float(t)),p,plant,prop)[4]
              -at_time(motion,float(t))['heading_rad'] for t in tt]
    ca = np.polynomial.polynomial.polyfit(tt-RELEASE,angles,degree)
    cv = np.polynomial.polynomial.polyfit(tt-RELEASE,history['speed'][-count:],degree)
    cn = np.polynomial.polynomial.polyfit(tt-RELEASE,history['load'][-count:],degree)
    n, v = history['load'][-1], history['speed'][-1]
    nh = math.sqrt(n*n-1.)
    heading_rate = G0*nh/v
    heading_accel = G0*((n/nh)*cn[1]/v-nh*cv[1]/v**2)
    alpha_rate = ca[1]
    alpha_accel = 2*ca[2] if degree == 2 else 0.
    omega = float(heading_rate+alpha_rate)
    omega_dot = float(heading_accel+alpha_accel)
    state = (*state[:5],omega)
    target_v = alpha_servo.planar_target_motion(scenario)[1]
    pn = guidance(anchor,target_v,scenario,p.navigation_gain)['pn_horizontal_request_g']
    ps, pressure, av, alpha, gain, drag, accel = forces(RELEASE,state)
    assert abs(math.degrees(alpha)-anchor['alpha_deg']) < 1e-9
    error = pn-accel/G0
    prior = at_time(motion,RELEASE-CONTROL_DT)
    prior_state = measured_state(prior,p,plant,prop)
    previous_error = (guidance(prior,target_v,scenario,p.navigation_gain)['pn_horizontal_request_g']
                      -forces(prior['time_s'],prior_state)[-1]/G0)
    passive = plant.sample(v,state[4]-state[1],omega,pressure,ps.mass_kg,0.,p,alpha)
    moment_per_delta = p.stability_distance_m*pressure*p.fin_force_area_per_rad
    delta = (passive.inertia_kg_m2*omega_dot-passive.moment_nm)/moment_per_delta
    assert abs(delta) < p.horizontal_fin_limit_rad
    assert abs(pressure*p.fin_force_area_per_rad*delta) < ps.mass_kg*G0*p.fin_lateral_acceleration_limit_g
    proportional = p.acceleration_pid_kp*error
    derivative = p.acceleration_pid_kd*(error-previous_error)/CONTROL_DT
    integral_after_tick = delta/p.horizontal_fin_limit_rad-proportional-derivative
    integral_before_tick = integral_after_tick-p.acceleration_pid_ki*error*CONTROL_DT
    assert abs(integral_after_tick) < p.acceleration_pid_integral_limit
    restart = dict(time_s=RELEASE,state=state,pid_integral=float(integral_before_tick),
                   previous_error=float(previous_error))
    details = dict(release_time_s=RELEASE,state=state,derivative_degree=degree,
        derivative_observation_times_s=list(map(float,tt)),
        observed_total_alpha_deg=anchor['alpha_deg'],observed_total_load_g=float(n),
        predicted_total_load_g=math.hypot(accel/G0,1.),pn_request_g=pn,
        inferred_alpha_rate_deg_s=math.degrees(alpha_rate),
        inferred_body_rate_deg_s=math.degrees(omega),
        inferred_body_angular_acceleration_rad_s2=omega_dot,
        inferred_fin_angle_deg=math.degrees(delta),
        pid_integral_before_release_tick=integral_before_tick,
        pid_integral_after_release_tick=integral_after_tick,
        previous_pid_error_g=previous_error,release_pid_error_g=error,
        inferred_control_moment_nm=moment_per_delta*delta,
        target_position_m=[guidance(anchor,target_v,scenario,p.navigation_gain)[k]
                           for k in ('target_x_m','target_z_m')])
    return restart,details,motion


def integral_bound(motion, early_trace, p, scenario, plant, config, forces, table):
    """Necessary integral bound, conditional on the same pre-release motion.

    At 1.48 s use PN to match the derivative memory used at release. Earlier
    requests take their largest allowed values. This is not a reconstructed
    game command history or proof that the moment equation can realize it.
    """
    prop = alpha_servo.PiecewisePropulsion.from_config(config)
    cap = math.sqrt(p.command_limit_g**2-1.)
    integral = loose_bound = measured_motion_bound = 0.
    previous_error = 0.
    for tick in range(round(RELEASE/CONTROL_DT)):
        t = round(tick*CONTROL_DT,10)
        gain = float(np.interp(t,*np.array(table).T))
        maximum = gain*cap
        if t < .2:
            actual = at_time(early_trace,t)['horizontal_specific_force_g']
        else:
            row = at_time(motion,t)
            actual = forces(t,measured_state(row,p,plant,prop))[-1]/G0
        assert actual >= 0.
        request = maximum
        if t >= RELEASE-CONTROL_DT-1e-10:
            request = guidance(row,alpha_servo.planar_target_motion(scenario)[1],
                               scenario,p.navigation_gain)['pn_horizontal_request_g']
        error = request-actual
        proposed = integral+p.acceleration_pid_ki*error*CONTROL_DT
        output = (p.acceleration_pid_kp*error+proposed+
                  p.acceleration_pid_kd*(error-previous_error)/CONTROL_DT)
        assert abs(output) < 1., (t,output)
        integral, previous_error = proposed,error
        measured_motion_bound += p.acceleration_pid_ki*(maximum-actual)*CONTROL_DT
        loose_bound += p.acceleration_pid_ki*maximum*CONTROL_DT
    return dict(integral_before_release_with_PN_previous_error=integral,
        integral_before_release_upper_bound_on_measured_motion=measured_motion_bound,
        integral_before_release_upper_bound_even_at_zero_actual_load=loose_bound,
        last_error_g=previous_error,initial_integral_at_launch=0.,
        initial_motion_before_0p2_s='fixed model pre-guidance trajectory',
        hypothetical_max_request_history_is_not_a_plant_trajectory=True)


def plot(results, baseline, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('alpha_deg','AoA [deg]',np.degrees(obs['alpha'])),
              ('wind_normal_specific_force_g','Total normal load [g]',obs['load']),
              ('speed_kmh','Speed [km/h]',obs['speed']*3.6),
              ('commanded_horizontal_acceleration_g','Requested horizontal load [g]',None)]
    fig,axes = plt.subplots(2,2,figsize=(13,8.8))
    for ax,(field,label,measured) in zip(axes.flat,panels):
        ax.axvspan(0,RELEASE,color='#e9edf0',alpha=.8,zorder=0)
        ax.axvline(RELEASE,color='#747b83',ls=':',lw=1.2)
        if measured is not None:
            ax.plot(obs['time'],measured,'o-',color='#252525',lw=1,ms=4,label='2.57 measurements',zorder=5)
        ax.plot([r['time_s'] for r in baseline],[r[field] for r in baseline],
                color='#c77c27',ls='--',lw=1.6,label='Previous orange: timed manual capture')
        main = results['integral_upper_bound'].samples
        matched = results['matched_initial_control'].samples
        alt = results['two_point_rate'].samples
        ax.fill_between([r['time_s'] for r in matched],
            np.minimum([r[field] for r in matched],[r[field] for r in alt]),
            np.maximum([r[field] for r in matched],[r[field] for r in alt]),
            color='#8061a7',alpha=.2)
        ax.plot([r['time_s'] for r in matched],[r[field] for r in matched],
                color='#8061a7',ls=':',lw=2,
                label='Moment-matched I = 0.319 (exceeds controller bound)')
        ax.plot([r['time_s'] for r in main],[r[field] for r in main],
                color='#147f9b',lw=2,label='Release with I upper bound = 0.220; PN + PID')
        ax.set(xlim=(.2,3.5),xlabel='Missile flight time [s]',ylabel=label)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False,fontsize=9.5)
    fig.suptitle('R-77-1 | release from measured motion at 1.50 s | manual capture disabled',fontsize=15,y=.988)
    fig.text(.08,.025,
        'Frozen 2.57.1.107; N = 4; unchanged PID, plant, drag and damping; confirmed target course 180°.\n'
        'Shaded time is initialization data, not a forward prediction. After release, no measured state or force is imposed.\n'
        'Position, attitude and body rate use pre-release observations. Initial PID memory remains unobserved; two conditional seeds are shown.\n'
        'Purple preserves the inferred initial moment but needs more integral than the current zero-initialized controller can accumulate.',
        fontsize=9,color='#444')
    fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.17,hspace=.30,wspace=.24)
    fig.savefig(prefix.with_suffix('.png'),dpi=165,facecolor='white')
    plt.close(fig)


def main():
    protected = protected_hashes()
    source_paths = [ROOT/'outputs/r77_1_target_course_corrected.json',
        ROOT/'outputs/r77_1_target_course_corrected_trajectory.tsv',
        ROOT/'scripts/check_r77_1_capture_strategy.py',ROOT/'scripts/run_r77_1_focus_moment.py']
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    reference = json.loads(source_paths[0].read_text())
    config,p,raw_limit = inputs()
    p = replace(p,fin_force_area_per_rad=reference['fixed_parameters']['fin_force_area_per_rad'],guidance_delay_s=0.)
    assert asdict(p) == reference['fixed_parameters']
    assert digest(config) == reference['configuration_sha256']
    assert hashes[str(source_paths[3])] == reference['plant_sha256']
    assert (VERSION,COMMIT) == (reference['source_version'],reference['source_commit'])
    scenario = dict(CASES[0][3])
    assert scenario == reference['scenarios']['confirmed_180']
    _,obs = load_observations()
    assert protected[str(ROOT/'data/replays/r77_1_level_20260824.tsv')] == reference['replay_sha256']
    plant = FocusPlant(config,p,'legacy_control_scale',raw_limit)
    forces,force_hash = frozen_force_function(config,p,scenario)
    seed = reference['measured_motion_seed']
    table = tuple(map(tuple,reference['raw_time_to_gain_table']))
    warm,details,motion = initialize(obs,seed,config,p,scenario,plant,forces)
    alternative,alt_details,_ = initialize(obs,seed,config,p,scenario,plant,forces,degree=1)
    full_pn = read_trace(source_paths[1],'confirmed_180_pn_only')
    bound = integral_bound(motion,full_pn,p,scenario,plant,config,forces,table)
    assert abs(bound['last_error_g']-warm['previous_error']) < 1e-12
    assert warm['pid_integral'] > bound['integral_before_release_upper_bound_even_at_zero_actual_load']
    starts = dict(matched_initial_control=warm,two_point_rate=alternative,
                  integral_upper_bound=dict(warm,pid_integral=bound['integral_before_release_with_PN_previous_error']),
                  zero_integral_diagnostic=dict(warm,pid_integral=0.))
    results,runners = {},{}
    for name,restart in starts.items():
        runner,pid_hash = build_runner(plant,False,False,request_gain_table=table,pn_only=True,restart=restart)
        assert pid_hash == reference['pid_block_sha256']
        result = runner(config,p,scenario,max_time_s=END,plant_dt_s=.002)
        enrich(result.samples,p,scenario)
        results[name],runners[name] = result,runner
        print(name,json.dumps({part:{k:row[k] for k in ('time_s','alpha_deg',
            'wind_normal_specific_force_g','speed_kmh','commanded_horizontal_acceleration_g',
            'horizontal_fin_deflection_rad','pid_integral_output')}
            for part,row in [('initial',result.samples[0]),('final',result.samples[-1])]}),flush=True)

    # Focused verification: exact known-state restart; no post-release data in
    # initialization; unmodified force/PID blocks; no scheduled capture switch.
    known = at_time(full_pn,RELEASE)
    known_restart = dict(time_s=RELEASE,state=tuple(known[k] for k in
        ('speed_kmh','heading_rad','x_m','z_m','body_heading_rad','body_rate_rad_s')),
        pid_integral=known['pid_integral_output'],previous_error=known['pid_error_g'],
        fin_deflection=known['horizontal_fin_deflection_rad'],
        command=known['commanded_horizontal_alpha_rad'],
        commanded_acceleration_g=known['commanded_horizontal_acceleration_g'],
        pid_proportional=known['pid_proportional_output'],pid_derivative=known['pid_derivative_output'],after_control=True)
    known_restart['state'] = (known_restart['state'][0]/3.6,*known_restart['state'][1:])
    resume,_ = build_runner(plant,False,False,request_gain_table=table,pn_only=True,restart=known_restart)
    resumed = resume(config,p,scenario,max_time_s=END).samples
    old = [r for r in full_pn if r['time_s'] >= RELEASE-1e-10]
    assert len(old) == len(resumed)
    restart_errors = {f:max(abs(a[f]-b[f]) for a,b in zip(old,resumed)) for f in FIELDS}
    assert max(restart_errors.values()) < 1e-8,restart_errors
    poisoned = {k:np.asarray(v).copy() for k,v in obs.items()}
    for k in ('speed','load','alpha'):
        poisoned[k][obs['time'] > RELEASE] *= 2.
    changed,_,_ = initialize(poisoned,seed,config,p,scenario,plant,forces)
    assert changed == warm
    assert not any(r['capture_active'] for result in results.values() for r in result.samples)
    main_rows = results['matched_initial_control'].samples
    first = main_rows[0]
    assert abs(first['time_s']-RELEASE) < 1e-10
    assert abs(first['alpha_deg']-details['observed_total_alpha_deg']) < 1e-9
    assert abs(first['horizontal_fin_deflection_rad']-math.radians(details['inferred_fin_angle_deg'])) < 1e-12
    assert abs(first['body_angular_acceleration_rad_s2']-details['inferred_body_angular_acceleration_rad_s2']) < 1e-10
    cap = math.sqrt(p.command_limit_g**2-1.)
    request_error = max(abs(r['commanded_horizontal_acceleration_g']-np.clip(r['raw_pn_request_g'],-cap,cap))
                        for result in results.values() for r in result.samples[:-1])
    assert request_error < 1e-10
    fine = runners['matched_initial_control'](config,p,scenario,max_time_s=END,plant_dt_s=.001)
    assert fine.event == results['matched_initial_control'].event and len(fine.samples) == len(main_rows)
    step_errors = {f:max(abs(a[f]-b[f]) for a,b in zip(main_rows,fine.samples)) for f in FIELDS}
    assert max(step_errors.values()) < .01,step_errors
    bounded_fine = runners['integral_upper_bound'](config,p,scenario,max_time_s=END,plant_dt_s=.001)
    bounded_rows = results['integral_upper_bound'].samples
    assert len(bounded_rows) == len(bounded_fine.samples)
    bounded_step_errors = {f:max(abs(a[f]-b[f]) for a,b in zip(bounded_rows,bounded_fine.samples)) for f in FIELDS}
    assert max(bounded_step_errors.values()) < .01,bounded_step_errors
    assert protected_hashes() == protected
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in source_paths)
    frames = []
    for name,result in results.items():
        t = [r['time_s'] for r in result.samples]
        for i,time in enumerate(obs['time']):
            if not RELEASE-1e-10 <= time <= END+1e-10: continue
            row = dict(case=name,time_s=float(time),observed_alpha_deg=math.degrees(obs['alpha'][i]),
                       observed_load_g=float(obs['load'][i]),observed_speed_kmh=float(obs['speed'][i]*3.6))
            for f in ('alpha_deg','wind_normal_specific_force_g','speed_kmh','commanded_horizontal_acceleration_g'):
                row[f] = float(np.interp(time,t,[r[f] for r in result.samples]))
            frames.append(row)
    prefix = ROOT/'outputs/r77_1_release_at_1p5'
    report = dict(source_version=VERSION,source_commit=COMMIT,scenario=scenario,
        release_time_s=RELEASE,end_time_s=END,fixed_parameters=asdict(p),parameters_fitted=[],
        manual_capture_enabled=False,post_release_measurements_in_dynamics=False,
        guidance='same N=4 PN, continuously available target state',
        initialization=details,alternative_rate_initialization=alt_details,
        integral_bound=bound,moment_matched_integral_exceeds_launch_history_bound=True,
        primary_case='integral_upper_bound',
        unobserved_launch_pose_seed=seed,controller_initialization_is_model_assumption=True,
        zero_integral_is_diagnostic_not_primary=True,initial_states=starts,
        force_block_sha256=force_hash,pid_block_sha256=reference['pid_block_sha256'],
        plant_sha256=reference['plant_sha256'],configuration_sha256=digest(config),
        replay_sha256=reference['replay_sha256'],source_hashes=hashes,
        known_state_restart_max_difference=restart_errors,half_step_max_difference=step_errors,
        bounded_initialization_half_step_max_difference=bounded_step_errors,
        pn_request_max_residual_g=request_error,future_observation_leak_check_passed=True,
        protected_inputs_unchanged=True,frames=frames)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),frames)
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
        [dict(case=name,**row) for name,result in results.items() for row in result.samples])
    baseline = read_trace(source_paths[1],'confirmed_180_timed_capture')
    plot(results,baseline,obs,prefix)
    print(json.dumps(dict(initialization=details,alternative=alt_details,integral_bound=bound,
                         restart_errors=restart_errors,half_step=step_errors)),flush=True)
    print(prefix.with_suffix('.png'),flush=True)


if __name__ == '__main__':
    main()
