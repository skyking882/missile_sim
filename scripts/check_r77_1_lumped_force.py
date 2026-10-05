"""Test aggregate force and a single force application point on the 2.57 replay.

This is a conditional balance check, NOT a forward flight simulation. Measured
speed, AoA and total normal load drive the reconstruction. Keep the historical
thrust/mass, fixed-altitude density, reference area and baseline drag curve.
The normal load includes thrust; remove T*sin(alpha) before interpreting aero.

Compare the current wind-axis polar with an alternative body-axis baseline X0
and normal-force projection. The latter fits no AoA drag coefficient in its
zero-extra-drag case. It does not identify wing/body/fin shares. Separately test
the narrower hypothesis that the entire normal force acts at one fixed arm,
with no other moment except nonnegative rate damping. Unsigned replay angles
require a positive horizontal turn and level-flight decomposition assumption.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import effective_cda0
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2 as G0
from audit_r77_1_257_input_mapping import protected_hashes
from check_r77_1_257_plant_constraints import load_observations, VERSION, COMMIT
from run_alpha_servo_candidate import candidate_inputs, R77_1_INDUCED_DRAG_K


def cumulative(time, value):
    return np.r_[0., np.cumsum(np.diff(time) * (value[:-1] + value[1:]) / 2)]


def grid_for(time, subdivisions):
    return np.r_[np.concatenate([np.linspace(a,b,subdivisions,endpoint=False)
                                for a,b in zip(time,time[1:])]),time[-1]]


def least_squares_nonnegative_last(matrix, rhs):
    scales = np.linalg.norm(matrix,axis=0)
    result = np.linalg.lstsq(matrix/scales, rhs, rcond=None)[0]/scales
    if result[-1] < 0:
        result[:-1] = np.linalg.lstsq(matrix[:,:-1],rhs,rcond=None)[0]
        result[-1] = 0.
    return result


def analyze(subdivisions):
    p, o = load_observations()
    cfg, _ = candidate_inputs('su_r_77_1')
    prop = PiecewisePropulsion.from_config(cfg)
    atmo = StandardAtmosphere().sample(6300.)
    time = grid_for(o['time'],subdivisions)
    speed,alpha,load = [np.interp(time,o['time'],o[k]) for k in ('speed','alpha','load')]
    mass = np.array([prop.sample(float(t)).mass_kg for t in time])
    thrust = np.array([prop.sample(float(t)).thrust_n for t in time])
    pressure = .5*atmo.density_kg_m3*speed**2
    qs = pressure*o['sref']
    lift = mass*G0*load-thrust*np.sin(alpha)
    baseline = pressure*np.array([effective_cda0(float(v/atmo.speed_of_sound_mps),cfg) for v in speed])
    # Alternative: X=X0. Invert the measured wind-normal force to find N.
    body_normal = (lift+baseline*np.sin(alpha))/np.cos(alpha)
    body_drag = baseline/np.cos(alpha)+lift*np.tan(alpha)
    wind_basis = qs*(14*alpha)**2
    body_basis = qs*alpha**2/np.cos(alpha)
    drag_wind = baseline+R77_1_INDUCED_DRAG_K*wind_basis
    axial_wind = (thrust*np.cos(alpha)-drag_wind)/mass
    axial_body = (thrust*np.cos(alpha)-body_drag)/mass
    wind_velocity = speed[0]+cumulative(time,axial_wind)
    body_velocity = speed[0]+cumulative(time,axial_body)
    assert np.allclose(body_drag,baseline*np.cos(alpha)+body_normal*np.sin(alpha),rtol=1e-12)
    assert np.allclose(lift,body_normal*np.cos(alpha)-baseline*np.sin(alpha),rtol=1e-12)

    windows=[]
    for start in (.2,.5,.7):
        mask=time >= start-1e-10
        t=time[mask]; v=speed[mask]; m=mass[mask]
        dv=v[-1]-v[0]
        free_wind=np.trapezoid((thrust[mask]*np.cos(alpha[mask])-baseline[mask])/m,t)
        free_body=np.trapezoid(axial_body[mask],t)
        k_wind=(free_wind-dv)/np.trapezoid(wind_basis[mask]/m,t)
        b_body=(free_body-dv)/np.trapezoid(body_basis[mask]/m,t)
        v_body=v[0]+cumulative(t,axial_body[mask])
        native=o['time'][o['time']>=start]
        v_obs=np.interp(native,o['time'],o['speed'])
        residual=3.6*(np.interp(native,t,v_body)-v_obs)
        windows.append(dict(start_s=start,end_s=3.5,observed_end_speed_kmh=float(v[-1]*3.6),
            zero_extra_body_end_speed_kmh=float(v_body[-1]*3.6),
            zero_extra_body_worst_native_speed_error_kmh=float(max(abs(residual))),
            endpoint_matched_wind_K=float(k_wind),endpoint_matched_body_alpha2_coefficient=float(b_body),
            body_alpha2_coefficient_is_unconstrained=True))

    # Positive horizontal force/turn, near-level reconstruction. At these
    # high loads the vertical fraction is small, but it remains an assumption.
    horizontal_fraction=np.sqrt(1-1/load**2)
    heading=cumulative(time,G0*np.sqrt(load**2-1)/speed)
    theta=alpha*horizontal_fraction+heading
    normal_h=body_normal*horizontal_fraction
    inertia=mass*p.length_m**2/12
    moment_results=[]
    moment_curves={}
    for start in (.7,1.1):
        mask=time>=start-1e-10
        t=time[mask]; th=theta[mask]; nh=normal_h[mask]; I=inertia[mask]
        native=o['time'][o['time']>=start]
        th_native=np.interp(native,t,th)
        delta=th_native-th_native[0]
        widths=np.diff(t); mids=(t[:-1]+t[1:])/2
        kernel=np.maximum(native[:,None]-mids[None,:],0)*widths
        # Integrate the observed piecewise-linear body angle; avoid a second
        # derivative of sparse rounded screenshot data.
        omega=np.diff(th)/widths
        n_mid=(nh[:-1]+nh[1:])/2
        I_mid=(I[:-1]+I[1:])/2
        q_mid=(pressure[mask][:-1]+pressure[mask][1:])/2
        v_mid=(speed[mask][:-1]+speed[mask][1:])/2
        force_column=kernel@(n_mid/I_mid)
        damping_column=kernel@(-q_mid/(v_mid*I_mid)*omega)
        matrix=np.column_stack((native-start,damping_column))
        for name,arm in [('aft_single_force',-p.stability_distance_m),
                         ('forward_sign_counterfactual',p.stability_distance_m)]:
            coefficients=least_squares_nonnegative_last(matrix,delta-arm*force_column)
            prediction=matrix@coefficients+arm*force_column
            moment_results.append(dict(start_s=start,case=name,signed_arm_m=arm,
                fitted_initial_rate_rad_s=float(coefficients[0]),fitted_nonnegative_damping_m4=float(coefficients[1]),
                max_body_angle_error_deg=float(max(abs(np.degrees(prediction-delta)))),
                observed_first_interval_mean_rate_rad_s=float((th_native[1]-th_native[0])/(native[1]-native[0])),
                measured_body_rate_available=False))
            if start==.7:
                moment_curves[name]=[native.tolist(),np.degrees(prediction).tolist()]
        if start==.7:
            moment_curves['reconstructed_observation']=[native.tolist(),np.degrees(delta).tolist()]

    frame_rows=[]
    for i,t in enumerate(o['time']):
        frame_rows.append(dict(time_s=float(t),observed_speed_kmh=float(o['speed'][i]*3.6),
            observed_load_g=float(o['load'][i]),alpha14_load_g=float(o['baseline_load'][i]),
            observed_alpha_deg=float(np.degrees(o['alpha'][i])),
            wind_polar_speed_budget_kmh=float(np.interp(t,time,wind_velocity)*3.6),
            body_projection_speed_budget_kmh=float(np.interp(t,time,body_velocity)*3.6),
            reconstructed_wind_lift_n=float(np.interp(t,time,lift)),
            conditional_body_normal_n=float(np.interp(t,time,body_normal))))
    return dict(source_version=VERSION,source_commit=COMMIT,reference_area_m2=o['sref'],
        method='conditional measured-state balance; not a forward simulation',
        body_model='X equals the existing zero-AoA baseline; N inferred from measured wind-normal force',
        force_partition_identified=False,game_CyK_identified=False,
        fixed_PID=[p.acceleration_pid_kp,p.acceleration_pid_ki,p.acceleration_pid_kd,p.acceleration_pid_integral_limit],
        windows=windows,moment_tests=moment_results,frames=frame_rows,moment_curves=moment_curves)


def main():
    before=protected_hashes()
    report=analyze(80)
    fine=analyze(160)
    speed_difference=max(abs(a['body_projection_speed_budget_kmh']-b['body_projection_speed_budget_kmh'])
                         for a,b in zip(report['frames'],fine['frames']))
    moment_difference=max(abs(a['max_body_angle_error_deg']-b['max_body_angle_error_deg'])
                          for a,b in zip(report['moment_tests'],fine['moment_tests']))
    assert speed_difference<.01 and moment_difference<.1
    report['verification']=dict(subdivisions=[80,160],max_speed_budget_difference_kmh=speed_difference,
        max_moment_error_difference_deg=moment_difference,protected_inputs_unchanged=protected_hashes()==before)
    assert report['verification']['protected_inputs_unchanged']
    report['replay_sha256']=hashlib.sha256((ROOT/'data/replays/r77_1_level_20260824.tsv').read_bytes()).hexdigest()
    prefix=ROOT/'outputs/r77_1_lumped_force_check'
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    with prefix.with_suffix('.tsv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['frames'][0]),delimiter='\t')
        writer.writeheader(); writer.writerows(report['frames'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,10))
    rows=report['frames'];t=[r['time_s'] for r in rows]
    axes[0].plot(t,[r['observed_load_g'] for r in rows],'o-',color='#222',ms=4,label='Game replay')
    axes[0].plot(t,[r['alpha14_load_g'] for r in rows],color='#167d9a',label='Aggregate lift: 14/rad + thrust normal force')
    axes[0].set(ylabel='Total normal load [g]',title='Aggregate-force check at observed speed and AoA')
    axes[1].plot(t,[r['observed_speed_kmh'] for r in rows],'o-',color='#222',ms=4,label='Game replay')
    axes[1].plot(t,[r['wind_polar_speed_budget_kmh'] for r in rows],color='#167d9a',label='Current wind-axis polar, K=0.0767')
    axes[1].plot(t,[r['body_projection_speed_budget_kmh'] for r in rows],'--',color='#bd7829',label='Body-axis X0 + normal-force projection, no extra alpha drag')
    axes[1].set(ylabel='Speed budget [km/h]',title='Measured-state axial balance; all curves anchored at 0.2 s')
    colors={'aft_single_force':'#b85969','forward_sign_counterfactual':'#bd7829','reconstructed_observation':'#222'}
    labels={'aft_single_force':'All normal force at aft 0.175 m','forward_sign_counterfactual':'Opposite arm sign (diagnostic only)',
            'reconstructed_observation':'Reconstructed body-angle change'}
    for name,(t,angle) in report['moment_curves'].items():
        axes[2].plot(t,angle,'o-',color=colors[name],ms=3,label=labels[name])
    axes[2].set(ylabel='Body-angle change [deg]',xlabel='Flight time [s]',
                title='Single-point force hypothesis: best initial rate and nonnegative damping')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best');ax.set_xlim(.15,3.55)
    fig.suptitle('R-77-1 | aggregate force and moment consistency | frozen 2.57.1.107',fontsize=14)
    fig.text(.10,.019,'Conditional on replay timing, displayed AoA/load, fixed 6300 m altitude and nominal propulsion.\n'
        'The force split and actual fin input are unobserved. This checks consistency, not the game source formula.\n'
        'No PID or production model changes. The single-point test assumes no other aerodynamic moment.',fontsize=9,color='#444')
    fig.subplots_adjust(left=.10,right=.98,top=.92,bottom=.11,hspace=.36)
    fig.savefig(prefix.with_suffix('.png'),dpi=160,facecolor='white');plt.close(fig)
    print(json.dumps({k:report[k] for k in ('windows','moment_tests','verification')},indent=2))
    print(prefix.with_suffix('.png'))


if __name__=='__main__':
    main()
