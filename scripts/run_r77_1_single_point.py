"""Free horizontal-plane evaluation of one aerodynamic force application point.

The existing controller, scheduler, guidance and initial conditions are reused
verbatim in a private function copy. Production files/defaults are not edited.
Only the force/moment closure changes. The vertical 1 g algebraic support is
still an external approximation, NOT a full three-dimensional single-point FM.

At zero control and rate, retain the fitted WIND-axis slope 14/rad. The control
effectiveness and input scale remain prior candidate assumptions, not observed
fin deflections. All coefficient contributions are summed into one body-normal
force N, used in translation and M=arm*N. Body axial force X is obtained from
the unchanged wind-drag law; no second induced-drag term is added. Rate damping,
when present, is represented by the same force, never a separate moment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from aim120_model import alpha_servo
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import effective_cda0, effective_cda_alpha
from aim120_model.units import G0_MPS2 as G0
from audit_r77_1_257_input_mapping import protected_hashes
from check_r77_1_257_plant_constraints import VERSION, COMMIT, load_observations
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import candidate_inputs, write_table


@dataclass(frozen=True)
class PointSample(alpha_servo.FinMomentSample):
    body_axial_force_n: float
    wind_lift_n: float
    wind_drag_n: float
    force_rotation_residual_n: float
    moment_identity_residual_nm: float


def build_runner(config, signed_arm):
    sound = StandardAtmosphere().sample(CASES[0][3]['launch_altitude_m']).speed_of_sound_mps

    def point_sample(speed, ah, omega, pressure, mass, control, p, total_alpha):
        # For an aft point arm<0, rotation adds +ell*omega to local incidence.
        local = math.atan2(speed * math.sin(ah) - signed_arm * omega,
                           speed * math.cos(ah))
        drag = pressure * (effective_cda0(speed / sound, config)
                           + effective_cda_alpha(total_alpha, 0., speed / sound, config))
        cosine, sine = math.cos(ah), math.sin(ah)
        # 14/rad remains a wind-axis zero-input/rate derivative; do not silently
        # relabel it as a body-normal coefficient. D is held fixed under the
        # coordinate conversion, including the previously selected K=0.0767.
        passive = (pressure * p.lift_area_per_rad * local + drag * sine) / cosine
        raw = math.copysign(1., signed_arm) * pressure * p.fin_force_area_per_rad * control
        limit = mass * G0 * p.fin_lateral_acceleration_limit_g
        controlled = float(np.clip(raw, -limit, limit))
        damping_coefficient = pressure / speed * p.rate_damping_derivative_m4
        damping_normal = -damping_coefficient * omega / signed_arm
        normal = passive + controlled + damping_normal
        axial = (drag - normal * sine) / cosine
        lift = normal * cosine - axial * sine
        inertia = mass * p.length_m ** 2 / 12.
        moment = signed_arm * normal
        return PointSample(
            local, raw, controlled, limit, passive,
            signed_arm * controlled, signed_arm * passive, normal,
            moment, inertia, moment / inertia, signed_arm * damping_normal,
            damping_coefficient / inertia, axial, lift, drag,
            max(abs(drag - (axial*cosine + normal*sine)),
                abs(lift - (normal*cosine - axial*sine))),
            abs(moment - signed_arm * normal))

    # A diagnostic-only copy retains the complete existing controller and
    # integration path. Exact match assertions fail if its interface drifts.
    source = inspect.getsource(alpha_servo.run_alpha_servo)
    old_acceleration = '        a_normal = normal_gain * ah / propulsion.mass_kg'
    assert source.count(old_acceleration) == 1
    source = source.replace(old_acceleration,
        '        point = point_sample(speed, ah, s[5], qbar, propulsion.mass_kg,\n'
        '                             fin_deflection, p, alpha)\n'
        '        a_normal = (point.wind_lift_n + propulsion.thrust_n * sinc * ah) / propulsion.mass_kg')
    assert source.count('fin_deflection, p)') == 2
    source = source.replace('fin_moment_sample(s[0]', 'point_sample(s[0]')
    source = source.replace('fin_deflection, p)', 'fin_deflection, p, alpha)')
    insert = '        samples.append(row)'
    assert source.count(insert) == 1
    source = source.replace(insert,
        '        row.update(body_normal_force_n=moment.equivalent_normal_force_n,\n'
        '                   body_axial_force_n=moment.body_axial_force_n,\n'
        '                   wind_lift_n=moment.wind_lift_n,\n'
        '                   force_rotation_residual_n=moment.force_rotation_residual_n,\n'
        '                   moment_identity_residual_nm=moment.moment_identity_residual_nm,\n'
        '                   signed_arm_m=signed_arm)\n' + insert)
    guard = '        r0, r1 = relative(time_s, state), relative(next_time, following)'
    assert source.count(guard) == 1
    source = source.replace(guard,
        '        next_alpha = (following[4] - following[1] + math.pi) % (2*math.pi) - math.pi\n'
        '        if abs(next_alpha) >= math.radians(80.):\n'
        '            event = "single_point_projection_domain_limit_80deg"\n'
        '            break\n' + guard)
    original = inspect.getsource(alpha_servo.run_alpha_servo)
    start = '        if time_s >= tick * control_dt_s - 1e-10:'
    end = '        boundaries = [time_s + plant_dt_s, tick * control_dt_s, end_time]'
    controller = original[original.index(start):original.index(end)]
    assert source[source.index(start):source.index(end)] == controller
    namespace = dict(vars(alpha_servo), point_sample=point_sample, signed_arm=signed_arm)
    exec(compile(source, '<single-point diagnostic>', 'exec'), namespace)
    return namespace['run_alpha_servo'], hashlib.sha256(controller.encode()).hexdigest()


def run_cases(step):
    base_area = .4151096915927259
    config, baseline = candidate_inputs('su_r_77_1',
        fin_force_area_per_rad=4*base_area,
        restoring_force_area_per_rad=2*base_area,
        rate_damping_derivative_m4=14.41)
    assert (baseline.acceleration_pid_kp, baseline.acceleration_pid_ki,
            baseline.acceleration_pid_kd, baseline.acceleration_pid_integral_limit) == (.0071,.005,.00015,1.)
    results = {'previous_candidate': alpha_servo.run_alpha_servo(
        config, baseline, CASES[0][3], max_time_s=3.5, plant_dt_s=step)}
    metadata = {}
    for name, arm, damping in (
        ('aft_local_damping', -baseline.stability_distance_m, 0.),
        ('aft_all_damping_same_point', -baseline.stability_distance_m, 14.41),
        ('forward_sign_diagnostic', baseline.stability_distance_m, 14.41),
    ):
        # The former separate restoring derivative is unused here. One force
        # slope supplies both translational lift and the restoring moment.
        p = replace(baseline, rate_damping_derivative_m4=damping,
                    restoring_force_area_per_rad=baseline.lift_area_per_rad)
        runner, digest = build_runner(config, arm)
        results[name] = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=step)
        metadata[name] = dict(signed_arm_m=arm, additional_rate_derivative_m4=damping,
            wind_zero_input_slope_per_rad=14., control_force_area_per_rad=p.fin_force_area_per_rad,
            control_limit_rad=p.horizontal_fin_limit_rad, controller_block_sha256=digest)
    return results, metadata


def summarize(result, observations):
    rows = result.samples
    t = np.array([r['time_s'] for r in rows])
    frames = []
    for i, time in enumerate(observations['time']):
        row = dict(time_s=float(time), compared=bool(time <= t[-1]+1e-10))
        for field, obs in (
            ('alpha_deg', np.degrees(observations['alpha'][i])),
            ('wind_normal_specific_force_g', observations['load'][i]),
            ('speed_kmh', observations['speed'][i]*3.6),
        ):
            row['observed_'+field] = float(obs)
            row['predicted_'+field] = (float(np.interp(time,t,[r[field] for r in rows]))
                                       if row['compared'] else None)
        for field in ('horizontal_specific_force_g','body_rate_rad_s','body_heading_rad','heading_rad'):
            row['predicted_'+field] = (float(np.interp(time,t,[r[field] for r in rows]))
                                       if row['compared'] else None)
        frames.append(row)
    peaks=[]
    for before, row, after in zip(rows, rows[1:], rows[2:]):
        if row['time_s'] < .5:
            continue
        a,b,c = [x['alpha_deg'] for x in (before,row,after)]
        kind = 'peak' if a<b>c else 'dip' if a>b<c else None
        if kind:
            peaks.append(dict(time_s=row['time_s'],alpha_deg=b,kind=kind))
    return dict(event=result.event, end_time_s=float(t[-1]), frames=frames,
        extrema=peaks, maximum_alpha_deg=max(r['alpha_deg'] for r in rows),
        maximum_load_g=max(r['wind_normal_specific_force_g'] for r in rows),
        minimum_horizontal_load_g=min(r['horizontal_specific_force_g'] for r in rows),
        final_heading_deg=math.degrees(rows[-1]['heading_rad']),
        final_body_heading_deg=math.degrees(rows[-1]['body_heading_rad']),
        final_body_rate_rad_s=rows[-1]['body_rate_rad_s'])


def save_figure(results, observations, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'previous_candidate': ('#89918e','--','Previous independent-moment candidate'),
        'aft_local_damping': ('#b97820','-','Aft point: local damping only'),
        'aft_all_damping_same_point': ('#167d9a','-','Aft point: all damping through same force'),
        'forward_sign_diagnostic': ('#985a87',':','Forward point: sign diagnostic'),
    }
    fig, axes = plt.subplots(2,2,figsize=(12,8.7),sharex=True)
    panels=[('alpha_deg','AoA magnitude [deg]','alpha'),
            ('wind_normal_specific_force_g','Total normal load [g]','load'),
            ('speed_kmh','Speed [km/h]','speed'),
            ('horizontal_specific_force_g','Signed horizontal load [g]',None)]
    for ax,(field,label,obs_key) in zip(axes.flat,panels):
        if obs_key:
            values=observations[obs_key]
            if obs_key=='alpha': values=np.degrees(values)
            if obs_key=='speed': values=values*3.6
            ax.scatter(observations['time'],values,s=23,color='#222222',label='2.57 replay',zorder=5)
        for name,result in results.items():
            # Its >600 g blow-up would hide the two main cases. The sign
            # diagnostic is reported explicitly below and in the data files.
            if name == 'forward_sign_diagnostic':
                continue
            color,style,title=styles[name]
            ax.plot([r['time_s'] for r in result.samples],[r[field] for r in result.samples],
                    color=color,ls=style,lw=1.7,label=title)
            if result.samples[-1]['time_s'] < 3.5-1e-8:
                ax.plot(result.samples[-1]['time_s'],result.samples[-1][field],'x',color=color,ms=8)
        ax.axvline(.5,color='#aaaaaa',ls=':',lw=.8)
        ax.set(ylabel=label,xlim=(0,3.55)); ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[1,1].axhline(0,color='#555',lw=.8)
    axes[1,1].text(.03,.96,'Replay gives magnitude only; no signed load points shown.',
                   va='top',transform=axes[1,1].transAxes,fontsize=9,color='#555')
    for ax in axes[1,:]: ax.set_xlabel('Flight time [s]')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False,fontsize=10)
    fig.suptitle('R-77-1 | single-point aerodynamic force | free horizontal integration',fontsize=15,y=.985)
    fig.text(.075,.022,
        'Frozen 2.57.1.107; identical PID, guidance, drag polar and launch state. No replay state is fed into the simulations.\n'
        'One body-normal N and axial X at the same point; moment = signed arm x N. Rate damping also enters translation.\n'
        'Vertical 1 g support remains imposed. 14/rad is the zero-control/rate wind-lift slope; input effectiveness is assumed.\n'
        'Forward-point sign diagnostic is not plotted here: it reaches the 80-degree domain boundary at 0.822 s.',
        fontsize=9,color='#444')
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.14,hspace=.25,wspace=.22)
    fig.savefig(prefix.with_suffix('.png'),dpi=160,facecolor='white');plt.close(fig)


def main():
    before=protected_hashes()
    results,metadata=run_cases(.002)
    # Read replay only after all free trajectories have finished.
    _,observations=load_observations()
    summaries={name:summarize(result,observations) for name,result in results.items()}

    # One focused final pass: actual shared force/moment identities, unchanged
    # controller/source inputs, and time-step refinement with held PID tick.
    fine,_=run_cases(.001)
    verification={}
    for name,result in results.items():
        coarse=result.samples; refined=fine[name].samples
        common=[r for r in coarse if r['time_s'] <= refined[-1]['time_s']+1e-10]
        ft=[r['time_s'] for r in refined]
        errors={field:max(abs(r[field]-float(np.interp(r['time_s'],ft,[x[field] for x in refined])))
                           for r in common)
                for field in ('alpha_deg','speed_kmh','wind_normal_specific_force_g')}
        assert result.event == fine[name].event
        assert errors['alpha_deg']<.05 and errors['speed_kmh']<.05 and errors['wind_normal_specific_force_g']<.05, (name,errors)
        verification[name]=dict(half_step_max_difference=errors,
            terminal_time_difference_s=abs(coarse[-1]['time_s']-refined[-1]['time_s']))
        if name!='previous_candidate':
            max_rotation=max(r['force_rotation_residual_n'] for r in coarse)
            max_moment=max(r['moment_identity_residual_nm'] for r in coarse)
            assert max_rotation<1e-7 and max_moment<1e-7
            assert all(r['rate_damping_moment_nm']*r['body_rate_rad_s']<=1e-8 for r in coarse)
            verification[name].update(max_force_rotation_residual_n=max_rotation,
                                      max_moment_identity_residual_nm=max_moment)
    assert len({m['controller_block_sha256'] for m in metadata.values()})==1
    assert protected_hashes()==before
    report=dict(source_version=VERSION,source_commit=COMMIT,
        scope='free horizontal-plane single-point candidate; imposed vertical 1g support',
        raw_PID=[.0071,.005,.00015,1.], fitted_parameters=[], metadata=metadata,
        variants=summaries, verification=verification, protected_inputs_unchanged=True)
    prefix=ROOT/'outputs/r77_1_single_point_forward'
    prefix.parent.mkdir(exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),[dict(case=name,**row) for name,s in summaries.items() for row in s['frames']])
    shared=list(results['previous_candidate'].samples[0])
    shared += [k for k in results['aft_local_damping'].samples[0] if k not in shared]
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
        [dict(case=name,**{k:row.get(k) for k in shared}) for name,result in results.items() for row in result.samples])
    save_figure(results,observations,prefix)
    print(json.dumps({name:{k:v for k,v in summary.items() if k!='frames'}
                      for name,summary in summaries.items()},indent=2))
    print(json.dumps(verification,indent=2))
    print(prefix.with_suffix('.png'))


if __name__=='__main__':
    main()
