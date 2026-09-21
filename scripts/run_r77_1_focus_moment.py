"""Aerodynamic-center moment candidate, frozen 2.57 R-77-1 inputs.

Preserve the entire existing translational force function: aggregate wind lift
14/rad, drag polar, propulsion and algebraic vertical support. Replace only
horizontal attitude dynamics by M_cm=-ell*N_body+M_ac. M_ac is a control couple
and a rate-damping couple; no added translational control force is introduced.
This is a planar ROM hypothesis, not identified War Thunder engine semantics.

Compare the previous assumed control-moment scale to a no-fit hypothesis in
which raw finsAoa is the maximum zero-rate trim angle in radians. Raw PID gains
are never changed. Static roots and a frozen-speed turning step precede the
free closed-loop trajectories. Replay data are read only for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2 as G0
from audit_r77_1_257_input_mapping import protected_hashes
from check_r77_1_257_plant_constraints import VERSION, COMMIT, load_observations
from gen0_alpha_servo_model import CASES
from run_alpha_servo_candidate import candidate_inputs, write_table
from run_r77_1_single_point import summarize


@dataclass(frozen=True)
class FocusSample(alpha_servo.FinMomentSample):
    body_axial_force_n: float
    aerodynamic_center_moment_nm: float
    normalized_control: float
    force_rotation_residual_n: float
    moment_transfer_residual_nm: float


class FocusPlant:
    def __init__(self, config, p, mode, raw_trim_limit):
        self.config, self.p, self.mode = config, p, mode
        self.raw_trim_limit = raw_trim_limit
        self.atmo = StandardAtmosphere().sample(CASES[0][3]['launch_altitude_m'])
        # Preserve the former damping to first order at zero incidence: the
        # local-inflow term becomes a couple independent of actual alpha.
        self.damping_m4 = (p.rate_damping_derivative_m4
                           + p.restoring_area_per_rad*p.stability_distance_m**2)

    def body_force(self, speed, alpha_h, total_alpha):
        p = self.p
        pressure = .5*self.atmo.density_kg_m3*speed**2
        lift = pressure*p.lift_area_per_rad*alpha_h
        drag = pressure*(effective_cda0(speed/self.atmo.speed_of_sound_mps,self.config)
            + effective_cda_alpha(total_alpha,0.,speed/self.atmo.speed_of_sound_mps,self.config))
        c,s=math.cos(alpha_h),math.sin(alpha_h)
        normal=lift*c+drag*s
        axial=drag*c-lift*s
        residual=max(abs(lift-(normal*c-axial*s)),abs(drag-(axial*c+normal*s)))
        return normal,axial,lift,drag,residual

    def sample(self, speed, ah, omega, pressure, mass, control, p, total_alpha):
        u=control/p.horizontal_fin_limit_rad
        normal,axial,_,_,residual=self.body_force(speed,ah,total_alpha)
        if self.mode=='legacy_control_scale':
            equivalent_control=pressure*p.fin_force_area_per_rad*control
        elif self.mode=='raw_trim_angle':
            trim=u*self.raw_trim_limit
            # Evaluate at the commanded angle, independent of actual alpha.
            # Static horizontal trim, without the imposed vertical support.
            equivalent_control=self.body_force(speed,trim,abs(trim))[0]
        else:
            raise ValueError(self.mode)
        limit=mass*G0*p.fin_lateral_acceleration_limit_g
        limited=float(np.clip(equivalent_control,-limit,limit))
        control_moment=p.stability_distance_m*limited
        damping_coefficient=pressure/speed*self.damping_m4
        damping=-damping_coefficient*omega
        focus=control_moment+damping
        restoring=-p.stability_distance_m*normal
        moment=restoring+focus
        inertia=mass*p.length_m**2/12.
        return FocusSample(ah,-equivalent_control,-limited,limit,normal,
            control_moment,restoring,normal,moment,inertia,moment/inertia,
            damping,damping_coefficient/inertia,axial,focus,u,residual,
            abs(moment-(-p.stability_distance_m*normal+focus)))


def build_runner(plant, error_scale=1.):
    original=inspect.getsource(alpha_servo.run_alpha_servo)
    source=original
    assert source.count('fin_deflection, p)')==2
    source=source.replace('fin_moment_sample(s[0]','focus_sample(s[0]')
    source=source.replace('fin_deflection, p)','fin_deflection, p, alpha)')
    insert='        samples.append(row)'
    assert source.count(insert)==1
    source=source.replace(insert,
        '        row.update(body_normal_force_n=moment.equivalent_normal_force_n,\n'
        '                   body_axial_force_n=moment.body_axial_force_n,\n'
        '                   aerodynamic_center_moment_nm=moment.aerodynamic_center_moment_nm,\n'
        '                   normalized_control=moment.normalized_control,\n'
        '                   force_rotation_residual_n=moment.force_rotation_residual_n,\n'
        '                   moment_transfer_residual_nm=moment.moment_transfer_residual_nm)\n'+insert)
    error_line='                error_g = commanded_acceleration_g - accel / G0'
    assert source.count(error_line)==1
    if error_scale!=1.:
        source=source.replace(error_line,error_line+'\n                error_g *= error_scale')
    # Preserve the whole force function and, apart from an explicitly selected
    # unit-conversion diagnostic, the whole existing PID/guidance block.
    a,b='    def forces(t, s):','    def record(t, s):'
    assert source[source.index(a):source.index(b)]==original[original.index(a):original.index(b)]
    a='        if time_s >= tick * control_dt_s - 1e-10:'
    b='        boundaries = [time_s + plant_dt_s, tick * control_dt_s, end_time]'
    controller=original[original.index(a):original.index(b)]
    changed=source[source.index(a):source.index(b)].replace('\n                error_g *= error_scale','')
    assert changed==controller
    namespace=dict(vars(alpha_servo),focus_sample=plant.sample,error_scale=error_scale)
    exec(compile(source,'<focus-moment diagnostic>','exec'),namespace)
    return namespace['run_alpha_servo'],hashlib.sha256(controller.encode()).hexdigest()


def root_in_domain(function):
    grid=np.linspace(0.,math.radians(80.),161)
    for left,right in zip(grid[:-1],grid[1:]):
        fl,fr=function(left),function(right)
        if abs(fl)<1e-10: return float(left)
        if fl*fr>0.: continue
        for _ in range(55):
            mid=(left+right)/2
            fm=function(mid)
            if fl*fm<=0.: right=mid
            else: left,fl=mid,fm
        return float((left+right)/2)
    return None


def static_and_step(plant, propulsion, dt=.002):
    p=plant.p
    speed=1600./3.6
    ps=propulsion.sample(1.)
    pressure=.5*plant.atmo.density_kg_m3*speed**2

    def sample(alpha,omega,u):
        return plant.sample(speed,alpha,omega,pressure,ps.mass_kg,
                            u*p.horizontal_fin_limit_rad,p,abs(alpha))

    def heading_rate(alpha):
        return (pressure*p.lift_area_per_rad*alpha+ps.thrust_n*math.sin(alpha))/(ps.mass_kg*speed)

    static=[]
    for u in np.linspace(0.,1.,21):
        root=root_in_domain(lambda alpha:sample(alpha,0.,u).moment_nm)
        turn_root=root_in_domain(lambda alpha:sample(alpha,heading_rate(alpha),u).moment_nm)
        static.append(dict(normalized_control=float(u),
            static_trim_deg=None if root is None else math.degrees(root),
            steady_turn_alpha_deg=None if turn_root is None else math.degrees(turn_root),
            static_moment_residual_nm=None if root is None else sample(root,0.,u).moment_nm))
    # A freely changing flight heading distinguishes steady-turn incidence
    # from zero-rate static trim. No PID or guidance in this frozen plant step.
    u=.5
    target=root_in_domain(lambda alpha:sample(alpha,heading_rate(alpha),u).moment_nm)
    state=np.zeros(2)
    rows=[]
    end=4.
    for index in range(round(end/dt)+1):
        time=index*dt
        if index%max(1,round(.02/dt))==0:
            rows.append(dict(time_s=time,alpha_deg=math.degrees(state[0]),body_rate_rad_s=float(state[1])))
        if index==round(end/dt): break
        def rhs(s):
            moment=sample(float(s[0]),float(s[1]),u)
            return np.array([s[1]-heading_rate(s[0]),moment.angular_acceleration_rad_s2])
        k1=rhs(state);k2=rhs(state+dt*k1/2);k3=rhs(state+dt*k2/2);k4=rhs(state+dt*k3)
        state+=dt*(k1+2*k2+2*k3+k4)/6
    final=rows[-1]['alpha_deg']; maximum=max(r['alpha_deg'] for r in rows)
    return dict(frozen_speed_kmh=1600.,frozen_propulsion_time_s=1.,
        static_curve=static,step_input_fraction=u,
        steady_turn_alpha_deg=math.degrees(target),step_end_alpha_deg=final,
        step_max_alpha_deg=maximum,step_overshoot_deg=max(0.,maximum-math.degrees(target)),
        step=rows)


def inputs():
    area=.4151096915927259
    config,p=candidate_inputs('su_r_77_1',fin_force_area_per_rad=4*area,
        restoring_force_area_per_rad=2*area,rate_damping_derivative_m4=14.41)
    raw_path=ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx'
    raw=json.loads(raw_path.read_text())['rocket']
    assert (p.acceleration_pid_kp,p.acceleration_pid_ki,p.acceleration_pid_kd,p.acceleration_pid_integral_limit)==(.0071,.005,.00015,1.)
    return config,p,raw['finsAoaHor']


def run_cases(step=.002,include_si=False):
    config,p,raw_trim=inputs()
    plants={mode:FocusPlant(config,p,mode,raw_trim) for mode in ('legacy_control_scale','raw_trim_angle')}
    benchmarks={mode:static_and_step(plant,PiecewisePropulsion.from_config(config),step)
                for mode,plant in plants.items()}
    results={'previous_candidate':alpha_servo.run_alpha_servo(config,p,CASES[0][3],max_time_s=3.5,plant_dt_s=step)}
    modes=[('focus_legacy_control','legacy_control_scale',1.),('focus_trim_g_error','raw_trim_angle',1.)]
    if include_si: modes.append(('focus_trim_si_error','raw_trim_angle',G0))
    metadata={}
    for name,mode,scale in modes:
        runner,digest=build_runner(plants[mode],scale)
        results[name]=runner(config,p,CASES[0][3],max_time_s=3.5,plant_dt_s=step)
        metadata[name]=dict(control_mapping=mode,pid_error_scale_from_g=scale,
            damping_derivative_m4=plants[mode].damping_m4,raw_trim_limit_rad=raw_trim,
            source_controller_sha256=digest)
    return results,benchmarks,metadata,plants


def save_figure(results,benchmarks,obs,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={
        'previous_candidate':('#929894','--','Previous candidate'),
        'focus_legacy_control':('#b87925','-','Focus moment: independent control gain'),
        'focus_trim_g_error':('#167d9a','-','Focus moment: raw trim limit, error in g'),
        'focus_trim_si_error':('#975b83','-','Raw trim limit, error in m/s² (hypothesis)'),
    }
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    panels=[('alpha_deg','AoA [deg]','alpha'),('wind_normal_specific_force_g','Total normal load [g]','load'),
            ('speed_kmh','Speed [km/h]','speed'),('commanded_horizontal_acceleration_g','Requested horizontal load [g]',None)]
    for ax,(field,label,key) in zip(axes[:2,:].flat,panels):
        if key:
            val=obs[key]
            if key=='alpha': val=np.degrees(val)
            if key=='speed': val=val*3.6
            ax.scatter(obs['time'],val,s=22,color='#222',label='2.57 replay',zorder=5)
        for name,result in results.items():
            color,style,title=styles[name]
            ax.plot([r['time_s'] for r in result.samples],[r[field] for r in result.samples],
                    color=color,ls=style,lw=1.7,label=title)
        ax.set(xlim=(0,3.55),ylabel=label,xlabel='Flight time [s]')
        ax.axvline(.5,color='#aaa',ls=':',lw=.8)
    for mode,color,title in [('legacy_control_scale','#b87925','Independent control gain'),
                              ('raw_trim_angle','#167d9a','Raw trim-angle hypothesis')]:
        b=benchmarks[mode]
        axes[2,0].plot([r['normalized_control'] for r in b['static_curve']],
                       [r['static_trim_deg'] for r in b['static_curve']],color=color,label=title)
        axes[2,1].plot([r['time_s'] for r in b['step']],[r['alpha_deg'] for r in b['step']],color=color,label=title)
        axes[2,1].axhline(b['steady_turn_alpha_deg'],color=color,ls=':',lw=1)
    axes[2,0].set(xlabel='Normalized control output u',ylabel='Zero-rate trim AoA [deg]',title='Static trim at 1600 km/h')
    axes[2,1].set(xlabel='Time since fixed input [s]',ylabel='AoA [deg]',title='Frozen-speed plant step: u = 0.5; no PID')
    axes[2,1].legend(fontsize=9,frameon=False,loc='lower right')
    for ax in axes.flat:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.946),ncol=2,frameon=False,fontsize=10)
    fig.suptitle('R-77-1 | aerodynamic-center moment | frozen 2.57.1.107',fontsize=16,y=.984)
    fig.text(.075,.024,'Aggregate wind lift stays 14/rad; original drag and thrust stay in translation. M_cm = -ell N + control couple + damping couple.\n'
        'All runs retain raw PID gains and guidance code. Any error-unit change is explicitly labelled; no coefficients are fitted here.\n'
        'Planar candidate with imposed vertical 1 g support; control mapping, inertia, damping and focal-point interpretation remain assumptions.',
        fontsize=9,color='#444')
    fig.subplots_adjust(left=.075,right=.98,top=.85,bottom=.11,hspace=.39,wspace=.22)
    fig.savefig(prefix.with_suffix('.png'),dpi=160,facecolor='white');plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-si-error',action='store_true')
    parser.add_argument('--screen-only',action='store_true')
    args=parser.parse_args()
    before=protected_hashes()
    results,benchmarks,metadata,plants=run_cases(include_si=args.include_si_error)
    _,obs=load_observations()
    summaries={name:summarize(result,obs) for name,result in results.items()}
    if args.screen_only:
        print(json.dumps(dict(benchmarks={k:{f:v for f,v in b.items() if f not in ('step','static_curve')} for k,b in benchmarks.items()},
            static_limits={k:b['static_curve'][-1] for k,b in benchmarks.items()},
            variants={name:{k:v for k,v in s.items() if k!='frames'} for name,s in summaries.items()}),indent=2))
        return
    fine,fine_b,_,_=run_cases(.001,args.include_si_error)
    verification={}
    for name,result in results.items():
        assert result.event==fine[name].event and abs(result.samples[-1]['time_s']-3.5)<1e-9
        assert len(result.samples)==len(fine[name].samples)
        errors={field:max(abs(a[field]-b[field]) for a,b in zip(result.samples,fine[name].samples))
                for field in ('alpha_deg','speed_kmh','wind_normal_specific_force_g')}
        assert max(errors.values())<.01,(name,errors)
        verification[name]=dict(half_step_max_difference=errors)
        if name!='previous_candidate':
            assert max(r['force_rotation_residual_n'] for r in result.samples)<1e-7
            assert max(r['moment_transfer_residual_nm'] for r in result.samples)<1e-7
            assert all(r['rate_damping_moment_nm']*r['body_rate_rad_s']<=1e-8 for r in result.samples)
    for mode,b in benchmarks.items():
        assert abs(b['step_end_alpha_deg']-b['steady_turn_alpha_deg'])<.05
        assert max(abs(a['alpha_deg']-c['alpha_deg']) for a,c in zip(b['step'],fine_b[mode]['step']))<.001
        assert max(abs(row['static_moment_residual_nm'] or 0.) for row in b['static_curve'])<1e-7
        # At fixed V, mass, control and rate the focal moment is independent
        # of alpha. This checks the actual implemented transfer, not labels.
        plant=plants[mode];p=plant.p;v=1600/3.6;q=.5*plant.atmo.density_kg_m3*v*v
        moments=[plant.sample(v,a,.2,q,180.,.3*p.horizontal_fin_limit_rad,p,abs(a)).aerodynamic_center_moment_nm
                 for a in (0.,.1,.3,.5)]
        assert max(moments)-min(moments)<1e-8
    assert protected_hashes()==before
    report=dict(source_version=VERSION,source_commit=COMMIT,raw_PID=[.0071,.005,.00015,1.],
        fitted_parameters=[],metadata=metadata,benchmarks=benchmarks,variants=summaries,
        verification=verification,protected_inputs_unchanged=True)
    prefix=ROOT/'outputs/r77_1_focus_moment'
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),[dict(case=name,**row) for name,s in summaries.items() for row in s['frames']])
    fields=list(results['previous_candidate'].samples[0])
    fields += [k for k in results['focus_legacy_control'].samples[0] if k not in fields]
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
        [dict(case=name,**{k:row.get(k) for k in fields}) for name,result in results.items() for row in result.samples])
    save_figure(results,benchmarks,obs,prefix)
    print(json.dumps(dict(benchmarks={k:{f:v for f,v in b.items() if f not in ('step','static_curve')} for k,b in benchmarks.items()},
        variants={name:{k:v for k,v in s.items() if k!='frames'} for name,s in summaries.items()},verification=verification),indent=2))
    print(prefix.with_suffix('.png'))


if __name__=='__main__':
    main()
