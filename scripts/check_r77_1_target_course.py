"""Isolate the confirmed 180-degree target course with all model coefficients fixed."""
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math

import numpy as np

from check_r77_1_capture_strategy import (
    ROOT, FocusPlant, inputs, CASES, protected_hashes, load_observations,
    VERSION, COMMIT, FIELDS, digest, build_runner, enrich, analyze, write_table,
)
from aim120_model.alpha_servo import planar_target_motion
from reconstruct_r77_1_guidance_geometry import reconstruct, guidance, at_time, cumulative_integral


def read_trace(path, case):
    with path.open() as stream:
        return [{k:(v == 'True' if v in ('True', 'False') else float(v))
                 for k,v in row.items() if k != 'case'}
                for row in csv.DictReader(stream, delimiter='\t') if row['case'] == case]


def plot(results, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    strategies = [('original_capture', '#147f9b', 'Original capture + handover'),
                  ('timed_capture', '#c77c27', 'Timed capture: 35.25 g, 1.80 s'),
                  ('pn_only', '#8061a7', 'PN only: N = 4')]
    panels = [('alpha_deg', 'AoA [deg]', np.degrees(obs['alpha'])),
              ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
              ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6)]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10), sharex=True, sharey='row')
    for col, (name, color, title) in enumerate(strategies):
        for j, (field, label, measured) in enumerate(panels):
            ax = axes[j, col]
            ax.plot(obs['time'], measured, 'o-', color='#252525', ms=3.5, lw=1, zorder=4)
            for course, style, width in [('old_210', '--', 1.5), ('confirmed_180', '-', 2.1)]:
                rows = results[course+'_'+name].samples
                ax.plot([r['time_s'] for r in rows], [r[field] for r in rows],
                        color=color, ls=style, lw=width, alpha=.65 if course == 'old_210' else 1.)
            ax.set(xlim=(0,3.5))
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            if col == 0: ax.set_ylabel(label)
            if j == 0: ax.set_title(title, fontsize=11)
            if j == 2: ax.set_xlabel('Missile flight time [s]')
    fig.suptitle('R-77-1 | only target course changed: 210° → 180°', fontsize=17, y=.986)
    fig.legend(handles=[Line2D([],[],color='#252525',marker='o',lw=1,label='2.57 measurements'),
        Line2D([],[],color='#555',lw=2,label='Confirmed 180°: parallel opposite courses'),
        Line2D([],[],color='#555',lw=1.5,ls='--',label='Previous incorrect 210° course')],
        loc='upper center',bbox_to_anchor=(.5,.957),ncol=3,frameon=False,fontsize=10)
    fig.text(.065,.024,
        'Frozen 2.57.1.107; same PID, N = 4, force/moment plant, damping, drag and launch schedule. No refit or time shift.\n'
        'Launch heading 0°; target initially 8 km away at 30° bearing; both 1200 km/h at 6300 m. All curves are free forward runs.\n'
        'Previously fitted plant and capture coefficients are retained only to isolate the geometry change.',fontsize=9,color='#444')
    fig.subplots_adjust(left=.065,right=.986,top=.86,bottom=.11,hspace=.17,wspace=.075)
    fig.savefig(prefix.with_suffix('.png'),dpi=165,facecolor='white')
    plt.close(fig)


def main():
    protected = protected_hashes()
    paths = [ROOT/'outputs/r77_1_pn_only.json', ROOT/'outputs/r77_1_pn_only_trajectory.tsv',
             ROOT/'outputs/r77_1_capture_fit.json', ROOT/'outputs/r77_1_capture_fit_trajectory.tsv',
             ROOT/'scripts/run_r77_1_focus_moment.py']
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    previous, fit = json.loads(paths[0].read_text()), json.loads(paths[2].read_text())
    config, p, raw_limit = inputs()
    p = replace(p, fin_force_area_per_rad=previous['fixed_parameters']['fin_force_area_per_rad'], guidance_delay_s=0.)
    assert asdict(p) == previous['fixed_parameters'] == fit['fixed_parameters']
    assert digest(config) == previous['configuration_sha256'] == fit['configuration_sha256']
    assert (VERSION,COMMIT) == (previous['source_version'],previous['source_commit'])
    assert hashes[str(paths[4])] == previous['plant_sha256']
    raw = json.loads((ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx').read_text())['rocket']['guidance']['guidanceAutopilot']
    table = tuple(tuple(raw[k]) for k in sorted((k for k in raw if k.startswith('timeToGain')),
                                               key=lambda k:int(k.removeprefix('timeToGain'))))
    assert [list(row) for row in table] == previous['raw_time_to_gain_table']
    assert raw['propNavMult'] == p.navigation_gain == 4.
    _, obs = load_observations()
    assert protected[str(ROOT/'data/replays/r77_1_level_20260824.tsv')] == previous['replay_sha256']
    scenarios = {'confirmed_180': dict(CASES[0][3])}
    # Omitted heading reproduces the exact historical floating-point operations.
    scenarios['old_210'] = {k:v for k,v in scenarios['confirmed_180'].items()
                            if k not in ('target_heading_deg','target_course_reference')}
    early = fit['best_early_window']
    options = dict(original_capture=dict(retain_command=True),
                   timed_capture=dict(retain_command=False,
                       timed_capture=(early['capture_cap_g'],early['capture_duration_s'])),
                   pn_only=dict(retain_command=False,pn_only=True))
    plant = FocusPlant(config,p,'legacy_control_scale',raw_limit)
    results, runners, summaries = {}, {}, {}
    for strategy, opts in options.items():
        runner, pid_hash = build_runner(plant,pn_capture=False,request_gain_table=table,**opts)
        assert pid_hash == previous['pid_block_sha256']
        runners[strategy] = runner
        for course, scenario in scenarios.items():
            name = course+'_'+strategy
            result = runner(config,p,scenario,max_time_s=3.5,plant_dt_s=.002)
            enrich(result.samples,p,scenario)
            results[name] = result
            summary = analyze(result,obs)
            exits = [b['time_s'] for a,b in zip(result.samples,result.samples[1:])
                     if a['capture_active'] and not b['capture_active']]
            summary['capture_exit_time_s'] = exits[0] if exits else None
            summary.pop('handover_samples')
            summaries[name] = summary
            print(name,json.dumps({k:summary[k] for k in ('alpha_at_native_times_deg',
                'early_topology_matches','capture_exit_time_s','extrema')}),flush=True)

    # Conditional measured-motion PN check uses the same pre-guidance pose.
    before = [r for r in results['confirmed_180_pn_only'].samples if r['time_s'] <= .2]
    initial = before[-1]
    seed = dict(heading_rad=initial['heading_rad'],x_m=initial['x_m'],z_m=initial['z_m'],
        flown_m=float(cumulative_integral(np.array([r['speed_kmh']/3.6 for r in before]),
                                         np.array([r['time_s'] for r in before]))[-1]))
    motion = reconstruct(obs,seed)
    conditional = {course:[dict(r,**guidance(r,planar_target_motion(scenario)[1],scenario,p.navigation_gain))
                          for r in motion] for course,scenario in scenarios.items()}

    # Focused verification: historical reproduction, actual target path, fixed
    # request/controller mapping and convergence with the same 20 ms PID period.
    reproduction, half_step = {}, {}
    for strategy, path, case in [('original_capture',paths[1],'blue_capture_reference'),
                                 ('pn_only',paths[1],'pn_only'),
                                 ('timed_capture',paths[3],'early_capture')]:
        old = read_trace(path,case)
        new = results['old_210_'+strategy].samples
        assert len(old) == len(new)
        reproduction[strategy] = {f:max(abs(a[f]-b[f]) for a,b in zip(old,new)) for f in FIELDS}
        assert max(reproduction[strategy].values()) < 1e-9
        corrected = results['confirmed_180_'+strategy]
        fine = runners[strategy](config,p,scenarios['confirmed_180'],max_time_s=3.5,plant_dt_s=.001)
        assert corrected.event == fine.event and len(corrected.samples) == len(fine.samples)
        half_step[strategy] = {f:max(abs(a[f]-b[f]) for a,b in zip(corrected.samples,fine.samples)) for f in FIELDS}
        assert max(half_step[strategy].values()) < .01
        for row in corrected.samples:
            assert abs(row['target_z_m']-4000.) < 1e-9
            assert abs(row['target_x_m']-(8000*math.cos(math.pi/6)-1200/3.6*row['time_s'])) < 1e-9
            assert abs(row['horizontal_fin_deflection_rad']-np.clip(row['pid_raw_output'],-1.,1.)*raw_limit) < 1e-12
        assert all(abs(a[f]-b[f]) < 1e-12 for a,b in zip(new,corrected.samples)
                   if a['time_s'] <= .3 for f in FIELDS)
    request_residuals = []
    cap = math.sqrt(p.command_limit_g**2-1.)
    for row in results['confirmed_180_pn_only'].samples[:-1]:
        gain = float(np.interp(row['time_s'],*np.array(table).T))
        request_residuals.append(abs(row['commanded_horizontal_acceleration_g']-
                                     np.clip(gain*row['raw_pn_request_g'],-cap,cap)))
    assert max(request_residuals) < 1e-10
    assert protected_hashes() == protected
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)
    prefix = ROOT/'outputs/r77_1_target_course_corrected'
    report = dict(source_version=VERSION,source_commit=COMMIT,scenarios=scenarios,
        scenario_heading_deg={'old_210':210.,'confirmed_180':180.},
        target_course_source='User confirmed parallel opposite courses: launch 0 deg, target 180 deg',
        fixed_parameters=asdict(p),raw_time_to_gain_table=table,parameters_fitted=[],
        timed_capture_parameters=options['timed_capture']['timed_capture'],
        prior_fits_revalidated=False,plant_and_pid_unchanged=True,
        pid_block_sha256=previous['pid_block_sha256'],plant_sha256=previous['plant_sha256'],
        configuration_sha256=digest(config),replay_sha256=previous['replay_sha256'],
        results=summaries,baseline_reproduction_max_difference=reproduction,
        half_step_max_difference=half_step,pn_request_max_residual_g=max(request_residuals),
        measured_motion_seed=seed,measured_motion_assumptions=['level flight','positive horizontal turn',
            'unobserved 0-0.2 s pose from fixed model; interpolated measured V and G thereafter'],
        measured_motion_pn_at_2p1_s={k:at_time(v,2.1) for k,v in conditional.items()},
        source_hashes=hashes,protected_inputs_unchanged=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
                [dict(case=name,**row) for name,result in results.items() for row in result.samples])
    write_table(prefix.with_suffix('.tsv'),
                [dict(case=name,**row) for name,s in summaries.items() for row in s['frames']])
    write_table(prefix.with_name(prefix.name+'_measured_motion_pn.tsv'),
                [dict(target_course=name,**row) for name,rows in conditional.items() for row in rows])
    plot(results,obs,prefix)
    print(json.dumps(dict(baseline_reproduction=reproduction,half_step=half_step,
                         conditional_pn_at_2p1_s={k:at_time(v,2.1)['pn_horizontal_request_g'] for k,v in conditional.items()})),flush=True)
    print(prefix.with_suffix('.png'),flush=True)


if __name__ == '__main__':
    main()
