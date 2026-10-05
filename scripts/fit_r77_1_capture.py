"""Fit only a capture request ceiling and duration on the frozen 2.57 plant.

Capture keeps the existing angular-error request, clips it at G_capture, and
switches directly to PN at 0.30 + T_capture. No release filter or angle exit.
Replay states are used only to score free-flight predictions at native times.
"""
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


def measures(result, obs):
    rows = result.samples
    assert abs(rows[-1]['time_s']-3.5) < 1e-9
    t = np.array([r['time_s'] for r in rows])
    a = np.interp(obs['time'], t, [r['alpha_deg'] for r in rows])
    error = np.abs(a-np.degrees(obs['alpha']))
    full = obs['time'] >= .5
    early = full & (obs['time'] <= 2.1)
    a07, a09, a11, a13 = np.interp([.7, .9, 1.1, 1.3], obs['time'], a)
    return dict(max_alpha_error_deg=float(max(error[full])),
        early_max_alpha_error_deg=float(max(error[early])),
        mean_absolute_alpha_error_deg=float(np.mean(error[full])),
        early_topology_matches=bool(a07 < a09 > a11 < a13),
        change_0p9_to_1p1_deg=float(a11-a09),
        change_1p1_to_1p3_deg=float(a13-a11))


def plot(results, obs, selected, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = (
        ('alpha_deg', 'AoA [deg]', np.degrees(obs['alpha'])),
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', None),
        ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
        ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6),
    )
    cap, duration = selected['capture_cap_g'], selected['capture_duration_s']
    styles = {
        'blue_reference': ('#177e9b', '--', 'Previous blue capture'),
        'early_capture': ('#ce7826', '-', f'Early fit: {cap:.2f} g; T = {duration:.2f} s shown'),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for ax, (field, label, measured) in zip(axes.flat, panels):
        if measured is not None:
            ax.plot(obs['time'], measured, 'o-', color='#252525', lw=1., ms=4.5,
                    label='2.57 replay', zorder=5)
        for name, (color, ls, title) in styles.items():
            rows = results[name].samples
            ax.plot([r['time_s'] for r in rows], [r[field] for r in rows],
                    color=color, ls=ls, lw=1.9, label=title)
        ax.axvspan(.9, 1.1, color='#e3e8ec', alpha=.55, zorder=0)
        ax.axvline(.3+duration, color='#ce7826', lw=1., ls=':', alpha=.8)
        ax.set(xlim=(.2, 3.5), xlabel='Missile flight time [s]', ylabel=label)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .952),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle('R-77-1 | two capture parameters; fixed plant and PID', fontsize=16, y=.984)
    fig.text(.08, .024,
        'Fit: smallest worst AoA error at native replay samples, 0.5–2.1 s. Later samples, load and speed are unfitted checks.\n'
        f'Capture starts at 0.30 s; direct switch to PN at {0.3+duration:.2f} s (dotted line). Shading: measured AoA decline.\n'
        f'T >= {duration:.2f} s gives the same early fit; the shortest minimizing duration is shown.\n'
        '2.57.1.107; raw PID, N = 4, launch gain table, initial state and all force/moment coefficients fixed. No time shift.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.08, right=.98, top=.855, bottom=.145, hspace=.27, wspace=.24)
    fig.savefig(prefix.with_suffix('.png'), dpi=165, facecolor='white')
    plt.close(fig)


def main():
    protected = protected_hashes()
    old_path = ROOT/'outputs/r77_1_pn_only.json'
    old_trace = ROOT/'outputs/r77_1_pn_only_trajectory.tsv'
    plant_path = ROOT/'scripts/run_r77_1_focus_moment.py'
    raw_path = ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx'
    extras = {path:hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (old_path, old_trace, plant_path, raw_path)}
    old = json.loads(old_path.read_text())
    config, p, raw_limit = inputs()
    p = replace(p, fin_force_area_per_rad=old['fixed_parameters']['fin_force_area_per_rad'],
                guidance_delay_s=0.)
    assert asdict(p) == old['fixed_parameters']
    assert digest(config) == old['configuration_sha256']
    assert extras[plant_path] == old['plant_sha256']
    assert (VERSION, COMMIT) == (old['source_version'], old['source_commit'])
    autopilot = json.loads(raw_path.read_text())['rocket']['guidance']['guidanceAutopilot']
    table = tuple(tuple(autopilot[key]) for key in sorted(
        (k for k in autopilot if k.startswith('timeToGain')),
        key=lambda k:int(k.removeprefix('timeToGain'))))
    assert [list(k) for k in table] == old['raw_time_to_gain_table']
    assert p.navigation_gain == autopilot['propNavMult'] == 4.
    _, obs = load_observations()
    assert protected[str(ROOT/'data/replays/r77_1_level_20260824.tsv')] == old['replay_sha256']
    plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    cap_max = math.sqrt(p.command_limit_g**2-1.)
    cache = {}

    def runner_for(cap, duration):
        runner, pid_hash = build_runner(plant, False, False, request_gain_table=table,
                                       timed_capture=(cap, duration))
        assert pid_hash == old['pid_block_sha256']
        return runner

    def evaluate(cap, duration):
        key = (round(float(cap), 8), round(float(duration), 2))
        if key not in cache:
            runner = runner_for(*key)
            result = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)
            cache[key] = dict(capture_cap_g=key[0], capture_duration_s=key[1],
                              **measures(result, obs))
        return cache[key]

    # Bound the two-dimensional search by the raw request cap and observed
    # time span. Refine around both full-window and early-window minima.
    for i, duration in enumerate(np.arange(.02, 3.201, .04)):
        for cap in [*np.arange(0., 49., 5.), cap_max]:
            evaluate(cap, duration)
        if (i+1) % 10 == 0:
            best = min(cache.values(), key=lambda r:r['max_alpha_error_deg'])
            print('coarse', len(cache), json.dumps(best), flush=True)
    seeds = []
    for objective in ('max_alpha_error_deg', 'early_max_alpha_error_deg'):
        ranked = sorted(cache.values(), key=lambda r:(r[objective], r['capture_duration_s']))
        seeds += ranked[:3]
    for i, seed in enumerate(seeds):
        caps = np.clip(np.arange(seed['capture_cap_g']-5., seed['capture_cap_g']+5.01, .25),
                       0., cap_max)
        durations = np.clip(np.arange(seed['capture_duration_s']-.04,
                                      seed['capture_duration_s']+.041, .02), .02, 3.2)
        for duration in durations:
            for cap in caps:
                evaluate(cap, duration)
        print('refine', i+1, len(cache), flush=True)
    selected = min(cache.values(), key=lambda r:(r['max_alpha_error_deg'],
        r['mean_absolute_alpha_error_deg'], r['capture_duration_s']))
    early_best = min(cache.values(), key=lambda r:(r['early_max_alpha_error_deg'],
        r['capture_duration_s'], r['max_alpha_error_deg']))
    shape_candidates = [r for r in cache.values() if r['early_topology_matches']]
    shape_best = min(shape_candidates, key=lambda r:r['max_alpha_error_deg']) if shape_candidates else None
    print('selected', json.dumps(selected), 'early_best', json.dumps(early_best),
          'shape_candidates', len(shape_candidates), 'shape_best', json.dumps(shape_best), flush=True)

    reference_runner, pid_hash = build_runner(plant, False, True, request_gain_table=table)
    assert pid_hash == old['pid_block_sha256']
    runners = dict(blue_reference=reference_runner,
        fitted_capture=runner_for(selected['capture_cap_g'], selected['capture_duration_s']),
        early_capture=runner_for(early_best['capture_cap_g'], early_best['capture_duration_s']))
    results, summaries = {}, {}
    for name, runner in runners.items():
        result = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)
        enrich(result.samples, p)
        summary = analyze(result, obs)
        if name != 'blue_reference':
            switches = [i for i in range(1, len(result.samples))
                        if result.samples[i-1]['capture_active'] and not result.samples[i]['capture_active']]
            assert len(switches) == 1
            i = switches[0]
            summary['capture_exit_time_s'] = result.samples[i]['time_s']
            summary['handover_samples'] = result.samples[i-1:i+2]
        summary.update(measures(result, obs))
        results[name], summaries[name] = result, summary

    # Final focused verification: old blue and the zero-duration PN limit,
    # exact guidance equation, fixed PID/plant, and integration convergence.
    with old_trace.open() as stream:
        old_rows = list(csv.DictReader(stream, delimiter='\t'))
    reproduction = {}
    for name, case, result in (
        ('blue', 'blue_capture_reference', results['blue_reference']),
        ('zero_duration', 'pn_only', runner_for(selected['capture_cap_g'], 0.)(
            config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)),
    ):
        previous = [r for r in old_rows if r['case'] == case]
        assert len(previous) == len(result.samples)
        reproduction[name] = {field:max(abs(float(a[field])-b[field])
            for a,b in zip(previous, result.samples)) for field in FIELDS}
        assert max(reproduction[name].values()) < 1e-9, reproduction
    rows = results['fitted_capture'].samples
    cap, duration = selected['capture_cap_g'], selected['capture_duration_s']
    residuals = []
    for row in rows[:-1]:  # Terminal telemetry holds the preceding command.
        t = row['time_s']
        capturing = .3-1e-10 <= t < .3+duration-1e-10
        assert row['capture_active'] == capturing
        request = float(np.clip(row['raw_capture_request_g'], -cap, cap)) if capturing else row['raw_pn_request_g']
        request = float(np.clip(request*np.interp(t, *np.array(table).T), -cap_max, cap_max))
        residuals.append(abs(request-row['commanded_horizontal_acceleration_g']))
        assert abs(row['horizontal_fin_deflection_rad']
                   -np.clip(row['pid_raw_output'], -1., 1.)*raw_limit) < 1e-12
    assert max(residuals) < 1e-10, max(residuals)
    for field in ('commanded_horizontal_acceleration_g', 'horizontal_fin_deflection_rad',
                  'pid_proportional_output', 'pid_integral_output', 'pid_derivative_output'):
        assert rows[-1][field] == rows[-2][field]
    fine = runners['fitted_capture'](config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
    assert fine.event == results['fitted_capture'].event and len(fine.samples) == len(rows)
    step_error = {field:max(abs(a[field]-b[field]) for a,b in zip(rows, fine.samples)) for field in FIELDS}
    assert max(step_error.values()) < .01, step_error
    early_rows = results['early_capture'].samples
    early_fine = runners['early_capture'](config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
    assert early_fine.event == results['early_capture'].event and len(early_fine.samples) == len(early_rows)
    early_step_error = {field:max(abs(a[field]-b[field]) for a,b in zip(early_rows, early_fine.samples))
                        for field in FIELDS}
    assert max(early_step_error.values()) < .01, early_step_error
    assert protected_hashes() == protected
    assert asdict(p) == old['fixed_parameters'] and digest(config) == old['configuration_sha256']
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == h for path,h in extras.items())

    prefix = ROOT/'outputs/r77_1_capture_fit'
    report = dict(source_version=VERSION, source_commit=COMMIT, fixed_parameters=asdict(p),
        fitted_parameters=['capture_cap_g', 'capture_duration_s'],
        strategy='clip existing capture request at Gcap from 0.30 to 0.30+Tcap, then direct PN',
        raw_time_to_gain_table=table, target_information='ideal continuous TRK as requested',
        fit_window_s=[.5, 3.5], early_diagnostic_window_s=[.5, 2.1],
        objective='minimum maximum absolute AoA error on native replay times; no state replay or time shift',
        unfitted_observations=['overload_g','speed_kmh'],
        search_bounds=dict(capture_cap_g=[0., cap_max], capture_duration_s=[.02, 3.2]),
        search_resolution=dict(coarse_cap_g=5., coarse_duration_s=.04,
                               refined_cap_g=.25, refined_duration_s=.02),
        evaluated_pairs=len(cache), selected=selected, best_early_window=early_best,
        plotted_candidate='early_capture',
        early_duration_identified=False,
        early_duration_note='At the selected ceiling, T >= 1.80 s gives the same states through 2.10 s; shortest minimizer shown.',
        native_shape_matching_pairs=len(shape_candidates), best_native_shape_match=shape_best,
        shape_definition='AoA(0.7) < AoA(0.9) > AoA(1.1) < AoA(1.3)',
        search_is_bounded_not_a_uniqueness_proof=True,
        pid_block_sha256=old['pid_block_sha256'], plant_sha256=extras[plant_path],
        configuration_sha256=digest(config), replay_sha256=old['replay_sha256'],
        results=summaries, baseline_reproduction=reproduction,
        request_equation_max_residual_g=max(residuals), half_step_max_difference=step_error,
        early_half_step_max_difference=early_step_error,
        plant_and_pid_unchanged=True, protected_inputs_unchanged=True, runtime_defaults_changed=False)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_name(prefix.name+'_scan.tsv'), list(cache.values()))
    write_table(prefix.with_suffix('.tsv'), [dict(case=name, **r)
        for name, summary in summaries.items() for r in summary['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), [dict(case=name, **r)
        for name, result in results.items() for r in result.samples])
    plot(results, obs, early_best, prefix)
    print('verified', json.dumps(dict(step_error=step_error, request_residual=max(residuals))), flush=True)
    print(prefix.with_suffix('.png'), flush=True)


if __name__ == '__main__':
    main()
