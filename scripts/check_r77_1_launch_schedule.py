"""Remove the assumed 0.5 s controller gate with the fitted plant frozen.

Compare the historical gate, immediate control, and a request-gain candidate
using the raw 2.57 timeToGain knots. The table's location on the request path
is a hypothesis; a zero request gain is not treated as a locked control fin.
"""
from dataclasses import asdict, replace
import csv
import hashlib
import json

import numpy as np

from check_r77_1_capture_strategy import (
    ROOT, FocusPlant, inputs, CASES, protected_hashes, load_observations,
    VERSION, COMMIT, FIELDS, digest, build_runner, enrich, analyze, write_table,
)


def plot(results, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'old_0p5_gate': ('#777777', '--', 'Previous: PID held off until 0.5 s'),
        'immediate_control': ('#c17b28', '-', 'PID and capture active from launch'),
        'blkx_request_gain': ('#137e99', '-', 'PID active; raw timeToGain on request'),
    }
    panels = (
        ('alpha_deg', 'AoA [deg]', np.degrees(obs['alpha'])),
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', None),
        ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
        ('horizontal_fin_deflection_rad', 'Equivalent control angle [deg]', None),
        ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6),
        ('pid_integral_output', 'PID integral output', None),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5))
    for ax, (field, label, measured) in zip(axes.flat, panels):
        if measured is not None:
            mask = obs['time'] <= 2.1
            ax.plot(obs['time'][mask], measured[mask], 'o-', color='#222',
                    lw=1., ms=4.5, label='2.57 replay', zorder=5)
        for name, (color, ls, title) in styles.items():
            rows = [r for r in results[name].samples if r['time_s'] <= 2.1]
            values = np.array([r[field] for r in rows])
            if field == 'horizontal_fin_deflection_rad':
                values = np.degrees(values)
            ax.plot([r['time_s'] for r in rows], values, color=color, ls=ls,
                    lw=1.7, label=title)
        ax.set(xlim=(0., 2.1), xlabel='Missile flight time [s]', ylabel=label)
        ax.axvspan(.3, .31, color='#b3d4da', alpha=.6, zorder=0)
        ax.axvline(.5, color='#999', lw=.8, ls=':', alpha=.7)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .952),
               ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle('R-77-1 | launch timing only | identical plant and PID gains', fontsize=16, y=.985)
    fig.text(.08, .025,
        'Frozen 2.57.1.107; A_delta = 2.73404 m²/rad. Same plant, initial state, capture and continuous handover; no refit.\n'
        'Raw timeToGain: (0.30 s, 0), (0.31 s, 1), linearly interpolated on the acceleration request before its cap.\n'
        'Request-gain placement is a candidate interpretation. All runs use the original 20 ms PID period; no time shift.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.13, hspace=.32, wspace=.24)
    fig.savefig(prefix.with_suffix('.png'), dpi=160, facecolor='white')
    plt.close(fig)


def main():
    protected = protected_hashes()
    previous_path = ROOT/'outputs/r77_1_capture_strategy.json'
    previous_trace = ROOT/'outputs/r77_1_capture_strategy_trajectory.tsv'
    raw_path = ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx'
    plant_path = ROOT/'scripts/run_r77_1_focus_moment.py'
    extra = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (previous_path, previous_trace, raw_path, plant_path)}
    previous = json.loads(previous_path.read_text())
    raw = json.loads(raw_path.read_text())['rocket']
    autopilot = raw['guidance']['guidanceAutopilot']
    knots = [autopilot[key] for key in sorted(
        (key for key in autopilot if key.startswith('timeToGain')),
        key=lambda key: int(key.removeprefix('timeToGain')))]
    assert len(knots) >= 2 and np.isfinite(knots).all()
    assert np.all(np.diff(np.array(knots)[:, 0]) > 0)
    assert knots == [[.3, 0.], [.31, 1.]]
    config, baseline, raw_limit = inputs()
    baseline = replace(baseline, fin_force_area_per_rad=previous['fixed_parameters']['fin_force_area_per_rad'])
    assert asdict(baseline) == previous['fixed_parameters']
    assert digest(config) == previous['configuration_sha256']
    assert (previous['source_version'], previous['source_commit']) == (VERSION, COMMIT)
    _, obs = load_observations()
    assert protected[str(ROOT/'data/replays/r77_1_level_20260824.tsv')] == previous['replay_sha256']
    variants = {'old_0p5_gate': (.5, ()), 'immediate_control': (0., ()),
                'blkx_request_gain': (0., tuple(tuple(k) for k in knots))}
    results, summaries, runners, parameters, pid_hashes = {}, {}, {}, {}, set()
    for name, (delay, table) in variants.items():
        p = replace(baseline, guidance_delay_s=delay)
        assert {k:v for k,v in asdict(p).items() if k != 'guidance_delay_s'} == {
            k:v for k,v in asdict(baseline).items() if k != 'guidance_delay_s'}
        plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
        runner, pid_hash = build_runner(plant, pn_capture=False, retain_command=True,
                                       request_gain_table=table)
        pid_hashes.add(pid_hash)
        parameters[name], runners[name] = p, runner
        results[name] = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)
        rows = results[name].samples
        enrich(rows, p)
        for row in rows:
            row['request_time_gain'] = float(np.interp(row['time_s'], *np.array(knots).T)) if table else 1.
        summaries[name] = analyze(results[name], obs)
        summaries[name]['first_nonzero_control_s'] = next(
            (r['time_s'] for r in rows if abs(r['horizontal_fin_deflection_rad']) > 1e-12), None)
        summaries[name]['first_positive_request_s'] = next(
            (r['time_s'] for r in rows if r['commanded_horizontal_acceleration_g'] > 1e-12), None)
        print(name, json.dumps({k:summaries[name][k] for k in (
            'first_nonzero_control_s', 'first_positive_request_s', 'capture_exit_time_s',
            'alpha_at_native_times_deg', 'early_topology_matches', 'extrema')}), flush=True)

    # Focused verification of unchanged dynamics, historical receipt and launch semantics.
    assert pid_hashes == {previous['pid_block_sha256']}
    with previous_trace.open() as f:
        old = [r for r in csv.DictReader(f, delimiter='\t') if r['case'] == 'continuous_handover']
    assert len(old) == len(results['old_0p5_gate'].samples)
    reproduction = {field: max(abs(float(a[field])-b[field])
        for a,b in zip(old, results['old_0p5_gate'].samples)) for field in FIELDS}
    assert max(reproduction.values()) < 1e-9
    assert summaries['immediate_control']['first_nonzero_control_s'] == 0.
    assert summaries['blkx_request_gain']['first_nonzero_control_s'] == 0.
    assert summaries['blkx_request_gain']['first_positive_request_s'] == .32
    assert all(abs(r['commanded_horizontal_acceleration_g']) < 1e-12
               for r in results['blkx_request_gain'].samples if r['time_s'] <= .3)
    verification = {}
    for name, result in results.items():
        fine = runners[name](config, parameters[name], CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
        assert result.event == fine.event and len(result.samples) == len(fine.samples)
        assert abs(result.samples[-1]['time_s']-3.5) < 1e-9
        errors = {field:max(abs(a[field]-b[field]) for a,b in zip(result.samples,fine.samples))
                  for field in FIELDS}
        assert max(errors.values()) < .01, (name, errors)
        assert [r['capture_active'] for r in result.samples] == [r['capture_active'] for r in fine.samples]
        verification[name] = errors
    assert digest(config) == previous['configuration_sha256']
    assert protected_hashes() == protected
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in extra.items())
    prefix = ROOT/'outputs/r77_1_launch_schedule'
    prefix.with_suffix('.json').write_text(json.dumps(dict(
        source_version=VERSION, source_commit=COMMIT, source_sha256=extra[raw_path],
        raw_time_to_gain_table=knots, source_field='rocket.guidance.guidanceAutopilot.timeToGain*',
        fixed_parameters=asdict(baseline), variant_guidance_delays_s={k:v[0] for k,v in variants.items()},
        pid_block_sha256=next(iter(pid_hashes)), plant_sha256=extra[plant_path],
        configuration_sha256=digest(config), replay_sha256=previous['replay_sha256'],
        table_interpretation='candidate: multiply raw acceleration request before cap; PID active from launch',
        table_interpretation_is_verified_game_semantics=False, fitted_parameters=[],
        results=summaries, baseline_reproduction_max_difference=reproduction,
        half_step_max_difference=verification, plant_and_pid_unchanged=True,
        protected_inputs_unchanged=True, runtime_defaults_changed=False), indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'), [dict(case=name, **r)
        for name,s in summaries.items() for r in s['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), [dict(case=name, **r)
        for name,result in results.items() for r in result.samples])
    plot(results, obs, prefix)
    print(json.dumps(dict(baseline_reproduction=reproduction, verification=verification)), flush=True)
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
