"""Change only capture end to the first recorded TRK frame; no refitting."""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math

import numpy as np

from run_r77_1_current_full_forward import (
    ROOT, FocusPlant, inputs, protected_hashes, VERSION, COMMIT,
    FIELDS, digest, build_runner, enrich, write_table, sha,
)


def read_trace(path):
    with path.open() as stream:
        return [{k: (v == 'True' if v in ('True', 'False') else float(v))
                 for k, v in r.items()} for r in csv.DictReader(stream, delimiter='\t')]


def plot(old, new, observed, report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    teal, orange = '#167f99', '#c77c27'
    end = max(old[-1]['time_s'], new[-1]['time_s'], observed[-1]['time_s'])
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), sharex='col')
    fields = [('alpha_deg', 'AoA [deg]'),
              ('wind_normal_specific_force_g', 'Normal load [g]'),
              ('speed_kmh', 'Speed [km/h]')]
    for j, (field, label) in enumerate(fields):
        for col in range(2):
            ax = axes[j, col]
            ax.plot([r['time_s'] for r in observed], [r[field] for r in observed],
                    'o-', color='#252525', ms=3.4, lw=1., zorder=5)
            for rows, color, style in [(old, orange, '--'), (new, teal, '-')]:
                ax.plot([r['time_s'] for r in rows], [r[field] for r in rows],
                        color=color, ls=style, lw=2.)
            ax.axvspan(report['last_non_trk_time_s'], report['capture_end_s'],
                       color=teal, alpha=.08, zorder=0)
            ax.axvline(report['capture_end_s'], color=teal, ls=':', lw=1.)
            ax.axvline(report['baseline_capture_end_s'], color=orange, ls=':', lw=1.)
            ax.set(xlim=(0, end+.1 if col == 0 else 3.5), ylabel=label)
            if col == 1:
                values = [r[field] for rows in (old, new, observed) for r in rows if r['time_s'] <= 3.5]
                lo, hi = min(values), max(values)
                ax.set_ylim(lo-.07*(hi-lo), hi+.09*(hi-lo))
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 2:
                ax.set_xlabel('Missile flight time [s]')
    axes[0, 0].set_title('Complete forward runs', fontsize=12)
    axes[0, 1].set_title('Early-flight detail', fontsize=12)
    fig.suptitle('R-77-1 | capture ends at first observed TRK: 0.90 s', fontsize=16, y=.985)
    fig.legend(handles=[
        Line2D([], [], color='#252525', marker='o', lw=1., label='Game measurements'),
        Line2D([], [], color=orange, ls='--', lw=2., label='Previous: capture until 2.10 s'),
        Line2D([], [], color=teal, lw=2., label='This test: capture until 0.90 s'),
    ], loc='upper center', bbox_to_anchor=(.5, .948), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.07, .028,
        'Only capture duration changes: 1.80 s -> 0.60 s; same start at 0.30 s and cap at 35.25 g. Then PN N = 4.\n'
        '2.57.1.107; target course 180 deg; identical PID, plant, initial state and integration. No state injection or refit.\n'
        'Shading marks the sampled transition bracket: non-TRK at 0.70 s, first TRK at 0.90 s. Seeker acquisition is not simulated.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.07, right=.98, top=.865, bottom=.13, hspace=.20, wspace=.22)
    fig.savefig(prefix.with_suffix('.png'), dpi=170, facecolor='white')
    plt.close(fig)


def main():
    baseline_path = ROOT/'outputs/r77_1_current_full_forward.json'
    baseline_trace = ROOT/'outputs/r77_1_current_full_forward_trajectory.tsv'
    replay_path = ROOT/'data/replays/r77_1_level_20260824.tsv'
    baseline = json.loads(baseline_path.read_text())
    protected = protected_hashes() | {str(p): sha(p) for p in (
        baseline_path, baseline_trace, ROOT/'scripts/run_r77_1_focus_moment.py',
        ROOT/'scripts/check_r77_1_capture_strategy.py', ROOT/'scripts/run_r77_1_current_full_forward.py')}
    assert all(sha(ROOT/path) == expected for path, expected in baseline['protected_input_hashes'].items())
    config, p, raw_limit = inputs()
    p = replace(p, fin_force_area_per_rad=baseline['fixed_parameters']['fin_force_area_per_rad'],
                guidance_delay_s=baseline['fixed_parameters']['guidance_delay_s'])
    assert asdict(p) == baseline['fixed_parameters']
    assert config == baseline['fixed_configuration']
    assert digest(config) == baseline['configuration_sha256']
    assert (VERSION, COMMIT) == (baseline['source_version'], baseline['source_commit'])
    with replay_path.open() as stream:
        native = list(csv.DictReader(stream, delimiter='\t'))
    first_index = next(i for i, r in enumerate(native) if r['seeker_state'].strip() == 'TRK')
    first, previous = native[first_index], native[first_index-1]
    trk_time = float(first['missile_flight_time_s'])
    table = tuple(tuple(r) for r in baseline['raw_time_to_gain_table'])
    duration = trk_time-baseline['capture_start_s']
    assert duration > 0.
    plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    runner, pid_hash = build_runner(plant, pn_capture=False, retain_command=False,
        request_gain_table=table, timed_capture=(baseline['capture_cap_g'], duration))
    assert pid_hash == baseline['pid_block_sha256']
    result = runner(config, p, baseline['scenario'], max_time_s=config['performance']['lifetime_s'],
                    plant_dt_s=.002, control_dt_s=.02)
    rows = result.samples
    enrich(rows, p, baseline['scenario'])
    old = read_trace(baseline_trace)
    observed = [dict(time_s=float(r['missile_flight_time_s']),
        alpha_deg=float(r['angle_of_attack_deg']),
        wind_normal_specific_force_g=float(r['overload_g']), speed_kmh=float(r['speed_kmh'])) for r in native]
    fields = ('alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh')
    comparison = []
    for obs in observed:
        item = dict(time_s=obs['time_s'])
        for field in fields:
            item[field+'_measured'] = obs[field]
            for name, trace in [('baseline', old), ('trk', rows)]:
                item[field+'_'+name] = float(np.interp(obs['time_s'],
                    [r['time_s'] for r in trace], [r[field] for r in trace])) if obs['time_s'] <= trace[-1]['time_s'] else None
        comparison.append(item)

    # One focused pass: identical pre-switch evolution and continuous state,
    # exact switch to PN, unchanged inputs and complete terminal integration.
    shared = [(a, b) for a, b in zip(old, rows) if a['time_s'] < trk_time-1e-10]
    pre_error = {f:max(abs(a[f]-b[f]) for a, b in shared) for f in FIELDS}
    assert max(pre_error.values()) == 0.
    switch = next(r for r in rows if abs(r['time_s']-trk_time) < 1e-10)
    old_switch = next(r for r in old if abs(r['time_s']-trk_time) < 1e-10)
    assert all(switch[f] == old_switch[f] for f in (*fields, 'body_rate_rad_s', 'x_m', 'z_m'))
    assert all(not r['capture_active'] for r in rows if r['time_s'] >= trk_time-1e-10)
    request_limit = math.sqrt(p.command_limit_g**2-1.)
    pn_residual = max(abs(r['commanded_horizontal_acceleration_g']-
        np.clip(r['raw_pn_request_g'], -request_limit, request_limit))
        for r in rows[:-1] if r['time_s'] >= trk_time-1e-10)
    assert pn_residual < 1e-10
    assert result.event in ('fuse', 'closest_approach', 'lifetime')
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
    assert all(sha(ROOT/path) == expected for path, expected in protected.items())
    prefix = ROOT/'outputs/r77_1_capture_until_trk'
    report = dict(source_version=VERSION, source_commit=COMMIT,
        fixed_parameters=asdict(p), configuration_sha256=digest(config), pid_block_sha256=pid_hash,
        scenario=baseline['scenario'], first_trk_frame=first, preceding_frame=previous,
        last_non_trk_time_s=float(previous['missile_flight_time_s']), capture_end_s=trk_time,
        baseline_capture_end_s=baseline['capture_end_s'], capture_duration_s=duration,
        capture_start_s=baseline['capture_start_s'], capture_cap_g=baseline['capture_cap_g'],
        changed_parameters=['capture_duration_s'], fitted_parameters=[],
        event=result.event, terminal_time_s=rows[-1]['time_s'],
        minimum_distance_m=result.minimum_distance_m, terminal_state=rows[-1],
        baseline_terminal_time_s=baseline['terminal_time_s'],
        native_replay_frame_count=len(native),
        compared_frame_count=sum(r['time_s'] <= rows[-1]['time_s'] for r in observed),
        comparison=comparison,
        handover=dict(before=rows[rows.index(switch)-1], after=switch,
                      baseline_same_time=old_switch),
        verification=dict(pre_switch_max_difference=pre_error, state_continuous_at_switch=True,
            pn_request_max_residual_g=pn_residual, protected_inputs_unchanged=True),
        measured_input_use='First observed TRK time sets a fixed schedule; no measured kinematic state is injected.',
        seeker_acquisition_modelled=False, protected_input_hashes=protected)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), rows)
    write_table(prefix.with_name(prefix.name+'_comparison.tsv'), comparison)
    plot(old, rows, observed, report, prefix)
    print(json.dumps({k:report[k] for k in ('capture_end_s', 'capture_duration_s', 'event',
        'terminal_time_s', 'minimum_distance_m', 'compared_frame_count', 'verification')}, indent=2))
    for r in comparison:
        if r['time_s'] in (.9, 1.1, 1.3, 1.5, 2.1, 3.5, 6., 7.8):
            print(json.dumps(r))
    print('handover requests', report['handover']['before']['commanded_horizontal_acceleration_g'],
          switch['commanded_horizontal_acceleration_g'])


if __name__ == '__main__':
    main()
