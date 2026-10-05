"""Run the current frozen R-77-1 candidate from launch to its terminal event."""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import math

import numpy as np

from check_r77_1_capture_strategy import (
    ROOT, FocusPlant, inputs, protected_hashes, VERSION, COMMIT,
    FIELDS, digest, build_runner, enrich, write_table,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figures(rows, measured, report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color = '#c77c27'
    time = np.array([r['time_s'] for r in rows])
    end = time[-1]
    right = max(end, measured[-1]['time_s']) + .15
    capture_end = report['capture_end_s']
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), sharex='col')
    fields = [('alpha_deg', 'AoA [deg]'),
              ('wind_normal_specific_force_g', 'Normal load [g]'),
              ('speed_kmh', 'Speed [km/h]')]
    for row_index, (field, label) in enumerate(fields):
        for col in range(2):
            ax = axes[row_index, col]
            ax.plot([r['time_s'] for r in measured], [r[field] for r in measured],
                    'o-', color='#252525', ms=3.4, lw=1., zorder=4)
            ax.plot(time, [r[field] for r in rows], color=color, lw=2.)
            ax.axvline(capture_end, color='#9a9a9a', ls=':', lw=1.)
            if col == 0:
                ax.axvline(7., color='#708b9e', ls='--', lw=.9)
                ax.axvline(end, color=color, ls=':', lw=1.)
            ax.set(xlim=(0., right if col == 0 else 3.5), ylabel=label)
            if col == 1:
                early = [r[field] for r in rows if r['time_s'] <= 3.5]
                early += [r[field] for r in measured if r['time_s'] <= 3.5]
                lo, hi = min(early), max(early)
                ax.set_ylim(lo - .07*(hi-lo), hi + .09*(hi-lo))
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            if row_index == 2:
                ax.set_xlabel('Missile flight time [s]')
    axes[0, 0].set_title('Complete forward run', fontsize=12)
    axes[0, 1].set_title('Early-flight detail: same run', fontsize=12)
    fig.suptitle('R-77-1 | continuous forward evaluation | frozen 2.57.1.107',
                 fontsize=16, y=.986)
    fig.legend(handles=[
        Line2D([], [], color='#252525', marker='o', lw=1., label='Game measurements'),
        Line2D([], [], color=color, lw=2., label='Current candidate: capture then PN'),
        Line2D([], [], color='#9a9a9a', ls=':', label='Capture ends: 2.10 s'),
        Line2D([], [], color='#708b9e', ls='--', label='Motor cutoff: 7.00 s'),
    ], loc='upper center', bbox_to_anchor=(.5, .951), ncol=2, frameon=False, fontsize=10)
    fig.text(.07, .024,
        f'Launch at t = 0; no measured-state injection or time shift. Terminal event: {report["event"]} at {end:.3f} s.\n'
        'Capture cap 35.25 g for 1.80 s from t = 0.30 s; then PN N = 4. PID, plant and coefficients unchanged.\n'
        'Initial horizontal AoA 11 deg and zero body rate retained. Level-flight model at 6300 m; measurements only overlaid afterwards.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.125, hspace=.20, wspace=.22)
    fig.savefig(prefix.with_suffix('.png'), dpi=170, facecolor='white')
    plt.close(fig)

    fig, (ax, distance_ax) = plt.subplots(1, 2, figsize=(12.8, 5.5))
    x = np.array([r['x_m'] for r in rows])/1000
    z = np.array([r['z_m'] for r in rows])/1000
    tx = np.array([r['target_x_m'] for r in rows])/1000
    tz = np.array([r['target_z_m'] for r in rows])/1000
    ax.plot(x, z, color=color, lw=2., label='Missile prediction')
    ax.plot(tx, tz, color='#397c96', lw=2., label='Target: course 180 deg')
    ax.scatter([x[0], tx[0]], [z[0], tz[0]], c=[color, '#397c96'], marker='s', s=40)
    ax.scatter([x[-1]], [z[-1]], color=color, marker='*', s=120, zorder=4)
    for sec in (2., 4., 6.):
        if sec < end:
            px, pz = np.interp(sec, time, x), np.interp(sec, time, z)
            ax.plot(px, pz, 'o', color=color, ms=4)
            ax.annotate(f'{sec:g} s', (px, pz), xytext=(5, -13), textcoords='offset points', fontsize=9)
    ax.set(xlabel='Along initial missile heading [km]', ylabel='Horizontal cross-track [km]',
           title='Horizontal plane: simulated positions')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=9, loc='lower right', frameon=False)
    distance_ax.plot(time, [r['distance_m']/1000 for r in rows], color=color, lw=2.)
    distance_ax.set(xlabel='Missile flight time [s]', ylabel='Missile-target distance [km]',
                    xlim=(0, end+.15), title='Separation from the same forward run')
    distance_ax.text(.04, .92,
        f'{report["event"]} at {end:.3f} s\nTerminal separation: {rows[-1]["distance_m"]:.2f} m\n'
        f'Terminal speed: {rows[-1]["speed_kmh"]:.0f} km/h',
        transform=distance_ax.transAxes, va='top', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=5.))
    for a in (ax, distance_ax):
        a.grid(alpha=.18)
        a.spines[['top', 'right']].set_visible(False)
    fig.suptitle('R-77-1 | launch 0 deg; target course 180 deg', fontsize=15, y=.98)
    fig.text(.075, .025,
        'Initial separation 8 km at 30 deg bearing; both start at 1200 km/h. Fixed altitude 6300 m.\n'
        'No measured x/y trajectory is available here. Replay displayed camera distance is not used as target range.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.18, wspace=.24)
    fig.savefig(prefix.with_name(prefix.name+'_path.png'), dpi=170, facecolor='white')
    plt.close(fig)


def main():
    baseline_path = ROOT/'outputs/r77_1_target_course_corrected.json'
    baseline_trace = ROOT/'outputs/r77_1_target_course_corrected_trajectory.tsv'
    replay_path = ROOT/'data/replays/r77_1_level_20260824.tsv'
    extra_paths = [baseline_path, baseline_trace, ROOT/'scripts/run_r77_1_focus_moment.py',
                   ROOT/'scripts/check_r77_1_capture_strategy.py']
    protected_before = protected_hashes() | {str(p): sha(p) for p in extra_paths}
    baseline = json.loads(baseline_path.read_text())
    config, p, raw_limit = inputs()
    p = replace(p, fin_force_area_per_rad=baseline['fixed_parameters']['fin_force_area_per_rad'],
                guidance_delay_s=baseline['fixed_parameters']['guidance_delay_s'])
    assert asdict(p) == baseline['fixed_parameters']
    assert digest(config) == baseline['configuration_sha256']
    assert (VERSION, COMMIT) == (baseline['source_version'], baseline['source_commit'])
    assert sha(extra_paths[2]) == baseline['plant_sha256']
    assert sha(replay_path) == baseline['replay_sha256']
    profile = json.loads((ROOT/'missiles/su_r_77_1.json').read_text())
    provenance = profile['provenance']
    assert (provenance['source_version'], provenance['source_commit']) == (VERSION, COMMIT)
    for name, expected in provenance['source_sha256'].items():
        assert sha(ROOT/'data/datamine'/name) == expected

    scenario = dict(baseline['scenarios']['confirmed_180'])
    capture = tuple(baseline['timed_capture_parameters'])
    table = tuple(tuple(row) for row in baseline['raw_time_to_gain_table'])
    plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    runner, pid_hash = build_runner(plant, pn_capture=False, retain_command=False,
                                   request_gain_table=table, timed_capture=capture)
    assert pid_hash == baseline['pid_block_sha256']

    # No replay arrays or restart state enter the forward integrator.
    result = runner(config, p, scenario, max_time_s=config['performance']['lifetime_s'],
                    plant_dt_s=.002, control_dt_s=.02)
    rows = result.samples
    enrich(rows, p, scenario)
    assert result.event != 'numerical_failure'
    end = rows[-1]['time_s']

    with replay_path.open() as stream:
        native = list(csv.DictReader(stream, delimiter='\t'))
    measured = [dict(time_s=float(r['missile_flight_time_s']),
                     alpha_deg=float(r['angle_of_attack_deg']),
                     wind_normal_specific_force_g=float(r['overload_g']),
                     speed_kmh=float(r['speed_kmh'])) for r in native]
    comparison = []
    fields = ('alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh')
    times = np.array([r['time_s'] for r in rows])
    for obs in measured:
        supported = times[0] <= obs['time_s'] <= end
        item = dict(time_s=obs['time_s'], within_forward_run=supported)
        for field in fields:
            predicted = float(np.interp(obs['time_s'], times, [r[field] for r in rows])) if supported else None
            item.update({field+'_measured': obs[field], field+'_predicted': predicted,
                         field+'_error': None if predicted is None else predicted-obs[field]})
        comparison.append(item)

    # Focused final verification: reproduce the earlier free trajectory exactly
    # and verify the newly extended run preserves all input/interface invariants.
    with baseline_trace.open() as stream:
        old = [r for r in csv.DictReader(stream, delimiter='\t')
               if r['case'] == 'confirmed_180_timed_capture']
    assert len(rows) >= len(old) and rows[0]['time_s'] == 0.
    # The old run stopped at 3.5 s before that control tick; the full run
    # continues and records the new command at 3.5 s. Compare held inputs
    # before that endpoint, while comparing physical states through it.
    reproduction = {field: max(abs(float(a[field])-b[field]) for a, b in zip(old[:-1], rows))
                    for field in FIELDS}
    assert max(reproduction.values()) < 1e-9, reproduction
    endpoint_state_difference = {field: abs(float(old[-1][field])-rows[len(old)-1][field])
                                 for field in fields}
    assert max(endpoint_state_difference.values()) < 1e-9
    assert all(b['time_s'] > a['time_s'] for a, b in zip(rows, rows[1:]))
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
    assert all(abs(r['target_z_m']-4000.) < 1e-9 for r in rows)
    mapping_error = max(abs(r['horizontal_fin_deflection_rad']-
                           np.clip(r['pid_raw_output'], -1., 1.)*raw_limit) for r in rows)
    assert mapping_error < 1e-12
    assert all(r['thrust_n'] == 0. for r in rows if r['time_s'] >= 7.)
    assert all(r['rate_damping_moment_nm']*r['body_rate_rad_s'] <= 1e-8 for r in rows)
    protected_after = protected_hashes() | {str(p): sha(p) for p in extra_paths}
    assert protected_after == protected_before

    prefix = ROOT/'outputs/r77_1_current_full_forward'
    report = dict(source_version=VERSION, source_commit=COMMIT,
        scenario=scenario, fixed_parameters=asdict(p), fixed_configuration=config,
        capture_cap_g=capture[0], capture_duration_s=capture[1],
        capture_start_s=table[0][0], capture_end_s=table[0][0]+capture[1],
        raw_time_to_gain_table=table, guidance='timed_capture_then_pn',
        event=result.event, terminal_time_s=end, minimum_distance_m=result.minimum_distance_m,
        terminal_state=rows[-1], native_replay_frame_count=len(measured),
        compared_frame_count=sum(r['within_forward_run'] for r in comparison),
        comparison=comparison,
        extrema={field: dict(time_s=max(rows, key=lambda r:r[field])['time_s'],
                            value=max(r[field] for r in rows)) for field in fields},
        replay_used_only_after_forward_integration=True, midflight_state_injection=False,
        fitted_parameters=[], initial_state_assumptions=dict(horizontal_aoa_deg=p.initial_alpha_deg,
            body_rate_rad_s=p.initial_body_rate_rad_s, pid_integral=0.),
        pid_block_sha256=pid_hash, configuration_sha256=digest(config),
        verification=dict(early_trajectory_reproduction_max_difference=reproduction,
            previous_endpoint_state_difference=endpoint_state_difference,
            fin_mapping_max_error_rad=mapping_error, protected_inputs_unchanged=True,
            post_burn_thrust_zero=True, no_extrapolation_past_terminal_event=True),
        protected_input_hashes=protected_before)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), rows)
    write_table(prefix.with_name(prefix.name+'_comparison.tsv'), comparison)
    figures(rows, measured, report, prefix)
    print(json.dumps({k: report[k] for k in ('event', 'terminal_time_s', 'minimum_distance_m',
          'compared_frame_count', 'native_replay_frame_count', 'extrema', 'verification')}, indent=2))
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
