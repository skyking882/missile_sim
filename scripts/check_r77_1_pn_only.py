"""Blue launch-table candidate with all manual capture/handover bypassed.

Use ideal continuously available target position and velocity as requested.
Only the outer request path changes; preserve the fitted plant, PID, raw
launch gain table, initial state, control period and translational dynamics.
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


def plot(results, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = (
        ('alpha_deg', 'AoA [deg]', np.degrees(obs['alpha'])),
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', None),
        ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
        ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6),
    )
    styles = {
        'blue_capture_reference': ('#137e99', '--', 'Previous blue: launch table + manual capture'),
        'pn_only': ('#c47b26', '-', 'PN only: N = 4, same launch table'),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for ax, (field, label, measured) in zip(axes.flat, panels):
        if measured is not None:
            ax.plot(obs['time'], measured, 'o-', color='#222', lw=1., ms=4.5,
                    label='2.57 replay', zorder=5)
        for name, (color, ls, title) in styles.items():
            rows = results[name].samples
            ax.plot([r['time_s'] for r in rows], [r[field] for r in rows],
                    color=color, ls=ls, lw=1.8, label=title)
        ax.set(xlim=(0., 3.5), xlabel='Missile flight time [s]', ylabel=label)
        ax.axvspan(.9, 1.1, color='#e5e9ed', alpha=.45, zorder=0)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .954),
               ncol=1, frameon=False, fontsize=10)
    fig.suptitle('R-77-1 | PN only with the blue candidate\'s launch timing', fontsize=16, y=.989)
    fig.text(.08, .025,
        'Frozen 2.57.1.107; A_delta = 2.73404 m²/rad. Plant, PID and initial state unchanged; no refit.\n'
        'PN uses continuous target information (TRK assumption). Manual capture, 15° exit and 2.5 s release are bypassed.\n'
        'Raw timeToGain: (0.30 s, 0), (0.31 s, 1), applied to the request. PID active from launch. No time shift.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.15, hspace=.3, wspace=.24)
    fig.savefig(prefix.with_suffix('.png'), dpi=165, facecolor='white')
    plt.close(fig)


def main():
    protected = protected_hashes()
    previous_path = ROOT/'outputs/r77_1_launch_schedule.json'
    previous_trace = ROOT/'outputs/r77_1_launch_schedule_trajectory.tsv'
    raw_path = ROOT/'data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx'
    plant_path = ROOT/'scripts/run_r77_1_focus_moment.py'
    extra = {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (previous_path, previous_trace, raw_path, plant_path)}
    previous = json.loads(previous_path.read_text())
    autopilot = json.loads(raw_path.read_text())['rocket']['guidance']['guidanceAutopilot']
    table = tuple(tuple(autopilot[key]) for key in sorted(
        (k for k in autopilot if k.startswith('timeToGain')),
        key=lambda k: int(k.removeprefix('timeToGain'))))
    assert [list(k) for k in table] == previous['raw_time_to_gain_table']
    config, base, raw_limit = inputs()
    base = replace(base, fin_force_area_per_rad=previous['fixed_parameters']['fin_force_area_per_rad'])
    assert asdict(base) == previous['fixed_parameters']
    p = replace(base, guidance_delay_s=previous['variant_guidance_delays_s']['blkx_request_gain'])
    assert p.guidance_delay_s == 0. and p.navigation_gain == autopilot['propNavMult'] == 4.
    assert digest(config) == previous['configuration_sha256']
    assert (previous['source_version'], previous['source_commit']) == (VERSION, COMMIT)
    assert extra[plant_path] == previous['plant_sha256']
    _, obs = load_observations()
    assert protected[str(ROOT/'data/replays/r77_1_level_20260824.tsv')] == previous['replay_sha256']
    plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    results, summaries, runners, pid_hashes = {}, {}, {}, set()
    for name, pure in (('blue_capture_reference', False), ('pn_only', True)):
        runner, pid_hash = build_runner(plant, pn_capture=False, retain_command=not pure,
                                       request_gain_table=table, pn_only=pure)
        runners[name] = runner
        pid_hashes.add(pid_hash)
        result = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)
        enrich(result.samples, p)
        for row in result.samples:
            row['request_time_gain'] = float(np.interp(row['time_s'], *np.array(table).T))
        results[name] = result
        summaries[name] = analyze(result, obs)
        if pure:
            summaries[name]['capture_exit_time_s'] = None
            summaries[name]['handover_samples'] = []
        print(name, json.dumps({k:summaries[name][k] for k in (
            'alpha_at_native_times_deg', 'alpha_change_0p9_to_1p1_deg',
            'early_topology_matches', 'capture_exit_time_s', 'extrema')}), flush=True)

    # Final verification: reproduce blue, verify the complete PN request, and retain dynamics.
    assert pid_hashes == {previous['pid_block_sha256']}
    with previous_trace.open() as f:
        old = [r for r in csv.DictReader(f, delimiter='\t') if r['case'] == 'blkx_request_gain']
    assert len(old) == len(results['blue_capture_reference'].samples)
    reproduction = {field:max(abs(float(a[field])-b[field])
        for a,b in zip(old, results['blue_capture_reference'].samples)) for field in FIELDS}
    assert max(reproduction.values()) < 1e-9, reproduction
    cap = math.sqrt(p.command_limit_g**2-1.)
    rows = results['pn_only'].samples
    # The terminal 3.5 s telemetry row reuses the 3.48 s controller output.
    # Only the preceding rows are fresh command evaluations.
    control_rows = rows[:-1]
    expected = [float(np.clip(r['request_time_gain']*r['raw_pn_request_g'], -cap, cap))
                for r in control_rows]
    pn_residual = max(abs(r['commanded_horizontal_acceleration_g']-target)
                      for r,target in zip(control_rows, expected))
    assert pn_residual < 1e-10, pn_residual
    assert all(rows[-1][field] == rows[-2][field] for field in (
        'commanded_horizontal_acceleration_g', 'horizontal_fin_deflection_rad',
        'pid_proportional_output', 'pid_integral_output', 'pid_derivative_output'))
    assert not any(r['capture_active'] for r in rows)
    assert next(r['time_s'] for r in rows if r['commanded_horizontal_acceleration_g'] > 1e-12) == .32
    verification = {}
    for name, result in results.items():
        fine = runners[name](config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
        assert result.event == fine.event and len(result.samples) == len(fine.samples)
        assert abs(result.samples[-1]['time_s']-3.5) < 1e-9
        errors = {field:max(abs(a[field]-b[field]) for a,b in zip(result.samples, fine.samples))
                  for field in FIELDS}
        assert max(errors.values()) < .01, (name, errors)
        verification[name] = errors
    assert protected_hashes() == protected
    assert digest(config) == previous['configuration_sha256']
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == h for path,h in extra.items())
    prefix = ROOT/'outputs/r77_1_pn_only'
    prefix.with_suffix('.json').write_text(json.dumps(dict(
        source_version=VERSION, source_commit=COMMIT, fixed_parameters=asdict(p),
        raw_time_to_gain_table=table, target_information='ideal continuous TRK as requested',
        pn_request='clip(timeGain(t)*4*max(closing_speed,0)*LOS_rate, +/-horizontal_reqAccelMax)',
        manual_capture_enabled=False, capture_exit_enabled=False, release_filter_enabled=False,
        unused_parameters_in_pn_only=['capture_time_constant_s','release_time_constant_s','release_angle_deg'],
        table_placement_is_verified_game_semantics=False, fitted_parameters=[],
        pid_block_sha256=next(iter(pid_hashes)), plant_sha256=extra[plant_path],
        configuration_sha256=digest(config), replay_sha256=previous['replay_sha256'],
        results=summaries, baseline_reproduction_max_difference=reproduction,
        pn_request_max_residual_g=pn_residual,
        pn_request_verified_control_rows=len(control_rows),
        terminal_sample_reuses_last_controller_output=True, half_step_max_difference=verification,
        plant_and_pid_unchanged=True, protected_inputs_unchanged=True,
        runtime_defaults_changed=False), indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'), [dict(case=name, **r)
        for name,s in summaries.items() for r in s['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), [dict(case=name, **r)
        for name,result in results.items() for r in result.samples])
    plot(results, obs, prefix)
    print(json.dumps(dict(baseline_reproduction=reproduction, pn_request_residual_g=pn_residual,
                         verification=verification)), flush=True)
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
