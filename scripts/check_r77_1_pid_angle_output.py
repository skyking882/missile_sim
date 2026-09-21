"""Compare PID angle mappings on the frozen orange focal-moment plant.

No fit or runtime-default change. The main alternative changes only the final
output mapping. A separate diagnostic moves the existing anti-windup gate to
the new output limit, without changing gains, integral bounds or error units.
With --normalized-only, compare the historical pi/2 scale with the current
normalized-output-times-raw-radians mapping, retaining the same plant gains.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_r77_1_focus_moment import (
    ROOT, FocusPlant, alpha_servo, inputs, CASES, protected_hashes,
    load_observations, VERSION, COMMIT, write_table, summarize,
)


CASES_TO_RUN = ("orange_baseline", "angle_output_only", "angle_output_aw_limit")


def build_angle_runner(plant, case, raw_limit):
    original = inspect.getsource(alpha_servo.run_alpha_servo)
    source = original
    assert source.count('fin_deflection, p)') == 2
    source = source.replace('fin_moment_sample(s[0]', 'focus_sample(s[0]')
    source = source.replace('fin_deflection, p)', 'fin_deflection, p, alpha)')
    old_mapping = 'fin_deflection = max(-1.0, min(1.0, output)) * p.horizontal_fin_limit_rad'
    new_mapping = 'fin_deflection = max(-raw_limit, min(raw_limit, output))'
    old_aw = 'abs(output) > 1.0 and output * error_g > 0.0'
    new_aw = 'abs(output) > raw_limit and output * error_g > 0.0'
    if case in ('angle_output_only', 'angle_output_aw_limit'):
        assert source.count(old_mapping) == 1
        source = source.replace(old_mapping, new_mapping)
    if case == 'angle_output_aw_limit':
        assert source.count(old_aw) == 1
        source = source.replace(old_aw, new_aw)
    # Assert that the complete guidance/PID block differs only at the named
    # interface and (separate diagnostic only) its saturation gate.
    start = '        if time_s >= tick * control_dt_s - 1e-10:'
    end = '        boundaries = [time_s + plant_dt_s, tick * control_dt_s, end_time]'
    controller = original[original.index(start):original.index(end)]
    modified = source[source.index(start):source.index(end)]
    assert modified.replace(new_mapping, old_mapping).replace(new_aw, old_aw) == controller
    start, end = '    def forces(t, s):', '    def record(t, s):'
    assert source[source.index(start):source.index(end)] == original[original.index(start):original.index(end)]
    insert = '        samples.append(row)'
    source = source.replace(insert,
        '        row.update(pid_raw_output=pid_proportional+pid_integral+pid_derivative,\n'
        '                   capture_active=bool(capturing))\n' + insert)
    namespace = dict(vars(alpha_servo), focus_sample=plant.sample, raw_limit=raw_limit)
    exec(compile(source, '<PID angle-output diagnostic>', 'exec'), namespace)
    return namespace['run_alpha_servo'], hashlib.sha256(controller.encode()).hexdigest()


def run_cases(step, cases=CASES_TO_RUN):
    config, p, raw_limit = inputs()
    historical = replace(p, horizontal_fin_limit_rad=p.horizontal_fin_limit_rad*math.pi/2,
                          vertical_fin_limit_rad=p.vertical_fin_limit_rad*math.pi/2)
    results = {}
    for case in cases:
        parameters = p if case == 'normalized_raw_radians' else historical
        plant = FocusPlant(config, parameters, 'legacy_control_scale', raw_limit)
        runner, digest = build_angle_runner(plant, case, raw_limit)
        results[case] = runner(config, parameters, CASES[0][3], max_time_s=3.5, plant_dt_s=step)
    return results, dict(
        raw_angle_limit_rad=raw_limit,
        original_angle_limit_rad=historical.horizontal_fin_limit_rad,
        control_force_area_per_rad=p.fin_force_area_per_rad,
        damping_derivative_m4=plant.damping_m4,
        raw_pid=[p.acceleration_pid_kp, p.acceleration_pid_ki,
                 p.acceleration_pid_kd, p.acceleration_pid_integral_limit],
        original_controller_sha256=digest,
        original_plant_sha256=hashlib.sha256(inspect.getsource(FocusPlant).encode()).hexdigest(),
    )


def time_windows(rows, predicate):
    windows = []
    start = None
    for i, row in enumerate(rows):
        if predicate(row) and start is None:
            start = row['time_s']
        if start is not None and (not predicate(row) or i == len(rows)-1):
            windows.append([start, row['time_s']])
            start = None
    return windows


def save_figure(results, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = (
        ('orange_baseline', '#bd7929', '-', 'Previous orange: normalized output'),
        ('normalized_raw_radians', '#147f9b', '-', 'Normalized output x raw finsAoa [rad]'),
        ('angle_output_only', '#147f9b', '-', 'PID output in radians: mapping only'),
        ('angle_output_aw_limit', '#9468a8', '--', 'Radians + matching anti-windup limit'),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.8))
    panels = (
        ('alpha_deg', 'AoA [deg]', np.degrees(obs['alpha'])),
        ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
        ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6),
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', None),
        ('horizontal_fin_deflection_rad', 'Control angle sent to plant [deg]', None),
        ('pid_raw_output', 'Raw PID output (before angle mapping)', None),
    )
    for ax, (field, label, observed) in zip(axes.flat, panels):
        if observed is not None:
            ax.plot(obs['time'], observed, 'o-', color='#222222', lw=1, ms=4,
                    label='2.57 replay', zorder=5)
        for case, color, linestyle, title in styles:
            if case not in results:
                continue
            rows = results[case].samples
            values = np.array([r[field] for r in rows])
            if field == 'horizontal_fin_deflection_rad':
                values = np.degrees(values)
            ax.plot([r['time_s'] for r in rows], values,
                    color=color, ls=linestyle, lw=1.7, label=title)
        ax.set(xlim=(0.2, 3.5), ylabel=label, xlabel='Flight time [s]')
        ax.axvspan(.9, 1.3, color='#e5e9ed', alpha=.45, zorder=0)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .948),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle('R-77-1 | PID output-angle test | frozen 2.57.1.107', fontsize=16, y=.982)
    mapping_note = ('Blue changes only the angle scale: delta = clip(PID, +/- 1) x raw finsAoa; no extra pi/2.\n'
        if 'normalized_raw_radians' in results else
        'Solid blue changes only delta = clip(PID, +/- raw finsAoa). Purple also aligns the existing anti-windup gate.\n')
    fig.text(.075, .025,
        'Same orange force/moment plant, guidance, raw PID gains, error in g, integration and initial state; no refit.\n'
        + mapping_note +
        'Angle is an assumed control input, not a measured fin angle. Shaded region: replay peak/dip interval.',
        fontsize=9, color='#444444')
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.12, hspace=.36, wspace=.24)
    fig.savefig(prefix.with_suffix('.png'), dpi=160, facecolor='white')
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalized-only', action='store_true')
    args = parser.parse_args()
    cases = ('orange_baseline', 'normalized_raw_radians') if args.normalized_only else CASES_TO_RUN
    before = protected_hashes()
    _, obs = load_observations()  # Validates exact source version/commit/hashes.
    results, metadata = run_cases(.002, cases)
    fine, fine_metadata = run_cases(.001, cases)
    assert metadata == fine_metadata
    verification, summaries = {}, {}
    raw_limit = metadata['raw_angle_limit_rad']
    for case, result in results.items():
        rows = result.samples
        assert result.event == fine[case].event
        assert abs(rows[-1]['time_s']-3.5) < 1e-9
        assert len(rows) == len(fine[case].samples)
        errors = {field: max(abs(a[field]-b[field]) for a,b in zip(rows, fine[case].samples))
                  for field in ('alpha_deg','wind_normal_specific_force_g','speed_kmh')}
        assert max(errors.values()) < .01, (case, errors)
        mapping_error = max(abs(r['horizontal_fin_deflection_rad'] - (
            np.clip(r['pid_raw_output'], -1., 1.)*metadata['original_angle_limit_rad']
            if case == 'orange_baseline' else
            np.clip(r['pid_raw_output'], -1., 1.)*raw_limit if case == 'normalized_raw_radians' else
            np.clip(r['pid_raw_output'], -raw_limit, raw_limit)))
            for r in rows)
        assert mapping_error < 1e-12
        verification[case] = dict(half_step_max_difference=errors,
                                  angle_mapping_max_residual_rad=float(mapping_error))
        summary = summarize(result, obs)
        summary['capture_exit_time_s'] = next((r['time_s'] for r in rows if not r['capture_active']), None)
        limit = 1. if case in ('orange_baseline', 'normalized_raw_radians') else raw_limit
        summary['angle_saturation_windows_s'] = time_windows(rows, lambda r: abs(r['pid_raw_output']) > limit)
        summary['control_force_saturation_windows_s'] = time_windows(rows, lambda r: r['fin_control_force_saturated'])
        summary['alpha_change_0p9_to_1p1_deg'] = float(
            np.interp(1.1, [r['time_s'] for r in rows], [r['alpha_deg'] for r in rows])
            - np.interp(.9, [r['time_s'] for r in rows], [r['alpha_deg'] for r in rows]))
        summaries[case] = summary
    assert protected_hashes() == before
    prefix = ROOT/'outputs'/('r77_1_normalized_raw_angle' if args.normalized_only else 'r77_1_pid_angle_output')
    report = dict(source_version=VERSION, source_commit=COMMIT, metadata=metadata,
        fitted_parameters=[], variants=summaries, verification=verification,
        protected_inputs_unchanged=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),
        [dict(case=case, **frame) for case, summary in summaries.items() for frame in summary['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
        [dict(case=case, **row) for case, result in results.items() for row in result.samples])
    save_figure(results, obs, prefix)
    print(json.dumps(dict(metadata=metadata, variants={case:{key:value for key,value in summary.items()
        if key != 'frames'} for case,summary in summaries.items()}, verification=verification), indent=2))
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
