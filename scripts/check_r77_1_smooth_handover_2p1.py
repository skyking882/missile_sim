"""Isolate the same request smoothstep after holding capture until 2.1 s."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math

import numpy as np

import check_r77_1_capture_strategy as capture
from check_r77_1_smooth_handover import DURATIONS_S, build_smooth_runner
from run_r77_1_current_full_forward import sha

ROOT = capture.ROOT
START_S = 2.1
PROBE_TIMES_S = (2.1, 2.3, 2.5, 2.7, 3.1, 3.5)
STATE_FIELDS = (
    'alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh', 'heading_rad',
    'x_m', 'z_m', 'body_heading_rad', 'body_rate_rad_s', 'mass_kg', 'thrust_n',
    'drag_n', 'horizontal_alpha_rad', 'vertical_alpha_rad', 'altitude_m',
    'distance_m', 'target_x_m', 'target_z_m', 'inertia_kg_m2',
)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def control_rows(rows):
    return [row for row in rows if abs(row['time_s'] - round(row['time_s'] / .02) * .02) < 1e-10]


def native_observations(path):
    with path.open() as stream:
        native = list(csv.DictReader(stream, delimiter='\t'))
    return [dict(time_s=float(row['missile_flight_time_s']),
                 alpha_deg=float(row['angle_of_attack_deg']),
                 wind_normal_specific_force_g=float(row['overload_g']),
                 speed_kmh=float(row['speed_kmh'])) for row in native]


def read_trace(path):
    with path.open() as stream:
        return [{key: (value == 'True' if value in ('True', 'False') else float(value))
                 for key, value in row.items()} for row in csv.DictReader(stream, delimiter='\t')]


def interpolate(rows, time_s, field):
    return float(np.interp(time_s, [row['time_s'] for row in rows], [row[field] for row in rows]))


def common_numeric_difference(left, right):
    assert len(left) == len(right)
    keys = [key for key in left[0] if key in right[0]
            and (is_number(left[0][key]) or isinstance(left[0][key], bool))
            and (is_number(right[0][key]) or isinstance(right[0][key], bool))]
    return {key: max(abs(float(a[key]) - float(b[key])) for a, b in zip(left, right)) for key in keys}


def shape_window(rows, observed, start, duration):
    descent = [row for row in rows if 2.1 - 1e-12 <= row['time_s'] <= 3.5 + 1e-12]
    switch = [row for row in rows if 2.08 - 1e-12 <= row['time_s'] <= 2.30 + 1e-12]
    alpha_min = min(descent, key=lambda row: row['alpha_deg'])
    alpha_max = max(descent, key=lambda row: row['alpha_deg'])
    load_min = min(descent, key=lambda row: row['wind_normal_specific_force_g'])
    load_max = max(descent, key=lambda row: row['wind_normal_specific_force_g'])
    fin_min = min(switch, key=lambda row: row['horizontal_fin_deflection_rad'])
    derivative_peak = max(switch, key=lambda row: abs(row['pid_derivative_output']))
    probes = {}
    for time_s in PROBE_TIMES_S:
        measured = next(item for item in observed if abs(item['time_s'] - time_s) < 1e-12)
        probes[f'{time_s:.1f}'] = dict(
            alpha_deg_predicted=interpolate(rows, time_s, 'alpha_deg'),
            alpha_deg_measured=measured['alpha_deg'],
            wind_normal_specific_force_g_predicted=interpolate(rows, time_s, 'wind_normal_specific_force_g'),
            wind_normal_specific_force_g_measured=measured['wind_normal_specific_force_g'],
            speed_kmh_predicted=interpolate(rows, time_s, 'speed_kmh'),
            speed_kmh_measured=measured['speed_kmh'],
            request_g=interpolate(rows, time_s, 'commanded_horizontal_acceleration_g'),
            fin_deg=math.degrees(interpolate(rows, time_s, 'horizontal_fin_deflection_rad')),
        )
    return dict(
        handover_end_s=start + duration,
        alpha_peak_2p1_to_3p5=dict(time_s=alpha_max['time_s'], alpha_deg=alpha_max['alpha_deg']),
        alpha_trough_2p1_to_3p5=dict(time_s=alpha_min['time_s'], alpha_deg=alpha_min['alpha_deg']),
        load_peak_2p1_to_3p5=dict(time_s=load_max['time_s'],
                                 wind_normal_specific_force_g=load_max['wind_normal_specific_force_g']),
        load_trough_2p1_to_3p5=dict(time_s=load_min['time_s'],
                                   wind_normal_specific_force_g=load_min['wind_normal_specific_force_g']),
        reverse_fin_at_2p1=math.degrees(next(row['horizontal_fin_deflection_rad'] for row in rows
                                            if abs(row['time_s'] - start) < 1e-10)) < 0.,
        minimum_fin_2p08_to_2p30_deg=math.degrees(fin_min['horizontal_fin_deflection_rad']),
        minimum_fin_time_s=fin_min['time_s'],
        largest_abs_derivative_2p08_to_2p30=derivative_peak['pid_derivative_output'],
        derivative_peak_time_s=derivative_peak['time_s'],
        probes=probes,
    )


def plot(results, observed, start, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {
        0.: ('#c77c27', '-', 'Hard switch at 2.10 s'),
        .1: ('#387cc0', '-', 'Smooth: T = 0.10 s'),
        .3: ('#148d81', '--', 'Smooth: T = 0.30 s'),
        .6: ('#9662ad', '-.', 'Smooth: T = 0.60 s'),
    }
    curves = [(results[duration].samples, *styles[duration]) for duration in DURATIONS_S]
    end_time = max(result.samples[-1]['time_s'] for result in results.values())
    fields = [('alpha_deg', 'AoA [deg]'),
              ('wind_normal_specific_force_g', 'Normal load [g]'),
              ('speed_kmh', 'Speed [km/h]')]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), sharex='col')
    for index, (field, label) in enumerate(fields):
        for col in range(2):
            ax = axes[index, col]
            ax.plot([row['time_s'] for row in observed], [row[field] for row in observed],
                    'o', color='#252525', ms=3.4, zorder=6)
            for rows, color, style, _title in curves:
                ax.plot([row['time_s'] for row in rows], [row[field] for row in rows],
                        color=color, ls=style, lw=1.85)
            ax.axvline(start, color='#666', ls=':', lw=.8)
            if col == 0:
                ax.axvline(7., color='#708b9e', ls='--', lw=.8)
                ax.set(xlim=(0., max(end_time, observed[-1]['time_s']) + .12), ylabel=label)
            else:
                ax.set(xlim=(1.8, 3.5), ylabel=label)
                values = [row[field] for rows, *_ in curves for row in rows if 1.8 <= row['time_s'] <= 3.5]
                values += [row[field] for row in observed if 1.8 <= row['time_s'] <= 3.5]
                lo, hi = min(values), max(values)
                ax.set_ylim(lo - .07 * (hi - lo), hi + .09 * (hi - lo))
            ax.grid(alpha=.16)
            ax.spines[['top', 'right']].set_visible(False)
            if index == 2:
                ax.set_xlabel('Missile flight time [s]')
    axes[0, 0].set_title('Complete forward runs', fontsize=12)
    axes[0, 1].set_title('Descent detail: 1.8-3.5 s', fontsize=12)
    fig.suptitle('R-77-1 | smooth request handover after capture until 2.10 s', fontsize=16, y=.986)
    handles = [Line2D([], [], color='#252525', marker='o', lw=0., label='Game measurements')]
    handles += [Line2D([], [], color=color, ls=style, lw=1.85, label=title)
                for _rows, color, style, title in curves]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .953), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.07, .024,
             'Request = (1-w) x last delivered capture request + w x live PN request; w = 3s² - 2s³.\n'
             'Same 35.25 g capture from 0.30 s to 2.10 s. Only handover duration varies; no PID reset, plant change or state injection.\n'
             'Frozen 2.57.1.107; target course 180 deg. T values are sensitivity cases, not identified game parameters.',
             fontsize=9, color='#444')
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.13, hspace=.20, wspace=.22)
    fig.savefig(prefix.with_suffix('.png'), dpi=170, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), sharex=True)
    panels = [
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', 1., True),
        ('wind_normal_specific_force_g', 'Actual normal load [g]', 1., False),
        ('horizontal_fin_deflection_rad', 'Control angle [deg]', 180. / math.pi, True),
        ('pid_derivative_output', 'PID derivative contribution', 1., True),
    ]
    for ax, (field, label, factor, stepped) in zip(axes.flat, panels):
        if field == 'wind_normal_specific_force_g':
            ax.plot([row['time_s'] for row in observed], [row[field] for row in observed],
                    'o', color='#252525', ms=4.2, zorder=6)
        for duration in DURATIONS_S:
            color, style, _title = styles[duration]
            rows = [row for row in results[duration].samples if 1.95 <= row['time_s'] <= 2.85]
            times = [row['time_s'] for row in rows]
            values = [factor * row[field] for row in rows]
            if stepped:
                ax.step(times, values, where='post', color=color, ls=style, lw=1.7)
            else:
                ax.plot(times, values, color=color, ls=style, lw=1.7)
        ax.axvline(start, color='#666', ls=':', lw=.8)
        ax.set(xlim=(1.95, 2.85), ylabel=label, xlabel='Missile flight time [s]')
        ax.grid(alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Handover request, load and unchanged PID response', fontsize=16, y=.982)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.075, .026,
             'Commands are held for each 20 ms control interval. Smooth describes the underlying reference, sampled at the existing PID rate.\n'
             'The same PID stays active; its integral and previous-error states continue without resetting. Origin is the last request delivered to the PID.',
             fontsize=9, color='#444')
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.145, hspace=.28, wspace=.25)
    fig.savefig(prefix.with_name(prefix.name + '_control.png'), dpi=170, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='outputs/r77_1_smooth_handover_2p1')
    args = parser.parse_args()
    prefix = ROOT / args.output_prefix
    baseline_path = ROOT / 'outputs/r77_1_current_full_forward.json'
    baseline_trace_path = ROOT / 'outputs/r77_1_current_full_forward_trajectory.tsv'
    replay_path = ROOT / 'data/replays/r77_1_level_20260824.tsv'
    extra = [baseline_path, baseline_trace_path,
             ROOT / 'scripts/run_r77_1_focus_moment.py',
             ROOT / 'scripts/check_r77_1_capture_strategy.py',
             ROOT / 'scripts/check_r77_1_smooth_handover.py',
             ROOT / 'scripts/run_r77_1_current_full_forward.py']
    protected = capture.protected_hashes() | {str(path): sha(path) for path in extra}
    baseline = json.loads(baseline_path.read_text())
    assert all(sha(ROOT / path) == expected for path, expected in baseline['protected_input_hashes'].items())
    config, p, raw_limit = capture.inputs()
    p = replace(p, fin_force_area_per_rad=baseline['fixed_parameters']['fin_force_area_per_rad'],
                guidance_delay_s=baseline['fixed_parameters']['guidance_delay_s'])
    assert asdict(p) == baseline['fixed_parameters']
    assert config == baseline['fixed_configuration']
    assert (capture.VERSION, capture.COMMIT) == (baseline['source_version'], baseline['source_commit'])
    assert p.acceleration_pid_kp == .0071 and p.acceleration_pid_ki == .005
    assert p.acceleration_pid_kd == .00015 and p.acceleration_pid_integral_limit == 1.
    table = tuple(tuple(row) for row in baseline['raw_time_to_gain_table'])
    timed_capture = (baseline['capture_cap_g'], baseline['capture_duration_s'])
    assert timed_capture == (35.25, 1.8)
    assert abs(table[0][0] + timed_capture[1] - START_S) < 1e-12
    old = read_trace(baseline_trace_path)
    observed = native_observations(replay_path)
    assert len(observed) == 46 == baseline['native_replay_frame_count']
    plant = capture.FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    results, summaries, checks, comparisons, shapes = {}, {}, {}, [], {}
    reproduction = None
    for duration in DURATIONS_S:
        runner, pid_hash = build_smooth_runner(plant, table, timed_capture, duration)
        assert pid_hash == baseline['pid_block_sha256']
        result = runner(config, p, baseline['scenario'], max_time_s=config['performance']['lifetime_s'],
                        plant_dt_s=.002, control_dt_s=.02)
        capture.enrich(result.samples, p, baseline['scenario'])
        results[duration] = result
        rows = result.samples
        assert result.event in ('fuse', 'closest_approach', 'lifetime')
        assert all(math.isfinite(value) for row in rows for value in row.values() if is_number(value))
        times = np.array([row['time_s'] for row in rows])
        assert all(later['time_s'] > earlier['time_s'] for earlier, later in zip(rows, rows[1:]))
        for obs in observed:
            supported = bool(times[0] <= obs['time_s'] <= times[-1])
            item = dict(handover_duration_s=duration, time_s=obs['time_s'], within_forward_run=supported)
            for field in ('alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh'):
                predicted = interpolate(rows, obs['time_s'], field) if supported else None
                item[field + '_measured'] = obs[field]
                item[field + '_predicted'] = predicted
                item[field + '_error'] = None if predicted is None else predicted - obs[field]
            comparisons.append(item)
        assert all(item['within_forward_run'] for item in comparisons[-len(observed):])

        switch_index = next(index for index, row in enumerate(rows) if abs(row['time_s'] - START_S) < 1e-10)
        switch = rows[switch_index]
        previous = rows[switch_index - 1]
        ticks = control_rows(rows)
        assert ticks[-1]['time_s'] < rows[-1]['time_s'] or abs(ticks[-1]['time_s'] - rows[-1]['time_s']) < 1e-10
        pre = [row for row in rows if row['time_s'] < START_S - 1e-10]
        pre_old = [row for row in old if row['time_s'] < START_S - 1e-10]
        pre_error = common_numeric_difference(pre_old, pre)
        assert max(pre_error.values()) == 0.
        state_at_switch = {field: abs(switch[field] - next(row[field] for row in old
                                                           if abs(row['time_s'] - START_S) < 1e-10))
                           for field in STATE_FIELDS}
        assert max(state_at_switch.values()) == 0.
        if duration == 0.:
            assert len(rows) == len(old) and result.event == baseline['event']
            reproduction = common_numeric_difference(old, rows)
            assert max(reproduction.values()) == 0.
            assert abs(switch['commanded_horizontal_acceleration_g'] - old[switch_index]['commanded_horizontal_acceleration_g']) == 0.
        else:
            assert switch['handover_origin_request_g'] == previous['commanded_horizontal_acceleration_g']
            assert abs(switch['commanded_horizontal_acceleration_g'] - previous['commanded_horizontal_acceleration_g']) < 1e-12
            assert switch['handover_pn_weight'] == 0.
            assert switch['pid_integral_output'] != 0.
            assert switch['pid_integral_output'] != old[switch_index]['pid_integral_output']
        assert all(abs(row['handover_origin_request_g'] - switch['handover_origin_request_g']) < 1e-12
                   for row in rows[switch_index:])
        assert all(not row['capture_active'] for row in rows if row['time_s'] >= START_S - 1e-10)

        limit = math.sqrt(p.command_limit_g ** 2 - 1.)
        residuals, integral_errors, post_pn = [], [], []
        for before, row in zip(ticks, ticks[1:]):
            if row['time_s'] < START_S - 1e-10:
                continue
            phase = np.clip((row['time_s'] - START_S) / duration, 0., 1.) if duration else 1.
            weight = float(phase * phase * (3. - 2. * phase))
            pn = float(np.clip(row['raw_pn_request_g'], -limit, limit))
            expected = (1. - weight) * row['handover_origin_request_g'] + weight * pn
            residuals.append(abs(row['commanded_horizontal_acceleration_g'] - expected))
            if row['time_s'] >= START_S + duration - 1e-10:
                post_pn.append(abs(row['commanded_horizontal_acceleration_g'] - pn))
            expected_i = np.clip(before['pid_integral_output'] + p.acceleration_pid_ki * row['pid_error_g'] * .02,
                                 -p.acceleration_pid_integral_limit, p.acceleration_pid_integral_limit)
            proposed = row['pid_proportional_output'] + expected_i + row['pid_derivative_output']
            if abs(proposed) > 1. and proposed * row['pid_error_g'] > 0.:
                expected_i = before['pid_integral_output']
            integral_errors.append(abs(row['pid_integral_output'] - expected_i))
        assert max(residuals) < 1e-10
        assert max(integral_errors) < 1e-12
        assert (not post_pn) or max(post_pn) < 1e-10
        assert all(abs(row['commanded_horizontal_acceleration_g']) <= limit + 1e-10 for row in rows)
        assert all(abs(row['horizontal_fin_deflection_rad'] - np.clip(row['pid_raw_output'], -1., 1.) * raw_limit) < 1e-12
                   for row in rows)
        assert all(row['thrust_n'] == 0. for row in rows if row['time_s'] >= 7.)
        shapes[str(duration)] = shape_window(rows, observed, START_S, duration)
        summaries[str(duration)] = dict(
            event=result.event, terminal_time_s=rows[-1]['time_s'],
            minimum_distance_m=result.minimum_distance_m, handover_start_s=START_S,
            handover_end_s=START_S + duration,
            request_at_start_g=switch['commanded_horizontal_acceleration_g'],
            origin_request_g=switch['handover_origin_request_g'],
            fin_at_start_deg=math.degrees(switch['horizontal_fin_deflection_rad']),
            derivative_at_start=switch['pid_derivative_output'],
            integral_at_start=switch['pid_integral_output'],
            compared_frame_count=len(observed),
            shape=shapes[str(duration)],
        )
        checks[str(duration)] = dict(
            pre_switch_max_difference=pre_error,
            physical_state_at_2p1_max_difference=state_at_switch,
            request_formula_max_residual_g=float(max(residuals)),
            integral_update_max_residual=float(max(integral_errors)),
            post_handover_pn_max_residual_g=float(max(post_pn)) if post_pn else 0.,
            origin_frozen_after_start=True,
            no_recapture_after_start=True,
            pid_block_sha256=pid_hash,
            final_event=result.event,
        )

    first = results[0.].samples
    for duration in DURATIONS_S[1:]:
        other = results[duration].samples
        shared = [(a, b) for a, b in zip(first, other) if a['time_s'] < START_S - 1e-10]
        assert max(abs(a[field] - b[field]) for a, b in shared for field in capture.FIELDS) == 0.
        assert max(abs(a['commanded_horizontal_acceleration_g'] - b['commanded_horizontal_acceleration_g'])
                   for a, b in shared) == 0.
        assert max(abs(a['pid_integral_output'] - b['pid_integral_output']) for a, b in shared) == 0.
        assert max(abs(a['pid_derivative_output'] - b['pid_derivative_output']) for a, b in shared) == 0.
        switch_a = next(row for row in first if abs(row['time_s'] - START_S) < 1e-10)
        switch_b = next(row for row in other if abs(row['time_s'] - START_S) < 1e-10)
        assert max(abs(switch_a[field] - switch_b[field]) for field in STATE_FIELDS) == 0.

    assert all(sha(ROOT / path) == expected for path, expected in protected.items())
    report = dict(
        source_version=capture.VERSION, source_commit=capture.COMMIT,
        fixed_parameters=asdict(p), scenario=baseline['scenario'],
        fixed_configuration=config, configuration_sha256=capture.digest(config),
        capture_cap_g=timed_capture[0], capture_duration_s=timed_capture[1],
        capture_start_s=table[0][0], capture_end_s=START_S,
        raw_time_to_gain_table=table, handover_durations_s=DURATIONS_S,
        changed_parameters=['request_handover_duration_s'], fitted_parameters=[],
        formula='n_req=(1-w)*n_last_capture+w*n_PN(t); w=3*s^2-2*s^3; s=clip((t-2.1)/T,0,1)',
        origin='Last delivered PID acceleration request; not actual load or measured AoA',
        pid_block_sha256=baseline['pid_block_sha256'],
        results=summaries, comparisons=comparisons, shape=shapes,
        verification=dict(
            per_case=checks,
            hard_switch_reproduction_max_difference=reproduction,
            protected_inputs_unchanged=True,
            force_rhs_and_pid_blocks_unchanged=True,
            native_replay_frame_count=len(observed),
            compared_frame_count=len(observed),
            no_extrapolation_past_terminal_event=True,
            midflight_state_injection=False,
        ),
        protected_input_hashes=protected, runtime_defaults_changed=False,
        measured_input_use='Native 46-frame overlay after free forward runs; displayed_distance unused; no kinematic state injection',
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    capture.write_table(prefix.with_name(prefix.name + '_trajectory.tsv'),
                        [dict(handover_duration_s=duration, **row)
                         for duration, result in results.items() for row in result.samples])
    capture.write_table(prefix.with_name(prefix.name + '_comparison.tsv'), comparisons)
    plot(results, observed, START_S, prefix)
    print(json.dumps(summaries, indent=2))
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
