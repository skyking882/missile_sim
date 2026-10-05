"""Isolate smooth capture-to-PN request transfer without changing PID or plant."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import inspect
import json
import math

import numpy as np

import check_r77_1_capture_strategy as capture
from check_r77_1_capture_until_trk import read_trace
from run_r77_1_current_full_forward import sha

ROOT = capture.ROOT
DURATIONS_S = (0., .1, .3, .6)


def build_smooth_runner(plant, request_gain_table, timed_capture, duration_s):
    if not math.isfinite(duration_s) or duration_s < 0.:
        raise ValueError('Handover duration must be finite and nonnegative')
    # Obtain the exact source already checked by the existing experimental
    # factory. Only its return value changes in this local copy; no source file
    # or global is modified, and all existing baseline receipts remain valid.
    factory = inspect.getsource(capture.build_runner)
    anchor = "return namespace['run_alpha_servo'], hashlib.sha256(pid.encode()).hexdigest()"
    assert factory.count(anchor) == 1
    factory = factory.replace(anchor, 'return source, namespace, hashlib.sha256(pid.encode()).hexdigest()')
    factory_namespace = dict(vars(capture))
    exec(compile(factory, '<capture factory source export>', 'exec'), factory_namespace)
    original, namespace, pid_hash = factory_namespace['build_runner'](
        plant, pn_capture=False, retain_command=False,
        request_gain_table=request_gain_table, timed_capture=timed_capture)
    source = original
    anchor = '    command = 0.0\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor +
        '    handover_started = False\n'
        '    handover_origin_g = 0.0\n'
        '    handover_weight = 0.0\n')
    # Blend after the existing schedule and acceleration limits, using the
    # last request actually delivered to this PID, not the measured load.
    anchor = '            desired_alpha = ps.mass_kg * a_cmd / gain\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '            if time_s >= capture_end - 1e-10:\n'
        '                if not handover_started:\n'
        '                    handover_origin_g = commanded_acceleration_g\n'
        '                    handover_started = True\n'
        '                if handover_duration_s > 0.:\n'
        '                    phase = max(0., min(1., (time_s-capture_end)/handover_duration_s))\n'
        '                    handover_weight = phase*phase*(3.-2.*phase)\n'
        '                    a_cmd = (1.-handover_weight)*handover_origin_g*G0 + handover_weight*a_cmd\n'
        '                else:\n'
        '                    handover_weight = 1.\n' + anchor)
    anchor = '        samples.append(row)'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '        row.update(handover_origin_request_g=handover_origin_g,\n'
        '                   handover_pn_weight=handover_weight,\n'
        '                   handover_active=bool(handover_started and handover_weight < 1.))\n' + anchor)
    for start, end in [
        ('    def forces(t, s):', '    def record(t, s):'),
        ('                # One PID: error in wind-normal g, output in fin-travel', '            record(time_s, state)'),
        ('        def rhs(t, s):', '        def shifted(s, k, h):'),
    ]:
        assert source[source.index(start):source.index(end)] == original[original.index(start):original.index(end)]
    namespace['handover_duration_s'] = duration_s
    exec(compile(source, '<smooth request handover>', 'exec'), namespace)
    return namespace['run_alpha_servo'], pid_hash


def plot(results, old, observed, start, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {0.: ('#858b92', '--', 'Direct switch at 0.90 s'),
              .1: ('#387cc0', '-', 'Smooth: T = 0.10 s'),
              .3: ('#148d81', '-', 'Smooth: T = 0.30 s'),
              .6: ('#9662ad', '-', 'Smooth: T = 0.60 s')}
    curves = [(old, '#c77c27', ':', 'Previous capture until 2.10 s')]
    curves += [(results[d].samples, *styles[d]) for d in DURATIONS_S]
    end_time = max(r.samples[-1]['time_s'] for r in results.values())
    fields = [('alpha_deg', 'AoA [deg]'),
              ('wind_normal_specific_force_g', 'Normal load [g]'),
              ('speed_kmh', 'Speed [km/h]')]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), sharex='col')
    for j, (field, label) in enumerate(fields):
        for col in range(2):
            ax = axes[j, col]
            ax.plot([r['time_s'] for r in observed], [r[field] for r in observed],
                    'o-', color='#252525', ms=3.2, lw=1., zorder=6)
            for rows, color, style, title in curves:
                ax.plot([r['time_s'] for r in rows], [r[field] for r in rows],
                        color=color, ls=style, lw=1.8)
            ax.axvline(start, color='#666', ls=':', lw=.8)
            ax.set(xlim=(0., max(end_time, old[-1]['time_s'])+.1) if col == 0 else (.6, 2.3), ylabel=label)
            if col == 1:
                values = [r[field] for rows, *_ in curves for r in rows if .6 <= r['time_s'] <= 2.3]
                values += [r[field] for r in observed if .6 <= r['time_s'] <= 2.3]
                lo, hi = min(values), max(values)
                ax.set_ylim(lo-.06*(hi-lo), hi+.08*(hi-lo))
            ax.grid(alpha=.16)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 2:
                ax.set_xlabel('Missile flight time [s]')
    axes[0, 0].set_title('Complete forward runs', fontsize=12)
    axes[0, 1].set_title('Handover detail: 0.6-2.3 s', fontsize=12)
    fig.suptitle('R-77-1 | smooth request handover starting at first observed TRK', fontsize=16, y=.986)
    handles = [Line2D([], [], color='#252525', marker='o', lw=1., label='Game measurements')]
    handles += [Line2D([], [], color=c, ls=s, lw=1.8, label=label) for _, c, s, label in curves]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .953), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.07, .026,
        'Request = (1-w) x last delivered capture request + w x live PN request; w = 3s² - 2s³.\n'
        'Same 35.25 g capture until 0.90 s. Only handover duration varies; no PID reset, gain change, plant change or state injection.\n'
        'Frozen 2.57.1.107; target course 180 deg. T values are sensitivity cases, not identified game parameters.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.13, hspace=.20, wspace=.22)
    fig.savefig(prefix.with_suffix('.png'), dpi=170, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.7), sharex=True)
    fields = [('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', 1.),
              ('horizontal_fin_deflection_rad', 'Control angle [deg]', 180./math.pi),
              ('pid_derivative_output', 'PID derivative contribution', 1.),
              ('pid_integral_output', 'PID integral contribution', 1.)]
    for ax, (field, label, factor) in zip(axes.flat, fields):
        for duration in DURATIONS_S:
            color, style, _ = styles[duration]
            rows = [r for r in results[duration].samples if .8 <= r['time_s'] <= 1.66]
            ax.step([r['time_s'] for r in rows], [factor*r[field] for r in rows],
                    where='post', color=color, ls=style, lw=1.7)
        ax.axvline(start, color='#666', ls=':', lw=.8)
        ax.set(xlim=(.8, 1.65), ylabel=label, xlabel='Missile flight time [s]')
        ax.grid(alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Handover input and unchanged PID response', fontsize=16, y=.982)
    fig.legend(handles=handles[2:], loc='upper center', bbox_to_anchor=(.5, .934), ncol=2, frameon=False, fontsize=9.5)
    fig.text(.075, .027,
        'Commands are held for each 20 ms control interval. Smooth describes the underlying reference, sampled at the existing PID rate.\n'
        'The same PID stays active; its integral and previous-error states continue without resetting.', fontsize=9, color='#444')
    fig.subplots_adjust(left=.075, right=.98, top=.805, bottom=.145, hspace=.32, wspace=.25)
    fig.savefig(prefix.with_name(prefix.name+'_control.png'), dpi=170, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='outputs/r77_1_smooth_handover')
    args = parser.parse_args()
    prefix = ROOT/args.output_prefix
    baseline_path = ROOT/'outputs/r77_1_current_full_forward.json'
    direct_path = ROOT/'outputs/r77_1_capture_until_trk.json'
    baseline = json.loads(baseline_path.read_text())
    direct = json.loads(direct_path.read_text())
    extra = [baseline_path, direct_path, ROOT/'outputs/r77_1_current_full_forward_trajectory.tsv',
             ROOT/'outputs/r77_1_capture_until_trk_trajectory.tsv',
             ROOT/'scripts/run_r77_1_focus_moment.py', ROOT/'scripts/check_r77_1_capture_strategy.py']
    protected = capture.protected_hashes() | {str(p):sha(p) for p in extra}
    assert all(sha(ROOT/path) == expected for path, expected in direct['protected_input_hashes'].items())
    config, p, raw_limit = capture.inputs()
    p = replace(p, fin_force_area_per_rad=baseline['fixed_parameters']['fin_force_area_per_rad'],
                guidance_delay_s=baseline['fixed_parameters']['guidance_delay_s'])
    assert asdict(p) == baseline['fixed_parameters'] == direct['fixed_parameters']
    assert config == baseline['fixed_configuration']
    assert (capture.VERSION, capture.COMMIT) == (baseline['source_version'], baseline['source_commit'])
    table = tuple(tuple(r) for r in baseline['raw_time_to_gain_table'])
    start = direct['capture_end_s']
    timed_capture = (direct['capture_cap_g'], direct['capture_duration_s'])
    old = read_trace(extra[2])
    direct_rows = read_trace(extra[3])
    observed = [dict(time_s=r['time_s'], **{f:r[f+'_measured'] for f in (
        'alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh')}) for r in direct['comparison']]
    plant = capture.FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    results, summaries, checks, comparisons = {}, {}, {}, []
    for duration in DURATIONS_S:
        runner, pid_hash = build_smooth_runner(plant, table, timed_capture, duration)
        assert pid_hash == baseline['pid_block_sha256']
        result = runner(config, p, baseline['scenario'], max_time_s=config['performance']['lifetime_s'],
                        plant_dt_s=.002, control_dt_s=.02)
        capture.enrich(result.samples, p, baseline['scenario'])
        results[duration] = result
        rows = result.samples
        assert result.event in ('fuse', 'closest_approach', 'lifetime')
        assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
        for obs in observed:
            item = dict(handover_duration_s=duration, time_s=obs['time_s'],
                        within_forward_run=obs['time_s'] <= rows[-1]['time_s'])
            for field in ('alpha_deg', 'wind_normal_specific_force_g', 'speed_kmh'):
                item[field+'_measured'] = obs[field]
                item[field+'_predicted'] = float(np.interp(obs['time_s'], [r['time_s'] for r in rows],
                    [r[field] for r in rows])) if item['within_forward_run'] else None
            comparisons.append(item)
        transition = [r for r in rows if start-1e-10 <= r['time_s'] <= start+max(duration, .02)+1e-10]
        early = [r for r in rows if 1. <= r['time_s'] <= 2.]
        switch_index = next(i for i, r in enumerate(rows) if abs(r['time_s']-start) < 1e-10)
        switch = rows[switch_index]
        summaries[str(duration)] = dict(event=result.event, terminal_time_s=rows[-1]['time_s'],
            minimum_distance_m=result.minimum_distance_m, handover_start_s=start, handover_end_s=start+duration,
            request_at_start_g=switch['commanded_horizontal_acceleration_g'],
            fin_at_start_deg=math.degrees(switch['horizontal_fin_deflection_rad']),
            derivative_at_start=switch['pid_derivative_output'],
            minimum_fin_during_handover_deg=math.degrees(min(r['horizontal_fin_deflection_rad'] for r in transition)),
            largest_abs_derivative_during_handover=max(abs(r['pid_derivative_output']) for r in transition),
            minimum_alpha_1_to_2_s=dict(time_s=min(early, key=lambda r:r['alpha_deg'])['time_s'],
                                      alpha_deg=min(r['alpha_deg'] for r in early)),
            compared_frame_count=sum(r['time_s'] <= rows[-1]['time_s'] for r in observed))

        # Focused checks on actual generated controller behavior, beyond the
        # interpolation formula: baseline equivalence, causality, saturation,
        # and preserving PID state evolution across the mode boundary.
        common = [(a, b) for a, b in zip(direct_rows, rows) if a['time_s'] < start-1e-10]
        pre_error = {f:max(abs(a[f]-b[f]) for a, b in common) for f in capture.FIELDS}
        assert max(pre_error.values()) == 0.
        if duration == 0.:
            assert len(rows) == len(direct_rows) and result.event == direct['event']
            reproduction = {f:max(abs(a[f]-b[f]) for a, b in zip(direct_rows, rows)) for f in capture.FIELDS}
            assert max(reproduction.values()) == 0.
        else:
            old_switch = next(r for r in old if abs(r['time_s']-start) < 1e-10)
            assert all(switch[f] == old_switch[f] for f in capture.FIELDS)
            assert switch['pid_integral_output'] == old_switch['pid_integral_output']
            assert switch['handover_origin_request_g'] == rows[switch_index-1]['commanded_horizontal_acceleration_g']
        limit = math.sqrt(p.command_limit_g**2-1.)
        residuals = []
        integral_errors = []
        for before, row in zip(rows, rows[1:-1]):
            if row['time_s'] < start-1e-10:
                continue
            phase = np.clip((row['time_s']-start)/duration, 0., 1.) if duration else 1.
            weight = phase**2*(3.-2.*phase)
            pn = float(np.clip(row['raw_pn_request_g'], -limit, limit))
            expected = (1.-weight)*row['handover_origin_request_g']+weight*pn
            residuals.append(abs(row['commanded_horizontal_acceleration_g']-expected))
            expected_i = np.clip(before['pid_integral_output'] + p.acceleration_pid_ki*row['pid_error_g']*.02,
                                 -p.acceleration_pid_integral_limit, p.acceleration_pid_integral_limit)
            proposed = row['pid_proportional_output'] + expected_i + row['pid_derivative_output']
            if abs(proposed) > 1. and proposed*row['pid_error_g'] > 0.:
                expected_i = before['pid_integral_output']
            integral_errors.append(abs(row['pid_integral_output']-expected_i))
        assert max(residuals) < 1e-10
        assert max(integral_errors) < 1e-12
        assert all(abs(r['commanded_horizontal_acceleration_g']) <= limit+1e-10 for r in rows)
        assert all(abs(r['horizontal_fin_deflection_rad']-np.clip(r['pid_raw_output'], -1., 1.)*raw_limit) < 1e-12 for r in rows)
        assert all(r['thrust_n'] == 0. for r in rows if r['time_s'] >= 7.)
        checks[str(duration)] = dict(pre_switch_max_difference=pre_error,
            request_formula_max_residual_g=max(residuals), integral_update_max_residual=max(integral_errors),
            pid_block_sha256=pid_hash, final_event=result.event)
    assert all(sha(ROOT/path) == expected for path, expected in protected.items())
    report = dict(source_version=capture.VERSION, source_commit=capture.COMMIT,
        fixed_parameters=asdict(p), scenario=baseline['scenario'], configuration_sha256=capture.digest(config),
        capture_cap_g=timed_capture[0], capture_end_s=start, first_trk_frame=direct['first_trk_frame'],
        handover_durations_s=DURATIONS_S, changed_parameters=['request_handover_duration_s'], fitted_parameters=[],
        formula='n_req=(1-w)*n_last_capture+w*n_PN(t); w=3*s^2-2*s^3; s=clip((t-t_start)/T,0,1)',
        origin='Last delivered PID acceleration request; not actual load or measured AoA',
        results=summaries, comparisons=comparisons,
        verification=dict(per_case=checks, direct_switch_reproduction_max_difference=reproduction,
            protected_inputs_unchanged=True, force_rhs_and_pid_blocks_unchanged=True),
        protected_input_hashes=protected, runtime_defaults_changed=False,
        measured_input_use='Fixed first-TRK schedule only; no kinematic state injection; observations used for comparison')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    capture.write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),
        [dict(handover_duration_s=d, **r) for d, result in results.items() for r in result.samples])
    capture.write_table(prefix.with_name(prefix.name+'_comparison.tsv'), comparisons)
    plot(results, old, observed, start, prefix)
    print(json.dumps(summaries, indent=2))
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
