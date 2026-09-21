"""Isolate capture request and handover reset on the frozen fitted plant.

The four cases form a two-by-two comparison. Only the capture-phase request
and the one-time command reset may change. PID code, force/moment plant,
coefficients, initial state and integration schedule are shared. No fitting.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math

import numpy as np

from run_r77_1_focus_moment import (
    ROOT, FocusPlant, alpha_servo, inputs, CASES, protected_hashes,
    load_observations, VERSION, COMMIT, write_table, summarize,
)
from check_r77_1_pid_angle_output import time_windows


VARIANTS = {
    "baseline": dict(pn_capture=False, retain_command=False),
    "continuous_handover": dict(pn_capture=False, retain_command=True),
    "pn_capture_only": dict(pn_capture=True, retain_command=False),
    "pn_and_continuous": dict(pn_capture=True, retain_command=True),
}
FIELDS = ("alpha_deg", "wind_normal_specific_force_g", "speed_kmh",
          "horizontal_fin_deflection_rad", "pid_raw_output",
          "total_aerodynamic_moment_nm")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_runner(plant, pn_capture, retain_command, request_gain_table=(), pn_only=False,
                 timed_capture=None, restart=None):
    original = inspect.getsource(alpha_servo.run_alpha_servo)
    source = original
    edits = []
    if pn_only or timed_capture is not None:
        assert not pn_capture and not retain_command
        assert not (pn_only and timed_capture is not None)
        start = '            a_cmd = 0.0\n'
        end = '            max_a = math.sqrt'
        guidance = original[original.index(start):original.index(end)]
        if timed_capture is None:
            replacement = (
                '            a_cmd = 0.0\n'
                '            if time_s >= p.guidance_delay_s - 1e-10:\n'
                '                a_cmd = p.navigation_gain * max(closing, 0.0) * los_rate\n')
        else:
            assert request_gain_table
            cap_g, duration = timed_capture
            assert 0. <= cap_g <= plant.p.command_limit_g and duration >= 0.
            replacement = (
                '            a_cmd = 0.0\n'
                '            capturing = capture_start - 1e-10 <= time_s < capture_end - 1e-10\n'
                '            if time_s >= p.guidance_delay_s - 1e-10:\n'
                '                a_cmd = p.navigation_gain * max(closing, 0.0) * los_rate\n'
                '                if capturing:\n'
                '                    transverse_target_speed = (rx * target_v[1] - rz * target_v[0]) / max(distance, 1e-9)\n'
                '                    lead = math.asin(max(-1.0, min(1.0, transverse_target_speed / speed)))\n'
                '                    error = (math.atan2(rz, rx) + lead - heading + math.pi) % (2 * math.pi) - math.pi\n'
                '                    capture_a = speed * math.sin(error) / p.capture_time_constant_s\n'
                '                    a_cmd = max(-capture_cap_g * G0, min(capture_cap_g * G0, capture_a))\n')
        edits.append((guidance, replacement))
        handover = (
            '            if capturing:\n'
            '                command = desired_alpha\n'
            '            elif time_s > release_t + 1e-10:\n'
            '                fraction = -math.expm1(-control_dt_s / p.release_time_constant_s)\n'
            '                command += fraction * (desired_alpha - command)\n')
        edits.append((handover, '            command = desired_alpha\n'))
        edits.append(('    capturing = True\n', '    capturing = False\n'))
    if pn_capture:
        edits.append(("                if capturing:\n                    a_cmd = capture_a\n",
                      "                if capturing:\n                    a_cmd = pn\n"))
    if retain_command:
        edits.append((
            "                    # Begin handover from delivered incidence, not an\n"
            "                    # unreachable acceleration-derived angle request.\n"
            "                    command = (state[4] - state[1] + math.pi) % (2 * math.pi) - math.pi\n",
            "                    # Hold the preceding request at the handover tick.\n"))
    if request_gain_table:
        # Optional launch-schedule diagnostic. Zero requested load does not
        # disable the PID or force its control angle to zero.
        anchor = '            max_a = math.sqrt(max((G0 * p.command_limit_g) ** 2 - gravity ** 2, 0.0))\n'
        edits.append((anchor, '            a_cmd *= request_gain(time_s)\n' + anchor))
    if restart is not None:
        assert pn_only and not retain_command and timed_capture is None
        anchor = '    samples: list[dict[str, float]] = []\n'
        edits.append((anchor, anchor +
            '    state = tuple(restart["state"])\n'
            '    time_s = restart["time_s"]\n'
            '    assert len(state) == 6 and all(math.isfinite(v) for v in state)\n'
            '    assert 0. <= time_s < end_time\n'
            '    tick = round(time_s / control_dt_s)\n'
            '    assert abs(tick * control_dt_s - time_s) < 1e-10\n'
            '    pid_integral = restart["pid_integral"]\n'
            '    previous_error = restart["previous_error"]\n'
            '    fin_deflection = restart.get("fin_deflection", 0.)\n'
            '    command = restart.get("command", 0.)\n'
            '    commanded_acceleration_g = restart.get("commanded_acceleration_g", 0.)\n'
            '    pid_proportional = restart.get("pid_proportional", 0.)\n'
            '    pid_derivative = restart.get("pid_derivative", 0.)\n'
            '    minimum = math.hypot(target0[0]+time_s*target_v[0]-state[2],\n'
            '                         target0[1]+time_s*target_v[1]-state[3])\n'
            '    if restart.get("after_control", False):\n'
            '        tick += 1\n'))
        anchor = '    while time_s < end_time - 1e-10:\n'
        edits.append((anchor,
            '    if restart.get("after_control", False):\n'
            '        record(time_s, state)\n' + anchor))
    for old, new in edits:
        assert source.count(old) == 1
        source = source.replace(old, new)
    restored = source
    for old, new in reversed(edits):
        restored = restored.replace(new, old)
    assert restored == original
    pid_start = '                # One PID: error in wind-normal g, output in fin-travel'
    pid_end = '            record(time_s, state)'
    pid = original[original.index(pid_start):original.index(pid_end)]
    assert source[source.index(pid_start):source.index(pid_end)] == pid

    # Use exactly the fitted focal plant instead of the core default plant.
    assert source.count('fin_deflection, p)') == 2
    source = source.replace('fin_moment_sample(s[0]', 'focus_sample(s[0]')
    source = source.replace('fin_deflection, p)', 'fin_deflection, p, alpha)')
    source = source.replace('        samples.append(row)',
        '        row.update(pid_raw_output=pid_proportional+pid_integral+pid_derivative,\n'
        '                   capture_active=bool(capturing))\n'
        '        samples.append(row)')
    def request_gain(time_s):
        return float(np.interp(time_s, [r[0] for r in request_gain_table],
                               [r[1] for r in request_gain_table]))
    namespace = dict(vars(alpha_servo), focus_sample=plant.sample, request_gain=request_gain)
    if restart is not None:
        namespace['restart'] = dict(restart)
    if timed_capture is not None:
        namespace.update(capture_cap_g=cap_g, capture_start=request_gain_table[0][0],
                         capture_end=round(request_gain_table[0][0]+duration, 10))
    exec(compile(source, '<capture-strategy diagnostic>', 'exec'), namespace)
    return namespace['run_alpha_servo'], hashlib.sha256(pid.encode()).hexdigest()


def enrich(rows, p, scenario=None):
    _, (tvx, tvz) = alpha_servo.planar_target_motion(CASES[0][3] if scenario is None else scenario)
    for row in rows:
        v, heading = row['speed_kmh']/3.6, row['heading_rad']
        rx, rz = row['target_x_m']-row['x_m'], row['target_z_m']-row['z_m']
        distance = math.hypot(rx, rz)
        rvx, rvz = tvx-v*math.cos(heading), tvz-v*math.sin(heading)
        closing = -(rx*rvx+rz*rvz)/distance
        los_rate = (rx*rvz-rz*rvx)/distance**2
        lead = math.asin(np.clip((rx*tvz-rz*tvx)/distance/v, -1., 1.))
        error = (math.atan2(rz, rx)+lead-heading+math.pi)%(2*math.pi)-math.pi
        row.update(
            capture_heading_error_deg=math.degrees(error),
            raw_capture_request_g=v*math.sin(error)/p.capture_time_constant_s/alpha_servo.G0,
            raw_pn_request_g=p.navigation_gain*max(closing, 0.)*los_rate/alpha_servo.G0,
            pid_error_g=row['commanded_horizontal_acceleration_g']-row['horizontal_specific_force_g'],
            horizontal_alpha_rate_deg_s=math.degrees(row['body_rate_rad_s']
                -row['horizontal_specific_force_g']*alpha_servo.G0/v),
        )


def analyze(result, obs):
    rows = result.samples
    report = summarize(result, obs)
    t = np.array([r['time_s'] for r in rows])
    alpha = np.array([r['alpha_deg'] for r in rows])
    a07, a09, a11, a13, a15 = np.interp([.7, .9, 1.1, 1.3, 1.5], t, alpha)
    report.update(
        alpha_at_native_times_deg=dict(zip(('0.7', '0.9', '1.1', '1.3', '1.5'),
                                          map(float, (a07, a09, a11, a13, a15)))),
        alpha_change_0p9_to_1p1_deg=float(a11-a09),
        alpha_change_1p1_to_1p3_deg=float(a13-a11),
        early_topology_matches=bool(a07 < a09 > a11 < a13),
        angle_saturation_windows_s=time_windows(rows, lambda r: abs(r['pid_raw_output']) > 1.),
        force_saturation_windows_s=time_windows(rows, lambda r: r['fin_control_force_saturated']),
        request_cap_windows_s=time_windows(rows, lambda r:
            abs(r['commanded_horizontal_acceleration_g']) >= math.sqrt(50**2-1)-1e-8),
        capture_exit_time_s=None, handover_samples=[],
    )
    for i, row in enumerate(rows):
        if not row['capture_active']:
            report['capture_exit_time_s'] = row['time_s']
            report['handover_samples'] = rows[max(i-1, 0):i+2]
            break
    return report


def plot_results(results, obs, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'baseline': ('#70757b', '--', 'Existing capture + reset'),
        'continuous_handover': ('#147f9b', '-', 'Same capture; retain command'),
        'pn_capture_only': ('#c67b23', '-', 'PN during capture; keep reset'),
        'pn_and_continuous': ('#9565ad', ':', 'PN during capture; retain command'),
    }
    panels = (
        ('alpha_deg', 'AoA magnitude [deg]', np.degrees(obs['alpha'])),
        ('commanded_horizontal_acceleration_g', 'Requested horizontal load [g]', None),
        ('wind_normal_specific_force_g', 'Total normal load [g]', obs['load']),
        ('horizontal_fin_deflection_rad', 'Equivalent control angle [deg]', None),
        ('speed_kmh', 'Speed [km/h]', obs['speed']*3.6),
        ('total_aerodynamic_moment_nm', 'Net attitude moment [kNm]', None),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5))
    for ax, (field, label, observed) in zip(axes.flat, panels):
        if observed is not None:
            mask = (obs['time'] >= .4) & (obs['time'] <= 2.)
            ax.plot(obs['time'][mask], observed[mask], 'o-', color='#222',
                    lw=1., ms=4, label='2.57 replay', zorder=5)
        for name, (color, ls, title) in styles.items():
            rows = [r for r in results[name].samples if .4 <= r['time_s'] <= 2.]
            values = np.array([r[field] for r in rows])
            if field == 'horizontal_fin_deflection_rad':
                values = np.degrees(values)
            if field == 'total_aerodynamic_moment_nm':
                values = values/1000.
            ax.plot([r['time_s'] for r in rows], values, color=color, ls=ls,
                    lw=1.8, label=title)
        ax.set(xlim=(.4, 2.), xlabel='Flight time [s]', ylabel=label)
        ax.axvspan(.9, 1.1, color='#e5e9ed', alpha=.45, zorder=0)
        ax.axvline(1.44, color='#777', lw=.8, ls=':', alpha=.65)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .947),
               ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle('R-77-1 | capture strategy isolation | frozen plant and PID', fontsize=16, y=.985)
    fig.text(.08, .027,
        '2.57.1.107; A_delta = 2.73404 m²/rad. All plant coefficients and raw PID terms held fixed; no refit.\n'
        'Shading: measured 0.9–1.1 s decline. Dotted vertical line: original handover at 1.44 s.\n'
        'PN-only request tests capture authority; it is not claimed as the game guidance law. Full traces run to 3.5 s.',
        fontsize=9, color='#444')
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.13, hspace=.32, wspace=.24)
    fig.savefig(prefix.with_suffix('.png'), dpi=160, facecolor='white')
    plt.close(fig)


def main():
    fit_path = ROOT/'outputs/r77_1_control_moment_fit.json'
    old_trajectory = ROOT/'outputs/r77_1_control_moment_fit_trajectory.tsv'
    protected = protected_hashes()
    extra = [fit_path, old_trajectory, ROOT/'scripts/run_r77_1_focus_moment.py']
    extra_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extra}
    config, p0, raw_limit = inputs()
    fit = json.loads(fit_path.read_text())
    assert (fit['source_version'], fit['source_commit']) == (VERSION, COMMIT)
    assert asdict(p0) == fit['frozen_parameters']
    assert digest(config) == fit['frozen_configuration_sha256']
    p = replace(p0, fin_force_area_per_rad=fit['selected']['minimax_AoA']['control_force_area_per_rad'])
    _, obs = load_observations()
    assert hashlib.sha256((ROOT/'data/replays/r77_1_level_20260824.tsv').read_bytes()).hexdigest() == fit['replay_sha256']

    results, runners, pid_hashes, summaries = {}, {}, set(), {}
    plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
    for name, options in VARIANTS.items():
        runner, pid_hash = build_runner(plant, **options)
        runners[name] = runner
        pid_hashes.add(pid_hash)
        results[name] = runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.002)
        enrich(results[name].samples, p)
        summaries[name] = analyze(results[name], obs)
        print(name, json.dumps({k: summaries[name][k] for k in (
            'capture_exit_time_s', 'alpha_at_native_times_deg',
            'alpha_change_0p9_to_1p1_deg', 'early_topology_matches', 'extrema')}), flush=True)

    # One final verification: baseline receipt, frozen code/inputs, and time-step sensitivity.
    assert len(pid_hashes) == 1
    with old_trajectory.open() as stream:
        old = [r for r in csv.DictReader(stream, delimiter='\t') if r['case'] == 'minimax_AoA']
    assert len(old) == len(results['baseline'].samples)
    reproduction = {field: max(abs(float(a[field])-b[field])
        for a, b in zip(old, results['baseline'].samples)) for field in FIELDS}
    assert max(reproduction.values()) < 1e-9, reproduction
    preswitch = {field: max(abs(a[field]-b[field]) for a, b in zip(
        results['baseline'].samples, results['continuous_handover'].samples)
        if a['time_s'] < summaries['baseline']['capture_exit_time_s']-1e-9) for field in FIELDS}
    assert max(preswitch.values()) == 0., preswitch
    verification = {}
    for name, result in results.items():
        fine = runners[name](config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=.001)
        assert result.event == fine.event
        assert len(result.samples) == len(fine.samples)
        assert abs(result.samples[-1]['time_s']-3.5) < 1e-9
        errors = {field: max(abs(a[field]-b[field]) for a, b in zip(result.samples, fine.samples))
                  for field in FIELDS}
        assert max(errors.values()) < .01, (name, errors)
        assert [r['capture_active'] for r in result.samples] == [r['capture_active'] for r in fine.samples]
        mapping = max(abs(r['horizontal_fin_deflection_rad']
            -float(np.clip(r['pid_raw_output'], -1., 1.))*raw_limit) for r in result.samples)
        assert mapping < 1e-12
        assert all(r['rate_damping_moment_nm']*r['body_rate_rad_s'] <= 1e-8 for r in result.samples)
        verification[name] = dict(half_step_max_difference=errors,
                                  angle_mapping_residual_rad=mapping)
    assert digest(config) == fit['frozen_configuration_sha256']
    assert protected_hashes() == protected
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == extra_hashes[str(path)] for path in extra)

    prefix = ROOT/'outputs/r77_1_capture_strategy'
    report = dict(source_version=VERSION, source_commit=COMMIT,
        fitted_parameters=[], fixed_parameters=asdict(p), configuration_sha256=digest(config),
        pid_block_sha256=next(iter(pid_hashes)), plant_sha256=extra_hashes[str(extra[-1])],
        replay_sha256=fit['replay_sha256'], baseline_fit_sha256=extra_hashes[str(fit_path)],
        variants=VARIANTS, results=summaries, baseline_reproduction_max_difference=reproduction,
        retain_command_preswitch_max_difference=preswitch, verification=verification,
        protected_inputs_unchanged=True, runtime_defaults_changed=False,
        replay_used_only_for_comparison=True, unknown_game_guidance=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'), [dict(case=name, **frame)
        for name, summary in summaries.items() for frame in summary['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'), [dict(case=name, **row)
        for name, result in results.items() for row in result.samples])
    plot_results(results, obs, prefix)
    print(json.dumps(dict(baseline_reproduction=reproduction,
                         preswitch_unchanged=preswitch, verification=verification)), flush=True)
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
