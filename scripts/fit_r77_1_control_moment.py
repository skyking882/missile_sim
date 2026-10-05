"""Fit only the angle-to-control-moment gain of the frozen focal plant.

Free forward runs; no replay state is fed into dynamics. Minimize the largest
AoA residual at native 0.2--3.5 s timestamps, and separately assess the early
peak/dip/recovery signs. All other coefficients and the normalized-to-radian
input mapping remain fixed. Results are calibration, not game semantics.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math

import numpy as np

from check_r77_1_pid_angle_output import build_angle_runner, time_windows
from run_r77_1_focus_moment import (
    ROOT, FocusPlant, inputs, CASES, protected_hashes,
    load_observations, VERSION, COMMIT, write_table, summarize,
)


def main():
    before = protected_hashes()
    config, baseline, raw_limit = inputs()
    _, obs = load_observations()
    base_values = asdict(baseline)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    actual = np.degrees(obs['alpha'])
    cache = {}
    summaries = []
    digests = set()

    def forward(area, step=.002):
        p = replace(baseline, fin_force_area_per_rad=float(area))
        assert {k:v for k,v in asdict(p).items() if k != 'fin_force_area_per_rad'} == {
            k:v for k,v in base_values.items() if k != 'fin_force_area_per_rad'}
        plant = FocusPlant(config, p, 'legacy_control_scale', raw_limit)
        runner, digest = build_angle_runner(plant, 'normalized_raw_radians', raw_limit)
        digests.add(digest)
        return runner(config, p, CASES[0][3], max_time_s=3.5, plant_dt_s=step)

    def evaluate(area):
        area = round(float(area), 9)
        if area in cache:
            return cache[area]
        result = forward(area)
        rows = result.samples
        valid = abs(rows[-1]['time_s']-3.5) < 1e-9 and max(r['alpha_deg'] for r in rows) < 80.
        t = np.array([r['time_s'] for r in rows])
        alpha = np.array([r['alpha_deg'] for r in rows])
        pred = np.interp(obs['time'], t, alpha)
        residual = pred-actual
        a07,a09,a11,a13,a15,a19 = np.interp([.7,.9,1.1,1.3,1.5,1.9], t, alpha)
        # Native-frame shape constraints, with no smoothing or shifted times.
        topology = bool(a09 > a07 and a09 > a11 and a13 > a11)
        active = np.flatnonzero(t >= .5)
        peak = int(active[np.argmax(alpha[active])])
        entry = dict(
            control_force_area_per_rad=area,
            multiplier=area/baseline.fin_force_area_per_rad,
            valid=bool(valid),
            worst_alpha_error_deg=float(max(abs(residual))),
            worst_error_time_s=float(obs['time'][np.argmax(abs(residual))]),
            early_topology_matches=topology,
            alpha_rise_0p7_to_0p9_deg=float(a09-a07),
            alpha_change_0p9_to_1p1_deg=float(a11-a09),
            alpha_change_1p1_to_1p3_deg=float(a13-a11),
            alpha_0p7_deg=float(a07), alpha_0p9_deg=float(a09),
            alpha_1p1_deg=float(a11), alpha_1p3_deg=float(a13),
            alpha_1p5_deg=float(a15), alpha_1p9_deg=float(a19),
            peak_alpha_deg=float(alpha[peak]), peak_time_s=float(t[peak]),
            capture_exit_time_s=next((r['time_s'] for r in rows if not r['capture_active']), None),
            angle_saturation_windows_s=time_windows(rows, lambda r: abs(r['pid_raw_output']) > 1.),
            force_saturation_windows_s=time_windows(rows, lambda r: r['fin_control_force_saturated']),
            event=result.event, end_time_s=float(t[-1]),
        )
        cache[area] = (entry, result)
        summaries.append(entry)
        return cache[area]

    # A declared broad interval, spanning 128x in the sole fitted coefficient.
    grid = baseline.fin_force_area_per_rad*np.geomspace(.125,16.,49)
    for i, area in enumerate(grid):
        evaluate(area)
        if i % 16 == 0:
            print(f'Coarse scan: {i+1}/{len(grid)}', flush=True)
    original = evaluate(baseline.fin_force_area_per_rad)
    coarse = [evaluate(a)[0] for a in grid]
    indices = sorted(range(len(grid)), key=lambda i: coarse[i]['worst_alpha_error_deg']
                     if coarse[i]['valid'] else math.inf)
    centers = []
    for i in indices:
        if all(abs(i-j) > 1 for j in centers):
            centers.append(i)
        if len(centers) == 2:
            break
    # Capture release is sampled at 20 ms, so use explicit local grids rather
    # than assume a differentiable or unimodal closed-loop objective.
    for i in centers:
        for area in np.linspace(grid[max(0,i-1)], grid[min(len(grid)-1,i+1)], 61):
            evaluate(area)
        print(f'Refined neighborhood around A_delta={grid[i]:.4f}', flush=True)
    valid = [r for r in summaries if r['valid']]
    best = min(valid, key=lambda r:r['worst_alpha_error_deg'])
    topology_pool = [r for r in valid if r['early_topology_matches']]
    if topology_pool:
        topo = min(topology_pool, key=lambda r:r['worst_alpha_error_deg'])
        closest = int(np.argmin(abs(grid-topo['control_force_area_per_rad'])))
        for area in np.linspace(grid[max(0,closest-1)],grid[min(len(grid)-1,closest+1)],61):
            evaluate(area)
        valid = [r for r in summaries if r['valid']]
        best = min(valid, key=lambda r:r['worst_alpha_error_deg'])
        topology_pool = [r for r in valid if r['early_topology_matches']]
        topo = min(topology_pool, key=lambda r:r['worst_alpha_error_deg'])
    else:
        topo = None
    selected = {'previous_gain': original[0], 'minimax_AoA': best}
    if topo is not None and topo['control_force_area_per_rad'] != best['control_force_area_per_rad']:
        selected['early_shape_only'] = topo
    # Show the starting-response tradeoff without fitting another coefficient.
    onset = min(valid,key=lambda r:abs(r['alpha_0p7_deg']-float(np.interp(.7,obs['time'],actual))))
    selected['match_0p7_only'] = onset

    verification = {}
    for name, entry in selected.items():
        area = entry['control_force_area_per_rad']
        result = evaluate(area)[1]
        fine = forward(area,.001)
        assert result.event == fine.event
        assert len(result.samples) == len(fine.samples)
        errors = {field:max(abs(a[field]-b[field]) for a,b in zip(result.samples,fine.samples))
                  for field in ('alpha_deg','wind_normal_specific_force_g','speed_kmh')}
        assert max(errors.values()) < .01,(name, errors)
        rows = result.samples
        assert all(r['rate_damping_moment_nm']*r['body_rate_rad_s'] <= 1e-8 for r in rows)
        error = max(abs(r['horizontal_fin_deflection_rad']-
                        np.clip(r['pid_raw_output'],-1,1)*raw_limit) for r in rows)
        assert error < 1e-12
        verification[name] = dict(half_step_max_difference=errors, angle_mapping_residual_rad=float(error))
    assert len(digests) == 1
    assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == config_hash
    assert protected_hashes() == before

    prefix = ROOT/'outputs/r77_1_control_moment_fit'
    frame_results = {name:summarize(evaluate(row['control_force_area_per_rad'])[1],obs)
                     for name,row in selected.items()}
    report = dict(
        source_version=VERSION, source_commit=COMMIT,
        replay_sha256=hashlib.sha256((ROOT/'data/replays/r77_1_level_20260824.tsv').read_bytes()).hexdigest(),
        fitted_variable='fin_force_area_per_rad only',
        objective='minimize maximum absolute AoA residual at all native 0.2--3.5 s replay timestamps',
        fit_window_s=[.2,3.5], coefficient_interval_m2_per_rad=[float(grid[0]),float(grid[-1])],
        early_shape_check='alpha(0.7)<alpha(0.9)>alpha(1.1)<alpha(1.3); diagnostic, not the scalar objective',
        frozen_parameters=base_values, frozen_configuration_sha256=config_hash,
        controller_block_sha256=next(iter(digests)),
        selected=selected, comparisons=frame_results, verification=verification,
        scan=sorted(summaries,key=lambda r:r['control_force_area_per_rad']),
        scan_count=len(cache), valid_scan_count=len(valid),
        topology_matching_count=len(topology_pool),
        observations_are_calibration_data_not_holdout=True,
        coefficient_is_effective_not_identified_game_derivative=True,
        runtime_defaults_changed=False, protected_inputs_unchanged=True,
    )
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_table(prefix.with_suffix('.tsv'),[dict(case=name,**r) for name,s in frame_results.items() for r in s['frames']])
    write_table(prefix.with_name(prefix.name+'_trajectory.tsv'),[
        dict(case=name,**r) for name,entry in selected.items()
        for r in evaluate(entry['control_force_area_per_rad'])[1].samples])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'previous_gain':('#929894','--','Previous gain'),
              'minimax_AoA':('#137d9a','-','Smallest worst AoA error'),
              'early_shape_only':('#9866a4',':','Early peak/dip signs'),
              'match_0p7_only':('#b87925','--','Closest to 0.7 s AoA')}
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,field,observed,label in zip(axes.flat,
        ('alpha_deg','wind_normal_specific_force_g','speed_kmh'),
        (actual,obs['load'],obs['speed']*3.6),('AoA [deg]','Total normal load [g]','Speed [km/h]')):
        ax.plot(obs['time'],observed,'o-',color='#222',lw=1,ms=4,label='2.57 replay',zorder=5)
        for name,entry in selected.items():
            color,ls,title=styles[name]
            rows=evaluate(entry['control_force_area_per_rad'])[1].samples
            ax.plot([r['time_s'] for r in rows],[r[field] for r in rows],
                color=color,ls=ls,lw=1.7,label=f"{title}: A={entry['control_force_area_per_rad']:.3f}")
        ax.set(xlabel='Flight time [s]',ylabel=label,xlim=(.2,3.5))
        ax.axvspan(.9,1.3,color='#e5e9ed',alpha=.4,zorder=0)
    scan=sorted(valid,key=lambda r:r['control_force_area_per_rad'])
    ax=axes[1,1]
    ax.plot([r['control_force_area_per_rad'] for r in scan],[r['worst_alpha_error_deg'] for r in scan],color='#666',lw=1)
    for name,row in selected.items():
        ax.plot(row['control_force_area_per_rad'],row['worst_alpha_error_deg'],'o',color=styles[name][0],ms=6)
    ax.set(xlabel='Control-force derivative A_delta [m²/rad]',ylabel='Worst sampled AoA error [deg]',xscale='log',
           title='One coefficient varied; native timestamps')
    for ax in axes.flat:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False,fontsize=9.5)
    fig.suptitle('R-77-1 | fit only control-moment gain | frozen 2.57.1.107',fontsize=16,y=.986)
    fig.text(.08,.025,
        'delta = clip(PID, +/-1) x raw finsAoa [rad]. PID, damping, inertia, guidance, lift and drag are fixed.\n'
        'Only AoA is fitted. Load and speed are forward comparisons. Peak/dip topology is assessed separately.\n'
        'Same replay used for calibration; no independent validation. No time shift or AoA display scaling.',fontsize=9,color='#444')
    fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.15,hspace=.3,wspace=.23)
    fig.savefig(prefix.with_suffix('.png'),dpi=160,facecolor='white');plt.close(fig)
    print(json.dumps(dict(selected=selected,scan_count=len(cache),topology_matching_count=len(topology_pool),verification=verification),indent=2),flush=True)
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
