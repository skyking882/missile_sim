#!/usr/bin/env python3
"""Plot existing JARI and free-model trajectories without running the model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prepare_r771_jari_only as ds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jari_only/trajectory_comparison'
CACHE = ROOT / 'outputs/jari_only/vector_restoring/free_predictions.npz'
ANGLES = (0, 5, 10, 15, 30, 45, 60)
ORANGE = '#dc6b13'


def interp_xyz(t, ts, xyz):
    if t[0] < ts[0] or t[-1] > ts[-1]:
        raise ValueError('Interpolation timestamps exceed stored support')
    return np.column_stack([np.interp(t, ts, xyz[:, j]) for j in range(3)])


def load_case(angle, cached):
    case = f'angle_p{angle}'
    _, source = ds.load_case(case)
    tm = cached[case + '__t']
    pm = cached[case + '__position_xyz_m']
    t, p, target = (source[k] for k in ('t', 'position_xyz_m', 'target_xyz_m'))
    m = interp_xyz(t, tm, pm)
    if not np.array_equal(p[0], m[0]):
        raise ValueError('Source and model initial positions differ')
    if not all(np.isfinite(a).all() for a in (t, p, m, target)):
        raise ValueError('Nonfinite trajectory data')
    delta = m-p
    errors = {'3d': np.linalg.norm(delta, axis=1),
              'horizontal_xz': np.linalg.norm(delta[:, [0, 2]], axis=1),
              'vertical_y': np.abs(delta[:, 1])}
    metrics = {}
    for window, mask in (('early_0_to_nominal_3p5s', t <= 3.5+1e-6),
                         ('full_common_support', np.ones(len(t), bool))):
        ids = np.flatnonzero(mask)
        metrics[window] = {'sample_count': len(ids), 'actual_last_time_s': float(t[ids[-1]])}
        for name, error in errors.items():
            peak = ids[np.argmax(error[mask])]
            metrics[window][name] = {'rms_m': float(np.sqrt(np.mean(error[mask]**2))),
                                     'max_m': float(error[peak]), 'max_time_s': float(t[peak])}
    checkpoints = []
    for time in (1., 2., 3.5, 7., float(t[-1])):
        a = interp_xyz(np.array([time]), t, p)[0]
        b = interp_xyz(np.array([time]), tm, pm)[0]
        d = b-a
        checkpoints.append({'time_s': time, 'source_xyz_m': a.tolist(), 'model_xyz_m': b.tolist(),
                            'model_minus_source_xyz_m': d.tolist(), 'distance_3d_m': float(np.linalg.norm(d))})
    stats = {'common_support_s': [float(t[0]), float(t[-1])], 'source_rows': len(t),
             'model_cache_support_s': [float(tm[0]), float(tm[-1])],
             'metrics': metrics, 'checkpoints': checkpoints,
             'altitude_m': {name: {'min': float(a[:, 1].min()), 'max': float(a[:, 1].max()),
                                  'min_relative_to_launch': float(a[:, 1].min()-p[0, 1]),
                                  'max_relative_to_launch': float(a[:, 1].max()-p[0, 1])}
                            for name, a in (('jari', p), ('model', m))},
             'end_delta_xyz_m': delta[-1].tolist(),
             'end_position_difference_3d_m': float(errors['3d'][-1]),
             'end_range_to_exported_target_m': {'jari': float(np.linalg.norm(p[-1]-target[-1])),
                                               'model': float(np.linalg.norm(m[-1]-target[-1]))}}
    return (t, p, m, target, errors), stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, stats = {}, {}
    with np.load(CACHE) as cached:
        for angle in ANGLES:
            arrays[angle], stats[f'angle_p{angle}'] = load_case(angle, cached)
    result = {'cases': stats, 'model_configuration': 'research/legacy/config/r771_jari_only_vector_restoring.json',
              'model_cache': str(CACHE.relative_to(ROOT)), 'source_dataset': 'outputs/jari_only/dataset',
              'comparison': 'Cached free model interpolated to each case original JARI timestamps; no extrapolation',
              'coordinates': 'World X/Z horizontal, world Y altitude; XYZ differences are model minus JARI',
              'early_window': 'Original samples through nominal 3.5 s, with 1e-6 s tolerance for stored clock rounding',
              'endpoint_note': 'Each panel ends at its JARI cutoff; not a model intercept or kill criterion'}
    (OUT/'smaller_angles_results.json').write_text(json.dumps(result, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.grid': True, 'grid.alpha': .22})
    for kind, filename, title in (
        ('plan', 'smaller_angles_xz.png', 'Horizontal trajectories — equal X/Z distance scale in every panel'),
        ('height', 'smaller_angles_height.png', 'Vertical trajectories — World Y relative to launch altitude (6300 m)'),
        ('error', 'smaller_angles_position_error.png', 'Position differences at the same time — cached free model vs JARI')):
        fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
        handles = labels = None
        for angle, ax in zip(ANGLES, axes.flat):
            t, p, m, target, errors = arrays[angle]
            s = stats[f'angle_p{angle}']
            if kind == 'plan':
                ax.plot(p[:, 0]/1000, p[:, 2]/1000, color='black', lw=2.2, label='JARI')
                ax.plot(m[:, 0]/1000, m[:, 2]/1000, color=ORANGE, lw=1.8, label='Free model')
                ax.plot(target[:, 0]/1000, target[:, 2]/1000, '--', color='#4c849c', lw=1.3, label='Target')
                for check in s['checkpoints']:
                    time = check['time_s']
                    if time not in (3.5, t[-1]):
                        continue
                    a, b = np.array(check['source_xyz_m']), np.array(check['model_xyz_m'])
                    ax.plot([a[0]/1000, b[0]/1000], [a[2]/1000, b[2]/1000], color='#8c9097', lw=1.2)
                    ax.scatter(a[0]/1000, a[2]/1000, color='black', s=20, zorder=4)
                    ax.scatter(b[0]/1000, b[2]/1000, color=ORANGE, s=20, zorder=4)
                    ax.annotate('3.5 s' if time == 3.5 else 'cutoff', (a[0]/1000, a[2]/1000),
                                xytext=(-6, 10), textcoords='offset points', fontsize=8.5)
                bounds = np.concatenate((p, m, target))[:, [0, 2]]/1000
                mid = (bounds.min(axis=0)+bounds.max(axis=0))/2
                span = max(np.ptp(bounds, axis=0))*1.14
                ax.set_xlim(mid[0]-span/2, mid[0]+span/2)
                ax.set_ylim(mid[1]-span/2, mid[1]+span/2)
                ax.set_aspect('equal', adjustable='box')
                ax.set(xlabel='World X (km)', ylabel='World Z (km)')
            elif kind == 'height':
                ax.plot(t, p[:, 1]-6300, color='black', lw=2.2, label='JARI')
                ax.plot(t, m[:, 1]-6300, color=ORANGE, lw=1.8, label='Free model')
                ax.axhline(0, color='#4c849c', ls='--', lw=1.2, label='Launch / target altitude')
                ax.set(xlabel='Time (s)', ylabel='Y − 6300 m (m)', xlim=(0, t[-1]))
            else:
                ax.plot(t, errors['3d'], color='#9343a2', lw=2.2, label='3D difference')
                ax.plot(t, errors['horizontal_xz'], color=ORANGE, ls='--', lw=1.5, label='Horizontal XZ difference')
                ax.plot(t, errors['vertical_y'], color='#3276a3', lw=1.5, label='Absolute Y difference')
                ax.set(xlabel='Time (s)', ylabel='Position difference (m)', xlim=(0, t[-1]))
                rms = s['metrics']['full_common_support']['3d']['rms_m']
                ax.text(.04, .94, f'RMS {rms:.1f} m\nCutoff {s["end_position_difference_3d_m"]:.1f} m',
                        va='top', transform=ax.transAxes, fontsize=9,
                        bbox={'facecolor': 'white', 'edgecolor': '#ccc', 'alpha': .9})
            ax.set_title(f'{angle}° offset | 0–{t[-1]:.3f} s')
            handles, labels = ax.get_legend_handles_labels()
        last = axes.flat[-1]
        last.axis('off')
        last.legend(handles, labels, loc='upper left', fontsize=11)
        notes = ('Markers / connectors: equal times\nCutoff: final JARI exported timestamp\n\nEach panel uses equal X/Z units.\nTarget offset changes; launch\naltitude and attitude stay fixed.' if kind == 'plan' else
                 'Vertical scales vary by panel.\n0° / 5° include loft climbs.\nSame initial altitude does not\nmake the full flight planar.' if kind == 'height' else
                 '3D = norm(model position − JARI)\nHorizontal = norm of X/Z difference\nVertical = absolute Y difference\n\nEach case ends at its own\nshared source cutoff.')
        last.text(.03, .55, notes, va='top', fontsize=11, transform=last.transAxes, linespacing=1.7)
        fig.suptitle(title+'\nSmaller target offsets: 0°, 5°, 10°, 15°, 30°, 45°, 60°; existing caches only', fontsize=15)
        fig.savefig(OUT/filename, dpi=170)
        plt.close(fig)
    for case, s in stats.items():
        print(case, 'RMS', s['metrics']['full_common_support']['3d']['rms_m'],
              'end', s['end_position_difference_3d_m'], 'deltaXYZ', s['end_delta_xyz_m'])


if __name__ == '__main__':
    main()
