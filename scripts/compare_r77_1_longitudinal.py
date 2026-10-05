#!/usr/bin/env python3
"""R-77-1 conditional translation comparison; no CN/Cm fit or closed loop.

V, AoA and displayed wind-normal specific-force G are observed inputs.
Nominal altitude is 6300 m, gamma=0 for the axial gravity term, and AoA/G
are treated as coplanar, aligned values. This is not a recovered 3-D plane.
Both clocks are anchored at the first frame (0.2 s); no timing fit is made.
The existing drag-area law is held fixed, once interpreted as wind D and
once as body A. These are competing field semantics, not a coordinate test.
Reconstructed forces are observational constraints, not predictions.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from run_rebuild_scenarios import load_rebuild_profile
from aim120_model.atmosphere import StandardAtmosphere
from aim120_model.drag_models import effective_cda0, effective_cda_alpha
from aim120_model.longitudinal_dynamics import (
    BodyAeroCoefficients, LongitudinalProperties, LongitudinalState,
    evaluate_longitudinal,
)
from aim120_model.propulsion import PiecewisePropulsion
from aim120_model.units import G0_MPS2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def compare(clock, time, speed, alpha, load, config):
    air = StandardAtmosphere().sample(6300.0)
    prop = PiecewisePropulsion.from_config(config)
    area = math.pi * config["geometry"]["caliber_m"]**2 / 4
    predictions = np.zeros((len(time), 2)) + speed[0] * 3.6
    rows = []
    for i, (lo, hi) in enumerate(zip(time[:-1], time[1:])):
        # Split at burnout exactly; use interval integrals instead of noisy
        # point derivatives. Observed acceleration is only an interval mean.
        edges = sorted({float(lo), float(hi), *([7.0] if lo < 7.0 < hi else [])})
        sums = np.zeros(8)
        observed_accel = (speed[i + 1] - speed[i]) / (hi - lo)
        for begin, end in zip(edges[:-1], edges[1:]):
            count = math.ceil((end - begin) / 0.005)
            dt = (end - begin) / count
            for t in begin + (np.arange(count) + 0.5) * dt:
                v, a, gload = (float(np.interp(t, time, x)) for x in (speed, alpha, load))
                p = prop.sample(float(t))
                qs = 0.5 * air.density_kg_m3 * v * v * area
                mach = v / air.speed_of_sound_mps
                force0 = qs / area * effective_cda0(mach, config)
                force_alpha = qs / area * effective_cda_alpha(a, 0.0, mach, config)
                fixed = force0 + force_alpha
                s, c = math.sin(a), math.cos(a)
                observed_lift = p.mass_kg * G0_MPS2 * gload - p.thrust_n * s
                observed_drag = p.thrust_n * c - p.mass_kg * observed_accel
                observed_axial = observed_drag * c - observed_lift * s
                observed_normal = observed_drag * s + observed_lift * c
                # I=1 and Cm=q=0 are unused interface values: only the two
                # translational force projections are consumed, never q_dot.
                properties = LongitudinalProperties(p.mass_kg, 1.0, area,
                    config["geometry"]["caliber_m"], air.density_kg_m3, p.thrust_n)
                state = LongitudinalState(v, 0.0, a, 0.0)
                as_wind = BodyAeroCoefficients((fixed*c-observed_lift*s)/qs,
                                              (fixed*s+observed_lift*c)/qs, 0.0)
                as_body = BodyAeroCoefficients(fixed/qs,
                                              (observed_lift+fixed*s)/(c*qs), 0.0)
                acc_d = evaluate_longitudinal(state, properties, as_wind).speed_dot_mps2
                acc_a = evaluate_longitudinal(state, properties, as_body).speed_dot_mps2
                sums += dt * np.array([acc_d, acc_a, fixed, observed_drag,
                                       observed_axial, observed_normal, force0, force_alpha])
        predictions[i + 1] = predictions[i] + 3.6*sums[:2]
        mean = sums / (hi-lo)
        rows.append(dict(clock=clock, start_s=lo, end_s=hi,
            observed_speed_end_kmh=speed[i+1]*3.6,
            observed_delta_v_kmh=(speed[i+1]-speed[i])*3.6,
            d_hypothesis_delta_v_kmh=sums[0]*3.6,
            a_hypothesis_delta_v_kmh=sums[1]*3.6,
            d_hypothesis_speed_end_kmh=predictions[i+1,0],
            a_hypothesis_speed_end_kmh=predictions[i+1,1],
            d_residual_acceleration_mps2=mean[0]-observed_accel,
            a_residual_acceleration_mps2=mean[1]-observed_accel,
            fixed_force_mean_n=mean[2], reconstructed_D_mean_n=mean[3],
            reconstructed_A_mean_n=mean[4], reconstructed_N_mean_n=mean[5],
            fixed_zero_aoa_force_mean_n=mean[6], fixed_aoa_force_mean_n=mean[7]))
    return predictions, rows


def main():
    with (ROOT / "data/replays/r77_1_level_20260824.tsv").open() as stream:
        source = list(csv.DictReader(stream, delimiter="\t"))
    t = np.array([float(r["missile_flight_time_s"]) for r in source])
    replay = np.array([float(r["replay_time_s"]) for r in source])
    tr = t[0] + replay - replay[0]
    v = np.array([float(r["speed_kmh"])/3.6 for r in source])
    a = np.radians([float(r["angle_of_attack_deg"]) for r in source])
    g = np.array([float(r["overload_g"]) for r in source])
    config = load_rebuild_profile("su_r_77_1")["_model_config"]
    # Only propulsion, geometry and existing drag are consumed from config.
    pd, native = compare("displayed_flight_time", t, v, a, g, config)
    pr, replay_rows = compare("replay_intervals_first_frame_anchor", tr, v, a, g, config)
    rows = native + replay_rows
    OUT.mkdir(exist_ok=True)
    with (OUT / "r77_1_longitudinal_comparison.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.8))
    blue, orange = "#276f9a", "#cf6a24"
    ax = axes[0,0]
    ax.scatter(t, v*3.6, s=17, c="black", label="Game screenshots", zorder=4)
    for k, (color, name) in enumerate(((blue,"Fixed law as wind D"),(orange,"Fixed law as body A"))):
        ax.plot(t, pd[:,k], color=color, label=name)
        ax.plot(tr, pr[:,k], color=color, ls="--", alpha=.6)
    ax.set(title="Conditional cumulative speed (not closed loop)", ylabel="Speed (km/h)")
    ax.legend(fontsize=8)
    mid = (t[1:]+t[:-1])/2
    ax = axes[0,1]
    for key,color,label in (("d_residual_acceleration_mps2",blue,"D hypothesis"),
                            ("a_residual_acceleration_mps2",orange,"A hypothesis")):
        ax.plot(mid, [r[key] for r in native], ".-", color=color, alpha=.8,label=label)
    ax.axhline(0,color="black",lw=.7)
    ax.set(title="Interval mean acceleration error",ylabel="Predicted - observed (m/s²)")
    ax.legend(fontsize=8)
    ax = axes[0,2]
    ax.plot(t,np.degrees(a),"o-",ms=3,color="#604495",label="Observed AoA (deg)")
    ax.plot(t,g,"s-",ms=3,color="#26805c",label="Observed displayed G")
    ax.set(title="Observations used as inputs",ylabel="Degrees / g")
    ax.legend(fontsize=8)
    for ax,key,title in ((axes[1,0],"reconstructed_D_mean_n","Wind-axis D: reconstruction vs fixed law"),
                         (axes[1,1],"reconstructed_A_mean_n","Body-axis A: reconstruction vs fixed law")):
        ax.plot(mid,np.array([r["fixed_force_mean_n"] for r in native])/1000,
                color=blue,label="Same fixed force value")
        ax.plot(mid,np.array([r[key] for r in native])/1000,"o",ms=3,
                color="black",label="From flight-time intervals")
        mr=(tr[1:]+tr[:-1])/2
        ax.plot(mr,np.array([r[key] for r in replay_rows])/1000,"x",ms=3,
                color=orange,label="From replay intervals")
        ax.axhline(0,color="black",lw=.7)
        ax.set(title=title,ylabel="Interval mean force (kN)")
        ax.legend(fontsize=8)
    ax=axes[1,2]
    ax.plot(mid,np.diff(t),"o-",ms=3,label="Displayed flight time")
    ax.plot(mid,np.diff(tr),"x-",ms=3,label="Replay timestamp")
    ax.set(title="Timing resolution: no fitted time shift",ylabel="Adjacent-frame interval (s)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(7,color="#888888",ls=":",lw=1)
        ax.set_xlabel("Time since launch (s)")
        ax.grid(alpha=.18)
    fig.suptitle("R-77-1 | observed-input longitudinal comparison",x=.06,ha="left",fontsize=19)
    fig.text(.06,.922,"Launch: 1200 km/h, 6300 m; target: 1200 km/h, 6300 m, 8 km / 30° LOS geometry. Game hit confirmed.",fontsize=10)
    fig.text(.06,.035,
        "V, AoA and G are measured inputs; cumulative curves start at the first frame (0.2 s, 1237 km/h). No CN/Cm, fin or controller fit.\n"
        "Assumptions: fixed 6300 m density, zero along-velocity gravity, coplanar aligned AoA/G. Reconstructed forces are not predictions.\n"
        "Dashed speed curves use replay intervals anchored to the same first frame. Dotted line: nominal 7 s burnout.\n"
        "Fixed law includes raw CxK=1.65 plus shared Mach shape and AoA² coefficient 9; the complete law is not a raw blkx coefficient.",
        fontsize=9,color="#444444")
    fig.subplots_adjust(left=.06,right=.98,top=.865,bottom=.17,hspace=.35,wspace=.28)
    fig.savefig(OUT / "r77_1_longitudinal_comparison.png",dpi=160,facecolor="white")
    plt.close(fig)

    # One direct output check: correct interval count, finite numbers, and
    # cumulative integration agrees with the exported interval increments.
    assert len(source)==46 and len(rows)==90
    assert all(math.isfinite(float(value)) for r in rows for key,value in r.items() if key!="clock")
    for prediction, series in ((pd,native),(pr,replay_rows)):
        for k,key in enumerate(("d_hypothesis_delta_v_kmh","a_hypothesis_delta_v_kmh")):
            assert np.allclose(np.diff(prediction[:,k]),[r[key] for r in series])
    for name,time,prediction,series in (("flight",t,pd,native),("replay",tr,pr,replay_rows)):
        errors=prediction-v[:,None]*3.6
        print(name,"last_time",round(time[-1],4),"MAE_kmh",np.mean(abs(errors),axis=0).round(2),
              "final_bias_kmh",errors[-1].round(2),"predicted_final_kmh",prediction[-1].round(2))
        for label,lo,hi in (("early",.2,.9),("large_AoA",.9,2.1),("release",2.1,4.1),
                            ("late_powered",4.1,7),("after_burnout",7,8)):
            selected=[r for r in series if r["start_s"]>=lo-1e-9 and r["end_s"]<=hi+1e-9]
            if selected:
                span=sum(r["end_s"]-r["start_s"] for r in selected)
                residual=[sum(r[f"{h}_hypothesis_delta_v_kmh"]-r["observed_delta_v_kmh"] for r in selected) for h in ("d","a")]
                print(label,"span",round(span,3),"deltaV_error_kmh",np.round(residual,2),
                      "mean_accel_error",np.round(np.array(residual)/3.6/span,2))
    print("clock_delta_range_s",float(min(tr-t)),float(max(tr-t)))
    print("CHECK PASSED: 46 observations, 90 finite interval rows, cumulative increments agree.")


if __name__=="__main__":
    main()
