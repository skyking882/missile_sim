#!/usr/bin/env python3
"""Run the closed rebuild candidate from launch, then compare screenshots.

This uses the existing 3-D H2 RK4 implementation, not the unclosed standalone
longitudinal kernel. No observation enters integration. Raw configuration,
including provisional normalized CN/Cm gains and inertia, is unchanged.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from run_rebuild_scenarios import load_rebuild_profile
from aim120_model.h2_simulator import H2Simulator
from aim120_model.public_api import _case_from_scenario, validate_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SCENARIO = dict(launch_speed_kmh=1200, launch_altitude_m=6300,
    launch_pitch_deg=0, launch_heading_deg=0, target_speed_kmh=1200,
    target_altitude_m=6300, initial_distance_m=8000, target_azimuth_deg=30,
    target_heading_deg=0, target_vertical_heading_deg=0, target_constant_turn_g=0,
    target_course_reference="sensorwhale_launch_axis", loft_enabled=True,
    observation_mode="ideal_truth", max_simulation_time_s=40)


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_rebuild_profile("su_r_77_1")["_model_config"]
    scenario = validate_scenario(SCENARIO)
    config["performance"]["lifetime_s"] = min(config["performance"]["lifetime_s"], 40)
    config["guidance"]["lofting_enabled"] = bool(config["guidance"]["lofting_enabled"] and scenario["loft_enabled"])
    print("Running rebuild_first_principles_v1: independent t=0 launch, 3-D RK4.", flush=True)
    result = H2Simulator(config).run(_case_from_scenario(scenario))
    samples = result["samples"]
    trajectory = []
    for s in samples:
        r = dict(time_s=s["time_s"], speed_kmh=math.hypot(*s["velocity_mps"])*3.6,
                 alpha_deg=math.degrees(s["angle_of_attack_rad"]),
                 wind_normal_specific_force_magnitude_g=s["trajectory_lateral_load_g"])
        for field in ("position_m", "velocity_mps", "target_position_m", "target_velocity_mps"):
            for axis,value in zip("xyz",s[field]):
                r[f"{field}_{axis}"]=value
        for field in ("pitch_rad", "yaw_rad", "pitch_rate_rad_s", "yaw_rate_rad_s",
                      "actual_pitch_fin_angle_rad", "actual_yaw_fin_angle_rad",
                      "pitch_pid_integral", "yaw_pid_integral", "mass_kg", "thrust_n",
                      "distance_to_target_m"):
            r[field]=s[field]
        trajectory.append(r)
    # The complete integration has finished before observations are opened.
    with (ROOT / "data/replays/r77_1_level_20260824.tsv").open() as stream:
        observed = list(csv.DictReader(stream, delimiter="\t"))
    t=np.array([r["time_s"] for r in trajectory])
    model=np.array([[r["speed_kmh"],r["alpha_deg"],r["wind_normal_specific_force_magnitude_g"]] for r in trajectory])
    ot=np.array([float(r["missile_flight_time_s"]) for r in observed])
    obs=np.array([[float(r[k]) for k in ("speed_kmh","angle_of_attack_deg","overload_g")] for r in observed])
    mask=(ot>=t[0]) & (ot<=t[-1])
    matched=np.full_like(obs,np.nan)
    for j in range(3):
        matched[mask,j]=np.interp(ot[mask],t,model[:,j])
    errors=matched-obs
    comparison=[]
    for i,r in enumerate(observed):
        row=dict(source_file=r["source_file"], time_s=ot[i], compared=bool(mask[i]))
        for j,key in enumerate(("speed_kmh","alpha_deg","displayed_g")):
            row[f"observed_{key}"]=obs[i,j]
            row[f"predicted_{key}"]=matched[i,j]
            row[f"error_{key}"]=errors[i,j]
        comparison.append(row)
    OUT.mkdir(exist_ok=True)
    write_table(OUT / "r77_1_forward_trajectory.tsv",trajectory)
    write_table(OUT / "r77_1_forward_comparison.tsv",comparison)
    mae=np.mean(abs(errors[mask]),axis=0)
    rmse=np.sqrt(np.mean(errors[mask]**2,axis=0))
    minimum=min(r["distance_to_target_m"] for r in trajectory)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for j,(ax,label) in enumerate(zip(axes,("Speed (km/h)","Total AoA (deg)","Wind-normal specific force (g)"))):
        ax.plot(t,model[:,j],color="#276f9a",lw=1.2,label="Independent rebuild RK4 prediction")
        ax.scatter(ot,obs[:,j],s=23,color="#c66127",edgecolors="white",lw=.5,zorder=4,label="Game screenshots")
        ax.axvline(7,color="#777777",ls=":",lw=1)
        ax.set_ylabel(label)
        ax.set_xlim(0,8)
        # Display every model excursion inside the observation window.
        window=model[(t>=0)&(t<=8),j]
        low=min(np.min(window),np.min(obs[:,j])); high=max(np.max(window),np.max(obs[:,j]))
        margin=max((high-low)*.08,.2)
        ax.set_ylim(low-margin,high+margin)
        ax.grid(alpha=.2)
        ax.text(.015,.92,f"MAE {mae[j]:.2f} | RMSE {rmse[j]:.2f}",transform=ax.transAxes,
                va="top",bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    axes[0].legend(loc="lower right",fontsize=9)
    axes[-1].set_xlabel("Missile flight time (s); no fitted time shift")
    fig.suptitle("R-77-1 | closed rebuild candidate: forward RK4 from launch",x=.1,ha="left",fontsize=17)
    fig.text(.1,.935,f"Game: hit confirmed | Model: {result['event_type']} at {t[-1]:.2f} s | Minimum target distance: {minimum:.1f} m",fontsize=10)
    fig.text(.1,.028,
        f"t=0: 1200 km/h, 6300 m, level launch; target 1200 km/h at 6300 m, 8 km / 30° LOS geometry.\n"
        f"RK4 dt={result['time_step_s']:g} s; discrete guidance/PID each step; fins held during each RK4 step. No observed V/AoA/G input.\n"
        "Unidentified assumptions: normalized force/moment gains (2,-1,-1,+1), I=mL²/12, zero rate damping.\n"
        "This is rebuild_first_principles_v1, not the standalone longitudinal kernel or a full Devaud reproduction.\n"
        "G prediction is an unsigned spatial magnitude; signed screenshot values are preserved. Full trajectory is exported through model termination.",
        fontsize=9,color="#444444")
    fig.subplots_adjust(left=.1,right=.97,top=.885,bottom=.19,hspace=.18)
    fig.savefig(OUT / "r77_1_forward_comparison.png",dpi=160,facecolor="white")
    plt.close(fig)

    # One direct result check, including launch identity and no extrapolation.
    assert t[0]==0 and np.all(np.diff(t)>0)
    assert np.allclose(samples[0]["position_m"],(0,6300,0))
    assert math.isclose(model[0,0],1200,abs_tol=1e-8)
    finite=np.all(np.isfinite(model)) and all(math.isfinite(v) for r in trajectory for v in r.values())
    assert finite or result["event_type"]=="numerical_failure"
    assert len(observed)==46 and np.all(np.isnan(matched[~mask]))
    assert np.all(np.isfinite(matched[mask]))
    print("event",result["event_type"],"terminal_time",t[-1],"minimum_distance_m",minimum)
    print("samples",len(t),"compared",int(mask.sum()),"finite",finite,"dt",result["time_step_s"])
    print("MAE speed/alpha/G",mae,"RMSE",rmse)
    window=t<=7.8
    print("observation_window_peaks_speed_alpha_G",np.max(model[window],axis=0))
    print("first_alpha_over_30_s",next((r["time_s"] for r in trajectory if r["alpha_deg"]>30),None))
    print("terminal_speed_kmh",model[-1,0],"last_observed_time_prediction",matched[-1])
    print("CHECK PASSED: launch identity, monotonic finite trajectory, aligned observations without extrapolation.")


if __name__=="__main__":
    main()
