#!/usr/bin/env python3
import math, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-skyking-Documents-missle-sim/e7247f0a-1f65-4c63-b4c5-632b9bb928cd/scratchpad")
from ablate import run, at, adeg, release, modes
import json
from pathlib import Path
PROJ = Path("/Users/skyking/Documents/missle_sim")
sys.path.insert(0, "/private/tmp/claude-501/-Users-skyking-Documents-missle-sim/e7247f0a-1f65-4c63-b4c5-632b9bb928cd/scratchpad/src_exp")
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from aim120_model.public_api import simulate

print("== dt convergence (review item 6) ==")
print(f"{'config':26s} {'dt':>6} {'a@1.5':>6} {'a@2.3':>6} {'rel':>5} {'switches':>9} {'term':>10} {'dist':>7}")
CFG = [("P+Fcmd+Fenv+Z (no G,no L)", dict(P=True,Fcmd=True,Fenv=True,Z=True)),
       ("all + latch",               dict(P=True,Fcmd=True,Fenv=True,Z=True,G=True,L=True))]
for tag,kw in CFG:
    for dt in (0.02, 0.01, 0.002):
        r = run(dt=dt, **kw); sm = r["summary"]
        print(f"{tag:26s} {dt:6.3f} {adeg(at(r,1.5)):6.2f} {adeg(at(r,2.3)):6.2f} {release(r):5.2f} "
              f"{modes(r):9d} {sm['termination_event'][:10]:>10} {sm['minimum_distance_m']:7.1f}")

print("\n== frozen off-axis bands (su_r_77, no loft, 6500 m, 25 s) ==")
def off(az, t=25.0):
    return dict(launch_speed_kmh=1200.0, launch_altitude_m=6500.0, launch_pitch_deg=0.0,
                launch_heading_deg=0.0, target_speed_kmh=1200.0, target_altitude_m=6500.0,
                initial_distance_m=8000.0, target_azimuth_deg=az, target_heading_deg=0.0,
                target_vertical_heading_deg=0.0, target_constant_turn_g=0.0,
                max_simulation_time_s=t, loft_enabled=False)
def run77(scen, **kw):
    prof = json.loads((PROJ/"missiles/su_r_77.json").read_text(encoding="utf-8"))
    cfg,_ = build_h2_candidate_config(prof, load_runtime_defaults(str(PROJ/"config/profile_h2_runtime_defaults.json")))
    c = cfg["control"]
    c["_exp_fix_projection"]=kw.get("P",False)
    c["_exp_fin_deflection_limit_deg"]=26.4 if kw.get("Fcmd") else None
    c["_exp_envelope_mode"]="full_fin_trim" if kw.get("Fenv") else "alpha_max"
    c["_exp_zeta"]=0.5 if kw.get("Z") else 1.0
    c["_exp_flight_gain_on_capture"]=kw.get("G",False)
    c["_exp_capture_latch"]=kw.get("L",False)
    c["_exp_physical_damping"]=False
    prof["_model_config"]=cfg
    return simulate(prof, scen)["summary"]
BAND = {80:"!=fuse and 15<d<40 and 25<maxG<50", 90:"!=fuse"}
print(f"{'az':>4} {'config':26s} {'min_d':>8} {'maxG':>7} {'term':>10}   frozen band")
for az in (80, 90):
    for tag,kw in (("shipped baseline", {}), ("all + latch", dict(P=True,Fcmd=True,Fenv=True,Z=True,G=True,L=True)),
                   ("G + L only", dict(G=True,L=True))):
        s = run77(off(az), **kw)
        print(f"{az:4d} {tag:26s} {s['minimum_distance_m']:8.1f} {s['maximum_trajectory_normal_g']:7.1f} "
              f"{s['termination_event'][:10]:>10}   {BAND[az] if tag=='shipped baseline' else ''}")
