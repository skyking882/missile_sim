#!/usr/bin/env python3
"""80 deg robustness: the frozen band's lower bound is 15 m and we sit at 16.5 m."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-skyking-Documents-missle-sim/e7247f0a-1f65-4c63-b4c5-632b9bb928cd/scratchpad")
PROJ = Path("/Users/skyking/Documents/missle_sim")
sys.path.insert(0, str(SP / "src_exp"))
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from aim120_model.public_api import simulate

def off(az, v_kmh=1200.0, t=25.0, dt=None):
    sc = dict(launch_speed_kmh=v_kmh, launch_altitude_m=6500.0, launch_pitch_deg=0.0,
              launch_heading_deg=0.0, target_speed_kmh=1200.0, target_altitude_m=6500.0,
              initial_distance_m=8000.0, target_azimuth_deg=az, target_heading_deg=0.0,
              target_vertical_heading_deg=0.0, target_constant_turn_g=0.0,
              max_simulation_time_s=t, loft_enabled=False)
    if dt: sc["simulation_dt_s"] = dt
    return sc

def run(scen, zeta=0.5, full=True):
    prof = json.loads((PROJ / "missiles/su_r_77.json").read_text(encoding="utf-8"))
    cfg, _ = build_h2_candidate_config(
        prof, load_runtime_defaults(str(PROJ / "config/profile_h2_runtime_defaults.json")))
    c = cfg["control"]
    c["_exp_fix_projection"] = full
    c["_exp_fin_deflection_limit_deg"] = 26.4 if full else None
    c["_exp_envelope_mode"] = "full_fin_trim" if full else "alpha_max"
    c["_exp_zeta"] = zeta
    c["_exp_flight_gain_on_capture"] = full
    c["_exp_capture_latch"] = full
    c["_exp_physical_damping"] = False
    prof["_model_config"] = cfg
    return simulate(prof, scen)["summary"]

WN_REF = 6.87            # omega_n(1.6 s) at the A7 operating point
TAU_Q = 2 * 0.5 / WN_REF # the tau_q that zeta=0.5 corresponds to
print(f"P+Fcmd+Fenv+Z+G+L on su_r_77, 80 deg.  frozen band: !=fuse, 15 < d < 40, 25 < maxG < 50")
print(f"tau_q(zeta=0.5) = {TAU_Q:.4f} s at omega_n = {WN_REF} rad/s\n")

print(f"{'perturbation':30s} {'min_d':>8} {'maxG':>7} {'term':>10}  {'band':>6}")
print("-" * 68)
def line(tag, sm):
    ok = (sm["termination_event"] != "proximity_fuse"
          and 15.0 < sm["minimum_distance_m"] < 40.0
          and 25.0 < sm["maximum_trajectory_normal_g"] < 50.0)
    print(f"{tag:30s} {sm['minimum_distance_m']:8.1f} {sm['maximum_trajectory_normal_g']:7.1f} "
          f"{sm['termination_event'][:10]:>10}  {'PASS' if ok else 'FAIL':>6}")
    return ok, sm["minimum_distance_m"]

res = []
res.append(line("nominal (dt=0.02)", run(off(80))))
for dt in (0.010, 0.002):
    res.append(line(f"dt = {dt:.3f}", run(off(80, dt=dt))))
for k in (0.95, 1.05):
    res.append(line(f"tau_q x {k:.2f}  (zeta {0.5*k:.4f})", run(off(80), zeta=0.5 * k)))
for qk, vk in ((0.97, 0.98489), (1.03, 1.01489)):
    res.append(line(f"q_inf x {qk:.2f}  (V x {vk:.4f})", run(off(80, v_kmh=1200.0 * vk))))
print("-" * 68)
ds = [d for _, d in res]
print(f"min_d 范围 {min(ds):.1f} .. {max(ds):.1f} m   （冻结下界 15.0 m，裕度 {min(ds)-15.0:+.1f} m）")
print(f"全部 PASS: {all(ok for ok, _ in res)}")
print("\n对照 90 deg：")
line("nominal (dt=0.02)", run(off(90)))

print("\n" + "="*78)
print("分开测：commit-2 集（P+Fenv+G+L，无 Fcmd/无 Z）vs commit-3 集（+Fcmd+tau_q）")
print("="*78)
def run2(scen, P=False, Fenv=False, G=False, L=False, Fcmd=False, zeta=1.0):
    prof = json.loads((PROJ / "missiles/su_r_77.json").read_text(encoding="utf-8"))
    cfg, _ = build_h2_candidate_config(
        prof, load_runtime_defaults(str(PROJ / "config/profile_h2_runtime_defaults.json")))
    c = cfg["control"]
    c["_exp_fix_projection"] = P
    c["_exp_fin_deflection_limit_deg"] = 26.4 if Fcmd else None
    c["_exp_envelope_mode"] = "full_fin_trim" if Fenv else "alpha_max"
    c["_exp_zeta"] = zeta
    c["_exp_flight_gain_on_capture"] = G
    c["_exp_capture_latch"] = L
    c["_exp_physical_damping"] = False
    prof["_model_config"] = cfg
    return simulate(prof, scen)["summary"]

SETS = [("shipped (legacy)", {}),
        ("commit2: P+Fenv+G+L", dict(P=True, Fenv=True, G=True, L=True)),
        ("commit3: +Fcmd+tau_q", dict(P=True, Fenv=True, G=True, L=True, Fcmd=True, zeta=0.5))]
for az in (80, 90):
    print(f"\n--- {az} deg ---")
    print(f"{'set':26s} {'dt':>6} {'min_d':>7} {'maxG':>7} {'term':>10}  band")
    for tag, kw in SETS:
        for dt in (0.020, 0.010, 0.002):
            sm = run2(off(az, dt=dt), **kw)
            if az == 80:
                ok = (sm["termination_event"] != "proximity_fuse"
                      and 15.0 < sm["minimum_distance_m"] < 40.0
                      and 25.0 < sm["maximum_trajectory_normal_g"] < 50.0)
            else:
                ok = sm["termination_event"] != "proximity_fuse"
            print(f"{tag:26s} {dt:6.3f} {sm['minimum_distance_m']:7.1f} "
                  f"{sm['maximum_trajectory_normal_g']:7.1f} {sm['termination_event'][:10]:>10}  "
                  f"{'PASS' if ok else 'FAIL'}")
