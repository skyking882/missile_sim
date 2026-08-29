#!/usr/bin/env python3
"""Factorial ablation of the five switches (not the sequential ladder)."""
from __future__ import annotations
import json, math, sys, itertools
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-skyking-Documents-missle-sim/e7247f0a-1f65-4c63-b4c5-632b9bb928cd/scratchpad")
PROJ = Path("/Users/skyking/Documents/missle_sim")
sys.path.insert(0, str(SP / "src_exp"))
from aim120_model.profile_adapter import build_h2_candidate_config, load_runtime_defaults
from aim120_model.public_api import simulate

A7 = dict(loft_enabled=True, observation_mode="ideal_truth",
          target_course_reference="statshark_relative_to_los",
          launch_speed_kmh=1200, launch_altitude_m=6300, launch_pitch_deg=0,
          launch_heading_deg=0, target_speed_kmh=1200, target_altitude_m=6300,
          initial_distance_m=8000, target_azimuth_deg=30, target_heading_deg=0,
          target_vertical_heading_deg=0, target_constant_turn_g=0, max_simulation_time_s=40.0)

def run(P=False, Fcmd=False, Fenv=False, Z=False, G=False, L=False, dt=None, tau=0.30):
    prof = json.loads((PROJ / "missiles/su_r_77_1.json").read_text(encoding="utf-8"))
    prof["guidance"]["midcourse_lead_turn"]["tau_capture_s"] = tau
    cfg, _ = build_h2_candidate_config(
        prof, load_runtime_defaults(str(PROJ / "config/profile_h2_runtime_defaults.json")))
    c = cfg["control"]
    c["_exp_fix_projection"] = P                       # P: normalise the cos-alpha projection
    c["_exp_fin_deflection_limit_deg"] = 26.4 if Fcmd else None   # Fcmd: command full fin
    c["_exp_envelope_mode"] = "full_fin_trim" if Fenv else "alpha_max"  # Fenv: trim-solved envelope
    c["_exp_zeta"] = 0.5 if Z else 1.0                 # Z: damping
    c["_exp_flight_gain_on_capture"] = G               # G: datamine launch gain on PCC
    c["_exp_capture_latch"] = L                        # L: one-shot handoff
    c["_exp_physical_damping"] = False
    prof["_model_config"] = cfg
    sc = dict(A7)
    if dt: sc["simulation_dt_s"] = dt
    return simulate(prof, sc)

def at(res, t): return min(res["samples"], key=lambda x: abs(float(x["time_s"]) - t))
def adeg(s): return math.degrees(s["angle_of_attack_rad"])
def release(res):
    prev = None
    for s in res["samples"]:
        t = float(s["time_s"])
        if t < 0.5: continue
        r = s["pcc_capture_ratio"]
        if prev is not None and prev >= 1.0 > r: return t
        prev = r
    return float("nan")
def modes(res):
    out, prev = [], None
    for s in res["samples"]:
        if s["pcc_capture_mode"] != prev:
            out.append(round(float(s["time_s"]), 2)); prev = s["pcc_capture_mode"]
    return len(out) - 1

GAME = {0.7: 19.4, 0.9: 23.2, 1.5: 23.7, 1.9: 23.5, 2.3: 20.6, 2.9: 14.5}
def report(tag, res):
    sm = res["summary"]
    a = {t: adeg(at(res, t)) for t in GAME}
    e = at(res, 1.5)
    return (f"{tag:22s} {release(res):5.2f} " +
            " ".join(f"{a[t]:6.1f}" for t in GAME) +
            f" {at(res,0.9)['j_psi_actual_deg']:7.2f}"
            f" {e['pcc_envelope_g']:6.2f} {e['trajectory_lateral_load_g']:6.2f}"
            f" {modes(res):3d} {sm['termination_event'][:5]:>5s} {sm['minimum_distance_m']:6.1f}")

hdr = (f"{'switches':22s} {'rel':>5} " + " ".join(f"{('a@'+str(t)):>6}" for t in GAME) +
       f" {'Jψ@0.9':>7} {'a_env':>6} {'G@1.5':>6} {'sw':>3} {'term':>5} {'dist':>6}")
print("A7, su_r_77_1, dt=0.02, tau_c=0.30\n")
print(hdr); print("-" * len(hdr))
print(f"{'GAME':22s} {'2.1-2.3':>5} " + " ".join(f"{GAME[t]:6.1f}" for t in GAME) +
      f" {4.7:7.2f} {'—':>6} {15.8:6.2f} {1:3d} {'fuse':>5} {'~0':>6}")
print("-" * len(hdr))

SINGLE = [("baseline (none)", {}),
          ("P only", dict(P=True)), ("Fcmd only", dict(Fcmd=True)),
          ("Fenv only", dict(Fenv=True)), ("Z only", dict(Z=True)),
          ("G only", dict(G=True)),      ("L only", dict(L=True))]
for tag, kw in SINGLE:
    print(report(tag, run(**kw)))
print("-" * len(hdr))
COMBOS = [("P+Fcmd+Z (=ladder)", dict(P=True, Fcmd=True, Z=True)),
          ("P+Fcmd+Fenv+Z",      dict(P=True, Fcmd=True, Fenv=True, Z=True)),
          ("P+Fcmd+Fenv+Z+G",    dict(P=True, Fcmd=True, Fenv=True, Z=True, G=True)),
          ("all + latch",        dict(P=True, Fcmd=True, Fenv=True, Z=True, G=True, L=True)),
          ("G+Z only",           dict(G=True, Z=True)),
          ("P+Fcmd+Fenv+G",      dict(P=True, Fcmd=True, Fenv=True, G=True))]
for tag, kw in COMBOS:
    print(report(tag, run(**kw)))
