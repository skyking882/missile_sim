"""Forward loft comparison with the user-accepted, frozen R-77-1 plant.

There is no optimization. Speed and target motion remain replay inputs; all
missile states and the fixed 48 Hz PID evolve continuously from creation.
The two loft interpretations use existing raw numbers and differ only in the
presence of vertical PN during the observed active-loft window.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_r771_simple_joint as frozen

shared = frozen.shared
old = shared.old
model = shared.model
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "config/r771_simple_joint_frozen.json"
OUT = old.batch.BASE / "frozen_loft"
H_NAMES = ("H110", "H27.5", "H55native", "H55clone")
L_MATCH = {"H110": "L110", "H27.5": "L27.5", "H55native": "L55", "H55clone": "L55"}
MODES = ("pn_plus_loft", "loft_exclusive_vertical")
SUBSTEPS = 10
EVENTS = (("q_peak1", "omega", .35, .60, "max"),
          ("q_valley1", "omega", .60, .84, "min"),
          ("q_peak2", "omega", .85, 1.18, "max"),
          ("alpha_peak1", "alpha", .38, .76, "max"))
WINDOWS = (("full", 0., 2.45), ("passive", 0., .30),
           ("first_recovery", .60, 1.15),
           ("frozen_continuation", 2.55, 3.5))
UNITS = {"theta": "rad", "alpha": "rad", "gamma": "rad", "omega": "rad/s", "x": "m", "y": "m"}


def read_snapshot():
    cfg = json.loads(SNAPSHOT.read_text())
    source = json.loads((ROOT / cfg["source_results"]).read_text())
    coeff = np.asarray(cfg["coefficients"], float)
    assert cfg["source_candidate"] == source["chosen_candidate"] == "affine"
    assert np.array_equal(coeff, np.asarray(source["final_coefficients"]))
    assert cfg["fixed_reference_damping_s_minus1"] == frozen.KW
    assert cfg["source_strict_fit_acceptance"] == source["predeclared_fit_acceptance"]
    return cfg, coeff, source


def map_target_clock(packet_age, b5_age, target_packet_age):
    """Same endpoint offsets/interpolation as old.clock_mapping, allowing H ties.

    H110's packet at age zero coincides with the creation anchor. H55native has
    two packet stamps mapping to one b5 age: retain the resulting flat segment.
    This is a timestamp mapping, not a derivative or state interpolation law.
    """
    positive = packet_age > 0.
    pa, ba = packet_age[positive], b5_age[positive]
    if np.all(np.diff(ba) > 0.) and np.all(ba > 0.):
        return old.clock_mapping(pa, ba, target_packet_age)
    xp, fp = np.r_[0., pa], np.r_[0., ba]
    assert np.all(np.diff(xp) > 0.) and np.all(np.diff(fp) >= 0.)
    mapped = np.interp(target_packet_age, xp, fp)
    left, right = target_packet_age < xp[0], target_packet_age > xp[-1]
    mapped[left] = target_packet_age[left]
    mapped[right] = target_packet_age[right] + fp[-1] - xp[-1]
    return mapped


def make_loft_shot(raw):
    """Reuse no-loft creation parsing, with explicit observation-clock boundaries."""
    source = raw["r"]
    packet_age, native_t = np.asarray(source["t"]), np.asarray(raw["b5"])
    assert np.all(np.diff(packet_age) > 0.) and np.all(np.diff(native_t) >= 0.)
    if native_t[0] > 0. and np.all(np.diff(native_t) > 0.):
        unlofted = dict(raw, p=dict(raw["p"], loft_enabled=False))
        shot = old.make_shot(unlofted, "b5")
        shot["p"] = dict(raw["p"])
    else:
        creation = raw["case"]["creation"]
        velocity = np.asarray(creation["u12_2"], float)
        position = np.asarray(creation["starting_pos"], float)
        theta0 = old.creation_pitch(creation["u16_1"])
        omega0 = -float(creation["u12_3"][2])
        gamma0 = math.atan2(velocity[1], velocity[0])
        speed0 = float(np.hypot(velocity[0], velocity[1]))
        initial = np.array([theta0, omega0, gamma0, position[0], position[1]])
        # All decoded observations remain below. External input interpolation
        # uses the last packet at a positive duplicate tick, and creation at 0.
        keep = np.r_[np.diff(native_t) > 0., True] & (native_t > 0.)
        replay = dict(source, t=np.r_[0., native_t[keep]],
                      speed2=np.r_[speed0, source["speed2"][keep]],
                      missile_packet0_s=0.)
        target_age = source["target_t"] - source["missile_packet0_s"]
        replay["target_t"] = map_target_clock(packet_age, native_t, target_age)
        for key, value in (("x", position[0]), ("y", position[1]),
                           ("theta", theta0), ("gamma", gamma0),
                           ("alpha", theta0-gamma0), ("omega", omega0)):
            replay[key] = np.r_[value, source[key][keep]]
        observed = {key: np.asarray(source[key], float) for key in
                    ("theta", "gamma", "alpha", "omega", "x", "y", "speed2")}
        observed["t"] = native_t.copy()
        shot = {"name": old.batch.NAMES[raw["case"]["case"]], "p": dict(raw["p"]),
                "replay": replay, "observed": observed, "initial": initial,
                "initial_speed_mps": speed0, "clock": "b5",
                "first_packet_age_s": float(packet_age[0]),
                "first_b5_age_s": float(native_t[0]), "sample_count": len(native_t),
                "creation_quat": creation["u16_1"],
                "creation_omega_z_raw": creation["u12_3"][2]}
    duplicates = []
    for i in np.flatnonzero(np.diff(native_t) == 0.):
        duplicates.append({"row_indices_zero_based": [int(i), int(i+1)],
            "b5_age_s": float(native_t[i]),
            "packet_span_s": float(packet_age[i+1]-packet_age[i]),
            "later_minus_earlier": {key: float(source[key][i+1]-source[key][i])
                for key in ("speed2", "theta", "omega", "gamma", "alpha", "x", "y")}})
    shot["clock_boundary"] = {"duplicate_native_ticks": duplicates,
        "zero_age_native_observation_count": int(np.sum(native_t == 0.)),
        "external_input_tie_policy": "last packet at each positive b5 tick; creation input at t=0",
        "observation_policy": "all native rows retained and equally counted in ordinary RMSE, including same-tick rows",
        "target_clock": "no-loft clock_mapping equations; nondecreasing extension retains any flat packet-to-b5 segment",
        "creation_speed_mps": shot["initial_speed_mps"],
        "first_native_speed_mps": float(source["speed2"][0]),
        "creation_minus_first_native_speed_mps": float(shot["initial_speed_mps"]-source["speed2"][0])}
    assert shot["replay"]["t"][0] == 0.
    assert shot["replay"]["speed2"][0] == shot["initial_speed_mps"]
    assert np.all(np.diff(shot["replay"]["t"]) > 0.)
    return shot


def mode_shot(shot, mode):
    p = dict(shot["p"])
    assert p["loft_enabled"] and p["loft_reference"] == "flight_path"
    assert p["loft_omega_unit"] == "rad_per_s" and "geometric_loft_mode" not in p
    if mode == "loft_exclusive_vertical":
        p["pn_gain"] = 0.
    return dict(shot, p=p)


def event_record(t, y, lo, hi, kind):
    return old.event(t, y, lo, hi, kind)


def metrics(shot, prediction):
    obs = shot["observed"]
    rec = {"windows": {}, "events": {}, "units": UNITS}
    for label, lo, hi in WINDOWS:
        mask = (obs["t"] >= lo-1e-9) & (obs["t"] <= hi+1e-9)
        assert np.any(mask)
        w = {"native_sample_count": int(np.sum(mask))}
        for key in UNITS:
            delta = np.interp(obs["t"][mask], prediction["t"], prediction[key])-obs[key][mask]
            w[key+"_rmse"] = float(np.sqrt(np.mean(delta**2)))
            w[key+"_mean_error"] = float(np.mean(delta))
        rec["windows"][label] = w
    for label, key, lo, hi, kind in EVENTS:
        measured = event_record(obs["t"], obs[key], lo, hi, kind)
        predicted = event_record(prediction["t"], prediction[key], lo, hi, kind)
        rec["events"][label] = {"observed_native": measured, "predicted": predicted,
            "time_error_s": predicted["time_s"]-measured["time_s"],
            "value_error": predicted["value"]-measured["value"],
            "relative_amplitude_error": (predicted["value"]-measured["value"])/abs(measured["value"])}
    return rec


def forward_guidance_summary(shot, prediction, mode):
    variant = mode_shot(shot, mode)
    requests = []
    for tick in range(round(3.5*48)+1):
        t = tick/48
        state = np.array([np.interp(t, prediction["t"], prediction[key])
                          for key in ("theta", "omega", "gamma", "x", "y")])
        requests.append(model.request(t, state, variant["replay"], variant["p"]))
    return {"clock": "forward states at48Hz request evaluations,0..3.5s",
        "loft_cap_active_tick_count": int(sum(abs(r["loft_raw_g"]-r["loft_limited_g"]) > 1e-10 for r in requests)),
        "loft_raw_range_g": [min(r["loft_raw_g"] for r in requests), max(r["loft_raw_g"] for r in requests)],
        "request_range_g": [min(r["a_req_g"] for r in requests), max(r["a_req_g"] for r in requests)],
        "request_limit_active_tick_count": int(sum(abs(r["a_req_g"]) >= variant["p"]["req_accel_max_g"]-1e-10 for r in requests)),
        "loft_active_all_ticks": all(r["loft_active"] for r in requests),
        "geometric_mode_enabled": False}


def measured_state_diagnostic(shot, coeff, mode):
    """Reconstruct PID on replay states, not a decoded fin or a free-flight fit."""
    variant = mode_shot(shot, mode)
    p, r = variant["p"], variant["replay"]
    pid, held = (0., 0., 0.), 0.
    rows = []
    keys = ("t", "pn_original_g", "pn_for_command_g", "loft_raw_g", "loft_limited_g",
            "loft_projected_g", "gravity_g", "request_g", "feedback_g", "held_u_rad", "qdot_rad_s2")
    for tick in range(round(2.5*48)+1):
        t = tick/48
        state = np.array([np.interp(t, r["t"], r[key]) for key in ("theta", "omega", "gamma", "x", "y")])
        req = model.request(t, state, r, p)
        original = model.request(t, state, r, shot["p"])
        plant = frozen.surface(t, state, held, variant, coeff)
        gain = model.interpolate_table((tick+1)/48, p["time_gain"])
        if gain > 1e-12:
            raw_u, pid = old.pid_step_with_dt(req["a_req_g"]-plant["feedback_g"], pid, p, 1/48)
            held = gain*raw_u
        else:
            assert pid == (0., 0., 0.)
            held = 0.
        plant = frozen.surface(t, state, held, variant, coeff)
        rows.append([t, original["pn_g"], req["pn_for_command_g"], req["loft_raw_g"],
            req["loft_limited_g"], req["loft_projected_g"], req["gravity_compensation_g"],
            req["a_req_g"], plant["feedback_g"], held, plant["qdot"]])
    a = np.asarray(rows)
    curve = {key: a[:, j] for j, key in enumerate(keys)}
    obs = shot["observed"]
    # Do not differentiate or interpolate a state across the tied observations.
    # Native interval predictions are evaluated only at strictly positive dt.
    interval = []
    for i in range(len(obs["t"])-1):
        lo, hi = obs["t"][i:i+2]
        if lo < .3-1e-9 or hi > 2.45+1e-9 or hi <= lo:
            continue
        tm = (lo+hi)/2
        state = np.array([.5*(obs[key][i]+obs[key][i+1]) for key in ("theta", "omega", "gamma", "x", "y")])
        j = np.searchsorted(curve["t"], tm, side="right")-1
        f = frozen.surface(tm, state, curve["held_u_rad"][j], variant, coeff)
        interval.append([lo, hi, obs["omega"][i+1]-obs["omega"][i], f["qdot"]*(hi-lo),
                         obs["gamma"][i+1]-obs["gamma"][i], f["gamma_dot"]*(hi-lo)])
    a = np.asarray(interval)
    diag = {"description": "Measured-state conditional PID with frozen plant feedback and same48Hz/end-gain lifecycle; held_u is reconstructed, not decoded fin",
        "native_positive_interval_count": len(a),
        "delta_omega_rmse_rad_s": float(np.sqrt(np.mean((a[:,3]-a[:,2])**2))),
        "delta_gamma_rmse_rad": float(np.sqrt(np.mean((a[:,5]-a[:,4])**2))),
        "loft_cap_active_tick_count": int(np.sum(np.abs(curve["loft_raw_g"]-curve["loft_limited_g"]) > 1e-10)),
        "request_peak_g": float(np.max(curve["request_g"])),
        "original_pn_range_g": [float(np.min(curve["pn_original_g"])), float(np.max(curve["pn_original_g"]))],
        "loft_raw_range_g": [float(np.min(curve["loft_raw_g"])), float(np.max(curve["loft_raw_g"]))],
        "held_u_range_rad": [float(np.min(curve["held_u_rad"])), float(np.max(curve["held_u_rad"]))]}
    curve.update({"interval_"+key: a[:,j] for j,key in enumerate(("start_s", "end_s", "observed_delta_omega",
        "predicted_delta_omega", "observed_delta_gamma", "predicted_delta_gamma"))})
    return curve, diag


def draw(shots, predictions, l_predictions, diagnostics):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22})
    colors = {MODES[0]: "#15856f", MODES[1]: "#3168b3"}
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for col, name in enumerate(H_NAMES):
        obs, lobs = shots[name]["observed"], shots[L_MATCH[name]]["observed"]
        for row, (key, scale, label) in enumerate((("omega", 1., "Pitch rate [rad/s]"),
                ("alpha", 180/math.pi, "Incidence [deg]"), ("theta", 180/math.pi, "Pitch [deg]"),
                ("gamma", 180/math.pi, "Flight path [deg]"))):
            ax = axes[row,col]
            ax.plot(obs["t"], obs[key]*scale, ".", color="#202020", markersize=3, label="Decoded loft")
            for mode in MODES:
                pr = predictions[name][mode]
                ax.plot(pr["t"], pr[key]*scale, color=colors[mode], lw=1.7,
                        label="Frozen PN + loft" if mode == MODES[0] else "Frozen loft, vertical PN absent")
            ax.plot(lobs["t"], lobs[key]*scale, ".", color="#aaaaaa", markersize=2, label="Decoded no-loft")
            lp = l_predictions[L_MATCH[name]]
            ax.plot(lp["t"], lp[key]*scale, color="#888888", ls="--", lw=1., label="Frozen no-loft")
            ax.set_xlim(0., 2.45)
            if col == 0: ax.set_ylabel(label)
            if row == 0: ax.set_title(name)
            if row == 3: ax.set_xlabel("Native b5 age [s]")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5,.001))
    fig.suptitle("Frozen four-coefficient R-77-1 plant: loft forward transfer, no retuning", y=.995)
    fig.tight_layout(rect=(0,.065,1,.975))
    fig.savefig(OUT / "loft_forward.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for col,name in enumerate(H_NAMES):
        obs = shots[name]["observed"]
        for row,(key,scale,label) in enumerate((("omega",1.,"Pitch rate [rad/s]"),("alpha",180/math.pi,"Incidence [deg]"))):
            ax=axes[row,col]
            visible=(obs["t"]>=2.45)&(obs["t"]<=3.5)
            ax.plot(obs["t"][visible],obs[key][visible]*scale,".",color="#202020",markersize=3)
            for mode in MODES:
                pr=predictions[name][mode]
                visible=(pr["t"]>=2.45)&(pr["t"]<=3.5)
                ax.plot(pr["t"][visible],pr[key][visible]*scale,color=colors[mode],lw=1.7)
            ax.set_xlim(2.45,3.5)
            ax.axvline(2.55,color="#999999",ls=":")
            if col==0:ax.set_ylabel(label)
            if row==0:ax.set_title(name)
            if row==1:ax.set_xlabel("Native b5 age [s]")
    fig.suptitle("Frozen continuation (metric window 2.55–3.50 s); same colors as forward figure")
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUT / "loft_continuation.png",dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3,4,figsize=(16,9),sharex=True)
    for col,name in enumerate(H_NAMES):
        plus=diagnostics[name][MODES[0]]
        exclusive=diagnostics[name][MODES[1]]
        ax=axes[0,col]
        for key,label,color in (("pn_original_g","Original vertical PN","#9150a4"),
                ("loft_projected_g","Projected loft","#d17f25"),("gravity_g","Gravity","#777777")):
            ax.plot(plus["t"],plus[key],color=color,label=label)
        ax.set_title(name)
        for mode in MODES:
            curve=diagnostics[name][mode]
            label="PN + loft" if mode==MODES[0] else "Loft only"
            axes[1,col].plot(curve["t"],curve["request_g"],color=colors[mode],label=label)
            axes[2,col].plot(curve["t"],np.degrees(curve["held_u_rad"]),color=colors[mode],label=label)
        axes[1,col].plot(plus["t"],plus["feedback_g"],color="#202020",ls="--",label="Frozen-plant feedback at replay state")
        for row in range(3):axes[row,col].set_xlim(0.,2.45)
        axes[2,col].set_xlabel("Native b5 age [s]")
    axes[0,0].set_ylabel("Measured-state components [g]")
    axes[1,0].set_ylabel("Required / feedback [g]")
    axes[2,0].set_ylabel("Reconstructed held input [deg]")
    for row in range(3):axes[row,0].legend(fontsize=7,loc="best")
    fig.suptitle("Conditional guidance diagnostic on replay states; input is reconstructed, not decoded fin")
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(OUT / "loft_guidance_components.png",dpi=180)
    plt.close(fig)


def main():
    cfg, coeff, source = read_snapshot()
    raw, p = old.batch.load_all()
    shots = {name: make_loft_shot(raw[name]) for name in H_NAMES}
    shots.update({name:old.make_shot(raw[name],"b5") for name in shared.NAMES})
    for shot in shots.values():
        assert shot["p"]["tail_arm_m"] == cfg["surface"]["tail_arm_m"]
        assert [shot["p"][key] for key in ("kp","ki","kd","integral_limit")] == [.002125,.006481,.000023,1.]
        for key,value in cfg["fixed_plant_parameters"].items():
            if key != "density":assert shot["p"][key] == value
        area=math.pi*shot["p"]["caliber_m"]**2/4*shot["p"]["wing_area_mult"]
        assert abs(area-cfg["surface"]["reference_area_m2"]) < 1e-15
    OUT.mkdir(parents=True,exist_ok=True)
    predictions, diagnostics, records, diag_records, arrays = {},{},{},{},{}
    l_predictions={name:frozen.simulate(shots[name],coeff,substeps=SUBSTEPS) for name in shared.NAMES}
    for name,pr in l_predictions.items():
        assert pr is not None
        for key,value in pr.items():
            if isinstance(value,np.ndarray):arrays[name+"__no_loft__"+key]=value
    for name in H_NAMES:
        predictions[name],diagnostics[name],records[name],diag_records[name]={},{},{},{}
        for mode in MODES:
            pr=frozen.simulate(mode_shot(shots[name],mode),coeff,substeps=SUBSTEPS)
            if pr is None:raise RuntimeError(name+" "+mode+" exceeded existing frozen simulator validity guard")
            curve,diag=measured_state_diagnostic(shots[name],coeff,mode)
            predictions[name][mode],diagnostics[name][mode]=pr,curve
            records[name][mode],diag_records[name][mode]=metrics(shots[name],pr),diag
            records[name][mode]["forward_guidance"]=forward_guidance_summary(shots[name],pr,mode)
            for key,value in pr.items():
                if isinstance(value,np.ndarray):arrays[name+"__"+mode+"__"+key]=value
            for key,value in curve.items():arrays[name+"__"+mode+"__diagnostic__"+key]=value
        for key,value in shots[name]["observed"].items():arrays[name+"__observed__"+key]=value
    comparisons={}
    for name in H_NAMES:
        rec={}
        for window,*_ in WINDOWS:
            rec[window]={key:records[name][MODES[1]]["windows"][window][key+"_rmse"] /
                records[name][MODES[0]]["windows"][window][key+"_rmse"] for key in UNITS}
        comparisons[name]={"exclusive_over_additive_rmse_ratios":rec}
    pairs={}
    for name in H_NAMES:
        l_name=L_MATCH[name]
        rec={}
        for label,key,lo,hi,kind in EVENTS:
            ho,loft=shots[name]["observed"],shots[l_name]["observed"]
            h=event_record(ho["t"],ho[key],lo,hi,kind)
            l=event_record(loft["t"],loft[key],lo,hi,kind)
            rec[label]={"H":h,"L":l,"H_over_L_abs_amplitude":abs(h["value"]/l["value"])}
        pairs[name]={"no_loft_reference":l_name,"events":rec}
    # One focused verification: exact frozen-source equality, old no-loft
    # reproduction, representative H step halving, all arrays and lifecycle.
    prior=np.load(frozen.OUT / "trajectories.npz")
    lp=l_predictions["L110"]
    reproduction={key:float(np.max(np.abs(lp[key]-prior["L110__affine__"+key])))
                  for key in ("t","theta","omega","alpha","gamma","y","u")}
    assert max(reproduction.values()) < 1e-12
    lm=shared.metrics(shots["L110"],lp)
    saved=source["records"]["L110"]["affine"]
    metric_diff=max(abs(lm["windows"][w][key+"_rmse"]-saved["windows"][w][key+"_rmse"])
                   for w in lm["windows"] for key in (*shared.STATE_SCALES,"y"))
    assert metric_diff < 1e-12
    fine=frozen.simulate(mode_shot(shots["H110"],MODES[0]),coeff,substeps=2*SUBSTEPS)
    assert fine is not None
    hpred=predictions["H110"][MODES[0]]
    halving={key:float(np.max(np.abs(hpred[key]-np.interp(hpred["t"],fine["t"],fine[key]))))
             for key in ("theta","omega","alpha","gamma","y")}
    assert halving["omega"] < 1e-5 and halving["alpha"] < 1e-6
    assert all(np.isfinite(v).all() for v in arrays.values())
    identities,first_active={},{}
    for name in H_NAMES:
        identities[name],first_active[name]={},{}
        for mode in MODES:
            pr=predictions[name][mode]
            identities[name][mode]=float(np.max(np.abs(pr["moment_nm"]+.4*pr["normal_body_n"]-pr["reference_couple_nm"])))
            first_active[name][mode]=pr["first_active"]["start_s"]
            assert first_active[name][mode] == 14/48
            assert identities[name][mode] < 1e-8
    result={"frozen_snapshot":str(SNAPSHOT),"frozen_coefficients":coeff.tolist(),
        "fixed_reference_damping_s_minus1":frozen.KW,"source_version":p["source_version"],
        "frozen_status":cfg["status"],"user_acceptance_does_not_change_prior_strict_fit_flags":True,
        "optimization_performed":False,"production_changed":False,"native_game_law_identified":False,
        "simulation":"Same frozen continuous creation-state RK4,48Hz/end-gain zero-memory PID; no state resets, fitted schedules, or time shifts",
        "horizon_s":3.5,"substeps_per_control_interval":SUBSTEPS,"metric_units":UNITS,
        "metric_definition":"ordinary RMSE interpolated prediction at every retained native b5 observation; all same-tick rows count; event extrema use native observed samples (no parabolic sub-sample claim)",
        "raw_loft_fields":{"loftElevation_deg":24.,"loftTargetElevation_deg":-3.5,
            "loftAngleToAccelMult":17.5,"loftTargetOmegaMax":.5,"propNavMult":6.},
        "loft_interpretations":{
            MODES[0]:"Existing flight-path loft g=17.5*(24deg-gamma),0.5rad/s loft-only cap, projected onto flight-path normal, additive PN, gravity, existing50g request limit",
            MODES[1]:"Same loft law; vertical PN absent throughout observed active-loft window (pn_gain=0); no timed/geometric switch inferred",
            "loftTargetElevation":"-3.5deg is retained raw but unused by both interpretations; no exit semantics established",
            "cap_units":"0.5rad/s is an existing conditional interpretation, not a decoded native unit"},
        "plant_transfer_limitation":"Linear force/moment law fitted near4deg first incidence peak transfers here to27deg; PN A/B cannot uniquely identify guidance under large-incidence plant extrapolation",
        "finsLatAccel_limitation":"Empirical frozen plant does not vary dynamically with this raw field; similar predictions cannot identify its game mapping",
        "target_velocity_provenance":"Packet-derived target_vxy from source loader is retained unchanged; only target timestamps are mapped to b5 age",
        "source_map":{"payload":str(old.batch.BASE / "source_payload.json"),
            "compact_creation_packet_epochs":str(old.batch.BASE / "batch_compact.json"),
            "creation_and_clock_helper":str(ROOT / "scripts/fit_r771_no_loft_control.py"),
            "frozen_force_moment":str(ROOT / "scripts/fit_r771_simple_joint.py"),
            "RK4_and_PID_lifecycle":str(ROOT / "scripts/fit_r771_shared_surface.py"),
            "guidance_request":str(ROOT / "scripts/try_r771_effective_moment.py")},
        "initial_inputs":{name:{"state_theta_omega_gamma_x_y":shots[name]["initial"].tolist(),
            "initial_speed_mps":shots[name]["initial_speed_mps"],
            "mass0_kg":shots[name]["p"]["mass0_kg"],"massEnd_kg":shots[name]["p"]["mass_end_kg"],
            "thrust_n":shots[name]["p"]["thrust_n"],"timeFire_s":shots[name]["p"]["time_fire_s"],
            "finsLatAccel_raw":shots[name]["p"]["finsLatAccel_custom"],
            "density_initial_kg_m3":model.isa_density(shots[name]["initial"][4]),
            "clock_boundary":shots[name]["clock_boundary"]} for name in H_NAMES},
        "records":records,"exclusive_vs_additive":comparisons,"H_vs_L_native_events":pairs,
        "measured_state_diagnostic":diag_records,
        "verification":{"snapshot_exact_saved_coefficient_equality":True,
            "L110_saved_array_reproduction_max_abs":reproduction,"L110_saved_rmse_max_abs_difference":metric_diff,
            "H110_additive_step_halving_max_abs":halving,"all_arrays_finite":True,
            "surface_moment_identity_max_nm":identities,"first_active_start_s":first_active,
            "moment_identity_is_bookkeeping_not_independent_physics":True}}
    np.savez_compressed(OUT / "trajectories.npz",**arrays)
    (OUT / "results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    draw(shots,predictions,l_predictions,diagnostics)
    print(json.dumps({"output":str(OUT),"records":{name:{mode:records[name][mode]["windows"]["full"] for mode in MODES} for name in H_NAMES},"verification":result["verification"]},indent=2))


if __name__ == "__main__":
    main()
