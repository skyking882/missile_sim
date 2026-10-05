#!/usr/bin/env python3
"""Repeatable, version-separated comparison with recorded game observations.

The default runner is the actual GUI's general H2 runtime. Native missile
states evolve freely from creation; only the decoded target track is external.
Screenshot inputs retain the historical scenario assumptions, not reconstructed
target motion. No time shift, observation filtering, or extrapolation is used.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")

import numpy as np

from aim120_model.dynamics import SimState
from aim120_model.h2_simulator import H2Simulator
from aim120_model.public_api import _case_from_scenario, _summarize, simulate, validate_scenario
from aim120_model.target import TargetState
from missile_gui.library import _load_one
from compare_rebuild_replays import CASES as SCREENSHOT_CASES
from run_rebuild_scenarios import SCENARIO

BASE = ROOT / "outputs/unified_game_replays"
NATIVE_BASE = ROOT / "outputs/r771_replay_batch_20260928"
WINDOWS = (("full", 0., math.inf), ("early_0_3p5", 0., 3.5),
           ("transient", .3, 1.25), ("recovery", 1.25, 3.5),
           ("late", 3.5, math.inf))
UNITS = {"speed_kmh": "km/h", "alpha_deg": "deg", "load_g": "g",
         "pitch_deg": "deg", "omega_rad_s": "rad/s", "gamma_deg": "deg",
         "x_m": "m", "y_m": "m"}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def capture_profiles(path: Path) -> dict:
    """Explicit GUI loader snapshot, so later adapter edits cannot move baseline."""
    if path.exists():
        return json.loads(path.read_text())
    profiles = {}
    for missile in ("cn_pl12", "su_r_77", "su_r_77_1", "us_aim_120a"):
        profiles["2.57.1.107/" + missile] = _load_one(ROOT / "missiles" / (missile + ".json"), ROOT)
    profiles["2.59.0.27/su_r_77_1"] = _load_one(
        ROOT / "data/profile_sets/2.59.0.27/missiles/su_r_77_1.json", ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, indent=2))
    return profiles


class DecodedTarget:
    """Piecewise-linear positions with their own consistent piecewise velocity.

    Target timestamps are mapped to native b5 age by the existing clock mapping.
    Flat mapped intervals use their last target sample for external input only;
    all original target and missile rows are preserved in acceptance artifacts.
    """
    def __init__(self, t, xyz):
        self.raw_t, self.raw_xyz = np.asarray(t), np.asarray(xyz)
        keep = np.r_[np.diff(self.raw_t) > 0., True]
        self.t, self.xyz = self.raw_t[keep], self.raw_xyz[keep]
        assert np.all(np.diff(self.t) > 0.)
        assert self.t[0] <= 0. <= self.t[-1], "creation outside target support"
        self.initial_state = self.state_at(0.)

    def state_at(self, time_s):
        if time_s < self.t[0]-1e-8 or time_s > self.t[-1]+1e-8:
            raise ValueError("target observation support exceeded; no extrapolation")
        index = int(np.clip(np.searchsorted(self.t, time_s, side="right")-1, 0, len(self.t)-2))
        dt = self.t[index+1]-self.t[index]
        velocity = (self.xyz[index+1]-self.xyz[index])/dt
        position = self.xyz[index] + (time_s-self.t[index])*velocity
        return TargetState(tuple(position), tuple(velocity))


class CreationH2(H2Simulator):
    """Script-local initialization bridge; production equations are unchanged."""
    def initial_state(self, case):
        return replace(case["_creation_state"])


def load_cases(profile_snapshot: Path | None = None):
    """Return five screenshot cases, seven native cases, and one scalar anchor."""
    profile_snapshot = profile_snapshot or BASE / "baseline/runtime_profiles.json"
    profiles = capture_profiles(profile_snapshot)
    cases = []
    limitations = {
        "30degpl12": "A2 target actually maneuvered; straight historical target is an approximation. Full trace is not an independent plant anchor.",
        "30degr77": "A4 historical 1500 km/h launch retained; screenshot folder and old slow label do not set speed or target course.",
        "30degpl12slow": "A3 historical straight, LOS-relative target geometry; exact target maneuver is unavailable.",
        "pl12level": "A7 historical setup; target course interpretation was not independently confirmed.",
        "r77-1": "User-confirmed parallel opposite courses: target absolute heading180 degrees. Launch speed is approximate and target track unavailable.",
    }
    for folder, missile, filename, count, overrides in SCREENSHOT_CASES:
        scenario = dict(SCENARIO, **overrides)
        with (ROOT / "data/replays" / filename).open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == count
        usable = [r for r in rows if r["missile_flight_time_s"] != "NA"]
        obs = {"t": np.array([float(r["missile_flight_time_s"]) for r in usable])}
        for dest, source in (("speed_kmh", "speed_kmh"), ("alpha_deg", "angle_of_attack_deg"), ("load_g", "overload_g")):
            obs[dest] = np.array([float(r[source]) if r[source] != "NA" else np.nan for r in usable])
        cases.append({"id": folder, "dataset": "screenshot_2.57", "version": "2.57.1.107",
            "profile": copy.deepcopy(profiles["2.57.1.107/"+missile]), "scenario": scenario,
            "observed": obs, "source_count": count, "source_file": "data/replays/"+filename,
            "source_rows": rows, "game_outcome": "hit_user_confirmed_2026-09-08",
            "input_limitations": [limitations[folder], "Loft toggle and ideal-truth observation follow historical comparison; no original screenshot field establishes either.",
                "Camera distance is never target range. Screenshot load includes thrust and is signed; runtime magnitude is unsigned. Small-alpha display bias below about1.5 degrees remains uncorrected."],
            "input_provenance": "tests/test_replay_anchors.py and data/replays/README.md; no fitted input or time offset"})

    import analyze_r771_batch_loft as batch
    from analyze_r771_frozen_loft import make_loft_shot, map_target_clock
    from fit_r771_no_loft_control import creation_pitch, make_shot
    raw_shots, _ = batch.load_all()
    payload = json.loads((NATIVE_BASE / "source_payload.json").read_text())
    rocket = json.loads(payload["parent_parameters"]["native_text"])["rocket"]
    runtime_rocket = json.loads((ROOT/"data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx").read_text())["rocket"]
    shared_fields = ("mass", "massEnd", "force", "timeFire", "caliber", "length", "wingAreaMult",
        "CxK", "CyMaxAoA", "finsAoaHor", "finsAoaVer", "distFromCmToStab")
    assert all(rocket[key] == runtime_rocket[key] for key in shared_fields)
    assert rocket["guidance"]["guidanceAutopilot"] == runtime_rocket["guidance"]["guidanceAutopilot"]
    for name, raw in raw_shots.items():
        shot = make_loft_shot(raw) if raw["p"]["loft_enabled"] else make_shot(raw, "b5")
        observed = shot["observed"]
        creation = raw["case"]["creation"]
        pos = np.asarray(creation["starting_pos"], float)
        velocity = np.asarray(creation["u12_2"], float)
        quat = np.asarray(creation["u16_1"], float)
        qx, qy, qz, qw = quat
        pitch = creation_pitch(quat)
        yaw = math.atan2(2*(qx*qz-qw*qy), 1-2*(qy*qy+qz*qz))
        # Keep absolute altitude, translate horizontal launch origin for H2 event geometry.
        translation = np.array([pos[0], 0., pos[2]])
        target_rows = raw["case"]["target_rows"]
        # shot.replay resets packet origin; derive packet ages explicitly here.
        target_t_packet = np.array([r["time_ms"] for r in target_rows], float)/1000-raw["r"]["missile_packet0_s"]
        target_t = map_target_clock(raw["r"]["t"], raw["b5"], target_t_packet)
        target_xyz = np.array([[r[k] for k in ("x_m", "y_m", "z_m")] for r in target_rows], float)-translation
        target = DecodedTarget(target_t, target_xyz)
        profile = copy.deepcopy(profiles["2.59.0.27/su_r_77_1"])
        config = profile["_model_config"]
        config["aerodynamics"]["fins_lateral_acceleration_g"] = float(raw["case"]["finsLatAccel"])
        config["guidance"]["lofting_enabled"] = bool(raw["case"]["loftEnabled"])
        # The replay parent has the same inspected source propulsion/geometry/PID;
        # retain the old actual-runtime loft mapping (20/.75) in baseline.
        native_obs = {"t": observed["t"], "speed_kmh": np.linalg.norm(raw["r"]["velocity3"], axis=1)*3.6,
            "alpha_deg": np.degrees(observed["alpha"]), "pitch_deg": np.degrees(observed["theta"]),
            "omega_rad_s": observed["omega"], "gamma_deg": np.degrees(observed["gamma"]),
            "x_m": observed["x"]-pos[0], "y_m": observed["y"]-pos[1]}
        state = SimState(tuple(pos-translation), tuple(velocity), pitch, yaw,
            -float(creation["u12_3"][2]), -float(creation["u12_3"][1]), float(config["geometry"]["initial_mass_kg"]))
        horizon = min(float(observed["t"][-1]), float(target.t[-1]))
        initial_target = target.initial_state
        relative = np.asarray(initial_target.position)-np.asarray(state.position)
        scenario = dict(SCENARIO, loft_enabled=bool(raw["case"]["loftEnabled"]),
            target_course_reference="absolute_world", launch_speed_kmh=float(np.linalg.norm(velocity)*3.6),
            launch_altitude_m=float(pos[1]), launch_pitch_deg=math.degrees(pitch), launch_heading_deg=math.degrees(yaw),
            target_altitude_m=initial_target.position[1], target_speed_kmh=float(np.linalg.norm(initial_target.velocity)*3.6),
            initial_distance_m=float(np.hypot(relative[0], relative[2])),
            target_azimuth_deg=math.degrees(math.atan2(relative[2], relative[0])),
            target_heading_deg=math.degrees(math.atan2(initial_target.velocity[2], initial_target.velocity[0])),
            target_vertical_heading_deg=math.degrees(math.atan2(initial_target.velocity[1], math.hypot(initial_target.velocity[0], initial_target.velocity[2]))),
            max_simulation_time_s=horizon)
        dt = np.diff(observed["t"])
        theta_residual = np.diff(observed["theta"])-.5*(observed["omega"][:-1]+observed["omega"][1:])*dt
        positions = np.column_stack((raw["r"]["x"], raw["r"]["y"], raw["r"]["z"]))
        position_residual = np.diff(positions, axis=0)-.5*(raw["r"]["velocity3"][:-1]+raw["r"]["velocity3"][1:])*dt[:, None]
        cases.append({"id": name, "dataset": "native_2.59", "version": "2.59.0.28",
            "profile": profile, "scenario": scenario, "observed": native_obs,
            "source_count": len(observed["t"]), "source_file": str((NATIVE_BASE/"source_payload.json").relative_to(ROOT)),
            "source_rows": list(range(len(observed["t"]))), "game_outcome": "not_independently_established_by_this_payload",
            "input_provenance": "creation u12_2/u12_3/u16_1 and decoded target positions; source b5 age retained including ties",
            "input_limitations": ["b7 velocity, b8 pitch rate, quaternion and Euler channels are replay-decoded candidates, not native telemetry.",
                "H2 uses pitch/yaw and omits small creation roll; no bank response or unknown native fin input is identified.",
                "Target piecewise velocity is derived on mapped b5 clock; packet-derived target velocity is not mixed with that clock.",
                "Source2.59.0.27 runtime snapshot used with matching inspected2.59.0.28 propulsion/geometry/PID; actual baseline loft defaults20/.75 are preserved and disclosed."],
            "raw_loft": dict(rocket["guidance"]["guidanceAutopilot"], loftEnabled=bool(raw["case"]["loftEnabled"])),
            "source_parameter_equality": {"runtime_source_version": "2.59.0.27", "replay_source_version": "2.59.0.28",
                "checked_equal_top_level_fields": list(shared_fields), "entire_guidance_autopilot_equal": True},
            "native_custom_fins_lat_accel": raw["case"]["finsLatAccel"],
            "creation": creation, "_creation_state": state, "_target_model": target,
            "target_source_t": target_t, "target_source_xyz": target_xyz,
            "clock_diagnostics": {"native_tick_ties": int(np.sum(dt == 0)),
                "zero_age_rows": int(np.sum(observed["t"] == 0)),
                "theta_minus_integrated_omega_max_interval_deg": float(np.degrees(np.max(np.abs(theta_residual)))),
                "position_minus_integrated_velocity_max_interval_m": float(np.max(np.linalg.norm(position_residual, axis=1))),
                "target_support_s": [float(target.t[0]), float(target.t[-1])],
                "clock_boundary": shot.get("clock_boundary", {})}})

    # One weak scalar game anchor, kept apart from frame-accurate replay evidence.
    scalar_case = json.loads((ROOT/"configs/aim120a_v1_cases.json").read_text())["cases"][1]
    ic = scalar_case["initial_conditions"]
    scenario = dict(SCENARIO, loft_enabled=False, launch_speed_kmh=ic["start_speed_kmh"],
        launch_altitude_m=ic["launch_altitude_m"], target_speed_kmh=ic["target_speed_kmh"],
        target_altitude_m=ic["target_altitude_m"], initial_distance_m=ic["initial_target_distance_m"],
        target_azimuth_deg=ic["target_azimuth_deg"], max_simulation_time_s=50.)
    cases.append({"id": "AIM120A_15km_38deg_scalar", "dataset": "scalar_2.57", "version": "2.57.1.107",
        "profile": copy.deepcopy(profiles["2.57.1.107/us_aim_120a"]), "scenario": scenario,
        "observed": {}, "source_count": 1, "source_file": "README.md:166 and configs/aim120a_v1_cases.json",
        "game_outcome": "historical_game_or_external_time16.7s_only",
        "scalar_anchor": {"flight_time_s": 16.7},
        "input_limitations": ["Only a rounded historical time anchor is available; no game trajectory, impact time precision, or numerical miss distance is supplied.",
            "Loft-disabled scenario follows frozenv1 case; current general H2 runtime is compared, not v1 regression output."],
        "input_provenance": "historical v1 geometry and README scalar; version2.57 only"})
    return cases


def current_runtime(case, profile):
    """Return actual general H2 output, with creation bridge only for native cases."""
    if case["dataset"] != "native_2.59":
        return simulate(profile, case["scenario"])
    normalized = validate_scenario(case["scenario"])
    run_case = _case_from_scenario(normalized)
    run_case["_creation_state"] = case["_creation_state"]
    run_case["_target_model"] = case["_target_model"]
    config = copy.deepcopy(profile["_model_config"])
    config["guidance"]["lofting_enabled"] = normalized["loft_enabled"]
    config["performance"]["lifetime_s"] = normalized["max_simulation_time_s"]
    result = CreationH2(config).run(run_case)
    return {"samples": result["samples"], "summary": _summarize(result, config, True,
            normalized["loft_enabled"], normalized["loft_enabled"]),
            "model": {k: result[k] for k in ("model_label", "time_step_s", "integrator")}}


def sample_series(samples, case):
    """Normalize runtime samples to observed axes, preserving their definitions."""
    t = np.array([s["time_s"] for s in samples])
    values = {"t": t,
        "speed_kmh": np.array([math.hypot(*s["velocity_mps"])*3.6 for s in samples]),
        "load_g": np.array([s["trajectory_lateral_load_g"] for s in samples]),
        "alpha_deg": np.array([math.degrees(s["pitch_alpha_rad"] if case["dataset"] == "native_2.59" else s["angle_of_attack_rad"]) for s in samples]),
        "pitch_deg": np.array([math.degrees(s["pitch_rad"]) for s in samples]),
        "omega_rad_s": np.array([s["pitch_rate_rad_s"] for s in samples]),
        "gamma_deg": np.array([math.degrees(math.atan2(s["velocity_mps"][1], s["velocity_mps"][0])) for s in samples]),
        "x_m": np.array([s["position_m"][0] for s in samples]),
        "y_m": np.array([s["position_m"][1]-case["scenario"]["launch_altitude_m"] for s in samples])}
    assert np.all(np.diff(t) > 0.) and all(np.isfinite(v).all() for v in values.values())
    return values


def extremum(t, value, mode, lo=-math.inf, hi=math.inf):
    mask = (t >= lo) & (t <= hi) & np.isfinite(value)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return None
    pick = int(ids[np.argmax(value[ids]) if mode == "max" else np.argmin(value[ids])])
    return {"time_s": float(t[pick]), "value": float(value[pick])}


def compare_series(case, prediction, summary):
    """Reusable acceptance function for a future candidate's normalized arrays."""
    obs, mt = case["observed"], prediction["t"]
    record = {k: case[k] for k in ("id", "dataset", "version", "source_count", "source_file", "game_outcome", "input_limitations", "input_provenance")}
    record.update(scenario=case["scenario"], model_summary=summary,
        runtime_model_source=case["profile"].get("_runtime_adapter", case["profile"]["_model_config"].get("runtime_adapter")),
        runtime_profile_provenance=case["profile"].get("provenance"),
        supplied_missile_trajectory_after_creation=False, units=UNITS, windows={}, events={},
        runtime_control=case["profile"]["_model_config"]["control"],
        runtime_loft={k: case["profile"]["_model_config"]["guidance"].get(k) for k in
            ("lofting_elevation_deg", "target_elevation_deg", "angle_to_acceleration_multiplier",
             "loft_omega_max_deg_s", "omega_max_deg_s")})
    if not obs:
        record["scalar_anchor"] = case["scalar_anchor"]
        record["flight_time_error_s"] = summary["flight_time_s"]-case["scalar_anchor"]["flight_time_s"]
        record["frame_comparison_available"] = False
        return record, {}
    ot = obs["t"]
    matched = (ot >= mt[0]-1e-10) & (ot <= mt[-1]+1e-10)
    aligned = {"t": ot, "compared": matched}
    record.update(telemetry_rows=len(ot), compared_rows=int(matched.sum()),
        original_rows_preserved=True, time_shift_s=0., extrapolated_rows=0,
        observed_support_s=[float(ot[0]), float(ot[-1])], model_support_s=[float(mt[0]), float(mt[-1])],
        unrepresented_rows=int((~matched).sum()))
    for key in obs:
        if key == "t":
            continue
        aligned["observed_"+key] = obs[key]
        val = np.full(len(ot), np.nan)
        val[matched] = np.interp(ot[matched], mt, prediction[key])
        aligned["model_"+key] = val
    for name, lo, hi in WINDOWS:
        mask = matched & (ot >= lo-1e-9) & (ot <= hi+1e-9)
        window = {"compared_rows": int(mask.sum()), "metrics": {}}
        for key in obs:
            if key == "t":
                continue
            valid = mask & np.isfinite(obs[key])
            if not np.any(valid):
                continue
            e = aligned["model_"+key][valid]-obs[key][valid]
            window["metrics"][key] = {"n": int(valid.sum()), "rmse": float(np.sqrt(np.mean(e*e))),
                "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)), "max_abs": float(np.max(np.abs(e))),
                "last_compared_residual": float(e[-1])}
        record["windows"][name] = window
    event_defs = [(key+"_max", key, "max", -math.inf, math.inf) for key in obs if key != "t"]
    if "omega_rad_s" in obs:
        event_defs.extend((("q_peak1", "omega_rad_s", "max", .35, .60),
            ("q_valley1", "omega_rad_s", "min", .60, .84),
            ("q_peak2", "omega_rad_s", "max", .85, 1.18),
            ("alpha_peak1", "alpha_deg", "max", .38, .76)))
    for name, key, mode, lo, hi in event_defs:
        observed = extremum(ot, obs[key], mode, lo, hi)
        predicted = extremum(mt, prediction[key], mode, max(lo, float(ot[0])), min(hi, float(ot[-1])))
        if observed and predicted:
            record["events"][name] = {"observed": observed, "model": predicted,
                "value_error": predicted["value"]-observed["value"],
                "time_error_s": predicted["time_s"]-observed["time_s"]}
    if "clock_diagnostics" in case:
        record["clock_diagnostics"] = case["clock_diagnostics"]
        record["raw_loft"] = case["raw_loft"]
        record["native_custom_fins_lat_accel"] = case["native_custom_fins_lat_accel"]
        record["source_parameter_equality"] = case["source_parameter_equality"]
    return record, aligned


def plot_cases(cases, arrays, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    for group, fields, filename in (("screenshot_2.57", ("speed_kmh", "alpha_deg", "load_g"), "screenshots_257.png"),
            ("native_2.59", ("speed_kmh", "alpha_deg", "omega_rad_s", "gamma_deg", "y_m"), "native_259.png")):
        selected = [c for c in cases if c["dataset"] == group]
        fig, axes = plt.subplots(len(selected), len(fields), figsize=(4.1*len(fields), 2.55*len(selected)), squeeze=False)
        for row, case in enumerate(selected):
            ident, observed = case["id"], case["observed"]
            for col, key in enumerate(fields):
                ax = axes[row, col]
                ax.scatter(observed["t"], observed[key], s=13, facecolors="white", edgecolors="#213d50", linewidths=.7, label="Game observations", zorder=3)
                ax.plot(arrays[ident+"__model__t"], arrays[ident+"__model__"+key], color="#c34f38", lw=1.3, label=label)
                ax.set_xlim(0., max(observed["t"][-1], .1))
                ax.set_ylabel(key.replace("_", " ")+" ("+UNITS[key]+")")
                ax.set_xlabel("Missile age (s)")
                ax.grid(alpha=.18)
                if col == 0:
                    ax.set_title(ident, loc="left", weight="bold")
            if row == 0:
                axes[row, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(group+": free missile dynamics from creation", x=.04, ha="left", fontsize=18)
        fig.text(.04, .015, "Original game rows retained. Model interpolation only; no fitted time shifts or extrapolation.\n"+
            ("Historical straight-target assumptions and unsigned runtime G remain explicit limitations." if group.startswith("screenshot") else
             "Native b5 clock including tied observations; decoded target position is external, missile V/pose are not."), fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .97))
        fig.savefig(output/filename, dpi=135)
        plt.close(fig)


def run_suite(label="Current GUI general H2 baseline", output_dir=None, profile_snapshot=None, runner=None, config_transform=None):
    """Run all cases. runner(case, profile) may supply future public-style output."""
    output = Path(output_dir or BASE/"baseline")
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(profile_snapshot)
    arrays, records = {}, []
    runner = runner or current_runtime
    for case in cases:
        profile = copy.deepcopy(case["profile"])
        if config_transform:
            transformed = config_transform(copy.deepcopy(profile["_model_config"]), case)
            profile["_model_config"] = transformed
            case = dict(case, profile=profile)
        result = runner(case, profile)
        prediction = sample_series(result["samples"], case)
        record, aligned = compare_series(case, prediction, result["summary"])
        record["model_runtime_metadata"] = result.get("model", {})
        records.append(record)
        for key, value in prediction.items():
            arrays[case["id"]+"__model__"+key] = value
        for key, value in aligned.items():
            arrays[case["id"]+"__aligned__"+key] = value
        if case["dataset"] == "native_2.59":
            arrays[case["id"]+"__target_source__t"] = case["target_source_t"]
            arrays[case["id"]+"__target_source__xyz"] = case["target_source_xyz"]
        full = record.get("windows", {}).get("full", {}).get("metrics", {})
        print(case["id"], record.get("compared_rows"), {k: round(v["rmse"], 3) for k, v in full.items()}, result["summary"]["termination_event"], flush=True)
    output.joinpath("results.json").write_text(json.dumps(jsonable({"label": label,
        "profile_snapshot": str(profile_snapshot or BASE/"baseline/runtime_profiles.json"),
        "data_versions_separated": True, "native_speed_is_free_state": True,
        "no_time_shift_or_extrapolation": True, "cases": records,
        "acceptance": "Exact equivalence is not declared: missing scenario inputs and unresolved native observation timing/state meanings are disclosed per case."}), indent=2, allow_nan=False))
    np.savez_compressed(output/"trajectories.npz", **arrays)
    plot_cases(cases, arrays, output, label)
    return records, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE/"baseline")
    parser.add_argument("--profile-snapshot", type=Path, default=BASE/"baseline/runtime_profiles.json")
    args = parser.parse_args()
    run_suite(output_dir=args.output_dir, profile_snapshot=args.profile_snapshot)


if __name__ == "__main__":
    main()
