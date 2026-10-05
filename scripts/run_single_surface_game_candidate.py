#!/usr/bin/env python3
"""Compare the opt-in freely integrated game candidate with all 13 anchors.

The two force bases and two pre-existing loft cap interpretations are declared
before the run. This script never feeds observed missile states to the plant.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from aim120_model.aerodynamics import quaternion_from_pitch_yaw
from aim120_model.dynamics import SimState
from aim120_model.public_api import _case_from_scenario, validate_scenario
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients
from aim120_model.target import TargetModel, TargetState
import compare_unified_game_replays as harness

OUT = ROOT/"outputs/unified_game_replays/single_surface_candidate"
SNAPSHOT = ROOT/"outputs/unified_game_replays/corrected_input/runtime_profiles.json"
FROZEN = ROOT/"config/r771_simple_joint_frozen.json"


def runner_for(force_basis, substeps, cap_unit="deg_s", coefficients=SurfaceCoefficients()):
    def run(case, profile):
        config = copy.deepcopy(profile["_model_config"])
        scenario = validate_scenario(case["scenario"])
        config["guidance"]["lofting_enabled"] = scenario["loft_enabled"]
        simulation_case = _case_from_scenario(scenario)
        if "_creation_state" in case:
            state = dataclasses.replace(case["_creation_state"])
            # Native source stores xyzw. Keep all creation quaternion components.
            x,y,z,w = case["creation"]["u16_1"]
            state.orientation_quaternion = (w,x,y,z)
            target_at = case["_target_model"].state_at
        else:
            initial = simulation_case["initial_conditions"]
            pitch, yaw = map(math.radians, (initial["launch_angle_deg"], initial["launch_yaw_deg"]))
            quaternion = quaternion_from_pitch_yaw(pitch, yaw)
            from aim120_model.aerodynamics import body_axes_from_quaternion
            forward = body_axes_from_quaternion(quaternion).forward
            state = SimState((0., initial["launch_altitude_m"], 0.),
                tuple(x*initial["start_speed_kmh"]/3.6 for x in forward), pitch,yaw,0.,0.,
                config["geometry"]["initial_mass_kg"], orientation_quaternion=quaternion)
            target_model = TargetModel(initial, config["atmosphere"]["gravity_mps2"])
            target0 = target_model.initial_state
            if target_model.turn_g != 0:
                raise ValueError("historical suite requires straight target; no invented maneuver")
            target_at = lambda t: TargetState(tuple(x+t*v for x,v in zip(target0.position,target0.velocity)), target0.velocity)
        model = SingleSurfaceGame(config, coefficients=coefficients, force_basis=force_basis,
            plant_substeps=substeps, loft_cap_unit=cap_unit)
        result = model.run(state, target_at, end_time_s=scenario["max_simulation_time_s"])
        result["model"]["coefficients"] = dataclasses.asdict(coefficients)
        result["model"]["raw_PID_preserved"] = True
        result["model"]["finsLatAccel_dynamic_mapping_identified"] = False
        return result
    return run


def suite(force_basis, substeps, cap_unit="deg_s", coefficients=SurfaceCoefficients(), output=None):
    folder = output or OUT/(force_basis+"_"+cap_unit)
    records, arrays = harness.run_suite(label="Free single surface: "+force_basis+" / "+cap_unit,
        output_dir=folder, profile_snapshot=SNAPSHOT,
        runner=runner_for(force_basis,substeps,cap_unit,coefficients))
    return records, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--variant", choices=("wind_linear", "body_linear", "all"), default="all")
    parser.add_argument("--cap-unit", choices=("deg_s", "rad_s", "both"), default="both")
    args = parser.parse_args()
    frozen_before = FROZEN.read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ("wind_linear","body_linear") if args.variant == "all" else (args.variant,)
    units = ("deg_s","rad_s") if args.cap_unit == "both" else (args.cap_unit,)
    manifest = {"model": "free 3D game-replay candidate with one equivalent surface",
        "frozen_configuration_sha256": hashlib.sha256(frozen_before).hexdigest(),
        "frozen_source_version": json.loads(frozen_before)["source_version"],
        "common_coefficients": dataclasses.asdict(SurfaceCoefficients()),
        "predeclared_force_bases": list(variants), "predeclared_loft_cap_units": list(units),
        "fit_performed": False, "production_changed": False, "goal_complete": False,
        "observation_forcing_after_creation": False,
        "raw_PID_preserved": True, "control_clock_hz":48, "plant_substeps":args.substeps,
        "loft_raw_targetElevation_unused_by_existing_simple_law": True,
        "force_formula": "wind: L=q*S*Ca*local_alpha in wind-normal axes; body: project(q*S*Ca*local_alpha in body normal axes) perpendicular to flow; existing wind D used once",
        "moment_formula": "qdot=Ku*(1+b*(V/V0-1))*u-A*Ka*(alpha+l*omega/V)-B*Kw*omega; A=(q/m)/(q0/m0), B=A*V0/V; M=-l*Fbody_normal+Mref, same station",
        "limitations": ["Shared moment coefficients originally calibrated conditionally on 2.59R771; their transfer is being tested, not asserted.",
            "Other-missile Ca and control gain are not independently identified by this candidate.",
            "Source observation, target track and scenario limitations remain in each comparison.",
            "No maximum-AoA envelope or Jari agreement is inferred from a force-law algebraic check.",
            "Raw finsLatAccel has no identified dynamic mapping in this empirical family."]}
    results = {}
    for force in variants:
        for unit in units:
            records, _ = suite(force,args.substeps,unit)
            results[force+"_"+unit] = [{"id":r["id"], "compared_rows":r.get("compared_rows"),
                "unrepresented_rows":r.get("unrepresented_rows"), "summary":r["model_summary"],
                "full_metrics":r.get("windows",{}).get("full",{}).get("metrics",{}),
                "events":r.get("events",{})} for r in records]
    manifest["variants"] = results
    if FROZEN.read_bytes() != frozen_before:
        raise AssertionError("frozen configuration changed")
    (OUT/"candidate_summary.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
