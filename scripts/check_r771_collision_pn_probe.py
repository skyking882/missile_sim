#!/usr/bin/env python3
"""Two game-only collision/head-on probes with original configured loft.

Public JARI exports are interim simulator references. PN/loft reconstructed
on their states is diagnostic only; each local candidate integrates freely
from its creation state. No fitting, new capture term, or runtime modification.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")
import numpy as np
import check_r771_horizontal_sinc_cap as best
import compare_r771_jari_offaxis as comparison
import fit_r771_jari_capture as fields
from aim120_model.aerodynamics import body_axes_from_quaternion, wind_normal_basis
from aim120_model.guidance import estimate_time_to_go_s, interpolate_table, pn_acceleration
from aim120_model.math3d import dot, norm, scale, sub
from aim120_model.single_surface_game import SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUT = ROOT / "outputs/jari_offaxis/collision_pn_probe"
CASES = ("headon", "collision")
WINDOWS = (("activation_0p3_0p7", .3, .7), ("startup_14over48_0p7", 14 / 48, .7),
           ("early_0_3p5", 0., 3.5), ("platform_0p7_2p3", .7, 2.3), ("full_common", 0., math.inf))
CHANNELS = ("speed_mps", "alpha_deg", "normal_specific_load_g", "body_yaw_deg", "body_yaw_rate_deg_s", "body_pitch_deg")


def serializable(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return comparison.jsonable(value)


def protected():
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in ("src", "config", "configs", "schemas", "docs", "scripts", "data", "outputs/jari_offaxis")
            for p in (ROOT / folder).rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and ".git" not in p.parts
            and OUT not in p.parents and p != Path(__file__).resolve()}


def stats(a):
    a = np.asarray(a)
    return {"n": len(a), "mean": float(np.mean(a)), "RMS": float(np.sqrt(np.mean(a * a))),
            "min": float(np.min(a)), "max": float(np.max(a))}


def source_geometry(run, ref, cfg):
    """Current source-state geometry and existing candidate equations only."""
    t = ref["t"]
    q = fields.gap.continuous_quaternions(ref["quaternion_xyzw"][:, [3, 0, 1, 2]])
    g = fields.gap.trajectory_fields(t, ref["position_xyz_m"], ref["velocity_xyz_mps"],
        run["scenario"]["target_position"], run["scenario"]["target_velocity"], q)
    observed = fields.observed_fields(run, ref)
    observed["body_pitch_deg"] = g["body_pitch_deg"]
    observed["body_yaw_rate_5point_deg_s"] = np.degrees(-fields.gap.body_rates(q, t, 5)[:, 1])
    arrays = {"t": t, **{k: g[k] for k in ("los_rate_xyz_rad_s", "los_bearing_rate_rad_s", "body_yaw_deg", "body_pitch_deg")},
              "source_controls_raw": np.asarray([s["controls"] for s in run["samples"]]),
              "source_omega_raw": ref["omega_xyz_export_units"], "source_reported_loft": [s.get("loft") for s in run["samples"]]}
    values = {k: [] for k in ("full_relative_horizontal_PN_G", "full_relative_world_PN_mps2", "candidate_unslewed_request_G",
                             "hit_gain", "candidate_loft_G", "candidate_loft_horizontal_gate", "candidate_loft_current_weight", "candidate_loft_display_minimum_weight")}
    target_v = tuple(run["scenario"]["target_velocity"])
    target0 = np.asarray(run["scenario"]["target_position"])
    gc = cfg["guidance"]
    minimum_weight = 1.
    initial_range = None
    for i, ts in enumerate(t):
        rel = tuple(target0 + ts * np.asarray(target_v) - ref["position_xyz_m"][i])
        rv = sub(target_v, tuple(ref["velocity_xyz_mps"][i]))
        axes = body_axes_from_quaternion(tuple(q[i]))
        wb = wind_normal_basis(axes, tuple(ref["velocity_xyz_mps"][i]))
        pn0, closing, los = pn_acceleration(rel, rv, tuple(ref["velocity_xyz_mps"][i]), float(gc["pn_gain"]))
        distance = norm(rel)
        if initial_range is None:
            initial_range = distance
        tgo = estimate_time_to_go_s(distance, closing, rv, cfg)
        hit = interpolate_table(tgo, gc["time_to_hit_gain_table"])
        pn0 = np.asarray(pn0) * hit
        alternative = float(gc["pn_gain"]) * hit * np.cross(los, -np.asarray(rv))
        right = np.asarray(wb.right)
        mixed = pn0 + right * float((alternative - pn0) @ right)
        threshold = 1.3 * math.radians(abs(float(gc["loft_omega_max_deg_s"])))
        multiplier = float(gc["angle_to_acceleration_multiplier"])
        z = max(-60., min(60., multiplier * (norm(los) / threshold - 1.)))
        current_weight = 1. / (1. + math.exp(z))
        minimum_weight = min(minimum_weight, current_weight)
        xi = max(0., min(1., distance / initial_range))
        eta = math.atan2(-dot(rel, axes.up), dot(rel, axes.forward))
        positive = float(gc["lofting_elevation_deg"]) - multiplier * max(eta, 0.)
        loft = .75 * xi ** 2 * (positive * minimum_weight + float(gc["target_elevation_deg"]) * (1. - minimum_weight))
        z_gate = max(-60., min(60., multiplier * (1. - abs(los[1]) / math.radians(abs(float(gc["loft_omega_max_deg_s"]))))))
        gate = (1. / (1. + math.exp(-z_gate))) / (1. / (1. + math.exp(-max(-60., min(60., multiplier)))))
        gate = max(0., min(1., gate)) if abs(los[1]) != 0. else 1.
        loft *= gate
        boundary = best.dec.outer_boundaries(tuple(mixed), loft, wb, G0_MPS2, float(gc["maximum_lateral_acceleration_g"]) * G0_MPS2)
        for k, v in {"full_relative_horizontal_PN_G": float(mixed @ right) / G0_MPS2,
                     "full_relative_world_PN_mps2": mixed, "candidate_unslewed_request_G": boundary["unslewed_request_G"],
                     "hit_gain": hit, "candidate_loft_G": loft, "candidate_loft_horizontal_gate": gate,
                     "candidate_loft_current_weight": current_weight, "candidate_loft_display_minimum_weight": minimum_weight}.items():
            values[k].append(v)
    arrays.update({k: np.asarray(v) for k, v in values.items()})
    rel0 = np.asarray(run["samples"][0]["target"]) - ref["position_xyz_m"][0]
    rv0 = np.asarray(target_v) - ref["velocity_xyz_mps"][0]
    initial = {"position_m": ref["position_xyz_m"][0], "missile_velocity_mps": ref["velocity_xyz_mps"][0],
               "quaternion_xyzw": ref["quaternion_xyzw"][0], "target_velocity_mps": target_v,
               "r_m": rel0, "relative_velocity_mps": rv0, "LOS_vector_unflushed_rad_s": np.cross(rel0, rv0) / float(rel0 @ rel0),
               "horizontal_LOS_rate_rad_s": float(g["los_bearing_rate_rad_s"][0]),
               "offaxis_deg": float(np.degrees(np.arctan2(rel0[2], rel0[0])))}
    return observed, arrays, initial


def metrics(observed, pred):
    result = {}
    for name, lo, hi in WINDOWS:
        mask = (observed["t"] >= lo - 1e-9) & (observed["t"] <= min(hi, pred["t"][-1]) + 1e-9)
        ts = observed["t"][mask]
        result[name] = {key: stats(np.interp(ts, pred["t"], pred[key]) - observed[key][mask]) for key in CHANNELS}
    return result


def sampled_rows(observed, a, pred):
    rows = []
    for nominal in (0., 14 / 48, 1 / 3, .375, 5 / 12, .5, 13 / 24, 7 / 12, 2 / 3):
        i = int(np.argmin(abs(a["t"] - nominal)))
        ts = float(a["t"][i])
        row = {"actual_source_time_s": ts, "source_horizontal_LOS_rate_rad_s": float(a["los_bearing_rate_rad_s"][i]),
               "source_state_existing_full_relative_PN_G": float(a["full_relative_horizontal_PN_G"][i]),
               "source_state_candidate_loft_G_diagnostic": float(a["candidate_loft_G"][i]),
               "source_state_candidate_loft_horizontal_gate_diagnostic": float(a["candidate_loft_horizontal_gate"][i]),
               "source_controls_raw": a["source_controls_raw"][i], "source_omega_raw": a["source_omega_raw"][i]}
        for key in CHANNELS:
            row[key] = {"source": float(observed[key][i]), "local_free": float(np.interp(ts, pred["t"], pred[key]))}
        row["source_body_yaw_rate_5point_deg_s"] = float(observed["body_yaw_rate_5point_deg_s"][i])
        rows.append(row)
    return rows


def plot(all_cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(7, 2, figsize=(15, 15), sharex=True)
    for j, cid in enumerate(CASES):
        o, a, p = all_cases[cid]
        am, om, pm = a["t"] <= 3.5, o["t"] <= 3.5, p["t"] <= 3.5
        ax[0, j].plot(a["t"][am], a["full_relative_horizontal_PN_G"][am], color="#9247ac", label="Source-state existing full horizontal PN")
        ax[0, j].set_ylabel("Horizontal PN (G)")
        for row, key, units in ((1,"body_yaw_deg","Yaw (deg)"),(2,"body_yaw_rate_deg_s","Yaw rate (deg/s)"),
                                (3,"body_pitch_deg","Pitch (deg)"),(4,"alpha_deg","AoA (deg)"),
                                (5,"speed_mps","Speed (m/s)"),(6,"normal_specific_load_g","Normal load (G)")):
            ax[row, j].plot(o["t"][om], o[key][om], color="#137e83", label="JARI source")
            ax[row, j].plot(p["t"][pm], p[key][pm], color="#dc8c30", label="Current best actual free")
            if row == 2:
                ax[row,j].plot(o["t"][om], o["body_yaw_rate_5point_deg_s"][om], "--", color="#528baa", label="Source quaternion 5-point")
            ax[row, j].set_ylabel(units)
        ax[0, j].set_title(cid + ": original configured loft")
        for axis in ax[:,j]:
            axis.set_xlim(0, 3.5); axis.grid(alpha=.2); axis.axvspan(.3,.7,color="#cbd9e5",alpha=.18); axis.legend(fontsize=7)
        ax[-1,j].set_xlabel("Original time (s)")
    fig.suptitle("War Thunder game: paired target-velocity probe, original configured loft\nSource-state PN diagnostics do not feed either local free trajectory", fontsize=12)
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(OUT/"paired_original_configured_loft.png",dpi=145);plt.close(fig)


def main():
    receipt = protected()
    # The previous head-on forward completed but saving failed on numpy.bool_.
    # It is counted as an actual failed-save run, never described as recovered.
    json.dumps(serializable({"antiwindup_blocked": np.bool_(True), "ticks": [{"flag": np.bool_(False)}]}))
    prior = json.loads((OUT/"results.json").read_text())
    runs = {c: json.loads((OUT/(c+"_config_response.json")).read_text()) for c in CASES}
    assert runs["headon"]["profile_source"] == runs["collision"]["profile_source"]
    assert runs["headon"]["backend"] == runs["collision"]["backend"]
    assert runs["headon"]["profile_source"] == json.loads((comparison.DEFAULT_MANIFEST.parent/"jari_259_30_response.json").read_text())["profile_source"]
    assert all(r["scenario"]["loft_state"] == "config" for r in runs.values())
    assert all(runs["headon"]["scenario"][k] == runs["collision"]["scenario"][k] for k in runs["headon"]["scenario"] if k != "target_velocity")
    profiles = json.loads(comparison.SNAPSHOT.read_text())
    coefficients = best.dec.audit.drag.DragCoefficients(**json.loads((best.dec.audit.drag.OUTPUT/"fit.json").read_text())["coefficients"])
    cases = {}; arrays = {}; plot_cases = {}; new_free = 0
    for cid in CASES:
        run = runs[cid]
        request = json.loads((OUT/(cid+"_config_request.json")).read_text())
        assert all(run["scenario"][k] == v for k,v in request.items())
        ref, diagnostic = comparison.parse_reference(run)
        profile, source = comparison.profile_for(run, {}, profiles)
        cfg = copy.deepcopy(profile["_model_config"])
        initial, target_at = comparison.free_inputs(run, ref, cfg)
        model = best.make_model(cfg, coefficients)
        path = OUT/(cid+"_best_free.json")
        if path.exists():
            raise FileExistsError("This bounded probe requires exactly one new free run per case, not a silent rerun")
        result = model.run(dataclasses.replace(initial), target_at, end_time_s=cfg["performance"]["lifetime_s"])
        new_free += 1
        path.write_text(json.dumps(serializable(result),ensure_ascii=False)+"\n")
        pred = fields.prediction_fields(result, model, run["scenario"])
        observed, diag_arrays, source_initial = source_geometry(run, ref, cfg)
        assert model.config == cfg and model.coefficients == SurfaceCoefficients() and model.drag_coefficients == coefficients
        assert model.controller_hz == 48 and model.plant_substeps == 4 and model.arm == .4
        assert model.relative_blend == 1. and model.request_slew_g_s == 110.
        assert (model.amplitude_coefficient,model.transition_coefficient) == (.75,1.3)
        target_error = max(float(np.linalg.norm(np.asarray(s["target"]) - (np.asarray(run["scenario"]["target_position"]) + s["t"]*np.asarray(run["scenario"]["target_velocity"])))) for s in run["samples"])
        assert target_error < 1e-7
        activation = (ref["t"] >= .3) & (ref["t"] <= .7)
        summary = {k:stats(diag_arrays[k][activation]) for k in ("los_bearing_rate_rad_s","full_relative_horizontal_PN_G","candidate_loft_G","candidate_loft_horizontal_gate")}
        summary["yaw_rate_3point_deg_s"] = stats(observed["body_yaw_rate_deg_s"][activation])
        summary["yaw_rate_5point_deg_s"] = stats(observed["body_yaw_rate_5point_deg_s"][activation])
        cases[cid] = {"actual_source_profile":run["profile_source"],"actual_backend":run["backend"],"scenario":run["scenario"],
            "source_version_bridge":source,"source_diagnostics":diagnostic,"actual_source_creation":source_initial,
            "source_target_linear_motion_max_error_m":target_error,"source_summary":run["summary"],"local_free_summary":result["summary"],
            "raw_PID":cfg["control"]["pid"],"source_state_activation_diagnostic":summary,"free_vs_source_errors":metrics(observed,pred),
            "sampled_actual_values":sampled_rows(observed,diag_arrays,pred),
            "frozen_invariants_checked":True,"local_force_moment_closure":best.dec.audit.drag.force_bookkeeping(model,result)}
        for label, payload in (("reference",observed),("diagnostic",diag_arrays),("local_free",pred)):
            for k,v in payload.items():
                if k != "source_reported_loft": arrays[cid+"__"+label+"__"+k] = np.asarray(v)
        plot_cases[cid] = observed, diag_arrays, pred
    assert abs(cases["collision"]["actual_source_creation"]["horizontal_LOS_rate_rad_s"]) < 1e-16
    assert np.array_equal(runs["headon"]["samples"][0]["velocity"],runs["collision"]["samples"][0]["velocity"])
    assert np.array_equal(runs["headon"]["samples"][0]["quaternion"],[0.,0.,0.,1.])
    assert protected() == receipt
    np.savez(OUT/"trajectories.npz",**arrays)
    plot(plot_cases)
    output = {"status":"completed_two_original_configured_loft_source_and_free_cases", "prior_off_request_rejection":prior["actual_HTTP_attempts"],
        "new_simulation_POST_attempts_total":3,"successful_new_JARI_source_cases":2,"new_free_forward_runs":new_free+1,
        "distinct_free_cases":2,"successfully_saved_free_runs":new_free,"failed_save_free_runs":1,
        "failed_save_reason":"First head-on free run completed, then JSON serialization rejected numpy.bool_ in tick antiwindup_blocked; process ended, no state was recovered. One explicit root-authorized rerun completed the case.",
        "loft_mode":"original configured loft; not disabled", "fitting_scans_new_coefficients":0,"reference_feedback_in_runtime":False,
        "case_pair_only_scenario_difference":"target_velocity", "all_source_profiles_backends_equal_existing_259_reference":True,
        "cases":cases,"protected_size_mtime_receipts_equal":True,"protected_file_count":len(receipt),
        "limitations":["Public JARI model is an interim reference, not captured game truth or private source.",
            "The collision-course invariant applies only at creation; thrust and loft change the later LOS rate.",
            "PN and loft diagnostics are the existing candidate equations evaluated on source states, not exported JARI internal commands.",
            "Candidate loft one-way weight here commits only at exported display samples; omitted 48Hz states prevent exact source-cadence guidance replay.",
            "Source controls are preserved as raw indices0/1; their fin-angle or controlYaw semantics are not assumed.",
            "Raw source omega units/frame remain unverified; quaternion3/5 rates are separate diagnostic estimates.",
            "Normal load is display-cadence trajectory-normal specific load, including normal thrust; derivative onset/endpoints have cadence uncertainty.",
            "This new pair is not mixed into the previous candidate acceptance gate."]}
    (OUT/"results.json").write_text(json.dumps(serializable(output),indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(serializable({"status":output["status"],"new_free_forward_runs_actual":new_free+1,"cases":{c:{"activation":cases[c]["source_state_activation_diagnostic"],"errors":cases[c]["free_vs_source_errors"]} for c in CASES}}),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
