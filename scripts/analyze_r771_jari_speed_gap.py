#!/usr/bin/env python3
"""Conditional axial-force accounting for the existing Jari game references.

No free trajectory is rerun and no coefficient is fitted. Observed missile
states are used only to evaluate forces and a diagnostic force integral.
Jari remains an interim War Thunder reference; this is not game telemetry.
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")

import numpy as np

from compare_r771_jari_offaxis import (FROZEN, SNAPSHOT, free_inputs,
    jsonable, parse_reference, profile_for, sha256, split_runs)
from aim120_model.aerodynamics import body_axes_from_quaternion, pitch_yaw_from_quaternion
from aim120_model.drag_models import area_basis, effective_cda0, interpolate_cx_vs_mach
from aim120_model.dynamics import SimState
from aim120_model.single_surface_game import SingleSurfaceGame, SurfaceCoefficients

OUTPUT = ROOT / "outputs/jari_offaxis/speed_gap"
MANIFEST = ROOT / "outputs/jari_offaxis/reference/manifest.json"
HALF = ROOT / "outputs/jari_offaxis/comparison/half/trajectories.npz"
LOCAL_GLOBAL = ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/config/gameparams.blkx"
OLD_GLOBAL = ROOT / "outputs/r771_force_frames/global_257_source.json"
LOCAL_ROCKET = ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
WINDOWS = (("activation_0_0p7", 0., .7), ("turn_0p7_3p5", .7, 3.5),
           ("late_burn_3p5_7", 3.5, 7.), ("post_burn", 7., math.inf))


def stats(x):
    x = np.asarray(x)
    return {"n": len(x), "mean": float(np.mean(x)), "rmse": float(np.sqrt(np.mean(x*x))),
            "min_median_max": np.quantile(x, [0., .5, 1.]).tolist()} if len(x) else {"n": 0}


def integrate(t, x, lo, hi):
    lo, hi = max(lo, float(t[0])), min(hi, float(t[-1]))
    if hi <= lo:
        return 0.
    inside = (t > lo) & (t < hi)
    ti = np.r_[lo, t[inside], hi]
    xi = np.r_[np.interp(lo, t, x), x[inside], np.interp(hi, t, x)]
    return float(np.trapezoid(xi, ti))


def cumulative(t, x):
    return np.r_[0., np.cumsum(.5*(x[1:]+x[:-1])*np.diff(t))]


def case(entry, profiles):
    run = split_runs(entry["export_file"])[0]
    ref, parser_diagnostic = parse_reference(run)
    profile, source = profile_for(run, entry, profiles)
    config = copy.deepcopy(profile["_model_config"])
    state0, _ = free_inputs(run, ref, config)
    model = SingleSurfaceGame(config)
    model.initialize(state0)
    t = ref["t"]
    v = ref["speed_mps"]
    tangent = ref["tangent"]
    samples = run["samples"]
    axes = [body_axes_from_quaternion((q[3], *q[:3])) for q in ref["quaternion_xyzw"]]
    forward = np.asarray([a.forward for a in axes])
    cosine = np.sum(forward*tangent, axis=1)
    prop = [model.propulsion.sample(float(tt)) for tt in t]
    raw_mass = np.asarray([p.mass_kg for p in prop])
    raw_thrust = np.asarray([p.thrust_n for p in prop])
    scalar_derivative = np.gradient(v, t, edge_order=2)
    vector_derivative = np.sum(np.gradient(ref["velocity_xyz_mps"], t, axis=0,
                                         edge_order=2)*tangent, axis=1)
    gravity_tangent = model.gravity*tangent[:, 1]
    required = raw_thrust*cosine-raw_mass*(scalar_derivative+gravity_tangent)
    required_vector = raw_thrust*cosine-raw_mass*(vector_derivative+gravity_tangent)
    evaluated = []
    for sample, axis in zip(samples, axes):
        q = sample["quaternion"]
        q = (q[3], *q[:3])
        pitch, yaw = pitch_yaw_from_quaternion(q)
        state = SimState(tuple(sample["position"]), tuple(sample["velocity"]),
            pitch, yaw, 0., 0., sample["mass"], orientation_quaternion=q)
        force = model.evaluate(state, sample["t"], (0., 0.))
        atmosphere = model.atmosphere.sample(sample["position"][1])
        mach = sample["speed"]/atmosphere.speed_of_sound_mps
        base = force.qbar*effective_cda0(mach, config)
        # Counterfactual UNPROJECTED body normal, with no source omega units
        # guessed. Rate=0 here; this is neither applied nor a fitted closure.
        nb = force.qbar*model.area*model.coefficients.ca*(
            force.alpha_pitch*np.asarray(axis.up)+force.alpha_yaw*np.asarray(axis.right))
        extra_body_projection = -float(np.dot(nb, np.asarray(sample["velocity"])/sample["speed"]))
        evaluated.append((force.drag_n, base, force.drag_n-base, force.qbar,
            mach, atmosphere.density_kg_m3, atmosphere.speed_of_sound_mps,
            force.alpha_pitch, force.alpha_yaw, extra_body_projection))
    local = np.asarray(evaluated)
    sd = area_basis(config, "caliber_area")
    cd_base = np.asarray([s["cd_base"] for s in samples])
    cd_aoa = np.asarray([s["cd_aoa"] for s in samples])
    site_base = np.asarray([s["drag_base_n"] for s in samples])
    site_aoa = np.asarray([s["drag_aoa_n"] for s in samples])
    site_drag = site_base+site_aoa
    q_site = site_base/(sd*cd_base)
    local_base_cd_caliber = local[:, 1]/(local[:, 3]*sd)
    alpha = np.radians(ref["alpha_deg"])
    sin_sq = np.sin(alpha)**2
    k = np.divide(cd_aoa, config["aerodynamics"]["cx_vs_aoa"]*sin_sq,
                  out=np.zeros(len(t)), where=sin_sq > 1e-12)
    residual = required-local[:, 0]
    excess_site = site_drag-local[:, 0]
    # These integrals retain observed V, orientation, altitude and forces.
    # They are cumulative accounting, NOT a forward integration prediction.
    force_integral_speed = v+cumulative(t, excess_site/raw_mass)
    endpoint_excluded = np.ones(len(t), dtype=bool)
    endpoint_excluded[:2] = False
    endpoint_excluded[-2:] = False
    derivative_valid = endpoint_excluded & (np.abs(t-model.propulsion.burn_time_s) > 2.*np.median(np.diff(t)))
    row = {"case_id": entry["actual_reference_id"], "source": source,
        "source_export": entry["export_file"], "source_sha256": sha256(entry["export_file"]),
        "source_rows": len(t), "original_time_support_s": [float(t[0]), float(t[-1])],
        "parser_diagnostics": parser_diagnostic, "local_force_config": {
            "density": "StandardAtmosphere.sample(observed ASL); standard ISA",
            "sound_speed": "sqrt(1.4*287.05287*T_ISA(ASL))",
            "reference_area_m2": model.area, "caliber_area_m2": sd,
            "wing_area_multiplier": config["geometry"]["wing_area_multiplier"],
            "zero_alpha": "q*S_w*CxK*interpolate_cx_vs_mach(M, runtime drag_model.cx_vs_mach)",
            "zero_alpha_table_source": "project 1943_law_x1_10_linear_interpolation candidate; not a table present in the raw rocket",
            "wing_multiplier_in_base_used_exactly_once": True,
            "aoa_drag": "q*S_caliber*9*min(alpha_pitch^2+alpha_yaw^2,1.2^2); no Mach shape",
            "cx_k": config["aerodynamics"]["cx_k"],
            "cx_aoa": config["aerodynamics"]["cx_vs_aoa"],
            "alpha_cap_rad": config["drag_model"]["alpha_drag_cap_rad"],
            "cap_ever_active": bool(np.any(local[:, 7]**2+local[:, 8]**2 > config["drag_model"]["alpha_drag_cap_rad"]**2)),
            "mach_support": [float(local[:, 4].min()), float(local[:, 4].max())]},
        "propulsion_checks": {
            "mass_export_minus_raw_max_kg": float(np.max(np.abs(ref["mass"]-raw_mass))),
            "thrust_export_minus_raw_max_excluding_creation_n": float(np.max(np.abs(ref["thrust"][1:]-raw_thrust[1:]))),
            "creation_export_thrust_n": float(ref["thrust"][0]),
            "creation_raw_thrust_n": float(raw_thrust[0]),
            "creation_display_thrust_is_initialization_exception": True,
            "mass_and_thrust_are_not_fitted": True},
        "field_semantics_checks": {
            "export_acc_drag_times_mass_minus_display_drag_max_n": float(np.max(np.abs(ref["acc_drag"]*ref["mass"]-site_drag))),
            "export_acc_thrust_times_mass_minus_display_thrust_max_n": float(np.max(np.abs(ref["acc_thrust"]*ref["mass"]-ref["thrust"]))),
            "inferred_q_site_divided_by_local_q": stats(q_site/local[:, 3]),
            "inferred_q_basis": "q_site=drag_base_n/(S_caliber*cd_base); agreement with ISA supports this basis, not private API source proof",
            "acc_lift_not_assumed_total_overload": True},
        "derivative_policy": {
            "source_median_step_s": float(np.median(np.diff(t))),
            "scalar": "second-order np.gradient(speed, source time)",
            "crosscheck": "tangent dot second-order gradient(world velocity)",
            "excluded": "first and last two displayed samples; within two display intervals of raw 7 s motor cutoff",
            "source_48_hz_endpoints_not_reconstructed": True,
            "valid_rows": int(np.sum(derivative_valid))},
        "windows": {}, "residual_alpha_bins": {}, "residual_speed_bins": {},
        "snapshots": []}
    for name, lo, hi in WINDOWS:
        mask = derivative_valid & (t >= lo) & (t <= hi)
        row["windows"][name] = {
            "requested_support_s": [lo, None if math.isinf(hi) else hi],
            "valid_derivative_sample_support_s": [float(t[mask][0]), float(t[mask][-1])] if np.any(mask) else None,
            "required_drag_n": stats(required[mask]), "local_drag_n": stats(local[mask, 0]),
            "required_minus_local_drag_n": stats(residual[mask]),
            "required_minus_site_display_drag_n": stats((required-site_drag)[mask]),
            "scalar_minus_vector_derivative_required_drag_n": stats((required-required_vector)[mask]),
            "required_minus_local_equivalent_delta_cd_on_S_caliber": stats((residual/(local[:, 3]*sd))[mask]),
            "unprojected_body_normal_extra_drag_n_rate_zero": stats(local[mask, 9]),
            "required_minus_local_minus_body_projection_n": stats((residual-local[:, 9])[mask]),
            "site_minus_local_base_drag_n": stats((site_base-local[:, 1])[mask]),
            "site_cd_base_divided_by_local_caliber_normalized_base_cd": stats((cd_base/local_base_cd_caliber)[mask]),
            "site_minus_local_aoa_drag_n": stats((site_aoa-local[:, 2])[mask]),
            "site_vs_local_force_integral_excess_speed_mps": integrate(t, excess_site/raw_mass, lo, hi),
            "same_integral_thrust_cosine_loss_mps": integrate(t, raw_thrust*(1-cosine)/raw_mass, lo, hi),
            "force_integral_is_conditional_not_free_prediction": True}
    for lo, hi in ((0., 2.), (2., 10.), (10., 25.), (25., 40.), (40., 60.)):
        mask = derivative_valid & (ref["alpha_deg"] >= lo) & (ref["alpha_deg"] < hi)
        row["residual_alpha_bins"][f"{lo:g}_{hi:g}_deg"] = stats((residual/(local[:, 3]*sd))[mask])
    for lo, hi in ((330., 420.), (420., 600.), (600., 900.), (900., 1300.)):
        mask = derivative_valid & (v >= lo) & (v < hi)
        row["residual_speed_bins"][f"{lo:g}_{hi:g}_mps"] = stats((residual/(local[:, 3]*sd))[mask])
    for tt in (.5, 1., 1.5, 2., 2.5, 3., 3.5, 6.5, 7.5):
        if tt > t[-1]:
            continue
        i = int(np.argmin(abs(t-tt)))
        row["snapshots"].append({"time_s": float(t[i]), "speed_mps": float(v[i]),
            "alpha_deg": float(ref["alpha_deg"][i]), "required_drag_n": float(required[i]),
            "local_drag_n": float(local[i, 0]), "site_display_drag_n": float(site_drag[i]),
            "local_base_drag_n": float(local[i, 1]), "local_aoa_drag_n": float(local[i, 2]),
            "site_base_drag_n": float(site_base[i]), "site_aoa_drag_n": float(site_aoa[i]),
            "source_cd_base_on_inferred_caliber_area": float(cd_base[i]),
            "local_cd_base_on_caliber_area": float(local_base_cd_caliber[i]),
            "local_1943_table_cx": float(interpolate_cx_vs_mach(local[i, 4])),
            "source_to_local_base_cd_ratio": float(cd_base[i]/local_base_cd_caliber[i]),
            "thrust_cosine_loss_n": float(raw_thrust[i]*(1-cosine[i])),
            "inferred_aoa_sin2_multiplier_over_9": float(k[i]), "derivative_valid": bool(derivative_valid[i])})
    arrays = {**ref, "required_drag_n": required, "required_vector_drag_n": required_vector,
        "local_drag_n": local[:, 0], "local_base_drag_n": local[:, 1], "local_aoa_drag_n": local[:, 2],
        "source_cd_base": cd_base, "local_cd_base_on_caliber_area": local_base_cd_caliber,
        "q_local_pa": local[:, 3], "mach_local": local[:, 4], "site_display_drag_n": site_drag,
        "site_base_drag_n": site_base, "site_aoa_drag_n": site_aoa, "q_inferred_site_pa": q_site,
        "raw_mass_kg": raw_mass, "raw_thrust_n": raw_thrust,
        "body_cosine": cosine, "required_minus_local_drag_n": residual,
        "body_normal_extra_rate_zero_n": local[:, 9], "derivative_valid": derivative_valid,
        "source_aoa_sin2_multiplier_over_9": k, "conditional_force_integral_speed_mps": force_integral_speed}
    return row, arrays


def plot(cases, arrays, prior):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
    colors = {"fixed_RateAngle_baseline": "#8c8c8c", "half_horizontal_PN_and_loft_gate": "#7552a8"}
    for column, degree in enumerate((0, 30, 60)):
        name = f"jari_259_{degree}"
        a = arrays[name]
        t = a["t"]
        top, middle, bottom = ax[:, column]
        top.plot(t, a["speed_mps"], "o-", markersize=2, color="#2176bd", label="Jari reference")
        for variant, color in colors.items():
            prefix = name+"__"+variant+"__"
            top.plot(prior[prefix+"t"], prior[prefix+"speed_mps"], color=color,
                     label="Original free model" if variant.startswith("fixed") else "Half-PN free candidate")
        top.plot(t, a["conditional_force_integral_speed_mps"], "--", color="#cd8720",
                 label="Observed-state force integral (conditional)")
        top.set(title=f"{degree} degree game shot", ylabel="Speed (m/s)", xlim=(0, min(7., t[-1])))
        valid = a["derivative_valid"]
        req = np.where(valid, a["required_drag_n"]/a["raw_mass_kg"], np.nan)
        middle.plot(t, req, color="#111111", label="Required from velocity + thrust cosine")
        middle.plot(t, a["site_display_drag_n"]/a["raw_mass_kg"], color="#2176bd", alpha=.75, label="Jari diagnostic drag")
        middle.plot(t, a["local_drag_n"]/a["raw_mass_kg"], color="#7552a8", label="Local drag at observed state")
        middle.set(ylabel="Drag / mass (m/s²)", xlim=(0, min(7., t[-1])))
        cd = a["required_minus_local_drag_n"]/(a["q_local_pa"]*math.pi*.2**2/4)
        pts = bottom.scatter(a["alpha_deg"][valid], cd[valid], c=a["speed_mps"][valid],
                             cmap="viridis", vmin=330, vmax=1250, s=10)
        bottom.axhline(0., color="#777777", lw=.7)
        bottom.set(xlabel="Observed incidence (degree)", ylabel="Missing drag / (q S_caliber)")
        for aa in (top, middle):
            aa.grid(alpha=.2)
            aa.set_xlabel("Original time (s)")
        bottom.grid(alpha=.2)
        if column == 0:
            top.legend(fontsize=7)
            middle.legend(fontsize=7)
    fig.colorbar(pts, ax=ax[2, :], label="Observed speed (m/s)", shrink=.8)
    fig.suptitle("War Thunder game model: observed-state axial accounting, no parameter fit\n"
                 "Drag gap at the reference speed and attitude; conditional integrals are not free predictions")
    fig.savefig(OUTPUT / "axial_force_accounting.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = [MANIFEST, FROZEN, SNAPSHOT, HALF, ROOT / "scripts/compare_r771_jari_offaxis.py",
                 ROOT / "src/aim120_model/single_surface_game.py", ROOT / "src/aim120_model/drag_models.py",
                 LOCAL_GLOBAL, OLD_GLOBAL, ROOT / "scripts/import_datamine_missiles.py",
                 ROOT / "src/aim120_model/profile_adapter.py", LOCAL_ROCKET]
    manifest = json.loads(MANIFEST.read_text())
    protected += [Path(e["export_file"]) for e in manifest["cases"]]
    before = {str(p): sha256(p) for p in protected}
    frozen = json.loads(FROZEN.read_text())
    coeff = SurfaceCoefficients()
    if frozen["coefficients"] != [coeff.ca, coeff.speed_slope, coeff.control_gain, coeff.restoring_gain] or frozen["fixed_reference_damping_s_minus1"] != coeff.rate_damping:
        raise AssertionError("frozen inner coefficients differ from evaluation defaults")
    profiles = json.loads(SNAPSHOT.read_text())
    cases, arrays = {}, {}
    for entry in manifest["cases"]:
        row, data = case(entry, profiles)
        cases[row["case_id"]] = row
        arrays[row["case_id"]] = data
    # No coefficient/model is inferred here: cross-case interpolation of the
    # site diagnostic factor only tests whether a common speed shape exists.
    base = arrays["jari_259_0"]
    base_mask = (base["alpha_deg"] > .05) & base["derivative_valid"]
    order = np.argsort(base["speed_mps"][base_mask])
    base_v = base["speed_mps"][base_mask][order]
    base_k = base["source_aoa_sin2_multiplier_over_9"][base_mask][order]
    base_cd = base["source_cd_base"][base_mask][order]
    transfer = {}
    for degree in (30, 60):
        name = f"jari_259_{degree}"
        a = arrays[name]
        mask = a["derivative_valid"] & (a["alpha_deg"] > .5) & (a["speed_mps"] >= base_v[0]) & (a["speed_mps"] <= base_v[-1])
        predicted_k = np.interp(a["speed_mps"], base_v, base_k)
        transfer[name] = {"common_speed_multiplier_error": stats((a["source_aoa_sin2_multiplier_over_9"]-predicted_k)[mask]),
            "common_source_base_cd_speed_shape_error": stats((a["source_cd_base"]-np.interp(a["speed_mps"], base_v, base_cd))[mask]),
            "method": "0 degree diagnostic k=cd_aoa/[9 sin²(source incidence)] versus speed, interpolated to separate reference; no coefficient fit, no dynamics insertion",
            "shape_candidate": "q*S_caliber*9*k(M)*sin²(alpha); k speed/Mach semantics and .28 shared source unverified"}
    prior = np.load(HALF)
    plot(cases, arrays, prior)
    # One focused numerical and preservation pass after producing the result.
    checks = {"all_input_hashes_unchanged": before == {str(p): sha256(p) for p in protected},
        "frozen_coefficients_equal_evaluation_defaults": True, "source_rows": sum(c["source_rows"] for c in cases.values()),
        "all_accounting_arrays_finite": all(np.isfinite(v).all() for a in arrays.values() for v in a.values()),
        "export_mass_within_1e_minus4_kg": all(c["propulsion_checks"]["mass_export_minus_raw_max_kg"] < 1e-4 for c in cases.values()),
        "export_thrust_exact_except_creation": all(c["propulsion_checks"]["thrust_export_minus_raw_max_excluding_creation_n"] == 0 for c in cases.values()),
        "export_drag_diagnostic_component_sum_within_1e_minus8_n": all(c["field_semantics_checks"]["export_acc_drag_times_mass_minus_display_drag_max_n"] < 1e-8 for c in cases.values())}
    if not all(value for key, value in checks.items() if key != "source_rows"):
        raise AssertionError(checks)
    document = {"status": "conditional_axial_accounting_complete", "parameter_fit_performed": False,
        "free_model_or_guidance_changed": False, "free_predictions_are_preexisting_only": True,
        "common_frozen_coefficients": dataclasses.asdict(coeff), "input_hashes": before,
        "classification_rules_predeclared": ["raw mass/thrust mismatch", "same observed-state local drag residual",
            "unprojected body-normal drag counterfactual over/under closure", "common AoA/Mach shape versus timing-only gap"],
        "required_drag_equation": "D_required=T_raw*(body_forward dot v_hat)-m_raw*(d|v|/dt+g*v_y/|v|)",
        "existing_shared_source_check": {
            "local_27_shellBallisticsParams": json.loads(LOCAL_GLOBAL.read_text())["shellBallisticsParams"],
            "recorded_57_shellBallisticsParams": json.loads(OLD_GLOBAL.read_text())["shellBallisticsParams"],
            "source_flag_import": "scripts/import_datamine_missiles.py:664 records useCxiMach as provenance metadata only",
            "runtime_consumption": "No Cxi token/implementation in inspected src/aim120_model; profile adapter selects alpha_drag_mach_shape=false",
            "existing_source_contains_multiplier_function_or_table": False,
            "source_search_scope": "local shellBallisticsParams plus importer/profile adapter/aim120_model core; no broader datamine survey",
            "interpretation": "useCxiMach=true makes a missing speed/Mach mapping plausible, but does not prove the measured Jari multiplier formula; .28 shared source remains unverified"},
        "base_drag_source_accounting": {
            "raw_rocket_Cx_fields": {key: value for key, value in json.loads(LOCAL_ROCKET.read_text())["rocket"].items() if "cx" in key.lower()},
            "raw_rocket_has_Cx_Mach_table": False,
            "raw_dragCx_imported_or_consumed_in_inspected_importer_adapter": False,
            "local_base_algebra": "q*S_caliber*1.5*1.4*project_Cx1943_x1p10(M); no other area multiplier",
            "source_cd_base_equals_raw_Cx_Mach_times_CxK": "unverified: no raw Cx(M) table/function in inspected inputs; cannot equate project table to game law",
            "source_ratio_shape_explanation": "source/local base-Cd ratio varies with speed (about1.10 near338m/s to0.86 near625-1200m/s), so a constant wing-area normalization cannot close it",
            "joint_correction_requirement": "0 degree speed agreement partly cancels excess local base drag against deficient AoA drag; correcting only AoA multiplier would expose the base mismatch"},
        "units": {"t": "seconds unchanged", "speed": "m/s", "forces": "N", "mass": "kg", "density": "kg/m³", "q": "Pa", "area": "m²", "angles": "radians in force laws, degrees in source/plots"},
        "cases": cases, "cross_case_aoa_drag_shape": transfer, "verification": checks,
        "limitations": ["Jari .28 is an interim reference, not actual game telemetry.",
            "Entire .28 rocket equals local .27; .28 shared global drag settings remain unverified.",
            "Display output is ~24 Hz; original 48 Hz internal endpoints and terminal impulse details cannot be reconstructed.",
            "Conditional force integrals freeze observed speed, attitude and altitude and cannot be scored as free model predictions.",
            "Body-normal extra is a zero-rate counterfactual; no unverified export omega frame/unit is used.",
            "Wind normal force was projected orthogonal to velocity because existing CxAoA already represents wind drag; adding full body-normal projection risks double counting.",
            "The reference alpha drag speed multiplier is an algebraic diagnostic; its shape is not fitted or promoted into the local model."]}
    (OUTPUT / "results.json").write_text(json.dumps(jsonable(document), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(OUTPUT / "accounting_arrays.npz", **{name+"__"+key: value for name, a in arrays.items() for key, value in a.items()})
    print(json.dumps({"output": str(OUTPUT), "verification": checks,
        "turn_window": {name: {"same_state_drag_deficit_mean_n": row["windows"]["turn_0p7_3p5"]["required_minus_local_drag_n"]["mean"],
            "conditional_missing_drag_delta_v_mps": row["windows"]["turn_0p7_3p5"]["site_vs_local_force_integral_excess_speed_mps"]} for name, row in cases.items()},
        "shape_transfer_rmse": {name: item["common_speed_multiplier_error"]["rmse"] for name, item in transfer.items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
