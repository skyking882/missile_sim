#!/usr/bin/env python3
"""Offline JARI pitch-response diagnostic; no coefficient or runtime changes.

The only counterfactual substitutes the components of one total local
incidence rotation vector for the two independent projected incidence angles.
It is evaluated on recorded source states, never as a free trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aim120_model.effective_surface_core import EffectiveSurfaceGame, read_raw
import fit_r771_jari_only as base
import prepare_r771_jari_only as ds

OUT = ROOT / "outputs/jari_only/pitch_diagnosis"
CONFIG = ROOT / "research/legacy/config/r771_jari_only_corrected.json"
CASE = "angle_p75"


def protected() -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for folder in ("src", "scripts", "config", "configs", "data", "outputs/jari_offaxis",
                           "outputs/jari_only/dataset", "outputs/jari_only/calibration",
                           "outputs/jari_only/correction", "outputs/jari_only/control_forcing")
            for p in (ROOT / folder).rglob("*") if p.is_file()
            and p.resolve() != Path(__file__).resolve() and "__pycache__" not in p.parts}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def conditioned(source: dict[str, np.ndarray], model: EffectiveSurfaceGame) -> dict[str, np.ndarray]:
    t, velocity = source["t"], source["velocity_xyz_mps"]
    w = -source["omega_export_raw"]
    bv = np.column_stack([np.sum(velocity * source[k], axis=1)
                          for k in ("body_forward", "body_up", "body_right")])
    # Exactly the existing station velocity convention; this is not changed.
    local = bv + np.column_stack((np.zeros(len(t)), -model.ell*w[:, 2], model.ell*w[:, 1]))
    ap = np.arctan2(-local[:, 1], local[:, 0])
    ay = np.arctan2(-local[:, 2], local[:, 0])
    lateral_length = np.linalg.norm(local[:, 1:], axis=1)
    alpha = np.arctan2(lateral_length, local[:, 0])
    rotation_components = np.zeros((len(t), 2))
    nonzero = lateral_length > 0.
    rotation_components[nonzero] = (alpha[nonzero, None] * -local[nonzero, 1:]
                                   / lateral_length[nonzero, None])
    # [pitch, yaw] component magnitudes/signs of the shortest nose-to-flow
    # angular displacement. The angular velocity axes remain [X,Y,Z].
    ps = np.asarray([.5*model.atmosphere.sample(float(p[1])).density_kg_m3*v*v/model.mass_thrust(float(time))[0]/model.qm0
                     for time, p, v in zip(t, source["position_xyz_m"], source["speed_mps"])])
    damping_scale = ps*model.v0/source["speed_mps"]
    ku, ka, kw = (model.p[k] for k in ("control_response_rad_s2", "restoring_response_rad_s2_per_rad", "damping_response_per_s"))
    controls = source["controls_export_raw"]
    control_terms = np.column_stack((-ku*ps*controls[:, 0], ku*ps*controls[:, 1]))
    restoring_terms = np.column_stack((ka*ps*ay, -ka*ps*ap))
    vector_restoring_terms = np.column_stack((ka*ps*rotation_components[:, 1], -ka*ps*rotation_components[:, 0]))
    damping_terms = -kw*damping_scale[:, None]*w[:, 1:]
    model_wd = control_terms+restoring_terms+damping_terms
    vector_wd = control_terms+vector_restoring_terms+damping_terms
    actual_wd, acceleration = [], []
    for i in range(len(t)):
        state = tuple(source["position_xyz_m"][i])+tuple(velocity[i])+tuple(source["quaternion_xyzw_normalized"][i])+tuple(w[i])
        d, diagnostic = model.plant(float(t[i]), state, tuple(controls[i]))
        actual_wd.append(d[-2:])
        acceleration.append(diagnostic["acceleration_xyz_mps2"])
    assert np.allclose(actual_wd, model_wd, rtol=1e-12, atol=1e-12)
    cosine = np.sqrt(1-source["body_forward"][:, 1]**2)
    rate_terms = np.column_stack((w[:, 2]*source["body_up"][:, 1],
                                  -w[:, 1]*source["body_right"][:, 1]))/cosine[:, None]
    return {"t": t, "omega_body": w, "omega_from_quaternion": source["quaternion_body_rate_5point_xyz_rad_s"],
            "omega_derivative_5point": ds.derivative(w, t, 5),
            "omega_derivative_3point": ds.derivative(w, t, 3),
            "quaternion_rate_derivative_5point": ds.derivative(source["quaternion_body_rate_5point_xyz_rad_s"], t, 5),
            "local_velocity_body": local, "independent_incidence_pitch_yaw_rad": np.column_stack((ap, ay)),
            "total_incidence_rotation_pitch_yaw_rad": rotation_components,
            "total_local_incidence_rad": alpha, "pressure_mass_scale": ps, "damping_scale": damping_scale,
            "control_terms_wdot_YZ": control_terms, "restoring_terms_wdot_YZ": restoring_terms,
            "vector_restoring_terms_wdot_YZ": vector_restoring_terms, "damping_terms_wdot_YZ": damping_terms,
            "frozen_model_wdot_YZ": model_wd, "offline_vector_wdot_YZ": vector_wd,
            "frozen_model_acceleration": np.asarray(acceleration), "world_pitch_rate_terms_rad_s": rate_terms}


def residual_metrics(source: dict[str, np.ndarray], a: dict[str, np.ndarray], lo: float, hi: float) -> dict:
    mask = (source["t"] >= lo)&(source["t"] <= hi)&source["derivative_5point_valid_mask"]
    actual = a["omega_derivative_5point"][:, 1:]
    output = {}
    for name in ("frozen_model_wdot_YZ", "offline_vector_wdot_YZ"):
        error = a[name][mask]-actual[mask]
        output[name] = {axis: {"rmse_rad_s2": base.rmse(error[:, i]), "mean_signed_rad_s2": float(error[:, i].mean())}
                        for i, axis in enumerate(("horizontal_Y", "vertical_Z"))}
    output["rows"] = int(mask.sum())
    output["source_omega_vs_quaternion_vertical_rate_rmse_rad_s"] = base.rmse((a["omega_from_quaternion"]-a["omega_body"])[mask, 2])
    output["source_omega_derivative_vs_quaternion_derivative_vertical_rmse_rad_s2"] = base.rmse((a["omega_derivative_5point"]-a["quaternion_rate_derivative_5point"])[mask, 2])
    output["3point_vs_5point_source_vertical_wdot_rmse_rad_s2"] = base.rmse((a["omega_derivative_3point"]-a["omega_derivative_5point"])[mask, 2])
    output["same_state_vertical_linear_acceleration_error_rmse_mps2"] = base.rmse((a["frozen_model_acceleration"]-source["acceleration_5point_xyz_mps2"])[mask, 1])
    return output


def prediction_geometry(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    f, up, right = ds.body_axes_xyzw(p["quaternion_xyzw"])
    w = p["body_omega_xyz_rad_s"]
    cosine = np.sqrt(1-f[:, 1]**2)
    return {"flight_path_elevation_deg": np.degrees(np.arcsin(p["velocity_xyz_mps"][:, 1]/p["speed_mps"])),
            "world_pitch_rate_terms_rad_s": np.column_stack((w[:, 2]*up[:, 1], -w[:, 1]*right[:, 1]))/cosine[:, None]}


def plot(source: dict[str, np.ndarray], a: dict[str, np.ndarray], cached: dict[str, dict[str, np.ndarray]],
         geometry: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    mask = (source["t"] <= 3.5)
    t = source["t"][mask]
    colors = {"current_launch_only": "#a2a7af", "advance_one_tick": "#2a7d53"}
    labels = {"current_launch_only": "Current closed loop", "advance_one_tick": "JARI controls, +dt"}
    ax = axs[0, 0]
    ax.plot(t, source["body_pitch_deg"][mask], "k", lw=2, label="JARI")
    for key, p in cached.items():
        ax.plot(p["t"], p["body_pitch_deg"], color=colors[key], ls="--" if key.startswith("current") else "-", label=labels[key])
    ax.set_title("World nose pitch [deg]")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(t, source["controls_export_raw"][mask, 1], "k", lw=2, label="JARI vertical export")
    ax.plot(cached["current_launch_only"]["t"], cached["current_launch_only"]["controls_raw"][:, 1], color=colors["current_launch_only"], ls="--", label="Current vertical export")
    ax.set_title("Vertical control: recorded channels")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    valid = mask & source["derivative_5point_valid_mask"] & (source["t"] >= .75)
    tt = source["t"][valid]
    ax.plot(tt, a["omega_derivative_5point"][valid, 2], "k", lw=2, label="Derivative of source omega")
    ax.plot(tt, a["quaternion_rate_derivative_5point"][valid, 2], color="#777777", ls=":", label="Independent quaternion estimate")
    ax.plot(tt, a["frozen_model_wdot_YZ"][valid, 1], color="#b64f37", label="Frozen plant on source state")
    ax.plot(tt, a["offline_vector_wdot_YZ"][valid, 1], color="#32699f", label="Total-angle components, offline only")
    ax.set_title("Same-state vertical angular acceleration [rad/s²]")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    for key, label, color, ls in (("control_terms_wdot_YZ", "Control", "#474747", "-"),
                                  ("restoring_terms_wdot_YZ", "Independent-angle restoring", "#b64f37", "-"),
                                  ("vector_restoring_terms_wdot_YZ", "Total-angle restoring, offline", "#32699f", "-"),
                                  ("damping_terms_wdot_YZ", "Damping", "#936bab", ":")):
        ax.plot(t, a[key][mask, 1], color=color, ls=ls, label=label)
    ax.set_title("Same-state vertical angular-response terms [rad/s²]")
    ax.legend(fontsize=8)
    ax = axs[2, 0]
    ax.plot(t, source["body_pitch_deg"][mask], "k", label="JARI nose")
    gamma = np.degrees(np.arcsin(source["velocity_xyz_mps"][:, 1]/source["speed_mps"]))
    ax.plot(t, gamma[mask], "k--", label="JARI flight path")
    p = cached["advance_one_tick"]
    ax.plot(p["t"], p["body_pitch_deg"], color=colors["advance_one_tick"], label="Forced nose")
    ax.plot(p["t"], geometry["advance_one_tick"]["flight_path_elevation_deg"], color=colors["advance_one_tick"], ls="--", label="Forced flight path")
    ax.set_title("Nose and flight-path elevation [deg]")
    ax.legend(fontsize=8)
    ax = axs[2, 1]
    ax.plot(t, np.degrees(a["independent_incidence_pitch_yaw_rad"][mask, 0]), color="#b64f37", label="atan2(-vY, vX)")
    ax.plot(t, np.degrees(a["total_incidence_rotation_pitch_yaw_rad"][mask, 0]), color="#32699f", label="alpha × (-vY) / transverse speed")
    ax.set_title("Local vertical restoring-angle candidates [deg]")
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.set_xlim(0, 3.5)
        ax.grid(alpha=.2)
        ax.axvspan(.75, 2.5, color="#888888", alpha=.06)
    for ax in axs[2]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("75° JARI pitch diagnostic: fixed coefficients; one offline geometry counterfactual", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(OUT/"pitch_overlay.png", dpi=150)
    plt.close(fig)


def main() -> None:
    before = protected()
    dependencies = [CONFIG, base.RAW, ROOT/"src/aim120_model/effective_surface_core.py", ds.OUT/"manifest.json",
                    ds.OUT/"observations.npz", ds.OUT/"creation_inputs.json",
                    ROOT/"outputs/jari_only/control_forcing/predictions.npz"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dependencies}
    raw = read_raw(base.RAW)
    config = json.loads(CONFIG.read_text())
    metadata, source = ds.load_case(CASE)
    model = EffectiveSurfaceGame(raw, config, ds.creation_input(CASE))
    a = conditioned(source, model)
    with np.load(dependencies[-1], allow_pickle=False) as z:
        cached = {name: {key.split("__", 1)[1]: z[key].copy() for key in z.files if key.startswith(name+"__")}
                  for name in ("current_launch_only", "advance_one_tick")}
    geom = {name: prediction_geometry(p) for name, p in cached.items()}
    windows = {"0p75_to_2p5s": residual_metrics(source, a, .75, 2.5),
               "1p5_to_2s": residual_metrics(source, a, 1.5, 2.),
               "2p5_to_3p5s": residual_metrics(source, a, 2.5, 3.5)}
    phase_checks = {}
    mask = (source["t"] >= .75)&(source["t"] <= 2.5)&source["derivative_5point_valid_mask"]
    ku = model.p["control_response_rad_s2"]
    for tick in (-1, 0, 1):
        controls = base.interpolate(source["controls_export_raw"], source["t"], source["t"][mask]+tick*model.dt)
        pred = ku*a["pressure_mass_scale"][mask]*controls[:, 1]+a["restoring_terms_wdot_YZ"][mask, 1]+a["damping_terms_wdot_YZ"][mask, 1]
        error = pred-a["omega_derivative_5point"][mask, 2]
        phase_checks[str(tick)] = {"rmse_rad_s2": base.rmse(error), "mean_signed_rad_s2": float(error.mean())}
    existing_case_checks = {}
    # Existing arrays only: these checks are counterevidence, not selection.
    for case in ("angle_p0", "angle_p30", "angle_p60"):
        _, s = ds.load_case(case)
        aa = conditioned(s, EffectiveSurfaceGame(raw, config, ds.creation_input(case)))
        existing_case_checks[case] = residual_metrics(s, aa, .75, 2.5)
    checkpoints = {}
    source_gamma = np.degrees(np.arcsin(source["velocity_xyz_mps"][:, 1]/source["speed_mps"]))
    for second in (.5, .625, .75, .875, 1., 1.125, 1.5, 1.75, 2., 2.5, 3., 3.5):
        i = int(np.argmin(abs(source["t"]-second)))
        t = source["t"][i]
        row = {"source_time_s": float(t), "source_pitch_deg": float(source["body_pitch_deg"][i]),
               "source_flight_path_deg": float(source_gamma[i]), "source_bank_deg": float(source["body_bank_deg"][i]),
               "source_controls": source["controls_export_raw"][i],
               "source_pitch_rate_vertical_rotation_deg_s": float(np.degrees(a["world_pitch_rate_terms_rad_s"][i, 0])),
               "source_pitch_rate_yaw_bank_coupling_deg_s": float(np.degrees(a["world_pitch_rate_terms_rad_s"][i, 1])),
               "independent_pitch_angle_deg": float(np.degrees(a["independent_incidence_pitch_yaw_rad"][i, 0])),
               "total_angle_pitch_component_deg": float(np.degrees(a["total_incidence_rotation_pitch_yaw_rad"][i, 0])),
               "local_yaw_angle_deg": float(np.degrees(a["independent_incidence_pitch_yaw_rad"][i, 1])),
               "vertical_wdot_control_restore_damping_rad_s2": [a[k][i, 1] for k in ("control_terms_wdot_YZ", "restoring_terms_wdot_YZ", "damping_terms_wdot_YZ")],
               "source_vertical_wdot_rad_s2": float(a["omega_derivative_5point"][i, 2]),
               "frozen_vertical_wdot_rad_s2": float(a["frozen_model_wdot_YZ"][i, 1]),
               "offline_vector_vertical_wdot_rad_s2": float(a["offline_vector_wdot_YZ"][i, 1])}
        for name, p in cached.items():
            row[name] = {"pitch_deg": float(base.interpolate(p["body_pitch_deg"], p["t"], t)),
                         "flight_path_deg": float(base.interpolate(geom[name]["flight_path_elevation_deg"], p["t"], t)),
                         "controls_recorded": base.interpolate(p["controls_raw"], p["t"], t),
                         "pitch_rate_terms_deg_s": np.degrees(base.interpolate(geom[name]["world_pitch_rate_terms_rad_s"], p["t"], t))}
        checkpoints[str(second)] = row
    peak_windows = (("first", .333, .75), ("second", .875, 1.333), ("third", 1.4, 2.))
    peaks = {}
    for name, lo, hi in peak_windows:
        peaks[name] = {}
        for label, t, pitch in (("JARI", source["t"], source["body_pitch_deg"]),
                               *((name, p["t"], p["body_pitch_deg"]) for name, p in cached.items())):
            ids = np.flatnonzero((t >= lo)&(t <= hi))
            i = ids[int(np.argmax(pitch[ids]))]
            peaks[name][label] = {"time_s": float(t[i]), "pitch_deg": float(pitch[i])}
    early = source["t"] <= 3.5
    p = cached["advance_one_tick"]
    error = base.interpolate(p["body_pitch_deg"], p["t"], source["t"][early])-source["body_pitch_deg"][early]
    onset = {}
    for threshold in (.1, .25, .5, 1.):
        ids = np.flatnonzero((error[:-2] < -threshold)&(error[1:-1] < -threshold)&(error[2:] < -threshold))
        onset[str(threshold)] = float(source["t"][ids[0]]) if len(ids) else None
    qmask = (source["t"] > .3)&(source["t"] < 3.5)&source["derivative_5point_valid_mask"]
    r = a["omega_from_quaternion"][qmask, 2]
    w = a["omega_body"][qmask, 2]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"arrays.npz", **a)
    plot(source, a, cached, geom)
    assert before == protected()
    assert all(digest(ROOT/p) == h for p, h in hashes.items())
    result = {"case": CASE, "source": metadata["source_path"], "status": "offline_diagnostic_only",
              "network_requests": 0, "new_free_predictions": 0, "game_replay_observations_used": False,
              "production_changes": False, "new_coefficients_or_selection": False,
              "formulas": {"world_pitch": "asin(f.y)", "world_pitch_rate": "(omega.z*up.y - omega.y*right.y)/sqrt(1-f.y^2)",
                           "frozen_angles": "ap=atan2(-local_v.y,local_v.x); ay=atan2(-local_v.z,local_v.x)",
                           "offline_total_angle_components": "rho=hypot(local_v.y,local_v.z); alpha=atan2(rho,local_v.x); beta_p=alpha*(-local_v.y)/rho; beta_y=alpha*(-local_v.z)/rho; zero if rho=0",
                           "angular_axis_signs": "wdot.Y=-ku*S*uH + ka*S*beta_y - kw*D*omega.Y; wdot.Z=+ku*S*uV - ka*S*beta_p - kw*D*omega.Z",
                           "pure_axis_equivalence": "For positive local_v.x and exactly one nonzero transverse component, beta_p=ap and beta_y=ay; therefore the pure pitch/yaw response is identical.",
                           "small_pitch_large_yaw_ratio": "beta_p/ap approaches alpha*cot(alpha) when vertical incidence is small"},
              "windows": windows, "source_control_phase_checks": phase_checks,
              "existing_fit_case_counterevidence": existing_case_checks,
              "source_vertical_omega_quaternion_correlation": float(np.corrcoef(r, w)[0, 1]),
              "source_vertical_rate_rms_rad_s": base.rmse(w), "source_quaternion_vertical_rate_error_rms_rad_s": base.rmse(r-w),
              "forced_negative_pitch_onset_three_consecutive_source_rows_threshold_deg": onset,
              "peaks": peaks, "checkpoints": checkpoints,
              "limits": ["Source 24Hz display cadence does not reveal intermediate 48Hz controls or actuator processing.",
                         "Exported controls are not independently identified physical fin angles.",
                         "Total-angle mapping strongly reduces same-state vertical residual but has not been run freely or adopted.",
                         "Horizontal angular residual remains large; pure-pitch 0 degree residual remains unchanged.",
                         "Independent quaternion rates use display derivatives; startup discontinuities are excluded from angular-residual windows."],
              "verification": {"decomposed_current_angular_derivative_matches_production_plant": True,
                               "protected_files_unchanged": True, "protected_file_count": len(before),
                               "direct_dependencies_sha256_unchanged": hashes}}
    (OUT/"results.json").write_text(json.dumps(ds.jsonable(result), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"windows": windows, "peaks": peaks, "pitch_onset": onset,
                      "protected_files_unchanged": True, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
