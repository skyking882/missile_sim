#!/usr/bin/env python3
"""Compare free War Thunder candidates with an actual Jari trajectory export.

Jari is an interim reference, not game telemetry. Raw exports and every row
are retained. Version and scenario come from the completed export, never from
the planned web form. No reference missile state enters a candidate after its
creation state. No fitted clock, forcing, knots, or parameter optimization is
performed. The three declared candidates separate force basis from loft law.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/missle_sim_mpl")

import numpy as np

from aim120_model.aerodynamics import body_axes_from_quaternion, pitch_yaw_from_quaternion
from aim120_model.dynamics import SimState
from aim120_model.math3d import add, clamp, clamp_norm, cross, dot, norm, scale, sub
from aim120_model.single_surface_game import RateAngleLoftGame, SingleSurfaceGame, SurfaceCoefficients
from aim120_model.target import TargetState
from aim120_model.units import G0_MPS2

DEFAULT_MANIFEST = ROOT / "outputs/jari_offaxis/reference/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/jari_offaxis/comparison"
SNAPSHOT = ROOT / "outputs/unified_game_replays/corrected_input/runtime_profiles.json"
FROZEN = ROOT / "config/r771_simple_joint_frozen.json"
VARIANTS = (
    ("wind_simple", "wind_linear", SingleSurfaceGame),
    ("body_simple", "body_linear", SingleSurfaceGame),
    ("wind_rate_angle", "wind_linear", RateAngleLoftGame),
)
WINDOWS = (("activation_0_0p7", 0., .7), ("turn_0p7_3p5", .7, 3.5),
           ("burn_0_7", 0., 7.), ("full", 0., math.inf))


class HorizontalRelativePNGame(RateAngleLoftGame):
    """Game-only diagnostic: replace the horizontal PN component, no fit."""
    gate_loft = False
    relative_blend = 1.
    variant_label = "A_horizontal_PN"

    def request(self, state, target, time_s, evaluation):
        original, details = super().request(state, target, time_s, evaluation)
        horizontal_rate = abs(details["los_rate_rad_s"][1])
        # The exact zero branch keeps all planar trajectories and diagnostic
        # dictionaries bit-for-bit identical to the accepted loft baseline.
        if horizontal_rate == 0.:
            return original, details
        cfg = self.config["guidance"]
        relative_velocity = sub(target.velocity, state.velocity)
        alternative = scale(cross(details["los_rate_rad_s"], scale(relative_velocity, -1.)),
                            float(cfg["pn_gain"])*details["hit_gain"])
        basis = evaluation.wind_basis
        previous_pn = details["pn_mps2"]
        pn = add(previous_pn, scale(basis.right,
            self.relative_blend*(dot(alternative, basis.right)-dot(previous_pn, basis.right))))
        gate = 1.
        if self.gate_loft and cfg.get("lofting_enabled", False):
            steepness = float(cfg["angle_to_acceleration_multiplier"])
            threshold = math.radians(abs(float(cfg["loft_omega_max_deg_s"])))
            z = clamp(steepness*(1.-horizontal_rate/max(threshold, 1e-12)), -60., 60.)
            # Normalize the sigmoid's zero-rate value to exactly one.
            gate = (1./(1.+math.exp(-z))) / (1./(1.+math.exp(-clamp(steepness, -60., 60.))))
            gate = clamp(gate, 0., 1.)
        loft_g = details["loft_g"]*gate
        maximum = float(cfg["maximum_lateral_acceleration_g"])*self.gravity
        commanded = add(pn, (0., loft_g*self.gravity, 0.))
        commanded = clamp_norm(sub(commanded, scale(basis.forward, dot(commanded, basis.forward))), maximum)
        required = clamp_norm(add(commanded, (0., self.gravity, 0.)), maximum)
        request = (dot(required, basis.up)/G0_MPS2, dot(required, basis.right)/G0_MPS2)
        details = {**details, "pn_mps2": pn, "loft_g": loft_g,
            "outer_AB_variant": self.variant_label,
            "horizontal_los_rate_rad_s": horizontal_rate, "horizontal_loft_gate": gate,
            "original_loft_g": details["loft_g"], "original_pn_mps2": previous_pn}
        self.request_records[time_s] = {"request_g": request, **details}
        return request, details


class HorizontalRelativePNGatedLoftGame(HorizontalRelativePNGame):
    gate_loft = True
    variant_label = "B_horizontal_PN_and_loft_gate"


class HalfHorizontalRelativePNGatedLoftGame(HorizontalRelativePNGatedLoftGame):
    relative_blend = .5
    variant_label = "half_horizontal_PN_and_loft_gate"


AB_VARIANTS = (("A_horizontal_PN", HorizontalRelativePNGame),
               ("B_horizontal_PN_and_loft_gate", HorizontalRelativePNGatedLoftGame))
HALF_VARIANTS = ((HalfHorizontalRelativePNGatedLoftGame.variant_label,
                  HalfHorizontalRelativePNGatedLoftGame),)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def vector(value, size, label):
    answer = np.asarray(value, dtype=float)
    if answer.shape != (size,) or not np.isfinite(answer).all():
        raise ValueError(f"{label}: expected {size} finite components")
    return answer


def split_runs(path):
    document = json.loads(Path(path).read_text())
    runs = document.get("runs", [document])
    if not isinstance(runs, list) or not runs:
        raise ValueError("export contains no runs")
    return runs


def parse_reference(run):
    """Read the recorded public schema-3 export, with explicit unit checks."""
    if int(run.get("schema", 0)) < 3:
        raise ValueError("unsupported export schema; no guessed column mapping")
    axes = run.get("axes", {})
    if "ASL" not in str(axes.get("y", "")) or "deg" not in str(axes.get("units", "")):
        raise ValueError("world-Y absolute altitude / degree units not established in export")
    samples = run.get("samples", [])
    if len(samples) < 3:
        raise ValueError("fewer than three source samples")
    t = np.asarray([float(s["t"]) for s in samples])
    if not np.isfinite(t).all() or t[0] != 0. or np.any(np.diff(t) <= 0.):
        raise ValueError("export must include creation at t=0 and strictly increasing original time")
    position = np.stack([vector(s["position"], 3, "position") for s in samples])
    velocity = np.stack([vector(s["velocity"], 3, "velocity") for s in samples])
    quaternion_xyzw = np.stack([vector(s["quaternion"], 4, "quaternion") for s in samples])
    omega = np.stack([vector(s["omega"], 3, "omega") for s in samples])
    speed = np.linalg.norm(velocity, axis=1)
    if np.any(speed <= 1e-6):
        raise ValueError("zero reference speed")
    tangent = velocity / speed[:, None]
    axes_forward = np.stack([body_axes_from_quaternion((q[3], *q[:3])).forward for q in quaternion_xyzw])
    incidence = np.degrees(np.arctan2(np.linalg.norm(np.cross(tangent, axes_forward), axis=1),
                                      np.sum(tangent * axes_forward, axis=1)))
    aoa = np.asarray([float(s["aoa"]) for s in samples])
    if not np.isfinite(aoa).all() or np.any(aoa < -1e-10):
        raise ValueError("export AoA is not finite unsigned incidence in degree units")
    # This derived load includes thrust normal and removes gravity. It is a
    # display-cadence kinematic diagnostic, not the API's specific-force norm.
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    specific = acceleration + np.array([0., G0_MPS2, 0.])
    load = np.linalg.norm(specific - np.sum(specific * tangent, axis=1)[:, None] * tangent, axis=1) / G0_MPS2
    output = {"t": t, "speed_mps": speed, "alpha_deg": aoa,
              "geometric_incidence_deg": incidence, "normal_specific_load_g": load,
              "tangent": tangent, "position_xyz_m": position, "velocity_xyz_mps": velocity,
              "quaternion_xyzw": quaternion_xyzw, "omega_xyz_export_units": omega}
    for key in ("acc_drag", "acc_lift", "acc_thrust", "mass", "thrust", "range"):
        if all(key in s for s in samples):
            values = np.asarray([float(s[key]) for s in samples])
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite source {key}")
            output[key] = values
    diagnostics = {
        "source_rows": len(t), "source_support_s": [float(t[0]), float(t[-1])],
        "source_time_unchanged": True,
        "time_step_s_min_median_max": np.quantile(np.diff(t), [0., .5, 1.]),
        "reported_speed_minus_velocity_norm_max_mps": max(abs(float(s["speed"])-v) for s, v in zip(samples, speed)),
        "reported_aoa_minus_geometric_incidence_max_deg": float(np.max(np.abs(aoa-incidence))),
        "load_semantics": "unsigned trajectory-normal specific load derived from display velocity; includes thrust, removes gravity; endpoints less accurate",
        "acc_lift_semantics": "site display component; exact force projection is not established; never treated as total overload",
        "omega_semantics": "source omega frame/unit not independently established; creation zero only is supported; no rate-history comparison",
        "loft_semantics": "original config was selected; samples.loft=null does not establish Jari internal loft activation",
        "source_api_summary": run.get("api_summary", {}),
    }
    return output, diagnostics


def profile_for(run, entry, profiles):
    source = run.get("provenance", run.get("profile_source", {}))
    version = source.get("version") or run.get("backend", {}).get("version")
    if version is None and entry.get("raw_source", {}).get("uploaded_to_jari"):
        if not entry.get("uploaded_source_consumption_verified", False):
            raise ValueError("uploaded raw source was not verified as consumed by calculation")
        version = entry["raw_source"]["game_version"]
    local_version = {"2.57.1.107": "2.57.1.107", "2.59.0.27": "2.59.0.27",
                     "2.59.0.28": "2.59.0.27"}.get(version)
    if local_version is None:
        raise ValueError(f"unverified or unsupported actual Jari version {version!r}")
    missile = source.get("missile") or run.get("missile", "").split(".")[0]
    if missile != "su_r_77_1":
        raise ValueError(f"not original R-77-1: {missile}")
    profile = copy.deepcopy(profiles[local_version + "/su_r_77_1"])
    if version == "2.59.0.28":
        # This source equality was used by the existing native harness. Verify
        # it here before reusing the .27 runtime profile for a .28 reference.
        payload = json.loads((ROOT / "outputs/r771_replay_batch_20260928/source_payload.json").read_text())
        source_rocket = json.loads(payload["parent_parameters"]["native_text"])["rocket"]
        local_rocket = json.loads((ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx").read_text())["rocket"]
        if source_rocket != local_rocket:
            raise ValueError(".28 replay-parent/.27 entire rocket parameter equality failed")
    return profile, {"reference_version": version, "local_profile_version": local_version,
                     "reference_provenance": source, "entire_28_replay_parent_27_rocket_equal": version == "2.59.0.28",
                     "version_bridge_source": "outputs/r771_replay_batch_20260928/source_payload.json:parent_parameters.native_text; actual remote resource has its own provenance hash",
                     "old_257_reference": version == "2.57.1.107"}


def free_inputs(run, reference, config):
    scenario = run["scenario"]
    if scenario.get("guidance") != "perfect-track":
        raise ValueError("only explicitly ideal, permanently tracked external target is supported")
    target0 = vector(scenario["target_position"], 3, "scenario target_position")
    target_velocity = vector(scenario["target_velocity"], 3, "scenario target_velocity")
    loft = scenario.get("loft_state", "config")
    if loft in ("off", False):
        config["guidance"]["lofting_enabled"] = False
    elif loft in ("on", True):
        config["guidance"]["lofting_enabled"] = True
    elif loft != "config":
        raise ValueError(f"unrecognized explicit loft_state {loft!r}")
    qx, qy, qz, qw = reference["quaternion_xyzw"][0]
    quaternion = (qw, qx, qy, qz)
    pitch, yaw = pitch_yaw_from_quaternion(quaternion)
    omega = reference["omega_xyz_export_units"][0]
    if np.max(np.abs(omega)) > 1e-12:
        raise ValueError("nonzero creation omega frame not established; no guessed conversion")
    state = SimState(tuple(reference["position_xyz_m"][0]), tuple(reference["velocity_xyz_mps"][0]),
        pitch, yaw, 0., 0., config["geometry"]["initial_mass_kg"], orientation_quaternion=quaternion)
    if "mass" in reference and abs(reference["mass"][0]-state.mass) > 1e-6:
        raise ValueError("reference creation mass differs from version-pinned raw profile")
    def target_at(time_s):
        return TargetState(tuple(target0+time_s*target_velocity), tuple(target_velocity))
    return state, target_at


def prediction_arrays(result, model):
    samples = result["samples"]
    t = np.asarray([s["time_s"] for s in samples])
    velocity = np.asarray([s["velocity_mps"] for s in samples])
    speed = np.linalg.norm(velocity, axis=1)
    answer = {"t": t, "speed_mps": speed,
        "alpha_deg": np.degrees([s["angle_of_attack_rad"] for s in samples]),
        "normal_specific_load_g": np.asarray([s["trajectory_lateral_load_g"] for s in samples]),
        "position_xyz_m": np.asarray([s["position_m"] for s in samples]),
        "velocity_xyz_mps": velocity, "tangent": velocity/speed[:, None]}
    components = []
    for sample in samples:
        q = sample["orientation_quaternion"]
        pitch, yaw = pitch_yaw_from_quaternion(q)
        state = SimState(tuple(sample["position_m"]), tuple(sample["velocity_mps"]), pitch, yaw,
            sample["pitch_rate_rad_s"], sample["yaw_rate_rad_s"], sample["mass_kg"], orientation_quaternion=q)
        force = model.evaluate(state, sample["time_s"], sample["held_fin_rad"])
        components.append((force.drag_n/force.mass, np.linalg.norm(force.wind_lift_force)/force.mass,
                           force.thrust/force.mass))
    for i, key in enumerate(("acc_drag", "acc_lift", "acc_thrust")):
        answer[key] = np.asarray(components)[:, i]
    if not all(np.isfinite(v).all() for v in answer.values()):
        raise ValueError("nonfinite free prediction")
    return answer


def control_diagnostic(result, config):
    """Inspect the already-run controller without changing or fitting it."""
    limits = np.radians([config["aerodynamics"]["horizontal_fin_aoa_limit_deg"],
                         config["aerodynamics"]["vertical_fin_aoa_limit_deg"]])
    maximum = config["guidance"]["maximum_lateral_acceleration_g"]
    rows = [tick for tick in result["ticks"] if tick["time_s"] <= 3.5]
    t = np.asarray([r["time_s"] for r in rows])
    request = np.asarray([r["request_g"] for r in rows])
    held = np.asarray([r["held_fin_rad"] for r in rows])
    feedback = np.asarray([r["feedback_g"] for r in rows])
    pn = np.asarray([r["pn_mps2"] for r in rows])/G0_MPS2
    fin_ratio = np.abs(held)/limits
    saturated = np.linalg.norm(request, axis=1) >= maximum*.999
    def intervals(mask):
        indices = np.flatnonzero(mask)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(indices) else []
        return [[float(t[g[0]]), float(t[g[-1]])] for g in groups]
    return {"window_s": [0., 3.5], "first_active_s": result["summary"]["first_active_s"],
        "request_peak_total_g": float(np.max(np.linalg.norm(request, axis=1))),
        "request_peak_abs_pitch_yaw_g": np.max(np.abs(request), axis=0),
        "raw_PN_peak_g_before_final_request_projection": float(np.max(np.linalg.norm(pn, axis=1))),
        "request_at_50g_cap_intervals_s": intervals(saturated),
        "held_fin_peak_abs_pitch_yaw_rad": np.max(np.abs(held), axis=0),
        "held_fin_peak_fraction_of_limit_pitch_yaw": np.max(fin_ratio, axis=0),
        "held_fin_at_limit_intervals_s": intervals(np.any(fin_ratio >= .999, axis=1)),
        "source_control_units_comparable": False,
        "source_control_note": "Jari exports dimensionless-looking components with magnitude ceiling 1; physical/control normalization is not established and is not equated with local fin radians",
        "ticks": {"time_s": t, "request_pitch_yaw_g": request, "held_fin_pitch_yaw_rad": held,
                  "feedback_pitch_yaw_g": feedback, "raw_PN_xyz_g": pn,
                  "loft_g": [r["loft_g"] for r in rows], "flight_gain": [r["gain"] for r in rows]}}


def errors(reference, prediction):
    t = reference["t"]
    support = (t >= prediction["t"][0]) & (t <= prediction["t"][-1])
    interpolated = {}
    for key in ("speed_mps", "alpha_deg", "normal_specific_load_g", "position_xyz_m", "tangent"):
        values = prediction[key]
        interpolated[key] = (np.interp(t[support], prediction["t"], values) if values.ndim == 1
            else np.column_stack([np.interp(t[support], prediction["t"], values[:, k]) for k in range(values.shape[1])]))
    tangent = interpolated["tangent"]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(tangent, reference["tangent"][support]), axis=1),
                                np.sum(tangent*reference["tangent"][support], axis=1)))
    windows = {}
    for name, lo, hi in WINDOWS:
        selected = (t[support] >= lo) & (t[support] <= hi)
        metrics = {}
        for key in ("speed_mps", "alpha_deg", "normal_specific_load_g", "position_xyz_m"):
            delta = interpolated[key][selected]-reference[key][support][selected]
            metrics[key] = {"rmse": np.sqrt(np.mean(delta**2, axis=0)),
                            "bias": np.mean(delta, axis=0), "max_abs": np.max(np.abs(delta), axis=0)} if len(delta) else None
        metrics["tangent_error_deg"] = {"rms": np.sqrt(np.mean(angle[selected]**2)), "max": np.max(angle[selected])} if selected.any() else None
        windows[name] = {"reference_rows": int(((t >= lo)&(t <= hi)).sum()),
                         "compared_rows": int(selected.sum()), "metrics": metrics}
    events = {}
    for source, a in (("reference", reference), ("prediction", prediction)):
        early = np.flatnonzero(a["t"] <= 3.5)
        ix = early[np.argmax(a["alpha_deg"][early])]
        events[source] = {"early_peak_alpha_deg": float(a["alpha_deg"][ix]), "early_peak_time_s": float(a["t"][ix])}
    return {"windows": windows, "events": events, "compared_rows": int(support.sum()),
            "unrepresented_source_rows": np.flatnonzero(~support), "support_mask": support}


def plot_case(case_id, reference, predictions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = ("speed_mps", "alpha_deg", "normal_specific_load_g")
    labels = ("Speed (m/s)", "Unsigned AoA (deg)", "Normal specific load (g), derived Jari")
    colors = ("#686b70", "#a05080", "#be502f")
    for ax, key, label in zip(axes[0], fields, labels):
        ax.plot(reference["t"], reference[key], "o", ms=2.8, mfc="none", color="#24566a", label="Jari interim reference")
        for (name, pred), color in zip(predictions.items(), colors):
            keep = pred["t"] <= reference["t"][-1]
            ax.plot(pred["t"][keep], pred[key][keep], color=color, lw=1.2, label=name)
        ax.set(xlabel="Time (s)", ylabel=label); ax.grid(alpha=.18)
    for i, ax in enumerate(axes[1]):
        start = reference["position_xyz_m"][0, i]
        ax.plot(reference["t"], reference["position_xyz_m"][:, i]-start, "o", ms=2.8, mfc="none", color="#24566a")
        for (name, pred), color in zip(predictions.items(), colors):
            keep = pred["t"] <= reference["t"][-1]
            ax.plot(pred["t"][keep], pred["position_xyz_m"][keep, i]-start, color=color, lw=1.2)
        ax.set(xlabel="Time (s)", ylabel=f"{'XYZ'[i]} displacement (m)");ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{case_id}: matched exported creation and constant target; fixed coefficients; no clock shift", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96));fig.savefig(output / f"{case_id}_overlay.png", dpi=150);plt.close(fig)


def native_non_regression(profiles, arrays, variants=AB_VARIANTS,
                          baseline_arrays=None, baseline_records=None):
    """Use existing native fixtures unchanged; no projection or fitted gates."""
    import compare_unified_game_replays as harness
    import run_single_surface_game_candidate as candidate
    cases = {c["id"]: c for c in harness.load_cases(SNAPSHOT)}
    checks = {}
    limits = {"alpha_deg": .01, "omega_rad_s": .001, "gamma_deg": .01, "y_m": .1}
    for name in ("H110", "H27.5", "H55native"):
        case = cases[name]
        predictions, reports = {}, {}
        cached_baseline = baseline_arrays is not None and baseline_records is not None
        if cached_baseline:
            prefix = f"native_{name}__baseline__"
            predictions["baseline"] = {key[len(prefix):]: baseline_arrays[key]
                for key in baseline_arrays.files if key.startswith(prefix)}
            reports["baseline"] = {"windows": baseline_records[name]["baseline_quality"]}
            if not predictions["baseline"]:
                raise ValueError(f"Missing recorded native baseline: {name}")
            for key, value in predictions["baseline"].items():
                arrays[prefix+key] = value
        original_class = candidate.SingleSurfaceGame
        try:
            for variant, cls in (variants if cached_baseline else (("baseline", RateAngleLoftGame), *variants)):
                candidate.SingleSurfaceGame = cls
                result = candidate.runner_for("wind_linear", 4, "rad_s")(case, case["profile"])
                prediction = harness.sample_series(result["samples"], case)
                report, _ = harness.compare_series(case, prediction, result["summary"])
                predictions[variant], reports[variant] = prediction, report
                for key, value in prediction.items():
                    arrays[f"native_{name}__{variant}__{key}"] = value
        finally:
            candidate.SingleSurfaceGame = original_class
        for key, value in case["observed"].items():
            arrays[f"native_{name}__observed__{key}"] = value
        baseline = predictions["baseline"]
        t = case["observed"]["t"]
        check = {"fixture_projection_performed": False, "original_source_rows": len(t),
                 "source_has_small_nonzero_horizontal_motion": True,
                 "baseline_quality": reports["baseline"]["windows"], "variants": {}}
        for variant, _ in variants:
            pred = predictions[variant]
            support = ((t >= max(pred["t"][0], baseline["t"][0])) &
                       (t <= min(pred["t"][-1], baseline["t"][-1])))
            metric_checks = {}
            for channel in (*limits, "speed_kmh"):
                observed = case["observed"][channel][support]
                a = np.interp(t[support], baseline["t"], baseline[channel])
                b = np.interp(t[support], pred["t"], pred[channel])
                multiplier = 1./3.6 if channel == "speed_kmh" else 1.
                base_rmse = float(np.sqrt(np.mean(((a-observed)*multiplier)**2)))
                new_rmse = float(np.sqrt(np.mean(((b-observed)*multiplier)**2)))
                allowed = .1 if channel == "speed_kmh" else max(.01*base_rmse, limits[channel])
                metric_checks[channel] = {"baseline_rmse": base_rmse, "candidate_rmse": new_rmse,
                    "rmse_change": new_rmse-base_rmse, "allowed_worsening": allowed,
                    "prediction_delta_rms": float(np.sqrt(np.mean(((b-a)*multiplier)**2))),
                    "prediction_delta_max_abs": float(np.max(np.abs((b-a)*multiplier))),
                    "units": "m/s" if channel == "speed_kmh" else harness.UNITS[channel],
                    "passed": new_rmse-base_rmse <= allowed}
            # Native fixtures have no independently measured load channel.
            # Bound the predicted load change instead of inventing observed G.
            a = np.interp(t[support], baseline["t"], baseline["load_g"])
            b = np.interp(t[support], pred["t"], pred["load_g"])
            load_delta = float(np.sqrt(np.mean((b-a)**2)))
            metric_checks["load_g"] = {"independent_observed_load_available": False,
                "prediction_delta_rms_g": load_delta, "prediction_delta_max_abs_g": float(np.max(np.abs(b-a))),
                "allowed_rms_change_g": .01, "passed": load_delta <= .01}
            check["variants"][variant] = {"quality": reports[variant]["windows"],
                "common_support_rows": int(support.sum()), "checks": metric_checks,
                "passed": all(v["passed"] for v in metric_checks.values())}
        checks[name] = check
    return checks


def run_ab(manifest, output, substeps):
    """Fixed k=c=1 diagnostic, preserving the previous comparison artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    original_results_path = DEFAULT_OUTPUT / "results.json"
    old = json.loads(original_results_path.read_text())
    old_cases = {c["case_id"]: c for c in old["cases"]}
    old_arrays = np.load(DEFAULT_OUTPUT / "trajectories.npz")
    profiles = json.loads(SNAPSHOT.read_text())
    frozen_before = FROZEN.read_bytes()
    previous_results_before = original_results_path.read_bytes()
    arrays, records, invariance = {}, [], {}
    zero_reference = None
    for entry in manifest["references"]:
        case_id = entry["id"]
        run = split_runs(entry["export_path"])[0]
        reference, diagnostic = parse_reference(run)
        profile, source = profile_for(run, entry, profiles)
        config = copy.deepcopy(profile["_model_config"])
        state, target_at = free_inputs(run, reference, config)
        baseline_prediction = {key.split("__")[-1]: old_arrays[key] for key in old_arrays.files
            if key.startswith(case_id+"__wind_rate_angle__")}
        baseline_record = old_cases[case_id]["variants"]["wind_rate_angle"]
        record = {"case_id": case_id, "source": source, "actual_scenario": run["scenario"],
            "export_sha256": sha256(entry["export_path"]), "reference_diagnostics": diagnostic,
            "actual_reference_summary": run["summary"], "baseline_record": baseline_record, "variants": {}}
        predictions = {"fixed_RateAngle_baseline": baseline_prediction}
        results = {}
        for name, cls in AB_VARIANTS:
            model = cls(config, force_basis="wind_linear", plant_substeps=substeps, loft_cap_unit="rad_s")
            result = model.run(dataclasses.replace(state), target_at,
                               end_time_s=config["performance"]["lifetime_s"])
            prediction = prediction_arrays(result, model)
            record["variants"][name] = {**errors(reference, prediction), "summary": result["summary"],
                "model": result["model"], "early_control_diagnostic": control_diagnostic(result, config),
                "predicted_y_displacement_min_max_m": [float(np.min(prediction["position_xyz_m"][:, 1])-state.position[1]),
                                                        float(np.max(prediction["position_xyz_m"][:, 1])-state.position[1])],
                "end_speed_mps": float(prediction["speed_mps"][-1])}
            predictions[name], results[name] = prediction, result
        for name, prediction in predictions.items():
            for key, value in prediction.items():
                arrays[f"{case_id}__{name}__{key}"] = value
        for key, value in reference.items():
            arrays[f"{case_id}__reference__{key}"] = value
        if entry["offaxis_deg"] == 0:
            baseline_model = RateAngleLoftGame(config, force_basis="wind_linear", plant_substeps=substeps, loft_cap_unit="rad_s")
            baseline_result = baseline_model.run(dataclasses.replace(state), target_at,
                end_time_s=config["performance"]["lifetime_s"])
            for name, result in results.items():
                invariance[name] = {key: result[key] == baseline_result[key] for key in ("samples", "ticks", "summary")}
                if not all(invariance[name].values()):
                    raise AssertionError("Jari zero-off-axis pass-through is not exactly invariant")
            zero_reference = (state, target_at, config)
        plot_case(case_id, reference, predictions, output)
        records.append(record)
    if zero_reference is None:
        raise ValueError("AB requires the real planar zero-off-axis export")
    state, target_at, config = zero_reference
    no_loft_config = copy.deepcopy(config)
    no_loft_config["guidance"]["lofting_enabled"] = False
    baseline = RateAngleLoftGame(no_loft_config, force_basis="wind_linear", plant_substeps=substeps, loft_cap_unit="rad_s").run(
        dataclasses.replace(state), target_at, end_time_s=config["performance"]["lifetime_s"])
    no_loft_invariance = {}
    for name, cls in AB_VARIANTS:
        result = cls(no_loft_config, force_basis="wind_linear", plant_substeps=substeps, loft_cap_unit="rad_s").run(
            dataclasses.replace(state), target_at, end_time_s=config["performance"]["lifetime_s"])
        no_loft_invariance[name] = {key: result[key] == baseline[key] for key in ("samples", "ticks", "summary")}
        if not all(no_loft_invariance[name].values()):
            raise AssertionError("No-loft vertical-plane pass-through is not exactly invariant")
    native_checks = native_non_regression(profiles, arrays)
    if FROZEN.read_bytes() != frozen_before or original_results_path.read_bytes() != previous_results_before:
        raise AssertionError("Frozen configuration or original comparison changed")
    document = {"reference_label": "Jari .28 interim videogame reference; real game validation pending",
        "experiment": "Fixed horizontal relative-vector PN / horizontal LOS-rate loft gate",
        "parameter_fit_performed": False, "coefficients_k_c": [1., 1.],
        "common_inner_coefficients": dataclasses.asdict(SurfaceCoefficients()),
        "accepted_loft_coefficients": [.75, 1.3], "raw_PID_changed": False,
        "native_algorithm_identified": False, "evidence_grade": "D game mechanism hypothesis",
        "formula_A": "PN_new=PN_old+wind_right*(dot(N*hit_gain*cross(LOSrate,-Vrel),wind_right)-dot(PN_old,wind_right)); old up/tangent retained",
        "formula_B": "b_new=b_accepted*sigmoid(raw17.5*(1-abs(LOSrate_world_y)/radians(abs(raw0.5))))/sigmoid(raw17.5); exactly zero horizontal rate returns original request",
        "no_missile_reference_after_creation": True, "clock_shift_s": 0., "production_changed": False,
        "global_28_gameparams_verified": old.get("global_28_gameparams_verified", False),
        "global_source_limitation": old.get("global_source_limitation"),
        "control_hz": 48, "plant_substeps": substeps, "cases": records,
        "planar_zero_loft_pass_through": invariance, "planar_zero_no_loft_pass_through": no_loft_invariance,
        "native_near_axis_non_regression": native_checks,
        "native_non_regression_passed": all(v["passed"] for c in native_checks.values() for v in c["variants"].values()),
        "verification": {"frozen_bytes_unchanged": True, "original_comparison_bytes_unchanged": True,
            "all_arrays_finite": all(np.isfinite(a).all() for a in arrays.values()), "all_reference_rows_retained": True}}
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "results.json").write_text(json.dumps(jsonable(document), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "native_non_regression_passed": document["native_non_regression_passed"],
                     "planar_loft_exact": invariance, "planar_no_loft_exact": no_loft_invariance}))


def run_half(manifest, output, substeps):
    """One fixed half-blend run per case; all prior baselines are reused."""
    output.mkdir(parents=True, exist_ok=True)
    old_path, ab_path = DEFAULT_OUTPUT / "results.json", DEFAULT_OUTPUT / "ab/results.json"
    old, ab = json.loads(old_path.read_text()), json.loads(ab_path.read_text())
    if substeps != old["plant_substeps"] or substeps != ab["plant_substeps"]:
        raise ValueError("Half-blend must use the recorded baseline plant substeps")
    old_cases = {c["case_id"]: c for c in old["cases"]}
    old_arrays = np.load(DEFAULT_OUTPUT / "trajectories.npz")
    ab_arrays = np.load(DEFAULT_OUTPUT / "ab/trajectories.npz")
    prior_files = [p for folder in (DEFAULT_OUTPUT, DEFAULT_OUTPUT / "ab")
                   for p in folder.iterdir() if p.is_file()]
    prior_hashes = {p: sha256(p) for p in prior_files}
    frozen_before = FROZEN.read_bytes()
    profiles = json.loads(SNAPSHOT.read_text())
    arrays, records, invariance = {}, [], {}
    variant, cls = HALF_VARIANTS[0]
    for entry in manifest["references"]:
        case_id = entry["id"]
        run = split_runs(entry["export_path"])[0]
        reference, diagnostic = parse_reference(run)
        profile, source = profile_for(run, entry, profiles)
        config = copy.deepcopy(profile["_model_config"])
        state, target_at = free_inputs(run, reference, config)
        prefix = case_id+"__wind_rate_angle__"
        baseline_prediction = {key[len(prefix):]: old_arrays[key]
            for key in old_arrays.files if key.startswith(prefix)}
        baseline_record = old_cases[case_id]["variants"]["wind_rate_angle"]
        if config["control"]["pid"] != baseline_record["raw_PID"]:
            raise AssertionError("Raw PID differs from the recorded baseline")
        model = cls(config, force_basis="wind_linear", plant_substeps=substeps, loft_cap_unit="rad_s")
        result = model.run(dataclasses.replace(state), target_at,
                           end_time_s=config["performance"]["lifetime_s"])
        prediction = prediction_arrays(result, model)
        control = control_diagnostic(result, config)
        candidate_record = {**errors(reference, prediction), "summary": result["summary"],
            "model": result["model"], "early_control_diagnostic": control,
            "predicted_y_displacement_min_max_m": [float(np.min(prediction["position_xyz_m"][:, 1])-state.position[1]),
                                                    float(np.max(prediction["position_xyz_m"][:, 1])-state.position[1])],
            "end_speed_mps": float(prediction["speed_mps"][-1])}
        predictions = {"fixed_RateAngle_baseline": baseline_prediction, variant: prediction}
        for name, pred in predictions.items():
            for key, value in pred.items():
                arrays[f"{case_id}__{name}__{key}"] = value
        for key, value in reference.items():
            if not np.array_equal(value, old_arrays[f"{case_id}__reference__{key}"]):
                raise AssertionError("Reference arrays differ from the original export comparison")
            arrays[f"{case_id}__reference__{key}"] = value
        if entry["offaxis_deg"] == 0:
            checks = {"all_saved_prediction_arrays_exact": all(np.array_equal(value, baseline_prediction[key])
                        for key, value in prediction.items()),
                      "summary_exact": result["summary"] == baseline_record["summary"],
                      "saved_early_control_diagnostic_exact": jsonable(control) == baseline_record["early_control_diagnostic"]}
            if not all(checks.values()):
                raise AssertionError("Half-blend planar pass-through differs from the cached baseline")
            invariance = {**checks, "baseline_rerun": False,
                "full_samples_and_ticks_rechecked": False,
                "full_dictionary_limitation": "Previous baseline artifacts save numeric prediction arrays and early controller diagnostics, not full sample/tick dictionaries; no baseline rerun was authorized",
                "unchanged_exact_zero_return_branch": True,
                "previous_B_full_samples_ticks_summary_check": ab["planar_zero_loft_pass_through"]["B_horizontal_PN_and_loft_gate"]}
        plot_case(case_id, reference, predictions, output)
        records.append({"case_id": case_id, "source": source, "actual_scenario": run["scenario"],
            "export_sha256": sha256(entry["export_path"]), "reference_diagnostics": diagnostic,
            "actual_reference_summary": run["summary"], "baseline_record": baseline_record,
            "baseline_reused": True, "variants": {variant: candidate_record}})
    if not invariance:
        raise ValueError("Half-blend requires the recorded planar zero-off-axis export")
    native_checks = native_non_regression(profiles, arrays, HALF_VARIANTS, ab_arrays,
                                          ab["native_near_axis_non_regression"])
    if FROZEN.read_bytes() != frozen_before or any(sha256(p) != digest for p, digest in prior_hashes.items()):
        raise AssertionError("Frozen configuration or previous comparison artifacts changed")
    document = {"reference_label": "Jari .28 interim videogame reference; real game validation pending",
        "experiment": "Fixed half horizontal relative-vector PN with unchanged B loft gate",
        "coefficients_k_c": [.5, 1.], "parameter_fit_performed": False,
        "formula_horizontal": "PN_half=PN_old+0.5*wind_right*(dot(N*hit_gain*cross(LOSrate,-Vrel),wind_right)-dot(PN_old,wind_right)); all other B terms unchanged",
        "formula_gate": ab["formula_B"], "common_inner_coefficients": dataclasses.asdict(SurfaceCoefficients()),
        "accepted_loft_coefficients": [.75, 1.3], "raw_PID_changed": False,
        "native_algorithm_identified": False, "evidence_grade": "D game mechanism hypothesis",
        "no_missile_reference_after_creation": True, "clock_shift_s": 0., "production_changed": False,
        "global_28_gameparams_verified": old.get("global_28_gameparams_verified", False),
        "global_source_limitation": old.get("global_source_limitation"),
        "baseline_sources": [str(old_path), str(ab_path)], "baseline_simulations_rerun": 0,
        "new_Jari_case_simulations": 3, "new_native_fixture_simulations": 3,
        "control_hz": 48, "plant_substeps": substeps, "cases": records,
        "planar_zero_loft_pass_through": invariance,
        "planar_zero_no_loft_pass_through": {"rerun": False, "unchanged_exact_zero_return_branch": True,
            "previous_B_full_samples_ticks_summary_check": ab["planar_zero_no_loft_pass_through"]["B_horizontal_PN_and_loft_gate"]},
        "native_near_axis_non_regression": native_checks,
        "native_non_regression_passed": all(v["passed"] for c in native_checks.values() for v in c["variants"].values()),
        "minimum_range_semantics": old["minimum_range_semantics"],
        "verification": {"frozen_bytes_unchanged": True, "previous_comparison_and_AB_files_unchanged": True,
            "all_arrays_finite": all(np.isfinite(a).all() for a in arrays.values()),
            "all_reference_rows_retained": True, "no_extrapolation_or_clock_fit": True}}
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "results.json").write_text(json.dumps(jsonable(document), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "native_non_regression_passed": document["native_non_regression_passed"],
                     "planar_saved_arrays_and_summary_exact": invariance}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--ab", action="store_true", help="Run only the authorized fixed k=c=1 A/B plus preservation checks")
    parser.add_argument("--half", action="store_true", help="Run fixed k=.5,c=1 only, reusing recorded Jari/native baselines")
    parser.add_argument("--validate-export", type=Path, help="Only check a recorded export; do not run/fake a new reference")
    args = parser.parse_args()
    if args.ab and args.half:
        parser.error("Choose one bounded experiment: --ab or --half")
    if args.validate_export:
        for run in split_runs(args.validate_export):
            reference, diagnostic = parse_reference(run)
            print(json.dumps(jsonable({"file": str(args.validate_export), "schema": run["schema"],
                "rows": len(reference["t"]), "diagnostic": diagnostic}), indent=2, allow_nan=False))
        return
    manifest = json.loads(args.manifest.read_text())
    if args.half:
        run_half(manifest, args.output / "half" if args.output == DEFAULT_OUTPUT else args.output, args.substeps)
        return
    if args.ab:
        run_ab(manifest, args.output / "ab" if args.output == DEFAULT_OUTPUT else args.output, args.substeps)
        return
    profiles = json.loads(SNAPSHOT.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    before = FROZEN.read_bytes()
    document = {"reference_label": "Jari interim War Thunder test; true game revalidation pending",
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "predeclared_candidates": [{"name": n, "force_basis": f, "outer_class": c.__name__,
            "role": "current_primary_baseline" if c is RateAngleLoftGame else "force_or_loft_diagnostic"} for n, f, c in VARIANTS],
        "common_coefficients": dataclasses.asdict(SurfaceCoefficients()), "loft_coefficients_fixed": [.75, 1.3],
        "parameter_fit_performed": False, "control_hz": 48, "plant_substeps": args.substeps,
        "missile_reference_forcing_after_creation": False, "clock_shift_s": 0.,
        "production_changed": False, "true_game_validation": False,
        "global_28_gameparams_verified": False,
        "global_source_limitation": "Recorded .28 parent contains missile rocket parameters, manifest and runbook only; no .28 shared gameparams to verify CxAoA/applyWingAreaMultToCxAoA/useCxiMach. Current local drag uses caliber area times CxAoA=9 times capped angle squared, no Mach multiplier; Cx(M) table times CxK for zero-angle drag.",
        "minimum_range_semantics": "Candidate reports separation until first fuse boundary; no post-fuse closest point is flown or scored.",
        "cases": []}
    arrays = {}
    for entry in manifest.get("references", manifest.get("cases", [])):
        case_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", entry.get("case_id", entry.get("id", "reference")))
        path = entry.get("export_file", entry.get("export_path"))
        record = {"case_id": case_id, "planned_settings": entry.get("predeclared_settings", entry.get("settings")),
                  "actual_capture_status": entry.get("status"), "source_rows_deleted": 0}
        if not path:
            record.update(status="awaiting_actual_reference", reason="No completed Jari export; planned settings never become truth")
            document["cases"].append(record); continue
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        try:
            runs = split_runs(path)
            if len(runs) != 1:
                raise ValueError("manifest case must identify one actual completed run")
            run = runs[0]
            reference, diagnostic = parse_reference(run)
            profile, source = profile_for(run, entry, profiles)
            config = copy.deepcopy(profile["_model_config"])
            state, target_at = free_inputs(run, reference, config)
            record.update(status="compared", export_file=str(path), export_sha256=sha256(path),
                actual_scenario=run["scenario"], source=source, reference_diagnostics=diagnostic,
                actual_reference_summary=run.get("summary", {}), variants={})
            for key, values in reference.items():
                arrays[f"{case_id}__reference__{key}"] = values
            predictions = {}
            for name, basis, cls in VARIANTS:
                model = cls(config, force_basis=basis, plant_substeps=args.substeps, loft_cap_unit="rad_s")
                result = model.run(dataclasses.replace(state), target_at, end_time_s=config["performance"]["lifetime_s"])
                prediction = prediction_arrays(result, model)
                record["variants"][name] = {**errors(reference, prediction), "summary": result["summary"],
                    "model": result["model"], "raw_PID": config["control"]["pid"],
                    "early_control_diagnostic": control_diagnostic(result, config)}
                for key, values in prediction.items():
                    arrays[f"{case_id}__{name}__{key}"] = values
                predictions[name] = prediction
            plot_case(case_id, reference, predictions, args.output)
        except (ValueError, KeyError, FileNotFoundError) as error:
            record.update(status="not_compared", reason=str(error))
        document["cases"].append(record)
    if FROZEN.read_bytes() != before:
        raise AssertionError("frozen configuration modified")
    document["verification"] = {"frozen_bytes_unchanged": True,
        "all_saved_arrays_finite": all(np.isfinite(a).all() for a in arrays.values()),
        "all_source_rows_retained": True, "no_extrapolation_or_clock_fit": True}
    document["status"] = ("comparison_complete" if document["cases"] and all(c["status"] == "compared" for c in document["cases"])
                          else "reference_or_mapping_incomplete")
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    (args.output / "results.json").write_text(json.dumps(jsonable(document), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": document["status"], "cases": [(c["case_id"], c["status"]) for c in document["cases"]]}))


if __name__ == "__main__":
    main()
