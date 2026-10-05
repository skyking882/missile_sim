#!/usr/bin/env python3
"""Game-only, script-local four-coefficient drag calibration and free A/B.

Only the existing zero-degree Jari export supplies the fit. The other angles
and original near-axis fixtures are checks, not fitting inputs. Core, GUI,
frozen attitude coefficients, PID, loft fields and previous artifacts are
untouched. This is a D-grade Jari hypothesis, not an identified game formula.
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

import compare_r771_jari_offaxis as comparison
import compare_unified_game_replays as harness
from aim120_model.aerodynamics import body_axes_for_state, pitch_yaw_from_quaternion
from aim120_model.drag_models import area_basis, effective_cda0
from aim120_model.dynamics import SimState
from aim120_model.math3d import add, dot, scale
from aim120_model.single_surface_game import RateAngleLoftGame, SingleSurfaceGame, SurfaceCoefficients
from aim120_model.units import G0_MPS2

OUTPUT = ROOT / "outputs/jari_offaxis/drag_fit"
MANIFEST = comparison.DEFAULT_MANIFEST
HALF_OUTPUT = comparison.DEFAULT_OUTPUT / "half"
HALF_ARRAYS = HALF_OUTPUT / "trajectories.npz"
STRATEGIES = (("rate_angle", "fixed_RateAngle_baseline"),
              ("half_PN_gate", "half_horizontal_PN_and_loft_gate"))


@dataclasses.dataclass(frozen=True)
class DragCoefficients:
    base_a: float
    base_b: float
    aoa_a: float
    aoa_b: float

    def __post_init__(self):
        if not all(math.isfinite(x) and .1 <= x <= 5. for x in dataclasses.astuple(self)):
            raise ValueError("first-family coefficients must be positive and within [.1,5]")

    def factors(self, mach):
        m = max(float(mach), 1.)
        return self.base_a+self.base_b/m**4, self.aoa_a+self.aoa_b/m


class FittedDragMixin:
    """Replace all force bookkeeping, preserving the frozen angular law."""
    def __init__(self, *args, drag_coefficients, **kwargs):
        self.drag_coefficients = drag_coefficients
        self.drag_mach_min = math.inf
        self.drag_mach_max = -math.inf
        super().__init__(*args, **kwargs)

    def evaluate(self, state, time_s, held):
        original = super().evaluate(state, time_s, held)
        axes = body_axes_for_state(state)
        speed = math.sqrt(sum((v-w)**2 for v, w in zip(state.velocity, self.wind)))
        atmosphere = self.atmosphere.sample(state.position[1])
        mach = speed/atmosphere.speed_of_sound_mps
        self.drag_mach_min = min(self.drag_mach_min, mach)
        self.drag_mach_max = max(self.drag_mach_max, mach)
        r0, k = self.drag_coefficients.factors(mach)
        base = original.qbar*effective_cda0(mach, self.config)*r0
        aoa = original.qbar*area_basis(self.config, "caliber_area")*float(
            self.config["aerodynamics"]["cx_vs_aoa"])*k*math.sin(original.alpha_total)**2
        drag_n = base+aoa
        drag = scale(original.wind_basis.forward, -drag_n)
        aerodynamic = add(original.wind_lift_force, drag)
        specific = scale(add(aerodynamic, scale(axes.forward, original.thrust)), 1./original.mass)
        acceleration = add(specific, (0., -self.gravity, 0.))
        feedback = (dot(specific, original.wind_basis.up)/G0_MPS2,
                    dot(specific, original.wind_basis.right)/G0_MPS2)
        lever = (-self.arm*dot(aerodynamic, axes.up), -self.arm*dot(aerodynamic, axes.right))
        couple = tuple(original.inertia*q-l for q, l in zip(original.angular_acceleration, lever))
        return dataclasses.replace(original, acceleration=acceleration,
            specific_acceleration=specific, aerodynamic_force=aerodynamic,
            wind_drag_force=drag, feedback_g=feedback, lever_moment=lever,
            reference_couple=couple, drag_n=drag_n)


class FittedRateAngleGame(FittedDragMixin, RateAngleLoftGame):
    pass


class FittedHalfPNGame(FittedDragMixin, comparison.HalfHorizontalRelativePNGatedLoftGame):
    pass


CLASSES = {"rate_angle": FittedRateAngleGame, "half_PN_gate": FittedHalfPNGame}


def stats(x):
    return {"n": len(x), "rmse": float(np.sqrt(np.mean(x*x))), "bias": float(np.mean(x)),
            "max_abs": float(np.max(np.abs(x)))} if len(x) else {"n": 0}


def calibration_inputs(entry, profiles):
    run = comparison.split_runs(entry["export_file"])[0]
    ref, diagnostic = comparison.parse_reference(run)
    profile, source = comparison.profile_for(run, entry, profiles)
    config = copy.deepcopy(profile["_model_config"])
    model = SingleSurfaceGame(config)
    atmosphere = [model.atmosphere.sample(y) for y in ref["position_xyz_m"][:, 1]]
    qbar = .5*np.asarray([a.density_kg_m3 for a in atmosphere])*ref["speed_mps"]**2
    mach = ref["speed_mps"]/np.asarray([a.speed_of_sound_mps for a in atmosphere])
    base_old = qbar*np.asarray([effective_cda0(float(m), config) for m in mach])
    aoa_unit = qbar*area_basis(config, "caliber_area")*config["aerodynamics"]["cx_vs_aoa"]*np.sin(
        np.radians(ref["geometric_incidence_deg"]))**2
    site_base = np.asarray([s["drag_base_n"] for s in run["samples"]])
    site_aoa = np.asarray([s["drag_aoa_n"] for s in run["samples"]])
    base_ratio = site_base/base_old
    aoa_factor = np.divide(site_aoa, aoa_unit, out=np.zeros(len(mach)), where=aoa_unit > 1e-10)
    valid = np.ones(len(mach), dtype=bool)
    valid[:2] = False
    valid[-2:] = False
    valid &= mach >= 1.
    aoa_valid = valid & (ref["geometric_incidence_deg"] >= .5)
    return {"run": run, "reference": ref, "diagnostic": diagnostic, "source": source,
            "config": config, "mach": mach, "qbar": qbar, "base_ratio": base_ratio,
            "aoa_factor": aoa_factor, "base_valid": valid, "aoa_valid": aoa_valid,
            "base_old_n": base_old, "aoa_unit_n": aoa_unit,
            "site_base_n": site_base, "site_aoa_n": site_aoa}


def fit_and_check(inputs):
    train = inputs["jari_259_0"]
    m = train["mach"]
    bmask, kmask = train["base_valid"], train["aoa_valid"]
    base = np.linalg.lstsq(np.column_stack((np.ones(bmask.sum()), m[bmask]**-4)),
                           train["base_ratio"][bmask], rcond=None)[0]
    aoa = np.linalg.lstsq(np.column_stack((np.ones(kmask.sum()), m[kmask]**-1)),
                          train["aoa_factor"][kmask], rcond=None)[0]
    coefficients = DragCoefficients(*base, *aoa)
    record = {"coefficients": dataclasses.asdict(coefficients),
        "base_formula": "base_current(M)*(base_a+base_b/max(M,1)^4)",
        "aoa_formula": "q*S_caliber*raw_CxAoA*(aoa_a+aoa_b/max(M,1))*sin(total_geometric_incidence)^2",
        "fit_method": "separate unweighted two-column linear least squares, no trajectory fit",
        "fit_source_case": "jari_259_0", "fit_source_sha256": comparison.sha256(train["source_path"]),
        "base_fit_rows": np.flatnonzero(bmask).tolist(), "aoa_fit_rows": np.flatnonzero(kmask).tolist(),
        "base_fit_mach_support": [float(m[bmask].min()), float(m[bmask].max())],
        "aoa_fit_mach_support": [float(m[kmask].min()), float(m[kmask].max())],
        "filter": "exclude first/last two displayed samples; M>=1; AoA geometric incidence>=0.5degree for multiplier",
        "extension": "max(M,1) only; below training Mach domain unverified",
        "other_angles_used_in_fit": False, "cross_angle_checks_are_blind": False,
        "evidence_grade": "D Jari game-model empirical hypothesis; .28 global not verified",
        "same_state_closure": {}}
    for name, x in inputs.items():
        r0 = coefficients.base_a+coefficients.base_b/np.maximum(x["mach"], 1.)**4
        k = coefficients.aoa_a+coefficients.aoa_b/np.maximum(x["mach"], 1.)
        bm, am = x["base_valid"], x["aoa_valid"]
        new_base, new_aoa = x["base_old_n"]*r0, x["aoa_unit_n"]*k
        record["same_state_closure"][name] = {
            "base_ratio_error": stats((r0-x["base_ratio"])[bm]),
            "aoa_multiplier_error": stats((k-x["aoa_factor"])[am]),
            "base_force_error_n": stats((new_base-x["site_base_n"])[bm]),
            "aoa_force_error_n": stats((new_aoa-x["site_aoa_n"])[am]),
            "total_force_error_n": stats((new_base+new_aoa-x["site_base_n"]-x["site_aoa_n"])[bm]),
            "conditional_not_free_prediction": True}
    record["first_family_closure_gate_passed"] = all(
        v["base_ratio_error"]["rmse"] < .02 and v["aoa_multiplier_error"]["rmse"] < .05
        for v in record["same_state_closure"].values())
    return coefficients, record


def selected_arrays(container, prefix):
    out = {key[len(prefix):]: container[key] for key in container.files if key.startswith(prefix)}
    if not out:
        raise ValueError("missing cached baseline: "+prefix)
    return out


def pair_metrics(observed, old, new, fields, windows, tolerances=None):
    t = observed["t"]
    support = (t >= max(old["t"][0], new["t"][0])) & (t <= min(old["t"][-1], new["t"][-1]))
    report = {}
    for name, lo, hi in windows:
        window = support & (t >= lo-1e-9) & (t <= hi+1e-9)
        channels = {}
        for field in fields:
            valid = window & np.isfinite(observed[field])
            if not np.any(valid):
                continue
            a, b = np.interp(t[valid], old["t"], old[field]), np.interp(t[valid], new["t"], new[field])
            base, candidate = stats(a-observed[field][valid]), stats(b-observed[field][valid])
            channel = {"old": base, "new": candidate, "rmse_change": candidate["rmse"]-base["rmse"],
                       "prediction_delta": stats(b-a)}
            if tolerances and field in tolerances:
                allowed = max(.01*base["rmse"], tolerances[field])
                channel.update(allowed_worsening=allowed, passed=candidate["rmse"]-base["rmse"] <= allowed)
            channels[field] = channel
        report[name] = {"original_reference_rows": int(((t >= lo-1e-9)&(t <= hi+1e-9)).sum()),
                        "common_compared_rows": int(window.sum()), "metrics": channels}
    return report


def force_bookkeeping(model, result):
    maximum = {"aero_sum_n": 0., "drag_tangent_n": 0., "specific_sum_n": 0.,
               "angular_couple_closure_nm": 0., "lever_nm": 0., "unchanged_angular_law_rad_s2": 0.}
    ids = sorted(set((0, len(result["samples"])-1, *[int(np.argmin(abs(
        np.asarray([s["time_s"] for s in result["samples"]])-t))) for t in (.5, 1., 2., 3.5, 7.)])))
    for i in ids:
        s = result["samples"][i]
        q = s["orientation_quaternion"]
        pitch, yaw = pitch_yaw_from_quaternion(q)
        state = SimState(tuple(s["position_m"]), tuple(s["velocity_mps"]), pitch, yaw,
            s["pitch_rate_rad_s"], s["yaw_rate_rad_s"], s["mass_kg"], orientation_quaternion=tuple(q))
        f = model.evaluate(state, s["time_s"], tuple(s["held_fin_rad"]))
        original = SingleSurfaceGame.evaluate(model, state, s["time_s"], tuple(s["held_fin_rad"]))
        axes = body_axes_for_state(state)
        errors = {"aero_sum_n": np.max(np.abs(np.asarray(f.aerodynamic_force)-np.asarray(f.wind_lift_force)-np.asarray(f.wind_drag_force))),
            "drag_tangent_n": abs(dot(f.wind_drag_force, f.wind_basis.forward)+f.drag_n),
            "specific_sum_n": np.max(np.abs(f.mass*np.asarray(f.specific_acceleration)-np.asarray(f.aerodynamic_force)-f.thrust*np.asarray(axes.forward))),
            "angular_couple_closure_nm": np.max(np.abs(f.inertia*np.asarray(f.angular_acceleration)-np.asarray(f.lever_moment)-np.asarray(f.reference_couple))),
            "lever_nm": np.max(np.abs(np.asarray(f.lever_moment)+model.arm*np.asarray([dot(f.aerodynamic_force, axes.up), dot(f.aerodynamic_force, axes.right)]))),
            "unchanged_angular_law_rad_s2": np.max(np.abs(np.asarray(f.angular_acceleration)-np.asarray(original.angular_acceleration)))}
        for key, value in errors.items():
            maximum[key] = max(maximum[key], float(value))
    if max(maximum.values()) > 1e-8:
        raise AssertionError(maximum)
    return {"checked_saved_sample_indices": ids, "maximum_absolute_errors": maximum,
            "one_surface_arm_m": model.arm, "free_couple_at_same_station": True}


def event_summary(reference, prediction, result, reference_summary):
    early = prediction["t"] <= 3.5
    i = np.flatnonzero(early)[np.argmax(prediction["alpha_deg"][early])]
    ri = np.flatnonzero(reference["t"] <= 3.5)[np.argmax(reference["alpha_deg"][reference["t"] <= 3.5])]
    return {"alpha_peak_deg": float(prediction["alpha_deg"][i]), "alpha_peak_time_s": float(prediction["t"][i]),
        "reference_alpha_peak_deg": float(reference["alpha_deg"][ri]), "reference_alpha_peak_time_s": float(reference["t"][ri]),
        "alpha_peak_time_error_s": float(prediction["t"][i]-reference["t"][ri]),
        "end_speed_mps": float(prediction["speed_mps"][-1]), "reference_end_speed_mps": reference_summary["end_v"],
        "end_speed_error_mps": float(prediction["speed_mps"][-1]-reference_summary["end_v"]),
        "flight_time_error_s": result["summary"]["flight_time_s"]-reference_summary["final_time"],
        "vertical_displacement_min_max_m": [float(np.min(prediction["position_xyz_m"][:, 1])-reference["position_xyz_m"][0, 1]),
            float(np.max(prediction["position_xyz_m"][:, 1])-reference["position_xyz_m"][0, 1])],
        "summary": result["summary"]}


def near_events(observed, previous, predicted):
    events = {}
    for field in ("alpha_deg", "omega_rad_s"):
        events[field+"_peak_before_3p5"] = {name: harness.extremum(a["t"], a[field], "max", 0., 3.5)
            for name, a in (("observed", observed), ("old", previous), ("new", predicted))}
    events["end_values_at_own_endpoints"] = {name: {"time_s": float(a["t"][-1]),
        "speed_kmh": float(a["speed_kmh"][-1]), "y_m": float(a["y_m"][-1])}
        for name, a in (("observed", observed), ("old", previous), ("new", predicted))}
    valid = (observed["t"] >= max(previous["t"][0], predicted["t"][0])) & (observed["t"] <= min(previous["t"][-1], predicted["t"][-1]))
    last = int(np.flatnonzero(valid)[-1])
    t = float(observed["t"][last])
    events["last_common_observed_time_s"] = t
    events["last_common_residuals"] = {name: {field: float(np.interp(t, a["t"], a[field])-observed[field][last])
        for field in ("speed_kmh", "y_m")}
        for name, a in (("old", previous), ("new", predicted))}
    events["original_rows_outside_common_support"] = np.flatnonzero(~valid).tolist()
    return events


def plot_shapes(inputs, coefficients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for name, x in inputs.items():
        for ax, value, mask in zip(axes, (x["base_ratio"], x["aoa_factor"]), (x["base_valid"], x["aoa_valid"])):
            ax.plot(x["mach"][mask], value[mask], "o", ms=2, alpha=.6, label=name+" source")
    m = np.linspace(1., 4., 160)
    axes[0].plot(m, coefficients.base_a+coefficients.base_b/m**4, color="black", label="Fit from zero-degree only")
    axes[1].plot(m, coefficients.aoa_a+coefficients.aoa_b/m, color="black", label="Fit from zero-degree only")
    for ax, label in zip(axes, ("Base drag ratio r0", "AoA sin² multiplier K")):
        ax.set(xlabel="Mach from local ISA", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Four positive coefficients fitted to existing 0-degree Jari export; other angles check transfer")
    fig.savefig(OUTPUT / "coefficient_shapes.png", dpi=150)
    plt.close(fig)


def plot_case(name, reference, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    colors = ("#8c8c8c", "#9564ac", "#cd7820", "#15965a")
    for ax, key, label in zip(axes[0], ("speed_mps", "alpha_deg", "normal_specific_load_g"),
                             ("Speed (m/s)", "Unsigned incidence (degree)", "Normal specific load (g)")):
        ax.plot(reference["t"], reference[key], "o", ms=2.4, mfc="none", color="#2176bd", label="Jari reference")
        for (variant, p), color in zip(predictions.items(), colors):
            ax.plot(p["t"], p[key], color=color, lw=1.2, label=variant)
        ax.set(xlabel="Original time (s)", ylabel=label, xlim=(0, reference["t"][-1]))
        ax.grid(alpha=.2)
    for coordinate, ax in enumerate(axes[1]):
        offset = reference["position_xyz_m"][0, coordinate]
        ax.plot(reference["t"], reference["position_xyz_m"][:, coordinate]-offset, "o", ms=2.4, mfc="none", color="#2176bd")
        for (variant, p), color in zip(predictions.items(), colors):
            ax.plot(p["t"], p["position_xyz_m"][:, coordinate]-offset, color=color, lw=1.2)
        ax.set(xlabel="Original time (s)", ylabel="XYZ"[coordinate]+" displacement (m)", xlim=(0, reference["t"][-1]))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(name+": freely integrated creation only, frozen inner/PID/loft; drag fit used no off-axis data")
    fig.savefig(OUTPUT / (name+"_overlay.png"), dpi=150)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior_paths = [p for d in (comparison.DEFAULT_OUTPUT, ROOT / "outputs/jari_offaxis/reference",
                               ROOT / "outputs/jari_offaxis/speed_gap") for p in d.rglob("*") if p.is_file()]
    protected = set(prior_paths+[comparison.FROZEN, comparison.SNAPSHOT,
        ROOT / "src/aim120_model/single_surface_game.py", ROOT / "src/aim120_model/drag_models.py",
        ROOT / "scripts/compare_r771_jari_offaxis.py", ROOT / "scripts/compare_unified_game_replays.py",
        ROOT / "scripts/run_single_surface_game_candidate.py", harness.NATIVE_BASE / "source_payload.json"])
    before = {str(p): comparison.sha256(p) for p in protected}
    frozen = json.loads(comparison.FROZEN.read_text())
    surface = SurfaceCoefficients()
    if frozen["coefficients"] != [surface.ca, surface.speed_slope, surface.control_gain, surface.restoring_gain] or frozen["fixed_reference_damping_s_minus1"] != surface.rate_damping:
        raise AssertionError("frozen inner law differs")
    manifest = json.loads(MANIFEST.read_text())
    profiles = json.loads(comparison.SNAPSHOT.read_text())
    inputs = {}
    for entry in manifest["cases"]:
        x = calibration_inputs(entry, profiles)
        x["source_path"] = Path(entry["export_file"])
        inputs[entry["actual_reference_id"]] = x
    coefficients, fit = fit_and_check(inputs)
    fit_cached_path = OUTPUT / "fit.json"
    same_cached_fit = fit_cached_path.exists() and json.loads(fit_cached_path.read_text()) == comparison.jsonable(fit)
    (OUTPUT / "fit.json").write_text(json.dumps(comparison.jsonable(fit), indent=2, allow_nan=False)+"\n")
    plot_shapes(inputs, coefficients)
    if not fit["first_family_closure_gate_passed"]:
        print(json.dumps({"status": "first_family_conditionally_failed_no_free_runs", "fit": fit}, allow_nan=False))
        return
    old_arrays = np.load(HALF_ARRAYS)
    old_records = json.loads((HALF_OUTPUT / "results.json").read_text())
    arrays, records, force_checks = {}, {}, {}
    computed_this_invocation = {"Jari": 0, "near": 0}
    recovered_saved_new_runs = {"Jari": 0, "near": 0}
    full_folder = OUTPUT / "full_runs"
    full_folder.mkdir(exist_ok=True)
    for name, x in inputs.items():
        reference, run = x["reference"], x["run"]
        predictions, variants = {}, {}
        for strategy, old_variant in STRATEGIES:
            previous = selected_arrays(old_arrays, name+"__"+old_variant+"__")
            predictions["old_"+strategy] = previous
            config = copy.deepcopy(x["config"])
            state, target_at = comparison.free_inputs(run, reference, config)
            model = CLASSES[strategy](config, drag_coefficients=coefficients, coefficients=surface,
                force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
            full_path = full_folder / (name+"__"+strategy+".json")
            if same_cached_fit and full_path.exists():
                result = json.loads(full_path.read_text())
                model.initialize(dataclasses.replace(state))
                recovered_saved_new_runs["Jari"] += 1
            else:
                result = model.run(dataclasses.replace(state), target_at, end_time_s=config["performance"]["lifetime_s"])
                computed_this_invocation["Jari"] += 1
            result["model"]["drag_coefficients"] = dataclasses.asdict(coefficients)
            result["model"]["drag_fit_source"] = "jari_259_0 diagnostic only"
            predicted = comparison.prediction_arrays(result, model)
            predictions["new_drag_"+strategy] = predicted
            force_checks[name+"__"+strategy] = force_bookkeeping(model, result)
            variants[strategy] = {"matched_old_vs_new": pair_metrics(reference, previous, predicted,
                ("speed_mps", "alpha_deg", "normal_specific_load_g"), comparison.WINDOWS),
                "new_vs_reference": comparison.errors(reference, predicted),
                "events": event_summary(reference, predicted, result, run["summary"]),
                "raw_PID": config["control"]["pid"], "frozen_surface": dataclasses.asdict(model.coefficients),
                "original_loft_fields": {key: config["guidance"][key] for key in ("lofting_elevation_deg", "target_elevation_deg", "angle_to_acceleration_multiplier", "loft_omega_max_deg_s")},
                "accepted_loft_coefficients": [model.amplitude_coefficient, model.transition_coefficient],
                "mach_visited": [model.drag_mach_min, model.drag_mach_max],
                "below_M1_extension_visited": model.drag_mach_min < 1.,
                "full_result_path": str(full_folder / (name+"__"+strategy+".json"))}
            full_path.write_text(json.dumps(comparison.jsonable(result), separators=(",", ":"), allow_nan=False)+"\n")
            for key, value in predicted.items():
                arrays[name+"__new_drag_"+strategy+"__"+key] = value
            for key, value in previous.items():
                arrays[name+"__old_"+strategy+"__"+key] = value
        for key, value in reference.items():
            arrays[name+"__reference__"+key] = value
        # Four consistent old/new colors and curve labels.
        predictions = {key: predictions[key] for key in ("old_rate_angle", "old_half_PN_gate", "new_drag_rate_angle", "new_drag_half_PN_gate")}
        plot_case(name, reference, predictions)
        records[name] = {"source": x["source"], "original_scenario": run["scenario"],
                         "source_rows": len(reference["t"]), "variants": variants}
        print(json.dumps({"finished_case": name, "new_flight_times": {k: v["events"]["summary"]["flight_time_s"] for k, v in variants.items()}}), flush=True)
    near_cases = {c["id"]: c for c in harness.load_cases(comparison.SNAPSHOT)}
    near = {}
    near_limits = {"alpha_deg": .01, "omega_rad_s": .001, "gamma_deg": .01, "y_m": .1}
    for name in ("H110", "H27.5", "H55native"):
        case = near_cases[name]
        variants = {}
        for strategy, old_variant in STRATEGIES:
            old_prefix = f"native_{name}__"+("baseline" if strategy == "rate_angle" else old_variant)+"__"
            previous = selected_arrays(old_arrays, old_prefix)
            config = copy.deepcopy(case["profile"]["_model_config"])
            state = dataclasses.replace(case["_creation_state"])
            qx, qy, qz, qw = case["creation"]["u16_1"]
            state.orientation_quaternion = (qw, qx, qy, qz)
            target_at = case["_target_model"].state_at
            model = CLASSES[strategy](config, drag_coefficients=coefficients, coefficients=surface,
                force_basis="wind_linear", plant_substeps=4, loft_cap_unit="rad_s")
            full_path = full_folder / ("native_"+name+"__"+strategy+".json")
            if same_cached_fit and full_path.exists():
                result = json.loads(full_path.read_text())
                model.initialize(dataclasses.replace(state))
                recovered_saved_new_runs["near"] += 1
            else:
                result = model.run(state, target_at, end_time_s=case["scenario"]["max_simulation_time_s"])
                computed_this_invocation["near"] += 1
            result["model"]["drag_coefficients"] = dataclasses.asdict(coefficients)
            result["model"]["drag_fit_source"] = "jari_259_0 diagnostic only"
            full_path.write_text(json.dumps(comparison.jsonable(result), separators=(",", ":"), allow_nan=False)+"\n")
            predicted = harness.sample_series(result["samples"], case)
            force_checks["native_"+name+"__"+strategy] = force_bookkeeping(model, result)
            metrics = pair_metrics(case["observed"], previous, predicted,
                ("speed_kmh", "alpha_deg", "omega_rad_s", "gamma_deg", "y_m"), harness.WINDOWS, near_limits)
            variants[strategy] = {"same_original_creation_and_target": True, "fixture_projection": False,
                "matched_old_vs_new": metrics, "summary": result["summary"],
                "events": near_events(case["observed"], previous, predicted),
                "full_angular_height_gate_passed": all(v["passed"] for k, v in metrics["full"]["metrics"].items() if k in near_limits),
                "raw_PID": config["control"]["pid"], "frozen_surface": dataclasses.asdict(model.coefficients),
                "original_loft_fields": {key: config["guidance"][key] for key in ("lofting_elevation_deg", "target_elevation_deg", "angle_to_acceleration_multiplier", "loft_omega_max_deg_s")},
                "accepted_loft_coefficients": [model.amplitude_coefficient, model.transition_coefficient],
                "mach_visited": [model.drag_mach_min, model.drag_mach_max]}
            for key, value in predicted.items():
                arrays[f"native_{name}__new_drag_{strategy}__{key}"] = value
            for key, value in previous.items():
                arrays[f"native_{name}__old_{strategy}__{key}"] = value
        near[name] = {"source_rows": len(case["observed"]["t"]), "variants": variants}
        for key, value in case["observed"].items():
            arrays[f"native_{name}__observed__{key}"] = value
        print(json.dumps({"finished_native_case": name, "full_angular_height_gate": {k: v["full_angular_height_gate_passed"] for k, v in variants.items()}}), flush=True)
    # One focused final verification of delivered new simulations and preserved
    # inputs; old baselines are never rerun for reassurance.
    preservation = before == {str(p): comparison.sha256(p) for p in protected}
    summaries = [variant["events"]["summary"] for row in records.values() for variant in row["variants"].values()]
    # First contact is found on the linear integration segment, then the RK
    # state is integrated to that first root. Their difference is below1mm;
    # the already-accepted event tolerance must not demand exact RK equality.
    all_fuse = all(s["termination_event"] == "proximity_fuse" and abs(s["minimum_range_m"]-10.) < .001 for s in summaries)
    raw_pid = profiles["2.59.0.27/su_r_77_1"]["_model_config"]["control"]["pid"]
    rows = [variant for data in (*records.values(), *near.values()) for variant in data["variants"].values()]
    verification = {"all_existing_inputs_and_artifacts_unchanged": preservation,
        "source_Jari_rows_retained": sum(v["source_rows"] for v in records.values()),
        "all_arrays_finite": all(np.isfinite(value).all() for value in arrays.values()),
        "raw_PID_all_12_preserved": all(v["raw_PID"] == raw_pid for v in rows),
        "frozen_inner_all_12_preserved": all(v["frozen_surface"] == dataclasses.asdict(surface) for v in rows),
        "accepted_loft_coefficients_all_12_preserved": all(v["accepted_loft_coefficients"] == [.75, 1.3] for v in rows),
        "all_6_Jari_first_fuse_minimum_distances_10m_within_1mm": all_fuse,
        "maximum_first_fuse_boundary_deviation_m": max(abs(s["minimum_range_m"]-10.) for s in summaries),
        "force_bookkeeping_and_angular_law_passed": True,
        "baseline_simulations_rerun": 0, "new_Jari_free_simulations": 6, "new_original_native_free_simulations": 6,
        "simulation_counts_are_unique_delivered_configurations": True,
        "new_free_integrations_this_invocation": computed_this_invocation,
        "saved_new_runs_recovered_without_integration": recovered_saved_new_runs,
        "control_clock_hz": 48, "plant_substeps": 4, "clock_shift_s": 0., "observed_missile_after_creation": False}
    required = ("all_existing_inputs_and_artifacts_unchanged", "all_arrays_finite", "raw_PID_all_12_preserved",
                "frozen_inner_all_12_preserved", "accepted_loft_coefficients_all_12_preserved", "all_6_Jari_first_fuse_minimum_distances_10m_within_1mm")
    if not all(verification[k] for k in required) or verification["source_Jari_rows_retained"] != 631:
        raise AssertionError(verification)
    document = {"status": "four_coefficient_Jari_drag_fit_and_free_AB_complete", "fit": fit,
        "cases": records, "native_near_axis": near, "force_bookkeeping": force_checks,
        "verification": verification, "preserved_input_hashes": before,
        "production_or_GUI_changed": False, "fit_family_changed_after_offaxis_checks": False,
        "near_gate_all_passed": all(v["full_angular_height_gate_passed"] for row in near.values() for v in row["variants"].values()),
        "limitations": ["Jari remains interim; real game validation pending.",
            "Entire .28 rocket equals local .27 but .28 shared global is not verified.",
            "Four coefficients fit the existing zero-degree Jari diagnostic, not an identified native Cxi function.",
            "Other angles were already inspected before this fit; transfer checks are not blind.",
            "Source display decimation and unknown omega frame remain; derived normal load is not acc_lift.",
            "Below training Mach/altitude/AoA domains is an unvalidated numerical extension.",
            "Native near-axis speed/attitude are replay-derived candidate observations; no independent native normal-load channel is available."]}
    np.savez_compressed(OUTPUT / "trajectories.npz", **arrays)
    (OUTPUT / "results.json").write_text(json.dumps(comparison.jsonable(document), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(OUTPUT), "coefficients": dataclasses.asdict(coefficients),
                      "verification": verification, "near_gate_all_passed": document["near_gate_all_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
