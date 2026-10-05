"""Independent extrema of declared simple game-force hypotheses.

This probe does not change or claim to reproduce the game or Jari solver.
The four uncapped mid-speed reference loads calibrate one positive slope;
the low-speed/capped points and all angles remain transfer diagnostics.
One explicitly labelled two-coefficient sensitivity tests whether suppressing
the nominal thrust-normal projection explains the near-38 degree branch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jari_formula_research"
G = 9.80665


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def bounded_minimum(function, lower, upper):
    ratio = (math.sqrt(5.) - 1.) / 2.
    a, b = lower, upper
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(48):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = function(d)
    candidates = [lower, upper, (a + b) / 2.]
    return min(candidates, key=function)


def main():
    result_path = OUT / "results.json"
    result = json.loads(result_path.read_text())
    reference = json.loads((ROOT / result["reference"]).read_text())
    raw_path = ROOT / "data/datamine_2.59.0.27/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
    raw = json.loads(raw_path.read_text())["rocket"]
    frozen_path = ROOT / "config/r771_simple_joint_frozen.json"
    protected = frozen_path.read_bytes()
    frozen = json.loads(protected)["coefficients"][0]
    settings = reference["settings"]
    t0 = settings["sea_level_temperature_c_input"] + 273.15
    temp = t0 - .0065 * settings["asl_m"]
    pressure = 101325. * (temp / t0) ** (G / (287.05287 * .0065))
    rho = pressure / (287.05287 * temp)
    area = math.pi * raw["caliber"] ** 2 / 4 * raw["wingAreaMult"]
    mass = settings["displayed_current_mass_kg"]
    points = reference["points"]
    velocity = np.array([p["speed_mps"] for p in points])
    observed = np.array([p["max_overload_g"] for p in points])
    observed_angle = np.array([p["aerodynamic_aoa_deg"] for p in points])
    anchor = np.array([2, 3, 4, 5])
    scale = .5 * rho * velocity ** 2 * area / (mass * G)
    thrust_g = raw["force"] / (mass * G)
    alpha = np.linspace(0., math.pi / 2, 18001)
    candidates = {
        "wind_linear": (alpha, math.pi / 2, "f(alpha)=alpha"),
        "body_linear": (alpha * np.cos(alpha), math.pi / 2, "f(alpha)=alpha*cos(alpha)"),
        "body_cos_squared": (alpha * np.cos(alpha) ** 2, math.pi / 2, "f(alpha)=alpha*cos(alpha)^2"),
        "body_sine": (np.sin(alpha) * np.cos(alpha), math.pi / 2, "f(alpha)=sin(alpha)*cos(alpha)"),
        "wind_linear_saturates_at_raw_angle": (np.minimum(alpha, raw["CyMaxAoA"]), math.pi / 2, "f(alpha)=min(alpha,1.3), 1.3 interpreted radians"),
        "wind_linear_raw_angle_domain": (alpha, raw["CyMaxAoA"], "f(alpha)=alpha; feasible alpha<=1.3 rad"),
    }
    sin_alpha = np.sin(alpha)
    command_cap = raw["guidance"]["guidanceAutopilot"]["reqAccelMax"]

    def evaluate(name, coefficient, gravity=0., thrust_multiplier=1., step=1):
        shape, upper, _ = candidates[name]
        active = alpha <= upper + 1e-14
        aa = alpha[active][::step]
        force = coefficient * scale[:, None] * shape[active][::step][None, :] + thrust_multiplier * thrust_g * sin_alpha[active][::step][None, :] - gravity
        maximum_index = np.argmax(force, axis=1)
        maximum = force[np.arange(len(velocity)), maximum_index]
        crossing = force >= command_cap
        has_crossing = np.any(crossing, axis=1)
        chosen = np.where(has_crossing, np.argmax(crossing, axis=1), maximum_index)
        return np.minimum(maximum, command_cap), np.degrees(aa[chosen]), maximum

    def record(name, coefficient, gravity=0., thrust_multiplier=1.):
        load, angle, uncapped = evaluate(name, coefficient, gravity, thrust_multiplier)
        return {"coefficient": float(coefficient), "thrust_normal_multiplier": float(thrust_multiplier),
                "fixed_gravity_offset_g": float(gravity), "predicted_envelope_g": load.tolist(),
                "predicted_envelope_angle_deg": angle.tolist(), "uncapped_maximum_g": uncapped.tolist(),
                "rmse_all8_g": rmse(observed, load), "rmse_anchor4_g": rmse(observed[anchor], load[anchor]),
                "transfer_rmse_low2_g": rmse(observed[:2], load[:2]), "transfer_rmse_cap2_g": rmse(observed[-2:], load[-2:]),
                "angle_rmse_all8_deg": rmse(observed_angle, angle),
                "angle_rmse_anchor4_deg": rmse(observed_angle[anchor], angle[anchor])}

    comparison = {}
    for name, (_, upper, formula) in candidates.items():
        comparison[name] = {"formula": formula, "feasible_angle_deg": [0., math.degrees(upper)], "conventions": {}}
        for gravity in (0., 1.):
            cost = lambda coefficient: rmse(observed[anchor], evaluate(name, coefficient, gravity)[0][anchor])
            coefficient = bounded_minimum(cost, 5., 40.)
            tolerance = cost(coefficient) + .1
            interval = []
            for bound in (5., 40.):
                if cost(bound) <= tolerance:
                    interval.append(bound)
                    continue
                outside, inside = bound, coefficient
                for _ in range(36):
                    middle = (outside + inside) / 2
                    if cost(middle) <= tolerance:
                        inside = middle
                    else:
                        outside = middle
                interval.append(inside)
            comparison[name]["conventions"][str(int(gravity))] = {
                "frozen_slope": record(name, frozen, gravity),
                "one_coefficient_mid_load_calibration": {
                    **record(name, coefficient, gravity),
                    "objective_sensitivity_coefficient_band": interval,
                    "band_rule": "anchor load RMSE <= minimum + 0.1g; sensitivity band, not a statistical identification interval"}}

    # This is a sensitivity, not an assertion that thrust projection can change.
    # Same four mid-speed points calibrate C and K against both reported axes.
    best = None
    for coefficient in np.linspace(5., 40., 71):
        for multiplier in np.linspace(0., 1.5, 61):
            load, angle, _ = evaluate("body_cos_squared", coefficient, thrust_multiplier=multiplier, step=10)
            cost = rmse(observed[anchor], load[anchor]) ** 2 + rmse(observed_angle[anchor], angle[anchor]) ** 2
            if best is None or cost < best[0]:
                best = cost, coefficient, multiplier
    _, c0, k0 = best
    for coefficient in np.linspace(max(5., c0 - .5), min(40., c0 + .5), 41):
        for multiplier in np.linspace(max(0., k0 - .025), min(1.5, k0 + .025), 41):
            load, angle, _ = evaluate("body_cos_squared", coefficient, thrust_multiplier=multiplier, step=10)
            cost = rmse(observed[anchor], load[anchor]) ** 2 + rmse(observed_angle[anchor], angle[anchor]) ** 2
            if cost < best[0]:
                best = cost, coefficient, multiplier
    sensitivity = record("body_cos_squared", best[1], thrust_multiplier=best[2])
    sensitivity.update({"classification": "two_coefficient_mid_load_and_angle_calibration_sensitivity_not_game_model",
        "objective": "sum of anchor load RMSE squared / (1g)^2 and anchor angle RMSE squared / (1deg)^2",
        "coefficient_bounds": [5., 40.], "thrust_multiplier_bounds": [0., 1.5],
        "inference": "A small fitted thrust multiplier implies angle/force semantics, not permission to remove physical thrust from the plant."})
    result.update({"status": "bounded_public_source_and_independent_extrema_analysis_complete_not_model_validation",
        "assumed_environment": {"rho_kg_m3": rho, "area_m2": area, "mass_kg": mass,
            "raw_thrust_n": raw["force"], "thrust_normal_max_g": thrust_g, "command_cap_g": command_cap},
        "candidate_registration": {"count": len(candidates), "shape_parameters_fitted": False,
            "coefficient_bounds": [5., 40.], "anchor_indices": anchor.tolist(),
            "angle_resolution_deg": .005, "load_formula": "(q*S*C*f(alpha)+K*T*sin(alpha))/(m*g)-fixed_gravity_offset",
            "envelope_rule": "Independently maximize over declared angle domain. If 50g is attainable, report first ascending-angle crossing as a declared tie rule; Jari's own tie rule is unknown."},
        "independent_envelope_candidates": comparison, "two_coefficient_sensitivity": sensitivity,
        "limitations": ["Jari game/model version, force equations, optimality definition and aerodynamic angle semantics remain unavailable.",
            "No additional Jari calculation or service quota consumed.",
            "Raw CyMaxAoA=1.3 as angle threshold/domain is an explicit hypothesis; no primary-source support was found.",
            "Calibration coefficients are effective combinations of density, area and force slope; no coefficient is promoted to the core model.",
            "No fin equilibrium or closed-loop reachability constraints inferred from the eight tooltip points.",
            "The two-coefficient sensitivity deliberately changes the thrust projection multiplier, so its fit is evidence of a possible semantic degeneracy, not a verified plant law."],
        "verification": {"all8_reference_points_retained": True, "all_predictions_finite": True,
            "frozen_configuration_unchanged": frozen_path.read_bytes() == protected,
            "core_profile_pid_default_changes": False}})
    assert result["verification"]["frozen_configuration_unchanged"]
    for item in comparison.values():
        for convention in item["conventions"].values():
            for entry in convention.values():
                assert np.isfinite(entry["predicted_envelope_g"]).all()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(velocity, observed, "ko-", label="Jari tooltip maximum")
    axes[1].plot(velocity, observed_angle, "ko-", label="Jari aerodynamic angle")
    for name in ("wind_linear", "body_linear", "body_cos_squared", "body_sine"):
        row = comparison[name]["conventions"]["0"]["one_coefficient_mid_load_calibration"]
        axes[0].plot(velocity, row["predicted_envelope_g"], ".--", label=name)
        axes[1].plot(velocity, row["predicted_envelope_angle_deg"], ".--", label=name)
    axes[0].plot(velocity, sensitivity["predicted_envelope_g"], "s:", label="cos squared: two-coefficient sensitivity")
    axes[1].plot(velocity, sensitivity["predicted_envelope_angle_deg"], "s:", label="cos squared: two-coefficient sensitivity")
    axes[0].set(xlabel="Speed [m/s]", ylabel="Predicted maximum load [g]", title="Independent maximization; four mid-speed loads calibrate slope")
    axes[1].set(xlabel="Speed [m/s]", ylabel="Maximizing angle [deg]", title="Angles are predicted; capped-load tie rule is declared")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "independent_envelopes.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"output": str(result_path), "two_coefficient_sensitivity": sensitivity,
        "calibrated_zero_gravity": {name: x["conventions"]["0"]["one_coefficient_mid_load_calibration"] for name, x in comparison.items()}}, indent=2))


if __name__ == "__main__":
    main()
