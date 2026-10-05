"""One fixed distance-shaped biased-PN trial on the frozen replay plant.

This compares request laws, without calibration or production changes. The
selected proxy is inspired by biased-PN / lead-angle shaping, not an identified
native game law or an exact implementation of a particular published law.
"""
from __future__ import annotations

import json
import math

import numpy as np
import matplotlib.pyplot as plt

import analyze_r771_frozen_loft as loft

frozen, shared, old, model = loft.frozen, loft.shared, loft.old, loft.model
OUT = old.batch.BASE / "bpn_comparison"
MODES = ("current_pn", "distance_shaped_bpn")
SUBSTEPS = 10


def shaped_bias(p, gamma, los, distance, initial_range):
    xi = float(np.clip(distance / initial_range, 0., 1.))
    eta = gamma - los
    eta_cmd = p["loft_elevation_rad"] * xi
    raw_g = p["loft_angle_to_accel_mult"] * (eta_cmd - eta)
    return raw_g, eta, eta_cmd, xi


def make_bpn_request(original_request, initial_ranges):
    def request(t, state, replay, p, loft_armed=True, force_geometric_active=None):
        original = original_request(t, state, replay, p, loft_armed, force_geometric_active)
        if not p["loft_enabled"]:
            return original
        assert "geometric_loft_mode" not in p
        assert p["loft_reference"] == "flight_path" and p["loft_omega_unit"] == "rad_per_s"
        gamma = float(state[2])
        los = math.radians(original["los_elevation_deg"])
        raw_g, eta, eta_cmd, xi = shaped_bias(p, gamma, los, original["range_m"], initial_ranges[id(replay)])
        speed = float(np.interp(t, replay["t"], replay["speed2"]))
        cap_g = p["loft_omega_max_deg_s"] * speed / model.G
        limited_g = model.clamp(raw_g, -cap_g, cap_g)
        normal = np.array([-math.sin(gamma), math.cos(gamma)])
        loft_normal_mps2 = limited_g * model.G * normal[1]
        max_accel = p["req_accel_max_g"] * model.G
        command_normal = model.clamp(original["pn_for_command_g"] * model.G + loft_normal_mps2,
                                     -max_accel, max_accel)
        required_vector = command_normal * normal + np.array([0., model.G])
        req_norm = float(np.linalg.norm(required_vector))
        if req_norm > max_accel:
            required_vector *= max_accel / req_norm
        required_normal = model.clamp(float(np.dot(required_vector, normal)), -max_accel, max_accel)
        return dict(original, a_req_g=required_normal / model.G,
                    error_g=required_normal / model.G - original["actual_g"],
                    loft_raw_g=raw_g, loft_limited_g=limited_g,
                    loft_projected_g=loft_normal_mps2 / model.G,
                    loft_reference_rad=eta, loft_target_rad=eta_cmd,
                    bpn_normalized_range=xi)
    return request


def guidance_summary(shot, pred, request):
    rows = []
    for tick in range(round(3.5 * 48) + 1):
        t = tick / 48
        state = np.array([np.interp(t, pred["t"], pred[k]) for k in ("theta", "omega", "gamma", "x", "y")])
        rows.append(request(t, state, shot["replay"], shot["p"]))
    return {
        "clock": "forward states at48Hz,0..3.5s",
        "loft_cap_active_tick_count": int(sum(abs(r["loft_raw_g"] - r["loft_limited_g"]) > 1e-10 for r in rows)),
        "request_limit_active_tick_count": int(sum(abs(r["a_req_g"]) >= shot["p"]["req_accel_max_g"] - 1e-10 for r in rows)),
        "loft_raw_range_g": [min(r["loft_raw_g"] for r in rows), max(r["loft_raw_g"] for r in rows)],
        "request_range_g": [min(r["a_req_g"] for r in rows), max(r["a_req_g"] for r in rows)],
    }


def draw(shots, predictions, lo, hi, filename):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22})
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    for col, name in enumerate(loft.H_NAMES):
        obs = shots[name]["observed"]
        for row, (key, factor, label) in enumerate((("omega", 1., "Pitch rate [rad/s]"),
                 ("alpha", 180/math.pi, "Incidence [deg]"), ("theta", 180/math.pi, "Pitch [deg]"),
                 ("gamma", 180/math.pi, "Flight path [deg]"))):
            ax = axes[row, col]
            ax.plot(obs["t"], obs[key] * factor, ".", color="#202020", ms=3, label="Decoded replay")
            for mode, color, curve_label in ((MODES[0], "#15856f", "Current PN + loft"),
                                      (MODES[1], "#8b48ad", "Distance-shaped BPN proxy")):
                pr = predictions[name][mode]
                ax.plot(pr["t"], pr[key]*factor, color=color, lw=1.7, label=curve_label)
            ax.set_xlim(lo, hi)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0: ax.set_title(name)
            if row == 3: ax.set_xlabel("Native b5 age [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed guidance comparison; frozen plant/PID, replay speed and target inputs; no calibration")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main():
    cfg, coeff, _ = loft.read_snapshot()
    raw, p0 = old.batch.load_all()
    shots = {n: loft.make_loft_shot(raw[n]) for n in loft.H_NAMES}
    shots.update({n: old.make_shot(raw[n], "b5") for n in shared.NAMES})
    for s in shots.values():
        assert s["p"]["tail_arm_m"] == .4
        assert [s["p"][k] for k in ("kp", "ki", "kd", "integral_limit")] == [.002125, .006481, .000023, 1.]
        for key, value in cfg["fixed_plant_parameters"].items():
            if key != "density": assert s["p"][key] == value
    initial_ranges = {id(s["replay"]): float(np.linalg.norm(model.target_at(s["replay"], 0.)[0] - s["initial"][3:5]))
                      for s in shots.values()}
    original_request = model.request
    bpn_request = make_bpn_request(original_request, initial_ranges)
    predictions, records, arrays = {}, {}, {}
    try:
        for name, shot in shots.items():
            predictions[name], records[name] = {}, {}
            for mode, request in ((MODES[0], original_request), (MODES[1], bpn_request)):
                model.request = request
                pred = frozen.simulate(shot, coeff, substeps=SUBSTEPS)
                if pred is None:
                    raise RuntimeError(f"{name} {mode} exceeded existing .6rad incidence validity guard")
                predictions[name][mode] = pred
                records[name][mode] = loft.metrics(shot, pred)
                records[name][mode]["forward_guidance"] = guidance_summary(shot, pred, request)
                for key, value in pred.items():
                    if isinstance(value, np.ndarray): arrays[name + "__" + mode + "__" + key] = value
            for key, value in shot["observed"].items(): arrays[name + "__observed__" + key] = value
        # One focused verification pass, with no retuning or additional variants.
        archives = np.load(loft.OUT / "trajectories.npz")
        baseline_diff, no_loft_diff, sample_counts, moment_identity, first_active = {}, {}, {}, {}, {}
        for name, shot in shots.items():
            prior_mode = "pn_plus_loft" if name in loft.H_NAMES else "no_loft"
            baseline_diff[name] = {key: float(np.max(np.abs(predictions[name][MODES[0]][key] - archives[name + "__" + prior_mode + "__" + key])))
                                  for key in predictions[name][MODES[0]] if isinstance(predictions[name][MODES[0]][key], np.ndarray)}
            assert max(baseline_diff[name].values()) < 1e-12
            obs = shot["observed"]
            assert len(obs["t"]) == len(raw[name]["b5"]) == shot["sample_count"]
            sample_counts[name] = len(obs["t"])
            moment_identity[name], first_active[name] = {}, {}
            for mode in MODES:
                pr = predictions[name][mode]
                moment_identity[name][mode] = float(np.max(np.abs(pr["moment_nm"] + .4*pr["normal_body_n"] - pr["reference_couple_nm"])))
                first_active[name][mode] = pr["first_active"]["start_s"]
                assert moment_identity[name][mode] < 1e-8 and first_active[name][mode] == 14/48
            if name in shared.NAMES:
                no_loft_diff[name] = {key: float(np.max(np.abs(predictions[name][MODES[0]][key] - predictions[name][MODES[1]][key])))
                                      for key in predictions[name][MODES[0]] if isinstance(predictions[name][MODES[0]][key], np.ndarray)}
                assert max(no_loft_diff[name].values()) == 0.
                assert original_request(0., shot["initial"], shot["replay"], shot["p"]) == bpn_request(0., shot["initial"], shot["replay"], shot["p"])
        model.request = bpn_request
        fine = frozen.simulate(shots["H110"], coeff, substeps=2*SUBSTEPS)
        assert fine is not None
        coarse = predictions["H110"][MODES[1]]
        halving = {key: float(np.max(np.abs(coarse[key] - np.interp(coarse["t"], fine["t"], fine[key]))))
                   for key in ("theta", "omega", "alpha", "gamma", "y")}
        assert halving["omega"] < 1e-5 and halving["alpha"] < 1e-6
        assert all(np.isfinite(value).all() for value in arrays.values())
        # Same-state shaping check: eta=0, xi=1 yields24deg; xi=0 yields0.
        checks = [shaped_bias(shots["H110"]["p"], .2, .2, d, 100.) for d in (100., 0.)]
        assert checks[0][2] == math.radians(24.) and checks[1][2] == 0.
        assert checks[0][0] == 17.5*math.radians(24.) and checks[1][0] == 0.
    finally:
        model.request = original_request
    comparisons = {}
    for name in shots:
        comparisons[name] = {window: {key: records[name][MODES[1]]["windows"][window][key+"_rmse"] /
                                      records[name][MODES[0]]["windows"][window][key+"_rmse"]
                                      for key in loft.UNITS}
                             for window, *_ in loft.WINDOWS}
    result = {
        "frozen_snapshot": str(loft.SNAPSHOT), "frozen_coefficients": coeff.tolist(),
        "fixed_reference_damping_s_minus1": frozen.KW, "source_version": p0["source_version"],
        "optimization_performed": False, "production_changed": False, "native_game_law_identified": False,
        "simulation": "Continuous creation-state frozen RK4; fixed48Hz/end-gain zero-memory PID; measuredV(t) and packet-derived target are external",
        "horizon_s": 3.5, "substeps_per_control_interval": SUBSTEPS,
        "metric_units": loft.UNITS,
        "metric_definition": "Ordinary RMSE at every retained native b5 observation; same-tick rows count equally; native observed event extrema, no parabolic observation-peak estimate",
        "baseline": "Current PN+additive loft already has biased-PN structure; this trial changes only the existing vertical loft bias formula",
        "candidate": {
            "formula": "eta=gamma-lambda; xi=clip(R/R0,0,1); eta_cmd=loftElevation_rad*xi; vertical_bias_g=loftAngleToAccelMult*(eta_cmd-eta)",
            "lambda": "actual target inertial LOS elevation; no predicted-intercept solver",
            "R0": "range from decoded creation-state position to unchanged target_at(t=0), computed separately for each case",
            "projection": "same worldvertical-to-flight-pathnormal n_y projection as baseline",
            "no_loft": "bias=0; returns original PN request unmodified and exactly reproduces all baseline arrays",
            "family_provenance": "Chosen distance-shapedbiasedPN proxy inspired by BPN/APG families; not exact paper law or known game semantics",
        },
        "raw_loft_fields": {"loftElevation": 24., "loftTargetElevation": -3.5, "loftAngleToAccelMult": 17.5, "loftTargetOmegaMax": .5, "propNavMult": 6.},
        "common_prototype_interpretations": {"loftElevation": "degrees converted to radians", "loftAngleToAccelMult": "g/rad in both laws",
            "loftTargetOmegaMax": "Existing0.5rad/s loft-only acceleration cap, held common solely to isolate bias formula; game unit/meaning unidentified",
            "loftTargetElevation": "Raw-3.5 retained but unconsumed in both; no invented exit gate",
            "other": "PN6, timeToHit gain, timeToGain output scheduling,50g command/vector clipping, gravity compensation, PID, creation mapping unchanged"},
        "scope_limitations": ["Replay-conditioned kinematic fit cannot establish hit probability, range, drag/energy efficiency, or fullflight advantage",
            "Frozen linear plant fitted near4deg transfers to27deg loft regime; guidance-only comparison cannot uniquely identify large-incidence game dynamics",
            "Frozen empirical plant does not dynamically use raw finsLatAccel variation; similar crosscase predictions do not identify its game mapping"],
        "initial_range_m": {name: initial_ranges[id(shot["replay"])] for name, shot in shots.items()},
        "records": records, "candidate_over_baseline_rmse_ratios": comparisons,
        "verification": {"frozen_coefficients_exact": True, "baseline_saved_array_reproduction_max_abs": baseline_diff,
            "no_loft_all_array_difference_max_abs": no_loft_diff, "native_sample_counts_all_retained": sample_counts,
            "all_arrays_finite": True, "H110_BPN_step_halving_max_abs": halving,
            "same_state_range_shape_eta_cmd_rad": [checks[0][2], checks[1][2]],
            "same_state_range_shape_bias_g": [checks[0][0], checks[1][0]],
            "first_active_start_s": first_active, "surface_moment_identity_max_nm": moment_identity,
            "moment_identity_is_bookkeeping_not_independent_physics": True, "request_patch_restored": model.request is original_request},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    draw(shots, predictions, 0., 2.45, "comparison.png")
    print(json.dumps({"output": str(OUT), "full_window": {name: {mode: records[name][mode]["windows"]["full"] for mode in MODES} for name in shots},
                      "verification": {"baseline_max_abs": max(max(v.values()) for v in baseline_diff.values()), "no_loft_max_abs": max(max(v.values()) for v in no_loft_diff.values()), "step_halving": halving}}, indent=2))


if __name__ == "__main__":
    main()
