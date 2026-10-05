"""Conditional request identification with the frozen R-77-1 plant and PID.

The replay has no measured fin angle or body rate. A derivative-based inverse
is therefore reported only as a sensitivity diagnostic. A low-dimensional
request is then identified by complete forward runs; no replay state enters
the integrator and no physical or PID coefficient is optimized.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math

import numpy as np

import audit_r77_1_open_loop_rotation as rotation
import check_r77_1_capture_strategy as capture
from aim120_model.propulsion import PiecewisePropulsion
from check_r77_1_smooth_handover_2p1 import control_rows, native_observations, read_trace
from run_r77_1_current_full_forward import sha


ROOT = capture.ROOT
PREFIX = ROOT / "outputs/r77_1_fixed_pid_request"
KNOTS = np.array([.30, .55, .80, 1.05, 1.30, 1.60, 1.90, 2.15, 2.40, 2.75, 3.10])
PN_ONLY_AT = 3.50
INITIAL_VALUES = np.array([35., 35., 35., 35., 35., 35., 35., 25., 16., 14., 12.])


def runner_factory(plant, gain_table, values, knots=KNOTS):
    """Replace only the PN request expression in the existing verified runner."""
    factory = inspect.getsource(capture.build_runner)
    old = "return namespace['run_alpha_servo'], hashlib.sha256(pid.encode()).hexdigest()"
    assert factory.count(old) == 1
    factory = factory.replace(old, "return source, namespace, hashlib.sha256(pid.encode()).hexdigest()")
    namespace = dict(vars(capture))
    exec(compile(factory, "<capture factory export>", "exec"), namespace)
    source, run_namespace, pid_hash = namespace["build_runner"](
        plant, pn_capture=False, retain_command=False,
        request_gain_table=gain_table, pn_only=True)
    old = "            a_cmd *= request_gain(time_s)\n"
    assert source.count(old) == 1
    source = source.replace(old,
        "            a_cmd = candidate_pre_gain(time_s, a_cmd / G0) * G0\n" + old)
    # The complete force, moment, RK4 and original PID text must be identical.
    original, _, _ = namespace["build_runner"](
        plant, pn_capture=False, retain_command=False,
        request_gain_table=gain_table, pn_only=True)
    assert source.replace("            a_cmd = candidate_pre_gain(time_s, a_cmd / G0) * G0\n", "") == original

    def candidate_pre_gain(time_s, pn_g):
        if time_s <= knots[-1]:
            return float(np.interp(time_s, knots, values))
        if time_s >= PN_ONLY_AT:
            return pn_g
        fraction = (time_s - knots[-1]) / (PN_ONLY_AT - knots[-1])
        weight = fraction * fraction * (3. - 2. * fraction)
        return (1. - weight) * float(values[-1]) + weight * pn_g

    run_namespace["candidate_pre_gain"] = candidate_pre_gain
    exec(compile(source, "<fixed PID request candidate>", "exec"), run_namespace)
    return run_namespace["run_alpha_servo"], pid_hash, candidate_pre_gain


def conditional_inverse(obs, p, config, plant, baseline_rows):
    """Infer a request from four permissible derivative stencils.

    This uses the baseline controller state at 0.2 s, because the replay does
    not observe the earlier fin command. It is a diagnostic, not the fitted
    request used in free forward evaluation.
    """
    rotation.PLANT = plant
    rotation.RHO = plant.atmo.density_kg_m3
    propulsion = PiecewisePropulsion.from_config(config)
    motion = rotation.kinematics_and_bounds(obs, p, propulsion)
    arrays = rotation.nominal_plant_arrays(obs, p, propulsion, motion)
    model_horizontal_g = []
    for t, total_alpha, ah, qbar, mass in zip(obs["time"], obs["alpha"], motion["alpha_h"],
                                               arrays["qbar"], arrays["mass"]):
        thrust = propulsion.sample(float(t)).thrust_n
        sinc = math.sin(total_alpha) / total_alpha if total_alpha else 1.
        model_horizontal_g.append((qbar * p.lift_area_per_rad + thrust * sinc) * ah /
                                  (mass * rotation.G0))
    model_horizontal_g = np.asarray(model_horizontal_g)
    baseline_at_start = next(r for r in baseline_rows if abs(r["time_s"] - .2) < 1e-10)
    inferred = {}
    for name, (count, degree, _) in rotation.SCHEMES.items():
        scheme = rotation.scheme_result(name, count, degree, obs, motion, arrays, p)
        grid = np.arange(.20, 3.501, .02)
        wanted_fin = np.interp(grid, obs["time"], scheme["required_delta"])
        # The PID receives the fixed model's force, not the replay load.
        actual_g = np.interp(grid, obs["time"], model_horizontal_g)
        last_error = baseline_at_start["commanded_horizontal_acceleration_g"] - baseline_at_start["horizontal_specific_force_g"]
        integral = baseline_at_start["pid_integral_output"]
        rows = []
        for t, delta, load_g in zip(grid[1:], wanted_fin[1:], actual_g[1:]):
            u = delta / p.horizontal_fin_limit_rad
            denom = p.acceleration_pid_kp + p.acceleration_pid_ki * .02 + p.acceleration_pid_kd / .02
            error = (u - integral + p.acceleration_pid_kd * last_error / .02) / denom
            proposed = float(np.clip(integral + p.acceleration_pid_ki * error * .02,
                                     -p.acceleration_pid_integral_limit, p.acceleration_pid_integral_limit))
            raw = p.acceleration_pid_kp * error + proposed + p.acceleration_pid_kd * (error - last_error) / .02
            if abs(raw) > 1. and raw * error > 0.:
                # Saturated output does not uniquely determine the error.
                rows.append(dict(time_s=float(t), equivalent_fin_rad=float(delta),
                                 inferred_request_g=None, reason="PID saturation / nonunique"))
            else:
                integral = proposed
                last_error = error
                rows.append(dict(time_s=float(t), equivalent_fin_rad=float(delta),
                                 inferred_request_g=float(load_g + error), reason="conditional inverse"))
        inferred[name] = rows
    return inferred, dict(native_time_s=obs["time"].tolist(),
                          measured_horizontal_g=np.sqrt(obs["load"]**2 - 1.).tolist(),
                          model_horizontal_g=model_horizontal_g.tolist())


def evaluate(run, config, p, scenario, observed, values, horizon):
    result = run(config, p, scenario, max_time_s=horizon,
                 plant_dt_s=.002, control_dt_s=.02)
    rows = result.samples
    if result.event == "numerical_failure" or rows[-1]["time_s"] < min(horizon, 3.5) - .01:
        return None, result
    times = np.array([row["time_s"] for row in rows])
    samples = [o for o in observed if .5 <= o["time_s"] <= min(3.5, times[-1])]
    residual = []
    for o in samples:
        t = o["time_s"]
        # Total AoA and normal load have their own scales; speed is held out.
        residual.extend([
            (np.interp(t, times, [r["alpha_deg"] for r in rows]) - o["alpha_deg"]) / 1.5,
            (np.interp(t, times, [r["wind_normal_specific_force_g"] for r in rows]) - o["wind_normal_specific_force_g"]) / 1.2,
        ])
    # Favor a small number of broad request changes over frame-to-frame commands.
    residual.extend(.012 * np.diff(values, n=2))
    return np.asarray(residual), result


def identify(run, config, p, scenario, observed, values):
    residual, _ = evaluate(run, config, p, scenario, observed, values, 3.52)
    assert residual is not None
    history = []
    for iteration in range(12):
        before = float(np.linalg.norm(residual))
        jacobian = np.empty((len(residual), len(values)))
        for i in range(len(values)):
            perturbed = values.copy()
            perturbed[i] += .4
            values[:] = perturbed
            r, _ = evaluate(run, config, p, scenario, observed, perturbed, 3.52)
            values[i] -= .4
            assert r is not None
            jacobian[:, i] = (r - residual) / .4
        step = np.linalg.solve(jacobian.T @ jacobian + .08 * np.eye(len(values)), -jacobian.T @ residual)
        accepted = False
        for fraction in (1., .5, .25, .125):
            trial = np.clip(values + fraction * np.clip(step, -8., 8.), 0., 49.)
            previous = values.copy()
            values[:] = trial
            trial_residual, _ = evaluate(run, config, p, scenario, observed, trial, 3.52)
            if trial_residual is not None and np.linalg.norm(trial_residual) < before - 1e-4:
                residual = trial_residual
                accepted = True
                break
            values[:] = previous
        history.append(dict(iteration=iteration, objective_before=before,
                            objective_after=float(np.linalg.norm(residual)), accepted=accepted))
        print(f"inverse iteration {iteration}: {before:.3f} -> {np.linalg.norm(residual):.3f}", flush=True)
        if not accepted or before - np.linalg.norm(residual) < .015:
            break
    return values, history


def comparison(rows, observed):
    times = np.array([r["time_s"] for r in rows])
    result = []
    for o in observed:
        if o["time_s"] > times[-1]:
            continue
        item = dict(time_s=o["time_s"])
        for key in ("alpha_deg", "wind_normal_specific_force_g", "speed_kmh"):
            item[key + "_measured"] = o[key]
            item[key + "_predicted"] = float(np.interp(o["time_s"], times, [r[key] for r in rows]))
            item[key + "_error"] = item[key + "_predicted"] - o[key]
        result.append(item)
    return result


def window_errors(rows, start, end):
    selected = [r for r in rows if start - 1e-12 <= r["time_s"] <= end + 1e-12]
    return {field: dict(max_abs=float(max(abs(r[field + "_error"]) for r in selected)),
                        mean_abs=float(np.mean([abs(r[field + "_error"]) for r in selected])))
            for field in ("alpha_deg", "wind_normal_specific_force_g", "speed_kmh")}


def control_checks(rows, p, raw_limit):
    limit_g = math.sqrt(p.command_limit_g ** 2 - 1.)
    assert all(math.isfinite(value) for row in rows for value in row.values()
               if isinstance(value, (float, int)))
    assert all(b["time_s"] > a["time_s"] for a, b in zip(rows, rows[1:]))
    assert all(abs(row["commanded_horizontal_acceleration_g"]) <= limit_g + 1e-10 for row in rows)
    mapping = max(abs(row["horizontal_fin_deflection_rad"] -
                      np.clip(row["pid_raw_output"], -1., 1.) * raw_limit) for row in rows)
    assert mapping < 1e-12
    force_excess = max(abs(row["fin_control_force_n"]) - row["fin_control_force_limit_n"] for row in rows)
    assert force_excess < 1e-8
    ticks = control_rows(rows)
    integral_error = 0.
    for before, row in zip(ticks, ticks[1:]):
        expected_i = np.clip(before["pid_integral_output"] +
                             p.acceleration_pid_ki * row["pid_error_g"] * .02,
                             -p.acceleration_pid_integral_limit,
                             p.acceleration_pid_integral_limit)
        proposed = row["pid_proportional_output"] + expected_i + row["pid_derivative_output"]
        if abs(proposed) > 1. and proposed * row["pid_error_g"] > 0.:
            expected_i = before["pid_integral_output"]
        integral_error = max(integral_error, abs(row["pid_integral_output"] - expected_i))
    assert integral_error < 1e-12
    assert all(row["thrust_n"] == 0. for row in rows if row["time_s"] >= 7.)
    assert all(row["rate_damping_moment_nm"] * row["body_rate_rad_s"] <= 1e-8 for row in rows)
    return dict(fin_mapping_error_rad=float(mapping), pid_integral_update_error=float(integral_error),
                maximum_force_cap_excess_n=float(force_excess),
                max_abs_delivered_request_g=float(max(abs(r["commanded_horizontal_acceleration_g"]) for r in rows)),
                fin_force_saturation_ticks=sum(bool(r["fin_control_force_saturated"]) for r in ticks),
                pid_output_saturation_ticks=int(sum(abs(r["pid_raw_output"]) > 1. for r in ticks)))


def plot(old, new, reduced, measured, inverse, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), sharex="col")
    for j, (field, label) in enumerate((("alpha_deg", "Total AoA [deg]"),
                                       ("wind_normal_specific_force_g", "Total normal load [g]"),
                                       ("speed_kmh", "Speed [km/h]"))):
        for col in range(2):
            ax = axes[j, col]
            ax.plot([r["time_s"] for r in measured], [r[field] for r in measured],
                    "o", color="#222", ms=3.5, label="2.57 measured")
            for rows, color, title in ((old, "#c77c27", "Previous"),
                                       (new, "#237d91", "11-knot request"),
                                       (reduced, "#739340", "6-knot reduction")):
                ax.plot([r["time_s"] for r in rows], [r[field] for r in rows],
                        color=color, lw=1.7, ls="--" if rows is reduced else "-", label=title)
            ax.set(xlim=(0., 7.9 if col == 0 else 3.5), ylabel=label)
            ax.grid(alpha=.2)
            if j == 2:
                ax.set_xlabel("Flight time [s]")
    fig.suptitle("R-77-1 | fixed plant and PID: request identification", y=.98)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True)
    ax = axes[0]
    for name, data in inverse.items():
        ax.plot([r["time_s"] for r in data], [r["inferred_request_g"] for r in data],
                lw=1, alpha=.65, label=name)
    ax.plot([r["time_s"] for r in new], [r["commanded_horizontal_acceleration_g"] for r in new],
            color="#237d91", lw=2, label="Fitted equivalent request")
    ax.plot([r["time_s"] for r in old], [r["commanded_horizontal_acceleration_g"] for r in old],
            color="#c77c27", ls="--", label="Previous request")
    ax.plot([r["time_s"] for r in reduced], [r["commanded_horizontal_acceleration_g"] for r in reduced],
            color="#739340", ls=":", label="6-knot reduction")
    ax.set(ylabel="Horizontal request [g]", xlim=(0., 3.6), ylim=(-20, 55))
    ax.legend(frameon=False, ncol=2)
    ax = axes[1]
    for rows, color, label in ((old, "#c77c27", "Previous"), (new, "#237d91", "11-knot"),
                               (reduced, "#739340", "6-knot")):
        ax.plot([r["time_s"] for r in rows],
                [math.degrees(r["horizontal_fin_deflection_rad"]) for r in rows],
                color=color, lw=1.6, ls=":" if rows is reduced else "-", label=label)
    ax.axhline(math.degrees(.460812), color="#666", ls=":")
    ax.set(xlabel="Flight time [s]", ylabel="Equivalent fin angle [deg]", xlim=(0., 3.6))
    ax.grid(alpha=.2)
    fig.suptitle("Conditional inverse and forward PID response", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(prefix.with_name(prefix.name + "_control.png"), dpi=160)
    plt.close(fig)


def main():
    baseline_path = ROOT / "outputs/r77_1_current_full_forward.json"
    baseline_trace = ROOT / "outputs/r77_1_current_full_forward_trajectory.tsv"
    replay = ROOT / "data/replays/r77_1_level_20260824.tsv"
    frozen = [ROOT / name for name in (
        "src/aim120_model/alpha_servo.py", "scripts/run_r77_1_focus_moment.py",
        "scripts/check_r77_1_capture_strategy.py", "missiles/su_r_77_1.json",
        "config/profile_h2_rebuild_defaults.json")]
    hashes_before = {str(path): sha(path) for path in [baseline_path, baseline_trace, replay, *frozen]}
    baseline = json.loads(baseline_path.read_text())
    assert (capture.VERSION, capture.COMMIT) == (baseline["source_version"], baseline["source_commit"])
    config, p, raw_limit = capture.inputs()
    p = replace(p, fin_force_area_per_rad=baseline["fixed_parameters"]["fin_force_area_per_rad"],
                guidance_delay_s=baseline["fixed_parameters"]["guidance_delay_s"])
    assert asdict(p) == baseline["fixed_parameters"]
    assert capture.digest(config) == baseline["configuration_sha256"]
    table = tuple(tuple(r) for r in baseline["raw_time_to_gain_table"])
    scenario = baseline["scenario"]
    assert scenario["target_heading_deg"] == 180.
    observed = native_observations(replay)
    old = read_trace(baseline_trace)
    plant = capture.FocusPlant(config, p, "legacy_control_scale", raw_limit)
    baseline_run, base_pid = capture.build_runner(plant, False, False,
        request_gain_table=table,
        timed_capture=(baseline["capture_cap_g"], baseline["capture_duration_s"]))
    assert base_pid == baseline["pid_block_sha256"]
    reproduced = baseline_run(config, p, scenario, max_time_s=config["performance"]["lifetime_s"],
                              plant_dt_s=.002, control_dt_s=.02)
    assert len(reproduced.samples) == len(old)
    baseline_difference = max(abs(a[k] - b[k]) for a, b in zip(reproduced.samples, old)
                              for k in capture.FIELDS)
    assert baseline_difference == 0.
    rotation.REFERENCE, rotation.CONFIG, rotation.PARAMETERS, rotation.PLANT, rotation.OBS = rotation.load_current()
    rotation.RHO = rotation.PLANT.atmo.density_kg_m3
    inverse, inverse_loads = conditional_inverse(rotation.OBS, p, config, plant, old)
    values = INITIAL_VALUES.copy()
    run, pid_hash, request = runner_factory(plant, table, values)
    assert pid_hash == base_pid
    values[:], optimization = identify(run, config, p, scenario, observed, values)
    result = run(config, p, scenario, max_time_s=config["performance"]["lifetime_s"],
                 plant_dt_s=.002, control_dt_s=.02)
    rows = result.samples
    capture.enrich(rows, p, scenario)
    simple_knots = np.array([.30, .55, .80, 1.60, 1.90, 3.10])
    simple_values = np.interp(simple_knots, KNOTS, values)
    simple_run, simple_pid_hash, _ = runner_factory(plant, table, simple_values, simple_knots)
    assert simple_pid_hash == pid_hash
    reduced = simple_run(config, p, scenario, max_time_s=config["performance"]["lifetime_s"],
                         plant_dt_s=.002, control_dt_s=.02)
    reduced_rows = reduced.samples
    capture.enrich(reduced_rows, p, scenario)
    assert result.event not in ("numerical_failure", "vertical_trim_unavailable")
    assert reduced.event not in ("numerical_failure", "vertical_trim_unavailable")
    candidate_checks = control_checks(rows, p, raw_limit)
    reduced_checks = control_checks(reduced_rows, p, raw_limit)
    assert hashes_before == {str(path): sha(path) for path in [baseline_path, baseline_trace, replay, *frozen]}
    old_comparison, new_comparison = comparison(old, observed), comparison(rows, observed)
    reduced_comparison = comparison(reduced_rows, observed)
    def geometry_at(rows, t):
        ts = [r["time_s"] for r in rows]
        return {key: float(np.interp(t, ts, [r[key] for r in rows]))
                for key in ("heading_rad", "x_m", "z_m", "distance_m", "raw_pn_request_g",
                            "alpha_deg", "wind_normal_specific_force_g", "speed_kmh")}
    report = dict(source_version=capture.VERSION, source_commit=capture.COMMIT,
        method="conditional derivative/PID inverse plus low-dimensional forward request identification",
        fixed_parameters=asdict(p), scenario=scenario, knots_s=KNOTS.tolist(),
        pre_gain_request_knots_g=values.tolist(), pn_only_after_s=PN_ONLY_AT,
        reduced_check=dict(knots_s=simple_knots.tolist(), request_knots_g=simple_values.tolist(),
                           event=reduced.event, terminal_time_s=reduced_rows[-1]["time_s"],
                           minimum_distance_m=reduced.minimum_distance_m),
        raw_time_gain_table=table, raw_pid_block_sha256=pid_hash,
        baseline_reproduction_max_difference=baseline_difference,
        optimization_history=optimization, event=result.event,
        terminal_time_s=rows[-1]["time_s"], minimum_distance_m=result.minimum_distance_m,
        inverse_stencil_ranges={name: dict(min_request_g=float(min(r["inferred_request_g"] for r in data if r["inferred_request_g"] is not None)),
                                            max_request_g=float(max(r["inferred_request_g"] for r in data if r["inferred_request_g"] is not None)),
                                            ambiguous_points=sum(r["inferred_request_g"] is None for r in data),
                                            command_cap_exceedance_points=int(sum(r["inferred_request_g"] is not None and
                                                abs(r["inferred_request_g"]) > math.sqrt(p.command_limit_g**2 - 1.)
                                                for r in data)))
                                for name, data in inverse.items()},
        inverse_force_comparison=inverse_loads,
        comparisons=dict(baseline=old_comparison, candidate=new_comparison,
                         reduced=reduced_comparison),
        free_geometry_at_s={str(t): {name: geometry_at(case, t)
                                    for name, case in (("baseline", old), ("candidate", rows),
                                                       ("reduced", reduced_rows))}
                            for t in (2.1, 3.5, 5.5, 6.6)},
        window_errors={name: {str((a, b)): window_errors(data, a, b)
                              for a, b in ((.5, 1.6), (1.9, 3.5), (5.1, 7.1))}
                       for name, data in (("baseline", old_comparison), ("candidate", new_comparison),
                                          ("reduced", reduced_comparison))},
        verification=dict(candidate=candidate_checks, reduced=reduced_checks,
                          baseline_reproduction_max_difference=baseline_difference,
                          protected_inputs_unchanged=True),
        assumptions=["level flight", "positive horizontal turn", "20ms interpolation of 19 rounded replay frames",
                     "baseline PID state at 0.2s for conditional inverse", "unknown fin and body rate"],
        interpretation="An equivalent command conditional on the frozen plant; not a uniquely identified game guidance law.",
        protected_hashes=hashes_before)
    PREFIX.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    capture.write_table(PREFIX.with_name(PREFIX.name + "_trajectory.tsv"), rows)
    capture.write_table(PREFIX.with_name(PREFIX.name + "_reduced_trajectory.tsv"), reduced_rows)
    with PREFIX.with_name(PREFIX.name + "_inverse.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stencil", "time_s", "equivalent_fin_rad", "inferred_request_g", "reason"], delimiter="\t")
        writer.writeheader()
        for name, data in inverse.items():
            for row in data:
                writer.writerow(dict(stencil=name, **row))
    plot(old, rows, reduced_rows, observed, inverse, PREFIX)
    print(json.dumps({key: report[key] for key in ("pre_gain_request_knots_g", "event", "terminal_time_s",
                                                 "minimum_distance_m", "inverse_stencil_ranges")}, indent=2))


if __name__ == "__main__":
    main()
