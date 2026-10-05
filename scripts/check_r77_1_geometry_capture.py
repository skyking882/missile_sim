"""Test geometry-only capture exits on the frozen R-77-1 candidate.

This is a bounded guidance diagnostic.  It keeps the 2.57.1.107 inputs,
plant, damping, propulsion, initial state, target course, raw PID and hard
capture-to-PN handover unchanged.  The only changed branch is the instant at
which the initial capture request is latched off.  There is no seeker-pointing
state here, so the two angles tested below are explicitly geometry proxies:

* ``body``: absolute LOS angle relative to the simulated body heading;
* ``velocity``: absolute LOS angle relative to the simulated velocity heading.

The thresholds are diagnostic sensitivity values, not inferred game values.
Replay values are used only for comparisons after each free forward integration.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

import check_r77_1_capture_strategy as capture
from run_r77_1_current_full_forward import (
    ROOT, FocusPlant, inputs, protected_hashes, VERSION, COMMIT,
    FIELDS, digest, write_table, sha,
)


THRESHOLDS_DEG = (3.0, 6.0, 10.0, 15.0)
CAPTURE_CAP_G = 35.25
CAPTURE_START_S = 0.30
BASELINE_CAPTURE_END_S = 2.10
FIRST_TRK_S = 0.90

# State and controller fields used for the causal boundary checks.  The
# geometry branch must be invisible to the original trajectory until its
# first latched exit; at the exit tick the physical state is still sampled
# before the newly selected command is integrated.
STATE_FIELDS = ("speed_kmh", "heading_rad", "x_m", "z_m",
                "body_heading_rad", "body_rate_rad_s")
CONTROLLER_FIELDS = (
    "commanded_horizontal_acceleration_g", "commanded_horizontal_alpha_rad",
    "horizontal_fin_deflection_rad", "pid_raw_output",
    "pid_proportional_output", "pid_integral_output", "pid_derivative_output",
    "total_aerodynamic_moment_nm",
)
PRE_SWITCH_FIELDS = STATE_FIELDS + CONTROLLER_FIELDS + (
    "alpha_deg", "wind_normal_specific_force_g", "horizontal_specific_force_g",
)


def parse_value(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def read_table(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: parse_value(value) for key, value in row.items()}
                for row in csv.DictReader(stream, delimiter="\t")]


def read_replay(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        native = list(csv.DictReader(stream, delimiter="\t"))
    return [dict(time_s=float(row["missile_flight_time_s"]),
                 alpha_deg=float(row["angle_of_attack_deg"]),
                 wind_normal_specific_force_g=float(row["overload_g"]),
                 speed_kmh=float(row["speed_kmh"])) for row in native], native


def export_factory_source(plant, table, duration_s):
    """Return the existing capture factory's generated source without editing it."""
    factory = inspect.getsource(capture.build_runner)
    anchor = "return namespace['run_alpha_servo'], hashlib.sha256(pid.encode()).hexdigest()"
    assert factory.count(anchor) == 1
    factory = factory.replace(
        anchor,
        "return source, namespace, hashlib.sha256(pid.encode()).hexdigest()",
    )
    factory_namespace = dict(vars(capture))
    exec(compile(factory, "<capture-factory-source-export>", "exec"), factory_namespace)
    return factory_namespace["build_runner"](
        plant, pn_capture=False, retain_command=False,
        request_gain_table=table,
        timed_capture=(CAPTURE_CAP_G, duration_s),
    )


def _wrapped_difference(lhs, rhs):
    return (lhs - rhs + math.pi) % (2.0 * math.pi) - math.pi


def geometry_angle_pair(row):
    """Return (body-axis proxy, velocity-axis proxy) in signed degrees."""
    los = math.atan2(row["target_z_m"] - row["z_m"],
                     row["target_x_m"] - row["x_m"])
    body = math.degrees(_wrapped_difference(los, row["body_heading_rad"]))
    velocity = math.degrees(_wrapped_difference(los, row["heading_rad"]))
    return body, velocity


def first_geometry_entry(rows, mode, threshold_deg):
    index = 0 if mode == "body" else 1
    for row in rows:
        if row["time_s"] < CAPTURE_START_S - 1e-10:
            continue
        angles = geometry_angle_pair(row)
        if abs(angles[index]) <= threshold_deg + 1e-12:
            return dict(time_s=float(row["time_s"]),
                        signed_deg=float(angles[index]),
                        body_signed_deg=float(angles[0]),
                        velocity_signed_deg=float(angles[1]))
    return None


def build_geometry_runner(plant, table, mode, threshold_deg):
    """Create one local source variant with a latched geometry exit."""
    source, namespace, pid_hash = export_factory_source(plant, table, 120.0)
    factory_source = source
    original = inspect.getsource(capture.alpha_servo.run_alpha_servo)

    # The existing factory sets this false because its timed branch owns the
    # capture state.  Reintroduce a local latched state before that branch.
    initialization = "    capturing = False\n"
    assert source.count(initialization) == 1
    source = source.replace(initialization, initialization +
        "    capture_started = False\n"
        "    capture_exit_latched = False\n"
        "    capture_exit_time = -1.0\n"
        "    capture_exit_body_signed_rad = 0.0\n"
        "    capture_exit_velocity_signed_rad = 0.0\n", 1)

    timed_state = "            capturing = capture_start - 1e-10 <= time_s < capture_end - 1e-10\n"
    assert source.count(timed_state) == 1
    source = source.replace(timed_state,
        "            capture_started = capture_started or time_s >= capture_start - 1e-10\n"
        "            capturing = (capture_started and not capture_exit_latched\n"
        "                         and time_s < capture_end - 1e-10)\n", 1)

    error_line = (
        "                error = (math.atan2(rz, rx) + lead - heading + math.pi) % (2.0 * math.pi) - math.pi\n"
    )
    if source.count(error_line) != 1:
        # Current source uses ``2 * math.pi`` in this generated branch.
        error_line = (
            "                error = (math.atan2(rz, rx) + lead - heading + math.pi) % (2 * math.pi) - math.pi\n"
        )
    assert source.count(error_line) == 1
    geometry_branch = error_line + (
        "                    los_heading = math.atan2(rz, rx)\n"
        "                    capture_body_signed_rad = _wrapped_difference(los_heading, state[4])\n"
        "                    capture_velocity_signed_rad = _wrapped_difference(los_heading, heading)\n"
        "                    selected_geometry_angle = (capture_body_signed_rad\n"
        "                        if geometry_mode == 'body' else capture_velocity_signed_rad)\n"
        "                    if (capturing and capture_started\n"
        "                            and abs(selected_geometry_angle) <= geometry_threshold_rad):\n"
        "                        capture_exit_latched = True\n"
        "                        capturing = False\n"
        "                        capture_exit_time = time_s\n"
        "                        capture_exit_body_signed_rad = capture_body_signed_rad\n"
        "                        capture_exit_velocity_signed_rad = capture_velocity_signed_rad\n"
    )
    source = source.replace(error_line, geometry_branch, 1)
    capture_block = (
        "                    capture_a = speed * math.sin(error) / p.capture_time_constant_s\n"
        "                    a_cmd = max(-capture_cap_g * G0, min(capture_cap_g * G0, capture_a))\n"
    )
    assert source.count(capture_block) == 1
    source = source.replace(capture_block,
        "                if capturing:\n"
        "                    capture_a = speed * math.sin(error) / p.capture_time_constant_s\n"
        "                    a_cmd = max(-capture_cap_g * G0, min(capture_cap_g * G0, capture_a))\n", 1)

    # Add geometry telemetry to the existing factory row without touching the
    # original record/force/RK4 equations.  Computing these in record() also
    # preserves the signed angles before guidance starts.
    row_anchor = (
        "        row.update(pid_raw_output=pid_proportional+pid_integral+pid_derivative,\n"
        "                   capture_active=bool(capturing))\n"
    )
    assert source.count(row_anchor) == 1
    row_replacement = (
        "        record_los_heading = math.atan2(rel[2], rel[0])\n"
        "        record_body_signed = _wrapped_difference(record_los_heading, s[4])\n"
        "        record_velocity_signed = _wrapped_difference(record_los_heading, s[1])\n"
        "        row.update(pid_raw_output=pid_proportional+pid_integral+pid_derivative,\n"
        "                   capture_active=bool(capturing),\n"
        "                   capture_exit_latched=bool(capture_exit_latched),\n"
        "                   capture_exit_time_s=float(capture_exit_time),\n"
        "                   capture_body_boresight_signed_deg=math.degrees(record_body_signed),\n"
        "                   capture_body_boresight_abs_deg=abs(math.degrees(record_body_signed)),\n"
        "                   capture_velocity_los_signed_deg=math.degrees(record_velocity_signed),\n"
        "                   capture_velocity_los_abs_deg=abs(math.degrees(record_velocity_signed)),\n"
        "                   capture_exit_body_signed_deg=math.degrees(capture_exit_body_signed_rad),\n"
        "                   capture_exit_velocity_signed_deg=math.degrees(capture_exit_velocity_signed_rad))\n"
    )
    source = source.replace(row_anchor, row_replacement, 1)

    # The local changes must leave the plant and raw PID blocks inherited from
    # the factory.  The factory itself already verifies the PID block hash;
    # these additional checks make the boundary explicit in the receipt.
    pid_start = "                # One PID: error in wind-normal g, output in fin-travel"
    pid_end = "            record(time_s, state)"
    assert source[source.index(pid_start):source.index(pid_end)] == original[original.index(pid_start):original.index(pid_end)]
    force_start, force_end = "    def forces(t, s):", "    def record(t, s):"
    assert source[source.index(force_start):source.index(force_end)] == factory_source[factory_source.index(force_start):factory_source.index(force_end)]
    rhs_start, rhs_end = "        def rhs(t, s):", "        def shifted(s, k, h):"
    assert source[source.index(rhs_start):source.index(rhs_end)] == factory_source[factory_source.index(rhs_start):factory_source.index(rhs_end)]

    namespace = dict(namespace)
    namespace.update(
        _wrapped_difference=_wrapped_difference,
        geometry_mode=str(mode),
        geometry_threshold_rad=math.radians(float(threshold_deg)),
    )
    exec(compile(source, "<geometry-capture-diagnostic>", "exec"), namespace)
    return namespace["run_alpha_servo"], pid_hash, dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        factory_source_sha256=hashlib.sha256(factory_source.encode()).hexdigest(),
    )


def interpolate(rows, time_s, field):
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    if time_s < times[0] or time_s > times[-1]:
        return None
    return float(np.interp(time_s, times, [row[field] for row in rows]))


def comparison_rows(case_name, mode, threshold_deg, rows, orange, trk, measured):
    result = []
    for obs in measured:
        item = dict(case=case_name, geometry_mode=mode,
                    threshold_deg=float(threshold_deg), time_s=obs["time_s"])
        for field in ("alpha_deg", "wind_normal_specific_force_g", "speed_kmh"):
            item[field + "_measured"] = obs[field]
            item[field + "_geometry"] = interpolate(rows, obs["time_s"], field)
            item[field + "_orange"] = interpolate(orange, obs["time_s"], field)
            item[field + "_trk"] = interpolate(trk, obs["time_s"], field)
            predicted = item[field + "_geometry"]
            item[field + "_geometry_error"] = (None if predicted is None
                                                else predicted - obs[field])
        result.append(item)
    return result


def _around(rows, index):
    if index is None:
        return []
    return [dict(time_s=float(row["time_s"]),
                 alpha_deg=float(row["alpha_deg"]),
                 load_g=float(row["wind_normal_specific_force_g"]),
                 requested_g=float(row["commanded_horizontal_acceleration_g"]),
                 fin_deg=math.degrees(row["horizontal_fin_deflection_rad"]),
                 capture_active=bool(row["capture_active"]),
                 body_angle_signed_deg=float(row["capture_body_boresight_signed_deg"]),
                 velocity_angle_signed_deg=float(row["capture_velocity_los_signed_deg"]))
            for row in rows[max(0, index - 1):index + 2]]


def summarize_case(case_name, mode, threshold_deg, result, measured, first_trk_s):
    rows = result.samples
    start_index = next((i for i, row in enumerate(rows)
                        if row["time_s"] >= CAPTURE_START_S - 1e-10), None)
    exit_index = next((i for i, row in enumerate(rows)
                       if i >= (start_index or 0)
                       and not bool(row["capture_active"])), None)
    exit_row = rows[exit_index] if exit_index is not None else None
    early = [row for row in rows if 0.30 <= row["time_s"] <= 3.50]
    post_exit = ([row for row in early if row["time_s"] >= exit_row["time_s"]]
                 if exit_row is not None else [])
    samples = {}
    times = np.asarray([.7, .9, 1.1, 1.3, 1.5])
    alpha_at = [interpolate(rows, float(t), "alpha_deg") for t in times]
    for t, value in zip(times, alpha_at):
        samples[f"{t:.1f}"] = value
    topology = (all(value is not None for value in alpha_at)
                and alpha_at[0] < alpha_at[1] > alpha_at[2] < alpha_at[3])
    summary = dict(
        case=case_name, geometry_mode=mode, threshold_deg=float(threshold_deg),
        event=result.event, terminal_time_s=float(rows[-1]["time_s"]),
        minimum_distance_m=float(result.minimum_distance_m),
        capture_exit_time_s=(None if exit_row is None else float(exit_row["time_s"])),
        capture_exit_index=exit_index,
        exit_before_first_trk=(None if exit_row is None else bool(exit_row["time_s"] < first_trk_s - 1e-10)),
        exit_body_signed_deg=(None if exit_row is None else float(exit_row["capture_body_boresight_signed_deg"])),
        exit_body_abs_deg=(None if exit_row is None else float(exit_row["capture_body_boresight_abs_deg"])),
        exit_velocity_signed_deg=(None if exit_row is None else float(exit_row["capture_velocity_los_signed_deg"])),
        exit_velocity_abs_deg=(None if exit_row is None else float(exit_row["capture_velocity_los_abs_deg"])),
        request_before_exit_g=(None if exit_index is None else float(rows[max(0, exit_index - 1)]["commanded_horizontal_acceleration_g"])),
        request_after_exit_g=(None if exit_row is None else float(exit_row["commanded_horizontal_acceleration_g"])),
        fin_before_exit_deg=(None if exit_index is None else math.degrees(rows[max(0, exit_index - 1)]["horizontal_fin_deflection_rad"])),
        fin_after_exit_deg=(None if exit_row is None else math.degrees(exit_row["horizontal_fin_deflection_rad"])),
        alpha_before_exit_deg=(None if exit_index is None else float(rows[max(0, exit_index - 1)]["alpha_deg"])),
        alpha_after_exit_deg=(None if exit_row is None else float(exit_row["alpha_deg"])),
        load_before_exit_g=(None if exit_index is None else float(rows[max(0, exit_index - 1)]["wind_normal_specific_force_g"])),
        load_after_exit_g=(None if exit_row is None else float(exit_row["wind_normal_specific_force_g"])),
        early_peak_alpha_deg=float(max(row["alpha_deg"] for row in early)),
        early_peak_alpha_time_s=float(max(early, key=lambda row: row["alpha_deg"])["time_s"]),
        early_min_alpha_deg=float(min(row["alpha_deg"] for row in early)),
        early_min_alpha_time_s=float(min(early, key=lambda row: row["alpha_deg"])["time_s"]),
        post_exit_min_alpha_deg=(None if not post_exit else float(min(row["alpha_deg"] for row in post_exit))),
        post_exit_min_alpha_time_s=(None if not post_exit else float(min(post_exit, key=lambda row: row["alpha_deg"])["time_s"])),
        alpha_at_native_times_deg=samples,
        early_topology_matches=bool(topology),
        compared_frame_count=sum(obs["time_s"] <= rows[-1]["time_s"] + 1e-10 for obs in measured),
        handover_samples=_around(rows, exit_index),
    )
    return summary


def _plot_mode(cases, mode, measured, orange, trk, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {3.0: "#3978a8", 6.0: "#398b70", 10.0: "#c38f2f", 15.0: "#bd5757"}
    fields = (("alpha_deg", "AoA [deg]"),
              ("wind_normal_specific_force_g", "Normal load [g]"),
              ("speed_kmh", "Speed [km/h]"))
    end = max(case["result"].samples[-1]["time_s"] for case in cases)
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.2), sharex="col")
    for row_index, (field, label) in enumerate(fields):
        for col in range(2):
            ax = axes[row_index, col]
            ax.plot([row["time_s"] for row in measured], [row[field] for row in measured],
                    "o-", color="#252525", ms=3.3, lw=1.0, zorder=7)
            for base_rows, color, style, label_text in (
                (orange, "#c77c27", ":", "Orange baseline: capture to 2.10 s"),
                (trk, "#167f99", "--", "TRK-time baseline: capture to 0.90 s"),
            ):
                ax.plot([row["time_s"] for row in base_rows], [row[field] for row in base_rows],
                        color=color, ls=style, lw=1.7, label=label_text)
            for case in cases:
                rows = case["result"].samples
                threshold = case["threshold_deg"]
                ax.plot([row["time_s"] for row in rows], [row[field] for row in rows],
                        color=colors[threshold], lw=1.55,
                        label=f"{mode} threshold {threshold:g}°")
                if case["summary"]["capture_exit_time_s"] is not None:
                    ax.axvline(case["summary"]["capture_exit_time_s"],
                               color=colors[threshold], lw=.7, alpha=.35)
            ax.axvline(BASELINE_CAPTURE_END_S, color="#c77c27", ls=":", lw=.8, alpha=.8)
            ax.axvline(FIRST_TRK_S, color="#167f99", ls="--", lw=.8, alpha=.8)
            if col == 0:
                ax.set_xlim(0.0, end + .12)
                ax.axvline(7.0, color="#708b9e", ls="--", lw=.7, alpha=.7)
            else:
                ax.set_xlim(.2, 3.5)
            ax.set_ylabel(label)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 2:
                ax.set_xlabel("Missile flight time [s]")
    axes[0, 0].set_title(f"Complete forward runs | H_{mode}", fontsize=12)
    axes[0, 1].set_title(f"Early detail | H_{mode}", fontsize=12)
    legend_handles = [Line2D([], [], color="#252525", marker="o", lw=1., label="Game measurements"),
                      Line2D([], [], color="#c77c27", ls=":", lw=1.7, label="Orange baseline: 2.10 s"),
                      Line2D([], [], color="#167f99", ls="--", lw=1.7, label="TRK-time baseline: 0.90 s")]
    legend_handles.extend(Line2D([], [], color=colors[t], lw=1.55, label=f"{mode} {t:g}°")
                          for t in THRESHOLDS_DEG)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, .952),
               ncol=3, frameon=False, fontsize=9.0)
    fig.suptitle(f"R-77-1 | geometry capture exit diagnostic ({mode}) | frozen 2.57.1.107",
                 fontsize=16, y=.987)
    fig.text(.07, .025,
             "Only initial capture exit changes. Thresholds are diagnostic proxies; no seeker pointing or epsilon state is modeled.\n"
             "All runs use the same 35.25 g capture request, hard PN handover, plant, damping, raw PID and target course 180°.\n"
             "Vertical markers: orange 2.10 s, TRK-time 0.90 s, colored lines: first geometry-threshold entry.",
             fontsize=8.8, color="#444")
    fig.subplots_adjust(left=.07, right=.98, top=.835, bottom=.13, hspace=.20, wspace=.22)
    fig.savefig(prefix.with_suffix(".png"), dpi=170, facecolor="white")
    plt.close(fig)


def _plot_angles(cases_by_mode, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {3.0: "#3978a8", 6.0: "#398b70", 10.0: "#c38f2f", 15.0: "#bd5757"}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True)
    panels = (("capture_body_boresight_signed_deg", "Signed LOS − body heading [deg]", "body"),
              ("capture_body_boresight_abs_deg", "|LOS − body heading| [deg]", "body"),
              ("capture_velocity_los_signed_deg", "Signed LOS − velocity heading [deg]", "velocity"),
              ("capture_velocity_los_abs_deg", "|LOS − velocity heading| [deg]", "velocity"))
    for ax, (field, ylabel, _) in zip(axes.flat, panels):
        for mode, cases in cases_by_mode.items():
            style = "-" if mode == "body" else "--"
            for case in cases:
                rows = case["result"].samples
                threshold = case["threshold_deg"]
                ax.plot([row["time_s"] for row in rows], [row[field] for row in rows],
                        color=colors[threshold], ls=style, lw=1.5, label=f"{mode} {threshold:g}°")
                if case["summary"]["capture_exit_time_s"] is not None:
                    ax.axvline(case["summary"]["capture_exit_time_s"],
                               color=colors[threshold], ls=style, lw=.7, alpha=.35)
        ax.axvline(BASELINE_CAPTURE_END_S, color="#c77c27", ls=":", lw=.8)
        ax.axvline(FIRST_TRK_S, color="#167f99", ls="--", lw=.8)
        if "abs" in field:
            # Each mode's threshold is useful only as a visual reference in
            # its corresponding absolute-angle panel.
            for threshold in THRESHOLDS_DEG:
                ax.axhline(threshold, color=colors[threshold], ls=":", lw=.55, alpha=.35)
        ax.set_xlim(.0, 3.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Missile flight time [s]")
    axes[0, 0].set_title("Body-axis proxy", fontsize=12)
    axes[1, 0].set_title("Velocity-axis proxy", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .952),
               ncol=4, frameon=False, fontsize=8.7)
    fig.suptitle("R-77-1 | signed geometry proxies and latched exit points", fontsize=15, y=.985)
    fig.text(.075, .025,
             "Angles are calculated from the simulated target LOS, body heading and velocity heading.\n"
             "Solid: body-exit cases; dashed: velocity-exit cases. Vertical lines mark each exit; these are not measured seeker tracking errors.",
             fontsize=8.8, color="#444")
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.13, hspace=.27, wspace=.24)
    fig.savefig(prefix.with_suffix(".png"), dpi=170, facecolor="white")
    plt.close(fig)


def main():
    baseline_path = ROOT / "outputs/r77_1_current_full_forward.json"
    baseline_trace_path = ROOT / "outputs/r77_1_current_full_forward_trajectory.tsv"
    trk_path = ROOT / "outputs/r77_1_capture_until_trk.json"
    trk_trace_path = ROOT / "outputs/r77_1_capture_until_trk_trajectory.tsv"
    replay_path = ROOT / "data/replays/r77_1_level_20260824.tsv"
    baseline = json.loads(baseline_path.read_text())
    trk_report = json.loads(trk_path.read_text())
    orange = read_table(baseline_trace_path)
    trk = read_table(trk_trace_path)
    measured, native = read_replay(replay_path)

    extra_paths = [baseline_path, baseline_trace_path, trk_path, trk_trace_path,
                   ROOT / "scripts/run_r77_1_focus_moment.py",
                   ROOT / "scripts/check_r77_1_capture_strategy.py",
                   ROOT / "scripts/run_r77_1_current_full_forward.py",
                   replay_path]
    protected_before = protected_hashes() | {str(path): sha(path) for path in extra_paths}

    config, p, raw_limit = inputs()
    p = replace(p,
        fin_force_area_per_rad=baseline["fixed_parameters"]["fin_force_area_per_rad"],
        guidance_delay_s=baseline["fixed_parameters"]["guidance_delay_s"])
    assert asdict(p) == baseline["fixed_parameters"]
    assert config == baseline["fixed_configuration"]
    assert digest(config) == baseline["configuration_sha256"]
    assert (VERSION, COMMIT) == (baseline["source_version"], baseline["source_commit"])
    assert (VERSION, COMMIT) == ("2.57.1.107", "3f20645862c538b46c94e3f60ee30da9dfacd81a")
    assert len(measured) == 46
    assert abs(float(trk_report["capture_end_s"]) - FIRST_TRK_S) < 1e-10

    table = tuple(tuple(row) for row in baseline["raw_time_to_gain_table"])
    plant = FocusPlant(config, p, "legacy_control_scale", raw_limit)

    # Re-run one old orange case with the unchanged factory.  This catches a
    # stale generated-source assumption before any geometry branch is tested.
    base_source, base_namespace, base_pid_hash = export_factory_source(
        plant, table, baseline["capture_duration_s"])
    base_namespace = dict(base_namespace)
    exec(compile(base_source, "<geometry-baseline-reproduction>", "exec"), base_namespace)
    base_runner = base_namespace["run_alpha_servo"]
    base_result = base_runner(config, p, baseline["scenario"],
                              max_time_s=config["performance"]["lifetime_s"],
                              plant_dt_s=.002, control_dt_s=.02)
    assert base_pid_hash == baseline["pid_block_sha256"]
    assert base_result.event == baseline["event"]
    assert len(base_result.samples) == len(orange)
    base_reproduction = {field: max(abs(row[field] - float(old[field]))
                           for row, old in zip(base_result.samples, orange))
                         for field in FIELDS}
    assert max(base_reproduction.values()) < 1e-9, base_reproduction

    # A second, unmodified reference keeps capture active for the whole free
    # run.  It is the causal reference for thresholds whose first entry is
    # later than the historical orange line's 2.10 s handover.  The orange
    # trace remains the user-facing comparison line; this reference prevents
    # us from mistaking its own PN handover for a geometry effect.
    reference_source, reference_namespace, reference_pid_hash = export_factory_source(
        plant, table, 120.0)
    reference_namespace = dict(reference_namespace)
    exec(compile(reference_source, "<geometry-long-capture-reference>", "exec"),
         reference_namespace)
    reference_runner = reference_namespace["run_alpha_servo"]
    reference_result = reference_runner(
        config, p, baseline["scenario"],
        max_time_s=config["performance"]["lifetime_s"],
        plant_dt_s=.002, control_dt_s=.02)
    assert reference_pid_hash == baseline["pid_block_sha256"]
    assert reference_result.event in ("fuse", "closest_approach", "lifetime")
    reference_rows = reference_result.samples
    assert all(math.isfinite(value) for row in reference_rows
               for value in row.values() if isinstance(value, (float, int)))
    orange_geometry_entries = {
        f"{mode}_{threshold:g}deg": first_geometry_entry(orange, mode, threshold)
        for mode in ("body", "velocity") for threshold in THRESHOLDS_DEG
    }
    reference_geometry_entries = {
        f"{mode}_{threshold:g}deg": first_geometry_entry(reference_rows, mode, threshold)
        for mode in ("body", "velocity") for threshold in THRESHOLDS_DEG
    }
    assert all(entry is not None for entry in orange_geometry_entries.values())
    assert all(entry is not None for entry in reference_geometry_entries.values())

    cases_by_mode = {}
    summaries = {}
    all_comparisons = []
    pid_hashes = set()
    source_receipts = {}
    for mode in ("body", "velocity"):
        cases = []
        for threshold_deg in THRESHOLDS_DEG:
            case_name = f"{mode}_{threshold_deg:g}deg"
            runner, pid_hash, source_info = build_geometry_runner(
                plant, table, mode, threshold_deg)
            result = runner(config, p, baseline["scenario"],
                            max_time_s=config["performance"]["lifetime_s"],
                            plant_dt_s=.002, control_dt_s=.02)
            rows = result.samples
            assert result.event in ("fuse", "closest_approach", "lifetime")
            assert rows and all(row["time_s"] < 120.0 + 1e-9 for row in rows)
            assert all(math.isfinite(value) for row in rows
                       for value in row.values() if isinstance(value, (float, int)))
            assert all(b["time_s"] > a["time_s"] for a, b in zip(rows, rows[1:]))
            assert all(bool(rows[i]["capture_exit_latched"]) <= bool(rows[i + 1]["capture_exit_latched"])
                       for i in range(len(rows) - 1))
            exit_flags = [bool(row["capture_exit_latched"]) for row in rows
                          if row["time_s"] >= CAPTURE_START_S - 1e-10]
            assert exit_flags == sorted(exit_flags)
            assert all(abs(row["horizontal_fin_deflection_rad"]
                           - np.clip(row["pid_raw_output"], -1., 1.) * raw_limit) < 1e-12
                       for row in rows)
            assert all(row["thrust_n"] == 0. for row in rows if row["time_s"] >= 7.)
            assert all(row["rate_damping_moment_nm"] * row["body_rate_rad_s"] <= 1e-8
                       for row in rows)
            summary = summarize_case(case_name, mode, threshold_deg, result,
                                     measured, FIRST_TRK_S)
            exit_time = summary["capture_exit_time_s"]
            assert exit_time is not None
            orange_entry = orange_geometry_entries[case_name]
            reference_entry = reference_geometry_entries[case_name]
            assert abs(exit_time - reference_entry["time_s"]) < 1e-10, (
                case_name, exit_time, reference_entry)

            reference_by_time = {row["time_s"]: row for row in reference_rows}
            orange_by_time = {row["time_s"]: row for row in orange}
            pre_reference = [row for row in rows
                             if row["time_s"] < exit_time - 1e-10]
            reference_pairs = [(row, reference_by_time[row["time_s"]])
                               for row in pre_reference
                               if row["time_s"] in reference_by_time]
            assert len(reference_pairs) == len(pre_reference)
            pre_reference_diff = {
                field: max(abs(float(left[field]) - float(right[field]))
                           for left, right in reference_pairs)
                for field in PRE_SWITCH_FIELDS
            }
            assert max(pre_reference_diff.values()) < 1e-12, (
                case_name, pre_reference_diff)

            # The visible orange line itself hands over at 2.10 s.  Therefore
            # direct equality to it is asserted up to that handover; cases
            # exiting later are explicitly marked as expected post-orange
            # comparisons rather than silently called mismatches.
            orange_limit = min(exit_time, BASELINE_CAPTURE_END_S)
            pre_orange = [row for row in rows
                          if row["time_s"] < orange_limit - 1e-10]
            orange_pairs = [(row, orange_by_time[row["time_s"]])
                            for row in pre_orange
                            if row["time_s"] in orange_by_time]
            assert len(orange_pairs) == len(pre_orange)
            pre_orange_diff = {
                field: max(abs(float(left[field]) - float(right[field]))
                           for left, right in orange_pairs)
                for field in PRE_SWITCH_FIELDS
            }
            assert max(pre_orange_diff.values()) < 1e-12, (
                case_name, pre_orange_diff)

            exit_row = next(row for row in rows
                            if abs(row["time_s"] - exit_time) < 1e-10)
            reference_exit_row = reference_by_time[exit_row["time_s"]]
            exit_state_diff = {
                field: abs(float(exit_row[field]) - float(reference_exit_row[field]))
                for field in STATE_FIELDS
            }
            assert max(exit_state_diff.values()) < 1e-12, (
                case_name, exit_state_diff)
            exit_controller_delta = {
                field: float(exit_row[field]) - float(reference_exit_row[field])
                for field in CONTROLLER_FIELDS
            }
            summary.update(
                orange_geometry_first_entry=orange_entry,
                reference_geometry_first_entry=reference_entry,
                orange_entry_delta_s=float(exit_time - orange_entry["time_s"]),
                orange_entry_after_baseline_handover=bool(
                    orange_entry["time_s"] > BASELINE_CAPTURE_END_S + 1e-10),
                pre_reference_max_difference=pre_reference_diff,
                pre_orange_until_handover_max_difference=pre_orange_diff,
                exit_state_max_difference_to_long_capture_reference=exit_state_diff,
                exit_controller_delta_from_long_capture_reference=exit_controller_delta,
            )
            cases.append(dict(case=case_name, mode=mode, threshold_deg=float(threshold_deg),
                              result=result, summary=summary))
            summaries[case_name] = summary
            all_comparisons.extend(comparison_rows(case_name, mode, threshold_deg,
                                                   rows, orange, trk, measured))
            pid_hashes.add(pid_hash)
            source_receipts[case_name] = source_info
        cases_by_mode[mode] = cases

    assert pid_hashes == {baseline["pid_block_sha256"]}
    assert all(summary["compared_frame_count"] == len(measured)
               for summary in summaries.values())
    assert protected_hashes() | {str(path): sha(path) for path in extra_paths} == protected_before

    summary_rows = list(summaries.values())
    prefix = ROOT / "outputs/r77_1_geometry_capture"
    report = dict(
        source_version=VERSION, source_commit=COMMIT,
        scenario=baseline["scenario"], fixed_parameters=asdict(p),
        fixed_configuration=config, configuration_sha256=digest(config),
        capture_start_s=CAPTURE_START_S, capture_cap_g=CAPTURE_CAP_G,
        baseline_references=dict(
            orange_capture_end_s=BASELINE_CAPTURE_END_S,
            trk_capture_end_s=FIRST_TRK_S,
            first_trk_frame=trk_report["first_trk_frame"],
            orange_terminal_time_s=float(orange[-1]["time_s"]),
            trk_terminal_time_s=float(trk[-1]["time_s"]),
            orange_geometry_first_entries=orange_geometry_entries,
            long_capture_geometry_first_entries=reference_geometry_entries,
        ),
        geometry_definitions=dict(
            body="abs(wrap(atan2(target_z−missile_z,target_x−missile_x)−body_heading))",
            velocity="abs(wrap(atan2(target_z−missile_z,target_x−missile_x)−velocity_heading))",
            angle_units="degrees for thresholds and telemetry; radians inside generated source",
            exit_rule="first entry at or below threshold after t=0.30 s, latched permanently",
            seeker_pointing_modeled=False,
            seeker_tracking_error_modeled=False,
        ),
        thresholds_deg=THRESHOLDS_DEG,
        changed_parameters=["geometry_mode", "geometry_threshold_deg"],
        fitted_parameters=[],
        cases=summaries,
        source_receipts=source_receipts,
        comparisons=all_comparisons,
        native_replay_frame_count=len(measured),
        measured_input_use="Replay fields are overlaid after free integration; no state or future exit time is injected.",
        verification=dict(
            factory_baseline_reproduction_max_difference=base_reproduction,
            pid_block_sha256=next(iter(pid_hashes)),
            pid_block_matches_baseline=True,
            force_block_unchanged=True,
            rhs_block_unchanged=True,
            protected_inputs_unchanged=True,
            no_measured_state_injection=True,
            all_cases_full_forward=True,
            all_cases_finite=True,
            capture_exit_latched_no_reentry=True,
            geometry_exit_matches_long_capture_first_entry=True,
            pre_exit_state_and_controller_match_long_capture_reference=True,
            pre_exit_state_and_controller_match_orange_until_2p10=True,
            orange_post_handover_entry_deltas={
                name: summary["orange_entry_delta_s"]
                for name, summary in summaries.items()
                if summary["orange_entry_after_baseline_handover"]},
            compared_frame_count=len(measured),
        ),
        protected_input_hashes=protected_before,
        runtime_defaults_changed=False,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_table(prefix.with_suffix(".tsv"), summary_rows)
    write_table(prefix.with_name(prefix.name + "_trajectory.tsv"),
                [dict(case=case["case"], geometry_mode=case["mode"],
                      threshold_deg=case["threshold_deg"], **row)
                 for cases in cases_by_mode.values() for case in cases
                 for row in case["result"].samples])
    write_table(prefix.with_name(prefix.name + "_comparison.tsv"), all_comparisons)
    _plot_mode(cases_by_mode["body"], "body", measured, orange, trk,
               prefix.with_name(prefix.name + "_body"))
    _plot_mode(cases_by_mode["velocity"], "velocity", measured, orange, trk,
               prefix.with_name(prefix.name + "_velocity"))
    _plot_angles(cases_by_mode, prefix.with_name(prefix.name + "_angles"))

    print(json.dumps(dict(
        factory_baseline_reproduction=base_reproduction,
        cases={name: {key: summary[key] for key in (
            "capture_exit_time_s", "exit_before_first_trk", "exit_body_abs_deg",
            "exit_velocity_abs_deg", "early_peak_alpha_deg", "early_min_alpha_deg",
            "post_exit_min_alpha_deg", "event", "terminal_time_s")}
                      for name, summary in summaries.items()},
        verification=report["verification"],
    ), indent=2, allow_nan=False))
    print(prefix.with_suffix(".json"))
    print(prefix.with_name(prefix.name + "_body.png"))
    print(prefix.with_name(prefix.name + "_velocity.png"))
    print(prefix.with_name(prefix.name + "_angles.png"))


if __name__ == "__main__":
    main()
