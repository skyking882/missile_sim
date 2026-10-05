"""Conditional replay geometry, without running or fitting the missile plant.

Level flight converts total normal load into horizontal acceleration. The
sign of the turn is assumed positive. Two explicit 0--0.2 s histories replace
the unobserved launch interval; neither uses a simulated pose. Body direction
additionally requires an AoA-plane assumption, which is not measured here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "outputs/r77_1_measured_geometry_diagnostic"
REPLAY = ROOT / "data/replays/r77_1_level_20260824.tsv"
REPORT = ROOT / "outputs/r77_1_current_full_forward.json"
TRACE = ROOT / "outputs/r77_1_current_full_forward_trajectory.tsv"
RAW = ROOT / "data/datamine/aces.vromfs.bin_u/gamedata/weapons/rocketguns/su_r_77_1.blkx"
G0 = 9.80665


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integral(y, t):
    return np.r_[0., np.cumsum(.5 * (y[:-1] + y[1:]) * np.diff(t))]


def geometry(t, speed, chi, x, z, body, scenario, navigation_gain):
    bearing = math.radians(scenario["target_azimuth_deg"])
    course = math.radians(scenario["target_heading_deg"])
    vt = scenario["target_speed_kmh"] / 3.6
    vx, vz = vt * math.cos(course), vt * math.sin(course)
    rx = scenario["initial_distance_m"] * math.cos(bearing) + vx * t - x
    rz = scenario["initial_distance_m"] * math.sin(bearing) + vz * t - z
    distance = np.hypot(rx, rz)
    los = np.arctan2(rz, rx)
    transverse = (rx * vz - rz * vx) / distance
    assert np.all(np.abs(transverse / speed) < 1.)
    reference = los + np.arcsin(transverse / speed)
    rvx, rvz = vx - speed * np.cos(chi), vz - speed * np.sin(chi)
    closing = -(rx * rvx + rz * rvz) / distance
    los_rate = (rx * rvz - rz * rvx) / distance**2
    return dict(los_deg=np.degrees(los), velocity_heading_deg=np.degrees(chi),
                body_heading_deg=np.degrees(body), reference_deg=np.degrees(reference),
                lead_velocity_error_deg=np.degrees(reference - chi),
                lead_body_error_deg=np.degrees(reference - body),
                los_body_error_deg=np.degrees(los - body),
                pn_horizontal_request_g=navigation_gain * np.maximum(closing, 0.) * los_rate / G0,
                target_range_m=distance,
                intercept_transverse_residual_mps=transverse - speed * np.sin(reference - los))


def reconstruct(obs, scenario, navigation_gain, launch, projection, step=.002):
    t = np.linspace(0., 3.5, round(3.5 / step) + 1)
    v = np.interp(t, np.r_[0., obs["time_s"]],
                  np.r_[scenario["launch_speed_kmh"] / 3.6, obs["speed_mps"]])
    n = np.interp(t, obs["time_s"], obs["total_load_g"])
    alpha = np.radians(np.interp(t, obs["time_s"], obs["total_alpha_deg"]))
    assert np.all(n > 1.)
    nh = np.sqrt(n*n - 1.)
    turn_rate = G0 * nh / v
    start = int(round(obs["time_s"][0] / step))
    if launch == "straight":
        # Split at the first observation; no trapezoidal smearing of the jump.
        chi = np.r_[np.zeros(start), integral(turn_rate[start:], t[start:])]
    else:
        assert launch == "first_load_held"
        chi = integral(turn_rate, t)
    x, z = integral(v * np.cos(chi), t), integral(v * np.sin(chi), t)
    if projection == "normal_aligned":
        # Assumes body incidence and measured nongravitational normal force
        # lie in the same plane. No lift, drag or moment coefficient is used.
        ah = np.arctan(np.tan(alpha) * nh / n)
    else:
        assert projection == "all_horizontal"
        ah = alpha.copy()
    out = dict(time_s=t, speed_mps=v, total_load_g=n, horizontal_load_g=nh,
               total_alpha_deg=np.degrees(alpha), horizontal_alpha_deg=np.degrees(ah),
               velocity_turn_rate_deg_s=np.degrees(turn_rate), x_m=x, z_m=z,
               flown_m=integral(v, t))
    out.update(geometry(t, v, chi, x, z, chi + ah, scenario, navigation_gain))
    # Without an incidence-plane assumption, positive horizontal AoA only
    # supplies these much wider geometric bounds. They are not a confidence interval.
    out["unresolved_body_error_lower_deg"] = out["lead_velocity_error_deg"] - out["total_alpha_deg"]
    out["unresolved_body_error_upper_deg"] = out["lead_velocity_error_deg"].copy()
    return {k: value[start:] for k, value in out.items()}


def sample(curve, t):
    return {k: float(np.interp(t, curve["time_s"], v)) for k, v in curve.items()}


def crossing(curve, key):
    y, t = curve[key], curve["time_s"]
    idx = np.flatnonzero((y[:-1] > 0.) & (y[1:] <= 0.))
    return [float(t[i] - y[i] * (t[i+1] - t[i]) / (y[i+1] - y[i])) for i in idx]


def plot(obs, curves, baseline, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True)
    c = curves["first_load_held__normal_aligned"]
    t = c["time_s"]
    orange, blue, purple, dark = "#c77c27", "#237b98", "#8062a7", "#252525"
    ax = axes[0, 0]
    ax.plot(baseline["time_s"], baseline["alpha_deg"], color=orange, lw=2,
            label="Saved forward model")
    ax.plot(obs["time_s"], obs["total_alpha_deg"], "o-", color=dark, ms=4, lw=1.3,
            label="Game: measured total AoA")
    ax.set(title="A  |  Observed AoA and saved model", ylabel="Total AoA [deg]", ylim=(0, 29))
    ax.legend(loc="lower right", frameon=False)

    ax = axes[0, 1]
    for key, color, style, label in [
        ("reference_deg", purple, "--", "Instantaneous constant-speed reference"),
        ("body_heading_deg", orange, "-", "Body heading: conditional estimate"),
        ("velocity_heading_deg", blue, "-", "Velocity heading: reconstructed"),
        ("los_deg", dark, ":", "Target LOS: reconstructed"),
    ]:
        ax.plot(t, c[key], color=color, ls=style, lw=2, label=label)
        values = np.array([curve[key] for curve in curves.values()])
        ax.fill_between(t, values.min(axis=0), values.max(axis=0), color=color, alpha=.1)
    ax.set(title="B  |  Directions from measured motion", ylabel="Angle from launch direction [deg]", ylim=(0, 76))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    ax = axes[1, 0]
    for key, color, style, label in [
        ("lead_velocity_error_deg", blue, "--", "Reference minus velocity heading"),
        ("lead_body_error_deg", orange, "-", "Reference minus body heading"),
        ("los_body_error_deg", dark, ":", "LOS minus body heading"),
    ]:
        values = np.array([curve[key] for curve in curves.values()])
        ax.plot(t, c[key], color=color, ls=style, lw=2, label=label)
        ax.fill_between(t, values.min(axis=0), values.max(axis=0), color=color, alpha=.14)
    ax.axhline(0., color="#999", lw=.8)
    ax.set(title="C  |  Candidate geometry errors", ylabel="Signed angular error [deg]", ylim=(-28, 64))
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 1]
    ax.plot(baseline["time_s"], baseline["commanded_horizontal_acceleration_g"],
            color=orange, ls="--", lw=1.7, label="Saved model request to PID")
    ax.plot(t, c["pn_horizontal_request_g"], color=blue, lw=2,
            label="PN on reconstructed motion; N = 4")
    values = np.array([curve["pn_horizontal_request_g"] for curve in curves.values()])
    ax.fill_between(t, values.min(axis=0), values.max(axis=0), color=blue, alpha=.15)
    ax.plot(obs["time_s"], np.sqrt(obs["total_load_g"]**2-1.), "o-", color=dark, ms=4, lw=1.3,
            label="Game: horizontal load under level flight")
    ax.set(title="D  |  Actual load and candidate requests", ylabel="Horizontal load / request [g]", ylim=(0, 44))
    ax.legend(loc="upper left", frameon=False, fontsize=8.7)

    for ax in axes.flat:
        ax.axvspan(.7, .9, color="#a8bbca", alpha=.19)
        ax.axvline(2.1, color="#777", ls=":", lw=1)
        ax.set_xlim(.2, 3.5)
        ax.set_xticks([.2, .5, 1., 1.5, 2., 2.5, 3., 3.5])
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Reported missile flight time [s]")
    fig.suptitle("R-77-1 | measured-motion geometry | frozen 2.57.1.107", fontsize=17, y=.98)
    fig.legend(handles=[Line2D([], [], color="#a8bbca", lw=7, alpha=.5,
                               label="Last non-TRK to first TRK observation: 0.7–0.9 s"),
                        Line2D([], [], color="#777", ls=":", label="Model handover time: 2.1 s")],
               loc="upper center", bbox_to_anchor=(.5, .944), ncol=2, frameon=False)
    fig.text(.07, .026,
        "19 native observations, 0.2–3.5 s; V, G and AoA interpolated linearly. Level flight and positive turn assumed.\n"
        "Body estimate additionally assumes normal force and incidence align; total AoA treated as horizontal is a sensitivity case.\n"
        "Bands span two assumed 0–0.2 s histories and these two AoA projections; they are not uncertainty/confidence bounds.\n"
        "Target: 8 km, 30° bearing, 1200 km/h, absolute course 180°. No plant/PID run, coefficient fit or time shift.",
        fontsize=9, color="#444")
    fig.subplots_adjust(left=.07, right=.98, top=.865, bottom=.16, hspace=.24, wspace=.20)
    fig.savefig(prefix.with_suffix(".png"), dpi=175, facecolor="white")
    plt.close(fig)


def main():
    report = json.loads(REPORT.read_text())
    paths = [REPLAY, REPORT, TRACE, RAW, Path(__file__)]
    paths += [Path(p) if Path(p).is_absolute() else ROOT/p for p in report["protected_input_hashes"]]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for p, expected in report["protected_input_hashes"].items():
        path = Path(p) if Path(p).is_absolute() else ROOT/p
        assert sha(path) == expected, f"Saved model source changed: {p}"
    assert report["source_version"] == "2.57.1.107"
    assert report["source_commit"] == "3f20645862c538b46c94e3f60ee30da9dfacd81a"
    raw = json.loads(RAW.read_text())["rocket"]["guidance"]["guidanceAutopilot"]
    n_gain = raw["propNavMult"]
    assert n_gain == report["fixed_parameters"]["navigation_gain"] == 4.
    scenario = report["scenario"]
    assert scenario["target_course_reference"] == "absolute_world" and scenario["target_heading_deg"] == 180
    with REPLAY.open() as f:
        native = [r for r in csv.DictReader(f, delimiter="\t") if float(r["missile_flight_time_s"]) <= 3.5]
    assert len(native) == 19
    obs = {key: np.array([float(r[source]) / scale for r in native]) for key, source, scale in (
        ("time_s", "missile_flight_time_s", 1.), ("speed_mps", "speed_kmh", 3.6),
        ("total_load_g", "overload_g", 1.), ("total_alpha_deg", "angle_of_attack_deg", 1.))}
    assert np.all(np.diff(obs["time_s"]) > 0.)
    curves = {launch+"__"+projection: reconstruct(obs, scenario, n_gain, launch, projection)
              for launch in ("first_load_held", "straight")
              for projection in ("normal_aligned", "all_horizontal")}
    with TRACE.open() as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if float(r["time_s"]) <= 3.5 + 1e-9]
    fields = ["time_s", "speed_kmh", "alpha_deg", "heading_rad", "body_heading_rad", "x_m", "z_m",
              "raw_pn_request_g", "commanded_horizontal_acceleration_g"]
    baseline = {k: np.array([float(r[k]) for r in rows]) for k in fields}
    baseline.update(geometry(baseline["time_s"], baseline["speed_kmh"]/3.6, baseline["heading_rad"],
                             baseline["x_m"], baseline["z_m"], baseline["body_heading_rad"], scenario, n_gain))

    # Focused verification: native samples, geometry closure, integration and unchanged inputs.
    checks = {}
    checks["saved_pn_max_residual_g"] = float(np.max(np.abs(baseline["raw_pn_request_g"] - baseline["pn_horizontal_request_g"])))
    assert checks["saved_pn_max_residual_g"] < 1e-10
    fine_errors = []
    for name, curve in curves.items():
        assert all(np.isfinite(v).all() for v in curve.values())
        for i, t in enumerate(obs["time_s"]):
            row = sample(curve, t)
            for key in ("speed_mps", "total_load_g", "total_alpha_deg"):
                assert abs(row[key] - obs[key][i]) < 1e-9
        assert np.max(np.abs(curve["lead_velocity_error_deg"] - curve["lead_body_error_deg"] - curve["horizontal_alpha_deg"])) < 1e-10
        assert np.max(np.abs(curve["intercept_transverse_residual_mps"])) < 1e-9
        launch, projection = name.split("__")
        fine = reconstruct(obs, scenario, n_gain, launch, projection, step=.001)
        for key in ("x_m", "z_m", "velocity_heading_deg", "lead_body_error_deg", "pn_horizontal_request_g"):
            fine_errors.append(float(np.max(np.abs(curve[key] - np.interp(curve["time_s"], fine["time_s"], fine[key])))))
    checks["half_step_max_difference_across_m_deg_g"] = max(fine_errors)
    assert max(fine_errors) < .001
    assert before == {p: sha(ROOT/p) for p in before}
    checks["protected_inputs_unchanged"] = True

    output_rows = []
    for name, curve in curves.items():
        for r in native:
            output_rows.append(dict(reconstruction=name, sample_id=r["sample_id"], seeker_state=r["seeker_state"],
                                    **sample(curve, float(r["missile_flight_time_s"]))))
    with PREFIX.with_suffix(".tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    crossings = {name: {key: crossing(c, key) for key in ("lead_body_error_deg", "los_body_error_deg")}
                 for name, c in curves.items()}
    anchors = {str(t): {name: sample(c, t) for name, c in curves.items()} for t in (1.6, 1.9, 2.1, 2.3, 2.7, 3.1)}
    intervals = []
    c = curves["first_load_held__normal_aligned"]
    for t0, t1 in zip(obs["time_s"][:-1], obs["time_s"][1:]):
        a, b = sample(c, t0), sample(c, t1)
        intervals.append(dict(start_s=float(t0), end_s=float(t1), **{
            key+"_rate_deg_s": (b[key] - a[key])/(t1-t0)
            for key in ("horizontal_alpha_deg", "velocity_heading_deg", "body_heading_deg", "reference_deg")}))
    result = dict(source_version=report["source_version"], source_commit=report["source_commit"],
        scenario=scenario, window_s=[.2, 3.5], native_frames=len(native),
        source_hashes=before, parameters_fitted=[], simulation_performed=False,
        reconstruction={
            "heading": "integral(g*sqrt(n_total^2-1)/V), assuming level positive turn",
            "position": "integral(V*cos(chi), V*sin(chi)), initial missile position (0,0)",
            "launch_sensitivity": "0--0.2 s: straight or first measured total load held backward; V linear from launch speed",
            "body": "theta=chi+atan(tan(alpha_total)*sqrt(n_total^2-1)/n_total); incidence/normal-force alignment assumed",
            "body_sensitivity": "alpha_h=alpha_total, a planar limiting approximation",
            "body_identifiability": "Without incidence-plane assumption, alpha_h is only in [0,alpha_total] under positive incidence",
            "reference": "Current model's instantaneous constant-speed lead direction; not measured game guidance",
            "interpolation": "linear on reported missile timestamps; no fitted timing or smoothing",
            "band_meaning": "explicit launch/projection sensitivity cases, not probability or complete uncertainty bounds",
            "uncovered_uncertainty": ["rounded timestamps", "load/alpha display rounding", "vertical motion", "turn/incidence sign", "target state uncertainty", "unknown initial maneuver"]},
        first_trk_bracket_reported_s=[.7, .9],
        observed_aoa_plateau_reported_s=[1.5, 1.6], observed_early_total_load_peak_reported_s=2.1,
        zero_crossings_s=crossings, anchors=anchors, native_interval_rates=intervals,
        saved_model_at_2p1=sample(baseline, 2.1), verification=checks)
    PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    plot(obs, curves, baseline, PREFIX)
    print(json.dumps(dict(zero_crossings_s=crossings, verification=checks,
                         at_2p1={k: {f: v[f] for f in ("lead_body_error_deg", "lead_velocity_error_deg", "los_body_error_deg", "pn_horizontal_request_g")}
                                 for k, v in anchors["2.1"].items()}), indent=2))
    print(PREFIX.with_suffix(".png"))


if __name__ == "__main__":
    main()
