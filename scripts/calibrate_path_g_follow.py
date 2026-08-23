#!/usr/bin/env python3
"""Sweep path-G / cascade knobs against SensorWhale and the 80 deg ranking shot.

This is a local engineering calibration. It does not claim the protected
SensorWhale engine or War Thunder solver has been reproduced.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aim120_model.profile_adapter import (  # noqa: E402
    build_h2_candidate_config,
    load_runtime_defaults,
)
from aim120_model.public_api import simulate  # noqa: E402


REFERENCE_PATH = (
    ROOT
    / "data"
    / "reference_external"
    / "sensorwhale_pl12_off_axis_38_course0_20260818.json"
)
RANKING_IDS = ("cn_pl12", "su_r_77", "jp_aam4", "il_derby")


def _load_defaults() -> dict[str, Any]:
    return copy.deepcopy(
        load_runtime_defaults(str((ROOT / "config" / "profile_h2_runtime_defaults.json").resolve()))
    )


def _apply_knobs(
    defaults: dict[str, Any],
    *,
    cn_alpha: float,
    share: float,
    path_tau: float,
    close_ki: float,
) -> dict[str, Any]:
    patched = copy.deepcopy(defaults)
    body_lift = dict(patched.get("legacy_body_lift") or {})
    body_lift["cn_alpha_per_rad"] = float(cn_alpha)
    body_lift["fin_translation_share"] = float(share)
    patched["legacy_body_lift"] = body_lift
    patched["path_rate_time_constant_s"] = float(path_tau)
    patched["path_close_integral_gain_per_s"] = float(close_ki)
    return patched


def _simulate(missile_id: str, scenario: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads((ROOT / "missiles" / f"{missile_id}.json").read_text(encoding="utf-8"))
    config, _assumptions = build_h2_candidate_config(profile, defaults)
    profile["_model_config"] = config
    profile["_runtime_unsupported"] = []
    return simulate(profile, scenario)


def _follow_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    ratios: list[float] = []
    lags: list[float] = []
    for sample in samples:
        command = sample.get("controller_specific_force_command_g") or (0.0, 0.0)
        cmd = math.hypot(float(command[0]), float(command[1]))
        meas = float(sample.get("actual_overload_g", 0.0))
        fin = math.hypot(
            float(sample.get("pitch_requested_fin_command", 0.0)),
            float(sample.get("yaw_requested_fin_command", 0.0)),
        )
        if cmd < 5.0:
            continue
        lags.append(abs(meas - cmd) / cmd)
        if fin < 0.95:
            ratios.append(meas / cmd)
    def _median(values: list[float]) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    return {
        "unsaturated_follow_ratio": _median(ratios),
        "command_follow_err_frac": _median(lags),
        "unsaturated_samples": float(len(ratios)),
        "commanded_samples": float(len(lags)),
    }


def _aoa_at_peak_g(samples: list[dict[str, Any]]) -> tuple[float, float]:
    peak = max(samples, key=lambda sample: float(sample.get("actual_overload_g", 0.0)))
    return (
        float(peak.get("actual_overload_g", 0.0)),
        math.degrees(float(peak.get("angle_of_attack_rad", 0.0))),
    )


def _heading_at(samples: list[dict[str, Any]], time_s: float) -> float:
    def _heading(sample: dict[str, Any]) -> float:
        velocity = sample["velocity_mps"]
        return math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))

    origin = _heading(samples[0])
    for sample in samples:
        if float(sample["time_s"]) >= time_s:
            return abs(_heading(sample) - origin)
    return abs(_heading(samples[-1]) - origin)


def _off_axis_80() -> dict[str, Any]:
    return {
        "launch_speed_kmh": 1200.0,
        "launch_altitude_m": 6500.0,
        "launch_pitch_deg": 0.0,
        "launch_heading_deg": 0.0,
        "target_speed_kmh": 1200.0,
        "target_altitude_m": 6500.0,
        "initial_distance_m": 8000.0,
        "target_azimuth_deg": 80.0,
        "target_heading_deg": 0.0,
        "target_vertical_heading_deg": 0.0,
        "target_constant_turn_g": 0.0,
        "max_simulation_time_s": 20.0,
        "loft_enabled": False,
    }


def evaluate_knobs(knobs: dict[str, float], reference: dict[str, Any]) -> dict[str, Any]:
    defaults = _apply_knobs(
        _load_defaults(),
        cn_alpha=knobs["cn_alpha"],
        share=knobs["share"],
        path_tau=knobs["path_tau"],
        close_ki=knobs["close_ki"],
    )
    whale_scenario = dict(reference["local_scenario"])
    whale_scenario["max_simulation_time_s"] = 20.0
    whale = _simulate("cn_pl12", whale_scenario, defaults)
    whale_samples = whale["samples"]
    peak_g, aoa_at_peak = _aoa_at_peak_g(whale_samples)
    peak_aoa = max(math.degrees(float(sample["angle_of_attack_rad"])) for sample in whale_samples)
    ranking: dict[str, Any] = {}
    ranking_scenario = _off_axis_80()
    ranking_runs: dict[str, dict[str, Any]] = {}
    for missile_id in RANKING_IDS:
        result = _simulate(missile_id, ranking_scenario, defaults)
        ranking_runs[missile_id] = result
        summary = result["summary"]
        ranking[missile_id] = {
            "event": summary["termination_event"],
            "min_range_m": summary["minimum_distance_m"],
            "peak_g": summary["maximum_actual_g"],
            "max_cmd_g": max(
                math.hypot(*sample["controller_specific_force_command_g"])
                for sample in result["samples"]
            ),
            "heading_1s_deg": _heading_at(result["samples"], 1.0),
        }
    follow = _follow_metrics(ranking_runs["cn_pl12"]["samples"])
    ranking_ok = (
        ranking["su_r_77"]["event"] == "proximity_fuse"
        and ranking["jp_aam4"]["event"] == "proximity_fuse"
        and ranking["il_derby"]["event"] != "proximity_fuse"
        and ranking["cn_pl12"]["event"] != "proximity_fuse"
        and ranking["jp_aam4"]["min_range_m"] < ranking["il_derby"]["min_range_m"]
        and ranking["jp_aam4"]["heading_1s_deg"] > ranking["il_derby"]["heading_1s_deg"]
        and ranking["jp_aam4"]["heading_1s_deg"] > ranking["cn_pl12"]["heading_1s_deg"]
    )
    ref = reference["public_result_summary"]
    peak_g_err = abs(peak_g - float(ref["peak_g"]))
    aoa_peak_err = abs(aoa_at_peak - float(ref["aoa_at_peak_g_deg"]))
    peak_aoa_err = abs(peak_aoa - float(ref["peak_aoa_deg"]))
    follow_ratio = follow["unsaturated_follow_ratio"]
    follow_err = 1.0 if math.isnan(follow_ratio) else abs(1.0 - follow_ratio)
    command_err = follow["command_follow_err_frac"]
    if math.isnan(command_err):
        command_err = 1.0
    whale_miss = whale["summary"]["termination_event"] != "proximity_fuse"
    # SensorWhale's public peak_g=38 is not a PN command this local 38 deg
    # geometry issues, so following is scored on the 80 deg discriminator.
    score = (
        (0.0 if ranking_ok else 100.0)
        + (0.0 if whale_miss else 20.0)
        + 0.02 * peak_g_err
        + 0.04 * aoa_peak_err
        + 0.03 * peak_aoa_err
        + 12.0 * follow_err
        + 6.0 * command_err
    )
    return {
        "knobs": knobs,
        "score": score,
        "ranking_ok": ranking_ok,
        "whale_event": whale["summary"]["termination_event"],
        "whale_peak_g": peak_g,
        "whale_aoa_at_peak_deg": aoa_at_peak,
        "whale_peak_aoa_deg": peak_aoa,
        "whale_peak_g_err": peak_g_err,
        "follow": follow,
        "ranking": ranking,
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    return evaluate_knobs(payload, reference)


def _grid(cn_alphas: list[float], shares: list[float], taus: list[float], kis: list[float]) -> list[dict[str, float]]:
    return [
        {"cn_alpha": cn_alpha, "share": share, "path_tau": path_tau, "close_ki": close_ki}
        for cn_alpha in cn_alphas
        for share in shares
        for path_tau in taus
        for close_ki in kis
    ]


def pl12_trim_share(cn_alpha: float) -> float:
    """finsLatAccel-as-authority start: share*fins_g + body(delta_max) = fins_g at q_base."""

    diameter = 0.203
    wing_mult = 1.4
    area = math.pi * diameter ** 2 / 4.0 * wing_mult
    q_base = 0.5 * 1.225000018 * (1800.0 / 3.6) ** 2
    delta_max = 0.375092
    mass = 198.0
    gravity = 9.80665
    fins_g = 41.4036
    body_g = q_base * area * cn_alpha * delta_max / (mass * gravity)
    return max(0.0, 1.0 - body_g / fins_g)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("baseline", "follow", "plant", "full"),
        default="full",
    )
    args = parser.parse_args()
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    defaults = _load_defaults()
    body_lift = defaults.get("legacy_body_lift") or {}
    baseline = {
        "cn_alpha": float(body_lift.get("cn_alpha_per_rad", 2.0)),
        "share": float(body_lift.get("fin_translation_share", 1.0)),
        "path_tau": float(defaults.get("path_rate_time_constant_s", 0.35)),
        "close_ki": float(defaults.get("path_close_integral_gain_per_s", 0.0)),
    }
    print(
        json.dumps(
            {
                "pl12_trim_share_cn2": pl12_trim_share(2.0),
                "baseline_knobs": baseline,
            },
            indent=2,
        )
    )
    if args.baseline_only or args.stage == "baseline":
        report = evaluate_knobs(baseline, reference)
        print(json.dumps(report, indent=2))
        return 0
    points = [baseline]
    if args.stage in {"follow", "full"}:
        points.extend(
            _grid(
                [baseline["cn_alpha"]],
                [baseline["share"]],
                [0.12, 0.18, 0.25, 0.35],
                [0.0, 1.5, 3.0, 6.0],
            )
        )
    if args.stage in {"plant", "full"}:
        points.extend(
            _grid(
                [1.5, 2.0, 2.5],
                [round(pl12_trim_share(2.0), 3), 0.935, 1.0],
                [0.18, 0.25, 0.35],
                [0.0, 3.0],
            )
        )
    unique: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for point in points:
        key = (point["cn_alpha"], point["share"], point["path_tau"], point["close_ki"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {pool.submit(_worker, point): point for point in unique}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            knobs = report["knobs"]
            print(
                f"score={report['score']:.3f} rank={report['ranking_ok']} "
                f"cn={knobs['cn_alpha']:g} share={knobs['share']:g} "
                f"tau={knobs['path_tau']:g} ki={knobs['close_ki']:g} "
                f"pl12_80g={report['ranking']['cn_pl12']['peak_g']:.1f}/"
                f"{report['ranking']['cn_pl12']['max_cmd_g']:.1f} "
                f"follow={report['follow']['unsaturated_follow_ratio']:.3f} "
                f"whaleG={report['whale_peak_g']:.1f}",
                flush=True,
            )
    reports.sort(key=lambda item: (not item["ranking_ok"], item["score"]))
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "calibrate_path_g_follow.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    winner = reports[0]
    print(json.dumps({"winner": winner, "written": str(out_path), "n": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
