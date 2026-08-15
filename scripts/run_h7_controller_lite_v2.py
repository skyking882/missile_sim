"""Run the one-shot offline Plan 7 Lite v2 controller comparison.

This script reads existing H7 request/response artifacts only.  It never opens
StatShark, submits Calculate, edits raw evidence, or refits the H6R plant.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAR_THUNDER_ROOT = PROJECT_ROOT.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "statshark_h7_controller_id"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h7_controller_lite_v2"
PLAN_PATH = WAR_THUNDER_ROOT / "plan7_lite.md"
LEDGER_PATH = RAW_ROOT / "calculate_ledger.json"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "h7_controller_lite"
LEGACY_FITS_PATH = LEGACY_DIR / "candidate_fits.json"
LEGACY_REPORT_PATH = LEGACY_DIR / "H7_CONTROLLER_LITE_REPORT.md"
H6R_REPORT_PATH = PROJECT_ROOT / "outputs" / "h6_fin_dynamics_recovery" / "H6R_FIN_DYNAMICS_REPORT.md"
H6R_PLANT_PATH = PROJECT_ROOT / "outputs" / "h6_fin_dynamics_recovery" / "effective_yaw_plant_fit.json"
H6R_FIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "fit_h6r_effective_yaw_plant.py"


CASE_SPECS = (
    {
        "case_id": "H7_PID_NOM_F010",
        "role": "fit_A",
        "response": "H7_R1_PID_DIFFERENTIAL_retry01.response.json",
        "request": "H7_R1_PID_DIFFERENTIAL_retry01.request.json",
        "result_index": 0,
    },
    {
        "case_id": "H7_DYN_NOM_F100",
        "role": "fit_B",
        "response": "H7_R3_DYNAMIC_REVERSAL.response.json",
        "request": "H7_R3_DYNAMIC_REVERSAL.request.json",
        "result_index": 2,
    },
    {
        "case_id": "H7_LIM_NOM_F003",
        "role": "validation",
        "response": "H7_R2_LIMIT_SCHEDULE_retry01.response.json",
        "request": "H7_R2_LIMIT_SCHEDULE_retry01.request.json",
        "result_index": 0,
    },
    {
        "case_id": "H7_R3N_NOM_F0020",
        "role": "stress_low_authority",
        "response": "H7_R3N_NEAR_ENVELOPE_RETRY.response.json",
        "request": "H7_R3N_NEAR_ENVELOPE_RETRY.request.json",
        "result_index": 0,
    },
    {
        "case_id": "H7_R3N_NOM_F0025",
        "role": "stress_low_authority",
        "response": "H7_R3N_NEAR_ENVELOPE_RETRY.response.json",
        "request": "H7_R3N_NEAR_ENVELOPE_RETRY.request.json",
        "result_index": 1,
    },
)

MODEL_ORDER = ("L1_frozen_signed_lag", "E0_static_envelope", "E1_magnitude_envelope_lag")
FIT_MODELS = ("E0_static_envelope", "E1_magnitude_envelope_lag")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().relative_to(WAR_THUNDER_ROOT.resolve()).as_posix()


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_array(values) -> np.ndarray:
    return np.asarray([float(value) if finite_number(value) else np.nan for value in values], dtype=float)


def parse_request_body(request_outer):
    post_data = request_outer.get("postData")
    if isinstance(post_data, str):
        return json.loads(post_data)
    if isinstance(post_data, dict):
        return post_data
    raise ValueError("request artifact has no parseable postData")


def request_arm(body, missile_id):
    for custom in body.get("CustomMissiles", []):
        if custom.get("Id") == missile_id:
            parameters = custom.get("Parameters", {})
            pids = parameters.get("pids") or []
            pid = pids[0] if pids else {}
            authority = parameters.get("finsLatAccel")
            if not finite_number(authority) or float(authority) <= 0.0:
                raise ValueError("finsLatAccel must be a positive finite request-side authority")
            return {
                "finsLatAccel": float(authority),
                "p": pid.get("p"),
                "i": pid.get("i"),
                "d": pid.get("d"),
                "intgLim": pid.get("intgLim"),
            }
    raise ValueError("missile ID not found in request CustomMissiles")


def crossing_mask(times: np.ndarray, command: np.ndarray):
    crossing_times = []
    for index in range(1, len(command)):
        if command[index - 1] * command[index] < 0.0:
            crossing_times.append(float(0.5 * (times[index - 1] + times[index])))
    near = np.zeros(len(times), dtype=bool)
    for crossing_time in crossing_times:
        near |= np.abs(times - crossing_time) <= 0.10
    return crossing_times, near


def load_case(spec):
    response_path = RAW_ROOT / "responses" / spec["response"]
    request_path = RAW_ROOT / "requests" / spec["request"]
    response = read_json(response_path)
    request_body = parse_request_body(read_json(request_path))

    result_index = int(spec["result_index"])
    response_ids = list(response.get("missileIds", []))
    results = response.get("results", [])
    request_ids = list(request_body.get("Missiles", []))
    if result_index >= len(results) or result_index >= len(response_ids) or result_index >= len(request_ids):
        raise ValueError("result index is outside request/response arrays")
    missile_id = response_ids[result_index]
    if request_ids[result_index] != missile_id:
        raise ValueError("request/response missileIds are not same-index aligned")

    result = results[result_index]
    raw_times = numeric_array(result.get("times", []))
    raw_command = numeric_array(result.get("aCmdYaw", []))
    raw_current_g = numeric_array(result.get("currentG", []))
    raw_g_load = numeric_array(result.get("gLoad", []))
    raw_current_gain = numeric_array(result.get("currentGain", []))
    if not (len(raw_times) == len(raw_command) == len(raw_current_g) == len(raw_g_load)):
        raise ValueError("times/aCmdYaw/currentG/gLoad lengths differ")

    finite_active = np.isfinite(raw_times) & np.isfinite(raw_command) & np.isfinite(raw_current_g)
    active_indices = np.flatnonzero(finite_active & (np.abs(raw_command) > 1.0e-12))
    if len(active_indices) == 0:
        raise ValueError("no finite active guidance samples")
    active_start = int(active_indices[0])
    active_end = int(active_indices[-1])
    if not bool(np.all(finite_active[active_start : active_end + 1])):
        raise ValueError("active guidance samples are not contiguous")

    prefix_coverage = len(raw_current_gain) >= len(raw_times)
    if prefix_coverage:
        current_gain_prefix = raw_current_gain[: len(raw_times)]
        active_current_gain = current_gain_prefix[active_start : active_end + 1]
    else:
        active_current_gain = np.asarray([], dtype=float)
    finite_active_gain = active_current_gain[np.isfinite(active_current_gain)]

    times = raw_times[active_start : active_end + 1]
    command = raw_command[active_start : active_end + 1]
    current_g = raw_current_g[active_start : active_end + 1]
    g_load = raw_g_load[active_start : active_end + 1]
    signed_output = np.sign(command) * current_g
    crossing_times, near_crossing = crossing_mask(times, command)
    fit_mask = (
        (np.abs(command) >= 0.25)
        & (~near_crossing)
        & np.isfinite(signed_output)
        & np.isfinite(g_load)
    )
    if not bool(np.any(fit_mask)):
        raise ValueError("preprocessing removed every sample")
    first_included = int(np.flatnonzero(fit_mask)[0])
    dt = np.diff(times)

    return {
        "case_id": spec["case_id"],
        "role": spec["role"],
        "response_path": response_path,
        "request_path": request_path,
        "missile_id": missile_id,
        "arm": request_arm(request_body, missile_id),
        "times": times,
        "command": command,
        "current_g": current_g,
        "g_load": g_load,
        "signed_output": signed_output,
        "fit_mask": fit_mask,
        "crossing_times": crossing_times,
        "first_included": first_included,
        "initial_signed_output": float(signed_output[first_included]),
        "initial_magnitude": float(current_g[first_included]),
        "dt_median_s": float(np.median(dt)),
        "raw_audit": {
            "raw_times_length": int(len(raw_times)),
            "raw_currentGain_length": int(len(raw_current_gain)),
            "raw_lengths_exact_match": bool(len(raw_current_gain) == len(raw_times)),
            "currentGain_prefix_covers_times": bool(prefix_coverage),
            "prefix_alignment_policy": "same raw indices only; no interpolation; audit-only",
            "active_currentGain_finite_count": int(len(finite_active_gain)),
            "active_currentGain_unique": [float(value) for value in np.unique(finite_active_gain)],
            "active_start_index_raw": active_start,
            "active_end_index_raw": active_end,
        },
    }


def simulate_legacy_signed_lag(case, parameters):
    gain, tau = [float(value) for value in parameters]
    times = case["times"]
    command = case["command"]
    prediction = np.empty(len(times), dtype=float)
    prediction[0] = case["initial_signed_output"]
    for index in range(1, len(times)):
        dt = max(float(times[index] - times[index - 1]), 0.0)
        alpha = math.exp(-dt / tau)
        prediction[index] = alpha * prediction[index - 1] + gain * (1.0 - alpha) * command[index - 1]
    return prediction


def envelope_target(case, gain, fraction, authority_source="request"):
    command_magnitude = np.abs(case["command"])
    if authority_source == "request":
        authority = np.full(len(command_magnitude), case["arm"]["finsLatAccel"], dtype=float)
    elif authority_source == "backend_gLoad":
        authority = np.abs(case["g_load"])
    else:
        raise ValueError("unsupported authority source")
    return np.minimum(gain * command_magnitude, fraction * authority)


def simulate_static_envelope(case, parameters, authority_source="request"):
    gain, fraction = [float(value) for value in parameters]
    magnitude = envelope_target(case, gain, fraction, authority_source=authority_source)
    return np.sign(case["command"]) * magnitude


def simulate_magnitude_envelope_lag(case, parameters):
    gain, fraction, tau = [float(value) for value in parameters]
    times = case["times"]
    target = envelope_target(case, gain, fraction, authority_source="request")
    magnitude = np.empty(len(times), dtype=float)
    magnitude[0] = case["initial_magnitude"]
    for index in range(1, len(times)):
        dt = max(float(times[index] - times[index - 1]), 0.0)
        alpha = math.exp(-dt / tau)
        magnitude[index] = alpha * magnitude[index - 1] + (1.0 - alpha) * target[index - 1]
    return np.sign(case["command"]) * magnitude


def simulate(model, case, parameters):
    if model == "L1_frozen_signed_lag":
        return simulate_legacy_signed_lag(case, parameters)
    if model == "E0_static_envelope":
        return simulate_static_envelope(case, parameters)
    if model == "E1_magnitude_envelope_lag":
        return simulate_magnitude_envelope_lag(case, parameters)
    raise ValueError("unsupported model")


def fit_spec(model):
    if model == "E0_static_envelope":
        return {
            "parameter_names": ["K", "rho"],
            "lower": np.asarray([1.0e-4, 1.0e-4]),
            "upper": np.asarray([100.0, 10.0]),
            "seeds": [[0.1, 0.5], [0.5, 0.7], [1.0, 0.7], [1.0, 1.0]],
        }
    if model == "E1_magnitude_envelope_lag":
        return {
            "parameter_names": ["K", "rho", "tau_s"],
            "lower": np.asarray([1.0e-4, 1.0e-4, 1.0e-3]),
            "upper": np.asarray([100.0, 10.0, 200.0]),
            "seeds": [
                [0.5, 0.7, 0.02],
                [0.5, 0.7, 0.2],
                [0.5, 0.7, 1.0],
                [1.0, 1.0, 5.0],
            ],
        }
    raise ValueError("unsupported fitted model")


def equal_case_residual(log_parameters, model, cases):
    parameters = np.exp(np.asarray(log_parameters, dtype=float))
    residuals = []
    for case in cases:
        mask = case["fit_mask"]
        residual = simulate(model, case, parameters)[mask] - case["signed_output"][mask]
        residuals.append(residual / math.sqrt(len(residual)))
    joined = np.concatenate(residuals)
    if not bool(np.all(np.isfinite(joined))):
        return np.full(1, 1.0e6, dtype=float)
    return joined


def fit_model(model, cases):
    spec = fit_spec(model)
    starts = []
    best = None
    for seed in spec["seeds"]:
        seed_array = np.asarray(seed, dtype=float)
        result = least_squares(
            lambda log_parameters: equal_case_residual(log_parameters, model, cases),
            np.log(seed_array),
            bounds=(np.log(spec["lower"]), np.log(spec["upper"])),
            method="trf",
            loss="linear",
            max_nfev=400,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        parameters = np.exp(result.x)
        case_rmses = []
        for case in cases:
            mask = case["fit_mask"]
            residual = simulate(model, case, parameters)[mask] - case["signed_output"][mask]
            case_rmses.append(float(np.sqrt(np.mean(np.square(residual)))))
        objective = float(np.mean(np.square(case_rmses)))
        record = {
            "initial_parameters": [float(value) for value in seed_array],
            "final_parameters": [float(value) for value in parameters],
            "equal_case_mean_squared_rmse": objective,
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "optimality": float(result.optimality),
        }
        starts.append(record)
        if best is None or objective < best["equal_case_mean_squared_rmse"]:
            best = dict(record)
    best.update(
        {
            "parameter_names": spec["parameter_names"],
            "fit_case_ids": [case["case_id"] for case in cases],
            "loss_weighting": "each fit case contributes its mean squared residual equally",
            "multi_start_count": len(starts),
            "starts": starts,
        }
    )
    return best


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def evaluate(model, case, parameters):
    prediction = simulate(model, case, parameters)
    mask = case["fit_mask"] & np.isfinite(prediction)
    residual = prediction[mask] - case["signed_output"][mask]
    observed = case["signed_output"][mask]
    observed_peak = float(np.max(np.abs(observed)))
    positive = mask & (case["command"] > 0.0)
    negative = mask & (case["command"] < 0.0)
    return {
        "rmse_g": rmse(residual),
        "mae_g": float(np.mean(np.abs(residual))),
        "rmse_over_observed_peak": rmse(residual) / observed_peak if observed_peak > 0.0 else None,
        "observed_peak_g": observed_peak,
        "predicted_peak_g": float(np.max(np.abs(prediction[mask]))),
        "positive_window_rmse_g": rmse(prediction[positive] - case["signed_output"][positive]),
        "negative_window_rmse_g": rmse(prediction[negative] - case["signed_output"][negative]),
    }


def lineage_audit(cases, e0_parameters):
    report_text = H6R_REPORT_PATH.read_text(encoding="utf-8")
    fit_script_text = H6R_FIT_SCRIPT_PATH.read_text(encoding="utf-8")
    case_audits = []
    proxy_differences = []
    for case in cases:
        mask = case["fit_mask"] & np.isfinite(case["g_load"]) & (np.abs(case["g_load"]) > 0.0)
        request_prediction = simulate_static_envelope(case, e0_parameters, authority_source="request")
        backend_prediction = simulate_static_envelope(case, e0_parameters, authority_source="backend_gLoad")
        difference = request_prediction[mask] - backend_prediction[mask]
        difference_rmse = rmse(difference)
        proxy_differences.append(difference_rmse)
        authority = case["arm"]["finsLatAccel"]
        equality_fraction = float(np.mean(np.isclose(np.abs(case["g_load"][mask]), authority, rtol=0.0, atol=1.0e-12)))
        case_audits.append(
            {
                "case_id": case["case_id"],
                "request_finsLatAccel_g": authority,
                "positive_backend_gLoad_samples": int(np.sum(mask)),
                "excluded_nonpositive_backend_gLoad_samples": int(np.sum(case["fit_mask"] & (np.abs(case["g_load"]) <= 0.0))),
                "gLoad_equals_request_authority_fraction": equality_fraction,
                "request_vs_backend_envelope_prediction_rmse_g": difference_rmse,
            }
        )
    max_difference = max(value for value in proxy_differences if value is not None)
    return {
        "backend_gLoad_policy": "diagnostic only; not used by selected candidate",
        "local_h6r_boundary": "signed currentG is an input to the frozen moment-only plant",
        "local_h6r_synthesizes_backend_gLoad": False,
        "independent_model_authority": "request Parameters.finsLatAccel",
        "h6r_report_cap_semantics_present": "effective available-G/control-authority cap" in report_text,
        "h6r_report_controller_handoff_present": "including saturation by `gLoad`" in report_text,
        "h6r_fit_consumes_currentG": "current_g_reported" in fit_script_text and "u_eff_g" in fit_script_text,
        "h6r_fit_uses_available_g_as_plant_input": "available_g_reported" in fit_script_text,
        "proxy_equivalence_threshold_rmse_g": 0.01,
        "max_request_vs_backend_envelope_prediction_rmse_g": max_difference,
        "proxy_equivalence_pass": bool(max_difference <= 0.01),
        "case_audits": case_audits,
    }


def make_report(result):
    fits = result["fits"]
    comparison = result["comparison"]
    selection = result["selection"]
    lineage = result["lineage_audit"]
    lines = [
        "# H7 Controller Lite v2",
        "",
        "## Result",
        "",
        f"- Selected effective candidate: **{selection['selected_candidate'] or 'none'}**.",
        f"- Authority proxy gate: **{'PASS' if lineage['proxy_equivalence_pass'] else 'FAIL'}**.",
        f"- Dynamic lag retained: **{'yes' if selection['dynamic_retained'] else 'no'}**.",
        "- Calculate actions in this stage: **0**.",
        "- Existing raw evidence, old H7 outputs, and the frozen H6R plant were not modified.",
        "",
        "The selected boundary uses request-side `finsLatAccel` as an independently known case authority. Backend `gLoad` is used only to audit that proxy and is not a model input.",
        "",
        "## Parameters",
        "",
        "| model | parameters |",
        "|---|---|",
    ]
    for model in MODEL_ORDER:
        entry = fits[model]
        parameters = ", ".join(
            f"{name}={value:.9g}" for name, value in zip(entry["parameter_names"], entry["parameters"])
        )
        lines.append(f"| {model} | {parameters} |")
    lines.extend(
        [
            "",
            "## Frozen comparison",
            "",
            "| case | role | model | RMSE (g) | RMSE / peak | + window | - window |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for case in comparison["cases"]:
        for model in MODEL_ORDER:
            metrics = case["models"][model]
            positive = "n/a" if metrics["positive_window_rmse_g"] is None else f"{metrics['positive_window_rmse_g']:.6f}"
            negative = "n/a" if metrics["negative_window_rmse_g"] is None else f"{metrics['negative_window_rmse_g']:.6f}"
            lines.append(
                f"| {case['case_id']} | {case['role']} | {model} | {metrics['rmse_g']:.6f} | "
                f"{metrics['rmse_over_observed_peak']:.6f} | {positive} | {negative} |"
            )
    dynamic = selection["dynamic_gate"]
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"- E0 beats frozen L1 on every validation/stress case: **{selection['e0_beats_legacy_all_heldout']}**.",
            f"- E0 absolute held-out precision gate: **{selection['e0_absolute_precision_pass']}**.",
            f"- E1 mean held-out improvement over E0: **{dynamic['mean_heldout_improvement_percent']:.3f}%**.",
            f"- E1 no-case >10% degradation: **{dynamic['no_case_degrades_over_10pct']}**.",
            f"- E1 tau is at least the median sample interval: **{dynamic['tau_resolved_above_dt']}**.",
            "",
            "E1 is rejected when its fitted time constant falls below the sample interval or when it does not improve the frozen held-out cases. No second-order model is attempted.",
            "",
            "## Authority lineage",
            "",
            "- The frozen H6R moment-only plant consumes signed `currentG`; it does not currently generate the server `gLoad` trajectory.",
            "- `finsLatAccel` is request-side and independently known. In authority-limited cases it agrees with positive backend `gLoad`; in the high-authority R3 case the command branch remains below the envelope.",
            f"- Maximum request-authority versus backend-envelope diagnostic prediction RMSE: `{lineage['max_request_vs_backend_envelope_prediction_rmse_g']:.9g} g`.",
            "- This validates the proxy only on the five existing cases. It does not identify the server's internal saturation mechanism.",
            "",
            "## currentGain audit",
            "",
            "The raw prefix is aligned before active-window cropping. It is audited but never interpolated or fitted.",
            "",
            "| case | times | currentGain | exact length | prefix covers times | active values |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for case in comparison["cases"]:
        audit = case["currentGain_audit"]
        values = ", ".join(f"{value:g}" for value in audit["active_currentGain_unique"])
        lines.append(
            f"| {case['case_id']} | {audit['raw_times_length']} | {audit['raw_currentGain_length']} | "
            f"{audit['raw_lengths_exact_match']} | {audit['currentGain_prefix_covers_times']} | {values} |"
        )
    lines.extend(
        [
            "",
            "## Stop condition",
            "",
            "The one-shot offline comparison is complete. No new holdout, Calculate, capture, plant refit, authority sweep, R3B, or R4 was run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError("refusing to overwrite existing h7_controller_lite_v2 outputs")

    protected_paths = [LEDGER_PATH, LEGACY_FITS_PATH, LEGACY_REPORT_PATH, H6R_PLANT_PATH]
    protected_before = {relative_path(path): sha256_file(path) for path in protected_paths}
    ledger = read_json(LEDGER_PATH)
    if int(ledger.get("calculate_actions_used", -1)) != 6 or int(ledger.get("server_calculate_actions_used", -1)) != 6:
        raise ValueError("unexpected Calculate ledger count")
    if int(ledger.get("current_authorized_server_actions", -1)) != 0:
        raise ValueError("server actions must remain unauthorized for this offline workflow")

    cases = [load_case(spec) for spec in CASE_SPECS]
    fit_cases = [case for case in cases if case["role"].startswith("fit_")]
    legacy = read_json(LEGACY_FITS_PATH)["fits"]["M1"]
    legacy_parameters = [float(value) for value in legacy["parameters"]]
    if len(legacy_parameters) != 2:
        raise ValueError("legacy M1 parameter shape changed")

    fitted = {model: fit_model(model, fit_cases) for model in FIT_MODELS}
    fits = {
        "L1_frozen_signed_lag": {
            "parameter_names": ["K", "tau_s"],
            "parameters": legacy_parameters,
            "source": relative_path(LEGACY_FITS_PATH),
            "refit": False,
        }
    }
    for model in FIT_MODELS:
        fits[model] = {
            "parameter_names": fitted[model]["parameter_names"],
            "parameters": fitted[model]["final_parameters"],
            "fit": fitted[model],
            "refit": True,
        }

    comparison_cases = []
    for case in cases:
        model_metrics = {
            model: evaluate(model, case, fits[model]["parameters"])
            for model in MODEL_ORDER
        }
        comparison_cases.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "included_samples": int(np.sum(case["fit_mask"])),
                "duration_s": float(case["times"][-1] - case["times"][0]),
                "dt_median_s": case["dt_median_s"],
                "request_authority_finsLatAccel_g": case["arm"]["finsLatAccel"],
                "currentGain_audit": case["raw_audit"],
                "models": model_metrics,
            }
        )

    lineage = lineage_audit(cases, fits["E0_static_envelope"]["parameters"])
    heldout = [case for case in comparison_cases if case["role"] in ("validation", "stress_low_authority")]
    e0_beats_legacy = all(
        case["models"]["E0_static_envelope"]["rmse_g"]
        < case["models"]["L1_frozen_signed_lag"]["rmse_g"]
        for case in heldout
    )
    e0_precision = all(
        case["models"]["E0_static_envelope"]["rmse_over_observed_peak"] <= 0.25
        for case in heldout
    )
    e0_mean = float(np.mean([case["models"]["E0_static_envelope"]["rmse_g"] for case in heldout]))
    e1_mean = float(np.mean([case["models"]["E1_magnitude_envelope_lag"]["rmse_g"] for case in heldout]))
    e1_improvement = (1.0 - e1_mean / e0_mean) * 100.0
    e1_non_degradation = all(
        case["models"]["E1_magnitude_envelope_lag"]["rmse_g"]
        <= 1.10 * case["models"]["E0_static_envelope"]["rmse_g"]
        for case in heldout
    )
    fitted_tau = float(fits["E1_magnitude_envelope_lag"]["parameters"][2])
    median_dt = float(np.median([case["dt_median_s"] for case in cases]))
    tau_resolved = fitted_tau >= median_dt
    dynamic_retained = bool(e1_improvement >= 20.0 and e1_non_degradation and tau_resolved)
    if not lineage["proxy_equivalence_pass"] or not e0_beats_legacy or not e0_precision:
        selected = None
    elif dynamic_retained:
        selected = "E1_magnitude_envelope_lag"
    else:
        selected = "E0_static_envelope"

    source_files = {}
    for case in cases:
        for path in (case["request_path"], case["response_path"]):
            source_files[relative_path(path)] = sha256_file(path)
    source_files.update(
        {
            relative_path(PLAN_PATH): sha256_file(PLAN_PATH),
            relative_path(LEDGER_PATH): sha256_file(LEDGER_PATH),
            relative_path(LEGACY_FITS_PATH): sha256_file(LEGACY_FITS_PATH),
            relative_path(H6R_REPORT_PATH): sha256_file(H6R_REPORT_PATH),
            relative_path(H6R_PLANT_PATH): sha256_file(H6R_PLANT_PATH),
        }
    )

    result = {
        "schema_version": 1,
        "artifact": "H7_CONTROLLER_LITE_V2_ONE_SHOT_OFFLINE_COMPARISON",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "offline_once": True,
            "statshark_calculate_actions_this_stage": 0,
            "new_capture": False,
            "new_end_to_end_holdout": False,
            "plant_refit": False,
        },
        "source_files": source_files,
        "protected_hashes_before": protected_before,
        "lineage_audit": lineage,
        "fits": fits,
        "comparison": {"cases": comparison_cases},
        "selection": {
            "selected_candidate": selected,
            "e0_beats_legacy_all_heldout": e0_beats_legacy,
            "e0_absolute_precision_pass": e0_precision,
            "dynamic_retained": dynamic_retained,
            "dynamic_gate": {
                "mean_heldout_improvement_percent": e1_improvement,
                "required_improvement_percent": 20.0,
                "no_case_degrades_over_10pct": e1_non_degradation,
                "fitted_tau_s": fitted_tau,
                "median_dt_s": median_dt,
                "tau_resolved_above_dt": tau_resolved,
            },
            "stop": True,
        },
        "ledger": {
            "calculate_actions_before": int(ledger["calculate_actions_used"]),
            "server_calculate_actions_before": int(ledger["server_calculate_actions_used"]),
            "current_authorized_server_actions": int(ledger["current_authorized_server_actions"]),
        },
    }

    protected_after = {relative_path(path): sha256_file(path) for path in protected_paths}
    if protected_after != protected_before:
        raise RuntimeError("a protected ledger, legacy output, or H6R plant changed during the offline workflow")
    result["protected_hashes_after"] = protected_after
    result["protected_hashes_unchanged"] = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "fit_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison_artifact = {
        "schema_version": 1,
        "artifact": "H7_CONTROLLER_LITE_V2_MODEL_COMPARISON",
        "generated_at_utc": result["generated_at_utc"],
        "cases": comparison_cases,
        "selection": result["selection"],
        "lineage_audit": lineage,
    }
    (OUTPUT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison_artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "H7_CONTROLLER_LITE_V2_REPORT.md").write_text(make_report(result), encoding="utf-8")

    print(json.dumps({
        "selected_candidate": selected,
        "calculate_actions_this_stage": 0,
        "calculate_ledger_total": ledger["calculate_actions_used"],
        "proxy_equivalence_pass": lineage["proxy_equivalence_pass"],
        "dynamic_retained": dynamic_retained,
        "output_dir": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
