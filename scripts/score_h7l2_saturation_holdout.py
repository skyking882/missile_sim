from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path


CASE_ID = "H7L2_SATURATION_ENVELOPE_HOLDOUT_01"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "statshark_h7_controller_lite_v2_saturation_holdout"
H7_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "statshark_h7_controller_id"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "h7_controller_lite_v2_saturation_holdout"
PREDICTION_PATH = OUTPUT_ROOT / "H7L2_SATURATION_HOLDOUT_PREDICTION.json"
REQUEST_PATH = RAW_ROOT / "requests" / (CASE_ID + ".request.json")
RESPONSE_PATH = RAW_ROOT / "responses" / (CASE_ID + ".response.json")
SCHEMA_PATH = RAW_ROOT / "network_evidence" / (CASE_ID + ".schema.json")
CAPTURE_PATH = RAW_ROOT / "network_evidence" / (CASE_ID + ".capture.json")
TOKEN_AUDIT_PATH = H7_RAW_ROOT / "network_evidence" / "H7_TURNSTILE_TOKEN_AUDIT_20260814.json"
LEDGER_PATH = H7_RAW_ROOT / "calculate_ledger.json"
RESULT_PATH = OUTPUT_ROOT / "H7L2_SATURATION_HOLDOUT_RESULT.json"
REPORT_PATH = OUTPUT_ROOT / "H7L2_SATURATION_HOLDOUT_REPORT.md"


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def canonical_capture_sha256(capture):
    canonical = dict(capture)
    canonical["capture_artifact_sha256"] = None
    text = json.dumps(canonical, ensure_ascii=False, indent=2) + "\n"
    return sha256_bytes(text.encode("utf-8"))


def sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def longest_true_run(times, mask, nominal_dt):
    best = {"samples": 0, "duration_s": 0.0, "start_s": None, "end_s": None}
    start = None
    end = None
    samples = 0
    for index, passed in enumerate(mask):
        contiguous = end is None or times[index] - end <= nominal_dt * 1.5
        if passed and (samples == 0 or contiguous):
            if samples == 0:
                start = times[index]
            end = times[index]
            samples += 1
        elif passed:
            start = times[index]
            end = times[index]
            samples = 1
        else:
            samples = 0
            start = None
            end = None
        duration = samples * nominal_dt
        if duration > best["duration_s"]:
            best = {
                "samples": samples,
                "duration_s": duration,
                "start_s": start,
                "end_s": end,
            }
    return best


def main():
    prediction = read_json(PREDICTION_PATH)
    request = read_json(REQUEST_PATH)
    response = read_json(RESPONSE_PATH)
    schema = read_json(SCHEMA_PATH)
    capture = read_json(CAPTURE_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    ledger = read_json(LEDGER_PATH)

    assert prediction["status"] == "frozen_before_single_calculate"
    assert prediction["authorization"]["maximum_new_server_calculate_actions"] == 1
    assert prediction["authorization"]["automatic_retries"] == 0
    assert prediction["frozen_sources"]["calculate_ledger_count_before"] == 7
    assert sha256_file(PROJECT_ROOT / prediction["frozen_sources"]["effective_controller_module_path"]) == prediction["frozen_sources"]["effective_controller_module_sha256"]
    assert capture["http_status"] == 200 and capture["response_nonempty"] is True
    assert schema["pass"] is True and capture["schema_pass"] is True
    assert capture["payload_preflight"]["pass"] is True
    assert capture["payload_preflight"]["missile_id"] == prediction["intervention"]["selected_custom_id"]
    assert capture["turnstile_guard"]["unique_against_prior_submitted_requests"] is True
    assert capture["turnstile_guard"]["request_allowed_to_server"] is True
    for gate in ("requestWillBeSent", "requestPaused", "responseReceived", "loadingFinished", "response_body_read", "http_status_read"):
        assert capture["event_gates"][gate] is True
    assert sha256_file(REQUEST_PATH) == capture["request_artifact_sha256"]
    assert sha256_file(RESPONSE_PATH) == capture["response_artifact_sha256"]
    assert canonical_capture_sha256(capture) == capture["capture_artifact_sha256"]
    assert token_audit["submitted_server_request_count"] == 8
    assert token_audit["submitted_actions"][-1]["case_id"] == CASE_ID

    existing_actions = [action for action in ledger["actions"] if action.get("case_id") == CASE_ID]
    if ledger["calculate_actions_used"] == 7:
        assert sha256_file(LEDGER_PATH) == prediction["frozen_sources"]["calculate_ledger_sha256_before"]
        assert not existing_actions
    else:
        assert ledger["calculate_actions_used"] == 8
        assert len(existing_actions) == 1

    payload = json.loads(request["postData"])
    custom_id = prediction["intervention"]["selected_custom_id"]
    assert payload["Missiles"] == [custom_id]
    assert len(payload["CustomMissiles"]) == 1
    assert payload["CustomMissiles"][0]["Id"] == custom_id
    assert float(payload["CustomMissiles"][0]["Parameters"]["finsLatAccel"]) == float(prediction["intervention"]["finsLatAccel_g"])
    assert payload["Timestep"] == 0.02
    assert response["missileIds"] == [custom_id]
    assert len(response["results"]) == 1

    row = response["results"][0]
    times = [float(value) for value in row["times"]]
    commands = [None if value is None else float(value) for value in row["aCmdYaw"]]
    observed_magnitude = [None if value is None else float(value) for value in row["currentG"]]
    assert len(times) == len(commands) == len(observed_magnitude)
    assert len(times) > 1
    assert all(value is None or value >= 0.0 for value in observed_magnitude)
    nominal_dt = median([times[index] - times[index - 1] for index in range(1, len(times))])
    assert abs(nominal_dt - 0.02) < 1e-6

    model = prediction["frozen_model"]
    acceptance = prediction["frozen_acceptance"]
    k_value = float(model["K"])
    cap = float(model["predicted_effective_cap_g"])
    predicted = [None if command is None else sign(command) * min(k_value * abs(command), cap) for command in commands]
    signed_observed = [
        None if command is None or magnitude is None else sign(command) * magnitude
        for command, magnitude in zip(commands, observed_magnitude)
    ]

    predicted_saturated_mask = [
        command is not None and k_value * abs(command) >= cap for command in commands
    ]
    predicted_saturated_run = longest_true_run(times, predicted_saturated_mask, nominal_dt)
    predicted_saturated_samples = sum(1 for passed in predicted_saturated_mask if passed)

    ratio_min = float(acceptance["near_envelope_ratio_min"])
    ratio_max = float(acceptance["near_envelope_ratio_max"])
    envelope_ratios = [None if magnitude is None else magnitude / cap for magnitude in observed_magnitude]
    observed_near_envelope_mask = [
        ratio is not None and ratio_min <= ratio <= ratio_max for ratio in envelope_ratios
    ]
    observed_near_envelope_run = longest_true_run(times, observed_near_envelope_mask, nominal_dt)

    crossing_times = []
    previous_sign = 0.0
    previous_time = None
    for current_time, command in zip(times, commands):
        if command is None:
            continue
        current_sign = sign(command)
        if current_sign != 0.0 and previous_sign != 0.0 and current_sign != previous_sign:
            crossing_times.append((previous_time + current_time) / 2.0)
        if current_sign != 0.0:
            previous_sign = current_sign
            previous_time = current_time

    min_command = float(acceptance["minimum_abs_aCmdYaw_g"])
    crossing_exclusion = float(acceptance["sign_crossing_exclusion_s_each_side"])
    valid_indices = []
    for index, (current_time, command, observed) in enumerate(zip(times, commands, signed_observed)):
        if command is None or observed is None or abs(command) < min_command:
            continue
        if any(abs(current_time - crossing) <= crossing_exclusion for crossing in crossing_times):
            continue
        valid_indices.append(index)
    assert valid_indices

    errors = [predicted[index] - signed_observed[index] for index in valid_indices]
    absolute_errors = [abs(value) for value in errors]
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    mae = sum(absolute_errors) / len(absolute_errors)
    bias = sum(errors) / len(errors)
    observed_peak = max(abs(signed_observed[index]) for index in valid_indices)
    predicted_peak = max(abs(predicted[index]) for index in valid_indices)
    command_peak = max(abs(commands[index]) for index in valid_indices)
    normalized_rmse = rmse / observed_peak if observed_peak > 0 else float("inf")
    metric_threshold = float(acceptance["rmse_over_observed_peak_max"])

    predicted_saturation_pass = predicted_saturated_run["duration_s"] >= float(acceptance["minimum_predicted_continuous_saturated_duration_s"])
    observed_envelope_pass = observed_near_envelope_run["duration_s"] >= float(acceptance["minimum_observed_continuous_near_envelope_duration_s"])
    metric_pass = normalized_rmse <= metric_threshold
    overall_pass = predicted_saturation_pass and observed_envelope_pass and metric_pass

    state_value = row.get("state")
    terminal_state = state_value[-1] if isinstance(state_value, list) and state_value else state_value
    distance_values = [float(value) for value in row.get("distanceToTarget", []) if value is not None]
    minimum_distance = min(distance_values) if distance_values else None

    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "PASS" if overall_pass else "FAIL",
        "no_refit_performed": True,
        "scope": "Frozen aCmdYaw-to-effective-currentG envelope mapping only; terminal hit or miss is recorded but was not a frozen controller gate.",
        "calculate": {
            "new_server_actions": 1,
            "automatic_retries": 0,
            "ledger_total_after": 8,
            "missile_id": custom_id,
            "http_status": capture["http_status"],
            "response_bytes": capture["response_bytes"],
        },
        "frozen_model": model,
        "intervention": prediction["intervention"],
        "trajectory_context": {
            "terminal_state": terminal_state,
            "minimum_distance_to_target_m": minimum_distance,
            "ui_observation_from_operator": "missed_target_timeout",
            "controller_gate_dependency": False,
        },
        "sample_policy": {
            "total_samples": len(times),
            "used_metric_samples": len(valid_indices),
            "nominal_dt_s": nominal_dt,
            "minimum_abs_aCmdYaw_g": min_command,
            "sign_crossing_count": len(crossing_times),
            "sign_crossing_exclusion_s_each_side": crossing_exclusion,
        },
        "metrics": {
            "rmse_g": rmse,
            "mae_g": mae,
            "bias_predicted_minus_observed_g": bias,
            "p95_absolute_error_g": percentile(absolute_errors, 0.95),
            "max_absolute_error_g": max(absolute_errors),
            "rmse_over_observed_peak": normalized_rmse,
            "rmse_over_observed_peak_max": metric_threshold,
            "observed_peak_abs_currentG_g": observed_peak,
            "predicted_peak_abs_currentG_g": predicted_peak,
            "command_peak_abs_aCmdYaw_g": command_peak,
            "maximum_observed_envelope_ratio": max(ratio for ratio in envelope_ratios if ratio is not None),
            "predicted_saturated_samples": predicted_saturated_samples,
            "predicted_saturated_fraction": float(predicted_saturated_samples) / len(times),
            "predicted_longest_continuous_saturated": predicted_saturated_run,
            "observed_longest_continuous_near_envelope": observed_near_envelope_run,
        },
        "gates": {
            "http_200_nonempty_json": True,
            "one_missile_id_one_result": schema["missile_ids_one_to_one"],
            "required_same_index_arrays": schema["required_arrays_pass"] and schema["same_index_lengths"],
            "payload_preflight": capture["payload_preflight"]["pass"],
            "fresh_token": capture["turnstile_guard"]["unique_against_prior_submitted_requests"],
            "capture_hashes": True,
            "predicted_continuous_saturation": predicted_saturation_pass,
            "observed_continuous_near_envelope": observed_envelope_pass,
            "frozen_rmse_metric": metric_pass,
            "overall": overall_pass,
        },
        "hashes": {
            "request_artifact_sha256": capture["request_artifact_sha256"],
            "response_artifact_sha256": capture["response_artifact_sha256"],
            "capture_artifact_sha256": capture["capture_artifact_sha256"],
            "prediction_artifact_sha256": capture["prediction_artifact_sha256"],
        },
    }
    write_json(RESULT_PATH, result)

    report = """# H7L2 高饱和包线验证

结论：**{status}**。冻结模型未重拟合。

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| RMSE / 实测峰值 | {nrmse:.6f} | <= {nrmse_max:.4f} |
| RMSE | {rmse:.6f} g | — |
| 实测峰值 | {observed_peak:.6f} g | — |
| 冻结预测包线 | {cap:.6f} g | — |
| 最大实测/预测包线比 | {max_ratio:.6f} | — |
| 预测连续饱和 | {pred_duration:.3f} s | >= {pred_required:.3f} s |
| 实测连续近包线 | {obs_duration:.3f} s | >= {obs_required:.3f} s |

轨迹界面显示“脱靶/超时”；原始响应的终态为 `{terminal_state}`，最小目标距离为 {minimum_distance} m。该终态被记录，但它不是本次冻结的 `aCmdYaw -> currentG` 控制器门禁，不能据此单独判定控制器映射失败。

执行边界：新增 Calculate 1 次、无重试、ledger 总数 8；HTTP、schema、payload、token 与 hash 门禁通过；没有重拟合，也没有继续其他场景。
""".format(
        status=result["status"],
        nrmse=normalized_rmse,
        nrmse_max=metric_threshold,
        rmse=rmse,
        observed_peak=observed_peak,
        cap=cap,
        max_ratio=result["metrics"]["maximum_observed_envelope_ratio"],
        pred_duration=predicted_saturated_run["duration_s"],
        pred_required=float(acceptance["minimum_predicted_continuous_saturated_duration_s"]),
        obs_duration=observed_near_envelope_run["duration_s"],
        obs_required=float(acceptance["minimum_observed_continuous_near_envelope_duration_s"]),
        terminal_state=terminal_state,
        minimum_distance=("n/a" if minimum_distance is None else "{:.3f}".format(minimum_distance)),
    )
    write_text(REPORT_PATH, report)

    action = {
        "index": 8,
        "case_id": CASE_ID,
        "utc": capture["calculate_at_utc"],
        "status": "captured_http_200_valid_schema_saturation_holdout_" + result["status"].lower(),
        "raw_capture": True,
        "valid_for_analysis": True,
        "http_status": capture["http_status"],
        "request_id": capture["request_id"],
        "postData_sha256": capture["postData_sha256"],
        "request_artifact_sha256": capture["request_artifact_sha256"],
        "response_artifact_sha256": capture["response_artifact_sha256"],
        "capture_artifact_sha256": capture["capture_artifact_sha256"],
        "request_path": "data/raw/statshark_h7_controller_lite_v2_saturation_holdout/" + capture["request_path"],
        "response_path": "data/raw/statshark_h7_controller_lite_v2_saturation_holdout/" + capture["response_path"],
        "capture_path": "data/raw/statshark_h7_controller_lite_v2_saturation_holdout/network_evidence/" + CASE_ID + ".capture.json",
        "schema_path": "data/raw/statshark_h7_controller_lite_v2_saturation_holdout/" + capture["schema_path"],
        "prediction_path": "outputs/h7_controller_lite_v2_saturation_holdout/H7L2_SATURATION_HOLDOUT_PREDICTION.json",
        "result_path": "outputs/h7_controller_lite_v2_saturation_holdout/H7L2_SATURATION_HOLDOUT_RESULT.json",
        "event_gates": capture["event_gates"],
        "turnstile_header": "X-Turnstile-Token present; fresh page-native token",
        "turnstile_guard": capture["turnstile_guard"],
        "missile_id": custom_id,
        "timestep_sent": payload["Timestep"],
        "timestep_required": 0.02,
        "payload_preflight_pass": capture["payload_preflight"]["pass"],
        "response_schema_pass": schema["pass"],
        "holdout_status": result["status"],
        "rmse_over_observed_peak": normalized_rmse,
        "rmse_over_observed_peak_max": metric_threshold,
        "predicted_continuous_saturation_pass": predicted_saturation_pass,
        "observed_continuous_near_envelope_pass": observed_envelope_pass,
        "trajectory_terminal_state": terminal_state,
        "operator_ui_observation": "missed_target_timeout",
        "no_refit_performed": True,
        "note": "One explicitly authorized high-saturation envelope holdout. The trajectory miss/timeout is retained as context and is not substituted for the frozen controller-mapping gates.",
    }
    ledger["stage"] = "H7L2-saturation-envelope-single-holdout-executed-and-paused"
    ledger["authorization_policy"] = "explicit_single_saturation_envelope_holdout_calculate_consumed_then_pause"
    ledger["current_authorized_server_actions"] = 0
    ledger["used_turnstile_token_sha256"] = token_audit["used_turnstile_token_sha256"]
    ledger["calculate_actions_used"] = 8
    ledger["server_calculate_actions_used"] = 8
    if existing_actions:
        ledger["actions"] = [action if item.get("case_id") == CASE_ID else item for item in ledger["actions"]]
    else:
        ledger["actions"].append(action)
    ledger["status"] = "h7l2_saturation_envelope_holdout_" + result["status"].lower() + "_paused"
    ledger["next_action"] = "pause_after_single_saturation_holdout_no_refit_no_more_calculate"
    ledger["note"] = "Eight server submissions are counted. The single high-saturation envelope holdout is scored without refit and the workflow is paused."
    ledger["h7l2_saturation_holdout_completed"] = True
    ledger["h7l2_saturation_holdout_status"] = result["status"]
    ledger["h7l2_saturation_holdout_result_path"] = "outputs/h7_controller_lite_v2_saturation_holdout/H7L2_SATURATION_HOLDOUT_RESULT.json"
    ledger["calculate_budget"] = {
        "current_authorized_server_actions": 0,
        "already_counted_server_actions": 8,
        "authorization_statement": "single saturation-envelope holdout authorization consumed; score recorded; pause",
        "execution_constraint": "no refit and no further Calculate without new explicit authorization",
    }
    write_json(LEDGER_PATH, ledger)

    print(json.dumps({
        "case_id": CASE_ID,
        "status": result["status"],
        "rmse_over_observed_peak": normalized_rmse,
        "acceptance_max": metric_threshold,
        "predicted_continuous_saturation_s": predicted_saturated_run["duration_s"],
        "observed_continuous_near_envelope_s": observed_near_envelope_run["duration_s"],
        "observed_peak_g": observed_peak,
        "predicted_cap_g": cap,
        "terminal_state": terminal_state,
        "minimum_distance_to_target_m": minimum_distance,
        "new_calculate": 1,
        "ledger_total": 8,
        "result_path": str(RESULT_PATH),
        "report_path": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
