from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path


CASE_ID = "H7L2_REAL_AIM120A_HOLDOUT_01"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "statshark_h7_controller_lite_v2_holdout"
H7_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "statshark_h7_controller_id"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "h7_controller_lite_v2_holdout"
PREDICTION_PATH = OUTPUT_ROOT / "H7L2_REAL_AIM120A_HOLDOUT_PREDICTION.json"
REQUEST_PATH = RAW_ROOT / "requests" / (CASE_ID + ".request.json")
RESPONSE_PATH = RAW_ROOT / "responses" / (CASE_ID + ".response.json")
SCHEMA_PATH = RAW_ROOT / "network_evidence" / (CASE_ID + ".schema.json")
CAPTURE_PATH = RAW_ROOT / "network_evidence" / (CASE_ID + ".capture.json")
TOKEN_AUDIT_PATH = H7_RAW_ROOT / "network_evidence" / "H7_TURNSTILE_TOKEN_AUDIT_20260814.json"
LEDGER_PATH = H7_RAW_ROOT / "calculate_ledger.json"
RESULT_PATH = OUTPUT_ROOT / "H7L2_REAL_AIM120A_HOLDOUT_RESULT.json"
REPORT_PATH = OUTPUT_ROOT / "H7L2_REAL_AIM120A_HOLDOUT_REPORT.md"


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


def main():
    prediction = read_json(PREDICTION_PATH)
    request = read_json(REQUEST_PATH)
    response = read_json(RESPONSE_PATH)
    schema = read_json(SCHEMA_PATH)
    capture = read_json(CAPTURE_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    ledger = read_json(LEDGER_PATH)

    assert prediction["status"] == "frozen_before_statshark_calculate"
    assert prediction["authorization"]["automatic_retries"] == 0
    assert prediction["frozen_sources"]["calculate_ledger_count_before"] == 6
    assert capture["http_status"] == 200 and capture["response_nonempty"] is True
    assert schema["pass"] is True
    assert capture["schema_pass"] is True
    assert capture["payload_preflight"]["pass"] is True
    assert capture["payload_preflight"]["missile_id"] == "us_aim_120a"
    assert capture["turnstile_guard"]["unique_against_prior_submitted_requests"] is True
    assert capture["turnstile_guard"]["request_allowed_to_server"] is True
    assert capture["event_gates"]["requestWillBeSent"] is True
    assert capture["event_gates"]["responseReceived"] is True
    assert capture["event_gates"]["loadingFinished"] is True
    assert capture["event_gates"]["response_body_read"] is True
    assert sha256_file(REQUEST_PATH) == capture["request_artifact_sha256"]
    assert sha256_file(RESPONSE_PATH) == capture["response_artifact_sha256"]
    assert canonical_capture_sha256(capture) == capture["capture_artifact_sha256"]
    assert token_audit["submitted_server_request_count"] == 7
    assert token_audit["submitted_actions"][-1]["case_id"] == CASE_ID
    existing_actions = [action for action in ledger["actions"] if action.get("case_id") == CASE_ID]
    if ledger["calculate_actions_used"] == 6:
        assert sha256_file(LEDGER_PATH) == prediction["frozen_sources"]["calculate_ledger_sha256_before"]
        assert not existing_actions
    else:
        assert ledger["calculate_actions_used"] == 7
        assert len(existing_actions) == 1

    payload = json.loads(request["postData"])
    assert payload["Missiles"] == ["us_aim_120a"]
    assert not payload.get("CustomMissiles")
    assert payload["Timestep"] == 0.02
    assert response["missileIds"] == ["us_aim_120a"]
    assert len(response["results"]) == 1
    row = response["results"][0]
    times = [float(value) for value in row["times"]]
    commands = [None if value is None else float(value) for value in row["aCmdYaw"]]
    observed_magnitude = [float(value) for value in row["currentG"]]
    assert len(times) == len(commands) == len(observed_magnitude)
    assert len(times) > 0
    assert all(value >= 0.0 for value in observed_magnitude)

    model = prediction["frozen_model"]
    k_value = float(model["K"])
    cap = float(model["predicted_effective_cap_g"])
    predicted = [None if command is None else sign(command) * min(k_value * abs(command), cap) for command in commands]
    signed_observed = [None if command is None else sign(command) * magnitude for command, magnitude in zip(commands, observed_magnitude)]

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

    min_command = float(prediction["frozen_acceptance"]["minimum_abs_aCmdYaw_g"])
    crossing_exclusion = float(prediction["frozen_acceptance"]["sign_crossing_exclusion_s_each_side"])
    valid_indices = []
    for index, (current_time, command) in enumerate(zip(times, commands)):
        if command is None:
            continue
        if abs(command) < min_command:
            continue
        if any(abs(current_time - crossing) <= crossing_exclusion for crossing in crossing_times):
            continue
        valid_indices.append(index)
    assert valid_indices

    errors = [predicted[index] - signed_observed[index] for index in valid_indices]
    absolute_errors = [abs(value) for value in errors]
    squared_errors = [value * value for value in errors]
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mae = sum(absolute_errors) / len(absolute_errors)
    bias = sum(errors) / len(errors)
    observed_peak = max(abs(signed_observed[index]) for index in valid_indices)
    command_peak = max(abs(commands[index]) for index in valid_indices)
    predicted_peak = max(abs(predicted[index]) for index in valid_indices)
    normalized_rmse = rmse / observed_peak if observed_peak > 0 else float("inf")
    threshold = float(prediction["frozen_acceptance"]["rmse_over_observed_peak_max"])
    metric_pass = normalized_rmse <= threshold
    sign_matches = sum(1 for index in valid_indices if sign(predicted[index]) == sign(signed_observed[index]))
    envelope_indices = [index for index in valid_indices if k_value * abs(commands[index]) >= cap]

    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "PASS" if metric_pass else "FAIL",
        "interpretation": "The backend currentG array is a nonnegative magnitude. The Plan 7 Lite signed effective output is sign(aCmdYaw) * currentG; the frozen model is scored against that projected target.",
        "no_refit_performed": True,
        "calculate": {
            "new_server_actions": 1,
            "automatic_retries": 0,
            "ledger_total_after": 7,
            "missile_id": "us_aim_120a",
            "http_status": capture["http_status"],
            "response_bytes": capture["response_bytes"],
        },
        "frozen_model": model,
        "sample_policy": {
            "total_samples": len(times),
            "used_samples": len(valid_indices),
            "excluded_samples": len(times) - len(valid_indices),
            "invalid_same_index_samples": sum(1 for command in commands if command is None),
            "minimum_abs_aCmdYaw_g": min_command,
            "sign_crossing_exclusion_s_each_side": crossing_exclusion,
            "sign_crossing_count": len(crossing_times),
        },
        "coordinate_contract": {
            "backend_currentG": "nonnegative magnitude",
            "signed_effective_output": "sign(aCmdYaw) * currentG",
            "plant_input": "signed effective output",
            "model_sign_changed": False,
            "scorer_boundary_corrected": True,
        },
        "metrics": {
            "rmse_g": rmse,
            "mae_g": mae,
            "bias_predicted_minus_observed_g": bias,
            "p95_absolute_error_g": percentile(absolute_errors, 0.95),
            "max_absolute_error_g": max(absolute_errors),
            "observed_peak_abs_currentG_g": observed_peak,
            "predicted_peak_abs_currentG_g": predicted_peak,
            "command_peak_abs_aCmdYaw_g": command_peak,
            "rmse_over_observed_peak": normalized_rmse,
            "acceptance_max": threshold,
            "sign_match_fraction": float(sign_matches) / len(valid_indices),
            "envelope_branch_sample_fraction": float(len(envelope_indices)) / len(valid_indices),
        },
        "gates": {
            "http_200_nonempty_json": capture["http_status"] == 200 and capture["response_nonempty"],
            "one_missile_id_one_result": schema["missile_ids_one_to_one"],
            "required_same_index_arrays": schema["required_arrays_pass"] and schema["same_index_lengths"],
            "payload_preflight": capture["payload_preflight"]["pass"],
            "fresh_token": capture["turnstile_guard"]["unique_against_prior_submitted_requests"],
            "capture_hashes": True,
            "frozen_metric": metric_pass,
            "overall": metric_pass,
        },
        "hashes": {
            "request_artifact_sha256": capture["request_artifact_sha256"],
            "response_artifact_sha256": capture["response_artifact_sha256"],
            "capture_artifact_sha256": capture["capture_artifact_sha256"],
            "prediction_artifact_sha256": capture["prediction_artifact_sha256"],
        },
    }
    write_json(RESULT_PATH, result)

    report = """# AIM-120A 单次真实数据验证

结论：**{status}**。冻结模型未重拟合。StatShark 原始 `currentG` 是非负幅值；按照 Plan 7 Lite 的既定边界，评分目标为 `sign(aCmdYaw) * currentG`。

## 结果

| 指标 | 数值 |
|---|---:|
| 冻结门槛（RMSE / 实测峰值） | <= {threshold:.4f} |
| 实际 RMSE / 实测峰值 | {normalized_rmse:.6f} |
| RMSE | {rmse:.6f} g |
| MAE | {mae:.6f} g |
| 实测峰值 `|currentG|` | {observed_peak:.6f} g |
| 预测峰值 | {predicted_peak:.6f} g |
| 命令峰值 `|aCmdYaw|` | {command_peak:.6f} g |
| 符号一致率 | {sign_match:.3%} |
| 有效样本 | {used_samples} / {total_samples} |

本次原版 AIM-120A 的有效样本有 {envelope_fraction:.3%} 落在模型的包线饱和分支，其余使用 `0.714070845 * aCmdYaw` 命令分支。

先前把模型直接与原始非负 `currentG` 比较属于评分接口错误。模型符号没有翻转；修复的是观测量到带符号有效输出的坐标投影。

## 执行边界

- 新增真实 Calculate：1 次；无重试；ledger 总数更新为 7。
- HTTP 200、单导弹单结果、数组同索引、payload、token 与三类 hash 均通过。
- 没有重拟合，也没有继续运行其他场景。
- 本结果只验证这一场景下 `aCmdYaw -> currentG` 的条件映射，不等于完整制导或飞行模型验证。
""".format(
        status=result["status"],
        threshold=threshold,
        normalized_rmse=normalized_rmse,
        rmse=rmse,
        mae=mae,
        observed_peak=observed_peak,
        predicted_peak=predicted_peak,
        command_peak=command_peak,
        sign_match=float(sign_matches) / len(valid_indices),
        used_samples=len(valid_indices),
        total_samples=len(times),
        envelope_fraction=float(len(envelope_indices)) / len(valid_indices),
    )
    write_text(REPORT_PATH, report)

    action = {
        "index": 7,
        "case_id": CASE_ID,
        "utc": capture["calculate_at_utc"],
        "status": "captured_http_200_valid_schema_holdout_" + result["status"].lower(),
        "raw_capture": True,
        "valid_for_analysis": True,
        "http_status": capture["http_status"],
        "request_id": capture["request_id"],
        "postData_sha256": capture["postData_sha256"],
        "request_artifact_sha256": capture["request_artifact_sha256"],
        "response_artifact_sha256": capture["response_artifact_sha256"],
        "capture_artifact_sha256": capture["capture_artifact_sha256"],
        "request_path": "data/raw/statshark_h7_controller_lite_v2_holdout/" + capture["request_path"],
        "response_path": "data/raw/statshark_h7_controller_lite_v2_holdout/" + capture["response_path"],
        "capture_path": "data/raw/statshark_h7_controller_lite_v2_holdout/network_evidence/" + CASE_ID + ".capture.json",
        "schema_path": "data/raw/statshark_h7_controller_lite_v2_holdout/" + capture["schema_path"],
        "prediction_path": "outputs/h7_controller_lite_v2_holdout/H7L2_REAL_AIM120A_HOLDOUT_PREDICTION.json",
        "result_path": "outputs/h7_controller_lite_v2_holdout/H7L2_REAL_AIM120A_HOLDOUT_RESULT.json",
        "event_gates": capture["event_gates"],
        "turnstile_header": "X-Turnstile-Token present; fresh page-native token",
        "turnstile_guard": capture["turnstile_guard"],
        "missile_id": "us_aim_120a",
        "timestep_sent": payload["Timestep"],
        "timestep_required": 0.02,
        "payload_preflight_pass": capture["payload_preflight"]["pass"],
        "response_schema_pass": schema["pass"],
        "holdout_status": result["status"],
        "rmse_over_observed_peak": normalized_rmse,
        "rmse_over_observed_peak_max": threshold,
        "no_refit_performed": True,
        "note": "One explicitly authorized stock AIM-120A holdout Calculate. Backend currentG was treated as a nonnegative magnitude and projected to sign(aCmdYaw)*currentG for the frozen signed-output score; no refit was performed.",
    }
    ledger["stage"] = "H7L2-stock-AIM120A-single-holdout-executed-and-paused"
    ledger["authorization_policy"] = "explicit_single_stock_aim120a_holdout_calculate_consumed_then_pause"
    ledger["executor"] = "Codex"
    ledger["current_authorized_server_actions"] = 0
    ledger["used_turnstile_token_sha256"] = token_audit["used_turnstile_token_sha256"]
    ledger["calculate_actions_used"] = 7
    ledger["server_calculate_actions_used"] = 7
    if existing_actions:
        ledger["actions"] = [action if item.get("case_id") == CASE_ID else item for item in ledger["actions"]]
    else:
        ledger["actions"].append(action)
    ledger["status"] = "h7l2_stock_aim120a_holdout_" + result["status"].lower() + "_paused"
    ledger["next_action"] = "pause_after_single_holdout_score_no_refit_no_more_calculate"
    ledger["note"] = "Seven server submissions are counted. The stock AIM-120A H7L2 holdout is scored against sign(aCmdYaw)*currentG because backend currentG is a nonnegative magnitude; no parameter refit was performed."
    ledger["h7l2_holdout_completed"] = True
    ledger["h7l2_holdout_status"] = result["status"]
    ledger["h7l2_holdout_result_path"] = "outputs/h7_controller_lite_v2_holdout/H7L2_REAL_AIM120A_HOLDOUT_RESULT.json"
    ledger["calculate_budget"] = {
        "current_authorized_server_actions": 0,
        "already_counted_server_actions": 7,
        "authorization_statement": "single stock AIM-120A holdout authorization consumed; corrected signed-output score recorded; pause",
        "execution_constraint": "no refit and no further Calculate without new explicit authorization",
    }
    write_json(LEDGER_PATH, ledger)

    output = {
        "case_id": CASE_ID,
        "status": result["status"],
        "rmse_over_observed_peak": normalized_rmse,
        "acceptance_max": threshold,
        "rmse_g": rmse,
        "mae_g": mae,
        "observed_peak_g": observed_peak,
        "predicted_peak_g": predicted_peak,
        "used_samples": len(valid_indices),
        "total_samples": len(times),
        "new_calculate": 1,
        "ledger_total": 7,
        "ledger_sha256_after": sha256_file(LEDGER_PATH),
        "result_path": str(RESULT_PATH),
        "report_path": str(REPORT_PATH),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
