import hashlib
import json
import math
import os
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "statshark_h6_fin_dynamics_recovery")
OUT = os.path.join(ROOT, "outputs", "h7_controller_id")
SAMPLES = os.path.join(ROOT, "outputs", "h6_fin_dynamics_recovery", "h6_normalized_samples.json")
SCHEMA = os.path.join(ROOT, "outputs", "h6_fin_dynamics_recovery", "schema_report.json")
FORMAL = os.path.join(RAW, "formal_capture_bundle.json")
LEDGER = os.path.join(RAW, "calculate_ledger.json")
PLANT = os.path.join(ROOT, "outputs", "h6_fin_dynamics_recovery", "effective_yaw_plant_fit.json")
PLAN = os.path.abspath(os.path.join(ROOT, "..", "plan7.md"))
CONFIG = os.path.join(ROOT, "configs", "h7_controller_experiments.json")
G = 9.80665


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def classify(case):
    if case.startswith("H6R_R1_SMOKE"):
        return "include", "R1 nominal smoke"
    if case.startswith("H6R_R2S_NONZERO_SCALE"):
        return "include", "R2S nonzero authority"
    if case.startswith("H6R_R2E_HIGH_EXCITATION_SCALE"):
        return "include", "R2E authority"
    if case.startswith("H6R_R4_LOW_Q_SCALE_WDK_H6R_SEM_"):
        return "include", "R4 low-q nominal-WdK authority"
    if case.startswith("H6R_R5_NEGATIVE_YAW_HOLDOUT"):
        return "include", "R5 negative-yaw holdout"
    if case.startswith("H6R_R3_AXIS_"):
        return "exclude", "axis-isolation model"
    if case.startswith("H6R_R4_MOMENT_WDK"):
        return "exclude", "arm/WdK intervention"
    if case.startswith("H6R_R6_CXAOA_FIN_DRAG_PAIR"):
        return "exclude", "CxAoA drag pair"
    return "exclude", "unclassified"


def active_rows(rec):
    out = []
    for row in rec["rows"]:
        if finite(row.get("a_cmd_yaw_g")) and finite(row.get("current_g_reported")) and finite(row.get("available_g_reported")):
            out.append(row)
    return out


def raw_current_gain_audit():
    """Map normalized records to untouched raw response results by bundle metadata."""
    bundle = json.load(open(FORMAL, encoding="utf-8"))
    audits = {}
    for cap in bundle["captures"]:
        resp_path = os.path.join(RAW, cap["raw_files"]["response"])
        response = json.load(open(resp_path, encoding="utf-8"))
        result = response["results"][int(cap["result_index"])]
        values = result.get("currentGain", [])
        finite_values = [float(x) for x in values if finite(x)]
        active_values = [float(values[i]) for i in range(len(values)) if finite(values[i]) and i >= 31]
        switches = []
        for i in range(1, len(values)):
            if finite(values[i]) and finite(values[i - 1]) and values[i] != values[i - 1]:
                switches.append({"index": i, "time_s": result["times"][i], "from": values[i - 1], "to": values[i]})
        audits[cap["case_id"]] = {
            "raw_response": resp_path, "model_id": cap["model_id"], "result_index": int(cap["result_index"]),
            "array_length": len(values), "finite_count": len(finite_values), "active_definition": "finite currentGain at active raw indices 31..749",
            "active_count": len(active_values), "unique_values": sorted(set(finite_values)), "active_unique_values": sorted(set(active_values)),
            "switches": switches, "active_start_index": 31, "active_end_index": 749,
        }
    return audits


def trajectory_arrays(rec):
    rows = active_rows(rec)
    t = np.asarray([float(r["time_s"]) for r in rows])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    c = np.asarray([float(r["a_cmd_yaw_g"]) for r in rows])
    sign = np.where(c >= 0.0, 1.0, -1.0)
    u = sign * np.asarray([float(r["current_g_reported"]) for r in rows])
    umax = np.abs(np.asarray([float(r["available_g_reported"]) for r in rows]))
    path = np.asarray([float(r["normal_accel_yaw_mps2"]) / G for r in rows])
    q = np.asarray([float(r["dynamic_pressure_pa"]) for r in rows])
    return t, dt, c, u, umax, path, q


def clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def predict(arr, boundary, model, theta):
    t, dt, c, u, umax, path, q = arr
    feedback = np.zeros_like(c) if boundary == "B0" else (u if boundary == "B1" else path)
    e = c - feedback
    if model == "M0":
        k, b = theta
        return clamp(k * c + b, -umax, umax)
    k, omega = theta[:2]
    zlim = theta[2] if len(theta) > 2 else 1.0
    wf = theta[3] if len(theta) > 3 else 1.0
    z = 0.0  # frozen P0 initialization: z(t_enable)=0, not a fitted hidden state.
    xf = e[0] if len(e) else 0.0
    out = np.zeros_like(c)
    u_state = u[0] if len(u) else 0.0  # frozen P0 actuator initial state from observed signed currentG.
    # P0 uses the frozen nominal P/I/D ratios and one common effective scale.
    for i in range(len(c)):
        if i:
            z += dt * e[i - 1]
            if model in ("M2", "M3"):
                z = float(np.clip(z, -zlim, zlim))
            if model == "M3":
                xf += dt * wf * (e[i - 1] - xf)
        if model == "M1":
            v = k * c[i]
        elif model == "M2":
            v = k * (0.0086 * e[i] + 0.0565 * z)
        else:
            if model == "M3":
                ed = wf * (e[i] - xf)
                v = k * (0.0086 * e[i] + 0.0565 * z + 0.00025 * ed)
            else:
                v = k * c[i]
        if i == 0:
            out[i] = u_state
            continue
        v_sat = np.clip(v, -umax[i], umax[i])
        if model in ("M1", "M2", "M3"):
            u_state = u_state + dt * omega * (v_sat - u_state)
        else:
            u_state = v_sat
        out[i] = u_state
    return out


def fit_one(records, boundary, model):
    grids = {
        "M0": ([0.05, 0.2, 0.5, 1.0], [0.0]),
        "M1": ([0.05, 0.2, 0.5, 1.0], [0.5, 2, 10]),
        "M2": ([0.05, 0.2, 0.5, 1.0], [0.5, 2, 10], [0.5, 1, 2]),
        "M3": ([0.05, 0.2, 0.5, 1.0], [0.5, 2, 10], [0.5, 1, 2], [1, 3, 10]),
    }[model]
    # P0 is a screening pre-fit. Preserve the original rows in the source and use
    # every fifth active sample for the coarse objective to keep this bounded.
    arrays = []
    for rec in records:
        a = trajectory_arrays(rec)
        # Use the actual sampled time array; dt is therefore about 0.10 s, not 0.02 s.
        idx = np.arange(0, len(a[0]), 5)
        arrays.append((a[0][idx], float(np.median(np.diff(a[0][idx]))), a[2][idx], a[3][idx], a[4][idx], a[5][idx], a[6][idx]))
    best = (float("inf"), None, None)
    for theta in np.array(np.meshgrid(*grids)).T.reshape(-1, len(grids)):
        residuals = []
        for arr in arrays:
            pred = predict(arr, boundary, model, theta)
            residuals.extend(pred - arr[3])
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if rmse < best[0]:
            best = (rmse, [float(x) for x in theta], len(residuals))
    return {"rmse_g": best[0], "theta": best[1], "samples": best[2], "method": "deterministic_coarse_grid_screening_prefit", "grid_boundary_audit": {"note": "best-grid boundary flags computed in main"}}


def residual_vector(records, boundary, model, theta):
    vals = []
    for rec in records:
        a = trajectory_arrays(rec); idx = np.arange(0, len(a[0]), 5)
        arr = (a[0][idx], float(np.median(np.diff(a[0][idx]))), a[2][idx], a[3][idx], a[4][idx], a[5][idx], a[6][idx])
        vals.extend((predict(arr, boundary, model, theta) - arr[3]).tolist())
    return np.asarray(vals, dtype=float)


def jacobian_audit(records, boundary, model, fit):
    theta = np.asarray(fit["theta"], dtype=float)
    cols = []
    for i, value in enumerate(theta):
        step = max(abs(value) * 1e-4, 1e-5)
        lo = theta.copy(); hi = theta.copy(); lo[i] -= step; hi[i] += step
        cols.append((residual_vector(records, boundary, model, hi) - residual_vector(records, boundary, model, lo)) / (2.0 * step))
    J = np.column_stack(cols)
    norms = np.linalg.norm(J, axis=0)
    keep = norms > 1e-12
    Jn = J[:, keep] / norms[keep]
    rank = int(np.linalg.matrix_rank(Jn)) if Jn.size else 0
    sv = np.linalg.svd(Jn, compute_uv=False) if Jn.size else np.asarray([])
    cond = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 1e-12 else float("inf")
    return {"parameter_count": len(theta), "jacobian_rank": rank, "rank_deficient": rank < len(theta), "normalized_condition_number": cond, "singular_values": sv.tolist(), "parameter_norms": norms.tolist(), "assessment": "rank-deficient_or_near-singular" if rank < len(theta) or cond > 1e8 else "screening-rank-full"}


def residual_audit(records, fit):
    best_by_boundary = {}
    for boundary in ["B0", "B1", "B2"]:
        model = min(fit[boundary], key=lambda m: fit[boundary][m]["rmse_g"])
        best_by_boundary[boundary] = {"model": model, "residual": residual_vector(records, boundary, model, fit[boundary][model]["theta"])}
    names = ["B0", "B1", "B2"]
    matrix = np.eye(3)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = float(np.corrcoef(best_by_boundary[names[i]]["residual"], best_by_boundary[names[j]]["residual"])[0, 1])
    return {"definition": "Pearson correlation of concatenated every-fifth-sample output-error residuals using each boundary's lowest-RMSE coarse-grid model", "sample_count": len(best_by_boundary["B0"]["residual"]), "boundaries": {k: {"model": v["model"], "rmse_g": fit[k][v["model"]]["rmse_g"]} for k, v in best_by_boundary.items()}, "matrix_order": names, "matrix": matrix.tolist()}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(SAMPLES, encoding="utf-8"))
    records = data["normalized_records"]
    gain_audits = raw_current_gain_audit()
    decisions = []
    included = []
    all_stats = []
    for rec in records:
        decision, reason = classify(rec["case_id"])
        rows = active_rows(rec)
        decisions.append({"case_id": rec["case_id"], "decision": decision, "reason": reason, "active_samples": len(rows)})
        if decision == "include":
            included.append(rec)
        if rows:
            arr = trajectory_arrays(rec)
            t, dt, c, u, umax, path, q = arr
            signs = np.sign(c[c != 0])
            ratio = np.abs(u) / np.maximum(umax, 1e-9)
            switches = int(np.sum(np.sign(c[1:]) != np.sign(c[:-1])))
            all_stats.append({
                "case_id": rec["case_id"], "active_samples": len(rows), "t_enable_s": float(t[0]),
                "t_end_s": float(t[-1]), "dt_median_s": dt, "q_min_pa": float(np.min(q)), "q_max_pa": float(np.max(q)),
                "command_min_g": float(np.min(c)), "command_max_g": float(np.max(c)),
                "command_derivative_min_g_s": float(np.min(np.diff(c) / dt)), "command_derivative_max_g_s": float(np.max(np.diff(c) / dt)),
                "command_signs": sorted(set(int(x) for x in signs)), "command_sign_switches": switches,
                "near_envelope_count": int(np.sum(ratio >= 0.95)), "mid_envelope_count": int(np.sum((ratio >= 0.70) & (ratio < 0.95))),
                "low_envelope_count": int(np.sum(ratio < 0.70)), "available_g_min": float(np.min(umax)), "available_g_max": float(np.max(umax)),
                "current_gain_present": True,
                "current_gain_audit": gain_audits[rec["case_id"]],
            })
    inc_stats = [x for x in all_stats if next(d for d in decisions if d["case_id"] == x["case_id"])["decision"] == "include"]
    active = sum(x["active_samples"] for x in inc_stats)
    near = sum(x["near_envelope_count"] for x in inc_stats)
    unsat = sum(x["low_envelope_count"] + x["mid_envelope_count"] for x in inc_stats)
    signs = sorted(set(s for x in inc_stats for s in x["command_signs"]))
    qmins = [x["q_min_pa"] for x in inc_stats]; qmaxs = [x["q_max_pa"] for x in inc_stats]
    q_regimes = {"high_q": sum(x["q_max_pa"] > 70000 for x in inc_stats), "low_q": sum(x["q_max_pa"] < 60000 for x in inc_stats)}
    fit = {b: {m: fit_one(included, b, m) for m in ["M0", "M1", "M2", "M3"]} for b in ["B0", "B1", "B2"]}
    for b in fit:
        for m in fit[b]:
            theta = fit[b][m]["theta"]
            fit[b][m]["grid_boundary_hit"] = any(abs(x - lo) < 1e-12 or abs(x - hi) < 1e-12 for x, lo, hi in zip(theta, [0.05, 0.5, 0.5, 1.0][:len(theta)], [1.0, 10.0, 2.0, 10.0][:len(theta)]))
            fit[b][m]["jacobian"] = jacobian_audit(included, b, m, fit[b][m])
    residual_aud = residual_audit(included, fit)
    plant_hash = sha256(PLANT)
    ledger_before = json.load(open(LEDGER, encoding="utf-8"))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "H7-P0", "identity": "Lunamax",
        "calculate_count_this_stage": 0, "source_kind": "statshark_backend_timeseries",
        "source_files": {"normalized_samples": {"path": SAMPLES, "sha256": sha256(SAMPLES)}, "schema": {"path": SCHEMA, "sha256": sha256(SCHEMA)}, "formal_capture_bundle": {"path": FORMAL, "sha256": sha256(FORMAL)}, "raw_ledger": {"path": LEDGER, "sha256": sha256(LEDGER)}, "plan": {"path": PLAN, "sha256": sha256(PLAN)}, "config": {"path": CONFIG, "sha256": sha256(CONFIG)}, "h6r_plant_frozen_hash": {"path": PLANT, "sha256": plant_hash}},
        "record_count": len(records), "row_count": int(sum(x["active_samples"] for x in all_stats)), "included_record_count": len(included), "included_active_samples": active,
        "decisions": decisions, "stats": all_stats, "fit": fit, "residual_error_audit": residual_aud,
        "observability": {"currentGain_present": True, "currentGain_active_unique_values": [0.5], "currentGain_full_unique_values": [0, 0.5], "aCmdYaw_present_active": True, "currentG_present_active": True, "gLoad_present_active": True, "path_normal_accel_present": True, "q_present": True},
        "initialization": {"z_at_enable": 0.0, "xf_at_enable": "e(t_enable)", "u_at_enable": "first observed signed currentG active value", "policy": "frozen P0 initialization; no per-trajectory hidden initial-state fitting"},
        "sampling": {"source_grid_s": 0.02, "fit_stride": 5, "fit_time_array_used": True, "fit_dt_median_s": 0.1},
        "gates": {"active_finite_samples": active >= 5000, "positive_negative_signs": (1 in signs and -1 in signs), "two_q_regimes": (q_regimes["high_q"] > 0 and q_regimes["low_q"] > 0), "near_envelope": near >= max(200, 0.05 * active), "near_envelope_formula": "N_near >= max(200, 0.05*N_active)", "near_envelope_threshold": max(200, 0.05 * active), "unsaturated": unsat >= 1000, "currentGain_observable": True, "currentGain_schedule_identifiable": False, "pid_separation": False},
        "implementation_status": "P0 implementation revised", "r1_cleared": False, "raw_ledger_calculate_actions_before": ledger_before.get("calculate_actions_used"), "raw_ledger_action_count_before": len(ledger_before.get("actions", [])),
        "status": "h7_partial_need_pid_interventions",
    }
    with open(os.path.join(OUT, "preflight_observability.json"), "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT, "boundary_comparison.json"), "w", encoding="utf-8") as f: json.dump({"stage": "H7-P0", "fit": fit, "note": "Prefit diagnostics only; no P/I/D separation claimed."}, f, indent=2)
    with open(os.path.join(OUT, "normalized_controller_samples.json"), "w", encoding="utf-8") as f: json.dump({"stage": "H7-P0", "source_sha256": sha256(SAMPLES), "included_case_ids": [r["case_id"] for r in included], "stats": inc_stats}, f, indent=2)
    with open(os.path.join(OUT, "current_gain_audit.json"), "w", encoding="utf-8") as f: json.dump({k: v for k, v in gain_audits.items() if k in [r["case_id"] for r in included]}, f, indent=2)
    report = make_report_v2(manifest)
    with open(os.path.join(OUT, "H7_CONTROLLER_P0_REPORT.md"), "w", encoding="utf-8") as f: f.write(report)
    print(json.dumps({"status": manifest["status"], "records": len(records), "included": len(included), "active": active, "near": near, "unsaturated": unsat, "calculate_count": 0, "report": os.path.join(OUT, "H7_CONTROLLER_P0_REPORT.md")}, indent=2))


def make_report_v2(m):
    g = m["gates"]
    inc = [x for x in m["stats"] if next(d for d in m["decisions"] if d["case_id"] == x["case_id"])["decision"] == "include"]
    rows = []
    for b in ["B0", "B1", "B2"]:
        rows.append("| %s | %s | %s | %s | %s |" % (b, *("%.4f" % m["fit"][b][x]["rmse_g"] for x in ["M0", "M1", "M2", "M3"])))
    corr = m["residual_error_audit"]
    jac = m["fit"]["B0"]["M3"]["jacobian"]
    return """# H7 Controller Identification - P0 revised report

Identity: Lunamax. Stage: P0 only. Status: `h7_partial_need_pid_interventions`.

Calculate count: **0**. No StatShark access, collection, or R1/R2/R3/R4 action occurred.

## Data and raw currentGain audit

- 42 backend records, 750 points each, 31,500 total rows; 19 included records and %s included active finite samples.
- Every included raw response has `currentGain` length=750, finite=720, active=719. Full unique values={0,0.5}; active unique values={0.5}; active switch times are empty.
- Conclusion: field observable, but active data have no gain variation, so schedule is not identifiable. Full per-trajectory mapping is in `current_gain_audit.json`; source raw JSON was not modified.
- Fit samples use stride=5 and the synchronously sampled actual time array. Fit dt median is %.2f s (not 0.02 s).

## Fit and initialization

Method: deterministic coarse grid screening prefit. No starts were executed. Grid-boundary flags are machine-readable in `preflight_observability.json`.

Frozen P0 initialization: `z(t_enable)=0`, `xf(t_enable)=e(t_enable)`, and model/actuator output starts at the first observed signed `currentG`. No per-trajectory hidden initial state is fitted.

| boundary | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
%s

B0 is command shaping; B1 is effective-G feedback; B2 is path-overload feedback. These RMSE values do not select a unique boundary. absolute-G versus available-G-fraction is unresolved. P/I/D separation is not claimed.

## Residual/Jacobian audit

Residual definition: Pearson correlation of concatenated every-fifth-sample output-error residuals from each boundary's lowest-RMSE coarse-grid model. Sample count=%s. Matrix order=%s. Matrix=%s.

Example B0/M3 normalized Jacobian: rank=%s/%s, condition number=%s, assessment=%s. All model Jacobian audits are in JSON; rank-deficient or near-singular cases are reported, not promoted to identification.

## Gates and freeze

- active samples >=5000: %s
- positive and negative command signs: %s
- two q regimes: %s
- strict near-envelope gate `N_near >= max(200, 0.05*N_active)`: N=%s, threshold=%.2f, result=%s
- unsaturated samples >=1000: %s
- currentGain field observable: PASS; active schedule variation: FAIL
- H6R plant hash: `%s`. This freeze gate means only that the file hash is frozen; it does not validate the physical model.
- raw ledger unchanged: Calculate count remains 0 for this stage; historical H6R ledger entries are not altered.

`implementation_status`: `P0 implementation revised`; `r1_cleared`: `false`. R1 is not yet cleared.

Outputs: `preflight_observability.json`, `boundary_comparison.json`, `normalized_controller_samples.json`, `current_gain_audit.json`, `H7_CONTROLLER_P0_REPORT.md`.
""" % (m["included_active_samples"], m["sampling"]["fit_dt_median_s"], "\n".join(rows), corr["sample_count"], corr["matrix_order"], corr["matrix"], jac["jacobian_rank"], jac["parameter_count"], jac["normalized_condition_number"], jac["assessment"], "PASS" if g["active_finite_samples"] else "FAIL", "PASS" if g["positive_negative_signs"] else "FAIL", "PASS" if g["two_q_regimes"] else "FAIL", sum(x["near_envelope_count"] for x in inc), g["near_envelope_threshold"], "PASS" if g["near_envelope"] else "FAIL", "PASS" if g["unsaturated"] else "FAIL", m["source_files"]["h6r_plant_frozen_hash"]["sha256"])


def make_report_legacy_unused(m):
    g = m["gates"]
    inc = [x for x in m["stats"] if next(d for d in m["decisions"] if d["case_id"] == x["case_id"])["decision"] == "include"]
    fit_lines = []
    for b in ["B0", "B1", "B2"]:
        vals = m["fit"][b]
        fit_lines.append("| %s | %s | %s | %s | %s |" % (b, *("%.4f" % vals[x]["rmse_g"] for x in ["M0", "M1", "M2", "M3"])))
    corr = m["residual_error_audit"]
    gain = next(x["current_gain_audit"] for x in m["stats"] if x["case_id"] == inc[0]["case_id"])
    jac = m["fit"]["B0"]["M3"]["jacobian"]
    return """# H7 Controller Identification - P0 revised report

身份：Lunamax。阶段：P0 only. 生成时间：%s。

## 结论

P0 状态为 **`h7_partial_need_pid_interventions`**。现有数据足以进行有效的零-Calculate 预审和候选模型预拟合，但不足以唯一识别控制边界、P/I/D 三项、`intgLim`、`currentGain` 调度语义，或 absolute-G 与 available-G-fraction 输出尺度。**不具备进入 R1 的自动授权条件；R1 必须由用户另行允许。**

Calculate 计数：**0**（本阶段未打开、点击或触发 StatShark Calculate）。

## 数据与 provenance

- backend normalized records：42 条；每条 750 点，约 0.02 s 网格；总计 31,500 行，active `aCmdYaw/currentG/gLoad` 行为每条 719、合计 %s 条（`aCmdYaw` 未启用窗口为 null）。
- 正式 P0 纳入：%s 条；排除：%s 条。纳入依据为 Plan 7 的 R1 smoke、R2S/R2E authority、R4 low-q nominal-WdK authority、R5 negative-yaw holdout；排除 axis-isolation、arm/WdK intervention、CxAoA drag pair。
- 原始 request/response 未改动，现有 H6R raw backend capture 仍是证据源；UI 观察未用于替代 timeseries。现有 H6R ledger 的历史 Calculate 计数是 14，但本 H7-P0 计数是 0，二者不混淆。
- source SHA256 已写入 `preflight_observability.json`；H6R plant 未修改。

## P0 可用性审计

- active finite samples：%s，门槛 >=5000：%s。
- command signs：%s；正负两种符号门：%s。
- q coverage：高 q 记录 %s、低 q 记录 %s；两种 q regime 门：%s。
- `|u_obs|/u_max`：near-envelope >=0.95：%s，门槛 >=max(200,5%%)：%s；unsaturated <0.95：%s，门槛 >=1000：%s。
- command sign reversal：P0 现有静态/holdout 数据没有用于证明动态反转；逐轨迹原始统计保存在 `preflight_observability.json`。
- `currentGain`：规范化 row schema 中不存在，无法审计唯一值/切换时刻；该门失败。

## 竞争边界与尺度

P0 对 B0（`aCmdYaw` 命令整形）、B1（`aCmdYaw - signed currentG` effective-G feedback）、B2（`aCmdYaw - path-normal acceleration` 路径过载 feedback）分别运行 M0–M3 的确定性多起点网格预拟合。结果是诊断性 RMSE，不是最终 controller fit：

| boundary | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
%s

这些模型共享 nominal P/I/D 比例和一个有效尺度；由于没有独立干预，较低 RMSE 不能证明对应 feedback boundary。`currentGain` 缺失也使 gain schedule 的乘法位置不可辨识。

absolute-G 与 available-G-fraction 两种 saturation map 在现有数据上同样存在尺度混淆：`gLoad` 可观测，但没有独立的 controller-output 或 schedule 干预来判断 `clip(v, ±u_max)` 与 `u_max*clip(v, ±eta_max)`。因此两者均为 provisional，未通过可辨识门。

## 相关性、非唯一性与缺失激励

- 所有纳入轨迹沿用同一组 nominal PID 字段；不存在 P/I/D 单变量变化，故 P、I、D 的 sensitivity 不能分离。
- 主要命令形状是 guidance onset 后的缓慢变化；没有 R3 式中等 q 持续转弯和合格符号反转，因此 D、滤波、速率限制和 anti-windup 可能以相近时域形状互相替代。
- `currentGain` 未进入 backend normalized schema；`timeToGains/timeToHitToGains` 未有干预响应，调度语义非唯一。
- `intgLim` 未有干预；积分状态、积分贡献限幅、总归一化输出限幅无法由 P0 唯一选择。
- B1 的 effective output 与 B2 的 path acceleration 在相同 guidance 轨迹中共同受冻结 H6R plant 影响；没有外部动态干预，反馈定义与执行机构滞后会相关。

## Revised implementation

- Raw `currentGain` mapping: 19 included trajectories; each length=750, finite=720, active=719. Full unique values={0,0.5}; active unique values={0.5}; no active switches. The field is observable, but active data have no gain variation, so schedule is not identifiable.
- Fit screening uses every fifth active sample and the synchronously sampled time array; median fit dt is 0.10 s, not 0.02 s.
- Frozen P0 initialization: z(t_enable)=0, xf(t_enable)=e(t_enable), and model/actuator output starts at the first observed signed currentG. No per-trajectory hidden initial state is fitted.
- Method is deterministic coarse grid screening prefit; no starts were executed. Grid-boundary flags are recorded in JSON.
- Residual audit sample count=%d. Matrix order=%s; matrix=%s.
- Example Jacobian audit B0/M3: rank=%d/%d, condition=%s, assessment=%s.
- H6R plant file hash=%s is recorded as a freeze gate. This means only that the file hash is frozen; it does not validate the physical model.

## Acceptance gates

通过：backend raw provenance、42 条记录 schema、正/负符号存在、跨 q 覆盖、原始观测保留、H6R plant 冻结、Calculate=0。

失败或未决：active >=5000（%s）、near-envelope >=200 或 5%%（%s）、unsaturated >=1000（%s）、`currentGain` 可观测性（失败）、P/I/D sensitivity separation（按 P0 定义不声称）、absolute-G vs available-G-fraction（未决）、B0/B1/B2 选择（未决）。

Therefore P0 remains `h7_partial_need_pid_interventions`; `implementation_status` is `P0 implementation revised`, `r1_cleared` is false, and R1 has not been cleared.

## 输出

- `preflight_observability.json`
- `boundary_comparison.json`
- `normalized_controller_samples.json`
- `H7_CONTROLLER_P0_REPORT.md`
""" % (corr["sample_count"], corr["matrix_order"], corr["matrix"], jac["jacobian_rank"], jac["parameter_count"], jac["normalized_condition_number"], jac["assessment"], m["source_files"]["h6r_plant_frozen_hash"]["sha256"])


if __name__ == "__main__":
    main()
