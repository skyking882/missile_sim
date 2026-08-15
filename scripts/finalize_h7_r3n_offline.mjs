import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { stampCanonicalCaptureArtifactHash } from "./h7_cdp_capture_guard.mjs";

const CASE_ID = "H7_R3N_NEAR_ENVELOPE_RETRY";
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RAW_ROOT = path.join(PROJECT_ROOT, "data", "raw", "statshark_h7_controller_id");
const OUTPUT_DIR = path.join(PROJECT_ROOT, "outputs", "h7_controller_id");
const REL = {
  request: `requests/${CASE_ID}.request.json`,
  response: `responses/${CASE_ID}.response.json`,
  capture: `network_evidence/${CASE_ID}.capture.json`,
  schema: `network_evidence/${CASE_ID}.schema.json`,
  readback: `model_snapshots/${CASE_ID}_READBACK_20260814.json`,
  contrast: "payload_contrast_r3n.json",
};
const ABS = {
  request: path.join(RAW_ROOT, REL.request),
  response: path.join(RAW_ROOT, REL.response),
  capture: path.join(RAW_ROOT, REL.capture),
  schema: path.join(RAW_ROOT, REL.schema),
  readback: path.join(RAW_ROOT, REL.readback),
  contrast: path.join(RAW_ROOT, REL.contrast),
  ledger: path.join(RAW_ROOT, "calculate_ledger.json"),
  manifest: path.join(RAW_ROOT, "session_manifest.json"),
  config: path.join(PROJECT_ROOT, "configs", "h7_controller_experiments.json"),
  plan: path.join(PROJECT_ROOT, "..", "plan7.md"),
  plant: path.join(PROJECT_ROOT, "outputs", "h6_fin_dynamics_recovery", "effective_yaw_plant_fit.json"),
  analysis: path.join(OUTPUT_DIR, `${CASE_ID}_ANALYSIS.json`),
  report: path.join(OUTPUT_DIR, `${CASE_ID}_REPORT.md`),
};

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writeText(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, value, "utf8");
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function utcNow() {
  return new Date().toISOString();
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function numeric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function p95(values) {
  const finite = values.filter(Number.isFinite).map(Math.abs).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  return finite[Math.min(finite.length - 1, Math.ceil(0.95 * finite.length) - 1)];
}

function booleanWindows(times, flags) {
  const windows = [];
  let start = null;
  for (let index = 0; index < flags.length; index += 1) {
    if (flags[index] && start === null) start = index;
    if ((!flags[index] || index === flags.length - 1) && start !== null) {
      const end = flags[index] ? index : index - 1;
      windows.push({
        start_s: times[start],
        end_s: times[end],
        duration_s: times[end] - times[start],
        samples: end - start + 1,
      });
      start = null;
    }
  }
  return windows;
}

function thresholdRuns(times, values, threshold) {
  const runs = [];
  let start = null;
  let sign = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const currentSign = Number.isFinite(value) && Math.abs(value) >= threshold ? Math.sign(value) : 0;
    if (currentSign !== 0 && start === null) {
      start = index;
      sign = currentSign;
    } else if (currentSign !== sign && start !== null) {
      const end = index - 1;
      runs.push({
        sign,
        start_s: times[start],
        end_s: times[end],
        duration_s: times[end] - times[start],
        samples: end - start + 1,
      });
      start = currentSign !== 0 ? index : null;
      sign = currentSign || 0;
    }
  }
  if (start !== null) {
    const end = values.length - 1;
    runs.push({
      sign,
      start_s: times[start],
      end_s: times[end],
      duration_s: times[end] - times[start],
      samples: end - start + 1,
    });
  }
  return runs;
}

function signCrossingCount(values) {
  let previous = 0;
  let crossings = 0;
  for (const value of values) {
    if (!Number.isFinite(value) || value === 0) continue;
    const sign = Math.sign(value);
    if (previous !== 0 && sign !== previous) crossings += 1;
    previous = sign;
  }
  return crossings;
}

function maxAbsDerivative(times, values) {
  let maximum = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (!Number.isFinite(values[index]) || !Number.isFinite(values[index - 1])) continue;
    const dt = times[index] - times[index - 1];
    if (!(dt > 0)) continue;
    maximum = Math.max(maximum, Math.abs((values[index] - values[index - 1]) / dt));
  }
  return maximum;
}

function analyzeArm(row, arm) {
  const times = row.times || [];
  const command = row.aCmdYaw || [];
  const observed = row.currentG || [];
  const envelope = row.gLoad || [];
  const active = command.map((value, index) =>
    Number.isFinite(value) && Number.isFinite(observed[index]) && Number.isFinite(envelope[index]) && envelope[index] > 0,
  );
  const ratios = command.map((value, index) =>
    active[index] ? Math.abs(observed[index]) / envelope[index] : Number.NaN,
  );
  const strictFlags = ratios.map((value) => Number.isFinite(value) && value >= 0.95);
  const unsaturatedFlags = ratios.map((value) => Number.isFinite(value) && value < 0.95);
  const strictWindows = booleanWindows(times, strictFlags);
  const unsaturatedWindows = booleanWindows(times, unsaturatedFlags);
  const signRuns = thresholdRuns(times, command, 0.25);
  const hasPositiveRun = signRuns.some((run) => run.sign > 0 && run.duration_s >= 0.25);
  const hasNegativeRun = signRuns.some((run) => run.sign < 0 && run.duration_s >= 0.25);
  const finiteRatios = ratios.filter(Number.isFinite);
  const terminalTime = times.length ? times[times.length - 1] : null;
  const terminalDistance = Array.isArray(row.distanceToTarget) && row.distanceToTarget.length
    ? row.distanceToTarget[row.distanceToTarget.length - 1]
    : null;
  const criticalWindowCompleted = Number.isFinite(terminalTime) && terminalTime >= 14.9 && Number.isFinite(row.state);
  return {
    name: arm.name,
    missileId: arm.id,
    finsLatAccel: arm.finsLatAccel,
    times_length: times.length,
    finite_aCmdYaw_count: command.filter(Number.isFinite).length,
    active_controller_count: active.filter(Boolean).length,
    t_start_s: times[0] ?? null,
    t_end_s: terminalTime,
    aCmdYaw_min: Math.min(...command.filter(Number.isFinite)),
    aCmdYaw_max: Math.max(...command.filter(Number.isFinite)),
    aCmdYaw_p95_abs: p95(command),
    sign_crossings: signCrossingCount(command),
    sign_runs: signRuns,
    sign_reversal_valid: hasPositiveRun && hasNegativeRun,
    max_abs_derivative_g_s: maxAbsDerivative(times, command),
    max_abs_u_obs_over_u_max: finiteRatios.length ? Math.max(...finiteRatios) : null,
    strict_near_envelope_windows: strictWindows,
    strict_near_envelope_max_duration_s: strictWindows.length ? Math.max(...strictWindows.map((window) => window.duration_s)) : 0,
    strict_near_envelope_pass_1s: strictWindows.some((window) => window.duration_s >= 1.0),
    unsaturated_windows: unsaturatedWindows,
    unsaturated_max_duration_s: unsaturatedWindows.length ? Math.max(...unsaturatedWindows.map((window) => window.duration_s)) : 0,
    unsaturated_pass_1s: unsaturatedWindows.some((window) => window.duration_s >= 1.0),
    terminal_state: row.state ?? null,
    terminal_time_s: terminalTime,
    terminal_distance_m: terminalDistance,
    critical_window_completed: criticalWindowCompleted,
  };
}

function normalizePayloadContrast(rawContrast, readback) {
  const armsById = new Map(readback.selected.map((arm) => [arm.id, arm]));
  const allowedNameMetadataOmissions = [];
  const remaining = [];
  for (const difference of rawContrast.differences || []) {
    const match = String(difference.path || "").match(/Id=([^\]]+)\].*\.bulletName$/);
    const arm = match ? armsById.get(match[1]) : null;
    if (
      arm &&
      String(difference.path).endsWith(".Parameters.bulletName") &&
      !Object.prototype.hasOwnProperty.call(difference, "left") &&
      difference.right === arm.name
    ) {
      allowedNameMetadataOmissions.push({
        path: difference.path,
        value: difference.right,
        reason: "page serializer omits the local bulletName name-metadata field; no model parameter was changed",
      });
    } else {
      remaining.push(difference);
    }
  }
  const normalized = {
    ...rawContrast,
    schema_version: 3,
    stage: "post_request_payload_contrast_normalized",
    generated_at_utc: utcNow(),
    raw_differences: rawContrast.differences || [],
    allowed_name_metadata_omissions: allowedNameMetadataOmissions,
    differences: remaining,
    normalization_policy: "bulletName may be omitted by the native page serializer; any other request/readback difference fails",
    pass: Boolean(rawContrast.request_json_valid) && remaining.length === 0,
  };
  return normalized;
}

function updateLedger({ ledger, capture, schema, contrast, gate, analysisPath, reportPath, analysis }) {
  assert(!(ledger.actions || []).some((entry) => entry.case_id === CASE_ID), "R3N ledger entry already exists; refusing duplicate finalization");
  const tokenHash = capture.turnstile_guard?.token_sha256;
  const action = {
    index: 6,
    case_id: CASE_ID,
    utc: capture.calculate_at_utc,
    status: "captured_http_200_schema_pass_r3n_gate_failed",
    raw_capture: true,
    valid_for_analysis: schema.pass && contrast.pass,
    valid_for_controller_fit: false,
    http_status: capture.http_status,
    request_id: capture.request_id,
    postData_sha256: capture.postData_sha256,
    request_artifact_sha256: capture.request_artifact_sha256,
    response_artifact_sha256: capture.response_artifact_sha256,
    capture_artifact_sha256: capture.capture_artifact_sha256,
    request_path: capture.request_path,
    response_path: capture.response_path,
    capture_path: capture.request_path.replace(/^requests\//, "network_evidence/").replace(/\.request\.json$/, ".capture.json"),
    schema_path: capture.schema_path,
    payload_contrast_path: capture.payload_contrast_path,
    readback_path: capture.readback_path,
    analysis_path: analysisPath,
    report_path: reportPath,
    event_gates: capture.event_gates,
    turnstile_header: "X-Turnstile-Token present; fresh page-native token",
    turnstile_guard: capture.turnstile_guard,
    timestep_sent: 0.02,
    timestep_required: 0.02,
    response_schema: schema,
    schema_pass: schema.pass,
    payload_contrast_pass: contrast.pass,
    excitation_gate_pass: gate.pass,
    r3n_gate_pass: gate.pass,
    r3n_gate: gate,
    r4_cleared: false,
    note: "One explicitly authorized native R3N Calculate reached the server. HTTP/schema/payload gates passed; strict one-second near-envelope coverage failed, so Plan 7 stops before fit/freeze and R4.",
  };
  ledger.actions.push(action);
  ledger.stage = "H7-R3N-executed-stop-after-excitation-gate";
  ledger.authorization_policy = "explicit_single_r3n_calculate_consumed_then_pause";
  ledger.current_authorized_server_actions = 0;
  ledger.used_turnstile_token_sha256 = [...new Set([...(ledger.used_turnstile_token_sha256 || []), tokenHash])];
  ledger.calculate_actions_used = 6;
  ledger.server_calculate_actions_used = 6;
  ledger.preflight_token_aborts = ledger.preflight_token_aborts || 0;
  ledger.status = "r3n_http_200_schema_pass_excitation_gate_failed";
  ledger.r1_cleared = true;
  ledger.r2_cleared = true;
  ledger.r3_schema_cleared = true;
  ledger.r3_excitation_gate_pass = false;
  ledger.r3b_cleared = false;
  ledger.r3n_design_ready = true;
  ledger.r3n_calculate_authorized = false;
  ledger.r3n_completed = true;
  ledger.r3n_schema_cleared = schema.pass;
  ledger.r3n_payload_contrast_pass = contrast.pass;
  ledger.r3n_excitation_gate_pass = gate.pass;
  ledger.r4_cleared = false;
  ledger.r3n_analysis_path = analysisPath;
  ledger.r3n_report_path = reportPath;
  ledger.r3n_capture_path = capture.capture_path || REL.capture;
  ledger.next_action = "pause_after_r3n_gate_failure_no_fit_no_r4";
  ledger.note = "Six server submissions are counted. R3N used the single fresh-token authorization; no controller fit/freeze, plant refit, authority sweep, R3B, or R4 is allowed after the failed strict near-envelope gate.";
  ledger.calculate_budget = {
    ...(ledger.calculate_budget || {}),
    current_authorized_server_actions: 0,
    already_counted_server_actions: 6,
    authorization_statement: "single R3N authorization consumed; R3N gate failed; pause",
    execution_constraint: "stop after R3N gate failure; do not fit/freeze controller or execute R4",
  };
  return ledger;
}

function updateConfig(config, { readback, capture, schema, contrast, gate, analysisPath, reportPath }) {
  const r3n = (config.calculate_actions || []).find((entry) => entry.case_id === CASE_ID);
  assert(r3n, "R3N config entry missing");
  r3n.authorization_status = "executed_once_authorized_then_stopped";
  r3n.execution_contract.current_authorized_calculate_actions = 0;
  r3n.execution_contract.authorization_consumed = true;
  r3n.execution_contract.actual_server_calculate_count = 1;
  r3n.models = readback.selected.map((arm) => ({
    name: arm.name,
    custom_id: arm.id,
    finsLatAccel: arm.parameters.finsLatAccel,
  }));
  r3n.result = {
    http_status: capture.http_status,
    response_nonempty: capture.response_nonempty,
    valid_json: capture.valid_json,
    result_count: schema.result_count,
    missile_ids_one_to_one: schema.missile_ids_one_to_one,
    required_arrays_pass: schema.required_arrays_pass,
    currentGain_lengths: schema.currentGain_lengths,
    currentGain_length_matches_times: schema.currentGain_length_matches_times,
    request_id: capture.request_id,
    postData_sha256: capture.postData_sha256,
    request_artifact_sha256: capture.request_artifact_sha256,
    response_artifact_sha256: capture.response_artifact_sha256,
    capture_artifact_sha256: capture.capture_artifact_sha256,
    token_sha256: capture.turnstile_guard?.token_sha256,
    token_unique: capture.turnstile_guard?.unique_against_prior_submitted_requests,
    payload_contrast_pass: contrast.pass,
    schema_pass: schema.pass,
    excitation_gate_pass: gate.pass,
    r3n_gate: gate,
    capture_path: REL.capture,
    schema_path: REL.schema,
    readback_path: REL.readback,
    payload_contrast_path: REL.contrast,
    analysis_path: analysisPath,
    report_path: reportPath,
    valid_for_analysis: schema.pass && contrast.pass,
    valid_for_controller_fit: false,
  };
  const r4 = (config.calculate_actions || []).find((entry) => entry.case_id === "H7_R4_FROZEN_HOLDOUT");
  if (r4) {
    r4.status = "blocked_r3n_excitation_gate_failed";
    r4.authorization_status = "not_executed";
  }
  config.status = "r3n_http_200_schema_pass_excitation_gate_failed_paused";
  config.authorization_policy = "single_explicit_r3n_calculate_consumed_then_pause";
  config.turnstile_policy.used_turnstile_token_sha256 = [
    ...new Set([...(config.turnstile_policy.used_turnstile_token_sha256 || []), capture.turnstile_guard?.token_sha256]),
  ];
  config.calculate_budget = {
    ...(config.calculate_budget || {}),
    already_counted_server_actions: 6,
    current_authorized_server_actions: 0,
    authorization_statement: "single R3N authorization consumed; strict R3N gate failed",
    execution_constraint: "pause after R3N gate failure; no controller fit/freeze, plant refit, authority sweep, R3B, or R4",
  };
  return config;
}

function updateManifest(manifest, { config, capture, readback, schema, contrast, gate, analysisPath, reportPath }) {
  manifest.stage = "H7-R3N-executed-stop-after-excitation-gate";
  manifest.generated_at_utc = utcNow();
  manifest.status = "r3n_http_200_schema_pass_excitation_gate_failed_paused";
  manifest.calculate_count = 6;
  manifest.authorization_policy = "single_explicit_r3n_calculate_consumed_then_pause";
  manifest.current_authorized_server_actions = 0;
  manifest.r3n_calculate_authorized = false;
  manifest.r3n_completed = true;
  manifest.r3n_schema_cleared = schema.pass;
  manifest.r3n_payload_contrast_pass = contrast.pass;
  manifest.r3n_excitation_gate_pass = gate.pass;
  manifest.r4_cleared = false;
  manifest.r3n_valid_capture = {
    case_id: CASE_ID,
    raw_capture_path: REL.capture,
    request_path: REL.request,
    response_path: REL.response,
    schema_path: REL.schema,
    readback_path: REL.readback,
    payload_contrast_path: REL.contrast,
    analysis_path: analysisPath,
    report_path: reportPath,
    http_status: capture.http_status,
    request_id: capture.request_id,
    postData_sha256: capture.postData_sha256,
    request_artifact_sha256: capture.request_artifact_sha256,
    response_artifact_sha256: capture.response_artifact_sha256,
    capture_artifact_sha256: capture.capture_artifact_sha256,
    token_sha256: capture.turnstile_guard?.token_sha256,
    token_unique: capture.turnstile_guard?.unique_against_prior_submitted_requests,
    response_schema_pass: schema.pass,
    payload_contrast_pass: contrast.pass,
    excitation_gate_pass: gate.pass,
    r3n_gate: gate,
    missileIds: schema.missile_ids,
    clone_ids: readback.selected.map((arm) => arm.id),
  };
  manifest.r3n_clone_count = readback.selected.length;
  manifest.r3n_clone_names = readback.selected.map((arm) => arm.name);
  manifest.r3n_custom_ids = readback.selected.map((arm) => arm.id);
  manifest.next_gate = "pause_after_r3n_gate_failure_no_fit_no_r4";
  const planKey = Object.keys(manifest.sources || {}).find((key) => key.toLowerCase().endsWith("plan7.md"));
  const configKey = Object.keys(manifest.sources || {}).find((key) => key.toLowerCase().endsWith("h7_controller_experiments.json"));
  if (planKey) manifest.sources[planKey].sha256 = sha256File(ABS.plan);
  if (configKey) {
    manifest.sources[configKey].sha256 = sha256File(ABS.config);
    manifest.sources[configKey].path = ABS.config;
  }
  return manifest;
}

function buildReport({ capture, schema, contrast, gate, perArm, analysisPath }) {
  const lines = [
    "# H7 R3N Near-Envelope Retry Report",
    "",
    `- Case: ${CASE_ID}`,
    "- Actual server Calculate count for this case: 1; no retry.",
    `- HTTP: ${capture.http_status}; nonempty response: ${capture.response_nonempty}; valid JSON: ${capture.valid_json}.`,
    `- Token guard: unique=${capture.turnstile_guard?.unique_against_prior_submitted_requests}; allowed unchanged=${capture.turnstile_guard?.request_allowed_to_server}; token plaintext written=false.`,
    `- Schema: ${schema.pass ? "PASS" : "FAIL"}; three results and one-to-one missileIds=${schema.missile_ids_one_to_one}.`,
    `- Payload contrast: ${contrast.pass ? "PASS" : "FAIL"}; the only normalized omission is page-local bulletName name metadata.`,
    "",
    "## Arms",
    "",
    "| Arm | finsLatAccel | strict >=0.95 windows | unsaturated windows | sign reversal | critical window |",
    "|---|---:|---|---|---|---|",
  ];
  for (const arm of perArm) {
    const strict = arm.strict_near_envelope_windows.map((window) => `${window.start_s}–${window.end_s}s (${window.duration_s}s)`).join(", ") || "none";
    const unsat = arm.unsaturated_windows.map((window) => `${window.start_s}–${window.end_s}s (${window.duration_s}s)`).join(", ") || "none";
    lines.push(`| ${arm.name} | ${arm.finsLatAccel} | ${strict} | ${unsat} | ${arm.sign_reversal_valid ? "PASS" : "FAIL"} | ${arm.critical_window_completed ? "PASS" : "FAIL"} |`);
  }
  lines.push(
    "",
    "## R3N gate",
    "",
    `- Low-authority strict near-envelope continuous >=1.0 s: ${gate.low_authority_near_envelope_pass ? "PASS" : "FAIL"}.`,
    `- F100 unsaturated continuous >=1.0 s: ${gate.f100_unsaturated_pass ? "PASS" : "FAIL"}.`,
    `- Valid sign reversal: ${gate.sign_reversal_pass ? "PASS" : "FAIL"}.`,
    `- Critical window complete: ${gate.critical_window_pass ? "PASS" : "FAIL"}.`,
    `- Final R3N status: **${gate.pass ? "PASS" : "FAIL"}**.`,
    "",
    "R3N failed only because the strict near-envelope one-second coverage was absent. Under Plan 7, controller fit/freeze, plant refit, authority sweep, R3B, and R4 are not executed.",
    "",
    `Analysis artifact: ${analysisPath}`,
  );
  return `${lines.join("\n")}\n`;
}

function main() {
  const capture = readJson(ABS.capture);
  const schema = readJson(ABS.schema);
  const request = readJson(ABS.request);
  const response = readJson(ABS.response);
  const readback = readJson(ABS.readback);
  const rawContrast = readJson(ABS.contrast);
  assert(capture.case_id === CASE_ID, "capture case mismatch");
  assert(schema.case_id === CASE_ID, "schema case mismatch");
  assert(request.case_id === CASE_ID, "request case mismatch");
  assert(response.results?.length === 3, "R3N response result count is not three");
  assert(readback.selected?.length === 3, "R3N readback arm count is not three");

  const finalContrast = normalizePayloadContrast(rawContrast, readback);
  assert(finalContrast.pass, `non-planned payload differences remain: ${JSON.stringify(finalContrast.differences)}`);
  writeJson(ABS.contrast, finalContrast);

  const perArm = response.results.map((row, index) => analyzeArm(row, {
    name: readback.selected[index].name,
    id: readback.selected[index].id,
    finsLatAccel: readback.selected[index].parameters.finsLatAccel,
  }));
  const lowAuthorityArms = perArm.filter((arm) => arm.strict_near_envelope_pass_1s);
  const f100 = perArm.find((arm) => arm.name === "H7_R3N_NOM_F100");
  const gate = {
    http_200: capture.http_status === 200,
    nonempty_valid_json: capture.response_nonempty && capture.valid_json,
    three_missile_ids_one_to_one: schema.result_count === 3 && schema.missile_ids_one_to_one,
    payload_contrast_pass: finalContrast.pass,
    schema_pass: schema.pass,
    low_authority_near_envelope_threshold: 0.95,
    low_authority_near_envelope_required_duration_s: 1.0,
    low_authority_near_envelope_arms: lowAuthorityArms.map((arm) => ({ name: arm.name, max_duration_s: arm.strict_near_envelope_max_duration_s })),
    low_authority_near_envelope_pass: lowAuthorityArms.length > 0,
    f100_unsaturated_required_duration_s: 1.0,
    f100_unsaturated_max_duration_s: f100?.unsaturated_max_duration_s ?? 0,
    f100_unsaturated_pass: Boolean(f100?.unsaturated_pass_1s),
    sign_reversal_count: perArm.filter((arm) => arm.sign_reversal_valid).length,
    sign_reversal_pass: perArm.some((arm) => arm.sign_reversal_valid),
    critical_window_pass: perArm.every((arm) => arm.critical_window_completed),
    pass: capture.http_status === 200 && capture.response_nonempty && capture.valid_json && schema.pass && finalContrast.pass && lowAuthorityArms.length > 0 && Boolean(f100?.unsaturated_pass_1s) && perArm.some((arm) => arm.sign_reversal_valid) && perArm.every((arm) => arm.critical_window_completed),
  };

  capture.payload_contrast_pass = finalContrast.pass;
  capture.payload_contrast_normalization = {
    allowed_name_metadata_omissions: finalContrast.allowed_name_metadata_omissions,
    policy: finalContrast.normalization_policy,
  };
  capture.r3n_gate = gate;
  capture.schema_pass = schema.pass;
  capture.capture_artifact_sha256 = null;
  const finalCapture = stampCanonicalCaptureArtifactHash(crypto, capture);
  writeJson(ABS.capture, finalCapture);

  const analysis = {
    schema_version: 2,
    case_id: CASE_ID,
    generated_at_utc: utcNow(),
    status: gate.pass ? "r3n_gate_pass" : "r3n_schema_pass_no_near_envelope_coverage",
    calculate_count_this_action: 1,
    raw_capture: {
      capture_path: REL.capture,
      request_path: REL.request,
      response_path: REL.response,
      schema_path: REL.schema,
      readback_path: REL.readback,
      payload_contrast_path: REL.contrast,
      postData_sha256: finalCapture.postData_sha256,
      request_artifact_sha256: finalCapture.request_artifact_sha256,
      response_artifact_sha256: finalCapture.response_artifact_sha256,
      capture_artifact_sha256: finalCapture.capture_artifact_sha256,
    },
    token_guard: finalCapture.turnstile_guard,
    schema,
    payload_contrast: finalContrast,
    arms: perArm,
    r3n_gate: gate,
    controller_fit_allowed: false,
    r4_allowed: false,
    stop_reason: "strict near-envelope coverage gate failed; Plan 7 requires immediate pause",
    source_hashes: finalCapture.source_hashes,
  };
  writeJson(ABS.analysis, analysis);
  const report = buildReport({ capture: finalCapture, schema, contrast: finalContrast, gate, perArm, analysisPath: ABS.analysis });
  writeText(ABS.report, report);

  let ledger = readJson(ABS.ledger);
  ledger = updateLedger({ ledger, capture: { ...finalCapture, capture_path: REL.capture }, schema, contrast: finalContrast, gate, analysisPath: ABS.analysis, reportPath: ABS.report, analysis });
  writeJson(ABS.ledger, ledger);

  let config = readJson(ABS.config);
  config = updateConfig(config, { readback, capture: finalCapture, schema, contrast: finalContrast, gate, analysisPath: ABS.analysis, reportPath: ABS.report });
  writeJson(ABS.config, config);

  let manifest = readJson(ABS.manifest);
  manifest = updateManifest(manifest, { config, capture: finalCapture, readback, schema, contrast: finalContrast, gate, analysisPath: ABS.analysis, reportPath: ABS.report });
  writeJson(ABS.manifest, manifest);

  writeJson(path.join(RAW_ROOT, `network_evidence/${CASE_ID}.hash_reconciliation.json`), {
    schema_version: 1,
    case_id: CASE_ID,
    generated_at_utc: utcNow(),
    request_file_sha256: sha256File(ABS.request),
    response_file_sha256: sha256File(ABS.response),
    capture_canonical_sha256: finalCapture.capture_artifact_sha256,
    capture_file_bytes_sha256: sha256File(ABS.capture),
    ledger_entry_sha256_references: {
      request_artifact_sha256: finalCapture.request_artifact_sha256,
      response_artifact_sha256: finalCapture.response_artifact_sha256,
      capture_artifact_sha256: finalCapture.capture_artifact_sha256,
    },
    payload_contrast_pass: finalContrast.pass,
    schema_pass: schema.pass,
  });

  console.log(JSON.stringify({
    case_id: CASE_ID,
    actual_server_calculate_count: 1,
    r3n_gate_pass: gate.pass,
    strict_near_envelope_pass: gate.low_authority_near_envelope_pass,
    f100_unsaturated_pass: gate.f100_unsaturated_pass,
    sign_reversal_pass: gate.sign_reversal_pass,
    critical_window_pass: gate.critical_window_pass,
    final_capture_artifact_sha256: finalCapture.capture_artifact_sha256,
    ledger_server_calculate_actions_used: ledger.server_calculate_actions_used,
    status: analysis.status,
  }, null, 2));
}

main();
