import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  activatePageForCalculate,
  createTurnstileGuard,
  enableH7CaptureDomains,
  stampCanonicalCaptureArtifactHash,
} from "./h7_cdp_capture_guard.mjs";

const CASE_ID = "H7_R3N_NEAR_ENVELOPE_RETRY";
const PAGE_URL = "https://statshark.net/missilecalculator";
const CALC_URL = "/api/missiles/CalcMissileRange";
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RAW_ROOT = path.join(PROJECT_ROOT, "data", "raw", "statshark_h7_controller_id");
const TOKEN_AUDIT_PATH = path.join(RAW_ROOT, "network_evidence", "H7_TURNSTILE_TOKEN_AUDIT_20260814.json");
const ARM_NAMES = [
  "H7_R3N_NOM_F0020",
  "H7_R3N_NOM_F0025",
  "H7_R3N_NOM_F100",
];
const EXPECTED_FINS = [0.845158, 1.0564475, 42.2579];
const SCENE = {
  startSpeed: 1700,
  launchAltitude: 6500,
  launchAngle: 0,
  launchYaw: 0,
  closureRate: 900,
  initialTargetDistance: 15000,
  targetAltitude: 6500,
  targetAzimuth: -15,
  targetCourse: 90,
  targetConstantGTurn: 3,
  targetVerticalCourse: 0,
};
const SCENE_FORM_NAMES = {
  StartSpeed: "startSpeed",
  LaunchAltitude: "launchAltitude",
  LaunchAngle: "launchAngle",
  LaunchYaw: "launchYaw",
  ClosureRate: "closureRate",
  InitialTargetDistance: "initialTargetDistance",
  TargetAltitude: "targetAltitude",
  TargetAzimuth: "targetAzimuth",
  TargetCourse: "targetCourse",
  TargetConstantGTurn: "targetConstantGTurn",
  TargetVerticalCourse: "targetVerticalCourse",
};

const REL = {
  request: `requests/${CASE_ID}.request.json`,
  response: `responses/${CASE_ID}.response.json`,
  capture: `network_evidence/${CASE_ID}.capture.json`,
  schema: `network_evidence/${CASE_ID}.schema.json`,
  readback: `model_snapshots/${CASE_ID}_READBACK_20260814.json`,
  preContrast: "payload_contrast_r3n_pre_calculate.json",
  contrast: "payload_contrast_r3n.json",
};

function utcNow() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sha256Bytes(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function sha256Text(value) {
  return sha256Bytes(Buffer.from(String(value), "utf8"));
}

function sha256File(filePath) {
  return sha256Bytes(fs.readFileSync(filePath));
}

function writeText(relativePath, value) {
  const filePath = path.join(RAW_ROOT, relativePath);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, value, "utf8");
  return filePath;
}

function writeJson(relativePath, value) {
  return writeText(relativePath, `${JSON.stringify(value, null, 2)}\n`);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function diffValues(a, b, pointer = "$") {
  if (sameJson(a, b)) return [];
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    return [{ path: pointer, left: a, right: b }];
  }
  if (Array.isArray(a) !== Array.isArray(b)) {
    return [{ path: pointer, left: a, right: b }];
  }
  if (Array.isArray(a)) {
    const out = [];
    const length = Math.max(a.length, b.length);
    for (let i = 0; i < length; i += 1) {
      out.push(...diffValues(a[i], b[i], `${pointer}[${i}]`));
    }
    return out;
  }
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const out = [];
  for (const key of [...keys].sort()) {
    out.push(...diffValues(a[key], b[key], `${pointer}.${key}`));
  }
  return out;
}

function parameterDiffExceptFins(left, right) {
  const diffs = [];
  const keys = new Set([...Object.keys(left || {}), ...Object.keys(right || {})]);
  for (const key of [...keys].sort()) {
    if (["finsLatAccel", "bulletName", "name", "serverName"].includes(key)) continue;
    diffs.push(...diffValues(left?.[key], right?.[key], `$.parameters.${key}`));
  }
  return diffs;
}

function redactHeaders(headers) {
  const out = {};
  for (const [name, value] of Object.entries(headers || {})) {
    out[name] = name.toLowerCase() === "x-turnstile-token" ? "[REDACTED]" : value;
  }
  return out;
}

function findHeader(headers, wanted) {
  for (const [name, value] of Object.entries(headers || {})) {
    if (name.toLowerCase() === wanted.toLowerCase()) return value == null ? "" : String(value);
  }
  return null;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function findTarget() {
  const tabs = await (await fetch("http://127.0.0.1:9222/json/list")).json();
  const target = tabs.find((entry) => entry.type === "page" && entry.url === PAGE_URL);
  assert(target?.webSocketDebuggerUrl, "exact StatShark page target not found");
  return target;
}

async function connectCdp(target) {
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let sequence = 0;
  const pending = new Map();
  const eventHandlers = new Set();
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const callback = pending.get(message.id);
      pending.delete(message.id);
      callback(message);
      return;
    }
    for (const handler of eventHandlers) handler(message);
  };
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, (message) => {
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const result = await cdp("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (result?.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result?.result?.value;
  };
  return {
    ws,
    cdp,
    evaluate,
    onEvent(handler) { eventHandlers.add(handler); return () => eventHandlers.delete(handler); },
  };
}

async function waitFor(evaluate, label, expression, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await sleep(250);
  }
  throw new Error(`timeout waiting for ${label}`);
}

async function setNativeInput(evaluate, selector, value, emitKeys = false) {
  const expression = `(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) setter.call(el, ${JSON.stringify(String(value))});
    else el.value = ${JSON.stringify(String(value))};
    el.focus();
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    ${emitKeys ? "for (const type of [\"keydown\", \"keypress\", \"keyup\"]) el.dispatchEvent(new KeyboardEvent(type, { bubbles: true, key: \"0\", code: \"Digit0\" }));" : ""}
    ${emitKeys ? "" : "el.blur();"}
    return true;
  })()`;
  assert(await evaluate(expression), `native input missing: ${selector}`);
}

async function preparePage(cdp, evaluate, target, skipReload = false) {
  const pageLoadedAtUtc = utcNow();
  await cdp("Page.enable");
  await cdp("Runtime.enable");
  await cdp("Page.bringToFront");
  if (!skipReload) await cdp("Page.reload", { ignoreCache: false });
  await waitFor(evaluate, "page ready and page-native token", `(() => document.readyState === "complete" && (localStorage.getItem("turnstile_token") || "").length > 0 && !!localStorage.getItem("turnstile_timestamp"))()` , 30000);
  await sleep(1000);
  await waitFor(evaluate, "Turnstile interstitial to clear", `(() => ![...document.querySelectorAll("[role=dialog]")].some(d => /验证|Turnstile|challenge/i.test(d.innerText || "")))()`, 30000);

  for (const name of ARM_NAMES) {
    await setNativeInput(evaluate, "input#mat-input-0", name, true);
    await waitFor(evaluate, `native selection ${name}`, `(() => [...document.querySelectorAll("input[readonly]")].some(input => input.value === ${JSON.stringify(name)}))()`, 10000);
  }

  const sceneEntries = Object.entries(SCENE_FORM_NAMES);
  for (const [payloadName, formName] of sceneEntries) {
    await setNativeInput(evaluate, `[role="main"] input[formcontrolname="${formName}"], input[formcontrolname="${formName}"]`, SCENE[formName]);
  }

  const readbackAtUtc = utcNow();
  const readbackText = await evaluate(`(() => {
    const selectedNames = [...document.querySelectorAll("input[readonly]")].map(input => input.value).filter(Boolean);
    const custom = JSON.parse(localStorage.getItem("custom_missiles") || "[]");
    const selected = selectedNames.map(name => {
      const record = custom.find(item => item.name === name);
      return record ? { name: record.name, id: record.id, serverName: record.serverName, parameters: record.parameters } : null;
    });
    const scene = {};
    for (const [payloadName, formName] of Object.entries(${JSON.stringify(SCENE_FORM_NAMES)})) {
      const input = document.querySelector('input[formcontrolname="' + formName + '"]');
      scene[payloadName] = input ? Number(input.value) : null;
    }
    return JSON.stringify({
      schema_version: 1,
      case_id: ${JSON.stringify(CASE_ID)},
      page_url: location.href,
      target_id: ${JSON.stringify(target.id)},
      page_loaded_at_utc: ${JSON.stringify(pageLoadedAtUtc)},
      final_readback_at_utc: ${JSON.stringify(readbackAtUtc)},
      selected_names: selectedNames,
      selected,
      scene,
      turnstile_page_state: {
        token_present: (localStorage.getItem("turnstile_token") || "").length > 0,
        token_length: (localStorage.getItem("turnstile_token") || "").length,
        token_timestamp: localStorage.getItem("turnstile_timestamp") || null,
        token_plaintext_written: false
      }
    });
  })()`);
  const readback = JSON.parse(readbackText);
  assert(sameJson(readback.selected_names, ARM_NAMES), `selected arms mismatch: ${JSON.stringify(readback.selected_names)}`);
  assert(readback.selected.every(Boolean), "one or more selected arms has no localStorage record");
  for (let i = 0; i < ARM_NAMES.length; i += 1) {
    assert(readback.selected[i].name === ARM_NAMES[i], `readback name mismatch at arm ${i}`);
    assert(Number(readback.selected[i].parameters?.finsLatAccel) === EXPECTED_FINS[i], `finsLatAccel mismatch at arm ${i}`);
  }
  for (const [key, expected] of Object.entries({
    StartSpeed: SCENE.startSpeed,
    LaunchAltitude: SCENE.launchAltitude,
    LaunchAngle: SCENE.launchAngle,
    LaunchYaw: SCENE.launchYaw,
    ClosureRate: SCENE.closureRate,
    InitialTargetDistance: SCENE.initialTargetDistance,
    TargetAltitude: SCENE.targetAltitude,
    TargetAzimuth: SCENE.targetAzimuth,
    TargetCourse: SCENE.targetCourse,
    TargetConstantGTurn: SCENE.targetConstantGTurn,
    TargetVerticalCourse: SCENE.targetVerticalCourse,
  })) {
    assert(readback.scene[key] === expected, `scene readback mismatch ${key}: ${readback.scene[key]} != ${expected}`);
  }

  const nonPlannedDifferences = [];
  for (let i = 1; i < readback.selected.length; i += 1) {
    nonPlannedDifferences.push(...parameterDiffExceptFins(readback.selected[0].parameters, readback.selected[i].parameters).map((entry) => ({
      arm_a: readback.selected[0].name,
      arm_b: readback.selected[i].name,
      ...entry,
    })));
  }
  const payloadContrast = {
    schema_version: 1,
    case_id: CASE_ID,
    stage: "pre_calculate_final_readback",
    generated_at_utc: utcNow(),
    page_target: { id: target.id, url: target.url },
    selected_arms: readback.selected.map(({ name, id, serverName, parameters }) => ({ name, id, serverName, finsLatAccel: parameters.finsLatAccel })),
    scene: readback.scene,
    allowed_cross_arm_differences: ["custom missile name", "custom missile id/serverName", "parameters.bulletName as name metadata", "parameters.finsLatAccel"],
    non_planned_differences: nonPlannedDifferences,
    pass: nonPlannedDifferences.length === 0,
  };
  assert(payloadContrast.pass, `non-planned cross-arm differences: ${JSON.stringify(nonPlannedDifferences)}`);
  writeJson(REL.readback, readback);
  writeJson(REL.preContrast, payloadContrast);
  return { pageLoadedAtUtc, readbackAtUtc, readback, payloadContrast };
}

function buildSchema(responseText, httpStatus, responseBytes) {
  let parsed = null;
  let validJson = false;
  try { parsed = JSON.parse(responseText); validJson = true; } catch {}
  const results = Array.isArray(parsed?.results) ? parsed.results : [];
  const missileIds = Array.isArray(parsed?.missileIds) ? parsed.missileIds : [];
  const required = ["times", "aCmdYaw", "currentG", "currentGain"];
  const requiredArraysPass = results.length > 0 && results.every((row) => required.every((key) => Array.isArray(row?.[key])));
  const lengths = results.map((row) => row?.currentGain?.length ?? null);
  const timesLengths = results.map((row) => row?.times?.length ?? null);
  const sameIdCount = new Set(missileIds).size === missileIds.length;
  return {
    schema_version: 1,
    case_id: CASE_ID,
    http_status: httpStatus,
    response_bytes: responseBytes,
    response_nonempty: responseBytes > 0,
    valid_json: validJson,
    root_keys: parsed && typeof parsed === "object" ? Object.keys(parsed) : [],
    result_count: results.length,
    missile_ids: missileIds,
    result_array_keys: results.map((row) => Object.keys(row || {})),
    required_arrays_pass: requiredArraysPass,
    currentGain_lengths: lengths,
    times_lengths: timesLengths,
    currentGain_length_matches_times: requiredArraysPass && results.every((row) => row.currentGain.length === row.times.length),
    missile_ids_one_to_one: missileIds.length === results.length && sameIdCount,
    required_result_count: 3,
    required_currentGain_length: null,
    currentGain_length_policy: "R3N requires same-index currentGain length to match times; no fixed length is imposed",
    pass: httpStatus === 200 && responseBytes > 0 && validJson && results.length === 3 && missileIds.length === 3 && missileIds.length === results.length && sameIdCount && requiredArraysPass && results.every((row) => row.currentGain.length === row.times.length),
  };
}

function contrastRequestPayload(requestText, readback, schema) {
  let payload = null;
  let validJson = false;
  try { payload = JSON.parse(requestText); validJson = true; } catch {}
  const expectedIds = readback.selected.map((arm) => arm.id);
  const diffs = [];
  if (!validJson) diffs.push({ path: "$", reason: "request postData is not valid JSON" });
  if (validJson && !sameJson(payload.Missiles, expectedIds)) diffs.push({ path: "$.Missiles", expected: expectedIds, actual: payload.Missiles });
  const requestedCustom = Array.isArray(payload?.CustomMissiles) ? payload.CustomMissiles : [];
  if (requestedCustom.length !== expectedIds.length) diffs.push({ path: "$.CustomMissiles.length", expected: expectedIds.length, actual: requestedCustom.length });
  for (const arm of readback.selected) {
    const item = requestedCustom.find((entry) => entry?.Id === arm.id);
    if (!item) {
      diffs.push({ path: `$.CustomMissiles[Id=${arm.id}]`, reason: "missing custom model" });
      continue;
    }
    diffs.push(...diffValues(item.Parameters, arm.parameters, `$.CustomMissiles[Id=${arm.id}].Parameters`));
  }
  const expectedScene = {
    StartSpeed: 1700,
    LaunchAltitude: 6500,
    LaunchAngle: 0,
    LaunchYaw: 0,
    ClosureRate: 900,
    InitialTargetDistance: 15000,
    TargetAltitude: 6500,
    TargetAzimuth: -15,
    TargetCourse: 90,
    TargetConstantGTurn: 3,
    TargetVerticalCourse: 0,
    Timestep: 0.02,
  };
  for (const [key, value] of Object.entries(expectedScene)) {
    if (payload?.[key] !== value) diffs.push({ path: `$.${key}`, expected: value, actual: payload?.[key] });
  }
  return {
    schema_version: 2,
    case_id: CASE_ID,
    stage: "post_request_payload_contrast",
    generated_at_utc: utcNow(),
    request_json_valid: validJson,
    request_missile_ids: payload?.Missiles ?? null,
    request_custom_ids: requestedCustom.map((entry) => entry?.Id ?? null),
    response_missile_ids: schema.missile_ids,
    expected_missile_ids: expectedIds,
    planned_cross_arm_difference: "only Parameters.finsLatAccel; custom name/id are identity metadata",
    differences: diffs,
    pass: diffs.length === 0,
  };
}

async function run() {
  const calculateMode = process.argv.includes("--calculate");
  const target = await findTarget();
  const session = await connectCdp(target);
  const { cdp, evaluate } = session;
  try {
    const prepared = await preparePage(cdp, evaluate, target, process.argv.includes("--from-current"));
    if (!calculateMode) {
      console.log(JSON.stringify({ mode: "prepare-only", case_id: CASE_ID, target_id: target.id, selected: prepared.readback.selected_names, scene: prepared.readback.scene, payload_diff_pass: prepared.payloadContrast.pass }, null, 2));
      return;
    }

    console.log("r3n-capture: prepared");
    const usedTokenHashes = new Set(readJson(TOKEN_AUDIT_PATH).used_turnstile_token_sha256 || []);
    const priorCases = (readJson(TOKEN_AUDIT_PATH).submitted_actions || []).map((entry) => ({ case_id: entry.case_id, token_sha256: entry.token_sha256 }));
    const guard = createTurnstileGuard({ cryptoImpl: crypto, usedTokenHashes, priorCases });
    const requestDeferred = deferred();
    const pausedDeferred = deferred();
    const responseDeferred = deferred();
    const eventState = {
      requestWillBeSent: null,
      requestPaused: null,
      responseReceived: null,
      loadingFinished: null,
      responseBody: null,
      responseBodyError: null,
      guardError: null,
    };
    session.onEvent(async (message) => {
      try {
        if (message.method === "Network.requestWillBeSent" && message.params?.request?.url?.includes(CALC_URL)) {
          eventState.requestWillBeSent = message.params;
          requestDeferred.resolve(message.params);
        }
        if (message.method === "Network.responseReceived" && message.params?.response?.url?.includes(CALC_URL)) {
          eventState.responseReceived = message.params;
        }
        if (message.method === "Fetch.requestPaused" && message.params?.request?.url?.includes(CALC_URL)) {
          eventState.requestPaused = message.params;
          const decision = await guard.handlePausedRequest(message, cdp);
          pausedDeferred.resolve(decision);
        }
        if (message.method === "Network.loadingFinished" && eventState.responseReceived?.requestId === message.params?.requestId) {
          eventState.loadingFinished = message.params;
          try {
            eventState.responseBody = await cdp("Network.getResponseBody", { requestId: message.params.requestId });
          } catch (error) {
            eventState.responseBodyError = String(error?.message || error);
          }
          responseDeferred.resolve(true);
        }
      } catch (error) {
        eventState.guardError = String(error?.message || error);
        pausedDeferred.reject(error);
      }
    });

    console.log("r3n-capture: enabling domains");
    await enableH7CaptureDomains(cdp);
    console.log("r3n-capture: activating page");
    await activatePageForCalculate(cdp);
    const clickReadyAtUtc = utcNow();
    console.log("r3n-capture: evaluating native click");
    const clicked = await evaluate(`(() => {
      const button = document.querySelector("button.calculate-button") || [...document.querySelectorAll("button")].find(b => /计算/.test((b.innerText || "").trim()));
      if (!button || button.disabled) return false;
      button.click();
      return true;
    })()`);
    assert(clicked, "native Calculate button was not available or was disabled");

    let requestParams;
    try {
      requestParams = await Promise.race([requestDeferred.promise, sleep(30000).then(() => null)]);
    } catch (error) {
      throw error;
    }
    assert(requestParams, "no CalcMissileRange request observed after the single native click");
    let guardDecision;
    try {
      guardDecision = await Promise.race([pausedDeferred.promise, sleep(10000).then(() => null)]);
    } catch (error) {
      throw error;
    }
    assert(guardDecision, "Fetch token guard did not receive the CalcMissileRange request");

    if (guardDecision.preflight_token_abort) {
      const abort = {
        schema_version: 1,
        case_id: CASE_ID,
        status: "preflight_token_abort",
        generated_at_utc: utcNow(),
        target_id: target.id,
        request_id: requestParams.requestId,
        turnstile_guard: guardDecision,
        actual_server_calculate_count: 0,
        note: "Fetch.failRequest was used; no retry is permitted.",
      };
      writeJson(`network_evidence/${CASE_ID}.preflight_abort.json`, abort);
      throw new Error(`preflight token abort: ${JSON.stringify(guardDecision)}`);
    }

    await Promise.race([responseDeferred.promise, sleep(90000).then(() => null)]);
    const responseMeta = eventState.responseReceived?.response || {};
    const requestObject = requestParams.request || eventState.requestPaused?.request || {};
    const postData = requestObject.postData ?? eventState.requestPaused?.request?.postData ?? "";
    const responseBodyText = eventState.responseBody?.base64Encoded
      ? Buffer.from(eventState.responseBody.body || "", "base64")
      : Buffer.from(eventState.responseBody?.body ?? "", "utf8");
    const requestDocument = {
      schema_version: 3,
      case_id: CASE_ID,
      request_id: requestParams.requestId,
      url: requestObject.url || responseMeta.url || null,
      method: requestObject.method || "POST",
      headers: redactHeaders(requestObject.headers || eventState.requestPaused?.request?.headers || {}),
      postData,
      postData_sha256: sha256Text(postData),
      token_plaintext_written: false,
      request_artifact_sha256: null,
    };
    writeJson(REL.request, requestDocument);
    const requestArtifactSha256 = sha256File(path.join(RAW_ROOT, REL.request));
    fs.mkdirSync(path.dirname(path.join(RAW_ROOT, REL.response)), { recursive: true });
    fs.writeFileSync(path.join(RAW_ROOT, REL.response), responseBodyText);
    const responseArtifactSha256 = sha256File(path.join(RAW_ROOT, REL.response));
    const responseText = responseBodyText.toString("utf8");
    const schema = buildSchema(responseText, responseMeta.status ?? null, responseBodyText.length);
    writeJson(REL.schema, schema);
    const payloadContrast = contrastRequestPayload(postData, prepared.readback, schema);
    writeJson(REL.contrast, payloadContrast);
    const captureBeforeStamp = {
      schema_version: 3,
      case_id: CASE_ID,
      phase: "R3N",
      capture_method: "native_page_calculate_cdp_network_fetch_guard",
      request_id: requestParams.requestId,
      url: requestObject.url || responseMeta.url || null,
      http_status: responseMeta.status ?? null,
      response_bytes: responseBodyText.length,
      response_nonempty: responseBodyText.length > 0,
      valid_json: schema.valid_json,
      request_path: REL.request,
      response_path: REL.response,
      schema_path: REL.schema,
      payload_contrast_path: REL.contrast,
      readback_path: REL.readback,
      postData_sha256: requestDocument.postData_sha256,
      request_artifact_sha256: requestArtifactSha256,
      response_artifact_sha256: responseArtifactSha256,
      old_token_replay: guardDecision.unique_against_prior_submitted_requests === false,
      turnstile_guard: guardDecision,
      page_loaded_at_utc: prepared.pageLoadedAtUtc,
      final_readback_at_utc: prepared.readbackAtUtc,
      calculate_at_utc: clickReadyAtUtc,
      source_hashes: {
        plan7_md: sha256File(path.join(PROJECT_ROOT, "..", "plan7.md")),
        config_json: sha256File(path.join(PROJECT_ROOT, "configs", "h7_controller_experiments.json")),
        effective_yaw_plant_fit_json: sha256File(path.join(PROJECT_ROOT, "outputs", "h6_fin_dynamics_recovery", "effective_yaw_plant_fit.json")),
        capture_contract_guard_mjs: sha256File(path.join(PROJECT_ROOT, "scripts", "h7_cdp_capture_guard.mjs")),
      },
      network: {
        fetch_request_id: eventState.requestPaused?.requestId ?? null,
        network_request_id: eventState.requestWillBeSent?.requestId ?? requestParams.requestId,
        response_received_at_utc: eventState.responseReceived ? utcNow() : null,
        loading_finished_at_utc: eventState.loadingFinished ? utcNow() : null,
      },
      native_page_state: true,
      request_body_interception: false,
      direct_fetch_or_replay: false,
      token_plaintext_written: false,
      request_headers_redacted: true,
      event_gates: {
        requestWillBeSent: !!eventState.requestWillBeSent,
        responseReceived: !!eventState.responseReceived,
        loadingFinished: !!eventState.loadingFinished,
        response_body_read: !!eventState.responseBody && !eventState.responseBodyError,
        http_status_read: Number.isInteger(responseMeta.status),
        sha256_computed: true,
      },
      response_schema: schema,
      schema_pass: schema.pass,
      payload_contrast_pass: payloadContrast.pass,
      response_body_error: eventState.responseBodyError,
      click_ready_at_utc: clickReadyAtUtc,
    };
    const capture = stampCanonicalCaptureArtifactHash(crypto, captureBeforeStamp);
    writeJson(REL.capture, capture);
    const tokenAudit = readJson(TOKEN_AUDIT_PATH);
    tokenAudit.used_turnstile_token_sha256 = [...guard.usedTokenHashes];
    tokenAudit.submitted_server_request_count = Number(tokenAudit.submitted_server_request_count || 0) + 1;
    tokenAudit.generated_at_utc = utcNow();
    tokenAudit.submitted_actions = [...(tokenAudit.submitted_actions || []), {
      index: tokenAudit.submitted_server_request_count,
      case_id: CASE_ID,
      http_status: responseMeta.status ?? null,
      token_sha256: guardDecision.token_sha256,
    }];
    writeJson("network_evidence/H7_TURNSTILE_TOKEN_AUDIT_20260814.json", tokenAudit);
    console.log(JSON.stringify({
      mode: "calculate",
      case_id: CASE_ID,
      actual_server_calculate_count: 1,
      target_id: target.id,
      request_id: requestParams.requestId,
      http_status: responseMeta.status ?? null,
      response_bytes: responseBodyText.length,
      token_guard: guardDecision,
      schema_pass: schema.pass,
      payload_contrast_pass: payloadContrast.pass,
      request_artifact_sha256: requestArtifactSha256,
      response_artifact_sha256: responseArtifactSha256,
      capture_artifact_sha256: capture.capture_artifact_sha256,
      paths: REL,
    }, null, 2));
  } finally {
    session.ws.close();
  }
}

run().catch((error) => {
  console.error(error?.stack || String(error));
  process.exitCode = 1;
});
