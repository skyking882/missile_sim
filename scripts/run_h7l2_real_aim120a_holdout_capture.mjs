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

const SATURATION_MODE = process.argv.includes("--saturation");
const MANUAL_CALCULATE_MODE = process.argv.includes("--manual-calculate");
const CASE_ID = SATURATION_MODE ? "H7L2_SATURATION_ENVELOPE_HOLDOUT_01" : "H7L2_REAL_AIM120A_HOLDOUT_01";
const BASE_CUSTOM_NAME = "H7_R3N_NOM_F100";
const CUSTOM_NAME = BASE_CUSTOM_NAME;
const CUSTOM_ID = "custom_1786717189852_ktmd9vbh7";
const EXPERIMENT_ARM_LABEL = "H7L2_SAT_A165";
const SATURATION_AUTHORITY_G = 16.5;
const PAGE_URL = "https://statshark.net/missilecalculator";
const SATURATION_TARGET_ID = process.env.H7L2_SATURATION_TARGET_ID || "";
const CALC_URL = "/api/missiles/CalcMissileRange";
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RAW_ROOT = path.join(PROJECT_ROOT, "data", "raw", SATURATION_MODE ? "statshark_h7_controller_lite_v2_saturation_holdout" : "statshark_h7_controller_lite_v2_holdout");
const H7_RAW_ROOT = path.join(PROJECT_ROOT, "data", "raw", "statshark_h7_controller_id");
const TOKEN_AUDIT_PATH = path.join(H7_RAW_ROOT, "network_evidence", "H7_TURNSTILE_TOKEN_AUDIT_20260814.json");
const PREDICTION_PATH = path.join(PROJECT_ROOT, "outputs", SATURATION_MODE ? "h7_controller_lite_v2_saturation_holdout" : "h7_controller_lite_v2_holdout", SATURATION_MODE ? "H7L2_SATURATION_HOLDOUT_PREDICTION.json" : "H7L2_REAL_AIM120A_HOLDOUT_PREDICTION.json");
const BASE_REQUEST_PATH = path.join(H7_RAW_ROOT, "requests", "H7_R3N_NEAR_ENVELOPE_RETRY.request.json");
const SCENE = {
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
const REL = {
  request: `requests/${CASE_ID}.request.json`,
  response: `responses/${CASE_ID}.response.json`,
  schema: `network_evidence/${CASE_ID}.schema.json`,
  capture: `network_evidence/${CASE_ID}.capture.json`,
  preflightAbort: `network_evidence/${CASE_ID}.preflight_abort.json`,
};

function utcNow() { return new Date().toISOString(); }
function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function assert(condition, message) { if (!condition) throw new Error(message); }
function sha256Bytes(value) { return crypto.createHash("sha256").update(value).digest("hex"); }
function sha256Text(value) { return sha256Bytes(Buffer.from(String(value), "utf8")); }
function sha256File(filePath) { return sha256Bytes(fs.readFileSync(filePath)); }
function readJson(filePath) { return JSON.parse(fs.readFileSync(filePath, "utf8")); }
function writeText(relativePath, value) {
  const filePath = path.join(RAW_ROOT, relativePath);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, value, "utf8");
  return filePath;
}
function writeJson(relativePath, value) { return writeText(relativePath, `${JSON.stringify(value, null, 2)}\n`); }
function redactHeaders(headers) {
  return Object.fromEntries(Object.entries(headers || {}).map(([name, value]) => [
    name,
    name.toLowerCase() === "x-turnstile-token" ? "[REDACTED]" : value,
  ]));
}
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function inspectPayload(postData) {
  const differences = [];
  let payload = null;
  try { payload = JSON.parse(postData); } catch { differences.push({ path: "$", reason: "postData is not valid JSON" }); }
  if (payload) {
    if (!Array.isArray(payload.Missiles) || payload.Missiles.length !== 1) {
      differences.push({ path: "$.Missiles.length", expected: 1, actual: payload.Missiles?.length ?? null });
    }
    const missileId = payload.Missiles?.[0];
    if (SATURATION_MODE) {
      if (missileId !== CUSTOM_ID) differences.push({ path: "$.Missiles[0]", expected: CUSTOM_ID, actual: missileId ?? null });
      const custom = Array.isArray(payload.CustomMissiles) ? payload.CustomMissiles : [];
      if (custom.length !== 1) differences.push({ path: "$.CustomMissiles.length", expected: 1, actual: custom.length });
      if (custom[0]?.Id !== CUSTOM_ID) differences.push({ path: "$.CustomMissiles[0].Id", expected: CUSTOM_ID, actual: custom[0]?.Id ?? null });
      const parameters = custom[0]?.Parameters || null;
      if (parameters?.finsLatAccel !== SATURATION_AUTHORITY_G) {
        differences.push({ path: "$.CustomMissiles[0].Parameters.finsLatAccel", expected: SATURATION_AUTHORITY_G, actual: parameters?.finsLatAccel ?? null });
      }
      const basePayload = JSON.parse(readJson(BASE_REQUEST_PATH).postData);
      const base = (basePayload.CustomMissiles || []).find((entry) => entry?.Parameters?.finsLatAccel === 42.2579);
      if (!base) {
        differences.push({ path: "$source", reason: "preserved F100 base parameters not found" });
      } else if (parameters) {
        const expectedParameters = JSON.parse(JSON.stringify(base.Parameters));
        expectedParameters.finsLatAccel = SATURATION_AUTHORITY_G;
        if (JSON.stringify(parameters) !== JSON.stringify(expectedParameters)) {
          differences.push({ path: "$.CustomMissiles[0].Parameters", expected: "base F100 parameters with only finsLatAccel changed", actual: "additional parameter difference" });
        }
      }
    } else {
      if (typeof missileId !== "string" || !/aim[_-]?120a/i.test(missileId) || /^custom_/i.test(missileId)) {
        differences.push({ path: "$.Missiles[0]", expected: "stock AIM-120A id", actual: missileId ?? null });
      }
      if (Array.isArray(payload.CustomMissiles) && payload.CustomMissiles.length !== 0) {
        differences.push({ path: "$.CustomMissiles.length", expected: 0, actual: payload.CustomMissiles.length });
      }
      if (payload.CustomMissiles != null && !Array.isArray(payload.CustomMissiles)) {
        differences.push({ path: "$.CustomMissiles", expected: "absent or empty array", actual: typeof payload.CustomMissiles });
      }
    }
    if (!Array.isArray(payload.version) || payload.version.length !== 1 || payload.version[0] !== "local") {
      differences.push({ path: "$.version", expected: ["local"], actual: payload.version ?? null });
    }
    for (const [key, expected] of Object.entries(SCENE)) {
      if (payload[key] !== expected) differences.push({ path: `$.${key}`, expected, actual: payload[key] });
    }
  }
  return {
    pass: differences.length === 0,
    missile_id: payload?.Missiles?.[0] ?? null,
    scene: payload ? Object.fromEntries(Object.keys(SCENE).map((key) => [key, payload[key]])) : null,
    differences,
    parsed: payload,
  };
}

function buildSchema(responseText, httpStatus, responseBytes) {
  let parsed = null;
  let validJson = false;
  try { parsed = JSON.parse(responseText); validJson = true; } catch {}
  const results = Array.isArray(parsed?.results) ? parsed.results : [];
  const missileIds = Array.isArray(parsed?.missileIds) ? parsed.missileIds : [];
  const required = ["times", "aCmdYaw", "currentG"];
  const requiredArraysPass = results.length === 1 && required.every((key) => Array.isArray(results[0]?.[key]));
  const sameIndexLengths = requiredArraysPass && results[0].times.length === results[0].aCmdYaw.length && results[0].times.length === results[0].currentG.length;
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
    required_arrays: required,
    required_arrays_pass: requiredArraysPass,
    same_index_lengths: sameIndexLengths,
    times_length: results[0]?.times?.length ?? null,
    missile_ids_one_to_one: missileIds.length === 1 && results.length === 1,
    pass: httpStatus === 200 && responseBytes > 0 && validJson && missileIds.length === 1 && results.length === 1 && requiredArraysPass && sameIndexLengths,
  };
}

async function findTarget() {
  const tabs = await (await fetch("http://127.0.0.1:9222/json/list")).json();
  const candidates = tabs.filter((entry) => entry.type === "page" && entry.url === PAGE_URL && entry.webSocketDebuggerUrl);
  if (SATURATION_MODE) {
    const saturationTarget = candidates.find((entry) => entry.id === SATURATION_TARGET_ID);
    if (saturationTarget) return saturationTarget;
  }
  const diagnostics = [];
  let emptyStockPage = null;
  for (const candidate of candidates) {
    const probe = await connectCdp(candidate);
    try {
      const state = await probe.evaluate(`(() => {
        const text = document.body?.innerText || "";
        const values = [...document.querySelectorAll("input")].map((el) => String(el.value));
        const requiredValues = ["1700", "6500", "900", "15000", "-15", "90", "3"];
        return {
          matches: text.includes("AIM-120A") && requiredValues.every((value) => values.includes(value)),
          hasAIM120A: text.includes("AIM-120A"),
          inputValues: values.slice(0, 30),
          readonlyValues: [...document.querySelectorAll("input[readonly]")].map((el) => el.value),
          inputs: [...document.querySelectorAll("input")].slice(0, 18).map((el, index) => ({
            index,
            type: el.type,
            name: el.name,
            placeholder: el.placeholder,
            ariaLabel: el.getAttribute("aria-label"),
            value: el.value,
            className: el.className,
          })),
          textPrefix: text.slice(0, 240),
        };
      })()`);
      diagnostics.push({ target_id: candidate.id, ...state });
      if (SATURATION_MODE && state.readonlyValues.includes(BASE_CUSTOM_NAME)) return candidate;
      if (state.matches) return candidate;
      const numberInputCount = state.inputs.filter((entry) => entry.type === "number").length;
      const hasOldH7Custom = state.inputValues.some((value) => value.startsWith("H7_R3N_"));
      if (!hasOldH7Custom && numberInputCount === 11) {
        emptyStockPage = candidate;
      }
    } finally {
      probe.ws.close();
    }
  }
  if (emptyStockPage) return emptyStockPage;
  throw new Error(`no StatShark page target matched the verified AIM-120A scene: ${JSON.stringify(diagnostics)}`);
}

async function setInputValue(evaluate, selector, value, emitKeys = false) {
  const changed = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    if (setter) setter.call(el, ${JSON.stringify(String(value))});
    else el.value = ${JSON.stringify(String(value))};
    el.focus();
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    ${emitKeys ? 'for (const type of ["keydown", "keypress", "keyup"]) el.dispatchEvent(new KeyboardEvent(type, { bubbles: true, key: "0", code: "Digit0" }));' : ""}
    ${emitKeys ? "" : "el.blur();"}
    return true;
  })()`);
  assert(changed, `input not found: ${selector}`);
}

async function dispatchKey(cdp, key, code, windowsVirtualKeyCode, modifiers = 0) {
  const base = { key, code, windowsVirtualKeyCode, nativeVirtualKeyCode: windowsVirtualKeyCode, modifiers };
  await cdp("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
  await cdp("Input.dispatchKeyEvent", { type: "keyUp", ...base });
}

async function typeTrustedText(cdp, evaluate, selector, value) {
  const focused = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    el.focus();
    el.select();
    return document.activeElement === el;
  })()`);
  assert(focused, `input could not be focused: ${selector}`);
  await dispatchKey(cdp, "a", "KeyA", 65, 2);
  await dispatchKey(cdp, "Backspace", "Backspace", 8);
  await cdp("Input.insertText", { text: String(value) });
}

async function prepareStockPage(cdp, evaluate) {
  let readback = await evaluate(`(() => ({
    text: document.body?.innerText || "",
    numberValues: [...document.querySelectorAll('input[type="number"]')].map((el) => el.value),
    textValues: [...document.querySelectorAll('input[type="text"]')].map((el) => el.value),
  }))()`);
  if (SATURATION_MODE) {
    let storedInterventionReady = await evaluate(`(() => {
      const records = JSON.parse(localStorage.getItem("custom_missiles") || "[]");
      const record = records.find((entry) => entry?.name === ${JSON.stringify(BASE_CUSTOM_NAME)} && entry?.id === ${JSON.stringify(CUSTOM_ID)});
      return Number(record?.parameters?.finsLatAccel) === ${SATURATION_AUTHORITY_G};
    })()`);
    if (!storedInterventionReady) {
      const basePayload = JSON.parse(readJson(BASE_REQUEST_PATH).postData);
      const preservedBase = (basePayload.CustomMissiles || []).find((entry) => entry?.Id === CUSTOM_ID);
      assert(preservedBase?.Parameters?.finsLatAccel === 42.2579, "preserved F100 request base was not found");
      const localPrepared = await evaluate(`(() => {
        const records = JSON.parse(localStorage.getItem("custom_missiles") || "[]");
        const base = records.find((entry) => entry?.name === ${JSON.stringify(BASE_CUSTOM_NAME)} && entry?.id === ${JSON.stringify(CUSTOM_ID)});
        if (!base) return { pass: false, reason: "base custom F100 record not found" };
        const candidate = JSON.parse(JSON.stringify(base));
        candidate.parameters = ${JSON.stringify(preservedBase.Parameters)};
        candidate.parameters.finsLatAccel = ${SATURATION_AUTHORITY_G};
        const next = records.filter((entry) => entry?.name !== ${JSON.stringify(BASE_CUSTOM_NAME)} && entry?.id !== ${JSON.stringify(CUSTOM_ID)} && entry?.name !== "H7L2_SAT_A165" && entry?.id !== "custom_h7l2_sat_a165_20260814");
        next.push(candidate);
        localStorage.setItem("custom_missiles", JSON.stringify(next));
        location.reload();
        return { pass: true };
      })()`);
      assert(localPrepared?.pass, `custom saturation preparation failed: ${JSON.stringify(localPrepared)}`);
      let ready = false;
      for (let attempt = 0; attempt < 120; attempt += 1) {
        ready = await evaluate(`document.readyState === "complete" && (localStorage.getItem("turnstile_token") || "").length > 0`);
        if (ready) break;
        await sleep(250);
      }
      assert(ready, "page or page-native token did not become ready after custom record reload");
      storedInterventionReady = true;
    }
    const challengeClear = await evaluate(`![...document.querySelectorAll('[role="dialog"]')].some((el) => /验证|Turnstile|challenge/i.test(el.innerText || ""))`);
    assert(challengeClear, "Turnstile verification is still open");
    let customSelected = await evaluate(`(() => {
      const selected = [...document.querySelectorAll('input[readonly]')].map((el) => el.value);
      return selected.length === 1 && selected[0] === ${JSON.stringify(CUSTOM_NAME)};
    })()`);
    if (!customSelected) {
      await typeTrustedText(cdp, evaluate, "input#mat-input-0", CUSTOM_NAME);
      await sleep(400);
      await dispatchKey(cdp, "ArrowDown", "ArrowDown", 40);
      await sleep(100);
      await dispatchKey(cdp, "Enter", "Enter", 13);
      await sleep(300);
      for (let attempt = 0; attempt < 40; attempt += 1) {
        const added = await evaluate(`(() => {
          const alreadySelected = [...document.querySelectorAll('input[readonly]')].some((el) => el.value === ${JSON.stringify(CUSTOM_NAME)});
          if (alreadySelected) return true;
          const button = [...document.querySelectorAll("button")].find((el) => !el.disabled && (el.innerText || "").trim() === "add");
          if (!button) return false;
          button.click();
          return true;
        })()`);
        if (added) await sleep(250);
        customSelected = await evaluate(`(() => {
          const selected = [...document.querySelectorAll('input[readonly]')].map((el) => el.value);
          return selected.length === 1 && selected[0] === ${JSON.stringify(CUSTOM_NAME)};
        })()`);
        if (customSelected) break;
        await sleep(250);
      }
    }
    if (!customSelected) {
      const diagnostic = await evaluate(`(() => ({
        customRecordNames: JSON.parse(localStorage.getItem("custom_missiles") || "[]").map((entry) => entry?.name).filter((name) => /H7L2|H7_R3N/.test(name || "")),
        searchValue: document.querySelector("input#mat-input-0")?.value ?? null,
        readonlyValues: [...document.querySelectorAll("input[readonly]")].map((el) => el.value),
        optionTexts: [...document.querySelectorAll('[role="option"], mat-option')].map((el) => (el.innerText || "").trim()).slice(0, 20),
        buttons: [...document.querySelectorAll("button")].map((el) => ({
          text: (el.innerText || "").trim(),
          disabled: el.disabled,
          className: el.className,
          ariaPressed: el.getAttribute("aria-pressed"),
        })).filter((entry) => entry.text),
      }))()`);
      throw new Error(`custom saturation arm was not selected: ${JSON.stringify(diagnostic)}`);
    }
    await sleep(300);
  } else if (!SATURATION_MODE && (!readback.textValues.includes("AIM-120A") || !(await evaluate(`[...document.querySelectorAll('input[readonly]')].some((el) => el.value === "AIM-120A")`)))) {
    await setInputValue(evaluate, 'input.mat-mdc-autocomplete-trigger:not(.search-input)', "");
    await sleep(100);
    await setInputValue(evaluate, 'input.mat-mdc-autocomplete-trigger:not(.search-input)', "AIM-120A", true);
    let selected = false;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      selected = await evaluate(`[...document.querySelectorAll('input[readonly]')].some((el) => el.value === "AIM-120A")`);
      if (selected) break;
      await sleep(250);
    }
    for (let attempt = 0; !selected && attempt < 20; attempt += 1) {
      selected = await evaluate(`(() => {
        const options = [...document.querySelectorAll('[role="option"], mat-option')];
        const option = options.find((el) => (el.innerText || "").trim() === "AIM-120A");
        if (!option) return false;
        option.click();
        return true;
      })()`);
      if (selected) break;
      await sleep(250);
    }
    assert(selected, "AIM-120A stock autocomplete option did not appear or select");
    await sleep(500);
    const readonlySelected = await evaluate(`[...document.querySelectorAll('input[readonly]')].some((el) => el.value === "AIM-120A")`);
    if (!readonlySelected) {
      const added = await evaluate(`(() => {
        const button = [...document.querySelectorAll("button")].find((el) => !el.disabled && (el.innerText || "").trim() === "add");
        if (!button) return false;
        button.click();
        return true;
      })()`);
      assert(added, "AIM-120A was chosen but the Add button was not enabled");
      await sleep(500);
    }
  }

  const values = [1700, 6500, 0, 0, 900, 6500, 15000, -15, 90, 3, 0];
  const count = await evaluate(`document.querySelectorAll('input[type="number"]').length`);
  assert(count === values.length, `expected ${values.length} scene inputs, found ${count}`);
  for (let index = 0; index < values.length; index += 1) {
    const changed = await evaluate(`(() => {
      const el = document.querySelectorAll('input[type="number"]')[${index}];
      if (!el) return false;
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      if (setter) setter.call(el, ${JSON.stringify(String(values[index]))});
      else el.value = ${JSON.stringify(String(values[index]))};
      el.focus();
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      el.blur();
      return true;
    })()`);
    assert(changed, `scene input ${index} was not found`);
  }
  readback = await evaluate(`(() => ({
    text: document.body?.innerText || "",
    numberValues: [...document.querySelectorAll('input[type="number"]')].map((el) => el.value),
    textValues: [...document.querySelectorAll('input[type="text"]')].map((el) => el.value),
  }))()`);
  const expectedSelectedName = SATURATION_MODE ? CUSTOM_NAME : "AIM-120A";
  const readonlySelected = await evaluate(`[...document.querySelectorAll('input[readonly]')].some((el) => el.value === ${JSON.stringify(SATURATION_MODE ? CUSTOM_NAME : "AIM-120A")})`);
  assert(readonlySelected, `${expectedSelectedName} was not selected after preparation`);
  assert(JSON.stringify(readback.numberValues) === JSON.stringify(values.map(String)), `scene readback mismatch: ${JSON.stringify(readback.numberValues)}`);
  return { selected: expectedSelectedName, scene_input_values: readback.numberValues, text_input_values: readback.textValues };
}

async function connectCdp(target) {
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let sequence = 0;
  const pending = new Map();
  const handlers = new Set();
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const callback = pending.get(message.id);
      pending.delete(message.id);
      callback(message);
      return;
    }
    for (const handler of handlers) handler(message);
  };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, (message) => message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result));
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const result = await cdp("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result?.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result?.result?.value;
  };
  return { ws, cdp, evaluate, onEvent(handler) { handlers.add(handler); return () => handlers.delete(handler); } };
}

async function run() {
  const calculateMode = process.argv.includes("--calculate") || MANUAL_CALCULATE_MODE;
  const prepareOnly = process.argv.includes("--prepare-only");
  const inspectSaturationUi = process.argv.includes("--inspect-saturation-ui");
  const inspectSaturationEditor = process.argv.includes("--inspect-saturation-editor");
  const restoreSaturationLocal = process.argv.includes("--restore-saturation-local");
  assert(calculateMode || prepareOnly || inspectSaturationUi || inspectSaturationEditor || restoreSaturationLocal, "use an inspect mode, --prepare-only, --calculate, --manual-calculate, or --restore-saturation-local");
  assert(!MANUAL_CALCULATE_MODE || SATURATION_MODE, "manual Calculate mode is only enabled for the saturation holdout");
  assert(fs.existsSync(PREDICTION_PATH), "frozen prediction artifact is missing");
  const prediction = readJson(PREDICTION_PATH);
  assert(prediction.status === (SATURATION_MODE ? "frozen_before_single_calculate" : "frozen_before_statshark_calculate"), "prediction was not frozen before Calculate");
  assert(prediction.authorization?.maximum_new_server_calculate_actions === 1, "prediction authorization is not exactly one action");

  const target = await findTarget();
  const session = await connectCdp(target);
  const { cdp, evaluate } = session;
  try {
    await cdp("Page.bringToFront");
    if (restoreSaturationLocal) {
      assert(SATURATION_MODE, "local saturation restore requires --saturation");
      const basePayload = JSON.parse(readJson(BASE_REQUEST_PATH).postData);
      const preservedBase = (basePayload.CustomMissiles || []).find((entry) => entry?.Id === CUSTOM_ID);
      assert(preservedBase?.Parameters?.finsLatAccel === 42.2579, "preserved F100 restore base was not found");
      const restored = await evaluate(`(() => {
        const records = JSON.parse(localStorage.getItem("custom_missiles") || "[]");
        const record = records.find((entry) => entry?.name === ${JSON.stringify(BASE_CUSTOM_NAME)} && entry?.id === ${JSON.stringify(CUSTOM_ID)});
        if (!record) return false;
        record.parameters = ${JSON.stringify(preservedBase.Parameters)};
        localStorage.setItem("custom_missiles", JSON.stringify(records));
        return Number(record.parameters.finsLatAccel) === 42.2579;
      })()`);
      assert(restored, "F100 local browser record was not restored");
      const readback = await evaluate(`(() => {
        const records = JSON.parse(localStorage.getItem("custom_missiles") || "[]");
        const record = records.find((entry) => entry?.name === ${JSON.stringify(BASE_CUSTOM_NAME)} && entry?.id === ${JSON.stringify(CUSTOM_ID)});
        return { name: record?.name ?? null, id: record?.id ?? null, finsLatAccel: record?.parameters?.finsLatAccel ?? null };
      })()`);
      console.log(JSON.stringify({ mode: "restore-saturation-local", readback, actual_server_calculate_count: 0 }, null, 2));
      return;
    }
    if (inspectSaturationUi) {
      const ui = await evaluate(`(() => ({
        readonlyValues: [...document.querySelectorAll("input[readonly]")].map((el) => el.value),
        buttons: [...document.querySelectorAll("button")].map((el, index) => ({ index, text: (el.innerText || "").trim(), disabled: el.disabled, className: el.className })).filter((entry) => /edit|close|编辑|关闭/.test(entry.text)),
        inputs: [...document.querySelectorAll("input")].map((el, index) => ({ index, id: el.id, type: el.type, value: el.value, readonly: el.readOnly, formControlName: el.getAttribute("formcontrolname") })),
      }))()`);
      console.log(JSON.stringify({ mode: "inspect-saturation-ui", target_id: target.id, ui, actual_server_calculate_count: 0 }, null, 2));
      return;
    }
    if (inspectSaturationEditor) {
      const opened = await evaluate(`(() => {
        const edits = [...document.querySelectorAll("button")].filter((el) => (el.innerText || "").trim() === "edit");
        if (edits.length < 3) return false;
        edits[2].click();
        return true;
      })()`);
      assert(opened, "F100 edit button was not available");
      await sleep(500);
      const editor = await evaluate(`(() => {
        const dialog = document.querySelector('[role="dialog"]') || document.querySelector('mat-dialog-container');
        return {
          dialogText: (dialog?.innerText || "").slice(0, 4000),
          inputs: [...(dialog?.querySelectorAll("input") || [])].map((el, index) => ({ index, id: el.id, type: el.type, value: el.value, formControlName: el.getAttribute("formcontrolname"), placeholder: el.placeholder })),
          buttons: [...(dialog?.querySelectorAll("button") || [])].map((el, index) => ({ index, text: (el.innerText || "").trim(), disabled: el.disabled })),
        };
      })()`);
      console.log(JSON.stringify({ mode: "inspect-saturation-editor", target_id: target.id, editor, actual_server_calculate_count: 0 }, null, 2));
      return;
    }
    const pageReadback = await prepareStockPage(cdp, evaluate);
    if (prepareOnly) {
      console.log(JSON.stringify({ mode: "prepare-only", target_id: target.id, page_readback: pageReadback, actual_server_calculate_count: 0 }, null, 2));
      return;
    }
    const selectedVisible = await evaluate(`(() => {
      const text = document.body?.innerText || "";
      const readonlySelected = [...document.querySelectorAll('input[readonly]')].some((el) => el.value === ${JSON.stringify(SATURATION_MODE ? CUSTOM_NAME : "AIM-120A")});
      return readonlySelected && (text.includes("Selected Missile") || text.includes("已选导弹"));
    })()`);
    assert(selectedVisible, `page readback does not show selected ${SATURATION_MODE ? CUSTOM_NAME : "stock AIM-120A"}`);

    const tokenAudit = readJson(TOKEN_AUDIT_PATH);
    const usedTokenHashes = new Set(tokenAudit.used_turnstile_token_sha256 || []);
    const priorCases = (tokenAudit.submitted_actions || []).map((entry) => ({ case_id: entry.case_id, token_sha256: entry.token_sha256 }));
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
      payloadInspection: null,
      guardDecision: null,
    };

    session.onEvent(async (message) => {
      try {
        if (message.method === "Network.requestWillBeSent" && message.params?.request?.url?.includes(CALC_URL)) {
          eventState.requestWillBeSent = message.params;
          requestDeferred.resolve(message.params);
        }
        if (message.method === "Fetch.requestPaused" && message.params?.request?.url?.includes(CALC_URL)) {
          eventState.requestPaused = message.params;
          const inspection = inspectPayload(message.params.request.postData || "");
          eventState.payloadInspection = inspection;
          if (!inspection.pass) {
            await cdp("Fetch.failRequest", { requestId: message.params.requestId, errorReason: "BlockedByClient" });
            const decision = { preflight_payload_abort: true, request_allowed_to_server: false, differences: inspection.differences };
            eventState.guardDecision = decision;
            pausedDeferred.resolve(decision);
            return;
          }
          const decision = await guard.handlePausedRequest(message, cdp);
          eventState.guardDecision = decision;
          pausedDeferred.resolve(decision);
        }
        if (message.method === "Network.responseReceived" && message.params?.response?.url?.includes(CALC_URL)) {
          eventState.responseReceived = message.params;
        }
        if (message.method === "Network.loadingFinished" && eventState.responseReceived?.requestId === message.params?.requestId) {
          eventState.loadingFinished = message.params;
          try { eventState.responseBody = await cdp("Network.getResponseBody", { requestId: message.params.requestId }); }
          catch (error) { eventState.responseBodyError = String(error?.message || error); }
          responseDeferred.resolve(true);
        }
      } catch (error) {
        pausedDeferred.reject(error);
        responseDeferred.reject(error);
      }
    });

    await enableH7CaptureDomains(cdp);
    await activatePageForCalculate(cdp);
    const clickReadyAtUtc = utcNow();
    const calculateButtonReady = await evaluate(`(() => {
      const button = document.querySelector("button.calculate-button") || [...document.querySelectorAll("button")].find((b) => /Calculate|计算/i.test((b.innerText || "").trim()));
      return !!button && !button.disabled;
    })()`);
    assert(calculateButtonReady, "native Calculate button was missing or disabled");
    if (MANUAL_CALCULATE_MODE) {
      console.log(JSON.stringify({
        mode: "capture-armed-waiting-for-one-manual-click",
        case_id: CASE_ID,
        target_id: target.id,
        armed_at_utc: clickReadyAtUtc,
        actual_server_calculate_count: 0,
      }));
    } else {
      const clicked = await evaluate(`(() => {
        const button = document.querySelector("button.calculate-button") || [...document.querySelectorAll("button")].find((b) => /Calculate|计算/i.test((b.innerText || "").trim()));
        if (!button || button.disabled) return false;
        button.click();
        return true;
      })()`);
      assert(clicked, "native Calculate button was missing or disabled");
    }

    const requestTimeoutMs = MANUAL_CALCULATE_MODE ? 180000 : 20000;
    const requestParams = await Promise.race([requestDeferred.promise, sleep(requestTimeoutMs).then(() => null)]);
    assert(requestParams, "no CalcMissileRange request observed after the authorized native click");
    const calculateAtUtc = Number.isFinite(requestParams.wallTime)
      ? new Date(requestParams.wallTime * 1000).toISOString()
      : clickReadyAtUtc;
    const guardDecision = await Promise.race([pausedDeferred.promise, sleep(10000).then(() => null)]);
    assert(guardDecision, "preflight guard did not receive the CalcMissileRange request");
    if (guardDecision.preflight_payload_abort || guardDecision.preflight_token_abort) {
      writeJson(REL.preflightAbort, {
        schema_version: 1,
        case_id: CASE_ID,
        status: guardDecision.preflight_payload_abort ? "preflight_payload_abort" : "preflight_token_abort",
        generated_at_utc: utcNow(),
        target_id: target.id,
        request_id: requestParams.requestId,
        payload_inspection: eventState.payloadInspection ? { ...eventState.payloadInspection, parsed: undefined } : null,
        turnstile_guard: guardDecision,
        actual_server_calculate_count: 0,
        automatic_retries: 0,
      });
      throw new Error(`preflight abort: ${JSON.stringify(guardDecision)}`);
    }

    await Promise.race([responseDeferred.promise, sleep(45000).then(() => null)]);
    const responseMeta = eventState.responseReceived?.response || {};
    const requestObject = requestParams.request || eventState.requestPaused?.request || {};
    const postData = requestObject.postData ?? eventState.requestPaused?.request?.postData ?? "";
    const responseBytes = eventState.responseBody?.base64Encoded
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
    const responsePath = path.join(RAW_ROOT, REL.response);
    fs.mkdirSync(path.dirname(responsePath), { recursive: true });
    fs.writeFileSync(responsePath, responseBytes);
    const responseArtifactSha256 = sha256File(responsePath);
    const schema = buildSchema(responseBytes.toString("utf8"), responseMeta.status ?? null, responseBytes.length);
    writeJson(REL.schema, schema);
    const capture = stampCanonicalCaptureArtifactHash(crypto, {
      schema_version: 3,
      case_id: CASE_ID,
      phase: SATURATION_MODE ? "Plan7-Lite-v2-saturation-envelope-holdout" : "Plan7-Lite-v2-stock-holdout",
      capture_method: "native_page_calculate_cdp_network_fetch_guard",
      request_id: requestParams.requestId,
      url: requestObject.url || responseMeta.url || null,
      http_status: responseMeta.status ?? null,
      response_bytes: responseBytes.length,
      response_nonempty: responseBytes.length > 0,
      request_path: REL.request,
      response_path: REL.response,
      schema_path: REL.schema,
      postData_sha256: requestDocument.postData_sha256,
      request_artifact_sha256: requestArtifactSha256,
      response_artifact_sha256: responseArtifactSha256,
      prediction_path: path.relative(PROJECT_ROOT, PREDICTION_PATH).split(path.sep).join("/"),
      prediction_artifact_sha256: sha256File(PREDICTION_PATH),
      payload_preflight: {
        pass: eventState.payloadInspection.pass,
        missile_id: eventState.payloadInspection.missile_id,
        scene: eventState.payloadInspection.scene,
        intervention: SATURATION_MODE ? { experiment_arm_label: EXPERIMENT_ARM_LABEL, selected_custom_name: CUSTOM_NAME, custom_id: CUSTOM_ID, finsLatAccel_g: SATURATION_AUTHORITY_G, only_parameter_change: "finsLatAccel" } : null,
        differences: [],
      },
      turnstile_guard: guardDecision,
      calculate_at_utc: calculateAtUtc,
      captured_at_utc: utcNow(),
      source_hashes: {
        model_fit_results_json: sha256File(path.join(PROJECT_ROOT, "outputs", "h7_controller_lite_v2", "fit_results.json")),
        plan7_lite_md: sha256File(path.join(PROJECT_ROOT, "..", "plan7_lite.md")),
        effective_controller_py: sha256File(path.join(PROJECT_ROOT, "src", "aim120_model", "effective_controller.py")),
        ...(SATURATION_MODE ? {
          saturation_holdout_plan_md: sha256File(path.join(PROJECT_ROOT, "outputs", "h7_controller_lite_v2_saturation_holdout", "H7L2_SATURATION_HOLDOUT_PLAN.md")),
          base_r3n_request_json: sha256File(BASE_REQUEST_PATH),
        } : {}),
        capture_contract_guard_mjs: sha256File(path.join(PROJECT_ROOT, "scripts", "h7_cdp_capture_guard.mjs")),
      },
      native_page_state: true,
      request_body_interception: false,
      direct_fetch_or_replay: false,
      token_plaintext_written: false,
      request_headers_redacted: true,
      event_gates: {
        requestWillBeSent: !!eventState.requestWillBeSent,
        requestPaused: !!eventState.requestPaused,
        responseReceived: !!eventState.responseReceived,
        loadingFinished: !!eventState.loadingFinished,
        response_body_read: !!eventState.responseBody && !eventState.responseBodyError,
        http_status_read: Number.isInteger(responseMeta.status),
        sha256_computed: true,
      },
      schema_pass: schema.pass,
      response_body_error: eventState.responseBodyError,
      capture_artifact_sha256: null,
    });
    writeJson(REL.capture, capture);

    tokenAudit.used_turnstile_token_sha256 = [...guard.usedTokenHashes];
    tokenAudit.submitted_server_request_count = Number(tokenAudit.submitted_server_request_count || 0) + 1;
    tokenAudit.generated_at_utc = utcNow();
    tokenAudit.submitted_actions = [...(tokenAudit.submitted_actions || []), {
      index: tokenAudit.submitted_server_request_count,
      case_id: CASE_ID,
      http_status: responseMeta.status ?? null,
      token_sha256: guardDecision.token_sha256,
    }];
    fs.writeFileSync(TOKEN_AUDIT_PATH, `${JSON.stringify(tokenAudit, null, 2)}\n`, "utf8");

    console.log(JSON.stringify({
      case_id: CASE_ID,
      actual_server_calculate_count: 1,
      automatic_retries: 0,
      request_id: requestParams.requestId,
      missile_id: eventState.payloadInspection.missile_id,
      http_status: responseMeta.status ?? null,
      response_bytes: responseBytes.length,
      token_guard: guardDecision,
      payload_preflight_pass: eventState.payloadInspection.pass,
      schema_pass: schema.pass,
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
