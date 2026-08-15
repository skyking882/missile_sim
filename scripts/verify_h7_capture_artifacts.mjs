import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { verifyCanonicalCaptureArtifactHash } from "./h7_cdp_capture_guard.mjs";


export const LEGACY_EXACT_FILE_POLICY = "legacy_exact_file";
export const CANONICAL_NULL_FIELD_POLICY = "canonical_null_field";

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

export function determineCaptureHashPolicy(capture) {
  const schemaVersion = capture?.schema_version;
  const hasCaptureHash = hasOwn(capture, "capture_artifact_sha256");

  if (schemaVersion === 1 && !hasCaptureHash) {
    return {
      pass: true,
      name: LEGACY_EXACT_FILE_POLICY,
      schema_version: schemaVersion,
      capture_hash_mode: "capture_file_bytes",
      request_path_field: "request_path",
      response_path_field: "response_path",
      canonical_capture_hash: false,
    };
  }

  if ((schemaVersion === 2 || schemaVersion === 3) && hasCaptureHash) {
    return {
      pass: true,
      name: CANONICAL_NULL_FIELD_POLICY,
      schema_version: schemaVersion,
      capture_hash_mode: "null_field_canonical_json",
      request_path_field: schemaVersion === 2 ? "request_artifact_path" : "request_path",
      response_path_field: schemaVersion === 2 ? "response_artifact_path" : "response_path",
      canonical_capture_hash: true,
    };
  }

  return {
    pass: false,
    name: "unsupported_schema_hash_pair",
    schema_version: schemaVersion ?? null,
    capture_hash_mode: null,
    request_path_field: null,
    response_path_field: null,
    canonical_capture_hash: false,
    reason:
      "schema v1 must omit capture_artifact_sha256; schema v2/v3 must include it",
  };
}

export function exactFileSha256(filePath, cryptoImpl = crypto) {
  return cryptoImpl.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function tryExactFileSha256(filePath, cryptoImpl = crypto) {
  try {
    return { hash: exactFileSha256(filePath, cryptoImpl), error: null };
  } catch (error) {
    return { hash: null, error: String(error?.message || error) };
  }
}

export function resolveRawArtifact(relativePath, rawRoot) {
  if (typeof relativePath !== "string" || relativePath.length === 0) {
    throw new Error("artifact path is missing");
  }
  const resolvedRoot = path.resolve(rawRoot);
  const resolved = path.resolve(resolvedRoot, relativePath);
  const prefix = `${resolvedRoot}${path.sep}`;
  if (!resolved.startsWith(prefix)) {
    throw new Error(`artifact path escapes H7 raw root: ${relativePath}`);
  }
  return resolved;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function collectCaptureHashReferences(node, targetCaseId, inheritedCaseId = null, pointer = "$", out = []) {
  if (node == null || typeof node !== "object") return out;
  const currentCaseId = typeof node.case_id === "string" ? node.case_id : inheritedCaseId;
  if (
    currentCaseId === targetCaseId &&
    Object.prototype.hasOwnProperty.call(node, "capture_artifact_sha256")
  ) {
    out.push({ pointer, value: node.capture_artifact_sha256 });
  }
  if (Array.isArray(node)) {
    node.forEach((child, index) =>
      collectCaptureHashReferences(child, targetCaseId, currentCaseId, `${pointer}[${index}]`, out));
  } else {
    for (const [key, child] of Object.entries(node)) {
      if (child != null && typeof child === "object") {
        collectCaptureHashReferences(child, targetCaseId, currentCaseId, `${pointer}.${key}`, out);
      }
    }
  }
  return out;
}

function failedPathCheck(pathField, relativePath, error) {
  return {
    pass: false,
    path_field: pathField,
    relative_path: relativePath ?? null,
    computed: null,
    error,
  };
}

export function verifyCaptureArtifactHashes({
  capturePath,
  capture,
  ledgerEntry,
  artifactRoot,
  referenceDocuments = [],
  cryptoImpl = crypto,
}) {
  const policy = determineCaptureHashPolicy(capture);
  if (!policy.pass) {
    return {
      schema_version: capture?.schema_version ?? null,
      policy: policy.name,
      path_fields: null,
      checks: {
        schema_policy: policy,
      },
      pass: false,
    };
  }

  const requestPath = capture[policy.request_path_field];
  const responsePath = capture[policy.response_path_field];
  let requestCheck;
  let responseCheck;
  try {
    const resolved = resolveRawArtifact(requestPath, artifactRoot);
    const fileCheck = tryExactFileSha256(resolved, cryptoImpl);
    const ledgerHash = ledgerEntry?.request_artifact_sha256 ?? null;
    const captureHash = policy.canonical_capture_hash
      ? capture.request_artifact_sha256 ?? null
      : null;
    requestCheck = {
      pass:
        fileCheck.hash !== null &&
        fileCheck.hash === ledgerHash &&
        (!policy.canonical_capture_hash || fileCheck.hash === captureHash),
      policy: "exact_file_bytes",
      path_field: policy.request_path_field,
      relative_path: requestPath ?? null,
      recorded_in_capture: captureHash,
      recorded_in_ledger: ledgerHash,
      computed: fileCheck.hash,
      error: fileCheck.error,
    };
  } catch (error) {
    requestCheck = failedPathCheck(policy.request_path_field, requestPath, String(error?.message || error));
  }

  try {
    const resolved = resolveRawArtifact(responsePath, artifactRoot);
    const fileCheck = tryExactFileSha256(resolved, cryptoImpl);
    const ledgerHash = ledgerEntry?.response_artifact_sha256 ?? null;
    const captureHash = policy.canonical_capture_hash
      ? capture.response_artifact_sha256 ?? null
      : null;
    responseCheck = {
      pass:
        fileCheck.hash !== null &&
        fileCheck.hash === ledgerHash &&
        (!policy.canonical_capture_hash || fileCheck.hash === captureHash),
      policy: "exact_file_bytes",
      path_field: policy.response_path_field,
      relative_path: responsePath ?? null,
      recorded_in_capture: captureHash,
      recorded_in_ledger: ledgerHash,
      computed: fileCheck.hash,
      error: fileCheck.error,
    };
  } catch (error) {
    responseCheck = failedPathCheck(policy.response_path_field, responsePath, String(error?.message || error));
  }

  const canonicalCheck = policy.canonical_capture_hash
    ? verifyCanonicalCaptureArtifactHash(cryptoImpl, capture)
    : {
        pass: true,
        policy: LEGACY_EXACT_FILE_POLICY,
        recorded_capture_artifact_sha256: null,
        computed_capture_artifact_sha256: null,
        canonical_rule: "not used for schema v1; capture file bytes are authoritative",
      };

  const captureFileCheck = tryExactFileSha256(capturePath, cryptoImpl);
  const captureHash = policy.canonical_capture_hash
    ? canonicalCheck.computed_capture_artifact_sha256
    : captureFileCheck.hash;
  const ledgerCaptureHash = ledgerEntry?.capture_artifact_sha256 ?? null;
  const captureCheck = {
    pass:
      captureHash !== null &&
      ledgerCaptureHash === captureHash &&
      (!policy.canonical_capture_hash || canonicalCheck.pass),
    policy: policy.capture_hash_mode,
    recorded_in_capture: policy.canonical_capture_hash
      ? capture.capture_artifact_sha256 ?? null
      : null,
    recorded_in_ledger: ledgerCaptureHash,
    computed: captureHash,
    capture_file_sha256: captureFileCheck.hash,
    capture_file_error: captureFileCheck.error,
  };

  const references = [];
  for (const reference of referenceDocuments) {
    for (const entry of collectCaptureHashReferences(reference.document, capture.case_id)) {
      references.push({
        file: reference.file,
        pointer: entry.pointer,
        value: entry.value,
        expected: captureHash,
        pass: entry.value === captureHash,
      });
    }
  }

  const checks = {
    schema_policy: policy,
    canonical_capture_hash: canonicalCheck,
    request_artifact_hash: requestCheck,
    response_artifact_hash: responseCheck,
    capture_artifact_hash: captureCheck,
    references: {
      pass: references.length > 0 && references.every((entry) => entry.pass),
      count: references.length,
      entries: references,
    },
  };

  return {
    schema_version: capture.schema_version,
    policy: policy.name,
    path_fields: {
      request: policy.request_path_field,
      response: policy.response_path_field,
    },
    checks,
    pass: Object.values(checks).every((check) => check.pass),
  };
}

function isMainModule() {
  return Boolean(
    process.argv[1] &&
      path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url)),
  );
}

function main() {
  const caseId = process.argv[2];
  if (!caseId || !/^H7_[A-Za-z0-9_]+$/.test(caseId)) {
    process.stderr.write("usage: node scripts/verify_h7_capture_artifacts.mjs H7_CASE_ID\n");
    process.exit(2);
  }

  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const projectRoot = path.resolve(scriptDir, "..");
  const rawRoot = path.join(projectRoot, "data", "raw", "statshark_h7_controller_id");
  const capturePath = path.join(rawRoot, "network_evidence", `${caseId}.capture.json`);
  const capture = readJson(capturePath);
  if (capture.case_id !== caseId) {
    throw new Error(`capture case_id mismatch: expected ${caseId}, got ${capture.case_id}`);
  }

  const ledgerPath = path.join(rawRoot, "calculate_ledger.json");
  const ledger = readJson(ledgerPath);
  const ledgerEntries = (ledger.actions || []).filter((entry) => entry.case_id === caseId);
  const ledgerEntry = ledgerEntries.length === 1 ? ledgerEntries[0] : null;

  const referencePaths = [
    ledgerPath,
    path.join(rawRoot, "session_manifest.json"),
    path.join(projectRoot, "configs", "h7_controller_experiments.json"),
    path.join(projectRoot, "outputs", "h7_controller_id", `${caseId}_ANALYSIS.json`),
    path.join(rawRoot, "network_evidence", `${caseId}.hash_correction.json`),
  ].filter((filePath) => fs.existsSync(filePath));
  const referenceDocuments = referencePaths.map((filePath) => ({
    file: path.relative(projectRoot, filePath).replaceAll(path.sep, "/"),
    document: readJson(filePath),
  }));

  const result = verifyCaptureArtifactHashes({
    capturePath,
    capture,
    ledgerEntry,
    artifactRoot: rawRoot,
    referenceDocuments,
  });
  if (ledgerEntries.length !== 1) {
    result.checks.ledger_case_entry = {
      pass: false,
      expected_count: 1,
      actual_count: ledgerEntries.length,
    };
    result.pass = false;
  } else {
    result.checks.ledger_case_entry = {
      pass: true,
      expected_count: 1,
      actual_count: ledgerEntries.length,
    };
  }

  process.stdout.write(
    `${JSON.stringify({ verifier_schema_version: 2, case_id: caseId, ...result }, null, 2)}\n`,
  );
  process.exit(result.pass ? 0 : 1);
}

if (isMainModule()) {
  main();
}
