import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  computeCanonicalCaptureArtifactSha256,
  stampCanonicalCaptureArtifactHash,
} from "../scripts/h7_cdp_capture_guard.mjs";
import {
  CANONICAL_NULL_FIELD_POLICY,
  LEGACY_EXACT_FILE_POLICY,
  exactFileSha256,
  verifyCaptureArtifactHashes,
} from "../scripts/verify_h7_capture_artifacts.mjs";


function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function makeFixture(schemaVersion) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "h7-capture-validator-"));
  fs.mkdirSync(path.join(root, "requests"));
  fs.mkdirSync(path.join(root, "responses"));
  fs.mkdirSync(path.join(root, "network_evidence"));

  const requestRelativePath = `requests/H7_TEST_${schemaVersion}.request.json`;
  const responseRelativePath = `responses/H7_TEST_${schemaVersion}.response.json`;
  const requestPath = path.join(root, requestRelativePath);
  const responsePath = path.join(root, responseRelativePath);
  fs.writeFileSync(requestPath, `request-${schemaVersion}\n`, "utf8");
  fs.writeFileSync(responsePath, `response-${schemaVersion}\n`, "utf8");

  let capture = {
    schema_version: schemaVersion,
    case_id: `H7_TEST_${schemaVersion}`,
    captured_at_utc: "2026-08-14T00:00:00.000Z",
    request_sha256: "legacy-network-request-semantic-hash",
    response_sha256: "legacy-network-response-semantic-hash",
  };
  if (schemaVersion === 2) {
    capture.request_artifact_path = requestRelativePath;
    capture.response_artifact_path = responseRelativePath;
  } else {
    capture.request_path = requestRelativePath;
    capture.response_path = responseRelativePath;
  }

  const requestArtifactHash = exactFileSha256(requestPath);
  const responseArtifactHash = exactFileSha256(responsePath);
  if (schemaVersion === 1) {
    capture = {
      ...capture,
      // Deliberately unrelated to the request artifact bytes.
      request_sha256: "legacy-post-data-hash-not-an-artifact-hash",
    };
  } else {
    capture.request_artifact_sha256 = requestArtifactHash;
    capture.response_artifact_sha256 = responseArtifactHash;
    capture = stampCanonicalCaptureArtifactHash(crypto, capture);
  }

  const capturePath = path.join(root, "network_evidence", `H7_TEST_${schemaVersion}.capture.json`);
  writeJson(capturePath, capture);
  const captureArtifactHash = schemaVersion === 1
    ? exactFileSha256(capturePath)
    : computeCanonicalCaptureArtifactSha256(crypto, capture);
  const ledgerEntry = {
    case_id: capture.case_id,
    request_artifact_sha256: requestArtifactHash,
    response_artifact_sha256: responseArtifactHash,
    capture_artifact_sha256: captureArtifactHash,
  };
  const referenceDocuments = [{
    file: "calculate_ledger.json",
    document: { actions: [ledgerEntry] },
  }];

  return {
    root,
    capture,
    capturePath,
    ledgerEntry,
    referenceDocuments,
  };
}

function verifyFixture(fixture) {
  return verifyCaptureArtifactHashes({
    capturePath: fixture.capturePath,
    capture: fixture.capture,
    ledgerEntry: fixture.ledgerEntry,
    artifactRoot: fixture.root,
    referenceDocuments: fixture.referenceDocuments,
  });
}

test("schema v1 uses legacy exact-file hashes and ignores old request_sha256", () => {
  const fixture = makeFixture(1);
  const result = verifyFixture(fixture);
  assert.equal(result.policy, LEGACY_EXACT_FILE_POLICY);
  assert.equal(result.pass, true);
  assert.equal(result.checks.request_artifact_hash.pass, true);
  assert.equal(result.checks.response_artifact_hash.pass, true);
  assert.equal(result.checks.capture_artifact_hash.pass, true);
});

test("schema v2 uses canonical capture hash and artifact-path fields", () => {
  const fixture = makeFixture(2);
  const result = verifyFixture(fixture);
  assert.equal(result.policy, CANONICAL_NULL_FIELD_POLICY);
  assert.equal(result.path_fields.request, "request_artifact_path");
  assert.equal(result.path_fields.response, "response_artifact_path");
  assert.equal(result.pass, true);
});

test("schema v3 uses canonical capture hash and request/response path fields", () => {
  const fixture = makeFixture(3);
  const result = verifyFixture(fixture);
  assert.equal(result.policy, CANONICAL_NULL_FIELD_POLICY);
  assert.equal(result.path_fields.request, "request_path");
  assert.equal(result.path_fields.response, "response_path");
  assert.equal(result.pass, true);
});

test("changing only the embedded hash does not change canonical input", () => {
  const fixture = makeFixture(2);
  const original = computeCanonicalCaptureArtifactSha256(crypto, fixture.capture);
  const changedHash = { ...fixture.capture, capture_artifact_sha256: "different-hash" };
  assert.equal(computeCanonicalCaptureArtifactSha256(crypto, changedHash), original);
});

test("changing another capture field fails canonical validation", () => {
  const fixture = makeFixture(2);
  const changedCapture = { ...fixture.capture, captured_at_utc: "2026-08-14T00:00:01.000Z" };
  const result = verifyCaptureArtifactHashes({
    capturePath: fixture.capturePath,
    capture: changedCapture,
    ledgerEntry: fixture.ledgerEntry,
    artifactRoot: fixture.root,
    referenceDocuments: fixture.referenceDocuments,
  });
  assert.equal(result.pass, false);
  assert.equal(result.checks.canonical_capture_hash.pass, false);
});

test("ledger artifact reference mismatch fails validation", () => {
  const fixture = makeFixture(3);
  const badLedgerEntry = {
    ...fixture.ledgerEntry,
    response_artifact_sha256: "not-the-response-hash",
  };
  const result = verifyCaptureArtifactHashes({
    capturePath: fixture.capturePath,
    capture: fixture.capture,
    ledgerEntry: badLedgerEntry,
    artifactRoot: fixture.root,
    referenceDocuments: [{
      file: "calculate_ledger.json",
      document: { actions: [badLedgerEntry] },
    }],
  });
  assert.equal(result.pass, false);
  assert.equal(result.checks.response_artifact_hash.pass, false);
  assert.equal(result.checks.references.pass, true);
});
