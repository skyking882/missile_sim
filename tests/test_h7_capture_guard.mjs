import assert from "node:assert/strict";
import crypto from "node:crypto";
import test from "node:test";

import {
  computeCanonicalCaptureArtifactSha256,
  stampCanonicalCaptureArtifactHash,
  verifyCanonicalCaptureArtifactHash,
} from "../scripts/h7_cdp_capture_guard.mjs";


const SAMPLE = {
  schema_version: 1,
  case_id: "H7_TEST",
  nested: { value: 2 },
  capture_artifact_sha256: null,
};

test("canonical capture hash ignores the embedded hash value", () => {
  const first = computeCanonicalCaptureArtifactSha256(crypto, SAMPLE);
  const withOldHash = { ...SAMPLE, capture_artifact_sha256: "old-hash" };
  const second = computeCanonicalCaptureArtifactSha256(crypto, withOldHash);
  assert.equal(first, second);
});

test("canonical capture hash changes when evidence changes", () => {
  const first = computeCanonicalCaptureArtifactSha256(crypto, SAMPLE);
  const changed = { ...SAMPLE, nested: { value: 3 } };
  const second = computeCanonicalCaptureArtifactSha256(crypto, changed);
  assert.notEqual(first, second);
});

test("stamp returns a verified copy without mutating the input", () => {
  const stamped = stampCanonicalCaptureArtifactHash(crypto, SAMPLE);
  assert.equal(SAMPLE.capture_artifact_sha256, null);
  assert.equal(verifyCanonicalCaptureArtifactHash(crypto, stamped).pass, true);
  stamped.nested.value = 4;
  assert.equal(verifyCanonicalCaptureArtifactHash(crypto, stamped).pass, false);
});
