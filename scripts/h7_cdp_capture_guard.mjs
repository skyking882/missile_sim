// H7 StatShark CDP capture contract.
// The caller supplies a persistent CDP command function and the session-wide
// set of hashes for tokens already submitted to the server.  Token plaintext
// is never returned by this module.

export const TURNSTILE_HEADER = "x-turnstile-token";
export const CALC_URL_PATTERN = "*://*/api/missiles/CalcMissileRange";
export const CAPTURE_ARTIFACT_HASH_FIELD = "capture_artifact_sha256";
export const CAPTURE_ARTIFACT_CANONICAL_RULE =
  "SHA256(JSON.stringify(capture with capture_artifact_sha256=null, null, 2) + newline)";

export function sha256Hex(cryptoImpl, value) {
  return cryptoImpl
    .createHash("sha256")
    .update(String(value), "utf8")
    .digest("hex");
}

function cloneJsonValue(value) {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("capture must be a JSON object");
  }
  return JSON.parse(JSON.stringify(value));
}

// The hash field is always nulled before serialization. This prevents an old
// embedded hash from changing the next hash and makes post-write verification
// deterministic. Callers must finish every other capture field first.
export function canonicalCaptureArtifactJson(capture) {
  const canonical = cloneJsonValue(capture);
  canonical[CAPTURE_ARTIFACT_HASH_FIELD] = null;
  return `${JSON.stringify(canonical, null, 2)}\n`;
}

export function computeCanonicalCaptureArtifactSha256(cryptoImpl, capture) {
  return sha256Hex(cryptoImpl, canonicalCaptureArtifactJson(capture));
}

export function stampCanonicalCaptureArtifactHash(cryptoImpl, capture) {
  const stamped = cloneJsonValue(capture);
  stamped[CAPTURE_ARTIFACT_HASH_FIELD] = null;
  stamped[CAPTURE_ARTIFACT_HASH_FIELD] =
    computeCanonicalCaptureArtifactSha256(cryptoImpl, stamped);
  return stamped;
}

export function verifyCanonicalCaptureArtifactHash(cryptoImpl, capture) {
  const recorded = capture?.[CAPTURE_ARTIFACT_HASH_FIELD] ?? null;
  const computed = computeCanonicalCaptureArtifactSha256(cryptoImpl, capture);
  return {
    pass: typeof recorded === "string" && recorded === computed,
    recorded_capture_artifact_sha256: recorded,
    computed_capture_artifact_sha256: computed,
    canonical_rule: CAPTURE_ARTIFACT_CANONICAL_RULE,
  };
}

function findTurnstileHeader(headers) {
  for (const [name, value] of Object.entries(headers || {})) {
    if (name.toLowerCase() === TURNSTILE_HEADER) {
      return { name, value: value == null ? "" : String(value) };
    }
  }
  return null;
}

export function createTurnstileGuard({ cryptoImpl, usedTokenHashes, priorCases = [] }) {
  if (!cryptoImpl || typeof cryptoImpl.createHash !== "function") {
    throw new TypeError("cryptoImpl.createHash is required");
  }
  const used = usedTokenHashes instanceof Set
    ? usedTokenHashes
    : new Set(usedTokenHashes || []);
  const prior = Array.isArray(priorCases) ? priorCases : [];

  function inspectPausedRequest(event) {
    const request = event?.params?.request || {};
    const header = findTurnstileHeader(request.headers);
    const headerPresent = !!header && header.value.length > 0;
    const tokenHash = headerPresent ? sha256Hex(cryptoImpl, header.value) : null;
    const matched = tokenHash
      ? prior.find((entry) => entry && entry.token_sha256 === tokenHash)
      : null;
    const unique = !!tokenHash && !used.has(tokenHash);
    return {
      header_present: headerPresent,
      token_sha256: tokenHash,
      unique_against_prior_submitted_requests: unique,
      matched_prior_case_id: matched?.case_id ?? null,
      request_allowed_to_server: unique,
      preflight_token_abort: !unique,
    };
  }

  async function handlePausedRequest(event, cdp) {
    const decision = inspectPausedRequest(event);
    const requestId = event?.params?.requestId;
    if (!requestId) throw new Error("Fetch.requestPaused did not include requestId");

    if (!decision.request_allowed_to_server) {
      await cdp("Fetch.failRequest", {
        requestId,
        errorReason: "BlockedByClient",
      });
      return { ...decision, action: "preflight_token_abort" };
    }

    // The request is unchanged.  Record the hash before continuing so a
    // duplicate paused event in this session cannot pass the guard twice.
    used.add(decision.token_sha256);
    await cdp("Fetch.continueRequest", { requestId });
    return { ...decision, action: "continueRequest" };
  }

  return {
    usedTokenHashes: used,
    inspectPausedRequest,
    handlePausedRequest,
  };
}

export function fetchEnableParams() {
  return {
    patterns: [{ urlPattern: CALC_URL_PATTERN, requestStage: "Request" }],
  };
}

// Keep the page active immediately before the final native Calculate. This is
// part of the lifecycle contract: a background/frozen page can accept DOM
// inspection while failing to dispatch the native event or renew Turnstile.
export async function activatePageForCalculate(cdp) {
  await cdp("Page.bringToFront");
  await cdp("Page.setWebLifecycleState", { state: "active" });
  await cdp("Input.setIgnoreInputEvents", { ignore: false });
}

export async function enableH7CaptureDomains(cdp) {
  await cdp("Runtime.enable");
  await cdp("Page.enable");
  await cdp("Network.enable");
  await cdp("Fetch.enable", fetchEnableParams());
}
