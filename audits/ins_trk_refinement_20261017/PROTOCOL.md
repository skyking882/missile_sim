# INS/TRK refinement protocol — 2026-10-17

Scope: offline War Thunder game replay field reconstruction only. No production dev, raw WRPL or native decoder edits. Prior data/source baseline a68bbd33a92feddb95f0c8f15d8158df3b681223; final prior artifact run39069993344 / ID12115135336. No subagent dispatcher is available; separate numerical jobs are not independent AI reviewers. Local compute tools failed this session, so numerical work uses completed in-session GitHub Actions.

## 1. Quantization and reset map

Recreate exactly the previous244 zero-post-step candidate events (79 R77): b1 equals first48m+1 after b0, fresh current timer and both flag0. This is an input/time-defined subset, not proven knowledge of internal call count.

Test raw and cell-center OLS estimates of kappa and held-scene prediction. Treat two competing rate encoding intervals separately: signed toward-zero truncation (positive k->[k,k+1], negative->[k-1,k], zero->[-1,1]) and round-nearest (k+/-0.5), all in units q=15/32767rad/s. These are encoding hypotheses, not assertions of the WT serializer. For each fixed map use linear programming to find the smallest extra output half-width needed, allowing every prior rate within its cell. Fix Q and direction to decoded values; do not claim nonlinear Q/n quantization has been fully integrated.

Compare kappa1,.8,1-.926*h/(tau+h/2),1-h/(tau+h/2), with tau=.078525,h=1/48. Compare matrix order Pk(n)Q versus QPk(n), no Q and Q-transpose. For .8 compare current saved n, current Ps-derived ray, saved pointing and past-PV ray. Current endpoint projections are explicitly diagnostic, never independent direction reconstruction. Keep infeasible examples. Per-record variable choices in LP are feasibility witnesses, not predicted hidden states. Also measure a fixed kappa grid .77..83 in .002 increments on R77, without claiming best-grid selection is blind validation.

## 2. Phase controls

Use all931 captured transitions. Publish before/after ticks modulo48 histograms and interval coverage for every one of48 phase candidates, separately original501/repeated430, family, fresh flag0. Compare the same conditional .8 reset map by endpoint phase and gap without output-based exclusions. Test whether48m+1 is supported primarily for repeated conversions rather than universally. When reading anchor histories, check recurring code25 boundaries and whether they occur exactly after a configured/observed timeout rather than being new physical acquisitions. Do not equate mode flag change to a newly detected independent echo without evidence.

## 3. Ordinary update timing / causal ambiguity

Reproduce original ordinary-pair eligibility and5,484 R77 angle<54.9 samples (overlapping all-family32,189 angle<45). Compare same samples, preserving tick gaps1/2/3. Current n/Ps are labels only. Fit on other scenes and score held scene, all with past angular prediction plus current pointing field as conditional input.

Families: constant alpha; output-offset equivalent alpha(T)=1/(1+L*g(T)); implicit-new-velocity linearization alpha(T)=a/[1-(1-a)Tg(T)]. Also fit alpha separately by gap as diagnostic and test sequential1/48 updates repeatedly using the same current pointing (explicit constant-measurement assumption). Frozen g(T)=1/(.078525+T/2). Score direction and complete rate jointly, do not choose models using position-only losses. Stratify raw versus centered prior rate but never mix a centered comparison with a raw baseline silently. Rate labels remain raw; add cell-center diagnostics separately where appropriate. No inference that pointing is upstream measurement merely because an endpoint relation fits.

## Verification and artifacts

Check original delivery manifest on download; assert sample identities/counts. Unit tests for LP feasible/infeasible constructions, reset matrix order, projection and alpha algebra. Save every scored row, integer cells, excluded/nonconverged cases, fitted fold parameters and exact input/source hashes. Fixed .8 remains a candidate, not a claim of a recovered source constant. Do not weaken checks to hide failures. Deliver bounded new evidence rather than duplicate all previous reports.
