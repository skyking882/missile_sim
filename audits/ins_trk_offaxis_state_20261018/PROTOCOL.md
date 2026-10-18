# INS/TRK off-axis state audit — 2026-10-18

Question: in the frozen War Thunder replay records, are large-off-axis transitions less consistent because of angular geometry, mixed output semantics, old-prior errors or different saved stages? Do not assume the user's observed full-trajectory difference is identical to rate endpoint residual.

Freeze the successful replay-input/solver pipeline: no production solver, controller, guidance, raw WRPL, or native decoder modifications. No new fitted alpha, gains, weights, offsets, hidden measurements, or event-specific best model selection.

## Input and sample
Use completed three-family artifact from run39121722846 (source567cd783342961f9c5aba9005ce55437457744bb). Verify root manifest and original input manifest. Use its exact 931 transition records (501 first,430 subsequent), including232 R77, with exact replay/EID/tick/raw-index identity. Retain existing near244/near-R77-79 flags only as previous conditional subsets, not proof of instantaneous capture. No new WRPL decoding is claimed.

## Definitions before reading new scores
Off-axis grouping uses the LAST INS Ps target-estimate ray nP0=normalize(Ps0-own0), against body forward Q0[:,0]. This ray is itself an estimate, not independent target truth. Do NOT group using saved INS direction or bounded antenna angle. Primary groups: small<=20deg, middle(20,50), large>=50deg. Additional fixed bins0/20/40/60/180. Also record angle to velocity, both coordinate-axis pointing angles and per-axis demand beyond55deg. A radial angle>55deg does not automatically imply a per-axis mechanical limit.

Primary inference is R77-only, separately first/reacquisition; all families are sensitivity tables. For reacquisition only retain candidate scheduled phase48m+1 and report lag0/1/2; all first captures remain unknown timing even if a phase happens to fall inside their interval. Cross-tabulate angle with gap, fresh/flag, scene, previous timer and range(<1km,1-5km,>=5km). Sample counts accompany every result; empty cells are not zero errors. No causal claims from marginal averages alone. Repeated events within an entity/scene are not independent experiments.

## Fixed diagnostics
1. Same frozen baseline M=D_.8(n1)Q1. Report full vector, axial, transverse rate errors, RMS-relative energy and signed-truncation cell compatibility at fixed decoded Q/n. Separately report pureQ, noQ, bounded-pointing-axis and old-PV-axis variants, without fitting. Recover previous79-case baseline and reference predictions as guards.
2. Compare same-time n0 with pointing g0 and target ray nP0; n1 with g1 and nP1. Separate state-source difference from temporal direction jump.
3. Exact additive vector identity n1-n0=(nP0-n0)+(nPVpred-nP0)+(n1-nPVpred), where nPVpred uses old Ps/Vs and current own p. This is a diagnostic decomposition, NOT proof of causal internal ordering. Record all three chord vectors, and angles as nonadditive descriptive values.
4. Old-rate/PV geometric mismatch, PV constant-velocity position innovation, target-ray change and Q-body change. Do not interpret geometric mismatch alone as hidden memory, since PV/time semantics remain unknown.
5. Negative control: treating the entire cross-mode direction jump as ordinary correction, with frozen tau=.078525. Compare rate residual vs fixed baseline. Current direction is a diagnostic response, not autonomous measurement input.
6. Geometry check: D_.8(n)Q has singular values1,.8,.8 regardless of off-axis; norm(D(n)-D(m))=.2|sin angle(n,m)|. Independently verify and bound observed pointing-axis substitution changes. This can exclude intrinsic matrix ill-conditioning as the explanation, not exclude wrong input semantics.

## Local windows and checks
Use both supplied full anchor histories and, if accessible, exact raw windows around selected small/large representative transitions. Representative selection must not use response error. Review fixed#118 first/second,#148 first plus six pinned anomalies. At each window preserve raw order, split duplicate/nonincreasing ticks, and do not silently use future labels as predictors. A later TRK-to-TRK pair can diagnose ordinary-update compatibility without proving how the first state was generated.

Rerun calculations from the delivered data; verify vector formulas using separate expressions and exact record identities. Print compact tables and keep per-event data, adverse cases and the predeclared bins. Scientific limitations remain beside numerical claims. Research jobs are not independent AI subagents.
