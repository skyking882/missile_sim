# INS→TRK state provenance and update-order audit — frozen tests

Task: identify War Thunder game's serialized state transition without fitting new coefficients.

## Baseline and evidence
- Source branch `audit/ins-trk-three-models-20261018` and frozen 931 25→31 boundaries from run39100358600 artifact12126670224.
- Rate vector is full three-dimensional quantity; convention n_dot=n×w.
- Fixed comparison k=0.8, alpha_h=0.85, tau=0.078525 s, h=1/48s; these are *previously identified candidate endpoint parameters*, not game-source constants.
- Conditional input n1 is used ONLY in explicitly marked rate-projection diagnostics, not autonomous direction predictions.
- Main readouts are tests of constraints, clocks, stage geometry and failure signatures, **not selecting best parameter values**.
- The 79 near phase-1 R77 events, 244 near all families and 931 full boundary collection overlap.
- Some first TRK packets may occur after normal updates. Distinguish 'original capture' vs 'reacquisition'; record gap, phase, previous timer and state flags. Never treat 48m+1 as a universal first-capture timestamp.

## A. Is last serialized INS rate the live handover seed?
1. Confirm old encoded rate vs old PV-derived world geometric rate at the old saved tick. This comparison is *not* a test of whether internal seed equals serialized output.
2. Reconstruct **diagnostic** input `D_0.8(n1)^{-1}w1` and back-rotate using actual Q new, then calculate the discrepancy from serialized old rate. This inversion uses the target output and is NOT a predictor.
3. Decompose required seed adjustment into old-rate-parallel versus orthogonal components and into world PV-derived geometric rate correction. Record dot products and conditional shape residual, input event timer/age. Avoid claiming mechanism from improvement of one-event fits.
4. For anchors with actual full INS histories, check whether serialized old w remained constant, whether PV implied geometric w evolved, and whether rate changed at INS→TRK. Inspect old timer and flag changes.

## B. Is first saved TRK already post-update?
1. Treat 0 extra updates baseline as D_0.8(n_seed) Q(t_capture) old w, with n_seed from prior cached PV propagated to assumed capture tick. This model uses no current target output.
2. For scheduled reacquisitions only, define *precommitted* candidate capture tick `phase_after(old_tick)` and post ticks `new_tick-phase`. For unscheduled first captures, report other phase possibilities as an exploratory envelope, not as a selected best-case fit.
3. Step forward 0..saved lag using same past-only CV measurement proxy: target=old_Ps+old_Vs dt, own position linearly interpolated between known own states, current time angle ray m(t)=unit(target-own(t)); standard ordinary angle feedback with kinematically propagated n,w, alpha=.85/tau=.078525. Compare with no-post-update baseline on **both n and w**, in near/nonnear groups.
4. Provide a conditional diagnostic using final saved n as endpoint direction to check ordinary rate innovation c/(tau+T/2); label as output-conditioned, not predictive. If the step count is too uncertain / data do not support shared improvement, say so.
5. Inspect fixed anchor #118 (first, second), #148 first and six previously pinned mismatches separately. Do not select per-event best timing.

## C. Negative controls and audit
- If random/no-wrong phase assumptions perform similarly, timestamp inference is unidentifiable.
- If first-ins old rate not equal PV-geometric rate or timer very large, cached PV cannot be treated as independent tracking truth.
- Quantization check must be applied to full w and code, never rely on RMS alone.
- Run existing 931-event data NaN-poison guard: predicting current target state/rate from past-only inputs must not access new Ps/Vs/n/w. Current own p/v/Q are valid externally known inputs.
- Score fixed subsets including zero/later post ticks, timer and flags; record all failures.
- Do not modify production dev/raw/parser. No claim of multi-agent review when no dispatcher.
