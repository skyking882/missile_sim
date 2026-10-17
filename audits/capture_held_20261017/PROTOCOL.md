# Capture / held / clipping protocol, 2026-10-17

Scope: serialized state reconstruction in War Thunder game replays only. No production dynamics/guidance changes. Native replay bundle 9f28aeb; reuse verified Linux export from workflow 39055559128 and original frozen cohort definitions from snapshot 2dc243b. Current chat has no subagent dispatcher; separate scripts/jobs are not independent AI reviewers.

## Inputs and validation

Keep raw header flags, timer integer, after-timer flags, b5, packet time and epoch separately. Adjacent stream records are not deduplicated by tick; non-increasing b5 or unsupported layouts break a transition. Main identity inventory is the existing 501 entities; report all observed 25->31 crossings in their full recorded histories, separately first and repeated crossings. Report other header transitions separately. Do not reassign target identities using current Ps. Select representative #118 / #148 by fixed replay/EID, not residual. Other models use all eligible entities with family stratification.

All current direction/rate/Ps/Vs are response variables. They may enter expressly labelled conditional/diagnostic tests, never a purported forward prediction. New observed angle pair, range and own state may be inputs to conditional replay reconstruction. Frozen ordinary constants alpha_eff=0.926, tau=0.078525 s come from the previous report; they are fitted values, not identified engine source, and tests on previously used scenes are not new independent blind validation.

## Capture

At each 25->31 boundary with 1<=gap<=8 ticks, compare seeds w_old, Q_old w_old, Q_new w_old, Q_new Q_old^T w_old, zero, and past-Ps/Vs-derived transverse geometric rate. No per-event seed selection.

For each seed compare: (i) reset n to current angle ray with no rate correction; (ii) reset n to ray then predict/correct once using 1/48 s; (iii) same over the full saved interval; (iv) repeat same current ray once per b5 tick (explicit zero-order-measurement assumption); (v) preserve old n through an ordinary correction (negative control). Freeze alpha_eff and g(h)=1/(tau+h/2). Evaluate complete rate, longitudinal and transverse errors, direction and Ps jointly. Multi-correction use of the current angle does not identify hidden individual observations. First timer-positive/saturated rows are stratified, not silently discarded. No fit of hidden measurements to endpoint errors.

## Held and return to fresh

Define held from code31 and positive timer, not by equal rate/Ps/Vs outcomes. Stratify after-timer flags and whether the timer advance matches state interval within two timer cells. Compare fixed-rate rotation, frozen world direction, frozen body angles, and cached-Ps/Vs extrapolated LOS. Test exact integer/f32 cache equality as an outcome. Re-run no-new-angle propagation across entire held episodes from one seed; do not feed later saved direction/rate. Return-to-fresh tested separately for prior continuity versus resets, recording all failures.

## Angle clipping

Evaluate body yaw/pitch reconstructed from the saved n and Q versus the angle fields; also clipped reconstruction at fixed limits (55 deg for R77-1, other families only as separate exploratory groups). Count near-limit serialized angles and unbounded n angles beyond limit. Distinguish sensor direction censoring from physical state clipping. For saturated measurements a measured boundary value is not an equality constraint on unobserved LOS. Compare ordinary angle correction, no angular correction and projection of the prior to one-sided censored constraints; preserve unsaturated components. No fitted per-record missing angle.

## Evidence

Retain per-event predictions and raw source hashes, primitive decoder comparisons with old labels, counts/eligibility/failure strata, exact parameters, runtime environment and numerical checks. Separate diagnostic consequences from successful end-to-end reconstruction. Do not replace existing production model on the basis of a conditional replay fit.
