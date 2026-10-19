# Held recovery: source, pointing and acceptance contract

2026-10-19. War Thunder replay software only. Base6a166cf2e7e9c44bb489265c35be69c601e97736, prior completed retry delivery run39179706161. No production defaults, PN, control, physics or frozen profile changes. Do not add held-recovery cooldown, gate-width fit, best per-event gain or target reassignment by nearest output.

## Questions and fixed diagnostics

A. #118 last fresh→held→fresh mismatch. Preserve the prior retry simulation verbatim, including the premature accepted observation at2.4375s. Compare its cached-truth-PV pointing to directly recorded antenna angles, and to geometric candidates built from the LAST fresh state only: (1) target Ps+Vs*T, (2) fixed cached Ps, (3) n advanced with constant complete w and clipped in current body axes, (4) fixed body pointing angles, (5) fixed world pointing. Current recorded antenna and current n are output-conditioned diagnostics, never autonomous inputs.

Primary seed is identified algorithmically as first fresh code31 followed by flag0 code31 with increasing positive timer; inspect that whole continuous-held run until the next unsupported mode or timer discontinuity. Source record identity is exact replay/EID/raw index/tick. Target trajectory remains the independent fixed EID in prior metadata. #148 first six seconds are a fresh-stage negative control; additional held episodes are exploratory and only scored with independently covered target histories. Do not rename code29 to held without evidence.

At every scored saved point retain actual angles, angle integers, rate, cached Ps/Vs, timer, Q and source SHA. Determine what actually remains constant. Calculate full sphere/rotation geometry, not additive azimuth approximations. Native convention ray(y,p)=(cos p cos y,sin p,-cos p sin y), n_dot=n×w. Runtime uses C=diag(1,1,-1), polar n_runtime=C n_native, axial w_runtime=-C w_native; exact parity verified.

B. Recompute all old detector gate margins at the first divergence (not only first rejected reason) before modifying caches: signal margin, receiver range, rateMax, angleGateRate, clutter/Doppler, scalar distance and Doppler prediction and half ranges. Use the existing live scalar filter objects. New first acceptance is model-generated; no current game rate/direction is fed into this run. Separately evaluate last-game-fresh seeded state trajectories and clearly label them conditional restarts.

C. Inspect whether set_estimated_track overwrites cache time/value inconsistently or ignores the supplied angular state. Test actual contracts with synthetic timestamp cases and real last-fresh values where applicable. A proven software interface inconsistency can be patched opt-in; resemblance to WT output is not proof of an intended code path.

D. Association is a distinct hypothesis. Check last fresh Ps against independent fixed aircraft EID at matched packet time, nearby competing independent entities when present in bounded inputs, and the recorded designation key. Do not assume same key means same reflector. No current chaff cloud position is fabricated from its birth point. If bounded inputs lack chaff evidence, state that limitation rather than excluding it.

## Conditions and comparisons

Use both fixed old7.5deg and candidate15deg half-power conventions and RCS1/2m² for snapshot signal checks; preserve all results. Main replay trial uses prior fixed15deg/RCS2 and unchanged retry clock to isolate postcapture effects, with #148 control. No newly fitted parameters; old observer gains stay explicitly previously calibrated/assumed, not promoted to engine source.

Report held pointing prediction error separately from signal eligibility and actual acceptance. A current-recorded-axis check can reject a wrong predictive pointer but is not a forward solution. If all candidates allow too-early recovery, keep that failure instead of manufacturing cooldown.

## Execution and review

GitHub Actions executes deterministic bounded scripts because local container/Python currently fails. Jobs are not independent AI subagents. Verify prior manifest before reads, reproduce previous trace, check alternative vector expressions and timestamp/idempotence contracts. Publish all failures, limited sample counts and actual run receipts. Do not claim a fresh WRPL decode or closed-loop flight regression.
