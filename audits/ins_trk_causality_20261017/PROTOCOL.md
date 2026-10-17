# INS→TRK mechanism audit protocol · 2026-10-17

Question: distinguish an output/source switch, a coordinate-state handover, and hidden post-capture updates. Reconsider whether serialized pointing angles and Ps/Vs are estimator outputs rather than independent observations. Scope is War Thunder game replay reconstruction only; no production guidance/dynamics changes.

No callable subagent dispatcher was found in this session. Four separate numerical tasks are not independent AI agents. Execution uses GitHub Actions because local Python/container attempts failed. No task is left as promised background work.

## Frozen sources

Raw bundle 9f28aeb; existing verified Linux derived run39055559128; original 501-case reference run39055471867 (snapshot2dc243b). Boundary evidence run39065146998 includes931 capture pairs plus full #118/#148 histories. Prior audit source2954b2f. Retain native mode, raw record index, duplicate tick, timer, flags, source hashes, and family. Do not relabel targets using Ps. Current dev remains7a1c0c6.

Recover the original501-case tick window from the original dataset and select those entity histories from the actual Linux raw export, not reported summary numbers. Positive tick gaps1..3, code31/timer0/flag0 at both endpoints define ordinary pairs. Main all-family position regressions require both pointing angles <45 degrees and independent own-motion trapezoidal position residual <0.5m. R77 additional reporting may use <54.9 degrees, explicitly distinguished. Captures use the existing931 events, first501 and later430 reported separately, not called new blind data.

## A. Frame/state meaning

For INS and fresh TRK, compare saved rate with w_geo=(Vs-vm)×nPs/R; also Q*w, Q.T*w, and sign reversals. Use physical nPs=unit(Ps-p), not clamped INS direction. Check complete vectors, parallel and transverse components separately. Candidate frame relations are global, not selected per event. If raw INS rate is already world-geometric, Q multiplication cannot be described as a proved ordinary frame conversion without another state/reader explanation.

## B. Capture timing

Freeze alpha choices {0.926,1}, tau=0.078525s and h=1/48s. Fixed candidate capture tick is the first global48m+1 after old tick, only when contained in the observed bracket. This phase is a historical hypothesis, not an independently identified physical clock. Interpolate own position/velocity linearly and orientation by SO(3) interpolation between observed endpoints.

Compare Qold and Q(capture) seeds, reset-only vs reset-then-self-feedback, followed by one ordinary update per remaining native tick. Two input paths: (1) past Ps/Vs CV, which does not read current Ps/Vs/n/w; (2) current endpoint Ps/Vs back-propagated, explicitly an endpoint-conditioned diagnostic and NOT a forward observer. Report all fixed candidates, failure tails, phase eligibility, #118 first/second and #148 first. Never infer arbitrary hidden innovations to force a perfect fit. Current position in diagnostic2 already encodes endpoint direction; its direction fit is not independent evidence.

## C. Ps generation and causal equivalence

On ordinary matched pairs compare direct current pointing-ray reconstruction; previous PV prediction blended with current ray; previous angular-state prediction blended with current ray; current velocity substituted for prior velocity in the PV predictor (diagnostic only); and small time-offset forms P=p+R*unit(g+L*projection_tangent(V-vm)/R). Fit a single shared scalar per candidate using four scenes, test fifth. Do not conflate instantaneous geometry P=p+Rn with a proven computational direction.

Also examine whether n-to-pointing angular difference aligns with saved delta-w (output-stage alternative) rather than raw measurement innovation. This is a conditional field relation, not an independent input. Use identical eligibility across competing models, per-scene scores and separate full vs radial/transverse error measures. Distinguish the runtime alpha from any endpoint effective fit.

## D. Verification

Use independent algebra for geometric reconstruction, source hashes, poisoning of current endpoint Ps/Vs/n/w for past-only capture predictors, unchanged-bundle check, numeric comparison tolerances with actual maxima, and inspect worst cases instead of deleting them. Report precise source/label permissions per successful calculation. Retain deterministic code, full per-event output and training/test membership. No production observer is changed without evidence closing these contracts.
