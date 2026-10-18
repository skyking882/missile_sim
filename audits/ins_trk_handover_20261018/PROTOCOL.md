# Final bounded INS→TRK handover round · 2026-10-18

Scope: offline reverse reconstruction of War Thunder game replay fields. No production simulator, raw WRPL or native decoder edits. Parent commit 10cf1db7c09f500afd2d49724725ab27d75c23e3. Input delivery is completed run39074725895/artifact12116871425. Verify all manifest entries before computation. Local execution failed before analysis in this session; calculations will use completed GitHub Actions. No AI subagents are claimed.

## Frozen populations and parameters

All931 captured old/new record pairs retained. Main comparisons use the SAME244 phase1 endpoint/fresh/flag0 candidates from previous work, including SAME79 R77 events and SAME6 rate-cell failures. This cohort is not new blind data, and phase1 is not assumed for all first acquisitions. Do not select a different model per event or exclude the six failures. Fixed kappa=.8, h=1/48, tau=.078525. Per-scene alpha(h)=1-L_scene/h uses existing other-scene ordinary-fit coefficients; no refitting here.

## J. Direction and rate must share a consistent output stage

Given s=Qnew*wold, define F_s(m)=slerp(Rot(-s*h)m,m,alpha_h) and W_s(m)=s+c/(tau+h/2), c the oriented spherical correction from Rot(-s*h)m to F_s(m).

Compare (J0) explicit contraction P_.8(n_saved)s; (J1) old interpretation m=n_saved (direction before feedback, rate after feedback); (J2) infer m by solving ONLY F_s(m)=n_saved, then score W_s(m) against saved rate. J2 is conditional measurement elimination, NOT an independently generated observation or a forward capture model. Saved rate/Ps/Vs must not enter the inverse solver. Inspect inverse residual, uniqueness near n, implied displacement of m and required state/output phases. Matching rate cannot select J1/J2 if their difference lies below rate uncertainty. Keep all outputs.

## S. Did the last stored INS rate become stale before capture?

Use ONLY old Ps/Vs and endpoint own position/velocity to compute geometric-rate changes at fractions 0,half,full of the bracket and at the final tick before capture. Fixed candidate seeds: raw wold; wold+[w_geo(t)-w_geo(0)]; pure w_geo(0); pure w_geo(T). Q choices limited to old, midpoint, new attitude (fixed, not fitted). Main comparison uses Qnew throughout; Q timing sensitivity is separate. Current n is a diagnostic projection axis. Current rate is NEVER input to predictions. Evaluate complete vector errors and rate-cell feasibility (affine map when a deterministic geometric increment is added). Report ALL244 and R77 separately, plus original/repeated strata. Same old input errors occur in all models; this is not independent target-truth validation.

## I. Rotation-independent invariant

For the claimed map w1=P_.8(n1)Qw0 and ANY orthogonal Q, ||w0||²=(w1·n1)²+||w1-(w1·n1)n1||²/.8². Check whether intervals of possible norms overlap when old/new rates vary in their declared signed-truncation cells. Minimize positive quadratic on box by independent finite active-set enumeration; maximize at vertices. Interval overlap is necessary only, NOT sufficient for the actual Q. If disjoint, no choice of Q alone can fix the event at fixed n and kappa. Direction uncertainty can expand the bound by (1/kappa-1)*sin(eps_n)*max||w1||; report eps_n=0 and conservative .0031degrees sensitivity, conditional on the prior source-unit-vector encoding hypothesis.

For fixed-Q LP failures also give conservative orientation sensitivity: change of the linear map is bounded by ||w0||[(1-kappa)sin(eps_n)+2sin(eps_Q/2)]. eps_Q=.01/.1degrees are declared stress tests, NOT claims about actual Euler/quaternion precision. This can reject small orientation-error explanations but cannot verify one.

## Tests and handover

Verify rotation sign, spherical inverse closure for synthetic known states, rate-label poisoning guards, norm-box extrema against independent optimizer, deterministic candidate populations, and original fixture reproduction. Provide a single handover document distinguishing: reproduced field identities; fitted endpoint relations; conditional inversion diagnostics; rejected mechanisms; still-unidentified measurement generation. Include exact source hashes, counts, all six exceptions, commands, output files, package checksums and actual execution failures. Do not imply that an inverse-estimated measurement was independently observed.
