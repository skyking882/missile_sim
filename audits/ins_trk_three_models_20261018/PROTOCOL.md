# Three candidate mechanisms for INS→TRK serialized-state mismatch

Date: 2026-10-18. Scope is offline reconstruction of War Thunder game-replay fields, not real weapon performance or production guidance changes. The user requested all three previously proposed families: range-induced scaling, vector/state mixing, and nonorthogonal representation conversion.

## Frozen data and baseline

Restore completed handover run39100358600, artifact12126670224 (`missile-sim-ins-trk-handover-20261018`). Verify its complete root manifest. Use all931 saved code25→31 transitions; preserve the existing244 phase1/fresh/flag00 near-handover subset and79 R77 subset for primary comparison. Those are existing repeatedly studied observations, not new blind data. The full931 are a scope expansion over the previous79, not a new raw extraction. Independently recover the79 baseline0.028389deg/s and73/79 rate-cell compatibility before accepting new results.

Baseline M0=D_.8(n_new) Q_new. n_new is an explicitly conditional output/projection input, not an independently predicted radar observation. Rate units are native rad/s; q=15/32767rad/s. Fixed parameter0.8 is retained. Old and new rate encoding hypothesis: signed truncation cells including zero[-1,1], closed boundaries for compatibility. Q,n and auxiliary PV/range inputs are fixed decoded values; LP compatibility is conditional on them, not a complete nonlinear quantization proof.

## Input isolation and domains

Candidate builders receive old state and only new own-position/velocity/Q, recorded range, pointing angles and conditional n. Current target Ps/Vs and current rate must never be accessed. NaN-poison guards check this. New range is a serialized conditional input, not proven raw range measurement. Old Ps/Vs may be stale or refer to a prior track. No target reassignment from current outputs.

Nonpositive range or degenerate geometry (<1e-6 in declared units), Euler inverse condition number>1000 or polar metric |cos(elevation)|<1e-3 trigger an explicitly recorded baseline fallback rather than row removal. Score these fallbacks separately. No error-dependent exclusions. Report fresh/flag groups, first capture/reacquisition, phase offsets and missile family. A direct static map on a later saved TRK endpoint may include unmodelled normal updates; failure there is not alone a refutation of its instantaneous initialization interpretation.

## Family R: range conversion

Four input ratios: cached R_old/R_new; ||Ps_old-own_old||/R_new; ||Ps_old+Vs_old*T-own_new||/R_new; cached R_old/||Ps_old-own_old||. Test each ratio and its reciprocal. Test full-vector scaling r*M0 and transverse-only scaling D_(.8r)(n_new)Q_new,16 fixed variants. Also allow a shared partial correction M=M0+beta(Mcandidate-M0), beta in[0,1], fitted only on other scenes. beta=0 is a true baseline control. Do not choose ratio direction separately for each event.

## Family B: state-vector mixture

Alternative rates come only from (i) old Ps/Vs geometry; (ii) half-interval propagation using old Vs and linearly interpolated own state; (iii) full-interval propagation; (iv) old Vs with current conditional n/R and own velocity. For each, compare mixture before Q, DQ[(1-beta)w_old+beta u], and after Q, D[(1-beta)Qw_old+beta u]. Eight variants, shared beta constrained[0,1]. Report the fixed beta=.2 alternative as well as leave-scene fitted beta. These u vectors are not obtained by inversion of current response rate. Mixed state is a hypothesis, not a decoded field.

## Family J: nonorthogonal state representation

Native rotation convention Q=Ry(heading)Rz(pitch)Rx(bank). Derive Euler-rate→body-angular-velocity Jacobian Jb from this convention and verify it with finite differences of Q. Test Q_new Jb_new, Q_new inv(Jb_new), Q_new Jb_old, Q_new inv(Jb_old) as seed transformations, followed by D_.8(n_new). The matrix reconstructed from Q uses the canonical pitch branch; it is not proof that stored rate is Euler-rate data.

Also test a polar-rate metric and its inverse: scale the angular-velocity tangent component associated with yaw by cos(elevation), leave the elevation-associated and axial components unchanged. Express all axes in the world frame using new Q and conditional target n. This tests two possible angular-coordinate metric conventions without an arbitrary fitted3x3 matrix. Six fixed variants. A common small-departure interpolation between baseline and each exact transform is also fitted on other scenes with eta in[0,1]. No per-event matrix fitting.

## Fitting and comparison

For each held scene and target family, train only on other scenes in the near-handover244 subset, using same-family examples if at least10 are available; otherwise use other-scene near examples across families and label this fallback. Fit one scalar along each prespecified correction by bounded least squares on complete rate vectors. Fit and choose a variant within each family using only training loss, including baseline as a competing option. Apply the selected common variant and scalar unchanged to the held-scene near events and all broader events. Thus new Ps/Vs/rate do not enter individual predictors; training response rates enter parameter calibration only.

Report every fixed variant and every individual leave-scene fitted variant, not just the winning family. For primary79 and244 report complete/axial/transverse errors, maxima and quantization feasibility. For all931 and fresh/flag00 wider groups report complete errors and parameter transfer. New failures among previously baseline-compatible cases are counted explicitly. Use the same rows for baseline and candidate; no pooling overlapping sets as independent data.

## Structural diagnosis before interpreting fits

For each baseline M0, test whether arbitrary positive scaling alone can satisfy both rate cells. Substitute z=rho*x, impose rho*l0<=z<=rho*u0 and l1<=M0*z<=u1; solve linear programs for feasible rho interval. This is a per-event existence diagnostic using current rate as a label, not a forward model. Independently repeat with output variables and equality Mz=y. Also report decoded-point best nonnegative scaling and best nonnegative axial/transverse coefficients as diagnostic lower bounds, never as predicted coefficients.

The original six fixed-Q failures and the five arbitrary-Q norm failures stay pinned. Report all six separately, all successful baseline cases, and the wider population. A function manually constructed from five failures in the previous message is not validation and is not adopted as a model.

## Delivery

Retain full per-event predictions, scalar fit tables, domain fallbacks, raw source indices/hashes, unmodified inputs, test logs, baseline reconstruction checks, numerical reproduction and all adverse outcomes. Do not modify dev, source raw branch or native parser. Runtime is currently GitHub Actions because local execution calls failed; jobs are not AI subagents.
