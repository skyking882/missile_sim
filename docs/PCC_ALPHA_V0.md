# PCC-α v0 — Predicted Collision-Course Capture with Explicit α Saturation

**What.** The launch-capture (甩头) phase of the R-77 family is re-modeled as
an explicit guidance-to-collision law with an explicit achievable-G envelope,
replacing the calibrated timer (`hold_time_s`/`blend_time_s`/
`fin_error_ref_rad` direct routing) for profiles that opt in via
`guidance.midcourse_lead_turn.mode = "pcc_alpha"`. Default mode
`"timer_blend"` keeps every other profile bit-identical (verified: only
R-77-family tests moved; full suite back at the frozen 4-failed/238-passed/1-
xfail baseline).

**Why.** The four-model release discriminant
(`docs/CAPTURE_RELEASE_DISCRIMINANT.md`) refuted saturated-PN and left an
ε-driven capture law as the surviving mechanism family; the design was
specified in review (2026-08-26) as: predicted collision course → velocity-
vector error servo → α/fin achievable envelope → PN handover.

## Structure (one pass per guidance step)

1. **Reference direction** `d_c`: the existing loft-aware PIP solution
   (`midcourse_lead_turn_acceleration` — collision triangle at current speed,
   pitch floored/capped by the loft program; unchanged code).
2. **Capture law**: `a_cap = V·sin(ε)/τ_c` along the ⊥-v unit vector toward
   `d_c` (ε = heading error to `d_c`). The `χ̇_c` feed-forward of the full
   design is **not** in v0 (see residuals).
3. **Envelope**: `a_env = slope_force(η,q,m)·α_max + T·sin(α_max)/(m·g)`
   (`h2_dynamics.capture_alpha_envelope_g`, mirroring the force-channel
   branch fixed_cn > eta_law > slope_scale; **trajectory-normal,
   thrust-inclusive** — the same basis as the `physical_normal_g` feedback
   the v12 loop tracks and the basis the τ_c bracket was identified in).
4. **α inversion + routing**: `α_cmd = α_max·clip(R_cap,0,1)` with
   `R_cap = |a_cap|/a_env`; since this plant trims at α = δ
   (I·ω̇ = K(δ−α) − Cω), the fin-angle command that realizes `α_cmd` IS
   `α_cmd`, routed through the existing direct-fin path (weight = 1 engages
   control.py's tracking-mode integrator). The raw accel-PID alone was
   confirmed far too sluggish (α reached only ~5° at t=0.9 s in the first
   wiring attempt). The acceleration command (clipped to
   min(a_env, reqAccelMax)) is still emitted for telemetry/PID error state.
5. **Platform release** = `R_cap` falling through 1 (physical event; the
   commanded α shrinks continuously as `α_max·R`).
6. **CAPTURE→HOMING state machine** (per-run `guidance_state` dict):
   handoff when `R_cap < handoff_r_max` AND `ε < epsilon_enter_deg` AND
   closing, sustained `handoff_dwell_s`; recapture only when
   `ε > epsilon_exit_deg` AND `R_cap ≥ recapture_r_min` (both sustained).
   HOMING is the untouched PN(+loft) path.

## Config (all under `guidance.midcourse_lead_turn`, three-layer precedence)

| key | default | R-77/R-77-1 | note |
|---|---|---|---|
| `mode` | `timer_blend` | `pcc_alpha` | opt-in switch |
| `tau_capture_s` | 0.30 | 0.30 | interval-identified ≈[0.23, 0.42] s on the R-77-1 level replay (frame-classification sensitive — see discriminant doc); nominal, cross-flight validity untested |
| `capture_alpha_max_deg` | null (required for pcc) | 23.7 | measured replay plateau; physical provenance (full-fin trim vs autopilot limiter) undecided |
| `epsilon_enter_deg` | 2.0 | — | handoff hysteresis only, NOT a release threshold |
| `epsilon_exit_deg` | 15.0 | — | recapture guard; 5° chattered against boost-phase PIP drift (A6 40° shot picked up a 47 g mid-flight yank and missed) — robustness choice, not data-identified |
| `handoff_r_max` | 0.9 | — | |
| `recapture_r_min` | 1.0 | — | recapture only if the capture law would actually saturate; 0.5 still chattered |
| `handoff_dwell_s` | 0.1 | — | |
| `release_washout_time_s` | 0.0 (off) | 0.95 | v0.1, below |
| `los_rate_filter_time_constant_s` | 0.0 (off) | 0.0 | v0.1, below — implemented, calibrates to off |

Old timer keys remain in the R-77 profiles as §1e calibration records but are
**not consumed** in pcc mode.

## Acceptance (R-77-1 level shot, A7 geometry, vs 46-frame replay)

| t (s) | α mdl/game (°) | G mdl/game | note |
|---|---|---|---|
| 0.9 | 17.8 / 23.2 | 10.8 / 13.4 | plateau deficit (below) |
| 1.9 | 18.2 / 23.5 | 16.3 / 17.4 | G tracks well |
| 2.1 | 18.3 / 22.9 | 17.5 / 17.8 | R_cap = 1.27 |
| 2.3 | 18.2 / 20.6 | 18.6 / 17.1 | R crosses 1 at **t≈2.2 s** (game release 2.1–2.3) |
| 2.9 | 6.6 / 14.5 | 8.5 / 14.6 | model releases faster (below) |

Termination proximity fuse @ **7.36 s** (game ~7.2–7.4), min_dist 10 m. One
clean CAPTURE→HOMING transition at 2.6 s, no chatter.

**Off-axis healing (the structural prediction, now confirmed in-model):**
70° 82 m→**10.0 m proximity fuse** (UPDATE §5 item 6 healed); 80° 36.7 m
(below the old 40–100 m freeze, item 7)→**17.8 m**; 90° →18.6 m. Envelope
freezes in `test_rate_inner_fin_torque.py` re-based around the new structure
with dated comments; `pytest` back at the frozen 4/238/1 baseline.

## Known residuals (v0, in priority order)

1. **Plateau deficit**: α holds ~18.2° vs game 23.7°. With δ_cmd = α_cmd and
   a steady turn, trim gives α = δ − (C/K)·ω ≈ δ − 2ω_turn/ω_n — the same
   ~20° ceiling §1e's grid hit ("T1 unreachable"), now structurally
   attributed to the missing damping feed-forward in the fin command, not to
   capture parameters. v0.1 candidate: δ_cmd = α_cmd + 2ω/ω_n (needs a
   control-side pcc flag; do NOT touch the shared direct-routing path used by
   timer_blend).
2. **Release shape**: model α collapses to PN levels ~0.6 s after release;
   the game takes ~1.5 s (14.5° vs 6.6° at 2.9 s) — the discriminant's open
   release-shape question, now measurable in-model.
3. **No `χ̇_c` feed-forward / constant-speed PIP**: post-handoff ε wanders
   6–8° as the accelerating missile drifts the PIP; harmless after the
   ε_exit=15° + R≥1 recapture gates, but the v1 items (feed-forward, and
   `s_M(t_go)` from axial dynamics) exist precisely to shrink it.
4. Speed still runs hot (+250/+330/+200 km/h at 2.3/4.3/6.9) — drag-law
   thread (CN20), not a capture issue; endgame G residual unchanged.

## v0.1 (2026-08-26) — release washout, and a LOS-rate filter that calibrates to off

v0's residual #2: the modelled α collapses to PN levels ~0.6 s after release,
the replay takes ~1.5 s (20.6°@2.3 → 18.4 → 16.3 → 14.5 → 12.9 → 10.5 → 7.6°@3.7).
A separate quantitative pass established that **no PN-derived signal can supply
that shoulder** — 0.7 s after release the game is still riding ~65 % of the α
envelope while the true λ̇ is ≈0 — so the residual command has to be
capture-channel *memory*.  Two config-gated mechanisms were added; both default
to `0.0` = off, and with both off the v0 code path is reproduced bit-for-bit
(verified: full 371-sample A7 α/G/w/mode timeline identical at full float repr).

### A. Release washout (`release_washout_time_s`, R-77 family 0.95 s)

Two first-order states in the per-run `guidance_state`, both asymmetric (rise at
a fixed actuator-scale `PCC_WASHOUT_RISE_TIME_S` = 0.05 s, decay at
`release_washout_time_s`), sharing one `pcc_filter_step_s` Δt guard because
`guidance_command` runs twice per timestamp:

1. the α-inverted demand **as a body pitch/yaw vector**, lagged;
2. the direct-routing weight `w`, crossfaded 1 (CAPTURE) → 0 (HOMING).

The routed fin fraction is their product, so CAPTURE is exactly v0
(`w`=1, routed = α_max·R along ê), the handoff instant is bumpless (the demand
steps, the lag does not, and `w` is still 1), PID/PN authority (1−`w`) then
grows back at the same rate the residual fades, and at `w`=0 the direct term is
gone rather than double-counting PN's own feed-forward.  Handoff/recapture
*gates* are untouched.

Two structural choices were forced by measurement, not taste:

- **Lag the vector, not a magnitude with a live direction.** Once the velocity
  vector reaches the collision course the PIP error direction flips through
  zero; a decaying magnitude re-aimed along the flipped direction produced a
  20 g → 7 g → 20 g limit cycle across the shoulder.
- **Relax into the HOMING demand, not into zero.** HOMING's own command is run
  through the *same* a_n^-1 inversion so the lag has something to relax into.
  Decaying to a frozen vector flies the airframe well past the collision course
  (heading error 1.6° → 17° in 0.8 s) and trips recapture.
- A reversed demand is deliberately taken on the **slow** side of the
  asymmetry (`same_sense` test), or the lag snaps through zero and rings.

### B. LOS-rate low-pass (`los_rate_filter_time_constant_s`, R-77 family 0.0 = off)

`ideal_truth` hands PN a noiseless per-step λ̇, which no trackloop could follow,
and the unfiltered rate is the suspect for the modelled G being wigglier than
the game's.  The LOS-rate **vector** is low-passed in `guidance_state` before
the PN product is re-formed (`pn_acceleration_from_los_rate`;
`pn_acceleration` itself stays a pure measurement of the true rate, and the
telemetry field keeps the unfiltered vector).  Applied only in the pcc path,
so timer_blend is untouched whether or not the key is present.

**It calibrates to 0.0.**  τ_f ≥ 0.15 s loses the A7 intercept outright (0.3 s
of λ̇ lag in an endgame closing at >2 km/s is fatal: the run goes to lifetime
at 100–370 m); τ_f ≤ 0.10 s buys ~1–2 units of shoulder SSE while pushing
max G in 5.5–7.2 s further from the replay and leaving <1.5° of margin under
the recapture guard.  The mechanism is kept in the codebase (it is cheap, it is
the right shape for a trackloop surrogate, and it is the natural knob once
`observation_mode` is a real seeker rather than ideal truth), but it is not
adopted for the R-77 family on this evidence.

### Calibration (A7 geometry, su_r_77_1, shoulder-only fit)

Fit target: sum of squared error in α at t = 2.5 / 2.9 / 3.4 s vs the replay's
18.4 / 14.5 / 10.5°.  Everything else is reported, not fitted.  The grid was
run with `epsilon_exit_deg` already widened (below), so the shoulder is not
contaminated by recapture transients.

| τ_release | τ_f | SSE | α@2.5 | α@2.9 | α@3.4 | α@4.3 | G@4.9 | maxG 5.5–7.2 | A7 termination |
|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.00 | 24.70 | 17.79 | 12.24 | 6.11 | 1.30 | 12.28 | 34.89 | fuse 7.44 s |
| 0.60 | 0.20 | 18.84 | 17.79 | 12.48 | 6.71 | 1.31 | 11.51 | 41.37 | fuse 7.47 s |
| 0.60 | 0.30 | 16.52 | 17.79 | 12.57 | 6.97 | 1.41 | 10.89 | 41.30 | miss 47.6 m |
| 0.60 | 0.50 | 13.10 | 17.79 | 12.72 | 7.41 | 1.76 | 9.34 | 41.01 | miss 158 m |
| 0.80 | 0.00 | 14.24 | 17.92 | 13.19 | 6.99 | 1.55 | 13.10 | 38.43 | fuse 7.47 s |
| 0.80 | 0.20 | 9.43 | 17.92 | 13.39 | 7.68 | 1.35 | 12.34 | 41.10 | miss 56.3 m |
| 0.80 | 0.30 | 7.71 | 17.92 | 13.46 | 7.97 | 1.35 | 11.70 | 41.00 | miss 114 m |
| 0.80 | 0.50 | 5.29 | 17.92 | 13.59 | 8.45 | 1.68 | 10.00 | 55.98 | miss 203 m |
| **1.00** | **0.00** | **7.05** | 18.03 | 13.90 | 7.94 | 4.00 | 18.74 | 29.69 | fuse 7.46 s |
| 1.00 | 0.20 | 3.84 | 18.03 | 14.07 | 8.62 | 1.39 | 13.29 | 40.64 | miss 134 m |
| 1.00 | 0.30 | 2.78 | 18.03 | 14.13 | 8.92 | 1.35 | 12.52 | 40.48 | miss 200 m |
| 1.00 | 0.50 | 1.45 | 18.03 | 14.24 | 9.38 | 2.11 | 37.14 | 52.92 | miss 373 m |
| 1.20 | 0.00 | 25.64 | 18.12 | 14.45 | 5.44 | 18.14 | 21.29 | 46.89 | miss 124 m, handoff slips to 4.38 s |
| 1.20 | 0.20/0.30/0.50 | 25.64 | 18.12 | 14.45 | 5.44 | 18.14 | 23.6–25.5 | 56.77 | miss 95–189 m |
| 1.50 | 0.00–0.50 | 10.82 | 18.23 | 15.13 | 7.28 | 18.54 | 35.6–36.2 | 56.9 | miss 238–514 m, handoff 4.70 s |
| 1.80 | 0.00–0.50 | 4.51 | 18.33 | 15.63 | 8.71 | 17.10 | 42.45 | 55.9–56.7 | miss 530–763 m, handoff 5.02 s |

Grid winner among runs that still intercept: **τ_release = 1.0, τ_f = 0.0**
(SSE 7.05).  τ_release ≥ 1.2 pushes the handoff itself past 4.3 s and the shot
is lost; every τ_f ≥ 0.3 loses the shot.

**Local refinement** over τ_release ∈ [0.85, 1.15] × τ_f ∈ [0, 0.15], then
re-scored against *all* the hard R-77 gates (A7 shoulder + fuse, the A6 40°
double-fuse anchor, the 80°/90° off-axis freezes), moved the point to
**τ_release = 0.95 s, τ_f = 0.0**: SSE 8.43, α errors −0.40 / −0.76 / −2.77
(the |err| ≤ 3° acceptance band needs τ_release ≥ 0.92), and — the binding
constraint — the A6 40° shot only fuses for τ_release ≤ 0.98, missing by 12.8 m
at 1.00.  0.95 sits inside that window with margin on both sides.  The lowest
SSE reachable at all (3.58 at 1.10/0.10) was rejected: fuse at 7.50 s is
exactly on the acceptance boundary and the heading excursion comes within 1.2°
of the recapture guard.

### One extra parameter had to move: `epsilon_exit_deg` 15 → 25 (R-77 family)

The washout spends ~1 s of near-envelope turn *after* the collision course is
reached, so the post-handoff heading excursion grows from 11.4° (v0) to 19.4°.
v0's recapture guard `epsilon_exit_deg = 15°` — explicitly a robustness choice,
not a data-identified quantity — then fires, re-commands α_max in the reverse
direction and destroys the shot (40 g yank, miss, chatter).  With the guard at
25° the excursion is absorbed, the state machine keeps a single
CAPTURE→HOMING transition, and PN unwinds the over-turn on its own.  The
recapture *logic* is unchanged; only this per-profile threshold moved, and it
moved in the safe direction (fewer recaptures, the failure mode v0 already
documented at 5°).  This is a real cost: the excursion is a genuine model
defect, not just a threshold problem — see residuals.

### Results (A7, su_r_77_1, ideal_truth)

| t (s) | game α | v0 α | v0.1 α | game G | v0 G | v0.1 G |
|---|---|---|---|---|---|---|
| 1.5 | 23.7 | 18.17 | 18.61 | 15.8 | 13.99 | 14.30 |
| 2.5 | 18.4 | 15.16 | **18.00** | 16.2 | 16.65 | 19.68 |
| 2.9 | 14.5 | 6.63 | **13.74** | 14.6 | 8.53 | 17.22 |
| 3.4 | 10.5 | 4.87 | **7.73** | 12.3 | 7.58 | 11.66 |
| 4.3 | 3.6 | 0.74 | **3.75** | 5.6 | 1.60 | 7.85 |
| 4.9 | 1.3 | 3.21 | 6.60 | 2.0 | 8.46 | 16.87 |
| 6.9 | 4.0 | 1.64 | 2.51 | 17.1 | 7.51 | 11.01 |

Termination proximity fuse @ **7.46 s** / 10 m (game ~7.2–7.4, v0 7.36 s), one
clean CAPTURE→HOMING transition at 2.60 s, no chatter.

**Holdouts.** PL-12 (and every other timer_blend missile) is untouched —
`pytest` is back at the frozen 4-failed / 238-passed / 1-xfail baseline with the
4 failures still only the pre-existing missing-fixture H5 ones, and A7's PL-12
straight-target de-load arc anchor is green.  Off-axis, the extra second of
capture-channel memory is exactly what those shots were short of: 80° tightens
17.8 m → **14.9 m** (still no fuse) and 90° reaches the **proximity fuse** at
10 m / 9.55 s — the first of the four missiles to do so at 90°, and the
direction UPDATE_V1.0.2 §5 items 6/7 predicted for a geometry-scaled release.
Both R-77 bands in `test_rate_inner_fin_torque.py` were re-frozen with dated
comments; the other three missiles' bands are bit-identical.  70° regresses
slightly the other way (v0 10.0 m fuse → 10.5 m near-miss); it is not a frozen
assertion.

### Residuals after v0.1

1. **Plateau deficit unchanged**: α holds 18.6° vs the replay's 23.7°.  This is
   the missing damping feed-forward in the fin command (δ = α_cmd + 2ω/ω_n),
   still v0 residual #1, and it is explicitly **not** part of v0.1 — it needs a
   control-side pcc flag and must not touch the shared direct-routing path.
   Call it v0.1b.
2. **The over-turn is real, not just a threshold.** Carrying ~65 % of the
   envelope for another second in the capture direction genuinely flies the
   airframe ~19° past the collision course, and PN then has to unwind it:
   G@4.9 goes 8.5 → 16.9 (game 2.0) and max G in 5.5–7.2 s goes 22.0 → 31.1
   (game 17.1).  So v0.1 buys the shoulder (2.5–4.3 s, where every point is now
   within 3° in α) at the cost of the 4.5–6.5 s de-load valley, which v0 fitted
   better.  The discriminant doc predicted exactly this: reconstructing the
   post-release G as horizontal capture-direction turning fails its own
   closest-approach check by 1.7–1.9 km.  The honest reading is that the game's
   post-release G is at least partly **not** in the capture direction (vertical
   loft de-load is the obvious candidate), and no amount of tuning inside
   mechanism A can express that.  A vertical-plane-instrumented capture replay,
   or the game's own target-track reconstruction, is what would settle it.
3. Residuals 3 and 4 of v0 (no χ̇_c feed-forward / constant-speed PIP; speed
   running hot) are unchanged.

## Falsification path (unchanged from the discriminant)

Per-flight τ_c intervals from Q(t)=V·sinε/a_env must intersect across
geometries/altitudes; different-altitude level shots separate M2 (constant
ε_c) from this law (release-ε ∝ a_env·τ_c/V). New replays slot in as
`data/replays/*.tsv` + an anchor.

*Files (v0.1 adds): `guidance.py` (`pcc_filter_step_s`,
`pcc_washout_fin_fraction`, `pcc_washout_weight`, `pcc_filtered_los_rate`,
`pn_acceleration_from_los_rate`, `capture_routed_alpha_deg` telemetry),
`h2_simulator.py` (`pcc_routed_alpha_deg` sample field), the two new keys in
`profile_adapter.py`/schema/runtime defaults, and
`missiles/su_r_77*.json` (0.95 / 0.0 / `epsilon_exit_deg` 25).*

*Files: `guidance.py` (law, state machine, α inversion), `h2_dynamics.py`
(`capture_alpha_envelope_g`), `h2_simulator.py` (envelope wiring, per-run
guidance state, `pcc_*` telemetry), `profile_adapter.py`/schema/runtime
defaults (keys), `missiles/su_r_77*.json` (opt-in + provenance). CN20/M0
docs describe the pre-PCC baseline. Boundary: independent local engineering
candidate; no equivalence claim to any private implementation.*

## 2026-08-26c review — scalar washout rejected; closure gate; polar release

Review verdict on v0.1 (user, 2026-08-26): the scalar washout locks the
magnitude memory onto the stale horizontal capture direction, injecting
ΔΨ ≈ a_r·τ_r/V ≈ 15–20° of extra heading integral that PN must unwind — the
4.5–6.5 s G hump is *written into the filter*, not a tuning problem. Shipped
profiles reverted to `release_washout_time_s = 0.0` (v0 hard switch); the
`epsilon_exit_deg = 25` crutch removed with it; off-axis freezes restored to
the v0 bands. 0.95 s is retained only as an ablation value.

**Plant α→G closure audit — PASS, and now a frozen gate.** The review asked
why the closed loop showed ~18 g at α≈18.5° where the game needs ~23°.
`scripts/plant_g_closure_audit.py` (six replay frames, game TAS/α at 6300 m
ISA, model propulsion timeline, the simulator's own `forces_for_state_h2`):
worst |ΔG| = **0.17 g**, inverse α within 0.3° of the game plateau, plant ≡
standalone discriminant law (no packed-slope/body-lift double count, no
duplicated thrust-normal). The apparent mismatch is **entirely the hot-speed
(drag-thread) error** contaminating closed-loop comparisons. Frozen as
`tests/test_plant_g_closure.py` (suite now 4 failed / 240 passed / 1 xfail);
until the drag thread closes, α-point calibrations of guidance constants
must use game-q operating points, not closed-loop α.

**J_ψ diagnostic** (`j_psi_actual_deg` / `j_psi_capture_cmd_deg` /
`j_psi_homing_cmd_deg` per sample): cumulative horizontal heading integral,
actual and per-channel commanded. Standing acceptance rule: a release
smoother must not materially change the final heading integral.

**Polar release ablation** (`release_memory_mode = "polar"`,
`pcc_polar_washout_fin_fraction`: magnitude keeps the slow asymmetric decay;
direction is a fin-space unit vector slewed at 0.08 s toward the current
total specific-force demand, whose ≥1 g gravity-compensation share keeps it
non-degenerate). A7, τ = 0.95 held fixed from the scalar fit (no re-sweep):

| variant | α@2.5/2.9/3.4 | α@4.3/4.9 | maxG 5.5–7.2 | outcome | mode switches |
|---|---|---|---|---|---|
| game | 18.4/14.5/10.5 | 3.6/1.3 | 17.1 | fuse 7.2–7.4 | — |
| v0 hard switch | 15.2/6.6/4.9 | 0.7/3.2 | 22.0 | fuse 7.36/10 m | 1 |
| scalar 0.95 (honest ε_exit=15) | 18.0/13.7/7.7 | 17.8/16.7 | 56.4 | **miss 116 m** | 5 (chatter) |
| polar 0.95 | 18.0/8.8/8.8 | **3.8/1.3** | **22.0** | fuse 7.38/10 m | 1 |

Reading, per the review's pre-registered discrimination: the hump vanishes
and the de-load valley lands almost exactly on the game (α@4.3/4.9 =
3.8/1.3 vs 3.6/1.3) → **direction re-aiming was the missing piece**, and the
game's "stable, persistent" mid-course G is substantially the
gravity/vertical trim share the polar direction preserves. The remaining
un-suppliable gap is the α shoulder at 2.9–3.7 s (~8.8° vs 12–14.5°): no
horizontal-plane memory can carry it without trajectory damage → the
leading candidate is vertical-plane / total-incidence content, decidable
only with replays that record **altitude** (and ideally heading), plus the
0° off-axis control shot. Note the polar shoulder decays at ~τ/2 because
weight and magnitude multiply — a structural observation for any future fit,
not a license to re-sweep.

Ship state after this pass: everything off by default (`washout 0.0`,
`release_memory_mode "scalar"` inert, LOS filter 0.0); polar is one config
switch away. Next data, in order of discriminating power: altitude-recorded
level shot, 0° off-axis control, 30° horizontal vs 30° vertical off-axis
pair.
