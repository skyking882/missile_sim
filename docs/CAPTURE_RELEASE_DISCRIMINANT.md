# Launch-Capture Release Discriminant — What Ends the Alpha Plateau?

**Question.** The R-77-1 level-shot replay shows the seeker in TRK by t=0.9 s
but the launch-capture alpha plateau (23.7°) holding until t≈2.1–2.3 s. The
shipped model reproduces this with a calibrated timer (`hold_time_s`=1.4 s)
that structurally breaks extreme off-axis cases (UPDATE_V1.0.2 §5 items 6/7).
Four candidate mechanisms were pre-registered:

| | mechanism | release condition |
|---|---|---|
| M1 | Timer (shipped) | t > lock_delay + hold |
| M2 | Explicit velocity-vector capture | ε_PIP < ε_c |
| M3 | Plain PN from TRK, saturated | N·Vc·λ̇ falls below achievable G |
| M4 | Proportional capture V·ε/τ riding the α-limiter | same envelope exit, capture law |

M3 is the null hypothesis that would make an explicit capture mode
unnecessary (per the handover-transient literature, PN with a large initial
heading error produces a big initial command that a limiter clips).

**Method.** `scripts/capture_release_discriminant.py` (read-only, prints
tables). Game-data-driven: the missile's horizontal track is reconstructed by
integrating the replay's measured speed timeline and turning at the measured
G (minus 1 g level support, toward the target side); the scripted A7 geometry
supplies the straight head-on target. Per frame it computes ε_PIP, λ̇,
a_PN = 4·Vc·λ̇, a_cap = V·ε/τ, and the α-limited achievable envelope
G_env (measured G while α is pinned; η-law-scaled extension after). The sim
is not run; the model contributes only constants.

**Validation.** Integrated flown distance matches the replay's displayed
flown distance to ±0.05 km at every checked frame (t=0.2→7.8). The
reconstruction is only trusted to t≈2.4 s (through release): after release
the de-load-arc G cannot be attributed between horizontal/vertical/sign from
this data, so later ε/λ̇ are artifacts and the closest-approach check fails
(~1.7–1.9 km vs the real proximity kill) — a declared limitation, outside
the discriminant window.

**Plateau sanity.** During the pinned frames the packed-lift force law
(k(η)=0.574·η^0.242, slope finsLatAccel/finsAoa, η at 6300 m ISA) explains
8.0→11.9 g of the measured 13.4→17.8 g; the residual is a stable
+5.4→+5.9 g ≈ the thrust-normal share T·sinα/(mg) (implies T≈24 kN
mid-burn, plausible). Envelope construction self-consistent.

## Results (discriminant window 0.9–2.4 s; observed release 2.1–2.3 s)

| t (s) | seeker | α (°) | G_meas | ε_PIP (°) | λ̇ (rad/s) | a_PN (g) | a_cap,τ=0.8 (g) | R_PN |
|---|---|---|---|---|---|---|---|---|
| 0.9 | TRK | 23.2 | 13.4 | 25.3 | 0.0247 | 7.2 | 23.9 | 0.54 |
| 1.3 | TRK | 22.8 | 14.5 | 19.3 | 0.0210 | 6.5 | 19.3 | 0.45 |
| 1.6 | TRK | 23.7 | 16.6 | 14.1 | 0.0168 | 5.5 | 14.8 | 0.33 |
| 1.9 | TRK | 23.5 | 17.4 | 8.5 | 0.0109 | 3.6 | 9.2 | 0.21 |
| 2.1 | TRK | 22.9 | 17.8 | 4.6 | 0.0063 | 2.1 | 5.2 | 0.12 |
| 2.3 | TRK | 20.6 | 17.1 | 0.8 | 0.0011 | 0.4 | 0.9 | 0.02 |

**M3 saturated PN: REFUTED.** Raw PN demand peaks at R_PN=0.54 at TRK onset
and declines monotonically — it never reaches the achievable envelope the
airframe was demonstrably riding (α pinned, G climbing with q). No fixed
navigation constant rescues it: sustaining saturation to 2.1 s would need N
growing from ~8 to ~24 as λ̇→0. Refutation is N-independent and lives
entirely inside the validated window. **An explicit capture mechanism
therefore exists in-game after TRK** — TRK-to-PN-with-limiter cannot
produce the observed hold.

**Surviving family: ε-driven release, two sub-variants.**

- M2 threshold: ε crosses 5° at t=2.08, 2° at t=2.24 → ε_c ≈ 2–5° brackets
  the observed release exactly.
- M4 proportional capture saturating the α-limiter: V·ε/τ stays above the
  envelope until t=2.03 for τ=0.3 s (τ=0.8 unpins at 1.52 s — too early,
  which also explains why the shipped τ=0.8 + proportional fin routing
  undershoots the plateau: v1.0.2 §1e's "T1 unreachable"). τ≈0.3 needs no
  threshold parameter at all: release is the limiter exit. *Identification
  honesty (2026-08-26 review): the law's structure is theory-driven, but τ_c
  itself is identified on this flight — the last-pinned / first-unpinned frame
  pair brackets it via Q(t)=V·sinε/a_env to ≈0.23–0.42 s (t_s=1.9 / t_u=2.1
  classification; the classification rule itself needs pre-registering, since
  taking t_s=2.1 / t_u=2.3 instead shifts the bracket to ≈0.04–0.23 s and
  excludes 0.3), with 0.30 s as the in-interval nominal. Cross-flight validity
  untested.*
- M1 timer: lands at 2.2 s only because `hold_time_s` was calibrated on this
  very flight — zero predictive content, and it is the variant that breaks
  the 70°/80° envelope tests.

Both survivors make release geometry-dependent, which is the behavior the
off-axis regressions (items 6/7) are asking for.

**Degeneracy note.** For a head-on target the PIP direction, the LOS, and
pure pursuit nearly coincide, so this dataset cannot distinguish *which*
reference direction the capture steers to (PIP vs LOS), nor cleanly separate
M2-threshold from M4-τ≈0.3 (both fire within 2.0–2.2 s here). Off-axis
geometries separate all of these strongly.

## Follow-ups

1. **Cross-geometry test**: early frames (t<2.5 s) of the PL-12 / R-77 30°
   slow-launch replays would separate M2 vs M4 and PIP vs LOS — the A3
   anchors only bracket release to [1.4, 2.1] s. Re-digitize from the
   original recordings if they survive; a new 70–80° off-axis capture would
   be the sharpest possible discriminator.
2. **Structural refactor implied for `guidance.py`** regardless of which
   survivor wins: separate (i) the guidance-layer capture law (a_capture vs
   a_PN selection/blend) from (ii) an explicit achievable-G/α saturation,
   instead of the current ε→fin-fraction routing plus the w(t) time
   schedule. `hold_time_s` retires under either survivor.
3. The de-load arc's G-attribution problem (horizontal vs vertical vs sign)
   caps what any level-shot replay can say about t>2.4 s guidance — the
   endgame-G residual (§1e holdout regression) needs either the game's own
   target-track reconstruction or a vertical-plane-instrumented capture.

*Data: `data/replays/r77_1_level_20260824.tsv` (46 frames, seeker-state
resolved). Script: `scripts/capture_release_discriminant.py`. 2026-08-26.*
