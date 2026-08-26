# M0 Blind-Run Residuals

M0 (`config/profile_m0_strict.json`, see `docs/M0_PROVENANCE_AUDIT.md`
for the full parameter classification) is run once, blind, against the
standard scenario suite. **No parameter was adjusted based on the
results below.** M0 disagreeing with the game replays is the expected
and intended output of this exercise, not a defect.

Run with `scripts/run_m0_scenarios.py`; raw output reproduced inline
below. G values use the trajectory-normal (flow-normal) basis,
matching how the game displays G (see the audit's "trajectory-normal
display basis" row) -- same convention `tests/test_replay_anchors.py`
uses for the shipped, fitted runtime.

## Pre-registered predictions (written before any result below was read)

- **P1**: peak G overshoots game by ~1.7-1.9x at matched alpha.
- **P2**: 30-deg off-axis launches show no alpha plateau; crank is
  absent; expect degraded/failed intercepts on A3/A5-class geometries.
- **P3**: thrust/drag-dominated speed profiles remain within a few
  percent except during high-alpha phases.

---

## A1 -- 35 km head-on loft, cn_pl12

| Quantity | Game | M0 | Ratio (M0/game) | Delta |
|---|---|---|---|---|
| Time of flight | 33.26 s | 32.49 s | 0.977 | -0.77 s (-2.3%) |
| Terminal speed | 487 m/s | 522.4 m/s | 1.073 | +35.4 m/s (+7.3%) |
| Apex altitude | 8845 m | 8753.8 m | 0.990 | -91.2 m (-1.0%) |
| Minimum distance | 10 m | 10.0 m | 1.000 | ~0 |
| Termination | hit | `proximity_fuse` | -- | match |

M0's own Mach trace for this flight (`mach_by_time`): 2.97 @5s, 3.18
@10s, 2.72 @15s, 2.35 @20s, 2.04 @25s, 1.79 @30s -- i.e. this flight
cruises almost entirely inside M 1.8-3.2, the band where the audit's
five telemetry-anchored C_et(M) knots used to sit and where M0 now
only has a straight line bridging M 1.8 to M 3.5. Combined with the
uniform ~9.1% CdA0 reduction from removing the x1.10 global scale
(present at every Mach, not just in that gap), this is the most likely
driver of the +7.3% terminal-speed deviation in an otherwise low-alpha,
thrust/drag-dominated flight.

## A3 -- 30 deg slow launch, cn_pl12 (alpha timeline)

| t (s) | Game alpha (deg) | M0 alpha (deg) | Ratio | Delta |
|---|---|---|---|---|
| 0.7 | 11.5 | 0.30 | 0.026 | -11.2 |
| 1.2 | 14.1 | 3.67 | 0.260 | -10.4 |
| 1.4 | 15.2 | 8.28 | 0.545 | -6.9 |
| 2.1 | 10.4 | 9.04 | 0.869 | -1.4 |
| 3.2 | 5.1 | 6.22 | 1.220 | +1.1 |

Shape, not just amplitude, differs: the game plateaus/peaks early
(15.2 deg at t=1.4s) and decays by t=3.2s; M0 has no early response,
rises slowly, peaks late (~9 deg around t=2.1s) and is still above the
game's value by t=3.2s. M0 still reaches `proximity_fuse` at
min_distance=10.0 m, flight_time=6.83 s -- the intercept succeeds
despite the missing early turn-in.

## A7 -- level shot, cn_pl12 (de-load arc)

| t (s) | Game G | M0 G | G ratio | Game alpha (deg) | M0 alpha (deg) | alpha ratio |
|---|---|---|---|---|---|---|
| 2.9 | 8.9 | 18.57 | 2.09 | 6.3 | 6.84 | 1.09 |
| 3.4 | 5.9 | 16.42 | 2.78 | 4.1 | 5.70 | 1.39 |
| 3.9 | 3.6 | 13.38 | 3.72 | 2.5 | 4.38 | 1.75 |
| 4.4 | 1.4 | 9.75 | 6.96 | 1.2 | 3.01 | 2.51 |

Termination: game and M0 both `proximity_fuse` (M0: flight_time=7.32 s,
min_distance=10.0 m). The G ratio is smallest at t=2.9s (where alpha is
also most closely matched, 6.3 vs 6.84 deg) and grows sharply at later
times specifically because M0's alpha decay lags the game's -- i.e. the
"at matched alpha" comparison (t=2.9s, ratio 2.09x) is far closer to
P1's predicted band than the same-*time* comparisons at t=3.9-4.4s,
which mix in a second, slower-de-load effect P1 did not predict.

## R-77-1 level shot (A7 scenario, su_r_77_1)

Caveat: the mission-specified harness reuses A7's kinematic setup
(launch 1200 km/h @ 6300 m, 30 deg azimuth) as a stand-in; it is not a
geometry-matched replay of the actual R-77-1 game clip the "game"
numbers below were digitized from. Treat magnitude comparisons here as
indicative, not a clean controlled test (this matters most for the
speed row under P3, flagged again below).

| Quantity | Game | M0 | Ratio / delta |
|---|---|---|---|
| Alpha plateau window (t=0.9-2.1s) | 21.7-23.7 deg | 0.25-5.25 deg | M0 peak is 24% of the game plateau's low end |
| G @ 1.1s | 13.2 | 0.41 | 0.031x |
| G @ 2.1s | 17.8 | 9.49 | 0.533x |
| G @ 3.4s | 12.3 | 20.14 | 1.638x |
| G @ 5.6s | 8.5 | 7.97 | 0.938x |
| G @ 6.9s | 17.1 | 14.04 | 0.821x |
| alpha @ 6.9s | ~4 deg | 2.11 deg | 0.53x |
| Speed @ 2.3s | 1858 km/h | 2238.8 km/h | +20.5% |
| Speed @ 6.9s | 4032 km/h | 4387.5 km/h | +8.8% |
| Proximity fuse time | ~7.2-7.4 s | 7.33 s | inside window |

This is the sharpest illustration of what disabling midcourse actually
costs: M0 has essentially no lateral response during the window
(t=0.9-2.1s) where the game shows a hard, sustained ~22 deg launch
turn-in (G is roughly 2-32x too low across that window: 1.88x at
t=2.1s, 32.5x at t=1.1s), then *overshoots* once PN alone is
forced to close the resulting geometry error (G 64% too high at
t=3.4s), then settles back close to the game's late-arc values (within
6-18%) as intercept geometry converges. Despite this very different
transient, M0 still reaches `proximity_fuse` at 7.33s, inside the
game's own 7.2-7.4s window -- PN's terminal self-correction absorbs
several seconds of missing midcourse guidance in this particular
engagement.

## A5 -- windup-bug case, cn_pl12 (termination + min-distance only, per mission scope)

| Quantity | M0 |
|---|---|
| Termination | `proximity_fuse` |
| Minimum distance | 10.0 m |
| Flight time | 8.02 s |
| Peak G (trajectory-normal) | 27.52 @ t=2.48 s |
| Max \|PID integral\| | 0.209 rad |

No numeric game replay target is in scope for A5 (mission asks for
termination + min-distance only). Context, not a P1-P3 test: the
*shipped, fitted* runtime's own anchor test
(`test_a5_user_bug_case_anchor`) requires peak time <=2.7s and max
integral <=0.32 rad to prove the v1.0.2 anti-windup fix holds; M0's
peak lands at 2.48s with integral 0.209 rad -- comfortably inside
those structural thresholds even though M0 changes the plant, i.e. M0
does not reopen the windup pathology those thresholds were built to
catch.

## A6 -- statshark 40 deg 8 km double-fuse, cn_pl12 + su_r_77 (termination + min-distance only, per mission scope)

| Missile | Termination | Minimum distance | Flight time |
|---|---|---|---|
| cn_pl12 | `proximity_fuse` | 10.0 m | 8.74 s |
| su_r_77 | `proximity_fuse` | 10.0 m | 9.06 s |

Game/anchor requirement is qualitative ("both must still fuse", miss
distance <15 m): M0 satisfies it for both missiles.

---

## Prediction verdicts

### P1 -- "peak G overshoots game by ~1.7-1.9x at matched alpha" -- **HIT**

`packed_lift_slope_scale` moving 0.58 -> 1.0 is an exact 1/0.58 =
1.724x multiplier on `F_N` at any fixed (alpha, eta_q), a direct
algebraic consequence of the linear packed-lift law -- and since the
shipped 0.58 was fit precisely to make the model match the game's
displayed G at matched (alpha, eta) (R^2=0.97-0.998 per the audit),
this analytically predicts G_M0/G_game ~= 1.724 at matched alpha,
landing almost exactly inside the pre-registered 1.7-1.9x band. The
one empirical point in this dataset where alpha is genuinely closely
matched between M0 and the game (A7 @ t=2.9s: 6.84 vs 6.3 deg) gives a
G ratio of 2.09x -- same order, close to the predicted band. Ratios
measured at *matched time* instead grow far past that band later in
the same run (up to 6.96x by t=4.4s) purely because alpha itself stops
being matched (M0's de-load decays slower) -- a real secondary effect,
but not what P1 claimed, and the "at matched alpha" framing holds up
once that confound is accounted for.

### P2 -- "no alpha plateau, crank absent; expect degraded/failed intercepts on A3/A5-class geometries" -- **PARTIAL**

No-plateau clause: clean **HIT**. A3 peaks at 9.04 deg (vs the game's
14-15 deg plateau, wrong shape and wrong timing) and R-77-1's
launch-capture window shows 0.25-5.25 deg against the game's clean
21.7-23.7 deg plateau -- exactly the signature expected once the only
mechanism that produced early near-alpha_max effort (the launch-capture
handover) is removed.

Degraded/failed-intercept clause: **MISS**. All six M0 runs in this
suite (A1, A3, A7 x2 missiles, A5, A6 x2 missiles) still reach
`proximity_fuse` at essentially the 10 m trigger radius; none times
out or misses. This scenario suite uses only non-maneuvering,
straight/level or mildly climbing targets at moderate range, so PN's
terminal homing evidently has enough remaining time-to-go to absorb
several seconds of missing midcourse assist in these particular
geometries -- "degraded" shows up in the transient G/alpha/timing
history (confirmed above), not in this suite's binary hit/miss outcome.

### P3 -- "thrust/drag-dominated speed profiles remain within a few percent except during high-alpha phases" -- **PARTIAL**

A1, the one scenario here that is both a clean scenario-matched replay
comparison and genuinely low-alpha/thrust-drag-dominated, shows a
+7.3% terminal-speed deviation -- an order of magnitude smaller than
the largest G/alpha deviations above (up to +596% at A7 t=4.4s, -97%
at R-77-1 t=1.1s), consistent with the prediction's spirit that speed
holds up much better than lift/guidance, but larger than "a few
percent" taken literally, and *not* concentrated
in a high-alpha phase: the root cause identified above (uniform ~9.1%
CdA0 reduction from dropping the x1.10 scale, compounding with a Mach
1.8-3.5 interpolation gap this flight's cruise sits almost entirely
inside) is a zero-alpha-term effect. The R-77-1 speed points (+20.5%,
+8.8%) point the same direction but are not a clean test of this
prediction, since that scenario is a mission-specified kinematic
stand-in rather than a geometry-matched replay for that missile (see
caveat above).

---

## Pytest suite (unaffected by M0's additive files)

```
.venv/bin/python -m pytest -q
...
4 failed, 238 passed, 1 xfailed in 13.49s
```

Failures are the pre-existing, documented ones, all `FileNotFoundError`
for missing `data/raw/statshark_h4/*` / `outputs/h5_body_alpha2/*`
fixture files, unrelated to M0:

- `tests/test_h5_formal_ingest.py::test_h5_formal_matrix_preserves_failures_and_budget`
- `tests/test_h5_formal_ingest.py::test_h5_formal_alpha_gates_use_displayed_window_values`
- `tests/test_h5_formal_ingest.py::test_h5_formal_fit_remains_blocked_without_fabricated_parameters`
- `tests/test_h5_ingest.py::test_frozen_h4_visible_rows_have_required_alpha_and_source_boundary`

Plus the pre-existing `xfail(strict=True)`:
`tests/test_replay_anchors.py::test_a2_30deg_fast_pl12_anchor`.

No test imports or otherwise depends on `config/profile_m0_strict.json`
or `scripts/run_m0_scenarios.py`; both are new, additive files, and the
shipped `config/profile_h2_runtime_defaults.json` (which every existing
test and `missile_gui.library.scan_library()` call still reads) was not
modified.
