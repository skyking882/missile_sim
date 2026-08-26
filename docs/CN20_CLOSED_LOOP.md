# CN20 Closed-Loop Exam -- Fixed-CN Lift Law x Momentum-Tilt Induced Drag

**Harness: closed loop (guidance + control + rotation all active), unlike
`docs/ALPHA_PLAYBACK_R77_1.md`'s open-loop alpha-playback harness. This is an
exam harness -- CN_alpha is fixed at 19.9 and never adjusted based on the
results below.**

- Missile / scenario: `missiles/su_r_77_1.json` +
  `config/profile_h2_runtime_defaults.json`, R-77-1 level shot (the same A7
  geometry `scripts/run_m0_scenarios.py`'s `run_a7_r77_1()` and
  `tests/test_replay_anchors.py`'s A7 anchor use): launch 1200 km/h @ 6300 m,
  pitch/heading 0; target 1200 km/h @ 6300 m, 8000 m, azimuth 30 deg,
  heading 0, no turn; loft enabled, `ideal_truth`, `statshark_relative_to_los`,
  max 40 s.
- Game truth: `data/replays/r77_1_level_20260824.tsv` (46-frame digitized
  R-77-1 level-launch replay), compared at every frame's own
  `missile_flight_time_s`.
- Script: `scripts/cn20_closed_loop.py` (prints the tables reproduced below;
  performs no filesystem writes).
- Four variants, all built via
  `aim120_model.profile_adapter.build_h2_candidate_config` on an in-memory
  deep copy of the profile with the Step-1 keys injected into
  `aerodynamics`; `missiles/su_r_77_1.json` and
  `config/profile_h2_runtime_defaults.json` on disk are untouched:
  - **V1 shipped**: as-is (`packed_lift_force_eta_law` force law, shipped
    `cx_vs_aoa` alpha^2 induced drag).
  - **V2 shipped + momentum_tilt**: `induced_drag_mode="momentum_tilt"` added.
  - **V3 fixed_cn 19.9**: `packed_lift_fixed_cn={"cn_alpha_per_rad": 19.9}`
    added (overrides `packed_lift_force_eta_law` for the force channel).
  - **V4 fixed_cn 19.9 + momentum_tilt**: both keys added.

## Pre-registered predictions (written before any result below was read)

- **P1**: "de-load-arc force response of fixed_cn ~= eta-law variant (this
  flight cannot separate them)"
- **P2**: "crank release still early (~1.9 s vs 2.1) and a late catch-up G
  spike above game persists in all variants (guidance-side residual)"
- **P3**: "momentum_tilt variants track game speed within ~+/-40 km/h
  through t=4.3 (shipped drag runs ~+150 hot) and improve endgame G realism
  via the q feedback"

---

## (i) Per-variant summary

Game truth: alpha-release reference (see note below) `~= 2.1 s`; peak alpha
`23.7 deg`; late-arc peak G (`t>5.5s`) `17.1`; G at `t=2.1/3.4/6.9 s` =
`17.8 / 12.3 / 17.1`; speed(km/h) at `t=2.3/4.3/6.9 s` = `1858 / 2686 / 4032`;
termination = proximity fuse, `~7.2-7.4 s`.

Note on alpha-release: the replay is a sparse, ~0.15-0.25 s-spaced 46-frame
table. Applying "first tabulated row below 17 deg after the peak" literally
to that coarse grid lands at `t=2.7s` (game alpha: `...,2.5->18.4,
2.7->16.3`), not `2.1s`. The `~2.1s` reference instead comes from prior
analysis already in this repo -- `src/aim120_model/profile_adapter.py`'s
`midcourse_lead_turn.blend_time_s` assumption note and
`docs/M0_RESIDUALS.md`'s R-77-1 table both independently place the
launch-capture alpha-plateau's visible end at "about 2.1 s". The 17 deg rule
*is* applied, consistently, to each variant's own dense (`dt=0.02s`) model
curve below, which is the fair way to compare a continuous simulated curve
against that reference.

| variant | release t (s) | peak alpha (deg) | late-arc peak G (t>5.5) | G@2.1 | G@3.4 | G@6.9 | v@2.3 (km/h) | v@4.3 (km/h) | v@6.9 (km/h) | termination | min dist (m) | flight time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 shipped | 1.92 | 19.95 | 31.66 | 14.57 | 10.88 | 13.57 | 2110 | 3007 | 4173 | proximity_fuse | 10.00 | 7.404 |
| V2 shipped+momentum_tilt | 1.92 | 19.93 | 31.40 | 13.59 | 10.90 | 22.31 | 2017 | 2873 | 3936 | proximity_fuse | 10.00 | 7.579 |
| V3 fixed_cn 19.9 | 1.94 | 19.95 | 32.35 | 14.26 | 10.85 | 15.54 | 2110 | 3003 | 4154 | proximity_fuse | 10.00 | 7.411 |
| V4 fixed_cn 19.9+momentum_tilt | 1.92 | 19.91 | 31.98 | 13.43 | 10.79 | 23.45 | 2018 | 2873 | 3923 | proximity_fuse | 10.00 | 7.582 |
| **Game** | **~2.1** | **23.7** | **17.1** | **17.8** | **12.3** | **17.1** | **1858** | **2686** | **4032** | **proximity fuse** | **--** | **~7.2-7.4** |

All four variants reach `proximity_fuse` at `min_distance=10.00 m` (the
profile's `proximity_radius_m`). Peak alpha undershoots the game (~19.9-20.0
deg modeled vs 23.7 deg game) by nearly the same amount in all four
variants -- this is a pre-existing guidance-side characteristic of this
scenario, not something either Step-1 switch changes, and it confounds the
closed-loop speed comparison under P3 below (see verdicts).

## (ii) Mean |delta G|, de-load [2.3, 4.9] s and endgame [5.5, 7.1] s windows

| variant | de-load mean\|dG\| (n=17) | endgame mean\|dG\| (n=11) | combined |
|---|---|---|---|
| V1 shipped | 2.702 | 14.234 | 16.936 |
| V2 shipped+momentum_tilt | 2.337 | 13.699 | 16.036 |
| V3 fixed_cn 19.9 | 2.586 | 14.557 | 17.143 |
| V4 fixed_cn 19.9+momentum_tilt | 2.311 | 13.958 | 16.269 |

Best variant by combined (unweighted sum of the two window means, decided
before inspecting the per-frame table): **V2 (shipped + momentum_tilt)**,
combined `16.036`, narrowly ahead of V4 (`16.269`); both momentum_tilt
variants beat both non-momentum_tilt variants, and within each drag-mode
pair the shipped eta-law variant is marginally better than its fixed_cn
counterpart (V1 vs V3: `16.936` vs `17.143`; V2 vs V4: `16.036` vs `16.269`)
-- consistent with, not contradicting, P1's "cannot separate them" claim,
since the margins in every pair are 1.2-1.4% of the combined score.

## (iii) Per-frame table, best variant (V2: shipped + momentum_tilt)

```
id         t  G_game   G_mod     dG  a_game   a_mod   v_game    v_mod      dv
S001   0.200    4.40    1.80  -2.60    11.3     3.8   1237.0   1288.4    51.4
S002   0.400    1.20    5.87   4.67     4.3    11.6   1317.0   1373.9    56.9
S003   0.500    3.40    7.69   4.29     7.9    14.7   1401.0   1414.0    13.0
S004   0.700   10.40   10.21  -0.19    19.4    18.4   1475.0   1488.1    13.1
S005   0.900   13.40   11.70  -1.70    23.2    19.6   1529.0   1556.4    27.4
S006   1.100   13.20   12.69  -0.51    21.7    19.9   1579.0   1621.6    42.6
S007   1.300   14.50   13.49  -1.01    22.8    19.8   1622.0   1685.0    63.0
S008   1.500   15.80   14.03  -1.77    23.7    19.4   1662.0   1747.7    85.7
S009   1.600   16.60   14.17  -2.43    23.7    18.9   1706.0   1779.1    73.1
S010   1.900   17.40   14.08  -3.32    23.5    17.1   1759.0   1876.2   117.2
S011   2.100   17.80   13.59  -4.21    22.9    15.4   1802.0   1944.7   142.7
S012   2.300   17.10   12.91  -4.19    20.6    13.6   1858.0   2017.3   159.3
S013   2.500   16.20   12.64  -3.56    18.4    12.4   1927.0   2093.1   166.1
S014   2.700   15.40   12.61  -2.79    16.3    11.6   1997.0   2170.6   173.6
S015   2.900   14.60   12.45  -2.15    14.5    10.6   2064.0   2249.8   185.8
S016   3.100   13.70   12.05  -1.65    12.9     9.6   2127.0   2331.1   204.1
S017   3.200   13.00   11.75  -1.25    11.7     9.0   2185.0   2372.6   187.6
S018   3.400   12.30   10.90  -1.40    10.5     7.8   2245.0   2457.6   212.6
S019   3.500   11.10   10.36  -0.74     9.0     7.1   2317.0   2501.2   184.2
S020   3.700    9.80    9.00  -0.80     7.6     5.7   2391.0   2590.5   199.5
S021   3.800    9.00    8.17  -0.83     6.6     5.0   2457.0   2636.2   179.2
S022   4.000    7.70    6.24  -1.46     5.4     3.6   2535.0   2729.5   194.5
S023   4.100    6.70    5.14  -1.56     4.5     2.8   2605.0   2777.0   172.0
S024   4.300    5.60    2.94  -2.66     3.6     1.5   2686.0   2873.2   187.2
S025   4.500    4.20    2.60  -1.60     2.7     1.2   2758.0   2970.4   212.4
S026   4.600    2.90    3.80   0.90     1.9     1.7   2841.0   3019.0   178.0
S027   4.800    2.20    7.22   5.02     1.5     3.1   2915.0   3115.3   200.3
S028   4.900    2.00    9.15   7.15     1.3     3.8   2979.0   3162.6   183.6
S029   5.100    2.60   13.23  10.63     1.5     5.1   3065.0   3254.8   189.8
S030   5.200    3.00   15.32  12.32     1.6     5.8   3140.0   3299.3   159.3
S031   5.300    4.80   17.40  12.60     2.2     6.4   3204.0   3342.7   138.7
S032   5.500    5.60   21.45  15.85     2.3     7.4   3291.0   3425.4   134.4
S033   5.600    8.50   23.35  14.85     3.2     7.9   3366.0   3464.7    98.7
S034   5.800    8.70   26.76  18.06     3.1     8.6   3441.0   3539.3    98.3
S035   6.000   11.90   29.42  17.52     3.9     9.0   3527.0   3609.6    82.6
S036   6.100   12.10   30.39  18.29     3.7     9.1   3623.0   3643.5    20.5
S037   6.300   14.30   31.38  17.08     4.1     9.0   3708.0   3710.6     2.6
S038   6.500   15.70   30.79  15.09     4.3     8.5   3794.0   3779.3   -14.7
S039   6.600   16.10   29.74  13.64     4.2     8.0   3878.0   3815.3   -62.7
S040   6.800   16.10   25.64   9.54     4.0     6.6   3943.0   3893.1   -49.9
S041   6.900   17.10   22.31   5.21     4.0     5.6   4032.0   3935.7   -96.3
S042   7.100   16.50   10.93  -5.57     4.2     3.0   4036.0   3958.4   -77.6
S043   7.300    7.10    0.75  -6.35     2.0     0.2   4000.0   3913.4   -86.6
S044   7.500    5.70   19.22  13.52     1.7     5.6   3968.0   3866.5  -101.5
S045   7.700    3.60   23.77  20.17     1.2     7.0   3928.0   3842.8   -85.2
S046   7.800   -0.30   23.77  24.07     0.2     7.0   3895.0   3842.8   -52.2
```

(`G_game`/`a_game`/`v_game` are the replay's `overload_g`,
`angle_of_attack_deg`, `speed_kmh`; `_mod` columns are V2's nearest-sample
values at that same `missile_flight_time_s`; `dG`/`dv` = model - game.)

---

## Prediction verdicts

### P1 -- "de-load-arc force response of fixed_cn ~= eta-law variant (this flight cannot separate them)" -- **HIT**

Holding `induced_drag_mode` fixed and comparing only the force-channel
switch: V1 vs V3 differ by `<=0.31 g` at G@2.1/3.4 (`14.57` vs `14.26`,
`10.88` vs `10.85`), `0.12 g` in de-load-window mean\|dG\| (`2.702` vs
`2.586`, a 4.5% relative gap), and are identical in peak alpha (`19.95` vs
`19.95`) and within `0.02 s` in alpha-release time. V2 vs V4 (both
`momentum_tilt`) show the same pattern: de-load mean\|dG\| `2.337` vs
`2.311`, G@2.1 `13.59` vs `13.43`. Scoped to the de-load arc specifically,
the two force laws are indeed close to indistinguishable on this flight, as
predicted. (A larger, ~14% gap opens at `t=6.9s`, G@6.9 `13.57` vs `15.54`
for V1/V3 -- outside the de-load arc, in the late high-Mach/high-eta region
where the eta-law's `force_k` and the fixed-CN law diverge most; this does
not contradict the de-load-arc-scoped claim.)

### P2 -- "crank release still early (~1.9 s vs 2.1) and a late catch-up G spike above game persists in all variants (guidance-side residual)" -- **HIT**

Alpha-release time is `1.92-1.94 s` in all four variants against the `~2.1
s` game reference -- early, and matching the predicted `~1.9 s` figure
almost exactly. Late-arc peak G is `31.4-32.35` in all four variants against
the game's `17.1` (roughly `1.8-1.9x` too high), with a spread of less than
`1 g` out of `~32 g` (`<3%`) across all four variants despite the force law
and drag law both changing between them -- strong evidence this specific
residual is a guidance-side artifact (visible in the per-frame table above
as a sharp G ramp from `S029` (`t=5.1s`) that the game does not show until
much later and much smaller) rather than an aerodynamic-model artifact.
Both clauses hold.

### P3 -- "momentum_tilt variants track game speed within ~+/-40 km/h through t=4.3 (shipped drag runs ~+150 hot) and improve endgame G realism via the q feedback" -- **MISS**

Directionally right, magnitude and second clause wrong:

- **Speed tracking magnitude**: `momentum_tilt` cuts the "hot" bias
  substantially (V1 `+252/+321 km/h` at `t=2.3/4.3s` -> V2 `+159/+187
  km/h`), a genuine ~35-40% error reduction, but the closest of these
  (`+159 km/h`) is still 4x outside the predicted `+/-40 km/h` band.
- **"+150 hot" baseline**: does not hold in the closed loop. Shipped (V1)
  actually runs `+252/+321 km/h` hot at `t=2.3/4.3s`, hotter than the
  `+150`-ish figure this prediction was evidently anchored to (the
  open-loop alpha-playback exam's `D_1` deltas of `+152.6/+183.2 km/h` in
  `docs/ALPHA_PLAYBACK_R77_1.md`). The gap: that exam replays the
  replay's own measured `alpha(t)` open-loop, while this closed-loop
  exam's own guidance undershoots the replay's alpha (peak `19.9-20.0 deg`
  modeled vs `23.7 deg` game, all four variants, see (i) above) --
  less commanded turning compounds with the drag-law gap, so the same
  drag-law change produces a larger "hot" bias here than in the open-loop
  diagnostic.
- **Endgame G realism**: does not improve under `momentum_tilt` at the
  specific `t=6.9s` checkpoint the mission specified (V1 error
  `|13.57-17.1|=3.53 g` vs V2 error `|22.31-17.1|=5.21 g` -- worse), though
  the window-*averaged* endgame mean\|dG\| is marginally better
  (`14.234 -> 13.699`, ~3.8% relative). Mechanism visible in the per-frame
  table: `momentum_tilt`'s induced drag scales with the same late-arc alpha
  spike identified as a guidance-side residual under P2, so it amplifies
  that region's speed error rather than damping it -- by `t=6.9-7.8s` the
  momentum_tilt variants' speed delta has flipped sign, from hot to cold
  (`-96` to `-109 km/h`), rather than converging toward zero.

---

## Regression

`.venv/bin/python -m pytest -q` gives the exact pre-existing baseline:
`4 failed, 238 passed, 1 xfailed`. The 4 failures
(`test_h5_formal_ingest.py` x3, `test_h5_ingest.py` x1, pre-existing missing
fixture files unrelated to this change) and the 1 xfail (`test_a2_...`) are
unchanged from the baseline run captured before any Step-1 edit. Neither new
`aerodynamics` key (`packed_lift_fixed_cn`, `induced_drag_mode`) appears in
any shipped profile or the shipped runtime defaults, so every existing test
continues to exercise shipped behavior only.
