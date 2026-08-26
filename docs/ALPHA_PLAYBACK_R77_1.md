# Alpha-Playback Open-Loop Plant Exam -- R-77-1 Level Launch

**Harness: open-loop alpha playback. Guidance, control, and rotation are fully
bypassed. This is an exam harness -- no parameter was adjusted based on the
results below.**

- Data: `data/replays/r77_1_level_20260824.tsv` (46-frame digitized R-77-1
  level-launch replay, launch altitude ~6300 m, level flight, straight
  target; `displayed_distance_km`/`displayed_flown_distance_km` are the
  replay camera distance and are non-physical, unused here).
- Script: `scripts/alpha_playback.py` (prints the tables reproduced below;
  performs no filesystem writes).
- Model: `missiles/su_r_77_1.json` + `config/profile_h2_runtime_defaults.json`
  built via `aim120_model.profile_adapter.build_h2_candidate_config`, the
  same wiring `scripts/run_m0_scenarios.py` and the test suite use.
  `config/profile_m0_strict.json` is additionally built, the same way, to
  source the V_C ("M0 strict", `packed_lift_slope_scale=1.0`) constant.
- Every constant used below (`fins_lateral_acceleration_g`,
  `horizontal_fin_aoa_limit_deg`, `packed_lift_slope_scale`, the per-missile
  `packed_lift_force_eta_law`, propulsion stage thrust/duration/mass, `cx_k`,
  wing area multiplier, caliber, the shipped Cx(M) drag curve, `cx_vs_aoa`,
  and base indicated speed) is read out of the built config, not hardcoded.
  Mode 2's drag terms call the shipped `aim120_model.drag_models.effective_cda0`
  / `effective_cda_alpha` functions directly rather than re-deriving the
  Cx(M) interpolation by hand.

## Method

**Per-frame atmosphere reconstruction** (both modes), from the replay's own
speed and Mach: `a_sound = v/M`, `T_K = (a_sound/20.0468)^2`,
`rho = 1.225*(T_K/288.15)^4.2559`, `q = 0.5*rho*v^2`, `eta = q/q_base` with
`q_base = 0.5*1.225*(1800/3.6)^2`. Propulsion: `m(t) = 190 - (71/7)*t` for
`t<=7`, else `119`; `T(t) = 24350` N for `t<=7`, else `0` (sourced from
`config["propulsion"]["stages"][0]` via `PiecewisePropulsion`). Displayed-G
semantics: `G_disp = G_aero_path + T*sin(alpha)/(m*g)`.

**MODE 1** (state-matched, no integration) evaluates, at each frame's own
measured `(t, M, alpha)`, three force-law variants for the packed-lift path
G:

- `V_A` "constant 0.58": `k = packed_lift_slope_scale` (shipped default).
- `V_B` "shipped eta-law": `k = 0.574*clamp(eta, 0.35, 2.65)^0.242` (the
  per-missile `su_r_77_1.json` `packed_lift_force_eta_law`).
- `V_C` "M0 strict": `k = 1.0` (`config/profile_m0_strict.json`).
- `G_pred = k*A*eta*(alpha/alpha_max) + T*sin(alpha)/(m*g)`, with the
  alpha-disk clamp at `alpha_max` (alpha never reaches it in this data: max
  observed is 23.7 deg vs `alpha_max = 26.4026 deg`).

Frames with `t<0.5 s` or `alpha<1.0 deg` (S001, S002, S046) are excluded from
the mean-error statistics as launch transient / display noise, but are still
printed in the full table.

**MODE 2** (integrated open-loop energy exam) starts at the S001 state
(`t=0.2 s`, `v=1237/3.6 m/s`) and integrates `dv/dt = (T*cos(alpha) -
D_total)/m` with explicit Euler, `dt=0.005 s`, from `t=0.2` to `t=7.8 s`, at
a fixed 6300 m altitude (atmosphere held constant at the ISA-6300 sample for
the whole run). `alpha(t)` is linearly interpolated between the replay's own
`(missile_flight_time_s, angle_of_attack_deg)` samples. Two drag variants:

- `D_1` "shipped": `D = q*S_w*CxK*C_et(M) + q*S_d*cx_vs_aoa*alpha^2`, exactly
  the shipped induced-drag law (`effective_cda0`/`effective_cda_alpha` called
  directly on the built config; `S_d = pi*0.2^2/4`, `S_w = S_d*1.45`, `C_et`
  interpolated from the shipped 15-knot Cx(M) table).
- `D_2` "momentum-tilt diagnostic": `D = q*S_w*CxK*C_et(M) + L_aero*tan(alpha)`,
  where `L_aero = k_B(eta)*A*eta*(alpha/alpha_max)*m*g` uses the `V_B` force
  law. **This is a diagnostic only, not a model change**: it swaps the
  induced-drag term for a flight-path-tilt term built from the same `V_B`
  packed-lift force, to see whether the shipped induced-drag term or the
  packed-lift force itself better accounts for the replay's measured
  deceleration/acceleration.

A separate cross-check re-evaluates `V_B` at 3 frames (early/mid/late)
through the real model code path
(`aim120_model.h2_dynamics.forces_for_state_h2`): a `SimState` is built at
the frame's own level velocity and an altitude inverted from the same
per-frame `T_K` reconstruction (so the shipped `StandardAtmosphere` sample
reproduces the identical `T_K`/`rho`/`q` used by the closed-form analytic
formula), with `pitch = alpha` (so the state's AoA equals the frame alpha)
and fin angle `= alpha` as trim. `diagnostics.trajectory_pitch_normal_acceleration_g`
is compared directly against the closed-form `G_B`.

## MODE 1 -- per-frame table

```
id         t     M    eta  a_deg  G_game     G_A     G_B     G_C   |eA|   |eB|   |eC|  incl
S001   0.200  1.09  0.246   11.3    4.40    5.86    5.10    8.24   1.46   0.70   3.84     n
S002   0.400  1.16  0.280    4.3    1.20    2.42    2.09    3.45   1.22   0.89   2.25     n
S003   0.500  1.24  0.304    7.9    3.40    4.67    4.02    6.72   1.27   0.62   3.32     Y
S004   0.700  1.30  0.350   19.4   10.40   12.49   10.64   18.27   2.09   0.24   7.87     Y
S005   0.900  1.35  0.370   23.2   13.40   15.51   13.27   22.83   2.11   0.13   9.43     Y
S006   1.100  1.39  0.405   21.7   13.20   15.47   13.35   22.96   2.27   0.15   9.76     Y
S007   1.300  1.43  0.422   22.8   14.50   16.76   14.53   24.95   2.26   0.03  10.45     Y
S008   1.500  1.46  0.456   23.7   15.80   18.44   16.13   27.66   2.64   0.33  11.86     Y
S009   1.600  1.50  0.477   23.7   16.60   19.06   16.76   28.70   2.46   0.16  12.10     Y
S010   1.900  1.55  0.498   23.5   17.40   19.57   17.32   29.55   2.17   0.08  12.15     Y
S011   2.100  1.59  0.517   22.9   17.80   19.66   17.48   29.74   1.86   0.32  11.94     Y
S012   2.300  1.64  0.548   20.6   17.10   18.52   16.60   28.14   1.42   0.50  11.04     Y
S013   2.500  1.70  0.592   18.4   16.20   17.58   15.93   26.86   1.38   0.27  10.66     Y
S014   2.700  1.76  0.641   16.3   15.40   16.58   15.22   25.49   1.18   0.18  10.09     Y
S015   2.900  1.82  0.682   14.5   14.60   15.51   14.37   23.93   0.91   0.23   9.33     Y
S016   3.100  1.87  0.742   12.9   13.70   14.77   13.88   22.93   1.07   0.18   9.23     Y
S017   3.200  1.92  0.787   11.7   13.00   14.03   13.32   21.88   1.03   0.32   8.88     Y
S018   3.400  1.98  0.805   10.5   12.30   12.86   12.25   20.07   0.56   0.05   7.77     Y
S019   3.500  2.04  0.870    9.0   11.10   11.73   11.34   18.41   0.63   0.24   7.31     Y
S020   3.700  2.11  0.909    7.6    9.80   10.28   10.01   16.17   0.48   0.21   6.37     Y
S021   3.800  2.16  0.991    6.6    9.00    9.58    9.49   15.16   0.58   0.49   6.16     Y
S022   4.000  2.23  1.049    5.4    7.70    8.23    8.24   13.06   0.53   0.54   5.36     Y
S023   4.100  2.29  1.115    4.5    6.70    7.22    7.31   11.49   0.52   0.61   4.79     Y
S024   4.300  2.36  1.190    3.6    5.60    6.11    6.27    9.76   0.51   0.67   4.16     Y
S025   4.500  2.43  1.226    2.7    4.20    4.71    4.86    7.53   0.51   0.66   3.33     Y
S026   4.600  2.50  1.314    1.9    2.90    3.51    3.68    5.64   0.61   0.78   2.74     Y
S027   4.800  2.57  1.361    1.5    2.20    2.86    3.02    4.60   0.66   0.82   2.40     Y
S028   4.900  2.62  1.452    1.3    2.00    2.62    2.81    4.23   0.62   0.81   2.23     Y
S029   5.100  2.70  1.516    1.5    2.60    3.15    3.40    5.08   0.55   0.80   2.48     Y
S030   5.200  2.77  1.572    1.6    3.00    3.47    3.77    5.61   0.47   0.77   2.61     Y
S031   5.300  2.82  1.669    2.2    4.80    5.02    5.54    8.15   0.22   0.74   3.35     Y
S032   5.500  2.90  1.743    2.3    5.60    5.46    6.08    8.88   0.14   0.48   3.28     Y
S033   5.600  2.96  1.855    3.2    8.50    8.03    9.07   13.09   0.47   0.57   4.59     Y
S034   5.800  3.03  1.917    3.1    8.70    8.02    9.13   13.09   0.68   0.43   4.39     Y
S035   6.000  3.11  1.991    3.9   11.90   10.45   11.99   17.07   1.45   0.09   5.17     Y
S036   6.100  3.19  2.127    3.7   12.10   10.51   12.25   17.22   1.59   0.15   5.12     Y
S037   6.300  3.27  2.198    4.1   14.30   12.02   14.11   19.70   2.28   0.19   5.40     Y
S038   6.500  3.34  2.336    4.3   15.70   13.32   15.87   21.88   2.38   0.17   6.18     Y
S039   6.600  3.42  2.404    4.2   16.10   13.36   16.02   21.97   2.74   0.08   5.87     Y
S040   6.800  3.47  2.530    4.0   16.10   13.34   16.19   21.97   2.76   0.09   5.87     Y
S041   6.900  3.55  2.635    4.0   17.10   13.85   16.97   22.83   3.25   0.13   5.73     Y
S042   7.100  3.56  2.599    4.2   16.50   12.85   16.03   22.16   3.65   0.47   5.66     Y
S043   7.300  3.53  2.543    2.0    7.10    5.99    7.42   10.32   1.11   0.32   3.22     Y
S044   7.500  3.50  2.513    1.7    5.70    5.03    6.22    8.67   0.67   0.52   2.97     Y
S045   7.700  3.46  2.491    1.2    3.60    3.52    4.34    6.07   0.08   0.74   2.47     Y
S046   7.800  3.43  2.455    0.2   -0.30    0.58    0.71    1.00   0.88   1.01   1.30     n
```

(`M` = frame Mach; `eta` = q/q_base, ratio_max=4.0-capped, not binding in this
data; `incl` = included in the per-band error stats below.)

## MODE 1 -- mean |error| per Mach band

(frames with `t>=0.5 s` and `alpha>=1.0 deg` only; n=43 of 46 frames)

```
band            n  mean|eA|  mean|eB|  mean|eC|
M<1.7          10     2.056     0.256     9.993
1.7<=M<2.4     12     0.783     0.332     7.510
2.4<=M<2.9      7     0.520     0.770     2.736
M>=2.9         14     1.660     0.318     4.708
ALL            43     1.322     0.381     6.398
```

## MODE 1 -- code-path cross-check

`V_B` evaluated analytically (closed-form) vs through
`aim120_model.h2_dynamics.forces_for_state_h2` on the real `su_r_77_1`
config, at 3 frames (early/mid/late):

```
id         t  a_deg  alt_eq_m  M_frame   M_code  G_B_analytic   G_B_code  rel_diff_%
S006   1.100   21.7    6213.1   1.3900   1.3900       13.3526    13.3526      0.0002
      code-path AoA=21.7000 deg (frame 21.7000); q_analytic=61967.98 Pa, q_code=61968.17 Pa (rel diff 0.00031%)
S023   4.100    4.5    6106.8   2.2900   2.2900        7.3113     7.3114      0.0003
      code-path AoA=4.5000 deg (frame 4.5000); q_analytic=170674.03 Pa, q_code=170674.55 Pa (rel diff 0.00030%)
S040   6.800    4.0    6190.3   3.4700   3.4700       16.1899    16.1900      0.0003
      code-path AoA=4.0000 deg (frame 4.0000); q_analytic=387402.96 Pa, q_code=387404.14 Pa (rel diff 0.00031%)
```

`alt_eq_m` is the altitude whose `StandardAtmosphere` sample reproduces the
frame's own reconstructed `T_K` (6106.8-6213.1 m across the three frames,
consistent with the stated ~6300 m level-flight launch altitude). Mach,
AoA, and dynamic pressure recomputed by the real code path match the frame
inputs and the closed-form analytic values to within 0.0002-0.0003%
(requested tolerance was ~1%; no code-path wiring obstacle was encountered).
The residual ~0.0003% is consistent with the difference between the rounded
reconstruction constants `20.0468`/`4.2559` used here and their
full-precision equivalents (`sqrt(1.4*287.05287)=20.046812...`,
`9.80665/(287.05287*0.0065)-1=4.255881...`) inside the shipped
`StandardAtmosphere`.

## MODE 2 -- integrated v(t) vs game speed

Fixed-altitude atmosphere at 6300 m: `T=247.200 K`, `rho=0.63800 kg/m3`,
`a=315.188 m/s`.

```
id         t  M_game   v_game  v1_shipped   v2_diag      d1      d2
S001   0.200    1.09   1237.0      1237.0    1237.0     0.0     0.0
S002   0.400    1.16   1317.0      1324.2    1323.2     7.2     6.2
S003   0.500    1.24   1401.0      1368.3    1366.9   -32.7   -34.1
S004   0.700    1.30   1475.0      1451.7    1446.4   -23.3   -28.6
S005   0.900    1.35   1529.0      1526.5    1511.0    -2.5   -18.0
S006   1.100    1.39   1579.0      1598.9    1571.0    19.9    -8.0
S007   1.300    1.43   1622.0      1670.9    1629.8    48.9     7.8
S008   1.500    1.46   1662.0      1740.4    1683.6    78.4    21.6
S009   1.600    1.50   1706.0      1774.3    1709.0    68.3     3.0
S010   1.900    1.55   1759.0      1874.9    1782.8   115.9    23.8
S011   2.100    1.59   1802.0      1941.6    1831.3   139.6    29.3
S012   2.300    1.64   1858.0      2010.6    1883.7   152.6    25.7
S013   2.500    1.70   1927.0      2083.6    1943.1   156.6    16.1
S014   2.700    1.76   1997.0      2160.2    2008.7   163.2    11.7
S015   2.900    1.82   2064.0      2239.9    2079.6   175.9    15.6
S016   3.100    1.87   2127.0      2322.4    2154.8   195.4    27.8
S017   3.200    1.92   2185.0      2364.8    2194.1   179.8     9.1
S018   3.400    1.98   2245.0      2451.4    2275.6   206.4    30.6
S019   3.500    2.04   2317.0      2495.7    2317.9   178.7     0.9
S020   3.700    2.11   2391.0      2586.4    2406.0   195.4    15.0
S021   3.800    2.16   2457.0      2632.5    2451.3   175.5    -5.7
S022   4.000    2.23   2535.0      2726.1    2544.1   191.1     9.1
S023   4.100    2.29   2605.0      2773.5    2591.5   168.5   -13.5
S024   4.300    2.36   2686.0      2869.2    2687.8   183.2     1.8
S025   4.500    2.43   2758.0      2965.8    2785.5   207.8    27.5
S026   4.600    2.50   2841.0      3014.4    2834.8   173.4    -6.2
S027   4.800    2.57   2915.0      3112.1    2934.3   197.1    19.3
S028   4.900    2.62   2979.0      3161.0    2984.2   182.0     5.2
S029   5.100    2.70   3065.0      3258.9    3084.3   193.9    19.3
S030   5.200    2.77   3140.0      3307.9    3134.3   167.9    -5.7
S031   5.300    2.82   3204.0      3356.8    3184.1   152.8   -19.9
S032   5.500    2.90   3291.0      3454.3    3283.4   163.3    -7.6
S033   5.600    2.96   3366.0      3502.9    3332.7   136.9   -33.3
S034   5.800    3.03   3441.0      3599.8    3430.4   158.8   -10.6
S035   6.000    3.11   3527.0      3696.3    3527.2   169.3     0.2
S036   6.100    3.19   3623.0      3744.4    3575.2   121.4   -47.8
S037   6.300    3.27   3708.0      3840.5    3670.7   132.5   -37.3
S038   6.500    3.34   3794.0      3936.2    3765.2   142.2   -28.8
S039   6.600    3.42   3878.0      3984.1    3812.3   106.1   -65.7
S040   6.800    3.47   3943.0      4079.9    3907.0   136.9   -36.0
S041   6.900    3.55   4032.0      4127.9    3954.5    95.9   -77.5
S042   7.100    3.56   4036.0      4153.9    3979.4   117.9   -56.6
S043   7.300    3.53   4000.0      4105.0    3930.8   105.0   -69.2
S044   7.500    3.50   3968.0      4058.3    3886.1    90.3   -81.9
S045   7.700    3.46   3928.0      4012.7    3842.6    84.7   -85.4
S046   7.800    3.43   3895.0      3990.3    3821.4    95.3   -73.6
```

(`v_game`/`v1_shipped`/`v2_diag`/`d1`/`d2` all in km/h; `d1=v1_shipped-v_game`,
`d2=v2_diag-v_game`.)

## MODE 2 -- explicit speed deltas

```
   t_s  v_game_kmh    v1_kmh    v2_kmh   d1_kmh   d2_kmh
  2.30      1858.0    2010.6    1883.7    152.6     25.7
  4.30      2686.0    2869.2    2687.8    183.2      1.8
  6.90      4032.0    4127.9    3954.5     95.9    -77.5
  7.80      3895.0    3990.3    3821.4     95.3    -73.6
```

## MODE 2 -- delta sign-run / min / max summary

Computed directly (not eyeballed off the table above) across all 46 frame
times; `+`/`-`/`0` mark a maximal run of consecutive frames where
`predicted - game` has that sign:

```
D_1 shipped delta sign runs (sample_id@t -> sample_id@t, count):
  0 : S001@0.200s -> S001@0.200s  (n=1)
  + : S002@0.400s -> S002@0.400s  (n=1)
  - : S003@0.500s -> S005@0.900s  (n=3)
  + : S006@1.100s -> S046@7.800s  (n=41)
  min delta = -32.7 km/h at S003@0.500s
  max delta = +207.8 km/h at S025@4.500s
D_2 diagnostic delta sign runs (sample_id@t -> sample_id@t, count):
  0 : S001@0.200s -> S001@0.200s  (n=1)
  + : S002@0.400s -> S002@0.400s  (n=1)
  - : S003@0.500s -> S006@1.100s  (n=4)
  + : S007@1.300s -> S020@3.700s  (n=14)
  - : S021@3.800s -> S021@3.800s  (n=1)
  + : S022@4.000s -> S022@4.000s  (n=1)
  - : S023@4.100s -> S023@4.100s  (n=1)
  + : S024@4.300s -> S025@4.500s  (n=2)
  - : S026@4.600s -> S026@4.600s  (n=1)
  + : S027@4.800s -> S029@5.100s  (n=3)
  - : S030@5.200s -> S034@5.800s  (n=5)
  + : S035@6.000s -> S035@6.000s  (n=1)
  - : S036@6.100s -> S046@7.800s  (n=11)
  min delta = -85.4 km/h at S045@7.700s
  max delta = +30.6 km/h at S018@3.400s
```

## MODE 2 -- burnout peak speed

```
game measured peak            = 4036 km/h at t=7.1 s (S042)
D_1 shipped predicted peak    = 4178.2 km/h at t=7.005 s
D_2 diagnostic predicted peak = 4004.3 km/h at t=7.005 s
```

## Results summary (factual; state deltas only)

**MODE 1 (state-matched G, 43 in-stats frames of 46):**

- `ALL`-band mean |error|: `V_A` (constant 0.58) = 1.322 g; `V_B` (shipped
  eta-law) = 0.381 g; `V_C` (M0 strict, k=1.0) = 6.398 g.
- Per-band mean |error| for `V_B`: `M<1.7`=0.256 g, `1.7<=M<2.4`=0.332 g,
  `2.4<=M<2.9`=0.770 g, `M>=2.9`=0.318 g.
- Per-band mean |error| for `V_A`: `M<1.7`=2.056 g, `1.7<=M<2.4`=0.783 g,
  `2.4<=M<2.9`=0.520 g, `M>=2.9`=1.660 g.
- Per-band mean |error| for `V_C`: `M<1.7`=9.993 g, `1.7<=M<2.4`=7.510 g,
  `2.4<=M<2.9`=2.736 g, `M>=2.9`=4.708 g.
- `V_A` has the lowest mean |error| of the three variants only in the
  `2.4<=M<2.9` band (0.520 g vs `V_B`'s 0.770 g); `V_B` has the lowest mean
  |error| of the three variants in every other band and overall.
- Excluded frames (still shown in the table): S001 (`t=0.2s<0.5s`), S002
  (`t=0.4s<0.5s`), S046 (`alpha=0.2deg<1.0deg`).

**MODE 1 code-path cross-check (3 of 46 frames, early/mid/late):**

- S006 (`t=1.1s`): `G_B` analytic 13.3526 g vs code-path 13.3526 g, rel diff
  0.0002%.
- S023 (`t=4.1s`): `G_B` analytic 7.3113 g vs code-path 7.3114 g, rel diff
  0.0003%.
- S040 (`t=6.8s`): `G_B` analytic 16.1899 g vs code-path 16.1900 g, rel diff
  0.0003%.
- Recomputed Mach, AoA, and dynamic pressure from the code path matched the
  frame inputs and the analytic reconstruction at each of the 3 frames to
  within 0.0003%.

**MODE 2 (integrated speed, dt=0.005s, fixed 6300m, 46 frame-time
comparisons):**

- At `t=2.3s`: game 1858.0 km/h; `D_1` 2010.6 km/h (delta +152.6 km/h);
  `D_2` 1883.7 km/h (delta +25.7 km/h).
- At `t=4.3s`: game 2686.0 km/h; `D_1` 2869.2 km/h (delta +183.2 km/h);
  `D_2` 2687.8 km/h (delta +1.8 km/h).
- At `t=6.9s`: game 4032.0 km/h; `D_1` 4127.9 km/h (delta +95.9 km/h); `D_2`
  3954.5 km/h (delta -77.5 km/h).
- At `t=7.8s`: game 3895.0 km/h; `D_1` 3990.3 km/h (delta +95.3 km/h); `D_2`
  3821.4 km/h (delta -73.6 km/h).
- `D_1` delta (`predicted - game`) is 0 at S001 (`t=0.2s`, the start state),
  positive at S002 (`t=0.4s`, +7.2 km/h), negative for S003-S005
  (`t=0.5-0.9s`, range -32.7 to -2.5 km/h), then positive for every one of
  the remaining 41 frames from S006 (`t=1.1s`) through S046 (`t=7.8s`)
  (range +19.9 to +207.8 km/h). Over all 46 frames the minimum `D_1` delta
  is -32.7 km/h (S003, `t=0.5s`) and the maximum is +207.8 km/h (S025,
  `t=4.5s`).
- `D_2` delta is 0 at S001, positive at S002 (+6.2 km/h), negative for
  S003-S006 (`t=0.5-1.1s`, range -34.1 to -8.0 km/h), then mixed-sign with
  mostly positive runs from S007 through S029 (`t=1.3-5.1s`; single-frame
  negative dips at S021, S023, S026), then mixed-sign with mostly negative
  runs from S030 through S046 (`t=5.2-7.8s`; a single-frame positive dip at
  S035, `t=6.0s`, +0.2 km/h). Over all 46 frames the minimum `D_2` delta is
  -85.4 km/h (S045, `t=7.7s`) and the maximum is +30.6 km/h (S018,
  `t=3.4s`). Full sign-run breakdown above.
- Burnout peak speed: game 4036 km/h at `t=7.1s`; `D_1` predicted peak
  4178.2 km/h at `t=7.005s` (delta +142.2 km/h); `D_2` predicted peak 4004.3
  km/h at `t=7.005s` (delta -31.7 km/h).
