# M0 Provenance Audit

Classifies every numeric runtime parameter reachable by the H2
`critical_damped_v12` plant (the only `plant_model` exercised by
`config/profile_m0_strict.json`, and the only one used by `cn_pl12`,
`su_r_77`, `su_r_77_1` in this repo) into one of five classes:

- **DATAMINE** -- read verbatim (or by zero-parameter arithmetic on two
  datamine fields, e.g. `mass_lost_kg = mass - massEnd`) from the War
  Thunder public datamine, per the `parameter_sources` block already
  recorded in each `missiles/*.json` profile.
- **PHYSICS** -- a zero-parameter identity: rigid-body kinematics, a
  circle/rod geometry formula, a unit conversion, or an SI constant.
  Given the datamine inputs, no free number is chosen.
- **STANDARD-LAW** -- a published, externally-defined reference
  model (ISA atmosphere; the 1943 standard projectile drag-law shape)
  used as-is, not fitted to this project's game-replay telemetry.
- **REPLAY-FITTED** -- a numeric constant whose value was chosen by
  fitting or calibrating against digitized War Thunder replay
  telemetry (StatShark G/alpha frames, PL-12 35 km CdA inversion,
  etc.). Every row in this class states what M0 does with it.
- **ARCHITECTURAL-CHOICE** -- a discrete modeling decision (which
  equation form, which control path, which display basis) that is
  evidence-informed but is not itself a fitted numeric constant. Kept
  in M0; evidence is one line per row.

Sources consulted: `config/profile_h2_runtime_defaults.json` (the
`boundary` string), `docs/UPDATE_V1.0.1.md`, `docs/UPDATE_V1.0.2.md`,
`src/aim120_model/{h2_dynamics,profile_adapter,guidance,control,
drag_models,aerodynamics}.py`, `missiles/{cn_pl12,su_r_77,
su_r_77_1}.json` (each field's own `parameter_sources.kind`), and
`git log`/`git show` on `drag_models.py`, `profile_h2_runtime_defaults.json`
and `profile_adapter.py` for when/why each constant was introduced.

## Summary counts

| Class | Distinct parameters/groups audited | Disposition in M0 |
|---|---|---|
| DATAMINE | 22 | Unchanged |
| PHYSICS | 11 | Unchanged |
| STANDARD-LAW | 2 (ISA atmosphere; 1943 drag-law shape) | Unchanged (ISA); drag-law shape kept but rebuilt at scale x1.0 with the 5 telemetry-anchored knots removed |
| REPLAY-FITTED | 9 | **All 9 purged, structurally neutralized, or (the one historically-retired item) confirmed not applicable in M0** (see per-row disposition below) |
| ARCHITECTURAL-CHOICE | 7 | Kept as current architecture; logged with evidence |

Purged/neutralized REPLAY-FITTED list (detail in the table below):
`packed_lift_slope_scale` (0.58->1.0), the C_et(M) global x1.10 scale
(->x1.0), the 5 C_et(M) telemetry-anchored Mach knots (removed, not
merely rescaled), the whole `midcourse_lead_turn` mechanism (disabled),
its shared constants `turn_time_constant_s`/`lock_delay_s`/
`blend_time_s`/`fin_error_ref_rad`, the per-missile
`midcourse_lead_turn.blend_time_s=1.3` override on `su_r_77`/
`su_r_77_1` (moot once the mechanism is disabled), the per-missile
`packed_lift_force_eta_law` override on `su_r_77`/`su_r_77_1`
(stripped), and the IOG direct-fin-routing anti-windup "tracking mode"
heuristic thresholds in `control.py` (structurally inert once
midcourse is disabled -- no code change needed or made).

## Classification table

### DATAMINE

| Parameter | Value (example: PL-12 unless noted) | Evidence pointer |
|---|---|---|
| `geometry.initial_mass_kg`, `caliber_m`, `length_m` | 198 kg / 0.203 m / 3.93 m | `missiles/cn_pl12.json` `parameter_sources.{initial_mass_kg,caliber_m,length_m}.kind="datamine"`; fields `rocket.mass`/`caliber`/`length` |
| `geometry.wing_area_multiplier` | 1.4 | `rocket.wingAreaMult`, `parameter_sources.wing_area_multiplier` |
| `propulsion.stages[].thrust_n`, `duration_s` | 34960 N x2.5 s + 12880 N x5 s | `rocket.force`/`force1`, `rocket.timeFire`/`timeFire1` |
| `propulsion.stages[].mass_lost_kg`, `isp_s` | 38/28 kg; ISP derived | `kind="derived"`: zero-parameter arithmetic (`mass-massEnd`; `thrust/(mass_flow*g0)`) on two datamine numbers -- not a fit |
| `aerodynamics.cx_k` (CxK) | 1.6 (1.7 R-77, 1.65 R-77-1) | `rocket.CxK`; `profile_adapter.py:761` copies verbatim |
| `aerodynamics.fins_lateral_acceleration_g` (finsLatAccel, "A") | 41.4036 g (53.5864 g R-77/R-77-1) | `rocket.finsLatAccel`; UPDATE_V1.0.1.md Sec.5 "2.57 AIM-54 tuning diff verbatim-matches loadFactorMax/reqAccelMax 17->22 = published 'max load factor'" |
| `aerodynamics.fin_aoa_limit_rad` (finsAoa, "alpha_max") | 0.375092 rad (0.460812 rad R-77/R-77-1) | `rocket.finsAoaHor/Ver`; same AIM-54 diff evidence (0.218->0.436 rad = published 12.5deg->25deg) |
| `aerodynamics.distance_cm_to_stabilizer_m` (distFromCmToStab, "Delta") | 0.12 m (0.59 x caliber) | `rocket.distFromCmToStab`; UPDATE_V1.0.1.md Sec.6: 108-missile dimensional audit, 0.03-1.48 x caliber, median 0.65, self-consistent as static margin |
| `aerodynamics.lift_area_scale` | = wingAreaMult | `rocket.wingAreaMult` reused, `parameter_sources.lift_area_scale` |
| `performance.load_factor_max_g` (loadFactorMax, "n_max") | 38 g (50 g R-77/R-77-1) | `rocket.loadFactorMax` |
| `guidance.maximum_lateral_acceleration_g` (reqAccelMax, "n_req") | 38 g (50 g R-77/R-77-1) | `rocket.guidance.guidanceAutopilot.reqAccelMax` |
| `guidance.pn_gain` (propNavMult) | 4.0 | `rocket.guidance.guidanceAutopilot.propNavMult` |
| `guidance.lock_range_m` | 16000 m | `rocket.guidance.lockDistance` |
| `guidance.lofting_enabled`, `lofting_elevation_deg` | true, 20deg (24deg R-77/R-77-1) | `rocket.guidance.guidanceAutopilot.loftEnabled/loftElevation` |
| `guidance.proximity_radius_m` | 10 m | `rocket.proximityFuse.radius` |
| `guidance.maximum_angular_rate_deg_s` | 60 deg/s | `rocket.guidance.radarSeeker.rateMax` |
| `guidance.flight_time_gain_table` | `[[0.3,0.0],[0.31,1.0]]` | `rocket.guidance.guidanceAutopilot.timeToGain*` |
| `control.pid.p/i/d` (accelControlProp/Intg/Diff) | 0.0181/0.013/0.00025 (PL-12) | `rocket.guidance.guidanceAutopilot.accelControlProp/Intg/Diff` |
| `control.pid.integral_limit` (accelControlIntgLim) | 1.0 | `rocket.guidance.guidanceAutopilot.accelControlIntgLim` |
| `control.base_indicated_speed_kmh` (baseIndSpeed) | 1800 km/h (all 3 missiles declare it) | `rocket.guidance.guidanceAutopilot.baseIndSpeed` |
| `aerodynamics.global_cx_vs_aoa` (CxAoA, shared) | 9.0 | `gameparams.blkx: shellBallisticsParams.props.CxAoA`; `profile_adapter.py:57-58,438-441` |
| CxAoA area-basis rule (no `wingAreaMult` on the induced-drag area) | `S_d = pi*d^2/4` | `gameparams.applyWingAreaMultToCxAoA:false`; UPDATE_V1.0.1.md item 2 |

### PHYSICS (zero-parameter given the DATAMINE inputs above)

| Parameter | Formula | Evidence pointer |
|---|---|---|
| `I = m*L^2/12` | uniform-rod transverse inertia | `h2_dynamics.py:887-889`; standard rigid-body formula, no free constant beyond the datamine `m`,`L` |
| `K = N'*Delta` | torque = normal-force-slope x arm | `h2_dynamics.py:972-977` comment "Spec Sec.5"; pure moment-arm mechanics |
| `C = 2*sqrt(K*I)` (given zeta=1, logged separately below) | textbook critical-damping coefficient | `h2_dynamics.py:982-988` |
| `omega_n = sqrt(K/I)` | natural-frequency identity | derived from K, I above |
| Quaternion attitude kinematics `Qdot = 1/2 Q (x) (0,omega)` | exact rigid-body identity | `aerodynamics.py:145` `quaternion_derivative`; no fitted term |
| `mach = V/a`, `q = 1/2 rho V^2` | definitions | `aerodynamics.py:206`, standard fluid-dynamics definitions |
| `S_d = pi*d^2/4` (induced-drag / body reference area) | circle area from caliber | `drag_models.py:16-28`, `area_basis()` |
| Thrust projection `F_T = T*f_hat` | vector projection along body axis | `h2_dynamics.py:473` |
| `gravity_mps2 = 9.80665` | standard gravity (exact SI/CODATA value) | `profile_adapter.py:104`, `UNIVERSAL_H2_LAYER["atmosphere"]` |
| finsAoa alpha-disk (radial saturation) | `limit_unit_disk` at the datamine `finsAoa` radius | `math3d.limit_unit_disk`; zero extra parameter once `finsAoa` (datamine) is fixed |
| eta_q = min(q/q_base, 4) numerator | `q/q_base` dynamic-pressure ratio | `control.py:60` `base_indicated_speed_schedule`; definition, not a fit (the cap=4 itself is logged as ARCHITECTURAL-CHOICE below) |

### STANDARD-LAW

| Parameter | Value | Evidence pointer |
|---|---|---|
| ISA atmosphere `rho(h)`, `a(h)` | International Standard Atmosphere | `src/aim120_model/atmosphere.py`; published reference model, not fitted to this project's replays |
| 1943 standard-projectile drag-law shape (10 Mach/Cx knots, M0 scale x1.0) | see "Drag Mach curve" deep-dive below | `docs/UPDATE_V1.0.1.md` Sec.4 bracket table; `drag_models.py:31-50` comment "1943-law Cx(M)" |

### REPLAY-FITTED -- purged or neutralized in M0

| Parameter | Shipped value | Fit evidence | **What M0 does with it** |
|---|---|---|---|
| `packed_lift_slope_scale` (k_s) | 0.58 | UPDATE_V1.0.1.md Sec.5: 3-flight, 59-frame joint replay fit, R^2=0.97-0.998; UPDATE_V1.0.2.md Sec.2: cross-missile AAM-4 judge frame, merged k_s=0.577+/-0.018 | **Set to 1.0** in `config/profile_m0_strict.json`. `F_N = m g0 finsLatAccel eta_q disk(alpha/alpha_max)` -- the literal reading of `finsLatAccel` as full lift authority, no fitted multiplier |
| C_et(M) global scale | x1.10 | UPDATE_V1.0.1.md Sec.4: "苏式1943型定律形状 x1.10" (1943-law shape x1.10), reconciled against the same 35 km telemetry inversion that produced the anchor knots | **Set to x1.0**: the 10 surviving shape knots in `config/profile_m0_strict.json`'s `m0_overrides.drag_model_cx_vs_mach` are each the shipped value / 1.10 |
| C_et(M) 5 telemetry-anchored knots (M 1.93, 2.01, 2.29, 2.66, 3.10) | 0.352/0.342/0.314/0.288/0.270 | UPDATE_V1.0.1.md Sec.4 bracket diagram labels these "DATA"; PL-12 35 km replay CdA inversion, +/-1.6% | **Removed outright** (not rescaled -- there is no shape value to rescale, they replaced the shape). M0's table has no knot in this Mach band; linear interpolation bridges M 1.8 -> M 3.5 directly. See "Drag Mach curve" below for the consequence |
| `midcourse_lead_turn.enabled` + `turn_time_constant_s`/`lock_delay_s`/`blend_time_s`/`fin_error_ref_rad` | true; 0.8/0.8/0.5/0.15 s,s,s,rad | UPDATE_V1.0.1.md Sec.7.1: "candidate absent from datamine"; `turn_time_constant_s` "fit within the declared 0.3-0.8s range to the PL-12 30 deg slow-launch replay, SSE 7.4"; `fin_error_ref_rad` added in v1.0.2 as a "回放标定候选" (replay-calibrated candidate) | **`enabled: false`** in `config/profile_m0_strict.json`. Zeroes both the acceleration term (`guidance.py:342`) and the IOG direct-fin-routing fraction (`guidance.py:359-374`) for every missile |
| `su_r_77`/`su_r_77_1` per-profile `guidance.midcourse_lead_turn.blend_time_s` override | 1.3 s (shared default 0.5 s) | `missiles/su_r_77.json` `parameter_sources."midcourse_lead_turn.blend_time_s".kind="calibrated"`: "由共享默认0.5s上调至1.3s：回放显示...是自动驾驶仪launch capture handover记忆时长" | **Moot in M0, no edit needed.** `profile_adapter.py:365-372` only reads a profile-level `enabled` if the profile sets one; neither profile does, so both fall through to the defaults-level `enabled=false` above, and the disabled mechanism ignores `blend_time_s` regardless of its source |
| `su_r_77`/`su_r_77_1` `aerodynamics.packed_lift_force_eta_law` | `k_force(eta)=0.574*clamp(eta,0.35,2.65)^0.242` | `missiles/su_r_77_1.json` `parameter_sources.packed_lift_force_eta_law.kind="fitted"`: single-flight, 46-frame R-77-1 level-shot replay fit, "单次飞行候选值，未跨弹验证" (single-flight candidate, not cross-missile validated); applied to `su_r_77` only by same-family analogy, no independent R-77 data | **Stripped** by `scripts/run_m0_scenarios.py` (`config["aerodynamics"].pop("packed_lift_force_eta_law", None)`, driven by `m0_overrides.strip_profile_aerodynamics_keys` in the M0 json) after `build_h2_candidate_config()` returns. Force channel then falls back to `force_k = packed_lift_slope_scale = 1.0`, identical to the moment channel |
| IOG direct-fin-routing anti-windup "tracking mode": `>=0.1` authority-fraction engage threshold, `+/-0.32 rad` integrator-tracking clamp | 0.1 (dimensionless fraction), 0.32 rad | UPDATE_V1.0.2.md Sec.1: "0.1 权限阈值...属启发式而非证明" (heuristic, not proof); `control.py:318-346` comment: clamp value tuned to the A5 replay-anchor unload arc | **Structurally inert, no code touched.** `direct_engaged` in `control.py:318` requires `abs(direct_axis_fraction) >= 0.1`, and `direct_axis_fraction` comes from `midcourse_fin_fraction`, which `guidance.py:373-374` forces to `(0.0, 0.0)` whenever `midcourse_active` is false. Disabling midcourse (row above) is what neutralizes this; M0 always takes the plain `saturated and pushing_fin` conditional-integration guard instead (a standard control-engineering anti-windup idiom with only an exact-saturation sentinel, `>=0.99` of travel -- not a value tuned to any specific replay number) |
| AIM-120A historical fitted drag scale, 0.2995 | 0.2995 | UPDATE_V1.0.1.md Sec.4 "历史注记": "v1.0.0时代的0.2995...C_et在M2-3段数值的经验近似,本机制落地后正式退役" (retired) | **N/A, not exercised.** `profile_adapter.py:67-69` comment: "frozen in configs/aim120a_h2.yaml only... Profile missiles use datamine CxK * interpolated 1943*1.10 Cx(M)". Never reachable from `cn_pl12`/`su_r_77`/`su_r_77_1`, listed here only for completeness |

Note: a ninth replay-derived finding, the +16% high-altitude thrust
correction (UPDATE_V1.0.1.md Sec.3, "未建模" -- not modeled), was
investigated but never implemented as a runtime parameter at all, so
there is nothing for M0 to purge; it remains a documented, un-coded
gap in both the shipped runtime and M0.

### ARCHITECTURAL-CHOICE -- kept, evidence-informed, not a fitted number

| Choice | What M0 keeps | Evidence (one line) |
|---|---|---|
| Packed-lift force frame: one combined (body+wing+tail) lift vector on the flow-normal pitch/yaw disk, rather than a split body-lift/tail-force model | `plant_semantics="fin_torque_body_aoa"`, `path_g_from_alpha=true` | No `CyK` field exists in the AAM datamine at all; `finsLatAccel`'s own AIM-54-diff-verified semantics ("max load factor") is a whole-airframe quantity, not a tail-only one -- the packed frame is the structural reading of that field, not a number fit to telemetry |
| Critical-damping rotation surrogate: single CM-to-neutral-point rigid rod, `zeta=1` (picked, not swept) | `I omega_dot = K(delta-alpha) - C*omega`, `zeta=1` | `zeta=1` is the parameterless "critical" value (fastest non-oscillatory response), chosen because no independent rotational-dynamics datamine field exists to identify a different damping ratio from -- not a value swept against replay data |
| v12 default autopilot path: raw PID output = fin angle directly, error in **g** (not m/s^2, not the rate-cascade candidate) | `pid_error_units="g"`, `acceleration_outer_rate_inner=false` | The `mps2` alternative was tried and *rejected on structural grounds*: it produced a bang-bang relay and a ~5 Hz G limit cycle regardless of actuator speed or step size (`config/profile_h2_runtime_defaults.json` boundary string) -- a stability pass/fail, not a value tuned to match a specific replay number |
| `t_go = R / max(Vc, kappa\|V_rel\|)`, `kappa=1` | `time_to_go_mode="closing_or_relative_speed"`, weight 1.0 | UPDATE_V1.0.1.md item 8: chosen to stop a beam-shot geometry from inflating `t_go` and self-suppressing gain (dissected from a 90-deg beam-shot pathology); `kappa=1` is the parameterless "take the max" choice, not a fitted weight |
| Drag mechanism: shared interpolated Cx(M) "shape factor" x per-missile datamine `CxK`, rather than an absolute per-missile Cx(M) table | `drag_shape_mode="interpolated_cx_1943_x1_10"` | `gameparams.shellBallisticsParams.useCxiMach:true` flag plus a `calc_cxi()` engine stub in the public DagorEngine headers -- the *mechanism* (shared curve exists) is evidence-backed; the curve's own values are handled separately above (STANDARD-LAW shape + REPLAY-FITTED scale/anchors, purged in M0) |
| Plain conditional-integration anti-windup guard (`saturated and pushing_fin`, `>=0.99` of travel) | `control.py`, always active (unconditionally, not just when IOG tracking-mode is inert) | Textbook anti-windup idiom (freeze the integrator when the actuator is pinned and the error is still pushing the same way); the `0.99` threshold is a near-100%-travel sentinel, not a number fit to any specific replay frame |
| Trajectory-normal (flow-normal) basis for **reporting** G against game telemetry, vs. the model's native body-axis `lateral_load_g` | `scripts/run_m0_scenarios.py` reads `trajectory_lateral_load_g`, matching `tests/test_replay_anchors.py::_actual_g` | UPDATE_V1.0.2.md Sec.5 item 1c/1d: the game's displayed G is thrust-inclusive trajectory-normal (verified two ways -- a 3-5g boost-phase gap that only trajectory-normal closes, and the R-77-1 burnout-cutoff A/B frame S041/S042 where the thrust-normal component crosses zero at cutoff but the fitted k only drifts 0.738->0.739). This is comparison-side bookkeeping, not a change to any force or the equations of motion -- it decides which already-computed diagnostic field is read for the residual tables |

## Drag Mach curve deep-dive (Step 1 special attention item)

The shipped `CX_1943_X1_10_TABLE` (`src/aim120_model/drag_models.py:34-50`,
introduced whole-cloth in commit `6bad659`, no earlier unscaled table
exists in git history to diff against) is 15 `(Mach, Cx)` knots.
`docs/UPDATE_V1.0.1.md` Sec.4 draws its own bracket diagram over that
exact table:

```
M    0.8   0.9   1.0   1.1   1.2   1.4   1.6   1.8  |1.93  2.01  2.29  2.66  3.10| 3.5   4.0
Cet .173  .209  .330  .402  .413  .396  .380  .358  |.352  .342  .314  .288  .270| 250   .231
                                shape ————————————— |———————— DATA ——————————————| shape
```

Ten knots (M 0.8-1.8, plus the extrapolated M 3.5/4.0) are labeled
"shape" -- the Soviet-style 1943 standard-projectile drag-law shape,
**times a fitted 1.10 global scale**. Five knots (M 1.93-3.10) are
labeled "DATA" -- PL-12 35 km telemetry-inverted CdA anchors that
*replaced* whatever the shape alone would have produced there, not a
rescaling of it.

M0 reconstructs "shape x 1.0" per the mission brief: divide each of
the ten shape knots by 1.10 (recovering the pre-scale standard-law
value), and drop the five DATA knots entirely rather than rescale
them, since there is no shape value at those five Mach numbers to
rescale -- only the replacement telemetry value. Several quotients
land on suspiciously round numbers (0.209/1.10=0.19 exactly,
0.396/1.10=0.36 exactly, 0.231/1.10=0.21 exactly), which corroborates
that the shipped table really is `round(shape_value * 1.10, 3)` for a
shape with round underlying values, not an independently-tabulated
15-point curve:

| Mach | Shipped Cx (shape x1.10) | M0 Cx (shape x1.0 = shipped/1.10) |
|---|---|---|
| 0.8 | 0.173 | 0.157273 |
| 0.9 | 0.209 | 0.19 |
| 1.0 | 0.330 | 0.3 |
| 1.1 | 0.402 | 0.365455 |
| 1.2 | 0.413 | 0.375455 |
| 1.4 | 0.396 | 0.36 |
| 1.6 | 0.380 | 0.345455 |
| 1.8 | 0.358 | 0.325455 |
| 1.93 | 0.352 (DATA) | *(removed)* |
| 2.01 | 0.342 (DATA) | *(removed)* |
| 2.29 | 0.314 (DATA) | *(removed)* |
| 2.66 | 0.288 (DATA) | *(removed)* |
| 3.10 | 0.270 (DATA) | *(removed)* |
| 3.5 | 0.250 | 0.227273 |
| 4.0 | 0.231 | 0.21 |

Consequence, flagged here so it is not mistaken for a bug when read
in the residuals doc: M0's Cx(M) curve has **no resolution between M
1.8 and M 3.5** -- exactly the band the five purged telemetry knots
used to cover -- and linear interpolation (the model's existing,
unmodified interpolation rule, `drag_models.py:80-94`) bridges that
gap as a straight line between the M 1.8 and M 3.5 shape knots.
Separately and additionally, because the x1.10 scale is removed from
*all ten* surviving knots (not just the ones inside the old gap),
M0's zero-AoA drag area `CdA0` is uniformly **~9.1% lower** than the
shipped runtime's at every Mach number, including the fully
shape-only M<1.8 band. Both effects are visible in
`docs/M0_RESIDUALS.md`'s A1 case, whose flight spends nearly all of
its cruise between Mach 1.79 and 3.18 -- almost entirely inside the
newly-unresolved gap.
