# missile_sim

**Local candidate 1.2.0rc3:** all supported missiles use the same response functions and common internal shapes, parameterized by each original resource. R-77-1-only motor/radar routing and per-object replacement constants have been removed. Original BLK/BLKX/profile coefficients, verified loader defaults, PN/PID, integration, state transitions and timing remain unchanged. Aerodynamic direction projection, lever-arm moment and angular damping now use equivalent expressions, preserving original coefficients, staged float32 operations and the braking limiter. Across four missile models and 1,056 static force/moment states, differences from the pre-rewrite rc2 reference were zero. Separate synthetic IR, ARH and SARH scenes each ran for 25 seconds and 1,200 frames with zero position or velocity differences; this covers those tested states and synthetic scenes, not replay or in-game trajectory validation.

`reconstructed/responses.py` uses dimensionless `data/response_shapes.json` parameters. Each motor stage keeps its own force/time/delay/mass and pressure knots, with the shared unit-integral shape `1+.08(1-2u)+.06(6u(1-u)-1)`. Antenna double-exponential shapes use each original half-angle and sidelobe sensitivity; their exponents/width ratios come from the previously accepted local response fit. Receiver normalization uses each original range/rcs/power, a shared 8th-power endpoint mean and `P/(P+1)`; those receiver choices are derived model choices, not native constants. Aircraft RCS uses `.04*Length*Span` and common log-aspect interpolation of original factors (84 original scale records included, or explicit `target_airframe`/RCS). Cloud expansion uses original global/resource scales with dimensionless ratios extracted from the accepted local rational fit; reflection keeps original body size, cloud cap, multiplier and envelope with a shared smooth saturation. Motor shape, RCS area/aspect and reflection parameters are declared model choices, not per-missile or trajectory/TRK fits. Original query/folding/size boundaries still belong to each missile.

After installing the wheel, run `python -m aim120_model.reconstructed --missile us_aim9m_sidewinder` (or `su_r_77_1`, `us_aim7f_sparrow`), or `python -m missile_gui.server`. The package example explicitly supplies a Su-30MKM radar signature, authored virtual IR bands and an independent PD illuminator; those bands are demonstration inputs, not a measured Su-30 heat source. Use `--input scene.json --output result.json` for another scene. The Python entry is `simulate(load_profile(id), scene)`; import `load_profile` from `aim120_model.reconstructed`.

Default IR requires explicit `target_ir` values: directional source signature `I_k/C0` before propagation, with game-internal `C0=24034.501953125` (not watts). An aircraft name alone never generates heat. Native engine B0/B1/Bab production is not closed, so the legacy F-16C/200 normalization is not established I/C0. The old FM producer is available only through explicit `infrared_source="legacy_fm_approximation"`, with unverified unit mapping labeled in results and all three `target_engine` controls required: throttle, afterburner and running. Scene body attitude takes priority over any velocity-derived direction. Guidance never receives invented IR range/velocity. SARH requires explicit sampled PD illumination and propagates hold/loss/reacquisition when support disappears. Radar targets require original `target_airframe` scales, a documented `target_aircraft`, or explicit RCS. Birth designation and virtual datalink are opt-in independent scene inputs.

The wheel includes 233 JSON-decoded original `.blk` missile resources (126 identities, including four unsupported TVC families) from 2.59.0.80. The existing 122 converted 2.59.0.27 profiles remain public catalog inputs; the opt-in legacy FM approximation separately identifies its frozen aircraft fields as 2.59.0.27. Data coverage is not execution coverage. CW/absolute Doppler, nonzero radar scans, clutter spectra/notches, TVC/command/beam riding, SARH chaff association and IR resolved targets/occlusion/sun/background are unsupported. Native carrier association is not reconstructed: the PD illuminator is an explicit virtual scene service. The common raw-scaled two-leg SARH response and receiver normalization are derived approximations: full native bistatic echo normalization remains unclosed. Both `model.SARH_signal_model` and each sample’s `bistatic.signal_model_status` expose this limitation.

Use `model_mode="legacy_surface"` for the previous shared approximation, or `"legacy_h2"` for frozen H2. The older 1.1.x descriptions below apply to those explicit modes.

[中文](README.md)

A War Thunder air-to-air missile flight simulator written in Python.

This project is a **semi-empirical, data-driven dynamics model built from public War Thunder air-to-air missile parameters and in-game experiments.**

---

## What it can do

The current model includes:

- 3D missile and target motion
- 5-DOF dynamics: 3D translation plus pitch / yaw
- Mach-dependent drag
- Angle-of-attack lift and extra drag
- Quaternion attitude integration
- Proportional navigation (PN)
- Mid- to long-range loft guidance
- A single PN law for all launch angles
- PID autopilot
- Effective control, restoring and damping response at one station
- Proximity fuse and ground-impact events
- Fixed 48 Hz control with midpoint integration; legacy RK4 remains available
- Local web GUI
- CSV / JSON telemetry export

There is no roll yet.

---

## Quick start

Python 3.10 or newer is required.

The simulation core uses only the Python standard library.

```bash
# Run the regression entry first to check the environment
python3 scripts/run_test_functions.py

# Open the GUI
./run_gui.sh
```

Windows:

```bash
run_gui.cmd
```

You can also replace `python3` with:

```bash
py -3
```

The GUI listens only on `127.0.0.1`. No network is required.

On start it selects the active parameter set from [`config/missile_profile_catalog.json`](config/missile_profile_catalog.json), currently the 2.59.0.27 official-server datamine. The 2.57.1.107 profile set and historical experiment path remain in `data/profile_sets/` and `missiles/` for replayability. Pick a missile, set the launch conditions, and run.

The GUI uses a shared single-surface framework with each missile’s mass, geometry, moment arm, engine stages, PID, PN, and Loft fields. Navigation and tracking are enabled. The active catalog contains 122 entries: 118 PN/PN+Loft entries are supported; four command/beam-guidance entries are explicitly unavailable. Shared effective coefficients use fixed nominal conditions and geometry scaling; accuracy has not been verified for every missile. Below the inverse-Mach drag law’s positive maximum, a continuous constant continuation gives finite low-speed forces. At zero speed, attitude defines direction without a fictitious speed. Loft is off by default. Comparison experiments and ideal-truth observation are absent from the GUI. The previous H2 interface remains available as `model_mode="legacy_h2"`, and frozen cases retain their original entry point.

Results include:

- A rotatable 3D trajectory
- Telemetry plots for speed, load factor, angle of attack, and more
- JSON and CSV export

The program only reads missile parameters. It does not modify the original files.

---

## A typical case

For example:

```text
Missile: AIM-120A
Altitude: 6.5 km
Shooter speed: 1200 km/h
Target speed: 1200 km/h
Initial range: 15 km
Off-boresight: 38°
Target aspect: head-on
```

The integrator runs until one of:

- hit / proximity fuse
- miss
- timeout
- ground impact

On the frozen AIM-120A regression case, this setup gives:

```text
12 m proximity fuse
t = 17.243 s
```

---

## How the model works, roughly

The missile is a reduced **5-DOF rigid body**.

Translation is the motion of the center of mass. Rotation is pitch and yaw. Attitude is a unit quaternion.

The aerodynamics are reduced-order, not full CFD or a high-dimensional aero database:

```text
per-missile datamine parameters
        +
a shared Mach drag model
        +
angle-of-attack lateral force
        +
extra drag
```

Guidance and control are separate:

```text
guidance law
  ↓
commanded lateral acceleration
  ↓
PID autopilot
  ↓
fin command
  ↓
missile dynamics
```

All launch angles use the same PN and parallel PID. Effective force and moment coefficients are scaled by missile geometry at a fixed reference condition; no angle-specific capture controller is added.

Historical H2 equations and parameter notes are retained in:

[`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)

This README only describes the structure. It does not unpack the whole evidence trail.

---

## How far can you trust it?

The goal is:

**physically consistent results that are useful for engineering comparisons.**

It is not:

**a frame-by-frame clone of the game's internal solver.**

The following anchors belong to the historical H2 model, not the current shared framework.

| Case | Model | Game / external |
| --- | ---: | ---: |
| 35 km head-on intercept | 33.26 s | 32.77 s |
| Single-frame load factor | 15.9 G | 15.0 G |
| AIM-120A, 15 km / 38° | 17.243 s | 16.7 s |

In the conditions already tested, the model's load-factor vs angle-of-attack relationship also lines up reasonably with replay data.

That means the model has captured some of the main dynamics. It is not enough to call it a validated game model.

---

## What is still inferred?

Not everything can be read straight from the datamine.

The larger uncertainties include:

- the exact drag law outside the calibrated Mach band
- whether every missile should share the same lift slope scale
- the attitude-control logic the game actually uses in the inertial phase
- exact rotational damping
- how thrust varies with altitude / ambient pressure

Those parts are labeled as such. They are not presented as known parameters.

Detailed evidence, and explanations that were ruled out, are in:

[`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)

---

## Known limits

Not modeled:

- roll dynamics
- center-of-gravity travel as fuel burns
- a full inertia tensor
- fin-actuator rate limits
- a complete nonlinear aero database

These regimes have also not been systematically validated:

- long low-speed glides
- extreme tail-chase
- a target in a sustained high-g turn
- a full long-range loft trajectory

There is also a known mismatch:

After 2.57, in-game rocket thrust appears to vary with ambient pressure.

One 6.8 km replay inverse suggests sustain thrust about 16% above the datamine nominal.

There is not enough data yet to reconstruct that law, so it is not in the model.

---

## Missile support

The active profile set (see `config/missile_profile_catalog.json`) has **122** missile profiles: **118** are connected to the current runtime and **4** are retained but unavailable because their guidance types are not yet supported. The root `missiles/` directory remains the 2.57.1.107 input for historical experiments.

Four unavailable profiles use command guidance or beam riding. All remain visible with a specific availability reason.

When the program sees one of those missiles it returns:

```text
Unsupported physics
```

It does not silently wrap them in the wrong PN model.

---

## Regression tests

AIM-120A has two frozen cases used to catch accidental code or config changes.

Both use:

```text
dt = 0.02 s
```

| Setup | Result | Time |
| --- | --- | ---: |
| 12 km, 10° off-boresight, head-on | 12 m fuse | 10.553 s |
| 15 km, 38° off-boresight, head-on | 12 m fuse | 17.243 s |

These historical cases use the frozen v1.0.0 config and 2.57.1.107 parameters:

```text
configs/aim120a_v1.json
configs/aim120a_v1_cases.json
```

The v1.0.1 dynamics use a separate runtime path.

These historical baselines are kept separate from the active profile set; updating the parameter source does not mean the trajectories have been revalidated.

---

## Repository layout

```text
src/aim120_model/   simulation core
src/missile_gui/    local web GUI
missiles/           frozen 2.57.1.107 historical experiment profiles
data/profile_sets/  versioned profile sets selected by the catalog
scenarios/          launch and target scenarios
config/             runtime defaults
configs/            frozen configs
schemas/            missile JSON schema
scripts/            run, import, and diagnostic tools
tests/              regression tests
```

More detailed technical notes:

- [`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md): full equations, parameter notes, and evidence
- [`V1_FREEZE.md`](V1_FREEZE.md): v1.0 freeze boundary

Raw evidence lives in:

```text
data/raw/
```

Generated results live in:

```text
outputs/
dist/
```

Those are not committed to git.

---

## License

No formal license has been chosen yet.

Until a real `LICENSE` file is in the repository, do not treat this as an open-source project that is cleared for public redistribution.
