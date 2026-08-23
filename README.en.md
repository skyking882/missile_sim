# missile_sim

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
- Initial turn for large off-boresight launches
- PID autopilot
- Fin-actuator dynamics
- Proximity fuse and ground-impact events
- Fixed-step RK4 integration
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

On start it scans missile parameter files in `missiles/`. Pick a missile, set the launch conditions, and run.

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

On a large off-boresight launch the model also does a short initial slew, pointing the nose toward the predicted intercept region before normal proportional navigation.

The full equations, parameter meanings, and the evidence behind each assumption are in:

[`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)

This README only describes the structure. It does not unpack the whole evidence trail.

---

## How far can you trust it?

The goal is:

**physically consistent results that are useful for engineering comparisons.**

It is not:

**a frame-by-frame clone of the game's internal solver.**

There are a few sanity-check anchors.

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

`missiles/` currently has 120 missile parameter files.

Of those:

- **116** can run on the current model
- **4** are unsupported

The remaining four use command guidance or beam riding, which is not the same physics as the current air-to-air PN plant.

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

These cases use the frozen v1.0.0 config:

```text
configs/aim120a_h2.yaml
```

The v1.0.1 dynamics use a separate runtime path.

They are kept apart on purpose, so work on the new plant cannot silently move the frozen regression baseline.

---

## Repository layout

```text
src/aim120_model/   simulation core
src/missile_gui/    local web GUI
missiles/           missile parameters
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
