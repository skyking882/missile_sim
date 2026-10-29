# missile_sim

**1.2.0rc3** provides approximate simulation of aerodynamics, rigid-body dynamics, engines, radar, target acquisition and guidance, using game parameters and response observations together with local response fits and general approximation functions.

Supported missiles share response functions while retaining their own mass, geometry, motor, seeker and control parameters. Original BLK, BLKX and profile coefficients are preserved. The public API and local GUI support infrared (IR), active radar (ARH) and semi-active radar (SARH) scenes with explicit PD illumination, including acquisition, tracking, loss and reacquisition, plus INS and PN/PID guidance control. Local fits and general approximations both contribute to the model; accuracy has not been calibrated for every missile across the full game environment.

Requires Python 3.10 or later. Download the release wheel, install it, then run from any directory:

```console
python -m pip install aim120a_local_model-1.2.0rc3-py3-none-any.whl
python -m aim120_model.reconstructed --missile us_aim9m_sidewinder
python -m aim120_model.reconstructed --missile su_r_77_1
python -m aim120_model.reconstructed --missile us_aim7f_sparrow
python -m missile_gui.server
```

The default demonstration scene explicitly supplies a Su-30MKM radar signature, authored virtual IR bands and an independent PD illuminator. The IR bands are demonstration inputs, not a measured Su-30 heat source. Use `--input scene.json --output result.json` for a custom scene. The Python entry is `simulate(load_profile(missile_id), scene)`; import `load_profile` from `aim120_model.reconstructed`.

Default IR requires explicit `target_ir` directional signatures `I_k/C0` before propagation, where `C0=24034.501953125` is a game-internal scale, not watts. An aircraft name alone does not supply IR intensity. Optional `infrared_source="legacy_fm_approximation"` uses the older FM heat-source approximation with an uncalibrated intensity-unit mapping; all three `target_engine` controls are required: `throttle`, `afterburner` and `running`. IR observations provide angles and angular rates only; PN uses the missile's own velocity.

SARH requires explicit PD illumination through `illuminator`; disabled, stale or mismatched illumination affects tracking. Echo intensity uses a two-leg signal approximation, identified by `SARH_signal_model` and `bistatic.signal_model_status` in the results. Radar targets can use `target_aircraft`, `target_airframe={length,span,aspects}` or explicit `target_rcs_m2`. `birth_designation` and `virtual_datalink` are explicit scene inputs. Scenes may specify body attitude with a quaternion.

The package retains 233 JSON-decoded original `.blk` resource variants representing 126 identities from 2.59.0.80, plus 122 converted 2.59.0.27 profiles. The optional legacy FM heat-source approximation uses frozen 2.59.0.27 aircraft/FM fields. Supported coverage includes zero-scan-sector relative-Doppler active radar, explicit PD semi-active illumination, point-source IR, gating and IRCCM. TVC, command/beam riding, CW/absolute Doppler, radar scans/clutter spectra, SARH chaff association and IR resolved targets/occlusion/sun/background remain unsupported. The PD illuminator is an independent scene input.

Select `model_mode="legacy_surface"` or `model_mode="legacy_h2"` for the older models. The 1.1.x descriptions below apply to those explicit modes.

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

The historical v1.0.0 AIM-120A regression uses frozen 2.57.1.107 parameters. The result below has not been revalidated with the current 2.59.0.27 parameter set:

```text
12 m proximity fuse
t = 17.243 s
```

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
- [`V1_FREEZE.md`](V1_FREEZE.md): historical v1.0.0 freeze record (not a model card for the current 2.59 version)

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
