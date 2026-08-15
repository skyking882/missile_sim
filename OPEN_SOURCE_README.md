# AIM-120A Local Model v1.0.0

An independent, auditable Python forward model for AIM-120A-like missile trajectory experiments. It uses only the Python standard library at runtime and does not modify or depend on game files.

## Quick start

Requirements: Python 3.10 or newer.

```text
python3 examples/run_v1.py
python3 scripts/run_test_functions.py
```

On Windows, `py -3` can be used instead of `python3`.

## Local GUI v1

After cloning, start the loopback-only GUI on Windows with:

```text
.\run_gui.cmd
```

On Linux or macOS:

```text
./run_gui.sh
```

The launcher creates or reuses `.venv`, checks the optional GUI dependencies, starts only on `127.0.0.1:8765`, and opens the browser. The GUI uses only Python and browser standard-library features: it does not download frontend assets, call external APIs, edit missile files, or write simulation results into experiment directories.

The first GUI release scans the read-only `missiles/*.json` library, preserves the scenario when switching missiles, runs one missile at a time through the single `simulate(missile_profile, scenario)` entry point, and provides a rotatable, zoomable, pannable 3D trajectory scene plus six interactive 2D trajectory/telemetry charts. The 3D scene uses stable colors and shapes for missile, target, engine-stage boundary, burnout, and termination markers, with hover telemetry. Scenario/result JSON and trajectory CSV export remain available. Of the 120 unit-explicit experimental profiles, 116 are runnable through `profile_h2_universal_v2`: a shared Python H2 model layer supplies effective drag, lift/Mach shape, loft/control, and numerical semantics, then the selected missile JSON supplies missile-specific data. The frozen AIM-120A configuration remains an independent regression artifact, and the universal mapping must preserve its trajectory anchors within 1 ms. Runtime-model assumptions are recorded in exported results. Four command/beam-guidance profiles remain `Unsupported physics` and are never silently mapped to PN.

To validate the imported profile set from the project root:

```text
py -3 -m missile_lab validate-profiles
```

## What is frozen

- Two-stage propulsion and continuous mass loss.
- Standard atmosphere, effective drag area, angle-of-attack drag and flow-normal lift.
- Three-dimensional target motion.
- Gain-scheduled proportional navigation, a minimal loft branch, acceleration PID and actuator dynamics.
- Proximity-fuse, ground, lifetime, range and numerical-failure events.
- A separate identified effective-controller mapping from signed yaw command to reported `currentG` magnitude. Its sign is restored only at the local plant boundary; this mapping is not wired into the H2 trajectory loop in v1.0.0.

For website-style target inputs, set `target_course_reference` to `statshark_relative_to_los`. In that convention, course `0` is head-on toward the launcher and course `180` is tail-away.

## Frozen checks

With the included 0.02 s fixed-step configuration:

| Case | Local terminal event | Local time |
|---|---:|---:|
| 1200/1200 km/h, 6.5 km altitude, 12 km, 10 deg off-axis, head-on | fuse at 12 m | 10.553 s |
| 1200/1200 km/h, 6.5 km altitude, 15 km, 38 deg off-axis, head-on | fuse at 12 m | 17.243 s |

The second case was manually compared with a 16.7 s external result under the same interpreted inputs, a difference of about 0.54 s (3.3%). One close case is a useful smoke check, not broad validation.

## Model boundary

This is a local engineering candidate, not a reproduction of any proprietary solver and not an official implementation from Gaijin Entertainment or StatShark. It is most useful for short-to-medium-range, straight-target, head-on through moderate off-axis exploration.

Not established in v1.0.0: long low-Mach glide accuracy, extreme tail chase, sustained maneuvering targets, full loft behavior, six-degree-of-freedom roll dynamics, or broad end-to-end equivalence with an external solver.

See `MODEL_CARD.md` for the exact freeze boundary and `RELEASE_MANIFEST.json` for file hashes.

## License status

No open-source license has been selected yet. Add a `LICENSE` file before publishing this repository as open source; see `LICENSE_PENDING.md`.
