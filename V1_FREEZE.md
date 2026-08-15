# v1.0.0 freeze record

- Frozen: 2026-08-14
- Runtime: Python 3.10+, standard library only
- Numerical step: 0.02 s fixed-step RK4 with explicit propulsion boundaries
- Public source: the allowlisted modules under `src/aim120_model/`
- Public inputs: `configs/aim120a_v1.json` and `configs/aim120a_v1_cases.json`
- Regression anchors: 10 deg / 12 km at 10.553325 s; 38 deg / 15 km at 17.242967 s

The trajectory path is the H2 forward model: target kinematics, proportional-navigation/loft command, acceleration PID and actuator, body/aerodynamic/propulsion dynamics, then terminal events. `EffectiveControllerEnvelope` is retained as a separate identified command-to-effective-output layer; v1.0.0 does not silently substitute it into the H2 trajectory control loop.

The release is constructed from an exact allowlist. It excludes experimental notebooks and fit scripts, generated outputs, raw external-service evidence, browser state, credentials, local runtime directories, and all game installation files. No network calculation or external UI action is part of the freeze process.

The values are an independent local engineering candidate. Publicly observable names and parameters do not establish equivalence to a proprietary implementation. The included regression cases detect code/config drift; they do not prove the untested envelope.
