"""Guidance-state scheduling invariants for the H2 run loop.

Until the 2026-08-29 time-semantics review, telemetry ``_sample()`` advanced
the stateful PCC guidance machinery (mode state machine, washout and LOS-rate
filters) itself, so the control trajectory implicitly required "exactly one
telemetry row per control step": decimating or densifying sampling would have
changed the flight path.  The run loop now evaluates ``guidance_command``
exactly once per timestamp and every telemetry row reuses that command.

These tests pin that boundary:

- exactly one guidance evaluation per timestamp, strictly increasing in time;
- one telemetry row per evaluation on an event-free run;
- ``_sample`` has no fallback path that could evaluate guidance on its own.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import aim120_model.h2_simulator as h2_simulator_module
from aim120_model.h2_simulator import H2Simulator
from aim120_model.public_api import simulate
from missile_gui.library import scan_library

ROOT = Path(__file__).resolve().parents[1]

# su_r_77 runs the PCC-alpha midcourse (stateful mode machine + filters), so a
# double evaluation at any timestamp would advance real state, not a pure
# function.  3 s of an 80-degree off-axis launch ends on the scenario time
# limit long before any fuse/ground event, keeping the sample grid regular.
_SCENARIO = dict(
    launch_speed_kmh=1200.0,
    launch_altitude_m=6500.0,
    launch_pitch_deg=0.0,
    launch_heading_deg=0.0,
    target_speed_kmh=1200.0,
    target_altitude_m=6500.0,
    initial_distance_m=8000.0,
    target_azimuth_deg=80.0,
    target_heading_deg=0.0,
    target_vertical_heading_deg=0.0,
    target_constant_turn_g=0.0,
    max_simulation_time_s=3.0,
    loft_enabled=False,
)


def _r77_profile() -> dict:
    profiles, errors = scan_library(ROOT / "missiles", ROOT)
    assert errors == []
    return next(p for p in profiles if p["missile_id"] == "su_r_77")


def test_one_guidance_evaluation_per_timestamp(monkeypatch) -> None:
    times: list[float] = []
    real_guidance_command = h2_simulator_module.guidance_command

    def counting_guidance_command(state, track, time_s, config, **kwargs):
        times.append(float(time_s))
        return real_guidance_command(state, track, time_s, config, **kwargs)

    monkeypatch.setattr(h2_simulator_module, "guidance_command", counting_guidance_command)
    result = simulate(_r77_profile(), _SCENARIO)

    assert result["summary"]["termination_event"] == "lifetime"
    assert len(times) > 100
    assert times == sorted(times)
    assert len(set(times)) == len(times), (
        "guidance_command ran twice at one timestamp: stateful PCC guidance "
        "would advance twice per control step"
    )
    # Every evaluation feeds exactly one grid telemetry row, in order.  The
    # final row may be an event-interpolated sample (the scenario time limit
    # surfaces as a "lifetime" event candidate), which by design reuses the
    # last evaluation instead of advancing guidance state at an interpolated
    # timestamp.
    sample_times = [float(sample["time_s"]) for sample in result["samples"]]
    assert sample_times[: len(times)] == times
    assert len(sample_times) - len(times) <= 1


def test_sample_cannot_evaluate_guidance_on_its_own() -> None:
    parameters = inspect.signature(H2Simulator._sample).parameters
    assert "guidance_state" not in parameters, (
        "_sample must not receive mutable guidance state; telemetry is a pure reader"
    )
    assert parameters["reuse_guidance"].default is inspect.Parameter.empty, (
        "reuse_guidance must be required so no sampling path can silently "
        "re-evaluate guidance"
    )
