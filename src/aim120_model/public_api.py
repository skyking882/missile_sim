"""Stable, side-effect-free entry point for local GUI and API consumers.

The browser never contains trajectory equations.  It submits a scenario here and
this module adapts it to the frozen H2 simulator contract.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

from .h2_simulator import H2Simulator
from .math3d import norm


SUPPORTED_ENGINE_TYPES = {"piecewise_constant_thrust"}
SUPPORTED_CONTROL_TYPES = {"pn_pid_first_order"}
SUPPORTED_TARGET_COURSE_REFERENCES = {
    "relative_to_los",
    "launch_axis",
    "absolute_world",
}


class SimulationInputError(ValueError):
    """A user-correctable scenario or profile error."""


class UnsupportedPhysicsError(ValueError):
    """The selected missile asks for a model this runtime does not implement."""


_FIELDS: dict[str, tuple[float, float, str]] = {
    "launch_speed_kmh": (0.0, 5000.0, "发射速度"),
    "launch_altitude_m": (0.0, 30000.0, "发射高度"),
    "launch_pitch_deg": (-90.0, 90.0, "发射俯仰角"),
    "launch_heading_deg": (-180.0, 180.0, "发射航向角"),
    "target_speed_kmh": (0.0, 5000.0, "目标速度"),
    "target_altitude_m": (0.0, 30000.0, "目标高度"),
    "initial_distance_m": (1.0, 200000.0, "初始距离"),
    "target_azimuth_deg": (-180.0, 180.0, "目标方位角"),
    "target_heading_deg": (-360.0, 360.0, "目标航向角"),
    "target_vertical_heading_deg": (-90.0, 90.0, "目标垂直航向"),
    "target_constant_turn_g": (-15.0, 15.0, "目标恒定转弯G"),
}


def validate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the public scenario schema."""

    if not isinstance(scenario, dict):
        raise SimulationInputError("场景必须是 JSON 对象。")
    normalized: dict[str, Any] = {}
    for key, (minimum, maximum, label) in _FIELDS.items():
        if key not in scenario or scenario[key] in (None, ""):
            raise SimulationInputError(f"缺少必填项：{label}。")
        value = scenario[key]
        if isinstance(value, bool):
            raise SimulationInputError(f"{label}必须是数字。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError(f"{label}必须是数字。") from exc
        if not math.isfinite(number):
            raise SimulationInputError(f"{label}必须是有限数字。")
        if number < minimum or number > maximum:
            raise SimulationInputError(
                f"{label}超出范围：应为 {minimum:g} 到 {maximum:g}。"
            )
        normalized[key] = number

    optional = scenario.get("max_simulation_time_s")
    if optional in (None, ""):
        normalized["max_simulation_time_s"] = None
    else:
        try:
            maximum_time = float(optional)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("最大模拟时间必须是数字。") from exc
        if not math.isfinite(maximum_time) or not 0.1 <= maximum_time <= 300.0:
            raise SimulationInputError("最大模拟时间应为 0.1 到 300 秒。")
        normalized["max_simulation_time_s"] = maximum_time

    observation_mode = scenario.get("observation_mode", "ideal_truth")
    if observation_mode not in {"ideal_truth", "sensor_track"}:
        raise SimulationInputError("制导观测必须是 ideal_truth 或 sensor_track。")
    normalized["observation_mode"] = str(observation_mode)

    # Loft remains a profile capability, but activation is an explicit public
    # scenario choice.  Keep the switch opt-in for backwards-compatible
    # callers and make its type strict so a browser/API client cannot
    # accidentally enable it with a truthy string or number.
    loft_enabled = scenario.get("loft_enabled", False)
    if not isinstance(loft_enabled, bool):
        raise SimulationInputError("loft_enabled 必须是布尔值。")
    normalized["loft_enabled"] = loft_enabled

    course_reference = scenario.get(
        "target_course_reference",
        "relative_to_los",
    )
    if course_reference not in SUPPORTED_TARGET_COURSE_REFERENCES:
        raise SimulationInputError(
            "目标航向参考系必须是 relative_to_los、launch_axis 或 absolute_world。"
        )
    normalized["target_course_reference"] = str(course_reference)

    simulation_dt = scenario.get("simulation_dt_s")
    if simulation_dt in (None, ""):
        normalized["simulation_dt_s"] = None
    else:
        try:
            simulation_dt = float(simulation_dt)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("仿真时间步必须是数字。") from exc
        if not math.isfinite(simulation_dt) or not 1.0e-4 <= simulation_dt <= 1.0:
            raise SimulationInputError("仿真时间步应为 0.0001 到 1 秒。")
        normalized["simulation_dt_s"] = simulation_dt

    datalink_enabled = scenario.get("datalink_enabled", True)
    if not isinstance(datalink_enabled, bool):
        raise SimulationInputError("datalink_enabled 必须是布尔值。")
    normalized["datalink_enabled"] = datalink_enabled

    datalink_disconnect = scenario.get("datalink_disconnect_time_s")
    if datalink_disconnect in (None, ""):
        normalized["datalink_disconnect_time_s"] = None
    else:
        try:
            datalink_disconnect = float(datalink_disconnect)
        except (TypeError, ValueError) as exc:
            raise SimulationInputError("Datalink 断开时间必须是数字。") from exc
        if not math.isfinite(datalink_disconnect) or datalink_disconnect < 0.0 or datalink_disconnect > 300.0:
            raise SimulationInputError("Datalink 断开时间应为 0 到 300 秒。")
        normalized["datalink_disconnect_time_s"] = datalink_disconnect

    drift_direction = scenario.get("inertial_drift_direction", [0.0, 0.0, 1.0])
    if drift_direction is None:
        drift_direction = [0.0, 0.0, 1.0]
    if isinstance(drift_direction, (str, bytes)):
        raise SimulationInputError("惯性漂移方向必须是三个数字。")
    try:
        drift_values = [float(value) for value in drift_direction]
    except (TypeError, ValueError) as exc:
        raise SimulationInputError("惯性漂移方向必须是三个数字。") from exc
    if len(drift_values) != 3 or not all(math.isfinite(value) for value in drift_values):
        raise SimulationInputError("惯性漂移方向必须是三个有限数字。")
    drift_norm = math.sqrt(sum(value * value for value in drift_values))
    if drift_norm <= 1.0e-12:
        raise SimulationInputError("惯性漂移方向不能是零向量。")
    normalized["inertial_drift_direction"] = [value / drift_norm for value in drift_values]
    return normalized


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise SimulationInputError("导弹配置不是有效对象。")
    if "model_family" in profile:
        unsupported = profile.get("_runtime_unsupported", [])
        if unsupported:
            raise UnsupportedPhysicsError("该导弹请求了不支持的模型类型：" + "、".join(unsupported))
        model_config = profile.get("_model_config")
        if not isinstance(model_config, dict):
            raise UnsupportedPhysicsError("该导弹没有可用的冻结或 profile candidate runtime 参数。")
        return model_config
    physics = profile.get("physics")
    if not isinstance(physics, dict):
        raise UnsupportedPhysicsError("该导弹没有可用的物理模型。")
    engine_type = physics.get("engine_type")
    control_type = physics.get("control_type")
    if engine_type not in SUPPORTED_ENGINE_TYPES:
        raise UnsupportedPhysicsError(f"不支持的发动机类型：{engine_type or '未声明'}。")
    if control_type not in SUPPORTED_CONTROL_TYPES:
        raise UnsupportedPhysicsError(f"不支持的控制类型：{control_type or '未声明'}。")
    model_config = profile.get("_model_config")
    if not isinstance(model_config, dict):
        raise UnsupportedPhysicsError("该导弹没有已加载的冻结模型参数。")
    return model_config


def _case_from_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "gui_scenario",
        "model_variant": "full",
        "observation_mode": scenario["observation_mode"],
        "loft_enabled": scenario["loft_enabled"],
        "simulation_dt_s": scenario["simulation_dt_s"],
        "datalink_enabled": scenario["datalink_enabled"],
        "datalink_disconnect_time_s": scenario["datalink_disconnect_time_s"],
        "inertial_drift_direction": scenario["inertial_drift_direction"],
        "initial_conditions": {
            "start_speed_kmh": scenario["launch_speed_kmh"],
            "launch_altitude_m": scenario["launch_altitude_m"],
            "launch_angle_deg": scenario["launch_pitch_deg"],
            "launch_yaw_deg": scenario["launch_heading_deg"],
            "target_speed_kmh": scenario["target_speed_kmh"],
            "target_altitude_m": scenario["target_altitude_m"],
            "initial_target_distance_m": scenario["initial_distance_m"],
            "target_azimuth_deg": scenario["target_azimuth_deg"],
            "target_course_deg": scenario["target_heading_deg"],
            "target_course_reference": scenario["target_course_reference"],
            "target_constant_g_turn": scenario["target_constant_turn_g"],
            "target_vertical_course_deg": scenario["target_vertical_heading_deg"],
        },
    }


def _command_g(sample: dict[str, Any]) -> float:
    values = sample.get("commanded_acceleration_g", [0.0, 0.0])
    return math.hypot(float(values[0]), float(values[1]))


def _event_name(event_type: str) -> str:
    return {
        "impact": "hit",
        "fuse": "proximity_fuse",
        "ground": "ground",
        "lifetime": "lifetime",
        "max_distance": "max_range",
        "numerical_failure": "numerical_failure",
    }.get(event_type, event_type)


def _summarize(
    result: dict[str, Any],
    config: dict[str, Any],
    limited_by_scenario: bool,
    scenario_loft_enabled: bool,
    profile_lofting_enabled: bool,
) -> dict[str, Any]:
    samples = result["samples"]
    terminal = samples[-1]
    speeds = [norm(tuple(sample["velocity_mps"])) for sample in samples]
    stages = config["propulsion"]["stages"]
    burnout_time = sum(float(stage["duration_s"]) for stage in stages)
    event = _event_name(str(result["event_type"]))
    track_errors = [float(sample.get("track_position_error_m", 0.0)) for sample in samples]
    radar_track_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("track_mode") == "radar_track"
    ]
    radar_lost_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("seeker_state") == "lost"
    ]
    lock_loss_times = [
        float(sample["time_s"])
        for sample in samples
        if sample.get("track_mode") == "ins_search"
    ]
    reacquire_times: list[float] = []
    had_lock_loss = False
    for sample in samples:
        if sample.get("track_mode") == "ins_search":
            had_lock_loss = True
        elif had_lock_loss and sample.get("track_mode") == "radar_track":
            reacquire_times.append(float(sample["time_s"]))
    reject_reasons = [
        str(sample.get("radar_reject_reason"))
        for sample in samples
        if sample.get("radar_reject_reason")
    ]
    return {
        "termination_event": event,
        "termination_detail": (
            "reached_scenario_time_limit"
            if event == "lifetime" and limited_by_scenario
            else event
        ),
        "flight_time_s": float(terminal["time_s"]),
        "terminal_distance_m": float(terminal["distance_to_target_m"]),
        "terminal_speed_kmh": speeds[-1] * 3.6,
        "terminal_altitude_m": float(terminal["position_m"][1]),
        "maximum_speed_kmh": max(speeds) * 3.6,
        "maximum_altitude_m": max(float(sample["position_m"][1]) for sample in samples),
        "maximum_commanded_g": max(_command_g(sample) for sample in samples),
        "maximum_actual_g": max(float(sample.get("actual_overload_g", 0.0)) for sample in samples),
        "maximum_trajectory_normal_g": max(float(sample.get("trajectory_lateral_load_g", 0.0)) for sample in samples),
        "minimum_distance_m": min(float(sample["distance_to_target_m"]) for sample in samples),
        "loft_enabled": bool(config["guidance"].get("lofting_enabled", False)),
        "scenario_loft_enabled": bool(scenario_loft_enabled),
        "profile_lofting_enabled": bool(profile_lofting_enabled),
        "burnout_time_s": burnout_time,
        "observation_mode": result.get("observation_mode", "ideal_truth"),
        "observation_provider": result.get("observation_provider", "ideal_truth"),
        "track_mode": terminal.get("track_mode"),
        "seeker_state": terminal.get("seeker_state"),
        "seeker_display_state": terminal.get("seeker_display_state"),
        "maximum_track_error_m": max(track_errors) if track_errors else 0.0,
        "first_radar_track_time_s": min(radar_track_times) if radar_track_times else None,
        "first_radar_lost_time_s": min(radar_lost_times) if radar_lost_times else None,
        "first_lock_loss_time_s": min(lock_loss_times) if lock_loss_times else None,
        "first_reacquire_time_s": min(reacquire_times) if reacquire_times else None,
        "last_radar_reject_reason": reject_reasons[-1] if reject_reasons else None,
        "last_observation_reject_reason": next(
            (
                str(sample.get("observation_reject_reason"))
                for sample in reversed(samples)
                if sample.get("observation_reject_reason")
            ),
            None,
        ),
    }


def _stage_markers(config: dict[str, Any]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    elapsed = 0.0
    stages = config["propulsion"]["stages"]
    for index, stage in enumerate(stages):
        elapsed += float(stage["duration_s"])
        label = "发动机燃尽" if index == len(stages) - 1 else f"{stage.get('name', index + 1)} 结束"
        markers.append({"time_s": elapsed, "label": label, "kind": "burnout" if index == len(stages) - 1 else "stage"})
    return markers


def _validate_surface_options(early_miss_s,clutter_model,clutter_min_depression_deg,multipath_gain,
                              launcher_radar_gimbal_deg,launcher_support,launcher_radar_geometry=None):
    if early_miss_s is not None and (isinstance(early_miss_s, bool) or not isinstance(early_miss_s, (int, float))
                                     or not math.isfinite(early_miss_s) or early_miss_s <= 0):
        raise SimulationInputError("提前脱靶判定时间必须是正数。")
    if clutter_model not in (None, "look_down", "geometric_mainlobe", "look_down_angle"):
        raise SimulationInputError("杂波模型必须是 look_down、geometric_mainlobe 或 look_down_angle。")
    if clutter_min_depression_deg is not None and (
            clutter_model != "look_down_angle" or isinstance(clutter_min_depression_deg, bool)
            or not isinstance(clutter_min_depression_deg, (int, float))
            or not 0 <= clutter_min_depression_deg < 90):
        raise SimulationInputError("下视门槛角只用于 look_down_angle，且应在 0–90° 之间。")
    if multipath_gain is not None and (isinstance(multipath_gain, bool) or not isinstance(multipath_gain, (int, float))
                                       or not 0 <= multipath_gain <= 1):
        raise SimulationInputError("多径增益应在 0–1 之间。")
    if launcher_radar_gimbal_deg is not None and (
            isinstance(launcher_radar_gimbal_deg, bool) or not isinstance(launcher_radar_gimbal_deg, (int, float))
            or not 0 < launcher_radar_gimbal_deg <= 180):
        raise SimulationInputError("载机雷达扫描半角应在 0–180° 之间。")
    from .radar_geometry import validate_launcher_radar_geometry
    try:
        validate_launcher_radar_geometry(launcher_radar_geometry)
    except ValueError as exc:
        raise SimulationInputError(str(exc)) from exc
    if launcher_radar_geometry is not None and (launcher_radar_gimbal_deg is not None or launcher_support is not None):
        raise SimulationInputError('launcher_radar_geometry cannot combine with launcher_radar_gimbal_deg or launcher_support')
    if launcher_support is not None:
        if not callable(launcher_support):
            raise SimulationInputError('launcher_support 必须是可调用对象。')
        if launcher_radar_gimbal_deg is not None:
            raise SimulationInputError('launcher_support 与 launcher_radar_gimbal_deg 不能同时使用。')


def simulate(
    missile_profile: dict[str, Any],
    scenario: dict[str, Any],
    *,
    model_mode: str = "general",
    target_factory: Callable[[Any], Any] | None = None,
    early_miss_s: float | None = None,
    clutter_model: str | None = None,
    clutter_min_depression_deg: float | None = None,
    cw_on_clear_beam: bool = False,
    multipath_gain: float | None = None,
    launcher_radar_gimbal_deg: float | None = None,
    launcher_radar_geometry: dict[str, Any] | None = None,
    require_seeker_lock: bool = False,
    launcher_support: Callable[[float, Any], str | tuple[str, tuple, tuple]] | None = None,
    seeker_search: dict[str, Any] | None = None,
    min_altitude_m: float | None = None,
    chaff_capture: dict[str, Any] | None = None,
    track_gate_model: dict[str, Any] | None = None,
    target_observer: str | None = None,
    angular_observer: str | None = None,
    launch_quaternion_xyzw: tuple[float, float, float, float] | None = None,
    guidance_input: Callable | None = None,
    infrared_seeker: Any | None = None,
) -> dict[str, Any]:
    """Run one missile and return a JSON-serializable, read-only result.

    ``infrared_seeker`` accepts InfraredSeekerConfig for optical sensor_track
    profiles. This opt-in includes game-style FOV gating, tracking suspension and
    deterministic flares via target.flares_at. Game-data assembly reads aircraft
    heat, global conversion/aspect constants and band ranges automatically;
    downstream range/velocity are truth-assisted simulation closure. None retains
    the existing observation path and output schema.

    This function performs no filesystem writes and is the only GUI calculation
    entry point. Opt-in, general runtime only: ``target_factory`` wraps the
    scenario target (see ``evasive_target.evasive_target_factory``);
    ``early_miss_s`` ends an unpowered missile that has been opening from a
    faster target for that long with ``termination_event == "early_miss"``;
    ``clutter_model="geometric_mainlobe"`` replaces the any-look-down clutter
    notch condition with a flat-earth main-beam ground intersection;
    ``"look_down_angle"`` requires the line of sight to point at least
    ``clutter_min_depression_deg`` (default 2) below the horizon;
    ``cw_on_clear_beam`` lets any chaff bundle capture a beaming target that
    has no ground clutter behind it (decoy path only);
    ``multipath_gain`` (0–1) moves the radar seeker's measured position toward
    the target's flat-surface image by gain times the profile's
    ``multipath_effect`` strength at the target height (see
    ``RadarSeekerObserver.enable_multipath``).
    ``launcher_radar_gimbal_deg`` drops the datalink for good once the launcher's
    own radar would lose the target (outside its body yaw/pitch limits, or in the clutter notch
    seen from the launcher; see ``SensorTrackProvider.enable_launcher_radar``);
    ``launcher_radar_geometry`` supplies explicit yaw/pitch degree-limit pairs,
    beam_half_angle_deg and a fixed body-to-world orientation_quaternion_xyzw.
    It uses the launcher's own circular beam after mechanical clipping; the
    scalar gimbal option has no aircraft beam/roll data and adds no beam width.
    ``require_seeker_lock`` lets the proximity fuse fire only after the seeker
    has tracked the target at least once. Callback support also accepts
    (reason, position, velocity) with native y-up estimates; empty reason uses
    that estimate in datalink, seeker search and INS. ``launcher_support(t, target_truth)``
    optionally supplies external launcher support at seconds since launch;
    return an empty string for support or a nonempty permanent loss reason.
    It requires the radar sensor_track provider and is mutually exclusive with
    ``launcher_radar_gimbal_deg``.
    ``seeker_search`` opts active-radar sensor_track into deterministic scanning,
    per-profile tx/rx antenna gain and RCS detection range, and acquisition dwell.
    An empty object uses D-grade scan defaults and zero additional dwell.
    This missile's lock_timeout_s independently sets the launch startup delay.
    Target models may expose rcs_m2 (default 1) with or without decoys_at.
    None leaves existing results and output keys unchanged. Other observers keep
    their existing behavior and report the scan as not_applicable when requested.
    ``track_gate_model`` enables predicted range/speed/angle return competition,
    notch tracking hazard, narrow initial/reacquisition windows, and datalink
    pointing after own lock; None preserves current behavior. In the object,
    datalink_after_lock defaults to True and acquire_half_deg to 5; False/None
    respectively restore the original tracking-gate acquisition/support rules.
    It requires active radar sensor_track and cannot combine with chaff_capture.
    ``chaff_capture`` enables one seeded vg_soft trial per timestamped chaff release.
    None retains the legacy decoy model; only active radar sensor_track supports it.
    See the evidence ledger for calibrated versus assumed parameters.
    ``min_altitude_m=50.0`` floors propagated INS estimates at 50 m in native
    y-up coordinates and zeros estimated vertical velocity on clipping, including
    drift. Pass it as a top-level keyword (or in WT_overlay's LAUNCH_OPTIONS).
    None disables the floor. Requires the general surface sensor_track runtime;
    direct radar/datalink observations keep their measured state unchanged.
    ``target_observer='unified_alpha_beta'`` opts the radar sensor_track provider
    into one world position-only observer (.8/.05, explicit experiment). Echo
    velocities never enter its state. Timestamped DL callbacks may return
    (reason, tracking.DatalinkObservation); old estimate tuples supply only an
    initial prior in this mode. No estimate means no truth fallback. Same-object
    reacquisition retains history; a new reflector starts with zero velocity.
    ``angular_observer='q_capture'`` enables the empirical full-rate observer
    for active radar sensor_track: previous body posture Q at acquisition,
    received LOS prediction/correction with fixed gain 11.23 /s, and prediction
    through missed echoes. It requires no replay rate/direction. None retains
    the existing observer. It cannot be combined with target_observer or
    guidance_input. This is a game approximation, not recovered WT code.
    ``launch_quaternion_xyzw`` overrides only the initial body-to-world
    attitude (forward/up/right body axes, scalar-last xyzw, normalized).
    The scenario velocity direction remains defined by launch pitch/heading.
    ``guidance_input(t, own_position, own_velocity) -> TrackSolution`` replaces
    the built-in observer. AngularObservation/RecordedSeekerHistory in
    angular_guidance support complete rate, causal acquisition boundaries and
    changing missile-relative motion. Full rate uses runtime world axes and
    n_dot = n x rate; all three components enter PN. None retains the built-in
    observation path. Only the general surface runtime supports this input.
    """

    _validate_surface_options(early_miss_s,clutter_model,clutter_min_depression_deg,multipath_gain,
                              launcher_radar_gimbal_deg,launcher_support,launcher_radar_geometry)
    from .radar_seeker import validate_seeker_search, validate_chaff_capture, validate_track_gate_model
    from .observation import validate_min_altitude, validate_target_observer
    try:
        validate_seeker_search(seeker_search)
        validate_chaff_capture(chaff_capture)
        validate_track_gate_model(track_gate_model)
        if chaff_capture is not None and track_gate_model is not None:
            raise ValueError("track_gate_model and chaff_capture are mutually exclusive")
        validate_min_altitude(min_altitude_m)
        validate_target_observer(target_observer)
        from .q_angular_observer import validate_angular_observer
        validate_angular_observer(angular_observer)
    except ValueError as exc:
        raise SimulationInputError(str(exc)) from exc
    if model_mode == "general" and "model_family" in missile_profile:
        if missile_profile.get("_runtime_unsupported"):
            raise UnsupportedPhysicsError("不支持的制导或物理类型：" + "、".join(missile_profile["_runtime_unsupported"]))
        from .surface_runtime import simulate_surface
        try:
            return simulate_surface(missile_profile, scenario, target_factory, early_miss_s, clutter_model,
                                    clutter_min_depression_deg, bool(cw_on_clear_beam), multipath_gain,
                                    launcher_radar_gimbal_deg, bool(require_seeker_lock),launcher_support,
                                    seeker_search,min_altitude_m=min_altitude_m,chaff_capture=chaff_capture,track_gate_model=track_gate_model,target_observer=target_observer,
                                    launch_quaternion_xyzw=launch_quaternion_xyzw,launcher_radar_geometry=launcher_radar_geometry,
                                    guidance_input=guidance_input,angular_observer=angular_observer,infrared_seeker=infrared_seeker)
        except ValueError as exc:
            raise SimulationInputError(str(exc)) from exc
    if model_mode not in {"general", "legacy_h2"}:
        raise SimulationInputError(f"未知模型模式：{model_mode}。")
    if min_altitude_m is not None:
        raise SimulationInputError('min_altitude_m needs the general surface sensor_track runtime')
    if target_factory is not None or early_miss_s is not None or clutter_model is not None or cw_on_clear_beam \
            or multipath_gain is not None or launcher_radar_gimbal_deg is not None or launcher_radar_geometry is not None or require_seeker_lock or launcher_support is not None \
            or seeker_search is not None or chaff_capture is not None or track_gate_model is not None or target_observer is not None or angular_observer is not None or launch_quaternion_xyzw is not None or guidance_input is not None or infrared_seeker is not None:
        raise SimulationInputError("外部目标模型、提前脱靶判定、杂波、多径、载机雷达和导引头锁定条件只支持通用 surface 运行时。")

    base_config = _validate_profile(missile_profile)
    normalized = validate_scenario(scenario)
    config = copy.deepcopy(base_config)
    profile_lofting_enabled = bool(config["guidance"].get("lofting_enabled", False))
    # The public scenario switch is the only activation control for GUI/API
    # runs.  A profile that does not expose loft capability remains disabled;
    # when enabled, the existing guidance gate/equation is left untouched.
    config["guidance"]["lofting_enabled"] = bool(
        normalized["loft_enabled"] and profile_lofting_enabled
    )
    physical_lifetime = float(config["performance"]["lifetime_s"])
    requested_limit = normalized["max_simulation_time_s"]
    limited_by_scenario = requested_limit is not None and float(requested_limit) < physical_lifetime
    if requested_limit is not None:
        config["performance"]["lifetime_s"] = min(physical_lifetime, float(requested_limit))
    result = H2Simulator(config).run(_case_from_scenario(normalized))
    if result["event_type"] == "numerical_failure":
        raise ArithmeticError("模拟器检测到非有限数值或积分未能正常终止。")
    return {
        "schema_version": 1,
        "missile": {
            "id": missile_profile.get("missile_id", missile_profile.get("id")),
            "name": missile_profile.get("display_name", missile_profile.get("name")),
            "status": missile_profile.get("model_status", missile_profile.get("status")),
            "parameter_source": {
                "version": missile_profile.get("provenance", {}).get("source_version"),
                "commit": missile_profile.get("provenance", {}).get("source_commit"),
            },
        },
        "scenario": normalized,
        "summary": _summarize(
            result,
            config,
            limited_by_scenario,
            normalized["loft_enabled"],
            profile_lofting_enabled,
        ),
        "markers": _stage_markers(config) + [{
            "time_s": float(result["terminal_time_s"]),
            "label": "终止事件",
            "kind": "termination",
        }],
        "samples": result["samples"],
        "model": {
            "label": result["model_label"],
            "aerodynamics": result["aero_model_version"],
            "geometry": result["force_geometry_version"],
            "control": result["control_model_version"],
            "integrator": result["integrator"],
            "time_step_s": result["time_step_s"],
            "observation_mode": result["observation_mode"],
            "requested_observation_mode": result.get("requested_observation_mode", result["observation_mode"]),
            "guidance_update_hz": result["guidance_update_hz"],
            "observation_provider": result.get("observation_provider", "ideal_truth"),
            "lock_state_machine": "TRK->INS+SRC->TRK" if result["observation_mode"] == "sensor_track" else "not_applicable",
            "radar_model": (
                "deterministic_gate_candidate_v1"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
                if result["observation_mode"] == "sensor_track"
                else "ideal_truth"
            ),
            "datalink_update": (
                "every_guidance_tick"
                if result.get("observation_provider") == "radar_datalink_ins_v1"
                else "not_applicable"
            ),
            "datalink_update_count": result.get("datalink_update_count", 0),
            "random_measurement_noise": False,
            "multipath_enabled": False,
            "sarh_model_enabled": False,
            "runtime_adapter": missile_profile.get("_runtime_adapter", "frozen_config"),
            "runtime_boundary": missile_profile.get("_runtime_boundary", missile_profile.get("_model_config", {}).get("reference", {}).get("source")),
            "runtime_assumptions": list(missile_profile.get("_runtime_assumptions", [])),
            "scenario_loft_enabled": bool(normalized["loft_enabled"]),
            "profile_lofting_enabled": profile_lofting_enabled,
        },
    }


from .surface_runtime import (create_surface_missile, launch_quaternion_xyzw_from_velocity_normal,
                              launch_quaternion_xyzw_from_dagor)


__all__ = [
    "SimulationInputError",
    "UnsupportedPhysicsError",
    "simulate",
    "create_surface_missile",
    "launch_quaternion_xyzw_from_velocity_normal",
    "launch_quaternion_xyzw_from_dagor",
    "validate_scenario",
]
