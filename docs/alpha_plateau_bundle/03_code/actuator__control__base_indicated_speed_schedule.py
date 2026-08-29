# ==== VERBATIM EXCERPT ====================================================
# source : src/aim120_model/control.py
# lines  : 22-82   (1-based, inclusive; original line numbers)
# note   : eta_q = (V_ind/baseIndSpeed)^2 的定义（CSV 的 fin_force_speed_scale 列）
# repo   : working tree at 2026-08-26, unmodified
# ==========================================================================

class BaseIndicatedSpeedSchedule:
    """Candidate speed schedule derived from dynamic pressure.

    ``dynamic_pressure_ratio`` is q/q_ref, where q_ref uses sea-level density
    and the profile's raw baseIndSpeed.  The three scale fields keep candidate
    placement explicit instead of silently rewriting the raw PID values.
    """

    mode: str
    base_indicated_speed_kmh: float | None
    indicated_speed_kmh: float
    dynamic_pressure_ratio: float
    pid_output_scale: float
    requested_fin_scale: float
    fin_force_scale: float


def base_indicated_speed_schedule(
    dynamic_pressure_pa: float,
    config: dict[str, Any],
    sea_level_density_kg_m3: float = 1.225000018,
) -> BaseIndicatedSpeedSchedule:
    """Return B0/B1/B2/B3 schedule factors without changing raw PID gains."""

    control = config["control"]
    mode = str(control.get("base_indicated_speed_mode", "none"))
    if mode not in BASE_INDICATED_SPEED_MODES:
        raise ValueError(f"unknown base_indicated_speed_mode: {mode}")
    q = max(float(dynamic_pressure_pa), 0.0)
    density = max(float(sea_level_density_kg_m3), 1e-12)
    indicated_speed_kmh = math.sqrt(2.0 * q / density) * 3.6
    raw_base = control.get("base_indicated_speed_kmh")
    base = None if raw_base is None else float(raw_base)
    if mode != "none" and (base is None or not math.isfinite(base) or base <= 0.0):
        raise ValueError("active baseIndSpeed candidate requires a positive per-profile base_indicated_speed_kmh")
    if base is None or not math.isfinite(base) or base <= 0.0:
        ratio = 1.0
    else:
        ratio = (indicated_speed_kmh / base) ** 2
    ratio_max = control.get("base_indicated_speed_ratio_max")
    if ratio_max is not None:
        ratio = min(ratio, float(ratio_max))
    if mode == "fin_authority_q":
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0, ratio
    elif mode == "matched_q":
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0 / max(ratio, 1e-12), ratio
    elif mode == "pid_output_q":
        pid_scale, fin_command_scale, fin_force_scale = ratio, 1.0, 1.0
    else:
        pid_scale, fin_command_scale, fin_force_scale = 1.0, 1.0, 1.0
    return BaseIndicatedSpeedSchedule(
        mode=mode,
        base_indicated_speed_kmh=base,
        indicated_speed_kmh=indicated_speed_kmh,
        dynamic_pressure_ratio=ratio,
        pid_output_scale=pid_scale,
        requested_fin_scale=fin_command_scale,
        fin_force_scale=fin_force_scale,
    )


