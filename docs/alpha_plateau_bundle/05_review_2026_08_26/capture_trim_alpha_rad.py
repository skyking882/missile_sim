# EXPERIMENTAL, scratchpad copy only -- not in the repo's src/.
# 满舵动态配平求根（复审 item 4/5）。插入位置：h2_dynamics.py 的 _add_many() 之前。

def capture_trim_alpha_rad(
    dynamic_pressure_pa: float,
    mass_kg: float,
    thrust_n: float,
    delta_max_rad: float,
    omega_n_rad_s: float,
    speed_mps: float,
    zeta: float,
    config: dict[str, Any],
) -> float:
    """EXPERIMENTAL (2026-08-26 review item 4/5): full-fin DYNAMIC trim alpha.

    At a steady turn the weathervane trims where the spring balances the rate
    damping, not where alpha == delta:

        0 = wn^2 (delta_max - alpha*) - 2 zeta wn omega*,   omega* ~= a_n(alpha*)/V

    which rearranges to the scalar root problem

        alpha* + (2 zeta / (wn V)) a_n(alpha*) = delta_max        [monotone in alpha*]

    a_n(alpha) is the SAME trajectory-normal, thrust-inclusive law
    capture_alpha_envelope_g already evaluates.  f(0) < 0 and f(delta_max) > 0,
    so plain bisection on [0, delta_max] is safe and needs no tuning.
    """

    if omega_n_rad_s <= 0.0 or speed_mps <= 0.0 or delta_max_rad <= 0.0:
        return max(delta_max_rad, 0.0)
    gravity = float(config["atmosphere"]["gravity_mps2"])
    lag_s = 2.0 * max(zeta, 0.0) / omega_n_rad_s          # tau_q = 2 zeta / wn

    def residual(alpha_rad: float) -> float:
        a_n_mps2 = capture_alpha_envelope_g(
            dynamic_pressure_pa, mass_kg, thrust_n, alpha_rad, config
        ) * gravity
        return alpha_rad + lag_s * a_n_mps2 / speed_mps - delta_max_rad

    lo, hi = 0.0, float(delta_max_rad)
    if residual(hi) <= 0.0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


