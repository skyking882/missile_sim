#!/usr/bin/env python3
"""Replay-forced plant identification (review §5 spec, 2026-08-26).

V(t), q_inf(t) and the flight-path turn rate are FORCED from the replay.  Only
the actuator + attitude plant is integrated, so the drag thread and the guidance
geometry cannot contaminate the identification.

    tau_d * ddelta/dt + delta = G_flight(t) * delta_max          (full-fin capture)
    dalpha/dt = q_b - gammadot_game(t)
    dq_b/dt   = s_K * wn0^2(t) * (delta - alpha - tau_q * q_b)

Fit s_K, tau_q (and q_b(t0) as a nuisance) to the replay alpha over
0.5 <= t <= 1.9 s.  2.1-2.9 s is held out entirely.  zeta is NOT fitted; it is
converted afterwards at an operating point via zeta = tau_q * wn / 2.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
PROJ = Path("/Users/skyking/Documents/missle_sim")
sys.path.insert(0, str(PROJ / "src"))
from aim120_model.atmosphere import StandardAtmosphere

G0 = 9.81
RHO0 = 1.225000018
BASE_IND_KMH = 1800.0
ETA_MAX = 4.0
FINS_LAT_G = 53.5864
SLOPE_SCALE = 0.58
ARM = 0.175
LENGTH = 3.71
IPM = LENGTH * LENGTH / 12.0
DELTA_MAX = 0.460812
FLIGHT_GAIN = [(0.30, 0.0), (0.31, 1.0)]

rows = list(csv.DictReader(open(PROJ / "data/replays/r77_1_level_20260824.tsv"), delimiter="\t"))
REPLAY = [(float(r["missile_flight_time_s"]), float(r["speed_kmh"]) / 3.6,
           float(r["overload_g"]), math.radians(float(r["angle_of_attack_deg"])))
          for r in rows if r["angle_of_attack_deg"] not in ("", "NA")]
REPLAY.sort()

def interp(table, x):
    if x <= table[0][0]: return table[0][1]
    if x >= table[-1][0]: return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]

V_TAB = [(t, v) for t, v, _, _ in REPLAY]
G_TAB = [(t, g) for t, _, g, _ in REPLAY]
A_TAB = [(t, a) for t, _, _, a in REPLAY]

def build_forcing(altitude_m):
    rho = StandardAtmosphere().sample(altitude_m).density_kg_m3
    def wn0_sq(t):
        V = interp(V_TAB, t)
        q = 0.5 * rho * V * V
        v_ind = math.sqrt(2.0 * q / RHO0) * 3.6
        eta = min((v_ind / BASE_IND_KMH) ** 2, ETA_MAX)
        sched = FINS_LAT_G * eta * SLOPE_SCALE
        return sched * G0 * ARM / (IPM * DELTA_MAX)
    def gammadot(t):
        V = max(interp(V_TAB, t), 1.0)
        g_meas = interp(G_TAB, t)
        a_h = math.sqrt(max((g_meas * G0) ** 2 - G0 * G0, 0.0))
        return a_h / V
    return wn0_sq, gammadot, rho

def simulate(s_K, tau_q, qb0, t0, t1, altitude_m, tau_d=0.08, alpha0=None, dt=0.002):
    wn0_sq, gammadot, _ = build_forcing(altitude_m)
    a0 = interp(A_TAB, t0) if alpha0 is None else alpha0
    # actuator has been driven by G_flight*delta_max since 0.305 s
    d0 = DELTA_MAX * (1.0 - math.exp(-max(t0 - 0.305, 0.0) / tau_d))
    y = [d0, a0, qb0]
    out = [(t0, list(y))]
    def deriv(t, y):
        d, a, qb = y
        return [(interp(FLIGHT_GAIN, t) * DELTA_MAX - d) / tau_d,
                qb - gammadot(t),
                s_K * wn0_sq(t) * (d - a - tau_q * qb)]
    t = t0
    n = int(round((t1 - t0) / dt))
    for _ in range(n):
        k1 = deriv(t, y)
        k2 = deriv(t + dt/2, [y[i] + dt/2*k1[i] for i in range(3)])
        k3 = deriv(t + dt/2, [y[i] + dt/2*k2[i] for i in range(3)])
        k4 = deriv(t + dt,   [y[i] + dt*k3[i]   for i in range(3)])
        y = [y[i] + dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(3)]
        t += dt
        out.append((t, list(y)))
    return out

FIT_LO, FIT_HI = 0.5, 1.9
HOLD_LO, HOLD_HI = 2.1, 2.9
FIT_PTS  = [(t, a) for t, _, _, a in REPLAY if FIT_LO <= t <= FIT_HI]
HOLD_PTS = [(t, a) for t, _, _, a in REPLAY if HOLD_LO <= t <= HOLD_HI]

QB0_MAX = math.radians(60.0)     # datamine rateMax = 60 deg/s

def sse(params, alt, pts, t_end, s_K=None, alpha0=None):
    tau_q, qb0 = params
    if tau_q < 0.0 or tau_q > 1.0 or abs(qb0) > QB0_MAX: return 1e9
    try:
        traj = simulate(s_K, tau_q, qb0, FIT_LO, t_end, alt, alpha0=alpha0)
    except (OverflowError, ValueError):
        return 1e9
    tab = [(t, y[1]) for t, y in traj]
    tot = 0.0
    for t, a in pts:
        d = interp(tab, t) - a
        if not math.isfinite(d) or abs(d) > 10.0: return 1e9
        tot += d * d
    return tot

def nelder_mead(f, x0, step, iters=120):
    n = len(x0)
    simplex = [list(x0)] + [[x0[i] + (step[i] if i == j else 0.0) for i in range(n)] for j in range(n)]
    vals = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < 1e-12: break
        cen = [sum(p[i] for p in simplex[:-1]) / n for i in range(n)]
        xr = [cen[i] + 1.0 * (cen[i] - simplex[-1][i]) for i in range(n)]; fr = f(xr)
        if fr < vals[0]:
            xe = [cen[i] + 2.0 * (cen[i] - simplex[-1][i]) for i in range(n)]; fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = [cen[i] + 0.5 * (simplex[-1][i] - cen[i]) for i in range(n)]; fc = f(xc)
            if fc < vals[-1]: simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    vals[i] = f(simplex[i])
    k = min(range(n + 1), key=lambda i: vals[i])
    return simplex[k], vals[k]


def fit_at(s_K, alt, pts=None, t_end=None):
    pts = pts or FIT_PTS
    t_end = t_end or (FIT_HI + 0.05)
    best = None
    for tq0 in (0.08, 0.16, 0.30):
        for qb0 in (0.0, math.radians(15)):
            x, v = nelder_mead(lambda p: sse(p, alt, pts, t_end, s_K=s_K), [tq0, qb0], [0.05, math.radians(8)])
            if best is None or v < best[1]: best = (x, v)
    return best

print("=" * 86)
print("replay-forced plant identification   (review 5 spec)")
print("窗口 0.5-1.9 s 拟合，2.1-2.9 s 完全留出；tau_d = 0.08 s 固定")
print("delta 由 G_flight(t)*delta_max 开环驱动 —— 检验的是【满舵捕获】假说")
print("=" * 86)

ALT = 6300.0
wn0_sq, _, rho = build_forcing(ALT)
wn0_16 = math.sqrt(wn0_sq(1.6))
print(f"\nh = {ALT:.0f} m, rho = {rho:.4f}；现有公式 omega_n(1.6 s) = {wn0_16:.2f} rad/s (s_K=1)")
print(f"代码当前 zeta = 1.0  =>  tau_q = 2/omega_n = {2.0/wn0_16:.3f} s")
print(f"若 zeta = 0.5        =>  tau_q            = {1.0/wn0_16:.3f} s\n")

print(f"{'s_K':>7} {'omega_n(1.6)':>13} {'tau_q':>8} {'q_b(t0)':>9} {'拟合RMS':>9} {'留出RMS':>9} {'zeta 换算':>10}")
print("-" * 86)
prof = []
for s_K in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    (tq, qb0), v = fit_at(s_K, ALT)
    traj = simulate(s_K, tq, qb0, FIT_LO, 3.0, ALT)
    tab = [(t, y[1]) for t, y in traj]
    hold = math.sqrt(sum((interp(tab,t)-a)**2 for t,a in HOLD_PTS)/len(HOLD_PTS))
    wn = math.sqrt(s_K * wn0_sq(1.6))
    prof.append((s_K, tq, v, wn))
    print(f"{s_K:7.2f} {wn:13.2f} {tq:8.4f} {math.degrees(qb0):9.2f} "
          f"{math.degrees(math.sqrt(v/len(FIT_PTS))):9.3f} {math.degrees(hold):9.3f} {tq*wn/2:10.3f}")

vmin = min(p[2] for p in prof)
print("-" * 86)
print(f"最优 SSE = {vmin:.5f}；沿 s_K 方向 SSE 变化 = "
      f"{max(p[2] for p in prof)/vmin:.2f}x（{min(p[0] for p in prof)}x .. {max(p[0] for p in prof)}x 的 s_K 跨度 64 倍）")
print(f"tau_q 跨这 64 倍 s_K 只从 {min(p[1] for p in prof):.4f} 变到 {max(p[1] for p in prof):.4f} s")

print("\n高度敏感性（s_K = 1 固定）：")
print(f"{'h (m)':>7} {'rho':>8} {'omega_n(1.6)':>13} {'tau_q':>8} {'拟合RMS':>9}")
for alt in (6000.0, 6300.0, 6600.0):
    (tq, qb0), v = fit_at(1.0, alt)
    w0, _, r = build_forcing(alt)
    print(f"{alt:7.0f} {r:8.4f} {math.sqrt(w0(1.6)):13.2f} {tq:8.4f} "
          f"{math.degrees(math.sqrt(v/len(FIT_PTS))):9.3f}")

print("\ns_K = 1 的逐点结果（h = 6300 m）：")
(tq, qb0), v = fit_at(1.0, ALT)
traj = simulate(1.0, tq, qb0, FIT_LO, 3.0, ALT)
tab = [(t, y[1]) for t, y in traj]; dtab = [(t, y[0]) for t, y in traj]
print(f"{'t':>5} {'游戏 a':>8} {'模型 a':>8} {'残差':>7} {'delta':>7} {'区':>8}")
for t, a in FIT_PTS + HOLD_PTS:
    m = interp(tab, t)
    print(f"{t:5.1f} {math.degrees(a):8.2f} {math.degrees(m):8.2f} {math.degrees(m-a):7.2f} "
          f"{math.degrees(interp(dtab,t)):7.2f} {('fit' if t<=FIT_HI else 'HOLDOUT'):>8}")
