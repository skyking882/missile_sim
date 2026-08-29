# 05 — 2026-08-26 外部复审的响应

复审提出两个优先级高于 ζ 的结构问题，两条都已在**实验副本**上实现并验证。
完整叙述见 `docs/ALPHA_PLATEAU_SYNTHETIC_DAMPING.md` §10。

## 新增的实验开关（只在 scratchpad 副本，仓库 `src/` 未改）

| 开关 | 默认 | 作用 |
|---|---|---|
| `control._exp_fix_projection` | `False` | **P** cos α 投影归一化 |
| `control._exp_fin_deflection_limit_deg` | `None` | **Fcmd** 直接舵指令幅值与 `capture_alpha_max_deg` 解耦 |
| `control._exp_envelope_mode` | `"alpha_max"` | **Fenv** 改成 `"full_fin_trim"` 时走配平求根 |
| `control._exp_zeta` | `1.0` | **Z** 缩放合成阻尼 |
| `control._exp_flight_gain_on_capture` | `False` | **G** datamine launch gain 也乘到捕获通道与直接舵路由 |
| `control._exp_capture_latch` | `False` | **L** 进入 HOMING 后不再回发射捕获 |
| `control._exp_physical_damping` | `False` | 只留 `K·Δ/V` 的单点尾舵阻尼 |

全部取默认值时与仓库出厂逐位一致。

## 文件

- `capture_trim_alpha_rad.py` — 满舵动态配平求根（逐字）。单调 + 二分，无调参。
- `ablate_factorial.py` — 因子化 ablation，重现 §10.3 的表。
- `dt_convergence_and_frozen_bands.py` — 重现 §10.4 的 dt 收敛表与 §10.5 的冻结带。

两个脚本都需要 scratchpad 的 `src_exp` 副本；脚本顶部的路径按需要改。

## 关键数字

- `a_env` 误差：13% → **1%**（t=1.5 s，19.25→17.25 g vs 实际 17.08 g）
- `J_ψ(0.9)`：旧三级 10.64° → 全部+latch **5.14°**（游戏重构 ≈ 4.7°）
- 释放时刻：旧三级 1.76 s → **2.02 s**（游戏 2.1–2.3 s）
- dt 收敛：无 latch 时 6/4/1 次模式切换且终止条件不同；有 latch 时三档全部 1 次、
  近炸 10.0 m、释放 2.02/2.02/2.01
- 80° 冻结带：旧三级引信起爆 10.0 m（挂）→ 全部+latch 16.5 m 不引信（**通过**）

## 仍未做

1. **replay-forced plant run** —— 强制 V(t)/q(t)/m(t)/T(t) 为游戏时间线，只积分 α/ω/δ，
   干净识别 `tau_q` / `ω_n` / `τ_act`。复审列为最优先，本轮**未做**。
2. `plant_g_closure_audit` 逐帧表未进包，那条结论目前仍不可独立复算。
3. 游戏侧高时间分辨率建立段、离轴 15/30/45° 组 —— 需要重新录。
