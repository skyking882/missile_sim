# 03_code — 逐行索引

四条路径，全部是 2026-08-26 工作树的**逐字摘录**，行号保留原文件的行号。

| 文件 | 源 | 原始行号 |
|---|---|---|
| `moment__h2_dynamics__forces_for_state_h2.py` | `src/aim120_model/h2_dynamics.py` | 374–1324 |
| `moment__h2_dynamics__diagnostics_fields.py` | 同上 | 40–160 |
| `attitude__dynamics__integration.py` | `src/aim120_model/dynamics.py` | 80–253 |
| `actuator__control__update_control_feedback.py` | `src/aim120_model/control.py` | 168–539 |
| `actuator__control__base_indicated_speed_schedule.py` | 同上 | 22–82 |
| `guidance__pcc_routing.py` | `src/aim120_model/guidance.py` | 188–541 |
| `guidance__guidance_command.py` | 同上 | 542–937 |

## `forces_for_state_h2` 里对 `fin_torque_body_aoa` **实际活着**的行

这个函数同时承载四个 plant 候选（`direct_fin_g` / `fin_torque_body_aoa` /
`body_cm_tail_force_moment` / `generalized_aero_moment`）。R-77-1 用的是
`fin_torque_body_aoa`（见 `04_config/su_r_77_1_resolved_runtime.json` 的
`control.plant_semantics`）。活的行是：

| 原始行号 | 内容 |
|---|---|
| 385 | `legacy_fin_torque_plant = plant_semantics == "fin_torque_body_aoa"` |
| 664–745 | `scheduled_fins_g`（力矩通道）与 `scheduled_fins_g_force`（力通道）的**拆分** |
| 668–671 | `scheduled_fins_g = finsLatAccel · eta_q`，随后 `*= packed_lift_slope_scale (0.58)` |
| 727–735 | 力通道走 `packed_lift_force_eta_law`：`k_force(η)=0.574·η^0.242` |
| **776–787** | **`ω·Δ/V` 在这里被算出来，注释明说"diagnostic only，不进风向标弹簧"** |
| 862–866 | `pitch_moment_fraction = limit_unit_disk((δ−α)/δ_max, …)` |
| 880–882 | `pitch_fin_moment_equivalent_g = scheduled_fins_g · moment_fraction` |
| 883–900 | 力通道：`path_g_from_alpha=True` ⇒ 升力跟 **α** 走，不跟 δ 走 |
| 906–910 | `control_force`（打包升力，沿 flow-normal 基） |
| 926–945 | `loadFactorMax` 只径向裁 `F_N`，不裁力矩通道 |
| **1073–1094** | **刚度与阻尼：`K = scheduled_fins_g·g·Δ/((L²/12)·δ_max)`，`C/I = 2·ω_n`（ζ=1 手填）** |
| 1146–1149 | `pitch_damping = pitch_residual_rate_damping`（= 2ω_n） |
| 1191–1208 | `ω̇ = moment_eq_g·g·Δ/(L²/12) − damping·ω`；`total_moment = tail + residual_damping` |
| 1241–1242 | 导出 `pitch/yaw_angular_acceleration_rad_s2` |

**等价形式**（把 1073–1094 与 1191–1208 合起来）：

    ω̇ = ω_n²·(δ − α) − 2ζ·ω_n·ω,     ω_n² = K = scheduled_fins_g·g·Δ / ((L²/12)·δ_max)

CSV 里 `pitch_total_moment_nm / inertia_kg_m2` 就是 ω̇（已作为 `pitch_angular_accel_rad_s2`
列预先算好），可以直接和 `pitch_natural_frequency_rad_s`、`actual_pitch_fin_angle_rad`、
`pitch_alpha_rad`、`pitch_rate_rad_s` 对拟合，逐点闭合上式。

## 舵执行器路径的关键行（`control.py`）

| 原始行号 | 内容 |
|---|---|
| 431–463 | `pid_output_semantics = "fin_angle_rad"`：PID 输出**就是**物理舵角，`finsAoa` 是唯一的角度钳位 |
| 456–460 | `requested_fin_angle = direct_fin_angle + (1−w)·pid_fin_angle`，w 即 `midcourse_weight` |
| 491–498 | `fin_torque_body_aoa` 专用：pitch/yaw 两轴的 requested 分数一起过 `limit_unit_disk` |
| **500** | **作动器：一阶滞后 `α_act = dt/(τ_act+dt)`，`τ_act = 0.08 s`（assumed，datamine 无此字段）** |
| 501–508 | `actual_fin_angle += α_act·(desired − actual)`；`desired = requested·authority·δ_max` |
| 39–82 | `eta_q = (V_ind/1800 km/h)²`，`fin_force_scale = eta_q`（上限 4） |

## 制导路由的关键行（`guidance.py`）

| 原始行号 | 内容 |
|---|---|
| 649 | `pcc_active`：`midcourse_lead_turn.mode == "pcc_alpha"` |
| 707 | `a_cap = V·sin(ε)/τ_c` |
| 709–714 | `R_cap = a_cap / a_env`，`a_env` 由 `h2_dynamics.capture_alpha_envelope_g` 在 α_max 处算出 |
| 754 | `alpha_full_rad = capture_alpha_max_deg` |
| **770–776** | **`fin_fraction_mag = clamp(α_max·R/δ_max, 0, 1)`；投影 `dot(demand_hat, up/right)` 未归一化 ⇒ 丢掉 cos α** |
| 841–845 | `midcourse_fin_fraction`（CAPTURE 时 w=1，直接舵） |
| 847–849 | `capture_routed_alpha_deg` 遥测（CSV 的 `pcc_routed_alpha_deg` 列） |
| 493–541 | CAPTURE/HOMING 状态机（迟滞门限） |

**注意** `capture_alpha_max_deg` 兼两个差事：(1) 770 行的舵指令幅度，(2) 709 行 `a_env` 的
释放门限。程序从未把实现出来的 α 与 `capture_alpha_max_deg` 比较过。
