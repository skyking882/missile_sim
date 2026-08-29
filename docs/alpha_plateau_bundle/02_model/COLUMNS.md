# 02_model — CSV 列说明

四个文件，同一发 A7 平射（8 km / 方位 30° / 6300 m / 直线目标 / `ideal_truth`）：

| 文件 | ζ | cos 投影 | `capture_alpha_max_deg` | dt |
|---|---|---|---|---|
| `a7_baseline_dt0.020.csv` | 1.0 | 未归一化（现状） | 23.7° | 0.02 s（**出厂值**） |
| `a7_three_step_fix_dt0.020.csv` | 0.5 | 已归一化 | 26.4°（满舵） | 0.02 s |
| `a7_baseline_dt0.002.csv` | 1.0 | 未归一化 | 23.7° | 0.002 s |
| `a7_three_step_fix_dt0.002.csv` | 0.5 | 已归一化 | 26.4° | 0.002 s |

`dt0.002` 那两个是给「ζ 是不是在补偿离散化/作动器」这一条准备的：出厂 dt 是 0.02 s，
作动器时间常数 0.08 s，两者只差 4 倍；把 dt 降到 0.002 s 后若 ζ 的需求量不变，
离散化就被排除了。`manifest.json` 里有每个文件的终止条件和脱靶量。

采样是**每个积分步一行**，不抽稀。

---

## 直接从 sample 导出的列（原名，未改）

**时间 / 质量 / 大气**
`time_s` `mass_kg` `mach` `dynamic_pressure_pa` `indicated_speed_kmh`
`base_indicated_speed_q_ratio` `fin_force_speed_scale`（= η_q，力/力矩斜率的动压标度）
`fin_authority_scale`

**姿态与角速度**（plant 积分的状态量）
`pitch_rad` `yaw_rad` `pitch_rate_rad_s` `yaw_rate_rad_s`

**迎角**
`angle_of_attack_rad`（总迎角，**无符号**）`pitch_alpha_rad` `yaw_alpha_rad`（有符号分量）

**舵：指令 → 请求 → 作动器 → 实际**
`pitch_fin_command` `yaw_fin_command`（归一化 −1..1）
`pitch_requested_fin_command` `yaw_requested_fin_command`（过 unit-disk 之后）
`pitch_requested_fin_angle_rad` `yaw_requested_fin_angle_rad`（作动器**输入**）
`actual_pitch_fin_angle_rad` `actual_yaw_fin_angle_rad`（作动器**输出**，δ）

舵机械上限 δ_max = `finsAoa` = 0.460812 rad = 26.40°，全程常数，不占列；
见 `04_config/datamine_fields_only.json`。

**plant 系数**
`pitch_natural_frequency_rad_s` `yaw_natural_frequency_rad_s`（ω_n = √K）
`pitch_residual_rate_damping_per_s` `yaw_residual_rate_damping_per_s`（**C/I = 2ζω_n，ζ=1 时即 2ω_n**）
`pitch_tail_rate_damping_per_s` `yaw_tail_rate_damping_per_s`（legacy plant 恒为 0）

**力矩分项**
`pitch_fin_moment_equivalent_g` `yaw_fin_moment_equivalent_g`（= scheduled_fins_g·(δ−α)/δ_max）
`pitch_tail_moment_nm` `yaw_tail_moment_nm`（弹簧项）
`pitch_residual_damping_moment_nm` `yaw_residual_damping_moment_nm`（阻尼项，= −C·ω）
`pitch_total_moment_nm` `yaw_total_moment_nm`（两者之和）
`pitch_tail_authority_fraction` `yaw_tail_authority_fraction`（= (δ−α)/δ_max，过 unit-disk）

**被算出来但没用的真实气动阻尼**
`pitch_tail_rate_incidence_rad` `yaw_tail_rate_incidence_rad`（**= ω·Δ/V**，h2_dynamics.py:780–781，
注释明说 "diagnostic only; it does not enter the weathervane spring"）
`pitch_tail_effective_incidence_rad` `yaw_tail_effective_incidence_rad`（= δ − α，**没有**减 ωΔ/V）

**力 / 载荷**
`trajectory_lateral_load_g`（游戏 HUD 那个 G：轨迹法向、含推力分量）
`trajectory_pitch_normal_acceleration_g` `trajectory_yaw_normal_acceleration_g`
`axial_specific_force_g` `thrust_n` `drag_n` `total_cda_m2` `cda_alpha_m2` `fin_drag_n`

**自动驾驶仪**
`pitch_pid_output` `yaw_pid_output` `pitch_pid_integral` `yaw_pid_integral`
`pid_feedback_pitch_g` `pid_feedback_yaw_g`
`midcourse_weight`（w：直接舵路由权重，CAPTURE 期为 1）

**制导 / 捕获**
`pcc_capture_mode`（capture/homing）`pcc_capture_ratio`（R_cap）
`pcc_envelope_g`（a_env）`pcc_routed_alpha_deg`（路由出去的舵角，度）
`heading_error_deg`（ε，到 PIP）`j_psi_actual_deg`（累计水平航向积分）
`distance_to_target_m` `closing_speed_mps`

---

## 派生列（本包新加，公式如下，都可以从上面的列自己复算）

| 列 | 公式 |
|---|---|
| `commanded_accel_body_pitch_g` / `..._yaw_g` | 原 `commanded_acceleration_g` 是二元组，拆成两列 |
| `altitude_m` | `position_m[1]` |
| `speed_mps` / `speed_kmh` | `|velocity_mps|` |
| `inertia_kg_m2` | `mass_kg · L²/12`，L = 3.71 m（**这是假设，见 04_config**） |
| `pitch_angular_accel_rad_s2` | `pitch_total_moment_nm / inertia_kg_m2` |
| `yaw_angular_accel_rad_s2` | `yaw_total_moment_nm / inertia_kg_m2` |
| `omega_total_rad_s` | `hypot(pitch_rate, yaw_rate)` |
| `delta_total_rad` | `hypot(actual_pitch_fin_angle, actual_yaw_fin_angle)` |
| `trim_deficit_rad` | `delta_total_rad − angle_of_attack_rad`（实测亏损） |
| `predicted_deficit_rad` | `2·omega_total / ω_n`（ζ=1 的预测；ω_n 取主导轴那一个） |
| `zeta_implied` | `trim_deficit_rad / predicted_deficit_rad` |

`zeta_implied` 只在**稳态转弯**（ω̇≈0）的平台段有意义，
瞬态和释放段会跑飞——别整段平均。
