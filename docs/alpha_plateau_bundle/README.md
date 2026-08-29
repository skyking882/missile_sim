# 迎角平台补充包 — R-77-1 / A7 平射

配套文档：`docs/ALPHA_PLATEAU_SYNTHETIC_DAMPING.md`（完整推导与实验记录）。
本包只装**原始材料**，不复述结论。

打包日期 2026-08-26，对应工作树未提交状态。**仓库源码未修改**；模型跑的是
`src/` 的 scratchpad 副本，副本上只加了两个实验开关（默认值与出厂逐位一致）。

---

## 目录

```
01_replay/   r77_1_level_20260824.tsv        46 帧游戏服务器回放逐帧读数
             README_replays.md               列语义与已知偏差

02_model/    a7_baseline_dt0.020.csv         现状（ζ=1，cos 投影未归一化，指令 23.7°）
             a7_three_step_fix_dt0.020.csv   三级修正（ζ=0.5，投影归一化，满舵 26.4°）
             a7_baseline_dt0.002.csv         同上，dt 缩到 1/10
             a7_three_step_fix_dt0.002.csv
             COLUMNS.md                      81 列逐列说明 + 派生列公式
             manifest.json                   每个文件的参数、行数、终止条件

03_code/     moment__h2_dynamics__forces_for_state_h2.py      力矩路径（374–1324 行）
             moment__h2_dynamics__diagnostics_fields.py       诊断字段表（40–160）
             attitude__dynamics__integration.py               姿态路径（80–253）
             actuator__control__update_control_feedback.py    舵执行器（168–539）
             actuator__control__base_indicated_speed_schedule.py  η_q 定义（22–82）
             guidance__pcc_routing.py                         制导路由前半（188–541）
             guidance__guidance_command.py                    制导路由后半（542–937）
             INDEX.md                        逐行索引：哪些行对 fin_torque_body_aoa 是活的

04_config/   datamine_fields_only.json       35 个 kind=="datamine" 字段 + 源字段名
             non_datamine_fields.json        25 个拟合/标定/识别/假设字段 + notes
             su_r_77_1_profile_raw.json      完整 profile 原文
             su_r_77_1_resolved_runtime.json 适配器展开后的运行时 config
             profile_h2_runtime_defaults.json  共享运行时默认值
             missile_profile.schema.json     profile schema

scripts/     check_plant_identity.py         只读 CSV，不 import 本仓库任何模块
```

---

## 这包是为哪个判别准备的

> ζ=0.5 是真的 effective damping，还是在补偿 K/I、舵输入、执行器或速度误差？

四个嫌疑各自对应的材料：

### 嫌疑 ①：K / I 弄错了

`ω_n` 逐点在 CSV 里（`pitch/yaw_natural_frequency_rad_s`），`K = ω_n²`。
`I = mass_kg · L²/12`（`inertia_kg_m2` 列）。

- `L = 3.71 m` 是 datamine（`04_config/datamine_fields_only.json`）。
- **`I = m·L²/12` 是假设**（匀质细杆），没有 datamine 依据。
- `K` 里的力矩斜率标度 `packed_lift_slope_scale = 0.58` 在
  `profile_h2_runtime_defaults.json`，来源是**力通道**的三飞行拟合，
  力矩通道没有独立数据集，直接沿用。

**必须知道的简并**：配平亏损 `δ−α = 2ζω/ω_n` 只约束 **ζ/√K** 这一个组合。
ζ 大一倍和 K 小到四分之一在这个观测量上**完全等价**。ζ≈0.5 是在
「K 按现有公式算」这个前提下得到的，两条推导（闭环扫描、在游戏工作点上开环反算）
共用同一个 K，所以它们不构成独立验证。要分开必须有第二个观测量 —— 见文档 §9。

### 嫌疑 ②：舵输入不是我们以为的那个

CSV 里舵的完整链条按顺序都在：

```
pitch_fin_command → pitch_requested_fin_command → pitch_requested_fin_angle_rad
                  → actual_pitch_fin_angle_rad
```

`pcc_routed_alpha_deg` 是制导路由出去的舵角（度）。
在捕获平台段可以直接验：`pcc_routed_alpha_deg == 23.7 · cos(angle_of_attack_rad)`
—— 这就是 cos 投影漏归一化（`guidance.py:770–776`，见 `03_code/INDEX.md`）。

### 嫌疑 ③：执行器

`pitch_requested_fin_angle_rad`（作动器输入）与 `actual_pitch_fin_angle_rad`
（输出）同时在列，一阶滞后 `α_act = dt/(τ_act+dt)` 见 `control.py:500`。

**`τ_act = 0.08 s` 是假设**（`non_datamine_fields.json` 里
`control.actuator_time_constant_s` = null，datamine 无此字段）。
出厂 dt = 0.02 s，与 τ_act 只差 4 倍 —— 所以额外给了 `dt0.002` 两个文件：
若 dt 缩到 1/10 后所需的 ζ 不变，离散化就被排除。

### 嫌疑 ④：速度误差

`speed_kmh` / `dynamic_pressure_pa` / `fin_force_speed_scale`（η_q）都在列，
可与 `01_replay/` 的 `speed_kmh` 逐帧对。平台段模型约 1963 km/h、回放 1759 km/h，
动压高约 25%。

**注意这条会污染任何闭环 α 比较**：G ∝ q·α，所以迎角偏低和动压偏高互相抵消。
`scripts/plant_g_closure_audit.py`（在主仓库里）已经单独验过：在**游戏自己的**
q/α 工作点上，plant 的 |ΔG| ≤ 0.17 g。所以标定 guidance 常数要用 game-q 工作点，
不能用闭环 α。

---

## 先跑这个

```bash
cd scripts
python3 check_plant_identity.py ../02_model/a7_baseline_dt0.020.csv
```

它只读 CSV，不 import 仓库任何模块，输出三段：

- **A** 逐点验 `I·ω̇ = K(δ−α) − C·ω`。baseline 残差 ~6e-9 rad/s²（浮点噪声）
  ⇒ 力矩恒等式在 CSV 内部自洽，ζ 不是在补 plant 内部的别的项。
- **B** 稳态段的隐含 ζ。baseline 给 **1.002**（0.962–1.012）⇒ 与代码里手填的 ζ=1 一致。
  把同一段拿去和回放的 δ/α 对，需要的是 ≈0.5。
- **C** 真实气动阻尼 `ζ_aero = ω_n·Δ/(2V)`，用 CSV 里那个**被算出来但没进力矩**的
  `yaw_tail_rate_incidence_rad`（= ω·Δ/V）反解。结果 0.00127，恒定，
  是代码里 ζ 的 **1/788**。

四个 CSV 都可以喂给它。

---

## 两个实验开关（只存在于 scratchpad 副本，不在仓库里）

| 开关 | 位置 | 默认 | 作用 |
|---|---|---|---|
| `control._exp_zeta` | `h2_dynamics.py` 把 `2.0 * ω_n` 换成 `2.0 * ζ * ω_n` | `1.0` | 缩放合成阻尼 |
| `control._exp_fix_projection` | `guidance.py` 投影加一次归一化 | `False` | 修 cos α 损失 |
| `control._exp_physical_damping` | 同上，改用 `K·Δ/V` 的真实气动阻尼 | `False` | 只留物理阻尼 |

取默认值时与仓库出厂逐位一致（已验）。

---

## 边界

独立本地工程候选的诊断材料，不对任何私有实现主张等价性。
`ζ`、`I = mL²/12`、力矩斜率 `0.58` 三者均**未被辨识**。
ζ ≈ 0.5 来自**单次飞行**，跨飞行、跨弹种普适性未验。
回放数据由项目所有者 2026-08-24 从服务器回放逐帧转录，
`angle_of_attack_deg` 在小迎角（≲1.5°）有已知显示偏差。
