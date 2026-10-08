# 单发回放扩展数据：AoA、弹目距离与原始字段

后续更新：[未知片段的新线索](unknown_202944.md) 从保留位流中找到了候选目标跟踪位置、速度及状态变化。新结果单独保存在 `outputs/unknown_202944`，本文原主表与原始提取均保留。

针对 `#2026.09.27 20.29.44.wrpl`，继续读取了此前未导出的姿态、创建记录和同步字段。最终目录为 `outputs/extended_202944_final`，导弹仍是实体 `0x00008002` / `pi28_r771_h_110`。主表有 165 行、77 列；保留原位置、时间、差分速度和轨迹长度，同时加入扩展字段。

## 主要结果

| 数据 | 本次结果 | 性质 |
|---|---|---|
| 导弹姿态 | 165 组滚转、航向、俯仰 | 直接解码，欧拉角顺序按上游 Dagor 实现解释 |
| 速度向量 | 模长约 350.11–1253.84 m/s | `b7` 按有符号 16 位解释的候选结果，保留原整数与上游输出 |
| AoA | 约 −5.10°～27.34° | 姿态与速度推算，假设零风；不是 Sensor 的原生 AoA 字段 |
| AoA 最大值时刻 | 回放 1.130 s；候选飞行时间约 0.583 s | 此时俯仰角约 28.90°，AoA 约 27.34°，两者不同 |
| 侧滑角 | 约 −0.616°～0.235° | 同样为零风假设下的估计，符号定义见下 |
| 弹目三维距离 | 首点 7626.37 m，末点 62.20 m | 导弹坐标与目标插值坐标的距离 |
| 目标 | `TargetAI_0`，实体 27650 / `0x00006c02`，F-15C | 由回放实体名称和任务目标指定；不是已解码的导引头锁定目标 ID |
| 目标、发射机、观察机轨迹 | 各 51 点 | 三个不同实体，各自保留 |
| 未解释片段 | 创建记录 817 bit；165 次同步各包含 607 bit 的跳过区 | 原始字节和有效位范围保留，尚未赋予完整导引头语义 |

末点发生在回放 7.384 秒，早于实体销毁的 7.445 秒。62.20 米不能称为脱靶量、爆炸距离或命中判据，也没有向销毁时刻外推。

## 文件

- [主数据表：missile_all_fields.dat](outputs/extended_202944_final/missile_all_fields.dat)：UTF-8 TAB 分隔，165 行、77 列，所有已导出的数值字段。
- [完整原始提取：raw_extended.json](outputs/extended_202944_final/raw_extended.json)：导弹、三架飞机、组件清单、创建字段、同步字段和原始同步片段。
- [创建记录：missile_creation_fields.json](outputs/extended_202944_final/missile_creation_fields.json)：包括初始位置、四元数、多个向量、创建时间、武器引用和 817 bit 未解释区。
- [逐次同步片段：weapon_sync_raw.jsonl](outputs/extended_202944_final/weapon_sync_raw.jsonl)：每行一次同步，字段、字节片段、首个有效位偏移和有效位数，足以继续研究被上游跳过的部分。
- [实体组件清单：missile_ecs_components.json](outputs/extended_202944_final/missile_ecs_components.json)：包括模板 `rocket+arh_shell`、标签、active 原始布尔值、traceHandle 和 Rocket 字段。active 的物理含义未确认，不据此判断导弹是否在飞行。
- [检查与统计：extended_summary.json](outputs/extended_202944_final/extended_summary.json)。

另有三个 `unit_*.dat` 文件，分别保存 `ShooterAI_0`、`TargetAI_0` 和 `Observer_0` 的位置与姿态。组件中的 `net::Object`、`shells::ShellRef` 为未公开的内部结构，本次只登记类型，没有把进程内存当作回放数据。这里的“完整”指本次已提取内容及其原始片段，不能理解为 WRPL 的全部物理/导引字段已被解释。

## 字段说明

| 列或前缀 | 含义与限制 |
|---|---|
| `replay_time_*` | 回放包时间，原始毫秒及秒 |
| `since_entity_created_s` | 相对实体创建事件的时间；与物理发射时钟尚未逐帧校对 |
| `x_m/y_m/z_m` | 上游解码的世界坐标，米，Y 向上 |
| `dt_s`, `derived_world_speed_*`, `sampled_path_length_m` | 与上一版相同的相邻采样差分及折线长度 |
| `*_rad_decoded`, `roll_deg/heading_deg/pitch_deg` | 原始欧拉角及角度换算，分别为 X 滚转、Y 航向、Z 俯仰 |
| `simulation_time_s_candidate` | `b5 / 48`，候选模拟时钟，单位秒 |
| `flight_time_sf1_s_candidate` | 同步中 `sf1` 的直接解码值，候选飞行时长；保留与其他时钟的差异 |
| `velocity_*_mps_candidate`, `speed_*_candidate` | `signed16(b7_raw) × 4000/65535` 及模长，不是 IAS；保留为候选解读 |
| `path_speed_mps_estimate` | 用回放包时间对位置作中心差分的速度，首尾使用单边差分 |
| `path_speed_simulation_clock_mps_estimate` | 使用候选模拟时钟作同样的位置差分 |
| `flight_path_angle_deg_estimate` | 候选速度的垂直方向角 |
| `aoa_deg_estimate_zero_wind` | 假设零风，把速度旋转到弹体系，取 `atan2(-v_body_y, v_body_x)` |
| `sideslip_deg_estimate_zero_wind` | `atan2(v_body_z, sqrt(v_body_x²+v_body_y²))`；按本次弹体系定义符号 |
| `total_incidence_deg_estimate_zero_wind` | 弹轴与速度方向的三维夹角，非负；不等于有符号 AoA |
| `aoa_from_position_difference_deg_estimate` | 以坐标差分速度代替 b7 候选向量的对照估计 |
| `target_*_m_interpolated` | 在目标的 51 个回放采样点间按回放时间线性插值；没有外推 |
| `target_bracketing_gap_ms` | 用于插值的两个目标采样点的时间间隔，便于判断稀疏程度 |
| `target_slant_distance_m` | 三维弹目距离；另有水平距离和目标相对高度 |
| `target_closing_speed_mps_estimate` | 相对速度沿视线的接近速度，正值表示接近 |
| `target_off_boresight_deg_estimate` | 弹轴与目标视线夹角；与 AoA 是不同量 |
| `target_velocity_vector_angle_deg_estimate` | 导弹速度方向与目标视线夹角 |
| `shooter_distance_m` | 导弹距发射机的三维距离，发射机位置同样插值 |
| `kinematic_*_accel_*_estimate` | 用候选模拟时间对速度求导得到切向/法向运动学加速度；法向列除以 9.80665，以 g 表示，不能称为可用过载或原生气动载荷 |
| `angular_rate_*_candidate` | 对 b8 作有符号解读，标度 `30/65535`；单位、坐标系、符号尚未充分验证 |
| `raw_*` | 上游原字段、未解释量、flags、原始整数和原上游解码值，不覆盖、不平滑、不删除 |

主表以外的创建字段沿用上游的 `u12_2`、`u16_1` 等名称，避免把猜测写成已确认语义。`eid2` 在本回放为 0，未作为目标实体编号使用。

## 交叉检查与仍需验证的部分

初始姿态欧拉角换算出的四元数与创建记录 `u16_1` 一致，归一化四元数的绝对点积为 1.0。初始 b7 速度候选与创建向量 `u12_2` 的矢量差约 0.082 m/s；候选速度与按模拟时钟进行的位置差分速度，中位绝对差约 0.051 m/s。两种速度方向计算出的 AoA 中位差约 0.0082°。这些是内部一致性证据，还不是与游戏 Sensor 的独立对照。

`b5/48` 的首值与创建记录 `creation_time=0.5625` 秒一致；它与回放包创建事件的 0.548 秒不相同。`sf1` 与 tick 推算年龄的最大差约 0.02083 秒，即一帧 48 Hz 步长，所以分别保留。未把这些不同时间统一改写为同一种“飞行时间”。

风速为零是依据当前任务配置采用的假设，没有从导弹传感器直接读出相对气流。AoA、侧滑角、角速度候选和过载估计仍需游戏画面/已知工况核对。导引头被跳过的原始位流已保存，但锁定状态、当前导引模式及其内部目标坐标尚未完整解码。

原始 replay SHA256 仍为 `de6c9a98be11cb2f3b803fa5997fb96d14374644e8fa146bd35d7a63181b0b0c`。没有修改 replay 或任务/武器配置。最终日志仅有两条已知的二进制 include 性能提示，无其他 E 级记录。

## 复现

`build_backend.sh` 包含新增的字段保留、原始片段保留和安全标量绑定，全部以修补脚本形式应用。在 WSL 中把 `PYTHONPATH` 指向 `/root/wt-replay-trial-20260927/build/python/main`，运行 `inspect_extended.py --game ... --replay ... --output 新目录`，然后运行 `analyze_extended.py 输出目录 --target-name TargetAI_0`。

本次分析器明确限定为该单发回放；目标通过实体名称指定。不能把该目标选择直接用于多枚导弹、多目标回放。
