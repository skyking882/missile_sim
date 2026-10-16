# missile_sim

[English](README.en.md)

一个用 Python 写的战争雷霆空空导弹飞行仿真器。

这个项目**基于战争雷霆现有空空导弹部分参数和游戏实验做的半经验半数据驱动动力学模型。**

---

## 能做什么

目前模型包括：

- 三维导弹与目标运动
- 5 自由度动力学：三维平动 + pitch / yaw
- 随 Mach 数变化的阻力
- 迎角相关的升力和附加阻力
- 四元数姿态积分
- 比例导引（PN）
- 中远程爬升制导
- 全发射角统一 PN 制导
- PID 自动驾驶仪
- 单面控制、恢复与阻尼响应
- 近炸引信和触地事件
- 48 Hz 控制与中点积分；历史 RK4 入口保留
- 本地网页 GUI
- CSV / JSON 遥测导出

暂时没有 roll。

---

## 快速开始

需要 Python 3.10 或以上。

仿真内核本身只使用 Python 标准库。

```bash
# 先跑回归测试，确认环境正常
python3 scripts/run_test_functions.py

# 打开 GUI
./run_gui.sh
```

Windows：

```bash
run_gui.cmd
```

也可以把 `python3` 换成：

```bash
py -3
```

GUI 只监听本机的 `127.0.0.1`，不需要联网。

启动后会按 [`config/missile_profile_catalog.json`](config/missile_profile_catalog.json) 选择活动参数集，目前为正式服 datamine **2.59.0.27**。旧版 2.57.1.107 参数集和实验读取路径仍保留在 `data/profile_sets/` 与 `missiles/`，供历史算例复现。选导弹、设置发射条件，然后直接运行即可。

GUI 使用统一单等效气动面框架，按各弹的质量、几何、力臂、发动机、PID、PN 与 Loft 字段运行导航和航迹层。活动库 122 项中支持 118 项 PN/PN+Loft；4 项指令/驾束制导显示为不支持。共享有效系数按固定名义工况和几何尺度迁移，是近似假设，尚未逐弹验证精度。低 Mach 阻力在原函数的正值极大点作连续常数延拓；零速从姿态定义方向，不添加伪速度。Loft 默认关闭，界面不提供实验对比或理想真值选项。旧 H2 接口以 `model_mode="legacy_h2"` 保留，冻结算例继续使用原入口。

结果包括：

- 可旋转的三维弹道
- 速度、过载、迎角等遥测曲线
- JSON 和 CSV 数据导出

程序只读取导弹参数，不会修改原始文件。

---

## 一个典型算例

例如：

```text
导弹：AIM-120A
高度：6.5 km
发射方速度：1200 km/h
目标速度：1200 km/h
初始距离：15 km
离轴角：38°
目标姿态：迎头
```

程序会一直积分到：

- 命中 / 近炸引信触发
- 脱靶
- 超时
- 撞地

为止。

历史 v1.0.0 的 AIM-120A 回归算例使用冻结的 2.57.1.107 参数；下面的结果不是用当前 2.59.0.27 参数重新验证的：

```text
12 m 近炸引信触发
t = 17.243 s
```

---

## 模型大概怎么工作

导弹采用一个简化的 **5 自由度刚体模型**。

平动部分计算导弹质心运动，转动部分计算 pitch 和 yaw，姿态使用四元数表示。

气动力模型是 reduced-order 的，而不是完整 CFD 或高维气动数据库：

```text
datamine 中的导弹参数
        +
统一的 Mach 阻力模型
        +
迎角相关横向气动力
        +
附加阻力
```

制导和控制是分开的：

```text
制导律
  ↓
期望横向加速度
  ↓
PID 自动驾驶仪
  ↓
舵面指令
  ↓
导弹动力学
```

全部发射角使用同一 PN 与普通并联 PID。单面的有效力和力矩按固定参考工况与各弹几何尺度迁移，不增加角度专用捕获控制器。

通用 surface 运行时先把请求投影到垂直于当前速度的平面，再限幅；PID 在同一平面比较实际加速度，最后分解到弹体系。动压补偿采用 `max(1, (baseIndSpeed / IAS_kmh)²)`：保留低速增益，不在高速额外削弱控制量。P/I/D、气动和力矩参数不因此改变。PN 倍率仍来自所选参数集；活动 2.59.0.27 的 R-77-1 为 N=6。

### 发射状态与雷达几何

`create_surface_missile` 可同时接收独立的 `launch_velocity_mps` 和 `launch_quaternion_xyzw`。载机速度、升力法向和迎角可通过 `launch_quaternion_xyzw_from_velocity_normal` 构造初始姿态。起控延迟只关闭控制输出，气动力、恢复/阻尼力矩和姿态仍继续积分；没有在此新增弹射速度或载机角速度继承假设。

导引头按自身配置分别限制 yaw/pitch，再以可达天线轴判断圆锥波束。载机雷达可用 `launcher_radar_geometry` 提供自己的轴限位、波束半角和姿态；旧的单一 gimbal 参数缺少波束信息，不自动附加 15°。

### 回放与完整 rate 输入

`simulate`、`simulate_surface` 和 `create_surface_missile` 均接受 `guidance_input(t, own_position, own_velocity)`，返回 `TrackSolution`。该入口直接替换观测输入，目标真值仍只供终止事件和遥测使用，不传给这个回调，也不再次对已估计的数据链做 αβ。

`aim120_model.angular_guidance` 提供可直接接入的 `AngularObservation`、`RecordedSeekerSample` 和 `RecordedSeekerHistory`。所有位置必须先换到同一原点，所有时间均相对离架时刻。`AngularObservation` 使用模型世界坐标和 `n_dot = n × rate`；PN 使用负的完整 rate，保留沿 LOS 分量。原生回放转换请使用 `AngularObservation.from_native`：direction、位置和速度为 `(x, y, -z)`，旋转向量 rate 为 `(-x, -y, z)`。

```python
from aim120_model.angular_guidance import AngularObservation

# angular 是同一时刻的方向、完整 rate、距离、闭合速度，以及回放导弹自身状态。
# from_native 的输入位置需先转换到与模型一致的原点。
angular = AngularObservation.from_native(**native_angular_state)
runtime = create_surface_missile(
    profile, **launch_options,
    guidance_input=lambda t, p, v: angular.track_for(t, p, v),
)
# 上面演示单条观测接口；整段回放应传入 RecordedSeekerHistory(samples)。
```

角输入先重建同一目标的绝对位置和速度，再按当前仿真导弹重新计算相对运动，同时保留轴向 rate 状态。因此仿真导弹偏离回放后，不会把“相对另一枚导弹的速度”直接当作当前输入。仅有飞机轨迹时仍走位置/速度 PN；本接口不假装能从轨迹唯一生成未知的轴向状态。

历史输入恢复经校验的 float32 物理帧时间，48 Hz 是当前物理/控制时钟，不是雷达测量频率。INS→TRK 之间先按旧 Ps/Vs 外推，到切换时刻才接管。默认边界为首条保存的 TRK；若另有截获时刻，可显式传入 `capture_times_s={first_saved_TRK_time: capture_time}` 和回放导弹的 `reference_motion(t)`。缺失首段用旧 INS 与首条 TRK 的目标速度状态约束，只在截获后补值，不用后两条 TRK 倒推。此补值使用新段首个已保存轴向 rate，是离线重建假设，不是已证实的 WT 初始化算法。同模式保存点之间插值，越过最后记录后按末状态外推；开头缺少状态会报错。

**回放飞机速度规则（仅记录）**：载机和目标飞机优先按真实位置历史及对应时间差分获得速度，避免使用可能受约 500 m/s 字段量程限制的速度；保留原始字段及速度来源。导弹自身速度不能因此一概替换。本库此次不修改回放导入器，这条规则由回放调用层负责。

历史 H2 方程和参数说明保留在：

[`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)

README 这里只介绍模型结构，不展开所有考据过程。

---

## 能信到什么程度？

目标是：

**得到物理上自洽、能够做工程比较的结果。**

不是：

**逐帧复刻游戏内部求解器。**

下面这些是旧版 2.57.1.107 参数和历史游戏实验得到的 sanity-check 锚点，不代表当前 2.59.0.27 参数集已在这些轨迹上重新验证。

| 算例 | 模型 | 游戏 / 外部结果 |
| --- | ---: | ---: |
| 35 km 迎头拦截 | 33.26 s | 32.77 s |
| 单帧过载对比 | 15.9 G | 15.0 G |
| AIM-120A，15 km / 38° | 17.243 s | 16.7 s |

在目前已经测过的条件下，模型的过载—迎角关系与回放数据也能比较好地对上。

这些结果说明模型抓住了一些主要动力学，但还不足以把它称为“已验证的游戏模型”。

---

## 哪些东西还是推测？

并不是所有东西都能从 datamine 里直接读出来。

目前比较大的不确定项包括：

- 标定区间之外的精确阻力律
- 不同导弹是否应该共用同一个升力比例系数
- 惯导阶段游戏内部真正使用的姿态控制逻辑
- 精确的转动阻尼
- 推力随高度 / 外界气压变化的规律

这些部分会明确标出来，不会假装成已知参数。

详细证据和被排除过的其他解释见：

[`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)

---

## 已知限制

目前没有建模：

- roll 动力学
- 燃料消耗导致的质心移动
- 精确转动惯量张量
- 舵机速率限制
- 完整非线性气动数据库

下面这些工况目前也没有经过系统验证：

- 长时间低速滑翔
- 极限尾追
- 目标持续高机动
- 完整远程爬升弹道

另外还有一个已经知道的 mismatch：

2.57 之后，游戏中的火箭发动机推力似乎会随外界气压变化。

例如一组 6.8 km 高度的回放反推出，续航段实际推力比 datamine 标称值高约 16%。

目前数据还不足以重建这个规律，所以暂时没有加入模型。

---

## 导弹支持情况

活动参数集（见 `config/missile_profile_catalog.json`）有 **122** 个导弹参数文件，其中 **118** 个接入当前运行时，**4** 个在库中保留但不可运行，因为其制导类型尚未支持。根目录 `missiles/` 保留 2.57.1.107 的历史实验输入。

四个不可运行 profile 使用指令制导或驾束制导，当前运行时会在库中保留并显示具体不可用原因。

程序遇到这些导弹时会明确返回：

```text
Unsupported physics
```

而不是偷偷套一个错误的比例导引模型。

---

## 回归测试

AIM-120A 保留了两组历史固定算例，用来检查其冻结的 v1.0.0 代码和配置有没有被意外改坏；它们使用 2.57.1.107 参数，不能视为 2.59.0.27 参数集的验证结果。

统一使用：

```text
dt = 0.02 s
```

| 条件 | 结果 | 时间 |
| --- | --- | ---: |
| 12 km，10° 离轴，迎头 | 12 m 引信触发 | 10.553 s |
| 15 km，38° 离轴，迎头 | 12 m 引信触发 | 17.243 s |

这两组算例使用的是冻结的 v1.0.0 配置与 2.57.1.107 参数：

```text
configs/aim120a_v1.json
configs/aim120a_v1_cases.json
```

v1.0.1 的新动力学走另一条运行路径。

这组历史基准与当前活动参数集分开保存；参数版本更新不代表这些轨迹已重新验证。

---

## 规避目标与逃脱窗口扫描

`simulate(profile, scenario, target_factory=...)` 可以替换场景目标，只在通用 surface 运行时可用，默认路径逐位不变；`legacy_h2` 传入时报错。`evasive_target.EvasiveTarget` 是一个闭环规避目标：`start_s` 之前与场景目标完全相同，之后按导弹最近一次的位置转向三九（水平速度垂直于导弹视线）或置尾（背离导弹），速度不变，水平转弯率受 `max_turn_g` 限制，竖直速度保持。导弹位置在每个 48 Hz 控制节拍内保持不变。

```bash
python3 scripts/sweep_escape_window.py --missile us_aim_120c_5 --ranges-km 15,20,30
```

对每个初始距离和规避方式扫描开始时刻，输出 CSV 到 `outputs/escape_window/`，并打印逃脱窗口：近炸引信以外的终止都算逃脱，窗口是网格分辨率下连续逃脱的开始时刻。非机动目标已被命中之后的开始时刻不再计算。

`simulate(..., early_miss_s=2.0)` 是扫描用的提速开关（同样 opt-in、只在通用 surface 运行时可用）：燃尽后的导弹若连续 `early_miss_s` 秒满足以下任一条件，就以 `early_miss` 结束——正在远离且比目标慢；正在远离且目标已在其后方；或按当前接近率在剩余时间内到不了。这不是证明：它假设接近率不会恢复（目标不回头、俯冲增速不大）。在两种导弹各约 250 个 FM 规避算例上，与跑满寿命的结果相比命中／逃脱分类零差异，200 m 以内的脱靶距离差 ≤ 0.2 m；被"到不了"截断的慢追击，`minimum_distance_m` 是截断时刻的值，会偏大。

这是运动学几何探针，不是飞机模型：没有能量损失、滚转建立时间、高度变化或箔条。窗口对转弯过载敏感，并继承导引头和气动模型的全部未验证项。CSV 里的 `radar_reject` 只是最后一次拒绝原因，不等于逃脱的唯一原因。

### 箔条（opt-in，证据等级 D）

`chaff.chaffing_factory(inner_factory, ChaffSpec(rcs_ratio=...), ChaffProgram(start_s, interval_s, per_salvo, total))` 给任意外部目标挂上箔条，只在 `sensor_track`（`SensorTrackProvider`）下生效。每包箔条在抛出时刻取目标真值位置，加 70 m/s 向下抛射速度，按 0.1 s 时间常数停在静止空气中；RCS 按 `signature.radar.profile = [0.05, 0.1, 12, 15]` 解释为 0.05–0.1 s 上升、12 s 前平台、15 s 衰减到 0（= `timeLife`）。这些字段语义都是假设。箔条峰值 RCS 与飞机 RCS 之比 `rcs_ratio` 无法从 `multiplier0` 推出，必须由调用方给定。

导引头传入 `decoys` 时走多反射体路径：飞机和每包箔条各自通过单目标路径的全部门限（含地杂波 notch，所以下视时近乎静止的箔条和 notch 中的飞机一起被滤掉）；与 α-β 预测最近的候选作为锚点，距离差 ≤ `distance.width`、多普勒差 ≤ `dopplerSpeed.width`、夹角 ≤ 接收天线 `angleHalfSens` 的反射体视为不可分辨，RCS/R⁴ 最大者被测量。同单元多包箔条不叠加功率，取单包最大。零包箔条时与单目标路径逐样本一致（有测试）。摘要增加 `first_decoy_track_time_s` 和 `decoy_track_time_s`。

### 几何主瓣杂波（opt-in，证据等级 D）

默认的地杂波 notch 条件是"反射体低于导弹"（`look_down`）。导弹末段会把高度追到与目标相差零点几米，同高度交战时这个布尔量来回翻转，notch 随之开关，结果像抛硬币。`simulate(..., clutter_model="geometric_mainlobe")` 改为平地几何：波束下沿（视线俯角 + 接收天线 `angleHalfSens`）触地的斜距 `h / sin(俯角 + 半角)` 不超过反射体距离加一个距离单元时，才认为反射体背后是地面杂波。8 km 同高打 10 km 外目标时下沿约 31 km 才触地，不再 notch；2 km 高度约 7.7 km 触地，notch 成立。只改变杂波判定，角度门、角度变化率门和视线角速度门不变；单目标与多反射体两条路径都适用。`look_down` 显式传入与默认逐样本一致（有测试）。

`clutter_model="look_down_angle"`（opt-in，证据等级 C/D）：视线俯角不小于 `clutter_min_depression_deg`（默认 2°）才认为反射体背后有地杂波。依据是玩家经验——下视才能 notch，同高度不行（地形不建模，平地）。几何主瓣模型会让高空同高度完全无法 notch，与这一经验不符；而默认 `look_down` 在同高度会随厘米级高度差翻转。门槛角需要游戏内标定：同一距离下比导弹低 200 m／500 m／1 km 各试一次 notch。

`cw_on_clear_beam=True`（opt-in，证据等级 C）：目标在三九（地面径向速度落在 notch 窗口内）且背后没有地杂波（上视或同高度）时，导引头离开 PD，箔条只要与锚点处于同一距离单元和波束内就直接夺取跟踪，不再比较 RCS；没有箔条时照常跟踪目标。依据是玩家报告"上视进三九会进入连续波，一发箔条就脱锁；不放箔条照样被打中"。只作用于多反射体（箔条）路径；摘要增加 `cw_mode_time_s`。

---

## 仓库结构

```text
src/aim120_model/   仿真内核
src/missile_gui/    本地网页 GUI
missiles/           冻结的 2.57.1.107 历史实验参数
data/profile_sets/  由目录清单选择的版本化参数集
scenarios/          发射与目标场景
config/             运行时默认配置
configs/            冻结配置
schemas/            导弹 JSON 校验规则
scripts/            运行、导入和诊断工具
tests/              回归测试
```

更详细的技术说明：

- [`docs/MODEL_STATUS_20261016.md`](docs/MODEL_STATUS_20261016.md)：当前模型、并行线程进展、验证边界和未解决问题
- [`docs/WT_RATE_RECONSTRUCTION_PROGRESS_20261016.md`](docs/WT_RATE_RECONSTRUCTION_PROGRESS_20261016.md)：#118/#148 完整 rate、坐标变换、普通更新与捕获策略审计
- [`docs/UPDATE_V1.0.1.md`](docs/UPDATE_V1.0.1.md)：完整方程、参数解释和证据链
- [`V1_FREEZE.md`](V1_FREEZE.md)：历史 v1.0.0 冻结记录（不是当前 2.59 模型卡）

原始证据放在：

```text
data/raw/
```

生成结果放在：

```text
outputs/
dist/
```

这些不会提交到 Git。

---

## License

目前还没有确定正式许可证。

在仓库加入明确的 `LICENSE` 文件之前，请不要把它视为已经授权公开分发的开源项目。
