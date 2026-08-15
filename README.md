# AIM-120A 本地候选模型（H1 / H2）

> 冻结公开候选版本：`v1.0.0`。核心运行时是 Python 3.10+，无第三方运行依赖。使用 `py -3 scripts/build_open_source_release.py` 可从白名单生成干净发布包；公开前仍需选择并加入正式许可证。

这是根据 `plan.md` 和 `plan2.md` 建立的独立、可审计的本地候选模型。它不修改 War Thunder 的任何现有文件；StatShark 只作为明确授权后记录的参考来源。

## 证据边界

- `data/reference_cases.json` 保存已有的 StatShark 页面结果；其中曲线终点和坐标轴读数均标记为近似值。
- `configs/aim120a_statshark.yaml` 保存从 StatShark 编辑界面读取的显式参数，以及明确标注为候选假设的低维气动解释。
- `src/aim120_model/` 是我们自己的 H1 实现。它的输出必须称为“本地候选模型结果”，不能称为 StatShark 复现。
- `data/raw/` 保存页面原始读数；当前包含一次用户授权的 20° 离轴 AIM-120A 页面悬浮提示读数。

## 当前实现

已实现：

- km/h、m/s、度、弧度和 g 的单位转换；
- 两段常推力/常质量流率发动机，并在 1.7 s、7.0 s 处精确切段；
- 标准大气、可切换参考面积解释、低维跨声速阻力函数和候选升力函数；
- 三维目标运动、恒定 g 转弯、12 m 近炸、撞地、80 s 寿命、80 km 航程和数值失败事件；
- 三维比例导引、时间增益插值、35 g 横向指令上限、60°/s 姿态角速度上限和最小 loft；
- “指令加速度 → PID → 舵面标度 → 力/力矩 → 实际过载”的一阶控制链；
- 四个与计划中 StatShark 消融定义等价的本地案例。

航向语义必须显式区分：历史本地案例默认使用 `target_course_reference="absolute_world"`；从 StatShark 网站输入转换的案例使用 `target_course_reference="statshark_relative_to_los"`，其中 `TargetCourse=0°` 表示目标沿初始视线朝向发射点（迎头），`180°` 表示远离发射点（尾追）。

尚未做：

- 参数拟合、复杂 Mach 表、完整六自由度滚转模型；
- 四个自定义消融模型的重新创建；当前页面只恢复了原始 `AIM-120A`。
- 将 UI 的 `Impact` 标签擅自解释为 `fuse` 或 `ground`；该字段在新读数中保留为未知。

## 运行

在本目录中执行：

```text
py -3 -m pytest -q
py -3 scripts/run_case.py --case baseline_full
py -3 scripts/run_case.py --all
py -3 scripts/compare_reference.py
```

运行结果只写入 `outputs/`。每个运行文件包含完整配置快照、案例定义、时间步、Git commit（当前安装目录不是 Git 仓库时为 `null`）、事件类型、终点摘要和完整时间序列。

`fit_parameters.py` 目前只输出“尚未授权/尚未满足分阶段辨识条件”的说明，不会偷偷拟合参数。

## 本地 GUI v1

Windows：

```powershell
.\run_gui.cmd
```

Linux / macOS：

```bash
./run_gui.sh
```

启动器会创建或复用 `.venv`、检查 GUI 可选依赖、只在 `127.0.0.1:8765` 启动服务并自动打开浏览器。GUI v1 使用 Python 标准库，不下载前端资源、不访问外部 API；按 `Ctrl+C` 关闭后会释放端口。端口冲突时可先关闭占用程序，或设置环境变量 `MISSILE_GUI_PORT` 后重试。

GUI 的范围刻意限制为：只读扫描 `missiles/*.json`、切换导弹、输入/导入场景、调用唯一入口 `aim120_model.simulate(missile_profile, scenario)`、查看轨迹与导出结果。结果区包含一个可旋转、缩放、平移和悬浮读数的三维轨迹场景，以及六张二维 P0 图；导弹、目标、发动机切段、燃尽和终止事件使用稳定的颜色与形状标记。网页只负责显示，不包含另一份计算公式，也不会保存或编辑导弹参数。模拟在内存中运行，不写入 `outputs/` 或任何旧实验目录。

当前库状态：

- 116 个 H2/常规舵面/固体火箭/PN 或 PN-loft profile：`Experimental`。全部通过 `profile_h2_universal_v2` 运行：公共 Python H2 模型层负责有效阻力、升力/Mach 形状、loft/control 和数值语义，所选 `missiles/*.json` 覆盖每枚导弹的质量、发动机、几何、气动、制导和 PID 数据。AIM-120A 冻结配置继续作为独立回归证据；公共映射必须在1 ms门禁内保持其冻结轨迹。
- `DE X4 RUHRSTAHL`、`AA.20`、`Fireflash`、`Starstreak`：`Unsupported physics`。前三者为指令制导，后者为驾束制导；不会静默退化为 PN。

导弹条目格式、支持状态和模型类型由 `missiles/*.json` 声明。非法 JSON 会在页面显示人话错误；详细异常仍保留在启动终端。GUI 端口默认为 `8765`。

## H2 修订

H2 保留 H1 输出，并把新结果隔离到 `outputs/h2/`。主要变化是：

- 用有效 `CdA0(M)` 和一维 `power_only` 7 s 锚点标定阻力标度；当前 `drag_scale=0.29949663`。
- 用相对气流的流向法向计算自然升力，单独记录轴向 G、法向 G、横向载荷、阻力功率和升力功率。
- 将物理法向 G 反馈、PID 请求、舵面权限和实际舵机响应分开；`guidance_no_control` 的零舵面权限仍保留控制器请求遥测。
- `.md/H2_REPORT.md` 记录了 H2.1–H2.4 验收、已有参考对比和未辨识边界。

运行 H2：

```text
py -3 scripts/fit_drag.py
py -3 scripts/run_h2.py --all
py -3 scripts/compare_h2.py
py -3 scripts/check_h2_convergence.py
py -3 scripts/check_h2_guidance.py
py -3 scripts/diagnose_h2_power_only.py
py -3 scripts/run_test_functions.py
```

## Missile Lab v1 profiles

导弹数据、场景和公共数值设置已经分开：

```text
missiles/*.json       # 单枚导弹的单位明确属性、模型族和来源
scenarios/*.json      # 发射与目标条件
config/defaults.json  # 时间步、重力和 smoke 公共设置
schemas/*.json        # profile JSON Schema
```

使用 `data/datamine/` 中的 sparse checkout 批处理导入非 TVC 空对空导弹：

```text
py -3 scripts/import_datamine_missiles.py
py -3 -m missile_lab validate-profiles
```

导入器按 `rocket.bulletName` 去重，保留 default、平台和吊舱变体的路径与 SHA-256；缺少 `bulletName` 的历史条目使用文件名作为可审计回退并保留原始路径。当前 datamine 版本扫描 230 个候选文件、124 个实体，生成 120 个非 TVC profile；MICA EM、R-73、R-73E 和 UK SRAAM 因 `thrustVectoringAngle` 记录在 `data/aam_non_tvc_manifest.json` 的排除清单中。所有 profile 都是 `experimental`，profile contract smoke 不是端到端飞行验证。GUI 对现有 H2 求解器支持的 116 个 profile 开放运行，全部使用 `profile_h2_universal_v2` 的公共 Python 模型层再叠加所选 JSON；X-4、AA.20、Fireflash、Starstreak 等 4 个非 PN 制导 profile 保持不可运行，绝不退化为 PN。

## Plan 8：观测层候选

H2 现在提供两个可切换的观测边界：`ideal_truth` 把目标真值复制成 `TrackSolution`，用于保持旧回归；`sensor_track` 对所有可运行 profile 开放。AIM-120A 继续使用其 datamine 映射的确定性主动雷达、逐 guidance tick 的 DL 和恒速 INS；没有 `sensor_model` 的其他 profile 使用明确标注的 `profile_kinematic_v1` fallback，只读取已有的 `seeker_type`、`lock_range_m` 和 `maximum_angular_rate_deg_s`，不复制 AIM-120A 的 Doppler、RCS、噪声或雷达常数。默认仍为 `ideal_truth`，因此开关不会改变旧路径。

雷达层目前只实现角度/角速率/距离/Doppler 门、标量 alpha-beta 门和 look-down Doppler/ground-clutter notch。`multipathEffect`、旁瓣、地形遮蔽、RCS/SNR 雷达方程、随机测量噪声、随机掉锁、SARH 和真实 IMU bias 均未实现；`50 m/s` notch 带是由 AIM-120A profile 的 `width/refWidth/signalWidthMin` 计算的本地候选，不是游戏隐藏公式的声明。

主动雷达失测时现在进入确定性的 `TRK -> INS+SRC`，短于 `prolongationTimeMax` 保留旧距离/Doppler 门，超过该时间重置搜索门但继续沿 INS 航迹搜索；重新通过搜索门后回到 `TRK`。通用 `profile_kinematic_v1` 也提供同样的锁定丢失/复锁切换，但不宣称拥有对应导弹的真实 seeker 方程。

表格目标轨迹使用 `time_s,x_m,y_m,z_m`，速度列可以成组提供以启用分段 cubic Hermite；查询不允许超出首尾时间。3/9 候选场景和运行入口：

```powershell
py -3 scripts\generate_notch_trajectory.py
py -3 scripts\run_radar_guidance_scenario.py `
  --missile us_aim_120a `
  --trajectory scenarios\trajectories\aim120_notch_39.csv `
  --observation-mode sensor_track `
  --output outputs\radar_guidance\aim120_notch_sensor_track.json
```

场景脚本拒绝覆盖已有结果；结果中的 `track_*`、`radar_*`、`seeker_state` 和 `datalink_connected` 字段只用于诊断，不反向修改动力学。该层仍是本地 candidate，不是 StatShark 或 War Thunder 求解器复现。
