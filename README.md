# missile_sim

**1.2.0rc3**：基于游戏参数与响应观测，结合局部响应拟合和通用近似函数，提供气动、刚体动力学、发动机、雷达、目标截获链路与导引系统的近似仿真。

各支持型号使用共享响应函数，并保留各自的质量、几何、发动机、导引头和控制参数。原始 `.blk`、`.blkx` 和 profile 系数保持不动。公共 API 与本地 GUI 支持红外（IR）、主动雷达（ARH）和显式 PD 照射的半主动雷达（SARH）场景，包含目标截获、跟踪、失锁与重新截获，以及 INS、PN/PID 导引控制。局部响应拟合与通用近似共同构成模型，尚未逐弹完成完整游戏环境下的精度标定。

需要 Python 3.10 或以上。下载发布页 wheel 后安装，可在任意目录运行：

```console
python -m pip install aim120a_local_model-1.2.0rc3-py3-none-any.whl
python -m aim120_model.reconstructed --missile us_aim9m_sidewinder
python -m aim120_model.reconstructed --missile su_r_77_1
python -m aim120_model.reconstructed --missile us_aim7f_sparrow
python -m missile_gui.server
```

命令默认使用包内演示场景，其中 Su-30MKM 雷达签名、虚拟红外波段强度和独立 PD 照射平台均为明确的场景输入；红外波段值不是实测 Su-30 热源。使用 `--input scene.json --output result.json` 可运行自定义场景。Python 入口为 `simulate(load_profile(missile_id), scene)`，`load_profile` 来自 `aim120_model.reconstructed`。

红外默认要求显式 `target_ir`，每个值为传播前的方向签名 `I_k/C0`，`C0=24034.501953125` 为游戏内部标度，并非瓦。飞机名称本身不提供红外强度。可选 `infrared_source="legacy_fm_approximation"` 使用旧 FM 热源近似，但其强度单位映射未经标定，必须同时提供 `target_engine` 的 `throttle`、`afterburner` 和 `running`。红外观测仅提供角度与角速度，PN 使用导弹自身速度。

SARH 要求显式 `illuminator` 提供 PD 照射；其关闭、过期或失配会影响跟踪状态。回波强度采用双程信号近似，结果中的 `SARH_signal_model` 和 `bistatic.signal_model_status` 会标示这一模型性质。雷达目标可使用 `target_aircraft`、`target_airframe={length,span,aspects}` 或显式 `target_rcs_m2`；`birth_designation`、`virtual_datalink` 需要场景显式给出。场景可指定机体四元数姿态。

包内保留 **2.59.0.80 JSON 解码 `.blk`** 的 233 个资源变体、126 个实体，以及 122 份 2.59.0.27 转换 profile。可选旧 FM 热源近似使用冻结的 2.59.0.27 飞机/FM 字段。当前支持零扫描扇区、相对 Doppler 主动雷达、显式 PD 半主动照射，以及点热源红外、门控和 IRCCM。TVC、指令/驾束、CW/绝对 Doppler、雷达扫描/杂波频谱、SARH 箔条关联、IR 有限目标/遮挡/太阳背景尚不支持；PD 照射平台由独立场景输入提供。

旧模型可通过 `model_mode="legacy_surface"` 或 `model_mode="legacy_h2"` 显式选择；下文 1.1.x 说明适用于这些模式。

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
