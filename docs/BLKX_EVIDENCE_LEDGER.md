# blkx 字段语义证据台账

建立于 2026-09-12。规则:每个字段的语义主张必须挂证据等级与出处;
**任何进入动力学模型的映射,等级低于"官方文档"的必须在 profile
parameter_sources 里保持 `kind: assumed` 并在此登记**。

证据等级:
- **A 官方铁证** = 官方文字 × datamine diff 数字逐位对应(双源交叉)
- **B 官方文档** = CDK wiki / dev blog 文字描述(单源,可能过时)
- **C 识别证据** = 控制器无关的回放识别结果(P0 类)
- **D 假设** = 无任何游戏侧证据,纯模型侧猜测

## 系统识别范围与状态

范围为飞行、推进、控制、制导、导引头及近炸事件参数。数值读取、语义
识别和运行时公式分别记录：原始数值可核对，不代表单位、内部参考量和
求解公式都已确认。完整 JSON 路径与版本组成字段身份；同名字段不能合并。

| 字段族 | 已有证据能支持的内容 | 尚需区分的内容 |
|---|---|---|
| `rocket.mass/massEnd/caliber/length` | 原始质量与几何量；有官方命名支持 | 内部面积、惯量、质心与长度之间的计算关系 |
| `force*/timeFire*/propulsion` | 分段推进数值与燃烧时长 | 分段质量节点、点火延迟及压力修正的完整执行顺序 |
| `CxK/dragCx/CxAoA` | 阻力相关参数及各自来源路径 | 不同字段作用的求解层、参考面积、Mach 形状和迎角附加阻力 |
| `finsAoaHor/Ver` | 配平迎角限制有 blog×diff 数值配对证据 | 分轴/合成限幅、动态越限、显示角标度 |
| `CyK/CyMaxAoA/wingAreaMult` | 显式数值及升力相关角色 | 引擎默认斜率、参考面积、乘子位置、封顶的先后次序 |
| `CyMachMult.CyMachMult*` | 完整 Mach/乘子节点 | 插值、端点延拓以及与 CyMaxAoA 的组合顺序 |
| `distFromCmToStab/WdK` | 稳定性/阻尼相关候选 | 单位、轴向分量、速度缩放、状态阶数；联合变参不能单独识别 |
| `finsLatAccel` | 原始数值及与控制结构的共现 | 单位、参考动压/质量、进入指令限幅还是力通道 |
| `guidanceAutopilot.accelControl* / pid*` | 平铺 PID 或带时间的 PID 组 | 误差/输出单位、微分滤波、积分限幅语义、时间表切换/插值 |
| `baseIndSpeed/propNavMult/reqAccelMax/timeToGain*/timeToHitToGain*` | 参数与表的数值、独立路径 | 指令计算顺序、速度基准、时间基准、捕获到跟踪的衔接 |
| `loft*` | 仰角及增益字段 | 进入/退出条件及与 PN、目标高度的组合 |
| `radarSeeker/opticalSeeker` 与距离/多普勒/角度门 | 可枚举原始门限、滤波及探测参数 | 门限执行顺序、滤波状态、丢锁/重捕获行为与可观测条件 |
| `proximityFuse.timeOut/radius` | 近炸事件参数，有完整独立路径 | 启用边界是否含端点及具体游戏碰撞执行时序 |
| `endSpeed/machMax/loadFactorMax/maxDistance/timeLife` | 性能/事件相关原始值 | 哪些为显示、哪些为事件终止、哪些为物理硬裁剪 |

2.58.0.36 对比基准为本地 2.57.1.107，共 120 个既有非 TVC AAM
profile（含既有只读类型；不代表全部游戏武器）。27 个 trim-angle 字段
变化、28 个 `distFromCmToStab` 变化、14 个新增显式 `CyK` 与
`CyMachMult`、6 个改用 `pid0/pid1`。120 个 `loadFactorMax` 均未变化。
`endSpeed` 在 86 个条目变化；最大速度参数变化不能直接换算成机动性提升。
原始版本：[2.57.1.107 → 2.58.0.36](https://github.com/gszabi99/War-Thunder-Datamine/compare/3f20645862c538b46c94e3f60ee30da9dfacd81a...829400da4e23a5d975b1ca278987b203bd441703)。

识别顺序是先固定观测量和版本，再用同条件游戏记录区分单个候选关系。
只可辨识的参数组合保留为组合：例如升力斜率×参考面积，以及一阶响应
时间常数。没有独立激励/观测时，不拆成唯一的气动力、舵偏和力矩参数。
一阶自由飞行候选的初始迎角 11°、0.5 s 制导延迟、
捕获/释放常数和等高垂直配平仍为
明确假设；它不提供 `WdK`、PID 或内部转动惯量的语义证据。

---

## A 级(铁证)

### finsAoaHor / finsAoaVer = 最大配平迎角(弹体 α 限幅)
- Dev blog [news/10111]:AIM-54 "**Maximum trim angle of attack** doubled
  from 12.5 to 25 degrees";"maximum G-force increased from 17G to 22G"。
- Datamine:us_aim_54a.blkx 在 **2.56.0.45**(commit ddbaafb0,2026-06-16)
  `finsAoaHor` 0.217616 rad(12.469°)→ 0.436332 rad(25.000°),
  `loadFactorMax` 17→22 同一提交同步变化。数字与 blog 逐位对应。
- **不是**舵面偏角:CDK wiki 写的 "maximum fin deflection angle" 与带数字
  的 dev blog 矛盾,以 blog 为准(CDK 文档降为过时)。
- 本地 2.57 值:R-77/R-77-1 = 0.460812 rad = **26.40°**;
  PL-12 = 0.375092 rad = **21.49°**。
- 回放交叉:PL-12 慢速 α 峰 15.2° < 21.5°,全程未触线,无平顶 ✓;
  R-77 族肩部 23.7–23.9° vs 限幅 26.4°,**差 2.5°**。
  肩部与可达配平迎角、自动驾驶仪限幅的关系尚需独立观测判别。
- 项目影响:profile `fin_aoa_limit_rad` 及 rebuild 内环把该字段当
  **舵偏限幅**用,语义错误,内环结构需按"α 限幅"重构。
- 全仓库分布(2026-09-12,rocketguns 655 文件普查):
  - 出现在 433 个文件:AAM 229/231、制导 AGM/ATGM ~185、MCLOS 19
    (operated=true,含 1944 年 X-4);**纯无控弹 0 个**(59 无控火箭、
    flare/chaff/炮弹全无)→ 只随控制回路出现,是控制/配平参数。
  - 取值按机动性单调:MCLOS 7–8.6°、R-60 10.3°、AIM-9L 14.3°、
    AIM-120A 15.4°、AIM-7F 17.2°、Magic-2 20.4°、PL-12 21.5°、R-77 26.4°。
  - `finsAoaHor == finsAoaVer` 全库 433/433 无一例外(通道从未分开调)。
  - 共现:带 finsAoa 的弹 **433/433 同时带 distFromCmToStab 与
    wingAreaMult**;finsLatAccel 仅 233/433;CyK 仅 8/433。
  - 命名读法(推断,非铁证):字段自最老 MCLOS 弹起存在,命名沿袭十年;
    "fins AoA" = 舵面把弹体保持到的迎角——在"不积分旋转、直接伺服 α"
    的 FM 里,舵偏限幅与配平 α 限幅是同一旋钮。全库只有这一个混合命名
    字段而无独立 δ 限幅,本身是 α 伺服语义的旁证(待 P1 判别)。
  - 开放问题:Hor/Ver 为通道限幅,对角线 total α 是否可达 √2×?判不动。

## B 级(官方文档,单源)

| 字段 | 官方语义 | 出处 | 本地值(R-77-1/PL-12) |
|---|---|---|---|
| `loadFactorMax` | 导弹最大 G(stat card) | CDK wiki + blog 17→22 配对(半 A 级) | 50 / 38 |
| `reqAccelMax` | 允许指令的最大 G | CDK wiki | 50 / 38 |
| `rocket.proximityFuse.timeOut` | 近炸引信延迟启用参数；不能作为制导直飞时长 | R-77-1 / PL-12 原始嵌套路径，2.57 与 2.58 核对 | 1.7 s / 1.7 s |
| `rocket.guidance.guidanceAutopilot.timeOut` | 自动驾驶仪时间参数；具体行为与基准时间仍待识别 | 必须与引信同名字段分开；R-77-1 / PL-12 用 timeToGain 表且无本字段 | 两者缺失；AIM-120A 为 0.6 s |
| `CxK` | 飞行阻力系数(浅) | CDK wiki | 1.65 / 1.6 |
| `mass`/`massEnd` | 发射前/燃尽后质量 | CDK wiki | — |
| `force`/`timeFire` | 推力/燃烧时长 | CDK wiki | — |
| `finsAoaHor/Ver` 上限之外的 `thrustVectoringAngle` | 推矢偏角 | CDK wiki | — |
| `CyMachMult` | 升力随 Mach 修正的显式表；插值/端点延拓待识别 | blog "lift coefficient…Mach" + 2.58 原始表 | 本地 120 个条目中 14 个新增；R-77 / PL-12 / AIM-120A 无该表 |
| `extPressureToThrustMult` | 外压推力修正；确切插值及外推待识别 | blog "pressure-dependent thrust" + 原始数值对 | 2.57 的 AIM-54 已存在；2.58 在现有条目中新增 6 个 |

完整字段路径是识别主键。R-77-1 / PL-12 的 1.7 s 来自
`rocket.proximityFuse.timeOut`；自动驾驶仪实际另有
`timeToGain0=[0.3,0]`、`timeToGain1=[0.31,1]`。
因此现有原始文件不能支持“发射自带 1.7 s 无制导窗口”，也不能把
回放中的 1.9–2.1 s 事件归因于这个引信参数。

## C 级(识别证据,2026-09-11 P0)

- 2.57 配平升力线性:升力面积 0.60–0.64 m²/rad(参考面积约定无关),
  α≤23.9° **无升力封顶** → 引擎默认升力律
  在该区间无 CyMaxAoA 型 cap。
- 2.57 升力**无 Mach 依赖**(M 1.1–3.5)→ 与 CyMachMult 为 2.58 新
  能力互证。
- 显示 G 语义 = 垂直速度、含推力(v1.0.2 标定 + P0 内部一致性)。

## D 级(纯假设——当前模型在用、零证据)

| 字段 | 模型当前用法 | 风险 |
|---|---|---|
| `finsLatAccel` | 舵面横向力/加速度能力 | 单位、参考点全猜 |
| `distFromCmToStab` | 尾舵力臂 | 2.58 大改暗示进旋转,但语义无证据 |
| `wingAreaMult` | 升力面积乘子 | 乘哪个面积未知;P0 判不动 1.4 vs 1.45 |
| `CyK`(显式弹) | 升力斜率 /rad | **内部参考面积未知**,无法与识别值换算 |
| `CyMaxAoA` | 升力系数上限 | 同上;break 角 = CyMaxAoA/CyK 是约定无关预言 |
| PID 组 / `propNavMult` / `baseInd*` | 制导内环 | 结构全猜 |

### 自定义角速度阻尼记录（2026-09-17）

当前独立力矩候选使用了**我们自行定义的附加阻尼**，证据等级为 **D：模型假设**：

\[
M_{\rm extra}=-\frac{\bar q}{V}D_\omega\omega.
\]

- 代码参数为 `rate_damping_derivative_m4`，单位为 m⁴；\(\bar q\) 为瞬时动压，\(V\) 为瞬时速度，\(\omega\) 为弹体角速度。它产生与角速度反向的力矩。
- **公式和数值均未确认为 WT 引擎实现。** 它不是从 `WdK`、`finsLatAccel` 或其他拆包字段换算得到的；不能称为“游戏阻尼”或“已经识别出的气动导数”。
- `scripts/check_r77_1_moment_response.py` 的 `independent_moments` 诊断候选取 **D_omega = 12 m⁴**，同时使用原控制力矩斜率的 4 倍、原恢复力矩斜率的 2 倍。该组合在同一组 R-77-1 初段回放上筛选，不属于独立验证，也不能把改善全部归因于阻尼一项。
- “只增加阻尼”的对照取 16 m⁴；运行入口的默认值仍为 **0**，没有把 12 m⁴ 提升为默认参数。原有局部旋转来流阻尼仍保留，默认值为 0 仅表示不加这一项。
- 对比冻结于 **2.57.1.107**，原始 PID 不变。大幅反复振荡减弱，但捕获切换、迎角和过载曲线仍有不匹配，完整模型尚未通过实测检验。
- 实现：`src/aim120_model/alpha_servo.py`；诊断结果：`outputs/r77_1_independent_moments.json` 和同名 PNG/TSV。

## 地层学普查与官方消融(2026-09-12,rocketguns 655 文件)

### 世代分层(字段随 FM 世代累加、从不改名)
- **gen0 MCLOS**(X-4/AS.30/Bullpup,游戏最老实装):
  plant 核心 = CxK, dragCx, wingAreaMult, distFromCmToStab, finsAoaHor/Ver,
  WdK, mass/massEnd/force/timeFire(X-4 已有 force1/timeFire1 两段推进);
  手操指令层 = operated, controlSensitivity, aimSensitivity, maxAxisValues,
  maxLateralAcceleration, rollStabilization, autoLeveling。
  **无 PID、无 propNavMult、无 finsLatAccel、无 G 限幅、无 timeOut。**
  AS.30 与 Bullpup 参数逐位相同(CyK=5.0, CyMaxAoA=1.8, wam=1.2,
  dist=0.02, finsAoa=0.15, WdK=0.1, ctrlSens=0.2)→ 模板默认值而非逐弹
  标定;X-4 **无 CyK**(吃引擎默认斜率,可用于跨弹体检验默认升力律)。
- **gen1 自动驾驶仪世代**(AIM-9B 起):+ guidanceAutopilot,
  accelControlProp/Intg/Diff(PID), propNavMult, baseIndSpeed, rateMax,
  finsLatAccel, loadFactorMax, reqAccelMax, timeOut。
- **2.58 本次条目变化**:+ 部分型号的 pid0/pid1 时间表、CyMachMult；
  extPressureToThrustMult 扩展至更多型号，该字段在 2.57 已存在。

### 字段归层证据
- `distFromCmToStab`:**213/213 纯无控火箭也携带** → plant 被动稳定
  (风标)参数,与制导无关。
- `WdK`:553 文件;默认 0.1;chaff/flare = 0(自由翻滚);**2.0–2.5 只
  出现在慢速滑翔炸弹**(Hs 293、Ki-148、Grom-1、AGM-123——低速最易
  "海豚跳"的武器)→ 语义候选:**角运动阻尼增益**。rebuild 缺失阻尼的
  头号游戏侧对应物([[rebuild-oscillation-root-cause]] 候选可具名化)。
- `wingAreaMult`:无控弹 **0/213**,带控弹 433/433 → 机动/升力世代字段,
  不是通用气动;"乘哪个面积"仍未知。
- `finsLatAccel`:229/233 与 guidanceAutopilot 共现,4 个例外全是
  **驾束制导**(Kh-66、9M127 Vikhr×2、Fireflash)→ 属于指令→机动侧
  的能力参数;rebuild 把它当 plant 舵面力使用,证据不支持(D 级警告)。
- **全库不存在任何转动惯量字段** → 旋转动力学要么由 mass/length 内部
  推导,要么是无惯量的一阶 α 松弛。一阶系统永不过冲 → P1 直接判别。

### 官方消融词典(Ki-148 `_event_broken_*` 家族——Gaijin 自己的对照实验)
| 故障变体 | 改动字段 | 语义结论 |
|---|---|---|
| broken_guidance | aimSensitivity 22.1→0、controlSensitivity 0.25→0、maxAxisValues 1→0 | 三者 = 指令通路增益 |
| broken_servo_controll | aimSensitivity → **−22.1**、**distFromCmToStab 0.5→35**、rollStabilization→false | 多个控制/稳定字段同时变化，是联合干预；支持稳定性相关候选，不能单独证明 dist 的力臂单位或恢复刚度公式 |
| broken_air_ballon / pipe | CxK 1.25→2.25、force 2510→500、timeFire 75→10 | CxK=阻力、force/timeFire=推进 |
| broken_fuel_leak | force→4350(2 s)+ force1=100、CxK→3.75 | 两段推进语义 |
| broken_engine | timeLife 90→12 | timeLife = 自毁时限 |
| broken_warhead | 仅战斗部/碰撞特效 | 战斗部与 FM 解耦 |

### gen0 模型五场景验证(2026-09-12,scripts/gen0_alpha_servo_model.py)
α 伺服 plant(准静态→一阶 τ=C/(dist·q̄))+ 捕获-保持-释放制导,
**只拟合两个标量**(C=1150 于 r77-1;PL-12 τ_c=0.70 于 pl12level;
R-77 族 τ_c=0.30、washout 0.95 s、ε_exit=15° 沿用 8 月识别值),
其余输入全部来自 datamine/P0。轴向用回放实测速度强制(阻力通道独立
存疑,见下)。结果(α RMSE°/G RMSE):r77-1 2.7/6.3(标定)、
30degr77 3.0/3.6(纯预测)、30degpl12slow 2.7/2.8(纯预测)、
pl12level 1.0/2.0(τ_c 标定)、30degpl12 3.2/6.2(预测;该场目标
实际机动过,比较受限)。五场全部命中(CPA 0.1–1.4 m),A7 拦截
7.18–7.68 s 对游戏 7.36 s。τ@标定 q̄=0.185 s 与瞬态帧独立估计 0.21 s
吻合;发射瞬态跨弹缩放 τ_PL12/τ_R77 观测 1.55 vs 1/(dist·q̄) 预测 1.64。
这些是给定实测速度的条件比较，不能证明独立自由飞行会命中或复现速度。
升力斜率来自历史 P0 回放识别，两个响应/捕获常数也使用过其中两组数据；
因此不能把其余曲线称为完全独立的验证集。一阶有效响应是候选结构，
不是游戏内部只有一阶状态的证据。
【2026-09-12 用户更正,重要】"游戏不会振"的说法被用户驳回且成立:
WdK 的存在及其在慢速滑翔炸弹上的较大取值支持阻尼相关解释，但仅凭
参数共现不能证明引擎状态阶数，或证明 k∝dist·q̄、c∝WdK 的具体关系。
带阻尼二阶旋转及其过阻尼慢模态是保留候选；一阶模型描述的是所测区域
的有效响应。只有在这些关系成立时，增大刚度而不增加阻尼才预言阻尼比
降低；不能把该条件推论当作 2.58 实测结论。
未闭合残差:①终末段 G 隆起(全场景 5.5–7 s,候选:末段 PN 增益/
导引头几何/目标运动偏差);②释放混合的 3 s 伪凹陷(主管线 polar
release 已有正确实现);③自由跑速度通道 RMSE 130–320 km/h =
阻力模型独立问题;④ V0(准静态)与 V2/V3 在全程 RMSE 上区分度弱,
τ 的证据主要在瞬态帧与跨弹缩放。

### gen0 隐含模型形状(预注册,待 P1 判别)
指令(controlSensitivity/aimSensitivity 增益,MCLOS)或自动驾驶仪指令
(gen1)→ 配平 α 目标(≤ finsAoa)→ **α 松弛动力学**
(刚度 ∝ distFromCmToStab·q̄,阻尼 = WdK,惯量:无字段)→
升力 = min(CyK·α, CyMaxAoA)·q̄·S_int·wingAreaMult → G(≤ 指令 G 限幅)。
候选对:① 二阶(惯量内部推导)vs ② 一阶 α 伺服。判别 = 阶跃响应有无过冲。

### MCLOS 试验台(P1 升级版)
AS.30/Bullpup = 已知答案(CyK=5.0/CyMaxAoA=1.8,break 20.63° 约定无关)
+ **玩家手操 = 可控开环激励**(阶跃/倍频舵指令)+ 慢速大弹响应肉眼可见。
一次拍摄可同时判:引擎内部参考面积 S_int、α 响应阶数、配平律;
对照组 = WdK=2.0 的 Hs 293/Ki-148 同实验(判 WdK 阻尼语义);
X-4 无 CyK,可反演引擎默认升力斜率与 AAM 隐式斜率交叉验证。

## 证据机器(获取新证据的五条流水线)

1. **blog×diff 配对(可批量)**:官方 changelog/dev blog 带数字的表述 ×
   gszabi99 逐补丁 diff,数字对上即 A 级。finsAoa 即首例。
2. **已知答案回放**:2.57 live 有 11 枚 MCLOS 弹显式 CyK=5.0 +
   CyMaxAoA=1.8(Bullpup/AS-30/Kh-23M…)。预言:其识别升力线在
   **α=1.8/5.0=20.63°** 处折平(break 角约定无关!)。拍一组回放用
   P0 管道反演:折角对上 → CyK/CyMaxAoA 语义 + 引擎内部参考面积
   S_int=(识别升力面积斜率)/5.0 一次钉死。**当前最高价值实验。**
3. **2.57→2.58 同弹自然实验**:dev 服重拍同场景,识别量变化比 ×
   字段变化比(wingAreaMult 1.45→1.5、CxK 1.65→1.4、CyMaxAoA 出现)。
4. **CDK 文档**:浅层字段兜底,与 blog 冲突时以带数字的 blog 为准。
5. **P0 型控制器无关识别**:回放五列(V/M/G/α/t)+ 燃烧史,
   高度由 V/M 反演。脚本 scripts/identify_lift_from_replays.py。

### 箔条与导引头分辨单元(2026-10-03,`chaff.py`、`radar_seeker._update_with_decoys`,opt-in)

| 字段(2.59.0.27) | 数值 | 模型中的语义 | 等级 |
|---|---|---|---|
| `countermeasure_*/…/rocket.timeLife` | 15 | 箔条存在时长 | D(数值 A,语义同 Ki-148 `timeLife`=自毁时限类推) |
| `…/rocket.startSpeed` | 70 | 相对载机的抛射速度,方向取竖直向下 | D(方向纯假设) |
| `…/rocket.CxK`、`mass` | 100、0.01 | 0.1 s 时间常数停于静止空气 | D |
| `…/signature.radar.profile` | [0.05, 0.1, 12, 15] | RCS 包络时间节点:上升、平台、衰减至 0 | D |
| `…/signature.radar.multiplier0` | 0.05(MAW 弹 0.2) | 未使用;缺少飞机自身签名基准,改由 `rcs_ratio` 输入 | — |
| `radarSeeker.distance.width` | 300 | 距离分辨单元 | D |
| `radarSeeker.dopplerSpeed.width` | 20 | 多普勒分辨单元(与 notch 半宽中的同一字段) | D |
| `radarSeeker.receiver.antenna.angleHalfSens` | 15 | 不可分辨的波束半角 | D |

| `radarSeeker.receiver.antenna.angleHalfSens` 用于杂波 | 15 | opt-in `geometric_mainlobe`:波束下沿触地斜距 ≤ 反射体距离 + `distance.width` 时有主瓣杂波(平地) | D |

| (无对应字段)下视门槛角 | 2°(假设) | opt-in `look_down_angle`:视线俯角 ≥ 门槛才有地杂波 | C/D(玩家报告:下视才能 notch、同高度不行;门槛值未标定) |

| (无对应字段)上视三九模式切换 | — | opt-in `cw_on_clear_beam`:无地杂波且三九时离开 PD,同距离/波束单元内箔条直接夺取 | C(玩家报告:上视进三九转连续波,一发箔条脱锁;无箔条仍被命中) |

全部为 D:多反射体选择规则(锚点 + 同单元 RCS/R⁴ 最大者)、箔条受地杂波 notch 同样过滤、同单元不叠加功率、主瓣杂波几何(平地、半功率半角、无旁瓣与距离折叠)都没有游戏侧证据。默认 `look_down` 的同高度开关伪影见 README。需要的判别实验:同一几何下有/无箔条、上视/下视各一组回放。

### 地面多径(2026-10-04,`radar_seeker.enable_multipath`,opt-in `multipath_gain`)

| 字段 | 数值 | 模型中的语义 | 等级 |
|---|---|---|---|
| `radarSeeker.multipathEffect` 横轴 | [0, 1, 60, 0](几乎全部雷达 AAM;LD-10、AGM-122 为 [0, 1, 0.5, 0];AIM-54 系列无此字段) | (离地高度 m, 强度) 节点,线性插值、端点保持;60 m 以上无多径 | A:2.37 "Seek & Destroy" 官方更新日志:机载雷达与雷达导引头导弹的多径最大高度"由 100 m 降到离地 60 m"。datamine 逐位对应:APG-68(v5) 三个 `transivers.*.multipathEffect` 在 2.36.0.26 为 [0,1,100,0]、2.37.0.9 为 [0,1,60,0];导弹字段在 2.36.0.16 不存在,2.36.0.26 为 [0,1,50,0],2.37.0.9 为 [0,1,60,0](PL-12、AIM-120C-5、AIM-7F、R-77 相同) |
| `radarSeeker.multipathEffect` 纵轴 | 1 → 0 | 作为测量点向镜像移动比例的乘子 | D(CDK wiki 只写 Array、"Controls radar multipath effects near terrain") |
| (无对应字段)`multipath_gain` | 0–1,扫描 0.5 / 1 | 测量点 = 目标与其平地镜像(-高度)连线上 gain×强度 处;0.5 × 强度 1 = 目标正下方地面点。门限、距离、多普勒仍用真目标 | D(玩家报告:导弹"从目标下方穿过/扎地";增益未标定) |

全部为 D:确定性偏置(无噪声、无相位起伏、不造成丢锁);只作用于导弹导引头,载机雷达与数据链的多径未建模(载机雷达有同形状的表,低空目标可能先丢载机锁定);平地 y=0;近炸引信只对目标(blkx `proximityFuse.hasUnitsOnlyMode=true`,对地近炸未建模)。判别实验:海面上目标在 20/30/40 m 定高直飞,高空分别以陡、缓末段俯冲角发射;回放读弹着点相对目标的前后距离,可直接反推测量点高度 h(1−2k),从而得到 k(h),并检验"陡俯冲脱靶量按 cos(俯冲角) 缩小"的几何预测。

**增益标定点(2026-10-05,C,单次玩家回忆)**:本机约 7000 m、1.5 马赫,AIM-120 系,目标贴地(高度回忆约 20 m,不确定),约 17 km 发射命中。模型(AIM-120C-5/120D,迎头,20 m):增益 0.5 最远 17 km,增益 1 只到 5 km。该命中排除增益 1,支持增益 ≤ 约 0.5;不能区分 0.5 与更小的值。数据链相关(C,玩家报告):目标不 notch 本机雷达时,导引头未锁上、仅靠数据链也能命中贴地目标。`inertialNavigationDriftSpeed`:AIM-120D 为 0,120C-5/PL-12/PL-15/R-77-1 为 2(数值 A;"0 = GNSS 辅助"的读法为 C,玩家说法)。

### 载机雷达数据链与导引头锁定(2026-10-05,`SensorTrackProvider.enable_launcher_radar`,opt-in `launcher_radar_gimbal_deg`、`require_seeker_lock`)

| 字段 | 数值 | 模型中的语义 | 等级 |
|---|---|---|---|
| (无对应字段)载机雷达扫描半角 | 60°(假设) | 目标离开载机速度方向的该锥角,数据链永久断开 | D(玩家报告:载机雷达丢锁则数据链断;各机雷达实际扫描/跟踪角未从 blkx 读取) |
| (无对应字段)载机雷达 notch | 沿用导引头的杂波规则与 notch 半宽 | 以载机位置判定下视杂波、目标对地径向速度落入 notch 半宽时载机丢锁,数据链永久断开 | C/D(玩家报告:目标在下方三九时载机雷达丢锁、数据链断开;载机雷达自身的多普勒宽度、杂波门槛未标定) |
| (无对应字段)导引头必须锁定过 | — | 导引头从未形成雷达跟踪时近炸引信不起爆 | C(玩家报告:导引头没锁上打不中) |

全部为 D:载机发射后按发射速度直线飞行(不 crank、不重锁);数据链断开后不恢复;此前的默认数据链直接给出目标真值,只要导引头未锁定就一直有效,会让下视三九目标在导引头全程 notch 的情况下仍被数据链引导命中。判别实验:下视三九目标,分别在发射后保持锁定与主动丢锁两种情况下各录一组回放,看导弹是否仍飞向目标、导引头开机后是否命中。

## 未决简并(诚实清单)

- 显示 α 的乘性标度(≤~10%)——影响所有斜率类结论的绝对值。
- 引擎内部升力参考面积(流水线 2 可解)。
- body/fin 升力拆分、全部力矩/阻尼通道(P1 断锁振铃 + wrpl 姿态)。
- wingAreaMult 是否进升力(2.58 麻雀 wam=3.25 大杠杆,流水线 3)。
