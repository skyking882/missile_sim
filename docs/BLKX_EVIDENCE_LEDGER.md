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

### 发射离轴角上限(2026-10-05,仅 WT_overlay 交战环境使用,missile_sim 内核未读取)

| 字段 | 数值 | 模型中的语义 | 等级 |
|---|---|---|---|
| `rocket.guidance.launchAngleMax` | [180, 90]:R-77-1、RVV-SD、PL-12A、AIM-120D、MICA EM;其余雷达 AAM(AIM-120A/B/C-5/C-7、R-77、RVV-AE、PL-12、SD-10A、Derby、R-Darter、AAM-4、Rb 99、AIM-54)无此字段 | 允许发射时目标相对机头的最大方位角、俯仰角(度);[180, 90] 即可全向(越肩)发射。实际可发射角还受载机雷达可转动范围限制(例:N035 ±120°、APG-63(V)3 ±70°) | 数值 A;语义 C(玩家:R-77-1 可全向发射,配合苏-30SM2 的 ±120° 雷达可越肩打;AIM-120C 不支持全向发射;金鹰受雷达范围限制打不了越肩) |
| (无对应字段)无 `launchAngleMax` 时的上限 | 统一 60° | 同上 | D(用户 2026-10-05 决定取 60°;这些弹的文件中只有导引头 `angleMax`/`lockAngleMax` 55° 或 60° 与 `uncageBeforeLaunch=true`、`lockAfterLaunch=true`,未找到游戏侧证据) |

实测(2026-10-05,`simulate`,8 km、1100 km/h、目标 15 km 直飞、sensor_track + look_down_angle 2° + CW + require_seeker_lock):R-77-1 在 120° 离轴命中、150° 寿命耗尽未中;AIM-120C-5 在 150° 仍命中(飞行 42 s)。内核本身不限制离轴发射,上限由交战环境按上表施加。判别实验:120C 在自定义战斗中逐步加大离轴角发射,记录无法发射或发射后丢失的角度。

## 未决简并(诚实清单)

- 显示 α 的乘性标度(≤~10%)——影响所有斜率类结论的绝对值。
- 引擎内部升力参考面积(流水线 2 可解)。
- body/fin 升力拆分、全部力矩/阻尼通道(P1 断锁振铃 + wrpl 姿态)。
- wingAreaMult 是否进升力(2.58 麻雀 wam=3.25 大杠杆,流水线 3)。

## 主动雷达 SRC 搜索与重新捕获（2026-10-06，opt-in 候选）

`seeker_search=None` 保留既有观测和输出；启用字典时，只对 profile 的主动雷达标志为真的 observer 应用扫描、方向增益和连续探测 dwell。活动参数集为 2.59.0.27 / `4ac255b829a27346b9300fccdc138362c2535fbe`。当前 21 个主动雷达 profile：`cn_pl12`、`cn_pl12a`、`cn_pl15`、`cn_sd10a`、`il_derby`、`ir_fakour_90`、`jp_aam4`、`r_darter`、`su_r_77`、`su_r_77_1`、`su_rvv_ae`、`swd_rb99`、`us_aim_120a`、`us_aim_120b`、`us_aim_120c_5`、`us_aim_120c_7`、`us_aim_120d`、`us_aim_54a`、`us_aim_54b`、`us_aim_54c`、`us_aim_54c_plus`。每弹从自己的归一化 profile 读取，未修改参数集或借用 R-77-1 常数。

| 输入／机制 | 本候选中的解释 | 证据 |
|---|---|---|
| `radarSeeker.transmitter/receiver.antenna.angleHalfSens` | tx/rx 分别作为高斯单程半功率半角，`exp(-ln(2)*(theta/half)^2)` | 字段数值 A；半功率解释和高斯形状 D |
| `...antenna.sideLobesSensitivity` | 每路单程线性功率增益下限 `10^(dB/10)`；双程相乘 | 数值 A；下限模型 D。独立字段 `radarSeeker.sideLobesAttenuation` 不叠加使用，其关系未识别 |
| `radarSeeker.receiver.range/rcs` | 参考距离/RCS；目标与箔条的探测距离按 `(sigma/reference_rcs * G_tx * G_rx)^(1/4)` 缩放，并保留原距离、角度、角速率、距离门、多普勒门和杂波门限 | 数值 A；四次根、阈值和有增益的诱饵强度竞争 D |
| `guidance.lockTimeOut` | 搜索时同一候选连续可探测的 dwell，允许显式 `dwell_s` 覆盖；缺失时零额外 dwell并保留来源说明 | 数值 A；把 timeout 解释为 dwell 是 D。当前 21 弹均为 0.75 s，仍逐弹读取 |
| 无导弹扫描参数字段 | 以当前可用数据链解／已初始化且有效的惯导解指向，否则机头；中心和波束按本弹 lockAngleMax 限幅；spiral/raster、20°半角、1.5 s一帧是可配置起点 | D；玩家报告提供 SRC 扫描、甩锁后直飞／变线差异的观察，尚无逐弹扫描轨迹或周期识别 |
| `radarSeeker.prolongationTimeMax` | 沿用原跟踪保持窗口；仅窗口到期后进入新搜索、重置扫描相位并重新累计 dwell；短检测缺口不混入新搜索 dwell | 数值 A；现有窗口机制沿用，本新增状态边界 D |
| 箔条 `identity` | 每个源／释放束的稳定身份，用来防止跨候选拼接 dwell；目标身份与箔条分开。外部箔条无稳定身份时，新搜索报告 `acquisition_identity_missing` | D，仿真接口约束 |

有差异的参数须各自生效：AIM-54 家族和 Fakour-90 的 tx/rx 半角为 6°/9°，旁瓣分别 −19 dB；其余当前主动弹为 15°/15°、−30 dB。AIM-54A/B 惯导漂移为 10 m/s，AIM-120D 与 PL-12A 为 0，其他当前主动弹为 2 m/s；漂移值来自各弹已有字段，零漂移不自动证明 GNSS 机制。运行输出 `model.seeker_search.parameter_sources` 对新增语义保留 `kind: assumed`、D 级来源；不改写原 profile 数值出处。

尚未完成的游戏判别：同几何下测量首次锁定／丢锁后直飞、转向、爬升和俯冲的重新捕获延时与比例；比较 AIM-120D/C-5 和 PL-12A/PL-12；扫描方向图、周期和 dwell 在独立游戏数据上校准前，不作为已恢复的游戏引擎真值。


## 起控延迟导入候选（2026-10-09，opt-in `--autopilot-timeout-delay`）

`guidanceAutopilot.timeOut` 的原始数值来自活动 datamine 2.59.0.27 / `4ac255b829a27346b9300fccdc138362c2535fbe`；解释为飞行时间控制增益的起控延迟是 **C 级回放识别**，依据 2026-10-08 五局游戏回放与 WT_overlay 的 `docs/wt_refly_findings.md` 第 7 条、`docs/wt_seeker_chaff_task.md` A。AIM-120C-5 / D 的 0.6 s 是被比较的数据值，不是所有弹的共同常数；其他弹沿用自己的 `timeOut`，其延迟语义尚未逐型验证。

只在没有任何 `timeToGain*` 字段时，导入开关生成 `[[timeOut, 0], [timeOut + 0.01, 1]]`。0.01 s 的过渡宽度为 **D 级假设**，没有识别出游戏的渐变曲线；延迟结束后的过载残差不在此候选中修正。生成 profile 的 `guidance.parameter_sources.flight_time_gain_table` 保持 `kind: assumed` 并指向本弹 `.timeOut`。不开开关的导入仍忽略该候选语义，已有 `timeToGain*` 优先，活动参数集与 catalog 不变。验证输出位于 WT_overlay `outputs/wt_seeker_chaff_ab_20261009/`，只使用任务书列出的五局，本机运行。


## 爬升俯仰目标候选（2026-10-09，opt-in `loft_pitch_target_convention`）

活动 2.59.0.27 的 `guidanceAutopilot.loftElevation` 原值继续逐弹读取。配置 `plus_los` 保留现有 `loftElevation + LOS elevation - body pitch`；`absolute` 仅把该俯仰误差改为 `loftElevation - body pitch`，共用原 `loft_common_scale` / `loftAngleToAccelMult` 和限幅。把 loftElevation 解释为绝对俯仰目标是 **C 级回放识别**，依据 2026-10-08 五局游戏回放、WT_overlay `docs/wt_refly_findings.md` 第 8 条及 `docs/wt_seeker_chaff_task.md` B，未宣称为官方字段语义。新配置的 `parameter_sources.loft_pitch_target_convention.kind` 为 `assumed`；已有标定系数未重新拟合。

只改变爬升阶段的偏置公式。原 LOS / 飞行路径角门限、视线角速率门限、不可恢复退出状态、PN、控制节拍和 plant 子步均沿用。默认 `plus_los`（缺失键也回退此值）；未知约定报错。A 与 B 分别开启重飞，B 的对比保持 A 关闭。五局本机输出及逐位默认兼容证据见 WT_overlay `outputs/wt_seeker_chaff_ab_20261009/` 与 `docs/wt_seeker_chaff_report.md`。剩余平飞高度与远距陡爬问题不在此候选中改动。

## 2026-10-09：C 雷达导引头 vg_soft 箔条捕获（opt-in）

- 项目对象：War Thunder 游戏行为。`chaff_capture=None` 默认仍走原目标/箔条路径；不修改活动 catalog、凹口参数、控制节拍或单等效表面架构。
- **C / calibrated**：2026-10-08 指定五局 `capture.py` 的每次投放条件捕获率随投放时目标对弹目 LOS 的对地径向速度绝对值下降，上/下视近似同形；箔条捕获可发生在目标下视凹口内，无可识别的近远非对称。分类为投放前 1 s 内 WT code31 且 dT≤300 m、距离≥1500 m、投放≤终止前4.5 s；4 s 内 off-target31 与本次投放束距离≤150 m。固定前四局 1865 机会 / 173 捕获用于标定；第五局保留验证。
- **D / assumed**：函数 `p_max*exp(-max(0, |vg_release|-v0)/scale)`、每次真实投放一个 keyed Bernoulli 抽样、固定 1.3 s 最早捕获延迟、距离 width / angleHalfSens 几何门、同时候选按最早 release 然后稳定 bundle identity 关联；捕获后保持该束至失效，再走原目标重截获/搜索。此函数形式、延迟分布和多投放关联尚非游戏源码事实。
- 运行时启用默认参数 `p_max=.5, v0_mps=25, scale_mps=58, delay_s=1.3, seed=0` 是原汇总曲线的候选值；四局冻结标定使用 `.57, 7, 72, 1.3`，只由 train 选择，第五局不调参。标定区间、train/heldout 分档统计和实际重飞几何门损失在 WT_overlay `docs/wt_seeker_chaff_report.md` C 节及独立 C 输出中报告。
- 实现：`radar_seeker.py` 仅新模式的 chaff 分支跳过地杂波凹口、硬 Doppler cell、旧 RCS/R⁴ 竞争；目标的原凹口和原搜索/保持门保留。`chaff.py` 传递 release 时间、目标状态及稳定 release/bundle identity。`public_api.simulate`、`simulate_surface`、`create_surface_missile` 接受显式 `chaff_capture`。启用输出的 `parameter_sources.vg_curve.kind=calibrated`，`functional_form/delay_s/association.kind=assumed`；不写入任何 profile 的默认参数来源。
- 限制：`capture.py` 机会之间可重叠，投放结果并非独立；数值拟合得到的是所用分类下观测的4 s捕获概率，作为抽样概率后还会被几何/时序/实际锁定门压低，不能称为实际导引头捕获率。箔条轨迹的 stop τ=.1 s、固定延迟和独立抽样都为 D 级，射手雷达和数据链真值漏洞属于 D 部分，此条未修。

## 2026-10-09：D 射手雷达航迹数据链（opt-in）

- **C / calibrated**：2026-10-08 指定五局同名 rocket 的导引头块 `code 25` 记录位置/速度在首次 `code 31` 前可作为接收端数据链估计的代理。首次与目标真位置相差 >50 m（先于任何 code31）是本实验的 WT 航迹失真判据；并不等于可直接观察的飞机雷达丢锁。code31 属于导弹自身导引头，不作射手雷达真量。来源为 WT_overlay `shooter_radar_at_loss.py` 既有分析方法及 D 输出逐发表。
- **D / assumed**：`launcher_support` 显式返回 `(reason, native_y_up_position, native_y_up_velocity)` 时，所有 datalink、seeker-search estimate 与 INS-reset 消费者使用此估计；旧字符串回调保持原路径。WT_overlay 的 `datalink_track_estimates=False` 默认关闭，开启从实际 STT/TWS track 的 `at(t)`/velocity 读取，coast 时发送外推值；缺少可用范围/速度航迹时报告 `track_estimate_unavailable` 并按原永久断链规则处理，不回退真值。ENU↔native 为 `(E,N,U)↔(E,U,-N)`，估计查询时间为导弹发射时刻加回调相对时间。
- **D / assumed**：飞机雷达 `radar_chaff_capture=None` 默认关闭。开启后每个被跟踪目标的一次投放只做一次 keyed Bernoulli，取投放时目标对射手 LOS 的对地径向速度；接受后延迟屏蔽该目标的雷达测量，沿用原 STT/TWS 外推和 timeout。`p_max*exp(-max(0,|vg|-v0)/scale)` 的形状沿用 C 的导弹候选，射手的函数形式/屏蔽时长仍是假设，原 notch/gimbal/Doppler/scan/coast 参数不动。
- 射手标定仅用前四局：隔离投放（.12 s 聚合、后4 s无下一投放）且在首次失真/首次code31之前、具备完整4 s记录，标签为4 s内首次>50 m。32机会/11标签；固定 `v0=25, scale=58, hold=8 s`，训练识别 nominal `p_max=.974`，95% profile likelihood 区间 `[.646,1.0]`，训练标签延迟中位 `1.838 s`。第五局冻结验证11机会/3标签、曲线期望3.096。小样本、竞争原因、只观察到接收端估计和稀疏同步行均限制因果/时长识别，不可称游戏源码规则。
- WT_overlay `support="replay_estimate"` 仅读取同名 rocket code25 的 `P_s/V_s`，Dagor 原点变换后转 native，按同步记录时间线性插值，末端沿最后速度外推。首个可用code25之前不作反向外推、不补目标真值，初始缺测按原永久断链语义处理；code25在截获后的数据链/弹载记忆来源不可区分，此模式是接收端估计对照，非独立飞机雷达实测。默认 `support="replay"` 字符串断链模式不变。
- 实现与验证范围：不修改活动 profile、A/B/C 的模型参数、单等效表面或48 Hz节拍。D 数据/默认兼容/经验未达成项详见 WT_overlay `outputs/wt_seeker_chaff_d_20261009/` 与 `docs/wt_seeker_chaff_report.md` D 节。


## 2026-10-09 opt-in 跟踪门模型（新任务书第1部分）

`track_gate_model=None` 默认关闭。旧 `chaff_capture=vg_soft` 和旧飞机雷达掷骰实现保留；两者与新导引头路径不组合。新路径不把逐次投放概率或1.3秒延迟当机制。

| 主张 / 参数 | 等级 | 来源与边界 | 本轮映射 |
|---|---|---|---|
| 飞机 PD acquisition 有凹口滤波，tracking 不做该滤波 | B | [Server Update 29.12.2023](https://forum.warthunder.com/t/server-update-29-12-2023/62261)，官方说明范围是列出的飞机雷达 | 导引头 tracking 不硬删除凹口回波是 C/D 模型解释，不扩称为官方全导弹证明 |
| 列出的主动弹有角坐标/角速度门；箔条 track 可阻挡真飞机，INS 使用最后角度/角速度 | B | [Update 2.43.0.24](https://warthunder.com/en/game/changelog/current/1691)，PL-12 / MICA-EM / Derby / AAM-4 / R-77 / AIM-120A/B | 以已接受反射体的运动预测为门中心，不以目标真值重新对心 |
| `angleGateRate` 数值 | A 数值 / D 映射 | 现有 profile 保留原 blkx 导入的30°/s；门的角速度差用两个 LOS 角速度向量之差 | 现有 `angle_gate_rate_deg_s`，不修改 profile 原件 |
| 下视凹口目标失锁 hazard | C | 指定5局，`wt_chaff_seduction_findings.md` 第7节及原无箔条曝光统计 | 绝对vg 0–25 / 25–50 / 50–75 m/s 为 .34 / .39 / .09 s⁻¹；≥75 不加本底；每控制时间抽样一次 `1-exp(-hazard*dt)` |
| 跟丢后的复锁窗口 | C/D | 五局中0–3° 44/54、3–8° 4/11；8–15° 0/17 | nominal ±5°，围绕当前数据链/INS估计；参数来源 assumed，未标定 |
| 距离门 / 速度门 | D | `distance.width 300` / `dopplerSpeed.width 20` 是数值，全宽/半宽和真实关联规则未知 | nominal ±150m / ±10m/s，assumed；独立可配置，不替换硬件距离/框架角/rateMax |
| 角度门宽、竞争、RCS | D | 官方有机制但无角度宽；回波竞争未识别 | nominal ±5°；`stronger`=RCS/R⁴；可选 `stronger_fluct` 为按种子的lognormal起伏，σ=.25 assumed，初值重飞不用起伏；飞机RCS与箔条峰值均沿用1，无翼展/aspect标定 |
| 箔条包络 | A 数值 / D 时间包络解释 | 既有 ChaffSpec 的 `[.05,.1,12,15]` | 直接复用既有包络，不加独立诱骗延迟；launcher分型及multiplier0三解读留第2部分 |

运行时 `model.parameter_sources` 及 `track_gate_model.parameter_sources` 登记 calibrated/assumed 与等级；未写入或替换活动 profile 文件。本轮仅实现和固定初值验证，不构成门宽、RCS或竞争规则已被游戏识别。

### 2026-10-09 opt-in 跟踪门补充（新任务书第1b部分）

- **C / assumed，接收端结构解释**：指定2026-10-08五局，2026-10-09分析 `WT_overlay/docs/wt_chaff_seduction_findings.md` 第7.3节：箔条锁结束后指向目标37/42复锁、记忆13/155。新 `track_gate_model.datalink_after_lock=True` 仅取消“第一次弹载锁定即永久断链”的 profile `reconnect_datalink:false` 假设；支援损失、用户关闭和定时断链仍永久生效。弹载跟踪时只用其测量制导和重置INS，无弹载跟踪时使用仍可用的外部数据链估计。profile原件不变。
- **C 现象 / D 宽度，assumed**：`wt_refly_findings.md` 第9条及本任务书1b-2的初次截获与记忆指向误差关系。独立 `acquire_half_deg=5` 初值限制首次截获围绕可用数据链/INS/当前指向；仍过原硬件、凹口及可选扫描/驻留。宽度留第2部分标定，不是blkx已知字段。显式 `acquire_half_deg=None, datalink_after_lock=False` 恢复第1部分结构。
- **D / assumed，离线回放初始化**：本次验证专用 `support="replay_estimate"` 首个码25须先于首次码31；用其记录速度反向外推到发射（明确使用未来首样的离线初始化），随后仅沿当时已可用最后码25的速度外推，不跨空窗插值未来记录；与 `replay` 使用同一已识别WT永久断链时刻。缺初始码25或首码25晚于首码31则估计不可用，绝不以目标真值或码31补值。码25仍可能是弹载记忆，不能称为独立射手雷达测量。四组合、缺数据及首样空窗分布登记于WT_overlay第1b报告。


## 2026-10-12：待更新备注（用户指定）

- **TRK 阶段 αβ 观测器**：修正目标状态观测处理，并纳入最终模型更新；已有估计的数据链输入不重复做 αβ。具体修正以截获和跟踪阶段的回放验证为准。
- **导引头捕获几何**：将 yaw、pitch 逐轴角度限制与受限指向周围的视线锥一起纳入最终模型更新。当前讨论案例为逐轴 ±55°；角度限制和锥角按各导弹自身参数读取或配置，不统一硬编码为 ±55° 和 15°。回放两个角度字段的机械含义仍待验证。

以上为待落实事项，不表示已完成实现或验证。


## 2026-10-16：opt-in IR 导引头与热诱弹

- **A 数值 / D 映射，assumed**：2.59.0.27 `su_r_73.blkx` 的 `opticalSeeker.fov=4.5, gateWidth=.75`；`us_aim9m_sidewinder.blkx` 的 `fov=3.6, bandMaskToReject=4, signalRelRejectedTreshold=.1, rejectionReactionTime=.02`；9L无这些拒斥字段。`InfraredSeekerConfig.from_game_weapon` 读取原值，`angles_are_full_width` 默认True并在provenance记录；角度按度、rateMax按度/秒、prolongationTimeMax作常速记忆时限、mask4对应emission2及拒斥能量占总能量比均是D解释，不能称引擎公式已确认。缩圈与闭眼可独立或同时启用。丢锁后恢复宽视场/保持窄门由 `gate_on_loss` 显式配置。
- **官方历史描述 / 当前实现D**：[Gaijin 2018-12-14 Air-to-air Missiles](https://warthunder.com/en/news/5947-development-air-to-air-missiles-en) 描述假热源进FOV后指向两源之间；实现提供 `centroid`（默认信号加权质心）及 `strongest`（另一D假设）两种竞争规则，不据旧版新闻断言2.59算法。当前 [官方托管社区wiki](https://wiki.warthunder.com/mechanics/5026-aerial-infrared-countermeasures-flares) 描述缩圈/暂时暂停；[CalvinAz 2023-09-06 玩家论坛](https://forum.warthunder.com/t/missile-irccm-aim-9m-vs-r-73/24366) 将R73归为FOV gating、9M归为tracking suspension，属玩家解释，非官方源码。无2.59回放校准。闭眼期间只外推最后可用状态，不偷用当前目标真值；超记忆时限无效，场景清除后允许重捕。
- **A 数值 / D 动力学和包络，assumed**：2.59.0.27 `countermeasure_split_launcher_jet_only_flare_large.blkx` 的bullet.rocket有mass=1、massEnd=.27、caliber=.05、CxK=2.58、force=.1、timeFire=4.4、timeLife=4.5、startSpeed=70、signature.infraRed.emissionProfile=[.4,4,4.4]及emission2=4.5。新FlareSpec保留这些游戏值；线性升/平台/降包络、CxK二次阻力、质量线性下降、固定初始喷射轴和密度模型属D。默认密度取现有StandardAtmosphere，允许显式覆盖；世界喷射方向、释放时刻、唯一身份由caller显式输入，继承释放时载机p/v；不借用箔条RCS、不用随机脱锁。末端物理寿命与发光停止独立。
- **D / assumed，观测与导航接口**：IR可见源采用emission/R²近似。默认飞机热强度按下述同版数据适配器装配，保留显式scalar回调；不把emission2伪装成温度/功率、也不把rangeBand0/1解释为已标定阈值。光学角度选源后，现有PN所需距离/速度显式使用truth-assisted closure，来源标签 `ir_truth_assisted_*`；没有声称IR测得精确位置速度，也没有伪造独立轴向角状态或启用radar q_capture。闭眼/丢失只常速预测。太阳、地面背景、遮挡、多光谱分类、热特征标定未实现。None路径及活动profile默认不变；TVC profile排除不放开。


### 2026-10-16：飞机热强度与三类热诱弹的游戏数据装配

- **A数值 / D公式，assumed**：精确commit `4ac255b829a27346b9300fccdc138362c2535fbe`（2.59.0.27）的 `aces.vromfs.bin_u/config/gameplay.blkx`：`fmConstants.thrustKgsToInfraRedBrightness=1`、`afterburnerThrustKgsToInfraRedBrightness=4.5`、`horsePowersToInfraRedBrightness=.42`；`sensorsConstants.engineInfraRedBrightnessAspectMultFront/Side/Rear=.0006/.06/1`，`engineAfterburnerInfraRedBrightnessAspectMultFront/Side/Rear=.05/.12/1`；engine/airframe/afterburner sourceMask=1/2/128。`airframeInfraRedBrightnessMult=200`、`airframeInfraRedTemperatureMachBase=.8`、`airframeInfraRedTemperatureRatioToBrightnessPwr=10`、高度乘子节点[0,1]/[5000,2.4]。这些是真实游戏参数；温度/推力变化到亮度的完整引擎公式没有源码证明。
- **A数值 / D插值，assumed**：同commit的 `aces.vromfs.bin_u/gamedata/flightmodels/fm/f_16c_block_50.blkx`：Engine0、EngineType0.Main.ThrustMax.ThrustMax0=7030、ThrustMaxCoeff_0_0=.99、AfterburnerBoost=1.25、MinThrMult=.02；`fm/su_30mkk.blkx`：Engine0/1共两台、ThrustMax0=7680、ThrustMaxCoeff_0_0=.94、AfterburnerBoost=1.15、MinThrMult=.05。新AircraftInfraredSignature读取各机型的完整高度/TAS表；按表中Type关联发动机实例，不能只数EngineType。默认D插值为高度/TAS(km/h)双线性、油门在Min/MaxThrMult之间线性、加力额外推力乘4.5、方位余弦分段插值。机体假设ratio=max(1,(1+.2Mach²)/(1+.2MachBase²))，用原始指数和高度乘子；不把FM WaterTemperature当IR温度或套用黑体材料参数。调用者只给机型与油门/加力/开关、必要的机头朝向；没有方向时用运动方向D近似，静止且无方向报错。
- **D归一化，assumed**：默认IR工厂不再需要手填target_emission/flare_gain/minimum_signal。用同版本F16C单台干推海平面静态原值7030×.99×1=6959.7作为固定发动机亮度基准，各机型保持真实发动机数量和推力比；机体以原始全局200为基准；flare band2保持raw emission2。分别按opticalSeeker.rangeBandN²加权，信号=sum(normalized_emission_N×rangeBandN²)/R²，以1为单位检测阈值并受rangeMax限距。该跨波段参考是可运行的最小归一化假设，不是已经恢复的游戏标准亮度/探测公式。旧标量回调接口原样可用，仍有独立provenance；不静默混波段或混版本。profile与thermal已知version不同时报错。
- **A数值 / D包络和运动，assumed**：普通`countermeasure_split_launcher_jet.blkx`，mass .09→.04、caliber .026、CxK2.58、force.1、timeFire/timeLife4.4、useStartSpeed=true、startSpeed55、emissionProfile[.4,4,4.4]、emission2=1；BOL`countermeasure_split_launcher_jet_bol.blkx`，mass .054→.028、caliber .026、CxK12、force.1、timeFire/timeLife1.8、useStartSpeed=true、startSpeed20、emissionProfile[.75,1.65,1.8]、emission2=.6；大口径沿上条large原值1→.27、.05、CxK2.58、70、4.5s、emission2=4.5。`*_bol_pod_aim9l/aim9m`的实际payload是普通发光值1与55m/s，不能按文件名套.6。默认密度用项目已有StandardAtmosphere；明确zero timeFire和useStartSpeed=false不会除零或硬塞epsilon。列表/嵌套rocket/显式ballistic按实际发光flare payload选取，chaff cloud及带同样emission2的AP/I炮弹不会被误认；多个flare要求指定bullet_index。
- 源文件经只读git show取得并缓存 `outputs/ir_datamine_20261016/`，清单固定version/commit；运行时依次读取本地冻结datamine、下述同版versioned字段资产或匹配清单的缓存，缺机型/版本不会套默认飞机或联网下载。没有checkout/pull/reset。`InfraredSeekerConfig.from_game_weapon(weapon_name, aircraft_model=...)`、`FlareSpec.from_game_weapon('regular'/'bol'/'large')`是默认游戏数据入口；IR选项仍opt-in，TVC profile排除与活动catalog不变。


### 2026-10-16：Air RB ≥13.0 完整 IR 范围

- **A数值与客户端换算**：同一2.59.0.27 commit 的`char.vromfs.bin_u/config/wpcost.blkx`中`economicRankHistorical`，`gui.vromfs.bin_u/globals/ranks_common_shared.nut`明确REALISTIC映射Historical，BR=`round_by_value(economicRank/3+1,0.1)`。筛选`unittags.type=aircraft`及rank≥36共101条；按`unittags.tags.hideBrForVehicle=true`排除10个隐藏BR核弹/killstreak派生条目，保留91机型。机型、BR、排除理由、源路径/原始SHA和挂载关系保存在`data/profile_sets/2.59.0.27/infrared/coverage.json`，运行必要原字段的同版抽取在`game_fields.json`，不再依赖ignored outputs才能装配。`scripts/import_infrared_game_data.py`只读精确源/缓存生成这些资产，不下载、不改变checkout。
- **A引用 / D数值插值**：91个unit均实际解析`fmFile`，共79个FM；包括共享FM、无扩展名和前导斜线的游戏路径。实际EngineN继承其EngineTypeN.Main并覆盖本实例Main，保留真实发动机数量/推力差异。F16C Block40、J8F、M346的推力系数表有缺格：只插值各行实际存在节点并在域边界夹紧，不补0/epsilon或借其他机型；缺格清单和D解释入coverage/provenance。J8F另缺`ThrustMax.VelocityType`，只对该精确FM记录TAS解释D，原字段不伪造。
- **A挂载 / D机制**：展开91个unit的WeaponSlots等真实武器引用及220个container，共40个optical AAM配置、21个bulletName。当前自动装配38配置：15 baseline、14 FOV gate、9 suspension。`jp_aam3`及`jp_aam3_default`只有mask4而无signalRelRejectedTreshold/rejectionReactionTime，同版global opticalSeeker也无该默认，记`mask_only_unresolved`并默认明确报错，不套9M参数；显式D override可研究，来源会记录。rangeBand0/1/2/3/6/7的实际值全部保留；3/6当前没有额外环境热源输入，未推断波长或补隐藏信号。
- **A发光载荷 / D运动**：完整集合实际挂载9种launcher源，均按真实bullet/rocket及payload index解析，覆盖emission2=1/.6/4.5。`FlareSpec.from_game_aircraft`只选该机真实挂载的发光payload，多个不同选项须指明launcher/index，不按机型名猜普通/BOL/大口径。沿用上节质量、阻力、发光及运动解释。
- 新公开入口`infrared_coverage()`返回完整名单；`load_infrared_missile_profile`按真实bulletName解析launcher/default/BOL别名且可检查射手机型挂载关系。19个bulletName已有flight profile；R73/R73E的IR seeker可运行但TVC flight exclusion保持。`InfraredSeekerConfig.from_game_weapon`的`aircraft_model`是目标热特征机型，独立的`launch_aircraft_model`用于核对射手挂载；不把二者混淆。原始字段是A，热归一化/Mach公式/插值/捕捉及真值辅助导航闭合仍是D；没有2.59 IR回放识别。
