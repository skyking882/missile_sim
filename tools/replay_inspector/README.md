# WRPL 导弹回放解析工具

从 War Thunder `.wrpl` 回放提取导弹实体、坐标和姿态，保留创建记录、同步原始位流，再导出几何估计和逐列来源标签。这是独立的离线工具，不接入或改变 `missile_sim` 的飞行动力学模型。

后端使用 [LivingTheDagor/WrplReplayParser](https://github.com/LivingTheDagor/WrplReplayParser)，固定源码版本 `221be5cac8e772ab41eefb0e7dac35ce6fb5c39b`。构建脚本自动应用本目录的修补脚本。

仓库不包含游戏资源、回放、原生后端二进制、解析日志或实验导出数据。运行时需要自己的本地游戏安装和回放，游戏资源只读。上游代码在构建时另行获取，不随本仓库分发，其原有版权和许可适用（BSD 3-Clause；上游含 Gaijin Dagor Engine 派生代码，另按 Dagor Engine 的 BSD 3-Clause 许可）。修补脚本中用于定位和替换的少量上游源码片段同样适用这些许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 范围与字段可信度

- `extract_missiles.py` 支持导弹列表、武器名称过滤和实体编号过滤。
- `inspect_extended.py` 保存导弹和飞机原始轨迹、创建字段、组件快照、同步字段及原始字节。
- `analyze_extended.py` 针对单枚导弹，用显式指定的飞机名称作几何对照目标，输出 77 列主表。
- `inspect_unknown_chunks.py` 和 `classify_fields.py` 是针对 `#2026.09.27 20.29.44.wrpl` / `pi28_r771_h_110` 的研究工具。候选位布局仅在这个样本验证过，不能视为通用导引头解码器。分类脚本保留该样本的身份校验。
- `READ__`：实际解码字段或原始位值；`CAND_READ__`：从位流读取但布局/含义未确认的候选数值；`CALC__`：直接换算；`EST__`：插值、差分或依赖假设的推算；`META__`：解析器或分析元信息。

AoA 依赖零风假设、姿态与速度候选；弹目距离依赖目标轨迹插值；法向运动学加速度不是原生气动过载。未知位流中的目标位置/速度只是候选内部跟踪状态。不能从末点距离判断命中，也不能由组件 `active=false` 判断飞行或导引头状态。

## 构建后端

环境：Ubuntu / Ubuntu WSL、Python 3.10+、Git、CMake、C++ 编译器、Python 开发头文件、`libunwind-dev`、`liblzma-dev`。已验证的原生构建环境为 Ubuntu、GCC 13.2、Python 3.12。Python 后处理脚本仅使用标准库。

在 Ubuntu/WSL 中进入本目录：

```bash
sudo apt-get install git cmake g++ python3-dev libunwind-dev liblzma-dev
bash build_backend.sh
export PYTHONPATH="$HOME/.cache/wrpl-replay-parser-221be5c/build/python/main"
```

也可以给 `build_backend.sh` 传入自选的构建根目录；此时相应地修改 `PYTHONPATH` 或 PowerShell 的 `-ModuleDirectory`。构建目录必须使用固定的上游版本，脚本会检查版本。构建会修改构建目录内的上游 checkout，不修改游戏文件。

## 基础导出：Windows PowerShell

从仓库根目录运行，替换路径：

```powershell
.\tools\replay_inspector\Export-ReplayMissiles.ps1 `
  -GameDirectory 'C:\Games\War Thunder' `
  -ReplayPath 'C:\Games\War Thunder\Replays\sample.wrpl' `
  -OutputDirectory '.\tools\replay_inspector\outputs\sample_basic'
```

自定义本地武器包追加 `-LocalPackage 'C:\Games\War Thunder\content\pkg_local'`。可用 `-Weapon 'su_r_77_1'` 过滤武器名/路径子串，或 `-Entity '0x00008002'` 选择当前回放内的实体。`-Distribution` 默认为 Ubuntu，`-ModuleDirectory` 可指定另一个 Linux 后端目录。输出目录必须尚不存在。

输出包括 `projectiles.csv`、每枚导弹的 TAB 分隔 `.dat`、`manifest.json`、输入记录与日志。PowerShell 入口在解析进程退出后审计最终日志，不把原生日志延迟写出当作零错误。

## 扩展导出：Ubuntu / WSL

在本目录运行；路径使用 Linux/WSL 格式：

```bash
export PYTHONPATH="$HOME/.cache/wrpl-replay-parser-221be5c/build/python/main"
python3 inspect_extended.py \
  --game '/mnt/c/Games/War Thunder' \
  --replay '/mnt/c/Games/War Thunder/Replays/sample.wrpl' \
  --output outputs/sample_extended
python3 analyze_extended.py outputs/sample_extended --target-name TargetAI_0
```

`inspect_extended.py` 默认在存在时挂载游戏的 `content/pkg_local`，也可用 `--local-package` 指定。目标名称必须从当前回放实体清单确认；它是分析选定的对照目标，并非已解码的导引头目标 ID。`analyze_extended.py` 验证源文件身份，因此源回放路径在运行后处理的环境中仍需可读。

## 已验证单发样本的未知片段与字段分类

以下命令仅用于上述 2026-09-27 单发样本。输入是扩展导出目录，不随本分支提交：

```bash
python3 inspect_unknown_chunks.py \
  --source outputs/sample_extended --output outputs/sample_candidates
python3 classify_fields.py \
  --source outputs/sample_extended --candidates outputs/sample_candidates \
  --output outputs/sample_classified
```

分类输出有 77 列主表和 25 列候选表、JSON/Markdown 字段字典、分类检查结果。它只修改输出副本的列名，不改变原数据单元格。输入目录、候选目录、输出目录均可指定；建议每次使用新的输出目录保存独立结果。

## 检查与研究记录

在能访问源回放路径的环境中运行 `python3 qa_exports.py 输出目录`，检查基础导出的源身份、数量、有限性、时间顺序和差分算术。完整后端解析不是游戏内传感器对照验证。

- [单发样本的扩展字段与估计定义](docs/extended_202944.md)
- [607 bit 未知片段的候选解释](docs/unknown_202944.md)

这些研究记录中的 `outputs/...` 指实验时本地生成的证据，不是仓库随附文件。单个旧版本回放测试通过，不保证其他游戏版本、导弹或多目标样本兼容。

## 对局级导出：inspect_match.py（wt-overlay-exports 分支）

面向整场对局（32 人空战）的导出，输出 `match_export.json`。在 `inspect_extended.py` 的单位/导弹轨迹之外，加入后端已解码但原导出丢弃的字段。需要 `build_backend.sh`（或 macOS 的 `build_backend_macos.sh`）中新增的三个补丁：`retain_fm_sync.py`、`retain_unit_mpi.py`、`expose_kill_fields.py`。游戏目录只需 `aces.vromfs.bin`、`game.vromfs.bin`、`char.vromfs.bin`（与回放同版本），可放在任意目录作为 `--game`。

```bash
PYTHONPATH="$HOME/.cache/wrpl-replay-parser-221be5c/build/python/main" python3 inspect_match.py \
  --game <含三个 vromfs 的目录> --replay '<目录>/#2026.10.08 11.49.38.wrpl' --output outputs/<名称>
```

macOS（Apple Silicon）：Dagor 向量库依赖 x86 SSE4，只能编译 x86_64 模块，用 Xcode 自带的通用 Python 3.9 在 Rosetta 下加载：`bash build_backend_macos.sh`，然后 `arch -x86_64 /usr/bin/python3 inspect_match.py ...`。

| 字段 | 来源 | 可信度（2026-10-08 五场回放核对） |
|---|---|---|
| `players[].team`、`units[].army` | MPlayer.team、BaseExtReflectable.unitArmyNo | 已解码；32 名玩家两者完全一致，无玩家的 AI 飞机用 army |
| `battle_messages` 击杀 | KillMessage：击杀者/被击杀单位、武器名、weapon_type、death_type | 已解码；但武器名常与实际命中的导弹不同（如 R-77 命中而记为 R-73），归因需结合几何 |
| `battle_messages` 损伤 | SevereDamage / CriticalDamage：受损/施害单位 | 已解码 |
| `countermeasures` | 名称以 `countermeasure` 开头的发射器生成的弹体实体：所有者、时间、起点、寿命、自旋 u12_3 | 时间与所有者已解码；箔条/红外弹区分为候选：自旋者寿命 1.7–4.5 s（红外弹），其余 15–20 s（箔条），五场无例外。原导出只收有位置轨迹的弹体，因此漏掉了全部干扰弹 |
| `rockets[].sync` 导引头片段 | 607 bit 片段位 11 的三个 float | 候选布局，但前缀 `00000011111` 状态下与最近飞机中位距离 25 m（n=6239），可作为导引头目标 |
| `fm_probe` 目标指示 (6,0) | TargetDesignationControlState v6 位置 | 候选；中位距最近敌机 92 m；282 次发射中 272 次在发射时指向导弹后来跟踪的目标 |
| `fm_probe` 传感器列表 `s#_l#` | SensorsControlStates field4_0x4 | 候选；内容是导弹实体号（敌我双方），不是飞机，含义未定 |
| `fm_probe` 仅方向的指示 (3,7) | TargetDesignationControlState v7 单位向量 | 候选；与最近目标中位夹角 9°，敌机/导弹/友机混杂，未能确认为 RWR |
| `fm_probe` 干扰控制 `ncm` | CounterMeasuresControlState | 其他飞机的同步里恒为 0，没有数据 |
| `fm_probe.velocity` | FMSync unpack_velocity(…, 500) | 已解码，与位置差分中位差约 4 m/s |
| `unit_mpi` | 解析器跳过的单位消息（类型、消息号、长度、前 24 字节） | 只有消息号；FM_DVM 0xf09a 中 96 次出现在 100 m 内有导弹时，可作受损证据 |

没有找到的：显式的导弹命中/引信事件、RWR 告警状态、导弹导引头/数据链开关。
