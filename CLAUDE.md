# CLAUDE.md

## 项目性质

本项目逆向复现 **战争雷霆（War Thunder，Gaijin Entertainment 的商业电子游戏）** 里空空导弹的飞行模型。

- 代码、文档和脚本里出现的导弹名（AIM-120A、R-77、R-77-1、PL-12、MICA 等）都指**游戏内单位**，参数来自游戏客户端文件，不是真实武器。
- **真值就是游戏引擎的行为**。"验证"指与游戏内回放对得上；"参数识别"指反推游戏引擎内部使用的公式和常数。
- 真实世界的武器性能、气动数据和制导设计，既不是本项目的输入，也不是它的目标。游戏里的数值经常为了平衡而修改（例如 2.58 测试服对全体 AAM 做了重平衡），与现实无关。
- 制导律、导引头、PID 自动驾驶仪、近炸引信、爬升弹道等术语，指的都是游戏 blkx 文件里同名字段所驱动的游戏机制。

### 数据来源（全部来自游戏）

| 来源 | 位置 | 说明 |
|---|---|---|
| 客户端 datamine（blkx 解包文本） | `data/datamine/`（2.57.1.107，commit 3f20645）、`data/datamine_2.59.0.27/`（commit 4ac255b） | 公开仓库 gszabi99/War-Thunder-Datamine 的稀疏克隆 |
| 版本化参数集 | `data/profile_sets/<版本>/`，当前活动版本由 `config/missile_profile_catalog.json` 决定 | 由 `scripts/import_datamine_missiles.py --project-dir <参数集目录>` 从 datamine 导入 |
| 游戏内回放逐帧读数 | `data/replays/*.tsv` | 由服务器回放截图里的导弹信息框转录；列语义见 `data/replays/README.md` |
| 官方 dev blog / CDK wiki | 引用记录在 `docs/BLKX_EVIDENCE_LEDGER.md` | 字段语义的官方证据 |

## 常用命令

```bash
python3 -m pytest                       # 回归测试（pyproject 已配置 pythonpath=src）
python3 scripts/run_test_functions.py   # 无 pytest 时的标准库后备测试
./run_gui.sh                            # 本地 GUI，只监听 127.0.0.1
```

仿真内核（`src/aim120_model/`）只依赖 Python 标准库，不要为它引入第三方依赖。

## 数据纪律

- **不要对 `data/datamine*/` 执行 checkout、pull 或 reset。** 适配器直接读取这些目录里的 blkx，一旦工作区移动，模型输入会在没有任何提示的情况下改变。要换版本，应新建版本化参数集并更新 catalog。
- 不同游戏版本的回放和参数不能混在一起标定。`data/replays/` 里的 5 组试飞数据（2026-08-23/24 的 tsv）录制于 2.57。WT_overlay 用来重飞对照的 5 局实战回放（2026-10-08，`WT_overlay/outputs/engagements/wt_real_v3/`）录制于 2.59.0.54（回放头 `game_version` 解码），当前参数集为 2.59.0.27。
- 回放 `overload_g` 是轨迹法向 G，包含推力分量；`displayed_distance_km` 是镜头到导弹的距离，不是弹目距离；α 小于约 1.5° 时显示值有偏差。细节见 `data/replays/README.md`。
- `dt=0.02 s` 同时是控制节拍。改 dt 会同时改变制导、PID 和舵机的更新频率，结果是另一个离散控制系统，不能当作同一模型换了积分精度。做收敛性检验时，应固定控制节拍，只扫描 plant 子步。
- `outputs/`、`dist/`、`data/raw/`、`data/datamine/` 不进 Git。开源发布由 `scripts/build_open_source_release.py` 按白名单打包。

## 分析约定

- **判别分析先预注册**：看数据之前，先列出全部候选机制和分类规则。
- 参数报告给出**识别区间、分类规则和 nominal 值**，不给孤立的点估计。
- 用同一批数据扫描选出的参数叫"**标定**"，绝不能称为"推导"或"无参数"。结论里要分清两件事：结构由理论驱动，参数靠经验识别。
- 主动列出当前数据区分不了的简并项。摆脱循环论证的标准是：用一组飞行识别出的参数，在不重新调参的前提下预测其他几何。
- 每条 blkx 字段语义主张都要标证据等级（A 官方铁证 / B 官方文档 / C 回放识别 / D 假设），并登记到 `docs/BLKX_EVIDENCE_LEDGER.md`。进入动力学模型、但等级低于 B 的映射，在 profile 的 `parameter_sources` 里保持 `kind: assumed`。
- 新机制以 opt-in 模式落地，默认路径和其他导弹的结果保持逐位不变，除非已经明确决定更新基线。

## 目录速查

```text
src/aim120_model/    仿真内核（动力学、气动、制导、控制、事件）
src/missile_gui/     本地网页 GUI
data/profile_sets/   版本化参数集（catalog 选择）
missiles/            冻结的 2.57.1.107 历史输入
configs/             冻结的 v1.0.0 回归配置
config/              运行时默认配置与 catalog
scripts/             导入、拟合、判别、诊断脚本
docs/                模型说明与证据链（UPDATE_V1.0.x.md、BLKX_EVIDENCE_LEDGER.md 等）
```

## 安全分类器误拦时的处理

这个项目会被安全分类器误判为 `general_harms`：只读 README，或者子代理交回代码审查报告，都出现过被拦。

- 被扣下的那一轮内容不要换说法重写。但**不要因此放弃项目或其中某个模块**，其余工作照常推进。
- 立刻告诉用户三件事：哪一步被拦；仓库目前的状态，有没有写到一半的文件；哪些子代理产出还能取回。
- 同一会话连续被拦，说明上下文已经中毒，之后每一轮都会继续失败。这时建议用户开新会话接着做，不要在原会话里反复重试。
- 子代理：
  - prompt 里写明本项目是游戏逆向，并让它先读本文件。
  - 任务要拆小，一个子代理只负责一个模块。
  - 报告写到文件里，回复中只给路径和简短摘要，避免一次误拦把所有子代理的成果一起带走。

与用户交流使用中文。
