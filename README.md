# 服务端 WRPL 原始流与配套解包工具

这是 2026-10-17 整理的独立资料分支，保存当前已经下载的五场 War Thunder **服务端录像原始文件**，以及实际使用的解包器、源码、补丁、构建依赖和初始化资源。`raw/` 下的 46 个 `.wrpl` 与原下载记录逐字节一致，合计 **166,695,211 bytes**；没有重新编码、拼接或替换成 JSON，也没有混入客户端录像。

## 录像清单

所有回放头的游戏版本均为 **2.59.0.54**，`playerNo = -2147483648`（服务端 sentinel），分段头的 session ID 和 part 编号均已逐文件核对。

| 原场次标签／目录 | Session ID | Session 十六进制 | 已下载 part | 文件数 | Bytes | 原下载结束记录 |
|---|---|---|---|---:|---:|---|
| `2026.10.08_11.49.38`（含 #118） | `1331658428351516369` | `127b0049000c0ed1` | 0, 1, 3, 5, 7, 9, 11 | 7 | 32,219,759 | part 13 返回 404 |
| `2026.10.08_12.03.42` | `1331658428351518589` | `127b0049000c177d` | 0, 1, 3, 5, 7, 9, 11, 13, 15 | 9 | 26,225,061 | part 17 返回 404 |
| `2026.10.08_12.11.04`（含 #148） | `1331658428351520915` | `127b0049000c2093` | 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 | 12 | 37,179,989 | part 23 返回 404 |
| `2026.10.08_12.23.06` | `1331658428351522800` | `127b0049000c27f0` | 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 | 11 | 33,752,585 | part 21 返回 404 |
| `2026.10.08_12.45.12` | `1331658428351525826` | `127b0049000c33c2` | 0, 1, 3, 5, 7, 9, 11 | 7 | 37,317,817 | part 13 返回 404 |

目录名沿用原客户端场次标签以关联历史研究；文件身份由服务端头和下载 session ID 确认。`0,1,3,5,...` 是这批既有下载使用的服务器分段序列。偶数编号没有下载，不能据此认定丢包；遇到下一奇数段 404 的记录，也不能单独证明整个服务器协议的完整性。

## 目录

```text
raw/<场次>/0000.wrpl ...            原下载字节流
metadata/server_download.json       原下载 URL、HTTP 状态、响应头、大小、SHA-256；路径改为包内相对路径
metadata/original_server_header_verification.json  原始逐段头校验记录
metadata/dependency_lock.json       上游、补丁和外部依赖的不可变提交／源码归档 SHA
metadata/tool_provenance.json       源码恢复、补丁和既有二进制的具体来源与限制
metadata/verification.json          本次封装实际检查结果
manifest.json                      包内文件清单和校验范围
SHA256SUMS                         除此文件外所有交付文件的 SHA-256
runtime/game_resources/             三个初始化用 vromfs；不含完整游戏安装或 datamine
 tools/extract_server.py             本资料分支的服务端解包入口
 tools/verify_bundle.py              标准库校验入口
 tools/build_backend_bundle.sh       使用包内源码和依赖的构建入口
 tools/backend/macos-x86_64-cpython39/ 既有 macOS x86_64 CPython 3.9 扩展
 tools/WrplReplayParser/             固定上游版本 + 已有本地补丁的源码
 tools/local_changes.patch           相对于固定上游归档的完整文本差异
 tools/replay_inspector/             历史提取／保留原始字段／构建修补脚本与说明
 tools/dependencies/                 四项额外构建依赖的固定提交源码归档及许可证
```

## 下载与校验

```bash
git clone --single-branch --branch server-replays-raw-20261017 \
  https://github.com/skyking882/missile_sim.git server-replays
cd server-replays
python3 tools/verify_bundle.py
```

本分支在所有交付的 `.gitattributes` 中以最后一条 `* -text` 禁止 Git 换行转换，确保克隆后的文件字节与校验清单一致。

校验入口不需要原生扩展，只使用 Python 标准库：检查完整交付文件 SHA-256、46 个原始流的原下载大小和 SHA、分组编号、WRPL magic、session、playerNo 和版本。也可只检查原始流：`python3 tools/verify_bundle.py --raw-only`。Linux 可直接 `sha256sum -c SHA256SUMS`，macOS 可 `shasum -a 256 -c SHA256SUMS`。

## 用已有后端解包

随包扩展为 **Mach-O x86_64、CPython 3.9**，链接系统 Python3.framework 3.9、libSystem 和 libc++。在本次 Apple Silicon Mac 上，使用 Rosetta 和 `/usr/bin/python3`（3.9.6）成功运行。它不能直接由 arm64 Python、其他 CPython ABI、Linux 或 Windows 加载。

在上述 macOS 环境，从分支根目录运行：

```bash
arch -x86_64 /usr/bin/python3 tools/extract_server.py \
  2026.10.08_11.49.38 --output extracted/2026.10.08_11.49.38
```

入口默认使用包内原生扩展、三份包内 vromfs 和 `raw/<场次>`；也支持显式目录，所有位置均可通过参数替换：

```bash
python3 tools/extract_server.py --help
python3 tools/extract_server.py raw/2026.10.08_12.11.04 \
  --module-dir build/backend/python/main \
  --game-resources runtime/game_resources \
  --output extracted/2026.10.08_12.11.04
```

每次输出目录必须尚不存在。一次进程解析一场，适配后端的进程级初始化与 ECS callback 生命周期。原生后端启动时会报告没有 `content/base/res/tanks`：本包没有坦克 grp 资源；本次空战回放的 packet、weapon probe 和 unit history 导出仍完成。输出不是对所有坦克模型、地图资源或回放功能的覆盖承诺。

输出包含：

- `packets.jsonl.gz`：解压后的逐包 payload 字节、bit 数、packet type 和 **packet time_ms**。这是原 WRPL 容器的派生解包结果。
- `weapon_raw.jsonl.gz`：后端成功解析的 WeaponSync 字段与对应原始 bit 片段的 byte slice hex；`start_bit`、`end_bit` 描述 bit 对齐，首尾字节可能含相邻 bit。
- `unit_history.json.gz`：原生单位历史的 `t_ms`、坐标、Euler 原字段，及 rocket 身份／武器关联。
- `getFmProbe.json.gz`、`getFmVelocity.json.gz`、`getUnitMpiProbe.json.gz`：已有探针保留的飞机状态、速度候选和 MPI 消息概况。
- `summary.json`、`parser.log`：计数、布局、后端身份、运行环境和原生日志。

**只有 `raw/` 下的 `.wrpl` 是原下载流。** 解压包、bit slice 和字段 JSON 分别属于派生字节、解码字段或候选语义层，不能互相替代。

## 从配套源码构建

上游固定为 [LivingTheDagor/WrplReplayParser，221be5cac8e772ab41eefb0e7dac35ce6fb5c39b](https://github.com/LivingTheDagor/WrplReplayParser/tree/221be5cac8e772ab41eefb0e7dac35ce6fb5c39b)。补丁来自本仓库历史提交 `6576e6b` 的 `tools/replay_inspector/`：保留原始 weapon 字段／字节、FM 和 unit MPI 探针、Python callback 清理、绑定和平台修复。原运行临时源树已被清理；此处从留存的固定上游源码归档与已有补丁脚本恢复，详见 `tool_provenance.json` 和 `local_changes.patch`。没有更改位流解码算法以适配本次上传。

额外依赖 pybind11 3.0.4、Tracy 0.13.1、libdwarf-lite 2.1.0 和 zstd 1.5.7 都附有不可变提交对应的源码归档；构建入口解压到构建目录，并通过 `FETCHCONTENT_SOURCE_DIR_*` 指向本地文件。**不需要重新下载 parser 或这些依赖。** 上游自身附带的 EASTL、mimalloc、fmt、cpptrace、zlib-ng 等保留在 `3rdParty/`，各自许可证保留原位置。

需要系统工具链：CMake ≥3.22、C/C++23 编译器、Python 开发头文件。Ubuntu/WSL 的历史工具环境为 GCC 13.2／Python 3.12，还需要 `libunwind-dev` 和 `liblzma-dev`；可用：

```bash
sudo apt-get install cmake g++ python3-dev libunwind-dev liblzma-dev
bash tools/build_backend_bundle.sh
python3 tools/extract_server.py 2026.10.08_11.49.38 \
  --module-dir build/backend/python/main --output extracted/linux-first
```

macOS 需要 Xcode clang、SDK、具有 x86_64 slice 和开发头文件的 Python 3.9、CMake，Apple Silicon 还需要 Rosetta。入口默认使用 Xcode 的 Python，固定 x86_64 和 SSE4.2 编译；可用 `PYTHON3` 指向兼容解释器：

```bash
PYTHON3=/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
  bash tools/build_backend_bundle.sh
arch -x86_64 /usr/bin/python3 tools/extract_server.py \
  2026.10.08_11.49.38 --module-dir build/backend/python/main \
  --output extracted/macos-rebuilt
```

本次验证运行的是配套的**既有二进制**，未在此次封装过程中完整重编译，也未验证源码／二进制逐位可复现。构建入口做过 shell 语法检查；跨平台重建结果仍待实际运行确认。历史 `tools/replay_inspector/README.md` 和其构建脚本保留原工具分支用法；本资料分支请优先使用这里的 `build_backend_bundle.sh` 和服务端入口。

## 时钟与字段边界

- 容器 packet timestamp、原生 unit history timestamp、WeaponSync 的 `b5` 分别保留。现有五场研究使用 `b5/48` 秒作为对应状态的物理 tick 候选，不能直接用 packet `time_ms` 替代；48 Hz 的观测不等于传感器扫描／接受率。
- `b7_*_raw_u16`、上游数值解码、未知 seeker bit block、Euler／rate 的语义需要分别审计。诸如 INS／TRK、LOS、锁定目标 ID、缓存／计时器等候选含义，并未因为成功解包而全部确认。
- 探针只保存后端成功走过的解析路径；state 数、bit layout 数和原始文件 SHA 可检查一致性，不能证明所有消息布局被完整理解，也不能单凭这些计数定位网络丢包。
- 本分支不修改导弹仿真、参数、增益或生产模型。现有模型进展仍在 `dev` 分支。

本包的 upstream parser、Dagor Engine 派生代码、第三方库与补丁锚点各自沿用原许可证，见源码 `LICENSE`、`tools/replay_inspector/THIRD_PARTY_NOTICES.md` 及 `tools/dependencies/licenses/`。录像和三份游戏资源为 Gaijin 游戏数据，其字节保留用于回放复核；本包不为它们另行授予软件许可证。
