# 导弹模型完整研究快照 · 2026-10-16

源码基线：[dev / 7a1c0c6](https://github.com/skyking882/missile_sim/commit/7a1c0c6e9ebc0a9e730ffaa5c2e79a40c2ae44f5)。
这是当前工作区的完整非隐藏文件快照：包含源码、版本参数、所有 outputs、dist、data/raw、两个冻结 datamine 的已检出文件，以及仍可取得的临时证据。Git 元数据、Python 虚拟环境、缓存和其他隐藏本机文件不包含在内，具体见 manifest.json。

共 6883 个文件，原始大小 2,523,794,992 bytes，压缩大小 1,111,306,595 bytes；按 48 MiB 分为 23 卷。全部文件和分卷都有 SHA-256。源码测试为 963 passed、45 failed、1 xfailed，完整日志位于归档的 sync_verification/；这是现况存档，未修复这些失败。

进展和未解问题：源码提交中的 docs/MODEL_STATUS_20261016.md、docs/WT_RATE_RECONSTRUCTION_PROGRESS_20261016.md。分析链接按恢复后的原路径工作。原线程中较早的 /tmp/local_replay_tvc_* 结果已无法取得；可取得的后续 TVC 结果在 outputs/tvc_burn_calibration_20261016 与 outputs/tvc_burnout_20261016，缺失的早期文件没有伪造或宣称已恢复。

## 恢复

```bash
git clone --single-branch --branch dev https://github.com/skyking882/missile_sim.git missile_sim
git -C missile_sim checkout 7a1c0c6e9ebc0a9e730ffaa5c2e79a40c2ae44f5
git clone --single-branch --branch research-snapshot-20261016 https://github.com/skyking882/missile_sim.git missile_research_snapshot
(cd missile_research_snapshot && shasum -a 256 -c SHA256SUMS)
cat missile_research_snapshot/snapshot.tar.gz.part-* > missile-research-20261016.tar.gz
tar -xzf missile-research-20261016.tar.gz -C missile_sim
```

合并后 tar.gz 的 SHA-256：`34ab9229c9e3cc87a95ce88b120a2ae5c00d75cb95c9d3b861b90f809688c7f1`。manifest.json 给出归档开始/结束 UTC、每个文件哈希、分卷哈希与排除路径；构建中未发现文件变动。恢复会写入快照文件，建议使用上面的新目录。
