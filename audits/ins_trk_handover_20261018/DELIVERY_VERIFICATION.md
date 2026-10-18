# 最终交接核验 · 2026-10-18

完成且成功：run39100358600，job117595197751。交付源码固定commit0d3cd221a427104b085c0d94df81e4a9925ef203；本轮主计算源74748de64ece10f195f2ac8cc685e201ecb7d92e，run39099843191。

## 实际验证

6项数学/接口测试通过。交付脚本从包内只读输入重新运行本轮及独立复核，6/6组结果通过；3组解析JSON完全相同，另3组最大浮点差7.105427357601002e-15。对比原先声明float绝对1e-8＋相对1e-11；所有整数、布尔、字符串、字典key和列表长度精确相同。比较容差不是物理模型误差验收阈值。

244个实际端点用独立六变量等式LP复核，全部与主四变量最小松弛LP相容性结论一致。244个实际端点的模长最小值改用另一种显式二次式及L-BFGS-B复核，区间端点最大差1.1368683772161603e-13个rate格，优化状态警告为0。任意Q模长恒等式数值误差最大8.33e-17rad/s。

原先6个R77反例未删除。补半区间几何rate变化后，3个原反例得到单元相容、未引入新失败、3个仍失败；但3个剩余反例的误差恶化，使79例RMS从.028389升为.038379deg/s。因此不把相容数76/79当作已验证统一修复。

反例#94旧rate与旧Ps/Vs的世界几何rate相差3.409999deg/s；#129为.126696deg/s。相同六例的自身运动时间比值仍约1，不能用整枚实体走了2倍标签时间修复。这些数值及原始行index、EID、rate整数和源JSON SHA都在review/independent_review.json及CASE_INDEX.csv中。

原始46个WRPL、166695211bytes及2814项校验在本轮前后保持。上一轮输入80条manifest通过。交付重跑前122个已有文件逐项检查；添加新验证记录后最终127个文件尺寸/SHA256通过，另有根MANIFEST.json，共128文件。

这是同一冻结回放派生数据上的重执行与算术复核，不是新盲测、独立AI subagent审查或生产模型完整回归。没有改dev、raw分支或native解码器。本会话本地执行失败在分析前即确认，实际数值计算均用已完成GitHub Actions，未声称本地重跑。

## 接手文件

Artifact: missile-sim-ins-trk-handover-20261018
ID:12126670224
ZIP bytes:26093178
GitHub上传ZIP SHA256:78b8d1ed253aaf6796f7af3ec4e93a9bd313e9bf38310c3580a470c2a186166c

先读HANDOVER.md，再读review/ADVERSE_CASES.md。包已包含上一轮prior/，不必拼接旧附件。运行：

```bash
python3 reproduce.py /absolute/path/to/new-output
```

该核验说明在打包完成后写入仓库；相同底层数值、VERIFICATION.md和verification/reproduction_verification.json已在包内。研究分支继承原始资料分支，不应整枝合并dev。
