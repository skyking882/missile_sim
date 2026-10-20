# 已完成的独立数值复核与交付

2026-10-20。执行与交付源码7fbc28e4acca44388bdc1188779ce07b7f21d05d；workflow39425706971。contract/job118637603745、propagation/job118637604181、review/job118637827017均success。RESULTS.md在执行后另行写入分支3e57a433531e0749f53422716e9f79191764b696；交付ZIP内主要阅读文件是review/TABLES.md和README.md，不含后写的中文RESULTS.md。

## 真正执行的三条工作流

两个并行CI job分别检查native输出坐标契约和139段held的三种传播解释。随后review job用独立的标量Rodrigues矩阵和世界方程RK4，复核在弹体系求得的解析区间解。这是同一助手组织的独立数值实现，不是AI subagent独立评审。

输入来自先前已完成的原生状态提取，按3个输入文件manifest验证；本轮不重新编译native、重解包WRPL或新增盲测数据。139段strict held、60枚R77、2831保存节点；931个捕获边界。逐条核对2831个held原始行SHA，以及1862个捕获端点SHA。

## 独立积分结果

采用每个保存区间8和16个RK4子步的世界方程数值积分，与精确的分段体轴解比较。16子步最大方向差：

| 候选 | 最大差deg |
|---|---:|
|世界系rate|2.593442990606907e-12|
|惯性rate的弹体系分量|5.3537495404069635e-12|
|相对弹体rate|1.3955260864357651e-8|

相对弹体候选8子步最大差2.2325905724443274e-7deg；细化后约16倍下降。候选间实际误差为度至数十度，不能由上述数值差解释。

240个合成旋转/协变性合同检查通过，最大合成RK角差2.7210497164162583e-10deg。纯弹体旋转、零相对rate、native负z射线符号等控制通过。独立标量射线计算与主实现最大差1.1711118186319425e-14deg。重新计算所有主传播误差与存档差0。

共同世界坐标旋转下，正确的W/B/R方程协变；把已经world表达的rate直接当作body数字额外乘Q则一般不协变。此项只诊断公式的坐标契约，不是用新的游戏旋转场景证明游戏源码存在bug。

## 交付实际重跑

组装交付后实际运行reproduce.py，重新计算contract、propagation及review。8个结果JSON/解压JSON文件全部逐值一致，最大数值差0。比较器保留float绝对1e-8+相对1e-11容差，整数/文本/结构必须相同；本次没有靠容差通过。此是同数据重现，不是独立新样本。

最终29个文件按大小与SHA256核验，加根MANIFEST.json共30文件，内容8,524,091bytes（不含根manifest）。生产src/tests/config/data与dev ab93a013dbe5ed8010e3f925a4bcbc1fa83d6f8d无差异，没有新PN或自主飞行消融。

Artifact: missile-sim-rate-frame-contract-reviewed-20261020
Artifact ID:12265159220
ZIP bytes:8,476,814
ZIP SHA256:066d0e2e515f55b07a570159bc38485f8dde9720838be6436e32f1fd8e0745b0

```bash
python3 reproduce.py /absolute/path/to/new-output
```

## 更正与结论边界

旧axial-rate实验actual_gimbal射线z符号与native parser不一致；本轮已更正该项研究对照，旧actual_gimbal结果撤回。没有修改原始parser或生产代码。恒world-rate held传播、capture lambda统计和world-up/body-up旧对照不调用错误ray，不受它影响。

139/139 held段的世界表达解释均比两种body解释好，支持raw序列化rate是世界表达；它不排除游戏内部采用body变量、输出再转world。Q初始化的标量相容不能升级成完整状态算法证明，也不能因world几何不需要lambda就直接从当前恒完整rate传播中删掉它。缺失的输入转换/局部接口误用是一项候选，不是已确认游戏bug。
