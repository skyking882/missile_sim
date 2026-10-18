# 三类机制实测：最终核验

日期2026-10-18。最终成功run39121722846，job117658505674。交付源码commit567cd783342961f9c5aba9005ce55437457744bb。主报告RESULTS.md，全部三类任务已执行完成，不再是待运行协议。

## 数值复现

从组装后的交付包，只读其输入，重新运行range、mixture、representation、review以及两套各6项数学/接口测试。9/9组比较通过，9组数值/结构JSON完全一致，最大数值差0。三份summary的运行环境provenance另行保留，不与物理数值混为一谈；输入与源代码SHA256必须一致。不同平台重跑可存在末位差异，脚本记录而不是默认为一致。

原始12项独立测试全部通过。两并行家族作业重复执行的6项不重复计算为额外测试。该结果不等于全项目生产回归或新的盲测录像验证。

## 使用不同表达的实际复核

- 输入完整上一轮127文件manifest通过。
- 61,446个保存的rate向量误差用math.fsum重新计算，最大差1.4210854715202004e-14deg/s。
- 1,200个有界标量拟合，用SciPy bounded least-squares另行复核闭式beta公式，最大系数差3.127498260369066e-13。
- 27,930次留场预测参数和测试向量核验；训练集合与测试场次不重合。
- 931个逐点轴向/横向诊断下界，另以非负最小二乘检查。此处系数按事件反推，仅用于区分是否存在分量比例解释，不是预测器。
- 任意正比例的931个量化相容性判断，在主range任务内使用四变量与七变量等式表达分别复核。

原始bundle校验在主任务和打包前后通过，没有修改dev、原始WRPL分支或native解析器。实际工作均通过已完成GitHub Actions，没有声称本地运行或多个AI subagent。

## 六例附加诊断（不新增可部署模型）

固定s=Qnew*wold，在点值上分别允许s_parallel和s_perp取任意非负系数。比较最小残差：

| 事件 | 自由整体缩放点残差deg/s | 自由轴向/横向双系数点残差deg/s |
|---|---:|---:|
| #94 | 0.03624855 | 0.00953715 |
| #102 | 0.01400587 | 0.01000543 |
| #118首捕 | 0.03228918 | 0.01834315 |
| #129 | 0.06585688 | 0.03789635 |
| #123 | 0.02794470 | 0.00448358 |
| #112 | 0.03559516 | 0.00964784 |

这些是使用答案系数的诊断下界，不是共同规则；未在这一步联合放开全部量化单元。#129仍留有明显的点值横向形状残差，不能因此已认定更复杂的系数模型解决了它。完整数值和931个事件在review/two_component_diagnostics.json.gz。

## Artifact

Name: missile-sim-three-models-tested-20261018
Artifact ID:12134417482
ZIP bytes:30121687
GitHub-uploaded ZIP SHA256:d6e6668484a8a23843648d1c3ce24c0cf1841ed15a298526a6ef729e309e6bcd

最终188个文件逐项尺寸/SHA256核验，另加根MANIFEST.json，共189文件。包内包含完整上次handover，不需要重新拼接旧附件；167MB原WRPL不重复装入。

```bash
python3 reproduce.py /absolute/path/to/new-output
```

先看RESULTS.md和review/ADVERSE_CASES.md。研究分支audit/ins-trk-three-models-20261018继承原始资料分支，不应整枝合并dev。这份核验说明在打包完成后写入仓库；包内VERIFICATION.md及verification/reproduction_verification.json已保存相同底层实际数据。
