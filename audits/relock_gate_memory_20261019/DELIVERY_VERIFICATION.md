# 已完成的交付复现与边界

2026-10-19。交付源码4447c23c862bfd897b18df27682b3dba086f0136，运行39263363711/job118096525347成功。本说明在交付完成后写入；包内VERIFICATION.json、receipts/REPRODUCTION.json已保存相同底层实际结果。

从组装包实际重跑general/scalar/review/final四阶段。14个JSON或解压JSON输出全部逐值一致，最大数值差0；没有依赖浮点容差隐藏差异。最终65个文件逐尺寸/SHA256核验，加根MANIFEST.json共66文件。内容47,456,340bytes，不含根manifest。生产src/tests/config/data与dev ab93a013dbe5ed8010e3f925a4bcbc1fa83d6f8d无差异。

原生扫描范围1,633,750行，R77选中129,079条、82实体。424held过程，139严格过程/2831held记录用于标量传播。二次原生索引扫描验证36,254fresh标识的实际同EID前序记录；排除21对跨其他状态的伪相邻fresh，两时钟共42。最终packet相邻fresh26,878对，state26,880对，来自76实体。

独立标量运算最大差2.28e-13m/s。9项合同检查、2831条SHA一致和当前答案字段NaN污染检查通过。沿预测LOS累计自身速度补偿的closing重构RMS0.488640m/s、最大1.695845；#118为0.258515/0.478960m/s。该误差是输出重构，不是自主捕获精度。

#118全部23条held，在参考15deg半功率/RCS2下20条信号相容。候选速度半窗10或20可拒绝这20条全部；300则一条不拒。较广558条早期信号相容held中，movingLOS记忆+10拒138、+20拒89、+300拒0。7个严格且参考目标接近的短时恢复均不被这些速度窗拒绝。

但通用fresh正样本中，10m/s会误拒308/26878，20m/s误拒6/26878；state口径20仍有4个反例。候选门并非已证明的通用接受条件，closing/last自主初始化及实际门宽语义仍未确定。其余绝大部分held尚需其他原因解释，不能统一加冷却或删除反例。

Artifact: missile-sim-relock-gate-memory-reviewed-20261019
Artifact ID:12192160369
ZIP bytes:45,944,120
GitHub ZIP SHA256:9797dd2d903962ada63d4870ba62eabc6f3f308796bd1bb1f3568227c58bc513

```bash
python3 reproduce.py /absolute/path/to/new-output
```

输入和全部反例、六个原始标量候选、movingLOS补偿方式、packet/state对照和独立邻接索引均随包保留。入口不重新解包WRPL或重扫原生全流，完整流扫描脚本及来源在workflows中。未运行新的自主飞行，也没有多AI subagent独立评审。
