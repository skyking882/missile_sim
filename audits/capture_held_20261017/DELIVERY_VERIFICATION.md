# 最终复现与独立自身运动时钟核验

最终成功工作流：39065146998，job117491103812。交付代码固定提交 fad77d7390b3123e69f22deb16cd89261ce2551b。运行完成于2026-10-17T20:05Z。

## 自身运动不是两倍时间

对两个固定锚点所有timer连续、同flag、相邻gap1至8tick的held区间，只用自身位置和速度：

vbar=(vm0+vm1)/2,

T_own=((p1-p0)·vbar)/(vbar·vbar).

不读取当前direction/rate进行时间判定。此为短区间梯形自身运动近似，不是直接测量引擎子步数。

| 集合 | 样本 | T_own/T中位数 | 按T的位移残差RMS | 按2T的位移残差RMS |
|---|---:|---:|---:|---:|
| #118 flag0 | 25 | 1.0000625492 | 0.00461061m | 35.9057307m |
| #148 flag0 | 143 | 1.0005116169 | 0.00284436m | 11.1812620m |
| #148 flag1 | 106 | 1.0001278114 | 0.00399923m | 28.7693502m |

因此#148 flag1中的近两倍角推进，不能由整枚实体实际走了2T、但标签只记录T来解释。这个检验没有确定额外角推进来自哪一内部调用；更新次数和保存前外推仍是竞争解释。

## 小包复现

从组装后的文件重新运行六项数学／接口测试：6 passed。重新计算六组结果JSON，三组逐项完全相同，另外三组仅有以下最大浮点差，数据结构、整数、字符串均完全一致：

| 结果 | JSON严格相同 | 最大绝对数值差 |
|---|---|---:|
| followup/followup.json | 否 | 1.7053025658242404e-13 |
| followup/fresh_return_predictions.json.gz | 否 | 7.16227077646181e-12 |
| followup/anchor_time_diagnostics.json.gz | 是 | 0 |
| phase/phase_results.json | 否 | 1.7763568394002505e-15 |
| censored_prior/censored_capture_results.json | 是 | 0 |
| censored_prior/censored_capture_events.json.gz | 是 | 0 |

不同字段单位不同，这张表只度量数值复现差异，不是模型拟合误差。浮点比较采用absolute1e-9＋relative1e-12；整数／字符串／key／列表长度逐项严格检查。最大差远小于这些界限。第一轮严格JSON复现检查失败，未被称为逐bit成功；差异检查日志、路径与具体数值保留在包内。Gzip时间戳不作为数值相同的必要条件。

原始bundle在最终打包前后仍通过46个WRPL、166695211bytes、2814项哈希校验。core原有19条交付文件manifest全部一致。最终交付59个文件逐项核对SHA256（另加根MANIFEST.json，共60个文件）。没有修改生产dev或原始raw分支。

Artifact: missile-sim-capture-held-20261017，ID12113310683，ZIP大小21747009bytes。
GitHub上传所得ZIP SHA256：90246310832a27e5c10592070fdd3f8147fec42d2c49b70014ac95f0666e05d0。

包中own-motion-clock.json保留每对独立自身运动计算。RESULTS.md是主研究报告；该核验说明在打包完成后写入GitHub，但上述全部原始数值与日志已在包内。当前会话本地执行工具不可用，实际计算通过已完成的GitHub Actions运行；没有声称本地重跑或多subagent评审。
