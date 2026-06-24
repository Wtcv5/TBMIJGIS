# IJGIS 风格论文修改意见汇总

> 适用稿件：*Spatially Explicit Rock–TBM Interaction Graph Sequences for Response-Consistent Interpretation of TBM Excavation Processes*  
> 核心修改原则：不再把文章包装成“强预测模型论文”，而是收敛为“空间实体—关系表达 + 响应一致性诊断”的 GIScience 论文。

---

## 0. 总体判断

当前稿件已经比早期版本明显收敛，主线基本成立：

> 将 TSP 地质体素、TBM 刀盘/护盾表面采样和运行监测响应统一到掘进里程框架下，构建几何约束的岩–机交互图序列，并通过未来响应预测任务学习候选岩–机关系的响应一致性重要性。

但目前仍存在三个主要问题：

1. **方法章仍有“方案文档感”**：Figure 1、Figure 2、Algorithm 1–3 与文字说明之间存在重复，导致方法显得像实施方案，而不是论文方法。
2. **解释性表述仍略强**：`graph-to-surface interpretation`、`hotspot map`、`surface relevance projection` 等词容易让审稿人误以为做了真实接触位置、接触压力或卡机风险反演。
3. **图表与实验结论需完全一致**：Figure 1 和 Figure 3 中不应出现后文没有真实实验对应的 ablation 项，也不能使用示意性但看似真实的 RMSE/MAE 数值。

最终目标是把论文调整为：

> **空间关系表达论文，而不是预测性能突破论文。**

---

## 1. 全文主张修改

### 1.1 必须降低主张强度

当前实验结果不支持“图序列模型显著提升预测精度”的强结论。BSLL 中 Full model 与 Persistence、Random-edge 基本持平；SJLS 中 Full model 仅表现出很小的 geometry-related numerical difference。

因此全文主张应统一为：

```text
The framework provides a spatially explicit rock–TBM entity-relation representation and response-consistent diagnostic relevance, while prediction is used as a forward consistency check rather than the primary evidence of forecasting superiority.
```

中文理解为：

> 本文的核心贡献不是预测误差显著降低，而是把 TBM 掘进过程从 monitoring-only sequence 转换为可检查的岩–机空间实体—关系过程；预测任务用于响应一致性检验，而不是作为模型优越性的唯一证据。

### 1.2 全文避免使用的强表述

建议删除或减少：

- large forecasting improvement
- robust prediction superiority
- accurate jamming warning
- physical interaction recovery
- contact pressure estimation
- contact force recovery
- calibrated risk map
- operational risk calibration
- hotspot map, unless clearly qualified as model-derived diagnostic relevance

建议替换为：

- forward response check
- response-consistent diagnostic relevance
- model-derived relevance
- component-resolved diagnostic relevance
- chainage-resolved relevance view
- spatially organised diagnostic pattern
- geometrically plausible candidate relation

---

## 2. 摘要修改意见

### 2.1 摘要总体方向正确，但“graph-to-surface interpretation route”仍偏强

当前摘要中提到：

```text
a graph-to-surface interpretation route that projects raw edge scores and derived surface relevance onto TBM surface and chainage spaces to produce response-consistent hotspot maps and interaction-evolution views
```

建议改为：

```text
a diagnostic relevance aggregation procedure that summarises raw rock–machine edge scores at surface-node, component, and chainage levels to support response-consistent spatial diagnostics.
```

这样可以避免“surface projection / hotspot map”带来的物理反演暗示。

### 2.2 摘要中保留“不建立 broad forecasting improvement”的边界

当前摘要已经写明：

- remains competitive with temporal baselines；
- SJLS is consistent with a small case-specific geometry-related MAE difference；
- experiments do not establish broad forecasting improvement；
- patterns are diagnostics, not proof of physical contact or geological causality。

这部分非常重要，应保留。

---

## 3. Introduction 修改意见

### 3.1 第一段方向正确，但应更突出“空间关系表达不足”

建议第一段保持目前逻辑：

- TBM 响应异常来自地质条件、机器几何和施工过程耦合；
- thrust、torque、penetration、advance rate、shield pressure 是外部表现；
- monitoring curves alone 难以回答 where / which component / how interaction evolves。

这一段是 IJGIS 风格的关键：先把问题定义为空间关系表达问题，而不是预测算法问题。

### 3.2 第二段应继续强调 tabular/sequence representation 的不足

当前逻辑正确：现有方法将 TSP 和 TBM 参数处理为 tabular variables 或 time-series variables，缺少“哪个 rock unit 与哪个 TBM component 形成候选作用关系”的表达。

建议保留，但可以更凝练：

```text
These methods preserve temporal dependence but usually discard the explicit spatial pairing between rock units and machine components.
```

### 3.3 第三段删掉过强的 “hotspot and evolution views”

当前第三段类似：

```text
projects the learned relevance back to TBM surface space and chainage space to generate response-consistent hotspot and evolution views
```

建议改成：

```text
aggregates the learned relevance at edge, surface-node, component, and chainage levels to generate response-consistent diagnostic views.
```

### 3.4 Contributions 建议重写为三条

建议写成：

```text
The main contributions are threefold.

First, a chainage-referenced rock–TBM spatial representation is developed to organise TSP-derived rock voxels, parameterised TBM surface samples, and monitoring responses in a unified excavation coordinate system.

Second, a geometry-constrained rock–machine relation graph sequence is constructed, in which candidate rock–TBM edges are screened by active-zone membership, distance threshold, normal compatibility, and excavation state.

Third, a response-supervised diagnostic procedure is introduced to evaluate forward response consistency and summarise learned rock–machine relevance at edge, component, and chainage levels, with explicit interpretation limits.
```

注意：贡献中不要写“significantly improves prediction accuracy”。

---

## 4. Related work 修改意见

### 4.1 保留三小节结构

建议目录如下：

```text
2.1 TBM response prediction and jamming-related analysis
2.2 Spatial representation of geological conditions in tunnel excavation
2.3 Graph-based spatiotemporal prediction and spatial interaction modelling
```

### 4.2 2.1 的落脚点要明确

这一节不是泛泛综述 TBM 预测，而是服务于 gap：

> existing TBM prediction studies mostly represent geological and machine variables as feature vectors or sequences, rather than explicit spatial relations between geological entities and TBM components.

### 4.3 2.2 的落脚点要明确

这一节不是泛泛讲 TSP，而是引出：

> TSP-derived geological information can be transformed into spatial entities, but a geological voxel field alone does not specify which rock units are candidate interaction partners for TBM components.

### 4.4 2.3 的落脚点要明确

这一节应强调：

- GNN 的意义不是“新算法”；
- 图模型在 GIScience 中的价值是显式表达空间实体与关系；
- 对 TBM 而言，边不能从黑箱相关性中自由学习，而必须受 machine geometry、excavation state、active zone 约束。

---

## 5. Methodology 总体修改意见

### 5.1 方法章最终建议目录

```text
3. Methodology
   3.1 Chainage-referenced rock–TBM representation
   3.2 Geometry-constrained rock–machine relation graph sequence
   3.3 Response-supervised graph-sequence learning and diagnostic relevance
```

不要再拆成 data alignment、rock node、TBM node、edge construction、GNN、GRU、interpretation 多个平级小节。

核心逻辑是三层：

```text
representation → relation graph sequence → response-supervised diagnostic learning
```

---

## 6. 3.1 修改意见：Chainage-referenced rock–TBM representation

### 6.1 删除“五阶段 workflow”

当前 3.1 中写：

```text
workflow contains five linked stages: subsurface sensing, spatial entity formalisation, geometry-compatible relation construction, graph-sequence learning, and graph-to-surface spatial interpretation
```

这与 Figure 1 的三栏结构冲突，也显得像旧版方案残留。

建议替换为：

```text
The framework is organised into three coupled levels: chainage-referenced rock–TBM representation, geometry-constrained relation graph construction, and response-supervised diagnostic learning.
```

### 6.2 3.1 压成三段

建议 3.1 只保留三类内容：

#### 第一段：预测任务定义

保留公式：

```text
r_hat_{t+h}=f_Theta(G_{t-K+1:t}, u_{t-K+1:t})
```

并定义：

- excavation step `t`
- graph snapshot `G_t`
- monitoring vector `u_t`
- future response `r_{t+h}`
- history window `K`
- forecasting horizon `h`

#### 第二段：空间实体定义

定义：

```text
D_geo = {(c_i, g_i)}
M_TBM = {p_j(t), n_j(t), rho_j}
```

其中：

- `c_i` 是 rock voxel coordinate；
- `g_i` 是 geological attributes；
- `p_j(t)` 是 TBM surface node position；
- `n_j(t)` 是 surface normal；
- `rho_j` 是 component label。

#### 第三段：里程对齐

保留：

```text
x_local(t)=ShieldMileage(t)-ShieldMileage_start
```

但不要展开过多实现细节。

### 6.3 把 0.05 m / 0.1 m aggregation 移到 4.2

这属于 experiment/implementation setting，不应放在方法的概念定义部分。

---

## 7. 3.2 修改意见：Geometry-constrained rock–machine relation graph sequence

### 7.1 标题加上 relation

当前标题：

```text
Geometry-constrained rock–machine graph sequence
```

建议改为：

```text
Geometry-constrained rock–machine relation graph sequence
```

这更准确地强调你的核心是“岩–机候选关系构建”。

### 7.2 保留图快照定义

这一公式必须保留：

```text
G_t=(V_t^r ∪ V_t^m, E_t^{rr} ∪ E_t^{mm} ∪ E_t^{rm})
```

这是全文方法的核心。

### 7.3 保留候选边筛选公式

候选边条件必须保留：

```text
d_ij(t) <= tau_edge
kappa_ij(t) >= eta_min
c_i in Omega_t
s_i(t)=1
```

这是你的空间关系建模贡献。

### 7.4 全文把 “physically plausible” 改成 “geometrically plausible”

因为你使用的是：

- active zone；
- distance threshold；
- normal compatibility；
- excavation state。

这些是几何和空间相容性约束，不是力学接触模型。

建议统一使用：

```text
geometrically plausible candidate relations
```

避免：

```text
physically plausible relations
```

### 7.5 Figure 3 保留，但 ablation 项必须和 Table 3 一致

Figure 3 中如果出现：

- No constraints；
- No E_rm；
- other not-tested variants。

但 Table 3 没有对应实验，就必须删除。

Figure 3 只保留后文真实实验中出现的 ablation：

- Full；
- Random-edge；
- No-prior；
- No-monitoring。

---

## 8. 3.3 修改意见：Response-supervised graph-sequence learning and diagnostic relevance

### 8.1 标题建议修改

当前标题：

```text
Response-supervised prediction and interaction relevance diagnostics
```

建议改为：

```text
Response-supervised graph-sequence learning and diagnostic relevance
```

这样更紧凑，也避免 interpretation 过强。

### 8.2 “Graph-to-surface spatial interpretation” 改名

当前小标题如果使用：

```text
Graph-to-surface spatial interpretation
```

建议改为：

```text
Diagnostic relevance aggregation
```

因为你做的是 raw edge score 的 degree-normalised aggregation，不是物理空间反演。

### 8.3 保留 raw score / attention / surface relevance 的区分

这一部分很重要，因为你解释了为什么不能直接聚合 post-softmax attention：

- raw edge score `s_ij^rm(t)` 用于解释；
- normalized attention `alpha_ij^rm(t)` 用于 message passing；
- surface relevance `C_j(t)` 用于 degree-normalised diagnostic aggregation。

这能防止审稿人质疑注意力解释方式。

### 8.4 softplus 和 degree-normalisation 的解释压缩

当前解释太长。主文建议保留一句：

```text
Softplus transforms raw pre-softmax scores into non-negative relevance summaries, while degree normalisation prevents surface nodes with more incident candidate edges from being trivially assigned higher relevance.
```

详细解释可放 Supplementary。

### 8.5 Algorithm 3 移到 Supplementary

建议：

- Algorithm 1 保留；
- Algorithm 2 可简化为文字；
- Algorithm 3 移到 Supplementary。

三个算法连续出现，会让文章像工程实现说明书。

---

## 9. Figure 1 修改意见

### 9.1 Figure 1 三栏结构正确，保持

最终 Figure 1 应是三栏：

```text
(a) Chainage-referenced rock–TBM representation
(b) Geometry-constrained interaction graph sequence
(c) Response-supervised prediction and diagnostic relevance
```

不要再拆成 a、b、c、d 四栏。

### 9.2 Figure 1 图注重写

建议图注：

```text
Figure 1. Overview of the proposed rock–TBM interaction graph-sequence framework. The figure summarises three methodological levels: chainage-referenced rock–TBM representation, geometry-constrained interaction graph-sequence construction, and response-supervised prediction with diagnostic relevance views. The relevance outputs are model-derived diagnostics and are not interpreted as contact force, contact pressure, or calibrated jamming risk.
```

### 9.3 Figure 1(c) 不要放虚构数值

如果 Figure 1(c) 里出现：

```text
Full 0.82
No prior 0.95
Random edges 0.98
No E_rm 1.15
```

必须删除。Figure 1 是方法总览，不应放与 Table 2/3 不一致的示意数值。

建议改为无数值示意：

```text
Full / Random edges / No prior / No monitoring
```

或者画 paired-delta schematic，不写具体数值。

### 9.4 Figure 1(c) ablation 项必须和 Table 3 一致

Table 3 目前包括：

- Full；
- Random-edge；
- No-prior；
- No-monitoring。

Figure 1(c) 不要写 No `E^{rm}`，除非后文真的做了这个 ablation。

---

## 10. Figure 2 修改意见

### 10.1 建议删除或移到 Supplementary

Figure 2 和 Figure 1(a)、3.1 文字重复：

- chainage space；
- rock entity；
- machine entity；
- monitoring response；
- graph snapshot；
- prediction target。

这些已经在 Figure 1 和 3.1 中表达过。

建议：

```text
Move Figure 2 to Supplementary as Figure S1, or delete it from the main text.
```

### 10.2 如果保留，图题要改

当前：

```text
Spatial entity formalisation before case-specific experiments
```

建议：

```text
Supplementary schematic of spatial entity definitions
```

---

## 11. Figure 3 修改意见

### 11.1 Figure 3 是必要图，建议保留

Figure 3 直接展示：

- active zone；
- distance and normal compatibility；
- candidate relation filtering；
- ablation design。

这是方法贡献图，比 Figure 2 更重要。

### 11.2 图注中保留边界声明

建议保留或强化：

```text
The ablation variants test relation plausibility and prior guidance; they are not interpreted as physical contact recovery experiments.
```

### 11.3 删除未实验的 ablation

若 Figure 3 出现 No constraints，但 Table 3 没有这个结果，则删除。

---

## 12. Figure 4 修改意见

### 12.1 Figure 4 可保留，但要降低“relevance projection”姿态

Figure 4 是 GNN–GRU architecture，对方法必要。

但图中如果写：

```text
relevance projection
surface relevance
```

建议改为：

```text
diagnostic relevance aggregation
component / chainage relevance
```

### 12.2 图中保留警示语

保留：

```text
Attention scores are model relevance, not contact force.
```

这句很重要。

---

## 13. Figure 5 修改意见

### 13.1 弱化 “Case roles in the argument”

当前 Figure 5 的 case role matrix 有点像答辩 PPT。建议删掉 high/low 矩阵。

保留普通 case overview：

- case name；
- start chainage；
- TSP source；
- train/val/test samples；
- horizon；
- test chainage interval；
- experimental use。

### 13.2 图注不要重复“不是 old/new data releases”

正文 4.1 说一次即可，图注不要重复。

---

## 14. 实验设计修改意见

### 14.1 4.1 保留 BSLL / SJLS 的不同角色

建议写成：

```text
BSLL is used to examine compact one-step and multi-step settings and to illustrate diagnostic relevance views. SJLS provides an external TSP-derived velocity field for prediction consistency and geometry-ablation checks.
```

### 14.2 4.2 中训练细节移到 Appendix

主文保留：

- `K=5`
- `h=1/3`
- `tau_edge=2.0 m`
- `eta_min=0.3`
- `tau_zone=5.0 m`
- train/val/test samples
- major baselines and ablations

移到 Appendix：

- hidden dimension；
- dropout；
- optimizer；
- learning rate；
- scheduler；
- batch size；
- early stopping；
- seed。

### 14.3 4.4 “Evaluation and reporting principle” 是亮点，保留

这一节提前声明：

```text
prediction metrics are used as forward response checks, whereas relevance diagnostics examine whether learned graph relations are spatially organised.
```

非常重要，应保留。

---

## 15. Results 修改意见

### 15.1 5.1 标题保留

```text
Forward prediction check across case roles
```

这个标题准确，不夸大。

### 15.2 5.1 结论保持诚实

继续明确：

- BSLL 不支持 graph sequence improve prediction；
- SJLS 是 small response-consistency difference；
- 不应解读为 large accuracy breakthrough。

### 15.3 5.2 中 “accuracy gain” 改为 “numerical difference”

建议：

```text
The numerical difference is small and should be interpreted as a case-specific geometry-related response-consistency pattern rather than a general accuracy gain.
```

### 15.4 5.3 是全文结果亮点，应加强过渡

在 5.1 后加一句：

```text
Because the prediction differences are small, the following sections focus on whether the graph representation produces spatially organised diagnostic relevance.
```

### 15.5 5.3 中 correlation 表述要谨慎

把：

```text
large-magnitude association
```

改成：

```text
high absolute correlation within the evaluated chainage interval
```

并紧跟：

```text
This should not be interpreted as a universal geological rule.
```

### 15.6 5.4 标题修改

当前：

```text
Component-resolved relevance along chainage
```

建议：

```text
Component-resolved diagnostic relevance along chainage
```

强调 diagnostic，不是物理量。

### 15.7 Figure 10 可移到 Supplementary

Figure 10 与 Figure 9 信息部分重复。若主文图太多，建议移到 Supplementary。

---

## 16. Discussion 修改意见

### 16.1 6.1 不要重复 MAE 数值

Discussion 不应重复 Results。建议第一段改成：

```text
The principal value of the proposed framework is not a large reduction in forecasting error, but the conversion of TBM excavation from a monitoring-only sequence into an inspectable spatial entity-relation process.
```

### 16.2 6.1 强调 monitoring-only baselines 的不足

建议写：

```text
Monitoring-only baselines can predict response variables, but they cannot organise TSP voxels, TBM surface components, and candidate rock–machine relations into a spatially inspectable structure.
```

### 16.3 6.2 保留

6.2 解释 prediction gains limited 的原因，必须保留：

- small test sets；
- mileage-ordered extrapolation；
- strong temporal autocorrelation；
- TSP and monitoring scale mismatch。

### 16.4 6.3 增加独立验证边界

建议增加：

```text
The current relevance diagnostics are model-derived and have not yet been independently validated by contact measurements, shield-pressure distribution, or expert-labelled interaction events.
```

---

## 17. Conclusion 修改意见

### 17.1 结论不要说 forecasting dominance

保留：

- remains competitive；
- no broad statistically detectable accuracy improvement；
- consistent evidence is interpretive。

### 17.2 最后一句替换

如果现在写：

```text
surface-based diagnostics
```

建议改成：

```text
component- and chainage-resolved diagnostic relevance
```

更稳。

---

## 18. 术语统一表

| 不建议使用 | 建议使用 | 原因 |
|---|---|---|
| physical plausibility | geometric plausibility | 没有真实力学接触建模 |
| hotspot map | diagnostic relevance view | 避免暗示真实风险/接触热点 |
| graph-to-surface interpretation | diagnostic relevance aggregation | 降低物理反演暗示 |
| contact relevance | response-consistent relevance | relevance 是预测任务监督下得到的 |
| risk map | diagnostic relevance field | 没有风险标签或风险标定 |
| accuracy gain | numerical difference / response-consistency difference | 实验提升很小 |
| prediction superiority | forward prediction check | 避免过度主张 |
| surface projection | component-chainage aggregation | 更贴合 Figure 9 的实际内容 |

---

## 19. 最终建议目录

```text
1. Introduction

2. Related work
   2.1 TBM response prediction and jamming-related analysis
   2.2 Spatial representation of geological conditions in tunnel excavation
   2.3 Graph-based spatiotemporal prediction and spatial interaction modelling

3. Methodology
   3.1 Chainage-referenced rock–TBM representation
   3.2 Geometry-constrained rock–machine relation graph sequence
   3.3 Response-supervised graph-sequence learning and diagnostic relevance

4. Case studies and experimental design
   4.1 Case roles and data sources
   4.2 Excavation-step samples and graph settings
   4.3 Baselines and graph ablation settings
   4.4 Evaluation and reporting principle

5. Results
   5.1 Forward prediction check across case roles
   5.2 Geometry ablation and response consistency
   5.3 Spatial organisation of learned diagnostic relevance
   5.4 Component-resolved diagnostic relevance along chainage
   5.5 Robustness and remaining checks

6. Discussion
   6.1 Added value of spatial entity-relation representation
   6.2 Why prediction gains are limited
   6.3 Interpretation boundary and future validation

7. Conclusion
```

---

## 20. 修改优先级清单

### 第一优先级：必须马上改

1. Figure 1 删除虚构数值。
2. Figure 1 / Figure 3 的 ablation 项与 Table 3 对齐。
3. 删除或移走 Figure 2。
4. 全文将 hotspot / graph-to-surface / surface projection 降级为 diagnostic relevance aggregation。
5. 全文将 physical plausibility 改为 geometric plausibility。
6. 3.1 删除五阶段 workflow。
7. 5.2 中 accuracy gain 改为 numerical difference。

### 第二优先级：建议尽快改

1. Algorithm 3 移到 Supplementary。
2. 4.2 训练细节移到 Appendix。
3. Figure 5 删 high/low role matrix。
4. Figure 10 移到 Supplementary。
5. 6.1 减少结果复述，转向解释空间表示价值。

### 第三优先级：润色阶段改

1. 统一 diagnostic relevance 术语。
2. 图注全部增加 interpretation boundary。
3. Related work 每节末尾增加 gap sentence。
4. Conclusion 再压缩，避免重复 Discussion。

---

## 21. 最终一句话定位

最终全文应收敛到下面这句话：

```text
This study does not claim broad forecasting dominance or physical contact recovery. Instead, it proposes a chainage-referenced rock–TBM interaction graph-sequence representation that constrains candidate rock–machine relations by geometry and uses response prediction to derive spatially organised diagnostic relevance.
```

中文理解为：

> 本文不主张普遍预测优势，也不主张恢复真实物理接触；本文提出的是一种里程参照下的岩–机交互图序列表达方法，通过几何约束限定候选岩–机关系，并借助响应预测任务获得空间组织化的诊断相关性。
