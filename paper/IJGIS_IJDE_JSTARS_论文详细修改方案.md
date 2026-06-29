# IJGIS / IJDE / J-STARS 审稿视角下的详细修改方案

> 适用对象：当前稿件《Geometry-Constrained Rock–TBM Spatial Interaction Graphs for Chainage-Resolved Diagnostic Analysis of Excavation Responses》  
> 目标期刊：International Journal of Geographical Information Science (IJGIS)、International Journal of Digital Earth (IJDE)、IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (J-STARS)  
> 总体判断：**Major Revision；严格看接近 Reject & Resubmit**  
> 核心建议：**不要再回到 GNN-GRU 深度预测主线，而应把论文坐实为“链式里程参考下的岩体体素—TBM部件—候选作用关系空间图数据模型与残差一致性诊断工作流”。**

---

## 0. 总体审稿判断

### 0.1 当前稿件的优点

当前稿件最重要的进步是：没有继续把研究包装成“GNN-GRU 预测模型”或“深度学习预测优于 LSTM”，而是将贡献收缩到更稳妥的方向：

- 构建 **chainage-referenced rock–TBM spatial interaction graph**；
- 将 **TSP-derived rock voxels** 与 **parameterized TBM surface components** 表达为异构空间实体；
- 用 active-zone membership、distance decay、surface-normal compatibility 和 excavation state 约束候选岩—机关系；
- 基于图序列聚合得到 **component–chainage spatial interaction descriptors**；
- 通过描述符与 **persistence residuals** 的一致性进行诊断验证；
- 明确不声称恢复接触力、接触压力、卡机概率或广义预测优势。

这个方向对 IJGIS / IJDE 有潜力，因为它的核心不再是单纯工程预测，而是：

> 一种面向移动工程装备与地下地质体之间动态空间关系的实体—关系表达与诊断工作流。

### 0.2 当前稿件的核心问题

尽管方向正确，但目前稿件仍存在几个决定性短板：

1. **实验样本量过小**  
   BSLL h=1 测试样本只有 8 个，BSLL h=3 只有 7 个，SJLS h=1 只有 17 个。以这样的样本量，仅靠 Spearman 相关矩阵不足以支撑强结论。

2. **评价证据过弱**  
   当前主要证据是 \(I_c(t)\) 与 persistence residual 的 Spearman 相关，但缺少 null model、ablation、permutation test、detrending 等可靠性验证。

3. **图模型贡献还没有充分发挥**  
   文中定义了岩—岩边 \(E^{rr}\)、机—机边 \(E^{mm}\)、岩—机边 \(E^{rm}\)，但结果主要只用了岩—机边上的加权平均异常值。审稿人可能质疑：这是否只是几何邻域加权统计，而不是图模型。

4. **TSP 异常分数 \(q_i\) 过于单一**  
   主方法只使用 P-wave velocity 构造低速异常分数，但 BSLL 数据包含 Vp、Vs、density、dynamic Young’s modulus、velocity ratio、Poisson’s ratio 等属性。若不做多属性敏感性分析，会被质疑指标选择任意。

5. **几何建模和复现细节不足**  
   需要补充 TBM 参数化几何尺寸、刀盘/护盾采样规则、每个部件节点数、TSP 坐标与盾首里程对齐方式、active zone 具体几何定义等。

6. **图件质量不够期刊级**  
   Figure 1、Figure 3、Figure 5 等图件存在信息拥挤、字体过小、流程图过简、图形表达不够专业的问题。

---

## 1. 总策略：论文定位必须稳定下来

### 1.1 不建议回到 GNN-GRU 主线

当前稿件最不应该做的是重新强化 GNN-GRU 或复杂深度预测模型。原因如下：

- 当前有效样本量太少；
- 没有充分的训练/验证/测试规模支撑深度模型；
- 若做 GNN-GRU，对比 LSTM、GRU、XGBoost、TGCN 等基线时很容易暴露统计不稳；
- IJGIS/IJDE 审稿人不会因为用了 GNN 就认可，反而会要求完整预测基线、消融和泛化验证；
- 当前更有说服力的贡献是空间实体—关系建模，而不是预测精度提升。

### 1.2 最终论文应定位为

建议全文统一成如下定位：

> 本文不是提出一个新的 TBM 响应深度预测模型，而是提出一种链式里程参考下的岩—机空间实体关系图模型。该模型将 TSP 地质体素、TBM 几何部件和监测响应统一到掘进里程坐标系下，通过 active zone、距离衰减、法向相容性和开挖状态约束候选岩—机关系，形成可审计、可追溯、可按部件和里程聚合的空间交互描述符，并通过与 persistence residual 的一致性检验其诊断信息。

### 1.3 需要贯穿全文的核心逻辑闭环

最终论文必须形成如下闭环：

```text
监测曲线只能表达响应的时间惯性
    ↓
TSP 体素只能表达地质异常的空间分布
    ↓
二者都不能显式回答“哪个地质体素与哪个 TBM 部件在某里程形成几何相容的候选作用关系”
    ↓
因此需要链式里程参考下的岩—机空间实体关系图
    ↓
该图生成部件—里程尺度的几何加权地质异常描述符
    ↓
若描述符与 persistence residual 一致，说明图表达提供了监测曲线和普通地质均值之外的空间诊断信息
```

---

## 2. 期刊匹配与投稿策略

### 2.1 IJGIS

#### 当前匹配度

中等，有潜力，但必须强化 GIScience 方法贡献。

#### IJGIS 审稿人会关注

- 是否提出了有普适意义的空间数据模型；
- 是否是空间实体、空间关系、拓扑/邻域/几何约束建模问题；
- 图是否只是算法工具，还是空间关系表达框架；
- 方法是否能迁移到其他移动工程装备—环境交互问题；
- 实验是否足够支撑空间关系表达的有效性。

#### IJGIS 修改重点

- 强化 spatial entity-relation graph 的数据模型贡献；
- 补充空间关系定义、图模式、图查询和追溯功能；
- 不要只讲 TBM 工程案例；
- 需要证明图表达带来了普通时间序列和普通地质异常均值之外的信息。

#### IJGIS 风险

若实验仍只停留在两个短区间、少量 Spearman 相关，容易被认为是工程案例诊断，而非 GIScience 方法论文。

---

### 2.2 IJDE

#### 当前匹配度

较高，是最推荐的投稿方向。

#### IJDE 审稿人会关注

- 地下工程数字孪生、三维地质表达、动态空间数据融合；
- 多源数据统一表达；
- 体素地质模型、TBM 几何体、监测数据的时空耦合；
- 是否有可视化、诊断、工程应用价值；
- 方法是否具有数字地球/数字孪生系统表达意义。

#### IJDE 修改重点

- 强化 TSP 三维地质体、TBM 几何体、里程动态更新；
- 突出三维空间交互表达；
- 可增加一个轻量原型系统或三维可视化查询示例；
- 讨论中可以使用 digital-twin-oriented TBM excavation analysis，但不要泛泛包装。

#### IJDE 风险

若只保留相关矩阵而缺少图可视化、空间追溯和工程解释，仍会显得结果不够实。

---

### 2.3 J-STARS

#### 当前匹配度

较低，不建议首投。

#### 原因

- J-STARS 更关注遥感、地球观测、传感器反演、空间观测数据处理；
- 当前稿件的遥感/地球观测主线不足；
- TSP 虽属于地球物理探测，但稿件没有以 TSP 反演、波速场重建、观测不确定性传播为主线；
- 当前更像地下工程空间图建模，不是典型 J-STARS 论文。

#### 如果必须投 J-STARS，需要转向

- 强化 TSP / 地球物理探测数据的三维重构；
- 增加 TSP 波速异常识别与监测响应耦合；
- 加入探测数据不确定性；
- 以地球观测/地下感知数据融合为主线，而不是空间关系图。

---

## 3. 题目修改方案

### 3.1 当前题目

```text
Geometry-Constrained Rock–TBM Spatial Interaction Graphs
for Chainage-Resolved Diagnostic Analysis of Excavation Responses
```

### 3.2 当前题目的问题

优点：

- 表达了 geometry-constrained；
- 表达了 rock–TBM spatial interaction graphs；
- 表达了 chainage-resolved diagnostic analysis；
- 没有夸大为 risk prediction 或 jamming forecasting。

问题：

- 稍长；
- “Excavation Responses” 偏泛；
- 对 IJGIS 来说 spatial representation 贡献不够突出；
- 对 IJDE 来说 component-level 工程诊断不够突出。

### 3.3 推荐题目

#### 推荐题目 1：偏 IJGIS

```text
A Geometry-Constrained Spatial Graph Representation for Chainage-Resolved Rock–TBM Interaction Diagnosis
```

优点：

- 突出 spatial graph representation；
- 更像 GIScience 方法论文；
- 适合强调空间实体关系建模。

缺点：

- 工程对象略弱；
- component-level 诊断没有出现在题目中。

#### 推荐题目 2：偏 IJDE，最推荐

```text
Chainage-Referenced Rock–TBM Spatial Interaction Graphs for Component-Level Diagnosis of Excavation Responses
```

优点：

- 保留 rock–TBM spatial interaction graphs；
- 强调 chainage-referenced；
- 强调 component-level diagnosis；
- 工程对象和空间表达比较平衡；
- 适合 IJDE，也不排斥 IJGIS。

缺点：

- geometry-constrained 没有放进题目，但可在摘要和方法中强化。

#### 推荐题目 3：更稳妥

```text
Geometry-Constrained Rock–TBM Interaction Graphs for Component–Chainage Diagnostic Analysis in Tunnel Excavation
```

优点：

- 更紧凑；
- 保留 geometry-constrained；
- 明确 component–chainage diagnostic analysis。

缺点：

- spatial representation 感稍弱。

### 3.4 最终建议

优先采用：

```text
Chainage-Referenced Rock–TBM Spatial Interaction Graphs for Component-Level Diagnosis of Excavation Responses
```

如果投 IJGIS，可改为：

```text
A Geometry-Constrained Spatial Graph Representation for Chainage-Resolved Rock–TBM Interaction Diagnosis
```

---

## 4. 摘要修改方案

### 4.1 当前摘要优点

当前摘要已经正确做到：

- 指出现有 monitoring-based TBM response models 保留时间依赖，但不能显式表达地质单元和 TBM 部件之间的空间关系；
- 提出 chainage-referenced rock–TBM spatial interaction graph；
- 说明 TSP-derived rock voxels 和 TBM surface components 被形式化为异构空间实体；
- 候选关系受 active zone、distance decay、surface-normal compatibility、excavation state 约束；
- 描述符与 persistence residuals 比较；
- 明确贡献不是 deep prediction model，而是 spatial entity-relation representation and residual-consistency diagnostic workflow。

### 4.2 当前摘要问题

- 防御性表述稍多；
- “不是什么”的表达过于集中；
- 实验验证描述偏弱；
- 没有突出“component-level”和“chainage-resolved”的诊断价值；
- 若补充 null model、permutation、ablation 后，应在摘要中体现。

### 4.3 建议摘要结构

建议摘要按照四段逻辑重写：

#### 第 1 句：问题背景

```text
TBM excavation is a moving rock–machine interaction process in which geological anomalies, machine geometry, and operational state jointly shape excavation responses.
```

中文含义：

> TBM 掘进是一个移动岩—机作用过程，地质异常、机器几何和运行状态共同影响掘进响应。

#### 第 2 句：现有不足

```text
Monitoring curves can capture temporal response persistence, but they do not explicitly encode which geological units are spatially adjacent to which TBM components or how these relations evolve with chainage.
```

中文含义：

> 监测曲线可以反映响应惯性，但不能显式编码地质体素与 TBM 部件的空间邻近关系及其随里程的演化。

#### 第 3–4 句：方法

```text
This study proposes a chainage-referenced rock–TBM spatial interaction graph that formalises TSP-derived rock voxels and parameterised TBM surface samples as heterogeneous spatial entities. Candidate rock–machine relations are screened using active-zone membership, distance decay, surface-normal compatibility, and excavation-state constraints.
```

#### 第 5 句：描述符

```text
Based on the resulting graph sequence, component–chainage diagnostic descriptors are derived by aggregating fixed-reference geological anomaly scores over geometry-weighted candidate relations.
```

#### 第 6 句：评价方式

如果补充统计检验后，可以写：

```text
The descriptors are evaluated against persistence residuals of TBM monitoring responses using descriptor–residual association, null-model comparisons, and sensitivity checks.
```

若还没补，则只能写：

```text
The descriptors are evaluated through their consistency with persistence residuals of TBM monitoring responses.
```

#### 第 7 句：案例结果

```text
Two tunnel cases demonstrate that the proposed graph representation organises rock–machine relations into interpretable component–chainage patterns and provides diagnostic information not explicitly encoded by monitoring curves alone.
```

#### 第 8 句：边界

```text
The proposed descriptors should be interpreted as geometry-weighted diagnostic summaries rather than measured contact forces, calibrated jamming probabilities, or universal operational risk rules.
```

### 4.4 摘要中不建议出现的表述

避免：

```text
broad forecasting superiority
```

避免反复说：

```text
not contact force, not contact pressure, not jamming probability, not calibrated risk
```

可以保留一次边界说明即可。

---

## 5. 引言修改方案

### 5.1 引言总体结构

建议将引言改成 4 个紧凑段落：

1. **TBM 掘进是移动岩—机空间作用过程**；
2. **现有监测模型和地质预报的断裂**；
3. **图作为空间实体—关系模型的必要性**；
4. **本文贡献**。

---

### 5.2 引言第 1 段：从工程过程引出空间问题

#### 目的

不要一开始就讲模型，而要先把“空间作用过程”立住。

#### 建议逻辑

- TBM 在连续推进过程中与前方和周边围岩发生作用；
- 地质异常进入或离开掌子面/护盾作用区；
- 相同的监测响应异常可能对应不同空间关系；
- 因此需要部件—里程尺度解释。

#### 可写内容

```text
TBM excavation is not only a temporal monitoring process but also a moving spatial interaction process. As the machine advances, geological units ahead of the face and around the shield continuously enter and leave the potential rock–machine interaction zone. Abnormal thrust, torque, penetration, advance rate, or shield-pressure behaviour may therefore reflect not only temporal response persistence but also the evolving spatial configuration between geological anomalies and TBM components.
```

#### 中文逻辑

> TBM 掘进不是单纯时间序列，而是移动空间作用过程。随着机器推进，掌子面前方和护盾周边的地质体不断进入或退出潜在岩—机作用区。推力、扭矩、贯入度、推进速度、护盾压力等异常，不仅可能来自响应惯性，也可能来自地质异常与 TBM 部件空间构型变化。

---

### 5.3 引言第 2 段：现有方法不足

#### 不要这样写

不要泛泛批评：

- LSTM 不好；
- GNN 更高级；
- 传统模型不能处理复杂非线性。

这样会显得像普通预测论文。

#### 应该这样写

强调：

- monitoring models 解决时间依赖；
- TSP/geological models 解决地质空间分布；
- 但缺少统一空间实体关系表达；
- 因此无法回答 component-level spatial relation questions。

#### 可写内容

```text
Existing monitoring-based models are useful for short-term response prediction because TBM responses often show strong persistence. However, they usually encode geological and operational factors as feature vectors or ordered sequences. Such representations do not explicitly preserve which TSP voxel is close to the cutterhead or shield at a given excavation step, whether this proximity is geometrically compatible with the local surface normal, or how the candidate rock–machine relation set changes as excavation advances.
```

#### 中文逻辑

> 现有监测预测模型有用，因为 TBM 响应短期内具有强惯性。但这些方法通常将地质和运行因素编码为特征向量或时间序列，无法显式保留某一 TSP 体素是否靠近刀盘/护盾、这种邻近是否与表面法向相容、候选岩—机关系如何随里程变化。

---

### 5.4 引言第 3 段：为什么用图，但不是为了深度学习

#### 这是全文最关键的论证

要明确：

- 图不是因为 GNN 流行才用；
- 图是空间实体关系问题的自然表达；
- 图可以表达异构实体、邻接、拓扑、候选作用关系；
- 先有空间关系图，再谈是否学习。

#### 可写内容

```text
A graph is needed here not primarily as a neural network backbone, but as a spatial data model. Rock voxels, TBM surface samples, rock–rock adjacency, machine-surface adjacency, and rock–machine candidate relations are heterogeneous spatial entities and relations. A graph provides an auditable way to organise these entities and relations before any predictive model is introduced.
```

#### 中文逻辑

> 本文使用图，不是因为要套用图神经网络，而是因为岩体体素、TBM表面采样点、岩体邻接、机器表面邻接和岩—机候选作用关系天然构成异构空间实体关系问题。图提供了一种可审计、可追溯的空间关系组织方式。

---

### 5.5 引言第 4 段：贡献

建议贡献写成三条：

#### Contribution 1

```text
A chainage-referenced spatial entity model that places TSP-derived rock voxels, TBM surface components, and monitoring responses in a common excavation coordinate system.
```

中文：

> 构建统一里程坐标下的 TSP 地质体素—TBM表面部件—监测响应空间实体模型。

#### Contribution 2

```text
A geometry-constrained rock–machine candidate relation graph using active-zone filtering, distance decay, surface-normal compatibility, and excavation-state constraints.
```

中文：

> 提出受 active zone、距离衰减、法向相容性和开挖状态约束的岩—机候选关系图。

#### Contribution 3

```text
Component–chainage spatial interaction descriptors and a residual-consistency diagnostic workflow that compares geometry-weighted TSP anomaly summaries with persistence residuals of TBM responses.
```

中文：

> 构建部件—里程尺度空间交互描述符，并提出与 TBM 响应 persistence residual 对比的残差一致性诊断工作流。

#### 若补充实验后可增强为

```text
... evaluated through residual association, null-model comparisons, and sensitivity checks.
```

---

## 6. 相关工作修改方案

### 6.1 建议重构为四节

当前 Related work 可改成：

```text
2.1 Monitoring-based TBM response modelling
2.2 Spatial representation of geological conditions in tunnel excavation
2.3 Graph data models for spatial entity–relation representation
2.4 Diagnostic descriptors and residual-based evaluation
```

---

### 6.2 2.1 Monitoring-based TBM response modelling

#### 目标

说明现有 TBM 预测模型解决的是“时间依赖”和“监测响应预测”，但不是“空间关系解释”。

#### 应覆盖内容

- 传统机器学习：SVR、Random Forest、XGBoost、GEP 等；
- 时间序列模型：RNN、LSTM、GRU；
- TBM 响应预测对象：advance rate、penetration、torque、thrust、shield pressure；
- 这些方法把输入看作监测序列或特征向量；
- 它们不能显式表达 TSP 体素与 TBM 部件之间的关系。

#### 本节结尾建议

```text
These studies are effective for modelling temporal dependence in TBM monitoring data, but they usually do not preserve explicit spatial relations between geological units and TBM components. This limits their use when the objective is not only response prediction but also component-level spatial diagnosis.
```

---

### 6.3 2.2 Spatial representation of geological conditions in tunnel excavation

#### 目标

说明 TSP 和三维地质模型能表达地质空间异常，但不能自动建立岩—机关系。

#### 应覆盖内容

- TSP / HSP / advance geological prediction；
- 三维地质模型；
- voxel field；
- chainage-referenced geological representation；
- geological anomaly field；
- 但体素场本身不说明哪个 TBM 部件受影响。

#### 本节结尾建议

```text
A voxelised TSP field provides spatial support for geological anomalies, but it does not by itself determine which geological units are relevant to the cutterhead or shield at a given chainage. Bridging geological voxel fields and machine surface geometry is therefore necessary for spatially explicit rock–machine interaction diagnosis.
```

---

### 6.4 2.3 Graph data models for spatial entity–relation representation

#### 目标

将本文从“算法图神经网络”转向“空间图数据模型”。

#### 应覆盖内容

- spatial graph；
- topology；
- adjacency；
- proximity；
- spatial relation；
- heterogeneous graph；
- graph database / graph data model；
- spatial interaction graph；
- GIScience 中图模型用于表达复杂空间关系的传统。

#### 本节结尾建议

```text
The relevance of graphs in this study lies not in black-box graph learning, but in their capacity to formalise heterogeneous spatial entities and constrained candidate relations in an inspectable data model.
```

---

### 6.5 2.4 Diagnostic descriptors and residual-based evaluation

#### 这是建议新增的小节

#### 目标

提前解释为什么用 persistence residual，而不是 RMSE/MAE 预测指标。

#### 应覆盖内容

- persistence 是短期 TBM 响应的强基线；
- residual 代表时间惯性未解释的响应变化；
- 描述符与 residual 的一致性用于诊断；
- 不是因果归因；
- 不是 operational risk rule。

#### 可写内容

```text
For compact diagnostic intervals, direct forecasting comparisons may be unstable and may obscure the representation question. A residual-based evaluation reframes the task: instead of asking whether a spatial graph outperforms persistence in absolute prediction, it asks whether the spatial descriptor encodes information consistent with the response change left after temporal persistence.
```

---

## 7. 方法部分修改方案

### 7.1 方法部分总结构

建议方法部分改成：

```text
3.1 Chainage alignment and spatial entity definition
3.2 Parameterised TBM surface representation
3.3 Geometry-constrained candidate relation graph
3.4 Component–chainage diagnostic descriptors
3.5 Persistence residual and descriptor-consistency evaluation
3.6 Null-model and sensitivity checks
```

---

### 7.2 增加“空间实体与关系数据模型表”

#### 必须新增 Table：Spatial entities and relations

| 对象 | 符号 | 空间含义 | 关键属性 | 后续用途 |
|---|---|---|---|---|
| Rock voxel | \(v_i^r\) | TSP 体素单元 | coordinate, Vp, Vs, E, ν, ρ, active state | geological anomaly |
| TBM surface node | \(v_j^m\) | 刀盘/护盾表面采样点 | coordinate, normal, component label | component exposure |
| Rock–rock edge | \(e_{ii'}^{rr}\) | 岩体空间邻接 | 6/18/26-neighbourhood | continuity / traceability |
| Machine–machine edge | \(e_{jj'}^{mm}\) | TBM 表面结构邻接 | radial/circumferential/axial adjacency | component topology |
| Rock–machine edge | \(e_{ij}^{rm}\) | 候选岩—机作用关系 | distance, normal compatibility, active-zone flag, weight | descriptor aggregation |

#### 作用

该表可以解决审稿人最可能提出的问题：

> 如果最后只是算一个加权平均，为什么需要图？

有了这个表，论文就能说明：

- 图不是单一邻域统计；
- 图是包含岩体、机器、空间邻接、表面结构、候选作用关系的实体—关系表达；
- \(I_c(t)\) 只是从图中派生出的一个诊断描述符；
- 未来可支持查询、追溯、学习和数字孪生表达。

---

### 7.3 坐标对齐必须细化

#### 当前问题

当前只写了类似：

```text
x_local(t) = ShieldMileage(t) - ShieldMileage_start
```

这不足以复现。

#### 必须回答的问题

- TSP 坐标原点在哪里？
- TSP 的 X 方向是否与 TBM 推进方向一致？
- TSP 数据的里程起点如何对应盾首里程？
- 是否存在偏移 \(\delta\)？
- 坐标对齐误差如何处理？
- 如果 TSP 坐标与监测里程不一致，会如何影响结果？

#### 建议新增小节：Chainage alignment and coordinate uncertainty

##### 定义 TSP 局部坐标

\[
x_i^{TSP}=X_i-X_{min}
\]

##### 定义 TBM 当前掌子面或盾首位置

\[
x_{face}(t)=Mileage(t)-Mileage_0
\]

##### 定义统一坐标

\[
\tilde{x}_i=x_i^{TSP}+\delta
\]

其中：

- \(\delta\) 表示 TSP 解释里程与 TBM 监测里程之间的偏移；
- 若有施工记录，\(\delta\) 来自 TSP 起点里程；
- 若无精确记录，可通过地质揭示或异常段位置进行对齐。

##### 对齐误差敏感性

建议加入：

\[
\delta \in \{-1,0,+1\}\,m
\]

或：

\[
\delta \in \{-0.5,0,+0.5\}\,m
\]

用于检查描述符相关性是否对里程对齐误差过敏。

#### 建议正文表述

```text
Because the proposed descriptors are chainage-resolved, coordinate alignment between the TSP voxel field and TBM monitoring mileage is a critical preprocessing step. We therefore treat the TSP-to-monitoring offset as an explicit alignment parameter and examine its influence in a sensitivity check.
```

---

### 7.4 TBM 参数化几何建模必须补充

#### 当前问题

稿件提到 TBM surface discretised into cutterhead、front shield、middle shield、tail shield，但缺少：

- TBM 直径；
- 刀盘厚度；
- 护盾长度；
- 前盾/中盾/尾盾分段边界；
- 表面采样分辨率；
- 每个部件节点数；
- BSLL 与 SJLS 节点数差异原因。

#### 必须新增 Table：Parameterized TBM geometry and sampling

| Component | Geometry | Axial range | Sampling rule | Node count | Notes |
|---|---|---|---|---:|---|
| Cutterhead | disk | \(x=x_f\) | radial × circumferential | \(n_1\) | face-contact support |
| Front shield | cylinder | \(x_f-L_1\) to \(x_f\) | axial × circumferential | \(n_2\) | shield-side support |
| Middle shield | cylinder | \(x_f-L_1-L_2\) to \(x_f-L_1\) | axial × circumferential | \(n_3\) | shield-side support |
| Tail shield | cylinder | \(x_f-L_s\) to \(x_f-L_1-L_2\) | axial × circumferential | \(n_4\) | shield-side support |

#### 如果没有真实 CAD 模型

可写：

```text
A simplified parameterised TBM geometry was used because detailed CAD geometry was unavailable. The simplification preserves the component-level spatial support needed for cutterhead–shield diagnostic comparison, rather than exact mechanical contact surfaces.
```

#### 如果有真实 TBM 参数

必须写清楚：

- 刀盘半径 \(R_c\)；
- 护盾半径 \(R_s\)；
- 刀盘厚度；
- 前盾长度 \(L_f\)；
- 中盾长度 \(L_m\)；
- 尾盾长度 \(L_t\)；
- 采样间距，例如 0.5 m；
- 刀盘按径向和周向采样；
- 护盾按轴向和周向采样。

---

### 7.5 Active zone 必须给出具体几何定义

#### 当前定义

\[
\Omega_t = \Omega_t^{face} \cup \Omega_t^{shield}
\]

但还不够具体。

#### 建议定义 face active zone

\[
\Omega_t^{face}=
\left\{
v_i^r \mid
0 \le x_i-x_{face}(t) \le L_f,\ 
\sqrt{y_i^2+z_i^2}\le R_f
\right\}
\]

其中：

- \(L_f\)：掌子面前方地质异常影响长度；
- \(R_f\)：掌子面作用半径；
- \(x_{face}(t)\)：当前掌子面或盾首位置。

#### 建议定义 shield active zone

\[
\Omega_t^{shield}=
\left\{
v_i^r \mid
-L_s \le x_i-x_{face}(t) \le 0,\ 
R_s \le \sqrt{y_i^2+z_i^2}\le R_s+\tau_{zone}
\right\}
\]

其中：

- \(L_s\)：护盾总长度；
- \(R_s\)：护盾半径；
- \(\tau_{zone}\)：护盾周边缓冲距离。

#### 解释

```text
The face zone describes geological units ahead of the cutterhead that may contribute to cutting resistance, whereas the shield zone describes geological units around the shield skin that may contribute to shield-side confinement or squeezing. The two zones are combined for graph construction but remain distinguishable through TBM component labels.
```

---

### 7.6 岩—机候选边公式保留，但要解释其边界

#### 当前候选边条件

\[
d_{ij}(t)\le \tau_{edge}
\]

\[
\kappa_{ij}(t)\ge \eta_{min}
\]

\[
c_i\in \Omega_t
\]

\[
s_i(t)=1
\]

其中：

\[
d_{ij}(t)=\|c_i-p_j(t)\|
\]

\[
\kappa_{ij}(t)=
\max\left(
0,
\frac{n_j(t)^\top(c_i-p_j(t))}{d_{ij}(t)+\epsilon}
\right)
\]

#### 权重

\[
w_{ij}^{rm}(t)=\exp\left(-\frac{d_{ij}(t)}{\tau_{edge}}\right)\kappa_{ij}(t)
\]

#### 必须强调

该权重不是：

- 接触力；
- 接触压力；
- 摩阻力；
- 卡机概率；
- 工程风险真值。

它只是：

> geometry-compatible candidate relation intensity

#### 建议正文表达

```text
The weight should be interpreted as a geometry-compatible candidate-relation intensity. It increases when a rock voxel is closer to the TBM surface node and better aligned with the outward surface normal. It does not represent measured contact force or contact pressure.
```

---

### 7.7 \(E^{rr}\) 和 \(E^{mm}\) 不要只定义不用

#### 当前风险

文中定义完整图：

\[
G_t=(V_t^r\cup V_t^m,E_t^{rr}\cup E_t^{mm}\cup E_t^{rm})
\]

但结果主要使用 \(E_t^{rm}\)。审稿人会问：

> 如果岩—岩边和机—机边没有用于计算，为什么需要完整图？

#### 解决方案 A：保留完整图，并增加图查询/追溯示例

建议新增一个图追溯实验：

```text
For a selected diagnostic chainage, the graph was queried to retrieve the top-contributing rock voxels for the most response-consistent TBM component.
```

定义边贡献：

\[
C_{ij}^{c}(t)=
\frac{
w_{ij}^{rm}(t)q_i
}{
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)q_i+\epsilon
}
\]

输出表：

| Chainage | Component | Rock voxel coordinate | Vp | Distance | Normal compatibility | Contribution |
|---|---|---|---:|---:|---:|---:|

作用：

- 证明图可以追溯到具体岩体体素；
- 证明描述符不是黑箱；
- 证明空间图支持诊断查询；
- 增强 IJGIS/IJDE 说服力。

#### 解决方案 B：弱化完整图

如果不增加追溯实验，则应承认：

```text
Although the graph formally includes rock–rock and machine-surface adjacency for structural completeness, the present diagnostic descriptor focuses on geometry-constrained rock–machine relations. Learning over the full heterogeneous graph is left for future work.
```

不推荐方案 B，因为会削弱图模型贡献。

#### 最终建议

采用 **方案 A**。

---

## 8. 描述符修改方案

### 8.1 \(A_c(t)\) 必须增加归一化版本

#### 当前定义

\[
A_c(t)=
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)
\]

问题：

- 受部件表面积影响；
- 受节点采样密度影响；
- 受候选边数量影响；
- cutterhead 与 shield 不能直接比较。

#### 建议增加节点归一化暴露

\[
\bar{A}_c(t)=
\frac{1}{|V_c^m|}
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)
\]

#### 如果有面积信息，增加面积归一化

\[
A_c^{area}(t)=
\frac{1}{S_c}
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)
\]

#### 建议使用方式

- 原始 \(A_c(t)\)：作为 supporting descriptor 或 appendix；
- \(\bar{A}_c(t)\)：用于跨部件比较；
- \(I_c(t)\)：用于主诊断分析。

---

### 8.2 \(I_c(t)\) 保留为主描述符

当前定义：

\[
I_c(t)=
\frac{
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)q_i
}{
\sum_{j\in V_c^m}
\sum_{i:(i,j)\in E_t^{rm}}
w_{ij}^{rm}(t)+\epsilon
}
\]

解释：

- \(A_c(t)\)：部件几何暴露；
- \(I_c(t)\)：部件周边几何相容的地质异常强度；
- \(I_c(t)\) 是主诊断描述符；
- 它不是响应预测值；
- 它不是风险概率；
- 它不是接触压力。

---

### 8.3 \(q_i\) 必须做多属性敏感性分析

#### 当前 \(q_i\)

\[
q_i^{Vp}=
clip\left(
\frac{
Q_{95}^{train}(Vp)-Vp_i
}{
Q_{95}^{train}(Vp)-Q_{5}^{train}(Vp)+\epsilon
},
0,1
\right)
\]

优点：

- 透明；
- 可复现；
- 跨案例一致；
- 不需要人为权重。

问题：

- 只用 Vp 过于单一；
- 没解释为什么低 Vp 是主要异常；
- 没检验 Vs、E、Vp/Vs、Poisson’s ratio 等是否改变结果。

#### 建议增加三组敏感性

##### 1. Vp-only

保留当前主版本：

\[
q_i^{Vp}
\]

##### 2. Vs-only

\[
q_i^{Vs}=
clip\left(
\frac{
Q_{95}^{train}(Vs)-Vs_i
}{
Q_{95}^{train}(Vs)-Q_{5}^{train}(Vs)+\epsilon
},
0,1
\right)
\]

##### 3. Vp+Vs 或 Vp+Vs+E

简单平均：

\[
q_i^{multi}=
\frac{1}{M}
\sum_{m=1}^{M}
q_i^{(m)}
\]

其中 \(m\) 可以包括：

- low Vp anomaly；
- low Vs anomaly；
- low dynamic Young’s modulus anomaly。

如果没有充分工程解释，不建议纳入 Vp/Vs 和 Poisson’s ratio。

#### 结果表建议

| Descriptor | BSLL h=1 best \(|ρ|\) | BSLL h=3 best \(|ρ|\) | SJLS h=1 best \(|ρ|\) | Interpretation |
|---|---:|---:|---:|---|
| Vp-only | ... | ... | ... | main transparent descriptor |
| Vs-only | ... | ... | ... | sensitivity |
| Vp+Vs | ... | ... | ... | robust if consistent |
| Vp+Vs+E | ... | ... | ... | optional |

#### 讨论写法

如果结果稳定：

```text
The descriptor–residual pattern remains broadly consistent under alternative transparent anomaly definitions, suggesting that the diagnostic signal is not an artefact of using Vp alone.
```

如果结果不稳定：

```text
The sensitivity to anomaly definition indicates that the current evidence should be interpreted as Vp-based diagnostic consistency rather than a general geological-risk index.
```

---

## 9. 实验修改方案：必须补强

### 9.1 当前实验最大问题

当前结果主要依赖 Spearman 相关：

\[
\rho(I_c(t),e_{t+h}^{(k)})
\]

问题：

- 样本量太少；
- 多组件、多响应变量、多 horizon 组合导致多重比较；
- 当前 table 是 unadjusted；
- 没有 null model；
- 没有证明几何约束真的比普通地质均值有用；
- 没有排除里程趋势导致的假相关。

### 9.2 必须新增实验 1：Null-model comparisons

建议新增 Section：

```text
5.4 Null-model comparisons
```

#### Null 1：Chainage-only baseline

直接用里程 \(x(t)\) 与 residual 做 Spearman：

\[
\rho(x(t),e_{t+h}^{(k)})
\]

目的：

> 证明 \(I_c(t)\) 的相关性不是简单由里程趋势造成。

#### Null 2：Global active-zone Vp anomaly

不考虑 TBM 部件、不考虑岩—机几何边，只算 active zone 内普通平均异常：

\[
I_{global}(t)=
\frac{1}{|\Omega_t|}
\sum_{i\in \Omega_t}q_i
\]

目的：

> 证明部件级几何关系比普通地质异常均值更有诊断信息。

#### Null 3：Distance-only descriptor

去掉法向相容性：

\[
w_{ij}^{dist}(t)=
\exp\left(-\frac{d_{ij}(t)}{\tau_{edge}}\right)
\]

目的：

> 证明 surface-normal compatibility 不是多余的。

#### Null 4：Uniform-edge descriptor

只要满足连边条件，就等权：

\[
w_{ij}^{uniform}=1
\]

目的：

> 证明距离衰减和几何权重有贡献。

#### Null 5：Component-label shuffle

随机打乱 TBM surface node 的 component label，再计算 \(I_c(t)\)。

目的：

> 证明 component-level pattern 不是随机部件分配也能得到的结果。

#### Null 6：Rock voxel shuffle

随机打乱 \(q_i\) 在 rock voxels 上的位置，保持 \(q_i\) 分布不变。

目的：

> 证明结果依赖地质异常的空间位置，而不是异常值分布本身。

#### Null 7：Random edge rewiring

保持候选边数量不变，但随机重连 rock voxel 和 TBM node。

目的：

> 证明 geometry-constrained candidate relations 有必要。

#### 建议主文只保留前 5 个

前 5 个已经足够，Null 6 和 Null 7 可放附录。

#### 结果表建议

| Case | Diagnostic pair | Proposed \(I_c\) | Chainage only | Global Vp | Distance only | Uniform edge | Component shuffle |
|---|---|---:|---:|---:|---:|---:|---:|
| BSLL h=3 | Front shield–shield pressure | -0.750 | ... | ... | ... | ... | ... |
| SJLS h=1 | Cutterhead–shield pressure | -0.912 | ... | ... | ... | ... | ... |
| SJLS h=1 | Tail shield–shield pressure | 0.775 | ... | ... | ... | ... | ... |

---

### 9.3 必须新增实验 2：Permutation test

#### 为什么必须做

因为样本量太小，普通 Spearman 相关容易偶然很高。

#### 推荐方法：circular shift permutation

由于数据按里程排序，不能完全随机打乱。建议用 circular shift。

步骤：

1. 保持 \(I_c(t)\) 顺序不变；
2. 将 residual 序列循环平移 \(s\) 个位置；
3. 每次计算 Spearman；
4. 重复 \(B\) 次或遍历所有可能 shift；
5. 得到 null distribution；
6. 计算 empirical p-value：

\[
p=
\frac{
1+\sum_{b=1}^{B}I(|\rho_b|\ge |\rho_{obs}|)
}{
B+1
}
\]

#### 对小样本的解释

BSLL n=7 或 n=8 时，置换检验不能当作严格显著性检验，只能作为 stability check。

建议写：

```text
Due to the compact BSLL intervals, permutation results are interpreted as stability checks rather than formal confirmatory inference.
```

#### 图件建议

新增 Figure：

- 灰色柱状图：null distribution；
- 红色竖线：observed rho；
- 三个 panel：
  1. BSLL h=3 front shield–shield pressure；
  2. SJLS h=1 cutterhead–shield pressure；
  3. SJLS h=1 tail shield–shield pressure。

---

### 9.4 必须新增实验 3：Spatial detrending / partial association

#### 目的

排除里程趋势造成的假相关。

#### 方法 A：一阶差分

\[
\Delta I_c(t)=I_c(t)-I_c(t-1)
\]

\[
\Delta e(t)=e(t)-e(t-1)
\]

计算：

\[
\rho(\Delta I_c(t),\Delta e(t))
\]

缺点：

- 小样本下差分会进一步减少样本；
- 容易放大噪声。

#### 方法 B：linear detrending，推荐

分别对 descriptor 和 residual 拟合里程趋势：

\[
I_c(t)=a_0+a_1x(t)+\epsilon_I(t)
\]

\[
e(t)=b_0+b_1x(t)+\epsilon_e(t)
\]

然后计算：

\[
\rho(\epsilon_I(t),\epsilon_e(t))
\]

#### 建议结果表

| Case | Pair | Raw ρ | Detrended ρ | Interpretation |
|---|---|---:|---:|---|
| SJLS h=1 | Cutterhead–shield pressure | -0.912 | ... | robust / weakened |
| SJLS h=1 | Tail shield–shield pressure | 0.775 | ... | robust / weakened |
| BSLL h=3 | Front shield–shield pressure | -0.750 | ... | compact evidence |

#### 解释方式

如果 detrended 后仍稳定：

```text
The association remains after removing a linear chainage trend, suggesting that the descriptor–residual co-variation is not solely driven by monotonic mileage variation.
```

如果变弱：

```text
The weakened detrended association indicates that part of the descriptor–residual pattern reflects chainage-local geological contrast. This is expected because the descriptor is designed to encode chainage-resolved geological exposure rather than chainage-independent causality.
```

---

### 9.5 必须处理多重比较问题

当前 Table 3 是 full unadjusted matrix。必须调整解释。

#### 建议处理方式

主文不再把 full matrix 当作“primary result”，而是：

- full matrix 放 Appendix；
- 正文只报告 primary diagnostic pairs；
- full matrix 用于 exploratory overview；
- primary pairs 才做 permutation、detrending、null model。

#### Primary diagnostic pairs 的选择

不要事后 cherry-picking。必须给选择理由：

1. shield pressure 是与卡机/护盾受压最相关的监测响应；
2. cutterhead 和 shield components 是最直接岩—机作用部件；
3. SJLS 有最清晰 TSP velocity contrast；
4. BSLL h=3 是多步诊断 horizon。

#### 可定义 primary pairs

- SJLS h=1：cutterhead \(I_c(t)\) vs shield pressure residual；
- SJLS h=1：tail shield \(I_c(t)\) vs shield pressure residual；
- BSLL h=3：front shield \(I_c(t)\) vs shield pressure residual；
- 可选：SJLS h=1 tail shield vs penetration residual。

#### FDR 校正

如果继续报告完整矩阵，建议做 Benjamini–Hochberg FDR，但由于样本量小，结果可能不显著。因此更稳妥的写法是：

```text
The full matrix is reported as an exploratory diagnostic map. Confirmatory interpretation is restricted to the preselected primary diagnostic pairs, which are further examined using null-model and sensitivity checks.
```

---

## 10. 结果部分重构方案

### 10.1 建议结果结构

```text
5.1 Constructed graph objects and relation statistics
5.2 Component–chainage descriptor patterns
5.3 Descriptor–residual association
5.4 Null-model and ablation comparisons
5.5 Sensitivity to geometry thresholds and anomaly definitions
5.6 Graph-based traceability example
```

---

### 10.2 5.1 Constructed graph objects and relation statistics

#### 当前已有内容

当前 Table 2 报告 rock nodes、TBM nodes、candidate edges 等。

#### 必须补充解释

- BSLL 与 SJLS 为什么 TBM nodes 不同；
- candidate-edge share 中 shield 占比高是否因为护盾表面积更大；
- 是否使用归一化暴露；
- graph construction 是否跨案例一致。

#### 建议新增说明

```text
The larger shield share mainly reflects the larger shield surface support and longer axial exposure. Therefore, raw geometric exposure is not used for direct cross-component comparison unless node-normalised or area-normalised.
```

#### 建议表格扩展

| Case | Rock nodes | TBM nodes | Candidate edges | Cutterhead edge share | Shield edge share | Mean edges per TBM node | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| BSLL | ... | ... | ... | ... | ... | ... | ... |
| SJLS | ... | ... | ... | ... | ... | ... | ... |

---

### 10.3 5.2 Component–chainage descriptor patterns

#### 目标

展示 \(I_c(t)\) 的部件—里程空间结构。

#### 当前问题

Figure 4 只有 heatmap，缺少对应 residual 曲线。

#### 修改建议

每个 case 做上下两层：

- 上层：component–chainage \(I_c(t)\) heatmap；
- 下层：selected response residual line；
- 用阴影标出高 \(I_c(t)\) 区间；
- 标注 test chainage interval；
- 标注主要 component。

#### 图注应说明

```text
The heatmap shows the spatial diagnostic descriptor, not measured contact pressure or risk probability. The line plot shows the persistence residual used for residual-consistency evaluation.
```

---

### 10.4 5.3 Descriptor–residual association

#### 当前 Table 3 问题

- 全矩阵过大；
- 所有值 unadjusted；
- 小样本；
- 审稿人会质疑 cherry-picking。

#### 修改建议

正文只保留 primary diagnostic pairs：

| Case | Component | Response residual | Raw ρ | Detrended ρ | Permutation p | Interpretation |
|---|---|---|---:|---:|---:|---|
| SJLS h=1 | Cutterhead | Shield pressure | -0.912 | ... | ... | strongest diagnostic pair |
| SJLS h=1 | Tail shield | Shield pressure | 0.775 | ... | ... | shield-side co-variation |
| BSLL h=3 | Front shield | Shield pressure | -0.750 | ... | ... | compact evidence |

完整矩阵放 Appendix。

#### 说明方式

```text
The full component–response matrix is used as an exploratory diagnostic map. To avoid over-interpreting multiple unadjusted correlations, the following analysis focuses on preselected primary diagnostic pairs related to shield-pressure residuals and major TBM components.
```

---

### 10.5 5.4 Null-model and ablation comparisons

#### 重点

证明 proposed \(I_c(t)\) 比以下 baseline 更有诊断信息：

- chainage-only；
- global Vp；
- distance-only；
- uniform edge；
- component shuffle。

#### 结果解读模板

如果 proposed 最强：

```text
The proposed descriptor shows stronger descriptor–residual association than the chainage-only, global-anomaly, and geometry-ablated descriptors. This suggests that the diagnostic information is not solely explained by mileage trend or TSP anomaly magnitude, but depends on the geometry-constrained component-level rock–machine relation.
```

如果 proposed 与 distance-only 差不多：

```text
The similarity between the proposed and distance-only descriptors indicates that proximity dominates the diagnostic signal in this case, whereas normal compatibility provides limited additional information. This result should be reported as a case-specific finding rather than hidden.
```

如果 component shuffle 也强：

```text
The comparable component-shuffle result indicates that component-level interpretation is unstable in this interval. The claim should therefore be restricted to overall rock–machine exposure rather than component-specific diagnosis for this case.
```

---

### 10.6 5.5 Sensitivity to geometry thresholds and anomaly definitions

#### 现有 threshold sensitivity 保留

当前稿件已有 \(\tau_{edge}\) 和 \(\eta_{min}\) 敏感性分析，但需要：

- 明确每个 panel 对应哪个 descriptor–response pair；
- 说明不是预测精度；
- 说明是 descriptor stability。

#### 新增 anomaly definition sensitivity

比较：

- Vp-only；
- Vs-only；
- Vp+Vs；
- Vp+Vs+E。

#### 建议表述

```text
The sensitivity analysis evaluates whether the diagnostic association depends on a narrow parameter setting or a single geological-anomaly definition.
```

---

### 10.7 5.6 Graph-based traceability example

#### 新增原因

这是提升 IJGIS/IJDE 说服力的关键。

#### 内容

选择一个典型里程，例如 SJLS 中 shield-pressure residual 明显变化的位置，输出：

- 哪个部件 \(c\)；
- 哪些 rock voxel 贡献最大；
- 这些 voxel 的 Vp / Vs / E；
- 与 TBM node 的距离；
- 法向相容性；
- 贡献比例；
- 空间位置。

#### 表格

| Rank | Chainage | Component | Rock voxel coordinate | Vp | Vs | Distance | \(\kappa\) | Edge weight | Contribution |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

#### 图件

- 三维或剖面图；
- TBM 部件半透明；
- 高贡献 rock voxels 用颜色突出；
- candidate edges 用细线表示；
- 图例说明 \(C_{ij}^c(t)\)。

#### 作用

这个实验能把“描述符”变成“可追溯空间诊断”，比单纯相关矩阵更符合 IJGIS/IJDE。

---

## 11. 图件修改方案

### 11.1 Figure 1：总框架图

#### 当前问题

- 像普通流程图；
- 小字多；
- 空间图贡献不突出；
- 缺少空间实体—关系模型感。

#### 建议改成四栏

```text
(1) Chainage alignment
    TSP voxel field + TBM mileage + monitoring data

(2) Spatial entity graph
    rock voxels + TBM surface nodes + monitoring responses

(3) Geometry-constrained candidate relations
    active zone + distance decay + normal compatibility + excavation state

(4) Diagnostic descriptors
    A_c(t), I_c(t), persistence residual consistency
```

#### 图形设计建议

- 左侧画 TSP 体素场；
- 中间画 TBM 刀盘/护盾表面采样；
- 中间偏右画候选边；
- 右侧画 component–chainage heatmap 和 residual line；
- 少放公式；
- 公式放图注或正文；
- 颜色不要太花；
- 采用期刊风格，线条简洁、字体统一。

---

### 11.2 Figure 2：Active zone 与候选边构造

#### 当前问题

- 概念可以，但空间几何表达还不够强。

#### 修改建议

展示：

- \(\Omega_t^{face}\)；
- \(\Omega_t^{shield}\)；
- 推进方向；
- TBM cutterhead / shield；
- rock voxel；
- TBM surface node；
- \(d_{ij}\)；
- \(n_j\)；
- \(\kappa_{ij}\)；
- \(w_{ij}^{rm}(t)\)。

#### 最好做成剖面 + 局部放大图

左边：

- 整体剖面；
- face zone 和 shield zone。

右边：

- 一个 rock voxel 到 TBM node 的边；
- 标注距离和法向角。

---

### 11.3 Figure 3：案例与样本划分

#### 当前问题

- 信息多但不够清晰；
- train/validation/test 与 case role 没有充分视觉化；
- BSLL / SJLS 对比不够直观。

#### 修改建议

做成三部分：

1. 两个隧道案例的里程轴；
2. TSP source 和 test interval；
3. train / validation / test 样本数量。

#### 建议图中包含

| Case | TSP source | Horizon | Train/Val/Test | Test interval | Role |
|---|---|---|---|---|---|
| BSLL | project voxel field | h=1, h=3 | 30/6/8; 29/6/7 | 41/42–48 m | compact diagnostic case |
| SJLS | external velocity field | h=1 | 76/16/17 | 99–115 m | clear TSP contrast case |

---

### 11.4 Figure 4：描述符热力图 + residual 曲线

#### 当前问题

只有 heatmap，读者需要自己对照表格理解。

#### 修改建议

每个 case 一个 panel：

- 上：component–chainage \(I_c(t)\) heatmap；
- 下：shield pressure residual 或主要 response residual；
- 若多个 response，可只选 primary response；
- 用同一 chainage 横轴；
- 高异常区加阴影；
- 标注 primary diagnostic pair。

---

### 11.5 Figure 5：相关矩阵

#### 当前问题

- 三个矩阵横向挤压；
- 标签小；
- unadjusted matrix 过度强调；
- 审稿人会聚焦多重比较问题。

#### 修改建议

- full matrix 放 appendix；
- 正文只放 primary pair bar chart；
- 若保留矩阵：
  - 字体放大；
  - 未通过稳定性检验的格子淡化；
  - 通过 permutation/detrending 的格子加黑框；
  - 图注明确 exploratory。

---

### 11.6 Figure 6：参数敏感性

#### 当前问题

- 不知道每个 panel 对应哪个 pair；
- 读者难以理解颜色含义。

#### 修改建议

每个 panel 标题写清楚：

```text
SJLS h=1: cutterhead descriptor vs shield-pressure residual
BSLL h=3: front-shield descriptor vs shield-pressure residual
```

图注强调：

```text
The plot evaluates descriptor stability under geometry-parameter variation, not prediction accuracy.
```

---

### 11.7 新增 Figure 7：Null-model comparison

#### 图形形式

条形图或点图。

#### 横轴

- Proposed；
- Chainage-only；
- Global Vp；
- Distance-only；
- Uniform edge；
- Component shuffle。

#### 纵轴

- signed Spearman \(ρ\)；
- 或 \(|ρ|\)。

建议用 signed \(ρ\)，避免丢失方向。

---

### 11.8 新增 Figure 8：Graph traceability example

#### 内容

展示某一典型里程：

- TBM cutterhead / shield；
- 高贡献 rock voxels；
- candidate rock–machine edges；
- 贡献排名；
- TSP anomaly 值。

#### 作用

该图对 IJDE 非常有价值，因为它体现数字孪生可视化和空间诊断能力。

---

## 12. 讨论部分修改方案

### 12.1 讨论结构建议

```text
6.1 Spatial data-model contribution
6.2 What the descriptors explain beyond monitoring curves
6.3 Case-specific interpretation and evidential strength
6.4 Interpretation boundary
6.5 Limitations and future work
```

---

### 12.2 6.1 Spatial data-model contribution

#### 目的

明确本文贡献不是预测精度，而是空间关系表达。

#### 建议表述

```text
The main contribution is not a higher-accuracy response predictor, but a chainage-referenced spatial relation model that makes the otherwise implicit rock–machine relation set explicit, inspectable, and queryable.
```

#### 中文逻辑

> 主要贡献不是更高精度的响应预测器，而是将原本隐含的岩—机关系显式化、可检查化、可查询化的链式里程空间关系模型。

---

### 12.3 6.2 What the descriptors explain beyond monitoring curves

#### 重点

说明：

- persistence 是强时间基线；
- monitoring curves 解释时间惯性；
- \(I_c(t)\) 解释某部件周围几何相容地质异常暴露；
- residual consistency 是更合理的评价问题。

#### 建议表述

```text
Comparing \(I_c(t)\) with persistence residuals reframes the evaluation question from “does the graph beat persistence?” to “does the graph encode spatial diagnostic information consistent with the response changes left after persistence?” This is a more defensible question for the current data scale.
```

---

### 12.4 6.3 Case-specific interpretation and evidential strength

#### 当前结论

SJLS 支撑更强，BSLL 更弱。

#### 应解释原因

- SJLS test interval 更长；
- SJLS TSP velocity contrast 更清晰；
- BSLL h=1/h=3 样本只有 8/7；
- BSLL residual 可能受操作控制、施工扰动影响更大；
- BSLL 的弱证据不否定图构建，但限制 case-specific residual-consistency claim。

#### 建议表述

```text
The contrast between SJLS and BSLL should not be treated as a success–failure dichotomy. Rather, it shows that descriptor usefulness depends on geological contrast, monitoring quality, operational stability, and the length of the evaluated chainage interval.
```

---

### 12.5 6.4 Interpretation boundary

#### 必须保留，但不要反复出现

一次集中说明即可：

```text
The proposed descriptors are geometry-weighted summaries of TSP-derived anomalies around TBM components. They are not measured contact force, contact pressure, jamming probability, causal attribution, or calibrated operational risk.
```

#### 不建议全文反复出现

不建议在摘要、方法、结果、讨论、结论中每处都重复完整边界。保留：

- Abstract 末尾一次；
- Method 描述符定义处一次；
- Discussion 一次。

---

### 12.6 6.5 Limitations and future work

#### 必须具体，不要泛泛说 future validation

应包括：

1. **更多隧道区间验证**  
   当前 BSLL/SJLS 只是小样本诊断案例，需要更多里程段和更多地质条件。

2. **独立工程验证**  
   需要结合地质揭示、卡机记录、护盾压力异常记录、施工处置记录。

3. **TSP 不确定性传播**  
   TSP 反演存在空间分辨率和解释误差，未来应引入 uncertainty-aware voxel model。

4. **多属性异常分数**  
   当前 Vp-only 保持透明，但未来可引入 Vs、E、Vp/Vs、ν 等多属性指标。

5. **全异构图学习扩展**  
   在更多样本下，可进一步让模型学习 \(E^{rr}\)、\(E^{mm}\)、\(E^{rm}\) 的联合表示。

6. **接触力/风险概率需要独立标定**  
   只有在有接触压力、护盾摩阻、卡机标签或可靠处置记录时，才可讨论 contact force 或 jamming probability。

---

## 13. 结论修改方案

### 13.1 当前结论应避免

不要写：

- “实现卡机风险精准预测”；
- “证明该方法可用于卡机风险预警”；
- “显著优于传统方法”；
- “识别真实接触关系”；
- “计算接触压力”。

### 13.2 建议结论三点

#### 第 1 点：空间实体关系模型

```text
This study develops a chainage-referenced spatial entity–relation framework that places TSP-derived rock voxels, parameterised TBM surface components, and monitoring responses in a common excavation coordinate system.
```

#### 第 2 点：几何约束候选关系与描述符

```text
The framework constructs geometry-constrained candidate rock–machine relations and derives component–chainage descriptors from geometry-weighted TSP anomaly summaries.
```

#### 第 3 点：诊断证据边界

```text
The two case studies show that the descriptors can organise rock–machine relations into interpretable component–chainage patterns and, in some intervals, show consistency with persistence residuals. The evidence supports a diagnostic spatial-representation claim rather than contact-force recovery, calibrated jamming risk, or broad predictive dominance.
```

#### 最后一句

```text
The proposed framework provides a spatial representation basis for future rock–machine interaction diagnosis and digital-twin-oriented TBM excavation analysis.
```

---

## 14. 图表清单：建议最终论文包含哪些图表

### 14.1 主文图

| 图号 | 内容 | 目的 | 是否必须 |
|---|---|---|---|
| Figure 1 | 总体框架：数据对齐—图构建—描述符—残差一致性 | 建立论文主线 | 必须 |
| Figure 2 | active zone 与候选边几何约束 | 说明方法核心 | 必须 |
| Figure 3 | BSLL/SJLS 案例、样本划分、TSP 数据源 | 说明实验设计 | 必须 |
| Figure 4 | component–chainage \(I_c(t)\) heatmap + residual 曲线 | 核心结果 | 必须 |
| Figure 5 | primary pairs 的 descriptor–residual association | 核心统计证据 | 必须 |
| Figure 6 | null-model comparison | 证明方法必要性 | 必须 |
| Figure 7 | geometry threshold sensitivity | 稳定性 | 建议 |
| Figure 8 | graph traceability example | 强化空间图贡献 | 强烈建议 |

### 14.2 主文表

| 表号 | 内容 | 目的 | 是否必须 |
|---|---|---|---|
| Table 1 | spatial entities and relations | 说明图数据模型 | 必须 |
| Table 2 | TBM geometry and sampling | 复现几何建模 | 必须 |
| Table 3 | case runs and sample partitions | 说明样本 | 必须 |
| Table 4 | graph construction statistics | 说明图规模 | 必须 |
| Table 5 | primary diagnostic pair association | 替代全矩阵主结果 | 必须 |
| Table 6 | null/ablation comparison | 证明几何约束价值 | 必须 |
| Table 7 | anomaly definition sensitivity | 回应 \(q_i\) 任意性质疑 | 建议 |
| Table 8 | top contributing voxels | 图追溯 | 建议 |

### 14.3 Appendix

| 附录 | 内容 |
|---|---|
| Appendix A | full Spearman matrix |
| Appendix B | all threshold sensitivity results |
| Appendix C | coordinate-offset sensitivity |
| Appendix D | additional anomaly definitions |
| Appendix E | implementation details and pseudo-code |

---

## 15. 最小可行修改包

如果时间有限，必须优先完成以下内容。

### 15.1 第一优先级：决定能否送审

#### 1. 补 null model / ablation

至少包括：

- chainage-only；
- global Vp；
- distance-only；
- uniform edge；
- component shuffle。

#### 2. 补 permutation 或 detrending

至少做：

- primary diagnostic pairs；
- circular shift permutation；
- linear detrending。

#### 3. 补 TBM 几何采样细节

包括：

- 直径/半径；
- 护盾长度；
- 部件划分；
- 采样分辨率；
- 每个部件节点数；
- BSLL/SJLS 节点数差异原因。

#### 4. 重画 Figure 1、Figure 4、Figure 5

这三张图决定第一印象。

#### 5. 减少防御性表述

边界保留，但不要反复重复“不是接触力、不是风险概率”。

---

### 15.2 第二优先级：决定 IJGIS/IJDE 说服力

#### 1. 增加 graph traceability example

输出 top-contributing voxels。

#### 2. 增加 \(q_i\) 多属性敏感性

至少 Vp-only、Vs-only、Vp+Vs。

#### 3. 增加 \(A_c(t)\) 归一化版本

节点归一化或面积归一化。

#### 4. 将 full Spearman matrix 移入 Appendix

正文只报告 primary diagnostic pairs。

---

### 15.3 第三优先级：增强但非必须

#### 1. 增加三维可视化或原型系统截图

尤其适合 IJDE。

#### 2. 增加坐标对齐误差敏感性

\(\delta \in \{-1,0,+1\}m\)。

#### 3. 增加更多隧道区间

如果数据允许，这是最强增强。

---

## 16. 推荐的最终论文结构

建议最终稿结构为：

```text
1 Introduction
    1.1 TBM excavation as a moving spatial interaction process
    1.2 Limitations of monitoring-only response models
    1.3 Graph as a spatial entity–relation model
    1.4 Contributions

2 Related work
    2.1 Monitoring-based TBM response modelling
    2.2 Spatial representation of geological conditions
    2.3 Spatial graph data models and relation modelling
    2.4 Residual-based diagnostic evaluation

3 Methodology
    3.1 Chainage alignment and spatial entity definition
    3.2 Parameterised TBM surface representation
    3.3 Geometry-constrained candidate relation graph
    3.4 Component–chainage diagnostic descriptors
    3.5 Persistence residual and descriptor-consistency evaluation
    3.6 Null-model and sensitivity checks

4 Case studies and experimental design
    4.1 BSLL and SJLS case roles
    4.2 TSP preprocessing and monitoring alignment
    4.3 TBM geometry and graph-construction settings
    4.4 Evaluation design and primary diagnostic pairs

5 Results
    5.1 Constructed graph objects and relation statistics
    5.2 Component–chainage descriptor patterns
    5.3 Descriptor–residual association
    5.4 Null-model and ablation comparisons
    5.5 Sensitivity to geometry thresholds and anomaly definitions
    5.6 Graph-based traceability example

6 Discussion
    6.1 Spatial data-model contribution
    6.2 Diagnostic meaning of component–chainage descriptors
    6.3 Case-specific evidential strength
    6.4 Interpretation boundary
    6.5 Limitations and future work

7 Conclusion
```

---

## 17. 可直接执行的修改清单

### 17.1 摘要

- [ ] 删除过多防御性表达；
- [ ] 保留一次边界说明；
- [ ] 强化 chainage-referenced、component-level、spatial relation graph；
- [ ] 若完成新增实验，摘要中加入 null-model comparisons and sensitivity checks。

### 17.2 引言

- [ ] 第一段从“移动岩—机空间作用过程”切入；
- [ ] 第二段说明 monitoring-only models 的空间关系不足；
- [ ] 第三段说明 graph 是 spatial data model，不是 GNN 装饰；
- [ ] 第四段列出三条贡献；
- [ ] 不要声称预测优越性。

### 17.3 相关工作

- [ ] 增加 residual-based diagnostic evaluation 小节；
- [ ] 强化 graph data model 文献逻辑；
- [ ] 减少普通 GNN 预测模型堆砌；
- [ ] 结尾明确本文空缺：geology–machine–monitoring 的 chainage-referenced spatial relation representation。

### 17.4 方法

- [ ] 新增 spatial entities and relations 表；
- [ ] 补 coordinate alignment 与 offset sensitivity；
- [ ] 补 TBM geometry and sampling 表；
- [ ] 补 active zone 具体公式；
- [ ] 增加 \(A_c(t)\) 归一化版本；
- [ ] 增加 \(q_i\) 多属性敏感性定义；
- [ ] 增加 null-model 方法；
- [ ] 增加 permutation / detrending 方法；
- [ ] 增加 graph traceability contribution 定义。

### 17.5 实验

- [ ] 重新定义 primary diagnostic pairs；
- [ ] full matrix 移入 appendix；
- [ ] 正文报告 raw、detrended、permutation；
- [ ] 做 null comparison；
- [ ] 做 geometry threshold sensitivity；
- [ ] 做 anomaly definition sensitivity；
- [ ] 做 top-contributing voxel traceability。

### 17.6 图表

- [ ] 重画 Figure 1；
- [ ] 重画 Figure 2；
- [ ] 重画 Figure 3；
- [ ] 重画 Figure 4，加入 residual 曲线；
- [ ] Figure 5 不再只放 full unadjusted matrix；
- [ ] 新增 null model comparison；
- [ ] 新增 traceability visualization；
- [ ] 表格补充实体关系、几何采样、主诊断对、消融结果。

### 17.7 讨论和结论

- [ ] 将“图表示必要性”放在讨论第一节；
- [ ] 明确描述符解释边界；
- [ ] 解释 SJLS 与 BSLL 差异；
- [ ] 限制结论，不说风险概率和接触力；
- [ ] future work 具体化。

---

## 18. 最终审稿式修改目标

修改后，论文应从当前的：

```text
我构造了一个岩—机图，并发现一些描述符和残差相关
```

提升为：

```text
本文提出一种链式里程参考下的岩—机空间实体关系图模型。该模型把 TSP 体素、TBM 部件和监测响应统一到同一掘进坐标系中，通过 active zone、距离衰减、法向相容性和开挖状态约束候选岩—机关系，生成可追溯的部件—里程诊断描述符。通过 primary residual association、null-model comparison、detrending/permutation 和 sensitivity checks，证明该空间关系表达提供了普通监测曲线和普通地质异常均值之外的诊断信息。
```

---

## 19. 最核心的一句话

这篇论文要想被 IJGIS / IJDE 接受，不能靠“相关系数看起来很高”，而要靠下面这个逻辑闭环：

> 监测曲线只能表达响应时间惯性；TSP 体素只能表达地质空间异常；本文的岩—机空间交互图把“地质异常—TBM部件—里程位置—候选作用关系”显式连接起来，并通过残差一致性、消融和置换检验证明这种空间关系表达确实提供了监测曲线和普通地质均值之外的诊断信息。

---
