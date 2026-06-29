# 论文术语体系建议：Dynamic Rock–TBM Spatial Relation Graph

> 适用稿件方向：TBM 掘进过程中的岩体–机器动态空间关系建模，以及真实护盾卡机事件的空间解释。  
> 使用原则：术语尽量少、定义必须清楚、避免将空间关系权重解释为接触力、压力、风险概率或因果归因。

---

## 1. 核心术语总表

| 层级 | 推荐英文术语 | 中文含义 | 使用建议 |
|---|---|---|---|
| 方法总称 | **dynamic rock–TBM spatial relation graph** | 动态岩体–TBM 空间关系图 | 全文核心术语，标题、摘要、引言和方法中统一使用 |
| 地质对象 | **rock voxel** | 岩体体素 | 不与 geological voxel、rock cell 混用 |
| 机器对象 | **component-labelled TBM surface node** | 带部件标签的 TBM 表面节点 | 用于 cutterhead、front shield、middle shield、tail shield 等部件 |
| 图边 / 关系 | **candidate rock–TBM spatial relation** | 候选岩体–TBM 空间关系 | 强调是几何上合理的候选关系，不是真实接触 |
| 图边实现 | **rock–TBM edge** | 岩体–TBM 边 | 在图结构、公式和算法描述中使用 |
| 边权 | **spatial relation weight** | 空间关系权重 | 不解释为 contact force、pressure proxy 或 risk score |
| 体素异常分数 | **geological anomaly score** | 地质异常分数 | 由 TSP 派生的 Vp、Vs 或其他地质属性计算 |
| 部件级关系支撑量 | **component-level relation support** | 部件级关系支撑量 | 表示部件在某步连接到多少空间关系支撑 |
| 部件级地质异常指数 | **component-level geological anomaly index** | 部件级地质异常指数 | 用于表示某部件关联的异常地质程度 |
| 护盾组地质异常指数 | **shield-group geological anomaly index** | 护盾组地质异常指数 | front/middle/tail shield 合并后的事件解释指标 |
| 边贡献 | **edge contribution** | 边贡献 | 单条边对部件级地质异常指数的贡献 |
| 溯源 | **edge-level provenance** | 边级溯源 | 用于解释指数来源于哪些 rock voxel、TBM node 和 candidate edge |
| 事件解释 | **event-centred spatial interpretation** | 事件中心空间解释 | 用于真实护盾卡机事件，不写成 failure prediction |

---

## 2. 核心变量命名

| 符号 | 推荐英文名称 | 中文含义 | 定义 / 解释 |
|---|---|---|---|
| \(q_i\) | **geological anomaly score** | 体素 \(i\) 的地质异常分数 | 通常由低 P 波速度或其他 TSP 地质属性得到 |
| \(w_{ij}(t)\) | **spatial relation weight** | 体素 \(i\) 与 TBM 表面节点 \(j\) 在步 \(t\) 的空间关系权重 | 由距离、法向相容性等空间约束计算 |
| \(A_c(t)\) | **component-level relation support** | 部件 \(c\) 在步 \(t\) 的关系支撑量 | 连接到部件 \(c\) 的候选边权重总和 |
| \(I_c(t)\) | **component-level geological anomaly index** | 部件 \(c\) 在步 \(t\) 的地质异常指数 | 对连接到部件 \(c\) 的候选边上的 \(q_i\) 加权聚合 |
| \(\Delta I_c(t)\) | **change in component-level geological anomaly index** | 部件级地质异常指数变化量 | 相邻掘进步之间的指数差 |
| \(I_{\mathrm{shield}}(t)\) | **shield-group geological anomaly index** | 护盾组地质异常指数 | front shield、middle shield、tail shield 的组合指标 |
| \(w_{ij}(t)q_i\) | **edge contribution** | 边贡献 | 单条候选岩体–TBM 边对指数的贡献 |

---

## 3. 推荐定义句

### 3.1 Dynamic rock–TBM spatial relation graph

> We define a dynamic rock–TBM spatial relation graph to represent candidate spatial relations between TSP-derived rock voxels and component-labelled TBM surface nodes during excavation.

中文理解：

> 本文定义动态岩体–TBM 空间关系图，用于表达掘进过程中 TSP 派生岩体体素与带部件标签的 TBM 表面节点之间的候选空间关系。

---

### 3.2 Candidate rock–TBM spatial relation

> A candidate rock–TBM spatial relation is established when a rock voxel and a TBM surface node satisfy active-zone membership, distance, normal-compatibility, and excavation-state constraints.

中文理解：

> 当岩体体素与 TBM 表面节点同时满足活动区、距离、法向相容性和掘进状态约束时，二者形成候选岩体–TBM 空间关系。

---

### 3.3 Spatial relation weight

> The spatial relation weight quantifies the geometric plausibility of a candidate rock–TBM spatial relation. It is not interpreted as contact force, shield pressure, or risk probability.

中文理解：

> 空间关系权重用于量化候选岩体–TBM 空间关系的几何合理性，不解释为接触力、护盾压力或风险概率。

---

### 3.4 Component-level relation support

> Component-level relation support measures the total spatial relation weight incident to a TBM component at a given excavation step.

中文理解：

> 部件级关系支撑量表示某一掘进步下连接到某 TBM 部件的空间关系权重总量。

---

### 3.5 Component-level geological anomaly index

> We define \(I_c(t)\) as a component-level geological anomaly index, obtained by aggregating TSP-derived geological anomaly scores over the candidate rock–TBM spatial relations incident to component \(c\) at excavation step \(t\).

中文理解：

> \(I_c(t)\) 是部件 \(c\) 在掘进步 \(t\) 的部件级地质异常指数，由连接到该部件的候选岩体–TBM 空间关系上的 TSP 地质异常分数加权聚合得到。

---

### 3.6 Edge-level provenance

> Edge-level provenance records the rock voxel, TBM surface node, component label, distance, normal compatibility, spatial relation weight, geological anomaly score, and edge contribution for each candidate rock–TBM edge.

中文理解：

> 边级溯源记录每条候选岩体–TBM 边对应的岩体体素、TBM 表面节点、部件标签、距离、法向相容性、空间关系权重、地质异常分数和边贡献。

---

## 4. 建议的方法章节标题

```text
3. Methodology

3.1 Rock voxels and component-labelled TBM surface nodes

3.2 Candidate rock–TBM spatial relations

3.3 Spatial relation weight and component-level relation support

3.4 Component-level geological anomaly index

3.5 Edge-level provenance
```

说明：

- 不使用 “readout”。
- 不使用 “diagnostic layer”。
- 不使用 “risk index”。
- 不使用 “contact relation”。
- 方法章节重点是对象、关系、权重、指数和溯源。

---

## 5. 建议的实验章节标题

```text
4. Case data and event-centred evaluation

4.1 Shield-sticking event and aligned data

4.2 Component-level geological anomaly indices

4.3 Comparison summaries and edge-level provenance
```

说明：

- 实验不是 failure prediction benchmark。
- 实验目标是 event-centred spatial interpretation。
- 对照对象只需要 global geological anomaly summary、pooled TBM anomaly summary 和 component-level geological anomaly index。

---

## 6. 建议的结果章节标题

```text
5. Results

5.1 Event-aligned geological anomaly indices

5.2 Value of component-level indexing

5.3 Edge-level provenance of the shield-sticking interpretation
```

说明：

- 结果应围绕真实护盾卡机事件展开。
- 不要堆 AUC、F1、precision@k 等预测型指标。
- 重点展示：TSP 异常、TBM 监测异常、部件级地质异常指数和边级溯源之间的空间解释链。

---

## 7. 推荐标题

### 稳妥版

> **A dynamic rock–TBM spatial relation graph for component-level geological anomaly indexing during shield-sticking events**

### 事件解释版

> **A dynamic rock–TBM spatial relation graph for event-centred interpretation of TBM shield sticking**

### 更偏 IJGIS 方法版

> **Component-level geological anomaly indexing using a dynamic rock–TBM spatial relation graph during TBM excavation**

推荐优先使用第一个标题，因为它同时保留了：

- dynamic spatial relation graph；
- component-level indexing；
- geological anomaly；
- shield-sticking event。

---

## 8. 推荐关键词

```text
dynamic spatial relation graph; TBM excavation; rock–TBM relation; geological anomaly index; shield sticking; edge-level provenance
```

如果期刊限制关键词数量，可压缩为：

```text
spatial relation graph; TBM excavation; shield sticking; geological anomaly index; edge-level provenance
```

---

## 9. 禁用或慎用术语

| 不建议术语 | 处理建议 | 原因 | 推荐替代 |
|---|---|---|---|
| readout | 删除 | 不像 IJGIS 规范术语，偏神经网络/传感器输出 | component-level geological anomaly index |
| anomaly readout field | 删除 | 太抽象，像自造概念 | geological anomaly index / trajectory |
| diagnostic readout | 删除 | 容易被理解成诊断模型 | event-centred spatial interpretation |
| geological exposure index | 慎用 | exposure 在 GIS 中可用，但 TBM 领域不够稳 | component-level geological anomaly index |
| relation layer | 慎用 | 可在引言解释一次，但不作为正式术语 | dynamic rock–TBM spatial relation graph |
| contact relation | 删除 | 容易被理解为真实接触 | candidate rock–TBM spatial relation |
| contact force | 删除 | 本文未做力学反演 | spatial relation weight |
| pressure proxy | 删除 | 容易被要求压力标定 | spatial relation weight |
| risk index | 删除 | 本文未做风险概率标定 | geological anomaly index |
| failure prediction | 删除 | 本文不是预测模型 | shield-sticking event interpretation |
| causal attribution | 删除 | 本文不能证明唯一因果 | spatial interpretation / edge-level provenance |

---

## 10. 术语使用边界

### 可以主张

> The proposed graph represents candidate spatial relations between anomalous rock voxels and TBM components.

> The component-level geological anomaly index summarizes the geological anomaly scores associated with each TBM component through candidate rock–TBM spatial relations.

> Edge-level provenance traces the event interpretation back to specific rock voxels, TBM surface nodes, and spatial relation attributes.

### 谨慎主张

> The shield-group geological anomaly index is consistent with the observed shield-sticking process in the event case.

> The graph provides additional spatial interpretation beyond global geological anomaly summaries and pooled TBM summaries.

### 不应主张

> The index estimates shield pressure.

> The edge weight represents contact force.

> The method predicts all shield-sticking events.

> The method proves the unique causal component of the event.

> The method produces a calibrated risk probability.

---

## 11. 摘要中的标准写法

> We propose a dynamic rock–TBM spatial relation graph to represent candidate spatial relations between TSP-derived rock voxels and component-labelled TBM surface nodes during excavation. Based on this graph, we define a component-level geological anomaly index to summarize the anomalous geological conditions associated with each TBM component at each excavation step.

> The method is evaluated through an event-centred spatial interpretation of a real TBM shield-sticking event. The shield-group geological anomaly index is compared with monitoring responses and simpler geological summaries, and edge-level provenance is used to trace the event interpretation back to specific rock voxels and TBM surface nodes.

---

## 12. 全文核心贡献句

> The contribution of this study is a dynamic rock–TBM spatial relation graph that converts TSP-derived geological anomalies into component-level geological anomaly indices and supports edge-level provenance for event-centred spatial interpretation of TBM shield sticking.

中文理解：

> 本文贡献是提出动态岩体–TBM 空间关系图，将 TSP 派生地质异常转化为部件级地质异常指数，并通过边级溯源支持 TBM 护盾卡机事件的空间解释。

---

## 13. 最终术语原则

1. **少造新词**：全文核心术语控制在 6–7 个。
2. **边界清楚**：空间关系权重不等于接触力、护盾压力或风险概率。
3. **对象清楚**：rock voxel、TBM surface node、rock–TBM edge 三类对象不能混用。
4. **指标清楚**：\(q_i\)、\(w_{ij}(t)\)、\(A_c(t)\)、\(I_c(t)\) 各有明确含义。
5. **事件清楚**：真实案例是 shield-sticking event，不是泛化预测 benchmark。
6. **贡献清楚**：本文贡献是 spatial relation graph + component-level geological anomaly index + edge-level provenance。
