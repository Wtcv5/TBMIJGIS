# IJGIS 导向专家审稿意见与逐条修改方案

> 适用稿件：**Spatially Explicit Rock–TBM Interaction Graph Sequences for Response-Consistent Interpretation of TBM Excavation Processes**  
> 整理目的：将专家式审稿意见转化为可执行修改清单，帮助稿件从“工程场景中的图神经网络应用”进一步提升为“具有 GIScience 方法贡献的空间显式图建模论文”。

---

## 0. 总体判断

当前稿件已经具备完整论文形态，主线较清楚：

```text
TSP 地质体素场
→ TBM 参数化表面节点
→ 几何约束岩–机候选边
→ 响应监督 GNN-GRU 图序列学习
→ TBM 表面热点与里程演化解释
→ 空间一致性评价
```

目前的核心优势是：稿件已经不再硬说“预测精度最好”，而是将贡献定位为：

> 在保持基本响应预测能力的同时，提供 monitoring-only temporal models 难以获得的空间显式岩–机交互解释。

这个定位是正确的，也更接近 IJGIS 的兴趣点。

但若按 IJGIS 审稿标准，当前稿件仍可能被判断为：

> **Major revision, with promising idea but insufficient GIScience-level validation.**

主要原因不是 idea 不成立，而是：

1. GIScience 方法贡献还需要进一步凸显；
2. 小样本下实验稳定性不足；
3. spatial consistency 指标仍可能被认为是几何约束诱导的结果；
4. attention / relevance 解释需要更多 sanity checks；
5. 图文一致性和核心图件表达仍需彻底修正。

---

## 1. IJGIS 审稿人最可能关注的核心问题

IJGIS 审稿人不会只问“这个模型能不能用于 TBM”，而会问：

1. 这篇文章对 geographic information science 的一般性贡献是什么？
2. 你的 spatial entity representation 与已有 spatio-temporal graph learning 有什么区别？
3. 你的 geometry-constrained candidate relation construction 是否具有一般空间关系建模意义？
4. learned relevance 是否真的是 response-supervised learning 的结果，而不是几何先验自然造成的？
5. 小样本下结果是否稳定？
6. attention-based interpretation 是否可信？
7. 几何阈值、active zone、采样分辨率变化后结果是否稳健？
8. 图件是否能清楚支撑 spatial interpretation 贡献？

因此，修改时要围绕一句话强化：

> **这不是“把 GNN 用在 TBM 上”，而是提出一种面向多空间支撑实体、几何约束候选关系和不完全可观测响应监督的空间显式交互图建模框架。**

---

# A 类：不改不能投的硬伤

---

## A1. 修正 Figure 3 的 caption 和图例

### 当前问题

当前 Figure 3 实际展示的是五个目标响应变量：

- Advance Rate；
- Torque；
- Thrust；
- Penetration；
- Shield Pressure。

每个子图中同时包含 Proposed、LSTM、TSP-LSTM、XGBoost、Persistence 等曲线。

但当前 caption 写成：

```text
Predicted vs. observed advance rate ... Each panel shows one method ...
```

这与图完全不一致。IJGIS 审稿人会认为这是低级图文错误，并进一步怀疑全文图表是否经过仔细核查。

### 修改方式

将 caption 改为：

```text
Figure 3. Predicted versus observed standardized monitoring responses for five target variables on the test set. Each panel corresponds to one response variable, including advance rate, torque, thrust, penetration, and shield pressure. Different curves represent the proposed framework and baseline models. The proposed framework achieves comparable response prediction performance while providing spatially explicit interaction interpretation unavailable to monitoring-only models.
```

### 图例建议

当前图例中的 MAE 容易与 Table 2 的 global MAE 混淆。建议二选一：

**方案 1：删除图例中的 MAE。** 只保留模型名称。

**方案 2：保留 MAE，但在 caption 中补充：**

```text
MAE values in the legend are variable-specific values for the displayed panel and should not be confused with the global multi-variable metrics in Table 2.
```

### 优先级

**最高。** 这是显性错误，不改不能投。

---

## A2. 重画 Figure 6，补充 TSP Vp 面板

### 当前问题

正文和 caption 都说 Figure 6 包含：

- panel (a)：Mean surface relevance；
- panel (b)：TSP P-wave velocity profile；
- panel (c–g)：五个监测响应变量。

但当前图中实际只有六个 panel：

- a：Mean Cj；
- b：Advance Rate；
- c：Torque；
- d：Thrust；
- e：Penetration；
- f：Shield Pressure。

没有 TSP Vp，也没有低波速异常红色标注。这个问题非常明显。

### 推荐重画结构

Figure 6 应改为七个 panel：

```text
a. Mean surface relevance \bar{C}(c)
b. TSP P-wave velocity Vp(c), with low-Vp zones highlighted
c. Advance Rate
d. Torque
e. Thrust
f. Penetration
g. Shield Pressure
```

### 推荐 caption

```text
Figure 6. Chainage evolution of mean surface relevance, TSP velocity, and monitoring responses on the test chainages. (a) Mean surface relevance averaged over all TBM surface nodes; (b) TSP P-wave velocity profile, with low-velocity zones below the 25th percentile highlighted; (c–g) standardized monitoring responses. The figure links geological context, learned interaction relevance, and machine response along excavation chainage.
```

### 图件设计建议

- panel (b) 用右侧或独立 y 轴显示 Vp，单位 m/s；
- 低于 25% 分位数的 Vp 区间用浅红背景；
- 所有 panel 共用 x 轴 chainage；
- 不要过度解释“因果”，只说 co-evolution / alignment / spatial reasoning。

### 优先级

**最高。** 当前 Figure 6 图文不一致，必须修。

---

## A3. 放大并重排 Figure 5

### 当前问题

Figure 5 是全文最核心的空间解释图，直接支撑 “graph-to-surface interpretation”。但当前图太小，左侧空白大，热点图主体、分区线、色标和文字都不够清楚。

IJGIS 对空间图件质量要求较高。核心空间解释图如果看不清，会极大削弱稿件说服力。

### 修改方式

建议 Figure 5 单独占半页到一页，重排为：

```text
Ch. 41 m      Ch. 45 m      Ch. 48 m
[hotspot]     [hotspot]     [hotspot]
```

### 图件细节要求

- 三个 chainage 横向排布；
- 去掉左侧无效空白；
- 每张图宽度至少占整页宽度的 30%；
- cutterhead / front / middle / tail shield 分界线加粗；
- Crown / Invert 标注加大；
- 色标字体加大；
- 标注 “unwrapped TBM surface coordinate”；
- 如果每个 chainage 独立 min–max rescale，caption 必须说明不同子图之间颜色强度不能直接比较绝对大小。

### 推荐 caption

```text
Figure 5. Response-consistent surface relevance hotspots on the unwrapped TBM surface at three representative test chainages. The colour indicates degree-normalised surface relevance Cj, rescaled within each chainage for visualization only. Vertical dashed lines separate cutterhead, front shield, middle shield, and tail shield regions. The maps show where the learned rock–machine interaction relevance concentrates on the TBM surface.
```

### 优先级

**最高。** 这是主贡献图。

---

## A4. 修改 Table 1，避免 Raw steps 误导

### 当前问题

当前 Table 1 中 “Raw steps” 容易引起误解。由于使用 sliding window，validation/test 的输入窗口会与 training chainage 部分重叠。虽然正文已经解释这不是数据泄漏，但表格本身还不够直观。

### 推荐表格

建议改成：

| Partition | Prediction-target chainage | Input chainage required | Effective samples |
|---|---:|---:|---:|
| Training | 5–34 m | 0–34 m | 30 |
| Validation | 35–40 m | 30–40 m | 6 |
| Test | 41–48 m | 36–48 m | 8 |
| Total | 5–48 m | 0–48 m | 44 |

### 推荐 caption

```text
Partitions are defined by prediction-target chainage. Input windows may overlap across partitions following the standard sliding-window evaluation setting, but target responses are strictly separated in chainage order.
```

### 需要正文补充

```text
The TSP forecast is assumed to be operationally available before excavation reaches the corresponding target chainage; no future monitoring response is used in constructing input windows.
```

### 优先级

**高。** 主要防止审稿人质疑 data leakage。

---

## A5. 修改 Figure 2 caption，四个阶段必须对应图中内容

### 当前问题

Figure 2 图中有 Stage 1–4，但 caption 对阶段描述不完整，且 Stage 含义和图中结构略有错位。

### 推荐 caption

```text
Figure 2. Framework pipeline. Stage 1 constructs spatial entities from the TSP voxel field, TBM surface geometry, and monitoring records. Stage 2 builds geometry-constrained graph snapshots through active-zone filtering and candidate rock–machine edge construction. Stage 3 performs response-supervised graph-sequence learning using a heterogeneous GNN encoder and GRU temporal encoder. Stage 4 projects learned relevance back to TBM surface and chainage spaces to generate hotspot maps and evolution views.
```

### 优先级

**中高。** 图文一致性问题，需要修。

---

## A6. 弱化 discussion 中的工程处置建议

### 当前问题

Figure 7 已经说明是 illustrative，不是 validated decision-support tool。但 Discussion 中仍出现类似：

```text
reduced advance rate, enhanced support
```

这容易被认为是未经验证的工程处置建议。

### 修改方式

将其替换为：

```text
closer monitoring, operational review, or parameter adjustment may be considered.
```

或：

```text
the indicator may prompt further expert inspection rather than directly prescribing operational actions.
```

### 优先级

**高。** 避免超出证据边界。

---

## A7. 处理 Cj 的 raw score 符号问题

### 当前问题

当前表面相关性定义为：

```math
C_j(t)=\frac{1}{d_j(t)}\sum_i s_{ij}^{rm}(t)
```

其中 `s_ij` 是 pre-softmax raw score。raw score 可能为负。如果存在正负混合，则：

- Mean Cj 可能受符号抵消影响；
- CV 分母可能接近 0；
- “higher Cj = stronger relevance” 的解释不够稳定。

### 推荐修改

将 `Cj` 定义为非负 relevance intensity：

```math
C_j(t)=\frac{1}{d_j(t)+\epsilon}\sum_{i:(i,j)\in E_t^{rm}} \mathrm{softplus}(s_{ij}^{rm}(t))
```

或者：

```math
C_j(t)=\frac{1}{d_j(t)+\epsilon}\sum_{i:(i,j)\in E_t^{rm}} \sigma(s_{ij}^{rm}(t))
```

其中 `softplus` 更适合保留 raw score 的相对强弱。

### 文中解释建议

```text
To avoid sign cancellation and to interpret surface relevance as an intensity-like quantity, raw edge scores are transformed by a monotonic non-negative function before degree-normalised aggregation. This transformation preserves the ranking of edge scores while making Cj suitable for hotspot mapping and component-level comparison.
```

### 优先级

**高。** 这是 attention interpretation 的基础定义问题。

---

# B 类：显著提高 IJGIS 说服力的补充实验

---

## B1. 增加 Geometry-only baseline

### 目的

证明热点图不是几何先验天然“造出来”的，而是 geometry constraint 与 response-supervised learning 共同作用的结果。

### 做法

不训练 GNN，不使用监测响应监督，直接使用几何先验：

```math
\pi_{ij}^{rm}=\exp(-d_{ij}/\tau)\kappa_{ij}
```

聚合得到 surface relevance，并计算空间一致性指标。

### 新增表格模板

| Variant | MAE | Rel.–Geo. r | Moran’s I | CV |
|---|---:|---:|---:|---:|
| Full framework | 0.481 | -0.31 | 0.42 | 0.38 |
| Geometry prior only | — |  |  |  |
| No geometric prior | 0.360 | -0.12 | 0.18 | 0.15 |
| Randomised edges | 0.490 | 0.05 | 0.07 | 0.08 |

### 预期论证

如果 Geometry-only 的 Moran’s I 较高，但 Rel.–Geo. r、edge perturbation 效果或 response alignment 弱于 Full framework，则可以说明：

```text
Geometry defines the plausible relation space, but response-supervised learning is needed to identify response-consistent relevance within that space.
```

### 优先级

**非常高。** 这是 IJGIS 审稿人最可能要求的 sanity check。

---

## B2. 增加 Response-label randomization test

### 目的

证明 learned relevance 与响应监督有关，而不是图结构自然产生的。

### 做法

- 打乱训练集未来响应标签 `r_{t+h}`；
- 重新训练模型；
- 计算 Rel.–Geo. r、Moran’s I、CV；
- 与 Full framework 对比。

### 新增表格模板

| Variant | Rel.–Geo. r | Moran’s I | CV |
|---|---:|---:|---:|
| Full framework | -0.31 | 0.42 | 0.38 |
| Response-label shuffled |  |  |  |

### 预期结论

```text
Shuffling response labels weakens geological alignment and destabilises the spatial relevance pattern, indicating that the learned relevance is not solely induced by graph geometry.
```

### 优先级

**非常高。** 直接回应“attention 是否真的由 response supervision 学到”。

---

## B3. 增加 Edge perturbation / deletion test

### 目的

验证高 relevance 边对预测任务是否重要。

### 做法

在测试阶段进行边遮蔽：

1. 删除 top 10% high-relevance edges；
2. 删除 bottom 10% low-relevance edges；
3. 随机删除 10% edges。

比较预测误差变化。

### 新增表格模板

| Perturbation | ΔMAE | ΔRMSE | Interpretation |
|---|---:|---:|---|
| Remove top 10% relevance edges |  |  | high-relevance edges are important |
| Remove bottom 10% relevance edges |  |  | low-relevance edges have limited effect |
| Remove random 10% edges |  |  | random baseline |

### 预期结论

```text
Removing high-relevance edges causes larger performance degradation than removing low-relevance or random edges, supporting the functional importance of learned edge relevance.
```

### 优先级

**高。** 这是 attention 解释可信度的重要证据。

---

## B4. 增加几何参数敏感性分析

### 目的

验证 geometry-constrained relation construction 是否稳健。

### 参数建议

| 参数 | 推荐取值 |
|---|---|
| Distance threshold τ | 1.5 / 2.0 / 2.5 m |
| Normal threshold ηmin | 0.1 / 0.3 / 0.5 |
| Active-zone radius | 3 / 5 / 7 m |

### 新增表格模板

| Parameter setting | MAE | Pearson r | Rel.–Geo. r | Moran’s I | CV |
|---|---:|---:|---:|---:|---:|
| τ = 1.5 |  |  |  |  |  |
| τ = 2.0 |  |  |  |  |  |
| τ = 2.5 |  |  |  |  |  |
| ηmin = 0.1 |  |  |  |  |  |
| ηmin = 0.3 |  |  |  |  |  |
| ηmin = 0.5 |  |  |  |  |  |
| zone = 3 m |  |  |  |  |  |
| zone = 5 m |  |  |  |  |  |
| zone = 7 m |  |  |  |  |  |

### 预期结论

```text
The spatial consistency indicators remain directionally stable within a reasonable range of geometric thresholds, suggesting that the proposed relation construction is not overly sensitive to a single parameter choice.
```

### 优先级

**高。** IJGIS 很看重空间参数和尺度效应。

---

## B5. 增加 random seed stability

### 目的

证明小样本下结果不是随机初始化偶然导致。

### 做法

至少 5 个 random seeds，最好 10 个。

### 表格模板 1：预测指标

| Method | MAE mean ± std | Pearson r mean ± std |
|---|---:|---:|
| LSTM |  |  |
| TSP-LSTM |  |  |
| Proposed |  |  |

### 表格模板 2：空间指标

| Variant | Rel.–Geo. r mean ± std | Moran’s I mean ± std | CV mean ± std |
|---|---:|---:|---:|
| Full framework |  |  |  |
| No geometric prior |  |  |  |
| Randomised edges |  |  |  |

### 预期结论

即使绝对值有波动，也要证明排序稳定：

```text
Full framework consistently yields higher spatial consistency than ablation variants across random seeds.
```

### 优先级

**高。** 测试集只有 8 个样本，必须补稳定性。

---

## B6. 增加 bootstrap CI / permutation test

### 目的

给 Table 4 中的单点指标补充不确定性说明。

### 当前问题

当前 Table 4 只有单点值：

- Rel.–Geo. r = -0.31；
- Moran’s I = 0.42；
- CV = 0.38。

但测试 chainage 只有 8 个，单点值说服力有限。

### 修改后 Table 4 模板

| Variant | Rel.–Geo. r | p / 95% CI | Moran’s I | 95% CI | CV | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Full framework | -0.31 |  | 0.42 |  | 0.38 |  |
| No geometric prior | -0.12 |  | 0.18 |  | 0.15 |  |
| Randomised edges | 0.05 |  | 0.07 |  | 0.08 |  |

### 方法建议

- Rel.–Geo. r：permutation test；
- Moran’s I 和 CV：bootstrap over chainages；
- 明确写：

```text
Due to the limited number of test chainages, the correlation should be interpreted as directional rather than statistically conclusive.
```

### 优先级

**高。** 增强小样本结果可信度。

---

## B7. 增加 per-variable prediction metrics

### 目的

Table 2 是五个变量的 global average，过于粗略。需要说明哪些响应变量更能从空间图结构中受益。

### 表格模板

| Variable | Persistence MAE | LSTM MAE | TSP-LSTM MAE | Proposed MAE | Proposed r |
|---|---:|---:|---:|---:|---:|
| Advance rate |  |  |  |  |  |
| Torque |  |  |  |  |  |
| Thrust |  |  |  |  |  |
| Penetration |  |  |  |  |  |
| Shield pressure |  |  |  |  |  |

### 预期论证

不需要 Proposed 每项最优，但需要说明：

```text
The graph framework is not uniformly superior in prediction, but it maintains response prediction capability while enabling spatial interpretation.
```

### 优先级

**中高。** 有助于解释 Table 2 中负 R² 和预测不占优的问题。

---

## B8. 增加 low-Vp / high-Cj spatial overlap 分析

### 目的

比 Rel.–Geo. r 更直观地验证 learned relevance 与地质异常是否空间一致。

### 做法

定义：

```text
low-Vp chainages: Vp below 25th percentile
high-Cj chainages: mean Cj above 75th percentile
```

计算：

```math
Overlap=\frac{|LowVp \cap HighCj|}{|HighCj|}
```

并与随机置换结果比较。

### 表格模板

| Metric | Observed | Random mean | p-value |
|---|---:|---:|---:|
| Low-Vp / high-Cj overlap |  |  |  |

### 预期结论

```text
High-relevance chainages are enriched in low-velocity geological zones compared with random expectation.
```

### 优先级

**中高。** 这是更 GIS 化的空间关系验证。

---

# C 类：写作与结构强化

---

## C1. 重写 Introduction 中的 GIScience framing

### 当前问题

引言中“this becomes a geographic-information representation problem”方向对，但语气略硬，容易被认为是强行贴 GIS 标签。

### 推荐改法

将其改成：

```text
From a geographic-information representation perspective, rock–TBM interaction involves heterogeneous spatial entities with different spatial supports, including voxelized geological fields, parameterized machine surfaces, and chainage-indexed monitoring responses. The central challenge is to organize these entities and their candidate relations into a representation that preserves spatial explicitness, relation plausibility, and excavation evolution.
```

### 目标

让 GIScience framing 更自然：不是“我说它是 GIS 问题”，而是“它具有典型 GIScience 中多空间支撑实体与空间关系表达问题的特征”。

---

## C2. 重写贡献点，让它更像 IJGIS

### 当前贡献点的问题

当前贡献点偏工程方法描述。建议提升为一般空间方法贡献。

### 推荐贡献点

#### Contribution 1: Multi-support spatial entity representation

```text
We propose a multi-support spatial entity representation that integrates TSP-derived rock voxels, parameterized TBM surface nodes, and chainage-indexed monitoring responses into dynamic graph snapshots.
```

#### Contribution 2: Geometry-constrained candidate relation modeling

```text
We introduce a geometry-constrained candidate relation model that restricts rock–machine interaction edges through distance, signed-normal compatibility, active-zone filtering, and excavation state.
```

#### Contribution 3: Indirectly supervised spatial relation learning

```text
We develop a response-supervised strategy to learn candidate relation relevance under incomplete observability, without requiring contact-force labels or sparse jamming-event labels.
```

#### Contribution 4: Graph-to-surface spatial interpretation

```text
We establish a graph-to-surface interpretation route that projects learned relevance back to TBM surface and chainage spaces, enabling component-aware hotspot mapping and excavation-evolution analysis.
```

---

## C3. Methodology 增加 implementation details 小节

### 建议新增内容

在 Methodology 或 Experiment setup 中增加：

- rock voxel 数量；
- active rock node 数量范围；
- TBM surface node 数量；
- 每个 graph 的 candidate edge 数量范围；
- GNN encoder 具体类型；
- message passing 公式；
- graph-level readout 方式；
- GRU 输入维度；
- loss 权重；
- 输入标准化方式；
- training epochs；
- random seed；
- early stopping 规则。

### 示例表格

| Quantity | Value / range |
|---|---:|
| TSP voxels | 21,609 |
| Raw excavation steps | 49 |
| Effective sequence samples | 44 |
| TBM surface nodes |  |
| Rock nodes per graph |  |
| Rock–machine edges per graph | 35,429–93,048 |
| Historical window K | 5 |
| Forecast horizon h | 1 |

### 目的

提升可复现性。

---

## C4. Results 结构建议重排

建议 Results 改成下面结构：

```text
4.1 Prediction performance as auxiliary supervision
4.2 Structural ablation
4.3 Spatial consistency validation
4.4 Robustness and sanity checks
4.5 Spatial interpretation outputs
4.6 Illustrative spatial decision scenario
```

### 各小节内容

#### 4.1 Prediction performance as auxiliary supervision

Table 2 + Figure 3。强调 prediction 是监督任务，不是唯一目标。

#### 4.2 Structural ablation

Table 3 + Figure 4。

#### 4.3 Spatial consistency validation

Table 4 + CI / permutation。

#### 4.4 Robustness and sanity checks

新增：geometry-only、label randomization、edge perturbation、threshold sensitivity、seed stability。

#### 4.5 Spatial interpretation outputs

Figure 5 + Figure 6。

#### 4.6 Illustrative spatial decision scenario

Figure 7。强调 illustrative，不做 validated decision support claim。

---

## C5. Discussion 语气从 “justify” 改为 “suggest”

### 当前风险

当前 discussion 里部分表述过强，例如：

```text
this trade-off is justified
```

### 推荐改法

改成：

```text
this trade-off may be acceptable when spatial interpretability is prioritised, although further validation is required.
```

类似地，将：

```text
the framework enables decision support
```

改成：

```text
the framework provides spatial information that could inform decision-support workflows.
```

### 原则

- 不说“证明”；多说“suggest / indicate / provide evidence”。
- 不说“风险预警工具”；多说“spatial reasoning view / heuristic cue”。
- 不说“可直接指导施工”；多说“may support expert inspection”。

---

## C6. Boundary and caution 提前出现

### 建议

在 Spatial interpretation outputs 小节结束处加入一句：

```text
The hotspot maps should be interpreted as response-consistent relevance fields under geometric screening, not as measured contact pressure, stress, or jamming probability.
```

### 目的

防止读者将热点图误读为真实物理接触场。

---

# D 类：最小修改版本与 IJGIS 强化版本

---

## D1. 最小修改版本：1–2 天内完成

如果短期只是要形成一个内部预审稿，至少完成：

1. 修 Figure 3 caption；
2. 重画 Figure 6 或删除 TSP panel 相关描述；
3. 放大 Figure 5；
4. Table 1 改成 target/input chainage 表；
5. Figure 2 caption 改成四阶段；
6. Discussion 处置建议弱化；
7. Cj 增加非负 transform 或说明 raw score 符号问题。

完成后，稿件可以达到：

```text
故事完整、图文基本一致、可作为内部预审稿。
```

---

## D2. IJGIS 强化版本：建议正式投稿前完成

如果目标是 IJGIS，建议进一步补充：

1. Geometry-only baseline；
2. Response-label randomization test；
3. Edge perturbation / deletion test；
4. τ / ηmin / active-zone sensitivity analysis；
5. Random seed stability；
6. Bootstrap confidence intervals / permutation p-values；
7. Per-variable prediction metrics；
8. Low-Vp / high-Cj spatial overlap analysis；
9. Introduction 和 contributions 改成更一般的 GIScience 表述；
10. Results 结构重排，突出 spatial consistency validation 和 robustness。

完成后，稿件才能更有底气回应 IJGIS 审稿人：

```text
This is not merely a domain-specific GNN application, but a spatial representation and constrained relation modeling framework for heterogeneous spatial interaction systems under incomplete observability.
```

---

# E 类：修改优先级总表

| Priority | Item | Type | Why it matters |
|---:|---|---|---|
| 1 | 修 Figure 3 caption | Hard bug | 当前图文不一致，容易被直接质疑 |
| 2 | 重画 Figure 6 | Hard bug | 当前缺少 TSP Vp panel，caption 与图不一致 |
| 3 | 放大 Figure 5 | Core evidence | 空间解释主图必须清晰 |
| 4 | 修改 Table 1 | Data clarity | 防止 data leakage 误解 |
| 5 | 修 Figure 2 caption | Consistency | 四阶段 pipeline 要对应图中内容 |
| 6 | Cj 非负 transform | Method rigor | raw score 可负，影响 relevance 解释 |
| 7 | 弱化处置建议 | Boundary | 避免未验证 decision support claim |
| 8 | Geometry-only baseline | Sanity check | 证明热点不是几何先验自然造成 |
| 9 | Label randomization | Sanity check | 证明 response supervision 有效 |
| 10 | Edge perturbation | Explanation validity | 证明 high relevance edges 功能重要 |
| 11 | Threshold sensitivity | Robustness | 证明几何参数不敏感 |
| 12 | Random seed stability | Reliability | 小样本必须报告稳定性 |
| 13 | Bootstrap / permutation | Uncertainty | 单点指标不够可信 |
| 14 | Per-variable metrics | Result clarity | 解释不同响应变量差异 |
| 15 | Low-Vp / high-Cj overlap | GIS validation | 增强空间关系验证 |
| 16 | 重写 contributions | IJGIS framing | 提升 GIScience 方法贡献 |
| 17 | Results 重排 | Presentation | 让逻辑更符合审稿习惯 |
| 18 | Discussion 语气弱化 | Trustworthiness | 避免 overclaim |

---

# F 类：最终专家判断

当前稿件已经具备 IJGIS 论文的雏形，但还需要补足证据强度。最关键的问题不是“idea 行不行”，而是：

```text
你必须证明空间解释不是由几何约束单独制造出来的，
而是由合理的空间关系建模 + 响应监督学习共同产生的，
并且这种解释在参数、随机种子、扰动和对照实验下是稳定的。
```

因此，最终修改策略应当是：

1. 先清除所有图文和表格硬伤；
2. 再补充 sanity checks 和 robustness；
3. 最后重写 Introduction / Contributions / Discussion，使其更符合 IJGIS 的空间表达与空间关系建模话语体系。

如果只完成 A 类修改，稿件可作为较完整的工程 GeoAI 初稿；如果完成 A+B+C 类修改，才更接近 IJGIS 的正式投稿水准。

