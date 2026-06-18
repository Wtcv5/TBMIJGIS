# TBM 表面分区级响应定位研究论文撰写与实现方案

> 版本定位：最终收敛版  
> 目标：指导论文写作、算法实现、实验组织与图表制作  
> 核心原则：不广而泛之，不做宏大数字孪生框架；只围绕“将 TBM 监测响应定位到刀盘与护盾局部表面区域”展开。

---

## 0. 最终研究定位

本文不再定位为“TBM 参数预测模型”“卡机风险预测模型”或“大型地下工程数字孪生框架”，而是聚焦一个更窄、更可验证的问题：

> 如何将原本仅沿里程变化的 TBM 监测响应，定位到刀盘和护盾的局部表面区域？

本文提出 **TBM 表面分区级响应定位方法**。其基本思想是：将 TBM 刀盘与护盾划分为工程上可解释的表面分区；对每个掘进步，基于距离、法向一致性与 active zone 筛选每个表面分区邻近的 TSP 岩体体素；将局部岩体属性与几何关系聚合为分区级交互特征；再利用短时程运行响应作为监督信号，学习不同表面分区与运行响应之间的响应一致性相关性；最终将监测响应从“一维里程曲线”重新映射到“TBM 表面局部区域”。

全文应始终坚持：

```text
不是预测下一步响应本身，而是用响应预测作为功能性验证；
不是恢复真实接触力，而是学习响应一致性相关性；
不是做滚刀级细节，而是做表面分区级定位；
不是构建宏大数字孪生平台，而是实现一个最小可用的表面响应定位原型。
```

---

## 1. 论文核心创新句

中文核心句：

> 本文提出一种 TBM 表面分区级响应定位方法，将原本仅沿里程变化的推力、扭矩、贯入度和推进速度等监测响应，重新定位到刀盘与护盾的局部表面区域，从而为 TBM 掘进过程中的岩–机作用解释提供空间化表达。

英文核心句：

> This study proposes a surface-patch-based response localization method that maps chainage-indexed TBM operational responses onto local cutterhead and shield surface regions, providing a spatially explicit representation for interpreting rock–machine interaction during TBM excavation.

---

## 2. 推荐题目

### 2.1 首选题目

**Surface-Patch-Based Response Localization of TBM Operational Responses Using Geometry-Constrained Rock–Machine Associations**

优点：

- 明确是 surface-patch-based；
- 明确是 response localization；
- 保留 geometry-constrained rock–machine associations；
- 不过度强调 prediction / risk / digital twin。

### 2.2 备选题目

**From Monitoring Curves to Surface Hotspots: Geometry-Constrained Response Localization for TBM Excavation Processes**

适合更强调故事性。

### 2.3 更偏 GIS / Digital Earth 的题目

**Spatial Localization of TBM Operational Responses on Cutterhead and Shield Surfaces Using Geological Voxel Associations**

适合强调空间定位与地质体素关联。

---

## 3. 全文概念边界

| 不建议写法 | 推荐写法 | 原因 |
|---|---|---|
| contact force | response-consistent relevance | 没有真实接触力标签或数值模拟校准 |
| contact pressure | surface hotspot / relevance score | 不能将学习分数解释为压力 |
| jamming probability | response-related tendency | 本文不做卡机标签预测 |
| risk hotspot | response-consistent hotspot | 避免把响应相关性说成风险真值 |
| physical contact | geometry-screened candidate association | 距离和法向只能说明候选关系 |
| TBM digital twin platform | prototype for surface response localization | 避免平台化过大 |
| cutter-level localization | cutterhead patch-level localization | 当前监测数据难以支撑单滚刀级定位 |

全文最核心概念统一为：

> **response-consistent surface relevance**  
> 响应一致性表面相关性

也可以在边层面写为：

> **response-consistent rock–machine association**  
> 响应一致性岩–机关联

---

## 4. 论文整体结构

```text
1. Introduction
2. Related Work
3. Methodology
   3.1 Surface partitioning of TBM cutterhead and shield
   3.2 Geometry-constrained rock–machine association construction
   3.3 Response-supervised surface-patch localization model
   3.4 Prototype visualization for surface response localization
4. Experiments and Results
   4.1 Data preparation and chainage alignment
   4.2 Validity of surface-patch association construction
   4.3 Functional validation through response prediction
   4.4 Ablation and spatial-topology tests
   4.5 Surface hotspot localization and evolution analysis
   4.6 Prototype demonstration
5. Discussion
6. Conclusion
```

说明：

- 方法部分不超过 4 个小节；
- 原型系统不单独做成宏大章节，可放在方法 3.4 和结果 4.6 中；
- 实验重点不是“预测模型性能排行榜”，而是证明表面分区空间关系有用。

---

# 5. Introduction 撰写方案

## 5.1 第一段：工程问题

目标：从 TBM 掘进响应出发，而不是从 GNN 出发。

建议内容：

- TBM 掘进过程中的推力、扭矩、贯入度、推进速度、盾体压力等监测响应反映了机器与围岩之间的复杂相互作用。
- 在软弱破碎、富水或高地应力地层中，局部围岩条件会引起 TBM 响应异常。
- 传统监测分析通常将这些响应表达为沿里程变化的曲线。
- 但工程人员真正关心的不只是“哪一段响应升高”，还包括“这种响应更可能关联到 TBM 的哪个局部区域”。

可写句：

> In current TBM monitoring practice, operational responses are commonly represented as chainage-indexed curves. Such curves are effective for detecting abnormal intervals, but they provide limited information on where the response-related rock–machine interaction may occur on the cutterhead or shield surface.

## 5.2 第二段：现有方法不足

目标：克制地指出 gap。

现有方法包括：

- 分段风险评估；
- 基于地质指标的风险计算；
- 基于 TBM 监测数据的 LSTM、XGBoost、Transformer 等预测模型。

不足不是“预测不准”，而是：

> 它们多数把掘进过程表达为曲线、表格或分段值，缺乏将响应变化定位到 TBM 局部表面的空间表达。

可写句：

> Existing monitoring-based models can learn temporal dependence in TBM responses, while segment-based risk models can provide interval-level assessments. However, both representations usually remain at the curve or segment level and do not explicitly localize response-related information on TBM surface regions.

## 5.3 第三段：本文思路

目标：提出“表面分区级响应定位”。

建议内容：

- 将护盾划分为轴向 × 周向表面分区；
- 将刀盘划分为径向环带 × 周向扇区；
- 利用 TSP 地质体素为每个分区提供局部围岩上下文；
- 利用几何邻近和法向一致性构建候选岩–机关联；
- 使用短时程监测响应作为弱监督，学习分区级响应一致性相关性。

## 5.4 第四段：科学问题

建议提出三个问题：

1. 如何以工程可解释的方式划分 TBM 刀盘与护盾表面，使其能够承载响应定位结果？
2. 如何将 TSP 岩体体素与 TBM 表面分区建立几何合理的候选关联？
3. 如何利用连续监测响应学习表面分区的响应一致性相关性，并验证这种空间定位表达优于单纯监测曲线或整体地质统计？

## 5.5 第五段：贡献

建议四条贡献，不要多。

1. 提出 TBM 表面分区级响应定位表达，将监测响应由一维里程曲线映射到刀盘与护盾局部表面区域。
2. 设计工程可解释的 TBM 表面分区方法：护盾采用轴向–周向分区，刀盘采用径向环带–周向扇区分区。
3. 构建几何约束岩体–表面分区候选关联，利用 TSP 体素属性形成局部岩–机交互特征。
4. 通过响应监督学习、基线对比、随机关联消融和原型可视化验证表面分区级定位的有效性。

---

# 6. Related Work 撰写方案

Related Work 不要写太长，建议 3 小节。

## 6.1 TBM response prediction and risk assessment

重点：说明已有方法主要是曲线、表格、分段风险。

写作逻辑：

- TBM 参数预测：推力、扭矩、贯入度、推进速度预测；
- 风险评估：基于地质条件、监测指标和专家规则；
- 不足：空间定位能力弱。

## 6.2 Spatial data integration in digital tunnelling

重点：TSP、地质体素、监测数据的空间对齐。

写作逻辑：

- 超前地质预报、地质素描、监测日志提供多源信息；
- GIS / Digital Earth 关注工程对象的空间组织；
- 本文把 TSP 地质体素和 TBM 表面分区通过几何规则关联。

## 6.3 Surface-based interpretation and visual analytics

重点：支持原型系统和热点表达。

写作逻辑：

- 可视分析不仅展示预测值，还要展示空间解释；
- 本文的核心可视对象是 cutterhead / shield surface hotspots；
- 不是做完整 WebGIS 平台，而是做表面响应定位原型。

---

# 7. Methodology 详细方案

## 7.1 总体流程

整体流程可写为：

```text
TSP 体素数据 + TBM 监测数据 + TBM 几何参数
        ↓
里程统一与掘进步索引
        ↓
刀盘/护盾表面工程分区
        ↓
几何约束筛选局部岩体体素
        ↓
构建表面分区级局部交互特征 P_t
        ↓
响应监督学习表面相关性 C_j(t)
        ↓
输出护盾/刀盘热点图与沿里程演化图
```

方法图 Figure 1 应该只画这条线，不要画太多复杂模块。

---

## 7.2 3.1 Surface partitioning of TBM cutterhead and shield

### 7.2.1 护盾表面分区

护盾是圆柱壳体，沿 TBM 掘进方向为 X 轴。定义：

- 护盾半径：\(R_s\)
- 护盾长度：\(L_s\)
- 轴向分区间隔：\(\Delta x_s\)
- 周向分区数：\(N_\theta\)

主实验推荐：

```text
Δx_s = 1 m
N_θ = 12
```

第 \(q\) 个轴向段、第 \(l\) 个周向段对应一个护盾 patch：

\[
S^{shield}_{q,l}
\]

其中心点：

\[
p_{q,l}(t)=
[x_{front}(t)-q\Delta x_s,
R_s\cos\theta_l,
R_s\sin\theta_l]
\]

其中：

\[
\theta_l=\frac{2\pi l}{N_\theta}
\]

外法向：

\[
n_{q,l}=[0,\cos\theta_l,\sin\theta_l]
\]

工程解释：

> 每个护盾 patch 表示某一轴向位置、某一周向方向上的局部护盾表面区域。

### 7.2.2 刀盘表面分区

刀盘采用粗粒度工程分区，不做单滚刀级建模。

主实验推荐：

```text
径向：3 个环带
  ring 1: 中心区
  ring 2: 主切削区
  ring 3: 边缘/保径区
周向：12 个扇区
```

得到：

```text
3 × 12 = 36 个刀盘 patch
```

第 \(k\) 个径向环带、第 \(l\) 个周向扇区定义为：

\[
S^{cutter}_{k,l}
\]

其中心点可近似为：

\[
p_{k,l}(t)=
[x_{face}(t), \bar r_k\cos\theta_l, \bar r_k\sin\theta_l]
\]

刀盘正面法向：

\[
n_{k,l}=[1,0,0]
\]

说明：

> 刀盘分区体现不同径向切削区的工程差异，可概括滚刀布置造成的径向差异，但本文不进行单滚刀级响应定位。

### 7.2.3 为什么这样分区

论文中必须解释：

1. 分区尺度与 TSP 体素尺度匹配，避免假精细化；
2. 护盾轴向–周向分区便于展开成二维图；
3. 刀盘径向–周向分区符合刀盘工程结构；
4. 不直接使用 GLB/STL 顶点，避免网格密度偏差。

---

## 7.3 3.2 Geometry-constrained rock–machine association construction

### 7.3.1 输入数据

每个 TSP 体素：

\[
v_i^r=(c_i,g_i)
\]

其中：

\[
c_i=(x_i,y_i,z_i)
\]

\[
g_i=[Vp_i,Vs_i,E_i,Vp/Vs_i,Pr_i,\rho_i]
\]

每个 TBM 表面 patch：

\[
s_j=(p_j,n_j,\rho_j)
\]

其中：

- \(p_j\)：patch 中心点；
- \(n_j\)：patch 外法向；
- \(\rho_j\)：patch 类型，取 cutterhead 或 shield。

### 7.3.2 候选关系规则

对每个掘进步 \(t\)，对每个 patch \(j\)，寻找候选岩体体素 \(i\)。候选关系满足三类约束。

#### 距离约束

\[
d_{ij}(t)=\|c_i-p_j(t)\|\le \tau
\]

主实验推荐：

\[
\tau=1.5m
\]

#### 法向一致性约束

\[
\kappa_{ij}(t)=
\max\left(0,
\frac{n_j(t)^T(c_i-p_j(t))}{\|c_i-p_j(t)\|+\epsilon}
\right)
\]

保留：

\[
\kappa_{ij}(t)\ge 0.2
\]

#### 最近邻数量约束

每个 patch 最多保留最近 \(k\) 个岩体体素：

\[
k=8
\]

若候选不足 8 个，则保留所有满足条件的候选体素。

### 7.3.3 分部件 active zone

#### 刀盘 active zone

刀盘关注掌子面前方局部区域：

\[
\Omega_t^{cutter}=
\{c_i | 0\le x_i-x_{face}(t)\le L_f,
\sqrt{y_i^2+z_i^2}\le R_c+\tau \}
\]

推荐：

\[
L_f=\tau \text{ 或 } 2\tau
\]

#### 护盾 active zone

护盾关注隧道边界周边环形区域：

\[
\Omega_t^{shield}=
\{c_i | x_{tail}(t)\le x_i\le x_{face}(t),
R_s\le \sqrt{y_i^2+z_i^2}\le R_s+\tau \}
\]

解释：

> 护盾不是与前方掌子面岩体作用，而是与盾体周边围岩产生局部空间关联，因此需要单独定义护盾 active zone。

---

## 7.4 3.3 Response-supervised surface-patch localization model

### 7.4.1 Patch-level local interaction features

对每个 patch \(j\)，将其邻近候选岩体体素集合记为：

\[
\mathcal N_j(t)
\]

构建分区级特征：

\[
F_j(t)=
[N_j,
\overline d_j,
\overline \kappa_j,
\overline E_j,
\overline{Pr}_j,
\overline{Vp/Vs}_j,
\sigma(E)_j,
\sigma(Pr)_j]
\]

其中：

- \(N_j\)：候选体素数量；
- \(\overline d_j\)：平均距离；
- \(\overline \kappa_j\)：平均法向一致性；
- \(\overline E_j\)：邻近岩体弹性模量均值；
- \(\overline{Pr}_j\)：邻近岩体泊松比均值；
- \(\overline{Vp/Vs}_j\)：波速比均值；
- \(\sigma(E)_j\)、\(\sigma(Pr)_j\)：局部异质性。

所有 patch 组成表面分区矩阵：

\[
P_t\in \mathbb R^{N_{patch}\times F}
\]

### 7.4.2 Monitoring input

监测序列：

\[
u_t=[Thrust_t,Torque_t,Penetration_t,AdvanceRate_t,RPM_t]
\]

若有盾体压力数据，可加入：

\[
ShieldPressure_t
\]

### 7.4.3 模型输入输出

输入：

\[
(P_{t-K+1:t},u_{t-K+1:t})
\]

输出：

\[
\hat r_{t+1}
\]

其中：

\[
r_{t+1}=[Thrust_{t+1},Torque_{t+1},Penetration_{t+1},AdvanceRate_{t+1}]
\]

主实验参数：

```text
K = 5
h = 1
step = 1 m
```

### 7.4.4 简化模型结构

不建议一开始用复杂 GNN，建议采用：

```text
Patch Encoder: MLP
Patch Attention / Gating: MLP + sigmoid
Surface Pooling: weighted pooling
Temporal Encoder: GRU
Response Head: MLP
```

计算流程：

1. 对每个 patch 特征编码：

\[
h_j(t)=MLP_{patch}(F_j(t))
\]

2. 计算 patch 响应相关性分数：

\[
c_j(t)=\sigma(MLP_{gate}(h_j(t)))
\]

3. 聚合表面表示：

\[
z_t=\sum_j c_j(t)h_j(t)
\]

也可拼接平均池化：

\[
z_t=[\sum_j c_j(t)h_j(t) \Vert mean_j(h_j(t))]
\]

4. 拼接监测向量：

\[
\tilde z_t=[z_t\Vert u_t]
\]

5. GRU 建模时间序列：

\[
s_t=GRU(\tilde z_{t-K+1:t})
\]

6. 输出后续响应：

\[
\hat r_{t+1}=MLP_{resp}(s_t)
\]

### 7.4.5 损失函数

所有响应变量先标准化：

\[
\bar r^{(m)}=\frac{r^{(m)}-\mu_m}{\sigma_m}
\]

使用 Huber loss：

\[
L=\sum_m\lambda_mHuber(\hat{\bar r}^{(m)}_{t+1}-\bar r^{(m)}_{t+1})
\]

原因：

- 不同响应量纲差异大；
- TBM 监测存在尖峰、停机、操作扰动；
- Huber loss 比 MSE 更稳健。

### 7.4.6 表面热点定义

patch 级热点分数：

\[
C_j(t)=c_j(t)
\]

若希望结合几何候选关系强度，可定义：

\[
C_j(t)=c_j(t)\cdot \frac{1}{|\mathcal N_j|+\epsilon}
\sum_{i\in\mathcal N_j}\kappa_{ij}\exp(-d_{ij}/\tau)
\]

但主文建议先用简单版本：

\[
C_j(t)=c_j(t)
\]

因为它更清楚：热点来自响应监督学习，而几何关系已经体现在 \(F_j(t)\) 中。

---

## 7.5 3.4 Prototype visualization for surface response localization

原型系统只做最小闭环，不做大型平台。

### 7.5.1 系统定位

中文：

> TBM 表面响应定位原型系统

英文：

> Prototype system for TBM surface response localization

### 7.5.2 系统模块

```text
Data alignment module
Surface partition module
Rock–surface association module
Response localization module
Visualization module
```

### 7.5.3 必须展示的三个视图

1. 护盾展开热点图

```text
x-axis: shield axial position
y-axis: circumferential sector
color: C_j(t)
```

2. 刀盘分区热点图

```text
radial ring × circumferential sector
color: C_j(t)
```

3. 沿里程热点演化图

```text
x-axis: chainage
y-axis: patch index / circumferential sector
color: C_j(t)
```

### 7.5.4 点击查询内容

点击某个 patch 后返回：

```text
patch type: cutterhead / shield
patch position: axial-ring / circumferential sector
candidate voxel count
mean E
mean Pr
mean Vp/Vs
mean distance
mean normal compatibility
corresponding monitoring response
```

### 7.5.5 技术实现建议

MVP：

```text
Python + Plotly / PyVista / Matplotlib
```

不急于做 WebGIS。

如果需要更好展示：

```text
Three.js + FastAPI
```

但论文阶段 Python 原型足够。

---

# 8. 算法伪代码

## Algorithm 1. TBM surface patch construction

```text
Input:
  TBM geometric parameters: cutterhead radius Rc, shield radius Rs, shield length Ls
  surface partition settings: shield axial interval Δxs, circumferential number Nθ, cutterhead ring number Nr
  excavation step t

Output:
  TBM surface patches S_t

1. Construct shield patches:
   For each axial segment q:
     For each circumferential sector l:
       Compute patch centroid p_j(t)
       Compute outward normal n_j
       Assign component label shield

2. Construct cutterhead patches:
   Divide cutterhead into Nr radial rings
   For each ring k:
     For each circumferential sector l:
       Compute patch centroid p_j(t)
       Assign normal n_j = [1,0,0]
       Assign component label cutterhead

3. Save all surface patches with local coordinates and component labels.
```

## Algorithm 2. Geometry-constrained rock–surface association

```text
Input:
  TSP voxel field {c_i, g_i}
  TBM surface patches {p_j, n_j}
  chainage step t
  distance threshold τ
  normal threshold ηmin
  maximum neighbours k

Output:
  Candidate association set A_t and patch-level features P_t

For each patch j:
  1. Select rock voxels in the component-specific active zone.
  2. Compute distance d_ij between patch centroid and voxel center.
  3. Retain voxels satisfying d_ij ≤ τ.
  4. Compute normal compatibility κ_ij.
  5. Retain voxels satisfying κ_ij ≥ ηmin.
  6. Keep the nearest k voxels if more than k candidates remain.
  7. Aggregate geological and geometric attributes into patch feature F_j(t).
Return P_t = {F_j(t)}.
```

## Algorithm 3. Response-supervised surface localization

```text
Input:
  patch feature sequence P_{t-K+1:t}
  monitoring sequence u_{t-K+1:t}
  response label r_{t+1}

Output:
  predicted response r_hat_{t+1}
  surface hotspot scores C_j(t)

1. Encode each patch feature F_j(t) using MLP_patch.
2. Compute patch relevance score C_j(t) using sigmoid gate.
3. Aggregate patch embeddings into surface representation z_t.
4. Concatenate z_t with monitoring vector u_t.
5. Feed the sequence into GRU.
6. Predict response r_hat_{t+1} using MLP response head.
7. Compute standardized Huber loss.
8. Use learned C_j(t) as response-consistent surface hotspot score.
```

---

# 9. 实验设计

## 9.1 数据组织

### TSP 体素数据

字段建议：

```text
X, Y, Z, Vp, Vs, ro, E, Vp_Vs, Pr
```

### TBM 几何参数

字段建议：

```text
cutterhead_radius
shield_radius
shield_length
advance_direction
initial_chainage
```

### TBM 监测数据

字段建议：

```text
chainage
thrust
torque
penetration
advance_rate
rpm
shield_pressure(optional)
```

### 里程对齐

统一到 1 m 掘进步：

```text
step = 1 m
```

对监测数据按每米聚合：

```text
mean, median, std, p95
```

主实验可以只用 mean，异常响应分析可用 p95。

---

## 9.2 数据划分

必须按里程顺序划分，不能随机划分。

推荐：

```text
train: first 70%
validation: next 15%
test: last 15%
```

原因：避免相邻里程泄漏。

---

## 9.3 Baseline

只保留必要基线，避免过多。

| 模型 | 输入 | 目的 |
|---|---|---|
| Persistence | 上一步响应 | 最低基线 |
| Monitoring-only GRU | 历史监测序列 | 检查监测自相关有多强 |
| TSP-statistics + GRU | 历史监测 + active zone 整体 TSP 统计 | 检查简单地质统计是否足够 |
| Surface-patch model | 历史监测 + 表面分区局部特征 | 验证表面空间定位是否有额外价值 |

最关键比较：

```text
Surface-patch model vs Monitoring-only GRU
Surface-patch model vs TSP-statistics + GRU
```

---

## 9.4 消融实验

必须做两个，最多做三个。

### 9.4.1 Randomized patch assignment

做法：

- 保持每个 patch 的候选体素数量不变；
- 随机打乱岩体体素与 patch 的对应关系；
- 重新聚合 patch 特征并训练/测试。

目的：

> 证明真实空间对应关系不是装饰性输入。

### 9.4.2 No normal compatibility

做法：

- 只用距离阈值；
- 去掉法向一致性约束。

目的：

> 证明法向一致性约束有助于避免不合理关联。

### 9.4.3 No surface partition

做法：

- 将所有 active zone 体素整体聚合为一个全局 TSP 特征；
- 不区分 TBM 表面分区。

目的：

> 证明表面分区表达比整体地质统计更有空间信息。

---

## 9.5 评价指标

连续响应预测：

```text
MAE
RMSE
R²
Pearson correlation
```

建议主表格用：

```text
MAE, RMSE, R²
```

相关性放补充或图中。

注意表述：

> Prediction performance is used as functional validation rather than the sole objective of this study.

---

## 9.6 分场景评价

如果数据量允许，可额外分场景：

```text
全部区段
高推力区段
高扭矩区段
低贯入度区段
低推进速度区段
地质突变区段
```

目的：

> 检查表面分区特征在困难掘进段是否更有价值。

不要强行做卡机风险分类，除非有可靠标签。

---

# 10. 图表设计

## Figure 1. Research framework

四栏：

```text
Data → Surface partition → Rock–surface association → Response localization
```

不要画复杂系统大架构。

## Figure 2. TBM surface partition

展示：

- 护盾轴向 × 周向分区；
- 刀盘 3 环带 × 12 周向扇区；
- 分区编号方式。

## Figure 3. Geometry-constrained association

展示：

- 某一掘进步；
- 表面 patch；
- 邻近岩体体素；
- 距离阈值和法向一致性筛选。

## Figure 4. Functional validation results

展示：

- 真实响应曲线；
- Monitoring-only GRU；
- TSP-statistics + GRU；
- Surface-patch model。

建议对一个或两个核心响应变量展示，比如 thrust 和 penetration。

## Figure 5. Surface hotspot map

展示：

- 护盾展开热点图；
- 刀盘分区热点图。

## Figure 6. Hotspot evolution along chainage

展示：

```text
x-axis: chainage
y-axis: patch / circumferential sector
color: C_j(t)
```

下方叠加推力、扭矩或贯入度曲线。

## Figure 7. Prototype demonstration

展示 3 个窗口：

1. 当前里程 TBM 表面热点；
2. 点击 patch 后的局部岩体属性；
3. 沿里程热点演化。

## Table 1. Data summary

内容：

```text
TSP 体素数量
监测记录数量
里程范围
表面 patch 数量
掘进步数量
训练/验证/测试样本数
```

## Table 2. Model comparison

列：

```text
Model, Input, MAE, RMSE, R²
```

## Table 3. Ablation study

列：

```text
Variant, Removed/changed component, MAE, RMSE, R²
```

---

# 11. Results 写作方案

## 11.1 Data alignment and surface-patch construction

必须报告：

- TSP 体素数量；
- 监测数据聚合方式；
- 护盾 patch 数；
- 刀盘 patch 数；
- 每个掘进步平均候选关联数。

关键结论：

> The surface partition provides a stable spatial support for localizing response-related information on the TBM surface.

## 11.2 Validity of geometry-constrained association

报告：

- 距离分布；
- 法向一致性分布；
- 每个 patch 候选体素数量分布；
- 去掉法向约束后的不合理连边对比。

关键结论：

> The candidate associations are spatially constrained around the cutterhead and shield surfaces, rather than arbitrary links between geological voxels and machine regions.

## 11.3 Functional validation through response prediction

报告：

- baselines 对比；
- 预测曲线；
- 核心指标。

写法边界：

不要写：

```text
The proposed model achieves state-of-the-art prediction performance.
```

建议写：

```text
The improved response prediction indicates that surface-patch associations preserve additional spatial information beyond monitoring-only sequences and aggregated geological statistics.
```

## 11.4 Ablation analysis

重点写 randomized patch assignment。

如果 randomized 下降明显，结论：

> The spatial correspondence between geological voxels and TBM surface patches contributes to the model performance.

如果下降不明显，要诚实解释：

> The current data may not provide sufficient spatial variation to fully distinguish the contribution of patch-level associations.

## 11.5 Surface hotspot localization

展示热点图并解释：

- 哪些护盾周向区域热点高；
- 哪些轴向段热点持续；
- 刀盘中心区、主切削区或边缘区是否有差异；
- 热点是否与推力、扭矩、贯入度变化同步。

边界：

> These hotspots indicate response-consistent surface relevance, not calibrated contact pressure or verified jamming locations.

## 11.6 Prototype demonstration

展示原型系统如何支持：

- 当前里程表面热点查看；
- patch 查询；
- 局部岩体属性查看；
- 沿里程热点演化分析。

不要把系统说成完整预警平台。

---

# 12. Discussion 写作方案

## 12.1 Why surface localization matters

强调：

- 曲线告诉我们“什么时候/哪一里程响应异常”；
- 表面定位进一步告诉我们“响应相关性集中在哪个 TBM 局部区域”；
- 这是 GIS / 空间分析贡献。

## 12.2 Why not cutter-level modeling

必须主动解释：

- 滚刀重要，但当前监测与地质数据尺度不足以支撑单滚刀级定位；
- 表面分区级是数据尺度、工程可解释性和可验证性之间的折中；
- 后续若有单刀载荷、刀具磨损或刀盘姿态数据，可扩展到滚刀级。

## 12.3 Scientific boundary

必须写清楚：

- 热点不是接触压力；
- 相关性不是因果；
- 响应预测是验证任务；
- TSP 属性有空间分辨率限制；
- 表面 patch 是工程分区，不是精细机械网格。

## 12.4 Limitations and future work

建议写四点：

1. 当前方法使用参数化 TBM 表面分区，后续可结合更精细的真实几何模型；
2. TSP 体素分辨率限制了热点定位精度；
3. 监测响应是弱监督信号，不能替代真实接触力或现场风险记录；
4. 后续可引入盾体分区压力、单刀载荷、刀具磨损和数值模拟结果进一步验证。

---

# 13. Conclusion 写作方案

结论三句话即可。

1. 本文提出 TBM 表面分区级响应定位方法，将刀盘与护盾划分为工程可解释的表面分区，并利用 TSP 岩体体素构建几何约束岩体–表面分区候选关联。
2. 通过短时程运行响应监督，模型学习表面分区的响应一致性相关性，并通过监测序列基线、整体地质统计基线和随机分区消融验证其空间信息价值。
3. 该方法将 TBM 监测响应从沿里程曲线推进到表面局部热点表达，为复杂地质条件下 TBM 掘进过程的空间化解释与可视分析提供了可实现路径。

---

# 14. 实现路线

## 14.1 项目目录建议

```text
tbm_surface_localization/
  data/
    tsp_voxels.csv
    tbm_monitoring.csv
    tbm_geometry.yaml
  src/
    01_align_chainage.py
    02_build_surface_patches.py
    03_construct_associations.py
    04_extract_patch_features.py
    05_train_model.py
    06_evaluate_baselines.py
    07_visualize_hotspots.py
  outputs/
    patch_features/
    models/
    figures/
    prototype/
  configs/
    experiment_main.yaml
```

## 14.2 Step 1：数据检查与里程对齐

输入：

```text
TSP_data_cropped.csv
TBM monitoring records
```

输出：

```text
chainage-indexed TSP voxel field
chainage-indexed monitoring table
```

要做：

- 检查坐标轴方向；
- 明确 X 是否为掘进方向；
- 将监测数据聚合到 1 m；
- 对 TSP 体素进行坐标平移，使其与 TBM 当前里程对应。

## 14.3 Step 2：构建 TBM 表面 patch

参数：

```text
shield_radius = 5.2 m 或根据工程确定
shield_length = 10 m 或根据工程确定
cutterhead_radius = 5.0 m 或根据工程确定
shield_axial_interval = 1 m
circumferential_sectors = 12
cutterhead_rings = 3
```

输出：

```text
surface_patches.csv
```

字段：

```text
patch_id
component
axial_id / ring_id
theta_id
center_x, center_y, center_z
normal_x, normal_y, normal_z
area(optional)
```

## 14.4 Step 3：构建岩体–patch 候选关联

参数：

```text
τ = 1.5 m
ηmin = 0.2
k = 8
```

输出：

```text
associations_t.csv
```

字段：

```text
step
chainage
patch_id
voxel_id
distance
normal_compatibility
E
Pr
Vp_Vs
```

## 14.5 Step 4：提取 patch 特征

输出：

```text
patch_features.npy 或 patch_features.parquet
```

形状：

```text
[T, N_patch, F]
```

推荐特征：

```text
candidate_count
mean_distance
mean_kappa
mean_E
mean_Pr
mean_Vp_Vs
std_E
std_Pr
```

## 14.6 Step 5：训练模型

模型：

```text
Patch MLP + Gate + GRU + Response Head
```

输入：

```text
P_{t-4:t}, u_{t-4:t}
```

输出：

```text
r_{t+1}
```

训练细节：

```text
optimizer: Adam
loss: Huber
batch size: 16 or 32
learning rate: 1e-3
standardization: yes
early stopping: validation loss
```

## 14.7 Step 6：基线与消融

基线：

```text
Persistence
Monitoring-only GRU
TSP-statistics + GRU
Surface-patch model
```

消融：

```text
Randomized patch assignment
No normal compatibility
No surface partition
```

## 14.8 Step 7：可视化

必须输出：

```text
shield_unfold_hotspot.png
cutterhead_patch_hotspot.png
chainage_hotspot_evolution.png
prediction_comparison.png
prototype_demo.png
```

---

# 15. 参数最终建议

| 模块 | 参数 | 主实验值 | 敏感性分析 |
|---|---:|---:|---:|
| 里程步长 | step | 1 m | 0.5 / 2 m |
| 护盾轴向分区 | Δx_s | 1 m | 0.5 / 2 m |
| 护盾周向分区 | N_θ | 12 | 8 / 16 |
| 刀盘径向环带 | N_r | 3 | 4 |
| 距离阈值 | τ | 1.5 m | 1.0 / 2.0 m |
| 法向阈值 | ηmin | 0.2 | 0 / 0.4 |
| 最近邻数量 | k | 8 | 4 / 16 |
| 历史窗口 | K | 5 | 3 / 10 |
| 预测步长 | h | 1 | 不建议主文扩展太多 |

主文只报告核心参数，敏感性分析可放附录或补充。

---

# 16. 写作中必须避免的问题

## 16.1 不要泛化为大框架

避免：

> 本文构建了面向 TBM 全过程施工的数字孪生智能分析平台。

改为：

> 本文实现了一个用于展示表面响应定位结果的轻量化原型。

## 16.2 不要声称真实物理接触

避免：

> The hotspot indicates high contact pressure.

改为：

> The hotspot indicates high response-consistent surface relevance.

## 16.3 不要声称卡机预测

避免：

> The method predicts jamming risk.

改为：

> The method provides spatial localization of response-related surface regions, which may support subsequent risk interpretation.

## 16.4 不要过度强调 AI 模型

避免：

> A powerful deep learning model is developed.

改为：

> A lightweight response-supervised model is used to validate whether surface-patch associations contain useful spatial information.

---

# 17. 最小可发表闭环

必须完成：

1. TBM 表面分区构建；
2. TSP 体素与表面分区候选关联；
3. patch 特征矩阵；
4. Monitoring-only vs Surface-patch model 对比；
5. Randomized patch assignment 消融；
6. 护盾/刀盘热点图；
7. 沿里程热点演化图；
8. 轻量原型展示。

只要这 8 件事完成，论文逻辑就是闭合的。

---

# 18. 最终摘要草稿

TBM operational responses, such as thrust, torque, penetration and advance rate, are commonly represented as chainage-indexed monitoring curves. Although such representations are useful for detecting abnormal intervals, they provide limited information on where response-related rock–machine interactions may occur on the cutterhead or shield surface. This study proposes a surface-patch-based response localization method for TBM excavation processes. The cutterhead and shield are divided into engineering-interpretable surface patches, and TSP-derived geological voxels are associated with each patch using distance, normal compatibility and component-specific active-zone constraints. Patch-level geological and geometric descriptors are then constructed and used, together with historical monitoring sequences, to predict short-horizon operational responses. The prediction task is used as a functional validation to learn response-consistent surface relevance rather than as the sole objective of the study. Experiments compare the proposed model with monitoring-only and aggregated geological baselines, and structural ablations are conducted by randomizing voxel–patch associations and removing geometric constraints. The learned relevance scores are mapped back to cutterhead and shield patches to generate surface hotspots and chainage-evolving localization views. The results demonstrate that surface-patch associations can preserve spatial information beyond monitoring curves and aggregated geological statistics, providing a practical spatial representation for interpreting TBM responses in complex geological conditions.

---

# 19. 中文摘要草稿

TBM 掘进过程中的推力、扭矩、贯入度和推进速度等运行响应通常被表达为沿里程变化的监测曲线。该类表达能够识别异常里程区段，但难以进一步说明响应变化更可能关联到刀盘或护盾的哪个局部表面区域。针对这一问题，本文提出一种 TBM 表面分区级响应定位方法。首先，将刀盘和护盾划分为工程可解释的表面分区；其次，利用距离、法向一致性和部件特定 active zone 约束，将 TSP 地质体素与各表面分区建立候选岩–机关联；然后，构建分区级地质—几何描述符，并结合历史监测序列预测短时程运行响应。本文将响应预测作为功能性验证任务，用于学习表面分区的响应一致性相关性，而非将其解释为真实接触力或卡机概率。实验通过与纯监测序列模型和整体地质统计模型对比，并结合随机分区关联和几何约束消融，验证表面分区空间关联的有效性。最终，学习得到的相关性被映射回刀盘和护盾表面，形成表面热点和沿里程演化视图。结果表明，表面分区级岩–机关联能够保留监测曲线和整体地质统计难以表达的空间信息，为复杂地质条件下 TBM 运行响应的空间化解释提供了一种可实现方法。

---

# 20. 最终一句话

> 本文的关键不在于提出一个更复杂的 TBM 响应预测模型，而在于把沿里程变化的监测响应重新定位到刀盘和护盾的局部表面分区，并通过几何约束岩体–表面关联证明这种空间定位表达具有可验证的信息价值。

