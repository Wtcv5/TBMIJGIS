我建议的最终章节结构
1. Introduction

引言不要泛泛说 TBM 卡机风险，也不要一上来讲 GNN。你要按四段写：

第一段：工程问题。
TBM 掘进响应异常与前方/周边地质条件密切相关，但地质体与刀盘、护盾之间的作用关系是随掘进推进变化的空间关系。

第二段：现有方法不足。
多数方法把 TSP、地质指标和 TBM 参数处理成表格变量或时间序列变量，能够做趋势预测，但难以回答“哪个岩体空间单元与哪个机器部件形成了候选作用关系”。

第三段：本文切入点。
不是直接追求更复杂预测模型，而是构建一种里程参照下的 rock–TBM interaction graph sequence，把岩体体素、TBM 表面采样、运行响应序列组织成可计算的时空图。

第四段：贡献。
贡献要保守，建议写成三条：

提出 chainage-referenced rock–TBM spatial representation；
提出 geometry-constrained rock–machine interaction graph sequence；
通过 response-supervised prediction 检验图关系的响应一致性，并输出 edge/component-level relevance diagnostics。

不要写“显著提高预测精度”或“实现卡机风险精确预警”。你的初稿结果已经明确：BSLL 不支持 graph sequence 提高预测，SJLS 也只是小幅 case-specific geometry-related difference。

2. Related work

这里建议只保留三小节，而且每节都要服务你的 gap。

2.1 TBM response prediction and jamming-related analysis

讲 TBM 参数预测、卡机风险、掘进响应异常识别。最后落到一句：

existing studies usually represent geological and machine variables as feature vectors or sequences, rather than explicit spatial relations between geological entities and TBM components.

2.2 Spatial representation of geological conditions in tunnel excavation

讲 TSP、超前地质预报、体素化、三维地质表达。重点不是综述所有地质建模，而是引出：

TSP-derived geological information can be transformed into chainage-referenced rock voxels, but this representation still needs to be linked to machine geometry.

2.3 Graph-based spatiotemporal prediction and spatial interaction modelling

讲 GNN、动态图、时空图预测。这里要强调 graph 的意义是表达空间实体与关系，不是简单“因为 GNN 新”。你可以借鉴 IJGIS 交通预测论文的逻辑：先定义 spatial graph、feature matrix 和 prediction problem，再进入 graph construction 和模型，而不是一开始堆网络结构。

3. Methodology

这一章必须收紧成三节。不要再拆成“数据对齐、岩体节点、TBM 节点、边、GNN、GRU、解释”六七节。

3.1 Chainage-referenced rock–TBM representation

这一节合并原来的数据对齐、岩体节点、TBM 表面节点、监测序列定义。

它要回答一个问题：

一个掘进步 t 下，岩体、TBM 和响应如何被统一表达？

具体写法：

先定义任务：

r
^
t+h
	​

=f
Θ
	​

(G
t−K+1:t
	​

,u
t−K+1:t
	​

)

你的方法文档和初稿里已经这样定义了：给定过去 K 个图快照和监测序列，预测未来 h 步响应。

然后定义三个对象：

rock voxels

D
geo
	​

={(c
i
	​

,g
i
	​

)}

TBM surface samples

M
TBM
	​

={p
j
	​

(t),n
j
	​

(t),ρ
j
	​

}

monitoring response sequence

u
t
	​

=[AdvanceRate,RPM,Torque,Thrust,Penetration,ShieldPressure]

这一节不要展开边构建。只讲“空间对象如何进入同一 chainage-referenced representation”。

3.2 Geometry-constrained rock–machine graph sequence

这是你的核心方法节。必须写得最扎实。

开头直接定义：

G
t
	​

=(V
t
r
	​

∪V
t
m
	​

, E
t
rr
	​

∪E
t
mm
	​

∪E
t
rm
	​

)

其中 E
rr
 表达岩体空间邻接，E
mm
 表达 TBM 表面结构邻接，E
rm
 表达岩–机候选交互边。你的方法文档里已经明确了这些边的含义。

然后按三个关系讲：

3.2.1 Rock–rock adjacency

一句话即可。岩体节点间用空间邻域表达地质连续性。

3.2.2 Machine–machine adjacency

一句话即可。刀盘/护盾表面节点之间用径向、周向、轴向邻接表达机器表面连续性。

3.2.3 Rock–machine candidate interaction edges

这是重点。候选边满足：

d
ij
	​

(t)≤τ,κ
ij
	​

(t)≥η
min
	​

,c
i
	​

∈Ω
t
	​

,s
i
	​

(t)=1

并说明 active zone、distance threshold、normal compatibility、excavation state 的意义。这里一定加一句边界：

These constraints do not identify real contact. They restrict the candidate interaction space to geometrically plausible rock–machine relations.

因为方法文档已经写明，这一步不是得到真实接触关系，而是避免模型学习空间上不合理的远距离岩–机关联。

3.3 Response-supervised prediction and interaction relevance diagnostics

这一节把模型和解释合并，不要单独搞一个大节叫 interpretation。因为你的 relevance 是从预测任务里来的，不是独立解释模型。

结构可以这样：

3.3.1 Graph-sequence encoding

每个 G
t
	​

 经 GNN 得到图级表示 z
t
	​

。

3.3.2 Temporal response prediction

把 z
t
	​

 与 u
t
	​

 拼接，再用 GRU 预测：

r
^
t+h
	​

3.3.3 Interaction relevance diagnostics

解释 α
ij
rm
	​

(t) 或 derived relevance。这里必须写清：

learned relevance is response-consistent model diagnostics, not contact force, contact pressure, jamming probability, or engineering risk.

初稿和方法文档都已经明确了这个边界。

如果你要保留 Figure 9 这种 component-resolved surface relevance，那么这里可以写成：

R
c
	​

(t)=
∣E
c
	​

(t)∣
1
	​

(i,j)∈E
c
	​

(t)
∑
	​

α
ij
rm
	​

(t)

其中 c 是 cutterhead、front shield、middle shield、tail shield。这样就和你 Figure 9 的 component-resolved relevance 对上，而不是泛泛讲“热力图”。

4. Case studies and experimental design

这一章不要叫 “Experiments” 就完了。你现在的实验不是单纯模型 benchmark，而是两个 case role。建议标题写成：

4. Case studies and experimental design
4.1 Case roles and data sources

这一节必须明确两个案例的角色：

BSLL DyK1017+205：内部 case，做 h=1 和 h=3；
SJLS Dyk1252+411：external TSP-derived velocity field case，做 h=1。

初稿里已经说明三组 formal runs：BSLL h=1、BSLL h=3、SJLS h=1。

这里不要把两个案例包装成“充分泛化验证”。更准确是：

BSLL is used to examine one-step and multi-step settings, while SJLS provides an external TSP contrast.

4.2 Excavation-step samples and graph settings

这里放样本量和参数：

BSLL h=1：train 30, val 6, test 8；
BSLL h=3：train 29, val 6, test 7；
SJLS h=1：train 76, val 16, test 17；
K=5；
τ
edge
	​

=2.0m，η
min
	​

=0.3，τ
zone
	​

=5.0m。

这些是你初稿已有实验设置。

4.3 Baselines and graph ablations

这里不要写成“baseline models comparison”那么大。建议叫：

Baselines and graph ablation settings

分两组：

Prediction baselines

Persistence；
monitoring-only model，如果你有；
full graph-sequence model。

Graph ablations

Random edges；
No geometry prior；
No E
rm
；
No monitoring。

你现在 Table 3 已经有 Full、Random-edge、No-prior、No-monitoring。

4.4 Evaluation and reporting principle

这一节很重要。你要提前告诉读者：

prediction metrics are used as forward response checks, whereas relevance diagnostics are used to examine whether the learned graph relations are spatially organized.

这能避免审稿人用“没有显著预测提升”直接否掉全文。

5. Results

这一章必须按你真实结果来，而不是按理想 story。

5.1 Forward prediction check across case roles

这一节只做预测结果。标题里用 check，不要用 “superior performance”。

内容顺序：

先给 Table 2；
再解释 Figure 6；
最后谨慎总结。

建议结论写成：

The prediction results do not support a broad claim of forecasting improvement. In BSLL, the full graph-sequence model remains comparable to Persistence and random-edge variants. In SJLS, the full model shows a small geometry-related difference, but this should not be interpreted as a large accuracy breakthrough.

这完全对应你的初稿结果：BSLL h=1 Full MAE 0.4147，Persistence 0.4146，Random edges 0.4145；BSLL h=3 Random edges 甚至略低；SJLS Full 0.2553 仅略低于 Persistence 0.2554 和 Random edges 0.2559。

5.2 Geometry ablation and response consistency

这一节专门讨论 Table 3 / Figure 7。

不要说“几何约束显著提升性能”。要说：

The ablation results separate prediction performance from geometry-related response consistency.

初稿也已经写了：ablation 分离两个问题，一是 graph sequence 是否提高预测，二是 geometry-constrained relations 是否产生 interpretable spatial diagnostics，二者不能混为一谈。

这节重点放 SJLS：

No-prior: 0.2570；
Random-edge: 0.2559；
Full: 0.2553。

强调这是小差异，而不是 breakthrough。

5.3 Spatial organisation of learned relevance

这一节对应 Figure 8。

这里是你文章真正能讲出 GIScience 味道的部分。不要再纠结预测提升，而是讲：

learned relevance has positive spatial autocorrelation；
differs from geometry-only reference patterns；
is not a positive proxy for node degree；
relevance–geology / relevance–response associations are case-specific diagnostics.

这些都是你 Figure 8 图注里已有的结论。

5.4 Component-resolved relevance along chainage

这一节对应 Figure 9。

这里可以写：

Figure 9 projects degree-normalised response-supervised relevance to TBM components along chainage.

但必须马上加边界：

Colour intensity is not measured contact pressure or physical force.

因为你的 Figure 9 图注就是这么写的。

这节是你的“空间诊断结果”，比预测性能更重要。

5.5 Robustness and remaining checks

这节可以短，但建议保留。你初稿已经有 robustness checks，并明确说当前证据只支持三个 bounded claims：

competitive with temporal baselines；
SJLS shows a small case-specific geometry-related difference；
saved checkpoints produce spatially organised surface relevance。

同时不支持 broad global accuracy improvement、physical interaction recovery 或 operational risk calibration。

这段非常重要，放在结果最后或 Discussion 开头都可以。

6. Discussion

Discussion 不要再重复结果。建议三节。

6.1 What the graph representation adds beyond monitoring-only prediction

这里讨论：即使预测提升有限，图表示仍然有价值，因为它提供了 spatially organized rock–machine relation diagnostics。

6.2 Why prediction gains are limited

这里主动解释：

样本量很小；
mileage-ordered split 是外推设置；
TBM response 本身强时间自相关；
Persistence 是很强的 baseline；
TSP 和 TBM 响应存在尺度差异。

这会让审稿人觉得你诚实，而不是硬吹。

6.3 Interpretation boundary and future validation

这里必须写：

relevance ≠ contact pressure；
relevance ≠ jamming probability；
未来需要接触压力、护盾压力分布、卡机事件标签、施工处置记录做验证。
7. Conclusion

结论别写“显著提高预测精度”。写四点：

proposed chainage-referenced rock–TBM graph-sequence representation；
geometry-constrained candidate relations provide a spatially plausible interaction structure；
forward prediction checks show competitive but not universally superior performance；
learned relevance provides spatially organized component-level diagnostics, with clear physical interpretation limits.
最终目录建议
1. Introduction

2. Related work
   2.1 TBM response prediction and jamming-related analysis
   2.2 Spatial representation of tunnel geological conditions
   2.3 Graph-based spatiotemporal prediction and spatial interaction modelling

3. Methodology
   3.1 Chainage-referenced rock–TBM representation
   3.2 Geometry-constrained rock–machine graph sequence
   3.3 Response-supervised prediction and interaction relevance diagnostics

4. Case studies and experimental design
   4.1 Case roles and data sources
   4.2 Excavation-step samples and graph settings
   4.3 Baselines and graph ablation settings
   4.4 Evaluation and reporting principle

5. Results
   5.1 Forward prediction check across case roles
   5.2 Geometry ablation and response consistency
   5.3 Spatial organisation of learned relevance
   5.4 Component-resolved relevance along chainage
   5.5 Robustness and remaining checks

6. Discussion
   6.1 What the graph representation adds beyond monitoring-only prediction
   6.2 Why prediction gains are limited
   6.3 Interpretation boundary and future validation

7. Conclusion

这个结构的好处是：你不再把文章写成“模型性能论文”，而是写成“空间关系表达 + 谨慎预测验证 + 空间诊断解释论文”。这比泛泛说“构建图序列用于预测预警”稳很多。