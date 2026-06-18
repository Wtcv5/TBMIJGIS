# 实验管线：模块关系图与数据流图

## 一、代码模块关系图

```mermaid
graph TB
    subgraph Data["📦 data/"]
        direction TB
        RAW["data/raw/<br/>tsp.csv<br/>monitoring.csv"]
        PROC["data/processed/"]
    end

    subgraph SRC_DATA["src/data/ — 数据加载"]
        TSP["tsp_loader.py<br/>load_tsp()<br/>build_voxel_field()"]
        MON["monitoring.py<br/>load_monitoring()<br/>aggregate_by_chainage()<br/>standardize_monitoring()"]
        GEOM["tbm_geometry.py<br/>build_tbm_surface()<br/>advance_surface()"]
        ALIGN["alignment.py<br/>build_excavation_steps()<br/>mileage_split()"]
    end

    subgraph SRC_GRAPH["src/graph/ — 图构建 (Algorithm 1)"]
        NODES["nodes.py<br/>build_rock_nodes()<br/>build_tbm_nodes()<br/>active_rock_mask()"]
        EDGES["edges.py<br/>build_rock_edges_26nn()<br/>build_tbm_surface_edges()<br/>compute_rm_candidate_edges()<br/>encode_rm_edge_features()"]
        SEQ["sequence.py<br/>build_graph_snapshot()<br/>build_graph_sequence()"]
    end

    subgraph SRC_MODELS["src/models/ — 模型"]
        BASELINE["baselines.py<br/>Persistence<br/>TSPXGBoost<br/>LSTMBaseline<br/>TSPLSTM"]
        GNN["gnn.py<br/>RockTBMEncoder<br/>StaticGraphModel"]
        FULL["graph_seq.py<br/>GraphSequenceModel<br/>DynamicGraphOnly"]
    end

    subgraph SRC_TRAIN["src/training/ — 训练评估"]
        METRICS["metrics.py<br/>compute_all_metrics()<br/>compute_per_variable_metrics()<br/>compute_scenario_metrics()"]
        TRAINER["trainer.py<br/>StandardizedHuberLoss<br/>EarlyStopping<br/>train_sequence_model()"]
    end

    subgraph SRC_VIZ["src/visualization/ — 可视化"]
        GRAPH_VIZ["graph_viz.py<br/>plot_graph_snapshot()<br/>plot_edge_statistics()"]
        HOTSPOT["hotspot.py<br/>aggregate_attention_to_surface()<br/>plot_shield_hotspot()<br/>plot_hotspot_vs_response()"]
        PRED_VIZ["prediction.py<br/>plot_prediction_comparison()<br/>plot_metrics_bar()<br/>plot_ablation_results()"]
    end

    subgraph SCRIPTS["scripts/ — MVP 运行入口"]
        MVP1["mvp1_build_graph.py<br/>图构建与可视化"]
        MVP4["mvp4_full_model.py<br/>完整训练管线"]
    end

    subgraph OUTPUT["outputs/"]
        FIG1["graph_snapshot_*.png"]
        FIG2["edge_statistics.png"]
        FIG3["pred_comparison.png"]
        MODEL_PT["best_model.pt"]
    end

    %% 数据流
    RAW --> TSP
    RAW --> MON
    TSP -->|"coords (N,3)<br/>attrs (N,6)"| NODES
    MON -->|"标准化序列"| ALIGN
    GEOM -->|"TBM surface (M nodes)"| NODES
    GEOM -->|"TBM surface (M nodes)"| EDGES
    ALIGN -->|"ExcavationStep[]"| SEQ

    NODES -->|"rock nodes<br/>tbm nodes"| EDGES
    NODES --> SEQ
    EDGES -->|"E_rr, E_mm, E_rm<br/>+ geometry_prior"| SEQ
    SEQ -->|"GraphSnapshot[]"| GNN
    SEQ -->|"GraphSnapshot[]"| FULL

    ALIGN -->|"u_t sequences"| BASELINE
    ALIGN -->|"u_t sequences"| FULL

    BASELINE --> TRAINER
    GNN --> TRAINER
    FULL --> TRAINER

    TRAINER --> METRICS
    METRICS -->|"MAE/RMSE/R²/r"| PRED_VIZ
    SEQ --> GRAPH_VIZ
    GNN -->|"α_ij attention"| HOTSPOT
    FULL -->|"α_ij attention"| HOTSPOT

    GRAPH_VIZ --> FIG1
    GRAPH_VIZ --> FIG2
    HOTSPOT --> OUTPUT
    PRED_VIZ --> FIG3
    TRAINER --> MODEL_PT

    MVP1 --> TSP
    MVP1 --> GEOM
    MVP1 --> SEQ
    MVP1 --> GRAPH_VIZ

    MVP4 --> SEQ
    MVP4 --> FULL
    MVP4 --> BASELINE
    MVP4 --> TRAINER
    MVP4 --> PRED_VIZ
```

---

## 二、算法数据流图（对应论文 Algorithm 1 & 2）

```mermaid
flowchart TD
    subgraph INPUTS["输入数据"]
        A1["TSP 体素<br/>D_geo = {c_i, g_i}<br/>X,Y,Z, Vp,Vs,ρ,E,Vp_Vs,Pr"]
        A2["TBM 参数化几何<br/>M_tbm = {p_j(0), n_j, ρ_j}<br/>cutterhead/front/middle/tail"]
        A3["TBM 监测数据<br/>chainage, AR, RPM, Tor, Thr, Pen, SP"]
    end

    subgraph ALIGN["Step 1: 数据对齐"]
        B1["里程聚合<br/>step_length = 1.0m"]
        B2["坐标转换<br/>x_local(t) = chainage(t) − chainage_start"]
        B3["掘进步定义<br/>t = 1,...,T"]
    end

    subgraph GRAPH_BUILD["Step 2: 图序列构建 (Algorithm 1)"]
        C1["岩体节点 V_t^r<br/>x_i^r = [c_i, g_i, s_i(t)]<br/>active zone Ω_t 筛选"]
        C2["TBM 表面节点 V_t^m<br/>x_j^m = [p_j(t), n_j(t), ρ_j]<br/>推进平移 p_j(t) = p_j(0) + x·d"]
        C3["岩体内部边 E_t^rr<br/>26 邻域"]
        C4["TBM 表面边 E_t^mm<br/>网格邻接"]
        C5["岩-机候选边 E_t^rm<br/>d_ij ≤ τ, κ_ij ≥ η_min<br/>几何先验 π_ij = e^{−d/τ}·κ"]
        C6["图快照 G_t<br/>=(V^r∪V^m, E^rr∪E^mm∪E^rm)"]
        C7["图序列<br/>G = {G_1,...,G_T}"]
    end

    subgraph ENCODE["Step 3: 响应监督编码 (Algorithm 2)"]
        D1["节点编码<br/>h_i^r = MLP_r(x_i^r)<br/>h_j^m = MLP_m(x_j^m)"]
        D2["岩-机边注意力<br/>α_ij = softmax(MLP_att([h_i‖h_j‖a_ij])<br/>+ β·log(π_ij+ε))"]
        D3["消息聚合<br/>m_j = Σ_i α_ij · MLP_msg([h_i, a_ij])"]
        D4["图读出<br/>z_t = [mean(h^m) ‖ max(h^m)]"]
    end

    subgraph TEMPORAL["Step 4: 图序列时序建模"]
        E1["拼接监测<br/>q_t = [z_t ‖ u_t]"]
        E2["GRU 时序编码<br/>s_t = GRU(q_{t-K+1},..., q_t)"]
        E3["响应预测<br/>r̂_{t+h} = MLP_resp(s_t)"]
        E4["损失函数<br/>L = Σ λ_m · Huber(r̂^{(m)} − r^{(m)})"]
    end

    subgraph INTERPRET["Step 5: 空间解释"]
        F1["表面热点<br/>C_j(t) = Σ_i α_ij / (|N_rm(j)|+ε)"]
        F2["盾体热点图<br/>(shield展开)"]
        F3["沿里程演化图<br/>热点 vs 监测响应"]
    end

    subgraph EVAL["Step 6: 实验验证"]
        G1["Baselines<br/>Persistence / XGBoost / LSTM / TSP-LSTM"]
        G2["Ablations<br/>No E_rm / No TSP / Randomized E_rm / ..."]
        G3["Metrics<br/>MAE / RMSE / R² / Pearson / Spearman"]
        G4["分场景评价<br/>高推力 / 高扭矩 / 低贯入度 / ..."]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3

    B3 --> C1
    B3 --> C2
    C1 --> C3
    C2 --> C4
    C1 --> C5
    C2 --> C5
    C3 --> C6
    C4 --> C6
    C5 --> C6
    C6 --> C7

    C7 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4

    D4 --> E1
    B3 -->|"u_t"| E1
    E1 --> E2 --> E3
    E3 --> E4

    D2 --> F1
    F1 --> F2
    F1 --> F3

    E3 --> G1
    E3 --> G2
    G1 --> G3
    G2 --> G3
    G3 --> G4
```

---

## 三、文件 ↔ 论文章节映射

| 论文章节 | 对应代码文件 | 关键函数 |
|---|---|---|
| 3.1 Geometry-constrained graph construction | `src/graph/sequence.py` | `build_graph_sequence()` |
| 3.1 岩体节点 | `src/graph/nodes.py` | `build_rock_nodes()` |
| 3.1 TBM 节点 | `src/graph/nodes.py` | `build_tbm_nodes()` |
| 3.1 候选边生成 | `src/graph/edges.py` | `compute_rm_candidate_edges()` |
| 3.1 边特征 + 几何先验 | `src/graph/edges.py` | `encode_rm_edge_features()` |
| 3.2 GNN 边注意力 | `src/models/gnn.py` | `RockTBMEncoder.forward()` |
| 3.2 图序列 GRU | `src/models/graph_seq.py` | `GraphSequenceModel.forward()` |
| 3.2 训练目标 | `src/training/trainer.py` | `StandardizedHuberLoss` |
| 3.3 表面热点映射 | `src/visualization/hotspot.py` | `aggregate_attention_to_surface()` |
| 4.3-4.5 Baseline + 消融 | `src/models/baselines.py` | `LSTMBaseline`, `TSPXGBoost` |
| 4.5 分场景评价 | `src/training/metrics.py` | `compute_scenario_metrics()` |
| 4.6 空间解释可视化 | `src/visualization/` | `plot_shield_hotspot()`, `plot_hotspot_vs_response()` |
