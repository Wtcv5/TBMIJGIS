"""图序列模型 — 实现论文 Algorithm 2.

Response-supervised rock-TBM graph sequence learning.
Full Model = GNN + GRU: G_{t-K+1:t}, u_{t-K+1:t} → r_{t+h}
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.graph.sequence import GraphSnapshot
from src.models.gnn import RockTBMEncoder


class GraphSequenceModel(nn.Module):
    """Full Model: GNN 编码图快照序列 + GRU 建模时序演化.

    论文 Section 7.6:
      z_tau = GNN(G_tau)                          # 图编码
      q_tau = [z_tau || u_tau]                    # 拼接监测
      s_t = GRU(q_{t-K+1}, ..., q_t)              # 时序建模
      r_hat_{t+h} = MLP_resp(s_t)                 # 响应预测
    """

    def __init__(self, rock_in_dim: int, tbm_in_dim: int, edge_in_dim: int,
                 monitoring_dim: int = 6, output_dim: int = 5,
                 rock_hidden: int = 128, tbm_hidden: int = 64,
                 edge_hidden: int = 64, gnn_layers: int = 3,
                 gru_hidden: int = 256, gru_layers: int = 2,
                 dropout: float = 0.1, residual_prediction: bool = False):
        super().__init__()
        self.residual_prediction = residual_prediction

        self.gnn = RockTBMEncoder(rock_in_dim, tbm_in_dim, edge_in_dim,
                                  rock_hidden, tbm_hidden, edge_hidden,
                                  gnn_layers, dropout)

        # z_t 维度: tbm_hidden * 2 (mean + max pooling) + rock_hidden (rock mean)
        graph_out_dim = tbm_hidden * 2 + rock_hidden
        self.graph_out_dim = graph_out_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # GRU 时序编码
        self.gru = nn.GRU(graph_out_dim + monitoring_dim, gru_hidden,
                          gru_layers, batch_first=True, dropout=dropout)

        # 响应预测头
        self.resp_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim),
        )
        if self.residual_prediction:
            nn.init.zeros_(self.resp_head[-1].weight)
            nn.init.zeros_(self.resp_head[-1].bias)

    def _encode_single_graph(self, snap: GraphSnapshot, tau: float
                             ) -> torch.Tensor:
        """编码单个图快照, 返回 z_t."""
        z_t, _ = self.gnn(snap.hetero_data, snap.rock_attrs,
                          snap.tbm_components, tau)
        return z_t

    def forward(self, graph_seqs: List[List[GraphSnapshot]],
                monitoring_seqs: torch.Tensor, tau: float = 2.0,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """前向传播.

        Args:
          graph_seqs: list of K-length lists of GraphSnapshots,
                      or None for no-graph baselines
          monitoring_seqs: (B, K, F) 监测序列
          tau: 距离阈值

        Returns:
          predictions: (B, output_dim)
          attentions: list of per-step alpha_ij (for hotspot generation)
        """
        B = monitoring_seqs.shape[0]
        K = monitoring_seqs.shape[1]
        device = monitoring_seqs.device

        # 编码每个图快照
        z_embeddings = []
        for snap_seq in graph_seqs:
            z_seq = []
            for snap in snap_seq:
                z_t = self._encode_single_graph(snap, tau)
                z_seq.append(z_t)
            z_embeddings.append(torch.stack(z_seq))  # (K, D_graph)

        if z_embeddings:
            z_tensor = torch.stack(z_embeddings)  # (B, K, D_graph)
        else:
            z_tensor = torch.zeros(B, K, self.graph_out_dim, device=device)

        # 拼接: q_tau = [z_tau || u_tau]
        q_seq = torch.cat([z_tensor, monitoring_seqs], dim=-1)  # (B, K, D_graph+F)

        # GRU 时序建模
        gru_out, _ = self.gru(q_seq)
        s_t = gru_out[:, -1, :]  # 取最后一步输出

        # 响应预测
        pred = self.resp_head(s_t)
        if self.residual_prediction:
            # Persistence is a strong baseline in TBM monitoring series. In
            # residual mode, the network learns a graph-conditioned correction
            # to the last observed response instead of relearning persistence.
            pred = monitoring_seqs[:, -1, [0, 2, 3, 4, 5]] + pred

        # 收集最后一步的注意力 (用于热点图)
        # 使用 softmax 前的原始分数 s_ij, 而非归一化后的 alpha_ij
        # 原因: alpha_ij 经 per-TBM-node softmax 归一化后 Σα=1,
        #        导致 C_sum 无区分度; s_ij 保留绝对量纲, 可区分交互强度
        attentions = None
        if return_attention:
            last_snaps = [seq[-1] for seq in graph_seqs]
            attentions = []
            for snap in last_snaps:
                _, att_tuple = self.gnn(snap.hetero_data, snap.rock_attrs,
                                       snap.tbm_components, tau, return_attention=True)
                # att_tuple = (s_ij, alpha_ij); 取 s_ij 用于可视化
                if att_tuple is not None:
                    attentions.append(att_tuple[0])  # s_ij: raw attention scores
                else:
                    attentions.append(None)

        return pred, attentions


class DynamicGraphOnly(GraphSequenceModel):
    """消融变体: 只用图序列, 不加监测序列.

    对应 ablation "No monitoring input".
    Uses a separate GRU with input_size = graph_out_dim (no monitoring concat).
    """

    def __init__(self, rock_in_dim: int, tbm_in_dim: int, edge_in_dim: int,
                 monitoring_dim: int = 6, output_dim: int = 5,
                 rock_hidden: int = 128, tbm_hidden: int = 64,
                 edge_hidden: int = 64, gnn_layers: int = 3,
                 gru_hidden: int = 256, gru_layers: int = 2,
                 dropout: float = 0.1, residual_prediction: bool = False):
        # Pass all args to parent to initialize GNN encoder etc.
        super().__init__(
            rock_in_dim, tbm_in_dim, edge_in_dim,
            monitoring_dim, output_dim, rock_hidden, tbm_hidden, edge_hidden,
            gnn_layers, gru_hidden, gru_layers, dropout, residual_prediction=False,
        )
        # Override GRU to accept only graph embeddings (no monitoring)
        self.gru = nn.GRU(tbm_hidden * 2 + rock_hidden, gru_hidden,
                          gru_layers, batch_first=True, dropout=dropout)

    def forward(self, graph_seqs: List[List[GraphSnapshot]],
                monitoring_seqs: torch.Tensor, tau: float = 2.0,
                return_attention: bool = False):
        B = monitoring_seqs.shape[0]
        device = monitoring_seqs.device

        z_embeddings = []
        for snap_seq in graph_seqs:
            z_seq = [self._encode_single_graph(snap, tau) for snap in snap_seq]
            z_embeddings.append(torch.stack(z_seq))

        if z_embeddings:
            z_tensor = torch.stack(z_embeddings)
        else:
            z_tensor = torch.zeros(B, monitoring_seqs.shape[1],
                                   self.graph_out_dim, device=device)

        gru_out, _ = self.gru(z_tensor)
        s_t = gru_out[:, -1, :]
        return self.resp_head(s_t), None


class NoGeometryPrior(GraphSequenceModel):
    """消融变体: 移除几何先验 β·π_ij.

    通过设置 GNN 编码器的 _disable_geometry_prior 标志,
    在前向传播中永久将 β 贡献置零, 而非临时修改参数.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn._disable_geometry_prior = True


class RandomEdgesGraphSeq(GraphSequenceModel):
    """消融变体: 随机化岩-机边.

    保留边数量但随机分配 rock-tbm 连接, 破坏几何结构.
    随机化后重新计算边特征, 确保边特征与端点一致.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._randomize_edges = True

    def _encode_single_graph(self, snap: GraphSnapshot, tau: float
                             ) -> torch.Tensor:
        """编码图快照, 在编码前随机化岩-机边并重新计算边特征."""
        data = snap.hetero_data.clone()
        if self._randomize_edges and ("rock", "interact", "tbm") in data.edge_types:
            edge_store = data["rock", "interact", "tbm"]
            n_edges = edge_store.edge_index.shape[1]
            if n_edges > 0:
                n_rock = data["rock"].x.shape[0]
                n_tbm = data["tbm"].x.shape[0]
                rand_src = torch.randint(0, n_rock, (n_edges,), device=edge_store.edge_index.device)
                rand_dst = torch.randint(0, n_tbm, (n_edges,), device=edge_store.edge_index.device)
                edge_store.edge_index = torch.stack([rand_src, rand_dst])

                # 重新计算边特征, 使其与随机端点一致
                rock_pos = data["rock"].x[:, :3]  # 坐标
                tbm_pos = data["tbm"].x[:, :3]
                tbm_norm = data["tbm"].x[:, 3:6]
                rock_g = snap.rock_attrs  # 地质属性

                delta = rock_pos[rand_src] - tbm_pos[rand_dst]
                dist = delta.norm(dim=1)
                # 法向一致性 κ = max(0, n_j^T delta / d)
                cos_sim = (tbm_norm[rand_dst] * delta).sum(dim=1)
                kappa = torch.clamp(cos_sim / (dist + 1e-8), min=0.0)
                geo_prior = torch.exp(-dist / tau) * kappa

                edge_store["edge_attrs"] = {
                    "distance": dist.cpu().numpy(),
                    "kappa": kappa.cpu().numpy(),
                    "delta_vec": delta.cpu().numpy(),
                    "geometry_prior": geo_prior.cpu().numpy(),
                }

        z_t, _ = self.gnn(data, snap.rock_attrs,
                          snap.tbm_components, tau)
        return z_t


class NoGeometricConstraints(GraphSequenceModel):
    """消融变体: 移除几何约束 (距离 + 法向一致性).

    此变体本身不需要特殊模型逻辑 — 它与 Full Model 结构相同.
    区别在于图构建阶段: 使用宽松参数 (大 tau, eta_min=0) 构建图序列,
    然后用该宽松图序列训练此模型.

    The calling pipeline needs to build relaxed_snapshots and pass the
    corresponding dataloader.
    """
    pass
