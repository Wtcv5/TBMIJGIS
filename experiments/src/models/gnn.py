"""GNN 图编码器 — 岩-机边注意力 + 图快照编码.

论文 Section 7 (Algorithm 2 step 4):
  1. Encode rock/TBM nodes via type-specific MLPs
  2. Encode rock-machine edge attributes
  3. Compute geometry prior π_ij
  4. Compute attention α_ij over geometry-screened edges
  5. Aggregate rock messages to TBM surface nodes
  6. Readout to graph representation z_t
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear as PyGLinear

from src.graph.edges import encode_rm_edge_features


class RockTBMEncoder(nn.Module):
    """异构图编码器: 岩体节点 + TBM 表面节点 + 岩-机边注意力.

    对应论文 7.4-7.5 节.
    """

    def __init__(self, rock_in_dim: int, tbm_in_dim: int, edge_in_dim: int,
                 rock_hidden: int = 128, tbm_hidden: int = 64,
                 edge_hidden: int = 64, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers

        # 节点 MLP 编码器 (论文 7.4: h_i^{r,0}, h_j^{m,0})
        self.rock_encoder = nn.Sequential(
            nn.Linear(rock_in_dim, rock_hidden),
            nn.ReLU(),
            nn.Linear(rock_hidden, rock_hidden),
        )
        self.tbm_encoder = nn.Sequential(
            nn.Linear(tbm_in_dim, tbm_hidden),
            nn.ReLU(),
            nn.Linear(tbm_hidden, tbm_hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
        )

        # 边注意力 MLP (论文 7.4: MLP_att)
        self.attention_mlp = nn.Sequential(
            nn.Linear(rock_hidden + tbm_hidden + edge_hidden + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 消息 MLP (论文 7.5: MLP_msg)
        self.message_mlp = nn.Sequential(
            nn.Linear(rock_hidden + edge_hidden, tbm_hidden),
            nn.ReLU(),
            nn.Linear(tbm_hidden, tbm_hidden),
        )

        # 节点更新 (论文 7.5: GRU_node)
        # 仅 TBM 节点需要更新: 岩-机交互是单向的 (rock→tbm),
        # 岩体节点作为地质先验不接收 TBM 反馈消息
        self.tbm_update = nn.GRUCell(tbm_hidden, tbm_hidden)

        # Heterogeneous conv layers for E_rr and E_mm propagation
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("rock", "adj", "rock"): SAGEConv(rock_hidden, rock_hidden),
                ("tbm", "adj", "tbm"): SAGEConv(tbm_hidden, tbm_hidden),
            }, aggr="mean")
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        # 几何先验偏置 β (可学习参数, 论文 7.4: β·π_ij)
        # 初始化为 1.0, 因为 π_ij ∈ [0,1], β·π 需要与 MLP 输出同量级
        self.beta = nn.Parameter(torch.tensor(1.0))
        self._disable_geometry_prior = False

    def forward(self, data: HeteroData, rock_attrs: torch.Tensor,
                tbm_components: torch.Tensor, tau: float = 2.0,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """编码单个图快照.

        返回: (z_t, attention_tuple) — 图级表示和岩-机边注意力
          attention_tuple: (s_ij, alpha_ij) 当 return_attention=True
            - s_ij: softmax 前的原始注意力分数 (用于热点图可视化, 有绝对量纲区分度)
            - alpha_ij: softmax 后的归一化注意力 (用于消息聚合, per-TBM-node 归一化)
          否则 None
        """
        # Step 4.1: 节点初始编码
        h_rock = self.rock_encoder(data["rock"].x)
        h_tbm = self.tbm_encoder(data["tbm"].x)

        # Step 4.1b: 同质边传播 (E_rr, E_mm) — 先融合邻域信息
        # 岩体节点在参与岩-机交互前应已融合邻域地质信息
        for conv in self.convs:
            h_dict = conv({"rock": h_rock, "tbm": h_tbm},
                          {("rock", "adj", "rock"): data["rock", "adj", "rock"].edge_index,
                           ("tbm", "adj", "tbm"): data["tbm", "adj", "tbm"].edge_index})
            h_rock = self.dropout(torch.relu(h_dict["rock"]))
            h_tbm = self.dropout(torch.relu(h_dict["tbm"]))

        # Step 4.2-4.4: 岩-机边注意力 (在邻域传播之后)
        s_ij = None
        alpha_ij = None
        if ("rock", "interact", "tbm") in data.edge_types:
            edge_idx = data["rock", "interact", "tbm"].edge_index
            edge_store = data["rock", "interact", "tbm"]
            if "edge_attrs" in edge_store:
                attrs = edge_store["edge_attrs"]
                device = h_rock.device

                # 提取边特征
                distances = torch.tensor(attrs["distance"], dtype=torch.float32, device=device)
                kappas = torch.tensor(attrs["kappa"], dtype=torch.float32, device=device)
                delta_vecs = torch.tensor(attrs["delta_vec"], dtype=torch.float32, device=device)
                geo_prior = torch.tensor(attrs["geometry_prior"], dtype=torch.float32, device=device)

                # a_ij^{rm} 特征 (论文 7.2)
                edge_feat = encode_rm_edge_features(
                    distances, kappas, delta_vecs, tau,
                    rock_attrs, tbm_components, edge_idx
                )
                edge_encoded = self.edge_encoder(edge_feat)

                # 注意力分数 (论文 7.4)
                src, dst = edge_idx[0], edge_idx[1]
                cat_input = torch.cat([
                    h_rock[src], h_tbm[dst], edge_encoded,
                    geo_prior.unsqueeze(1)
                ], dim=1)
                s_ij = self.attention_mlp(cat_input).squeeze(-1)
                # 几何先验偏置: β·π_ij (加法形式, 有界稳定)
                # 替代原来的 β·log(π_ij) — log 形式在 π→0 时梯度爆炸
                beta_val = 0.0 if self._disable_geometry_prior else self.beta
                s_ij = s_ij + beta_val * geo_prior

                # Softmax 归一化 (per TBM node) — 用于消息聚合
                alpha_ij = _scatter_softmax(s_ij, dst)

                # Step 4.5: 消息聚合 (论文 7.5)
                msg_input = torch.cat([h_rock[src], edge_encoded], dim=1)
                messages = self.message_mlp(msg_input)
                weighted_msgs = messages * alpha_ij.unsqueeze(1)

                # Scatter sum to TBM nodes
                tbm_msg = torch.zeros_like(h_tbm)
                tbm_msg = tbm_msg.index_add(0, dst, weighted_msgs)

                # TBM 节点更新
                h_tbm = self.tbm_update(tbm_msg, h_tbm)

        # Step 4.6: 图读出 (论文 7.5: mean + max pooling)
        # 同时从 TBM 节点和岩体节点读出，保留岩体全局地质信息
        z_t = torch.cat([
            h_tbm.mean(dim=0),
            h_tbm.max(dim=0).values,
            h_rock.mean(dim=0),
        ], dim=0)

        if return_attention:
            # 返回 (s_ij, alpha_ij): s_ij 用于热点图可视化 (有绝对量纲),
            # alpha_ij 用于消息聚合 (per-TBM-node 归一化)
            return z_t, (s_ij, alpha_ij)
        return z_t, None


def _scatter_softmax(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Per-index softmax, numerically stable.

    Handles nodes with no incoming edges by returning 0 attention.
    """
    max_scores = torch.zeros(indices.max().item() + 1, device=scores.device)
    max_scores = max_scores.index_reduce(0, indices, scores, "amax",
                                          include_self=False)
    shifted = scores - max_scores[indices]
    exp_s = torch.exp(shifted)
    exp_sum = torch.zeros(indices.max().item() + 1, device=scores.device)
    exp_sum = exp_sum.index_add(0, indices, exp_s)
    # 标记有入射边的节点，避免 exp_sum=0 时除以 epsilon 产生极大值
    has_incoming = torch.zeros(indices.max().item() + 1, dtype=torch.bool, device=scores.device)
    has_incoming[indices] = True
    safe_denom = exp_sum[indices]
    safe_denom = torch.where(has_incoming[indices], safe_denom, torch.ones_like(safe_denom))
    result = exp_s / (safe_denom + 1e-8)
    # 无入射边的边不应存在，但以防万一
    return result


# ── 静态图模型 (只用 G_t) ──────────────────────────────────────────

class StaticGraphModel(nn.Module):
    """静态图 baseline: G_t → r_{t+1}.

    用于对比验证单步图编码是否有效.
    """

    def __init__(self, rock_in_dim: int, tbm_in_dim: int, edge_in_dim: int,
                 output_dim: int = 5, rock_hidden: int = 128,
                 tbm_hidden: int = 64, edge_hidden: int = 64,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = RockTBMEncoder(
            rock_in_dim, tbm_in_dim, edge_in_dim,
            rock_hidden, tbm_hidden, edge_hidden,
            num_layers, dropout
        )
        self.predictor = nn.Linear(tbm_hidden * 2 + rock_hidden, output_dim)

    def forward(self, data: HeteroData, rock_attrs: torch.Tensor,
                tbm_components: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
        z_t, _ = self.encoder(data, rock_attrs, tbm_components, tau)
        return self.predictor(z_t)
