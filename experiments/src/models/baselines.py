"""基线模型.

论文定义的 baselines:
  - Persistence: r_hat_{t+h} = r_t
  - XGBoost: 表格特征 (聚合监测 + TSP 统计量)
  - LSTM: 纯监测序列
  - TSP-LSTM: 监测序列 + TSP active-zone 统计量
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


# ── Persistence ───────────────────────────────────────────────────

class Persistence:
    """Naive persistence baseline: r_hat_{t+h} = r_t.

    Uses the last observed monitoring values as the prediction.
    Note: the input X has 6 features (including RPM) but the target
    has only 5 (excluding RPM). The caller must select the correct columns.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X, target_cols=None):
        """Predict using last observed values.

        Args:
          X: (N, K, F) monitoring sequence
          target_cols: optional list of column indices to select from F features.
                       If None, returns all F features (caller must handle mismatch).
        """
        last = X[:, -1, :]
        if target_cols is not None:
            return last[:, target_cols]
        return last


# ── XGBoost with TSP descriptors ──────────────────────────────────

class TSPXGBoost:
    """XGBoost baseline with aggregated TSP statistics.

    将序列展平 + TSP 统计量拼接为表格特征.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6, **kwargs):
        self.model = MultiOutputRegressor(
            XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         verbosity=0, **kwargs)
        )

    def fit(self, X_seq: np.ndarray, tsp_stats: np.ndarray, y: np.ndarray):
        """X_seq: (N, K, F), tsp_stats: (N, TSP_features), y: (N, T)"""
        # Flatten sequence
        feat = X_seq.reshape(X_seq.shape[0], -1)
        if tsp_stats is not None:
            feat = np.concatenate([feat, tsp_stats], axis=1)
        self.model.fit(feat, y)
        return self

    def predict(self, X_seq: np.ndarray, tsp_stats: Optional[np.ndarray] = None
                ) -> np.ndarray:
        feat = X_seq.reshape(X_seq.shape[0], -1)
        if tsp_stats is not None:
            feat = np.concatenate([feat, tsp_stats], axis=1)
        return self.model.predict(feat)


# ── LSTM ──────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """纯时间序列 LSTM baseline.

    输入: 历史监测序列 u_{t-K+1:t}
    输出: 未来响应 r_{t+h}
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, F)"""
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # 取最后一层隐状态
        return self.fc(h)


# ── TSP-LSTM ──────────────────────────────────────────────────────

class TSPLSTM(nn.Module):
    """LSTM + TSP active-zone 统计量.

    输入: 监测序列 + 每个 step 的 TSP 统计特征
    """

    def __init__(self, input_dim: int = 6, tsp_stat_dim: int = 8,
                 hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.tsp_fc = nn.Linear(tsp_stat_dim, tsp_stat_dim * 2)
        self.fc = nn.Linear(hidden_dim + tsp_stat_dim * 2, output_dim)

    def forward(self, x_seq: torch.Tensor, tsp_stats: torch.Tensor
                ) -> torch.Tensor:
        """x_seq: (B, K, F), tsp_stats: (B, S)"""
        _, (h_n, _) = self.lstm(x_seq)
        h = h_n[-1]
        tsp_h = torch.relu(self.tsp_fc(tsp_stats))
        combined = torch.cat([h, tsp_h], dim=1)
        return self.fc(combined)


# ── SpatialAggLSTM ────────────────────────────────────────────────

class SpatialAggLSTM(nn.Module):
    """简单空间聚合基线: 距离加权岩体属性 + LSTM.

    对每个掘进步, 将 active zone 内的岩体属性按距 TBM 表面的距离
    加权聚合为固定维度向量, 拼接到 LSTM 输入.

    这是验证"图结构 vs 简单空间聚合"的关键对照:
      如果 Full Model 优于 SpatialAggLSTM → 图结构贡献成立
      如果 Full Model ≈ SpatialAggLSTM → 图结构可能是过度工程化
    """

    def __init__(self, input_dim: int = 6, spatial_dim: int = 16,
                 hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.lstm = nn.LSTM(input_dim + spatial_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor, spatial_seq: torch.Tensor
                ) -> torch.Tensor:
        """x_seq: (B, K, F), spatial_seq: (B, K, spatial_dim)"""
        combined = torch.cat([x_seq, spatial_seq], dim=-1)
        _, (h_n, _) = self.lstm(combined)
        h = h_n[-1]
        return self.fc(h)


def compute_spatial_agg_features(graph_seqs, n_components: int = 4,
                                 n_attrs: int = None) -> np.ndarray:
    """从图序列中提取距离加权空间聚合特征.

    对每个样本的每个时间步:
      1. 获取 active zone 岩体节点和 TBM 表面节点
      2. 按 TBM 部件分组, 对每个部件计算距离加权岩体属性聚合
      3. 拼接为固定维度向量

    返回: (N, K, spatial_dim) numpy 数组
    """
    # 统一 n_attrs: 从第一个样本的第一个快照推断
    if n_attrs is None and len(graph_seqs) > 0 and len(graph_seqs[0]) > 0:
        n_attrs = graph_seqs[0][0].rock_attrs.shape[1]

    samples = []
    for snap_seq in graph_seqs:
        step_features = []
        for snap in snap_seq:
            feat = _compute_single_spatial_agg(snap, n_components, n_attrs)
            step_features.append(feat)
        samples.append(np.stack(step_features))
    return np.stack(samples)


def _compute_single_spatial_agg(snap: 'GraphSnapshot', n_components: int = 4,
                                n_attrs: int = None) -> np.ndarray:
    """计算单个图快照的距离加权空间聚合特征.

    对每个 TBM 部件, 计算:
      - 距离加权岩体属性均值 (D_attr 维)
      - 入射岩体节点数 (1 维)
    总维度: n_components * (D_attr + 1)
    """
    data = snap.hetero_data
    rock_attrs = snap.rock_attrs.cpu().numpy()  # (N_rock, D)
    tbm_comp = snap.tbm_components.cpu().numpy()  # (M, 4) onehot

    if n_attrs is None:
        n_attrs = rock_attrs.shape[1] if rock_attrs.ndim > 1 else 1

    feat_dim = n_components * (n_attrs + 1)

    # 如果没有岩-机边, 返回零向量
    if ("rock", "interact", "tbm") not in data.edge_types:
        return np.zeros(feat_dim, dtype=np.float32)

    edge_idx = data["rock", "interact", "tbm"].edge_index.cpu().numpy()
    edge_store = data["rock", "interact", "tbm"]

    if "edge_attrs" not in edge_store or edge_idx.shape[1] == 0:
        return np.zeros(feat_dim, dtype=np.float32)

    distances = edge_store["edge_attrs"]["distance"]
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()
    else:
        distances = np.asarray(distances, dtype=np.float32)

    src, dst = edge_idx[0], edge_idx[1]
    comp_labels = tbm_comp.argmax(axis=1)  # (M,)

    # 距离权重: w = exp(-d / tau), tau 取中位数距离
    tau = max(np.median(distances), 1e-4)
    weights = np.exp(-distances / tau)

    # 按部件聚合
    comp_features = []
    for c in range(n_components):
        mask = comp_labels[dst] == c
        if mask.any():
            w_c = weights[mask]
            attrs_c = rock_attrs[src[mask]]
            if attrs_c.ndim == 1:
                attrs_c = attrs_c.reshape(-1, 1)
            # 距离加权均值
            w_sum = w_c.sum() + 1e-8
            weighted_mean = (w_c[:, None] * attrs_c).sum(axis=0) / w_sum
            n_edges = float(mask.sum())
            comp_features.append(np.concatenate([weighted_mean, [n_edges]]))
        else:
            comp_features.append(np.zeros(n_attrs + 1, dtype=np.float32))

    return np.concatenate(comp_features)
