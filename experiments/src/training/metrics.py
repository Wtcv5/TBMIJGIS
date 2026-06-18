"""评价指标.

论文定义的指标:
  MAE, RMSE, R², Pearson correlation, Spearman correlation
  + Bootstrap confidence intervals for small-sample reliability
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray
                        ) -> Dict[str, float]:
    """计算所有评价指标.

    Args:
      y_true: (N, T) 真实值
      y_pred: (N, T) 预测值

    Returns:
      字典: {mae, rmse, r2, pearson, spearman}
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 全局 pearson/spearman
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()
    if len(yt_flat) >= 3:
        pearson, _ = pearsonr(yt_flat, yp_flat)
        spearman, _ = spearmanr(yt_flat, yp_flat)
    else:
        pearson, spearman = 0.0, 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


def compute_per_variable_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 var_names: list = None) -> Dict[str, Dict[str, float]]:
    """逐变量计算指标."""
    if var_names is None:
        var_names = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]

    results = {}
    for i, name in enumerate(var_names):
        if i < y_true.shape[1]:
            results[name] = compute_all_metrics(y_true[:, i], y_pred[:, i])
    return results


def identify_geological_scenarios(geo_chainages: np.ndarray,
                                  geo_values: np.ndarray,
                                  test_chainages: np.ndarray,
                                  low_velocity_percentile: float = 25.0,
                                  transition_half_width: float = 5.0
                                  ) -> Dict[str, np.ndarray]:
    """基于 TSP 速度场识别地质场景, 为每个测试样本分配场景标签.

    场景定义:
      - "Low-velocity zone": TSP 速度低于 P25 百分位的区域 (潜在弱岩/断层)
      - "Transition zone": 低速区边界 ±transition_half_width 范围 (岩性过渡)
      - "Normal zone": 其余区域 (正常岩体)

    这是 GIS 导向的场景验证核心: 不只看全局指标, 而是关注模型在
    不同地质条件下的空间适应性.

    Args:
      geo_chainages: TSP 速度场的里程坐标
      geo_values: TSP 速度值
      test_chainages: 测试样本的里程坐标
      low_velocity_percentile: 低速异常阈值百分位
      transition_half_width: 过渡带半宽 (m)

    Returns:
      {scenario_name: boolean mask over test samples}
    """
    if len(geo_values) == 0 or len(test_chainages) == 0:
        return {"Normal zone": np.ones(len(test_chainages), dtype=bool)}

    threshold = np.percentile(geo_values, low_velocity_percentile)

    # 对每个测试里程, 插值获取 TSP 速度
    test_vp = np.interp(test_chainages, geo_chainages, geo_values)

    # 识别低速区里程段
    low_vp_mask = geo_values < threshold
    low_vp_chainages = geo_chainages[low_vp_mask]

    # 构建场景掩码
    low_zone = np.zeros(len(test_chainages), dtype=bool)
    transition_zone = np.zeros(len(test_chainages), dtype=bool)

    if len(low_vp_chainages) > 0:
        # 使用连续低速段识别 (合并间距 < 2*transition_half_width 的相邻低速段)
        low_vp_sorted = np.sort(low_vp_chainages)
        segments = []
        seg_start = low_vp_sorted[0]
        seg_end = low_vp_sorted[0]
        for ch in low_vp_sorted[1:]:
            if ch - seg_end <= 2 * transition_half_width:
                seg_end = ch
            else:
                segments.append((seg_start, seg_end))
                seg_start = ch
                seg_end = ch
        segments.append((seg_start, seg_end))

        # 基于连续段分配场景
        for i, ch in enumerate(test_chainages):
            in_low = False
            in_transition = False
            for seg_start, seg_end in segments:
                if seg_start <= ch <= seg_end:
                    in_low = True
                    break
                elif (seg_start - transition_half_width <= ch < seg_start
                      or seg_end < ch <= seg_end + transition_half_width):
                    in_transition = True
            if in_low:
                low_zone[i] = True
            elif in_transition:
                transition_zone[i] = True

    normal_zone = ~low_zone & ~transition_zone

    scenarios = {}
    if low_zone.any():
        scenarios["Low-velocity zone"] = low_zone
    if transition_zone.any():
        scenarios["Transition zone"] = transition_zone
    if normal_zone.any():
        scenarios["Normal zone"] = normal_zone

    return scenarios


def compute_attention_geology_correlation(C_mean: np.ndarray,
                                          test_chainages: np.ndarray,
                                          geo_chainages: np.ndarray,
                                          geo_values: np.ndarray
                                          ) -> Dict[str, float]:
    """定量分析注意力与地质属性的相关性.

    验证注意力的物理有效性: 如果模型学到的注意力与地质异常相关,
    则说明注意力不仅是统计拟合, 而是具有地质工程含义.

    Returns:
      {pearson_r, pearson_p, spearman_r, spearman_p, n_samples}
    """
    # 插值获取测试里程处的 TSP 速度
    test_vp = np.interp(test_chainages, geo_chainages, geo_values)

    # 注意力与速度的相关性 (负相关 = 低速区高注意力 = 物理合理)
    if len(C_mean) < 3:
        return {"pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_r": 0.0, "spearman_p": 1.0,
                "n_samples": len(C_mean)}

    pr, pp = pearsonr(C_mean, test_vp)
    sr, sp = spearmanr(C_mean, test_vp)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n_samples": len(C_mean),
    }


def compute_scenario_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             monitoring: np.ndarray,
                             scenario_masks: Dict[str, np.ndarray]
                             ) -> Dict[str, Dict[str, float]]:
    """分场景评价.

    For small scenario subsets, R² is unreliable (can be extremely negative
    when the subset variance is small).  We report MAE and RMSE as primary
    metrics and include R² only when the subset has >= 5 samples.
    """
    results = {}
    for name, mask in scenario_masks.items():
        n = mask.sum()
        if n < 2:
            continue
        m = compute_all_metrics(y_true[mask], y_pred[mask])
        # Flag unreliable R² for very small subsets
        if n < 5:
            m["r2_reliable"] = False
        else:
            m["r2_reliable"] = True
        m["n_samples"] = int(n)
        results[name] = m
    return results


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      n_boot: int = 1000, ci: float = 0.95,
                      seed: int = 42
                      ) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals for all metrics.

    Returns:
      {metric_name: {point, ci_low, ci_high}} for MAE, RMSE, R².
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    point = compute_all_metrics(y_true, y_pred)

    boot_mae, boot_rmse, boot_r2 = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        boot_mae.append(mean_absolute_error(yt, yp))
        boot_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        boot_r2.append(r2_score(yt, yp))

    alpha = (1 - ci) / 2
    results = {}
    for name, boot_vals in [("mae", boot_mae), ("rmse", boot_rmse), ("r2", boot_r2)]:
        vals = np.array(boot_vals)
        results[name] = {
            "point": float(point[name]),
            "ci_low": float(np.percentile(vals, 100 * alpha)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha))),
        }
    return results


def paired_permutation_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                            y_pred_b: np.ndarray, metric: str = "mae",
                            n_perm: int = 10000, seed: int = 42
                            ) -> Tuple[float, float]:
    """Paired permutation test for comparing two models.

    Tests H0: metric(model_a) == metric(model_b).
    Returns (delta, p_value) where delta = metric_b - metric_a.
    """
    rng = np.random.RandomState(seed)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_a = y_pred_a.reshape(-1, 1)
        y_pred_b = y_pred_b.reshape(-1, 1)

    def _metric(yt, yp):
        if metric == "mae":
            return mean_absolute_error(yt, yp)
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(yt, yp))
        elif metric == "r2":
            return r2_score(yt, yp)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    obs_a = _metric(y_true, y_pred_a)
    obs_b = _metric(y_true, y_pred_b)
    delta_obs = obs_b - obs_a

    n = len(y_true)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pa = np.where(swap[:, None] if y_true.ndim > 1 else swap, y_pred_b, y_pred_a)
        pb = np.where(swap[:, None] if y_true.ndim > 1 else swap, y_pred_a, y_pred_b)
        d = _metric(y_true, pb) - _metric(y_true, pa)
        if abs(d) >= abs(delta_obs):
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return float(delta_obs), float(p_value)


def compute_morans_i(values: np.ndarray, positions: np.ndarray,
                     k_neighbors: int = 8) -> float:
    """Compute Moran's I for spatial autocorrelation of attention on TBM surface.

    Uses k-nearest-neighbour spatial weights.

    Args:
      values: (N,) attention values at each TBM surface node.
      positions: (N, 3) 3D positions of TBM surface nodes.
      k_neighbors: number of nearest neighbours for spatial weights.

    Returns:
      Moran's I statistic.
    """
    from scipy.spatial import cKDTree

    n = len(values)
    if n < 3:
        return 0.0

    # Build k-NN spatial weights
    tree = cKDTree(positions)
    _, indices = tree.query(positions, k=min(k_neighbors + 1, n))

    # Binary spatial weight matrix (exclude self)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            W[i, j] = 1.0

    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    # Moran's I
    z = values - values.mean()
    s0 = W.sum()
    if s0 == 0:
        return 0.0

    numerator = 0.0
    for i in range(n):
        for j_idx in range(n):
            if W[i, j_idx] > 0:
                numerator += W[i, j_idx] * z[i] * z[j_idx]

    denominator = (z ** 2).sum()
    if denominator == 0:
        return 0.0

    I = (n / s0) * (numerator / denominator)
    return float(I)


def compute_component_cv(C_j: np.ndarray, tbm_components: np.ndarray) -> float:
    """Compute coefficient of variation of component-mean attention.

    Measures how much attention differentiates across TBM components.
    Higher CV = more spatially structured attention.

    Args:
      C_j: (N,) attention values at each TBM surface node.
      tbm_components: (N,) integer component labels (0=cutterhead,
        1=front shield, 2=middle shield, 3=tail shield).

    Returns:
      CV of component-mean attention values.
    """
    unique_components = np.unique(tbm_components)
    if len(unique_components) < 2:
        return 0.0

    comp_means = []
    for c in unique_components:
        mask = tbm_components == c
        if mask.sum() > 0:
            comp_means.append(C_j[mask].mean())

    comp_means = np.array(comp_means)
    if comp_means.mean() == 0:
        return 0.0

    return float(comp_means.std() / comp_means.mean())


def compute_spatial_consistency_metrics(
    C_j: np.ndarray,
    tbm_positions: np.ndarray,
    tbm_components: np.ndarray,
    test_chainages: np.ndarray,
    geo_chainages: np.ndarray,
    geo_values: np.ndarray,
) -> Dict[str, float]:
    """Compute all spatial consistency indicators for one model variant.

    Returns:
      {attention_geology_pearson_r, attention_geology_pearson_p,
       morans_i_mean, component_cv}
    """
    # Attention-geology correlation
    C_mean_per_chainage = []
    for ch in test_chainages:
        # Average C_j across all TBM nodes at this chainage
        C_mean_per_chainage.append(C_j.mean())

    C_mean_arr = np.array(C_mean_per_chainage)
    att_geo = compute_attention_geology_correlation(
        C_mean_arr, test_chainages, geo_chainages, geo_values
    )

    # Moran's I (mean across chainages)
    # For each chainage, compute Moran's I on the TBM surface
    moran_values = []
    # Group by chainage (approximate by x-coordinate)
    x_coords = tbm_positions[:, 0]
    for ch in np.unique(np.round(x_coords, 1)):
        mask = np.abs(x_coords - ch) < 0.3
        if mask.sum() < 5:
            continue
        mi = compute_morans_i(C_j[mask], tbm_positions[mask])
        moran_values.append(mi)

    morans_i_mean = float(np.mean(moran_values)) if moran_values else 0.0

    # Component CV
    cv = compute_component_cv(C_j, tbm_components)

    return {
        "attention_geology_pearson_r": att_geo["pearson_r"],
        "attention_geology_pearson_p": att_geo["pearson_p"],
        "morans_i_mean": morans_i_mean,
        "component_cv": cv,
    }
