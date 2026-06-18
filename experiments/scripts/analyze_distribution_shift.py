"""分析训练/测试集的地质分布偏移.

对比:
  1. 监测变量在训练/验证/测试集上的统计量
  2. TSP 地质属性在训练/验证/测试区间内的分布
  3. Persistence baseline 在各子集上的表现
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def analyze_distribution_shift():
    """分析分布偏移."""
    # 加载数据
    mon_df = pd.read_csv("data/raw/monitoring.csv")
    tsp_df = pd.read_csv("data/raw/tsp.csv")

    n_total = len(mon_df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = mon_df.iloc[:n_train]
    val_df = mon_df.iloc[n_train:n_train + n_val]
    test_df = mon_df.iloc[n_train + n_val:]

    print("=" * 70)
    print("Distribution Shift Analysis")
    print("=" * 70)
    print(f"\nSplit: Train={len(train_df)} (chainage 0-{train_df['chainage'].max():.1f}m), "
          f"Val={len(val_df)} ({val_df['chainage'].min():.1f}-{val_df['chainage'].max():.1f}m), "
          f"Test={len(test_df)} ({test_df['chainage'].min():.1f}-{test_df['chainage'].max():.1f}m)")

    feature_cols = ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]

    print("\n--- Monitoring Variable Statistics ---")
    print(f"{'Variable':<16} {'Train mean±std':<20} {'Test mean±std':<20} {'Shift (std)':<12} {'KS p-value':<12}")
    print("-" * 80)

    shift_results = []
    for col in feature_cols:
        train_mean, train_std = train_df[col].mean(), train_df[col].std()
        test_mean, test_std = test_df[col].mean(), test_df[col].std()
        shift = abs(test_mean - train_mean) / (train_std + 1e-8)
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(train_df[col], test_df[col])
        shift_results.append({"var": col, "shift": shift, "ks_p": ks_p})
        print(f"{col:<16} {train_mean:>7.2f}±{train_std:<9.2f} "
              f"{test_mean:>7.2f}±{test_std:<9.2f} {shift:>10.2f}    {ks_p:.4f}")

    # TSP 地质属性分布
    print("\n--- TSP Geological Attributes by Chainage Zone ---")
    # TSP X coordinate corresponds to chainage
    tsp_features = ["Vp", "Vs", "ro", "E", "Vp_Vs", "Pr"]

    # 计算各区间内的 TSP 体素统计
    train_chainage_max = train_df["chainage"].max()
    val_chainage_max = val_df["chainage"].max()
    test_chainage_max = test_df["chainage"].max()

    # TSP X 范围
    tsp_x_min, tsp_x_max = tsp_df["X"].min(), tsp_df["X"].max()
    print(f"TSP X range: [{tsp_x_min}, {tsp_x_max}]")

    # 按 X 划分 TSP 体素到各区间
    # 假设 X=0 对应 chainage=0, X 正方向为掘进方向
    train_tsp = tsp_df[(tsp_df["X"] >= 0) & (tsp_df["X"] <= train_chainage_max)]
    val_tsp = tsp_df[(tsp_df["X"] > train_chainage_max) & (tsp_df["X"] <= val_chainage_max)]
    test_tsp = tsp_df[(tsp_df["X"] > val_chainage_max) & (tsp_df["X"] <= test_chainage_max)]

    print(f"TSP voxels: Train={len(train_tsp)}, Val={len(val_tsp)}, Test={len(test_tsp)}")

    if len(train_tsp) > 0 and len(test_tsp) > 0:
        print(f"\n{'TSP Var':<10} {'Train mean±std':<20} {'Test mean±std':<20} {'Shift (std)':<12}")
        print("-" * 65)
        for col in tsp_features:
            if col in tsp_df.columns:
                t_mean, t_std = train_tsp[col].mean(), train_tsp[col].std()
                te_mean, te_std = test_tsp[col].mean(), test_tsp[col].std()
                shift = abs(te_mean - t_mean) / (t_std + 1e-8)
                print(f"{col:<10} {t_mean:>7.1f}±{t_std:<9.1f} "
                      f"{te_mean:>7.1f}±{te_std:<9.1f} {shift:>10.2f}")

    # Persistence baseline 性能
    print("\n--- Persistence Baseline Performance ---")
    for name, df_subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df_subset) > 1:
            # Persistence: predict next = current
            actual = df_subset[feature_cols].iloc[1:].values
            pred = df_subset[feature_cols].iloc[:-1].values
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            # R2
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(train_df[col], bins=15, alpha=0.6, label="Train", density=True, color="blue")
        ax.hist(test_df[col], bins=8, alpha=0.6, label="Test", density=True, color="red")
        ax.set_title(f"{col}\n(shift={shift_results[i]['shift']:.2f}, p={shift_results[i]['ks_p']:.3f})")
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/distribution_shift.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: outputs/distribution_shift.png")

    return shift_results


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    analyze_distribution_shift()
