"""深入分析分布偏移源 — 修正 TSP 坐标映射."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def analyze_shift_corrected():
    """修正后的分布偏移分析."""
    mon_df = pd.read_csv("data/raw/monitoring.csv")
    tsp_df = pd.read_csv("data/raw/tsp.csv")

    # TSP 坐标归一化 (与 build_voxel_field 一致)
    tsp_df["X_local"] = tsp_df["X"] - tsp_df["X"].min()  # X_local in [0, 48]

    n_total = len(mon_df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = mon_df.iloc[:n_train]
    val_df = mon_df.iloc[n_train:n_train + n_val]
    test_df = mon_df.iloc[n_train + n_val:]

    print("=" * 70)
    print("Corrected Distribution Shift Analysis")
    print("=" * 70)
    print(f"\nMonitoring: Train={len(train_df)} (0-{train_df['chainage'].max():.0f}m), "
          f"Val={len(val_df)} ({val_df['chainage'].min():.0f}-{val_df['chainage'].max():.0f}m), "
          f"Test={len(test_df)} ({test_df['chainage'].min():.0f}-{test_df['chainage'].max():.0f}m)")
    print(f"TSP X_local range: [{tsp_df['X_local'].min():.0f}, {tsp_df['X_local'].max():.0f}]")

    # TSP 地质属性按 chainage 区间统计
    tsp_features = ["Vp", "Vs", "ro", "E", "Vp_Vs", "Pr"]

    print("\n--- TSP Geological Attributes by Chainage Zone (corrected) ---")
    train_chainage_max = train_df["chainage"].max()
    val_chainage_max = val_df["chainage"].max()
    test_chainage_max = test_df["chainage"].max()

    train_tsp = tsp_df[(tsp_df["X_local"] >= 0) & (tsp_df["X_local"] <= train_chainage_max)]
    val_tsp = tsp_df[(tsp_df["X_local"] > train_chainage_max) & (tsp_df["X_local"] <= val_chainage_max)]
    test_tsp = tsp_df[(tsp_df["X_local"] > val_chainage_max) & (tsp_df["X_local"] <= test_chainage_max)]

    print(f"TSP voxels: Train={len(train_tsp)}, Val={len(val_tsp)}, Test={len(test_tsp)}")

    if len(train_tsp) > 0 and len(test_tsp) > 0:
        print(f"\n{'TSP Var':<10} {'Train mean±std':<22} {'Test mean±std':<22} {'Shift (std)':<12} {'KS p':<8}")
        print("-" * 75)
        for col in tsp_features:
            t_mean, t_std = train_tsp[col].mean(), train_tsp[col].std()
            te_mean, te_std = test_tsp[col].mean(), test_tsp[col].std()
            shift = abs(te_mean - t_mean) / (t_std + 1e-8)
            ks_stat, ks_p = stats.ks_2samp(train_tsp[col].values, test_tsp[col].values)
            flag = " ***" if ks_p < 0.01 else (" **" if ks_p < 0.05 else "")
            print(f"{col:<10} {t_mean:>8.1f}±{t_std:<10.1f} "
                  f"{te_mean:>8.1f}±{te_std:<10.1f} {shift:>10.2f}    {ks_p:.4f}{flag}")

    # 监测变量标准化后的偏移 (用训练集统计量标准化)
    feature_cols = ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
    print("\n--- Monitoring Variables (standardized by train stats) ---")
    print(f"{'Variable':<16} {'Train mean±std':<22} {'Test mean±std':<22} {'Shift':<10} {'KS p':<8}")
    print("-" * 75)
    for col in feature_cols:
        t_mean, t_std = train_df[col].mean(), train_df[col].std()
        # 标准化
        train_std = (train_df[col] - t_mean) / (t_std + 1e-8)
        test_std = (test_df[col] - t_mean) / (t_std + 1e-8)
        shift = abs(test_std.mean() - train_std.mean())
        ks_stat, ks_p = stats.ks_2samp(train_std.values, test_std.values)
        flag = " ***" if ks_p < 0.01 else (" **" if ks_p < 0.05 else "")
        print(f"{col:<16} {train_std.mean():>7.3f}±{train_std.std():<9.3f} "
              f"{test_std.mean():>7.3f}±{test_std.std():<9.3f} {shift:>8.3f}    {ks_p:.4f}{flag}")

    # 标准化后 Persistence baseline
    print("\n--- Persistence Baseline (standardized) ---")
    for name, df_subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df_subset) > 1:
            # 标准化
            t_mean = train_df[feature_cols].mean()
            t_std = train_df[feature_cols].std()
            std_vals = (df_subset[feature_cols] - t_mean) / (t_std + 1e-8)
            actual = std_vals.iloc[1:].values
            pred = std_vals.iloc[:-1].values
            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # 测试集内部变化分析
    print("\n--- Test Set Internal Variation ---")
    test_std_vals = (test_df[feature_cols] - train_df[feature_cols].mean()) / (train_df[feature_cols].std() + 1e-8)
    print(f"Test set std (standardized): {test_std_vals.std().values}")
    print(f"Test set range (standardized):")
    for col in feature_cols:
        vals = test_std_vals[col].values
        print(f"  {col}: [{vals.min():.2f}, {vals.max():.2f}], span={vals.max()-vals.min():.2f}")

    # 可视化: 沿 chainage 的地质属性变化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(tsp_features):
        ax = axes[i]
        # 按 X_local 取均值
        tsp_by_x = tsp_df.groupby("X_local")[col].mean()
        ax.plot(tsp_by_x.index, tsp_by_x.values, "b-", alpha=0.7, linewidth=0.8)
        ax.axvspan(0, train_chainage_max, alpha=0.2, color="blue", label="Train")
        ax.axvspan(train_chainage_max, val_chainage_max, alpha=0.2, color="yellow", label="Val")
        ax.axvspan(val_chainage_max, test_chainage_max, alpha=0.2, color="red", label="Test")
        ax.set_title(f"TSP {col}")
        ax.set_xlabel("chainage (m)")
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/tsp_geology_by_chainage.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: outputs/tsp_geology_by_chainage.png")

    # 监测变量沿 chainage 变化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.plot(mon_df["chainage"], mon_df[col], "ko-", markersize=3, linewidth=1)
        ax.axvspan(0, train_chainage_max, alpha=0.2, color="blue", label="Train")
        ax.axvspan(train_chainage_max, val_chainage_max, alpha=0.2, color="yellow", label="Val")
        ax.axvspan(val_chainage_max, test_chainage_max, alpha=0.2, color="red", label="Test")
        ax.set_title(col)
        ax.set_xlabel("chainage (m)")
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/monitoring_by_chainage.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved: outputs/monitoring_by_chainage.png")


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    analyze_shift_corrected()
