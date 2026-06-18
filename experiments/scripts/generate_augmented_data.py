"""生成式数据增广 — 基于物理约束和多元相关性的增广策略.

改进策略:
  1. Multivariate Gaussian: 使用完整协方差矩阵, 保留变量间相关性
  2. Physics-constrained: 尊重 AdvanceRate ≈ RPM × Penetration 的物理关系
  3. Sequence interpolation: 在相邻训练样本间插值, 保持时间连续性
  4. Local mixup: 只在相近样本间混合, 避免不合理的组合

约束:
  - 只增广训练集 (chainage 0-33m, 前 70%)
  - 增广样本的 chainage 仍在训练区间内
  - 保持监测变量间的物理相关性
  - 不触碰验证/测试集

输出:
  data/raw/monitoring_augmented.csv  (原始 + 增广样本)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


FEATURE_COLS = ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
TARGET_COLS = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]

# Physical constraints bounds (from training data distribution)
BOUNDS = {
    "AdvanceRate": (10, 100),
    "RPM": (2, 10),
    "Torque": (0.5, 8),
    "Thrust": (5, 40),
    "Penetration": (2, 30),
    "ShieldPressure": (0.2, 4.0),
}


def load_original_data(path: str | Path) -> pd.DataFrame:
    """加载原始监测数据."""
    return pd.read_csv(path)


def get_train_split(df: pd.DataFrame, n_total: int, train_ratio: float = 0.70
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按里程顺序划分训练集和其余集."""
    n_train = int(n_total * train_ratio)
    train_df = df.iloc[:n_train].copy()
    rest_df = df.iloc[n_train:].copy()
    return train_df, rest_df


def clip_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """截断到物理合理范围."""
    df = df.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df[col] = np.clip(df[col], lo, hi)
    return df


def enforce_physics_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """强制物理约束: AdvanceRate ≈ RPM × Penetration / 60.

    如果增广样本违反物理关系超过 30%, 调整 AdvanceRate.
    """
    df = df.copy()
    # AdvanceRate (mm/min) ≈ RPM (rev/min) × Penetration (mm/rev)
    expected_ar = df["RPM"] * df["Penetration"]
    ratio = df["AdvanceRate"] / (expected_ar + 1e-6)
    # If ratio is way off (outside [0.5, 2.0]), adjust AdvanceRate
    mask = (ratio < 0.5) | (ratio > 2.0)
    df.loc[mask, "AdvanceRate"] = expected_ar[mask] * np.clip(ratio[mask], 0.7, 1.4)
    return df


def multivariate_gaussian_augment(train_df: pd.DataFrame, n_aug: int = 5,
                                   noise_scale: float = 0.03, seed: int = 42
                                   ) -> pd.DataFrame:
    """多元高斯增广: 使用完整协方差矩阵保留变量间相关性.

    Args:
      n_aug: 每个原始样本生成的增广样本数
      noise_scale: 噪声相对于标准差的比例 (0.03 = 3%)
    """
    rng = np.random.default_rng(seed)
    features = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    # 计算均值和协方差矩阵
    mean = features.mean(axis=0)
    cov = np.cov(features.T)
    # 正则化协方差矩阵
    cov += np.eye(len(FEATURE_COLS)) * 1e-6

    # Cholesky 分解用于采样
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + np.eye(len(FEATURE_COLS)) * 1e-4)

    augmented = []
    for idx, row in train_df.iterrows():
        for aug_i in range(n_aug):
            # 生成相关噪声
            z = rng.standard_normal(len(FEATURE_COLS))
            correlated_noise = noise_scale * (L @ z)
            # 相对于该样本的绝对噪声
            new_features = row[FEATURE_COLS].to_numpy(dtype=np.float32) + correlated_noise

            chainage_offset = (aug_i + 1) * 0.01 + rng.uniform(0, 0.005)
            new_row = {"chainage": row["chainage"] + chainage_offset}
            for j, col in enumerate(FEATURE_COLS):
                new_row[col] = new_features[j]
            augmented.append(new_row)

    aug_df = pd.DataFrame(augmented)
    aug_df = clip_to_bounds(aug_df)
    aug_df = enforce_physics_constraints(aug_df)
    return aug_df


def sequence_interpolation_augment(train_df: pd.DataFrame, n_aug: int = 30,
                                     seed: int = 42) -> pd.DataFrame:
    """序列插值增广: 在相邻训练样本间线性插值, 保持时间连续性.

    生成训练区间内的密集采样, 模拟更精细的掘进过程.
    """
    rng = np.random.default_rng(seed)
    augmented = []

    train_sorted = train_df.sort_values("chainage").reset_index(drop=True)
    n = len(train_sorted)

    for i in range(n - 1):
        row_a = train_sorted.iloc[i]
        row_b = train_sorted.iloc[i + 1]

        # 在相邻样本间插值 2-3 个点
        n_interp = rng.integers(2, 4)
        for j in range(1, n_interp + 1):
            lam = j / (n_interp + 1)
            # 添加小幅扰动
            perturbation = rng.uniform(-0.05, 0.05)

            chainage = row_a["chainage"] + lam * (row_b["chainage"] - row_a["chainage"])
            chainage += rng.uniform(-0.005, 0.005)  # 避免与原始样本重合

            new_row = {"chainage": chainage}
            for col in FEATURE_COLS:
                base_val = row_a[col] + lam * (row_b[col] - row_a[col])
                new_row[col] = base_val * (1 + perturbation)
            augmented.append(new_row)

    # 如果不够 n_aug, 重复采样
    while len(augmented) < n_aug:
        i = rng.integers(0, n - 1)
        row_a = train_sorted.iloc[i]
        row_b = train_sorted.iloc[i + 1]
        lam = rng.uniform(0.2, 0.8)
        chainage = row_a["chainage"] + lam * (row_b["chainage"] - row_a["chainage"])
        chainage += rng.uniform(-0.005, 0.005)
        new_row = {"chainage": chainage}
        for col in FEATURE_COLS:
            new_row[col] = row_a[col] + lam * (row_b[col] - row_a[col])
        augmented.append(new_row)

    aug_df = pd.DataFrame(augmented[:n_aug])
    aug_df = clip_to_bounds(aug_df)
    return aug_df


def local_mixup_augment(train_df: pd.DataFrame, n_aug: int = 15, seed: int = 42
                         ) -> pd.DataFrame:
    """局部 Mixup: 只在 chainage 相近的样本间混合.

    避免不相邻地质区域的样本混合, 保持局部地质一致性.
    """
    rng = np.random.default_rng(seed)
    augmented = []

    train_sorted = train_df.sort_values("chainage").reset_index(drop=True)
    n = len(train_sorted)

    for i in range(n_aug):
        # 随机选择一个中心样本
        center_idx = rng.integers(0, n)
        # 在 ±3 个样本范围内选择混合对象
        window = 3
        lo = max(0, center_idx - window)
        hi = min(n, center_idx + window + 1)
        partner_idx = rng.integers(lo, hi)
        if partner_idx == center_idx:
            partner_idx = (partner_idx + 1) % n

        row_a = train_sorted.iloc[center_idx]
        row_b = train_sorted.iloc[partner_idx]
        lam = rng.beta(2.0, 2.0)  # 更集中在 0.5 附近

        chainage = lam * row_a["chainage"] + (1 - lam) * row_b["chainage"]
        chainage += 0.001 * (i + 1) + rng.uniform(0, 0.0005)

        new_row = {"chainage": chainage}
        for col in FEATURE_COLS:
            new_row[col] = lam * row_a[col] + (1 - lam) * row_b[col]
        augmented.append(new_row)

    aug_df = pd.DataFrame(augmented)
    aug_df = clip_to_bounds(aug_df)
    return aug_df


def physics_scenario_augment(train_df: pd.DataFrame, n_aug: int = 15,
                              seed: int = 42) -> pd.DataFrame:
    """物理场景增广: 基于物理关系生成特定场景的样本.

    生成几种典型掘进场景:
    - 高推力-低穿透 (硬岩)
    - 低推力-高穿透 (软岩)
    - 高转速-低扭矩 (正常掘进)
    - 低转速-高扭矩 (困难掘进)
    """
    rng = np.random.default_rng(seed)
    augmented = []

    features = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)

    # 场景定义: 相对于均值的偏移 (单位: std)
    scenarios = [
        # [AR, RPM, Torque, Thrust, Pen, SP]
        {"name": "hard_rock", "offset": [-0.5, -0.3, 0.8, 1.0, -0.8, 0.3]},
        {"name": "soft_rock", "offset": [0.5, 0.2, -0.5, -0.6, 0.8, -0.2]},
        {"name": "normal_fast", "offset": [0.6, 0.5, -0.2, -0.1, 0.4, 0.0]},
        {"name": "difficult_slow", "offset": [-0.8, -0.5, 0.6, 0.5, -0.5, 0.4]},
    ]

    n_per_scenario = n_aug // len(scenarios) + 1

    for scenario in scenarios:
        for i in range(n_per_scenario):
            # 随机选择一个训练样本作为基础
            base_idx = rng.integers(0, len(train_df))
            base_row = train_df.iloc[base_idx]

            # 添加场景偏移 + 小幅随机扰动
            chainage_offset = 0.0001 * (i + 1) + rng.uniform(0, 0.00005)
            new_row = {"chainage": base_row["chainage"] + chainage_offset}

            for j, col in enumerate(FEATURE_COLS):
                scenario_shift = scenario["offset"][j] * std[j] * rng.uniform(0.5, 1.0)
                noise = rng.normal(0, 0.02 * std[j])
                new_row[col] = base_row[col] + scenario_shift + noise

            augmented.append(new_row)

    aug_df = pd.DataFrame(augmented[:n_aug])
    aug_df = clip_to_bounds(aug_df)
    aug_df = enforce_physics_constraints(aug_df)
    return aug_df


def generate_augmented_data(original_path: str | Path,
                            output_path: str | Path,
                            n_mvg: int = 3,
                            n_seq: int = 30,
                            n_local_mix: int = 15,
                            n_scenario: int = 15,
                            seed: int = 42) -> pd.DataFrame:
    """生成增广数据集.

    将原始数据 + 增广样本合并, 按 chainage 排序.
    增广样本只来自训练区间, 验证/测试集保持不变.
    """
    original_df = load_original_data(original_path)
    n_total = len(original_df)

    train_df, rest_df = get_train_split(original_df, n_total)

    print(f"Original: {n_total} samples (train: {len(train_df)}, rest: {len(rest_df)})")

    # 生成增广样本
    mvg_aug = multivariate_gaussian_augment(train_df, n_aug=n_mvg, seed=seed)
    seq_aug = sequence_interpolation_augment(train_df, n_aug=n_seq, seed=seed + 1)
    local_mix_aug = local_mixup_augment(train_df, n_aug=n_local_mix, seed=seed + 2)
    scenario_aug = physics_scenario_augment(train_df, n_aug=n_scenario, seed=seed + 3)

    all_aug = pd.concat([mvg_aug, seq_aug, local_mix_aug, scenario_aug],
                        ignore_index=True)

    # 合并: 原始训练集 + 增广样本 + 原始验证测试集
    combined = pd.concat([train_df, all_aug, rest_df], ignore_index=True)
    combined = combined.sort_values("chainage").reset_index(drop=True)

    print(f"Augmented: {len(combined)} samples "
          f"(train+aug: {len(train_df) + len(all_aug)}, rest: {len(rest_df)})")
    print(f"Augmentation breakdown: mvg={n_mvg * len(train_df)}, "
          f"seq={n_seq}, local_mix={n_local_mix}, scenario={n_scenario}")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return combined


if __name__ == "__main__":
    original_path = "data/raw/monitoring.csv"
    output_path = "data/raw/monitoring_augmented.csv"

    combined = generate_augmented_data(
        original_path=original_path,
        output_path=output_path,
        n_mvg=3,          # 3 * 34 = 102 multivariate Gaussian samples
        n_seq=30,         # 30 sequence interpolation samples
        n_local_mix=15,   # 15 local mixup samples
        n_scenario=15,    # 15 physics scenario samples
        seed=42,
    )

    print("\nCombined dataset statistics:")
    print(combined.describe().round(2).to_string())
    print(f"\nTotal samples: {len(combined)}")
