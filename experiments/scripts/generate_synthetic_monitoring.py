"""生成合成 TBM 监测数据，用于验证实验管线.

数据范围与 TSP 对齐: 0-48m, step=1m.
输出: data/raw/monitoring.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_monitoring(n_steps: int = 49, step_length: float = 1.0, seed: int = 42
                        ) -> pd.DataFrame:
    """生成具有合理物理量纲和互相关性的合成 TBM 监测数据.

    TBM 典型值参考:
      AdvanceRate:  20-80 mm/rev
      RPM:          3-8 rev/min
      Torque:       1-5 MN·m
      Thrust:       10-30 MN
      Penetration:  5-20 mm/rev
      ShieldPressure: 0.5-2.5 MPa

    加入空间自相关 + 交叉相关，模拟地质变化引起的响应波动.
    """
    rng = np.random.default_rng(seed)
    chainage = np.arange(n_steps) * step_length

    # 生成潜在因子: 模拟沿里程的地质变化 (低频 + 高频噪声)
    t = chainage / chainage.max()  # [0, 1]
    geo_factor = (
        0.5 * np.sin(2 * np.pi * t * 1.5) +      # 低频地质趋势
        0.3 * np.cos(2 * np.pi * t * 0.7) +
        rng.normal(0, 0.1, n_steps)               # 随机波动
    )

    # 各变量基值 + 受地质因子调制的波动
    advance_rate = 45 + 15 * geo_factor + rng.normal(0, 3, n_steps)
    rpm = 6.0 + 0.5 * geo_factor + rng.normal(0, 0.3, n_steps)
    torque = 3.0 + 0.8 * geo_factor + rng.normal(0, 0.2, n_steps)
    thrust = 18.0 + 5.0 * geo_factor + rng.normal(0, 1.5, n_steps)
    penetration = 12.0 + 4.0 * geo_factor + rng.normal(0, 1.0, n_steps)
    shield_pressure = 1.5 + 0.5 * geo_factor + rng.normal(0, 0.15, n_steps)

    # 加入一些局部异常 (模拟困难掘进段)
    anomaly_mask = (t > 0.4) & (t < 0.55)
    torque[anomaly_mask] += 1.5 + rng.normal(0, 0.3, anomaly_mask.sum())
    thrust[anomaly_mask] += 8.0 + rng.normal(0, 2.0, anomaly_mask.sum())
    penetration[anomaly_mask] -= 4.0 + rng.normal(0, 0.5, anomaly_mask.sum())
    advance_rate[anomaly_mask] -= 10.0 + rng.normal(0, 2.0, anomaly_mask.sum())
    shield_pressure[anomaly_mask] += 0.8 + rng.normal(0, 0.2, anomaly_mask.sum())

    # 确保物理合理性 (截断)
    advance_rate = np.clip(advance_rate, 10, 100)
    rpm = np.clip(rpm, 2, 10)
    torque = np.clip(torque, 0.5, 8)
    thrust = np.clip(thrust, 5, 40)
    penetration = np.clip(penetration, 2, 30)
    shield_pressure = np.clip(shield_pressure, 0.2, 4.0)

    df = pd.DataFrame({
        "chainage": chainage,
        "AdvanceRate": advance_rate,
        "RPM": rpm,
        "Torque": torque,
        "Thrust": thrust,
        "Penetration": penetration,
        "ShieldPressure": shield_pressure,
    })
    return df


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_monitoring()
    df.to_csv(output_dir / "monitoring.csv", index=False)
    print(f"Generated {len(df)} monitoring records → {output_dir / 'monitoring.csv'}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
