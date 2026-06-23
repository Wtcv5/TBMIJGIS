"""Controlled tabular prediction benchmark for the SJLS Dyk1252+411 case.

This script tests whether the SJLS TSP-derived geology features improve
prediction under the real monitoring task. It deliberately separates:

1. monitoring-only predictors;
2. monitoring + known-ahead TSP geology predictors.

The benchmark is tabular and lightweight. It complements the full graph
pipeline by verifying that the real data actually contains usable predictive
signal from the external TSP field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.metrics import compute_all_metrics, compute_per_variable_metrics


FEATURE_COLS = ["AdvanceRate", "RPM", "Torque", "Thrust", "Penetration", "ShieldPressure"]
TARGET_COLS = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]
GEO_COLS = ["Vp_mean", "Vp_std", "Vs_mean", "Vp_Vs_mean", "Pr_mean", "E_mean", "low_vp_frac", "Vp_grad"]


def mileage_split(n_total: int, train_r: float, val_r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    idx = np.arange(n_total)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def build_samples(mon: pd.DataFrame, geo: pd.DataFrame, k: int, h: int):
    merged_geo = geo.set_index("chainage")
    x_mon = []
    x_geo = []
    y = []
    target_chainages = []
    for i in range(len(mon) - k - h + 1):
        target_idx = i + k + h - 1
        target_chainage = float(mon.iloc[target_idx]["chainage"])
        hist = mon.iloc[i : i + k][FEATURE_COLS].to_numpy(dtype=np.float64).reshape(-1)
        if target_chainage not in merged_geo.index:
            raise KeyError(f"Missing geology drivers for chainage {target_chainage}")
        geo_feat = merged_geo.loc[target_chainage, GEO_COLS].to_numpy(dtype=np.float64)
        x_mon.append(hist)
        x_geo.append(np.concatenate([hist, geo_feat]))
        y.append(mon.iloc[target_idx][TARGET_COLS].to_numpy(dtype=np.float64))
        target_chainages.append(target_chainage)
    return (
        np.asarray(x_mon, dtype=np.float64),
        np.asarray(x_geo, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(target_chainages, dtype=np.float64),
    )


def standardize_by_train(x: np.ndarray, train_idx: np.ndarray):
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sd = x[train_idx].std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd, {"mean": mu.ravel().tolist(), "std": sd.ravel().tolist()}


def target_standardize_by_train(y: np.ndarray, train_idx: np.ndarray):
    mu = y[train_idx].mean(axis=0, keepdims=True)
    sd = y[train_idx].std(axis=0, keepdims=True) + 1e-8
    return (y - mu) / sd, mu, sd


def fit_xgb(x_train, y_train, seed: int):
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
        )
    )
    model.fit(x_train, y_train)
    return model


def fit_rf(x_train, y_train, seed: int):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_ridge(x_train, y_train):
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2.0))
    model.fit(x_train, y_train)
    return model


def relative_improvement(baseline: dict, candidate: dict) -> dict:
    return {
        "mae_reduction_pct": 100.0 * (baseline["mae"] - candidate["mae"]) / baseline["mae"],
        "rmse_reduction_pct": 100.0 * (baseline["rmse"] - candidate["rmse"]) / baseline["rmse"],
        "r2_gain": candidate["r2"] - baseline["r2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitoring-path", default="data/processed/sjls_dyk1252_411/monitoring_sjls_dyk1252_411.csv")
    parser.add_argument("--geology-path", default="data/processed/sjls_dyk1252_411/geology_drivers_sjls_dyk1252_411.csv")
    parser.add_argument("--output-dir", default="outputs/sjls_dyk1252_411_tabular")
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--predict-horizon", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mon = pd.read_csv(args.monitoring_path)
    geo = pd.read_csv(args.geology_path)
    x_mon, x_geo, y_raw, target_chainages = build_samples(
        mon=mon, geo=geo, k=args.history_window, h=args.predict_horizon
    )
    train_idx, val_idx, test_idx = mileage_split(len(y_raw), args.train_ratio, args.val_ratio)
    del val_idx

    x_mon_s, x_mon_stats = standardize_by_train(x_mon, train_idx)
    x_geo_s, x_geo_stats = standardize_by_train(x_geo, train_idx)
    y, y_mu, y_sd = target_standardize_by_train(y_raw, train_idx)

    persistence_idx = [0, 2, 3, 4, 5]
    last_hist = x_mon.reshape(len(x_mon), args.history_window, len(FEATURE_COLS))[:, -1, :]
    persistence_preds = (last_hist[:, persistence_idx] - y_mu) / y_sd

    models = {
        "Persistence": None,
        "Monitoring XGBoost": fit_xgb(x_mon_s[train_idx], y[train_idx], args.seed),
        "Monitoring+TSP XGBoost": fit_xgb(x_geo_s[train_idx], y[train_idx], args.seed),
        "Monitoring RandomForest": fit_rf(x_mon_s[train_idx], y[train_idx], args.seed),
        "Monitoring+TSP RandomForest": fit_rf(x_geo_s[train_idx], y[train_idx], args.seed),
        "Monitoring Ridge": fit_ridge(x_mon_s[train_idx], y[train_idx]),
        "Monitoring+TSP Ridge": fit_ridge(x_geo_s[train_idx], y[train_idx]),
    }

    preds = {"Persistence": persistence_preds[test_idx]}
    for name, model in models.items():
        if name == "Persistence":
            continue
        x = x_geo_s if "+TSP" in name else x_mon_s
        preds[name] = model.predict(x[test_idx])

    metrics = {name: compute_all_metrics(y[test_idx], pred) for name, pred in preds.items()}
    per_variable = {
        name: compute_per_variable_metrics(y[test_idx], pred, TARGET_COLS)
        for name, pred in preds.items()
    }

    improvements = {
        "Monitoring+TSP XGBoost vs Monitoring XGBoost": relative_improvement(
            metrics["Monitoring XGBoost"], metrics["Monitoring+TSP XGBoost"]
        ),
        "Monitoring+TSP RandomForest vs Monitoring RandomForest": relative_improvement(
            metrics["Monitoring RandomForest"], metrics["Monitoring+TSP RandomForest"]
        ),
        "Monitoring+TSP Ridge vs Monitoring Ridge": relative_improvement(
            metrics["Monitoring Ridge"], metrics["Monitoring+TSP Ridge"]
        ),
    }
    audit = {
        "monitoring_path": args.monitoring_path,
        "geology_path": args.geology_path,
        "history_window": args.history_window,
        "predict_horizon": args.predict_horizon,
        "n_samples": int(len(y)),
        "split_counts": {
            "train": int(len(train_idx)),
            "test": int(len(test_idx)),
        },
        "test_chainage_min": float(target_chainages[test_idx].min()),
        "test_chainage_max": float(target_chainages[test_idx].max()),
        "feature_sets": {
            "monitoring_only": FEATURE_COLS,
            "monitoring_plus_tsp": FEATURE_COLS + GEO_COLS,
        },
        "scaling": "features and targets are fitted on training samples only",
        "intended_use": "controlled SJLS Dyk1252+411 real-data prediction benchmark",
        "x_mon_scaler": x_mon_stats,
        "x_geo_scaler": x_geo_stats,
    }

    (output_dir / "metrics_global.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "metrics_per_variable.json").write_text(
        json.dumps(per_variable, indent=2), encoding="utf-8"
    )
    (output_dir / "improvements.json").write_text(
        json.dumps(improvements, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    rows = []
    for name, m in metrics.items():
        rows.append({"model": name, **m})
    pd.DataFrame(rows).sort_values("mae").to_csv(output_dir / "metrics_global.csv", index=False)

    print("Global metrics, sorted by MAE:")
    print(pd.DataFrame(rows).sort_values("mae").round(4).to_string(index=False))
    print("\nTSP improvements:")
    print(json.dumps(improvements, indent=2))
    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
