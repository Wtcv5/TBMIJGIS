"""Run controlled model-search experiments for accuracy and interpretation.

This script generates YAML configs from a base config, optionally executes
``run_graph_sequence_case.py`` for each trial, and summarizes the resulting metrics.

Search runs intentionally disable expensive ablations, bootstrap intervals, and
permutation tests. After selecting a promising configuration, rerun it with
``--final-ablation`` to produce a full evidence package.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
BASE_CONFIG = EXP_DIR / "config" / "bsll_dyk1017_205.yaml"
GENERATED_CONFIG_DIR = EXP_DIR / "config" / "generated_search"
SUMMARY_DIR = EXP_DIR / "outputs" / "search"


def deep_set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` inside a nested dictionary."""
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_trials(
    seeds: list[int],
    search_epochs: int,
    search_patience: int,
) -> list[dict[str, Any]]:
    """Return a compact, high-signal search grid.

    The grid favors small-data regularization and geometry sensitivity instead
    of large model capacity. It is deliberately compact so runs remain feasible
    on CPU.
    """
    model_grid = [
        {
            "tag": "balanced",
            "model.dropout": 0.25,
            "training.weight_decay": 0.005,
            "training.learning_rate": 0.001,
            "model.gru_hidden_dim": 64,
            "model.gnn_layers": 2,
        },
        {
            "tag": "low_dropout",
            "model.dropout": 0.15,
            "training.weight_decay": 0.005,
            "training.learning_rate": 0.001,
            "model.gru_hidden_dim": 64,
            "model.gnn_layers": 2,
        },
        {
            "tag": "strong_reg",
            "model.dropout": 0.35,
            "training.weight_decay": 0.02,
            "training.learning_rate": 0.0008,
            "model.gru_hidden_dim": 64,
            "model.gnn_layers": 2,
        },
        {
            "tag": "wider_temporal",
            "model.dropout": 0.25,
            "training.weight_decay": 0.01,
            "training.learning_rate": 0.001,
            "model.gru_hidden_dim": 96,
            "model.gnn_layers": 2,
        },
    ]
    geometry_grid = [
        {"tag": "tau15_eta25", "graph.tau_edge": 1.5, "graph.distance_threshold": 1.5, "graph.normal_threshold": 0.25},
        {"tag": "tau20_eta30", "graph.tau_edge": 2.0, "graph.distance_threshold": 2.0, "graph.normal_threshold": 0.30},
        {"tag": "tau25_eta35", "graph.tau_edge": 2.5, "graph.distance_threshold": 2.5, "graph.normal_threshold": 0.35},
    ]
    trials: list[dict[str, Any]] = []
    for seed, model_opts, geom_opts in itertools.product(
        seeds, model_grid, geometry_grid
    ):
        tag = f"s{seed}_{model_opts['tag']}_{geom_opts['tag']}"
        params = {
            "seed": seed,
            "runtime.output_dir": f"outputs/search/{tag}",
            "runtime.run_ablations": False,
            "runtime.bootstrap_samples": 0,
            "runtime.permutation_samples": 0,
            "training.epochs": search_epochs,
            "training.patience": search_patience,
        }
        params.update({k: v for k, v in model_opts.items() if k != "tag"})
        params.update({k: v for k, v in geom_opts.items() if k != "tag"})
        trials.append({"tag": tag, "params": params})
    return trials


def materialize_trial(base_config: dict[str, Any], trial: dict[str, Any]) -> Path:
    cfg = deepcopy(base_config)
    for key, value in trial["params"].items():
        deep_set(cfg, key, value)
    path = GENERATED_CONFIG_DIR / f"{trial['tag']}.yaml"
    save_yaml(cfg, path)
    return path


def run_trial(config_path: Path, output_dir: Path, skip_completed: bool = True) -> None:
    if skip_completed and (output_dir / "metrics_global.json").exists():
        print(f"Skipping completed trial: {output_dir.relative_to(EXP_DIR)}")
        return
    subprocess.run(
        [sys.executable, "scripts/run_graph_sequence_case.py", "--config", str(config_path.relative_to(EXP_DIR))],
        cwd=EXP_DIR,
        check=True,
    )


def read_trial_metrics(output_dir: Path) -> dict[str, Any] | None:
    metrics_path = output_dir / "metrics_global.json"
    ablation_path = output_dir / "ablation_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        global_metrics = json.load(f)
    full = global_metrics.get("Full Model")
    if not full:
        return None
    row: dict[str, Any] = {
        "mae": full.get("mae"),
        "rmse": full.get("rmse"),
        "r2": full.get("r2"),
        "pearson": full.get("pearson"),
        "spearman": full.get("spearman"),
    }
    if ablation_path.exists():
        with ablation_path.open("r", encoding="utf-8") as f:
            row["ablation_metrics"] = json.load(f)
    return row


def summarize(trials: list[dict[str, Any]], summary_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        output_dir = EXP_DIR / trial["params"]["runtime.output_dir"]
        metrics = read_trial_metrics(output_dir)
        if metrics is None:
            continue
        row = {"tag": trial["tag"], **trial["params"], **metrics}
        rows.append(row)

    rows.sort(key=lambda r: (r.get("mae", float("inf")), r.get("rmse", float("inf"))))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = summary_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if rows:
        fieldnames = [
            "tag", "mae", "rmse", "r2", "pearson", "spearman",
            "seed", "graph.tau_edge",
            "graph.normal_threshold", "model.dropout",
            "training.weight_decay", "training.learning_rate",
            "model.gru_hidden_dim", "model.gnn_layers",
        ]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return rows


def write_final_config(base_config: dict[str, Any], best_trial: dict[str, Any]) -> Path:
    cfg = deepcopy(base_config)
    for key, value in best_trial["params"].items():
        deep_set(cfg, key, value)
    deep_set(cfg, "runtime.output_dir", "outputs/tuned_mileage_final")
    deep_set(cfg, "runtime.run_ablations", True)
    deep_set(cfg, "runtime.bootstrap_samples", 300)
    deep_set(cfg, "runtime.permutation_samples", 2000)
    path = EXP_DIR / "config" / "tuned_mileage_final.yaml"
    save_yaml(cfg, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally run model-search trials.")
    parser.add_argument("--base-config", default=str(BASE_CONFIG), help="Base YAML config.")
    parser.add_argument("--execute", action="store_true", help="Run each generated trial.")
    parser.add_argument("--seeds", default="42,123", help="Comma-separated seeds.")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of trials; 0 means all.")
    parser.add_argument("--search-epochs", type=int, default=30, help="Epochs per search trial.")
    parser.add_argument("--search-patience", type=int, default=6, help="Early-stopping patience per search trial.")
    parser.add_argument("--rerun-completed", action="store_true", help="Rerun trials even if metrics already exist.")
    parser.add_argument("--final-ablation", action="store_true", help="Write and run a full ablation config for the best completed trial.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = EXP_DIR / base_path
    base_config = load_yaml(base_path)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    trials = build_trials(seeds, args.search_epochs, args.search_patience)
    if args.max_runs > 0:
        trials = trials[:args.max_runs]

    config_paths = [materialize_trial(base_config, trial) for trial in trials]
    print(f"Generated {len(config_paths)} configs under {GENERATED_CONFIG_DIR.relative_to(EXP_DIR)}")

    if args.execute:
        for i, config_path in enumerate(config_paths, start=1):
            print(f"[{i}/{len(config_paths)}] Running {config_path.name}")
            output_dir = EXP_DIR / trials[i - 1]["params"]["runtime.output_dir"]
            run_trial(config_path, output_dir, skip_completed=not args.rerun_completed)

    rows = summarize(trials, SUMMARY_DIR / "summary.csv")
    if rows:
        best = rows[0]
        print("Best completed trial:")
        print(f"  {best['tag']}: MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}, R2={best['r2']:.4f}")
        if args.final_ablation:
            trial_by_tag = {trial["tag"]: trial for trial in trials}
            final_config = write_final_config(base_config, trial_by_tag[best["tag"]])
            print(f"Wrote final config: {final_config.relative_to(EXP_DIR)}")
            if args.execute:
                run_trial(
                    final_config,
                    EXP_DIR / "outputs" / "tuned_mileage_final",
                    skip_completed=not args.rerun_completed,
                )
    else:
        print("No completed trial metrics found yet.")


if __name__ == "__main__":
    main()
