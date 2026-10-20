from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.benchmarks import DEFAULT_BENCHMARK_CONFIGS, make_published_benchmark  # noqa: E402
from gpace.data import load_gpace_data  # noqa: E402
from gpace.metrics import regression_metrics  # noqa: E402
from gpace.splits import make_inner_splits, make_outer_split  # noqa: E402


CANDIDATES = {
    "reaction_norm_kernel": [
        {"alpha": alpha, "genotype_components": components, "environment_components": 12,
         "genotype_weight": 1.0, "environment_weight": 1.0, "interaction_weight": interaction}
        for alpha in (0.1, 1.0, 10.0)
        for components in (16, 32)
        for interaction in (0.5, 1.0, 2.0)
    ],
    "ceris_jgra": [
        {"genotype_components": components, "alpha": alpha, "min_window": 7,
         "max_window": max_window, "window_step": 7}
        for components in (16, 32)
        for alpha in (1.0, 10.0, 100.0)
        for max_window in (28, 42)
    ],
    "autogs_style": [
        {"n_markers": markers, "environment_components": components, "head": head,
         "head_params": params}
        for markers in (256, 512, 1024)
        for components in (8, 12)
        for head, params in (
            ("ridge", {"alpha": 10.0}),
            ("catboost", {"iterations": 700, "depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 5.0}),
        )
    ],
    "geformer_style": [
        {"genotype_components": components, "width": width, "heads": 4, "dropout": dropout,
         "learning_rate": learning_rate, "weight_decay": 0.0001, "batch_size": 256,
         "epochs": 120, "patience": 15}
        for components in (32, 64)
        for width in (64, 96)
        for dropout in (0.05, 0.15)
        for learning_rate in (0.0002, 0.0004)
    ],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Select published benchmark hyperparameters without using external datasets"
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--selection-split", type=int, default=1)
    result.add_argument("--inner-repeats", type=int, default=2)
    result.add_argument("--trials", type=int, default=12, help="Maximum configurations sampled per benchmark")
    result.add_argument(
        "--models", default="reaction_norm_kernel,ceris_jgra,geformer_style"
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=2026)
    return result


def _score(frame: pd.DataFrame) -> float:
    metrics = regression_metrics(frame)
    values = [
        0.40 * metrics["pcc"],
        0.30 * metrics["within_environment_pcc"],
        0.20 * metrics["within_environment_spearman"],
        0.10 * np.clip(metrics["r2"], -1.0, 1.0),
    ]
    return float(np.nansum(values))


def _frame(data, indices, predicted):
    return pd.DataFrame(
        {
            "Experiment": data.records.loc[indices, "Experiment"].to_numpy(),
            "Variety_ID": data.records.loc[indices, "Variety_ID"].to_numpy(),
            "observed": data.records.loc[indices, "grain.yield"].to_numpy(dtype=float),
            "predicted": predicted,
        }
    )


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = sorted(set(selected) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown benchmark IDs: {unknown}")
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length or None)
    outer = make_outer_split(data, args.selection_split)
    inner = make_inner_splits(data, outer.train, repeats=args.inner_repeats, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    outcomes = []
    best_models = {}

    for model_number, name in enumerate(selected):
        candidates = CANDIDATES[name]
        order = rng.permutation(len(candidates))[: min(args.trials, len(candidates))]
        best_score = -np.inf
        best_config = None
        for trial, candidate_index in enumerate(order):
            config = copy.deepcopy(candidates[int(candidate_index)])
            scores = []
            for fold in inner:
                model = make_published_benchmark(
                    name,
                    random_state=args.seed + model_number * 10_000 + trial * 100 + fold.repeat,
                    device=args.device,
                    **config,
                ).fit(data, fold.train)
                for scenario, indices in fold.evaluation_sets().items():
                    score = _score(_frame(data, indices, model.predict(data, indices)))
                    if np.isfinite(score):
                        scores.append(score)
            value = float(np.mean(scores)) if scores else -np.inf
            outcomes.append(
                {"model": name, "trial": trial, "candidate_index": int(candidate_index),
                 "objective": value, "config": json.dumps(config, sort_keys=True)}
            )
            if value > best_score:
                best_score = value
                best_config = config
        if best_config is None:
            best_config = copy.deepcopy(DEFAULT_BENCHMARK_CONFIGS[name])
        best_models[name] = best_config

    pd.DataFrame(outcomes).to_csv(output_dir / "search_trials.csv", index=False)
    result = {
        "models": best_models,
        "selection": {
            "sequence_length": args.sequence_length,
            "outer_training_split": args.selection_split,
            "inner_repeats": args.inner_repeats,
            "external_data_used": False,
            "objective": "0.40 PCC + 0.30 within-environment PCC + 0.20 within-environment Spearman + 0.10 clipped R2",
        },
    }
    (output_dir / "best_benchmark_config.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"trials: {output_dir / 'search_trials.csv'}")
    print(f"best config: {output_dir / 'best_benchmark_config.json'}")


if __name__ == "__main__":
    main()
