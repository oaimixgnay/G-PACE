from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.reporting import external_paired_cluster_bootstrap, summarize_metric_replicates  # noqa: E402


PRIMARY = "mean_residual_hybrid_catboost"
COMPARATORS = ("e2e", "e2e_catboost", "mean_residual_no_phenology_catboost")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Summarize frozen component-trait evidence")
    result.add_argument("--component-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--bootstrap-repeats", type=int, default=2000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def main() -> None:
    args = parser().parse_args()
    component_dir = Path(args.component_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    comparisons = []
    for trait_dir in sorted(path for path in component_dir.iterdir() if path.is_dir()):
        evaluation = trait_dir / "evaluation"
        metrics_path = evaluation / "yield_metrics.csv"
        predictions_path = evaluation / "yield_predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(predictions_path)
        available = set(metrics["model"])
        if PRIMARY not in available:
            raise ValueError(f"{trait_dir.name} lacks frozen primary model {PRIMARY}")
        summary = summarize_metric_replicates(metrics)
        summary.insert(0, "trait", trait_dir.name)
        summaries.append(summary)
        comparator_names = [name for name in COMPARATORS if name in available]
        paired = external_paired_cluster_bootstrap(
            predictions,
            PRIMARY,
            comparator_names,
            repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        paired.insert(0, "trait", trait_dir.name)
        comparisons.append(paired)
    if not summaries:
        raise FileNotFoundError(f"No component evaluation outputs found in {component_dir}")
    pd.concat(summaries, ignore_index=True).to_csv(
        output_dir / "component_model_summary.csv", index=False
    )
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(
            output_dir / "component_external_bootstrap.csv", index=False
        )
    print(f"summary: {output_dir / 'component_model_summary.csv'}")
    print(f"bootstrap: {output_dir / 'component_external_bootstrap.csv'}")


if __name__ == "__main__":
    main()
