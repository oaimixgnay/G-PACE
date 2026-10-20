from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.reporting import (  # noqa: E402
    external_paired_cluster_bootstrap,
    paired_replicate_bootstrap,
    summarize_metric_replicates,
)


SCENARIO_ORDER = ["cv0", "cv1", "cv00", "external_g_ne", "external_ng_ne"]
SCENARIO_LABELS = {
    "cv0": "CV0",
    "cv1": "CV1",
    "cv00": "CV00",
    "external_g_ne": "External G/new E",
    "external_ng_ne": "External new G/new E",
}
MODEL_LABELS = {
    "e2e": "E2E Ridge",
    "e2e_catboost": "E2E CatBoost",
    "hybrid_catboost": "Hybrid",
    "mean_residual_hybrid_catboost": "MR-Hybrid",
    "v3_mean_residual_catboost": "GPACE V3",
    "mean_residual_no_phenology_catboost": "MR without phenology",
    "mean_residual_unshrunk_catboost": "MR without shrinkage",
    "mean_residual_observed_phenology_catboost": "MR observed phenology",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Create paper tables, paired uncertainty estimates and figures"
    )
    result.add_argument("--evaluation-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument(
        "--primary-model", default="mean_residual_hybrid_catboost"
    )
    result.add_argument(
        "--comparators",
        default=(
            "e2e,e2e_catboost,hybrid_catboost,"
            "mean_residual_no_phenology_catboost,"
            "mean_residual_unshrunk_catboost,"
            "mean_residual_observed_phenology_catboost"
        ),
    )
    result.add_argument("--bootstrap-repeats", type=int, default=2000)
    result.add_argument("--seed", type=int, default=2026)
    return result














def _write_markdown(
    path: Path,
    summary: pd.DataFrame,
    primary_model: str,
    comparisons: pd.DataFrame,
) -> None:
    primary = summary.loc[summary["model"].eq(primary_model)].copy()
    primary["order"] = primary["scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    primary = primary.sort_values("order")
    lines = [
        "# GPACE paper evidence report",
        "",
        f"Primary model: `{primary_model}`",
        "",
        "## Primary model performance",
        "",
        "| Scenario | PCC | R2 | RMSE | Within-environment PCC | Top-20 recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {SCENARIO_LABELS.get(row.scenario, row.scenario)} "
            f"| {row.pcc_mean:.3f} | {row.r2_mean:.3f} | {row.rmse_mean:.3f} "
            f"| {row.within_environment_pcc_mean:.3f} | {row.top20_recall_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical outputs",
            "",
            "- `paired_replicate_differences.csv`: paired differences over split/seed evaluations.",
            "- `external_paired_cluster_bootstrap.csv`: paired cluster bootstrap on seed-averaged external predictions.",
            "- Positive differences favor the primary model for correlations and Top-20 recall.",
            "- Negative differences favor the primary model for RMSE.",
            "",
            f"Generated comparison rows: {len(comparisons)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(evaluation_dir / "yield_metrics.csv")
    predictions = pd.read_csv(evaluation_dir / "yield_predictions.csv")
    exposure_metrics = pd.read_csv(evaluation_dir / "exposure_metrics.csv")
    available_models = set(metrics["model"])
    if args.primary_model not in available_models:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    requested = [name.strip() for name in args.comparators.split(",") if name.strip()]
    comparators = [name for name in requested if name in available_models]
    missing = [name for name in requested if name not in available_models]

    summary = summarize_metric_replicates(metrics)
    paired = paired_replicate_bootstrap(
        metrics,
        args.primary_model,
        comparators,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    external = external_paired_cluster_bootstrap(
        predictions,
        args.primary_model,
        comparators,
        repeats=args.bootstrap_repeats,
        seed=args.seed + 1000,
    )
    exposure_summary = (
        exposure_metrics.groupby(["scenario", "exposure"], as_index=False)
        .agg(
            replicates=("pcc", "size"),
            pcc_mean=("pcc", "mean"),
            pcc_sd=("pcc", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
        )
    )

    summary.to_csv(output_dir / "scenario_model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_replicate_differences.csv", index=False)
    external.to_csv(output_dir / "external_paired_cluster_bootstrap.csv", index=False)
    exposure_summary.to_csv(output_dir / "exposure_summary.csv", index=False)
    metadata = {
        "evaluation_dir": str(evaluation_dir),
        "primary_model": args.primary_model,
        "comparators": comparators,
        "missing_requested_comparators": missing,
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_markdown(output_dir / "paper_results.md", summary, args.primary_model, paired)



    print(f"summary: {output_dir / 'scenario_model_summary.csv'}")
    print(f"paired comparisons: {output_dir / 'paired_replicate_differences.csv'}")
    print(f"external bootstrap: {output_dir / 'external_paired_cluster_bootstrap.csv'}")
    print(f"report: {output_dir / 'paper_results.md'}")


if __name__ == "__main__":
    main()
