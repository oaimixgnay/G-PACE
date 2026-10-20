from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = (
    "pcc",
    "r2",
    "rmse",
    "within_environment_pcc",
    "within_environment_spearman",
    "top20_recall",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Summarize fixed-configuration environment-group holdout experiments."
    )
    result.add_argument("--evaluation-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--holdout-scheme", choices=("region", "stress"), required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    return result


def main() -> None:
    args = parser().parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(evaluation_dir / "yield_metrics.csv")
    if "holdout_scheme" not in metrics:
        raise ValueError("This evaluation was not run with --environment-holdout")
    metrics = metrics.loc[metrics["holdout_scheme"].eq(args.holdout_scheme)].copy()
    if metrics.empty:
        raise ValueError(f"No {args.holdout_scheme} holdout metrics were found")
    available = [metric for metric in METRICS if metric in metrics]
    summary = (
        metrics.groupby(["heldout_group", "model"], as_index=False)[available]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(["heldout_group", "model"])
    summary.to_csv(output_dir / "environment_holdout_summary.csv", index=False)
    primary = summary.loc[summary["model"].eq(args.primary_model)]
    if primary.empty:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    lines = [
        f"# Frozen V2 {args.holdout_scheme} holdout evidence",
        "",
        f"Primary model: `{args.primary_model}`",
        "",
        "Each row withholds every CV environment in one group during fitting.",
        "The configuration is fixed before these analyses; seed variation is not a substitute for independent groups.",
        "",
        "| Held-out group | PCC | R2 | RMSE | Within-environment PCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.heldout_group} | {row.pcc_mean:.3f} | {row.r2_mean:.3f} "
            f"| {row.rmse_mean:.3f} | {row.within_environment_pcc_mean:.3f} |"
        )
    (output_dir / "environment_holdout_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )



    print(f"summary: {output_dir / 'environment_holdout_summary.csv'}")
    print(f"report: {output_dir / 'environment_holdout_results.md'}")


if __name__ == "__main__":
    main()
