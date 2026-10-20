"""Combine G2F predictions and compute paired environment-cluster bootstrap evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.metrics import regression_metrics
from gpace.reporting import external_paired_cluster_bootstrap, summarize_metric_replicates


MODEL_LABELS = {
    "g_pace": "G-PACE",
    "e2e_catboost": "End-to-end CatBoost",
    "reaction_norm_rff": "Reaction-norm (RFF RKHS approximation)",
    "ceris_jgra": "CERIS-JGRA",
    "ec_gblup": "EC-GBLUP (linear GxE)",
    "geformer_style": "GEFormer-style",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="G2F 2024 paired environment-cluster bootstrap report"
    )
    result.add_argument("--gpace-evaluation-dir", required=True)
    result.add_argument(
        "--benchmark-evaluation-dir",
        action="append",
        required=True,
        help="One or more benchmark evaluation directories; repeat this option to merge runs.",
    )
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="g_pace")
    result.add_argument("--bootstrap-repeats", type=int, default=2000)
    result.add_argument("--seed", type=int, default=20260915)
    return result


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"record_id", "Experiment", "Variety_ID", "observed", "predicted", "model", "scenario", "seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")
    frame = frame.loc[frame["scenario"].eq("external_2024")].copy()
    if frame.empty:
        raise ValueError(f"{path} does not contain external_2024 predictions")
    if frame.duplicated(["model", "scenario", "seed", "record_id"]).any():
        raise ValueError(f"{path} has duplicate model/seed/record prediction keys")
    return frame


def _metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scenario, seed), frame in predictions.groupby(["model", "scenario", "seed"], sort=True):
        rows.append({"model": model, "scenario": scenario, "seed": seed, **regression_metrics(frame)})
    return pd.DataFrame(rows)


def _write_report(path: Path, summary: pd.DataFrame, paired: pd.DataFrame, primary: str) -> None:
    lines = [
        "# G2F 2024 independent benchmark report",
        "",
        "The G2F 2024 labels were held out from fitting and hyperparameter selection.",
        "The reaction-norm baseline is a random-Fourier approximation of the RKHS reaction-norm kernel, used because an exact 78k x 78k Gram matrix is infeasible.",
        "",
        "## Mean score across random seeds",
        "",
        "| Model | PCC | R2 | RMSE | Within-environment PCC | Within-environment Spearman | Top-20% recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("model").itertuples():
        lines.append(
            f"| {MODEL_LABELS.get(row.model, row.model)} | {row.pcc_mean:.3f} | {row.r2_mean:.3f} | {row.rmse_mean:.3f} | "
            f"{row.within_environment_pcc_mean:.3f} | {row.within_environment_spearman_mean:.3f} | {row.top20_recall_mean:.3f} |"
        )
    lines.extend(["", "## Paired environment-cluster bootstrap: G-PACE minus comparator", ""])
    for comparator, group in paired.groupby("comparator_model", sort=True):
        lines.append(f"### {MODEL_LABELS.get(comparator, comparator)}")
        lines.extend(["", "| Metric | Difference | 95% CI | P(G-PACE better) |", "|---|---:|---:|---:|"])
        for row in group.itertuples():
            lines.append(
                f"| {row.metric} | {row.difference_primary_minus_comparator:.3f} | "
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.probability_primary_better:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- The bootstrap clusters at the environment level and resamples records inside each selected environment.",
            "- A confidence interval crossing zero is not evidence of a stable directional advantage.",
            "- Results concern the G2F 2024 data and cannot establish performance under climates outside the observed G2F support.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpace_dir = Path(args.gpace_evaluation_dir).resolve()
    benchmark_dirs = [Path(path).resolve() for path in args.benchmark_evaluation_dir]
    primary_predictions = _read_predictions(gpace_dir / "g2f_yield_predictions.csv")
    benchmark_predictions = pd.concat(
        [_read_predictions(directory / "g2f_benchmark_predictions.csv") for directory in benchmark_dirs],
        ignore_index=True,
    )
    predictions = pd.concat([primary_predictions, benchmark_predictions], ignore_index=True)
    models = list(dict.fromkeys(predictions["model"]))
    if args.primary_model not in models:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    metrics = _metric_table(predictions)
    summary = summarize_metric_replicates(metrics)
    comparators = [model for model in models if model != args.primary_model]
    paired = external_paired_cluster_bootstrap(
        predictions,
        args.primary_model,
        comparators,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    paths = {
        "predictions": output_dir / "g2f_combined_predictions.csv",
        "metrics": output_dir / "g2f_combined_metrics.csv",
        "summary": output_dir / "g2f_scenario_model_summary.csv",
        "paired_bootstrap": output_dir / "g2f_external_2024_paired_cluster_bootstrap.csv",
        "report": output_dir / "g2f_2024_benchmark_report.md",
        "manifest": output_dir / "g2f_2024_analysis_manifest.json",
    }
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired_bootstrap"], index=False)
    _write_report(paths["report"], summary, paired, args.primary_model)
    paths["manifest"].write_text(
        json.dumps(
            {
                "gpace_evaluation_dir": str(gpace_dir),
                "benchmark_evaluation_dirs": [str(directory) for directory in benchmark_dirs],
                "primary_model": args.primary_model,
                "comparators": comparators,
                "bootstrap_repeats": args.bootstrap_repeats,
                "bootstrap_method": "paired environment-and-record cluster bootstrap on seed-averaged predictions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
