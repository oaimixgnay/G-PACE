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


MODEL_LABELS = {
    "mean_residual_hybrid_catboost": "GPACE V2 (frozen)",
    "reaction_norm_kernel": "Environmental-kernel reaction norm",
    "ceris_jgra": "CERIS-JGRA",
    "autogs_style": "Auto-GS-style",
    "geformer_style": "GEFormer-style",
}
SCENARIOS = ["cv0", "cv1", "cv00", "external_g_ne", "external_ng_ne"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Create a direct benchmark report for fixed GPACE")
    result.add_argument("--evaluation-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    result.add_argument("--bootstrap-repeats", type=int, default=2000)
    result.add_argument("--seed", type=int, default=2026)
    return result





def _write_report(path: Path, summary: pd.DataFrame, primary: str, comparisons: pd.DataFrame) -> None:
    table = summary.loc[summary["scenario"].isin(SCENARIOS)].copy()
    table["model_label"] = table["model"].map(MODEL_LABELS).fillna(table["model"])
    lines = [
        "# Fixed GPACE benchmark report",
        "",
        f"Primary model: `{primary}`.",
        "",
        "All benchmark methods used the same requested daily-environment window, SNP data and training/test partitions. Hyperparameters were selected from internal training partitions only.",
        "",
        "## Mean performance across supplied split/seed replicates",
        "",
        "| Scenario | Model | PCC | R2 | RMSE | Within-environment PCC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for row in table.loc[table["scenario"].eq(scenario)].itertuples():
            lines.append(
                f"| {scenario} | {row.model_label} | {row.pcc_mean:.3f} | {row.r2_mean:.3f} | "
                f"{row.rmse_mean:.3f} | {row.within_environment_pcc_mean:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `geformer_style` is a data-matched GEFormer-style reimplementation, not the official Python-3.8 compiled release.",
            "- `autogs_style` is a data-matched Auto-GS-style workflow with training-only marker screening and model selection.",
            "- The paired external bootstrap is the formal basis for claims on external datasets; split-level differences alone are not treated as independent tests.",
            "",
            f"Paired comparison rows generated: {len(comparisons)}.",
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
    available = list(dict.fromkeys(metrics["model"]))
    if args.primary_model not in available:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    comparators = [model for model in available if model != args.primary_model]
    summary = summarize_metric_replicates(metrics)
    paired = paired_replicate_bootstrap(
        metrics, args.primary_model, comparators, repeats=args.bootstrap_repeats, seed=args.seed
    )
    external = external_paired_cluster_bootstrap(
        predictions, args.primary_model, comparators,
        repeats=args.bootstrap_repeats, seed=args.seed + 1000,
    )
    summary.to_csv(output_dir / "scenario_model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_replicate_differences.csv", index=False)
    external.to_csv(output_dir / "external_paired_cluster_bootstrap.csv", index=False)
    (output_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_dir": str(evaluation_dir), "primary_model": args.primary_model,
                "comparators": comparators, "bootstrap_repeats": args.bootstrap_repeats,
                "seed": args.seed,
            }, indent=2
        ), encoding="utf-8"
    )
    _write_report(output_dir / "published_benchmark_report.md", summary, args.primary_model, paired)

    print(f"summary: {output_dir / 'scenario_model_summary.csv'}")
    print(f"external bootstrap: {output_dir / 'external_paired_cluster_bootstrap.csv'}")
    print(f"report: {output_dir / 'published_benchmark_report.md'}")


if __name__ == "__main__":
    main()
