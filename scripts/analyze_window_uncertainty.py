from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.reporting import (  # noqa: E402
    external_paired_cluster_bootstrap,
    paired_replicate_bootstrap,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run paired uncertainty analyses for frozen-config early-season windows."
        )
    )
    result.add_argument(
        "--window-evaluation",
        action="append",
        required=True,
        metavar="WINDOW=EVALUATION_DIR",
        help="Repeat once per window; include the reference window, normally full.",
    )
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    result.add_argument("--reference-window", default="full")
    result.add_argument(
        "--model-comparators",
        default="hybrid_catboost,e2e_catboost,e2e",
        help="Comma-separated models compared with V2 within each time window.",
    )
    result.add_argument("--bootstrap-repeats", type=int, default=2000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def _parse_entry(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Each --window-evaluation value must be WINDOW=EVALUATION_DIR")
    window, directory = value.split("=", maxsplit=1)
    window = window.strip()
    path = Path(directory).resolve()
    if not window or not (path / "yield_metrics.csv").is_file():
        raise ValueError(f"Invalid window evaluation: {value}")
    if not (path / "yield_predictions.csv").is_file():
        raise ValueError(f"Prediction file is missing for window: {value}")
    return window, path


def _renamed(frame: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    result = frame.loc[frame["model"].eq(old)].copy()
    if result.empty:
        raise ValueError(f"Model '{old}' is absent from an evaluation directory")
    result["model"] = new
    return result


def _external_comparison(
    predictions: pd.DataFrame,
    primary_label: str,
    comparator_label: str,
    repeats: int,
    seed: int,
    comparison_type: str,
    window: str,
    reference: str | None = None,
) -> pd.DataFrame:
    result = external_paired_cluster_bootstrap(
        predictions,
        primary_label,
        [comparator_label],
        repeats=repeats,
        seed=seed,
    )
    result.insert(0, "comparison_type", comparison_type)
    result.insert(1, "window", window)
    result.insert(2, "reference", reference or "")
    return result


def _replicate_comparison(
    metrics: pd.DataFrame,
    primary_label: str,
    comparator_label: str,
    repeats: int,
    seed: int,
    comparison_type: str,
    window: str,
    reference: str | None = None,
) -> pd.DataFrame:
    result = paired_replicate_bootstrap(
        metrics,
        primary_label,
        [comparator_label],
        repeats=repeats,
        seed=seed,
    )
    result.insert(0, "comparison_type", comparison_type)
    result.insert(1, "window", window)
    result.insert(2, "reference", reference or "")
    return result


def _write_markdown(path: Path, external: pd.DataFrame) -> None:
    selected = external.loc[
        external["metric"].isin(["pcc", "r2", "rmse", "within_environment_pcc"])
    ].copy()
    selected = selected.sort_values(
        ["comparison_type", "window", "scenario", "metric"]
    )
    lines = [
        "# Window and model paired uncertainty",
        "",
        "Positive differences favor the primary model for PCC, R2 and ranking metrics.",
        "Negative RMSE differences favor the primary model.",
        "External intervals use paired environment-and-record cluster bootstrap on seed-averaged predictions.",
        "",
        "| Comparison | Window | Scenario | Metric | Difference | 95% CI | P(primary better) |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in selected.itertuples():
        comparison = (
            f"V2 {row.window} vs {row.reference}"
            if row.comparison_type == "window_vs_reference"
            else f"V2 vs {row.comparator_model}"
        )
        lines.append(
            f"| {comparison} | {row.window} | {row.scenario} | {row.metric} "
            f"| {row.difference_primary_minus_comparator:.4f} "
            f"| [{row.ci_low:.4f}, {row.ci_high:.4f}] "
            f"| {row.probability_primary_better:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    entries = [_parse_entry(value) for value in args.window_evaluation]
    windows = {window: directory for window, directory in entries}
    if len(windows) != len(entries):
        raise ValueError("Each time window may be specified only once")
    if args.reference_window not in windows:
        raise ValueError("reference-window must be included in --window-evaluation")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparators = [name.strip() for name in args.model_comparators.split(",") if name.strip()]

    metrics_by_window = {
        window: pd.read_csv(directory / "yield_metrics.csv")
        for window, directory in windows.items()
    }
    predictions_by_window = {
        window: pd.read_csv(directory / "yield_predictions.csv")
        for window, directory in windows.items()
    }
    reference_metrics = metrics_by_window[args.reference_window]
    reference_predictions = predictions_by_window[args.reference_window]
    if args.primary_model not in set(reference_metrics["model"]):
        raise ValueError(f"Primary model is absent from {args.reference_window}")

    external_rows = []
    replicate_rows = []
    for position, (window, _) in enumerate(entries):
        metrics = metrics_by_window[window]
        predictions = predictions_by_window[window]
        if args.primary_model not in set(metrics["model"]):
            raise ValueError(f"Primary model is absent from {window}")

        if window != args.reference_window:
            primary_label = f"v2_{window}"
            comparator_label = f"v2_{args.reference_window}"
            paired_predictions = pd.concat(
                [
                    _renamed(predictions, args.primary_model, primary_label),
                    _renamed(reference_predictions, args.primary_model, comparator_label),
                ],
                ignore_index=True,
            )
            paired_metrics = pd.concat(
                [
                    _renamed(metrics, args.primary_model, primary_label),
                    _renamed(reference_metrics, args.primary_model, comparator_label),
                ],
                ignore_index=True,
            )
            external_rows.append(
                _external_comparison(
                    paired_predictions,
                    primary_label,
                    comparator_label,
                    args.bootstrap_repeats,
                    args.seed + position,
                    "window_vs_reference",
                    window,
                    args.reference_window,
                )
            )
            replicate_rows.append(
                _replicate_comparison(
                    paired_metrics,
                    primary_label,
                    comparator_label,
                    args.bootstrap_repeats,
                    args.seed + position,
                    "window_vs_reference",
                    window,
                    args.reference_window,
                )
            )

        for comparator_position, comparator in enumerate(comparators):
            if comparator not in set(metrics["model"]):
                continue
            primary_label = f"v2_{window}"
            comparator_label = f"{comparator}_{window}"
            paired_predictions = pd.concat(
                [
                    _renamed(predictions, args.primary_model, primary_label),
                    _renamed(predictions, comparator, comparator_label),
                ],
                ignore_index=True,
            )
            paired_metrics = pd.concat(
                [
                    _renamed(metrics, args.primary_model, primary_label),
                    _renamed(metrics, comparator, comparator_label),
                ],
                ignore_index=True,
            )
            seed = args.seed + 1000 + position * 100 + comparator_position
            external_rows.append(
                _external_comparison(
                    paired_predictions,
                    primary_label,
                    comparator_label,
                    args.bootstrap_repeats,
                    seed,
                    "model_within_window",
                    window,
                )
            )
            replicate_rows.append(
                _replicate_comparison(
                    paired_metrics,
                    primary_label,
                    comparator_label,
                    args.bootstrap_repeats,
                    seed,
                    "model_within_window",
                    window,
                )
            )

    external = pd.concat(external_rows, ignore_index=True)
    replicates = pd.concat(replicate_rows, ignore_index=True)
    external.to_csv(output_dir / "external_paired_window_comparisons.csv", index=False)
    replicates.to_csv(output_dir / "paired_replicate_window_comparisons.csv", index=False)
    _write_markdown(output_dir / "window_uncertainty.md", external)
    print(f"external bootstrap: {output_dir / 'external_paired_window_comparisons.csv'}")
    print(f"paired replicates: {output_dir / 'paired_replicate_window_comparisons.csv'}")
    print(f"report: {output_dir / 'window_uncertainty.md'}")


if __name__ == "__main__":
    main()
