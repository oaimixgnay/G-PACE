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
        description="Summarize fixed-configuration early-season prediction windows."
    )
    result.add_argument(
        "--window-evaluation",
        action="append",
        required=True,
        metavar="WINDOW=EVALUATION_DIR",
        help="Repeat once per window; use full=PATH for the complete season.",
    )
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    return result


def _parse_entry(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Each --window-evaluation value must be WINDOW=EVALUATION_DIR")
    window, directory = value.split("=", maxsplit=1)
    window = window.strip()
    path = Path(directory).resolve()
    if not window or not (path / "yield_metrics.csv").is_file():
        raise ValueError(f"Invalid window evaluation: {value}")
    return window, path


def _window_order(value: str) -> tuple[int, int]:
    return (1, 0) if value == "full" else (0, int(value))


def main() -> None:
    args = parser().parse_args()
    entries = [_parse_entry(value) for value in args.window_evaluation]
    if len({window for window, _ in entries}) != len(entries):
        raise ValueError("Each time window may be specified only once")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for window, directory in entries:
        metrics = pd.read_csv(directory / "yield_metrics.csv")
        metrics["window_days"] = window
        frames.append(metrics)
    all_metrics = pd.concat(frames, ignore_index=True)
    available = [metric for metric in METRICS if metric in all_metrics]
    summary = (
        all_metrics.groupby(["window_days", "scenario", "model"], as_index=False)[available]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary["window_order"] = summary["window_days"].map(_window_order)
    summary = summary.sort_values(["window_order", "scenario", "model"]).drop(
        columns="window_order"
    )
    summary.to_csv(output_dir / "window_scenario_summary.csv", index=False)

    primary = summary.loc[summary["model"].eq(args.primary_model)].copy()
    if primary.empty:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    primary.to_csv(output_dir / "window_primary_summary.csv", index=False)
    lines = [
        "# Fixed-configuration early-season evidence",
        "",
        f"Primary model: `{args.primary_model}`",
        "",
        "The model configuration is frozen before changing the daily-data window.",
        "Values are mean results across available split/seed replicates.",
        "",
        "| Window | Scenario | PCC | R2 | RMSE | Within-environment PCC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.window_days} | {row.scenario} | {row.pcc_mean:.3f} "
            f"| {row.r2_mean:.3f} | {row.rmse_mean:.3f} "
            f"| {row.within_environment_pcc_mean:.3f} |"
        )
    (output_dir / "window_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



    print(f"summary: {output_dir / 'window_scenario_summary.csv'}")
    print(f"report: {output_dir / 'window_results.md'}")


if __name__ == "__main__":
    main()
