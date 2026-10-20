from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CALIBRATION_SCENARIOS = {
    "external_g_ne": "cv0",
    "external_ng_ne": "cv00",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Construct scenario-matched cross-conformal empirical prediction intervals "
            "and confidence-aware external selection summaries for a frozen model."
        )
    )
    result.add_argument("--predictions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    result.add_argument("--levels", default="0.80,0.90")
    result.add_argument("--selection-fractions", default="0.10,0.20")
    result.add_argument("--selection-level", type=float, default=0.90)
    result.add_argument(
        "--scale-floor-quantile",
        type=float,
        default=0.50,
        help="Calibration quantile of seed disagreement used as a minimum local scale.",
    )
    result.add_argument("--bootstrap-repeats", type=int, default=5000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def _numbers(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _ensembled_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "record_id", "Experiment", "Variety_ID", "split"]
    return (
        predictions.groupby(keys, as_index=False, sort=False)
        .agg(
            observed=("observed", "first"),
            predicted_mean=("predicted", "mean"),
            seed_prediction_sd=("predicted", "std"),
            seed_count=("seed", "nunique"),
        )
        .assign(seed_prediction_sd=lambda frame: frame["seed_prediction_sd"].fillna(0.0))
    )


def _finite_sample_quantile(scores: np.ndarray, level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("Interval levels must be in (0, 1)")
    scores = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    if not len(scores):
        raise ValueError("No finite calibration scores")
    rank = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * level)) - 1)
    return float(scores[rank])


def _calibrate(
    calibration: pd.DataFrame,
    levels: tuple[float, ...],
    floor_quantile: float,
) -> tuple[dict[float, float], float]:
    if not 0.0 <= floor_quantile <= 1.0:
        raise ValueError("scale-floor-quantile must be in [0, 1]")
    raw_scale = calibration["seed_prediction_sd"].to_numpy(float)
    scale_floor = float(np.quantile(raw_scale, floor_quantile))
    # A zero floor would make a scaled score undefined for a deterministic seed ensemble.
    scale_floor = max(scale_floor, np.finfo(float).eps)
    scale = np.maximum(raw_scale, scale_floor)
    score = np.abs(
        calibration["observed"].to_numpy(float) - calibration["predicted_mean"].to_numpy(float)
    ) / scale
    return {level: _finite_sample_quantile(score, level) for level in levels}, scale_floor


def _interval_summary(frame: pd.DataFrame, level: float) -> pd.DataFrame:
    lower = f"lower_{level:.2f}"
    upper = f"upper_{level:.2f}"
    covered = f"covered_{level:.2f}"
    width = f"width_{level:.2f}"
    return (
        frame.groupby(["scenario", "Experiment"], as_index=False)
        .agg(
            records=("record_id", "size"),
            coverage=(covered, "mean"),
            mean_width=(width, "mean"),
            median_width=(width, "median"),
            mean_lower=(lower, "mean"),
            mean_upper=(upper, "mean"),
        )
        .assign(interval_level=level)
    )


def _selection_score(group: pd.DataFrame, score_column: str, fraction: float) -> dict[str, float | int]:
    ordered = group.sort_values([score_column, "Variety_ID"], ascending=[False, True], kind="stable")
    count = max(1, int(np.ceil(len(group) * fraction)))
    selected = ordered.head(count)
    oracle = group.sort_values(["observed", "Variety_ID"], ascending=[False, True], kind="stable").head(count)
    baseline = float(group["observed"].mean())
    selected_mean = float(selected["observed"].mean())
    oracle_mean = float(oracle["observed"].mean())
    oracle_gain = oracle_mean - baseline
    gain = selected_mean - baseline
    return {
        "entries": len(group),
        "selected_entries": count,
        "selected_mean": selected_mean,
        "selection_gain": gain,
        "selection_efficiency": gain / oracle_gain if abs(oracle_gain) > 1e-12 else np.nan,
        "selection_regret": oracle_mean - selected_mean,
        "topk_recall": float(len(set(selected["Variety_ID"]) & set(oracle["Variety_ID"])) / count),
    }


def _selection_bootstrap(
    scores: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metrics = ("selection_gain", "selection_efficiency", "selection_regret", "topk_recall")
    for (scenario, fraction), group in scores.groupby(["scenario", "selection_fraction"], sort=False):
        point = group.pivot(index="Experiment", columns="strategy", values=list(metrics))
        if "mean_prediction" not in point.columns.get_level_values(1) or "lower_bound" not in point.columns.get_level_values(1):
            continue
        environments = point.index.to_numpy()
        for metric in metrics:
            difference = (
                point[(metric, "lower_bound")] - point[(metric, "mean_prediction")]
            ).dropna().to_numpy(float)
            if not len(difference):
                continue
            samples = rng.choice(difference, size=(repeats, len(difference)), replace=True).mean(axis=1)
            lower, upper = np.percentile(samples, [2.5, 97.5])
            higher_is_better = metric != "selection_regret"
            rows.append(
                {
                    "scenario": scenario,
                    "selection_fraction": fraction,
                    "comparison": "lower_bound minus mean_prediction",
                    "metric": metric,
                    "difference": float(difference.mean()),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "probability_lower_bound_better": float(
                        np.mean(samples > 0.0 if higher_is_better else samples < 0.0)
                    ),
                    "environments": len(environments),
                    "bootstrap_method": "paired environment-level bootstrap",
                }
            )
    return pd.DataFrame(rows)





def main() -> None:
    args = parser().parse_args()
    levels = _numbers(args.levels)
    fractions = _numbers(args.selection_fractions)
    if args.selection_level not in levels:
        raise ValueError("selection-level must be listed in --levels")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(Path(args.predictions).resolve())
    primary = predictions.loc[predictions["model"].eq(args.primary_model)].copy()
    if primary.empty:
        raise ValueError(f"Primary model is absent: {args.primary_model}")
    ensembled = _ensembled_predictions(primary)
    interval_parts = []
    calibration_rows = []
    for external_scenario, calibration_scenario in CALIBRATION_SCENARIOS.items():
        calibration = ensembled.loc[ensembled["scenario"].eq(calibration_scenario)].copy()
        external = ensembled.loc[ensembled["scenario"].eq(external_scenario)].copy()
        if calibration.empty or external.empty:
            raise ValueError(f"Missing {calibration_scenario} calibration or {external_scenario} external data")
        quantiles, floor = _calibrate(calibration, levels, args.scale_floor_quantile)
        local_scale = np.maximum(external["seed_prediction_sd"].to_numpy(float), floor)
        external["calibration_scenario"] = calibration_scenario
        external["scale_floor"] = floor
        external["local_scale"] = local_scale
        for level, quantile in quantiles.items():
            radius = quantile * local_scale
            lower = f"lower_{level:.2f}"
            upper = f"upper_{level:.2f}"
            external[lower] = external["predicted_mean"] - radius
            external[upper] = external["predicted_mean"] + radius
            external[f"width_{level:.2f}"] = 2.0 * radius
            external[f"covered_{level:.2f}"] = (
                (external["observed"] >= external[lower]) & (external["observed"] <= external[upper])
            ).astype(float)
            calibration_rows.append(
                {
                    "external_scenario": external_scenario,
                    "calibration_scenario": calibration_scenario,
                    "interval_level": level,
                    "calibration_units": len(calibration),
                    "calibration_score_quantile": quantile,
                    "scale_floor": floor,
                }
            )
        interval_parts.append(external)
    intervals = pd.concat(interval_parts, ignore_index=True)
    coverage = []
    for level in levels:
        by_environment = _interval_summary(intervals, level)
        aggregate = (
            intervals.groupby("scenario", as_index=False)
            .agg(
                records=("record_id", "size"),
                environments=("Experiment", "nunique"),
                coverage=(f"covered_{level:.2f}", "mean"),
                mean_width=(f"width_{level:.2f}", "mean"),
                median_width=(f"width_{level:.2f}", "median"),
            )
            .assign(interval_level=level)
        )
        coverage.append(aggregate)
        by_environment.to_csv(output_dir / f"interval_coverage_by_environment_{level:.2f}.csv", index=False)
    coverage_table = pd.concat(coverage, ignore_index=True)
    selection_rows = []
    lower_column = f"lower_{args.selection_level:.2f}"
    for (scenario, environment), group in intervals.groupby(["scenario", "Experiment"], sort=False):
        for fraction in fractions:
            for strategy, column in (("mean_prediction", "predicted_mean"), ("lower_bound", lower_column)):
                selection_rows.append(
                    {
                        "scenario": scenario,
                        "Experiment": environment,
                        "selection_fraction": fraction,
                        "strategy": strategy,
                        **_selection_score(group, column, fraction),
                    }
                )
    selection = pd.DataFrame(selection_rows)
    selection_summary = (
        selection.groupby(["scenario", "selection_fraction", "strategy"], as_index=False)
        .agg(
            environments=("Experiment", "nunique"),
            selection_gain=("selection_gain", "mean"),
            selection_efficiency=("selection_efficiency", "mean"),
            selection_regret=("selection_regret", "mean"),
            topk_recall=("topk_recall", "mean"),
        )
    )
    bootstrap = _selection_bootstrap(selection, args.bootstrap_repeats, args.seed)
    intervals.to_csv(output_dir / "external_cross_conformal_intervals.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "conformal_calibration_summary.csv", index=False)
    coverage_table.to_csv(output_dir / "external_interval_coverage.csv", index=False)
    selection.to_csv(output_dir / "confidence_selection_by_environment.csv", index=False)
    selection_summary.to_csv(output_dir / "confidence_selection_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "confidence_selection_bootstrap.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "Intervals use scenario-matched cross-conformal residual calibration: CV0 for\n"
        "external known-genotype/new-environment predictions and CV00 for external\n"
        "new-genotype/new-environment predictions. Seed disagreement provides a local\n"
        "scale but is floored using a calibration-only quantile. These are empirical\n"
        "transfer intervals, not distribution-free guarantees under environmental shift.\n"
        "Observed external outcomes are used only to report coverage and selection utility.\n",
        encoding="utf-8",
    )
    print(f"intervals: {output_dir / 'external_cross_conformal_intervals.csv'}")
    print(f"coverage: {output_dir / 'external_interval_coverage.csv'}")
    print(f"selection: {output_dir / 'confidence_selection_summary.csv'}")


if __name__ == "__main__":
    main()
