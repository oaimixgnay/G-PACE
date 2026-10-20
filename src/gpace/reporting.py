from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import regression_metrics


REPORT_METRICS = (
    "pcc",
    "r2",
    "rmse",
    "within_environment_pcc",
    "within_environment_spearman",
    "top20_recall",
)
HIGHER_IS_BETTER = {
    "pcc",
    "r2",
    "within_environment_pcc",
    "within_environment_spearman",
    "top20_recall",
}


def summarize_metric_replicates(metrics: pd.DataFrame) -> pd.DataFrame:
    available = [metric for metric in REPORT_METRICS if metric in metrics]
    grouped = metrics.groupby(["scenario", "model"], sort=False)[available]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_sd")
    count = grouped.size().rename("replicates")
    return pd.concat([count, mean, std], axis=1).reset_index()


def paired_replicate_bootstrap(
    metrics: pd.DataFrame,
    primary_model: str,
    comparator_models: Iterable[str],
    repeats: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    keys = ["scenario", "split", "seed"]
    available_metrics = [metric for metric in REPORT_METRICS if metric in metrics]

    primary = metrics.loc[metrics["model"].eq(primary_model), keys + available_metrics]
    for comparator_model in comparator_models:
        comparator = metrics.loc[
            metrics["model"].eq(comparator_model), keys + available_metrics
        ]
        paired = primary.merge(
            comparator,
            on=keys,
            how="inner",
            suffixes=("__primary", "__comparator"),
            validate="one_to_one",
        )
        for scenario, group in paired.groupby("scenario", sort=False):
            for metric in available_metrics:
                difference = (
                    group[f"{metric}__primary"] - group[f"{metric}__comparator"]
                ).dropna().to_numpy(dtype=float)
                if not len(difference):
                    continue
                sampled = rng.choice(
                    difference,
                    size=(repeats, len(difference)),
                    replace=True,
                ).mean(axis=1)
                low, high = np.percentile(sampled, [2.5, 97.5])
                if metric in HIGHER_IS_BETTER:
                    probability_better = float(np.mean(sampled > 0.0))
                else:
                    probability_better = float(np.mean(sampled < 0.0))
                rows.append(
                    {
                        "scenario": scenario,
                        "primary_model": primary_model,
                        "comparator_model": comparator_model,
                        "metric": metric,
                        "difference_primary_minus_comparator": float(difference.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "probability_primary_better": probability_better,
                        "replicates": int(len(difference)),
                        "bootstrap_method": "paired evaluation-replicate bootstrap",
                    }
                )
    return pd.DataFrame(rows)


def _seed_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scenario", "record_id", "Experiment", "Variety_ID"]
    return (
        predictions.groupby(keys, as_index=False, sort=False)
        .agg(observed=("observed", "first"), predicted=("predicted", "mean"))
    )


def _paired_prediction_frame(
    averaged: pd.DataFrame,
    scenario: str,
    primary_model: str,
    comparator_model: str,
) -> pd.DataFrame:
    keys = ["scenario", "record_id", "Experiment", "Variety_ID"]
    primary = averaged.loc[
        averaged["scenario"].eq(scenario) & averaged["model"].eq(primary_model),
        keys + ["observed", "predicted"],
    ]
    comparator = averaged.loc[
        averaged["scenario"].eq(scenario) & averaged["model"].eq(comparator_model),
        keys + ["observed", "predicted"],
    ]
    paired = primary.merge(
        comparator,
        on=keys,
        how="inner",
        suffixes=("__primary", "__comparator"),
        validate="one_to_one",
    )
    if len(paired) and not np.allclose(
        paired["observed__primary"], paired["observed__comparator"], equal_nan=True
    ):
        raise ValueError("Observed outcomes differ between paired model predictions")
    return paired


def external_paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    primary_model: str,
    comparator_models: Iterable[str],
    repeats: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    averaged = _seed_averaged_predictions(predictions)
    scenarios = [
        scenario
        for scenario in averaged["scenario"].drop_duplicates()
        if str(scenario).startswith("external")
    ]
    rows: list[dict[str, float | int | str]] = []

    for comparator_model in comparator_models:
        for scenario in scenarios:
            paired = _paired_prediction_frame(
                averaged, scenario, primary_model, comparator_model
            )
            if paired.empty:
                continue
            environments = paired["Experiment"].drop_duplicates().to_numpy()
            grouped = {name: group for name, group in paired.groupby("Experiment")}

            def scores(frame: pd.DataFrame, column: str) -> dict[str, float]:
                metric_frame = frame[
                    ["Experiment", "Variety_ID", "observed__primary", column]
                ].rename(
                    columns={
                        "observed__primary": "observed",
                        column: "predicted",
                    }
                )
                return regression_metrics(metric_frame)

            primary_point = scores(paired, "predicted__primary")
            comparator_point = scores(paired, "predicted__comparator")
            distributions = {metric: [] for metric in REPORT_METRICS}
            for _ in range(repeats):
                selected = rng.choice(
                    environments, size=len(environments), replace=True
                )
                chunks = []
                selection_chunks = []
                for cluster, environment in enumerate(selected):
                    group = grouped[environment]
                    positions = rng.integers(0, len(group), size=len(group))
                    sampled = group.iloc[positions].copy()
                    sampled["Experiment"] = f"{environment}__boot{cluster}"
                    chunks.append(sampled)
                    # Top-k recall is defined over a complete candidate set.
                    # Resampling rows duplicates varieties and biases a set-based
                    # top-k score downward, so preserve every material for this
                    # metric while still resampling environments as clusters.
                    complete_environment = group.copy()
                    complete_environment["Experiment"] = f"{environment}__boot{cluster}"
                    selection_chunks.append(complete_environment)
                bootstrap_frame = pd.concat(chunks, ignore_index=True)
                selection_frame = pd.concat(selection_chunks, ignore_index=True)
                primary_score = scores(bootstrap_frame, "predicted__primary")
                comparator_score = scores(bootstrap_frame, "predicted__comparator")
                primary_selection_score = scores(selection_frame, "predicted__primary")
                comparator_selection_score = scores(selection_frame, "predicted__comparator")
                for metric in REPORT_METRICS:
                    if metric == "top20_recall":
                        difference = (
                            primary_selection_score[metric]
                            - comparator_selection_score[metric]
                        )
                    else:
                        difference = primary_score[metric] - comparator_score[metric]
                    if np.isfinite(difference):
                        distributions[metric].append(difference)

            for metric in REPORT_METRICS:
                distribution = np.asarray(distributions[metric], dtype=float)
                if not len(distribution):
                    continue
                low, high = np.percentile(distribution, [2.5, 97.5])
                if metric in HIGHER_IS_BETTER:
                    probability_better = float(np.mean(distribution > 0.0))
                else:
                    probability_better = float(np.mean(distribution < 0.0))
                rows.append(
                    {
                        "scenario": scenario,
                        "primary_model": primary_model,
                        "comparator_model": comparator_model,
                        "metric": metric,
                        "difference_primary_minus_comparator": float(
                            primary_point[metric] - comparator_point[metric]
                        ),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "probability_primary_better": probability_better,
                        "records": int(len(paired)),
                        "environments": int(len(environments)),
                        "bootstrap_method": (
                            "paired environment-and-record cluster bootstrap on "
                            "seed-averaged predictions; top20 recall uses "
                            "environment-only resampling to preserve candidate sets"
                        ),
                    }
                )
    return pd.DataFrame(rows)
