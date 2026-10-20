from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_MODELS = ("g_pace", "ceris_jgra", "e2e_catboost", "reaction_norm_rff")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Decompose G2F external predictions into environment means and within-environment "
            "genotype deviations, then simulate phenotype-only selection budgets."
        )
    )
    result.add_argument("--predictions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--models", default=",".join(DEFAULT_MODELS))
    result.add_argument("--selection-fractions", default="0.10,0.20")
    result.add_argument("--phenotyping-budgets", default="0.10,0.20,0.30,0.50,0.70,1.00")
    result.add_argument("--random-repeats", type=int, default=2000)
    result.add_argument("--bootstrap-repeats", type=int, default=5000)
    result.add_argument("--random-state", type=int, default=20260915)
    return result


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _csv(value))
    if any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError("Fractions and budgets must be in (0, 1]")
    return values


def _correlation(y: np.ndarray, predicted: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) < 1e-12 or np.std(predicted) < 1e-12:
        return np.nan
    return float(pearsonr(y, predicted).statistic)


def _scores(group: pd.DataFrame, fraction: float) -> dict[str, float]:
    ordered_prediction = group.sort_values(["predicted", "Variety_ID"], ascending=[False, True], kind="stable")
    ordered_observed = group.sort_values(["observed", "Variety_ID"], ascending=[False, True], kind="stable")
    count = max(1, int(np.ceil(len(group) * fraction)))
    selected = ordered_prediction.head(count)
    oracle = ordered_observed.head(count)
    baseline = float(group["observed"].mean())
    selected_mean = float(selected["observed"].mean())
    oracle_mean = float(oracle["observed"].mean())
    return {
        "selected_mean": selected_mean,
        "oracle_mean": oracle_mean,
        "selection_gain": selected_mean - baseline,
        "selection_efficiency": (selected_mean - baseline) / (oracle_mean - baseline)
        if abs(oracle_mean - baseline) > 1e-12
        else np.nan,
        "topk_recall": float(len(set(selected["Variety_ID"]) & set(oracle["Variety_ID"])) / count),
    }


def _seed_average(source: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scenario", "record_id", "Experiment", "Variety_ID"]
    return source.groupby(keys, as_index=False).agg(observed=("observed", "first"), predicted=("predicted", "mean"))


def _component_tables(predictions: pd.DataFrame, models: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for model in models:
        model_frame = predictions.loc[predictions["model"].eq(model)].copy()
        model_frame["observed_environment_mean"] = model_frame.groupby("Experiment")["observed"].transform("mean")
        model_frame["predicted_environment_mean"] = model_frame.groupby("Experiment")["predicted"].transform("mean")
        model_frame["observed_deviation"] = model_frame["observed"] - model_frame["observed_environment_mean"]
        model_frame["predicted_deviation"] = model_frame["predicted"] - model_frame["predicted_environment_mean"]
        for environment, group in model_frame.groupby("Experiment", sort=True):
            observed_mean = float(group["observed_environment_mean"].iloc[0])
            predicted_mean = float(group["predicted_environment_mean"].iloc[0])
            y = group["observed_deviation"].to_numpy(float)
            p = group["predicted_deviation"].to_numpy(float)
            top = _scores(group, 0.20)
            rows.append(
                {
                    "model": model,
                    "Experiment": environment,
                    "entries": len(group),
                    "observed_environment_mean": observed_mean,
                    "predicted_environment_mean": predicted_mean,
                    "environment_mean_error": predicted_mean - observed_mean,
                    "environment_mean_absolute_error": abs(predicted_mean - observed_mean),
                    "deviation_pcc": _correlation(y, p),
                    "deviation_spearman": float(spearmanr(y, p).statistic),
                    "deviation_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                    "top20_recall": top["topk_recall"],
                }
            )
        environment_means = model_frame.groupby("Experiment", as_index=False).agg(
            observed=("observed_environment_mean", "first"), predicted=("predicted_environment_mean", "first")
        )
        y_dev = model_frame["observed_deviation"].to_numpy(float)
        p_dev = model_frame["predicted_deviation"].to_numpy(float)
        global_rows.append(
            {
                "model": model,
                "environments": len(environment_means),
                "environment_mean_pcc": _correlation(environment_means["observed"].to_numpy(float), environment_means["predicted"].to_numpy(float)),
                "environment_mean_rmse": float(np.sqrt(np.mean((environment_means["observed"] - environment_means["predicted"]) ** 2))),
                "deviation_pcc": _correlation(y_dev, p_dev),
                "deviation_rmse": float(np.sqrt(np.mean((y_dev - p_dev) ** 2))),
                "mean_environment_deviation_pcc": float(pd.DataFrame(rows).query("model == @model")["deviation_pcc"].mean()),
                "mean_environment_top20_recall": float(pd.DataFrame(rows).query("model == @model")["top20_recall"].mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(global_rows)


def _paired_bootstrap(by_environment: pd.DataFrame, primary: str, comparators: list[str], repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ("environment_mean_absolute_error", "deviation_pcc", "deviation_spearman", "deviation_rmse", "top20_recall")
    rows: list[dict[str, object]] = []
    primary_frame = by_environment.loc[by_environment["model"].eq(primary)]
    for comparator in comparators:
        other = by_environment.loc[by_environment["model"].eq(comparator)]
        paired = primary_frame.merge(other, on="Experiment", suffixes=("__primary", "__comparator"), validate="one_to_one")
        for metric in metrics:
            difference = (paired[f"{metric}__primary"] - paired[f"{metric}__comparator"]).to_numpy(float)
            difference = difference[np.isfinite(difference)]
            samples = rng.choice(difference, size=(repeats, len(difference)), replace=True).mean(axis=1)
            lower, upper = np.percentile(samples, [2.5, 97.5])
            higher_is_better = metric not in {"environment_mean_absolute_error", "deviation_rmse"}
            rows.append(
                {
                    "primary_model": primary,
                    "comparator_model": comparator,
                    "metric": metric,
                    "difference_primary_minus_comparator": float(difference.mean()),
                    "ci95_low": float(lower),
                    "ci95_high": float(upper),
                    "probability_primary_better": float(np.mean(samples > 0 if higher_is_better else samples < 0)),
                    "environments": len(difference),
                }
            )
    return pd.DataFrame(rows)


def _sparse_simulation(predictions: pd.DataFrame, models: tuple[str, ...], fractions: tuple[float, ...], budgets: tuple[float, ...], repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    for (model, environment), group in predictions.loc[predictions["model"].isin(models)].groupby(["model", "Experiment"], sort=False):
        for fraction in fractions:
            model_rows.append({"model": model, "Experiment": environment, "selection_fraction": fraction, **_scores(group, fraction)})
    model_scores = pd.DataFrame(model_rows)
    reference = predictions.loc[predictions["model"].eq(models[0])].copy()
    rng = np.random.default_rng(seed)
    baseline_rows: list[dict[str, object]] = []
    for environment, group in reference.groupby("Experiment", sort=False):
        n = len(group)
        observed = group[["Variety_ID", "observed"]].copy()
        for fraction in fractions:
            selected_count = max(1, int(np.ceil(n * fraction)))
            oracle = observed.sort_values(["observed", "Variety_ID"], ascending=[False, True], kind="stable").head(selected_count)
            baseline = float(observed["observed"].mean())
            oracle_gain = float(oracle["observed"].mean()) - baseline
            for budget in budgets:
                measured_count = int(np.ceil(n * budget))
                if measured_count < selected_count:
                    continue
                for repeat in range(repeats):
                    measured = observed.iloc[rng.choice(n, size=measured_count, replace=False)]
                    selected = measured.sort_values(["observed", "Variety_ID"], ascending=[False, True], kind="stable").head(selected_count)
                    selected_mean = float(selected["observed"].mean())
                    baseline_rows.append(
                        {
                            "Experiment": environment,
                            "selection_fraction": fraction,
                            "phenotyping_budget": budget,
                            "repeat": repeat + 1,
                            "selection_gain": selected_mean - baseline,
                            "selection_efficiency": (selected_mean - baseline) / oracle_gain if abs(oracle_gain) > 1e-12 else np.nan,
                            "topk_recall": float(len(set(selected["Variety_ID"]) & set(oracle["Variety_ID"])) / selected_count),
                        }
                    )
    return model_scores, pd.DataFrame(baseline_rows)





def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _csv(args.models)
    fractions = _floats(args.selection_fractions)
    budgets = _floats(args.phenotyping_budgets)
    source = pd.read_csv(args.predictions)
    required = {"model", "scenario", "record_id", "Experiment", "Variety_ID", "observed", "predicted"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    averaged = _seed_average(source)
    averaged = averaged.loc[averaged["scenario"].eq("external_2024") & averaged["model"].isin(models)].copy()
    models = tuple(model for model in models if model in set(averaged["model"]))
    if len(models) < 2:
        raise ValueError("At least two requested models must be available")
    by_environment, global_scores = _component_tables(averaged, models)
    comparisons = _paired_bootstrap(by_environment, "g_pace", [model for model in models if model != "g_pace"], args.bootstrap_repeats, args.random_state)
    model_scores, phenotype_only = _sparse_simulation(averaged, models, fractions, budgets, args.random_repeats, args.random_state)
    sparse_model_summary = model_scores.groupby(["model", "selection_fraction"], as_index=False).agg(
        environments=("Experiment", "nunique"), selection_gain=("selection_gain", "mean"), selection_efficiency=("selection_efficiency", "mean"), topk_recall=("topk_recall", "mean")
    )
    sparse_baseline_summary = phenotype_only.groupby(["selection_fraction", "phenotyping_budget"], as_index=False).agg(
        environments=("Experiment", "nunique"), selection_gain=("selection_gain", "mean"), selection_efficiency=("selection_efficiency", "mean"), topk_recall=("topk_recall", "mean")
    )
    by_environment.to_csv(output_dir / "g2f_environment_mean_and_deviation_by_environment.csv", index=False)
    global_scores.to_csv(output_dir / "g2f_environment_mean_and_deviation_summary.csv", index=False)
    comparisons.to_csv(output_dir / "g2f_decomposition_paired_environment_bootstrap.csv", index=False)
    model_scores.to_csv(output_dir / "g2f_model_selection_by_environment.csv", index=False)
    phenotype_only.to_csv(output_dir / "g2f_phenotyping_only_simulations.csv", index=False)
    sparse_model_summary.to_csv(output_dir / "g2f_model_selection_summary.csv", index=False)
    sparse_baseline_summary.to_csv(output_dir / "g2f_phenotyping_only_summary.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "Predictions are averaged across seeds before analysis. The decomposition separates each "
        "environment's mean yield from genotype deviations around that mean. The sparse baseline "
        "samples entries at random and can select only among phenotyped entries; model screens use "
        "the pre-existing zero-shot 2024 predictions and do not consume 2024 observed yields.\n",
        encoding="utf-8",
    )
    print(f"decomposition summary: {output_dir / 'g2f_environment_mean_and_deviation_summary.csv'}")
    print(f"sparse selection: {output_dir / 'g2f_model_selection_summary.csv'}")


if __name__ == "__main__":
    main()
