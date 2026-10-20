from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "mean_residual_hybrid_catboost": "GPACE V2",
    "reaction_norm_kernel": "Reaction norm",
    "ceris_jgra": "CERIS-JGRA",
    "geformer_style": "GEFormer-style",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Evaluate environment-wise breeding selection utility from frozen predictions. "
            "Predictions define selected entries; observed yield is used only afterward."
        )
    )
    result.add_argument("--predictions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    result.add_argument(
        "--models",
        default="mean_residual_hybrid_catboost,reaction_norm_kernel,ceris_jgra,geformer_style",
    )
    result.add_argument("--fractions", default="0.05,0.10,0.20")
    result.add_argument("--bootstrap-repeats", type=int, default=5000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _seed_average(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scenario", "record_id", "Experiment", "Variety_ID"]
    return (
        predictions.groupby(keys, as_index=False, sort=False)
        .agg(observed=("observed", "first"), predicted=("predicted", "mean"))
    )


def _selection_scores(group: pd.DataFrame, fraction: float) -> dict[str, float | int]:
    group = group.sort_values(["predicted", "Variety_ID"], ascending=[False, True], kind="stable")
    n = len(group)
    selected_count = max(1, int(np.ceil(n * fraction)))
    selected = group.head(selected_count)
    oracle = group.sort_values(["observed", "Variety_ID"], ascending=[False, True], kind="stable").head(selected_count)
    baseline = float(group["observed"].mean())
    selected_mean = float(selected["observed"].mean())
    oracle_mean = float(oracle["observed"].mean())
    selection_gain = selected_mean - baseline
    oracle_gain = oracle_mean - baseline
    return {
        "entries": n,
        "selected_entries": selected_count,
        "baseline_mean": baseline,
        "selected_mean": selected_mean,
        "oracle_mean": oracle_mean,
        "selection_gain": selection_gain,
        "oracle_gain": oracle_gain,
        "selection_efficiency": selection_gain / oracle_gain if abs(oracle_gain) > 1e-12 else np.nan,
        "selection_regret": oracle_mean - selected_mean,
        "topk_recall": float(len(set(selected["Variety_ID"]) & set(oracle["Variety_ID"])) / selected_count),
    }


def _paired_environment_bootstrap(
    scores: pd.DataFrame,
    primary: str,
    comparators: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ("selection_gain", "selection_efficiency", "selection_regret", "topk_recall")
    rows: list[dict[str, object]] = []
    for (scenario, fraction), part in scores.groupby(["scenario", "selection_fraction"], sort=False):
        primary_rows = part.loc[part["model"].eq(primary)]
        for comparator in comparators:
            other_rows = part.loc[part["model"].eq(comparator)]
            paired = primary_rows.merge(
                other_rows,
                on="Experiment",
                suffixes=("__primary", "__comparator"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            for metric in metrics:
                difference = (
                    paired[f"{metric}__primary"] - paired[f"{metric}__comparator"]
                ).to_numpy(float)
                valid = difference[np.isfinite(difference)]
                if not len(valid):
                    continue
                samples = rng.choice(valid, size=(repeats, len(valid)), replace=True).mean(axis=1)
                lower, upper = np.percentile(samples, [2.5, 97.5])
                higher_is_better = metric != "selection_regret"
                rows.append(
                    {
                        "scenario": scenario,
                        "selection_fraction": fraction,
                        "primary_model": primary,
                        "comparator_model": comparator,
                        "metric": metric,
                        "difference_primary_minus_comparator": float(valid.mean()),
                        "ci_low": float(lower),
                        "ci_high": float(upper),
                        "probability_primary_better": float(
                            np.mean(samples > 0.0 if higher_is_better else samples < 0.0)
                        ),
                        "environments": int(len(valid)),
                        "bootstrap_method": "paired environment-level bootstrap on seed-averaged predictions",
                    }
                )
    return pd.DataFrame(rows)





def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = _parse_csv(args.models)
    fractions = tuple(float(value) for value in _parse_csv(args.fractions))
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("Selection fractions must be in (0, 1]")
    predictions = pd.read_csv(Path(args.predictions).resolve())
    required = {"model", "scenario", "Experiment", "Variety_ID", "observed", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    averaged = _seed_average(predictions)
    external = averaged.loc[averaged["scenario"].astype(str).str.startswith("external")].copy()
    models = [model for model in requested if model in set(external["model"])]
    if args.primary_model not in models:
        raise ValueError(f"Primary model {args.primary_model} is unavailable")
    if len(models) < 2:
        raise ValueError("At least one comparator is required for selection utility analysis")
    rows: list[dict[str, object]] = []
    for (model, scenario, environment), group in external.loc[external["model"].isin(models)].groupby(
        ["model", "scenario", "Experiment"], sort=False
    ):
        for fraction in fractions:
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "Experiment": environment,
                    "selection_fraction": fraction,
                    **_selection_scores(group, fraction),
                }
            )
    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby(["scenario", "selection_fraction", "model"], as_index=False)
        .agg(
            environments=("Experiment", "nunique"),
            selected_mean_mean=("selected_mean", "mean"),
            selection_gain_mean=("selection_gain", "mean"),
            selection_gain_sd=("selection_gain", "std"),
            selection_efficiency_mean=("selection_efficiency", "mean"),
            selection_efficiency_sd=("selection_efficiency", "std"),
            selection_regret_mean=("selection_regret", "mean"),
            selection_regret_sd=("selection_regret", "std"),
            topk_recall_mean=("topk_recall", "mean"),
            topk_recall_sd=("topk_recall", "std"),
        )
    )
    comparisons = _paired_environment_bootstrap(
        scores,
        args.primary_model,
        [model for model in models if model != args.primary_model],
        args.bootstrap_repeats,
        args.seed,
    )
    scores.to_csv(output_dir / "selection_by_environment.csv", index=False)
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    comparisons.to_csv(output_dir / "selection_paired_bootstrap.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "For each environment, each model ranks entries and selects its predicted top-k.\n"
        "Observed yield is consulted only after selection. Selection efficiency is the\n"
        "realized gain divided by the environment-specific oracle gain; regret is the\n"
        "oracle selected mean minus the model-selected mean. External new-genotype\n"
        "results have only four environments and should be interpreted descriptively.\n",
        encoding="utf-8",
    )
    print(f"scores: {output_dir / 'selection_by_environment.csv'}")
    print(f"summary: {output_dir / 'selection_summary.csv'}")
    print(f"bootstrap: {output_dir / 'selection_paired_bootstrap.csv'}")


if __name__ == "__main__":
    main()
