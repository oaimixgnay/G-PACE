from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import load_gpace_data  # noqa: E402
from gpace.metrics import regression_metrics  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Map the Day-100 environmental applicability domain for frozen predictions. "
            "Novelty is calculated from daily environmental inputs only, never yield."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--predictions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--primary-model", default="mean_residual_hybrid_catboost")
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--pca-components", type=int, default=10)
    result.add_argument("--bootstrap-repeats", type=int, default=5000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def _average_primary_predictions(predictions: pd.DataFrame, primary: str) -> pd.DataFrame:
    frame = predictions.loc[
        predictions["model"].eq(primary)
        & predictions["scenario"].astype(str).str.startswith("external")
    ].copy()
    if frame.empty:
        raise ValueError(f"No external predictions for primary model {primary}")
    keys = ["scenario", "record_id", "Experiment", "Variety_ID"]
    return frame.groupby(keys, as_index=False).agg(
        observed=("observed", "first"), predicted=("predicted", "mean")
    )


def _environment_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, environment), group in predictions.groupby(["scenario", "Experiment"], sort=False):
        metric = regression_metrics(group[["Experiment", "Variety_ID", "observed", "predicted"]])
        rows.append({"scenario": scenario, "Experiment": environment, **metric})
    return pd.DataFrame(rows)


def _distance_table(data, pca_components: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = data.records
    train_indices = np.sort(
        frame.loc[frame["scenario"].eq("cross_validation"), "environment_index"].unique()
    )
    all_indices = np.arange(len(data.environment_ids))
    sequences = data.environment_sequences.astype(float, copy=True)
    # Daily weather has occasional missing values. Impute each variable using only
    # the internal training environments before constructing the novelty geometry.
    fill_value = np.nanmedian(sequences[train_indices], axis=(0, 1))
    if not np.isfinite(fill_value).all():
        missing = np.flatnonzero(~np.isfinite(fill_value)).tolist()
        raise ValueError(f"Daily variables entirely missing in training environments: {missing}")
    sequences = np.where(np.isnan(sequences), fill_value[None, None, :], sequences)
    values = sequences.reshape(len(all_indices), -1)
    scaler = StandardScaler().fit(values[train_indices])
    standardized = scaler.transform(values)
    component_count = min(pca_components, len(train_indices) - 1, standardized.shape[1])
    if component_count < 2:
        raise ValueError("At least two environment components are required")
    pca = PCA(n_components=component_count, random_state=2026).fit(standardized[train_indices])
    scores = pca.transform(standardized)
    train_scores = scores[train_indices]
    distance = np.sqrt(((scores[:, None, :] - train_scores[None, :, :]) ** 2).sum(axis=2))
    nearest_position = distance.argmin(axis=1)
    nearest_distance = distance[np.arange(len(all_indices)), nearest_position]
    train_distance = distance[train_indices]
    np.fill_diagonal(train_distance, np.inf)
    loo_distance = train_distance.min(axis=1)
    threshold = float(np.quantile(loo_distance, 0.95))
    scenario_by_environment = (
        frame.groupby("Experiment", as_index=True)["scenario"].first().to_dict()
    )
    table = pd.DataFrame(
        {
            "Experiment": data.environment_ids,
            "environment_index": all_indices,
            "data_scenario": [scenario_by_environment[name] for name in data.environment_ids],
            "pc1": scores[:, 0],
            "pc2": scores[:, 1],
            "nearest_training_environment": [data.environment_ids[train_indices[pos]] for pos in nearest_position],
            "nearest_training_distance": nearest_distance,
            "loo_distance_threshold_95": threshold,
            "outside_training_domain": nearest_distance > threshold,
        }
    )
    table.loc[train_indices, "nearest_training_distance"] = loo_distance
    table.loc[train_indices, "nearest_training_environment"] = [
        data.environment_ids[train_indices[np.argsort(train_distance[position])[0]]]
        for position in range(len(train_indices))
    ]
    table.loc[train_indices, "outside_training_domain"] = False
    return table, pca.explained_variance_ratio_


def _bootstrap_correlations(table: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, group in list(table.groupby("scenario", sort=False)) + [("external_all", table)]:
        for metric in ("rmse", "pcc", "r2", "within_environment_pcc", "top20_recall"):
            values = group[["nearest_training_distance", metric]].dropna()
            if len(values) < 4 or values.iloc[:, 0].std() < 1e-10 or values.iloc[:, 1].std() < 1e-10:
                continue
            observed_pearson = pearsonr(values.iloc[:, 0], values.iloc[:, 1])
            observed_spearman = spearmanr(values.iloc[:, 0], values.iloc[:, 1])
            samples = []
            for _ in range(repeats):
                positions = rng.integers(0, len(values), len(values))
                sample = values.iloc[positions]
                if sample.iloc[:, 0].std() > 1e-10 and sample.iloc[:, 1].std() > 1e-10:
                    samples.append(pearsonr(sample.iloc[:, 0], sample.iloc[:, 1]).statistic)
            low, high = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "environments": len(values),
                    "pearson_r": float(observed_pearson.statistic),
                    "pearson_p": float(observed_pearson.pvalue),
                    "pearson_ci_low": float(low),
                    "pearson_ci_high": float(high),
                    "spearman_rho": float(observed_spearman.statistic),
                    "spearman_p": float(observed_spearman.pvalue),
                    "bootstrap_method": "environment-level bootstrap correlation",
                }
            )
    return pd.DataFrame(rows)





def main() -> None:
    args = parser().parse_args()
    if args.sequence_length < 1:
        raise ValueError("Applicability analysis needs a fixed positive time horizon")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(Path(args.predictions).resolve())
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    domain, explained = _distance_table(data, args.pca_components)
    averaged = _average_primary_predictions(predictions, args.primary_model)
    performance = _environment_scores(averaged)
    external_domain = domain.loc[domain["data_scenario"].ne("cross_validation")].copy()
    # The four new-genotype environments are also present in the external-known
    # panel, so one environmental trajectory can legitimately have two outcome rows.
    external_domain = external_domain.merge(performance, on="Experiment", validate="one_to_many")
    correlations = _bootstrap_correlations(external_domain, args.bootstrap_repeats, args.seed)
    domain.to_csv(output_dir / "all_environment_domain_coordinates.csv", index=False)
    external_domain.to_csv(output_dir / "external_environment_applicability.csv", index=False)
    correlations.to_csv(output_dir / "novelty_performance_correlations.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
        }
    ).to_csv(output_dir / "environment_pca_variance.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "Environmental novelty is the Euclidean distance to the nearest training environment\n"
        "after fitting a standardizer and PCA only on 25 internal training environments'\n"
        "Day-100 daily trajectories. The 95th percentile of leave-one-training-environment\n"
        "distances is a descriptive applicability threshold, not a universal safety limit.\n"
        "Yield is never used to construct the distance. Correlations have only 12 and 4\n"
        "environments in the two external scenarios and should be presented with uncertainty.\n",
        encoding="utf-8",
    )
    print(f"domain: {output_dir / 'external_environment_applicability.csv'}")
    print(f"correlations: {output_dir / 'novelty_performance_correlations.csv'}")


if __name__ == "__main__":
    main()
