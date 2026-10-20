from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import EXPOSURE_COLUMNS, load_gpace_data  # noqa: E402


PRIMARY = "mean_residual_hybrid_catboost"
TRAITS = ("anthesis", "silking", "grain.number", "seed.size")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Describe genotype reaction norms against predicted exposure features. "
            "This is a concordance analysis, not a causal physiological model."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--evaluation-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--primary-model", default=PRIMARY)
    result.add_argument("--min-environments", type=int, default=4)
    return result


def _slope(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(values) < 3 or values["x"].std() < 1e-10:
        return float("nan")
    return float(np.polyfit(values["x"], values["y"], 1)[0])


def _correlation(x: pd.Series, y: pd.Series) -> tuple[int, float, float, float, float]:
    values = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(values) < 4 or values["x"].std() < 1e-10 or values["y"].std() < 1e-10:
        return len(values), np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(values["x"], values["y"])
    spearman = spearmanr(values["x"], values["y"])
    return len(values), float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue)





def main() -> None:
    args = parser().parse_args()
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    evaluation = Path(args.evaluation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(evaluation / "yield_predictions.csv")
    exposure = pd.read_csv(evaluation / "exposure_predictions.csv")
    predictions = predictions.loc[
        predictions["model"].eq(args.primary_model)
        & predictions["scenario"].astype(str).str.startswith("external")
    ].copy()
    exposure = exposure.loc[exposure["scenario"].astype(str).str.startswith("external")].copy()
    if predictions.empty or exposure.empty:
        raise ValueError("External primary yield and exposure predictions are required")
    group_keys = ["scenario", "record_id", "Experiment", "Variety_ID"]
    yield_mean = predictions.groupby(group_keys, as_index=False).agg(
        observed=("observed", "first"), predicted_yield=("predicted", "mean")
    )
    exposure_columns = [f"predicted__{name}" for name in EXPOSURE_COLUMNS]
    exposure_mean = exposure.groupby(group_keys, as_index=False)[exposure_columns].mean()
    table = yield_mean.merge(exposure_mean, on=group_keys, validate="one_to_one")

    raw_traits = data.records.loc[:, ["record_id", *TRAITS]].copy()
    raw_traits["asi"] = raw_traits["silking"] - raw_traits["anthesis"]
    table = table.merge(raw_traits, on="record_id", how="left", validate="one_to_one")
    slope_rows = []
    for (scenario, genotype), group in table.groupby(["scenario", "Variety_ID"], sort=False):
        if group["Experiment"].nunique() < args.min_environments:
            continue
        for exposure_name in EXPOSURE_COLUMNS:
            driver = f"predicted__{exposure_name}"
            observed_slope = _slope(group[driver], group["observed"])
            predicted_slope = _slope(group[driver], group["predicted_yield"])
            slope_rows.append(
                {
                    "scenario": scenario,
                    "Variety_ID": genotype,
                    "exposure": exposure_name,
                    "environment_count": group["Experiment"].nunique(),
                    "observed_slope": observed_slope,
                    "predicted_slope": predicted_slope,
                }
            )
    slopes = pd.DataFrame(slope_rows)
    if slopes.empty:
        raise ValueError("No reaction norms meet the requested minimum environment count")
    genotype_traits = table.groupby(["scenario", "Variety_ID"], as_index=False)[[*TRAITS, "asi"]].mean()
    slopes = slopes.merge(genotype_traits, on=["scenario", "Variety_ID"], how="left")
    slopes.to_csv(output_dir / "genotype_reaction_norm_slopes.csv", index=False)

    agreement_rows = []
    association_rows = []
    for (scenario, exposure_name), group in slopes.groupby(["scenario", "exposure"], sort=False):
        count, pcc, p_value, rho, rho_p = _correlation(group["observed_slope"], group["predicted_slope"])
        agreement_rows.append(
            {
                "scenario": scenario,
                "exposure": exposure_name,
                "n_genotypes": count,
                "slope_pcc": pcc,
                "slope_pcc_p": p_value,
                "slope_spearman": rho,
                "slope_spearman_p": rho_p,
            }
        )
        for trait in (*TRAITS, "asi"):
            for response in ("observed_slope", "predicted_slope"):
                count, pcc, p_value, rho, rho_p = _correlation(group[trait], group[response])
                association_rows.append(
                    {
                        "scenario": scenario,
                        "exposure": exposure_name,
                        "trait": trait,
                        "response_slope": response,
                        "n_genotypes": count,
                        "pcc": pcc,
                        "pcc_p": p_value,
                        "spearman": rho,
                        "spearman_p": rho_p,
                    }
                )
    pd.DataFrame(agreement_rows).to_csv(output_dir / "reaction_norm_agreement.csv", index=False)
    pd.DataFrame(association_rows).to_csv(output_dir / "reaction_norm_trait_associations.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "Reaction slopes are genotype-wise linear descriptions across external environments.\n"
        "Exposure predictors are the model's inferred stage-aligned representations.\n"
        "Association tests are descriptive, unadjusted screening results and must not\n"
        "be presented as causal physiology or genetic mechanism evidence.\n",
        encoding="utf-8",
    )
    print(f"slopes: {output_dir / 'genotype_reaction_norm_slopes.csv'}")
    print(f"agreement: {output_dir / 'reaction_norm_agreement.csv'}")
    print(f"associations: {output_dir / 'reaction_norm_trait_associations.csv'}")


if __name__ == "__main__":
    main()
