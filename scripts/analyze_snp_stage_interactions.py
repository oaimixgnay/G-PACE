from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import load_gpace_data  # noqa: E402


DEFAULT_SNPS = ("SYN25281", "PZE.106021419")
DEFAULT_STRESSES = ("Tmax.Flo", "Psi.Flo", "Ri.Fill")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Test yield-SNP by cross-fitted, phenology-aligned environmental-exposure "
            "interactions in the complete internal genotype-by-environment panel."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--exposure-predictions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--snps", default=",".join(DEFAULT_SNPS))
    result.add_argument("--stresses", default=",".join(DEFAULT_STRESSES))
    result.add_argument(
        "--stage-scenarios",
        default="cv1,cv00",
        help="Comma-separated out-of-genotype scenarios used to form stage exposures.",
    )
    result.add_argument("--pc-components", type=int, default=3)
    result.add_argument("--phenology-permutations", type=int, default=1000)
    result.add_argument("--random-state", type=int, default=20260915)
    return result


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _fdr_bh(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranks = np.arange(1, len(order) + 1, dtype=float)
    adjusted = valid.loc[order].to_numpy(dtype=float) * len(order) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def _residualize_two_way(values: np.ndarray, genotype_codes: np.ndarray, environment_codes: np.ndarray) -> np.ndarray:
    """Remove complete-panel genotype and environment fixed effects."""
    grand_mean = float(values.mean())
    genotype_sums = np.bincount(genotype_codes, weights=values)
    genotype_counts = np.bincount(genotype_codes)
    environment_sums = np.bincount(environment_codes, weights=values)
    environment_counts = np.bincount(environment_codes)
    return (
        values
        - genotype_sums[genotype_codes] / genotype_counts[genotype_codes]
        - environment_sums[environment_codes] / environment_counts[environment_codes]
        + grand_mean
    )


def _two_way_cluster_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    genotype_codes: np.ndarray,
    environment_codes: np.ndarray,
) -> np.ndarray:
    bread = np.linalg.pinv(design.T @ design)

    def meat_for(cluster: np.ndarray) -> np.ndarray:
        meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        for code in np.unique(cluster):
            rows = cluster == code
            score = design[rows].T @ residuals[rows]
            meat += np.outer(score, score)
        groups = len(np.unique(cluster))
        if groups > 1:
            meat *= groups / (groups - 1)
        return meat

    genotype_meat = meat_for(genotype_codes)
    environment_meat = meat_for(environment_codes)
    individual_meat = design.T @ ((residuals[:, None] ** 2) * design)
    denominator = max(len(residuals) - design.shape[1], 1)
    correction = (len(residuals) - 1) / denominator
    sandwich = (genotype_meat + environment_meat - individual_meat) * correction
    return bread @ sandwich @ bread


def _fit_twfe_arrays(
    y: np.ndarray,
    raw_stress: np.ndarray,
    dosage: np.ndarray,
    pcs: np.ndarray,
    genotype_codes: np.ndarray,
    environment_codes: np.ndarray,
) -> dict[str, float]:
    environment_totals = np.bincount(environment_codes, weights=raw_stress)
    environment_counts = np.bincount(environment_codes)
    within_environment = raw_stress - environment_totals[environment_codes] / environment_counts[environment_codes]
    stress_scale = float(within_environment.std(ddof=0))
    if stress_scale <= 1e-12:
        raise ValueError("The requested exposure has no within-environment variation")
    stress = within_environment / stress_scale

    covariates = [stress]
    for pc_values in pcs.T:
        covariates.append(pc_values * stress)
    covariates.append(dosage * stress)
    design = np.column_stack(
        [_residualize_two_way(value, genotype_codes, environment_codes) for value in covariates]
    )
    response = _residualize_two_way(y, genotype_codes, environment_codes)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    covariance = _two_way_cluster_covariance(design, residuals, genotype_codes, environment_codes)
    interaction_index = design.shape[1] - 1
    standard_error = float(np.sqrt(max(covariance[interaction_index, interaction_index], 0.0)))
    estimate = float(coefficients[interaction_index])
    degrees_of_freedom = min(len(np.unique(genotype_codes)) - 1, len(np.unique(environment_codes)) - 1)
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * student_t.sf(abs(statistic), df=degrees_of_freedom)) if np.isfinite(statistic) else np.nan
    return {
        "interaction_beta_per_allele_per_sd": estimate,
        "interaction_se_two_way_cluster": standard_error,
        "interaction_t": statistic,
        "interaction_p_two_way_cluster": p_value,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
        "stress_within_environment_sd": stress_scale,
        "degrees_of_freedom": degrees_of_freedom,
        "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def _fit_twfe(
    frame: pd.DataFrame,
    stress_column: str,
    dosage_column: str,
    pc_columns: list[str],
) -> dict[str, float]:
    # Refactorise here because leave-one-environment-out panels have a gap in the
    # original environment codes.
    genotype_codes, _ = pd.factorize(frame["Variety_ID"], sort=True)
    environment_codes, _ = pd.factorize(frame["Experiment"], sort=True)
    return _fit_twfe_arrays(
        frame["grain.yield"].to_numpy(dtype=float),
        frame[stress_column].to_numpy(dtype=float),
        frame[dosage_column].to_numpy(dtype=float),
        frame[pc_columns].to_numpy(dtype=float),
        genotype_codes.astype(int),
        environment_codes.astype(int),
    )


def _environment_effects(
    frame: pd.DataFrame,
    dosage_column: str,
    pc_columns: list[str],
    stress_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for environment, group in frame.groupby("Experiment", sort=True):
        y = group["grain.yield"].to_numpy(dtype=float)
        x = np.column_stack(
            [
                np.ones(len(group)),
                group[dosage_column].to_numpy(dtype=float),
                *[group[column].to_numpy(dtype=float) for column in pc_columns],
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        leverage = np.sum(x * (x @ np.linalg.pinv(x.T @ x)), axis=1)
        scaled = residuals / np.maximum(1.0 - leverage, 1e-8)
        covariance = np.linalg.pinv(x.T @ x) @ (x.T @ ((scaled[:, None] ** 2) * x)) @ np.linalg.pinv(x.T @ x)
        standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
        rows.append(
            {
                "Experiment": environment,
                "allele_effect_per_dosage": float(beta[1]),
                "allele_effect_se_hc3": standard_error,
                "ci95_low": float(beta[1] - 1.96 * standard_error),
                "ci95_high": float(beta[1] + 1.96 * standard_error),
                "n_genotypes": len(group),
                "mean_stage_aligned_stress": float(group[stress_column].mean()),
            }
        )
    return pd.DataFrame(rows)


def _phenology_shuffle_null(
    frame: pd.DataFrame,
    stress_column: str,
    dosage_column: str,
    pc_columns: list[str],
    repeats: int,
    random_state: int,
) -> tuple[np.ndarray, float]:
    genotype_codes, _ = pd.factorize(frame["Variety_ID"], sort=True)
    environment_codes, _ = pd.factorize(frame["Experiment"], sort=True)
    y = frame["grain.yield"].to_numpy(dtype=float)
    raw_stress = frame[stress_column].to_numpy(dtype=float)
    dosage = frame[dosage_column].to_numpy(dtype=float)
    pcs = frame[pc_columns].to_numpy(dtype=float)
    observed = _fit_twfe_arrays(
        y, raw_stress, dosage, pcs, genotype_codes.astype(int), environment_codes.astype(int)
    )["interaction_beta_per_allele_per_sd"]
    if repeats <= 0:
        return np.empty(0, dtype=float), observed
    rng = np.random.default_rng(random_state)
    null = np.empty(repeats, dtype=float)
    environment_indices = [np.flatnonzero(environment_codes == code) for code in np.unique(environment_codes)]
    for repeat in range(repeats):
        shuffled_stress = raw_stress.copy()
        for indices in environment_indices:
            shuffled_stress[indices] = rng.permutation(shuffled_stress[indices])
        null[repeat] = _fit_twfe_arrays(
            y, shuffled_stress, dosage, pcs, genotype_codes.astype(int), environment_codes.astype(int)
        )["interaction_beta_per_allele_per_sd"]
    return null, observed


def _load_panel(args: argparse.Namespace, stresses: tuple[str, ...]) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    records = data.records.loc[data.records["scenario"].eq("cross_validation")].copy()
    scenarios = set(_csv(args.stage_scenarios))
    predictions = pd.read_csv(args.exposure_predictions)
    required = {"Variety_ID", "Experiment", "scenario"} | {f"predicted__{item}" for item in stresses}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Exposure-prediction file is missing columns: {sorted(missing)}")
    predictions = predictions.loc[predictions["scenario"].isin(scenarios)].copy()
    if predictions.empty:
        raise ValueError(f"No rows found for stage scenarios: {sorted(scenarios)}")
    prediction_columns = [f"predicted__{item}" for item in stresses]
    exposures = predictions.groupby(["Variety_ID", "Experiment"], as_index=False)[prediction_columns].mean()
    exposures["Variety_ID"] = exposures["Variety_ID"].astype(str)
    panel = records.merge(exposures, on=["Variety_ID", "Experiment"], how="inner", validate="one_to_one")
    if panel.empty:
        raise ValueError("No raw yield records overlap the cross-fitted stage exposures")
    counts = panel.groupby("Variety_ID")["Experiment"].nunique()
    full_genotypes = counts.loc[counts.eq(records["Experiment"].nunique())].index
    panel = panel.loc[panel["Variety_ID"].isin(full_genotypes)].copy()
    if panel["Variety_ID"].nunique() < 30:
        raise ValueError("Too few fully cross-fitted genotypes for a SNP-by-stress analysis")

    genotype_path = Path(args.data_dir) / "7a-Genotyping_50K_41722.csv"
    genotype = pd.read_csv(genotype_path)
    genotype["Ind"] = genotype["Ind"].astype(str)
    genotype = genotype.set_index("Ind")
    marker_lookup = {_canonical(marker): marker for marker in genotype.columns}
    snp_columns: dict[str, str] = {}
    for snp in _csv(args.snps):
        marker = marker_lookup.get(_canonical(snp))
        if marker is None:
            raise ValueError(f"SNP {snp} was not found in {genotype_path.name}")
        snp_columns[snp] = marker
        panel[f"dosage__{snp}"] = panel["Variety_ID"].map(genotype[marker]).astype(float)

    all_ids = genotype.index.to_numpy()
    matrix = genotype.to_numpy(dtype=float)
    marker_sd = matrix.std(axis=0)
    informative = marker_sd > 1e-8
    matrix = (matrix[:, informative] - matrix[:, informative].mean(axis=0)) / marker_sd[informative]
    n_components = min(args.pc_components, matrix.shape[0] - 1, matrix.shape[1])
    pcs = PCA(n_components=n_components, random_state=args.random_state).fit_transform(matrix)
    pc_columns = [f"PC{index + 1}" for index in range(n_components)]
    pc_frame = pd.DataFrame(pcs, index=all_ids, columns=pc_columns)
    for column in pc_columns:
        panel[column] = panel["Variety_ID"].map(pc_frame[column]).astype(float)
        panel[column] = (panel[column] - panel[column].mean()) / panel[column].std(ddof=0)

    panel = panel.sort_values(["Variety_ID", "Experiment"]).reset_index(drop=True)
    panel["_genotype_code"], genotypes = pd.factorize(panel["Variety_ID"], sort=True)
    panel["_environment_code"], environments = pd.factorize(panel["Experiment"], sort=True)
    expected = len(genotypes) * len(environments)
    if len(panel) != expected:
        raise ValueError("The strict cross-fitted panel must contain every retained genotype in every environment")
    return panel, pc_columns, snp_columns








def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stresses = _csv(args.stresses)
    panel, pc_columns, snp_columns = _load_panel(args, stresses)
    interaction_rows: list[dict[str, object]] = []
    effect_rows: list[pd.DataFrame] = []
    loo_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    null_summary_rows: list[dict[str, object]] = []

    for snp, marker in snp_columns.items():
        dosage_column = f"dosage__{snp}"
        for stress in stresses:
            exposure_column = f"predicted__{stress}"
            fit = _fit_twfe(panel, exposure_column, dosage_column, pc_columns)
            interaction_rows.append(
                {
                    "SNP": snp,
                    "marker_column": marker,
                    "stress": stress,
                    "n_records": len(panel),
                    "n_genotypes": panel["Variety_ID"].nunique(),
                    "n_environments": panel["Experiment"].nunique(),
                    "genotypes_with_nonzero_dosage": int((panel.groupby("Variety_ID")[dosage_column].first() > 0).sum()),
                    **fit,
                }
            )
            effects = _environment_effects(panel, dosage_column, pc_columns, exposure_column)
            effects.insert(0, "SNP", snp)
            effects.insert(1, "stress", stress)
            effect_rows.append(effects)
            for heldout in sorted(panel["Experiment"].unique()):
                reduced = panel.loc[panel["Experiment"].ne(heldout)].copy()
                loo_fit = _fit_twfe(reduced, exposure_column, dosage_column, pc_columns)
                loo_rows.append({"SNP": snp, "stress": stress, "heldout_environment": heldout, **loo_fit})
            null, observed = _phenology_shuffle_null(
                panel,
                exposure_column,
                dosage_column,
                pc_columns,
                args.phenology_permutations,
                args.random_state + len(null_rows),
            )
            empirical_p = (
                float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))) if len(null) else np.nan
            )
            null_summary_rows.append(
                {
                    "SNP": snp,
                    "stress": stress,
                    "observed_interaction_beta": observed,
                    "null_mean": float(null.mean()) if len(null) else np.nan,
                    "null_sd": float(null.std(ddof=1)) if len(null) > 1 else np.nan,
                    "phenology_shuffle_repeats": len(null),
                    "empirical_two_sided_p": empirical_p,
                }
            )
            null_rows.extend(
                {
                    "SNP": snp,
                    "stress": stress,
                    "repeat": repeat + 1,
                    "shuffled_interaction_beta": value,
                    "observed_interaction_beta": observed,
                    "empirical_two_sided_p": empirical_p,
                }
                for repeat, value in enumerate(null)
            )

    interactions = pd.DataFrame(interaction_rows)
    interactions["interaction_q_fdr"] = _fdr_bh(interactions["interaction_p_two_way_cluster"])
    interactions.to_csv(output_dir / "snp_stage_interactions.csv", index=False)
    environment_effects = pd.concat(effect_rows, ignore_index=True)
    environment_effects.to_csv(output_dir / "environment_specific_allele_effects.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(output_dir / "leave_one_environment_out.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output_dir / "phenology_shuffle_null.csv", index=False)
    pd.DataFrame(null_summary_rows).to_csv(output_dir / "phenology_shuffle_summary.csv", index=False)
    pass
    pass
    manifest = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "exposure_predictions": str(Path(args.exposure_predictions).resolve()),
        "strict_stage_scenarios": list(_csv(args.stage_scenarios)),
        "panel_records": int(len(panel)),
        "panel_genotypes": int(panel["Variety_ID"].nunique()),
        "panel_environments": int(panel["Experiment"].nunique()),
        "snps": snp_columns,
        "stresses": list(stresses),
        "pc_components": pc_columns,
        "phenology_permutations": args.phenology_permutations,
        "model": (
            "two-way fixed-effects yield model (genotype and environment fixed effects), "
            "with PC-by-exposure covariates and two-way genotype/environment cluster-robust inference"
        ),
        "interpretation": (
            "This is a support analysis of SNP-dependent, phenology-aligned environmental response. "
            "It is not an independent causal validation of the GWAS loci."
        ),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Primary analysis uses only cv1/cv00 exposure predictions, so every retained genotype was "
        "excluded from the stage-model training data for each retained exposure prediction.\n"
        "The strict panel contains only genotypes with all 25 internal environments; it need not equal "
        "the 246-genotype BLUP-GWAS panel. Environment-specific allele effects are descriptive.\n"
        "The phenology-shuffle null permutes predicted stage-aligned exposures within environment.\n",
        encoding="utf-8",
    )
    print(f"interactions: {output_dir / 'snp_stage_interactions.csv'}")
    print(f"environment effects: {output_dir / 'environment_specific_allele_effects.csv'}")
    print(f"leave-one-environment-out: {output_dir / 'leave_one_environment_out.csv'}")
    print(f"phenology shuffle null: {output_dir / 'phenology_shuffle_null.csv'}")
    print(f"phenology shuffle summary: {output_dir / 'phenology_shuffle_summary.csv'}")


if __name__ == "__main__":
    main()
