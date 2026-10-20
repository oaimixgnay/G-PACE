from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_snp_stage_interactions import _csv, _fdr_bh, _load_panel, _residualize_two_way, _two_way_cluster_covariance  # noqa: E402
from gpace.data import DAILY_OUTPUT_COLUMNS, load_gpace_data  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compare aligned flowering-temperature SNP interactions to a fixed-calendar control."
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--exposure-predictions", required=True)
    result.add_argument("--aligned-interactions", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--snps", default="SYN25281,PZE.106021419")
    result.add_argument("--stage-scenarios", default="cv1,cv00")
    result.add_argument("--pc-components", type=int, default=3)
    result.add_argument("--random-state", type=int, default=20260915)
    return result


def _calendar_tmax(data, panel: pd.DataFrame) -> tuple[pd.Series, int, int]:
    cv = data.records.loc[data.records["scenario"].eq("cross_validation")]
    start = max(0, int(np.floor(cv["anthesis"].median())) - 1)
    end = min(data.environment_sequences.shape[1], int(np.ceil(cv["silking"].median())))
    if end <= start:
        raise ValueError("Could not construct a valid median flowering calendar window")
    column = DAILY_OUTPUT_COLUMNS.index("Tmax.air")
    values = {
        environment: float(np.nanmean(data.environment_sequences[index, start:end, column]))
        for index, environment in enumerate(data.environment_ids)
    }
    return panel["Experiment"].map(values).astype(float), start + 1, end


def _fit_calendar(frame: pd.DataFrame, calendar_tmax: np.ndarray, dosage_column: str, pc_columns: list[str]) -> dict[str, float]:
    genotype_codes, _ = pd.factorize(frame["Variety_ID"], sort=True)
    environment_codes, _ = pd.factorize(frame["Experiment"], sort=True)
    calendar = (calendar_tmax - calendar_tmax.mean()) / calendar_tmax.std(ddof=0)
    dosage = frame[dosage_column].to_numpy(dtype=float)
    covariates = [frame[column].to_numpy(dtype=float) * calendar for column in pc_columns]
    covariates.append(dosage * calendar)
    design = np.column_stack(
        [_residualize_two_way(value, genotype_codes, environment_codes) for value in covariates]
    )
    response = _residualize_two_way(frame["grain.yield"].to_numpy(dtype=float), genotype_codes, environment_codes)
    beta, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ beta
    covariance = _two_way_cluster_covariance(design, residuals, genotype_codes, environment_codes)
    estimate = float(beta[-1])
    se = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    df = min(frame["Variety_ID"].nunique() - 1, frame["Experiment"].nunique() - 1)
    statistic = estimate / se if se > 0 else np.nan
    p_value = float(2 * student_t.sf(abs(statistic), df=df)) if np.isfinite(statistic) else np.nan
    return {
        "interaction_beta_per_allele_per_sd": estimate,
        "interaction_se_two_way_cluster": se,
        "interaction_p_two_way_cluster": p_value,
        "ci95_low": estimate - 1.96 * se,
        "ci95_high": estimate + 1.96 * se,
    }





def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel, pc_columns, _ = _load_panel(args, ("Tmax.Flo",))
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    calendar_tmax, start_day, end_day = _calendar_tmax(data, panel)
    rows = []
    for snp in _csv(args.snps):
        dosage_column = f"dosage__{snp}"
        rows.append({"SNP": snp, **_fit_calendar(panel, calendar_tmax.to_numpy(), dosage_column, pc_columns)})
    calendar = pd.DataFrame(rows)
    calendar["interaction_q_fdr"] = _fdr_bh(calendar["interaction_p_two_way_cluster"])
    calendar["exposure_type"] = "fixed_calendar"
    calendar["calendar_start_day"] = start_day
    calendar["calendar_end_day"] = end_day
    aligned = pd.read_csv(args.aligned_interactions)
    aligned = aligned.loc[aligned["stress"].eq("Tmax.Flo") & aligned["SNP"].isin(calendar["SNP"])].copy()
    aligned = aligned.loc[:, [column for column in calendar.columns if column in aligned.columns]]
    aligned["exposure_type"] = "phenology_aligned"
    comparison = pd.concat([aligned, calendar], ignore_index=True)
    comparison.to_csv(output_dir / "aligned_vs_calendar_tmax_interactions.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "The fixed-calendar control uses Tmax.air averaged between the median internal anthesis and silking days. "
        "Unlike phenotype-aligned Tmax.Flo, it does not vary by genotype within an environment.\n",
        encoding="utf-8",
    )
    print(f"comparison: {output_dir / 'aligned_vs_calendar_tmax_interactions.csv'}")


if __name__ == "__main__":
    main()
