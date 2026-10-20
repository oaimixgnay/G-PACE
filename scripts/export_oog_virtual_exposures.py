from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import EXPOSURE_COLUMNS  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Export out-of-genotype virtual exposure phenotypes for supplementary GWAS. "
            "Only CV1/CV00 predictions are retained, where the genotype was excluded "
            "from that outer split's stage-model training data."
        )
    )
    result.add_argument("--exposure-predictions", required=True)
    result.add_argument("--output-dir", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    source = Path(args.exposure_predictions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(source)
    required = {"scenario", "split", "seed", "Variety_ID", "Experiment"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Exposure prediction file lacks columns: {sorted(missing)}")
    prediction_columns = [f"predicted__{name}" for name in EXPOSURE_COLUMNS]
    missing_predictions = set(prediction_columns) - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"Exposure prediction file lacks predictions: {sorted(missing_predictions)}")
    strict = predictions.loc[predictions["scenario"].isin(["cv1", "cv00"])].copy()
    if strict.empty:
        raise ValueError("No CV1/CV00 predictions found for out-of-genotype export")
    record_means = strict.groupby(
        ["Variety_ID", "Experiment", "scenario"], as_index=False
    )[prediction_columns].mean()
    phenotype = record_means.groupby("Variety_ID", as_index=False)[prediction_columns].mean()
    phenotype = phenotype.rename(
        columns={
            "Variety_ID": "Taxa",
            **{f"predicted__{name}": f"virtual_{name.replace('.', '_')}" for name in EXPOSURE_COLUMNS},
        }
    )
    phenotype.to_csv(output_dir / "oog_virtual_exposure_phenotypes.tsv", sep="\t", index=False)
    support = strict.groupby("Variety_ID", as_index=False).agg(
        outer_prediction_rows=("Experiment", "size"),
        unique_outer_splits=("split", "nunique"),
        unique_seeds=("seed", "nunique"),
        cv1_rows=("scenario", lambda value: int((value == "cv1").sum())),
        cv00_rows=("scenario", lambda value: int((value == "cv00").sum())),
    )
    support.to_csv(output_dir / "oog_virtual_exposure_support.csv", index=False)
    metadata = {
        "source": str(source),
        "scenarios_retained": ["cv1", "cv00"],
        "guarantee": (
            "Each retained exposure prediction was generated in an outer split where its "
            "genotype was absent from the stage-model training records."
        ),
        "aggregation": "mean over held-out environments, outer splits and seeds for each genotype",
        "warning": (
            "These are model-derived virtual exposure phenotypes. Association findings are "
            "supplementary consistency evidence, not independent proof of causal genes."
        ),
    }
    (output_dir / "oog_virtual_exposure_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"phenotypes: {output_dir / 'oog_virtual_exposure_phenotypes.tsv'}")
    print(f"support: {output_dir / 'oog_virtual_exposure_support.csv'}")


if __name__ == "__main__":
    main()
