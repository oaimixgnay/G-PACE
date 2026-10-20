from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPOSURE_COLUMNS = (
    "Tnight.Early",
    "Tnight.Flo",
    "Tnight.Fill",
    "Ri.Early",
    "Ri.Flo",
    "Ri.Fill",
    "Psi.Flo",
    "Psi.Fill",
    "Tmax.Early",
    "Tmax.Flo",
    "Tmax.Fill",
)

DAILY_SHARED_COLUMNS = (
    "D20.air",
    "D20cum.air",
    "D20.apex",
    "D20cum.apex",
    "Tmin.air",
    "Tmax.air",
    "Tmean.air",
    "RHmin.air",
    "RHmax.air",
    "RHmean.air",
    "Raincum",
    "Windspeedmax",
    "ET0.air",
    "Rad",
    "Ri",
    "VPD.air",
    "VPD.apex",
    "Tmax.apex",
    "nbHourSupTmax.apex",
    "Tnight",
)

DAILY_OUTPUT_COLUMNS = DAILY_SHARED_COLUMNS + (
    "soil.median3060",
    "soil.30",
    "soil.60",
)


@dataclass(frozen=True)
class GPACEData:
    records: pd.DataFrame
    genotype_ids: tuple[str, ...]
    genotype_matrix: np.ndarray
    marker_names: tuple[str, ...]
    environment_ids: tuple[str, ...]
    environment_sequences: np.ndarray
    environment_sequence_lengths: np.ndarray
    environment_feature_names: tuple[str, ...]
    genotype_partitions: pd.DataFrame
    environment_partitions: pd.DataFrame
    audit: dict[str, object]

    @property
    def exposure_columns(self) -> tuple[str, ...]:
        return EXPOSURE_COLUMNS


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")


def _collapse_exact_duplicates(
    frame: pd.DataFrame, keys: list[str], source: Path
) -> tuple[pd.DataFrame, int]:
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    remaining = frame[frame.duplicated(keys, keep=False)]
    if not remaining.empty:
        examples = remaining[keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{source.name} has conflicting duplicate keys: {examples}")
    return frame, duplicate_rows


def _load_genotypes(
    data_dir: Path,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], set[str], set[str]]:
    known_path = data_dir / "7a-Genotyping_50K_41722.csv"
    new_path = data_dir / "7c-Genotyping_50K_41722_nG.csv"
    known = pd.read_csv(known_path)
    new = pd.read_csv(new_path)
    _require_columns(known, ["Ind"], known_path)
    _require_columns(new, ["Ind"], new_path)
    if list(known.columns) != list(new.columns):
        raise ValueError("Known and new genotype matrices do not have identical marker columns")

    known_ids = set(known["Ind"].astype(str))
    new_ids = set(new["Ind"].astype(str))
    overlap = known_ids & new_ids
    if overlap:
        raise ValueError(f"Known and new genotype sets overlap: {sorted(overlap)[:5]}")

    combined = pd.concat([known, new], ignore_index=True)
    genotype_ids = tuple(combined["Ind"].astype(str))
    if len(set(genotype_ids)) != len(genotype_ids):
        raise ValueError("Duplicate genotype identifiers found after combining genotype matrices")
    matrix = combined.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Genotype matrix contains non-finite values")
    if not np.isin(matrix, [0.0, 1.0, 2.0]).all():
        raise ValueError("Genotype matrix must be encoded as 0, 1 or 2")
    return genotype_ids, matrix, tuple(combined.columns[1:]), known_ids, new_ids


def _environment_base(experiment: str) -> tuple[str, str]:
    experiment = str(experiment)
    if len(experiment) < 2 or experiment[-1] not in {"R", "W"}:
        raise ValueError(f"Cannot infer treatment from experiment name: {experiment}")
    base = experiment[:-1]
    if base == "Mar13":
        base = "13-Mar"
    return base, experiment[-1]


def _build_environment_sequences(
    daily: pd.DataFrame,
    experiments: Iterable[str],
    sequence_length: int | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    environment_ids = tuple(sorted(set(map(str, experiments))))
    source_rows: list[pd.DataFrame] = []

    for experiment in environment_ids:
        base, treatment = _environment_base(experiment)
        rows = daily.loc[daily["Env"].astype(str).eq(base)].sort_values("Date")
        if sequence_length is not None and len(rows) < sequence_length:
            raise ValueError(
                f"{experiment} has {len(rows)} daily records; {sequence_length} are required"
            )
        if sequence_length is not None:
            rows = rows.iloc[:sequence_length]
        source_rows.append(rows)

    target_length = sequence_length or max(len(rows) for rows in source_rows)
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    for experiment, rows in zip(environment_ids, source_rows):
        _, treatment = _environment_base(experiment)
        water_prefix = "WD" if treatment == "R" else "WW"
        shared = rows.loc[:, DAILY_SHARED_COLUMNS].to_numpy(dtype=np.float32)
        water = rows.loc[
            :, [f"{water_prefix}.median3060", f"{water_prefix}.30", f"{water_prefix}.60"]
        ].to_numpy(dtype=np.float32)
        values = np.concatenate([shared, water], axis=1)
        padded = np.full((target_length, values.shape[1]), np.nan, dtype=np.float32)
        padded[: len(values)] = values
        sequences.append(padded)
        lengths.append(len(values))

    return (
        environment_ids,
        np.stack(sequences).astype(np.float32, copy=False),
        np.asarray(lengths, dtype=np.int32),
    )


def load_gpace_data(
    data_dir: str | Path,
    sequence_length: int | None = 100,
    target_column: str = "grain.yield",
    require_complete_cv_target: bool = True,
) -> GPACEData:
    data_dir = Path(data_dir).resolve()
    if sequence_length is not None and sequence_length < 1:
        raise ValueError("sequence_length must be positive or None for the full season")
    yield_path = data_dir / "2b-GrainYield_components_BLUEs_level-1.csv"
    exposure_path = data_dir / "3-Indices_GenoEnv_level-1.csv"
    daily_path = data_dir / "1-Env_variables_daily.csv"
    genotype_split_path = data_dir / "9a-CrossValidation_Sets_Geno.csv"
    environment_split_path = data_dir / "9b-CrossValidation_Sets_Exp.csv"

    for path in (
        yield_path,
        exposure_path,
        daily_path,
        genotype_split_path,
        environment_split_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    yields = pd.read_csv(yield_path)
    exposures = pd.read_csv(exposure_path)
    daily = pd.read_csv(daily_path)
    genotype_partitions = pd.read_csv(genotype_split_path)
    environment_partitions = pd.read_csv(environment_split_path)

    identity = ["Experiment", "Code_ID", "Variety_ID"]
    _require_columns(yields, identity + [target_column, "type"], yield_path)
    _require_columns(exposures, identity + list(EXPOSURE_COLUMNS), exposure_path)
    _require_columns(daily, ["Env", "Date"] + list(DAILY_SHARED_COLUMNS), daily_path)
    _require_columns(genotype_partitions, ["Variety_ID", "Code_ID"], genotype_split_path)
    _require_columns(environment_partitions, ["Experiment"], environment_split_path)

    yields, duplicate_yield_rows = _collapse_exact_duplicates(yields, identity, yield_path)
    exposures, duplicate_exposure_rows = _collapse_exact_duplicates(
        exposures, identity, exposure_path
    )
    missing_target_rows = int(yields[target_column].isna().sum())

    records = exposures.merge(
        yields,
        on=identity,
        how="left",
        validate="one_to_one",
        suffixes=("_exposure", "_yield"),
    )
    if records[target_column].isna().any():
        records = records.loc[records[target_column].notna()].copy()
    # Downstream estimators use one canonical response name.  Keep the raw trait
    # column as well, so component-trait runs can share the same split machinery.
    if target_column != "grain.yield":
        records["grain.yield"] = records[target_column]

    genotype_ids, genotype_matrix, marker_names, known_ids, new_ids = _load_genotypes(data_dir)
    genotype_lookup = {name: i for i, name in enumerate(genotype_ids)}
    records["Variety_ID"] = records["Variety_ID"].astype(str)
    unknown_genotypes = sorted(set(records["Variety_ID"]) - set(genotype_lookup))
    if unknown_genotypes:
        raise ValueError(f"Records lack SNP data for genotypes: {unknown_genotypes[:10]}")

    cv_environments = set(environment_partitions["Experiment"].astype(str))

    def classify(row: pd.Series) -> str:
        genotype = row["Variety_ID"]
        experiment = str(row["Experiment"])
        if experiment in cv_environments and genotype in known_ids:
            return "cross_validation"
        if experiment not in cv_environments and genotype in known_ids:
            return "external_known_genotype"
        if experiment not in cv_environments and genotype in new_ids:
            return "external_new_genotype"
        raise ValueError(f"Unsupported genotype-environment record: {genotype}, {experiment}")

    records["scenario"] = records.apply(classify, axis=1)
    records["source_yield_type"] = records["type_yield"]
    records["source_exposure_type"] = records["type_exposure"]

    environment_ids, environment_sequences, environment_sequence_lengths = _build_environment_sequences(
        daily, records["Experiment"], sequence_length
    )
    environment_lookup = {name: i for i, name in enumerate(environment_ids)}
    records["genotype_index"] = records["Variety_ID"].map(genotype_lookup).astype(int)
    records["environment_index"] = records["Experiment"].map(environment_lookup).astype(int)
    records["record_id"] = np.arange(len(records), dtype=int)

    expected_cv = 25 * 246
    actual_cv = int(records["scenario"].eq("cross_validation").sum())
    if require_complete_cv_target and actual_cv != expected_cv:
        raise ValueError(f"Expected {expected_cv} cross-validation records, found {actual_cv}")

    split_columns = [f"training.set{i}" for i in range(1, 11)]
    _require_columns(genotype_partitions, split_columns, genotype_split_path)
    _require_columns(environment_partitions, split_columns, environment_split_path)
    valid_membership = {"yes", "no"}
    for frame, label in (
        (genotype_partitions, "genotype"),
        (environment_partitions, "environment"),
    ):
        values = set(frame[split_columns].stack().astype(str).str.lower())
        if values != valid_membership:
            raise ValueError(f"Unexpected {label} partition labels: {sorted(values)}")

    records = records.sort_values(["scenario", "Experiment", "Variety_ID"]).reset_index(drop=True)
    records["record_id"] = np.arange(len(records), dtype=int)
    audit = {
        "data_dir": str(data_dir),
        "duplicate_yield_rows_removed": duplicate_yield_rows,
        "duplicate_exposure_rows_removed": duplicate_exposure_rows,
        "prediction_target": target_column,
        "target_rows_with_missing_value_excluded": missing_target_rows,
        # Retained for compatibility with earlier validation and manifests.
        "yield_rows_with_missing_target_excluded": missing_target_rows,
        "record_counts": records["scenario"].value_counts().sort_index().to_dict(),
        "expected_cross_validation_records": expected_cv,
        "available_cross_validation_records": actual_cv,
        "complete_cross_validation_target_required": require_complete_cv_target,
        "genotype_count": len(genotype_ids),
        "known_genotype_count": len(known_ids),
        "new_genotype_count": len(new_ids),
        "environment_count": len(environment_ids),
        "marker_count": len(marker_names),
        "sequence_shape": list(environment_sequences.shape[1:]),
        "sequence_length_requested": sequence_length if sequence_length is not None else "full",
        "sequence_length_min": int(environment_sequence_lengths.min()),
        "sequence_length_max": int(environment_sequence_lengths.max()),
    }

    return GPACEData(
        records=records,
        genotype_ids=genotype_ids,
        genotype_matrix=genotype_matrix,
        marker_names=marker_names,
        environment_ids=environment_ids,
        environment_sequences=environment_sequences,
        environment_sequence_lengths=environment_sequence_lengths,
        environment_feature_names=DAILY_OUTPUT_COLUMNS,
        genotype_partitions=genotype_partitions,
        environment_partitions=environment_partitions,
        audit=audit,
    )
