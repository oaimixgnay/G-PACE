"""Data preparation for the independent G2F 2024 validation.

The G2F release is a separate maize hybrid trial network.  It is deliberately
kept outside :mod:`gpace.data`, which is tied to the original NG file layout.
Observed 2024 yield is retained only as a final evaluation label; no 2024
phenology or yield is used to fit the representation or yield models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GPACEData, EXPOSURE_COLUMNS


G2F_DAILY_COLUMNS = (
    "T2M_MIN",
    "T2M_MAX",
    "T2M",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_PAR_TOT",
    "GWETROOT",
    "WS2M",
    "T2MDEW",
    "VPD_KPA",
)

_FILES = {
    "train_traits": "1_Training_Trait_Data_2014_2023.csv",
    "train_meta": "2_Training_Meta_Data_2014_2023.csv",
    "train_weather": "4_Training_Weather_Data_2014_2023_full_year.csv",
    "genotypes": "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt",
    "test_template": "1_Submission_Template_2024.csv",
    "test_meta": "2_Testing_Meta_Data_2024.csv",
    "test_weather": "4_Testing_Weather_Data_2024_full_year.csv",
    "test_observed": "7_Testing_Observed_Values.csv",
}


@dataclass(frozen=True)
class G2FPreparedData:
    """Canonical data plus indices used by the G2F experiment runner."""

    data: GPACEData
    training_indices: np.ndarray
    external_indices: np.ndarray


def _require_files(data_dir: Path) -> dict[str, Path]:
    paths = {key: data_dir / name for key, name in _FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required G2F files:\n" + "\n".join(missing))
    return paths


def _parse_g2f_date(values: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    invalid = parsed.isna() & values.notna()
    if invalid.any() or (parsed.isna().any() and not allow_missing):
        examples = values.loc[invalid | parsed.isna()].head(5).tolist()
        raise ValueError(f"Could not parse G2F planting dates: {examples}")
    return parsed.dt.normalize()


def _load_hybrid_genotypes(path: Path) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], int]:
    # The first '<Numeric>' row is a format marker; the second row is the header.
    frame = pd.read_csv(path, sep="\t", skiprows=1, na_values=("NA", "NaN", "."))
    identifier = frame.columns[0]
    frame = frame.rename(columns={identifier: "Hybrid"})
    frame["Hybrid"] = frame["Hybrid"].astype(str)
    if frame["Hybrid"].duplicated().any():
        raise ValueError("G2F hybrid genotype matrix has duplicate hybrid identifiers")

    markers = frame.columns[1:]
    matrix = frame.loc[:, markers].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    missing_values = int(np.isnan(matrix).sum())
    if missing_values:
        # Marker-only imputation uses no phenotype information.  It is necessary
        # because PCA cannot accept missing calls and applies equally to 2024 hybrids.
        means = np.nanmean(matrix, axis=0)
        if np.isnan(means).any():
            raise ValueError("A G2F marker is missing for every hybrid")
        matrix = np.where(np.isnan(matrix), means[None, :], matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("G2F genotype matrix contains non-finite values after imputation")
    return tuple(frame["Hybrid"]), matrix, tuple(markers), missing_values


def _vpd_kpa(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    saturation = 0.6108 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual = 0.6108 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3))
    return np.maximum(0.0, saturation - actual)


def _prepare_weather(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {
        "Env", "Date", "T2M_MIN", "T2M_MAX", "T2M", "RH2M", "PRECTOTCORR",
        "ALLSKY_SFC_PAR_TOT", "GWETROOT", "WS2M", "T2MDEW",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source.name} is missing daily weather columns: {missing}")
    result = frame.copy()
    result["Env"] = result["Env"].astype(str)
    result["Date"] = pd.to_datetime(result["Date"].astype(str), format="%Y%m%d", errors="coerce")
    if result["Date"].isna().any():
        raise ValueError(f"{source.name} contains invalid daily dates")
    for column in required - {"Env", "Date"}:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["VPD_KPA"] = _vpd_kpa(result["T2M"].to_numpy(float), result["T2MDEW"].to_numpy(float))
    result = result.sort_values(["Env", "Date"]).drop_duplicates(["Env", "Date"], keep="last")
    return result


def _stage_mean(values: np.ndarray, start: int, end: int) -> float:
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end < start:
        return float("nan")
    selected = values[start : end + 1]
    return float(np.nanmean(selected)) if np.isfinite(selected).sum() >= 3 else float("nan")


def _exposure_vector(weather: pd.DataFrame, pollen_dap: float, silk_dap: float) -> np.ndarray:
    """Construct transparent stage-aligned labels from observed training phenology.

    Early spans planting to anthesis; flowering brackets pollen shedding and
    silking; grain filling is the following 50 days.  These labels are used only
    for 2014--2023 supervision.  At 2024 deployment they are predicted from
    genotype and daily environment.
    """

    pollen = int(round(float(pollen_dap)))
    silk = int(round(float(silk_dap)))
    if pollen < 1 or silk < pollen:
        return np.full(len(EXPOSURE_COLUMNS), np.nan)
    columns = {
        "night": weather["T2M_MIN"].to_numpy(float),
        "radiation": weather["ALLSKY_SFC_PAR_TOT"].to_numpy(float),
        "moisture": weather["GWETROOT"].to_numpy(float),
        "tmax": weather["T2M_MAX"].to_numpy(float),
    }
    early = (0, pollen - 1)
    flowering = (pollen - 3, silk + 3)
    filling = (silk + 4, silk + 53)
    return np.asarray(
        [
            _stage_mean(columns["night"], *early),
            _stage_mean(columns["night"], *flowering),
            _stage_mean(columns["night"], *filling),
            _stage_mean(columns["radiation"], *early),
            _stage_mean(columns["radiation"], *flowering),
            _stage_mean(columns["radiation"], *filling),
            _stage_mean(columns["moisture"], *flowering),
            _stage_mean(columns["moisture"], *filling),
            _stage_mean(columns["tmax"], *early),
            _stage_mean(columns["tmax"], *flowering),
            _stage_mean(columns["tmax"], *filling),
        ],
        dtype=np.float32,
    )


def _environment_sequences(
    weather: pd.DataFrame,
    planting: pd.Series,
    sequence_length: int | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    environment_ids = tuple(sorted(planting.index.astype(str)))
    source_rows: list[pd.DataFrame] = []
    for environment in environment_ids:
        rows = weather.loc[weather["Env"].eq(environment)].copy()
        rows = rows.loc[rows["Date"].ge(planting.loc[environment])].sort_values("Date")
        if sequence_length is not None and len(rows) < sequence_length:
            raise ValueError(
                f"{environment} has {len(rows)} post-planting weather days; "
                f"{sequence_length} are required"
            )
        source_rows.append(rows.iloc[:sequence_length] if sequence_length is not None else rows)

    target_length = sequence_length or max(len(rows) for rows in source_rows)
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    for rows in source_rows:
        values = rows.loc[:, G2F_DAILY_COLUMNS].to_numpy(np.float32)
        padded = np.full((target_length, len(G2F_DAILY_COLUMNS)), np.nan, dtype=np.float32)
        padded[: len(values)] = values
        sequences.append(padded)
        lengths.append(len(values))
    return environment_ids, np.stack(sequences), np.asarray(lengths, dtype=np.int32)


def load_g2f_2024_data(data_dir: str | Path, sequence_length: int | None = 100) -> G2FPreparedData:
    """Load the official G2F archive and create a G2F-specific GPACEData object."""

    if sequence_length is not None and sequence_length < 1:
        raise ValueError("sequence_length must be positive or None for complete post-planting weather")
    root = Path(data_dir).resolve()
    paths = _require_files(root)
    train_traits = pd.read_csv(paths["train_traits"])
    train_meta = pd.read_csv(paths["train_meta"])
    test_template = pd.read_csv(paths["test_template"])
    test_meta = pd.read_csv(paths["test_meta"])
    test_observed = pd.read_csv(paths["test_observed"])
    train_weather = _prepare_weather(pd.read_csv(paths["train_weather"]), paths["train_weather"])
    test_weather = _prepare_weather(pd.read_csv(paths["test_weather"]), paths["test_weather"])

    required_trait = {"Env", "Year", "Hybrid", "Date_Planted", "Pollen_DAP_days", "Silk_DAP_days", "Yield_Mg_ha"}
    if missing := sorted(required_trait - set(train_traits.columns)):
        raise ValueError(f"{paths['train_traits'].name} is missing columns: {missing}")
    for frame, source in ((test_template, paths["test_template"]), (test_observed, paths["test_observed"])):
        if missing := sorted({"Env", "Hybrid", "Yield_Mg_ha"} - set(frame.columns)):
            raise ValueError(f"{source.name} is missing columns: {missing}")
    if "Date_Planted" not in test_meta.columns:
        raise ValueError(f"{paths['test_meta'].name} is missing Date_Planted")

    train_traits = train_traits.loc[:, list(required_trait)].copy()
    train_traits["Hybrid"] = train_traits["Hybrid"].astype(str)
    train_traits["Env"] = train_traits["Env"].astype(str)
    for column in ("Yield_Mg_ha", "Pollen_DAP_days", "Silk_DAP_days"):
        train_traits[column] = pd.to_numeric(train_traits[column], errors="coerce")
    train_traits["Date_Planted"] = _parse_g2f_date(
        train_traits["Date_Planted"], allow_missing=True
    )
    training_rows_before_date_filter = int(len(train_traits))
    train_traits = train_traits.dropna(
        subset=["Yield_Mg_ha", "Pollen_DAP_days", "Silk_DAP_days", "Date_Planted"]
    )

    training = (
        train_traits.groupby(["Env", "Hybrid"], as_index=False)
        .agg(
            Year=("Year", "first"),
            Date_Planted=("Date_Planted", "first"),
            **{
                "grain.yield": ("Yield_Mg_ha", "mean"),
                "pollen_dap": ("Pollen_DAP_days", "mean"),
                "silk_dap": ("Silk_DAP_days", "mean"),
            },
        )
    )
    planting_train = training.groupby("Env")["Date_Planted"].first()

    test_meta = test_meta.loc[:, ["Env", "Date_Planted"]].copy()
    test_meta["Env"] = test_meta["Env"].astype(str)
    test_meta["Date_Planted"] = _parse_g2f_date(test_meta["Date_Planted"])
    if test_meta["Env"].duplicated().any():
        raise ValueError("Testing metadata has duplicate environment identifiers")
    observed = test_observed.loc[:, ["Env", "Hybrid", "Yield_Mg_ha"]].copy()
    observed["Env"] = observed["Env"].astype(str)
    observed["Hybrid"] = observed["Hybrid"].astype(str)
    observed["grain.yield"] = pd.to_numeric(observed["Yield_Mg_ha"], errors="coerce")
    test = test_template.loc[:, ["Env", "Hybrid"]].copy()
    test["Env"] = test["Env"].astype(str)
    test["Hybrid"] = test["Hybrid"].astype(str)
    test = test.merge(observed.loc[:, ["Env", "Hybrid", "grain.yield"]], on=["Env", "Hybrid"], how="left", validate="one_to_one")
    test = test.merge(test_meta, on="Env", how="left", validate="many_to_one")
    test = test.dropna(subset=["grain.yield"]).copy()
    test["Year"] = 2024

    weather = pd.concat([train_weather, test_weather], ignore_index=True)
    weather_envs = set(weather["Env"])
    missing_training_weather = sorted(set(training["Env"]) - weather_envs)
    training = training.loc[training["Env"].isin(weather_envs)].copy()
    missing_test_weather = sorted(set(test["Env"]) - weather_envs)
    if missing_test_weather:
        raise ValueError(f"2024 labeled records lack daily weather: {missing_test_weather}")

    planting = pd.concat(
        [training.groupby("Env")["Date_Planted"].first(), test.groupby("Env")["Date_Planted"].first()]
    )
    planting = planting.loc[~planting.index.duplicated(keep="last")]
    environment_ids, sequences, lengths = _environment_sequences(weather, planting, sequence_length)

    weather_by_environment = {
        environment: group.loc[group["Date"].ge(planting.loc[environment])].sort_values("Date").reset_index(drop=True)
        for environment, group in weather.groupby("Env", sort=False)
        if environment in set(training["Env"])
    }
    exposures = np.vstack(
        [
            _exposure_vector(weather_by_environment[row.Env], row.pollen_dap, row.silk_dap)
            for row in training.itertuples(index=False)
        ]
    )
    complete_exposure = np.isfinite(exposures).all(axis=1)
    excluded_exposure_rows = int((~complete_exposure).sum())
    training = training.loc[complete_exposure].reset_index(drop=True)
    exposures = exposures[complete_exposure]
    for position, column in enumerate(EXPOSURE_COLUMNS):
        training[column] = exposures[:, position]

    test = test.rename(columns={"Env": "Experiment", "Hybrid": "Variety_ID"})
    training = training.rename(columns={"Env": "Experiment", "Hybrid": "Variety_ID"})
    training["scenario"] = "training"
    test["scenario"] = "external_2024"
    for column in EXPOSURE_COLUMNS:
        test[column] = np.nan
    records = pd.concat([training, test], ignore_index=True, sort=False)
    records["Experiment"] = records["Experiment"].astype(str)
    records["Variety_ID"] = records["Variety_ID"].astype(str)
    records["source_yield_type"] = np.where(
        records["scenario"].eq("training"), "plot_mean", "official_2024_observed"
    )
    records["source_exposure_type"] = np.where(records["scenario"].eq("training"), "training_observed_phenology", "deployment_unobserved")

    genotype_ids, genotype_matrix, marker_names, genotype_missing = _load_hybrid_genotypes(paths["genotypes"])
    lookup = {identifier: index for index, identifier in enumerate(genotype_ids)}
    missing_genotypes = sorted(set(records["Variety_ID"]) - set(lookup))
    external_missing_genotypes = sorted(
        set(records.loc[records["scenario"].eq("external_2024"), "Variety_ID"]) - set(lookup)
    )
    if external_missing_genotypes:
        raise ValueError(
            "Official 2024 evaluation records without hybrid genotypes: "
            f"{external_missing_genotypes[:10]}"
        )
    training_missing_genotypes = int(
        records["scenario"].eq("training").mul(~records["Variety_ID"].isin(lookup)).sum()
    )
    records = records.loc[records["Variety_ID"].isin(lookup)].copy()
    environment_lookup = {identifier: index for index, identifier in enumerate(environment_ids)}
    records["genotype_index"] = records["Variety_ID"].map(lookup).astype(int)
    records["environment_index"] = records["Experiment"].map(environment_lookup).astype(int)
    if records["environment_index"].isna().any():
        raise RuntimeError("G2F records could not be mapped to environment sequences")
    records = records.sort_values(["scenario", "Year", "Experiment", "Variety_ID"]).reset_index(drop=True)
    records["record_id"] = np.arange(len(records), dtype=int)
    training_indices = records.index[records["scenario"].eq("training")].to_numpy(dtype=int)
    external_indices = records.index[records["scenario"].eq("external_2024")].to_numpy(dtype=int)

    audit = {
        "data_dir": str(root),
        "training_plot_records": int(len(train_traits)),
        "training_plot_records_without_planting_date_excluded": int(
            training_rows_before_date_filter - len(train_traits)
        ),
        "training_gxe_means": int(len(training_indices)),
        "external_2024_labeled_records": int(len(external_indices)),
        "training_environment_count": int(records.loc[training_indices, "Experiment"].nunique()),
        "external_2024_environment_count": int(records.loc[external_indices, "Experiment"].nunique()),
        "external_2024_submission_pairs_without_label": int(len(test_template) - len(external_indices)),
        "training_environments_without_weather_excluded": missing_training_weather,
        "training_rows_without_complete_stage_exposure_excluded": excluded_exposure_rows,
        "training_rows_without_hybrid_genotype_excluded": training_missing_genotypes,
        "genotype_missing_calls_imputed": genotype_missing,
        "genotype_count": len(genotype_ids),
        "marker_count": len(marker_names),
        "environment_count": len(environment_ids),
        "sequence_shape": list(sequences.shape[1:]),
        "sequence_length_requested": sequence_length if sequence_length is not None else "full",
        "sequence_length_min": int(lengths.min()),
        "sequence_length_max": int(lengths.max()),
        "stage_label_definition": "early=planting-to-pollen; flowering=pollen-3-to-silk+3; filling=silk+4-to-silk+53",
    }
    data = GPACEData(
        records=records,
        genotype_ids=genotype_ids,
        genotype_matrix=genotype_matrix,
        marker_names=marker_names,
        environment_ids=environment_ids,
        environment_sequences=sequences,
        environment_sequence_lengths=lengths,
        environment_feature_names=G2F_DAILY_COLUMNS,
        genotype_partitions=pd.DataFrame(),
        environment_partitions=pd.DataFrame(),
        audit=audit,
    )
    return G2FPreparedData(data, training_indices, external_indices)
