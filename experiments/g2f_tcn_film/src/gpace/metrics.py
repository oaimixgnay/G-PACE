from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) == 0 or np.std(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    pcc = _correlation(observed, predicted)
    r2 = float(r2_score(observed, predicted)) if len(observed) >= 2 else float("nan")
    mae = float(mean_absolute_error(observed, predicted))
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    if np.std(predicted) == 0:
        calibration_slope = float("nan")
        calibration_intercept = float("nan")
    else:
        calibration_slope, calibration_intercept = np.polyfit(predicted, observed, 1)

    within_pcc = []
    within_spearman = []
    top_recall = []
    for _, group in frame.groupby("Experiment"):
        y = group["observed"].to_numpy(dtype=float)
        prediction = group["predicted"].to_numpy(dtype=float)
        correlation = _correlation(y, prediction)
        if np.isfinite(correlation):
            within_pcc.append(np.clip(correlation, -0.999999, 0.999999))
        spearman = _correlation(rankdata(y), rankdata(prediction))
        if np.isfinite(spearman):
            within_spearman.append(spearman)
        count = max(1, int(np.ceil(0.2 * len(group))))
        observed_top = set(group.iloc[np.argsort(y)[-count:]]["Variety_ID"])
        predicted_top = set(group.iloc[np.argsort(prediction)[-count:]]["Variety_ID"])
        top_recall.append(len(observed_top & predicted_top) / count)

    mean_within_pcc = (
        float(np.tanh(np.mean(np.arctanh(within_pcc)))) if within_pcc else float("nan")
    )
    return {
        "n": float(len(frame)),
        "environment_count": float(frame["Experiment"].nunique()),
        "pcc": pcc,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "calibration_slope": float(calibration_slope),
        "calibration_intercept": float(calibration_intercept),
        "within_environment_pcc": mean_within_pcc,
        "within_environment_spearman": (
            float(np.mean(within_spearman)) if within_spearman else float("nan")
        ),
        "top20_recall": float(np.mean(top_recall)) if top_recall else float("nan"),
    }


def cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    repeats: int = 500,
    seed: int = 2026,
) -> pd.DataFrame:
    point = regression_metrics(frame)
    if repeats <= 0:
        return pd.DataFrame(
            [
                {"metric": key, "estimate": value, "ci_low": np.nan, "ci_high": np.nan}
                for key, value in point.items()
            ]
        )
    rng = np.random.default_rng(seed)
    environments = frame["Experiment"].drop_duplicates().to_numpy()
    samples: dict[str, list[float]] = {key: [] for key in point}
    grouped = {name: group for name, group in frame.groupby("Experiment")}

    for _ in range(repeats):
        selected = rng.choice(environments, size=len(environments), replace=True)
        chunks = []
        for bootstrap_environment, environment in enumerate(selected):
            group = grouped[environment]
            positions = rng.integers(0, len(group), size=len(group))
            sampled = group.iloc[positions].copy()
            sampled["Experiment"] = f"{environment}__boot{bootstrap_environment}"
            chunks.append(sampled)
        values = regression_metrics(pd.concat(chunks, ignore_index=True))
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)

    rows = []
    for key, estimate in point.items():
        distribution = samples[key]
        low, high = np.percentile(distribution, [2.5, 97.5]) if distribution else (np.nan, np.nan)
        rows.append({"metric": key, "estimate": estimate, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def exposure_metrics(observed: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    rows = []
    for column, name in enumerate(
        (
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
    ):
        y = observed[:, column]
        yhat = predicted[:, column]
        rows.append(
            {
                "exposure": name,
                "pcc": _correlation(y, yhat),
                "mae": float(mean_absolute_error(y, yhat)),
                "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            }
        )
    return pd.DataFrame(rows)
