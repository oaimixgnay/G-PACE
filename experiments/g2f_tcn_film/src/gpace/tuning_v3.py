from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import GPACEData, load_gpace_data
from .features import ProjectedInputs, interaction_design, phenology_design
from .metrics import regression_metrics
from .models import (
    EnvironmentMeanRegressor,
    YieldHeadRegressor,
    crossfit_exposures,
    make_exposure_model,
)
from .splits import make_inner_splits, make_outer_split


SCENARIO_WEIGHTS = {"cv0": 0.4, "cv1": 0.2, "cv00": 0.4}


@dataclass(frozen=True)
class V3SearchConfig:
    data_dir: str
    output_dir: str
    base_config: str
    sequence_length: int | None = None
    selection_split: int = 1
    inner_repeats: int = 3
    inner_holdout_fraction: float = 0.2
    trials: int = 100
    crossfit_folds: int = 3
    yield_genotype_components: tuple[int, ...] = (16, 32, 64, 128)
    seed: int = 2026
    device: str = "auto"


@dataclass
class _FeatureSet:
    x_direct_train: np.ndarray
    x_phenology_train: np.ndarray
    x_environment_train: np.ndarray
    train_genotype_indices: np.ndarray
    evaluation: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ]


@dataclass
class _V3Cache:
    repeat: int
    train_experiments: np.ndarray
    y_train: np.ndarray
    genomic_eigenvalues: np.ndarray
    genomic_eigenvectors: np.ndarray
    genomic_cross_all: np.ndarray
    genomic_target: np.ndarray
    genomic_row_positions: np.ndarray
    by_components: dict[int, _FeatureSet]


def _resolved_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _weighted(values: dict[str, float]) -> float:
    denominator = sum(SCENARIO_WEIGHTS[name] for name in values)
    return float(
        sum(SCENARIO_WEIGHTS[name] * value for name, value in values.items())
        / denominator
    )


def _genomic_terms(
    data: GPACEData,
    train_genotype_indices: np.ndarray,
    residual_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_genotypes, row_positions = np.unique(
        train_genotype_indices, return_inverse=True
    )
    matrix = np.asarray(data.genotype_matrix[unique_genotypes], dtype=np.float64)
    frequency = matrix.mean(axis=0) / 2.0
    scale = np.sqrt(2.0 * frequency * (1.0 - frequency))
    valid = scale > 1e-8
    center = 2.0 * frequency[valid]
    marker_scale = scale[valid]
    train_standardized = ((matrix[:, valid] - center) / marker_scale).astype(
        np.float32
    )
    all_standardized = (
        (np.asarray(data.genotype_matrix[:, valid], dtype=np.float64) - center)
        / marker_scale
    ).astype(np.float32)
    marker_count = max(1, train_standardized.shape[1])
    relationship = (train_standardized @ train_standardized.T) / marker_count
    cross_all = (all_standardized @ train_standardized.T) / marker_count
    eigenvalues, eigenvectors = np.linalg.eigh(relationship.astype(np.float64))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    genotype_target = (
        pd.Series(residual_target)
        .groupby(train_genotype_indices, sort=True)
        .mean()
        .reindex(unique_genotypes)
        .to_numpy(dtype=np.float64)
    )
    return (
        eigenvalues,
        eigenvectors,
        cross_all.astype(np.float32),
        genotype_target,
        row_positions,
    )


def _make_cache(
    data: GPACEData,
    inner_splits,
    stage_model: str,
    stage_params: dict[str, Any],
    environment_components: int,
    config: V3SearchConfig,
) -> list[_V3Cache]:
    caches: list[_V3Cache] = []
    for split in inner_splits:
        print(f"Preparing V3 features for inner repeat {split.repeat + 1}")
        training_exposures = crossfit_exposures(
            data,
            split.train,
            kind=stage_model,
            folds=config.crossfit_folds,
            random_state=config.seed + 10000 + split.repeat * 100,
            model_kwargs=stage_params,
        )
        final_stage = make_exposure_model(
            stage_model,
            random_state=config.seed + 20000 + split.repeat,
            **stage_params,
        ).fit(data, split.train)
        predicted_exposures = {
            name: final_stage.predict(data, indices)
            for name, indices in split.evaluation_sets().items()
        }
        y_train = data.records.loc[split.train, "grain.yield"].to_numpy(float)
        train_experiments = data.records.loc[split.train, "Experiment"].to_numpy()
        train_genotype_indices = data.records.loc[
            split.train, "genotype_index"
        ].to_numpy(dtype=int)
        environment_means = (
            pd.Series(y_train)
            .groupby(train_experiments, sort=False)
            .transform("mean")
            .to_numpy(dtype=np.float64)
        )
        genomic_terms = _genomic_terms(
            data, train_genotype_indices, y_train - environment_means
        )
        by_components: dict[int, _FeatureSet] = {}

        for components in config.yield_genotype_components:
            inputs = ProjectedInputs(
                genotype_components=components,
                environment_components=environment_components,
                random_state=config.seed + split.repeat,
            ).fit(data, split.train)
            train_genotype, train_environment = inputs.rows(data, split.train)
            direct_scaler = StandardScaler().fit(
                interaction_design(train_genotype, train_environment)
            )
            exposure_scaler = StandardScaler().fit(training_exposures)
            standardized_exposure = exposure_scaler.transform(training_exposures)
            phenology_scaler = StandardScaler().fit(
                phenology_design(train_genotype, standardized_exposure)
            )
            x_direct_train = direct_scaler.transform(
                interaction_design(train_genotype, train_environment)
            ).astype(np.float32)
            x_phenology_train = phenology_scaler.transform(
                phenology_design(train_genotype, standardized_exposure)
            ).astype(np.float32)
            evaluation = {}
            for scenario, indices in split.evaluation_sets().items():
                genotype, environment = inputs.rows(data, indices)
                x_direct = direct_scaler.transform(
                    interaction_design(genotype, environment)
                ).astype(np.float32)
                x_phenology = phenology_scaler.transform(
                    phenology_design(
                        genotype,
                        exposure_scaler.transform(predicted_exposures[scenario]),
                    )
                ).astype(np.float32)
                genotype_indices = data.records.loc[
                    indices, "genotype_index"
                ].to_numpy(dtype=int)
                frame = data.records.loc[
                    indices, ["Experiment", "Variety_ID", "grain.yield"]
                ].copy()
                frame = frame.rename(columns={"grain.yield": "observed"})
                evaluation[scenario] = (
                    x_direct,
                    x_phenology,
                    environment.astype(np.float32),
                    genotype_indices,
                    frame,
                )
            by_components[int(components)] = _FeatureSet(
                x_direct_train=x_direct_train,
                x_phenology_train=x_phenology_train,
                x_environment_train=train_environment.astype(np.float32),
                train_genotype_indices=train_genotype_indices,
                evaluation=evaluation,
            )
        caches.append(
            _V3Cache(
                repeat=split.repeat,
                train_experiments=train_experiments,
                y_train=y_train,
                genomic_eigenvalues=genomic_terms[0],
                genomic_eigenvectors=genomic_terms[1],
                genomic_cross_all=genomic_terms[2],
                genomic_target=genomic_terms[3],
                genomic_row_positions=genomic_terms[4],
                by_components=by_components,
            )
        )
        del final_stage
        _release_accelerator_memory()
    return caches


def _v3_parameters(trial, device: str, components: tuple[int, ...]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "yield_genotype_components": trial.suggest_categorical(
            "yield_genotype_components", list(components)
        ),
        "iterations": trial.suggest_int("catboost_iterations", 600, 1400, step=200),
        "depth": trial.suggest_int("catboost_depth", 4, 7),
        "learning_rate": trial.suggest_float(
            "catboost_learning_rate", 0.015, 0.10, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "catboost_l2_leaf_reg", 0.3, 20.0, log=True
        ),
        "random_strength": trial.suggest_float(
            "catboost_random_strength", 0.0, 1.5
        ),
        "task_type": "GPU" if device.startswith("cuda") else "CPU",
        "phenology_weight": trial.suggest_categorical(
            "phenology_weight", [0.0, 0.03, 0.1, 0.3]
        ),
        "residual_weight": trial.suggest_categorical(
            "residual_weight", [0.5, 0.75, 1.0]
        ),
        "mean_head": trial.suggest_categorical("mean_head", ["ridge", "krr"]),
        "mean_alpha": trial.suggest_float("mean_alpha", 1e-3, 1e3, log=True),
        "mean_calibration_strength": trial.suggest_categorical(
            "mean_calibration_strength", [0.0, 0.5, 1.0]
        ),
        "gblup_weight": trial.suggest_categorical(
            "gblup_weight", [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
    }
    if params["mean_head"] == "krr":
        params["mean_gamma"] = trial.suggest_float(
            "mean_gamma", 1e-4, 1.0, log=True
        )
    if params["gblup_weight"] > 0.0:
        params["gblup_alpha"] = trial.suggest_float(
            "gblup_alpha", 1e-3, 1e2, log=True
        )
    return params


def _selection_score(scores: dict[str, float], target_scale: float) -> float:
    pcc = scores["pcc"] if np.isfinite(scores["pcc"]) else -1.0
    within_pcc = (
        scores["within_environment_pcc"]
        if np.isfinite(scores["within_environment_pcc"])
        else -1.0
    )
    within_spearman = (
        scores["within_environment_spearman"]
        if np.isfinite(scores["within_environment_spearman"])
        else -1.0
    )
    r2 = scores["r2"] if np.isfinite(scores["r2"]) else -1.0
    r2 = float(np.clip(r2, -1.0, 1.0))
    relative_rmse = scores["rmse"] / max(target_scale, 1e-8)
    rmse_quality = 1.0 / (1.0 + relative_rmse)
    return float(
        0.20 * pcc
        + 0.25 * within_pcc
        + 0.15 * within_spearman
        + 0.20 * r2
        + 0.20 * rmse_quality
    )


def _gblup_predictions(
    cache: _V3Cache, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues = cache.genomic_eigenvalues
    eigenvectors = cache.genomic_eigenvectors
    target = cache.genomic_target
    inverse = 1.0 / (eigenvalues + alpha)
    projected_target = eigenvectors.T @ target
    dual = eigenvectors @ (inverse * projected_target)
    smoother_values = eigenvalues * inverse
    fitted = eigenvectors @ (smoother_values * projected_target)
    leverage = (eigenvectors**2) @ smoother_values
    loo = (fitted - leverage * target) / np.maximum(1e-6, 1.0 - leverage)
    train_prediction = loo[cache.genomic_row_positions]
    all_prediction = np.asarray(cache.genomic_cross_all @ dual, dtype=np.float64)
    return train_prediction, all_prediction


def _objective(data: GPACEData, caches: list[_V3Cache], config: V3SearchConfig):
    device = _resolved_device(config.device)

    def objective(trial) -> float:
        params = _v3_parameters(
            trial, device, config.yield_genotype_components
        )
        components = int(params["yield_genotype_components"])
        phenology_weight = float(params["phenology_weight"])
        residual_weight = float(params["residual_weight"])
        gblup_weight = float(params["gblup_weight"])
        head_params = {
            key: value
            for key, value in params.items()
            if key
            not in {
                "yield_genotype_components",
                "phenology_weight",
                "residual_weight",
                "mean_head",
                "mean_alpha",
                "mean_gamma",
                "mean_calibration_strength",
                "gblup_weight",
                "gblup_alpha",
            }
        }
        repeat_scores = []
        for cache in caches:
            features = cache.by_components[components]
            environment_means = (
                pd.Series(cache.y_train)
                .groupby(cache.train_experiments, sort=False)
                .transform("mean")
                .to_numpy(dtype=np.float64)
            )
            unique_environment_positions = (
                pd.Series(cache.train_experiments)
                .drop_duplicates(keep="first")
                .index.to_numpy(dtype=int)
            )
            mean_model = EnvironmentMeanRegressor(
                kind=str(params["mean_head"]),
                alpha=float(params["mean_alpha"]),
                gamma=params.get("mean_gamma"),
                calibration_strength=float(params["mean_calibration_strength"]),
            ).fit(
                features.x_environment_train[unique_environment_positions],
                environment_means[unique_environment_positions],
            )
            residual_target = cache.y_train - environment_means
            genomic_all = None
            if gblup_weight > 0.0:
                genomic_train, genomic_all = _gblup_predictions(
                    cache, float(params["gblup_alpha"])
                )
                residual_target = residual_target - gblup_weight * genomic_train
            x_train = np.concatenate(
                [
                    features.x_direct_train,
                    phenology_weight * features.x_phenology_train,
                ],
                axis=1,
            )
            residual_model = YieldHeadRegressor(
                "catboost",
                params=head_params,
                random_state=config.seed + trial.number * 100 + cache.repeat,
                device=device,
                scale_features=False,
            ).fit(x_train, residual_target)
            target_scale = float(np.std(cache.y_train))
            scenario_scores = {}
            for scenario, (
                x_direct,
                x_phenology,
                x_environment,
                genotype_indices,
                source,
            ) in features.evaluation.items():
                prediction = mean_model.predict(x_environment)
                prediction += residual_weight * residual_model.predict(
                    np.concatenate(
                        [x_direct, phenology_weight * x_phenology], axis=1
                    )
                )
                if genomic_all is not None:
                    prediction += gblup_weight * genomic_all[genotype_indices]
                frame = source.copy()
                frame["predicted"] = prediction
                scenario_scores[scenario] = _selection_score(
                    regression_metrics(frame), target_scale
                )
            weighted_score = _weighted(scenario_scores)
            repeat_scores.append(weighted_score)
            trial.set_user_attr(
                f"repeat_{cache.repeat}_weighted_v3_score", weighted_score
            )
        return float(np.mean(repeat_scores))

    return objective


def run_v3_search(config: V3SearchConfig) -> dict[str, Path]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for V3 search") from exc

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(Path(config.base_config).read_text(encoding="utf-8"))
    stage_model = str(base_config["stage_model"])
    stage_params = dict(base_config["stage_params"])
    stage_params["device"] = _resolved_device(config.device)
    environment_components = int(base_config.get("environment_components", 12))

    data = load_gpace_data(config.data_dir, sequence_length=config.sequence_length)
    selection_train = make_outer_split(data, config.selection_split).train
    inner_splits = make_inner_splits(
        data,
        selection_train,
        repeats=config.inner_repeats,
        holdout_fraction=config.inner_holdout_fraction,
        seed=config.seed,
    )
    caches = _make_cache(
        data,
        inner_splits,
        stage_model,
        stage_params,
        environment_components,
        config,
    )
    storage = f"sqlite:///{(output_dir / 'optuna_v3.db').as_posix()}"
    study = optuna.create_study(
        study_name="gpace_yield_v3",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=config.seed + 3, multivariate=True
        ),
    )
    completed = sum(trial.state.name == "COMPLETE" for trial in study.trials)
    remaining = max(0, config.trials - completed)
    if remaining:
        study.optimize(
            _objective(data, caches, config),
            n_trials=remaining,
            n_jobs=1,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError),
        )
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError("No V3 yield trial completed successfully") from exc
    best_params = _v3_parameters(
        best_trial,
        _resolved_device(config.device),
        config.yield_genotype_components,
    )
    yield_genotype_components = int(
        best_params.pop("yield_genotype_components")
    )
    best_config = {
        "stage_model": stage_model,
        "stage_params": stage_params,
        "sequence_length": config.sequence_length,
        "environment_components": environment_components,
        "yield_mode": "v3_mean_residual",
        "yield_genotype_components": yield_genotype_components,
        "yield_head": "catboost",
        "yield_params": best_params,
        "selection": {
            "outer_split_training_block": config.selection_split,
            "inner_repeats": config.inner_repeats,
            "scenario_weights": SCENARIO_WEIGHTS,
            "objective": (
                "weighted composite of PCC, within-environment PCC, "
                "within-environment Spearman, R2 and relative RMSE"
            ),
            "best_value": best_trial.value,
            "external_data_used": False,
            "base_stage_config": str(Path(config.base_config).resolve()),
        },
    }
    paths = {
        "best_config": output_dir / "best_config_v3.json",
        "trials": output_dir / "v3_trials.csv",
        "summary": output_dir / "search_summary_v3.json",
        "database": output_dir / "optuna_v3.db",
    }
    paths["best_config"].write_text(
        json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(paths["trials"], index=False)
    summary = {
        "search_config": asdict(config),
        "data_audit": data.audit,
        "inner_split_sizes": [
            {
                "repeat": split.repeat,
                "train": len(split.train),
                "cv0": len(split.cv0),
                "cv1": len(split.cv1),
                "cv00": len(split.cv00),
            }
            for split in inner_splits
        ],
        "best_config": best_config,
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths
