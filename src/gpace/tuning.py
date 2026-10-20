from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import GPACEData, EXPOSURE_COLUMNS, load_gpace_data
from .features import ProjectedInputs, interaction_design, phenology_design
from .metrics import regression_metrics
from .models import YieldHeadRegressor, crossfit_exposures, make_exposure_model
from .splits import InnerSplit, make_inner_splits, make_outer_split


SCENARIO_WEIGHTS = {"cv0": 0.4, "cv1": 0.2, "cv00": 0.4}


@dataclass(frozen=True)
class SearchConfig:
    data_dir: str
    output_dir: str
    sequence_length: int | None = 100
    selection_split: int = 1
    inner_repeats: int = 2
    inner_holdout_fraction: float = 0.2
    stage_trials: int = 30
    yield_trials: int = 60
    stage_model: str = "tcn_film"
    yield_heads: tuple[str, ...] = ("ridge", "krr", "catboost")
    crossfit_folds: int = 3
    environment_components: int = 12
    tcn_epochs: int = 120
    tcn_patience: int = 15
    seed: int = 2026
    device: str = "auto"


@dataclass
class _YieldCache:
    repeat: int
    x_direct_train: np.ndarray
    x_phenology_train: np.ndarray
    x_environment_train: np.ndarray
    train_experiments: np.ndarray
    y_train: np.ndarray
    evaluation: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]


def _resolved_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _storage_url(output_dir: Path) -> str:
    return f"sqlite:///{(output_dir / 'optuna.db').resolve().as_posix()}"


def _weighted(values: dict[str, float]) -> float:
    usable = {
        name: value
        for name, value in values.items()
        if name in SCENARIO_WEIGHTS and np.isfinite(value)
    }
    if not usable:
        return float("nan")
    denominator = sum(SCENARIO_WEIGHTS[name] for name in usable)
    return float(
        sum(SCENARIO_WEIGHTS[name] * value for name, value in usable.items())
        / denominator
    )


def _stage_parameters(trial, config: SearchConfig) -> dict[str, Any]:
    if config.stage_model == "ridge":
        return {
            "genotype_components": trial.suggest_categorical(
                "genotype_components", [16, 32, 64]
            ),
            "environment_components": trial.suggest_int(
                "environment_components", 6, 16, step=2
            ),
            "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        }
    return {
        "genotype_components": trial.suggest_categorical(
            "genotype_components", [16, 32, 64]
        ),
        "width": trial.suggest_categorical("width", [32, 64, 128]),
        "depth": trial.suggest_int("depth", 2, 4),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": config.tcn_epochs,
        "patience": config.tcn_patience,
        "validation_fraction": 0.15,
        "amp": True,
        "device": _resolved_device(config.device),
    }


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stage_objective(data: GPACEData, inner_splits: tuple[InnerSplit, ...], config):
    def objective(trial) -> float:
        params = _stage_parameters(trial, config)
        repeat_scores = []
        for split in inner_splits:
            model = make_exposure_model(
                config.stage_model,
                random_state=config.seed + trial.number * 100 + split.repeat,
                **params,
            ).fit(data, split.train)
            train_target = data.records.loc[split.train, EXPOSURE_COLUMNS].to_numpy(float)
            scale = train_target.std(axis=0)
            scale[scale < 1e-8] = 1.0
            scenario_scores = {}
            for scenario, indices in split.evaluation_sets().items():
                observed = data.records.loc[indices, EXPOSURE_COLUMNS].to_numpy(float)
                predicted = model.predict(data, indices)
                scenario_scores[scenario] = float(
                    np.sqrt(np.mean(((predicted - observed) / scale[None, :]) ** 2))
                )
            repeat_scores.append(_weighted(scenario_scores))
            trial.set_user_attr(
                f"repeat_{split.repeat}_normalized_rmse", _weighted(scenario_scores)
            )
            del model
            _release_accelerator_memory()
        return float(np.mean(repeat_scores))

    return objective


def _make_yield_cache(
    data: GPACEData,
    inner_splits: tuple[InnerSplit, ...],
    stage_model: str,
    stage_params: dict[str, Any],
    config: SearchConfig,
) -> list[_YieldCache]:
    caches = []
    genotype_components = int(stage_params["genotype_components"])
    for split in inner_splits:
        print(f"Preparing leakage-free yield features for inner repeat {split.repeat + 1}")
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

        inputs = ProjectedInputs(
            genotype_components=genotype_components,
            environment_components=config.environment_components,
            random_state=config.seed + split.repeat,
        ).fit(data, split.train)
        exposure_scaler = StandardScaler().fit(training_exposures)
        train_genotype, train_environment = inputs.rows(data, split.train)
        direct_scaler = StandardScaler().fit(
            interaction_design(train_genotype, train_environment)
        )
        standardized_training_exposure = exposure_scaler.transform(training_exposures)
        phenology_scaler = StandardScaler().fit(
            phenology_design(train_genotype, standardized_training_exposure)
        )
        x_direct_train = direct_scaler.transform(
            interaction_design(train_genotype, train_environment)
        ).astype(np.float32)
        x_phenology_train = phenology_scaler.transform(
            phenology_design(train_genotype, standardized_training_exposure)
        ).astype(np.float32)
        y_train = data.records.loc[split.train, "grain.yield"].to_numpy(float)
        train_experiments = data.records.loc[split.train, "Experiment"].to_numpy()
        evaluation = {}
        for scenario, indices in split.evaluation_sets().items():
            genotype, environment = inputs.rows(data, indices)
            x_direct_eval = direct_scaler.transform(
                interaction_design(genotype, environment)
            ).astype(np.float32)
            x_phenology_eval = phenology_scaler.transform(
                phenology_design(
                    genotype,
                    exposure_scaler.transform(predicted_exposures[scenario]),
                )
            ).astype(np.float32)
            frame = data.records.loc[
                indices, ["Experiment", "Variety_ID", "grain.yield"]
            ].copy()
            frame = frame.rename(columns={"grain.yield": "observed"})
            evaluation[scenario] = (
                x_direct_eval,
                x_phenology_eval,
                environment.astype(np.float32),
                frame,
            )
        caches.append(
            _YieldCache(
                split.repeat,
                x_direct_train,
                x_phenology_train,
                train_environment.astype(np.float32),
                train_experiments,
                y_train,
                evaluation,
            )
        )
        del final_stage
        _release_accelerator_memory()
    return caches


def _yield_parameters(
    trial, heads: tuple[str, ...], device: str
) -> tuple[str, str, dict]:
    mode = trial.suggest_categorical(
        "yield_mode", ["hybrid", "mean_residual_hybrid"]
    )
    head = trial.suggest_categorical("yield_head", list(heads))
    if head == "ridge":
        params = {"alpha": trial.suggest_float("ridge_alpha", 1e-4, 1e4, log=True)}
    elif head == "krr":
        params = {
            "alpha": trial.suggest_float("krr_alpha", 1e-3, 1e2, log=True),
            "gamma": trial.suggest_float("krr_gamma", 1e-5, 1e-1, log=True),
        }
    elif head == "catboost":
        params = {
            "iterations": trial.suggest_int("catboost_iterations", 400, 1400, step=100),
            "depth": trial.suggest_int("catboost_depth", 4, 8),
            "learning_rate": trial.suggest_float(
                "catboost_learning_rate", 0.01, 0.15, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "catboost_l2_leaf_reg", 0.1, 30.0, log=True
            ),
            "random_strength": trial.suggest_float("catboost_random_strength", 0.0, 2.0),
            "task_type": "GPU" if device.startswith("cuda") else "CPU",
        }
    else:
        raise ValueError(head)
    # Zero is a genuine E2E fallback; non-zero values test whether the
    # phenology-conditioned branch adds predictive information beyond daily E.
    params["phenology_weight"] = trial.suggest_categorical(
        "phenology_weight", [0.0, 0.03, 0.1, 0.3, 1.0, 2.0]
    )
    if mode == "mean_residual_hybrid":
        params["mean_alpha"] = trial.suggest_float("mean_alpha", 1e-2, 1e3, log=True)
        params["residual_weight"] = trial.suggest_categorical(
            "residual_weight", [0.5, 0.75, 1.0, 1.25]
        )
    return mode, head, params


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
    relative_rmse = scores["rmse"] / max(target_scale, 1e-8)
    rmse_quality = 1.0 / (1.0 + relative_rmse)
    return float(
        0.30 * pcc
        + 0.40 * within_pcc
        + 0.15 * within_spearman
        + 0.15 * rmse_quality
    )


def _yield_objective(caches: list[_YieldCache], config: SearchConfig):
    device = _resolved_device(config.device)

    def objective(trial) -> float:
        mode, head, params = _yield_parameters(trial, config.yield_heads, device)
        weight = float(params["phenology_weight"])
        mean_alpha = float(params.get("mean_alpha", 10.0))
        residual_weight = float(params.get("residual_weight", 1.0))
        head_params = {
            key: value
            for key, value in params.items()
            if key not in {"phenology_weight", "mean_alpha", "residual_weight"}
        }
        repeat_scores = []
        for cache in caches:
            x_train = np.concatenate(
                [cache.x_direct_train, weight * cache.x_phenology_train], axis=1
            )
            model_seed = config.seed + trial.number * 100 + cache.repeat
            mean_model = None
            residual_weight_used = 1.0
            target = cache.y_train
            if mode == "mean_residual_hybrid":
                environment_means = (
                    pd.Series(cache.y_train)
                    .groupby(cache.train_experiments, sort=False)
                    .transform("mean")
                    .to_numpy(float)
                )
                unique_positions = (
                    pd.Series(cache.train_experiments)
                    .drop_duplicates(keep="first")
                    .index.to_numpy(dtype=int)
                )
                mean_model = YieldHeadRegressor(
                    "ridge",
                    params={"alpha": mean_alpha},
                    random_state=model_seed,
                    device=device,
                ).fit(
                    cache.x_environment_train[unique_positions],
                    environment_means[unique_positions],
                )
                target = cache.y_train - environment_means
                residual_weight_used = residual_weight
            model = YieldHeadRegressor(
                head,
                params=head_params,
                random_state=model_seed,
                device=device,
                scale_features=False,
            ).fit(x_train, target)
            scenario_scores = {}
            target_scale = float(np.std(cache.y_train))
            for scenario, (
                x_direct,
                x_phenology,
                x_environment,
                source,
            ) in cache.evaluation.items():
                frame = source.copy()
                prediction = residual_weight_used * model.predict(
                    np.concatenate([x_direct, weight * x_phenology], axis=1)
                )
                if mean_model is not None:
                    prediction += mean_model.predict(x_environment)
                frame["predicted"] = prediction
                scores = regression_metrics(frame)
                scenario_scores[scenario] = _selection_score(scores, target_scale)
            repeat_scores.append(_weighted(scenario_scores))
            trial.set_user_attr(
                f"repeat_{cache.repeat}_weighted_composite_score",
                _weighted(scenario_scores),
            )
        return float(np.mean(repeat_scores))

    return objective


def _completed_best(study, label: str):
    try:
        return study.best_trial
    except ValueError as exc:
        raise RuntimeError(f"No {label} trial completed successfully") from exc


def run_search(config: SearchConfig) -> dict[str, Path]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for parameter search. Install with: pip install -e '.[search]'"
        ) from exc

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_gpace_data(config.data_dir, sequence_length=config.sequence_length)
    selection_train = make_outer_split(data, config.selection_split).train
    inner_splits = make_inner_splits(
        data,
        selection_train,
        repeats=config.inner_repeats,
        holdout_fraction=config.inner_holdout_fraction,
        seed=config.seed,
    )
    storage = _storage_url(output_dir)
    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)

    stage_study = optuna.create_study(
        study_name="gpace_stage",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    completed_stage = sum(trial.state.name == "COMPLETE" for trial in stage_study.trials)
    remaining_stage = max(0, config.stage_trials - completed_stage)
    if remaining_stage:
        stage_study.optimize(
            _stage_objective(data, inner_splits, config),
            n_trials=remaining_stage,
            n_jobs=1,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError),
        )
    best_stage_trial = _completed_best(stage_study, "stage")
    stage_params = _stage_parameters(best_stage_trial, config)

    caches = _make_yield_cache(
        data, inner_splits, config.stage_model, stage_params, config
    )
    yield_study = optuna.create_study(
        study_name="mean_residual_yield_v2",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.seed + 1, multivariate=True),
    )
    completed_yield = sum(trial.state.name == "COMPLETE" for trial in yield_study.trials)
    remaining_yield = max(0, config.yield_trials - completed_yield)
    if remaining_yield:
        yield_study.optimize(
            _yield_objective(caches, config),
            n_trials=remaining_yield,
            n_jobs=1,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError),
        )
    best_yield_trial = _completed_best(yield_study, "yield")
    yield_mode, yield_head, yield_params = _yield_parameters(
        best_yield_trial, config.yield_heads, _resolved_device(config.device)
    )

    best_config = {
        "stage_model": config.stage_model,
        "stage_params": stage_params,
        "sequence_length": config.sequence_length,
        "environment_components": config.environment_components,
        "yield_mode": yield_mode,
        "yield_head": yield_head,
        "yield_params": yield_params,
        "selection": {
            "outer_split_training_block": config.selection_split,
            "inner_repeats": config.inner_repeats,
            "scenario_weights": SCENARIO_WEIGHTS,
            "stage_objective": "weighted normalized RMSE (minimize)",
            "yield_objective": (
                "weighted composite: global PCC, within-environment PCC, "
                "within-environment Spearman and relative RMSE (maximize)"
            ),
            "stage_best_value": best_stage_trial.value,
            "yield_best_value": best_yield_trial.value,
            "external_data_used": False,
        },
    }
    paths = {
        "best_config": output_dir / "best_config.json",
        "stage_trials": output_dir / "stage_trials.csv",
        "yield_trials": output_dir / "yield_trials.csv",
        "summary": output_dir / "search_summary.json",
        "database": output_dir / "optuna.db",
    }
    paths["best_config"].write_text(
        json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stage_study.trials_dataframe().to_csv(paths["stage_trials"], index=False)
    yield_study.trials_dataframe().to_csv(paths["yield_trials"], index=False)
    summary = {
        "search_config": asdict(config),
        "data_audit": data.audit,
        "inner_split_sizes": [
            {
                "repeat": split.repeat,
                "train": len(split.train),
                **{name: len(rows) for name, rows in split.evaluation_sets().items()},
            }
            for split in inner_splits
        ],
        "best_config": best_config,
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths
