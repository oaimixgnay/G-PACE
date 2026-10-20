from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GPACEData, EXPOSURE_COLUMNS, load_gpace_data
from .metrics import cluster_bootstrap_intervals, exposure_metrics, regression_metrics
from .models import YieldModel, crossfit_exposures, make_exposure_model
from .splits import final_external_sets, make_environment_group_holdouts, make_outer_split


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str
    output_dir: str
    sequence_length: int | None = 100
    splits: tuple[int, ...] = tuple(range(1, 11))
    seeds: tuple[int, ...] = (2026,)
    stage_model: str = "ridge"
    crossfit_folds: int = 5
    genotype_components: int = 32
    environment_components: int = 12
    exposure_alpha: float = 10.0
    yield_alpha: float = 10.0
    bootstrap_repeats: int = 500
    tcn_epochs: int = 100
    tcn_patience: int = 12
    device: str = "auto"
    final_external: bool = True
    stage_params: dict | None = None
    yield_mode: str | None = None
    yield_head: str = "ridge"
    yield_params: dict | None = None
    yield_genotype_components: int | None = None
    paper_ablation: bool = False
    supplied_cv: bool = True
    environment_holdout: str | None = None
    model_suite: str = "full"
    target_column: str = "grain.yield"
    require_complete_cv_target: bool = True


def _prediction_frame(
    data: GPACEData,
    indices: np.ndarray,
    predicted: np.ndarray,
    model: str,
    scenario: str,
    split: int,
    seed: int,
) -> pd.DataFrame:
    source = data.records.loc[
        indices, ["record_id", "Experiment", "Variety_ID", "grain.yield"]
    ].copy()
    source = source.rename(columns={"grain.yield": "observed"})
    source["predicted"] = predicted
    source["model"] = model
    source["scenario"] = scenario
    source["split"] = split
    source["seed"] = seed
    return source


def _stage_kwargs(config: ExperimentConfig) -> dict:
    if config.stage_model == "ridge":
        result = {
            "genotype_components": config.genotype_components,
            "environment_components": config.environment_components,
            "alpha": config.exposure_alpha,
        }
    else:
        result = {
            "genotype_components": config.genotype_components,
            "epochs": config.tcn_epochs,
            "patience": config.tcn_patience,
            "device": config.device,
        }
    result.update(config.stage_params or {})
    result["device"] = config.device
    return result


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.model_suite not in {"full", "core", "primary", "biology"}:
        raise ValueError("model_suite must be 'full', 'core', 'primary' or 'biology'")
    if not (config.supplied_cv or config.final_external or config.environment_holdout):
        raise ValueError("At least one evaluation protocol must be enabled")
    data = load_gpace_data(
        config.data_dir,
        sequence_length=config.sequence_length,
        target_column=config.target_column,
        require_complete_cv_target=config.require_complete_cv_target,
    )
    all_predictions = []
    all_exposure_predictions = []
    metric_rows = []
    interval_rows = []
    stage_metric_rows = []

    def evaluate_task(
        train_indices: np.ndarray,
        evaluation_sets: dict[str, np.ndarray],
        split_number: int,
        seed: int,
        holdout_scheme: str | None = None,
        heldout_group: str | None = None,
    ) -> None:
        task_offset = 99 if split_number == 0 else split_number
        training_exposures = crossfit_exposures(
            data,
            train_indices,
            kind=config.stage_model,
            folds=config.crossfit_folds,
            random_state=seed + task_offset * 1000,
            model_kwargs=_stage_kwargs(config),
        )
        final_stage_model = make_exposure_model(
            config.stage_model,
            random_state=seed + task_offset * 1000 + 999,
            **_stage_kwargs(config),
        ).fit(data, train_indices)
        predicted_exposures = {
            name: final_stage_model.predict(data, indices)
            for name, indices in evaluation_sets.items()
        }

        for scenario, indices in evaluation_sets.items():
            observed_stage = data.records.loc[indices, EXPOSURE_COLUMNS].to_numpy(float)
            stage_scores = exposure_metrics(observed_stage, predicted_exposures[scenario])
            stage_scores["scenario"] = scenario
            stage_scores["split"] = split_number
            stage_scores["seed"] = seed
            stage_scores["holdout_scheme"] = holdout_scheme
            stage_scores["heldout_group"] = heldout_group
            stage_metric_rows.append(stage_scores)
            stage_frame = data.records.loc[
                indices, ["record_id", "Experiment", "Variety_ID"]
            ].copy()
            for column, name in enumerate(EXPOSURE_COLUMNS):
                stage_frame[f"observed__{name}"] = observed_stage[:, column]
                stage_frame[f"predicted__{name}"] = predicted_exposures[scenario][:, column]
            stage_frame["scenario"] = scenario
            stage_frame["split"] = split_number
            stage_frame["seed"] = seed
            stage_frame["holdout_scheme"] = holdout_scheme
            stage_frame["heldout_group"] = heldout_group
            all_exposure_predictions.append(stage_frame)

        model_specs = [
            ("g_only", "g_only", "ridge", None, None),
            ("e_only", "e_only", "ridge", None, None),
            ("g_plus_e", "g_plus_e", "ridge", None, None),
            ("e2e", "e2e", "ridge", None, None),
            ("gpace", "gpace", "ridge", None, "predicted"),
            ("hybrid", "hybrid", "ridge", {"phenology_weight": 1.0}, "predicted"),
            (
                "mean_residual_hybrid",
                "mean_residual_hybrid",
                "ridge",
                {"phenology_weight": 0.1},
                "predicted",
            ),
        ]
        if config.yield_head != "ridge":
            model_specs.extend(
                [
                    (
                        f"e2e_{config.yield_head}",
                        "e2e",
                        config.yield_head,
                        config.yield_params,
                        None,
                    ),
                    (
                        f"gpace_{config.yield_head}",
                        "gpace",
                        config.yield_head,
                        config.yield_params,
                        "predicted",
                    ),
                    (
                        f"hybrid_{config.yield_head}",
                        "hybrid",
                        config.yield_head,
                        config.yield_params,
                        "predicted",
                    ),
                    (
                        f"mean_residual_hybrid_{config.yield_head}",
                        "mean_residual_hybrid",
                        config.yield_head,
                        config.yield_params,
                        "predicted",
                    ),
                ]
            )
        oracle_suffix = "" if config.yield_head == "ridge" else f"_{config.yield_head}"
        model_specs.append(
            (
                f"oracle{oracle_suffix}",
                "oracle",
                config.yield_head,
                config.yield_params,
                "observed",
            )
        )
        model_specs.append(
            (
                f"hybrid_oracle{oracle_suffix}",
                "hybrid_oracle",
                config.yield_head,
                config.yield_params,
                "observed",
            )
        )
        if config.yield_mode == "v3_mean_residual":
            model_specs.append(
                (
                    f"v3_mean_residual{oracle_suffix}",
                    "v3_mean_residual",
                    config.yield_head,
                    config.yield_params,
                    "predicted",
                )
            )

        if config.paper_ablation:
            selected_params = dict(config.yield_params or {})
            no_phenology_params = {**selected_params, "phenology_weight": 0.0}
            unshrunk_params = {**selected_params, "residual_weight": 1.0}
            model_specs.extend(
                [
                    (
                        f"mean_residual_no_phenology{oracle_suffix}",
                        "mean_residual_hybrid",
                        config.yield_head,
                        no_phenology_params,
                        "predicted",
                    ),
                    (
                        f"mean_residual_unshrunk{oracle_suffix}",
                        "mean_residual_hybrid",
                        config.yield_head,
                        unshrunk_params,
                        "predicted",
                    ),
                    (
                        f"mean_residual_observed_phenology{oracle_suffix}",
                        "mean_residual_hybrid",
                        config.yield_head,
                        selected_params,
                        "observed",
                    ),
                ]
            )

        if config.model_suite != "full":
            if config.yield_head == "ridge":
                selected_name = "mean_residual_hybrid"
                core_names = {"e2e", "hybrid", selected_name}
            else:
                selected_name = f"mean_residual_hybrid_{config.yield_head}"
                core_names = {
                    "e2e",
                    f"e2e_{config.yield_head}",
                    f"hybrid_{config.yield_head}",
                    selected_name,
                }
            if config.model_suite == "primary":
                core_names = {selected_name}
            elif config.model_suite == "biology":
                core_names = {
                    "e2e",
                    f"e2e_{config.yield_head}",
                    selected_name,
                    f"mean_residual_no_phenology{oracle_suffix}",
                }
            model_specs = [spec for spec in model_specs if spec[0] in core_names]

        for model_name, mode, selected_head, selected_params, stage_source in model_specs:
            train_stage = None
            if stage_source == "predicted":
                train_stage = training_exposures
            elif stage_source == "observed":
                train_stage = data.records.loc[train_indices, EXPOSURE_COLUMNS].to_numpy(float)
            model = YieldModel(
                mode=mode,
                genotype_components=(
                    config.yield_genotype_components
                    if mode == "v3_mean_residual"
                    and config.yield_genotype_components is not None
                    else config.genotype_components
                ),
                environment_components=config.environment_components,
                alpha=config.yield_alpha,
                head=selected_head,
                head_params=selected_params,
                random_state=seed + task_offset * 100,
                device=config.device,
            ).fit(data, train_indices, train_stage)

            for scenario, indices in evaluation_sets.items():
                test_stage = None
                if stage_source == "predicted":
                    test_stage = predicted_exposures[scenario]
                elif stage_source == "observed":
                    test_stage = data.records.loc[indices, EXPOSURE_COLUMNS].to_numpy(float)
                predicted_yield = model.predict(data, indices, test_stage)
                prediction_frame = _prediction_frame(
                    data,
                    indices,
                    predicted_yield,
                    model_name,
                    scenario,
                    split_number,
                    seed,
                )
                prediction_frame["holdout_scheme"] = holdout_scheme
                prediction_frame["heldout_group"] = heldout_group
                all_predictions.append(prediction_frame)
                scores = regression_metrics(prediction_frame)
                metric_rows.append(
                    {
                        "model": model_name,
                        "scenario": scenario,
                        "split": split_number,
                        "seed": seed,
                        "holdout_scheme": holdout_scheme,
                        "heldout_group": heldout_group,
                        **scores,
                    }
                )
                intervals = cluster_bootstrap_intervals(
                    prediction_frame,
                    repeats=config.bootstrap_repeats,
                    seed=seed + task_offset * 10,
                )
                intervals["model"] = model_name
                intervals["scenario"] = scenario
                intervals["split"] = split_number
                intervals["seed"] = seed
                intervals["holdout_scheme"] = holdout_scheme
                intervals["heldout_group"] = heldout_group
                interval_rows.append(intervals)

    for seed in config.seeds:
        if config.supplied_cv:
            for split_number in config.splits:
                outer = make_outer_split(data, split_number)
                evaluate_task(outer.train, outer.evaluation_sets(), split_number, seed)

        if config.environment_holdout:
            for task_number, task in enumerate(
                make_environment_group_holdouts(data, config.environment_holdout), start=101
            ):
                evaluate_task(
                    task.train,
                    {task.scenario: task.test},
                    task_number,
                    seed,
                    holdout_scheme=task.scheme,
                    heldout_group=task.group,
                )

        if config.final_external:
            full_cv_training = data.records.index[
                data.records["scenario"].eq("cross_validation")
            ].to_numpy(dtype=int)
            evaluate_task(full_cv_training, final_external_sets(data), 0, seed)

    predictions = pd.concat(all_predictions, ignore_index=True)
    stage_predictions = pd.concat(all_exposure_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    intervals = pd.concat(interval_rows, ignore_index=True)
    stage_metrics = pd.concat(stage_metric_rows, ignore_index=True)

    paths = {
        "predictions": output_dir / "yield_predictions.csv",
        "exposure_predictions": output_dir / "exposure_predictions.csv",
        "metrics": output_dir / "yield_metrics.csv",
        "intervals": output_dir / "yield_metric_intervals.csv",
        "exposure_metrics": output_dir / "exposure_metrics.csv",
        "manifest": output_dir / "run_manifest.json",
    }
    predictions.to_csv(paths["predictions"], index=False)
    stage_predictions.to_csv(paths["exposure_predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    intervals.to_csv(paths["intervals"], index=False)
    stage_metrics.to_csv(paths["exposure_metrics"], index=False)
    manifest = {
        "config": asdict(config),
        "data_audit": data.audit,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths
