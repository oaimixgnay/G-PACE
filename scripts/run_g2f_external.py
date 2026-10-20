"""Run a frozen G-PACE replication on the independent G2F 2024 archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.g2f import load_g2f_2024_data
from gpace.metrics import cluster_bootstrap_intervals, exposure_metrics, regression_metrics
from gpace.models import YieldModel, crossfit_exposures, make_exposure_model


DEFAULT_STAGE_PARAMS = {
    "genotype_components": 16,
    "environment_components": 12,
    "alpha": 10.0,
}
DEFAULT_YIELD_PARAMS = {
    "iterations": 1400,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 6.9,
    "random_strength": 0.47,
    "phenology_weight": 0.03,
    "residual_weight": 0.75,
    "mean_alpha": 0.42,
}


def _numbers(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent G2F 2024 replication of genotype-conditioned phenology-aligned prediction"
    )
    result.add_argument("--data-dir", required=True, help="Directory containing the eight official G2F files")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--seeds", type=_numbers, default=(2026, 2027, 2028))
    result.add_argument("--crossfit-folds", type=int, default=3)
    result.add_argument("--bootstrap-repeats", type=int, default=1000)
    result.add_argument("--stage-model", choices=("ridge", "tcn", "tcn_film"), default="ridge")
    result.add_argument("--yield-head", choices=("ridge", "catboost"), default="catboost")
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--config-json",
        help="Optional frozen JSON with stage_params and yield_params. It is never tuned on 2024 labels.",
    )
    return result


def _stage_kwargs(stage_model: str, supplied: dict, device: str) -> dict:
    if stage_model == "ridge":
        allowed = {"genotype_components", "environment_components", "alpha"}
        return {key: value for key, value in supplied.items() if key in allowed}
    defaults = {
        "genotype_components": 16,
        "width": 64,
        "depth": 2,
        "kernel_size": 5,
        "dropout": 0.06,
        "learning_rate": 2.3e-4,
        "weight_decay": 4.3e-6,
        "batch_size": 512,
        "epochs": 120,
        "patience": 15,
        "validation_fraction": 0.15,
        "amp": True,
        "device": device,
    }
    defaults.update(supplied)
    defaults["device"] = device
    return defaults


def _prediction_frame(data, indices: np.ndarray, predicted: np.ndarray, model: str, scenario: str, seed: int) -> pd.DataFrame:
    frame = data.records.loc[indices, ["record_id", "Experiment", "Variety_ID", "grain.yield"]].copy()
    frame = frame.rename(columns={"grain.yield": "observed"})
    frame["predicted"] = predicted
    frame["model"] = model
    frame["scenario"] = scenario
    frame["seed"] = seed
    return frame


def _evaluate_task(data, train_indices, test_indices, scenario, seed, args, stage_params, yield_params):
    training_exposures = crossfit_exposures(
        data,
        train_indices,
        kind=args.stage_model,
        folds=args.crossfit_folds,
        random_state=seed,
        model_kwargs=_stage_kwargs(args.stage_model, stage_params, args.device),
    )
    stage = make_exposure_model(
        args.stage_model,
        random_state=seed + 999,
        **_stage_kwargs(args.stage_model, stage_params, args.device),
    ).fit(data, train_indices)
    test_exposures = stage.predict(data, test_indices)

    stage_truth = data.records.loc[train_indices, data.exposure_columns].to_numpy(float)
    stage_scores = exposure_metrics(stage_truth, training_exposures)
    stage_scores["scenario"] = f"{scenario}_training_crossfit"
    stage_scores["seed"] = seed

    outputs = []
    for label, mode, needs_exposure in (
        ("e2e_catboost" if args.yield_head == "catboost" else "e2e_ridge", "e2e", False),
        ("g_pace", "mean_residual_hybrid", True),
    ):
        model = YieldModel(
            mode=mode,
            genotype_components=int(stage_params.get("genotype_components", 16)),
            environment_components=int(stage_params.get("environment_components", 12)),
            alpha=10.0,
            head=args.yield_head,
            head_params=yield_params,
            random_state=seed,
            device=args.device,
        ).fit(data, train_indices, training_exposures if needs_exposure else None)
        prediction = model.predict(data, test_indices, test_exposures if needs_exposure else None)
        frame = _prediction_frame(data, test_indices, prediction, label, scenario, seed)
        outputs.append(frame)
    return outputs, stage_scores


def main() -> None:
    args = parser().parse_args()
    if args.sequence_length < 1:
        raise ValueError("Use a positive --sequence-length; 100 is the preregistered early-season setting")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    supplied = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    stage_params = dict(DEFAULT_STAGE_PARAMS)
    stage_params.update(supplied.get("stage_params", {}))
    yield_params = dict(DEFAULT_YIELD_PARAMS)
    yield_params.update(supplied.get("yield_params", {}))
    yield_params["task_type"] = "GPU" if args.device.startswith("cuda") else "CPU"

    prepared = load_g2f_2024_data(args.data_dir, sequence_length=args.sequence_length)
    data = prepared.data
    train = prepared.training_indices
    external = prepared.external_indices
    temporal = train[data.records.loc[train, "Year"].to_numpy(int) == 2023]
    temporal_train = train[data.records.loc[train, "Year"].to_numpy(int) < 2023]
    if not len(temporal) or not len(temporal_train):
        raise ValueError("G2F training archive does not contain the required pre-2023 and 2023 records")

    predictions: list[pd.DataFrame] = []
    stage_scores: list[pd.DataFrame] = []
    for seed in args.seeds:
        diagnostic, diagnostic_stage = _evaluate_task(
            data, temporal_train, temporal, "temporal_2023", seed, args, stage_params, yield_params
        )
        final, final_stage = _evaluate_task(
            data, train, external, "external_2024", seed, args, stage_params, yield_params
        )
        predictions.extend(diagnostic + final)
        stage_scores.extend((diagnostic_stage, final_stage))

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = []
    intervals = []
    for (model, scenario, seed), frame in prediction_frame.groupby(["model", "scenario", "seed"], sort=True):
        metrics.append({"model": model, "scenario": scenario, "seed": seed, **regression_metrics(frame)})
        interval = cluster_bootstrap_intervals(frame, repeats=args.bootstrap_repeats, seed=int(seed))
        interval["model"] = model
        interval["scenario"] = scenario
        interval["seed"] = seed
        intervals.append(interval)

    paths = {
        "predictions": output_dir / "g2f_yield_predictions.csv",
        "metrics": output_dir / "g2f_yield_metrics.csv",
        "intervals": output_dir / "g2f_yield_metric_intervals.csv",
        "stage_metrics": output_dir / "g2f_training_crossfit_exposure_metrics.csv",
        "manifest": output_dir / "g2f_run_manifest.json",
    }
    prediction_frame.to_csv(paths["predictions"], index=False)
    pd.DataFrame(metrics).to_csv(paths["metrics"], index=False)
    pd.concat(intervals, ignore_index=True).to_csv(paths["intervals"], index=False)
    pd.concat(stage_scores, ignore_index=True).to_csv(paths["stage_metrics"], index=False)
    manifest = {
        "protocol": {
            "temporal_diagnostic": "fit through 2022, evaluate 2023",
            "independent_external_evaluation": "fit 2014-2023, evaluate official 2024 labels",
            "guardrail": "2024 observed yield is not used for fitting, stage-label construction, or configuration selection",
        },
        "config": vars(args),
        "stage_params": stage_params,
        "yield_params": yield_params,
        "data_audit": data.audit,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
