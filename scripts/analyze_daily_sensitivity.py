from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import load_gpace_data  # noqa: E402
from gpace.experiment import _stage_kwargs  # noqa: E402
from gpace.metrics import regression_metrics  # noqa: E402
from gpace.models import YieldModel, crossfit_exposures, make_exposure_model  # noqa: E402
from gpace.splits import final_external_sets  # noqa: E402


DEFAULT_FEATURES = "Tmax.air,Tnight,VPD.air,Ri,Rad,soil.median3060"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Frozen-model daily block-permutation sensitivity. This is predictive "
            "sensitivity, not a causal variable-importance or stress-effect analysis."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--best-config", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--seeds", default="2026,2027,2028")
    result.add_argument("--crossfit-folds", type=int, default=3)
    result.add_argument("--features", default=DEFAULT_FEATURES)
    result.add_argument("--block-days", type=int, default=10)
    result.add_argument("--permutations", type=int, default=3)
    result.add_argument("--device", default="cuda")
    return result


def _seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _frame(data, indices: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    frame = data.records.loc[indices, ["Experiment", "Variety_ID", "grain.yield"]].copy()
    frame = frame.rename(columns={"grain.yield": "observed"})
    frame["predicted"] = predicted
    return frame


def _fit_models(data, best: dict, seed: int, folds: int, device: str):
    training = data.records.index[data.records["scenario"].eq("cross_validation")].to_numpy(int)
    stage_config = {
        "stage_model": best.get("stage_model", "tcn_film"),
        "stage_params": best.get("stage_params"),
        "genotype_components": int(best.get("stage_params", {}).get("genotype_components", 32)),
        "environment_components": int(best.get("environment_components", 12)),
        "tcn_epochs": int(best.get("stage_params", {}).get("epochs", 100)),
        "tcn_patience": int(best.get("stage_params", {}).get("patience", 12)),
        "device": device,
    }
    class Config:
        pass

    config = Config()
    for key, value in stage_config.items():
        setattr(config, key, value)
    kwargs = _stage_kwargs(config)
    training_exposures = crossfit_exposures(
        data, training, best.get("stage_model", "tcn_film"), folds, seed, kwargs
    )
    stage = make_exposure_model(
        best.get("stage_model", "tcn_film"), random_state=seed + 999, **kwargs
    ).fit(data, training)
    yield_model = YieldModel(
        mode="mean_residual_hybrid",
        genotype_components=int(best.get("stage_params", {}).get("genotype_components", 32)),
        environment_components=int(best.get("environment_components", 12)),
        head=best.get("yield_head", "catboost"),
        head_params=best.get("yield_params"),
        random_state=seed,
        device=device,
    ).fit(data, training, training_exposures)
    return stage, yield_model


def _predict_with_modified_environment(stage, yield_model, data, indices: np.ndarray) -> np.ndarray:
    exposures = stage.predict(data, indices)
    original = yield_model.inputs_.environment_embedding_
    try:
        yield_model.inputs_.environment_embedding_ = (
            yield_model.inputs_.environment_projector_.transform(data.environment_sequences)
        )
        return yield_model.predict(data, indices, exposures)
    finally:
        yield_model.inputs_.environment_embedding_ = original





def main() -> None:
    args = parser().parse_args()
    if args.sequence_length < 1 or args.block_days < 1 or args.permutations < 1:
        raise ValueError("sequence length, block days and permutations must be positive")
    best = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    features = tuple(value.strip() for value in args.features.split(",") if value.strip())
    feature_positions = {name: data.environment_feature_names.index(name) for name in features}
    rows: list[dict[str, object]] = []
    external = final_external_sets(data)
    for seed in _seeds(args.seeds):
        print(f"Fitting frozen V2 model for sensitivity seed {seed}.", flush=True)
        stage, yield_model = _fit_models(data, best, seed, args.crossfit_folds, args.device)
        for scenario, indices in external.items():
            base_prediction = _predict_with_modified_environment(stage, yield_model, data, indices)
            base_metrics = regression_metrics(_frame(data, indices, base_prediction))
            environment_indices = np.unique(
                data.records.loc[indices, "environment_index"].to_numpy(dtype=int)
            )
            for feature, position in feature_positions.items():
                for day_start in range(0, args.sequence_length, args.block_days):
                    day_stop = min(day_start + args.block_days, args.sequence_length)
                    for repeat in range(args.permutations):
                        rng = np.random.default_rng(
                            seed * 100000 + position * 1000 + day_start * 10 + repeat
                        )
                        sequence = data.environment_sequences.copy()
                        permutation = rng.permutation(environment_indices)
                        sequence[environment_indices, day_start:day_stop, position] = sequence[
                            permutation, day_start:day_stop, position
                        ]
                        altered = replace(data, environment_sequences=sequence)
                        prediction = _predict_with_modified_environment(
                            stage, yield_model, altered, indices
                        )
                        perturbed = regression_metrics(_frame(altered, indices, prediction))
                        rows.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "feature": feature,
                                "day_start": day_start + 1,
                                "day_stop": day_stop,
                                "repeat": repeat + 1,
                                "baseline_pcc": base_metrics["pcc"],
                                "baseline_rmse": base_metrics["rmse"],
                                "pcc": perturbed["pcc"],
                                "rmse": perturbed["rmse"],
                                "delta_pcc": base_metrics["pcc"] - perturbed["pcc"],
                                "delta_rmse": perturbed["rmse"] - base_metrics["rmse"],
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "daily_block_permutation.csv", index=False)
    summary = (
        result.groupby(["scenario", "feature", "day_start", "day_stop"], as_index=False)
        .agg(
            runs=("delta_pcc", "size"),
            delta_pcc_mean=("delta_pcc", "mean"),
            delta_pcc_sd=("delta_pcc", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
            delta_rmse_sd=("delta_rmse", "std"),
        )
    )
    summary.to_csv(output_dir / "daily_block_sensitivity_summary.csv", index=False)
    pass
    (output_dir / "README.txt").write_text(
        "Daily block permutation sensitivity for the frozen Day-100 V2 model.\n"
        "A positive delta_pcc or delta_rmse means that shuffling this variable within\n"
        "that day block reduced predictive performance. It is not a causal effect,\n"
        "physiological importance score, or a test of a stress mechanism.\n",
        encoding="utf-8",
    )
    print(f"raw: {output_dir / 'daily_block_permutation.csv'}")
    print(f"summary: {output_dir / 'daily_block_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
