"""Fit training-only G2F benchmark baselines and predict official 2024 labels.

The script selects each baseline's small configuration grid using the 2023
chronological holdout (fit <= 2022), then refits on 2014--2023 and evaluates
the official 2024 labels exactly once.  The 2024 labels never participate in
model fitting, window selection, or configuration selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.benchmarks import CERISJGRA, ECGBLUP, GEFormerStyle, RFFReactionNorm
from gpace.g2f import load_g2f_2024_data
from gpace.metrics import regression_metrics


RFF_CANDIDATES = (
    {"genotype_components": 32, "environment_components": 12, "rff_components": 128, "alpha": 10.0},
    {"genotype_components": 32, "environment_components": 12, "rff_components": 128, "alpha": 100.0},
    {"genotype_components": 32, "environment_components": 12, "rff_components": 256, "alpha": 10.0},
    {"genotype_components": 32, "environment_components": 12, "rff_components": 256, "alpha": 100.0},
)
CERIS_CANDIDATES = (
    {"genotype_components": 16, "alpha": 1.0, "min_window": 7, "max_window": 35, "window_step": 7},
    {"genotype_components": 16, "alpha": 10.0, "min_window": 7, "max_window": 35, "window_step": 7},
    {"genotype_components": 16, "alpha": 100.0, "min_window": 7, "max_window": 42, "window_step": 7},
    {"genotype_components": 32, "alpha": 1.0, "min_window": 7, "max_window": 35, "window_step": 7},
    {"genotype_components": 32, "alpha": 10.0, "min_window": 7, "max_window": 35, "window_step": 7},
    {"genotype_components": 32, "alpha": 100.0, "min_window": 7, "max_window": 42, "window_step": 7},
)
ECGBLUP_CANDIDATES = (
    {"genotype_components": 32, "environment_components": 8, "genotype_alpha": 10.0, "environment_alpha": 1.0, "interaction_alpha": 10.0},
    {"genotype_components": 32, "environment_components": 12, "genotype_alpha": 10.0, "environment_alpha": 1.0, "interaction_alpha": 30.0},
    {"genotype_components": 64, "environment_components": 8, "genotype_alpha": 10.0, "environment_alpha": 1.0, "interaction_alpha": 30.0},
    {"genotype_components": 64, "environment_components": 12, "genotype_alpha": 10.0, "environment_alpha": 1.0, "interaction_alpha": 100.0},
)
GEFORMER_CANDIDATES = (
    {"genotype_components": 64, "width": 64, "heads": 4, "dropout": 0.10, "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 512, "epochs": 100, "patience": 12},
    {"genotype_components": 64, "width": 96, "heads": 4, "dropout": 0.15, "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 512, "epochs": 120, "patience": 15},
)


def _numbers(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Training-only benchmark selection and official G2F 2024 evaluation"
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--seeds", type=_numbers, default=(2026, 2027, 2028))
    result.add_argument("--selection-seed", type=int, default=2026)
    result.add_argument("--selection-year", type=int, default=2023)
    result.add_argument(
        "--models",
        type=_names,
        default=("reaction_norm_rff", "ceris_jgra"),
        help="Comma-separated models: reaction_norm_rff, ceris_jgra, ec_gblup, geformer_style",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--config-json", type=Path, help="Use the archived selected model parameters without a new search")
    return result


def _frame(data, indices: np.ndarray, prediction: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    result = data.records.loc[
        indices, ["record_id", "Experiment", "Variety_ID", "grain.yield"]
    ].copy()
    result = result.rename(columns={"grain.yield": "observed"})
    result["predicted"] = np.asarray(prediction, dtype=float)
    result["model"] = model
    result["scenario"] = "external_2024"
    result["seed"] = seed
    return result


def _score(frame: pd.DataFrame) -> float:
    metrics = regression_metrics(frame)
    values = (
        0.40 * metrics["pcc"],
        0.30 * metrics["within_environment_pcc"],
        0.20 * metrics["within_environment_spearman"],
        0.10 * np.clip(metrics["r2"], -1.0, 1.0),
    )
    return float(np.nansum(values))


def _selection_frame(data, indices: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    result = data.records.loc[
        indices, ["Experiment", "Variety_ID", "grain.yield"]
    ].copy()
    return result.rename(columns={"grain.yield": "observed"}).assign(predicted=prediction)


def _make_model(name: str, config: dict, seed: int, device: str):
    if name == "reaction_norm_rff":
        return RFFReactionNorm(random_state=seed, **config)
    if name == "ceris_jgra":
        return CERISJGRA(random_state=seed, **config)
    if name == "ec_gblup":
        return ECGBLUP(random_state=seed, **config)
    if name == "geformer_style":
        return GEFormerStyle(random_state=seed, device=device, **config)
    raise ValueError(f"Unsupported G2F benchmark: {name}")


def _select_config(data, train: np.ndarray, holdout: np.ndarray, name: str, candidates, seeds: tuple[int, ...], device: str):
    trials: list[dict] = []
    best_config: dict | None = None
    best_value = -np.inf
    for candidate_id, candidate in enumerate(candidates):
        scores = []
        for seed in seeds:
            model = _make_model(name, candidate, seed, device).fit(data, train)
            score = _score(_selection_frame(data, holdout, model.predict(data, holdout)))
            scores.append(score)
        value = float(np.mean(scores))
        trials.append(
            {
                "model": name,
                "candidate_id": candidate_id,
                "objective_mean": value,
                "objective_by_seed": json.dumps(scores),
                "config": json.dumps(candidate, sort_keys=True),
            }
        )
        if value > best_value:
            best_value = value
            best_config = dict(candidate)
    if best_config is None:
        raise RuntimeError(f"No valid candidates for {name}")
    return best_config, trials


def main() -> None:
    args = parser().parse_args()
    if args.sequence_length < 1:
        raise ValueError("sequence-length must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_g2f_2024_data(args.data_dir, sequence_length=args.sequence_length)
    data = prepared.data
    train = prepared.training_indices
    external = prepared.external_indices
    years = data.records.loc[train, "Year"].to_numpy(dtype=int)
    selection_holdout = train[years == args.selection_year]
    selection_train = train[years < args.selection_year]
    if not len(selection_train) or not len(selection_holdout):
        raise ValueError(f"G2F archive cannot make the requested {args.selection_year} temporal split")

    available_models = {
        "reaction_norm_rff": RFF_CANDIDATES,
        "ceris_jgra": CERIS_CANDIDATES,
        "ec_gblup": ECGBLUP_CANDIDATES,
        "geformer_style": GEFORMER_CANDIDATES,
    }
    unknown = sorted(set(args.models) - set(available_models))
    if unknown:
        raise ValueError(f"Unknown --models values: {unknown}")
    models = {name: available_models[name] for name in args.models}
    selected: dict[str, dict] = {}
    trials: list[dict] = []
    if args.config_json is not None:
        frozen = json.loads(args.config_json.read_text(encoding="utf-8"))
        missing = sorted(set(models) - set(frozen.get("models", {})))
        if missing:
            raise ValueError(f"Frozen configuration lacks requested models: {missing}")
        selected = {name: dict(frozen["models"][name]) for name in models}
    else:
        for name, candidates in models.items():
            config, outcomes = _select_config(
                data, selection_train, selection_holdout, name, candidates, (args.selection_seed,), args.device
            )
            selected[name] = config
            trials.extend(outcomes)

    predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    for model_number, (name, config) in enumerate(selected.items()):
        for seed in args.seeds:
            model = _make_model(name, config, seed + model_number * 10000, args.device).fit(data, train)
            frame = _frame(data, external, model.predict(data, external), name, seed)
            predictions.append(frame)
            metrics.append({"model": name, "scenario": "external_2024", "seed": seed, **regression_metrics(frame)})

    prediction_table = pd.concat(predictions, ignore_index=True)
    paths = {
        "search_trials": output_dir / "g2f_benchmark_search_trials.csv",
        "selection": output_dir / "g2f_benchmark_config.json",
        "predictions": output_dir / "g2f_benchmark_predictions.csv",
        "metrics": output_dir / "g2f_benchmark_metrics.csv",
        "manifest": output_dir / "g2f_benchmark_manifest.json",
    }
    if trials:
        pd.DataFrame(trials).to_csv(paths["search_trials"], index=False)
    paths["selection"].write_text(
        json.dumps(
            {
                "models": selected,
                "parameter_mode": "archived_selected" if args.config_json else "historical_search",
                "parameter_source": str(args.config_json) if args.config_json else None,
                "selection_protocol": {
                    "fit_years": f"through {args.selection_year - 1}",
                    "selection_holdout": args.selection_year,
                    "selection_seed": args.selection_seed,
                    "external_2024_labels_used": False,
                    "objective": "0.40 PCC + 0.30 within-environment PCC + 0.20 within-environment Spearman + 0.10 clipped R2",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prediction_table.to_csv(paths["predictions"], index=False)
    pd.DataFrame(metrics).to_csv(paths["metrics"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "protocol": {
                    "selection": "archived selected configuration; no new search" if args.config_json else f"fit 2014-{args.selection_year - 1}, select on {args.selection_year}",
                    "official_external_evaluation": "refit 2014-2023, evaluate official 2024 labels",
                    "guardrail": "2024 observed yield is not used for fitting or configuration selection",
                    "reaction_norm_note": "RFF approximation of the genomic, environmental and GxE RKHS reaction norm; exact Gram kernel is computationally infeasible at G2F scale.",
                },
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "selected_models": selected,
                "data_audit": data.audit,
                "outputs": {name: str(path) for name, path in paths.items() if name != "search_trials" or trials},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, path in paths.items():
        if name == "search_trials" and not trials:
            continue
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
