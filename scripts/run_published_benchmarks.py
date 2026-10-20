from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.benchmarks import DEFAULT_BENCHMARK_CONFIGS, make_published_benchmark  # noqa: E402
from gpace.data import load_gpace_data  # noqa: E402
from gpace.metrics import cluster_bootstrap_intervals, regression_metrics  # noqa: E402
from gpace.splits import final_external_sets, make_outer_split  # noqa: E402


def _numbers(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, stop = map(int, part.split("-", maxsplit=1))
            values.extend(range(start, stop + 1))
        elif part:
            values.append(int(part))
    return tuple(dict.fromkeys(values))


def _sequence_length(value: str) -> int | None:
    days = int(value)
    if days < 0:
        raise argparse.ArgumentTypeError("sequence length must be non-negative")
    return days or None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Benchmark fixed GPACE against data-matched published GxE model families"
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--best-config", help="JSON from search_published_benchmarks.py")
    result.add_argument("--sequence-length", type=_sequence_length, default=100)
    result.add_argument("--splits", type=_numbers, default=tuple(range(1, 11)))
    result.add_argument("--seeds", type=_numbers, default=(2026, 2027, 2028))
    result.add_argument(
        "--models",
        default="reaction_norm_kernel,ceris_jgra,geformer_style",
        help="Comma-separated benchmark IDs",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=0,
        help="Per-replicate intervals; keep 0 and use the paired external bootstrap in reporting.",
    )
    result.add_argument(
        "--gpace-predictions",
        help="Frozen V2 yield_predictions.csv to merge as mean_residual_hybrid_catboost.",
    )
    result.add_argument(
        "--gpace-model", default="mean_residual_hybrid_catboost", help="Model label to retain from --gpace-predictions"
    )
    return result


def _prediction_frame(data, indices, prediction, model, scenario, split, seed):
    frame = data.records.loc[
        indices, ["record_id", "Experiment", "Variety_ID", "grain.yield"]
    ].copy()
    frame = frame.rename(columns={"grain.yield": "observed"})
    frame["predicted"] = np.asarray(prediction, dtype=float)
    frame["model"] = model
    frame["scenario"] = scenario
    frame["split"] = split
    frame["seed"] = seed
    return frame


def _append_scores(predictions, metrics, intervals, model, scenario, split, seed, repeats):
    scores = regression_metrics(predictions)
    metrics.append({"model": model, "scenario": scenario, "split": split, "seed": seed, **scores})
    if repeats:
        interval = cluster_bootstrap_intervals(predictions, repeats=repeats, seed=seed + split * 17)
        interval["model"] = model
        interval["scenario"] = scenario
        interval["split"] = split
        interval["seed"] = seed
        intervals.append(interval)


def _merge_fixed_gpace(path: Path, model: str, predictions, metrics, intervals, repeats: int) -> int:
    source = pd.read_csv(path)
    source = source.loc[source["model"].eq(model)].copy()
    required = {"record_id", "Experiment", "Variety_ID", "observed", "predicted", "scenario", "split", "seed"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"GPACE prediction file misses columns: {missing}")
    for (scenario, split, seed), group in source.groupby(["scenario", "split", "seed"], sort=False):
        group = group.loc[:, sorted(required)].copy()
        group["model"] = model
        predictions.append(group)
        _append_scores(group, metrics, intervals, model, scenario, int(split), int(seed), repeats)
    return len(source)


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = sorted(set(selected) - set(DEFAULT_BENCHMARK_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown benchmark IDs: {unknown}")
    config = dict(DEFAULT_BENCHMARK_CONFIGS)
    if args.best_config:
        loaded = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
        config.update(loaded.get("models", loaded))
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    all_predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    intervals: list[pd.DataFrame] = []

    def evaluate(train_indices, evaluation_sets, split, seed):
        for model_number, name in enumerate(selected):
            model = make_published_benchmark(
                name,
                random_state=seed + split * 100 + model_number,
                device=args.device,
                **config[name],
            ).fit(data, train_indices)
            for scenario, indices in evaluation_sets.items():
                frame = _prediction_frame(
                    data, indices, model.predict(data, indices), name, scenario, split, seed
                )
                all_predictions.append(frame)
                _append_scores(frame, metrics, intervals, name, scenario, split, seed, args.bootstrap_repeats)

    for seed in args.seeds:
        for split in args.splits:
            outer = make_outer_split(data, split)
            evaluate(outer.train, outer.evaluation_sets(), split, seed)
        train = data.records.index[data.records["scenario"].eq("cross_validation")].to_numpy(dtype=int)
        evaluate(train, final_external_sets(data), 0, seed)

    gpace_rows = 0
    if args.gpace_predictions:
        gpace_rows = _merge_fixed_gpace(
            Path(args.gpace_predictions).resolve(), args.gpace_model, all_predictions, metrics,
            intervals, args.bootstrap_repeats,
        )
    predictions = pd.concat(all_predictions, ignore_index=True)
    metric_table = pd.DataFrame(metrics)
    interval_table = pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(
        columns=["metric", "estimate", "ci_low", "ci_high", "model", "scenario", "split", "seed"]
    )
    predictions.to_csv(output_dir / "yield_predictions.csv", index=False)
    metric_table.to_csv(output_dir / "yield_metrics.csv", index=False)
    interval_table.to_csv(output_dir / "yield_metric_intervals.csv", index=False)
    manifest = {
        "sequence_length": args.sequence_length,
        "models": list(selected),
        "model_configs": {name: config[name] for name in selected},
        "seeds": list(args.seeds),
        "splits": list(args.splits),
        "external_data_used_for_selection": False,
        "frozen_gpace_prediction_rows_merged": gpace_rows,
        "implementation_notes": {
            "reaction_norm_kernel": "Environmental-relatedness reaction norm (Jarquín-style RKHS).",
            "ceris_jgra": "Training-only daily-window CERIS environmental-index search plus joint genomic regression.",
            "autogs_style": "Training-only marker screen, reduced environment features and selected yield head; data-matched Auto-GS-style implementation.",
            "geformer_style": "Compact gMLP-temporal-attention reimplementation inspired by GEFormer; not the Python-3.8 compiled official release.",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"predictions: {output_dir / 'yield_predictions.csv'}")
    print(f"metrics: {output_dir / 'yield_metrics.csv'}")
    print(f"manifest: {output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
