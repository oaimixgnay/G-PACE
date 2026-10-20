from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.tuning import SearchConfig, run_search


def _names(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _sequence_length(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sequence length must be non-negative")
    return parsed or None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Training-only Optuna search for Hybrid-GPACE stage and yield models"
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument(
        "--sequence-length",
        type=_sequence_length,
        default=100,
        help="Daily environmental days per environment; use 0 for the complete season",
    )
    result.add_argument("--selection-split", type=int, default=1)
    result.add_argument("--inner-repeats", type=int, default=2)
    result.add_argument("--inner-holdout-fraction", type=float, default=0.2)
    result.add_argument("--stage-trials", type=int, default=30)
    result.add_argument("--yield-trials", type=int, default=60)
    result.add_argument(
        "--stage-model", choices=("ridge", "tcn", "tcn_film"), default="tcn_film"
    )
    result.add_argument(
        "--yield-heads",
        type=_names,
        default=("ridge", "krr", "catboost"),
        help="Comma-separated subset of ridge,krr,catboost",
    )
    result.add_argument("--crossfit-folds", type=int, default=3)
    result.add_argument("--environment-components", type=int, default=12)
    result.add_argument("--tcn-epochs", type=int, default=120)
    result.add_argument("--tcn-patience", type=int, default=15)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    invalid = set(args.yield_heads) - {"ridge", "krr", "catboost"}
    if invalid:
        raise ValueError(f"Unknown yield heads: {sorted(invalid)}")
    config = SearchConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        selection_split=args.selection_split,
        inner_repeats=args.inner_repeats,
        inner_holdout_fraction=args.inner_holdout_fraction,
        stage_trials=args.stage_trials,
        yield_trials=args.yield_trials,
        stage_model=args.stage_model,
        yield_heads=args.yield_heads,
        crossfit_folds=args.crossfit_folds,
        environment_components=args.environment_components,
        tcn_epochs=args.tcn_epochs,
        tcn_patience=args.tcn_patience,
        seed=args.seed,
        device=args.device,
    )
    outputs = run_search(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
