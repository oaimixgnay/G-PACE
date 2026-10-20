from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.tuning_v3 import V3SearchConfig, run_v3_search  # noqa: E402


def _numbers(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))


def _sequence_length(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sequence length must be non-negative")
    return parsed or None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Search GPACE V3 yield decomposition without external labels"
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--base-config", required=True)
    result.add_argument("--sequence-length", type=_sequence_length, default=None)
    result.add_argument("--selection-split", type=int, default=1)
    result.add_argument("--inner-repeats", type=int, default=3)
    result.add_argument("--inner-holdout-fraction", type=float, default=0.2)
    result.add_argument("--trials", type=int, default=100)
    result.add_argument("--crossfit-folds", type=int, default=3)
    result.add_argument(
        "--yield-genotype-components",
        type=_numbers,
        default=(16, 32, 64, 128),
    )
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    config = V3SearchConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=args.base_config,
        sequence_length=args.sequence_length,
        selection_split=args.selection_split,
        inner_repeats=args.inner_repeats,
        inner_holdout_fraction=args.inner_holdout_fraction,
        trials=args.trials,
        crossfit_folds=args.crossfit_folds,
        yield_genotype_components=args.yield_genotype_components,
        seed=args.seed,
        device=args.device,
    )
    outputs = run_v3_search(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
