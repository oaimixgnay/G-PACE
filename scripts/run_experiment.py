from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.experiment import ExperimentConfig, run_experiment


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
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sequence length must be non-negative")
    return parsed or None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run leakage-resistant GPACE experiments")
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument(
        "--target-column",
        default="grain.yield",
        help="Response column in 2b-GrainYield_components_BLUEs_level-1.csv",
    )
    result.add_argument(
        "--allow-incomplete-cv-target",
        action="store_true",
        help="Allow targets with missing supplied-CV records; use only when supplied CV is skipped.",
    )
    result.add_argument(
        "--sequence-length",
        type=_sequence_length,
        default=-1,
        help="Daily environmental days per environment; use 0 for the complete season",
    )
    result.add_argument("--splits", type=_numbers, default=tuple(range(1, 11)))
    result.add_argument("--seeds", type=_numbers, default=(2026,))
    result.add_argument("--best-config", help="JSON written by search_hyperparameters.py")
    result.add_argument(
        "--stage-model", choices=("ridge", "tcn", "tcn_film"), default=None
    )
    result.add_argument("--yield-head", choices=("ridge", "krr", "catboost"), default=None)
    result.add_argument("--crossfit-folds", type=int, default=5)
    result.add_argument("--genotype-components", type=int, default=None)
    result.add_argument("--environment-components", type=int, default=None)
    result.add_argument("--exposure-alpha", type=float, default=10.0)
    result.add_argument("--yield-alpha", type=float, default=10.0)
    result.add_argument("--bootstrap-repeats", type=int, default=500)
    result.add_argument("--tcn-epochs", type=int, default=100)
    result.add_argument("--tcn-patience", type=int, default=12)
    result.add_argument("--device", default="auto")
    result.add_argument("--skip-final-external", action="store_true")
    result.add_argument(
        "--paper-ablation",
        action="store_true",
        help="Add no-phenology, unshrunk-residual and observed-phenology controls",
    )
    result.add_argument(
        "--skip-supplied-cv",
        action="store_true",
        help="Skip the ten supplied CV partitions (use with a grouped environment holdout).",
    )
    result.add_argument(
        "--environment-holdout",
        choices=("region", "stress"),
        help="Withhold complete CV environment groups using the frozen configuration.",
    )
    result.add_argument(
        "--model-suite",
        choices=("full", "core", "primary", "biology"),
        default="full",
        help="Use core baselines or only the selected mean-residual model to reduce runtime.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    best = {}
    if args.best_config:
        best = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    stage_model = args.stage_model or best.get("stage_model", "ridge")
    yield_head = args.yield_head or best.get("yield_head", "ridge")
    genotype_components = int(
        args.genotype_components
        if args.genotype_components is not None
        else best.get("stage_params", {}).get("genotype_components", 32)
    )
    environment_components = int(
        args.environment_components
        if args.environment_components is not None
        else best.get("environment_components", 12)
    )
    yield_genotype_components = best.get("yield_genotype_components")
    if yield_genotype_components is not None:
        yield_genotype_components = int(yield_genotype_components)
    sequence_length = (
        args.sequence_length
        if args.sequence_length != -1
        else best.get("sequence_length", 100)
    )
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=sequence_length,
        splits=args.splits,
        seeds=args.seeds,
        stage_model=stage_model,
        crossfit_folds=args.crossfit_folds,
        genotype_components=genotype_components,
        environment_components=environment_components,
        exposure_alpha=args.exposure_alpha,
        yield_alpha=args.yield_alpha,
        bootstrap_repeats=args.bootstrap_repeats,
        tcn_epochs=args.tcn_epochs,
        tcn_patience=args.tcn_patience,
        device=args.device,
        final_external=not args.skip_final_external,
        stage_params=best.get("stage_params"),
        yield_mode=best.get("yield_mode"),
        yield_head=yield_head,
        yield_params=best.get("yield_params"),
        yield_genotype_components=yield_genotype_components,
        paper_ablation=args.paper_ablation,
        supplied_cv=not args.skip_supplied_cv,
        environment_holdout=args.environment_holdout,
        model_suite=args.model_suite,
        target_column=args.target_column,
        require_complete_cv_target=not args.allow_incomplete_cv_target,
    )
    outputs = run_experiment(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
