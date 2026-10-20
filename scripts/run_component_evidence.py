from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.experiment import ExperimentConfig, run_experiment  # noqa: E402


DEFAULT_TARGETS = ("grain.number", "seed.size")


def _numbers(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            first, last = map(int, part.split("-", maxsplit=1))
            values.extend(range(first, last + 1))
        elif part:
            values.append(int(part))
    return tuple(dict.fromkeys(values))


def _slug(column: str) -> str:
    return column.replace(".", "_").replace(" ", "_")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Day-100 GPACE architecture for yield components. "
            "The default protocol is external-only, so it is a biological validation "
            "rather than another model-selection exercise."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--best-config", required=True)
    result.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--seeds", type=_numbers, default=(2026, 2027, 2028))
    result.add_argument("--splits", type=_numbers, default=tuple(range(1, 11)))
    result.add_argument("--include-cv", action="store_true")
    result.add_argument("--crossfit-folds", type=int, default=3)
    result.add_argument("--bootstrap-repeats", type=int, default=1000)
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.sequence_length < 1:
        raise ValueError("Component evidence uses a positive, fixed prediction horizon")
    best = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    if not targets:
        raise ValueError("At least one target column is required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}

    for target in targets:
        trait_dir = output_dir / _slug(target) / "evaluation"
        config = ExperimentConfig(
            data_dir=args.data_dir,
            output_dir=str(trait_dir),
            target_column=target,
            # Component traits are external-only here. Seed size has missing internal
            # BLUEs, so it must not be forced through the complete yield CV contract.
            require_complete_cv_target=False,
            sequence_length=args.sequence_length,
            splits=args.splits,
            seeds=args.seeds,
            stage_model=best.get("stage_model", "tcn_film"),
            stage_params=best.get("stage_params"),
            crossfit_folds=args.crossfit_folds,
            genotype_components=int(best.get("stage_params", {}).get("genotype_components", 32)),
            environment_components=int(best.get("environment_components", 12)),
            yield_mode=best.get("yield_mode"),
            yield_head=best.get("yield_head", "catboost"),
            yield_params=best.get("yield_params"),
            yield_genotype_components=best.get("yield_genotype_components"),
            bootstrap_repeats=args.bootstrap_repeats,
            device=args.device,
            paper_ablation=True,
            model_suite="biology",
            supplied_cv=args.include_cv,
            final_external=True,
        )
        print(f"Running frozen architecture for {target}.", flush=True)
        result = run_experiment(config)
        outputs[target] = {name: str(path) for name, path in result.items()}

    (output_dir / "component_run_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Frozen-architecture component-trait evidence",
                "targets": targets,
                "sequence_length": args.sequence_length,
                "include_cv": args.include_cv,
                "source_best_config": str(Path(args.best_config).resolve()),
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
