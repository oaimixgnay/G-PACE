from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpace.data import EXPOSURE_COLUMNS, load_gpace_data  # noqa: E402
from gpace.metrics import exposure_metrics  # noqa: E402
from gpace.models import make_exposure_model  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Create exhaustive genotype-held-out stage predictions for every internal "
            "genotype. This is a frozen-stage-model robustness analysis, not tuning."
        )
    )
    result.add_argument("--data-dir", required=True)
    result.add_argument("--best-config", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--sequence-length", type=int, default=100)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--seeds", default="2026,2027,2028")
    result.add_argument("--device", default="cuda")
    return result


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("At least one seed is required")
    return result


def _fold_assignments(data, genotype_ids: np.ndarray, folds: int, seed: int) -> dict[str, int]:
    metadata = data.genotype_partitions.loc[:, ["Variety_ID", "genetic_group"]].copy()
    metadata["Variety_ID"] = metadata["Variety_ID"].astype(str)
    groups = metadata.set_index("Variety_ID")["genetic_group"].reindex(genotype_ids).fillna("unknown")
    assignments: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for group in sorted(groups.unique()):
        members = groups.loc[groups.eq(group)].index.to_numpy(dtype=str)
        members = rng.permutation(members)
        for fold, block in enumerate(np.array_split(members, folds)):
            assignments.update({str(item): fold for item in block})
    counts = pd.Series(assignments).value_counts().sort_index()
    if len(counts) != folds or counts.max() - counts.min() > max(2, len(genotype_ids) // folds):
        raise RuntimeError("Could not construct balanced genotype folds")
    return assignments


def _stage_kwargs(best: dict, device: str) -> tuple[str, dict]:
    kind = str(best.get("stage_model", "tcn_film"))
    params = dict(best.get("stage_params") or {})
    params["device"] = device
    return kind, params


def main() -> None:
    args = parser().parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    data = load_gpace_data(args.data_dir, sequence_length=args.sequence_length)
    cv_indices = data.records.index[data.records["scenario"].eq("cross_validation")].to_numpy(dtype=int)
    cv = data.records.loc[cv_indices]
    genotype_ids = np.sort(cv["Variety_ID"].astype(str).unique())
    environment_ids = np.sort(cv["Experiment"].astype(str).unique())
    if len(cv_indices) != len(genotype_ids) * len(environment_ids):
        raise ValueError("Internal panel must be complete before exhaustive genotype cross-fitting")
    kind, params = _stage_kwargs(best, args.device)
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    assignments = _fold_assignments(data, genotype_ids, args.folds, seed=20260915)

    for seed in _seeds(args.seeds):
        for fold in range(args.folds):
            heldout = {genotype for genotype, assigned in assignments.items() if assigned == fold}
            test_indices = cv.index[cv["Variety_ID"].isin(heldout)].to_numpy(dtype=int)
            train_indices = cv.index[~cv["Variety_ID"].isin(heldout)].to_numpy(dtype=int)
            print(
                f"Stage cross-fit seed={seed}, fold={fold + 1}/{args.folds}, "
                f"train={len(train_indices)}, test={len(test_indices)}",
                flush=True,
            )
            model = make_exposure_model(kind, random_state=seed * 100 + fold, **params).fit(data, train_indices)
            predicted = model.predict(data, test_indices)
            observed = data.records.loc[test_indices, EXPOSURE_COLUMNS].to_numpy(dtype=float)
            scores = exposure_metrics(observed, predicted)
            metric_rows.append(
                {
                    "seed": seed,
                    "fold": fold + 1,
                    "heldout_genotypes": len(heldout),
                    "train_records": len(train_indices),
                    "test_records": len(test_indices),
                    **scores,
                }
            )
            frame = data.records.loc[test_indices, ["record_id", "Experiment", "Variety_ID"]].copy()
            for position, name in enumerate(EXPOSURE_COLUMNS):
                frame[f"observed__{name}"] = observed[:, position]
                frame[f"predicted__{name}"] = predicted[:, position]
            frame["scenario"] = "genotype_5fold_oof"
            frame["split"] = fold + 1
            frame["seed"] = seed
            frame["holdout_scheme"] = "complete_genotype_5fold"
            frame["heldout_group"] = f"fold_{fold + 1}"
            predictions.append(frame)

    result = pd.concat(predictions, ignore_index=True)
    expected = len(cv_indices) * len(_seeds(args.seeds))
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} OOF stage predictions, found {len(result)}")
    result.to_csv(output_dir / "exposure_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "exposure_metrics.csv", index=False)
    pd.DataFrame(
        {"Variety_ID": genotype_ids, "fold": [assignments[item] + 1 for item in genotype_ids]}
    ).to_csv(output_dir / "genotype_fold_assignments.csv", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Exhaustive no-genotype-leakage stage-exposure robustness analysis",
                "data_dir": str(Path(args.data_dir).resolve()),
                "best_config": str(Path(args.best_config).resolve()),
                "stage_model": kind,
                "stage_params": params,
                "sequence_length": args.sequence_length,
                "folds": args.folds,
                "seeds": list(_seeds(args.seeds)),
                "internal_records": len(cv_indices),
                "genotypes": len(genotype_ids),
                "environments": len(environment_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"exposure predictions: {output_dir / 'exposure_predictions.csv'}")
    print(f"metrics: {output_dir / 'exposure_metrics.csv'}")


if __name__ == "__main__":
    main()
