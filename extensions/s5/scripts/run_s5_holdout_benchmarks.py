#!/usr/bin/env python3
"""Add genuine fixed-configuration benchmarks to the existing day-100 S5 study.

Uses the project's original loader, grouped splits, model implementations and
metrics. Original core results are read only. Completed fits are resumable.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd

NEW_MODELS = ("reaction_norm_kernel", "ceris_jgra", "geformer_style")
CORE_MODELS = ("e2e", "e2e_catboost", "hybrid_catboost", "mean_residual_hybrid_catboost")
METRICS = ("pcc", "r2", "rmse", "within_environment_pcc",
           "within_environment_spearman", "top20_recall")
DEFAULT_SEEDS = (2026, 2027, 2028)
INPUT_FILES = (
    "7a-Genotyping_50K_41722.csv", "7c-Genotyping_50K_41722_nG.csv",
    "2b-GrainYield_components_BLUEs_level-1.csv", "3-Indices_GenoEnv_level-1.csv",
    "1-Env_variables_daily.csv", "9a-CrossValidation_Sets_Geno.csv",
    "9b-CrossValidation_Sets_Exp.csv",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def json_scores(scores):
    return {k: float(v) if np.isfinite(v) else None for k, v in scores.items()}


def summarize(frame):
    grouped = frame.groupby(["heldout_group", "model"])[list(METRICS)].agg(
        ["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index().sort_values(["heldout_group", "model"])


def expected_rows(data, indices):
    return data.records.loc[indices, ["record_id", "Experiment", "Variety_ID", "grain.yield"]].rename(
        columns={"grain.yield": "observed"}).reset_index(drop=True)


def verify_rows(actual, expected, context):
    columns = ["record_id", "Experiment", "Variety_ID", "observed"]
    if not set(columns).issubset(actual.columns):
        raise ValueError(f"{context}: missing prediction identity/target columns")
    a = actual[columns].copy()
    b = expected[columns].copy()
    for frame in (a, b):
        for col in columns[:3]:
            frame[col] = frame[col].astype(str)
    if a.record_id.duplicated().any() or len(a) != len(b):
        raise ValueError(f"{context}: duplicated or different test records")
    a = a.sort_values("record_id").reset_index(drop=True)
    b = b.sort_values("record_id").reset_index(drop=True)
    if not a[columns[:3]].equals(b[columns[:3]]) or not np.allclose(
            a.observed.astype(float), b.observed.astype(float), rtol=1e-8, atol=1e-8):
        raise ValueError(f"{context}: original and new test identities/observations differ")


def load_core(root, scheme, tasks, data, seeds, sequence_length, require_predictions=False):
    base = root / scheme
    manifest_path = base / "evaluation/run_manifest.json"
    provenance = {"scheme": scheme, "core_source": str(base), "prediction_membership_checked": False}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        if config.get("sequence_length") != sequence_length or sorted(config.get("seeds", [])) != sorted(seeds):
            raise ValueError(f"{manifest_path}: original sequence length/seeds differ")
        if config.get("environment_holdout") != scheme:
            raise ValueError(f"{manifest_path}: original holdout scheme differs")
        provenance["manifest_sha256"] = sha256(manifest_path)
    elif "day100" not in str(root) or sequence_length != 100:
        raise ValueError(f"No manifest confirming day-100 core results under {base}")
    else:
        print(f"[WARN] No {scheme} core manifest; using explicitly named day100 directory.", flush=True)
    metric_path = base / "evaluation/yield_metrics.csv"
    summary_path = base / "report/environment_holdout_summary.csv"
    groups = {task.group for _, task in tasks}
    if metric_path.exists():
        raw = pd.read_csv(metric_path)
        raw = raw[raw.model.isin(CORE_MODELS) & raw.heldout_group.isin(groups)].copy()
        for group in groups:
            for model in CORE_MODELS:
                rows = raw[(raw.heldout_group == group) & (raw.model == model)]
                if len(rows) != len(seeds) or sorted(rows.seed.tolist()) != sorted(seeds):
                    raise ValueError(f"Incomplete/duplicate core seeds: {scheme}/{group}/{model}")
                if "holdout_scheme" in rows and not rows.holdout_scheme.eq(scheme).all():
                    raise ValueError("Core metric scheme differs")
        summary = summarize(raw)
        provenance["metrics_sha256"] = sha256(metric_path)
    elif summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary.model.isin(CORE_MODELS) & summary.heldout_group.isin(groups)].copy()
        if summary.duplicated(["heldout_group", "model"]).any():
            raise ValueError("Duplicate original core summary rows")
        if len(summary) != len(groups) * len(CORE_MODELS):
            raise ValueError("Original core summary lacks models/groups")
        for metric in METRICS:
            if not summary[f"{metric}_count"].eq(len(seeds)).all():
                raise ValueError(f"Original {scheme} {metric} summary lacks three seed results")
        provenance["summary_sha256"] = sha256(summary_path)
        print(f"[WARN] {scheme}: merging original three-seed summary; raw core metrics unavailable.", flush=True)
    else:
        raise FileNotFoundError(f"No original metrics or summary under {base}")
    pred_path = base / "evaluation/yield_predictions.csv"
    if pred_path.exists():
        predictions = pd.read_csv(pred_path, dtype={"record_id": str, "Experiment": str, "Variety_ID": str})
        for split, task in tasks:
            expected = expected_rows(data, task.test)
            for model in CORE_MODELS:
                for seed in seeds:
                    rows = predictions[(predictions.model == model) & (predictions.seed == seed)
                                       & (predictions.heldout_group == task.group)]
                    verify_rows(rows, expected, f"core {scheme}/{task.group}/{model}/{seed}")
                    if not rows.scenario.eq(task.scenario).all() or not rows.split.eq(split).all():
                        raise ValueError("Original split/scenario differs from current split function")
        provenance["prediction_membership_checked"] = True
        provenance["predictions_sha256"] = sha256(pred_path)
    elif require_predictions:
        raise FileNotFoundError(f"Required original prediction file absent: {pred_path}")
    else:
        print(f"[WARN] {scheme}: no original predictions; exact core test membership cannot be checked.", flush=True)
    reference = root / "reference_cv_records.csv"
    proof_path = root / "reference_provenance.json"
    if reference.exists() and proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        for name, metadata in proof["files"].items():
            if sha256(root / name) != metadata["sha256"]:
                raise ValueError(f"Bundled original reference checksum differs: {name}")
        ref = pd.read_csv(reference, dtype={"record_id": str, "Experiment": str, "Variety_ID": str})
        for _, task in tasks:
            subset = ref[ref[f"{scheme}_group"].eq(task.group)]
            verify_rows(subset, expected_rows(data, task.test), f"archived CV reference {scheme}/{task.group}")
        provenance["reference_dataset_checked"] = True
        provenance["reference_provenance_sha256"] = sha256(proof_path)
        print(f"[REFERENCE OK] {scheme}: current CV records/targets and group membership match archived reference.", flush=True)
    return summary, provenance


def choose_core_root(requested, scheme, fallback):
    """Prefer server results; use actual archived summaries only if paths are absent."""
    for candidate in (requested, fallback):
        if any((candidate / scheme / sub).is_file() for sub in (
                "evaluation/yield_metrics.csv", "report/environment_holdout_summary.csv")):
            if candidate != requested:
                print(f"[CORE FALLBACK] {scheme}: server results absent; using bundled original summary {candidate / scheme}", flush=True)
            return candidate
    raise FileNotFoundError(f"No original {scheme} results under {requested} or bundled reference {fallback}")


def completed_run(directory, fingerprint, expected, identity):
    marker = directory / "complete.json"
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint inputs/config/code changed: {directory}. Use a new --output-dir.")
    for name, checksum in saved["files"].items():
        path = directory / name
        if not path.exists() or sha256(path) != checksum:
            raise ValueError(f"Corrupt checkpoint: {path}. Move this run directory aside and retry.")
    pred = pd.read_csv(directory / "predictions.csv", dtype={"record_id": str, "Experiment": str, "Variety_ID": str})
    verify_rows(pred, expected, f"checkpoint {directory}")
    if not np.isfinite(pred.predicted).all():
        raise ValueError("Checkpoint predictions are nonfinite")
    for key, value in identity.items():
        if not pred[key].eq(value).all():
            raise ValueError(f"Checkpoint identity differs: {key}")
    scores = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {**identity, **scores}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--data-dir", type=Path, default=Path("../玉米NG原始data"))
    p.add_argument("--best-config", type=Path, default=Path("outputs/published_benchmarks_day100/search/best_benchmark_config.json"))
    p.add_argument("--core-root", type=Path, default=Path("outputs/environment_holdouts_v2_day100"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/environment_holdouts_benchmarks_day100"))
    p.add_argument("--sequence-length", type=int, default=100, choices=[100])
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--models", default=",".join(NEW_MODELS))
    p.add_argument("--schemes", default="region,stress")
    p.add_argument("--groups", help="Optional comma-separated group names for a limited run")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--check-only", action="store_true", help="Load data and validate all inputs/partitions without fitting")
    p.add_argument("--import-check", action="store_true", help="Verify bundled original modules without loading data or training")
    p.add_argument("--require-core-predictions", action="store_true")
    return p


def main():
    args = parser().parse_args()
    root = args.project_root.resolve()
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    sys.path.insert(0, str(vendor))
    from _s5_gpace.data import load_gpace_data
    from _s5_gpace.splits import make_environment_group_holdouts
    from _s5_gpace.benchmarks import make_published_benchmark
    from _s5_gpace.metrics import regression_metrics
    print(f"[IMPORT OK] Bundled original modules: {vendor / '_s5_gpace'}", flush=True)
    if args.import_check:
        print("[OK] Data/splits/benchmarks/metrics and their dependencies imported; no models fitted.", flush=True)
        return

    def resolve(path):
        return (path if path.is_absolute() else root / path).resolve()

    data_dir, config_path = resolve(args.data_dir), resolve(args.best_config)
    core_root, output = resolve(args.core_root), resolve(args.output_dir)
    if output == core_root or core_root in output.parents or output in core_root.parents:
        raise ValueError("Use a separate output directory; original core results must remain read only")
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    schemes = tuple(x.strip() for x in args.schemes.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if sorted(seeds) != list(DEFAULT_SEEDS):
        raise ValueError("S5 requires exactly seeds 2026,2027,2028")
    if not models or len(set(models)) != len(models) or not set(models).issubset(NEW_MODELS):
        raise ValueError(f"Allowed unique models: {NEW_MODELS}")
    if not schemes or len(set(schemes)) != len(schemes) or not set(schemes).issubset({"region", "stress"}):
        raise ValueError("Allowed schemes: region,stress")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = config.get("selection", {})
    if selection.get("sequence_length") != 100 or selection.get("external_data_used") is not False:
        raise ValueError("Frozen benchmark selection must confirm day 100 and external_data_used=false")
    params = {model: config["models"][model] for model in models}
    # Resolve auto explicitly: the original GEFormer-style implementation expects
    # a concrete torch device when CUDA is unavailable.
    device = args.device
    if "geformer_style" in models:
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Select --device cpu explicitly if required.")
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    elif device == "auto":
        device = "cpu"
    print("[LOAD] Original data loader, sequence_length=100", flush=True)
    data = load_gpace_data(data_dir, sequence_length=args.sequence_length)
    all_tasks = {scheme: list(enumerate(make_environment_group_holdouts(data, scheme), start=101))
                 for scheme in schemes}
    allowed = set(args.groups.split(",")) if args.groups else None
    tasks_by_scheme = {scheme: [(n, t) for n, t in tasks if allowed is None or t.group in allowed]
                       for scheme, tasks in all_tasks.items()}
    if allowed and allowed - {t.group for tasks in all_tasks.values() for _, t in tasks}:
        raise ValueError("Unknown --groups value")
    tasks_by_scheme = {s: tasks for s, tasks in tasks_by_scheme.items() if tasks}
    if not tasks_by_scheme:
        raise ValueError("No selected tasks")
    print("[CHECK] Matching original core results/partitions", flush=True)
    cores, core_provenance = {}, {}
    core_sources = {}
    for scheme, tasks in tasks_by_scheme.items():
        core_sources[scheme] = choose_core_root(
            core_root, scheme, vendor.parent / "core_reference_day100")
        cores[scheme], core_provenance[scheme] = load_core(
            core_sources[scheme], scheme, tasks, data, seeds, 100, args.require_core_predictions)
    total = sum(len(tasks) for tasks in tasks_by_scheme.values()) * len(models) * len(seeds)
    for scheme, tasks in tasks_by_scheme.items():
        for split, task in tasks:
            print(f"[TASK] {scheme}/{task.group} split={split} train={len(task.train)} test={len(task.test)}", flush=True)
    print(f"[PLAN] {total} fits; models={models}; seeds={seeds}; device={device}", flush=True)
    if args.check_only:
        print("[OK] Checks passed; no models fitted.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another runner is using {output}") from None
        # Hash raw inputs and all actual gpace source files once, to prevent reusing a
        # checkpoint after changes to preprocessing, splits, models, metrics or data.
        print("[HASH] Recording data/source provenance", flush=True)
        source_hashes = {str(p.relative_to(vendor)): sha256(p) for p in sorted((vendor / "_s5_gpace").rglob("*.py"))}
        data_hashes = {name: sha256(data_dir / name) for name in INPUT_FILES}
        shared = {"sequence_length": 100, "params": params, "device": device,
                  "source_hashes": source_hashes, "data_hashes": data_hashes,
                  "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path)}
        manifest = {**shared, "seeds": seeds, "models": models, "schemes": list(tasks_by_scheme),
                    "core_provenance": core_provenance, "data_audit": data.audit,
                    "complete": False, "tasks": [], "metric_summary": "mean and sample SD over three seeds"}
        rows, run_dirs = {scheme: [] for scheme in tasks_by_scheme}, []
        counter, start_time = 0, time.monotonic()
        for scheme, tasks in tasks_by_scheme.items():
            for split, task in tasks:
                expected = expected_rows(data, task.test)
                train = expected_rows(data, task.train)
                partition = {"scheme": scheme, "group": task.group, "split": split,
                             "train_record_ids": train.record_id.astype(str).tolist(),
                             "test_record_ids": expected.record_id.astype(str).tolist(),
                             "train_environments": sorted(train.Experiment.unique().tolist()),
                             "test_environments": sorted(expected.Experiment.unique().tolist())}
                manifest["tasks"].append(partition)
                for model in models:
                    for seed in seeds:
                        counter += 1
                        identity = {"model": model, "scenario": task.scenario, "split": split,
                                    "seed": seed, "holdout_scheme": scheme, "heldout_group": task.group}
                        actual_seed = seed + split * 100 + NEW_MODELS.index(model)
                        fingerprint = digest({**shared, "params": params[model], "partition": partition, "identity": identity,
                                              "random_state": actual_seed})
                        directory = output / "runs" / scheme / task.group / model / str(seed)
                        run_dirs.append(directory)
                        cached = completed_run(directory, fingerprint, expected, identity)
                        if cached is not None:
                            print(f"[RESUME {counter}/{total}] {scheme}/{task.group}/{model}/{seed}", flush=True)
                            rows[scheme].append(cached)
                            continue
                        print(f"[FIT {counter}/{total}] {scheme}/{task.group}/{model}/{seed}", flush=True)
                        started = time.monotonic()
                        random.seed(actual_seed)
                        np.random.seed(actual_seed)
                        fitted = make_published_benchmark(model, random_state=actual_seed,
                                                           device=device, **params[model]).fit(data, task.train)
                        prediction = np.asarray(fitted.predict(data, task.test), dtype=float).reshape(-1)
                        if len(prediction) != len(expected) or not np.isfinite(prediction).all():
                            raise ValueError("Model produced invalid predictions")
                        frame = expected.copy()
                        frame["predicted"] = prediction
                        for key, value in identity.items():
                            frame[key] = value
                        scores = json_scores(regression_metrics(frame))
                        write_csv(directory / "predictions.csv", frame)
                        write_json(directory / "metrics.json", scores)
                        elapsed = time.monotonic() - started
                        write_json(directory / "complete.json", {
                            "fingerprint": fingerprint, "identity": identity,
                            "random_state": actual_seed, "elapsed_seconds": elapsed,
                            "files": {name: sha256(directory / name) for name in ("predictions.csv", "metrics.json")}})
                        rows[scheme].append({**identity, **scores})
                        del fitted, frame
                        gc.collect()
                        if "geformer_style" in models:
                            torch.cuda.empty_cache()
                        print(f"[DONE] {elapsed / 60:.1f} min; total elapsed {(time.monotonic() - start_time) / 60:.1f} min", flush=True)
                        write_csv(output / scheme / "evaluation/benchmark_yield_metrics.csv", pd.DataFrame(rows[scheme]))
                        write_json(output / "run_manifest.json", manifest)
        for scheme, metric_rows in rows.items():
            metrics = pd.DataFrame(metric_rows)
            if metrics.duplicated(["heldout_group", "model", "seed"]).any():
                raise ValueError("Duplicate new run identities")
            added = summarize(metrics)
            merged = pd.concat([cores[scheme], added], ignore_index=True).sort_values(["heldout_group", "model"])
            write_csv(output / scheme / "evaluation/benchmark_yield_metrics.csv", metrics)
            write_csv(output / scheme / "report/benchmark_holdout_summary.csv", added)
            write_csv(output / scheme / "report/environment_holdout_summary.csv", merged)
            pred_frames = [pd.read_csv(p / "predictions.csv", dtype={"record_id": str, "Experiment": str, "Variety_ID": str})
                           for p in run_dirs if p.relative_to(output).parts[1] == scheme]
            write_csv(output / scheme / "evaluation/benchmark_yield_predictions.csv", pd.concat(pred_frames, ignore_index=True))
        manifest["complete"] = True
        manifest["completed_fits"] = total
        manifest["full_s5_extension"] = (set(models) == set(NEW_MODELS) and set(tasks_by_scheme) == {"region", "stress"}
                                        and allowed is None)
        write_json(output / "run_manifest.json", manifest)
        archive = output / ("S5_7models_source_export.zip" if manifest["full_s5_extension"] else "S5_partial_source_export.zip")
        tmp_archive = archive.with_name(archive.name + ".tmp")
        with zipfile.ZipFile(tmp_archive, "w", zipfile.ZIP_DEFLATED) as z:
            for scheme in rows:
                for sub in ("evaluation", "report"):
                    for path in sorted((output / scheme / sub).glob("*.csv")):
                        z.write(path, path.relative_to(output))
            z.write(output / "run_manifest.json", "run_manifest.json")
            z.write(config_path, "frozen_benchmark_config.json")
            for scheme, source in core_sources.items():
                for sub in ("report/environment_holdout_summary.csv", "evaluation/yield_metrics.csv"):
                    path = source / scheme / sub
                    if path.exists():
                        z.write(path, f"original_core/{scheme}/{sub}")
        os.replace(tmp_archive, archive)
        print(f"[COMPLETE] {total} fits, genuine three-seed summaries; export: {archive}", flush=True)


if __name__ == "__main__":
    main()
