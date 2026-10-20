"""Reproduce day-100 grouped holdouts for the omitted EC-GBLUP comparator.

Uses the unchanged original GPACE data loader, split logic, benchmark estimator,
and metric definitions. This source was bundled alongside the original data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT.parent / "vendor"
sys.path.insert(0, str(VENDOR))
from _s5_gpace.data import load_gpace_data
from _s5_gpace.splits import make_environment_group_holdouts
from _s5_gpace.benchmarks import DEFAULT_BENCHMARK_CONFIGS, make_published_benchmark
from _s5_gpace.metrics import regression_metrics

METRICS = ["pcc", "r2", "rmse", "within_environment_pcc",
           "within_environment_spearman", "top20_recall"]
DEFAULT_CONFIG = DEFAULT_BENCHMARK_CONFIGS["ec_gblup"]


def fingerprint(paths):
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(str(p.name).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def original_cv_check(data, reference):
    src = pd.read_csv(reference, dtype={"Variety_ID": str, "Experiment": str})
    current = data.records[data.records.scenario.eq("cross_validation")]
    assert len(src) == len(current) == 6150
    src = src.sort_values("record_id").reset_index(drop=True)
    current = current.sort_values("record_id").reset_index(drop=True)
    assert np.array_equal(src.record_id.to_numpy(), current.record_id.to_numpy())
    for col in ["Experiment", "Variety_ID"]:
        assert (src[col].astype(str) == current[col].astype(str)).all(), col
    assert np.allclose(src.observed, current["grain.yield"], atol=1e-8, rtol=1e-8)
    for scheme in ["region", "stress"]:
        for task in make_environment_group_holdouts(data, scheme):
            expected = set(src.loc[src[f"{scheme}_group"].eq(task.group), "record_id"])
            actual = set(data.records.loc[task.test, "record_id"])
            assert expected == actual, (scheme, task.group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "input_original_data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "ec_gblup_day100")
    p.add_argument("--reference", type=Path, default=ROOT.parents[2] / "data/source_data/reference_cv_records.csv")
    args = p.parse_args()
    begun = time.monotonic()
    data = load_gpace_data(args.data_dir, sequence_length=100)
    original_cv_check(data, args.reference)
    print("[CHECK] Day-100 CV records/targets and both holdout groupings match original S5 data", flush=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    input_hash = fingerprint([*sorted(args.data_dir.glob("*.csv")), args.reference,
                              VENDOR / "_s5_gpace/benchmarks.py",
                              VENDOR / "_s5_gpace/metrics.py"])
    manifest = {"model": "ec_gblup", "sequence_length": 100, "seeds": [2026, 2027, 2028],
                "config": DEFAULT_CONFIG, "input_sha256": input_hash,
                "source": "original GPACE ECGBLUP fixed defaults; no holdout tuning",
                "group_count": {"region": 3, "stress": 5}, "complete": False}
    all_rows = []
    for scheme in ["region", "stress"]:
        for split, task in enumerate(make_environment_group_holdouts(data, scheme), start=101):
            for seed in (2026, 2027, 2028):
                identity = {"model": "ec_gblup", "scenario": task.scenario,
                            "split": split, "seed": seed, "holdout_scheme": scheme,
                            "heldout_group": task.group}
                path = output / scheme / task.group / f"{seed}.csv"
                expected = data.records.loc[task.test, ["record_id", "Experiment",
                                                       "Variety_ID", "grain.yield"]].rename(
                                                            columns={"grain.yield": "observed"})
                if path.exists():
                    frame = pd.read_csv(path, dtype={"Variety_ID": str})
                    assert len(frame) == len(expected) and np.array_equal(
                        frame.record_id, expected.record_id), f"Checkpoint mismatch: {path}"
                    assert np.allclose(frame.observed, expected.observed, atol=1e-8)
                    for k, v in identity.items():
                        assert frame[k].eq(v).all(), (path, k)
                    print("[RESUME]", scheme, task.group, seed, flush=True)
                else:
                    random_state = seed + split * 100 + 3
                    fitted = make_published_benchmark(
                        "ec_gblup", random_state=random_state, device="cpu", **DEFAULT_CONFIG
                    ).fit(data, task.train)
                    frame = expected.copy()
                    frame["predicted"] = fitted.predict(data, task.test)
                    assert np.isfinite(frame.predicted).all()
                    for k, v in identity.items():
                        frame[k] = v
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    frame.to_csv(tmp, index=False, float_format="%.17g")
                    os.replace(tmp, path)
                    print("[DONE]", scheme, task.group, seed,
                          f"{time.monotonic()-begun:.1f}s elapsed", flush=True)
                all_rows.append({**identity, **regression_metrics(frame)})
    metrics = pd.DataFrame(all_rows)
    assert len(metrics) == 24
    assert not metrics.duplicated(["holdout_scheme", "heldout_group", "seed"]).any()
    for scheme in ["region", "stress"]:
        subset = metrics[metrics.holdout_scheme.eq(scheme)]
        summary = subset.groupby(["heldout_group", "model"])[METRICS].agg(
            ["mean", "std", "count"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.reset_index().to_csv(output / f"{scheme}_summary.csv", index=False)
        subset.to_csv(output / f"{scheme}_seed_metrics.csv", index=False)
    manifest["complete"] = True
    manifest["fits"] = len(metrics)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[COMPLETE]", len(metrics), "genuine held-out fits",
          f"{time.monotonic()-begun:.1f}s", flush=True)


if __name__ == "__main__":
    main()
