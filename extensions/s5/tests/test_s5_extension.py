"""Synthetic validation only. Uses the original modules bundled with this package.

Usage: python tests/test_s5_extension.py
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "vendor"))
from _s5_gpace.data import GPACEData
from _s5_gpace.splits import make_environment_group_holdouts
from _s5_gpace.benchmarks import make_published_benchmark
from _s5_gpace.metrics import regression_metrics

spec = importlib.util.spec_from_file_location("s5_runner", PACKAGE / "scripts/run_s5_holdout_benchmarks.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_data():
    rng = np.random.default_rng(471)
    experiments = ("CamA", "GaiB", "KarC", "CraD", "DebE", "MarF", "GraG", "GraH")
    varieties = tuple(f"g{i}" for i in range(8))
    genotype = rng.integers(0, 3, (8, 24)).astype(np.float32)
    sequences = rng.normal(size=(8, 100, 3)).astype(np.float32)
    sequences += np.arange(8)[:, None, None] * 0.3
    records = []
    for e, experiment in enumerate(experiments):
        for g, variety in enumerate(varieties):
            records.append({"record_id": len(records), "Experiment": experiment,
                            "Variety_ID": variety, "environment_index": e, "genotype_index": g,
                            "grain.yield": 2 + e * 0.3 + genotype[g, :4].sum() * 0.15 + rng.normal() * 0.07,
                            "scenario": "cross_validation"})
    partitions = pd.DataFrame({"Experiment": experiments,
                               "EC": ["WD-HOT", "WD-HOT(day)", "WW-COOL", "WW-HOT",
                                      "WW-HOT(day)", "WD-HOT", "WW-COOL", "WW-HOT"]})
    return GPACEData(pd.DataFrame(records), varieties, genotype,
                    tuple(f"m{i}" for i in range(24)), experiments, sequences,
                    np.full(8, 100), ("a", "b", "c"), pd.DataFrame(), partitions,
                    {"synthetic_test_only": True})


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.temp = tempfile.TemporaryDirectory(prefix="s5_synthetic_test_")
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        for name in runner.INPUT_FILES:
            (self.root / "data" / name).write_text("synthetic test placeholder only", encoding="utf-8")
        self.config = json.loads((PACKAGE / "config/frozen_benchmark_config.json").read_text())
        (self.root / "config.json").write_text(json.dumps(self.config))
        for scheme in ("region", "stress"):
            frames, metrics = [], []
            for split, task in enumerate(make_environment_group_holdouts(self.data, scheme), 101):
                for model in runner.CORE_MODELS:
                    for seed in runner.DEFAULT_SEEDS:
                        frame = runner.expected_rows(self.data, task.test)
                        frame["predicted"] = frame.observed * 0.95 + np.sin(frame.record_id) * 0.05
                        identity = {"model": model, "seed": seed, "scenario": task.scenario,
                                    "heldout_group": task.group, "holdout_scheme": scheme, "split": split}
                        for key, value in identity.items():
                            frame[key] = value
                        frames.append(frame)
                        metrics.append({**identity, **regression_metrics(frame)})
            base = self.root / "core_day100" / scheme
            runner.write_csv(base / "evaluation/yield_predictions.csv", pd.concat(frames))
            runner.write_csv(base / "evaluation/yield_metrics.csv", pd.DataFrame(metrics))
            runner.write_json(base / "evaluation/run_manifest.json", {"config": {
                "sequence_length": 100, "seeds": runner.DEFAULT_SEEDS, "environment_holdout": scheme}})

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, models="reaction_norm_kernel,ceris_jgra", extra=()):
        argv = ["runner", "--project-root", str(self.root), "--data-dir", "data",
                "--best-config", "config.json", "--core-root", "core_day100",
                "--output-dir", "results", "--device", "cpu", "--models", models, *extra]
        with patch.object(sys, "argv", argv), patch("_s5_gpace.data.load_gpace_data", return_value=self.data), contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def test_real_models_and_resume_and_merge(self):
        self.invoke()
        for scheme, groups in (("region", 3), ("stress", 5)):
            merged = pd.read_csv(self.root / f"results/{scheme}/report/environment_holdout_summary.csv")
            self.assertEqual(len(merged), groups * 6)
            self.assertTrue(merged.pcc_count.eq(3).all())
            self.assertFalse(merged.duplicated(["heldout_group", "model"]).any())
            metrics = pd.read_csv(self.root / f"results/{scheme}/evaluation/benchmark_yield_metrics.csv")
            self.assertEqual(len(metrics), groups * 2 * 3)
            original = pd.read_csv(self.root / f"core_day100/{scheme}/evaluation/yield_metrics.csv")
            core_summary = runner.summarize(original)
            pd.testing.assert_frame_equal(
                merged[merged.model.isin(runner.CORE_MODELS)].reset_index(drop=True),
                core_summary.reset_index(drop=True), check_exact=False, check_dtype=False, rtol=1e-12)
        with patch("_s5_gpace.benchmarks.make_published_benchmark", side_effect=AssertionError("Resume attempted fit")):
            self.invoke()
        checkpoints = list((self.root / "results/runs").rglob("complete.json"))
        self.assertEqual(len(checkpoints), 48)
        with zipfile.ZipFile(self.root / "results/S5_partial_source_export.zip") as z:
            self.assertIn("region/report/environment_holdout_summary.csv", z.namelist())
            manifest = json.loads(z.read("run_manifest.json"))
            self.assertFalse(manifest["full_s5_extension"])
            self.assertEqual(manifest["completed_fits"], 48)
        # Corruption must be rejected instead of silently using altered results.
        pred = checkpoints[0].parent / "predictions.csv"
        with pred.open("a") as f:
            f.write("corrupted\n")
        with self.assertRaisesRegex(ValueError, "Corrupt checkpoint"):
            self.invoke()

    def test_extend_partial_without_refitting(self):
        self.invoke(models="reaction_norm_kernel", extra=("--schemes", "region", "--groups", "eastern_europe"))
        self.assertEqual(len(list((self.root / "results/runs").rglob("complete.json"))), 3)
        actual_factory = make_published_benchmark
        calls = []
        def record_factory(name, **kwargs):
            calls.append(name)
            return actual_factory(name, **kwargs)
        with patch("_s5_gpace.benchmarks.make_published_benchmark", side_effect=record_factory):
            self.invoke(models="reaction_norm_kernel,ceris_jgra", extra=("--schemes", "region", "--groups", "eastern_europe"))
        self.assertEqual(calls, ["ceris_jgra"] * 3)

    def test_wrong_core_membership_and_window(self):
        path = self.root / "core_day100/region/evaluation/yield_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "observed"] += 1
        runner.write_csv(path, frame)
        with self.assertRaisesRegex(ValueError, "observations differ"):
            self.invoke(extra=("--check-only",))
        path = self.root / "core_day100/region/evaluation/run_manifest.json"
        runner.write_json(path, {"config": {"sequence_length": 0, "seeds": runner.DEFAULT_SEEDS}})
        with self.assertRaisesRegex(ValueError, "sequence length/seeds differ"):
            self.invoke(extra=("--check-only",))

    def test_geformer_frozen_constructor(self):
        model = make_published_benchmark("geformer_style", random_state=2026,
                                         device="cpu", **self.config["models"]["geformer_style"])
        self.assertEqual(model.epochs, 120)
        self.assertEqual(model.genotype_components, 32)
        self.assertEqual(model.dropout, 0.05)

    def test_sd_is_sample_sd(self):
        frame = pd.DataFrame([{"heldout_group": "test", "model": "test", **{m: x for m in runner.METRICS}}
                              for x in (1.0, 2.0, 3.0)])
        summary = runner.summarize(frame)
        self.assertEqual(summary.pcc_std.iloc[0], 1.0)
        self.assertEqual(summary.pcc_mean.iloc[0], 2.0)

    def test_import_without_server_gpace_splits(self):
        # Reproduce the reported server layout: an existing gpace package with no
        # splits module. The runner must still use its complete isolated bundle.
        old_gpace = self.root / "src/gpace"
        old_gpace.mkdir(parents=True)
        (old_gpace / "__init__.py").write_text("raise RuntimeError('Wrong gpace imported')\n")
        env = {**os.environ, "PYTHONPATH": str(self.root / "src")}
        result = subprocess.run([sys.executable, str(PACKAGE / "scripts/run_s5_holdout_benchmarks.py"),
                                 "--project-root", str(self.root), "--import-check"],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[IMPORT OK]", result.stdout)

    def test_actual_archived_reference_fallback(self):
        fallback = PACKAGE / "core_reference_day100"
        ref = pd.read_csv(fallback / "reference_cv_records.csv",
                          dtype={"Experiment": str, "Variety_ID": str})
        records = ref.rename(columns={"observed": "grain.yield"}).copy()
        records["scenario"] = "cross_validation"
        envs = ref[["Experiment", "stress_group"]].drop_duplicates().rename(columns={"stress_group": "EC"})
        envs["EC"] = envs.EC.str.replace("_", "-", regex=False)
        data = SimpleNamespace(records=records, environment_partitions=envs)
        for scheme, n in (("region", 12), ("stress", 20)):
            chosen = runner.choose_core_root(self.root / "absent_core_day100", scheme, fallback)
            self.assertEqual(chosen, fallback)
            tasks = list(enumerate(make_environment_group_holdouts(data, scheme), 101))
            summary, provenance = runner.load_core(chosen, scheme, tasks, data, runner.DEFAULT_SEEDS, 100)
            self.assertEqual(len(summary), n)
            self.assertTrue(provenance["reference_dataset_checked"])
            self.assertFalse(provenance["prediction_membership_checked"])
        # Server results remain preferred. Invalid existing results are never
        # silently replaced by a different reference.
        self.assertEqual(runner.choose_core_root(self.root / "core_day100", "region", fallback),
                         self.root / "core_day100")
        corrupt = self.root / "corrupt_reference_day100"
        shutil.copytree(fallback, corrupt)
        with (corrupt / "reference_cv_records.csv").open("a") as f:
            f.write("\n")
        tasks = list(enumerate(make_environment_group_holdouts(data, "region"), 101))
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            runner.load_core(corrupt, "region", tasks, data, runner.DEFAULT_SEEDS, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
