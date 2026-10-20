"""Offline tests of configuration identity, holdout boundaries and resumption."""
import importlib.util
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import gpace.models as core
from gpace.data import EXPOSURE_COLUMNS
import run_g2f_tcn_film as runner
from merge_tcn_film_results import align_records, ORDER


class FakeStage:
    """Deterministic placeholder; these tests do not claim neural training."""
    def __init__(self, seed): self.seed = seed
    def fit(self, data, indices):
        self.offset = self.seed + float(np.sum(indices)) / 100.0
        return self
    def predict(self, data, indices):
        return np.tile((np.asarray(indices) + self.offset)[:, None], (1, len(EXPOSURE_COLUMNS)))


def fake_factory(kind, random_state, **kwargs):
    if kind != 'tcn_film': raise ValueError(kind)
    return FakeStage(random_state)


class ProtocolTests(unittest.TestCase):
    def test_frozen_european_stage_and_verified_g2f_yield(self):
        config = runner.verify_config(ROOT / 'config/g2f_tcn_film_frozen.json')
        self.assertEqual(config['stage_params']['width'], 64)
        self.assertEqual(config['stage_params']['depth'], 2)
        self.assertEqual(config['stage_params']['kernel_size'], 5)
        self.assertEqual(config['yield_params']['iterations'], 1400)

    def test_network_source_hash_ignores_python_ast_schema_version(self):
        source = (ROOT / 'src/gpace/models.py').read_text(encoding='utf-8')
        parsed = ast.parse(source)
        old_schema = copy.deepcopy(parsed)
        for node in ast.walk(old_schema):
            node._fields = tuple(field for field in node._fields if field != 'type_params')
        expected = runner.stage_network_hash()
        with patch.object(runner.ast, 'parse', return_value=old_schema):
            self.assertEqual(runner.stage_network_hash(), expected)
        with patch.object(runner.ast, 'dump', side_effect=AssertionError('Version-dependent AST serialization must not be used')):
            self.assertEqual(runner.stage_network_hash(), expected)

    def test_dual_axis_crossfit_matches_core_and_reuses_completed_folds(self):
        rows = pd.DataFrame([{'Variety_ID': f'G{g}', 'Experiment': f'E{e}'} for g in range(6) for e in range(6)])
        data = SimpleNamespace(records=rows)
        indices = rows.index.to_numpy()
        with patch.object(core, 'make_exposure_model', fake_factory):
            expected = core.crossfit_exposures(data, indices, 'tcn_film', 3, 2026)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'make_exposure_model', fake_factory), patch.object(runner, 'clear_accelerator', lambda: None):
            cache = Path(tmp)
            got = runner.crossfit(data, indices, {}, 3, 2026, cache, False)
            np.testing.assert_array_equal(expected, got)
            self.assertEqual(len(list(cache.glob('*.npz'))), 6)
            with patch.object(runner, 'make_exposure_model', side_effect=AssertionError('Resume unexpectedly refitted a fold')):
                resumed = runner.crossfit(data, indices, {}, 3, 2026, cache, True)
            np.testing.assert_array_equal(expected, resumed)
            with self.assertRaises(ValueError):
                runner.load_array(next(cache.glob('*.npz')), np.array([999]))

    def test_2024_records_never_enter_historical_training(self):
        rows = pd.DataFrame({'Year': [2014, 2022, 2023, 2024]})
        for name in EXPOSURE_COLUMNS: rows[name] = [1.0, 2.0, 3.0, np.nan]
        prepared = SimpleNamespace(data=SimpleNamespace(records=rows), training_indices=np.array([0, 1, 2]), external_indices=np.array([3]))
        tasks = runner.make_tasks(prepared, True)
        np.testing.assert_array_equal(tasks[0][1], [0, 1])
        np.testing.assert_array_equal(tasks[0][2], [2])
        np.testing.assert_array_equal(tasks[1][1], [0, 1, 2])
        self.assertTrue(all(3 not in train for _, train, _ in tasks))
        rows.loc[3, EXPOSURE_COLUMNS[0]] = 1000.0
        with self.assertRaises(ValueError): runner.make_tasks(prepared)

    def test_six_model_merge_refuses_changed_evaluation_labels(self):
        rows = [{'record_id': i, 'Experiment': 'E', 'Variety_ID': f'G{i}', 'observed': float(i),
                 'predicted': float(i) + 0.2, 'scenario': 'external_2024', 'seed': 2026, 'model': model}
                for model in ORDER for i in range(4)]
        baseline = pd.DataFrame(rows)
        new = baseline[baseline.model.eq('g_pace')].copy()
        self.assertEqual(len(align_records(new, baseline)), 24)
        wrong = baseline.copy()
        wrong.loc[wrong.model.eq('ec_gblup') & wrong.record_id.eq(0), 'observed'] = 99
        with self.assertRaises(ValueError): align_records(new, wrong)


if __name__ == '__main__':
    unittest.main()
