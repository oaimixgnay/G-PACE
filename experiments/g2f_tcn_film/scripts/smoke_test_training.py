#!/usr/bin/env python3
"""Small CPU integration test using synthetic data, never official G2F labels."""
from pathlib import Path
import sys
import tempfile
PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / 'src'), str(PACKAGE / 'scripts')]
import numpy as np
import pandas as pd
import torch
import joblib
from gpace.data import GPACEData, EXPOSURE_COLUMNS
from gpace.models import make_exposure_model, YieldModel
import run_g2f_tcn_film as runner


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(2026)
    matrix = rng.integers(0, 3, size=(18, 40)).astype(float)
    sequences = rng.normal(size=(8, 100, 10)).astype(np.float32)
    rows = []
    for e in range(8):
        for g in range(18):
            row = {'record_id': len(rows), 'Experiment': f'E{e}', 'Variety_ID': f'G{g}',
                   'environment_index': e, 'genotype_index': g, 'Year': 2020 + (e % 4) if e < 6 else 2024,
                   'grain.yield': 5.0 + e * 0.2 + matrix[g, 0] * 0.1}
            for i, name in enumerate(EXPOSURE_COLUMNS):
                row[name] = float(e + g * 0.02 + i * 0.1) if e < 6 else np.nan
            rows.append(row)
    records = pd.DataFrame(rows)
    data = GPACEData(records, tuple(f'G{g}' for g in range(18)), matrix, tuple(f'M{i}' for i in range(40)),
                    tuple(f'E{e}' for e in range(8)), sequences, np.full(8, 100), tuple(f'W{i}' for i in range(10)),
                    pd.DataFrame(), pd.DataFrame(), {'synthetic_smoke_test': True})
    train, test = np.arange(108), np.arange(108, 144)
    config = runner.verify_config(PACKAGE / 'config/g2f_tcn_film_frozen.json')
    params = {**config['stage_params'], 'epochs': 2, 'patience': 2, 'batch_size': 32, 'amp': False, 'device': 'cpu', 'verbose': True}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        oof = runner.crossfit(data, train, params, 2, 2026, folder / 'crossfit', False)
        stage = make_exposure_model('tcn_film', random_state=3025, **params).fit(data, train)
        predicted = stage.predict(data, test)
        assert oof.shape == (108, 11) and predicted.shape == (36, 11)
        assert np.isfinite(oof).all() and np.isfinite(predicted).all()
        runner.save_stage(stage, params, folder / 'stage')
        weights = torch.load(folder / 'stage/stage_weights.pt', map_location='cpu', weights_only=True)
        assert set(weights) == set(stage.network_.state_dict())
        preprocess = joblib.load(folder / 'stage/stage_preprocessing.joblib')
        assert preprocess['target_scaler'].n_features_in_ == 11
        model = YieldModel('mean_residual_hybrid', genotype_components=16, environment_components=12,
                           head='catboost', device='cpu', random_state=2026,
                           head_params={**config['yield_params'], 'iterations': 20, 'task_type': 'CPU'})
        model.fit(data, train, oof)
        yhat = model.predict(data, test, predicted)
        joblib.dump(model, folder / 'yield.joblib')
        np.testing.assert_allclose(joblib.load(folder / 'yield.joblib').predict(data, test, predicted), yhat)
        assert np.isfinite(yhat).all()
    print('[PASS] Synthetic TCN–FiLM, dual-axis crossfit, CatBoost and learned-weight saving; no actual G2F training.', flush=True)


if __name__ == '__main__':
    main()
