#!/usr/bin/env python3
"""Meaningful small-data checks; never interprets synthetic output as G2F evidence."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

import numpy as np
import pandas as pd

from gpace.data import GPACEData, EXPOSURE_COLUMNS
from gpace.g2f import G2FPreparedData, _FILES
from gpace.models import YieldModel
from ablation_models import StageAblationYieldModel
import paired_statistics as stats
import run_stage_ablation as runner


def fixture():
    rng = np.random.default_rng(47)
    genotype = rng.integers(0, 3, size=(12, 30)).astype(np.float32)
    seq = rng.normal(size=(8, 100, 10)).astype(np.float32)
    rows = []
    for env in range(8):
        for g in range(10 if env < 6 else 12):
            row = dict(record_id=len(rows), Experiment=f'E{env}', Variety_ID=f'G{g:02d}',
                       genotype_index=g, environment_index=env, Year=2018+env if env < 6 else 2024,
                       scenario='training' if env < 6 else 'external_2024')
            row['grain.yield'] = float(7 + env * .3 + genotype[g, :3].sum() * .2 + rng.normal(0, .1))
            row.update({k: rng.normal() if env < 6 else np.nan for k in EXPOSURE_COLUMNS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    train = frame.index[frame.scenario.eq('training')].to_numpy(int)
    test = frame.index[frame.scenario.eq('external_2024')].to_numpy(int)
    audit = {'training_gxe_means': len(train), 'external_2024_labeled_records': len(test),
             'external_2024_environment_count': 2}
    data = GPACEData(frame, tuple(f'G{x:02d}' for x in range(12)), genotype,
                    tuple(f'S{x}' for x in range(30)), tuple(f'E{x}' for x in range(8)),
                    seq, np.full(8, 100), tuple(f'W{x}' for x in range(10)),
                    pd.DataFrame(), pd.DataFrame(), audit)
    return G2FPreparedData(data, train, test)


def save_npz(path, indices, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, row_indices=indices, predictions=predictions)


def make_caches(prepared, source, seed):
    data, train, test = prepared.data, prepared.training_indices, prepared.external_indices
    rng = np.random.default_rng(seed)
    expected = np.zeros((len(train), 11))
    for axis, column in enumerate(('Variety_ID', 'Experiment')):
        groups = data.records.loc[train, column].astype(str).to_numpy()
        unique = np.unique(groups)
        np.random.default_rng(seed + axis).shuffle(unique)
        for fold, held in enumerate(np.array_split(unique, 3)):
            mask = np.isin(groups, held)
            values = rng.normal(size=(mask.sum(), 11))
            expected[mask] += values / 2
            path = source / 'runs/external_2024' / f'seed_{seed}' / 'crossfit' / f'axis{axis}_fold{fold}.npz'
            save_npz(path, train[mask], values)
    test_z = rng.normal(size=(len(test), 11))
    save_npz(source / 'runs/external_2024' / f'seed_{seed}' / 'test_exposures.npz', test, test_z)
    return expected, test_z


def model(config, seed, cls=StageAblationYieldModel, **kwargs):
    return cls(mode='mean_residual_hybrid', genotype_components=config['yield_genotype_components'],
               environment_components=config['yield_environment_components'], head='catboost',
               head_params={**config['yield_params'], 'task_type': 'CPU', 'thread_count': 1},
               random_state=seed, device='cpu', **kwargs)


def assert_raises(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return
    raise AssertionError(f'Expected {exception.__name__}')


def main():
    prepared = fixture()
    data, train, test = prepared.data, prepared.training_indices, prepared.external_indices
    config = json.loads((PACKAGE / 'config/g2f_tcn_film_frozen.json').read_text())
    config['yield_params']['iterations'] = 12
    config['yield_genotype_components'] = 3
    config['yield_environment_components'] = 2
    seeds = [2026, 2027, 2028]
    with tempfile.TemporaryDirectory(prefix='gpace_ablation_test_') as tmp:
        temp = Path(tmp)
        source, inputs, output = (temp / name for name in ('original/evaluation', 'inputs', 'output'))
        source.mkdir(parents=True)
        inputs.mkdir()
        source_predictions, all_new = [], []
        for seed in seeds:
            z, tz = make_caches(prepared, source, seed)
            cached, cached_test = runner.load_stage_features(data, train, test, source, seed)
            np.testing.assert_allclose(cached, z, atol=1e-15)
            np.testing.assert_allclose(cached_test, tz, atol=0)
            original = model(config, seed, YieldModel).fit(data, train, z)
            full = model(config, seed, include_stage=True).fit(data, train, z)
            no_stage = model(config, seed, include_stage=False).fit(data, train)
            np.testing.assert_allclose(original.predict(data, test, tz), full.predict(data, test, tz), atol=1e-12, rtol=0)
            direct_count = no_stage._design(data, test, None, False).shape[1]
            full_x = full._design(data, test, tz, False)
            np.testing.assert_allclose(full_x[:, :direct_count], no_stage._design(data, test, None, False), atol=0)
            assert full_x.shape[1] - direct_count == 11 * (1 + config['yield_genotype_components'])
            np.testing.assert_allclose(full.mean_regressor_.predict(full.inputs_.rows(data, test)[1]),
                                       no_stage.mean_regressor_.predict(no_stage.inputs_.rows(data, test)[1]), atol=0)
            np.testing.assert_allclose(no_stage.predict(data, test, tz * 1000), no_stage.predict(data, test), atol=0)
            changed = data.records.copy()
            changed.loc[test, 'grain.yield'] = -999
            changed_data = replace(data, records=changed)
            for m, include_stage in ((full, True), (no_stage, False)):
                values = tz if include_stage else None
                np.testing.assert_allclose(m.predict(data, test, values), m.predict(changed_data, test, values), atol=0)
            old_frame = data.records.loc[test, ['record_id', 'Experiment', 'Variety_ID', 'grain.yield']].rename(columns={'grain.yield': 'observed'}).copy()
            old_frame['predicted'], old_frame['model'], old_frame['seed'], old_frame['scenario'] = original.predict(data, test, tz), 'g_pace', seed, 'external_2024'
            source_predictions.append(old_frame)
            for include_stage in (True, False):
                f = runner.fit_one(data, train, test, z, tz, config, seed, include_stage,
                                   temp / 'checkpoint_test', 1, 'selftest', True)
                all_new.append(f)
        print('PASS: original full-path equality, actual feature deletion, unchanged mean branch, test-label isolation.')
        predictions = pd.concat(all_new, ignore_index=True)
        ensemble, env_frame, summary, paired = stats.compare(predictions, seeds, 200, 8)
        shuffled = predictions.sample(frac=1, random_state=19)
        pd.testing.assert_frame_equal(paired, stats.compare(shuffled, seeds, 200, 8)[3])
        equal = predictions.copy()
        lookup = equal[equal.model.eq(stats.FULL)].set_index(['seed', 'record_id']).predicted
        mask = equal.model.eq(stats.ABLATION)
        equal.loc[mask, 'predicted'] = [lookup.loc[(s, r)] for s, r in zip(equal.loc[mask, 'seed'], equal.loc[mask, 'record_id'])]
        diffs = stats.compare(equal, seeds, 200, 8)[3]
        np.testing.assert_allclose(diffs[['difference_full_minus_no_stage', 'ci_low', 'ci_high']], 0, atol=1e-12)
        assert_raises(lambda: stats.compare(predictions.iloc[1:], seeds))
        duplicated = pd.concat([predictions, predictions.iloc[[0]]])
        assert_raises(lambda: stats.compare(duplicated, seeds))
        constant = predictions.copy()
        constant.loc[constant.model.eq(stats.ABLATION) & constant.Experiment.eq('E6'), 'predicted'] = 8.
        constant_ci = stats.compare(constant, seeds, 200, 8)[3]
        assert constant_ci.loc[constant_ci.metric.eq('within_environment_pcc'), 'matched_valid_environments'].iat[0] == 1
        common = {k: np.logical_and.reduce([np.isfinite(env_frame[env_frame.model.eq(m)].set_index('Experiment').sort_index()[k]) for m in stats.MODELS])
                  for k in ('pcc', 'spearman', 'top10_efficiency', 'top20_efficiency')}
        base = env_frame[env_frame.model.eq(stats.FULL)].set_index('Experiment').sort_index()
        draws = np.array([[0, 0], [1, 0], [1, 1]])
        vector = stats.summarize_draws(base, draws, common)
        ids = base.index.to_numpy()
        full_ens = ensemble[ensemble.model.eq(stats.FULL)]
        for i, draw in enumerate(draws):
            chunks = [full_ens[full_ens.Experiment.eq(ids[j])].assign(Experiment=f'block{k}') for k, j in enumerate(draw)]
            brute = pd.concat(chunks)
            y, p = brute.observed.to_numpy(float), brute.predicted.to_numpy(float)
            np.testing.assert_allclose(vector['pcc'][i], stats.correlation(y, p), atol=1e-10)
            np.testing.assert_allclose(vector['rmse'][i], np.sqrt(np.mean((y-p)**2)), atol=1e-12)
            by_env = stats.environment_statistics(brute)
            np.testing.assert_allclose(vector['top20_recall'][i], by_env.top20_recall.mean(), atol=1e-12)
            np.testing.assert_allclose(vector['top20_gain'][i], by_env.top20_gain.mean(), atol=1e-12)
            fisher = np.tanh(np.mean(np.arctanh(np.clip(by_env.pcc, -.999999, .999999))))
            np.testing.assert_allclose(vector['within_environment_pcc'][i], fisher, atol=1e-12)
        print('PASS: paired intact-environment bootstrap agrees with explicit duplicate-block reconstruction; missing/duplicate rows rejected.')
        bad_path = runner.cache_paths(source, seeds[0], 3)[0]
        with np.load(bad_path) as c:
            rows, values = c['row_indices'].copy(), c['predictions'].copy()
        save_npz(bad_path, rows[::-1], values[::-1])
        assert_raises(lambda: runner.load_stage_features(data, train, test, source, seeds[0]))
        save_npz(bad_path, rows, values)
        print('PASS: changed cache row ordering rejected.')
        folder = temp / 'checkpoint_test/runs/seed_2028' / stats.ABLATION
        predictions_path = folder / 'predictions.csv'
        prior = predictions_path.read_bytes()
        runner.fit_one(data, train, test, None, None, config, 2028, False, temp / 'checkpoint_test', 1, 'selftest', True)
        assert predictions_path.read_bytes() == prior
        predictions_path.write_bytes(prior + b'\n')
        assert_raises(lambda: runner.fit_one(data, train, test, None, None, config, 2028, False, temp / 'checkpoint_test', 1, 'selftest', True))
        predictions_path.write_bytes(prior)
        print('PASS: verified completed fits resume; modified output rejected.')

        # Exercise the complete orchestration, source validation, locking,
        # persistence, report and lightweight ZIP with an injected tiny loader.
        release = temp / 'release'
        shutil.copytree(PACKAGE / 'src', release / 'src')
        shutil.copytree(PACKAGE / 'scripts', release / 'scripts')
        (release / 'config').mkdir()
        (release / 'config/g2f_tcn_film_frozen.json').write_text(json.dumps(config))
        old = {'status': 'complete', 'stage_params': config['stage_params'], 'yield_params': config['yield_params'],
               'config': {'stage_model': 'tcn_film', 'sequence_length': 100, 'crossfit_folds': 3, 'seeds': ','.join(map(str, seeds))},
               'code_sha256': {str(p.relative_to(release)): runner.sha256(p) for p in (release / 'src/gpace').glob('*.py')},
               'input_files': {}, 'data_audit': data.audit}
        for name in _FILES.values():
            path = inputs / name
            path.write_text('synthetic fixture; not real G2F data\n')
            old['input_files'][name] = {'sha256': runner.sha256(path)}
        (source / 'g2f_run_manifest.json').write_text(json.dumps(old))
        pd.concat(source_predictions).to_csv(source / 'g2f_yield_predictions.csv', index=False)
        saved_package, saved_loader, saved_argv = runner.PACKAGE, runner.load_g2f_2024_data, sys.argv
        try:
            runner.PACKAGE = release
            runner.load_g2f_2024_data = lambda *a, **k: prepared
            sys.argv = ['selftest', '--data-dir', str(inputs), '--existing-evaluation-dir', str(source),
                        '--output-dir', str(output), '--threads', '1', '--bootstrap-repeats', '100', '--resume']
            runner.main()
            assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'complete'
            hashes = {str(p): runner.sha256(p) for p in (output / 'runs').glob('seed_*/*/yield_model.joblib')}
            runner.main()
            assert hashes == {str(p): runner.sha256(p) for p in (output / 'runs').glob('seed_*/*/yield_model.joblib')}
            manifest = json.loads((output / 'run_manifest.json').read_text())
            assert manifest['vram_policy'].startswith('Both heads task_type=CPU')
            (inputs / _FILES['train_traits']).write_text('changed fixture\n')
            assert_raises(runner.main)
            with __import__('zipfile').ZipFile(output / 'g2f_stage_ablation_results.zip') as z:
                assert 'report/paired_environment_bootstrap.csv' in z.namelist()
                assert not any(n.endswith('.joblib') for n in z.namelist())
        finally:
            runner.PACKAGE, runner.load_g2f_2024_data, sys.argv = saved_package, saved_loader, saved_argv
        print('PASS: end-to-end orchestration, data checksums, report, ZIP and full-run resume.')
    print('ALL CHECKS PASSED (synthetic data only; no real G2F ablation result claimed).')


if __name__ == '__main__':
    main()
