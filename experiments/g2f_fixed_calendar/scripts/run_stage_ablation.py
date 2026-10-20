#!/usr/bin/env python3
"""Paired CPU yield-head ablation using frozen, out-of-fold TCN–FiLM exposures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile

# This release deliberately uses CPU for BOTH yield heads. It never fits or
# loads a neural network and cannot initialize a CUDA allocator.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gpace.data import EXPOSURE_COLUMNS
from gpace.g2f import _FILES, load_g2f_2024_data
from ablation_models import StageAblationYieldModel
from paired_statistics import FULL, ABLATION, MODELS, compare, write_report


def log(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, float_format='%.17g')
    temp.replace(path)


def parse_seeds(text):
    seeds = [int(s.strip()) for s in text.split(',') if s.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('This primary comparison requires three distinct seeds.')
    return seeds


def check_paths(args):
    source, output, inputs = (p.resolve() for p in (args.existing_evaluation_dir, args.output_dir, args.data_dir))
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('New output and original evaluation directories must be disjoint.')
    if output == inputs or inputs in output.parents or output in inputs.parents:
        raise ValueError('Output and original data directories must be disjoint.')
    return source, output, inputs


def cache_paths(source, seed, folds):
    root = source / 'runs' / 'external_2024' / f'seed_{seed}'
    return [root / 'crossfit' / f'axis{axis}_fold{fold}.npz'
            for axis in range(2) for fold in range(folds)] + [root / 'test_exposures.npz']


def preflight(args, config, seeds, source, inputs):
    missing = [inputs / name for name in _FILES.values() if not (inputs / name).is_file()]
    required = [source / 'g2f_run_manifest.json', source / 'g2f_yield_predictions.csv']
    for seed in seeds:
        required.extend(cache_paths(source, seed, 3))
    missing.extend(p for p in required if not p.is_file())
    if missing:
        raise FileNotFoundError(
            'Required source files are missing. This runner NEVER fabricates or replaces OOF exposures.\n'
            + '\n'.join(str(p) for p in missing)
            + '\nPoint --existing-evaluation-dir at the ORIGINAL server evaluation folder. '
              'The small exported results ZIP does not include the OOF .npz caches.')
    old = json.loads((source / 'g2f_run_manifest.json').read_text())
    if old.get('status') != 'complete':
        raise ValueError('The original full-model run must have completed successfully.')
    if old.get('stage_params') != config['stage_params'] or old.get('yield_params') != config['yield_params']:
        raise ValueError('Original run hyperparameters differ from the bundled frozen configuration.')
    original = old.get('config', {})
    if original.get('stage_model') != 'tcn_film' or original.get('sequence_length') != 100 or original.get('crossfit_folds') != 3:
        raise ValueError('Require original day-100 TCN–FiLM, dual-axis 3-fold cache.')
    old_seeds = [int(s) for s in str(original.get('seeds', '')).split(',') if s.strip()]
    if set(old_seeds) != set(seeds):
        raise ValueError(f'Seeds must match the completed source run: {old_seeds}')
    for p in sorted((PACKAGE / 'src/gpace').glob('*.py')):
        key = str(p.relative_to(PACKAGE))
        if old.get('code_sha256', {}).get(key) != sha256(p):
            raise ValueError(f'Bundled data/model implementation differs from the original run: {key}')
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'catboost', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'Missing dependency: {name}; activate the existing gpace environment.') from exc
    log(f'[CHECK] CPU-only paired heads; VRAM=0; threads={args.threads}; runtime={versions}')
    log('[CHECK] Both heads will use the same current CPU backend and fixed hyperparameters. No neural retraining.')
    return old, versions


def verify_data_hashes(inputs, old):
    result = {}
    for name in _FILES.values():
        path = inputs / name
        expected = old.get('input_files', {}).get(name, {}).get('sha256')
        if not expected:
            raise ValueError(f'Original manifest has no checksum for {name}')
        log(f'[VERIFY INPUT] {name}')
        result[name] = sha256(path)
        if result[name] != expected:
            raise ValueError(f'Input file changed since the stage-cache run: {name}')
    return result


def load_cache(path, expected_indices):
    with np.load(path, allow_pickle=False) as value:
        if not np.array_equal(value['row_indices'], expected_indices):
            raise ValueError(f'Cache row identities/order differ: {path}')
        predicted = value['predictions'].copy()
    if predicted.shape != (len(expected_indices), len(EXPOSURE_COLUMNS)) or not np.isfinite(predicted).all():
        raise ValueError(f'Invalid cached exposure array: {path}')
    return predicted


def load_stage_features(data, train, test, source, seed, folds=3):
    rows = data.records.loc[train]
    sums = np.zeros((len(train), len(EXPOSURE_COLUMNS)), dtype=float)
    counts = np.zeros(len(train), dtype=int)
    root = source / 'runs' / 'external_2024' / f'seed_{seed}'
    for axis, column in enumerate(('Variety_ID', 'Experiment')):
        groups = rows[column].astype(str).to_numpy()
        unique = np.unique(groups)
        if len(unique) < folds:
            raise ValueError(f'Too few {column} groups for the original fold assignment.')
        shuffled = unique.copy()
        np.random.default_rng(seed + axis).shuffle(shuffled)
        for fold, held in enumerate(np.array_split(shuffled, folds)):
            mask = np.isin(groups, held)
            if set(groups[mask]) & set(groups[~mask]):
                raise ValueError('Group holdout is not disjoint.')
            values = load_cache(root / 'crossfit' / f'axis{axis}_fold{fold}.npz', train[mask])
            sums[mask] += values
            counts[mask] += 1
    if not np.all(counts == 2):
        raise ValueError('Every training record must have exactly two held-out predictions.')
    return sums / counts[:, None], load_cache(root / 'test_exposures.npz', test)


def validate_prepared(prepared, old):
    data, train, test = prepared.data, prepared.training_indices, prepared.external_indices
    if not len(train) or not len(test) or len(set(train) & set(test)):
        raise ValueError('Empty or overlapping training/test indices.')
    if not data.records.loc[train, 'Year'].between(2014, 2023).all() or not data.records.loc[test, 'Year'].eq(2024).all():
        raise ValueError('Training/test years are incorrect.')
    if not data.records.loc[test, list(EXPOSURE_COLUMNS)].isna().all().all():
        raise ValueError('2024 measured exposure labels must remain missing.')
    audit = old.get('data_audit', {})
    if len(train) != audit.get('training_gxe_means') or len(test) != audit.get('external_2024_labeled_records'):
        raise ValueError('Prepared record counts differ from the completed source run.')
    if data.records.loc[test, 'Experiment'].nunique() != audit.get('external_2024_environment_count'):
        raise ValueError('External environment count differs from the completed source run.')
    return data, np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def original_predictions(source, data, test, seeds):
    frame = pd.read_csv(source / 'g2f_yield_predictions.csv', dtype={'Experiment': str, 'Variety_ID': str})
    frame = frame[frame.model.eq('g_pace') & frame.scenario.eq('external_2024')].copy()
    expected = data.records.loc[test, ['record_id', 'Experiment', 'Variety_ID', 'grain.yield']].rename(columns={'grain.yield': 'observed'})
    for seed in seeds:
        check_prediction(frame[frame.seed.eq(seed)], expected, 'g_pace', seed)
    return frame


def check_prediction(frame, expected, model, seed):
    required = ['record_id', 'Experiment', 'Variety_ID', 'observed', 'predicted', 'model', 'seed']
    if not set(required).issubset(frame) or len(frame) != len(expected):
        raise ValueError(f'Prediction table shape/columns differ for {model}, seed={seed}.')
    a, b = frame.sort_values('record_id').reset_index(drop=True), expected.sort_values('record_id').reset_index(drop=True)
    if a.record_id.duplicated().any() or not np.array_equal(a.record_id, b.record_id):
        raise ValueError('Prediction record IDs differ.')
    for col in ('Experiment', 'Variety_ID'):
        if not np.array_equal(a[col].astype(str), b[col].astype(str)):
            raise ValueError(f'Prediction {col} identities differ.')
    if not np.allclose(a.observed, b.observed, rtol=0, atol=1e-10) or not np.isfinite(a.predicted).all():
        raise ValueError('Prediction labels differ or predictions are non-finite.')
    if not a.model.eq(model).all() or not a.seed.eq(seed).all():
        raise ValueError('Prediction model/seed labels differ.')


def fit_one(data, train, test, exposures, test_exposures, config, seed, include_stage, output, threads, signature, resume):
    label = FULL if include_stage else ABLATION
    folder = output / 'runs' / f'seed_{seed}' / label
    folder.mkdir(parents=True, exist_ok=True)
    marker, csv_path, model_path = (folder / name for name in ('completed.json', 'predictions.csv', 'yield_model.joblib'))
    expected = data.records.loc[test, ['record_id', 'Experiment', 'Variety_ID', 'grain.yield']].rename(columns={'grain.yield': 'observed'})
    if resume and marker.is_file():
        done = json.loads(marker.read_text())
        if done.get('training_signature') != signature:
            raise ValueError(f'Stale completion marker: {marker}')
        for path in (csv_path, model_path):
            if not path.is_file() or done.get('sha256', {}).get(path.name) != sha256(path):
                raise ValueError(f'Completed output checksum failed: {path}')
        frame = pd.read_csv(csv_path, dtype={'Experiment': str, 'Variety_ID': str})
        check_prediction(frame, expected, label, seed)
        log(f'[RESUME] {label}, seed={seed}')
        return frame
    log(f'[FIT] {label}, seed={seed}; {len(train)} historical records; CPU threads={threads}')
    params = {**config['yield_params'], 'task_type': 'CPU', 'thread_count': threads}
    model = StageAblationYieldModel(
        mode='mean_residual_hybrid', genotype_components=config['yield_genotype_components'],
        environment_components=config['yield_environment_components'], alpha=10.,
        head='catboost', head_params=params, random_state=seed, device='cpu', include_stage=include_stage)
    with threadpool_limits(limits=threads):
        model.fit(data, train, exposures if include_stage else None)
        prediction = model.predict(data, test, test_exposures if include_stage else None)
    frame = expected.copy()
    frame['predicted'], frame['model'], frame['seed'], frame['scenario'] = prediction, label, seed, 'external_2024'
    check_prediction(frame, expected, label, seed)
    temp = model_path.with_name(model_path.name + '.tmp')
    joblib.dump(model, temp)
    temp.replace(model_path)
    atomic_csv(csv_path, frame)
    atomic_json(marker, {'training_signature': signature, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                         'include_stage': include_stage, 'backend': 'CPU', 'feature_count': model.regressor_.model_.n_features_in_,
                         'sha256': {p.name: sha256(p) for p in (csv_path, model_path)}})
    del model
    gc.collect()
    return frame


def export_results(output):
    files = [output / 'run_manifest.json', output / 'paired_predictions.csv',
             output / 'by_seed_summary.csv', output / 'original_vs_paired_cpu.csv']
    files += sorted((output / 'report').glob('*'))
    files += sorted((output / 'runs').glob('seed_*/*/completed.json'))
    target = output / 'g2f_stage_ablation_results.zip'
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_file():
                z.write(p, str(p.relative_to(output)))
    temp.replace(target)
    return target


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--existing-evaluation-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seeds', default='2026,2027,2028')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--bootstrap-repeats', type=int, default=5000)
    p.add_argument('--random-state', type=int, default=20261016)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1 or args.bootstrap_repeats < 1:
        raise ValueError('Threads and bootstrap repeats must be positive.')
    seeds = parse_seeds(args.seeds)
    config = json.loads((PACKAGE / 'config/g2f_tcn_film_frozen.json').read_text())
    source, output, inputs = check_paths(args)
    old, versions = preflight(args, config, seeds, source, inputs)
    if args.preflight_only:
        log('[CHECK COMPLETE] Files/configuration/dependencies checked. No data loading, fitting or output changes.')
        return
    # flock in the shell wrapper covers the whole run; direct invocation is also locked.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / (output.name + '.python.lock')).open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another process already owns this output directory.') from exc
    input_hashes = verify_data_hashes(inputs, old)
    cache_files = [p for seed in seeds for p in cache_paths(source, seed, 3)]
    code_paths = sorted([*list((PACKAGE / 'src/gpace').glob('*.py')), *list((PACKAGE / 'scripts').glob('*.py'))])
    signature_value = {'config': config, 'seeds': seeds, 'threads': args.threads, 'backend': 'CPU',
                       'python': sys.version.split()[0], 'versions': versions, 'inputs': input_hashes,
                       'source_manifest_sha256': sha256(source / 'g2f_run_manifest.json'),
                       'source_predictions_sha256': sha256(source / 'g2f_yield_predictions.csv'),
                       'stage_cache_sha256': {str(p.relative_to(source)): sha256(p) for p in cache_files},
                       'code_sha256': {str(p.relative_to(PACKAGE)): sha256(p) for p in code_paths}}
    signature = object_hash(signature_value)
    manifest_path = output / 'run_manifest.json'
    if manifest_path.is_file():
        if not args.resume or json.loads(manifest_path.read_text()).get('training_signature') != signature:
            raise ValueError('Output exists or data/code/configuration changed. Use --resume with identical inputs, or a new output directory.')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output has no matching manifest; use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'running', 'training_signature': signature, 'provenance': signature_value,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'source_evaluation': str(source), 'data_dir': str(inputs), 'output_dir': str(output),
                'protocol': 'Fixed-configuration paired CPU ablation; frozen TCN–FiLM crossfit/test features from completed historical run; no tuning.',
                'vram_policy': 'Both heads task_type=CPU, CUDA_VISIBLE_DEVICES empty; no neural training; GPU memory usage=0.',
                'stage_feature_policy': 'Full uses original dual-axis OOF train exposures; ablation deletes both Z and G×Z columns.',
                'original_full_policy': 'Original GPU predictions are a descriptive reference only, never paired with a CPU-only ablation as the primary result.'}
    atomic_json(manifest_path, manifest)
    try:
        log('[DATA] Loading original G2F inputs with the unchanged loader.')
        prepared = load_g2f_2024_data(inputs, sequence_length=100)
        data, train, test = validate_prepared(prepared, old)
        old_predictions = original_predictions(source, data, test, seeds)
        manifest['data_audit'] = data.audit
        atomic_json(manifest_path, manifest)
        all_frames = []
        for seed in seeds:
            log(f'[CACHE] Validate and combine original OOF stage predictions, seed={seed}')
            exposures, test_exposures = load_stage_features(data, train, test, source, seed)
            for include_stage in (True, False):
                frame = fit_one(data, train, test, exposures, test_exposures, config, seed,
                                include_stage, output, args.threads, signature, args.resume)
                all_frames.append(frame)
                atomic_csv(output / 'paired_predictions.csv', pd.concat(all_frames, ignore_index=True))
            del exposures, test_exposures
            gc.collect()
        predictions = pd.concat(all_frames, ignore_index=True)
        log('[REPORT] Ensemble-first comparison and paired environment bootstrap.')
        write_report(predictions, seeds, output / 'report', args.bootstrap_repeats, args.random_state)
        by_seed, references = [], []
        for seed in seeds:
            summary = compare(predictions[predictions.seed.eq(seed)], [seed], repeats=1)[2]
            summary['seed'] = seed
            by_seed.append(summary)
            old_full = old_predictions[old_predictions.seed.eq(seed)].copy()
            cpu_full = predictions[predictions.seed.eq(seed) & predictions.model.eq(FULL)].copy()
            fake_pair = pd.concat([old_full.assign(model=FULL), cpu_full.assign(model=ABLATION)], ignore_index=True)
            ref = compare(fake_pair, [seed], repeats=1)[2]
            ref['source'] = ref.model.map({FULL: 'original_full', ABLATION: 'paired_cpu_full'})
            ref['seed'] = seed
            references.append(ref.drop(columns='model'))
        atomic_csv(output / 'by_seed_summary.csv', pd.concat(by_seed, ignore_index=True))
        atomic_csv(output / 'original_vs_paired_cpu.csv', pd.concat(references, ignore_index=True))
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(manifest_path, manifest)
        result_zip = export_results(output)
        log(f'[COMPLETE] {output}')
        log(f'[SEND] {result_zip}')
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(manifest_path, manifest)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
