#!/usr/bin/env python3
"""Retrain the frozen European TCN–FiLM stage architecture on G2F history."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

import numpy as np
import pandas as pd

from gpace.data import EXPOSURE_COLUMNS
from gpace.g2f import _FILES, load_g2f_2024_data
from gpace.metrics import cluster_bootstrap_intervals, exposure_metrics, regression_metrics
from gpace.models import YieldModel, make_exposure_model
from gpace.reporting import summarize_metric_replicates


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def save_array(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def load_array(path, indices):
    with np.load(path, allow_pickle=False) as cached:
        if not np.array_equal(cached['row_indices'], indices):
            raise ValueError(f'Cached row order differs: {path}')
        values = cached['predictions']
    if values.shape != (len(indices), len(EXPOSURE_COLUMNS)) or not np.isfinite(values).all():
        raise ValueError(f'Invalid cached stage predictions: {path}')
    return values


def stage_network_hash():
    # Hash the frozen network source, not Python-version-dependent AST fields.
    source = (PACKAGE / 'src/gpace/models.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_TCNNetwork')
    fragment = ast.get_source_segment(source, definition)
    if fragment is None:
        raise ValueError('Cannot extract the frozen TCN–FiLM network definition.')
    return hashlib.sha256(fragment.encode('utf-8')).hexdigest()


def verify_config(path):
    config = json.loads(Path(path).read_text())
    reference = json.loads((PACKAGE / 'config/europe_frozen_reference.json').read_text())
    expected = {key: value for key, value in reference['stage_params'].items() if key != 'device'}
    if config['stage_model'] != 'tcn_film' or config['stage_params'] != expected:
        raise ValueError('Stage configuration must exactly match the frozen European reference.')
    if config['sequence_length'] != 100 or config['yield_mode'] != 'mean_residual_hybrid' or config['yield_head'] != 'catboost':
        raise ValueError('This package requires day-100 TCN–FiLM and hybrid CatBoost.')
    ridge = json.loads((PACKAGE / 'config/g2f_ridge_run_reference.json').read_text())
    expected_yield = {key: value for key, value in ridge['yield_params'].items() if key != 'task_type'}
    if config['yield_params'] != expected_yield:
        raise ValueError('Yield settings must match the verified G2F historical configuration.')
    expected_hash = json.loads((PACKAGE / 'config/network_signature.json').read_text())['europe_network_source_sha256']
    if stage_network_hash() != expected_hash:
        raise ValueError('Bundled TCN–FiLM network differs from the frozen European architecture.')
    return config


def preflight(data_dir, device, check_dependencies=True):
    data_dir = Path(data_dir).resolve()
    missing = [name for name in _FILES.values() if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'G2F input directory: {data_dir}\nMissing files: ' + ', '.join(missing))
    versions = {}
    packages = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'torch', 'catboost')
    absent = []
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            absent.append(name)
    if check_dependencies and absent:
        raise RuntimeError('Missing packages in the selected Python environment: ' + ', '.join(absent))
    if check_dependencies:
        import torch
        import catboost
        if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'GradScaler'):
            raise RuntimeError('The bundled European trainer requires torch.amp.GradScaler; use the existing gpace PyTorch 2.5 environment.')
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable. Select the existing CUDA-enabled gpace environment or explicitly use --device cpu.')
            free, total = torch.cuda.mem_get_info()
            log(f'[GPU] {torch.cuda.get_device_name(0)}; free={free / 2**30:.2f} GiB, total={total / 2**30:.2f} GiB')
    log(f'[CHECK] gpace loaded from {PACKAGE / "src/gpace"}')
    log(f'[CHECK] Python {sys.version.split()[0]}; packages={versions}')
    return versions


def make_tasks(prepared, diagnostic=False):
    data = prepared.data
    train = np.asarray(prepared.training_indices, dtype=int)
    external = np.asarray(prepared.external_indices, dtype=int)
    if len(set(train) & set(external)):
        raise ValueError('Historical training and 2024 evaluation indices overlap.')
    years = pd.to_numeric(data.records.loc[train, 'Year']).to_numpy(int)
    if not len(train) or not len(external) or np.any((years < 2014) | (years > 2023)):
        raise ValueError('Training must consist only of 2014–2023 records.')
    if not data.records.loc[external, 'Year'].eq(2024).all():
        raise ValueError('External evaluation must consist only of 2024 records.')
    if not np.isfinite(data.records.loc[train, list(EXPOSURE_COLUMNS)].to_numpy(float)).all():
        raise ValueError('Historical stage labels contain missing or infinite values.')
    if not data.records.loc[external, list(EXPOSURE_COLUMNS)].isna().all().all():
        raise ValueError('2024 measured stage labels must remain unavailable to the trainer.')
    tasks = []
    if diagnostic:
        earlier, last = train[years < 2023], train[years == 2023]
        if not len(earlier) or not len(last):
            raise ValueError('Cannot construct the optional 2023 temporal diagnostic.')
        tasks.append(('temporal_2023', earlier, last))
    tasks.append(('external_2024', train, external))
    return tasks


def clear_accelerator():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def crossfit(data, indices, stage_params, folds, seed, cache_dir, resume):
    """Same dual-axis split and seed rules as the European/G2F core function."""
    rows = data.records.loc[indices]
    total = np.zeros((len(indices), len(EXPOSURE_COLUMNS)), dtype=float)
    counts = np.zeros(len(indices), dtype=int)
    for axis, column in enumerate(('Variety_ID', 'Experiment')):
        groups = rows[column].astype(str).to_numpy()
        unique = np.unique(groups)
        if folds < 2 or folds > len(unique):
            raise ValueError(f'Invalid folds={folds} for {len(unique)} {column} groups.')
        shuffled = unique.copy()
        np.random.default_rng(seed + axis).shuffle(shuffled)
        for fold, held in enumerate(np.array_split(shuffled, folds)):
            mask = np.isin(groups, held)
            fit_indices, predict_indices = indices[~mask], indices[mask]
            if set(rows.loc[fit_indices, column]) & set(rows.loc[predict_indices, column]):
                raise RuntimeError(f'{column} holdout leaked.')
            path = cache_dir / f'axis{axis}_fold{fold}.npz'
            if resume and path.is_file():
                prediction = load_array(path, predict_indices)
                log(f'[RESUME] seed={seed} {column} fold={fold + 1}/{folds}')
            else:
                log(f'[CROSSFIT] seed={seed} {column} fold={fold + 1}/{folds}; fit={len(fit_indices)}, predict={len(predict_indices)}')
                model = make_exposure_model('tcn_film', random_state=seed + axis * 100 + fold, **stage_params)
                prediction = model.fit(data, fit_indices).predict(data, predict_indices)
                if prediction.shape != (len(predict_indices), len(EXPOSURE_COLUMNS)) or not np.isfinite(prediction).all():
                    raise RuntimeError('Stage model produced invalid crossfit predictions.')
                save_array(path, row_indices=predict_indices, predictions=prediction)
                del model
                clear_accelerator()
            total[mask] += prediction
            counts[mask] += 1
    if not np.all(counts == 2):
        raise RuntimeError('Each historical row must receive two held-out predictions.')
    return total / counts[:, None]


def save_stage(stage, config, folder):
    import joblib
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / 'stage_weights.pt.tmp'
    torch.save({key: value.detach().cpu() for key, value in stage.network_.state_dict().items()}, temp)
    temp.replace(folder / 'stage_weights.pt')
    temp = folder / 'stage_preprocessing.joblib.tmp'
    joblib.dump({'inputs': stage.inputs_, 'target_scaler': stage.target_scaler_}, temp)
    temp.replace(folder / 'stage_preprocessing.joblib')
    atomic_json(folder / 'stage_training.json', {
        'stage_model': 'tcn_film', 'stage_params': config,
        'best_epoch': stage.best_epoch_, 'best_validation_loss': stage.best_validation_loss_,
        'network_weights': 'stage_weights.pt', 'preprocessing': 'stage_preprocessing.joblib',
        'weight_initialization': 'Fresh random initialization; no European network weights loaded.'})


def prediction_frame(data, indices, values, model, scenario, seed):
    frame = data.records.loc[indices, ['record_id', 'Experiment', 'Variety_ID', 'grain.yield']].rename(columns={'grain.yield': 'observed'}).copy()
    frame['predicted'] = values
    frame['model'], frame['scenario'], frame['seed'] = model, scenario, seed
    return frame


def evaluate_task(data, train, test, scenario, seed, config, args, folder):
    import joblib
    task_dir = folder / 'runs' / scenario / f'seed_{seed}'
    task_dir.mkdir(parents=True, exist_ok=True)
    stage_params = {**config['stage_params'], 'device': args.device, 'verbose': True}
    training_exposures = crossfit(data, train, stage_params, args.crossfit_folds, seed, task_dir / 'crossfit', args.resume)
    final_path = task_dir / 'test_exposures.npz'
    weight_files = [task_dir / 'models' / name for name in ('stage_weights.pt', 'stage_preprocessing.joblib', 'stage_training.json')]
    if args.resume and final_path.is_file() and all(p.is_file() for p in weight_files):
        test_exposures = load_array(final_path, test)
        log(f'[RESUME] full-history stage model {scenario} seed={seed}')
    else:
        log(f'[STAGE FINAL] {scenario} seed={seed}; fit={len(train)}, predict={len(test)}')
        stage = make_exposure_model('tcn_film', random_state=seed + 999, **stage_params).fit(data, train)
        test_exposures = stage.predict(data, test)
        save_stage(stage, stage_params, task_dir / 'models')
        save_array(final_path, row_indices=test, predictions=test_exposures)
        del stage
        clear_accelerator()
    score = exposure_metrics(data.records.loc[train, list(EXPOSURE_COLUMNS)].to_numpy(float), training_exposures)
    score['scenario'], score['seed'] = f'{scenario}_training_crossfit', seed
    atomic_csv(task_dir / 'stage_metrics.csv', score)
    exposure_frame = data.records.loc[test, ['record_id', 'Experiment', 'Variety_ID']].copy()
    for i, name in enumerate(EXPOSURE_COLUMNS):
        exposure_frame[f'predicted_{name}'] = test_exposures[:, i]
    exposure_frame['scenario'], exposure_frame['seed'] = scenario, seed
    atomic_csv(task_dir / 'stage_predictions.csv', exposure_frame)
    predictions = []
    yield_params = {**config['yield_params'], 'task_type': 'GPU' if args.device == 'cuda' else 'CPU'}
    for label, mode, uses_stage in [('g_pace', 'mean_residual_hybrid', True)]:
        path, weight_path = task_dir / f'{label}_predictions.csv', task_dir / 'models' / f'{label}_yield.joblib'
        if args.resume and path.is_file() and weight_path.is_file():
            frame = pd.read_csv(path)
            if not np.array_equal(frame.record_id.to_numpy(), data.records.loc[test, 'record_id'].to_numpy()) or not np.isfinite(frame.predicted).all():
                raise ValueError(f'Invalid resumed yield predictions: {path}')
            log(f'[RESUME] yield model={label} {scenario} seed={seed}')
        else:
            log(f'[YIELD] model={label} {scenario} seed={seed}; fit={len(train)}')
            model = YieldModel(mode=mode, genotype_components=config['yield_genotype_components'],
                               environment_components=config['yield_environment_components'], alpha=10.0,
                               head='catboost', head_params=yield_params, random_state=seed, device=args.device)
            model.fit(data, train, training_exposures if uses_stage else None)
            prediction = model.predict(data, test, test_exposures if uses_stage else None)
            frame = prediction_frame(data, test, prediction, label, scenario, seed)
            temp = weight_path.with_name(weight_path.name + '.tmp')
            joblib.dump(model, temp)
            temp.replace(weight_path)
            atomic_csv(path, frame)
            del model
            clear_accelerator()
        predictions.append(frame)
    return pd.concat(predictions, ignore_index=True), score


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-dir', required=True, type=Path)
    result.add_argument('--output-dir', required=True, type=Path)
    result.add_argument('--config-json', type=Path, default=PACKAGE / 'config/g2f_tcn_film_frozen.json')
    result.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    result.add_argument('--seeds', default='2026,2027,2028')
    result.add_argument('--crossfit-folds', type=int, default=3)
    result.add_argument('--bootstrap-repeats', type=int, default=1000)
    result.add_argument('--temporal-diagnostic', action='store_true', help='Also train through 2022 and evaluate 2023; roughly doubles neural fits.')
    result.add_argument('--resume', action='store_true', help='Reuse completed folds/models only when data, configuration, code and runtime versions match.')
    result.add_argument('--preflight-only', action='store_true', help='Check configuration, bundled code, dependencies and input filenames without training.')
    return result


def main():
    args = parser().parse_args()
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds) or args.crossfit_folds < 2 or args.bootstrap_repeats < 0:
        raise ValueError('Use distinct seeds, at least two folds and nonnegative bootstrap repeats.')
    config = verify_config(args.config_json)
    versions = preflight(args.data_dir, args.device)
    if args.preflight_only:
        log('[CHECK COMPLETE] No model fitting or result-directory mutation performed.')
        return
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == data_dir or data_dir in output.parents:
        raise ValueError('Output directory must be separate from original input data.')
    output.mkdir(parents=True, exist_ok=True)
    log('[PROVENANCE] Hashing eight input files; this may take a few minutes on network storage.')
    input_files = {name: {'path': str(data_dir / name), 'sha256': sha256(data_dir / name)} for name in _FILES.values()}
    code_files = sorted([*list((PACKAGE / 'src/gpace').glob('*.py')), Path(__file__).resolve()])
    signature_value = {'config': config, 'seeds': seeds, 'crossfit_folds': args.crossfit_folds,
                       'device': args.device, 'versions': versions,
                       'inputs': {name: item['sha256'] for name, item in input_files.items()},
                       'code': {str(p.relative_to(PACKAGE)): sha256(p) for p in code_files}}
    signature = hashlib.sha256(json.dumps(signature_value, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'g2f_run_manifest.json'
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if not args.resume:
            raise ValueError('Output already contains a run. Use --resume or select a new output directory.')
        if old.get('training_signature') != signature:
            raise ValueError('Resume refused: input data/configuration/code/runtime versions differ. Use a new output directory.')
    elif any(output.iterdir()):
        raise ValueError('Output contains files without a matching run manifest. Use a new output directory.')
    manifest = {'status': 'running', 'training_signature': signature,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'config': {**vars(args), 'data_dir': str(data_dir), 'output_dir': str(output), 'config_json': str(args.config_json), 'stage_model': 'tcn_film', 'sequence_length': 100},
                'stage_params': config['stage_params'], 'yield_params': config['yield_params'],
                'protocol': config['protocol'], 'runtime_versions': versions,
                'input_files': input_files, 'code_sha256': signature_value['code'],
                'network_source_sha256': stage_network_hash()}
    atomic_json(manifest_path, manifest)
    log('[DATA] Loading and cleaning G2F historical data and 2024 evaluation records.')
    prepared = load_g2f_2024_data(data_dir, sequence_length=100)
    tasks = make_tasks(prepared, args.temporal_diagnostic)
    manifest['data_audit'] = prepared.data.audit
    atomic_json(manifest_path, manifest)
    log(f'[PLAN] {len(tasks)} task(s), {len(seeds)} seed(s), {2 * args.crossfit_folds + 1} stage fits per task/seed. Existing European weights are never loaded.')
    frames, scores = [], []
    started = time.monotonic()
    try:
        for scenario, train, test in tasks:
            for seed in seeds:
                frame, score = evaluate_task(prepared.data, train, test, scenario, seed, config, args, output)
                frames.append(frame)
                scores.append(score)
                atomic_csv(output / 'g2f_yield_predictions.csv', pd.concat(frames, ignore_index=True))
                log(f'[SAVED] {scenario} seed={seed}; elapsed={(time.monotonic() - started) / 3600:.2f} h')
        predictions = pd.concat(frames, ignore_index=True)
        metrics, intervals = [], []
        for (model, scenario, seed), frame in predictions.groupby(['model', 'scenario', 'seed'], sort=True):
            metrics.append({'model': model, 'scenario': scenario, 'seed': seed, **regression_metrics(frame)})
            log(f'[INTERVALS] {model} {scenario} seed={seed}; bootstrap={args.bootstrap_repeats}')
            interval = cluster_bootstrap_intervals(frame, repeats=args.bootstrap_repeats, seed=int(seed))
            interval['model'], interval['scenario'], interval['seed'] = model, scenario, seed
            intervals.append(interval)
        metrics_frame = pd.DataFrame(metrics)
        atomic_csv(output / 'g2f_yield_metrics.csv', metrics_frame)
        atomic_csv(output / 'g2f_yield_metric_intervals.csv', pd.concat(intervals, ignore_index=True))
        atomic_csv(output / 'g2f_training_crossfit_exposure_metrics.csv', pd.concat(scores, ignore_index=True))
        atomic_csv(output / 'g2f_model_run_summary.csv', summarize_metric_replicates(metrics_frame))
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        outputs={name: str(output / name) for name in ('g2f_yield_predictions.csv', 'g2f_yield_metrics.csv', 'g2f_yield_metric_intervals.csv', 'g2f_training_crossfit_exposure_metrics.csv', 'g2f_model_run_summary.csv')})
        atomic_json(manifest_path, manifest)
        log(f'[COMPLETE] Results and learned weights saved under {output}')
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
