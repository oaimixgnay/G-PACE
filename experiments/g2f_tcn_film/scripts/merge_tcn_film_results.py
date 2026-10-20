#!/usr/bin/env python3
"""Merge new TCN–FiLM G-PACE predictions with five historical comparators."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
sys.path.insert(0, str(PACKAGE / 'scripts'))
import pandas as pd
import numpy as np
from gpace.metrics import regression_metrics
from gpace.reporting import external_paired_cluster_bootstrap, summarize_metric_replicates
from run_g2f_tcn_film import atomic_csv, atomic_json, log, sha256

ORDER = ['g_pace', 'e2e_catboost', 'ec_gblup', 'reaction_norm_rff', 'ceris_jgra', 'geformer_style']
REQUIRED = ['record_id', 'Experiment', 'Variety_ID', 'observed', 'predicted', 'model', 'scenario', 'seed']


def read_predictions(path):
    frame = pd.read_csv(path)
    if not set(REQUIRED).issubset(frame):
        raise ValueError(f'Missing prediction columns: {path}')
    frame = frame.loc[frame.scenario.eq('external_2024'), REQUIRED].copy()
    if frame.empty or frame.duplicated(['model', 'seed', 'record_id']).any():
        raise ValueError(f'Empty/duplicate prediction records: {path}')
    if not np.isfinite(frame[['observed', 'predicted']].to_numpy(float)).all():
        raise ValueError(f'Nonfinite predictions or labels: {path}')
    return frame


def align_records(new, baseline):
    seeds = sorted(new.seed.unique())
    base = baseline.loc[baseline.seed.isin(seeds)].copy()
    if set(new.model) != {'g_pace'} or set(base.model) != set(ORDER):
        raise ValueError('New data must contain only G-PACE; baseline report must contain the original six models.')
    if set(new.seed) != set(base.seed):
        raise ValueError('New and baseline seed sets differ; comparisons refused.')
    key = ['seed', 'record_id', 'Experiment', 'Variety_ID']
    reference = new[new.model.eq('g_pace')].sort_values(key).reset_index(drop=True)
    for name, frame in [('new', new), ('baseline', base)]:
        for model, group in frame.groupby('model'):
            group = group.sort_values(key).reset_index(drop=True)
            if not group[key].equals(reference[key]) or not np.allclose(group.observed, reference.observed, rtol=0, atol=1e-10):
                raise ValueError(f'Mismatched records/labels in {name} model={model}; comparisons refused.')
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-evaluation-dir', required=True, type=Path)
    p.add_argument('--baseline-predictions', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--bootstrap-repeats', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260917)
    args = p.parse_args()
    manifest = json.loads((args.new_evaluation_dir / 'g2f_run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest['config']['stage_model'] != 'tcn_film':
        raise ValueError('Wait for a completed TCN–FiLM run before merging results.')
    new = read_predictions(args.new_evaluation_dir / 'g2f_yield_predictions.csv')
    original = read_predictions(args.baseline_predictions)
    baseline = align_records(new, original)
    retained = baseline[baseline.model.ne('g_pace')]
    combined = pd.concat([new, retained], ignore_index=True)
    combined['model'] = pd.Categorical(combined.model, categories=ORDER, ordered=True)
    combined = combined.sort_values(['model', 'seed', 'record_id']).reset_index(drop=True)
    # Keep the approved order without creating categorical Cartesian products
    # in the historical bootstrap helper's multi-key groupby.
    combined['model'] = combined['model'].astype(str)
    metrics = pd.DataFrame([{'model': model, 'scenario': scenario, 'seed': seed, **regression_metrics(group)}
                           for (model, scenario, seed), group in combined.groupby(['model', 'scenario', 'seed'], observed=True, sort=False)])
    summary = summarize_metric_replicates(metrics)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / 'g2f_combined_predictions.csv', combined)
    atomic_csv(output / 'g2f_combined_metrics.csv', metrics)
    atomic_csv(output / 'g2f_model_run_summary.csv', summary)
    atomic_csv(output / 'g2f_scenario_model_summary.csv', summary)
    ridge = baseline[baseline.model.eq('g_pace')].copy()
    ridge['model'] = 'g_pace_ridge_reference'
    comparison = pd.concat([ridge, new[new.model.eq('g_pace')]], ignore_index=True)
    ridge_metrics = pd.DataFrame([{'model': model, 'scenario': scenario, 'seed': seed, **regression_metrics(group)}
                                 for (model, scenario, seed), group in comparison.groupby(['model', 'scenario', 'seed'])])
    atomic_csv(output / 'g2f_ridge_vs_tcn_film_by_seed.csv', ridge_metrics)
    atomic_csv(output / 'g2f_ridge_vs_tcn_film_summary.csv', summarize_metric_replicates(ridge_metrics))
    log(f'[COMPARE] Six-model paired bootstrap, {args.bootstrap_repeats} repeats; only G-PACE was refitted.')
    paired = external_paired_cluster_bootstrap(combined, 'g_pace', ORDER[1:], repeats=args.bootstrap_repeats, seed=args.seed)
    atomic_csv(output / 'g2f_external_2024_paired_cluster_bootstrap.csv', paired)
    atomic_json(output / 'g2f_2024_analysis_manifest.json', {
        'stage_model': 'tcn_film', 'model_order': ORDER,
        'new_evaluation_dir': str(args.new_evaluation_dir.resolve()),
        'baseline_predictions': str(args.baseline_predictions.resolve()),
        'baseline_sha256': sha256(args.baseline_predictions), 'new_training_signature': manifest['training_signature'],
        'refitted_models': ['g_pace'], 'reused_fixed_comparators': ORDER[1:],
        'ridge_reference_retained': True, 'bootstrap_repeats': args.bootstrap_repeats,
        'guardrail': 'No comparator is tuned or selected using 2024 scores; same records, labels and seeds verified.'})
    log(f'[COMPLETE] Six-model report saved under {output}')


if __name__ == '__main__':
    main()
