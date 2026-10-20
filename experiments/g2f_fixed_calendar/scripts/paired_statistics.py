"""Ensemble-first statistics; paired bootstrap resamples entire environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FULL = 'g_pace_full'
ABLATION = 'g_pace_no_stage'
MODELS = (FULL, ABLATION)
ERRORS = {'rmse', 'environment_mean_rmse'}
KEYS = ['record_id', 'Experiment', 'Variety_ID']


def correlation(y, p):
    if len(y) < 3 or np.std(y) <= 1e-12 or np.std(p) <= 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def ensemble_predictions(predictions, seeds):
    required = set(KEYS + ['seed', 'model', 'observed', 'predicted'])
    if not required.issubset(predictions):
        raise ValueError(f'Missing prediction columns: {sorted(required - set(predictions))}')
    if predictions[list(required)].isna().any().any():
        raise ValueError('Prediction identifiers/labels/values contain missing data.')
    if not np.isfinite(predictions[['observed', 'predicted']].to_numpy(float)).all():
        raise ValueError('Non-finite predictions or labels.')
    if set(predictions.model) != set(MODELS):
        raise ValueError(f'Expected exactly the two paired models: {MODELS}')
    if predictions.duplicated(['model', 'seed', 'record_id']).any():
        raise ValueError('Duplicate model/seed/record predictions.')
    if set(predictions.seed.astype(int)) != set(seeds):
        raise ValueError('Missing or unexpected seeds.')
    aligned = None
    for (model, seed), frame in predictions.groupby(['model', 'seed']):
        frame = frame.sort_values('record_id')
        if frame.duplicated(['Experiment', 'Variety_ID']).any():
            raise ValueError('Each environment must have one record per hybrid.')
        reference = frame[KEYS + ['observed']].reset_index(drop=True)
        if aligned is None:
            aligned = reference
        elif not reference[KEYS].equals(aligned[KEYS]) or not np.allclose(
            reference.observed, aligned.observed, rtol=0, atol=1e-10
        ):
            raise ValueError(f'Record identities/labels differ for {model}, seed={seed}.')
    counts = predictions.groupby(['model', 'record_id']).size()
    if not counts.eq(len(seeds)).all():
        raise ValueError('Some records lack seed predictions; refusing an intersection-only comparison.')
    result = predictions.groupby(['model'] + KEYS, sort=True, as_index=False).agg(
        observed=('observed', 'first'), predicted=('predicted', 'mean')
    )
    return result


def environment_statistics(ensemble):
    rows = []
    for (model, env), frame in ensemble.groupby(['model', 'Experiment'], sort=True):
        frame = frame.sort_values('Variety_ID', kind='stable')
        y, p = frame.observed.to_numpy(float), frame.predicted.to_numpy(float)
        yd, pdv = y - y.mean(), p - p.mean()
        row = dict(model=model, Experiment=env, n=len(y), observed_mean=y.mean(), predicted_mean=p.mean(),
                   pcc=correlation(y, p), spearman=correlation(rankdata(y), rankdata(p)),
                   rmse=float(np.sqrt(np.mean((y - p)**2))),
                   sy=y.sum(), sp=p.sum(), syy=y @ y, spp=p @ p, syp=y @ p,
                   sse=np.sum((y - p)**2), dyy=yd @ yd, dpp=pdv @ pdv, dyp=yd @ pdv,
                   mean_error_squared=(y.mean() - p.mean())**2,
                   prediction_ties=int(pd.Series(p).duplicated(keep=False).sum()))
        # Stable hybrid-ID tie breaking is explicit and shared by both models.
        for q in (10, 20):
            k = int(np.ceil(q / 100 * len(y)))
            predicted_order = np.argsort(-p, kind='stable')[:k]
            observed_order = np.argsort(-y, kind='stable')[:k]
            gain = float(y[predicted_order].mean() - y.mean())
            ideal = float(y[observed_order].mean() - y.mean())
            row.update({f'top{q}_k': k, f'top{q}_gain': gain,
                        f'top{q}_recall': len(set(predicted_order) & set(observed_order)) / k,
                        f'top{q}_efficiency': gain / ideal if ideal > 1e-12 else np.nan,
                        f'top{q}_random_expected_recall': k / len(y)})
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_sums(sy, sp, syy, spp, syp, n):
    a = np.maximum(syy - sy * sy / n, 0)
    b = np.maximum(spp - sp * sp / n, 0)
    den = np.sqrt(a * b)
    return np.divide(syp - sy * sp / n, den, out=np.full_like(den, np.nan, dtype=float), where=den > 1e-12).clip(-1, 1)


def summarize_draws(frame, draws, common_masks):
    """One output value per environment draw, with duplicate blocks kept distinct."""
    def summed(name):
        return frame[name].to_numpy(float)[draws].sum(axis=1)

    def averaged(name, mask=None, fisher=False):
        v = frame[name].to_numpy(float)
        valid = np.isfinite(v) if mask is None else (mask & np.isfinite(v))
        if fisher:
            v = np.arctanh(np.clip(v, -.999999, .999999))
        v = np.where(valid, v, 0.)
        count = valid[draws].sum(axis=1)
        out = np.divide(v[draws].sum(axis=1), count,
                        out=np.full(len(draws), np.nan), where=count > 0)
        return np.tanh(out) if fisher else out

    n, sy, sp = summed('n'), summed('sy'), summed('sp')
    syy, spp, syp, sse = (summed(k) for k in ('syy', 'spp', 'syp', 'sse'))
    sst = syy - sy**2 / n
    dyy, dpp, dyp = (summed(k) for k in ('dyy', 'dpp', 'dyp'))
    den = np.sqrt(dyy * dpp)
    result = {
        'pcc': _corr_sums(sy, sp, syy, spp, syp, n),
        'r2': 1 - np.divide(sse, sst, out=np.full_like(sst, np.nan), where=sst > 1e-12),
        'rmse': np.sqrt(sse / n),
        'within_environment_pcc': averaged('pcc', common_masks['pcc'], fisher=True),
        'within_environment_spearman': averaged('spearman', common_masks['spearman']),
        'pooled_deviation_pcc': np.divide(dyp, den, out=np.full_like(den, np.nan), where=den > 1e-12),
        'environment_mean_rmse': np.sqrt(averaged('mean_error_squared')),
    }
    for q in (10, 20):
        for metric in ('gain', 'recall', 'efficiency'):
            key = f'top{q}_{metric}'
            result[key] = averaged(key, common_masks.get(key))
    return result


def compare(predictions, seeds, repeats=5000, random_state=20261016):
    if repeats < 1:
        raise ValueError('Bootstrap repeats must be positive.')
    ensemble = ensemble_predictions(predictions, seeds)
    per_environment = environment_statistics(ensemble)
    envs = sorted(ensemble.Experiment.unique())
    if len(envs) < 2:
        raise ValueError('At least two independent environments are required.')
    frames = {m: per_environment[per_environment.model.eq(m)].set_index('Experiment').loc[envs] for m in MODELS}
    common = {k: np.logical_and.reduce([np.isfinite(frames[m][k]) for m in MODELS])
              for k in ('pcc', 'spearman', 'top10_efficiency', 'top20_efficiency')}
    draws = np.random.default_rng(random_state).integers(0, len(envs), size=(repeats, len(envs)))
    point_draw = np.arange(len(envs))[None, :]
    points = {m: summarize_draws(frames[m], point_draw, common) for m in MODELS}
    boot = {m: summarize_draws(frames[m], draws, common) for m in MODELS}
    summary, differences = [], []
    for m in MODELS:
        summary.append({'model': m, 'n': len(ensemble[ensemble.model.eq(m)]),
                        'environment_count': len(envs), 'seeds': len(seeds),
                        **{k: float(v[0]) for k, v in points[m].items()}})
    for k in points[FULL]:
        difference = boot[FULL][k] - boot[ABLATION][k]
        finite = difference[np.isfinite(difference)]
        low, high = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
        favor = finite < 0 if k in ERRORS else finite > 0
        matched = int(common['pcc'].sum()) if k == 'within_environment_pcc' else (
            int(common['spearman'].sum()) if k == 'within_environment_spearman' else (
                int(common[k].sum()) if k in common else len(envs)))
        differences.append({'metric': k, 'full': float(points[FULL][k][0]),
                            'no_stage': float(points[ABLATION][k][0]),
                            'difference_full_minus_no_stage': float(points[FULL][k][0] - points[ABLATION][k][0]),
                            'ci_low': low, 'ci_high': high, 'environment_count': len(envs),
                            'matched_valid_environments': matched,
                            'valid_bootstrap_repeats': len(finite),
                            'bootstrap_fraction_favoring_full': float(favor.mean()) if len(finite) else np.nan,
                            'better_direction': 'lower' if k in ERRORS else 'higher'})
    return ensemble, per_environment, pd.DataFrame(summary), pd.DataFrame(differences)


def write_report(predictions, seeds, output, repeats=5000, random_state=20261016):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ['ensemble_predictions.csv', 'by_environment.csv', 'ensemble_summary.csv', 'paired_environment_bootstrap.csv']
    for name, frame in zip(names, compare(predictions, seeds, repeats, random_state)):
        temp = output / (name + '.tmp')
        frame.to_csv(temp, index=False)
        temp.replace(output / name)
    (output / 'statistics_protocol.json').write_text(json.dumps({
        'prediction_aggregation': 'Mean predictions across the declared seeds first, then calculate metrics.',
        'environment_aggregation': 'Equal weight for within-environment/selection metrics; Fisher-z for within-environment PCC.',
        'overall_metrics': 'All records pooled; environment bootstrap keeps each sampled environment intact, including duplicate draws.',
        'paired_difference': 'Full G-PACE minus no-stage; same environment draws for both models.',
        'bootstrap_unit': 'Environment, with replacement; no resampling of rows within environments.',
        'bootstrap_repeats': repeats, 'random_state': random_state,
        'correlation_missingness': 'Within-environment metrics use the intersection of valid environments for the two models; counts reported.',
        'within_pcc_difference': 'Difference of Fisher-z aggregate PCCs, not arithmetic mean of per-environment PCC differences.',
        'selection': 'k=ceil(q*n), equal environment weights, deterministic hybrid-ID tie breaking.',
        'inference': 'Percentile intervals, unadjusted for multiple comparisons. Bootstrap fractions are not P values.',
        'interpretation': 'Retrospective selection on observed trial yield, not a prospective genetic-gain experiment.'
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='2026,2027,2028')
    parser.add_argument('--bootstrap-repeats', type=int, default=5000)
    parser.add_argument('--random-state', type=int, default=20261016)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, dtype={'Experiment': str, 'Variety_ID': str})
    write_report(frame, [int(s) for s in args.seeds.split(',')], args.output_dir,
                 args.bootstrap_repeats, args.random_state)


if __name__ == '__main__':
    main()
