#!/usr/bin/env python3
"""Same-architecture fixed-calendar control with training-only window estimation."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

# The imported CPU-only ablation helper hides CUDA; restore the caller's selection
# before importing/initializing torch for the separately budgeted stage fits.
saved_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
import run_stage_ablation as old
if saved_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=saved_cuda
import paired_statistics as stats
from gpace.data import EXPOSURE_COLUMNS
from gpace.g2f import load_g2f_2024_data
from gpace.models import YieldModel,make_exposure_model
from calendar_targets import CalendarTargets
from gpu_budget import GPUBudget

log=old.log
sha256=old.sha256
atomic_json=old.atomic_json
atomic_csv=old.atomic_csv


def runtime_versions():
    return {p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','joblib','catboost','threadpoolctl','torch')}


def source_preflight(args,config,seeds):
    original=args.existing_evaluation_dir.resolve()
    paired=args.paired_ablation_dir.resolve()
    # Verify the original model/data source; the completed CPU prediction table
    # is the reused reference, so old OOF cache files are not needed here.
    for name in old._FILES.values():
        if not (args.data_dir/name).is_file():raise FileNotFoundError(args.data_dir/name)
    if not (original/'g2f_run_manifest.json').is_file():raise FileNotFoundError(original/'g2f_run_manifest.json')
    for name in ('run_manifest.json','paired_predictions.csv'):
        if not (paired/name).is_file():raise FileNotFoundError(f'Missing completed paired ablation: {paired/name}')
    manifest=json.loads((paired/'run_manifest.json').read_text())
    source=json.loads((original/'g2f_run_manifest.json').read_text())
    if source.get('status')!='complete' or source['stage_params']!=config['stage_params'] or source['yield_params']!=config['yield_params']:
        raise ValueError('Original run is incomplete or its configuration differs.')
    if source['config']['sequence_length']!=100 or source['config']['crossfit_folds']!=3:
        raise ValueError('Require the original day-100, dual-axis 3-fold reference.')
    for path in (PACKAGE/'src/gpace').glob('*.py'):
        if source['code_sha256'].get(str(path.relative_to(PACKAGE)))!=sha256(path):
            raise ValueError(f'Bundled model/data code differs from the original run: {path.name}')
    if manifest.get('status')!='complete' or manifest['provenance']['backend']!='CPU':
        raise ValueError('Require the completed paired CPU ablation as the yield-head reference.')
    provenance=manifest['provenance']
    if manifest.get('training_signature')!=old.object_hash(provenance):
        raise ValueError('Paired-reference manifest signature is inconsistent.')
    if provenance['config']!=config or provenance['threads']!=args.threads or provenance['seeds']!=seeds:
        raise ValueError('Configuration, seeds and thread count must match the completed paired CPU ablation.')
    versions=runtime_versions()
    for key,value in provenance['versions'].items():
        if versions.get(key)!=value:
            raise ValueError(f'Runtime {key} differs from the paired CPU reference ({versions.get(key)} vs {value}).')
    if versions['torch'].split('+')[0]!=source['runtime_versions']['torch'].split('+')[0]:
        raise ValueError('Use the same PyTorch release as the original stage run.')
    original_device=source['config']['device']
    if args.stage_device=='cuda' and original_device!='cuda':
        raise ValueError('GPU cache reuse requires an original GPU stage run.')
    if args.stage_device=='cuda' and versions['torch']!=source['runtime_versions']['torch']:
        raise ValueError('CUDA stage control must use the same PyTorch CUDA build as the reused phenology run.')
    return source,manifest,versions


def clear_device(device):
    gc.collect()
    if device=='cuda':
        import torch
        torch.cuda.empty_cache()


def atomic_npz(path,**values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def stage_task(data,fit,predict,targets,kind,config,seed,device,path,signature,resume,save_model=False):
    marker=path.with_suffix('.json')
    if resume and marker.is_file():
        done=json.loads(marker.read_text())
        if done.get('training_signature')!=signature or not path.is_file() or done['prediction_sha256']!=sha256(path):
            raise ValueError(f'Invalid cached stage completion: {path}')
        prediction=old.load_cache(path,predict)
        log(f'[RESUME STAGE] {kind}, seed={seed}, {path.name}')
        return prediction
    stage_data,windows=targets.training_data(fit,kind)
    log(f'[STAGE] {kind}, seed={seed}, fit={len(fit)}, predict={len(predict)}, windows={windows}')
    params={**config['stage_params'],'device':device,'verbose':True}
    model=make_exposure_model('tcn_film',random_state=seed,**params)
    model.fit(stage_data,fit)
    # Prediction sees only day-100 G/E inputs and does not read stage labels.
    prediction=model.predict(data,predict)
    if prediction.shape!=(len(predict),len(EXPOSURE_COLUMNS)) or not np.isfinite(prediction).all():
        raise ValueError('Invalid stage prediction.')
    if save_model:
        import torch
        weight_path=path.parent/(kind+'_stage_weights.pt')
        temp=weight_path.with_name(weight_path.name+'.tmp')
        torch.save({k:v.detach().cpu() for k,v in model.network_.state_dict().items()},temp);temp.replace(weight_path)
        prep_path=path.parent/(kind+'_stage_preprocessing.joblib')
        temp=prep_path.with_name(prep_path.name+'.tmp')
        joblib.dump({'inputs':model.inputs_,'target_scaler':model.target_scaler_},temp);temp.replace(prep_path)
    atomic_npz(path,row_indices=predict,predictions=prediction)
    atomic_json(marker,{'training_signature':signature,'kind':kind,'stage_seed':seed,'fit_rows':len(fit),
                        'predict_rows':len(predict),'fit_indices_sha256':hashlib.sha256(np.asarray(fit,dtype=np.int64).tobytes()).hexdigest(),
                        'windows':windows,'best_epoch':model.best_epoch_,'best_validation_loss':model.best_validation_loss_,
                        'prediction_sha256':sha256(path)})
    del model,stage_data
    clear_device(device)
    return prediction


def crossfit(data,train,test,targets,kind,config,seed,device,folder,signature,resume):
    rows=data.records.loc[train]
    total=np.zeros((len(train),len(EXPOSURE_COLUMNS)),dtype=float)
    counts=np.zeros(len(train),dtype=int)
    for axis,column in enumerate(('Variety_ID','Experiment')):
        groups=rows[column].astype(str).to_numpy()
        unique=np.unique(groups)
        if len(unique)<3:raise ValueError(f'Too few {column} groups for three folds.')
        np.random.default_rng(seed+axis).shuffle(unique)
        for fold,held in enumerate(np.array_split(unique,3)):
            mask=np.isin(groups,held)
            path=folder/f'axis{axis}_fold{fold}.npz'
            values=stage_task(data,train[~mask],train[mask],targets,kind,config,
                              seed+axis*100+fold,device,path,signature,resume)
            total[mask]+=values;counts[mask]+=1
    if not np.all(counts==2):raise ValueError('OOF count mismatch.')
    external=stage_task(data,train,test,targets,kind,config,seed+999,device,
                        folder/'test_exposures.npz',signature,resume,save_model=True)
    return total/counts[:,None],external


def fit_yield(data,train,test,z,tz,config,seed,kind,folder,signature,resume,threads):
    folder.mkdir(parents=True,exist_ok=True)
    path,weights,marker=(folder/name for name in ('predictions.csv','yield_model.joblib','completed.json'))
    expected=data.records.loc[test,['record_id','Experiment','Variety_ID','grain.yield']].rename(columns={'grain.yield':'observed'})
    if resume and marker.is_file():
        done=json.loads(marker.read_text())
        if done.get('training_signature')!=signature:raise ValueError('Stale yield completion marker.')
        for p in (path,weights):
            if not p.is_file() or done['sha256'][p.name]!=sha256(p):raise ValueError(f'Modified completed file: {p}')
        frame=pd.read_csv(path,dtype={'Experiment':str,'Variety_ID':str})
        old.check_prediction(frame,expected,kind,seed)
        log(f'[RESUME YIELD] {kind}, seed={seed}')
        return frame
    log(f'[YIELD CPU] {kind}, seed={seed}')
    model=YieldModel(mode='mean_residual_hybrid',genotype_components=config['yield_genotype_components'],
                     environment_components=config['yield_environment_components'],head='catboost',
                     head_params={**config['yield_params'],'task_type':'CPU','thread_count':threads},
                     random_state=seed,device='cpu')
    with threadpool_limits(limits=threads):
        model.fit(data,train,z);predicted=model.predict(data,test,tz)
    frame=expected.copy();frame['predicted']=predicted;frame['model']=kind;frame['seed']=seed;frame['scenario']='external_2024'
    old.check_prediction(frame,expected,kind,seed)
    temp=weights.with_name(weights.name+'.tmp');joblib.dump(model,temp);temp.replace(weights)
    atomic_csv(path,frame)
    atomic_json(marker,{'training_signature':signature,'model':kind,'seed':seed,'sha256':{p.name:sha256(p) for p in (path,weights)}})
    del model;gc.collect()
    return frame


def pair_report(frame,first,second,seeds,output,repeats,random_state):
    subset=frame[frame.model.isin([first,second])].copy()
    subset['model']=subset.model.map({first:stats.FULL,second:stats.ABLATION})
    ensemble,env,summary,ci=stats.compare(subset,seeds,repeats,random_state)
    for f in (ensemble,env,summary):f['model']=f.model.map({stats.FULL:first,stats.ABLATION:second})
    ci=ci.rename(columns={'full':'first_model_value','no_stage':'second_model_value',
                          'difference_full_minus_no_stage':'difference_first_minus_second',
                          'bootstrap_fraction_favoring_full':'bootstrap_fraction_favoring_first'})
    ci.insert(0,'second_model',second);ci.insert(0,'first_model',first)
    output.mkdir(parents=True,exist_ok=True)
    name=f'{first}_vs_{second}'
    for suffix,f in [('ensemble_predictions',ensemble),('by_environment',env),('summary',summary),('paired_bootstrap',ci)]:
        atomic_csv(output/f'{name}_{suffix}.csv',f)


def verify_paired_predictions(paired_path,data,test,seeds):
    frame=pd.read_csv(paired_path,dtype={'Experiment':str,'Variety_ID':str})
    # Existing two-model helper checks every seed and every record, never an intersection.
    stats.ensemble_predictions(frame,seeds)
    expected=data.records.loc[test,['record_id','Experiment','Variety_ID','grain.yield']].rename(columns={'grain.yield':'observed'})
    for seed in seeds:
        for label in stats.MODELS:
            old.check_prediction(frame[frame.seed.eq(seed)&frame.model.eq(label)],expected,label,seed)
    return frame


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--existing-evaluation-dir',type=Path,required=True)
    p.add_argument('--paired-ablation-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage-device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--seeds',default='2026,2027,2028')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--memory-budget-gb',type=float,default=20.)
    p.add_argument('--bootstrap-repeats',type=int,default=5000)
    p.add_argument('--random-state',type=int,default=20261016)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.threads<1 or args.bootstrap_repeats<1 or args.memory_budget_gb<=2:raise ValueError('Invalid resource/statistics settings.')
    if args.stage_device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    seeds=old.parse_seeds(args.seeds)
    config=json.loads((PACKAGE/'config/g2f_tcn_film_frozen.json').read_text())
    source_manifest,paired_manifest,versions=source_preflight(args,config,seeds)
    if args.preflight_only:
        if args.stage_device=='cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
        log('[CHECK COMPLETE] Configuration, dependencies and source files checked; no model fitting.')
        return
    output=args.output_dir.resolve()
    for source in (args.data_dir,args.existing_evaluation_dir,args.paired_ablation_dir):
        source=source.resolve()
        if output==source or output in source.parents or source in output.parents:
            raise ValueError('New output must be disjoint from every input/source run directory.')
    import fcntl
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=(output.parent/(output.name+'.lock')).open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise RuntimeError('Another job owns this output directory.') from exc
    data_hashes=old.verify_data_hashes(args.data_dir,source_manifest)
    if data_hashes!=paired_manifest['provenance']['inputs']:raise ValueError('Inputs differ from the paired CPU ablation.')
    code_paths=sorted([*list((PACKAGE/'src/gpace').glob('*.py')),*list((PACKAGE/'scripts').glob('*.py'))])
    provenance={'config':config,'seeds':seeds,'threads':args.threads,'stage_device':args.stage_device,'memory_budget_gb':args.memory_budget_gb,
                'versions':versions,'inputs':data_hashes,
                'paired_manifest_sha256':sha256(args.paired_ablation_dir/'run_manifest.json'),
                'paired_predictions_sha256':sha256(args.paired_ablation_dir/'paired_predictions.csv'),
                'source_manifest_sha256':sha256(args.existing_evaluation_dir/'g2f_run_manifest.json'),
                'code_sha256':{str(p.relative_to(PACKAGE)):sha256(p) for p in code_paths}}
    signature=old.object_hash(provenance)
    path=output/'calendar_run_manifest.json'
    if path.is_file():
        if not args.resume or json.loads(path.read_text()).get('training_signature')!=signature:
            raise ValueError('Resume mismatch in code/configuration/runtime/data; use a new output directory.')
    elif output.exists() and any(output.iterdir()):raise ValueError('Nonempty output without matching manifest.')
    output.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','training_signature':signature,'provenance':provenance,
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'window_policy':'Each fit uses medians from its historical fitting rows only; common boundaries across all G and E within that fit.',
              'comparison_policy':'No new parameter or window search. Target definition is the stage-model experimental difference. Yield heads share the CPU backend.',
              'stage_backend_policy':'CUDA: reuse completed CUDA phenology stage/CPU yield reference. CPU: refit BOTH phenology and calendar stage models and their CPU yield heads.',
              'test_policy':'2024 already inspected in earlier experiments; this is a retrospective robustness analysis, not a new blind evaluation.'}
    atomic_json(path,manifest)
    budget=GPUBudget(args.stage_device,args.memory_budget_gb)
    try:
        budget.start()
        log('[DATA] Loading day-100 model inputs and full historical weather for label construction.')
        prepared=load_g2f_2024_data(args.data_dir,sequence_length=100)
        full=load_g2f_2024_data(args.data_dir,sequence_length=None)
        data,train,test=old.validate_prepared(prepared,source_manifest)
        targets=CalendarTargets(data,full.data)
        reference=verify_paired_predictions(args.paired_ablation_dir/'paired_predictions.csv',data,test,seeds)
        frames=[reference[reference.model.eq(stats.ABLATION)].assign(model='no_stage')]
        if args.stage_device=='cuda':
            frames.append(reference[reference.model.eq(stats.FULL)].assign(model='phenology'))
            kinds=('calendar',)
        else:
            kinds=('phenology','calendar')
        for seed in seeds:
            for kind in kinds:
                folder=output/'runs'/f'seed_{seed}'/kind
                z,tz=crossfit(data,train,test,targets,kind,config,seed,args.stage_device,folder/'stage',signature,args.resume)
                frame=fit_yield(data,train,test,z,tz,config,seed,kind,folder/'yield',signature,args.resume,args.threads)
                frames.append(frame)
                atomic_csv(output/'combined_predictions.csv',pd.concat(frames,ignore_index=True))
                del z,tz;clear_device(args.stage_device)
        combined=pd.concat(frames,ignore_index=True)
        log('[STATISTICS] Same ensemble-first paired environment analysis as the completed ablation.')
        pair_report(combined,'phenology','calendar',seeds,output/'report',args.bootstrap_repeats,args.random_state)
        pair_report(combined,'calendar','no_stage',seeds,output/'report',args.bootstrap_repeats,args.random_state)
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                        data_audit=data.audit,observed_process_gpu_peak_gb=budget.peak_bytes/1e9)
        atomic_json(path,manifest)
        archive=output/'g2f_fixed_calendar_results.zip'
        temp=archive.with_name(archive.name+'.tmp')
        files=[path,output/'combined_predictions.csv',*sorted((output/'report').glob('*.csv')),
               *sorted((output/'runs').glob('seed_*/*/stage/*.json')),*sorted((output/'runs').glob('seed_*/*/yield/completed.json'))]
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,str(p.relative_to(output)))
        temp.replace(archive)
        log(f'[COMPLETE] {output}');log(f'[SEND] {archive}')
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}')
        atomic_json(path,manifest);raise
    finally:
        budget.close();fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
