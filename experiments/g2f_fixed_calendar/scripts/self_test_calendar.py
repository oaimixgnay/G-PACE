#!/usr/bin/env python3
"""Small real-Torch/real-CatBoost checks, including full CPU orchestration."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
import numpy as np
import pandas as pd
import torch
from gpace.data import GPACEData,EXPOSURE_COLUMNS
from gpace.g2f import G2FPreparedData,G2F_DAILY_COLUMNS,_FILES,_exposure_vector
from calendar_targets import CalendarTargets
import run_fixed_calendar as runner
import paired_statistics as stats


def fixture():
    rng=np.random.default_rng(47)
    genotype=rng.integers(0,3,size=(12,30)).astype(np.float32)
    weather=rng.normal(size=(8,180,10)).astype(np.float32)
    weather[:,:,0]+=10;weather[:,:,1]+=25;weather[:,:,5]+=200;weather[:,:,6]*=.01;weather[:,:,6]+=.5
    rows=[]
    for env in range(8):
        w=pd.DataFrame(weather[env],columns=G2F_DAILY_COLUMNS)
        for g in range(10 if env<6 else 12):
            p,s=52+g%5+env,54+g%5+env
            row=dict(record_id=len(rows),Experiment=f'E{env}',Variety_ID=f'G{g:02d}',genotype_index=g,
                     environment_index=env,Year=2018+env if env<6 else 2024,
                     pollen_dap=float(p) if env<6 else np.nan,silk_dap=float(s) if env<6 else np.nan,
                     scenario='training' if env<6 else 'external_2024')
            row['grain.yield']=float(7+env*.3+genotype[g,:3].sum()*.2+rng.normal(0,.1))
            label=_exposure_vector(w,p,s) if env<6 else np.full(11,np.nan)
            row.update(dict(zip(EXPOSURE_COLUMNS,label)));rows.append(row)
    records=pd.DataFrame(rows)
    train=records.index[records.scenario.eq('training')].to_numpy(int)
    test=records.index[records.scenario.eq('external_2024')].to_numpy(int)
    audit={'training_gxe_means':len(train),'external_2024_labeled_records':len(test),'external_2024_environment_count':2}
    full=GPACEData(records,tuple(f'G{x:02d}' for x in range(12)),genotype,tuple(f'S{x}' for x in range(30)),
                  tuple(f'E{x}' for x in range(8)),weather,np.full(8,180),G2F_DAILY_COLUMNS,pd.DataFrame(),pd.DataFrame(),audit)
    daily=replace(full,environment_sequences=weather[:,:100].copy(),environment_sequence_lengths=np.full(8,100))
    return G2FPreparedData(daily,train,test),G2FPreparedData(full,train,test)


def assert_raises(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected ValueError')


def main():
    torch.set_num_threads(1)
    prepared,full=fixture()
    data,train,test=prepared.data,prepared.training_indices,prepared.external_indices
    target=CalendarTargets(data,full.data)
    fit=train[~data.records.loc[train,'Variety_ID'].eq('G00').to_numpy()]
    labels,windows=target.training_data(fit,'calendar')
    assert labels.records.loc[labels.records.index.difference(fit),list(EXPOSURE_COLUMNS)].isna().all().all()
    for env,group in labels.records.loc[fit].groupby('Experiment'):
        values=group[list(EXPOSURE_COLUMNS)].to_numpy(float)
        np.testing.assert_allclose(values,np.repeat(values[:1],len(values),axis=0),atol=0)
    changed=data.records.copy()
    changed.loc[changed.index.difference(fit),['pollen_dap','silk_dap']]=1000
    changed_data=replace(data,records=changed)
    changed_target=CalendarTargets(changed_data,full.data)
    altered,other_windows=changed_target.training_data(fit,'calendar')
    assert windows==other_windows
    np.testing.assert_allclose(labels.records.loc[fit,list(EXPOSURE_COLUMNS)],altered.records.loc[fit,list(EXPOSURE_COLUMNS)],atol=0)
    assert_raises(lambda:target.fixed_windows(test))
    for row in labels.records.loc[fit].itertuples():
        w=pd.DataFrame(full.data.environment_sequences[row.environment_index],columns=G2F_DAILY_COLUMNS)
        expected=_exposure_vector(w,windows['pollen_boundary_dap'],windows['silk_boundary_dap'])
        actual=labels.records.loc[row.Index,list(EXPOSURE_COLUMNS)].to_numpy(float)
        np.testing.assert_allclose(expected,actual,atol=0)
    altered_full=full.data.environment_sequences.copy()
    altered_full[6:,100:]+=999
    t2=CalendarTargets(data,replace(full.data,environment_sequences=altered_full))
    alternate,w2=t2.training_data(fit,'calendar')
    assert windows==w2
    np.testing.assert_allclose(alternate.records.loc[fit,list(EXPOSURE_COLUMNS)],labels.records.loc[fit,list(EXPOSURE_COLUMNS)],atol=0)
    print('PASS: fitting-only windows, held-out/test phenology isolation, shared calendar labels, historical future-label weather only.')

    config=json.loads((PACKAGE/'config/g2f_tcn_film_frozen.json').read_text())
    config['stage_params'].update(genotype_components=3,width=8,depth=1,kernel_size=3,batch_size=16,epochs=1,patience=1)
    config['yield_params']['iterations']=8
    config['yield_genotype_components']=3;config['yield_environment_components']=2
    seeds=[2026,2027,2028]
    with tempfile.TemporaryDirectory(prefix='gpace_calendar_test_') as tmp:
        root=Path(tmp);release=root/'release';inputs=root/'inputs';original=root/'original';paired=root/'paired';out=root/'output'
        for folder in (inputs,original,paired):folder.mkdir()
        shutil.copytree(PACKAGE/'src',release/'src')
        shutil.copytree(PACKAGE/'scripts',release/'scripts')
        (release/'config').mkdir();(release/'config/g2f_tcn_film_frozen.json').write_text(json.dumps(config))
        hashes={}
        for name in _FILES.values():
            p=inputs/name;p.write_text('synthetic input; not actual G2F data\n');hashes[name]=runner.sha256(p)
        versions=runner.runtime_versions()
        source={'status':'complete','config':{'sequence_length':100,'crossfit_folds':3,'device':'cuda'},
                'stage_params':config['stage_params'],'yield_params':config['yield_params'],
                'runtime_versions':{'torch':versions['torch']},'data_audit':data.audit,
                'code_sha256':{str(p.relative_to(release)):runner.sha256(p) for p in (release/'src/gpace').glob('*.py')},
                'input_files':{name:{'sha256':value} for name,value in hashes.items()}}
        (original/'g2f_run_manifest.json').write_text(json.dumps(source))
        paired_manifest={'status':'complete','provenance':{'backend':'CPU','config':config,'threads':1,'seeds':seeds,
                         'versions':{k:v for k,v in versions.items() if k!='torch'},'inputs':hashes}}
        paired_manifest['training_signature']=runner.old.object_hash(paired_manifest['provenance'])
        (paired/'run_manifest.json').write_text(json.dumps(paired_manifest))
        predictions=[]
        for seed in seeds:
            for model in stats.MODELS:
                frame=data.records.loc[test,['record_id','Experiment','Variety_ID','grain.yield']].rename(columns={'grain.yield':'observed'}).copy()
                frame['predicted']=8+data.genotype_matrix[frame.Variety_ID.str[1:].astype(int).to_numpy(),:3].sum(axis=1)*.1
                frame['model']=model;frame['seed']=seed;frame['scenario']='external_2024';predictions.append(frame)
        pd.concat(predictions).to_csv(paired/'paired_predictions.csv',index=False,float_format='%.17g')
        saved_package,saved_loader,saved_argv=runner.PACKAGE,runner.load_g2f_2024_data,sys.argv
        try:
            runner.PACKAGE=release
            runner.load_g2f_2024_data=lambda *a,**k:prepared if k['sequence_length']==100 else full
            sys.argv=['selftest','--data-dir',str(inputs),'--existing-evaluation-dir',str(original),
                      '--paired-ablation-dir',str(paired),'--output-dir',str(out),'--stage-device','cpu',
                      '--threads','1','--bootstrap-repeats','100','--resume']
            runner.main()
            assert json.loads((out/'calendar_run_manifest.json').read_text())['status']=='complete'
            first={str(p):runner.sha256(p) for p in (out/'runs').glob('seed_*/*/stage/*.npz')}
            assert len(first)==42
            runner.main()
            assert first=={str(p):runner.sha256(p) for p in (out/'runs').glob('seed_*/*/stage/*.npz')}
            ci=pd.read_csv(out/'report/phenology_vs_calendar_paired_bootstrap.csv')
            assert set(ci.first_model)=={'phenology'} and set(ci.second_model)=={'calendar'}
            frame=pd.read_csv(out/'combined_predictions.csv')
            assert set(frame.model)=={'phenology','calendar','no_stage'}
            assert len(frame)==len(test)*3*3
            with __import__('zipfile').ZipFile(out/'g2f_fixed_calendar_results.zip') as z:
                assert 'report/phenology_vs_calendar_paired_bootstrap.csv' in z.namelist()
        finally:
            runner.PACKAGE,runner.load_g2f_2024_data,sys.argv=saved_package,saved_loader,saved_argv
    print('PASS: 42 real CPU TCN fits, six real CatBoost fits, same-device references, paired reports, ZIP and resume.')
    print('ALL CALENDAR CHECKS PASSED (synthetic data only; CUDA budget monitoring and full G2F training not tested locally).')


if __name__=='__main__':main()
