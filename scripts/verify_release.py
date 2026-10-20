"""Verify this assembly and recompute current G2F/permutation evidence.

The checks use supplied predictions and recorded experiment metadata.
"""
import argparse,hashlib,json,sys,ast,subprocess,platform
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gpace.metrics import regression_metrics

def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-metrics',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/validation.json')
    args=p.parse_args();checks=[]
    def record(name,**details):checks.append({'check':name,'status':'pass',**details});print('PASS:',name)
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path']
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],f'Hash mismatch: {item["path"]}')
    record('Assembly file integrity',files=len(manifest['files']))
    for item in json.loads((ROOT/'docs/experiment_source_inventory.json').read_text()):
        require(hashlib.sha256((ROOT/item['bundled_path']).read_bytes()).hexdigest()==item['sha256'],item['bundled_path'])
    record('G-PACE experiment source integrity')
    for path in ROOT.rglob('*.py'):
        if 'outputs' not in path.parts:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    for path in ROOT.rglob('*.sh'):subprocess.run(['bash','-n',str(path)],check=True)
    record('Python AST and Bash syntax')
    source=ROOT/'data/source_data'
    null=pd.read_csv(source/'phenology_shuffle_null.csv');summary=pd.read_csv(source/'phenology_shuffle_summary.csv')
    require(len(null)==6000 and len(summary)==6,'Permutation counts')
    for row in summary.itertuples():
        group=null[null.SNP.eq(row.SNP)&null.stress.eq(row.stress)]
        require(len(group)==1000,'Permutation group size')
        value=(1+(group.shuffled_interaction_beta.abs()>=abs(row.observed_interaction_beta)).sum())/1001
        require(abs(value-row.empirical_two_sided_p)<1e-12,'Permutation empirical P')
    record('Six within-environment exposure permutation summaries',draws=len(null))
    manifest=json.loads((ROOT/'data/run_records/g2f_tcn_film/g2f_run_manifest.json').read_text())
    require(manifest['status']=='complete' and manifest['config']['stage_model']=='tcn_film','Current TCN run manifest')
    require(manifest['config']['sequence_length']==100,'TCN sequence length')
    record('Current G2F manifest uses TCN-FiLM')
    if args.check_metrics:
        data=pd.read_csv(source/'g2f_combined_predictions.csv.gz')
        require(len(data)==170748,'Six-model row count')
        require(not data.duplicated(['model','seed','record_id']).any(),'Duplicated predictions')
        require(set(data.seed)=={2026,2027,2028},'Seed set')
        records=None
        for (model,seed),g in data.groupby(['model','seed']):
            require(len(g)==9486 and g.Experiment.nunique()==22,'Evaluation size')
            keys=g.set_index('record_id')[['Experiment','Variety_ID','observed']].sort_index()
            if records is None:records=keys
            pd.testing.assert_frame_equal(keys,records)
        rows=[{'model':model,'seed':seed,**regression_metrics(g)} for (model,seed),g in data.groupby(['model','seed'])]
        runs=pd.DataFrame(rows);ref=pd.read_csv(source/'g2f_model_run_summary.csv').set_index('model')
        keys=['pcc','r2','rmse','within_environment_pcc','within_environment_spearman','top20_recall']
        for model,g in runs.groupby('model'):
            for key in keys:
                require(np.isclose(g[key].mean(),ref.loc[model,key+'_mean'],atol=1e-10,rtol=0),f'{model} {key} mean')
                require(np.isclose(g[key].std(ddof=1),ref.loc[model,key+'_sd'],atol=1e-10,rtol=0),f'{model} {key} SD')
        record('Current G2F metrics recomputed from 170748 predictions',model_seed_groups=len(runs))
        old=pd.read_csv(ROOT/'data/frozen_comparators/g2f_predictions.csv.gz')
        columns=['model','seed','record_id','Experiment','Variety_ID','observed','predicted']
        a=data[data.model.ne('g_pace')][columns].sort_values(columns[:3]).reset_index(drop=True)
        b=old[old.model.ne('g_pace')][columns].sort_values(columns[:3]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-13,rtol=0)
        record('Five frozen comparators retained after replacing G-PACE')
        avg=data.groupby(['model','record_id','Experiment','Variety_ID'],as_index=False).agg(observed=('observed','first'),predicted=('predicted','mean'))
        envref=pd.read_csv(source/'g2f_per_environment.csv')
        require(len(envref)==132,'Per-environment reference size')
        for (model,env),g in avg.groupby(['model','Experiment']):
            row=envref[envref.model.eq(model)&envref.Experiment.eq(env)].iloc[0]
            require(np.isclose(g.observed.corr(g.predicted),row.deviation_pcc,atol=1e-10,rtol=0),'Per-environment PCC')
        record('All 132 current environment-model correlations recomputed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'python':platform.python_version(),'checks':checks,'scope':'Integrity and supplied-prediction recalculation.'},indent=2)+'\n')
    print('Validation report:',args.output)
if __name__=='__main__':main()
