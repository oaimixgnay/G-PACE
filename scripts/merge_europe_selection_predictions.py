"""Combine matched European predictions for six-model selection."""
import argparse,sys
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--core',required=True);p.add_argument('--benchmarks',required=True);p.add_argument('--ec-gblup',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
frames=[]
for path,models in [(a.core,['mean_residual_hybrid_catboost','e2e_catboost']),(a.benchmarks,['reaction_norm_kernel','ceris_jgra','geformer_style']),(a.ec_gblup,['ec_gblup'])]:
 d=pd.read_csv(path);frames.append(d[d.model.isin(models)&d.scenario.str.startswith('external')])
out=pd.concat(frames,ignore_index=True)
assert out.model.nunique()==6
assert not out.duplicated(['model','seed','scenario','record_id']).any()
assert out.groupby(['scenario','record_id']).observed.nunique().eq(1).all()
counts=out.groupby(['model','seed','scenario']).size()
for scenario in ['external_g_ne','external_ng_ne']:
 assert counts.xs(scenario,level='scenario').nunique()==1
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);out.to_csv(path,index=False);print(path)

