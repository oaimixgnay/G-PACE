"""Recompute environment decomposition and selection from all six models."""
from pathlib import Path
import pandas as pd
from analysis_components import _seed_average, _component_tables, _scores, _paired_bootstrap

root=Path(__file__).resolve().parent.parent
src=root.parents[0]/'data/source_data'
import argparse
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=root.parents[0]/'outputs/g2f_recomputed');args=p.parse_args()
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
models=('g_pace','e2e_catboost','ec_gblup','reaction_norm_rff','ceris_jgra','geformer_style')
raw=pd.read_csv(src/'g2f_combined_predictions.csv.gz')
avg=_seed_average(raw)
env,summary=_component_tables(avg,models)
selection=pd.DataFrame([{'model':model,'Experiment':experiment,'selection_fraction':fraction,**_scores(g,fraction)} for (model,experiment),g in avg.groupby(['model','Experiment']) for fraction in (.1,.2)])
boot=_paired_bootstrap(env,'g_pace',list(models[1:]),5000,20260915)
for frame,name in [(env,'g2f_per_environment.csv'),(summary,'g2f_decomposition_summary.csv'),(selection,'g2f_selection_per_environment.csv'),(boot,'g2f_paired_environment_bootstrap.csv')]:
    frame.to_csv(out/name,index=False)
print('Recomputed 132 environment rows, 264 selection rows and 25 paired-bootstrap rows.')
