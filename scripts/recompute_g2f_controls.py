"""Recompute paired CPU G2F comparisons from supplied record-level predictions."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/g2f_controls_recomputed')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('paired_recalculation',ROOT/'experiments/g2f_stage_ablation/scripts/paired_statistics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=pd.read_csv(ROOT/'data/reference_runs/g2f_fixed_calendar/combined_predictions.csv.gz')
    ref=pd.read_csv(ROOT/'data/reference_summaries/g2f_controls/ensemble_summary.csv').set_index('model')
    differences=pd.read_csv(ROOT/'data/reference_summaries/g2f_controls/paired_differences.csv')
    checked=[]
    for first,second in [('phenology','calendar'),('phenology','no_stage'),('calendar','no_stage')]:
        module.FULL=first;module.ABLATION=second;module.MODELS=(first,second)
        data=raw[raw.model.isin([first,second])].copy()
        ensemble,env,summary,paired=module.compare(data,[2026,2027,2028],5000,20261016)
        for row in summary.itertuples():
            for key in ref.columns:
                if key=='seeds':continue
                assert np.isclose(getattr(row,key),ref.loc[row.model,key],atol=1e-10,rtol=0),(row.model,key)
        expected=differences[differences.first_model.eq(first)&differences.second_model.eq(second)].set_index('metric')
        if expected.empty:
            expected=pd.read_csv(ROOT/'data/reference_runs/g2f_fixed_calendar/report/calendar_vs_no_stage_paired_bootstrap.csv').rename(columns={'difference_first_minus_second':'difference'}).set_index('metric')
        for row in paired.itertuples():
            for got,want in [('difference_full_minus_no_stage','difference'),('ci_low','ci_low'),('ci_high','ci_high')]:
                assert np.isclose(getattr(row,got),expected.loc[row.metric,want],atol=1e-10,rtol=0),(first,second,row.metric,got)
        paired=paired.rename(columns={'full':'first_value','no_stage':'second_value','difference_full_minus_no_stage':'difference','bootstrap_fraction_favoring_full':'bootstrap_fraction_favoring_first'})
        paired['first_model']=first;paired['second_model']=second
        target=args.output_dir/f'{first}_vs_{second}';target.mkdir(exist_ok=True)
        for frame,name in [(summary,'summary.csv'),(env,'by_environment.csv'),(paired,'paired_bootstrap.csv')]:frame.to_csv(target/name,index=False)
        checked.append({'first':first,'second':second,'metrics':len(paired),'bootstrap_repeats':5000})
        print('PASS:',first,'vs',second,'point estimates and paired percentile intervals')
    (args.output_dir/'validation.json').write_text(json.dumps({'comparisons':checked,'scope':'Recalculated from supplied CPU predictions, not a training rerun.'},indent=2)+'\n')
if __name__=='__main__':main()
