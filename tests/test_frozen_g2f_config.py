"""Ensure saved-parameter refits cannot retune or change original group offsets."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('frozen_g2f_runner',ROOT/'scripts/run_g2f_benchmarks.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def prepared():
    records=pd.DataFrame({'record_id':range(6),'Experiment':['H','H','T','T','T','T'],
        'Variety_ID':[f'G{i}' for i in range(6)],'grain.yield':[1.,2.,3.,4.,5.,6.],
        'Year':[2022,2023,2024,2024,2024,2024]})
    return SimpleNamespace(data=SimpleNamespace(records=records,audit={}),
        training_indices=np.array([0,1]),external_indices=np.array([2,3,4,5]))

class FixedConfigTests(unittest.TestCase):
    def test_original_two_groups_use_saved_parameters_and_seed_offsets(self):
        for filename,names in [('g2f_base_benchmark_config.json',['reaction_norm_rff','ceris_jgra']),
                               ('g2f_extension_benchmark_config.json',['ec_gblup','geformer_style'])]:
            with self.subTest(group=filename),tempfile.TemporaryDirectory() as temp:
                calls=[];config=ROOT/'configs'/filename
                expected=json.loads(config.read_text())['models']
                class Model:
                    def fit(self,data,indices):
                        np.testing.assert_array_equal(indices,[0,1]);return self
                    def predict(self,data,indices):
                        np.testing.assert_array_equal(indices,[2,3,4,5]);return np.array([3.1,4.2,4.8,5.9])
                def make(name,params,seed,device):
                    calls.append((name,params,seed));return Model()
                args=['run','--data-dir',temp,'--output-dir',temp,'--config-json',str(config),
                    '--models',','.join(names),'--device','cpu']
                with patch.object(sys,'argv',args),patch.object(runner,'load_g2f_2024_data',return_value=prepared()),\
                     patch.object(runner,'_select_config',side_effect=AssertionError('Saved refit attempted tuning')),\
                     patch.object(runner,'_make_model',side_effect=make):runner.main()
                self.assertEqual([x[0] for x in calls],[names[0]]*3+[names[1]]*3)
                self.assertEqual([x[2] for x in calls],[2026,2027,2028,12026,12027,12028])
                for name,params,_ in calls:self.assertEqual(params,expected[name])
                manifest=json.loads((Path(temp)/'g2f_benchmark_manifest.json').read_text())
                self.assertEqual(manifest['selected_models'],expected)
                self.assertNotIn('search_trials',manifest['outputs'])
    def test_missing_saved_model_stops_before_fitting(self):
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'bad.json';config.write_text('{"models":{}}')
            args=['run','--data-dir',temp,'--output-dir',temp,'--config-json',str(config)]
            with patch.object(sys,'argv',args),patch.object(runner,'load_g2f_2024_data',return_value=prepared()),\
                 patch.object(runner,'_make_model',side_effect=AssertionError('Unexpected fit')):
                with self.assertRaisesRegex(ValueError,'lacks requested models'):runner.main()
if __name__=='__main__':unittest.main()
