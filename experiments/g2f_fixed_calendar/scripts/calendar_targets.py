"""Training-only fixed calendar targets, with a fold-specific window estimator."""
from dataclasses import replace

import numpy as np
import pandas as pd

from gpace.data import EXPOSURE_COLUMNS
from gpace.g2f import _exposure_vector


class CalendarTargets:
    def __init__(self, day100, full_season):
        self.data = day100
        self.full_season = full_season
        if not day100.records[['record_id','Experiment','Variety_ID']].equals(
            full_season.records[['record_id','Experiment','Variety_ID']]
        ):
            raise ValueError('Full-season label source and day-100 inputs have different record identities/order.')
        if day100.environment_ids != full_season.environment_ids:
            raise ValueError('Environment ordering differs between weather sources.')
        if not np.array_equal(day100.environment_sequences, full_season.environment_sequences[:,:100], equal_nan=True):
            raise ValueError('Day-100 model inputs differ from the prefix of the full-weather label source.')
        self.channel_index = {k: full_season.environment_feature_names.index(k)
                              for k in ('T2M_MIN','T2M_MAX','ALLSKY_SFC_PAR_TOT','GWETROOT')}

    def fixed_windows(self, fit_indices):
        rows=self.data.records.loc[fit_indices]
        if len(rows)==0 or not rows.Year.between(2014,2023).all():
            raise ValueError('Calendar boundaries may only be estimated from the current historical fitting rows.')
        values=rows[['pollen_dap','silk_dap']].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Historical phenology dates required for the predefined window estimator.')
        pollen,silk=np.median(values,axis=0)
        p,s=int(round(float(pollen))),int(round(float(silk)))
        if p<3 or s<p:
            raise ValueError('Invalid historical median flowering dates.')
        return {'pollen_dap_median':float(pollen),'silk_dap_median':float(silk),
                'pollen_boundary_dap':p,'silk_boundary_dap':s,
                'early_zero_based_inclusive':[0,p-1],
                'flowering_zero_based_inclusive':[p-3,s+3],
                'filling_zero_based_inclusive':[s+4,s+53],
                'estimation_rows':len(rows),
                'estimator':'Record-weighted median of pollen/silk DAP in this fitting subset; rounded as in the original label construction.'}

    def training_data(self, fit_indices, kind):
        rows=self.data.records.copy(deep=True)
        # No held-out stage labels or measured test phenology are exposed to fit.
        rows.loc[:,list(EXPOSURE_COLUMNS)] = np.nan
        if kind=='phenology':
            rows.loc[fit_indices,list(EXPOSURE_COLUMNS)] = self.data.records.loc[fit_indices,list(EXPOSURE_COLUMNS)].to_numpy()
            windows={'definition':'Original genotype-specific observed-phenology labels; current fitting rows only.'}
        elif kind=='calendar':
            windows=self.fixed_windows(fit_indices)
            fitting=self.data.records.loc[fit_indices]
            lookup={}
            for env,index in fitting[['Experiment','environment_index']].drop_duplicates().itertuples(index=False,name=None):
                length=int(self.full_season.environment_sequence_lengths[index])
                series=self.full_season.environment_sequences[index,:length]
                weather=pd.DataFrame({k:series[:,v] for k,v in self.channel_index.items()})
                values=_exposure_vector(weather,windows['pollen_boundary_dap'],windows['silk_boundary_dap'])
                if not np.isfinite(values).all():
                    raise ValueError(f'Fixed calendar labels incomplete for {env}; refusing to change the training cohort.')
                lookup[env]=values
            labels=np.vstack([lookup[e] for e in fitting.Experiment])
            rows.loc[fit_indices,list(EXPOSURE_COLUMNS)] = labels
            windows['definition']='Same calendar boundaries for all genotypes and environments within this fitted stage model.'
        else:
            raise ValueError(kind)
        if not np.isfinite(rows.loc[fit_indices,list(EXPOSURE_COLUMNS)].to_numpy(float)).all():
            raise ValueError('Non-finite training labels.')
        other=rows.index.difference(fit_indices)
        assert rows.loc[other,list(EXPOSURE_COLUMNS)].isna().all().all()
        return replace(self.data,records=rows),windows
