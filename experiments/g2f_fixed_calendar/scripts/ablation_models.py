"""Ablate Z and G×Z while preserving the original yield fitting procedure."""
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from gpace.features import interaction_design
from gpace.models import YieldModel


@dataclass
class StageAblationYieldModel(YieldModel):
    include_stage: bool = True

    def _design(self, data, row_indices, exposures, fit):
        if self.mode != 'mean_residual_hybrid':
            raise ValueError('This experiment fixes the mean–residual hybrid architecture.')
        if self.include_stage:
            return super()._design(data, row_indices, exposures, fit)
        # Delete the columns entirely, rather than inserting all-zero columns.
        # G, E, G×E and their scaling are exactly the original direct block.
        genotype, environment = self.inputs_.rows(data, row_indices)
        direct = interaction_design(genotype, environment)
        if fit:
            self.direct_scaler_ = StandardScaler().fit(direct)
        return self.direct_scaler_.transform(direct).astype(np.float32)
