# Validation commands

The lightweight checks below operate on supplied numeric references or synthetic protocol inputs.

```bash
python scripts/verify_release.py --check-metrics
python scripts/recompute_g2f_controls.py
python analyses/g2f_six_models/recompute_g2f.py
python experiments/g2f_tcn_film/tests/test_protocol.py
python extensions/s5/tests/test_s5_extension.py
python tests/test_frozen_g2f_config.py
```

The release check validates file integrity, Python/Bash syntax, frozen experiment-source identities, the current G2F TCN manifest, six-model prediction metrics, comparator identity and all 132 model–environment correlations. It also recomputes empirical P values for six groups of 1,000 exposure permutations.

The paired-control recalculation checks point estimates and 5,000-draw whole-environment bootstrap intervals for phenology vs calendar, phenology vs no stage, and calendar vs no stage. The six-model post-processing recalculates environment performance, decomposition and candidate selection.

The existing TCN protocol and grouped-holdout extension tests contain five and seven tests, respectively. Real-data training is performed separately through the training workflows.

Two additional saved-parameter tests verify that comparator refits do not invoke tuning, preserve original group seed offsets, and reject incomplete parameter files before fitting.
