# G-PACE

Research code accompanying the maize prediction experiments reported in the manuscript. The repository contains data preparation, grouped validation, model fitting, comparator implementations, and statistical analyses. The Python package is named `gpace`.

[中文说明](README_CN.md) · [Input data](docs/DATA.md) · [Workflow reference](docs/WORKFLOWS.md)

## Code organization

| Component | Location | Purpose |
|---|---|---|
| Data preparation | `src/gpace/data.py`, `src/gpace/g2f.py`, `src/gpace/features.py` | Read phenotypes/genotypes/environmental sequences, construct stage targets, and fit feature transformations |
| Data separation | `src/gpace/splits.py` and experiment runners | Grouped genotype/environment validation, complete environment-group holdouts, historical-year training and 2024 evaluation |
| G-PACE | `src/gpace/models.py`, `src/gpace/experiment.py`, `experiments/g2f_tcn_film/` | TCN–FiLM stage prediction and environmental-mean/genotype-deviation yield prediction |
| Comparators | `src/gpace/benchmarks.py`, training runners, `extensions/s5/` | Direct CatBoost, EC-GBLUP, reaction norm, CERIS-JGRA, and an independent GEFormer-style implementation |
| Additional experiments | `workflows/windows.sh`, `workflows/environment_holdouts.sh`, `experiments/g2f_stage_ablation/`, `experiments/g2f_fixed_calendar/` | Input-window comparisons, grouped holdouts, stage removal and fixed-calendar controls |
| Statistical analyses | `scripts/analyze_*.py`, `analyses/g2f_six_models/` | Performance, decomposition, candidate selection, environmental applicability, reaction norms, candidate-marker interactions and exposure permutations |

## Installation

Python 3.10 or newer:

```bash
python -m pip install -e .
```

For model training, install the PyTorch build appropriate for the machine, then:

```bash
python -m pip install -e '.[train,dev]'
```

Training launchers use Linux/Bash. PyTorch and CatBoost device support must match the machine. `environment.yml` provides an alternative environment specification.

## Analyze supplied predictions

```bash
python scripts/verify_release.py --check-metrics
python scripts/recompute_g2f_controls.py
python analyses/g2f_six_models/recompute_g2f.py
```

Alternatively, run `bash workflows/analyze_predictions.sh`. These commands use the supplied prediction records and write numeric outputs to `outputs/`.

## Training workflows

Run from the repository root. Replace the input-directory arguments with the local locations described in `docs/DATA.md`.

```bash
# European main model, then comparators and analyses
bash workflows/europe.sh /absolute/path/to/european_data
bash workflows/europe_benchmarks.sh /absolute/path/to/european_data
bash workflows/windows.sh /absolute/path/to/european_data
bash workflows/environment_holdouts.sh /absolute/path/to/european_data
bash workflows/europe_evidence.sh /absolute/path/to/european_data

# G2F main model and paired controls
bash workflows/g2f.sh /absolute/path/to/g2f_data
bash workflows/g2f_controls.sh /absolute/path/to/g2f_data

# Original G2F comparator groups with saved selected parameters
bash workflows/g2f_comparators.sh /absolute/path/to/g2f_data
```

European experiments use the saved Day-100 configuration. G2F G-PACE retains the European TCN–FiLM architecture and retrains its weights on the historical G2F data. Its default comparison uses the supplied five comparator predictions. The comparator refit launcher loads the archived selected parameters, preserves the original two model groups and model-index seed offsets, and performs no new parameter search.

The main G2F comparison and paired robustness analyses retain their respective GPU and CPU yield-model backends. Stage-removal and fixed-calendar training require the full cross-fit cache tree produced by the G2F main training workflow. These caches are generated during fitting and retained in its output directory.

`PYTHON`, `DEVICE`, `THREADS`, `STAGE_DEVICE`, `DATA_DIR` and other supported overrides are documented directly in the relevant launcher. The original model implementations and fixed training configurations are preserved.

## Statistical definitions

- Seeds: 2026, 2027 and 2028.
- Primary environmental sequence length: 100 days; window experiments additionally use 30, 60, 90 and full-season inputs.
- Within-environment PCC: inverse Fisher transform of the equally weighted valid environment Fisher-z correlations.
- Candidate selection: predictions are averaged across seeds by record, candidates are ranked within environment, and selection outcomes are averaged across environments.
- CPU paired controls: 5,000 paired whole-environment bootstrap resamples; random seed 20261016.
- Candidate-marker permutation analysis: predicted exposure is shuffled within environment while genotype–yield correspondence is retained; 1,000 permutations per interaction.
- FiLM modulation: `[1 + gamma(G)] * H + beta(G)`.

## Repository contents

```text
configs/          Fixed model parameters and original comparator settings
src/gpace/         Data loaders, features, split logic, models and metrics
scripts/          Training, statistical analysis and integrity checks
experiments/      Isolated G2F retraining and paired-control implementations
extensions/s5/   Grouped-holdout comparator implementation
analyses/         G2F record-level post-processing
data/             Prediction references, numeric summaries and run metadata
docs/             Input specification and workflow reference
workflows/        Executable training and analysis launchers
```

Original datasets are obtained separately. Generated outputs, local raw data, model weights, caches and logs are excluded by `.gitignore`. Dependency ranges are installation requirements; completed-run metadata records the reported runtime versions. Checks on supplied predictions and protocol tests are described in `docs/VALIDATION.md`.
