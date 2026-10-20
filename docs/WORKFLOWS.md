# Workflow reference

Run commands from the repository root. All `DATA` arguments are paths to the required original input files.

| Task | Entry | Prerequisite |
|---|---|---|
| European main experiment | `bash workflows/europe.sh DATA` | European input files |
| European comparators | `bash workflows/europe_benchmarks.sh DATA` | Main experiment predictions |
| Input-window comparison | `bash workflows/windows.sh DATA` | European inputs |
| Region/stress complete holdouts | `bash workflows/environment_holdouts.sh DATA` | European inputs; launcher generates core results before added comparators |
| Stage reconstruction, components, reactions and interactions | `bash workflows/europe_evidence.sh DATA` | Main experiment and European comparator outputs |
| G2F main TCN–FiLM | `bash workflows/g2f.sh DATA` | G2F inputs; supplied frozen comparator predictions |
| G2F paired stage/calendar controls | `bash workflows/g2f_controls.sh DATA` | Full output/cache tree from the main G2F run |
| G2F comparator refits | `bash workflows/g2f_comparators.sh DATA` | G2F inputs; saved selected configurations |
| Supplied-prediction analyses | `bash workflows/analyze_predictions.sh` | Included prediction references |

## Saved parameters

| Experiment | Parameter source |
|---|---|
| European main TCN–FiLM and yield head | `configs/europe_day100.json` |
| European reaction norm/CERIS-JGRA/GEFormer-style | `configs/europe_benchmark_config.json` |
| European EC-GBLUP | `DEFAULT_BENCHMARK_CONFIGS` in `src/gpace/benchmarks.py` |
| Grouped-holdout additional comparators | `extensions/s5/config/frozen_benchmark_config.json` |
| G2F main TCN–FiLM and yield head | `experiments/g2f_tcn_film/config/g2f_tcn_film_frozen.json` |
| G2F reaction norm/CERIS-JGRA | `configs/g2f_base_benchmark_config.json` |
| G2F EC-GBLUP/GEFormer-style | `configs/g2f_extension_benchmark_config.json` |
| G2F direct CatBoost | `configs/g2f_runner_defaults.json` with `scripts/run_g2f_external.py` |

G2F comparator refits retain the original base group (reaction norm, CERIS-JGRA) and extension group (EC-GBLUP, GEFormer-style). Each group uses seeds 2026/2027/2028, with the existing runner's model-index offsets 0 and 10,000. `--config-json` loads the archived model parameters; omitting it retains the original historical-year selection procedure.

## Numeric analysis programs

- Performance and paired comparisons: `scripts/analyze_paper_results.py`, `scripts/analyze_published_benchmarks.py`, `scripts/analyze_environment_holdouts.py`.
- Input windows and uncertainty: `scripts/analyze_window_results.py`, `scripts/analyze_window_uncertainty.py`, `scripts/analyze_conformal_risk.py`.
- Applicability and selection: `scripts/analyze_environment_applicability.py`, `scripts/analyze_breeding_selection.py`.
- G2F decomposition and candidate selection: `analyses/g2f_six_models/recompute_g2f.py`, `scripts/analyze_g2f_decomposition_and_sparse_selection.py`.
- G2F paired CPU comparisons: `scripts/recompute_g2f_controls.py`.
- Component and reaction analyses: `scripts/analyze_component_traits.py`, `scripts/analyze_reaction_norms.py`.
- Candidate-marker interactions and exposure permutations: `scripts/analyze_snp_stage_interactions.py`, `scripts/analyze_alignment_negative_control.py`.

Analysis programs write CSV/JSON and numeric reports. Source and run metadata accompany the corresponding experiment directories.
