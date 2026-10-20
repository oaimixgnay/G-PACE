#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${1:?Provide the original data directory as the first argument}"
eval="outputs/paper_evidence_day100/evaluation"
"$PYTHON" scripts/run_complete_genotype_crossfit.py --data-dir "$DATA_DIR" --best-config configs/europe_day100.json --output-dir outputs/additional_evidence_day100/complete_genotype_5fold_stage --sequence-length 100 --folds 5 --seeds 2026,2027,2028 --device "$DEVICE"
exposure="outputs/additional_evidence_day100/complete_genotype_5fold_stage/exposure_predictions.csv"
"$PYTHON" scripts/analyze_snp_stage_interactions.py --data-dir "$DATA_DIR" --exposure-predictions "$exposure" --output-dir outputs/additional_evidence_day100/snp_stage_gxe_246 --sequence-length 100 --stage-scenarios genotype_5fold_oof --pc-components 3 --phenology-permutations 1000 --random-state 20260915
"$PYTHON" scripts/analyze_alignment_negative_control.py --data-dir "$DATA_DIR" --exposure-predictions "$exposure" --aligned-interactions outputs/additional_evidence_day100/snp_stage_gxe_246/snp_stage_interactions.csv --output-dir outputs/additional_evidence_day100/calendar_control_246 --sequence-length 100 --stage-scenarios genotype_5fold_oof
"$PYTHON" scripts/analyze_reaction_norms.py --data-dir "$DATA_DIR" --evaluation-dir "$eval" --output-dir outputs/biological_evidence_day100/reaction_norms --sequence-length 100
"$PYTHON" scripts/run_component_evidence.py --data-dir "$DATA_DIR" --best-config configs/europe_day100.json --output-dir outputs/biological_evidence_day100/components --sequence-length 100 --seeds 2026,2027,2028 --crossfit-folds 3 --bootstrap-repeats 1000 --device "$DEVICE"
"$PYTHON" scripts/analyze_component_traits.py --component-dir outputs/biological_evidence_day100/components --output-dir outputs/biological_evidence_day100/component_report --bootstrap-repeats 2000
"$PYTHON" scripts/analyze_environment_applicability.py --data-dir "$DATA_DIR" --predictions "$eval/yield_predictions.csv" --output-dir outputs/decision_evidence_day100/applicability --sequence-length 100 --pca-components 10 --bootstrap-repeats 5000 --seed 2026
"$PYTHON" scripts/analyze_conformal_risk.py --predictions "$eval/yield_predictions.csv" --output-dir outputs/uncertainty_evidence_day100
"$PYTHON" scripts/analyze_breeding_selection.py --predictions outputs/europe_six_model_external_predictions.csv --output-dir outputs/decision_evidence_day100/selection --models mean_residual_hybrid_catboost,e2e_catboost,ec_gblup,reaction_norm_kernel,ceris_jgra,geformer_style --fractions 0.05,0.10,0.20 --bootstrap-repeats 5000

