#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$(dirname -- "$package_dir")}"
project_dir="$(cd -- "$project_dir" && pwd)"
python_bin="${PYTHON:-python}"
data_dir="${DATA_DIR:-$project_dir/../G2F_2024_data}"
existing_dir="${EXISTING_EVALUATION_DIR:-$project_dir/outputs/g2f_2024_tcn_film_gpace_only_day100/evaluation}"
output_dir="${OUTPUT_DIR:-$project_dir/outputs/g2f_stage_ablation_day100_cpu}"
threads="${THREADS:-8}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"

args=(--data-dir "$data_dir" --existing-evaluation-dir "$existing_dir"
      --output-dir "$output_dir" --threads "$threads"
      --seeds "${SEEDS:-2026,2027,2028}"
      --bootstrap-repeats "${BOOTSTRAP_REPEATS:-5000}"
      --random-state "${RANDOM_STATE:-20261016}" --resume)

"$python_bin" "$package_dir/scripts/run_stage_ablation.py" "${args[@]}" --preflight-only
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
"$python_bin" "$package_dir/scripts/run_stage_ablation.py" "${args[@]}"
