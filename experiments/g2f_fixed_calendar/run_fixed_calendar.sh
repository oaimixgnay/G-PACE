#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$(dirname -- "$package_dir")}"
project_dir="$(cd -- "$project_dir" && pwd)"
python_bin="${PYTHON:-python}"
stage_device="${STAGE_DEVICE:-cuda}"
threads="${THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$stage_device" == cpu ]]; then export CUDA_VISIBLE_DEVICES=""; fi
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
args=(--data-dir "${DATA_DIR:-$project_dir/../G2F_2024_data}"
      --existing-evaluation-dir "${EXISTING_EVALUATION_DIR:-$project_dir/outputs/g2f_2024_tcn_film_gpace_only_day100/evaluation}"
      --paired-ablation-dir "${PAIRED_ABLATION_DIR:-$project_dir/outputs/g2f_stage_ablation_day100_cpu}"
      --output-dir "${OUTPUT_DIR:-$project_dir/outputs/g2f_fixed_calendar_day100_$stage_device}"
      --stage-device "$stage_device" --threads "$threads"
      --memory-budget-gb "${MEMORY_BUDGET_GB:-20}"
      --bootstrap-repeats "${BOOTSTRAP_REPEATS:-5000}" --resume)
"$python_bin" "$package_dir/scripts/run_fixed_calendar.py" "${args[@]}" --preflight-only
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
"$python_bin" "$package_dir/scripts/run_fixed_calendar.py" "${args[@]}"
