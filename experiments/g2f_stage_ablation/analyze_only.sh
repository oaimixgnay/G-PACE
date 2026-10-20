#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$(dirname -- "$package_dir")}"
output_dir="${OUTPUT_DIR:-$project_dir/outputs/g2f_stage_ablation_day100_cpu}"
"${PYTHON:-python}" "$package_dir/scripts/paired_statistics.py" \
  --predictions "$output_dir/paired_predictions.csv" \
  --output-dir "$output_dir/report" \
  --seeds "${SEEDS:-2026,2027,2028}" \
  --bootstrap-repeats "${BOOTSTRAP_REPEATS:-5000}" \
  --random-state "${RANDOM_STATE:-20261016}"
