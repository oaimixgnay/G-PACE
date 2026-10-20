#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$(dirname -- "$package_dir")}"
project_dir="$(cd -- "$project_dir" && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/g2f_2024_tcn_film_gpace_only_day100}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:-$project_dir/outputs/g2f_2024_comparator_extension_day100/report/g2f_combined_predictions.csv}"
"$PYTHON" "$package_dir/scripts/merge_tcn_film_results.py" \
    --new-evaluation-dir "$OUTPUT_DIR/evaluation" --baseline-predictions "$BASELINE_PREDICTIONS" \
    --output-dir "$OUTPUT_DIR/report" --bootstrap-repeats 2000 --seed 20260917
"$PYTHON" "$package_dir/scripts/analyze_g2f_decomposition_and_sparse_selection.py" \
    --predictions "$OUTPUT_DIR/report/g2f_combined_predictions.csv" --output-dir "$OUTPUT_DIR/decision" \
    --random-repeats 2000 --bootstrap-repeats 5000 --random-state 20260915
"$PYTHON" "$package_dir/scripts/export_tcn_film_results.py" --output-dir "$OUTPUT_DIR"
