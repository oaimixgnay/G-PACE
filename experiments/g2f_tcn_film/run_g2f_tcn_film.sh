#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${1:-$(dirname -- "$package_dir")}"
project_dir="$(cd -- "$project_dir" && pwd)"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$project_dir/../G2F_2024_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$project_dir/outputs/g2f_2024_tcn_film_gpace_only_day100}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-2026,2027,2028}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-1000}"
TEMPORAL_DIAGNOSTIC="${TEMPORAL_DIAGNOSTIC:-0}"
POSTPROCESS="${POSTPROCESS:-1}"
SMOKE_TEST="${SMOKE_TEST:-1}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:-$project_dir/outputs/g2f_2024_comparator_extension_day100/report/g2f_combined_predictions.csv}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

if ! command -v flock >/dev/null 2>&1; then
    printf '%s\n' '[ERROR] flock is required to prevent duplicate simultaneous training jobs.' >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
exec 9>"${OUTPUT_DIR}.lock"
if ! flock -n 9; then
    printf '%s\n' '[ERROR] A job is already using this output directory.' >&2
    exit 1
fi

args=(--data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/evaluation"
      --config-json "$package_dir/config/g2f_tcn_film_frozen.json"
      --device "$DEVICE" --seeds "$SEEDS" --crossfit-folds 3
      --bootstrap-repeats "$BOOTSTRAP_REPEATS" --resume)
if [[ "$TEMPORAL_DIAGNOSTIC" == 1 ]]; then args+=(--temporal-diagnostic); fi

"$PYTHON" "$package_dir/scripts/run_g2f_tcn_film.py" "${args[@]}" --preflight-only
if [[ "$SMOKE_TEST" == 1 ]]; then
    "$PYTHON" "$package_dir/scripts/smoke_test_training.py"
fi
printf '%s\n' '[RUN] Frozen European TCN–FiLM architecture; fresh G2F training weights.'
"$PYTHON" "$package_dir/scripts/run_g2f_tcn_film.py" "${args[@]}"

postprocess_status=0
if [[ "$POSTPROCESS" == 1 && -f "$BASELINE_PREDICTIONS" ]]; then
    if "$PYTHON" "$package_dir/scripts/merge_tcn_film_results.py" \
        --new-evaluation-dir "$OUTPUT_DIR/evaluation" \
        --baseline-predictions "$BASELINE_PREDICTIONS" \
        --output-dir "$OUTPUT_DIR/report" --bootstrap-repeats 2000 --seed 20260917; then
        if "$PYTHON" "$package_dir/scripts/analyze_g2f_decomposition_and_sparse_selection.py" \
            --predictions "$OUTPUT_DIR/report/g2f_combined_predictions.csv" \
            --output-dir "$OUTPUT_DIR/decision" \
            --random-repeats 2000 --bootstrap-repeats 5000 --random-state 20260915; then
            postprocess_status=0
        else postprocess_status=$?; fi
    else postprocess_status=$?; fi
elif [[ "$POSTPROCESS" == 1 ]]; then
    printf '%s\n' "[NOTICE] Primary training finished. Original six-model prediction report not found: $BASELINE_PREDICTIONS"
    printf '%s\n' '[NOTICE] After locating the baseline file, run postprocess_g2f_tcn_film.sh; no neural retraining is needed.'
fi

"$PYTHON" "$package_dir/scripts/export_tcn_film_results.py" --output-dir "$OUTPUT_DIR"
printf '%s\n' "[DONE] Results: $OUTPUT_DIR" "[SEND] $OUTPUT_DIR/g2f_tcn_film_results.zip"
if [[ "$postprocess_status" != 0 ]]; then
    printf '%s\n' '[NOTICE] Neural training finished and was exported; a later comparison/decision step failed. Inspect its traceback and use the postprocess-only command after correcting it.'
    exit "$postprocess_status"
fi
