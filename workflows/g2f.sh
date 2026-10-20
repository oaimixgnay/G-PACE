#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_DIR="${1:?Provide original G2F data directory}"
export BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:-$repo_dir/data/frozen_comparators/g2f_predictions.csv.gz}"
# Fresh TCN-FiLM G-PACE weights; the five historical comparator predictions stay fixed.
bash "$repo_dir/experiments/g2f_tcn_film/run_g2f_tcn_film.sh" "$repo_dir"
