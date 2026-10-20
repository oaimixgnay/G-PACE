#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_DIR="${1:?Provide original G2F data directory}"
# Requires the full evaluation/cache tree produced by workflows/g2f.sh.
bash "$repo_dir/experiments/g2f_stage_ablation/run_stage_ablation.sh" "$repo_dir"
bash "$repo_dir/experiments/g2f_fixed_calendar/run_fixed_calendar.sh" "$repo_dir"
