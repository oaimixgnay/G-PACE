#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${1:?Provide the original data directory as the first argument}"
for scheme in region stress; do
  out="outputs/environment_holdouts_v2_day100/$scheme"
  "$PYTHON" scripts/run_experiment.py --data-dir "$DATA_DIR" --output-dir "$out/evaluation" --best-config configs/europe_day100.json --sequence-length 100 --seeds 2026,2027,2028 --crossfit-folds 3 --bootstrap-repeats 0 --device "$DEVICE" --skip-supplied-cv --skip-final-external --environment-holdout "$scheme" --model-suite core
  "$PYTHON" scripts/analyze_environment_holdouts.py --evaluation-dir "$out/evaluation" --output-dir "$out/report" --holdout-scheme "$scheme" --primary-model mean_residual_hybrid_catboost
done
"$PYTHON" extensions/s5/scripts/run_s5_holdout_benchmarks.py --project-root . --data-dir "$DATA_DIR" --best-config configs/europe_benchmark_config.json --core-root outputs/environment_holdouts_v2_day100 --output-dir outputs/environment_holdouts_benchmarks_day100 --device "$DEVICE" --require-core-predictions
"$PYTHON" extensions/s5/scripts/run_ec_gblup_holdouts.py --data-dir "$DATA_DIR" --output-dir outputs/ec_gblup_holdouts

