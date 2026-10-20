#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${1:?Provide the original data directory as the first argument}"
out="outputs/published_benchmarks_day100"
"$PYTHON" scripts/run_published_benchmarks.py --data-dir "$DATA_DIR" --output-dir "$out/evaluation" --best-config configs/europe_benchmark_config.json --sequence-length 100 --splits 1-10 --seeds 2026,2027,2028 --device "$DEVICE" --gpace-predictions outputs/paper_evidence_day100/evaluation/yield_predictions.csv --gpace-model mean_residual_hybrid_catboost
"$PYTHON" scripts/analyze_published_benchmarks.py --evaluation-dir "$out/evaluation" --output-dir "$out/report" --primary-model mean_residual_hybrid_catboost --bootstrap-repeats 2000 --seed 2026


"$PYTHON" scripts/run_published_benchmarks.py --data-dir "$DATA_DIR" --output-dir outputs/europe_ec_gblup/evaluation --best-config configs/europe_benchmark_config.json --models ec_gblup --sequence-length 100 --splits 1-10 --seeds 2026,2027,2028 --device "$DEVICE"
"$PYTHON" scripts/merge_europe_selection_predictions.py --core outputs/paper_evidence_day100/evaluation/yield_predictions.csv --benchmarks "$out/evaluation/yield_predictions.csv" --ec-gblup outputs/europe_ec_gblup/evaluation/yield_predictions.csv --output outputs/europe_six_model_external_predictions.csv
