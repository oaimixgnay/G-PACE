#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON:-python}"
data_dir="${1:?Provide original G2F data directory}"
# Preserve the two original model groups and their model-index seed offsets.
"$python_bin" scripts/run_g2f_benchmarks.py --data-dir "$data_dir" --output-dir outputs/g2f_comparators/base --sequence-length 100 --models reaction_norm_rff,ceris_jgra --config-json configs/g2f_base_benchmark_config.json --seeds 2026,2027,2028 --device "${DEVICE:-cuda}"
"$python_bin" scripts/run_g2f_benchmarks.py --data-dir "$data_dir" --output-dir outputs/g2f_comparators/extension --sequence-length 100 --models ec_gblup,geformer_style --config-json configs/g2f_extension_benchmark_config.json --seeds 2026,2027,2028 --device "${DEVICE:-cuda}"
# This original runner also exports the direct CatBoost comparator.
"$python_bin" scripts/run_g2f_external.py --data-dir "$data_dir" --output-dir outputs/g2f_direct_reference --sequence-length 100 --seeds 2026,2027,2028 --crossfit-folds 3 --bootstrap-repeats 1000 --stage-model ridge --yield-head catboost --config-json configs/g2f_runner_defaults.json --device "${DEVICE:-cuda}"
