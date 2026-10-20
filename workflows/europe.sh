#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${1:?Provide the original data directory as the first argument}"
out="outputs/paper_evidence_day100"
"$PYTHON" scripts/validate_data.py --data-dir "$DATA_DIR"
"$PYTHON" scripts/run_experiment.py --data-dir "$DATA_DIR" --output-dir "$out/evaluation" --best-config configs/europe_day100.json --sequence-length 100 --splits 1-10 --seeds 2026,2027,2028 --crossfit-folds 3 --bootstrap-repeats 0 --device "$DEVICE" --paper-ablation
"$PYTHON" scripts/analyze_paper_results.py --evaluation-dir "$out/evaluation" --output-dir "$out/report" --bootstrap-repeats 2000 --seed 2026

