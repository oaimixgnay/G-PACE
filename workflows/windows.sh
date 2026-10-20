#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${1:?Provide the original data directory as the first argument}"
for window in 30 60 90 100 0; do
  label="$window"; [[ "$window" == 0 ]] && label="full"
  out="outputs/windows/$label"
  "$PYTHON" scripts/run_experiment.py --data-dir "$DATA_DIR" --output-dir "$out" --best-config configs/europe_day100.json --sequence-length "$window" --splits 1-10 --seeds 2026,2027,2028 --crossfit-folds 3 --bootstrap-repeats 0 --model-suite primary --device "$DEVICE"
done
"$PYTHON" scripts/analyze_window_results.py --window-evaluation 30=outputs/windows/30 --window-evaluation 60=outputs/windows/60 --window-evaluation 90=outputs/windows/90 --window-evaluation 100=outputs/windows/100 --window-evaluation full=outputs/windows/full --output-dir outputs/windows/report

