#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON:-python}"
"$python_bin" scripts/verify_release.py --check-metrics
"$python_bin" scripts/recompute_g2f_controls.py
"$python_bin" analyses/g2f_six_models/recompute_g2f.py
