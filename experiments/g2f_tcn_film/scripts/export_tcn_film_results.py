#!/usr/bin/env python3
"""Export source tables and manifests, excluding large learned-weight caches."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    output = args.output_dir.resolve()
    manifest = output / 'evaluation/g2f_run_manifest.json'
    if not manifest.is_file() or json.loads(manifest.read_text()).get('status') != 'complete':
        raise ValueError('Export requires a completed TCN–FiLM training run.')
    archive = output / 'g2f_tcn_film_results.zip'
    files = []
    for dirname in ('evaluation', 'report', 'decision'):
        root = output / dirname
        if not root.exists():
            continue
        for file in sorted(root.iterdir()):
            if file.is_file() and file.suffix in {'.csv', '.json', '.md'}:
                files.append(file)
    # Three external stage-prediction and training-summary files are useful for
    # auditing the new representation, while torch/joblib weights stay on server.
    for file in sorted((output / 'evaluation/runs').glob('*/seed_*/*')):
        if file.name in {'stage_predictions.csv', 'stage_metrics.csv'}:
            files.append(file)
    for file in sorted((output / 'evaluation/runs').glob('*/seed_*/models/stage_training.json')):
        files.append(file)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in files:
            z.write(file, str(file.relative_to(output)))
        z.writestr('README.txt', 'TCN–FiLM architecture with fresh G2F historical training weights.\nWeights and resumable fold caches remain on the server.\nEvaluation status is recorded in evaluation/g2f_run_manifest.json.\n')
    print(f'[EXPORT] {len(files)} source files: {archive}', flush=True)


if __name__ == '__main__':
    main()
