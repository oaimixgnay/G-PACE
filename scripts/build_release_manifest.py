"""Record hashes of tracked and non-ignored project files."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(
    ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
    cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8',
)
files = []
for name in sorted(set(result.stdout.split('\0'))):
    if not name or name == 'release_manifest.json':
        continue
    path = ROOT / name
    if path.is_file():
        files.append({
            'path': name,
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest = {'release': 'gpace-2026-10-20-v0.2.2', 'files': files}
(ROOT / 'release_manifest.json').write_text(
    json.dumps(manifest, indent=2) + '\n', encoding='utf-8',
)
print('Recorded', len(files), 'files')
