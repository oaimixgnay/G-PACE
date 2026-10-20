"""Record assembly file hashes; this does not validate scientific results."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[]
for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file() or p.name=='release_manifest.json' or any(x in rel.parts for x in ['.git','__pycache__','outputs']):continue
    files.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'release_manifest.json').write_text(json.dumps({'release':'gpace-2026-10-20-v0.2.2','files':files},indent=2)+'\n')
print('Recorded',len(files),'files')
