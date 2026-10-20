"""Record the actual machine and dependency versions used for a run."""
import argparse,importlib.metadata,json,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/environment.json'));a=p.parse_args()
packages={}
for name in ['numpy','pandas','scipy','scikit-learn','torch','catboost','optuna']:
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
info={'python':sys.version,'platform':platform.platform(),'packages':packages}
try:info['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True,timeout=10).stdout.strip()
except (OSError,subprocess.TimeoutExpired):info['gpu']=None
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(info,indent=2)+'\n');print(a.output)
