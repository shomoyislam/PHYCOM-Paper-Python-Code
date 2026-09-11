"""Save setup evidence even when training never starts (standard library only)."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import zipfile


def create_diagnostics(project, destination, output=None):
    project,destination = Path(project).resolve(),Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    target = destination/'PHYCOM_V3_2_DIAGNOSTICS.zip'
    temp = target.with_suffix('.tmp.zip')
    records = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   python=sys.version,executable=sys.executable,platform=platform.platform())
    for label,cmd in [('gpu',['nvidia-smi']),('compiler',['nvcc','--version']),
                      ('pip',[sys.executable,'-m','pip','--version'])]:
        try:
            result = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            records[label] = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except (OSError,subprocess.TimeoutExpired) as exc: records[label] = str(exc)
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('environment.json',json.dumps(records,indent=2)+'\n')
        for folder,prefix in [(project/'setup_logs','setup_logs')]+(
                [(Path(output)/'logs','training_logs')] if output else []):
            if folder.exists():
                for path in sorted(folder.rglob('*')):
                    if path.is_file(): archive.write(path,prefix+'/'+path.relative_to(folder).as_posix())
        if output:
            for name in ['run_manifest.json','FINAL_AUDIT.json']:
                path = Path(output)/name
                if path.exists(): archive.write(path,name)
    os.replace(temp,target)
    return target
