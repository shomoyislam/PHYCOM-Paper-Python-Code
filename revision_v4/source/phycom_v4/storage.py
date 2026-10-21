import os
import shutil
import zipfile
from pathlib import Path
from .common import read_json,atomic_json,file_hash,group_id
from .protocol import groups,models,spec_id


def safe_extract(archive,target):
    target=Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path=(target/info.filename).resolve()
            if not path.is_relative_to(target) or ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe path or symbolic link in archive')
        z.extractall(target)


def restore(root,input_path):
    root=Path(root)
    if input_path=='auto':
        found=list(Path('/kaggle/input').glob('**/PHYCOM_V4_*RESUME.zip'))
        found+=list(Path('/kaggle/input').glob('**/RUN_MANIFEST.json'))
        # A directory and its own archive are not silently resolved to a guessed version.
        if not found: return None
        if len(found)>1:
            raise ValueError('Several resume inputs found; set RESUME_INPUT to one exact path: '+str(found))
        input_path=str(found[0].parent if found[0].name=='RUN_MANIFEST.json' else found[0])
    if not input_path: return None
    p=Path(input_path)
    if not p.exists(): raise FileNotFoundError(input_path)
    if (root/'RUN_MANIFEST.json').exists():
        print('Existing output is authoritative; input restore skipped.',flush=True)
        return str(p)
    if root.exists() and any(root.iterdir()): raise ValueError('Restore destination must be empty')
    if p.is_file(): safe_extract(p,root)
    else: shutil.copytree(p,root,dirs_exist_ok=True)
    if not (root/'RUN_MANIFEST.json').exists():
        raise ValueError('Resume input must contain RUN_MANIFEST.json at its root')
    return str(p)


def verify_selection(root,c,run_id,freeze=False):
    root=Path(root);records={};covariances={}
    for g in groups(c):
        gd=root/'jobs'/group_id(g)
        done=read_json(gd/'TRAIN_COMPLETE.json')
        if done['run_id']!=run_id: raise RuntimeError('Training-group identity mismatch')
        for s in models(c,g):
            folder=gd/spec_id(s);d=read_json(folder/'training.json')
            if d['run_id']!=run_id or d['spec']!=s or d['group']!=g or not d['complete']:
                raise RuntimeError('Training record mismatch')
            if file_hash(folder/'best.pt')!=d['best_sha256']:
                raise RuntimeError('Selected checkpoint changed')
            records[folder.relative_to(root).as_posix()]=dict(best_sha256=d['best_sha256'],
                best_epoch=d['best_epoch'],best_val=d['best_val'],
                validation_residual_scale=d['validation_residual_scale'])
        for p in (gd/'covariances').glob('*.npz'):
            covariances[p.relative_to(root).as_posix()]=file_hash(p)
    content=dict(run_id=run_id,selected_models=records,covariances=covariances,
        policy='All checkpoints and validation-only choices fixed before any confirmation evaluation')
    target=root/'FROZEN_SELECTION.json'
    if target.exists() and read_json(target)!=content:
        raise RuntimeError('Frozen selection changed; refuse mixed or retuned confirmation results')
    if freeze and not target.exists(): atomic_json(target,content)
    return content


def bundle(root):
    root=Path(root)
    mode=read_json(root/'config.json')['mode'].upper()
    destinations=[]
    for kind in ('RESUME','REVIEW'):
        out=root.parent/f'PHYCOM_V4_{mode}_{kind}.zip';tmp=out.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.name.endswith('.tmp') or '__pycache__' in p.parts: continue
                rel=p.relative_to(root)
                if kind=='REVIEW' and (rel.parts[0]=='jobs' or rel.parts[0]=='logs'): continue
                z.write(p,rel.as_posix())
        os.replace(tmp,out);destinations.append(str(out))
    return destinations
