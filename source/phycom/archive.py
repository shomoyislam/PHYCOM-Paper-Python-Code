"""Read-only parent import, safe restore, and checkpoint lineage."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from .common import atomic_json, read_json, file_hash, jobs, job_id, model_names

PARENT_RUN_ID = '3bfd208073793897927a418fce0c95d661a66847da86a81f2099d8385c2ae2cc'
PARENT_CODE_HASH = '16b3ae2cffa69c0a51627c4fca2521d317721e133c39098e24e9fb412436ba07'


class SavedRun:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(); self.z = None
        if self.path.is_dir():
            if (self.path/'run_manifest.json').is_file(): self.prefix = self.path
            else:
                paths = list(self.path.rglob('run_manifest.json'))
                if len(paths)!=1: raise ValueError('Expected one run_manifest.json in saved run')
                self.prefix = paths[0].parent
        else:
            self.z = zipfile.ZipFile(self.path)
            manifests = [n for n in self.z.namelist() if n.endswith('/run_manifest.json') or n=='run_manifest.json']
            if len(manifests)!=1: self.z.close(); raise ValueError('Expected one run_manifest.json in ZIP')
            self.prefix = manifests[0][:-len('run_manifest.json')]
        self.manifest = json.loads(self.read('run_manifest.json'))

    def read(self, relative):
        return self.z.read(self.prefix+relative) if self.z else (self.prefix/relative).read_bytes()

    def exists(self, relative):
        return self.prefix+relative in self.z.namelist() if self.z else (self.prefix/relative).is_file()

    def close(self):
        if self.z: self.z.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def safe_target(root, relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or '\\' in relative:
        raise ValueError('Unsafe archive path')
    target = (Path(root)/relative).resolve()
    if not target.is_relative_to(Path(root).resolve()): raise ValueError('Unsafe archive path')
    return target


def restore_run(path, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Restore destination is not empty; resume existing output or use a new directory.')
    with SavedRun(path) as saved:
        if saved.manifest['config'].get('schema')!='phycom-v3.2':
            raise ValueError('RESUME_ARCHIVE must be V3.2; use PARENT_V31 for the older results.')
        if saved.z:
            members = [(i, i.filename[len(saved.prefix):]) for i in saved.z.infolist()
                       if i.filename.startswith(saved.prefix) and not i.is_dir()]
            for item, rel in members:
                target = safe_target(output, rel)
                if (item.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('Archive symlinks unsupported')
                target.parent.mkdir(parents=True, exist_ok=True)
                with saved.z.open(item) as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)
        else: shutil.copytree(saved.prefix, output, dirs_exist_ok=True)
    # Configuration, code and runtime are checked again by the orchestrator.


def find_parent(setting, roots):
    if setting.lower() in ('', 'none'): return None
    if setting.lower()!='auto':
        if not Path(setting).exists(): raise FileNotFoundError(f'PARENT_V31 does not exist: {setting}')
        return Path(setting)
    candidates = []
    for root in roots:
        root = Path(root)
        if not root.exists(): continue
        candidates += list(root.rglob('*.zip'))
        candidates += [p.parent for p in root.rglob('run_manifest.json')]
    for path in sorted(set(candidates), key=lambda p: (not p.is_file(),str(p))):
        try:
            with SavedRun(path) as saved:
                if saved.manifest.get('run_id')==PARENT_RUN_ID: return path
        except (ValueError, OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError): continue
    return None


def import_parent(root, cfg, rid, identity, parent):
    """All import work finishes before any worker starts; never execute parent code."""
    from .training import load_checkpoint, save_torch, publish_checkpoints, import_legacy_state
    import torch
    report_path = root/'parent_import.json'
    if report_path.exists(): return read_json(report_path)
    report = dict(parent=None, imported=[], status='fresh_run', reason='No matching V3.1 archive attached.')
    if parent is None or cfg['mode']=='smoke':
        atomic_json(report_path, report); return report
    with SavedRun(parent) as saved:
        man = saved.manifest
        if man.get('run_id')!=PARENT_RUN_ID or man.get('code_sha256')!=PARENT_CODE_HASH:
            raise ValueError('This is not the audited V3.1 parent run. Leave PARENT_V31 empty for a fresh run.')
        report['parent'] = dict(run_id=man['run_id'], code_sha256=man['code_sha256'], filename=Path(parent).name)
        atomic_json(root/'parent_run_manifest.json', man)
        if man['runtime_identity']!=identity:
            report.update(status='fresh_runtime_mismatch', reason='Parent numerical runtime differs; all fits start from scratch.')
            atomic_json(report_path, report); print(report['reason'], flush=True); return report
        old = man['config']
        for field in ['alpha','M','main_T','train_samples','val_samples','train_snr','batch_size',
                      'lr','patience','lr_patience','lr_factor','gradient_clip','seq_train','seq_val']:
            if cfg[field]!=old[field]: raise ValueError(f'Parent training setting mismatch: {field}')
        for job in jobs(cfg):
            for name in model_names(cfg,job):
                folder = root/'jobs'/job_id(job)/name
                rel = f'jobs/{job_id(job)}/{name}/'
                if not (saved.exists(rel+'last.pt') and saved.exists(rel+'best.pt')): continue
                a, b = saved.read(rel+'last.pt'), saved.read(rel+'best.pt')
                origin = dict(kind='continued_v3.1', parent_run_id=man['run_id'],
                    parent_last_sha256=hashlib.sha256(a).hexdigest(), parent_best_sha256=hashlib.sha256(b).hexdigest())
                if not (folder/'last.pt').exists():
                    last = torch.load(io.BytesIO(a),map_location='cpu',weights_only=True)
                    best = torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
                    if last['run_id']!=man['run_id'] or best['run_id']!=man['run_id']:
                        raise ValueError('Checkpoint parent identity mismatch')
                    ck = import_legacy_state(last,best,rid,job,name,origin)
                    save_torch(folder/'last.pt',ck); publish_checkpoints(folder,ck)
                else:
                    ck = load_checkpoint(folder/'last.pt')
                    if ck['run_id']!=rid or ck['origin']!=origin:
                        raise ValueError('Interrupted import conflicts with existing checkpoint')
                report['imported'].append(dict(job=job,model=name,**origin))
        report.update(status='imported',reason='Original weights, optimizer, scheduler, history and RNG retained.')
    atomic_json(report_path, report)
    print(f'Imported {len(report["imported"])} trained V3.1 model states; remaining fits start fresh.',flush=True)
    return report
