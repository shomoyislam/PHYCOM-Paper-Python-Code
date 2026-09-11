"""Kaggle Run All: verified setup, train, freeze, confirm, report, archive."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TRITON_CACHE_AUTOTUNING','1')


def environment():
    import importlib.metadata as md
    import platform
    import torch,numpy,scipy,matplotlib
    packages={}
    for name in ['mamba-ssm','causal-conv1d','triton','einops']:
        try: packages[name]=md.version(name)
        except md.PackageNotFoundError: packages[name]=None
    return dict(python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
        numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),packages=packages,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        deterministic_algorithms='warn_only; official CUDA kernels may be nondeterministic',
        parallelism='one independent job per GPU; timing serialized; no DataParallel')


def bundle(root,complete=False):
    target=root.parent/(root.name+('_RESULTS.zip' if complete else '_PARTIAL.zip'))
    temp=target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.tmp' and '__pycache__' not in p.parts:
                z.write(p,root.name+'/'+p.relative_to(root).as_posix())
    os.replace(temp,target); return target


def stop_workers(running):
    for proc,handle,job,path,start in running.values():
        if proc.poll() is None: proc.terminate()
    for proc,handle,job,path,start in running.values():
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        handle.close()


def execute_jobs(root,cfg,devices,phase):
    from phycom.common import jobs,job_id,read_json,SessionPaused
    from phycom.worker import checked_marker
    rid=read_json(root/'run_manifest.json')['run_id']
    marker='TRAINED.json' if phase=='train' else 'COMPLETE.json'
    todo=[job for job in jobs(cfg) if not checked_marker(root/'jobs'/job_id(job)/marker,rid)]
    running={}; completed=0; total=len(todo); paused=False
    logs=root/'logs'; logs.mkdir(exist_ok=True)
    try:
        while todo or running:
            if float(os.environ.get('PHYCOM_STOP_AT','0')) and time.time()>=float(os.environ['PHYCOM_STOP_AT']): paused=True
            for device in devices:
                if paused or device in running or not todo: continue
                job=todo.pop(0); name=job_id(job); path=logs/f'{phase}_{name}.log'
                handle=path.open('a'); cmd=[sys.executable,'-u','-m','phycom.worker','--root',str(root),
                    '--job',json.dumps(job),'--device',device,'--phase',phase]
                proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
                running[device]=(proc,handle,job,path,time.monotonic())
                print(f'{phase}: start {name} on {device}; {len(todo)} queued',flush=True)
            for device,(proc,handle,job,path,start) in list(running.items()):
                rc=proc.poll()
                if rc is None: continue
                handle.close(); del running[device]
                if rc==75: paused=True
                elif rc: raise RuntimeError(f'{phase} / {job_id(job)} failed (exit {rc}):\n'+path.read_text(errors='replace')[-9000:])
                else:
                    completed+=1; print(f'{phase}: completed {job_id(job)} ({completed}/{total})',flush=True)
            if paused and not running: raise SessionPaused('Session budget reached. Resume from the PARTIAL archive.')
            if running:
                for device,(_,_,job,path,start) in running.items():
                    lines=path.read_text(errors='replace').splitlines()
                    print(f'[{device}, {int(time.monotonic()-start)}s] '+(lines[-1][-220:] if lines else 'starting'),flush=True)
                time.sleep(1 if cfg['mode']=='smoke' else 20)
    finally: stop_workers(running)


def freeze_selection(root,cfg,rid):
    from phycom.common import jobs,job_id,model_names,stages,read_json,atomic_json,file_hash
    from phycom.training import load_checkpoint,publish_checkpoints
    target=root/'frozen_selection.json'
    if target.exists():
        previous=read_json(target)
        if previous['run_id']!=rid: raise ValueError('Frozen checkpoint run mismatch')
        if not all(file_hash(root/r['checkpoint'])==r['sha256'] for r in previous['checkpoints']):
            raise ValueError('A frozen checkpoint changed. Keep the original run unchanged.')
        return
    rows=[]
    for job in jobs(cfg):
        for name in model_names(cfg,job):
            folder=root/'jobs'/job_id(job)/name
            ck=load_checkpoint(folder/'last.pt')
            # Re-export from the committed transaction before hashing the selection.
            publish_checkpoints(folder,ck)
            for stage in stages(job):
                path=folder/('budget60.pt' if stage=='budget60' else 'best.pt')
                selected=load_checkpoint(path)
                rows.append(dict(job=job,model=name,stage=stage,checkpoint=path.relative_to(root).as_posix(),
                    sha256=file_hash(path),selected_epoch=selected['epoch'],validation_mse=selected['best_val']))
    atomic_json(target,dict(run_id=rid,frozen_utc=datetime.now(timezone.utc).isoformat(),
        selection='All fits completed; minimum validation MSE. No confirmation test results generated yet.',checkpoints=rows))


def copy_source(base,root):
    dest=root/'source'; dest.mkdir(exist_ok=True)
    for p in base.iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.md','.txt','.cff'):
            shutil.copy2(p,dest/p.name)
    for name in ['phycom','tests']:
        shutil.copytree(base/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if (base/'setup_logs').exists(): shutil.copytree(base/'setup_logs',root/'setup_logs',dirs_exist_ok=True)


def main():
    from phycom.common import base_config,read_json,atomic_json,digest,code_hash,jobs,model_names,SessionPaused,check_deadline
    from phycom.archive import restore_run,find_parent,import_parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='/kaggle/working/PHYCOM_V3_2_PUBLICATION' if Path('/kaggle').exists() else 'PHYCOM_V3_2_PUBLICATION')
    p.add_argument('--parent',default='auto',help='Auto-discover the audited V3.1 results, a path, or none')
    p.add_argument('--resume',default='',help='Own V3.2 partial/results ZIP to restore into an empty output')
    p.add_argument('--session-hours',type=float,default=10.5,help='Graceful pause budget per invocation; 0 disables')
    p.add_argument('--smoke',action='store_true'); p.add_argument('--no-mamba',action='store_true')
    p.add_argument('--single-gpu',action='store_true'); p.add_argument('--skip-setup',action='store_true')
    args=p.parse_args()
    if args.no_mamba and not args.smoke: p.error('--no-mamba is only allowed for a smoke test')
    if args.session_hours<0: p.error('--session-hours must be nonnegative')
    started=time.time()
    os.environ['PHYCOM_STOP_AT']=str(started+args.session_hours*3600 if args.session_hours else 0)
    base=Path(__file__).resolve().parent
    root=Path(args.output).expanduser().resolve(); os.chdir(base)
    if args.resume:
        if (root/'run_manifest.json').exists(): print('Using existing V3.2 output; restore is unnecessary.',flush=True)
        else: restore_run(args.resume,root)
    cfg=base_config(args.smoke)
    if args.no_mamba:
        cfg['include_mamba']=False
        for field in ['models','seq_models','stress_models']:
            cfg[field]=[m for m in cfg[field] if 'mamba' not in m]
    if not args.skip_setup:
        cmd=[sys.executable,'-u',str(base/'bootstrap.py')]
        if args.smoke: cmd+=['--smoke']
        if args.no_mamba: cmd+=['--no-mamba']
        subprocess.run(cmd,check=True)
    import torch
    if not args.smoke and not torch.cuda.is_available(): raise RuntimeError('Choose GPU T4 ×2 in Kaggle settings.')
    if cfg['include_mamba']:
        from bootstrap import MAMBA_PROBE
        if args.skip_setup: subprocess.run([sys.executable,'-c',MAMBA_PROBE],check=True)
        subprocess.run([sys.executable,'-u','-m','phycom.runtime_checks'],check=True)
    env=environment(); identity={k:env[k] for k in ['torch','numpy','scipy','cuda','cudnn','packages']}
    rid=digest(dict(code=code_hash(),config=cfg)); root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'run_manifest.json'
    if manifest_path.exists():
        man=read_json(manifest_path)
        if man['run_id']!=rid or man['runtime_identity']!=identity:
            raise RuntimeError('Existing output has different code/config/runtime. Use a NEW output directory; do not mix runs.')
        if man['status']=='complete' and (root/'FINAL_AUDIT.json').exists():
            print('Matching run already complete.',flush=True)
            archive=root.parent/(root.name+'_RESULTS.zip')
            if not archive.exists(): archive=bundle(root,True)
            print(f'Archive saved: {archive}',flush=True); return 0
    else:
        man=dict(run_id=rid,code_sha256=code_hash(),config=cfg,environment=env,runtime_identity=identity,
            created_utc=datetime.now(timezone.utc).isoformat(),status='in_progress')
    man['status']='in_progress'; atomic_json(manifest_path,man); atomic_json(root/'config.json',cfg)
    complete=False
    try:
        copy_source(base,root)
        (root/'requirements_frozen.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],
            capture_output=True,text=True,check=True).stdout)
        search_roots=[Path('/kaggle/input'),Path('/kaggle/working/PHYCOM_V3_1_PUBLICATION')]
        parent=find_parent(args.parent,search_roots) if not args.smoke and not (root/'parent_import.json').exists() else None
        import_parent(root,cfg,rid,identity,parent)
        from phycom.simulation import validate_channel
        if not (root/'channel_validation.json').exists():
            atomic_json(root/'channel_validation.json',validate_channel(count=5000 if args.smoke else 20000))
        if not read_json(root/'channel_validation.json')['passed']: raise RuntimeError('Jakes moment validation failed')
        devices=[f'cuda:{i}' for i in range(min(torch.cuda.device_count(),1 if args.single_gpu else 2))] if torch.cuda.is_available() else ['cpu']
        print(f'{cfg["mode"]}: {sum(len(model_names(cfg,j)) for j in jobs(cfg))} fits; up to {cfg["epochs"]} epochs; {devices}',flush=True)
        print('Phase 1: training and validation only.',flush=True)
        execute_jobs(root,cfg,devices,'train'); freeze_selection(root,cfg,rid)
        print('All models frozen. Phase 2: new confirmation channels and paired robustness tests.',flush=True)
        execute_jobs(root,cfg,devices,'evaluate'); check_deadline()
        print('Phase 3: computed tables, figures, serial timing and raw-data audit.',flush=True)
        from phycom.reporting import export
        export(root,cfg,torch.device(devices[0])); man['status']='complete'
        man['completed_utc']=datetime.now(timezone.utc).isoformat(); atomic_json(manifest_path,man); complete=True
    except SessionPaused as exc:
        man['status']='paused'; atomic_json(manifest_path,man)
        print(str(exc),flush=True); return 75
    except BaseException:
        man['status']='interrupted' if isinstance(sys.exc_info()[1],KeyboardInterrupt) else 'failed'
        atomic_json(manifest_path,man); (root/'failure_traceback.txt').write_text(traceback.format_exc()); raise
    finally:
        archive=bundle(root,complete); print(f'Archive saved: {archive}',flush=True)
    print('Complete. Read Paper_Data/RESULTS_SUMMARY.md and FINAL_AUDIT.json before finalizing claims.',flush=True)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        from diagnostics import create_diagnostics
        print('Diagnostics:',create_diagnostics(Path(__file__).parent,Path.cwd().parent),flush=True)
        raise SystemExit(1)
