#!/usr/bin/env python3
"""Self-contained Kaggle two-GPU experiment scheduler. No DDP batch-size change."""
import os
# Set before importing torch/numpy or creating a CUDA context.
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from phycom_v4.common import digest,source_hash,atomic_json,read_json,group_id,PauseRun
from phycom_v4.protocol import config,groups,models,plan,spec_id


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='/kaggle/working/PHYCOM_V4_RUN')
    p.add_argument('--smoke',action='store_true',help='Tiny non-publication run, all code paths')
    p.add_argument('--hours',type=float,default=10.,help='Work budget; packaging follows before session shutdown')
    p.add_argument('--devices',default='auto',help='auto, 0,1, a single CUDA index, or cpu (smoke only)')
    p.add_argument('--cpu-workers',type=int,default=2)
    p.add_argument('--resume-input',default='auto',help='Previous RESUME.zip, extracted output directory, auto or empty')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--worker',choices=['preflight','train','evaluate','timing'],help=argparse.SUPPRESS)
    p.add_argument('--group-index',type=int,help=argparse.SUPPRESS)
    return p.parse_args()


def runtime():
    import torch,numpy,scipy,pandas,matplotlib
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=numpy.__version__,
        scipy=scipy.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu_names=sorted(set(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))))


def preflight(c,device):
    import torch
    from phycom_v4.models import Estimator
    from phycom_v4.protocol import spec,resolved
    from phycom_v4.training import setup
    setup(c)
    for n in c['arrays']:
        for cond in ('none','gate','film_identity','concat_identity'):
            m=Estimator(resolved(spec(p=128,h=128,cond=cond),n),n,777).to(device)
            x=torch.randn(8,c['T'],2,n,n,device=device);snr=torch.zeros(8,1,device=device)
            y=m(x,snr);y.square().mean().backward()
            if not torch.isfinite(y).all(): raise FloatingPointError('GPU preflight failed')
    print(json.dumps(dict(preflight='passed',device=str(device),runtime=runtime())),flush=True)


def worker(a):
    import torch
    from phycom_v4.training import setup,train_group
    from phycom_v4.evaluation import evaluate_group
    from phycom_v4.timing import run_timing
    root=Path(a.output);c=config(a.smoke) if a.worker=='preflight' else read_json(root/'config.json')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    setup(c)
    try:
        if a.worker=='preflight': preflight(c,device);return 0
        run_id=read_json(root/'RUN_MANIFEST.json')['run_id']
        if a.worker=='timing': run_timing(root,c,run_id,device);return 0
        g=groups(c)[a.group_index];gd=root/'jobs'/group_id(g)
        if a.worker=='train': train_group(c,g,gd,device,run_id)
        else: evaluate_group(c,g,gd,device,run_id)
        return 0
    except PauseRun as e:
        print(str(e),flush=True);return 75


def device_slots(a):
    import torch
    if a.devices=='cpu':
        if not a.smoke: raise ValueError('Full publication run requires CUDA. Use --smoke for CPU validation.')
        return ['cpu']*max(1,a.cpu_workers)
    if not torch.cuda.is_available():
        if a.smoke: return ['cpu']*max(1,a.cpu_workers)
        raise RuntimeError('Enable Kaggle Settings > Accelerator > GPU T4 x2, then restart the session.')
    n=torch.cuda.device_count()
    if a.devices=='auto': return [str(i) for i in range(min(2,n))]
    ids=a.devices.split(',')
    if len(set(ids))!=len(ids) or any(not v.isdigit() or int(v)>=n for v in ids):
        raise ValueError('CUDA devices must be distinct visible indices, e.g. 0,1')
    return ids


def environment(slot,deadline=None):
    env=os.environ.copy()
    # Resolve user-visible indices against any existing CUDA_VISIBLE_DEVICES mask.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    original=visible.split(',') if visible else None
    env['CUDA_VISIBLE_DEVICES']='' if slot=='cpu' else (original[int(slot)] if original else slot)
    if deadline: env['PHYCOM_V4_STOP_AT']=str(deadline)
    else: env.pop('PHYCOM_V4_STOP_AT',None)
    env['PYTHONUNBUFFERED']='1'
    return env


def command(a,phase,index=None):
    args=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(Path(a.output).resolve()),'--worker',phase]
    if a.smoke: args.append('--smoke')
    if index is not None: args+=['--group-index',str(index)]
    return args


def dispatch(a,c,phase,slots,deadline):
    root=Path(a.output);logs=root/'logs';logs.mkdir(exist_ok=True)
    marker='TRAIN_COMPLETE.json' if phase=='train' else 'EVAL_COMPLETE.json'
    pending=[i for i,g in enumerate(groups(c)) if not (root/'jobs'/group_id(g)/marker).exists()]
    running={};paused=False;failed=[];last=0.
    try:
        while pending or running:
            if time.time()>=deadline: paused=True
            for k,slot in enumerate(slots):
                if k in running or not pending or paused or failed: continue
                i=pending.pop(0);gid=group_id(groups(c)[i]);p=logs/f'{phase}_{gid}.log'
                f=p.open('a');proc=subprocess.Popen(command(a,phase,i),stdout=f,stderr=subprocess.STDOUT,
                    env=environment(slot,deadline))
                running[k]=(proc,f,i,p)
                print(f'{phase}: {gid} -> {"CPU" if slot=="cpu" else "GPU "+slot}',flush=True)
            for k,(proc,f,i,p) in list(running.items()):
                code=proc.poll()
                if code is None: continue
                f.close();del running[k]
                if code==75: paused=True
                elif code:
                    failed.append(str(p));print(p.read_text()[-7000:],flush=True)
            if failed and running:
                # Keep the other worker's committed epochs; halt promptly on an error.
                for proc,f,i,p in running.values(): proc.terminate()
                for proc,f,i,p in running.values(): proc.wait();f.close()
                running.clear()
            if failed or (paused and not running): break
            if time.time()-last>=30:
                completed=sum(1 for _ in (root/'jobs').glob('*/'+marker))
                print(f'{phase}: {completed}/{len(groups(c))} groups complete; {len(running)} workers active; '
                    f'{max(0,(deadline-time.time())/3600):.2f} budget hours left.',flush=True)
                last=time.time()
            if running: time.sleep(1)
    finally:
        for proc,f,i,p in running.values():
            if proc.poll() is None: proc.terminate()
            proc.wait();f.close()
    if failed: raise RuntimeError('Worker failed; see '+', '.join(failed))
    return not paused and not pending


def snapshot_source(root):
    here=Path(__file__).resolve().parent;dest=Path(root)/'source'
    for name in ['run_revision.py','requirements.txt','README.md','REVIEWER_MAP.md','PROTOCOL.md','PROVENANCE.json']:
        if (here/name).exists():
            dest.mkdir(parents=True,exist_ok=True);shutil.copy2(here/name,dest/name)
    for name in ['phycom_v4','legacy_v32','tests']:
        shutil.copytree(here/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))


def main(a):
    c=config(a.smoke);tasks=plan(c);count=sum(len(x['models']) for x in tasks)
    if a.plan_only:
        print(json.dumps(dict(mode=c['mode'],fits=count,groups=len(tasks),tasks=tasks),indent=2));return 0
    slots=device_slots(a)
    for slot in sorted(set(slots)):
        subprocess.run(command(a,'preflight'),env=environment(slot),check=True)
    if a.preflight_only: return 0
    if a.hours<=0: raise ValueError('--hours must be positive')
    from phycom_v4.storage import restore,verify_selection,bundle
    root=Path(a.output);restored=restore(root,a.resume_input)
    root.mkdir(parents=True,exist_ok=True)
    rt=runtime();identity=dict(config=c,source_sha256=source_hash(),runtime=rt)
    run_id=digest(identity)
    manifest=root/'RUN_MANIFEST.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['run_id']!=run_id:
            raise RuntimeError('Resume refused: configuration, source or runtime differs. Use the original package/runtime '
                'or a NEW output directory. Never merge incompatible runs. Old identity: '+json.dumps(old['identity']))
        if read_json(root/'config.json')!=c: raise RuntimeError('Saved configuration changed')
    else:
        atomic_json(manifest,dict(run_id=run_id,identity=identity,expected_fits=count,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        atomic_json(root/'config.json',c);atomic_json(root/'EXPERIMENT_PLAN.json',tasks)
        snapshot_source(root)
    print(f'PHYCOM V4 {c["mode"]}: {count} fits; {len(slots)} independent workers. Run ID {run_id[:16]}.',flush=True)
    print('Selections are frozen before confirmation evaluation. This run does not assume an accuracy improvement.',flush=True)
    started=time.time();deadline=started+a.hours*3600;status='paused';failure=None
    try:
        if dispatch(a,c,'train',slots,deadline):
            verify_selection(root,c,run_id,freeze=True)
            if dispatch(a,c,'evaluate',slots,deadline):
                verify_selection(root,c,run_id)
                if time.time()<deadline:
                    code=subprocess.call(command(a,'timing'),env=environment(slots[0],deadline))
                    if code not in (0,75): raise RuntimeError('Isolated timing worker failed')
                    if code==0: status='complete'
    except (Exception,KeyboardInterrupt) as e:
        failure=f'{type(e).__name__}: {e}';status='failed'
        print(failure,flush=True)
    finally:
        # No GPU workers remain. Export tables even for a partial run and package for resume.
        from phycom_v4.reporting import generate_report
        try: generate_report(root,c,run_id)
        except Exception as e:
            status='failed';failure=(failure or '')+f'; Reporting: {type(e).__name__}: {e}'
            import traceback;traceback.print_exc()
        sessions=read_json(root/'SESSIONS.json') if (root/'SESSIONS.json').exists() else []
        sessions.append(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(started)),
            wall_seconds=time.time()-started,status=status,workers=slots,restored_from=restored,error=failure))
        atomic_json(root/'SESSIONS.json',sessions)
        atomic_json(root/'SESSION_STATUS.json',sessions[-1])
        print('Saving resume and review archives. Let this finish before ending the Kaggle session.',flush=True)
        for p in bundle(root): print('SAVED: '+p,flush=True)
    if failure: raise RuntimeError(failure)
    print('FINISHED: '+status.upper()+'. '+('Download REVIEW.zip and RESUME.zip.' if status=='complete' else
        'Download RESUME.zip; attach it as a private Kaggle input to the next session.'),flush=True)
    return 0


if __name__=='__main__':
    args=arguments()
    sys.exit(worker(args) if args.worker else main(args))
