"""Separate training and frozen evaluation workers, one independent job per GPU."""
import argparse
import gc
import json
from pathlib import Path
import torch
from .common import (read_json, atomic_json, stable_seed, job_id, model_names, stages,
                     file_hash, check_deadline, SessionPaused)
from .simulation import training_data
from .models import Estimator
from .training import train, load_checkpoint


def checked_marker(path,rid):
    if not path.exists(): return False
    if read_json(path)['run_id']!=rid: raise ValueError(f'Incompatible completion marker: {path}')
    return True


def head_diagnostics(model,cfg,job,device,stage):
    import numpy as np
    from .evaluation import sample_case
    from .simulation import real_input, real_target
    from .training import predict
    h,y,_ = sample_case(cfg,job,'main',0,30.)
    active=np.zeros(model.fc1.out_features,dtype=bool)
    positives=np.zeros(model.fc1.out_features,dtype=np.int64)
    def hook(module,inputs,output):
        pos=(output.detach()>0).cpu().numpy()
        active[:] |= pos.any(axis=0); positives[:] += pos.sum(axis=0)
    token=model.fc1.register_forward_hook(hook)
    try: predict(model,y,30.,device)
    finally: token.remove()
    weight=model.fc2.weight.detach().cpu().numpy().astype(np.float64)[:,active]
    U,s,_=np.linalg.svd(weight,full_matrices=False)
    tol=(s[0]*max(weight.shape)*np.finfo(float).eps) if len(s) else 0.
    rank=int(np.sum(s>tol)); U=U[:,:rank]
    target=real_target(h).reshape(len(h),-1).astype(np.float64)
    if model.ls_residual: target-=real_input(y)[:,-1].reshape(len(h),-1)
    target-=model.fc2.bias.detach().cpu().numpy()[None,:]
    residual=target-(target@U)@U.T
    bound=float(np.sum(residual**2)/np.sum(np.abs(h).astype(np.float64)**2))
    row=dict(**job,model=model.name,stage=stage,head_width=model.fc1.out_features,
        output_dimension=2*job['n']**2,observed_inactive_units=int((~active).sum()),
        observed_output_span_rank=rank,affine_span_nmse_lower_bound=bound,
        diagnostic_snr_db=30,test_sequences=len(h),minimum_positive_fraction=float(positives.min()/len(h)),
        interpretation='Observed on this frozen 30-dB set; not proof of global inactivity or a causal mechanism.')
    return row


def run_job(root,job,device,phase):
    man=read_json(root/'run_manifest.json'); cfg=man['config']; rid=man['run_id']
    torch.set_num_threads(cfg['torch_threads'])
    if device.type=='cuda': torch.cuda.set_device(device)
    folder=root/'jobs'/job_id(job); folder.mkdir(parents=True,exist_ok=True)
    marker=folder/('TRAINED.json' if phase=='train' else 'COMPLETE.json')
    if checked_marker(marker,rid): return
    names=model_names(cfg,job)
    if phase=='train':
        N=cfg['train_samples'] if job['kind']=='main' else cfg['seq_train']
        V=cfg['val_samples'] if job['kind']=='main' else cfg['seq_val']
        key=(job['kind'],job['n'],job['T'],job['seed'])
        trseed=stable_seed('train',*key); vseed=stable_seed('validation',*key)
        atomic_json(folder/'protocol.json',dict(job=job,run_id=rid,train_seed=trseed,val_seed=vseed,
            train_samples=N,val_samples=V,models=names,device=str(device),
            test_namespace=cfg['test_namespace'],selection='minimum validation MSE; no test use'))
        remaining=[name for name in names if not checked_marker(folder/name/'TRAINED.json',rid)]
        if remaining:
            check_deadline()
            print(f'Generate {job_id(job)}: train={N}, validation={V}',flush=True)
            tr=training_data(N,job['n'],job['T'],cfg['alpha'],trseed,cfg['M'],cfg['train_snr'])
            va=training_data(V,job['n'],job['T'],cfg['alpha'],vseed,cfg['M'],cfg['train_snr'])
            for name in remaining:
                check_deadline()
                model=Estimator(name,job['n'],seed=stable_seed('model',job['seed']))
                model=train(model,tr,va,cfg,job,folder/name,device,rid)
                atomic_json(folder/name/'TRAINED.json',dict(run_id=rid,job=job,model=name))
                del model; gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            del tr,va
    else:
        from .evaluation import evaluate_model,evaluate_baselines
        selection=read_json(root/'frozen_selection.json')
        if selection['run_id']!=rid: raise ValueError('Frozen selection run mismatch')
        frozen={r['checkpoint']:r['sha256'] for r in selection['checkpoints']}
        for name in names:
            dest=folder/name
            if checked_marker(dest/'COMPLETE.json',rid): continue
            for stage in stages(job):
                check_deadline()
                path=dest/('budget60.pt' if stage=='budget60' else 'best.pt')
                if file_hash(path)!=frozen[path.relative_to(root).as_posix()]:
                    raise ValueError('Selected checkpoint changed after freezing')
                model=Estimator(name,job['n']).to(device)
                model.load_state_dict(load_checkpoint(path)['state_dict']); model.eval()
                print(f'Evaluate {job_id(job)} / {name} / {stage}',flush=True)
                evaluate_model(model,cfg,job,dest/stage,device,stage,root)
                if job['kind']=='main' and not (dest/stage/'head_diagnostics.json').exists():
                    atomic_json(dest/stage/'head_diagnostics.json',head_diagnostics(model,cfg,job,device,stage))
                del model; gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            atomic_json(dest/'COMPLETE.json',dict(run_id=rid,job=job,model=name))
        # Classical estimates are computed once for each array/window, not once per training seed.
        first_seed=cfg['seeds'][0] if job['kind']=='main' else cfg['seq_seeds'][0]
        if job['seed']==first_seed:
            evaluate_baselines(cfg,job,folder/'baselines',root)
    atomic_json(marker,dict(run_id=rid,job=job,phase=phase))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--job',required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--phase',choices=['train','evaluate'],required=True)
    args=p.parse_args()
    try: run_job(Path(args.root),json.loads(args.job),torch.device(args.device),args.phase)
    except SessionPaused as exc:
        print(str(exc),flush=True); raise SystemExit(75)
