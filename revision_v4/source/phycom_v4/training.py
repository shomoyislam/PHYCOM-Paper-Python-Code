import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import seed,atomic_json,read_json,check_deadline,file_hash
from .protocol import models,spec_id
from .models import Estimator,parameter_count
from .data import training_arrays
from .linear import covariance_baselines


def setup(c):
    torch.set_num_threads(c['torch_threads'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=c['deterministic']
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(c['deterministic'])


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    torch.save(value,temp);os.replace(temp,path)


def load_checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def selected_model(c,g,s,folder,device,run_id):
    ck=load_checkpoint(Path(folder)/'best.pt')
    if ck['run_id']!=run_id or ck['spec']!=s or ck['group']!=g:
        raise RuntimeError('Selected checkpoint identity mismatch')
    m=Estimator(s,g['n'],seed('model',g['seed'])).to(device)
    m.load_state_dict(ck['state_dict']);m.eval()
    return m


def train_one(c,g,s,tr,va,folder,device,run_id,max_new_epochs=None):
    """One committed epoch is atomic; test arrays are not accepted by this API."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    done=folder/'training.json'
    if done.exists() and read_json(done).get('complete'):
        d=read_json(done)
        if d['run_id']!=run_id or file_hash(folder/'best.pt')!=d['best_sha256']:
            raise RuntimeError('Completed checkpoint identity/hash mismatch')
        return d
    model=Estimator(s,g['n'],seed('model',g['seed'])).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=c['lr_patience'],factor=c['lr_factor'])
    start=wait=0;best=float('inf');best_epoch=0;history=[];elapsed=0.;selected=None
    random.seed(seed('optimization',g['seed']));torch.manual_seed(seed('optimization',g['seed']))
    if (folder/'last.pt').exists():
        ck=load_checkpoint(folder/'last.pt')
        if ck['run_id']!=run_id or ck['spec']!=s or ck['group']!=g:
            raise RuntimeError('Resume refused: checkpoint/configuration mismatch')
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start,wait,best,best_epoch=ck['epoch'],ck['wait'],ck['best_val'],ck['best_epoch']
        history,elapsed,selected=ck['history'],ck['elapsed'],ck['best_state_dict']
        torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state(ck['cuda_rng'],device)
    bs=c['batch_size']
    for epoch in range(start,c['epochs']):
        if wait>=c['patience']: break
        if max_new_epochs is not None and epoch-start>=max_new_epochs: return None
        check_deadline();t0=time.perf_counter();model.train()
        gen=torch.Generator().manual_seed(seed('shuffle',g['seed'],epoch))
        ix=torch.randperm(len(tr[0]),generator=gen).to(device)
        total=torch.zeros((),device=device)
        for lo in range(0,len(ix),bs):
            if lo%(bs*100)==0: check_deadline()
            y,snr,h=(a[ix[lo:lo+bs]] for a in tr)
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(y,snr),h)
            loss.backward()
            if c['gradient_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
            opt.step();total+=loss.detach()*len(y)
        model.eval();validation=torch.zeros((),device=device)
        with torch.inference_mode():
            for lo in range(0,len(va[0]),c['eval_batch_size']):
                y,snr,h=(a[lo:lo+c['eval_batch_size']] for a in va)
                validation+=torch.nn.functional.mse_loss(model(y,snr),h)*len(y)
        tl=float(total.cpu())/len(tr[0]);vl=float(validation.cpu())/len(va[0])
        if not np.isfinite(tl+vl): raise FloatingPointError(f'Nonfinite training for {spec_id(s)}')
        elapsed+=time.perf_counter()-t0;scheduler.step(vl)
        history.append(dict(epoch=epoch+1,train_mse=tl,val_mse=vl,lr=opt.param_groups[0]['lr']))
        state=cpu_state(model)
        if vl<best: best,wait,best_epoch,selected=vl,0,epoch+1,state
        else: wait+=1
        ck=dict(run_id=run_id,spec=s,group=g,epoch=epoch+1,wait=wait,
            best_val=best,best_epoch=best_epoch,state_dict=state,best_state_dict=selected,
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),elapsed=elapsed,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        save_checkpoint(folder/'last.pt',ck)
        atomic_json(folder/'progress.json',dict(model=spec_id(s),epoch=epoch+1,best_epoch=best_epoch,best_val=best))
        if epoch==start or (epoch+1)%10==0:
            print(f"{g['family']} {g['n']}x{g['n']} seed={g['seed']} {spec_id(s)} epoch={epoch+1} val={vl:.6g}",flush=True)
    if selected is None: raise RuntimeError('No selected checkpoint')
    model.load_state_dict(selected);model.eval()
    beta=1.
    if s['residual']:
        cross=sq=0.
        with torch.inference_mode():
            for lo in range(0,len(va[0]),c['eval_batch_size']):
                y,snr,h=(a[lo:lo+c['eval_batch_size']] for a in va)
                delta=model(y,snr)-y[:,-1];e=y[:,-1]-h
                cross+=float((e.double()*delta.double()).sum().cpu())
                sq+=float(delta.double().square().sum().cpu())
        beta=float(np.clip(-cross/max(sq,1e-30),0,1))
    save_checkpoint(folder/'best.pt',dict(run_id=run_id,spec=s,group=g,state_dict=selected,
        best_epoch=best_epoch,best_val=best,validation_residual_scale=beta))
    window=c['convergence_window']
    old=min((x['val_mse'] for x in history[:-window]),default=history[0]['val_mse'])
    recent=max(0.,(old-best)/max(old,1e-30))
    record=dict(complete=True,run_id=run_id,spec=s,group=g,history=history,
        best_epoch=best_epoch,best_val=best,epochs_run=len(history),training_seconds=elapsed,
        parameters=parameter_count(model),stop_reason='validation_patience' if wait>=c['patience'] else 'epoch_cap',
        recent_relative_validation_gain=recent,convergence_window=window,validation_residual_scale=beta,
        precision=c['precision'],best_sha256=file_hash(folder/'best.pt'))
    atomic_json(done,record)
    return record


def train_group(c,g,root,device,run_id):
    from .common import atomic_npz
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'TRAIN_COMPLETE.json').exists():
        if read_json(root/'TRAIN_COMPLETE.json')['run_id']!=run_id: raise RuntimeError('Group identity mismatch')
        return
    check_deadline()
    arrays,h,y,snrs=training_arrays(c,g,'train')
    va,_,_,_=training_arrays(c,g,'validation')
    # Tiny windows fit comfortably in T4 VRAM. No DataLoader workers or inter-GPU copies.
    tr=tuple(torch.from_numpy(a).to(device) for a in arrays)
    vt=tuple(torch.from_numpy(a).to(device) for a in va)
    if not (root/'covariances.json').exists():
        covariance=covariance_baselines(c,g,h)
        info={}
        for name,(cov,mean,seconds) in covariance.items():
            atomic_npz(root/'covariances'/f'{name}.npz',cov=cov,mean=mean)
            info[name]=dict(fit_seconds=seconds,source='training_clean_channels_only')
        atomic_json(root/'covariances.json',info)
    del h,y,snrs,arrays,va
    for s in models(c,g):
        check_deadline()
        train_one(c,g,s,tr,vt,root/spec_id(s),device,run_id)
    atomic_json(root/'TRAIN_COMPLETE.json',dict(run_id=run_id,models=len(models(c,g))))


def predict(model,y,reported,device,batch=2048,features=False):
    from legacy_v32.simulation import real_input
    reports=np.broadcast_to(reported,(len(y),)).astype(np.float32)
    predictions=[];feats={};model.eval()
    with torch.inference_mode():
        for lo in range(0,len(y),batch):
            x=torch.from_numpy(real_input(y[lo:lo+batch])).to(device)
            s=torch.from_numpy(((reports[lo:lo+batch]-15)/15)[:,None].copy()).to(device)
            out=model(x,s,return_features=features)
            if features:
                out,parts=out
                for k,v in parts.items(): feats.setdefault(k,[]).append(v.cpu().numpy())
            a=out.cpu().numpy();predictions.append(a[:,0]+1j*a[:,1])
    return np.concatenate(predictions), {k:np.concatenate(v) for k,v in feats.items()}
