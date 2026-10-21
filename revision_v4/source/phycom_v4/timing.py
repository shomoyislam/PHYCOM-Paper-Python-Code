"""Isolated inference timing, after the training/evaluation workers have exited."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import read_json,atomic_json,check_deadline,group_id,seed
from .protocol import groups,models,spec_id
from .training import selected_model
from .data import component_cov,components
from .linear import LinearFilter
from legacy_v32.simulation import temporal_cov


def measure(fn,device,c):
    sync=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda:None)
    with torch.inference_mode():
        for _ in range(c['timing_warmup']): fn()
        sync()
        baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
        if baseline is not None: torch.cuda.reset_peak_memory_stats(device)
        values=[]
        for _ in range(c['timing_trials']):
            sync();t=time.perf_counter();out=fn();sync()
            values.append((time.perf_counter()-t)*1000)
        checksum=float(out.abs().sum().cpu())
    return dict(mean_batch_ms=float(np.mean(values)),median_batch_ms=float(np.median(values)),
        p95_batch_ms=float(np.quantile(values,.95)),trials=len(values),
        incremental_peak_allocated_bytes=(int(torch.cuda.max_memory_allocated(device)-baseline)
            if baseline is not None else None),checksum=checksum)


def model_ops(s,n,T):
    # Real multiply-add = 2 operations; excludes nonlinearities, BN and movement.
    h,p,q=s['hidden'],s['projection'],s['head']
    inp=2*n*n if s['encoder']=='raw' else p
    cnn=0 if s['encoder']=='raw' else T*(2*n*n*(2*16*9+16*8*9)+2*8*n*n*p)
    recurrent=T*8*h*(inp+h)
    head=2*(h+int(s['condition'].startswith('concat')))*q+2*q*2*n*n
    return int(cnn+recurrent+head)


def run_timing(root,c,run_id,device):
    root=Path(root);paper=root/'Paper_Data';paper.mkdir(exist_ok=True)
    done=paper/'TIMING_COMPLETE.json'
    if done.exists():
        if read_json(done)['run_id']!=run_id: raise RuntimeError('Timing run mismatch')
        return
    path=paper/'latency.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    existing={(x['group'],x['method'],int(x['batch'])) for x in rows}
    setup_rows=[]
    for g in groups(c):
        if g['seed']!=c['seeds'][0]: continue
        gid=group_id(g);n=g['n'];T=c['T']
        filters={}
        for name in read_json(root/'jobs'/gid/'covariances.json'):
            with np.load(root/'jobs'/gid/'covariances'/f'{name}.npz') as ar:
                t=time.perf_counter();f=LinearFilter(ar['cov'],n,T,ar['mean']);w=f.weight(15.)
                elapsed=time.perf_counter()-t
                filters[name]=(w,f.mean)
                setup_rows.append(dict(group=gid,method=name,snr=15.,seconds=elapsed,
                    scope='CPU eigendecomposition and cached filter construction; excludes covariance fitting'))
        # An optimized temporal-only filter avoids artificially inflating iid LMMSE cost.
        R=temporal_cov(T,c['alpha']);w_t=np.linalg.solve(R+10**(-1.5)*np.eye(T),R[:,-1])
        for bs in c['timing_batches']:
            rng=np.random.default_rng(seed('timing',n,bs))
            y=torch.from_numpy(rng.normal(size=(bs,T,2,n,n)).astype(np.float32)).to(device)
            cy=torch.complex(y[:,:,0],y[:,:,1]);flat=cy.reshape(bs,-1)
            snr=torch.zeros(bs,1,device=device)
            methods=[('LS_copy',lambda:cy[:,-1].clone(),0,None)]
            wt=torch.from_numpy(w_t.astype(np.float32)).to(device)
            methods.append(('Jakes_temporal_optimized',lambda:(cy*wt[None,:,None,None]).sum(1),0,int(4*T*n*n)))
            # Build each callable with bound defaults; tensors remain resident on the same GPU.
            for name,(w,mu) in filters.items():
                W=torch.from_numpy(w.astype(np.complex64)).to(device)
                mean=torch.from_numpy(mu.astype(np.complex64)).to(device)
                methods.append((name,lambda W=W,mean=mean:(flat-mean)@W.T+mean[-n*n:],0,int(8*T*n**4)))
            for name,fn,params,ops in methods:
                if (gid,name,bs) in existing: continue
                check_deadline()
                r=measure(fn,device,c)
                rows.append(dict(run_id=run_id,group=gid,n=n,family=g['family'],method=name,batch=bs,
                    device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    precision='complex64; cached real/complex filters',parameters=params,approx_real_ops=ops,**r))
                pd.DataFrame(rows).to_csv(path,index=False)
            for s in models(c,g):
                name=spec_id(s)
                if (gid,name,bs) in existing: continue
                check_deadline()
                m=selected_model(c,g,s,root/'jobs'/gid/name,device,run_id)
                r=measure(lambda:m(y,snr),device,c)
                rows.append(dict(run_id=run_id,group=gid,n=n,family=g['family'],method=name,batch=bs,
                    device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    precision='float32',parameters=sum(p.numel() for p in m.parameters()),
                    approx_real_ops=model_ops(s,n,T),**r))
                pd.DataFrame(rows).to_csv(path,index=False);del m
    pd.DataFrame(setup_rows).to_csv(paper/'linear_filter_setup_cost.csv',index=False)
    atomic_json(done,dict(run_id=run_id,isolated_workers_stopped=True,rows=len(rows),
        scope='Synchronized wall latency; resident inputs and outputs; no host transfer, pilot despreading or SNR acquisition',
        input='Identical T-window shapes at 15 dB; model weights from first predeclared seed',
        overhead='Includes Python dispatch and synchronization; LS includes output copy',
        hardware_comparison='Measurements on this device only; do not infer deployment or two-GPU inference speedup'))
