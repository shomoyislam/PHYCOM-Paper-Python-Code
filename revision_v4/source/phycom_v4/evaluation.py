import json
from pathlib import Path
import numpy as np
from .common import atomic_npz,atomic_json,read_json,check_deadline,seed,file_hash
from .protocol import models,spec_id
from .training import selected_model,predict
from .data import cases,make_case
from .linear import LinearFilter,oracle_estimate
from legacy_v32.simulation import ber_blocks,zf_rate_blocks


def spectral_info(a,center=False):
    a=np.asarray(a,np.float64)
    if center: a=a-a.mean(0)
    s=np.linalg.svd(a,compute_uv=False)
    # Models/features are stored in float32; a float64 rank tolerance would
    # misleadingly treat float32 rounding remnants as reliable dimensions.
    tol=float(max(a.shape)*float(np.finfo(np.float32).eps)*(float(s[0]) if len(s) else 0))
    rank=int(np.sum(s>tol));power=s*s
    prob=power/power.sum() if power.sum() else power
    effective=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))) if power.sum() else 0.
    condition=float(s[0]/s[rank-1]) if rank else None
    return dict(shape=list(a.shape),singular_values=s.tolist(),numerical_rank=rank,
        rank_tolerance=tol,rank_tolerance_precision='float32',
        full_spectrum_condition_number=float(s[0]/s[-1]) if len(s) and s[-1]>0 else None,
        domain_nullity=int(a.shape[1]-rank),condition_number_nonzero_spectrum=condition,
        effective_rank_variance_entropy=effective,
        stable_rank=float(power.sum()/power[0]) if len(s) and power[0] else 0.)


def diagnostics(model,y,reported,device,c):
    _,feats=predict(model,y[:c['diagnostics_samples']],np.asarray(reported)[:c['diagnostics_samples']],
        device,c['eval_batch_size'],True)
    r={'feature_spectra':{k:spectral_info(v,True) for k,v in feats.items()},
       'weight_spectra':{},'descriptive_only':True}
    for name,layer in [('projection',getattr(model,'proj',None)),('head_in',model.fc1),('head_out',model.fc2)]:
        if layer is not None: r['weight_spectra'][name]=spectral_info(layer.weight.detach().cpu().numpy())
    active=np.max(feats['head_relu'],axis=0)>1e-8
    r['active_head_units_on_diagnostic_windows']=int(active.sum())
    r['active_output_matrix']=spectral_info(model.fc2.weight.detach().cpu().numpy()[:,active])
    if hasattr(model,'gate'):
        import torch
        with torch.inference_mode():
            inputs=torch.tensor([[-1.],[0.],[1.]],device=device)
            value=model.gate(inputs)
            if model.condition=='gate': value=torch.sigmoid(value)
            elif model.condition=='gate_identity': value=2*torch.sigmoid(value)
            elif model.condition=='resgate': value=1+torch.sigmoid(value)
            else: value=1+torch.tanh(value)
            r['gate_values_snr_0_15_30']=value.cpu().tolist()
    return r


def result_arrays(c,g,case,h,y,pred,reported,outside,run_id):
    h=h[:,-1]
    power=np.sum(np.abs(h.astype(np.complex128))**2,axis=(1,2))
    err=pred.astype(np.complex128)-h
    errors=np.sum(np.abs(err)**2,axis=(1,2))
    # Exact risk identity: ||LS-H+c||² - ||LS-H||² = ||c||² + 2 Re <LS-H,c>.
    delta=pred.astype(np.complex128)-y[:,-1]
    ls_error=y[:,-1].astype(np.complex128)-h
    corr_sq=np.sum(np.abs(delta)**2,axis=(1,2))
    corr_cross=2*np.sum((ls_error.conj()*delta).real,axis=(1,2))
    r=dict(errors=errors,power=power,correction_energy=corr_sq,correction_cross=corr_cross,
           reported_snr=np.asarray(reported),outside_training_range=np.asarray(outside,dtype=np.uint8),
           run_id=np.array(run_id))
    if case['category']=='main':
        b=min(c['ber_blocks'],len(h));bits=c['bits_per_stream']*g['n']
        r['ber_errors']=ber_blocks(h[:b],pred[:b],case['snr'],bits,
            seed(c['confirmation_namespace'],'BER',g['n'],case['name'],g['family']))
        r['bits_per_block']=np.array(bits)
        b=min(c['rate_blocks'],len(h))
        r['rate']=zf_rate_blocks(h[:b],pred[:b],case['snr'])
    if not all(np.isfinite(v).all() for k,v in r.items() if k!='run_id'):
        raise FloatingPointError('Nonfinite evaluation arrays')
    return r


def evaluate_group(c,g,root,device,run_id):
    root=Path(root)
    complete=root/'EVAL_COMPLETE.json'
    if complete.exists():
        if read_json(complete)['run_id']!=run_id: raise RuntimeError('Evaluation identity mismatch')
        return
    estimators={}
    for s in models(c,g):
        folder=root/spec_id(s)
        estimators[spec_id(s)]=(selected_model(c,g,s,folder,device,run_id),
            read_json(folder/'training.json')['validation_residual_scale'])
    covariance={}
    for name in read_json(root/'covariances.json'):
        with np.load(root/'covariances'/f'{name}.npz') as a:
            covariance[name]=LinearFilter(a['cov'],g['n'],c['T'],a['mean'])
    case_records=[]
    for case in cases(c,g['family']):
        check_deadline()
        folder=root/'evaluation'/case['name'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'COMPLETE.json').exists():
            record=read_json(folder/'COMPLETE.json')
            if record['run_id']!=run_id: raise RuntimeError('Case identity mismatch')
            case_records.append(record);continue
        h,y,reported,labels,outside=make_case(c,g['n'],case)
        # A matched oracle uses true SNR; the reported-SNR oracle isolates scalar errors.
        candidates=['LS','Oracle_component_trueSNR','Oracle_component_reportedSNR']+list(covariance)+list(estimators)
        candidates += [name+'__val_scaled' for name,(m,beta) in estimators.items() if m.spec['residual']]
        hashes={}
        cache={}
        for name in candidates:
            check_deadline();target=folder/(name+'.npz')
            if target.exists():
                with np.load(target) as a:
                    if str(a['run_id'])!=run_id: raise RuntimeError('Raw-array identity mismatch')
                hashes[name]=file_hash(target);continue
            if name=='LS': pred=y[:,-1]
            elif name=='Oracle_component_trueSNR':
                pred=oracle_estimate(c,g['n'],case,y,case['snr'],labels)
            elif name=='Oracle_component_reportedSNR':
                pred=oracle_estimate(c,g['n'],case,y,reported,labels)
            elif name in covariance: pred=covariance[name].predict(y,reported)
            else:
                base=name.split('__')[0];m,beta=estimators[base]
                if base not in cache: cache[base]=predict(m,y,reported,device,c['eval_batch_size'])[0]
                pred=cache[base]
                if '__val_scaled' in name: pred=y[:,-1]+beta*(pred-y[:,-1])
                if not m.spec['residual'] or '__val_scaled' in name: cache.pop(base,None)
            atomic_npz(target,**result_arrays(c,g,case,h,y,pred,reported,outside,run_id))
            hashes[name]=file_hash(target)
        if case['name']=='main_snr30':
            for name,(m,beta) in estimators.items():
                target=root/name/'diagnostics.json'
                if not target.exists(): atomic_json(target,diagnostics(m,y,reported,device,c))
        record=dict(run_id=run_id,case=case,samples=len(h),files=hashes,
            outside_training_fraction=float(outside.mean()),reported_min=float(reported.min()),
            reported_max=float(reported.max()),overhead_noise_samples=c['noise_calibration_samples'] if case.get('calibration') else 0)
        atomic_json(folder/'COMPLETE.json',record);case_records.append(record)
        print(f"Evaluated {g['family']} {g['n']}x{g['n']} seed={g['seed']} {case['name']}",flush=True)
    atomic_json(complete,dict(run_id=run_id,cases=case_records))
