"""Fresh confirmation data, paired controls, and resumable raw block statistics."""
from pathlib import Path
import os
import numpy as np
from . import simulation as sim
from .common import stable_seed, atomic_json, read_json, check_deadline
from .training import predict


def channel_set(cfg, job, scenario='main', parameter=0):
    n,T = job['n'], job['T']
    count = cfg['test_samples'] if scenario in ('main','sequence') else cfg['stress_samples']
    alpha = parameter if scenario=='temporal' else cfg['alpha']
    rho = parameter if scenario=='spatial' else 0.
    seed = stable_seed(cfg['test_namespace'],'heldout_channels',scenario,parameter,n,T)
    return sim.test_channels(count,n,T,alpha,seed,rho=rho,M=cfg['M'])


def observation(h, cfg, job, scenario, parameter, snr):
    seed = stable_seed(cfg['test_namespace'],'heldout_noise',scenario,parameter,job['n'],job['T'],snr)
    return sim.observe(h,snr,np.random.default_rng(seed))


def row_base(job, model, stage, scenario, param, snr, cfg):
    return dict(suite=job['kind'],n=job['n'],T=job['T'],seed=job['seed'],model=model,
        stage=stage,scenario=scenario,parameter=float(param),snr_db=float(snr),
        test_namespace=cfg['test_namespace'])


def score(h, estimate, snr, cfg, seed, with_link=False, raw_path=None):
    metrics,e,p = sim.nmse(h,estimate)
    arrays = dict(error_energy=e,channel_energy=p)
    if with_link:
        C = min(cfg['ber_blocks'],len(h)); R = min(cfg['rate_blocks'],len(h))
        errors = sim.ber_blocks(h[:C],estimate[:C],snr,cfg['bits_per_block'],seed)
        ber = sim.bootstrap_mean(errors/cfg['bits_per_block'],seed=991,resamples=cfg['bootstrap_resamples'])
        rates = sim.zf_rate_blocks(h[:R],estimate[:R],snr)
        rate = sim.bootstrap_mean(rates,seed=991,resamples=cfg['bootstrap_resamples'])
        metrics.update(ber=ber['mean'],ber_ci95_low=ber['ci95_low'],ber_ci95_high=ber['ci95_high'],
            independent_ber_blocks=C,bits_per_block=cfg['bits_per_block'],zf_sum_rate=rate['mean'],
            rate_ci95_low=rate['ci95_low'],rate_ci95_high=rate['ci95_high'])
        arrays.update(ber_errors=errors,bits_per_block=np.array(cfg['bits_per_block']),rate_blocks=rates)
    if raw_path is not None:
        raw_path = Path(raw_path); raw_path.parent.mkdir(parents=True,exist_ok=True)
        temp = raw_path.with_suffix('.tmp')
        with temp.open('wb') as f: np.savez_compressed(f,**arrays)
        os.replace(temp,raw_path)
    return metrics


def conditions(cfg, job, stress):
    scenario = 'main' if job['kind']=='main' else 'sequence'
    out = [(scenario,0,s) for s in (cfg['snrs'] if job['kind']=='main' else [cfg['stress_snr']])]
    if job['kind']=='main' and stress:
        for scenario,grid in [('temporal',cfg['temporal_grid']),('spatial',cfg['spatial_grid'])]:
            out.extend((scenario,p,cfg['stress_snr']) for p in grid)
        for snr in [15.,30.]:
            for scenario,grid in [('snr_bias',cfg['snr_bias_grid']),('snr_error_std',cfg['snr_error_std_grid'])]:
                out.extend((scenario,p,snr) for p in grid)
    return out


def sample_case(cfg,job,scenario,parameter,snr):
    side_test = scenario.startswith('snr_')
    ch_scenario, ch_parameter = ('snr_sideinfo',0) if side_test else (scenario,parameter)
    hs = channel_set(cfg,job,ch_scenario,ch_parameter)
    y = observation(hs,cfg,job,ch_scenario,ch_parameter,snr)
    side = np.full(len(y),snr,dtype=np.float32)
    if scenario=='snr_bias': side += parameter
    elif scenario=='snr_error_std':
        seed = stable_seed(cfg['test_namespace'],'snr_error',job['n'],job['T'],snr)
        z = np.random.default_rng(seed).standard_normal(len(y))
        side = (snr+parameter*z).astype(np.float32)
    return hs[:,-1],y,side


def case_key(scenario,parameter,snr):
    return f'{scenario}_p{parameter:g}_snr{snr:g}'


def temporal_with_sideinfo(y, reported_snr, alpha=.8):
    """Fixed Jakes covariance with per-window reported noise variance."""
    R = sim.temporal_cov(y.shape[1],alpha)
    lam,V = np.linalg.eigh(R)
    noise = 10.**(-np.asarray(reported_snr,dtype=np.float64)/10)
    weights = ((V.T@R[:,-1])[None,:]/(lam[None,:]+noise[:,None]))@V.T
    return np.einsum('bt,btij->bij',weights,y).astype(np.complex64)


def cached_case(folder,key,make):
    json_path = folder/'cases'/(key+'.json'); raw_path = folder/'raw'/(key+'.npz')
    if json_path.exists() and raw_path.exists(): return read_json(json_path)
    check_deadline()
    row = make(raw_path)
    atomic_json(json_path,row)
    return row


def evaluate_model(model,cfg,job,folder,device,stage,root):
    folder = Path(folder); rows=[]
    for scenario,parameter,snr in conditions(cfg,job,model.name in cfg['stress_models']):
        def compute(raw_path):
            h,y,side = sample_case(cfg,job,scenario,parameter,snr)
            est = predict(model,y,side,device)
            row = row_base(job,model.name,stage,scenario,parameter,snr,cfg)
            row.update(score(h,est,snr,cfg,stable_seed(cfg['test_namespace'],'bits',job['n'],job['T'],snr),
                with_link=scenario=='main',raw_path=raw_path))
            row.update(raw_file=raw_path.relative_to(root).as_posix(),
                sideinfo_clipped=False,sideinfo_used=model.condition!='none')
            return row
        rows.append(cached_case(folder,case_key(scenario,parameter,snr),compute))
    atomic_json(folder/'metrics.json',rows)
    return rows


def evaluate_baselines(cfg,job,folder,root):
    folder=Path(folder); rows=[]
    for scenario,parameter,snr in conditions(cfg,job,True):
        check_deadline()
        side_test = scenario.startswith('snr_')
        names = ['LS','Jakes_LMMSE_fixed','Jakes_LMMSE_reported_snr'] if side_test else [
            'LS','LMMSE_instant_iid','Kalman_AR1_fixed','Jakes_LMMSE_fixed','Jakes_LMMSE_oracle']
        if scenario=='main': names += ['Perfect_CSI']
        # Share generated channels among all linear estimators for this case.
        h,y,side = sample_case(cfg,job,scenario,parameter,snr)
        if side_test:
            estimates=dict(LS=y[:,-1],Jakes_LMMSE_fixed=sim.temporal_lmmse(y,snr,cfg['alpha']),
                           Jakes_LMMSE_reported_snr=temporal_with_sideinfo(y,side,cfg['alpha']))
        else:
            estimates=sim.baselines(y,snr,parameter if scenario=='temporal' else cfg['alpha'],
                parameter if scenario=='spatial' else 0.)
            if scenario=='main': estimates['Perfect_CSI']=h
        for name in names:
            def compute(raw_path):
                row=row_base(job,name,'classical',scenario,parameter,snr,cfg)
                row['seed']=None; row['replicate_unit']='one shared Monte Carlo channel set'
                row.update(score(h,estimates[name],snr,cfg,
                    stable_seed(cfg['test_namespace'],'bits',job['n'],job['T'],snr),
                    with_link=scenario=='main',raw_path=raw_path))
                row['raw_file']=raw_path.relative_to(root).as_posix()
                row['noise_variance_information']='reported_snr' if name=='Jakes_LMMSE_reported_snr' else 'true_nominal_snr'
                if name=='LS': row['noise_variance_information']='not_required'
                if scenario in ('main','sequence') and name=='Jakes_LMMSE_oracle':
                    row['theoretical_nmse']=sim.temporal_lmmse_theory(job['T'],snr,cfg['alpha'])
                return row
            rows.append(cached_case(folder/name,case_key(scenario,parameter,snr),compute))
    atomic_json(folder/'metrics.json',rows)
    return rows
