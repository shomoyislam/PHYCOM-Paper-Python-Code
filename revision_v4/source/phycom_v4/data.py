"""Finite-sinusoid Jakes channels with physical ULA angular covariance controls."""
import numpy as np
from legacy_v32.simulation import jakes, temporal_cov, observe, real_input, real_target
from .common import seed, check_deadline


def ula_cov(n,mean_deg,spread_deg,spacing=.5):
    # Uniform angular power on [mean-spread,mean+spread], in the visible region.
    angles=np.deg2rad(np.linspace(mean_deg-spread_deg,mean_deg+spread_deg,513))
    a=np.exp(2j*np.pi*spacing*np.arange(n)[:,None]*np.sin(angles)[None,:])
    return (a@a.conj().T)/len(angles)


def psd_root(C):
    d,v=np.linalg.eigh((C+C.conj().T)/2)
    return (v*np.sqrt(np.maximum(d,0)))@v.conj().T


def components(c,family):
    if family=='iid': return [dict(alpha=c['alpha'],spatial=False,mean=0.,spread=0.)]
    if family=='spatial':
        return [dict(alpha=c['alpha'],spatial=True,mean=c['spatial_mean_deg'],spread=c['spatial_spread_deg'])]
    if family=='unknown_stats':
        return [dict(alpha=a,spatial=True,mean=c['spatial_mean_deg'],spread=d)
                for a in c['mixture_alphas'] for d in c['mixture_spreads']]
    raise ValueError(family)


def component_cov(c,n,component):
    S=(np.kron(ula_cov(n,component['mean'],component['spread'],c['spacing_wavelengths']),
                ula_cov(n,component['mean'],component['spread'],c['spacing_wavelengths']).T)
       if component['spatial'] else np.eye(n*n))
    return np.kron(temporal_cov(c['T'],component['alpha']),S).astype(np.complex128)


def channels(c,n,count,data_seed,comps,gaussian=False):
    rng=np.random.default_rng(data_seed)
    labels=rng.integers(0,len(comps),size=count)
    out=np.empty((count,c['T'],n,n),np.complex64)
    for k,comp in enumerate(comps):
        check_deadline()
        ix=np.where(labels==k)[0]
        if not len(ix): continue
        if gaussian:
            root=psd_root(component_cov(c,n,comp))
            z=(rng.standard_normal((len(ix),len(root)))+1j*rng.standard_normal((len(ix),len(root))))/np.sqrt(2)
            h=(z@root.T).reshape(-1,c['T'],n,n)
        else:
            h=jakes(len(ix),n,c['T'],comp['alpha'],rng,M=c['M'])
            if comp['spatial']:
                root=psd_root(ula_cov(n,comp['mean'],comp['spread'],c['spacing_wavelengths']))
                h=np.einsum('ij,btjk,lk->btil',root,h,root.conj(),optimize=True)
        out[ix]=h
    return out,labels


def training_arrays(c,group,split):
    count=c['train_samples'] if split=='train' else c['val_samples']
    key=seed(c['train_namespace'],group['family'],group['n'],group['seed'],split)
    h,labels=channels(c,group['n'],count,key,components(c,group['family']))
    rng=np.random.default_rng(seed(key,'observation'))
    snrs=rng.uniform(*c['train_snr'],count).astype(np.float32)
    y=observe(h,snrs,rng)
    return (real_input(y),((snrs-15)/15)[:,None],real_target(h[:,-1])), h, y, snrs


def cases(c,family):
    base=components(c,family)
    r=[dict(name=f'main_snr{snr}',snr=float(snr),components=base,gaussian=False,category='main',bias=0.,std=0.,clip=False)
       for snr in c['snrs']]
    if family=='iid':
        for snr in c['stress_true_snrs']:
            r.append(dict(name=f'report_reference_snr{snr:g}',snr=snr,components=base,gaussian=False,
                category='report_reference',bias=0.,std=0.,clip=False))
            for bias in c['snr_biases']:
                if bias==0: continue
                for clip in (False,True):
                    reported=snr+bias
                    category='report_in_range' if c['train_snr'][0]<=reported<=c['train_snr'][1] else 'report_extrapolation'
                    if clip: category='report_clipped'
                    r.append(dict(name=f'bias_snr{snr:g}_b{bias:g}_clip{int(clip)}',snr=snr,components=base,
                        gaussian=False,category=category,bias=bias,std=0.,clip=clip))
            for std in c['snr_error_stds']:
                for clip in (False,True):
                    r.append(dict(name=f'noise_snr{snr:g}_sd{std:g}_clip{int(clip)}',snr=snr,components=base,
                        gaussian=False,category='report_clipped' if clip else 'report_mixed_range',bias=0.,std=std,clip=clip))
            r.append(dict(name=f'calibrated_snr{snr:g}',snr=snr,components=base,gaussian=False,
                category='estimated_snr_extra_noise_samples',bias=0.,std=0.,clip=False,calibration=True))
        for a in (.3,.6,.95):
            r.append(dict(name=f'temporal_alpha{a}',snr=15.,components=[dict(base[0],alpha=a)],gaussian=False,
                category='temporal_shift',bias=0.,std=0.,clip=False))
        r.append(dict(name='physical_spatial_shift',snr=15.,components=components(c,'spatial'),gaussian=False,
            category='spatial_shift',bias=0.,std=0.,clip=False))
        for snr in (15.,30.):
            r.append(dict(name=f'gaussian_snr{snr:g}',snr=snr,components=base,gaussian=True,
                category='exact_gaussian_control',bias=0.,std=0.,clip=False))
    if family=='unknown_stats':
        comp=[dict(alpha=.75,spatial=True,mean=c['spatial_mean_deg'],spread=20.)]
        for snr in (15.,30.):
            r.append(dict(name=f'heldout_stats_snr{snr:g}',snr=snr,components=comp,gaussian=False,
                category='heldout_covariance_parameters',bias=0.,std=0.,clip=False))
    return r


def make_case(c,n,case):
    # All fits share test windows. Reporting perturbations share the same H and Y.
    key=seed(c['confirmation_namespace'],n,case['components'],case['gaussian'])
    count=c['test_samples'] if case['category']=='main' else c['stress_samples']
    h,labels=channels(c,n,count,key,case['components'],case['gaussian'])
    y=observe(h,case['snr'],np.random.default_rng(seed(key,'noise',case['snr'])))
    reported=np.full(count,case['snr']+case['bias'],np.float64)
    if case['std']:
        reported+=np.random.default_rng(seed(key,'report',case['snr'],case['std'])).normal(0,case['std'],count)
    if case.get('calibration'):
        rng=np.random.default_rng(seed(key,'noise_calibration',case['snr']))
        # L independent noise-only complex samples: mean power ~ Gamma(L, sigma²/L).
        L=c['noise_calibration_samples']
        variance=rng.gamma(L,10**(-case['snr']/10)/L,count)
        reported=-10*np.log10(variance)
    outside=(reported<c['train_snr'][0])|(reported>c['train_snr'][1])
    if case['clip']: reported=np.clip(reported,*c['train_snr'])
    return h,y,reported,labels,outside
