"""Second-moment baselines. Finite-SoS LMMSE is not asserted to be nonlinear MMSE."""
import time
import numpy as np
from .data import component_cov,components


class LinearFilter:
    def __init__(self,cov,n,T,mean=None):
        cov=np.asarray(cov,np.complex128)
        self.n,self.T=n,T
        self.mean=np.zeros(T*n*n,np.complex128) if mean is None else np.asarray(mean,np.complex128)
        self.d,self.v=np.linalg.eigh((cov+cov.conj().T)/2)
        self.d=np.maximum(self.d,0)
        self.cross=(cov[-n*n:]@self.v)
        self.cache={}

    def weight(self,snr):
        key=float(snr)
        if key not in self.cache:
            self.cache[key]=(self.cross/(self.d+10**(-key/10)))@self.v.conj().T
        return self.cache[key]

    def predict(self,y,snr):
        shape=(len(y),self.n,self.n)
        yy=y.reshape(len(y),-1)-self.mean
        snr=np.asarray(snr)
        if snr.ndim==0 or np.all(snr==snr.flat[0]):
            return (yy@self.weight(float(snr.flat[0])).T+self.mean[-self.n*self.n:]).reshape(shape).astype(np.complex64)
        if snr.shape!=(len(y),): raise ValueError('SNR must be scalar or one value per window')
        # Avoid solving a different dense system for every noisy reported SNR.
        z=yy@self.v.conj()
        z/=self.d[None,:]+10**(-snr[:,None]/10)
        return (z@self.cross.T+self.mean[-self.n*self.n:]).reshape(shape).astype(np.complex64)


def fit_empirical(h,n,T,shrinkage):
    t=time.perf_counter()
    a=h.reshape(len(h),-1).astype(np.complex128)
    mean=a.mean(0);a-=mean
    cov=a.T@a.conj()/len(a)
    cov=(1-shrinkage)*cov+shrinkage*np.trace(cov).real/len(cov)*np.eye(len(cov))
    return cov,mean,time.perf_counter()-t


def covariance_baselines(c,group,train_h):
    n=group['n'];T=c['T'];comp=components(c,group['family'])
    result={}
    for count in c['cov_samples']:
        count=min(count,len(train_h))
        cov,mean,elapsed=fit_empirical(train_h[:count],n,T,c['covariance_shrinkage'])
        result[f'Empirical_LMMSE_K{count}']=(cov,mean,elapsed)
        # Report a fixed regularized comparator as well, without weakening the
        # unregularized full-training-set reference used for neural comparisons.
        shrunk=.99*cov+.01*np.trace(cov).real/len(cov)*np.eye(len(cov))
        result[f'Empirical_LMMSE_shrink001_K{count}']=(shrunk,mean,elapsed)
    assumed=component_cov(c,n,components(c,'iid')[0])
    result['Jakes_LMMSE_fixed_iid']=(assumed,np.zeros(T*n*n),0.)
    # Knowledge oracle: full mixture second moment, no target conditioning.
    pooled=sum(component_cov(c,n,x) for x in comp)/len(comp)
    result['Oracle_training_distribution_LMMSE']=(pooled,np.zeros(T*n*n),0.)
    return result


def oracle_estimate(c,n,case,y,reported,labels):
    out=np.empty((len(y),n,n),np.complex64)
    for k,comp in enumerate(case['components']):
        ix=np.where(labels==k)[0]
        if len(ix):
            filt=LinearFilter(component_cov(c,n,comp),n,c['T'])
            sr=np.asarray(reported)
            out[ix]=filt.predict(y[ix],sr if sr.ndim==0 else sr[ix])
    return out
