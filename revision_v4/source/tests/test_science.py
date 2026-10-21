"""CPU checks of the scientific implementation; no publication-result assertions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from phycom_v4.protocol import config,spec,resolved,models,groups,param_count_formula,matched_raw_width
from phycom_v4.models import Estimator,parameter_count
from phycom_v4.data import channels,components,component_cov,cases,make_case,training_arrays
from phycom_v4.linear import LinearFilter,fit_empirical
from phycom_v4.training import setup,train_one,load_checkpoint
from phycom_v4.evaluation import result_arrays,spectral_info
from phycom_v4.common import seed,read_json
from legacy_v32.models import Estimator as Legacy
from legacy_v32.simulation import observe,temporal_lmmse,temporal_lmmse_theory


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup(config(True))

    def test_original_architectures_exact_initialization_and_prediction(self):
        for n in (2,4):
            x=torch.randn(7,5,2,n,n);snr=torch.randn(7,1)
            for name,s in [('lstm_none',spec()),('lstm_gate',spec(cond='gate')),
                ('lstm_film',spec(cond='film')),('lstm_resgate',spec(cond='resgate')),
                ('lstm_concat',spec(cond='concat')),('raw_lstm_gate',spec('raw',0,cond='gate')),
                ('lstm_wide_gate',spec(head=64,cond='gate')),('lsres_gate',spec(cond='gate',residual=True))]:
                old=Legacy(name,n,123).eval();new=Estimator(resolved(s,n),n,123).eval()
                self.assertEqual(set(old.state_dict()),set(new.state_dict()))
                for key in old.state_dict():
                    self.assertTrue(torch.equal(old.state_dict()[key],new.state_dict()[key]),key)
                with torch.no_grad():
                    self.assertTrue(torch.equal(old(x,snr),new(x,snr)),name)

    def test_identity_controls_start_at_same_function_with_trainable_condition(self):
        for n in (2,4):
            x=torch.randn(8,5,2,n,n);snr=torch.linspace(-1.5,1.5,8)[:,None]
            base=Estimator(resolved(spec(),n),n,123).eval()
            for cond in ('gate_identity','resgate_identity','film_identity','concat_identity'):
                m=Estimator(resolved(spec(cond=cond),n),n,123).eval()
                self.assertTrue(torch.allclose(base(x,snr),m(x,snr),atol=1e-7,rtol=0),cond)
                m(x,snr).square().mean().backward()
                grad=(m.gate.weight.grad if hasattr(m,'gate') else m.scale.weight.grad if hasattr(m,'scale')
                      else m.fc1.weight.grad[:,-1])
                self.assertGreater(float(grad.abs().sum()),0)

    def test_parameter_counts_and_matching(self):
        c=config(True)
        for g in groups(c):
            if g['seed']!=c['seeds'][0]: continue
            for s in models(c,g):
                self.assertEqual(parameter_count(Estimator(s,g['n'],42)),param_count_formula(s,g['n']))
        for p,h in [(32,32),(128,128)]:
            w=matched_raw_width(4,p,h)
            target=param_count_formula(spec(p=p,h=h),4)
            self.assertLess(abs(param_count_formula(spec('raw',0,w),4)-target)/target,.035)

    def test_physical_covariance_matches_generated_complex_channels(self):
        c=config(True);n=3;comp=components(c,'spatial')
        for gaussian in (False,True):
            h,_=channels(c,n,16000,998,comp,gaussian)
            z=h.reshape(len(h),-1).astype(np.complex128)
            empirical=z.T@z.conj()/len(z);target=component_cov(c,n,comp[0])
            self.assertLess(np.linalg.norm(empirical-target)/np.linalg.norm(target),.055)
            self.assertLess(abs(float(np.mean(np.abs(h)**2))-1),.04)

    def test_linear_filter_against_direct_complex_solve(self):
        c=config(True);n=3;T=c['T'];C=component_cov(c,n,components(c,'spatial')[0])
        rng=np.random.default_rng(4)
        y=(rng.normal(size=(12,T,n,n))+1j*rng.normal(size=(12,T,n,n))).astype(np.complex64)
        mu=(rng.normal(size=T*n*n)+1j*rng.normal(size=T*n*n))*.1
        f=LinearFilter(C,n,T,mu);snrs=np.linspace(0,30,len(y))
        actual=f.predict(y,snrs)
        for i,snr in enumerate(snrs):
            W=np.linalg.solve((C+10**(-snr/10)*np.eye(len(C))).T,C[-n*n:].T).T
            expected=W@(y[i].reshape(-1)-mu)+mu[-n*n:]
            np.testing.assert_allclose(actual[i].reshape(-1),expected,atol=3e-6,rtol=3e-6)
        np.testing.assert_allclose(f.predict(y,15),f.predict(y,np.full(len(y),15)),atol=0,rtol=0)

    def test_jakes_oracle_and_theoretical_risk(self):
        c=config(True);n=2;h,_=channels(c,n,24000,100,components(c,'iid'))
        for snr in (0,15,30):
            y=observe(h,snr,np.random.default_rng(444))
            f=LinearFilter(component_cov(c,n,components(c,'iid')[0]),n,c['T'])
            pred=f.predict(y,snr)
            np.testing.assert_allclose(pred,temporal_lmmse(y,snr),atol=3e-6,rtol=3e-6)
            measured=np.sum(np.abs(pred-h[:,-1])**2)/np.sum(np.abs(h[:,-1])**2)
            theory=temporal_lmmse_theory(c['T'],snr,c['alpha'])
            self.assertLess(abs(measured/theory-1),.035)

    def test_reporting_errors_share_observations_and_clip_masks(self):
        c=config(True);cc=cases(c,'iid')
        get=lambda name:next(x for x in cc if x['name']==name)
        base=make_case(c,4,get('report_reference_snr30'))
        biased=make_case(c,4,get('bias_snr30_b3_clip0'))
        clipped=make_case(c,4,get('bias_snr30_b3_clip1'))
        for other in (biased,clipped):
            np.testing.assert_array_equal(base[0],other[0]);np.testing.assert_array_equal(base[1],other[1])
        self.assertTrue(biased[4].all());self.assertTrue(clipped[4].all())
        self.assertTrue((clipped[2]==30).all());self.assertTrue((biased[2]==33).all())
        g=dict(family='iid',n=4,seed=c['seeds'][0]);a=training_arrays(c,g,'train')[1]
        b=training_arrays(c,g,'validation')[1]
        self.assertFalse(np.array_equal(a[:len(b)],b))

    def test_residual_zero_and_exact_paired_risk_identity(self):
        c=config(True);g=dict(family='iid',n=2,seed=1)
        m=Estimator(resolved(spec(residual=True),2),2,5).eval()
        x=torch.randn(8,5,2,2,2);s=torch.randn(8,1)
        self.assertTrue(torch.equal(m(x,s),x[:,-1]))
        case=cases(c,'iid')[1];h,y,r,lab,mask=make_case(c,2,case)
        pred=.7*y[:,-1]
        ar=result_arrays(c,g,case,h,y,pred,r,mask,'test')
        ls=result_arrays(c,g,case,h,y,y[:,-1],r,mask,'test')
        np.testing.assert_allclose(ar['errors']-ls['errors'],ar['correction_energy']+ar['correction_cross'],atol=1e-12)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        c=config(True);c.update(epochs=3,patience=5)
        g=dict(family='iid',n=2,seed=123);s=resolved(spec(cond='gate',residual=True),2)
        tr=tuple(torch.from_numpy(x) for x in training_arrays(c,g,'train')[0])
        va=tuple(torch.from_numpy(x) for x in training_arrays(c,g,'validation')[0])
        with tempfile.TemporaryDirectory() as td:
            full=Path(td)/'full';res=Path(td)/'res';dev=torch.device('cpu')
            train_one(c,g,s,tr,va,full,dev,'x')
            self.assertIsNone(train_one(c,g,s,tr,va,res,dev,'x',max_new_epochs=1))
            train_one(c,g,s,tr,va,res,dev,'x')
            aa=load_checkpoint(full/'best.pt');bb=load_checkpoint(res/'best.pt')
            self.assertEqual(aa['best_val'],bb['best_val'])
            self.assertEqual(aa['validation_residual_scale'],bb['validation_residual_scale'])
            for k in aa['state_dict']: self.assertTrue(torch.equal(aa['state_dict'][k],bb['state_dict'][k]),k)
            with self.assertRaises(RuntimeError): train_one(c,g,s,tr,va,res,dev,'different')

    def test_spectra_exposes_bottleneck_nullity(self):
        import json
        a=np.concatenate([np.eye(32),np.zeros((32,96))],axis=1)
        d=spectral_info(a)
        self.assertEqual(d['numerical_rank'],32);self.assertEqual(d['domain_nullity'],96)
        self.assertAlmostEqual(d['effective_rank_variance_entropy'],32)
        self.assertEqual(json.loads(json.dumps(d))['domain_nullity'],96)


if __name__=='__main__': unittest.main(verbosity=2)
