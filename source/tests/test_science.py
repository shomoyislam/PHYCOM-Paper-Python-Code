"""Checks of independent mathematical properties, not copied output constants."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from phycom import simulation as s
from phycom.models import Estimator


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_unitary_pilots_and_noise(self):
        for n in (2,4):
            np.testing.assert_allclose(s.dft(n)@s.dft(n).conj().T,np.eye(n),atol=2e-7)
        h=s.test_channels(12000,2,5,.8,77)
        y=s.observe(h,20,np.random.default_rng(88))
        self.assertAlmostEqual(float(np.mean(np.abs(y-h)**2)),.01,delta=.0002)

    def test_jakes_covariance_and_power(self):
        v=s.validate_channel(count=12000)
        self.assertTrue(v['passed'],v)
        self.assertAlmostEqual(s.temporal_cov(5,.8)[0,1],.8,places=10)

    def test_matched_lmmse_matches_theory(self):
        h=s.test_channels(16000,2,5,.8,17)
        y=s.observe(h,15,np.random.default_rng(18))
        est=s.temporal_lmmse(y,15)
        empirical=s.nmse(h[:,-1],est)[0]['nmse']
        expected=s.temporal_lmmse_theory(5,15)
        self.assertAlmostEqual(empirical,expected,delta=.001)
        self.assertLess(empirical,s.nmse(h[:,-1],y[:,-1])[0]['nmse'])
        np.testing.assert_allclose(est,s.joint_lmmse(y,15,.8,0),rtol=1e-5,atol=1e-6)

    def test_joint_lmmse_satisfies_normal_equations(self):
        n,T,rho=2,3,.7
        y=s.observe(s.test_channels(12,n,T,.8,19,rho),15,np.random.default_rng(20))
        R=s.temporal_cov(T,.8);C=np.kron(s.spatial_cov(n,rho),s.spatial_cov(n,rho).T)
        cov=np.kron(R,C)+10**(-1.5)*np.eye(T*n*n)
        cross=np.concatenate([R[-1,t]*C for t in range(T)],axis=1)
        W=np.linalg.solve(cov,cross.T).T
        direct=(y.reshape(len(y),-1)@W.T).reshape(-1,n,n)
        np.testing.assert_allclose(s.joint_lmmse(y,15,.8,rho),direct,atol=2e-6)

    def test_initial_ls_residual_and_shared_weights(self):
        for n in (2,4):
            a=Estimator('lstm_none',n,42);b=Estimator('lstm_gate',n,42)
            for key,value in a.state_dict().items():
                self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
            x=torch.randn(3,5,2,n,n);r=Estimator('lsres_gate',n,42)
            torch.testing.assert_close(r(x,torch.zeros(3,1)),x[:,-1],rtol=0,atol=0)

    def test_gate_and_backbone_shapes_and_gradients(self):
        for name in ['cnn_gate','raw_lstm_gate','lstm_none','lstm_concat','lstm_film','lstm_gate',
                     'lstm_resgate','gru_none','gru_gate','lsres_none','lsres_gate']:
            model=Estimator(name,4)
            p=model(torch.randn(2,3,2,4,4),torch.zeros(2,1))
            self.assertEqual(tuple(p.shape),(2,2,4,4))
            p.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_block_bootstrap_accounts_for_fading_variability(self):
        # Two populations of channel blocks, many bits per block: bit-binomial
        # uncertainty wrongly vanishes as payload size grows.
        blocks=np.r_[np.zeros(128),np.ones(128)*.4]
        ci=s.bootstrap_mean(blocks,resamples=2000)
        naive=1.96*np.sqrt(.2*.8/(256*10000))
        self.assertGreater((ci['ci95_high']-ci['ci95_low'])/2,10*naive)

    def test_ber_pairing_and_perfect_channel(self):
        h=np.broadcast_to(np.eye(2,dtype=np.complex64),(64,2,2)).copy()
        a=s.ber_blocks(h,h,25,1000,123)
        b=s.ber_blocks(h,h,25,1000,123)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(int(a.sum()),0)
        np.testing.assert_allclose(s.zf_rate_blocks(h,h,10),2*np.log2(1+10/2),rtol=1e-6)

    def test_interrupted_training_resumes_same_cpu_trajectory(self):
        from phycom import training as tr
        from phycom.common import base_config
        cfg=base_config(True);cfg['epochs']=3;cfg['patience']=3
        job=dict(kind='main',n=2,T=3,seed=42)
        train=s.training_data(96,2,3,.8,100)
        val=s.training_data(48,2,3,.8,200)
        with tempfile.TemporaryDirectory() as td:
            full=tr.train(Estimator('lstm_gate',2,42),train,val,cfg,job,Path(td)/'full',torch.device('cpu'),'test')
            writer=tr.save_torch
            def interrupt_after_saved_epoch(path,obj):
                writer(path,obj)
                if Path(path).name=='last.pt' and obj['epoch']==0:raise KeyboardInterrupt()
            with patch.object(tr,'save_torch',interrupt_after_saved_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    tr.train(Estimator('lstm_gate',2,42),train,val,cfg,job,Path(td)/'resume',torch.device('cpu'),'test')
            resumed=tr.train(Estimator('lstm_gate',2,42),train,val,cfg,job,Path(td)/'resume',torch.device('cpu'),'test')
            for name,weight in full.state_dict().items():
                torch.testing.assert_close(weight,resumed.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
