"""Regression tests for scientific pairing, continuation and frozen evidence."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from phycom.common import base_config,stable_seed,model_names,jobs,atomic_json
from phycom.models import Estimator
from phycom import simulation as sim,training as tr,evaluation as ev
from phycom.archive import safe_target


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_default_protocol_has_all_controls(self):
        cfg=base_config()
        self.assertEqual(sum(len(model_names(cfg,j)) for j in jobs(cfg)),129)
        self.assertEqual(cfg['seq_seeds'],cfg['seeds'])
        for stem in ['lstm','mamba','lsres','lstm_wide']:
            self.assertIn(stem+'_none',cfg['stress_models']); self.assertIn(stem+'_gate',cfg['stress_models'])

    def test_wide_control_matches_shared_parameters(self):
        narrow=Estimator('lstm_gate',4,123)
        for suffix in ['none','gate']:
            wide=Estimator('lstm_wide_'+suffix,4,123)
            self.assertEqual(wide.fc1.out_features,64); self.assertEqual(narrow.fc1.out_features,32)
            for key,value in narrow.state_dict().items():
                if key.startswith(('fc1.','fc2.','gate.')): continue
                self.assertTrue(torch.equal(value,wide.state_dict()[key]),key)
        a,b=Estimator('lstm_wide_none',4,123),Estimator('lstm_wide_gate',4,123)
        for key,value in a.state_dict().items(): self.assertTrue(torch.equal(value,b.state_dict()[key]),key)

    def test_fresh_test_set_and_pairing_across_seeds(self):
        cfg=base_config(True); a=dict(kind='main',n=2,T=5,seed=42); b=dict(a,seed=123)
        new=ev.channel_set(cfg,a); paired=ev.channel_set(cfg,b)
        old=sim.test_channels(len(new),2,5,.8,stable_seed('heldout_channels','main',0,2,5))
        np.testing.assert_array_equal(new,paired); self.assertFalse(np.array_equal(new,old))
        self.assertFalse(np.array_equal(ev.observation(new,cfg,a,'main',0,15),ev.observation(new,cfg,a,'main',0,30)))

    def test_sideinfo_mismatch_keeps_channel_observation_fixed(self):
        cfg=base_config(True); job=dict(kind='main',n=2,T=5,seed=42)
        h,y,s=ev.sample_case(cfg,job,'snr_bias',-3,30)
        hh,yy,ss=ev.sample_case(cfg,job,'snr_error_std',2,30)
        np.testing.assert_array_equal(h,hh); np.testing.assert_array_equal(y,yy)
        model=Estimator('lstm_none',2,42)
        np.testing.assert_array_equal(tr.predict(model,y,s,torch.device('cpu')),tr.predict(model,yy,ss,torch.device('cpu')))
        self.assertFalse(np.array_equal(s,ss))

    def test_reported_snr_lmmse_agrees_with_known_snr_at_zero_error(self):
        y=sim.observe(sim.test_channels(64,4,5,.8,90),15,np.random.default_rng(91))
        np.testing.assert_allclose(ev.temporal_with_sideinfo(y,np.full(64,15.)),sim.temporal_lmmse(y,15.),atol=1e-6)

    def test_parent_state_continuation_is_same_cpu_trajectory(self):
        cfg=base_config(True); job=dict(kind='main',n=2,T=3,seed=42)
        data=sim.training_data(96,2,3,.8,100); val=sim.training_data(48,2,3,.8,200)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            full=tr.train(Estimator('lstm_gate',2,42),data,val,cfg,job,root/'full',torch.device('cpu'),'new')
            tr.train(Estimator('lstm_gate',2,42),data,val,dict(cfg,epochs=1),job,root/'parent',torch.device('cpu'),'old')
            last=tr.load_checkpoint(root/'parent'/'last.pt'); best=tr.load_checkpoint(root/'parent'/'best.pt')
            for key in ['origin','best_state_dict','best_epoch','budget_state_dict','budget_val','budget_epoch']: last.pop(key)
            migrated=tr.import_legacy_state(last,best,'new',job,'lstm_gate',dict(kind='continued_v3.1'))
            tr.save_torch(root/'child'/'last.pt',migrated)
            child=tr.train(Estimator('lstm_gate',2,99),data,val,cfg,job,root/'child',torch.device('cpu'),'new')
            for key,value in full.state_dict().items(): torch.testing.assert_close(value,child.state_dict()[key],rtol=0,atol=0)
            ref=tr.load_checkpoint(root/'child'/'budget60.pt')
            for key,value in best['state_dict'].items(): torch.testing.assert_close(value,ref['state_dict'][key],rtol=0,atol=0)

    def test_committed_epoch_repairs_missing_selected_exports(self):
        cfg=base_config(True); cfg['epochs']=1; job=dict(kind='main',n=2,T=3,seed=42)
        data=sim.training_data(32,2,3,.8,100); val=sim.training_data(16,2,3,.8,200)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            with patch.object(tr,'publish_checkpoints',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tr.train(Estimator('lstm_gate',2,42),data,val,cfg,job,path,torch.device('cpu'),'test')
            self.assertTrue((path/'last.pt').exists()); self.assertFalse((path/'best.pt').exists())
            tr.train(Estimator('lstm_gate',2,42),data,val,cfg,job,path,torch.device('cpu'),'test')
            self.assertTrue((path/'budget60.pt').exists()); self.assertTrue((path/'best.pt').exists())

    def test_training_never_evaluates_and_freeze_detects_tampering(self):
        from phycom.worker import run_job
        from run_phycom import freeze_selection
        cfg=base_config(True); cfg.update(arrays=[2],models=['lstm_gate'],seq_lengths=[])
        job=dict(kind='main',n=2,T=5,seed=42)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); atomic_json(root/'run_manifest.json',dict(run_id='test',config=cfg))
            with patch.object(ev,'sample_case',side_effect=AssertionError('test accessed during training')):
                run_job(root,job,torch.device('cpu'),'train')
            freeze_selection(root,cfg,'test')
            path=root/'jobs/main_2x2_T5_seed42/lstm_gate/best.pt'
            ck=tr.load_checkpoint(path); ck['state_dict']['fc2.bias']+=1; tr.save_torch(path,ck)
            with self.assertRaises(ValueError): freeze_selection(root,cfg,'test')

    def test_parent_identity_and_archive_path_guards(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ['../escape','/absolute','good/../../escape','back\\slash']:
                with self.assertRaises(ValueError): safe_target(td,name)
            self.assertTrue(safe_target(td,'jobs/a/last.pt').is_relative_to(Path(td)))
        with self.assertRaises(ValueError):
            tr.import_legacy_state(dict(job={},model_name='bad'),dict(job={},model_name='bad'),'run',{},'lstm_gate',{})

    def test_seed_uncertainty_is_separate_from_conditional_monte_carlo(self):
        from phycom.reporting import paired_comparisons,summarize
        cfg=base_config(); cfg.update(snrs=[30])
        rows=[]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for seed,delta in zip(cfg['seeds'],[-.1,0.,.2]):
                for name,value in [('lstm_none',.5),('lstm_gate',.5+delta)]:
                    path=f'{seed}_{name}.npz'
                    np.savez(root/path,error_energy=np.full(64,value),channel_energy=np.ones(64))
                    rows.append(dict(suite='main',n=2,T=5,seed=seed,model=name,stage='extended',
                        scenario='main',parameter=0.,snr_db=30.,raw_file=path,nmse=value))
            paired=paired_comparisons(root,cfg,rows)
            self.assertEqual(len(paired),1)
            r=paired[0]
            self.assertGreater(r['nmse_diff_seed_std'],.1)
            self.assertLess(r['nmse_seed_t_ci95_low'],0.)
            self.assertGreater(r['nmse_seed_t_ci95_high'],0.)
            self.assertLess(r['nmse_mc_conditional_ci95_high']-r['nmse_mc_conditional_ci95_low'],1e-12)
            self.assertTrue(all(r['n_training_seeds']==3 for r in summarize(rows)))

    def test_optimizer_resume_preflight_cpu(self):
        from phycom.runtime_checks import optimizer_resume_probe
        for name in ['lstm_gate','lstm_wide_gate']:
            self.assertTrue(optimizer_resume_probe(torch.device('cpu'),name)['passed'])


if __name__=='__main__': unittest.main()
