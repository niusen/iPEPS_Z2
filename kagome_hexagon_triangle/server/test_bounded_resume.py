"""Small D=1/2 tests only. Never load or evolve a real server D9 state."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import workflow as base
from . import workflow_bounded_resume as job
from .run_D9_D15_bounded import settings
from .workflow_from_state import fingerprint as old_fingerprint
from .workflow_random_seeds import measure_snapshot
from ..simple_update import BlockState


def mock_measure(cfg,state,folder,device,state_hash,digest,su_ok):
    obs=dict(D=state.D,field=state.field,chi=cfg['chi'],energy_exchange=-.1,
             energy_total=-.12,magnetization_normalized=.1,ctm_converged=True,
             measurement_state_sha256=state_hash,settings_hash=digest,
             sites=[],bonds=[],triangles=[],schmidt_spectra=[])
    base.atomic_json(folder/'observables.json',obs)
    return obs


class SmallResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name);self.old=self.root/'old';self.old.mkdir()
        self.out=self.root/'new'
        torch.set_num_threads(1)

    def tearDown(self):self.tmp.cleanup()

    def fixture(self,source_D=1,dims=None):
        old=dict(field=.38,seed=1701,c3=True,second_order=True,dims=list(range(1,source_D+1)),
                 initial_state_sha256='small-fixture-original',measurement_device='cpu')
        base.atomic_json(self.old/'settings.json',old);digest=old_fingerprint(old)
        with base.writer_lock(self.old):pass
        for D in range(1,source_D+1):
            folder=self.old/f'D{D}';folder.mkdir()
            state=BlockState.random(D,.38,1701,c3=True)
            p=base.initialise_progress();p.update(stage=1,d_steps=7,elapsed_su_seconds=1.)
            p['stage_reports']=[dict(proxy_converged=False,tau=.007)]
            state.steps=7;base.save_checkpoint(folder/'final.pt',state,digest,p,{})
            if D<source_D:
                mock_measure(dict(chi=2),state,folder,None,base.sha256(folder/'final.pt'),digest,False)
                base.atomic_json(folder/'completed.json',dict(D=D,ready_for_next_D=True,
                    settings_hash=digest,state_sha256=base.sha256(folder/'final.pt'),su_proxy_converged=False))
            else:
                p['d_steps']=8;state.steps=8
                base.save_checkpoint(folder/'checkpoint.pt',state,digest,p,{})
        cfg=settings()|dict(source_D=source_D,dims=dims or [source_D],old_output=str(self.old),
            expected_initial_sha256=old['initial_state_sha256'],threads=1,measurement_device='cpu',
            source_stages=[dict(dt=.001,min_tau=.002,max_tau=.002)],
            stages=[dict(dt=.003,min_tau=.006,max_tau=.006),dict(dt=.001,min_tau=.002,max_tau=.002)],
            stable_checks=3,monitor_tau=.001,monitor_min_steps=1,checkpoint_steps=1,
            lambda_drift_tolerance=1e-30,rho_drift_tolerance=1e-30,
            chi=2,ctm_min=8,ctm_max=10,ctm_tolerance=1e-6,max_corner_cache_mb=1)
        return cfg

    def old_hashes(self):
        return {p.relative_to(self.old).as_posix():base.sha256(p) for p in self.old.rglob('*') if p.is_file()}

    def test_real_small_chain_and_restart(self):
        cfg=self.fixture(dims=[1,2]);before=self.old_hashes();calls=[]
        def measured(*args):
            state=args[1];calls.append((state.D,state.steps))
            return measure_snapshot(*args)
        with patch.object(BlockState,'random',side_effect=AssertionError('random forbidden')):
            with patch.object(job,'measure_snapshot',side_effect=measured):job.run(cfg,self.out)
        self.assertEqual(calls,[(1,2),(2,4)])
        self.assertEqual(before,self.old_hashes())
        m=json.loads((self.out/'source_manifest.json').read_text())
        self.assertEqual(m['inherited_steps'],8)
        self.assertEqual(Path(m['source_file']).name,'checkpoint.pt')
        self.assertEqual(base.sha256(self.out/'source_snapshot.pt'),m['source_sha256'])
        old=BlockState.load(self.out/'source_snapshot.pt');initial=BlockState.load(self.out/'D1/initial.pt')
        for a,b in zip(old.A+old.B,initial.A+initial.B):torch.testing.assert_close(a,b,rtol=0,atol=0)
        final1=BlockState.load(self.out/'D1/final.pt');initial2=BlockState.load(self.out/'D2/initial.pt')
        padded=base.expand_state(final1,2)
        for a,b in zip(padded.A+padded.B,initial2.A+initial2.B):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for D,added in ((1,2),(2,4)):
            done=json.loads((self.out/f'D{D}/completed.json').read_text())
            obs=json.loads((self.out/f'D{D}/observables.json').read_text())
            self.assertEqual(done['added_steps_this_job'],added)
            self.assertFalse(done['su_proxy_converged'])
            self.assertEqual(len(obs['sites']),27);self.assertEqual(len(obs['bonds']),54)
            self.assertTrue(math.isfinite(obs['energy_total']))
        final_hashes=[base.sha256(self.out/f'D{D}/final.pt') for D in (1,2)]
        with patch.object(base,'sweep',side_effect=AssertionError('no repeated SU')):
            with patch.object(job,'measure_snapshot',side_effect=AssertionError('no repeated measurement')):
                job.run(cfg,self.out)
        self.assertEqual(final_hashes,[base.sha256(self.out/f'D{D}/final.pt') for D in (1,2)])
        changed=copy.deepcopy(cfg);changed['lambda_drift_tolerance']=1e-4
        with self.assertRaises(ValueError):job.preflight(changed,self.out)

    def test_su_interruption_uses_new_checkpoint(self):
        cfg=self.fixture();real=base.sweep;calls=0
        def interrupted(*args):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt('synthetic interruption')
            return real(*args)
        with patch.object(base,'sweep',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):job.run(cfg,self.out)
        p=torch.load(self.out/'D1/checkpoint.pt',weights_only=False)['metadata']['progress']
        self.assertEqual(p['d_steps'],1)
        with patch.object(base,'sweep',wraps=real) as count:
            with patch.object(job,'measure_snapshot',side_effect=mock_measure):job.run(cfg,self.out)
            self.assertEqual(count.call_count,1)
        self.assertEqual(json.loads((self.out/'D1/completed.json').read_text())['added_steps_this_job'],2)

    def test_measurement_interruption_preserves_final_hash(self):
        cfg=self.fixture()
        with patch.object(job,'measure_snapshot',side_effect=RuntimeError('measurement interrupted')):
            with self.assertRaises(RuntimeError):job.run(cfg,self.out)
        digest=base.sha256(self.out/'D1/final.pt')
        with patch.object(base,'sweep',side_effect=AssertionError('SU already finished')):
            with patch.object(job,'measure_snapshot',side_effect=mock_measure):job.run(cfg,self.out)
        self.assertEqual(digest,base.sha256(self.out/'D1/final.pt'))

    def test_archive_and_source_read_lock(self):
        cfg=self.fixture(source_D=2);before=self.old_hashes()
        with base.writer_lock(self.old):
            with self.assertRaises(RuntimeError):job.preflight(cfg,self.out)
        before=self.old_hashes()
        with patch.object(job,'measure_snapshot',side_effect=mock_measure):job.run(cfg,self.out)
        self.assertEqual(before,self.old_hashes())
        self.assertEqual([r['D'] for r in job.collect(self.out)],[1,2])
        manifest=json.loads((self.out/'source_manifest.json').read_text())
        self.assertEqual(manifest['prior_dims'],[1])
        with (self.out/'source_snapshot.pt').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):job.preflight(cfg,self.out)

    def test_production_caps_and_invalid_config(self):
        cfg=settings()
        self.assertEqual(cfg['dims'],list(range(9,16)))
        cap=lambda stages:sum(math.ceil(s['max_tau']/s['dt']-1e-10) for s in stages)
        self.assertEqual(cap(cfg['source_stages']),2000);self.assertEqual(cap(cfg['stages']),9000)
        self.assertEqual(cfg['measurement_device'],'cuda:1');self.assertEqual(cfg['ctm_max'],50)
        self.assertEqual(cfg['lambda_drift_tolerance'],1e-5)
        cfg=self.fixture()
        with self.assertRaises(ValueError):job.validate(cfg,self.old/'nested')
        cfg['dims']=[]
        with self.assertRaises(ValueError):job.validate(cfg,self.out)


if __name__=='__main__':unittest.main(verbosity=2)
