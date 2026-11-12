"""Small CPU tests only; never starts a D5-D10 numerical job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..simple_update import BlockState
from . import workflow as base
from . import workflow_from_state as seeded
from .run_D5_D10_from_local_D5 import settings


class SeededChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(10)

    def config(self,source):
        cfg = settings()
        cfg.update(dims=[2,3],initial_state_file=str(source),initial_state_sha256=base.sha256(source),
                   measurement_device='cpu',chi=3,ctm_min=8,ctm_max=10,ctm_tolerance=1e6,
                   first_D_warmup=[],next_D_warmup=[],max_fine_chunks=1,
                   fine_stage=dict(dt=.03,min_tau=.06,max_tau=.06),
                   monitor_min_steps=1,monitor_tau=.03,stable_checks=1,checkpoint_steps=1,
                   lambda_drift_tolerance=0.,rho_drift_tolerance=0.)
        return cfg

    def test_real_D5_source_bundle(self):
        cfg = settings()
        seeded.validate_config(cfg)
        path,digest = seeded.validate_source(cfg)
        manifest = json.loads(path.with_suffix('.json').read_text())
        self.assertEqual(digest,manifest['verified_measurement']['input_sha256'])
        self.assertEqual(manifest['verified_measurement']['D'],5)
        self.assertLess(manifest['verified_measurement']['energy_exchange'],-.4166)

    def test_unconverged_stops_and_resume_continues_fine_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.pt'
            state = BlockState.random(2,seed=1701,c3=True)
            state.save(source)
            cfg = self.config(source)
            out = Path(tmp)/'run'
            with patch.object(BlockState,'random',side_effect=AssertionError('random initializer used')):
                self.assertFalse(seeded.run_seeded_chain(cfg,out))
                first = torch.load(out/'D2/checkpoint.pt',weights_only=False)['metadata']['progress']
                self.assertEqual(first['d_steps'],2)
                self.assertFalse((out/'D3').exists())
                self.assertFalse((out/'D2/observables.json').exists())
                self.assertFalse(seeded.run_seeded_chain(cfg,out))
                second = torch.load(out/'D2/checkpoint.pt',weights_only=False)['metadata']['progress']
                self.assertEqual(second['d_steps'],4)
                self.assertEqual(second['stage'],2)
                self.assertEqual(len(second['stage_reports']),2)
            initial = BlockState.load(out/'D2/initial.pt')
            for a,b in zip(state.A+state.B,initial.A+initial.B):
                self.assertTrue(torch.equal(a,b))

    def test_actual_small_chain_full_measurements_lineage_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.pt'
            BlockState.random(2,seed=1701,c3=True).save(source)
            cfg = self.config(source)
            cfg.update(lambda_drift_tolerance=10.,rho_drift_tolerance=10.)
            out = Path(tmp)/'run'
            with patch.object(BlockState,'random',side_effect=AssertionError('random initializer used')):
                self.assertTrue(seeded.run_seeded_chain(cfg,out))
            parent = json.loads((out/'D2/completed.json').read_text())
            child = json.loads((out/'D3/completed.json').read_text())
            self.assertTrue(parent['ready_for_next_D'] and child['ready_for_next_D'])
            self.assertEqual(child['parent_sha256'],base.sha256(out/'D2/final.pt'))
            initial = BlockState.load(out/'D3/initial.pt')
            before = BlockState.load(out/'D2/final.pt')
            for a,b in zip(before.A+before.B,initial.A+initial.B):
                self.assertTrue(torch.equal(a,b[:,:2,:2,:2]))
            obs = json.loads((out/'D3/observables.json').read_text())
            self.assertEqual((len(obs['sites']),len(obs['bonds']),len(obs['triangles'])),(27,54,18))
            with patch.object(seeded,'measure_final',side_effect=AssertionError('completed D remeasured')):
                self.assertTrue(seeded.run_seeded_chain(cfg,out))
            cfg2 = cfg | dict(measurement_device='cuda:1')
            self.assertEqual(seeded.fingerprint(cfg),seeded.fingerprint(cfg2))

    def test_mid_fine_interruption_resumes_without_repeating_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.pt'
            BlockState.random(2,seed=1701,c3=True).save(source)
            cfg = self.config(source)
            cfg.update(first_D_warmup=[dict(dt=.03,min_tau=.03,max_tau=.03)],
                       fine_stage=dict(dt=.03,min_tau=.12,max_tau=.12))
            out = Path(tmp)/'interrupted'
            original = base.sweep
            count = 0
            def interrupt(*args,**kwargs):
                nonlocal count
                count += 1
                if count == 3:
                    raise KeyboardInterrupt('simulated mid-fine interruption')
                return original(*args,**kwargs)
            with patch.object(base,'sweep',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    seeded.run_seeded_chain(cfg,out)
            progress = torch.load(out/'D2/checkpoint.pt',weights_only=False)['metadata']['progress']
            self.assertEqual((progress['stage'],progress['stage_steps'],progress['d_steps']),(1,1,2))
            self.assertFalse(seeded.run_seeded_chain(cfg,out))
            progress = torch.load(out/'D2/checkpoint.pt',weights_only=False)['metadata']['progress']
            self.assertEqual(progress['d_steps'],5)
            uninterrupted = Path(tmp)/'uninterrupted'
            self.assertFalse(seeded.run_seeded_chain(cfg,uninterrupted))
            a,b = BlockState.load(out/'D2/final.pt'),BlockState.load(uninterrupted/'D2/final.pt')
            for x,y in zip(a.A+a.B,b.A+b.B):
                self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=1e-11))

    def test_ctm_failure_blocks_next_D_and_retry_only_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.pt'
            BlockState.random(2,seed=1701,c3=True).save(source)
            cfg = self.config(source)
            cfg.update(lambda_drift_tolerance=10.,rho_drift_tolerance=10.)
            out = Path(tmp)/'run'
            original = seeded.measure_final
            def unconverged(*args):
                obs = original(*args)
                obs['ctm_converged'] = False
                base.atomic_json(args[2]/'observables.json',obs)
                return obs
            with patch.object(seeded,'measure_final',side_effect=unconverged):
                self.assertFalse(seeded.run_seeded_chain(cfg,out))
            self.assertFalse((out/'D3').exists())
            before = base.sha256(out/'D2/final.pt')
            with patch.object(base,'sweep',side_effect=AssertionError('SU rerun during measurement retry')):
                # Keep the second D out of this retry test, by interrupting
                # when the next dimension would first be initialized.
                original_evolve = seeded.evolve_to_convergence
                def only_D2(config,folder,*args):
                    if folder.name=='D3':
                        raise KeyboardInterrupt('stop after verified D2 retry')
                    return original_evolve(config,folder,*args)
                with patch.object(seeded,'evolve_to_convergence',side_effect=only_D2):
                    with self.assertRaises(KeyboardInterrupt):
                        seeded.run_seeded_chain(cfg,out)
            self.assertEqual(before,base.sha256(out/'D2/final.pt'))
            self.assertTrue(json.loads((out/'D2/completed.json').read_text())['ready_for_next_D'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SeededChecks))
    target = Path(__file__).resolve().parents[1]/'results/server_validation'
    target.mkdir(parents=True,exist_ok=True)
    (target/'seeded_chain_tests.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),success=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
