"""Tiny D2 CPU/GPU checks only. NEVER evolves any D7 state."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..simple_update import BlockState
from . import workflow as base
from . import workflow_random_seeds as batch
from .run_D7_random10 import settings


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


class RandomBatchChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(10)

    def config(self,seeds=(8101,8102),converge=True):
        cfg=settings()
        cfg.update(D=2,seeds=list(seeds),measurement_device='cpu',warmup=[],
                   fine_stage=dict(dt=.03,min_tau=.06,max_tau=.06),max_fine_chunks=1,
                   monitor_tau=.03,monitor_min_steps=1,checkpoint_steps=1,stable_checks=1,
                   chi=3,ctm_min=8,ctm_max=10,ctm_tolerance=1e6,
                   lambda_drift_tolerance=100. if converge else 0.,
                   rho_drift_tolerance=100. if converge else 0.)
        return cfg

    def test_production_defaults_and_validation(self):
        cfg=settings()
        batch.validate_batch(cfg)
        self.assertEqual(cfg['D'],7)
        self.assertEqual(len(set(cfg['seeds'])),10)
        self.assertEqual((cfg['field'],cfg['chi'],cfg['ctm_max'],cfg['measurement_device']),(.4,40,50,'cuda:1'))
        with self.assertRaises(ValueError):
            batch.validate_batch(cfg | dict(seeds=[7101,7101]))
        with self.assertRaises(ValueError):
            batch.validate_batch(cfg | dict(seeds=[]))

    def test_screen_defaults_exactly_9000_steps(self):
        from .run_D7_random10_screen import settings as screen_settings
        cfg=screen_settings()
        batch.validate_batch(cfg)
        self.assertEqual([round(s['max_tau']/s['dt']) for s in cfg['fixed_stages']],
                         [1333,2000,1667,2000,2000])
        self.assertEqual(sum(round(s['max_tau']/s['dt']) for s in cfg['fixed_stages']),9000)

    def screen_config(self,old=None):
        cfg=self.config(seeds=(8101,),converge=False)
        cfg.update(evolution_mode='fixed_screen',reuse_long_run=str(old) if old else '',
                   fixed_stages=[dict(dt=.03,min_tau=.06,max_tau=.06)])
        return cfg

    def test_fixed_screen_stops_at_budget_and_restart_does_not_add_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'screen'
            cfg=self.screen_config()
            self.assertTrue(batch.run_random_batch(cfg,out))
            row=csv_rows(out/'summary.csv')[0]
            self.assertEqual(row['status'],'screened')
            self.assertEqual(row['su_proxy_converged'],'False')
            self.assertEqual(row['screening_added_steps'],'2')
            with (patch.object(base,'sweep',side_effect=AssertionError('screen repeated SU')),
                  patch.object(batch,'measure_snapshot',side_effect=AssertionError('screen remeasured'))):
                self.assertTrue(batch.run_random_batch(cfg,out))

    def test_screen_reuses_long_candidate_without_more_SU(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old'
            oldcfg=self.config(seeds=(8101,))
            self.assertTrue(batch.run_random_batch(oldcfg,old))
            before=base.sha256(old/'seed8101/D2/checkpoint.pt')
            out=Path(tmp)/'screen'
            with patch.object(base,'sweep',side_effect=AssertionError('import evolved existing candidate')):
                self.assertTrue(batch.run_random_batch(self.screen_config(old),out))
            row=csv_rows(out/'summary.csv')[0]
            self.assertEqual((row['imported_steps'],row['screening_added_steps']),('2','0'))
            self.assertEqual(before,base.sha256(old/'seed8101/D2/checkpoint.pt'))

    def test_screen_recovers_saved_candidate_after_SU_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old'
            oldcfg=self.config(seeds=(8101,),converge=False)
            oldcfg['fine_stage']=dict(dt=.03,min_tau=.12,max_tau=.12)
            original=base.sweep
            count=0
            def fail_after_checkpoint(*args,**kwargs):
                nonlocal count
                count+=1
                if count==3:
                    raise RuntimeError('simulated SVD failure after two saved steps')
                return original(*args,**kwargs)
            with patch.object(base,'sweep',side_effect=fail_after_checkpoint):
                self.assertFalse(batch.run_random_batch(oldcfg,old))
            out=Path(tmp)/'screen'
            with patch.object(base,'sweep',side_effect=AssertionError('failed seed should only be measured')):
                self.assertTrue(batch.run_random_batch(self.screen_config(old),out))
            row=csv_rows(out/'summary.csv')[0]
            self.assertEqual((row['imported_steps'],row['screening_added_steps']),('2','0'))
            self.assertEqual(row['su_proxy_converged'],'False')

    def test_independent_initial_states_complete_outputs_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'batch'
            cfg=self.config()
            self.assertTrue(batch.run_random_batch(cfg,out))
            a=BlockState.load(out/'seed8101/random_source.pt')
            b=BlockState.load(out/'seed8102/random_source.pt')
            self.assertFalse(torch.equal(a.A[0],b.A[0]))
            self.assertEqual((a.seed,b.seed),(8101,8102))
            rows=csv_rows(out/'summary.csv')
            self.assertEqual([r['status'] for r in rows],['complete','complete'])
            self.assertEqual(len(csv_rows(out/'site_magnetization.csv')),54)
            self.assertEqual(len(csv_rows(out/'bond_energies.csv')),108)
            self.assertEqual(len(csv_rows(out/'triangle_observables.csv')),36)
            self.assertEqual({r['seed'] for r in csv_rows(out/'site_magnetization.csv')},{'8101','8102'})
            first=[base.sha256(out/f'seed{s}/D2/final.pt') for s in cfg['seeds']]
            with (patch.object(BlockState,'random',side_effect=AssertionError('completed source randomized')),
                  patch.object(batch,'measure_snapshot',side_effect=AssertionError('completed seed remeasured'))):
                self.assertTrue(batch.run_random_batch(cfg,out))
            self.assertEqual(first,[base.sha256(out/f'seed{s}/D2/final.pt') for s in cfg['seeds']])
            with self.assertRaises(ValueError):
                batch.run_random_batch(cfg | dict(field=.38),out)

    def test_SU_cap_measures_all_seeds_and_keeps_previous_snapshots_on_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'batch'
            cfg=self.config(converge=False)
            self.assertFalse(batch.run_random_batch(cfg,out))
            self.assertEqual([r['status'] for r in csv_rows(out/'summary.csv')],['needs_more_SU']*2)
            old=[base.sha256(out/f'seed{s}/D2/measurements/attempt0001/state.pt') for s in cfg['seeds']]
            with patch.object(BlockState,'random',side_effect=AssertionError('retry randomized')):
                self.assertFalse(batch.run_random_batch(cfg,out))
            self.assertEqual(old,[base.sha256(out/f'seed{s}/D2/measurements/attempt0001/state.pt') for s in cfg['seeds']])
            self.assertEqual(len(csv_rows(out/'all_measurements.csv')),4)
            for seed in cfg['seeds']:
                progress=torch.load(out/f'seed{seed}/D2/checkpoint.pt',weights_only=False)['metadata']['progress']
                self.assertEqual(progress['d_steps'],4)
                self.assertEqual(len(progress['stage_reports']),2)
                self.assertTrue((out/f'seed{seed}/D2/measurements/attempt0002/observables.json').exists())

    def test_measurement_interruption_retries_exact_snapshot_without_SU(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'batch'
            cfg=self.config(seeds=(8101,))
            with patch.object(batch,'measure_snapshot',side_effect=KeyboardInterrupt('interrupted measurement')):
                with self.assertRaises(KeyboardInterrupt):
                    batch.run_random_batch(cfg,out)
            pending=out/'seed8101/pending_measurement.json'
            self.assertTrue(pending.exists())
            snapshot=out/'seed8101/D2/measurements/attempt0001/state.pt'
            digest=base.sha256(snapshot)
            with (patch.object(base,'sweep',side_effect=AssertionError('measurement retry evolved state')),
                  patch.object(BlockState,'random',side_effect=AssertionError('measurement retry randomized'))):
                self.assertTrue(batch.run_random_batch(cfg,out))
            self.assertEqual(digest,base.sha256(snapshot))
            self.assertFalse(pending.exists())
            self.assertEqual(len(csv_rows(out/'all_measurements.csv')),1)

    def test_failed_seed_does_not_block_others_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'batch'
            cfg=self.config()
            original=batch.evolve_to_convergence
            def fail_first(config,*args):
                if config['seed']==8101:
                    raise RuntimeError('simulated seed-specific failure')
                return original(config,*args)
            with patch.object(batch,'evolve_to_convergence',side_effect=fail_first):
                self.assertFalse(batch.run_random_batch(cfg,out))
            rows=csv_rows(out/'summary.csv')
            self.assertEqual([r['status'] for r in rows],['failed','complete'])
            other=base.sha256(out/'seed8102/D2/final.pt')
            with patch.object(BlockState,'random',side_effect=AssertionError('existing sources randomized')):
                self.assertTrue(batch.run_random_batch(cfg,out))
            self.assertEqual(other,base.sha256(out/'seed8102/D2/final.pt'))
            self.assertFalse((out/'seed8101/error.json').exists())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable for local tiny validation')
    def test_tiny_gpu_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.config(seeds=(8101,))
            cpu=Path(tmp)/'cpu'
            gpu=Path(tmp)/'gpu'
            self.assertTrue(batch.run_random_batch(cfg,cpu))
            self.assertTrue(batch.run_random_batch(cfg | dict(measurement_device='cuda:0'),gpu))
            a,b=csv_rows(cpu/'summary.csv')[0],csv_rows(gpu/'summary.csv')[0]
            self.assertEqual(b['measurement_device'],'cuda:0')
            self.assertEqual(b['su_device'],'cpu')
            self.assertAlmostEqual(float(a['energy_total']),float(b['energy_total']),delta=1e-6)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RandomBatchChecks))
    out=Path(__file__).resolve().parents[1]/'results/server_validation'
    out.mkdir(parents=True,exist_ok=True)
    (out/'random_batch_tests.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        success=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
