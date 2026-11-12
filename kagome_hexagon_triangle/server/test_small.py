"""Only D1/D2/D3 checks. Never launches the server D4--D10 job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..simple_update import BlockState, sweep, embed, SZ
from ..honeycomb_ctm import HoneycombEnvironment, layers
from ..observables import measure
from .observables import measure_complete
from .run_D4_D10 import settings
from .workflow import expand_state, run_chain, sha256, writer_lock
from .device_support import measurement_state, initial_environment, resolve_measurement_device, immutable_settings


class SmallChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(10)
        torch.set_grad_enabled(False)

    def test_exact_embedding_and_new_rank(self):
        old = BlockState.random(2, seed=2501, c3=True)
        new = expand_state(old, 3)
        for kind in ('A','B'):
            for c in range(3):
                small, large = old.dressed(kind,c), new.dressed(kind,c)
                source_group, target_group = (old.A,new.A) if kind == 'A' else (old.B,new.B)
                self.assertTrue(torch.equal(source_group[c], target_group[c][:,:2,:2,:2]))
                self.assertTrue(torch.allclose(small, large[:, :2, :2, :2], atol=1e-15, rtol=1e-14))
                self.assertAlmostEqual(float(large.square().sum()), float(small.square().sum()), places=14)
        self.assertTrue(all(w[-1] == 0 for row in new.weights for w in row))
        for _ in range(3):
            sweep(new, .03)
        self.assertTrue(all(float(w[-1]) > 1e-9 for row in new.weights for w in row))
        self.assertTrue(all(torch.isfinite(t).all() for t in new.A+new.B))

    def test_complete_polarized_observables(self):
        s = BlockState.random(1)
        for t in s.A+s.B:
            t.zero_()
            t[0,0,0,0] = 1.
        env = HoneycombEnvironment.initial(1).run(layers(s), 2, 20, 1e-6)
        obs = measure_complete(s, env)
        self.assertEqual((len(obs['sites']),len(obs['bonds']),len(obs['triangles'])), (27,54,18))
        self.assertAlmostEqual(obs['energy_exchange'], .5, places=13)
        self.assertAlmostEqual(obs['energy_total'], .31, places=13)
        for bond in obs['bonds']:
            self.assertAlmostEqual(bond['szz'], .25, places=13)
            self.assertAlmostEqual(bond['sxx'], 0., places=13)
            self.assertAlmostEqual(bond['syy'], 0., places=13)
        for t in obs['triangles']:
            self.assertAlmostEqual(t['szszsz'], .125, places=13)
            self.assertAlmostEqual(t['connected_szszsz'], 0., places=13)
            self.assertAlmostEqual(t['sum_sz_variance'], 0., places=13)

    def test_random_cluster_three_body(self):
        s = BlockState.random(1, seed=2502)
        env = HoneycombEnvironment.initial(1).run(layers(s), 2, 20, 1e-6)
        obs = measure_complete(s, env)
        for t in obs['triangles']:
            a, b = s.A[t['a']].flatten(), s.B[t['b']].flatten()
            aa = float(a@embed(SZ,t['i'],6)@embed(SZ,t['j'],6)@a)
            bb = float(b@embed(SZ,t['t'],3)@b)
            self.assertAlmostEqual(t['szszsz'], aa*bb, places=13)

    def test_full_components_agree_with_existing_energy(self):
        s = BlockState.random(2, seed=2503, c3=True)
        sweep(s, .03)
        env = HoneycombEnvironment.initial(2).run(layers(s), 4, 20, 1e-6)
        old = measure(s, env)
        new = measure_complete(s, env, max_corner_cache_bytes=1024)
        self.assertAlmostEqual(old['energy_total'],new['energy_total'],places=12)
        self.assertAlmostEqual(old['magnetization_normalized'],new['magnetization_normalized'],places=12)

    def test_device_change_preserves_legacy_fingerprint(self):
        cfg = settings()
        legacy = {k:v for k,v in cfg.items() if k != 'measurement_device'}
        self.assertEqual(immutable_settings(cfg),legacy)
        self.assertEqual(immutable_settings(cfg | dict(measurement_device='cuda:1')),legacy)
        self.assertEqual(str(resolve_measurement_device('cpu')),'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available in this validation environment')
    def test_cuda_ctm_and_complete_observables(self):
        # This is D2 only. Use a local visible GPU for validation; it does
        # not change the server's default or replace the user's cuda:1 choice.
        device = resolve_measurement_device('cuda:0')
        source = BlockState.random(2,seed=2503,c3=True)
        sweep(source,.03)
        cpu_env = initial_environment(2,torch.device('cpu')).run(layers(source),4,20,1e-6)
        cpu_obs = measure_complete(source,cpu_env,max_corner_cache_bytes=1024)
        gpu_state = measurement_state(source,device)
        gpu_env = initial_environment(2,device)
        self.assertTrue(all(t.device == device for t in gpu_state.A+gpu_state.B))
        self.assertTrue(all(t.device.type == 'cpu' for t in source.A+source.B))
        self.assertTrue(all(t.device == device for f in gpu_env.faces for t in f.C+f.Ta+f.Tb))
        gpu_env.run(layers(gpu_state),4,20,1e-6)
        gpu_obs = measure_complete(gpu_state,gpu_env,max_corner_cache_bytes=1024)
        self.assertAlmostEqual(cpu_obs['energy_total'],gpu_obs['energy_total'],delta=1e-7)
        self.assertAlmostEqual(cpu_obs['magnetization_normalized'],gpu_obs['magnetization_normalized'],delta=1e-7)
        self.assertLess(max(abs(a['sz']-b['sz']) for a,b in zip(cpu_obs['sites'],gpu_obs['sites'])),1e-6)
        self.assertLess(max(abs(a['szszsz']-b['szszsz']) for a,b in zip(cpu_obs['triangles'],gpu_obs['triangles'])),1e-6)

    def test_interruption_resume_and_parent_lineage(self):
        cfg = settings()
        cfg.update(dims=[2,3], chi=3, ctm_min=8, ctm_max=10, monitor_min_steps=1,
                   monitor_tau=.03, checkpoint_steps=1, stable_checks=3,
                   lambda_drift_tolerance=0., rho_drift_tolerance=0.,
                   stages=[dict(dt=.03,min_tau=.12,max_tau=.12)])
        from . import workflow
        original = workflow.sweep
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt('simulated interruption')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory(prefix='chain_smoke_', dir=Path(__file__).parent) as tmp:
            out = Path(tmp)/'resumed'
            with patch.object(workflow, 'sweep', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run_chain(cfg, out, make_plots=False)
            payload = torch.load(out/'D2'/'checkpoint.pt', weights_only=False)
            self.assertEqual(payload['metadata']['progress']['stage_steps'],2)
            rows = run_chain(cfg, out, make_plots=False)
            self.assertEqual([r['D'] for r in rows],[2,3])
            child = json.loads((out/'D3'/'completed.json').read_text())
            self.assertEqual(child['parent_sha256'],sha256(out/'D2'/'final.pt'))
            first_hash = sha256(out/'D3'/'final.pt')
            with patch.object(workflow, 'measure_complete', side_effect=AssertionError('completed D remeasured')):
                run_chain(cfg, out, make_plots=False)
            self.assertEqual(first_hash,sha256(out/'D3'/'final.pt'))
            uninterrupted = Path(tmp)/'uninterrupted'
            run_chain(cfg, uninterrupted, make_plots=False)
            a, b = BlockState.load(out/'D3'/'final.pt'), BlockState.load(uninterrupted/'D3'/'final.pt')
            for x,y in zip(a.A+a.B,b.A+b.B):
                self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=1e-11))
            with writer_lock(out):
                with self.assertRaises(RuntimeError):
                    with writer_lock(out):
                        pass
            # The lock FILE still exists, but the released OS lock allows
            # immediate reuse; a stale pathname does not block the next run.
            with writer_lock(out):
                pass


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SmallChecks))
    validation = Path(__file__).resolve().parents[1]/'results'/'server_validation'
    validation.mkdir(parents=True, exist_ok=True)
    (validation/'test_results.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        cuda_available=torch.cuda.is_available(),success=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
