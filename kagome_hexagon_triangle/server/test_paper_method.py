"""Only D1/D2 regression checks; never starts a server D7 run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..simple_update import BlockState,rotate_tensor,embed,SZ
from ..geometry import edge_bonds,neighbour
from ..paper_method import physical_nn_su as su
from ..paper_method.c3_reduction import try_reduce,mode_product
from ..paper_method import workflow as protocol
from .run_D7_paper_method import settings as production_settings
from .workflow import sha256


def matrix(state,c,k):
    b=neighbour(c,k);order=[0]+[i+1 for i in range(3) if i!=k]+[k+1]
    a,t=state.A[c].clone(),state.B[b].clone()
    for kind,q,x in (('A',c,a),('B',b,t)):
        for leg,w in enumerate(state.lambdas(kind,q)):
            if leg!=k:
                shape=[1]*4;shape[leg+1]=len(w);x*=w.reshape(shape)
    a,t=a.permute(order),t.permute(order)
    return torch.einsum('pabi,qcdi,i->pabqcd',a,t,state.weights[c][k]).reshape(64*state.D**2,8*state.D**2)


def direct_gate(value,i,j,D,dt):
    shape=[2]*6+[D,D]+[2]*3+[D,D]
    active=[i,8+j];order=active+[k for k in range(13) if k not in active]
    x=value.reshape(shape).permute(order)
    x=(su.physical_gate(dt)@x.reshape(4,-1)).reshape([shape[k] for k in order])
    return x.permute([order.index(k) for k in range(13)]).reshape_as(value)


class PaperMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(10)

    def test_physical_gate_and_all_cross_updates_vs_direct_dense_SVD(self):
        for c in range(3):
            for k in range(3):
                for i,j in edge_bonds(k):
                    s=BlockState.random(2,field=.4,seed=610+c*9+k*3+i)
                    exact=direct_gate(matrix(s,c,k),i,j,2,.01)
                    u,w,v=torch.linalg.svd(exact,full_matrices=False)
                    exact=(u[:,:2]*w[:2])@v[:2]
                    su.update_cross(s,c,k,i,j,.01)
                    got=matrix(s,c,k)
                    self.assertLess(float((got/got.norm()-exact/exact.norm()).norm()),2e-12)

    def test_internal_and_uniform_field_are_exact_local_operations(self):
        s=BlockState.random(2,field=.4,seed=66)
        for i in range(6):
            j=(i+1)%6
            z=embed(SZ,i,6)@embed(SZ,j,6)
            from ..simple_update import SP,SM
            h=z+.5*(embed(SP,i,6)@embed(SM,j,6)+embed(SM,i,6)@embed(SP,j,6))
            exact=(torch.matrix_exp(-.01*h)@s.A[0].flatten(1)).reshape_as(s.A[0]);exact/=exact.norm()
            got=su.apply_inside(s.A[0],su.physical_gate(.01),[i,j],6)
            self.assertLess(float((got-exact).norm()),1e-12)
        for kind,n in (('A',6),('B',3)):
            g=su.field_diagonal(kind,.4,.01)
            h=sum(embed(SZ,i,n) for i in range(n))
            self.assertLess(float((torch.diag(g)-torch.matrix_exp(.004*h)).norm()),1e-12)

    def test_full_six_updates_match_C3_orbit_updates(self):
        a=BlockState.random(2,field=.4,seed=712,c3=True);b=copy.deepcopy(a);b.c3=False
        su.sweep(a,.003);su.sweep(b,.003)
        for c in range(3):
            for k in range(3):self.assertTrue(torch.allclose(a.weights[c][k],b.weights[c][k],atol=2e-11,rtol=0))
            for kind in ('A','B'):
                x=a.dressed(kind,c,power=1).flatten(1);y=b.dressed(kind,c,power=1).flatten(1)
                self.assertLess(float((x@x.mT-y@y.mT).norm()),2e-10)
        self.assertEqual(a.steps,1)

    def test_term_counts_and_polarized_product(self):
        self.assertEqual(len(su.orbit_representatives()),20)
        self.assertEqual(sum(t[0]=='cross' for t in su.orbit_representatives())*3,36)
        self.assertEqual(sum(t[0]=='internal' for t in su.orbit_representatives())*3,18)
        actual={(r,(k-r)%3,(i-2*r)%6,(j-r)%3) for term in su.orbit_representatives()
            if term[0]=='cross' for _,k,i,j in (term,) for r in range(3)}
        self.assertEqual(actual,{(c,k,i,j) for c in range(3) for k in range(3) for i,j in edge_bonds(k)})
        s=BlockState.random(1,field=.4,seed=5,c3=False)
        for group in (s.A,s.B):
            for t in group:t.zero_();t[0,0,0,0]=1
        su.sweep(s,.03)
        for group in (s.A,s.B):
            for t in group:self.assertAlmostEqual(float(t[0,0,0,0]),1.,places=12)

    def test_reduction_exact_C3_and_random_degenerate_bond_gauges(self):
        original=BlockState.random(2,field=.4,seed=801,c3=True)
        gauged=copy.deepcopy(original);gauged.c3=False
        gen=torch.Generator().manual_seed(802)
        for c in range(3):
            for k in range(3):
                q,_=torch.linalg.qr(torch.randn((2,2),generator=gen,dtype=torch.float64))
                gauged.A[c]=mode_product(gauged.A[c],q,k)
                b=neighbour(c,k);gauged.B[b]=mode_product(gauged.B[b],q,k)
        candidate,info=try_reduce(gauged)
        self.assertTrue(info['accepted'],info)
        self.assertLess(info['gauge_aligned_tensor_residual'],1e-10)
        self.assertLess(info['rho_proxy_drift'],1e-10)
        for group,kind,shift in ((candidate.A,'A',1),(candidate.B,'B',-1)):
            for c in range(3):self.assertTrue(torch.allclose(rotate_tensor(group[c],kind),group[(c+shift)%3],atol=1e-12,rtol=0))

    def test_far_state_reduction_rejected_without_modifying_source(self):
        s=BlockState.random(2,field=.4,seed=803,c3=False);before=copy.deepcopy(s)
        candidate,info=try_reduce(s)
        self.assertFalse(info['accepted']);self.assertIs(candidate,s)
        for x,y in zip(s.A+s.B,before.A+before.B):self.assertTrue(torch.equal(x,y))

    def test_reduction_accepts_small_perturbation(self):
        s=BlockState.random(2,field=.4,seed=804,c3=True);s.c3=False
        s.A[1].flatten()[0]+=1e-7
        got,info=try_reduce(s)
        self.assertTrue(info['accepted'],info);self.assertTrue(got.c3)

    def test_exact_C3_gauge_reduction_preserves_all_CTM_observables(self):
        from ..honeycomb_ctm import HoneycombEnvironment,layers
        from ..observables import measure
        s=BlockState.random(2,field=.4,seed=809,c3=True);s.c3=False
        gen=torch.Generator().manual_seed(810)
        for c in range(3):
            for k in range(3):
                q,_=torch.linalg.qr(torch.randn((2,2),generator=gen,dtype=torch.float64))
                s.A[c]=mode_product(s.A[c],q,k)
                b=neighbour(c,k);s.B[b]=mode_product(s.B[b],q,k)
        reduced,info=try_reduce(s);self.assertTrue(info['accepted'],info)
        measured=[]
        for state in (s,reduced):
            env=HoneycombEnvironment.initial(2).run(layers(state),chi=4,max_iterations=26,tolerance=1e-7)
            measured.append(measure(state,env))
        a,b=measured
        self.assertLess(abs(a['energy_total']-b['energy_total']),1e-9)
        self.assertLess(abs(a['magnetization_normalized']-b['magnetization_normalized']),1e-9)
        self.assertLess(max(abs(x['sz']-y['sz']) for x,y in zip(a['sites'],b['sites'])),1e-9)
        self.assertLess(max(abs(x['energy']-y['energy']) for x,y in zip(a['bonds'],b['bonds'])),1e-9)

    def config(self,seeds=(9101,)):
        return production_settings()|dict(D=2,seeds=list(seeds),threads=10,measurement_device='cpu',
            schedule=[dict(dt=.003,tau=.006),dict(dt=.001,tau=.002)],reduce_after_stage=1,
            chis=[2,3],ctm_min=8,ctm_max=14,ctm_tolerance=1e-5,
            checkpoint_steps=1,monitor_steps=1,stable_checks=1,
            lambda_drift_tolerance=1e9,rho_drift_tolerance=1e9)

    def test_production_settings_and_bounded_work(self):
        cfg=production_settings();protocol.validate(cfg)
        self.assertEqual(cfg['D'],7);self.assertEqual(cfg['seeds'],list(range(7101,7111)))
        self.assertEqual(cfg['chis'],[40,60]);self.assertEqual(cfg['measurement_device'],'cuda:1')
        self.assertEqual(sum(round(s['tau']/s['dt']) for s in cfg['schedule']),9000)
        self.assertEqual(cfg['ctm_max'],50)

    def test_tiny_batch_full_outputs_and_skip_without_new_evolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';cfg=self.config((9101,9102))
            self.assertTrue(protocol.run(cfg,out))
            import csv
            rows=list(csv.DictReader((out/'summary.csv').open()))
            self.assertEqual([r['seed'] for r in rows],['9101','9102'])
            self.assertTrue(all(r['status']=='protocol_complete' for r in rows))
            self.assertTrue(all(r['steps']=='4' for r in rows))
            for seed in cfg['seeds']:
                root=out/f'seed{seed}'
                self.assertFalse(BlockState.load(root/'initial.pt').c3)
                final=BlockState.load(root/'final.pt');self.assertEqual(final.steps,4)
                reduction=json.loads((root/'c3_reduction.json').read_text())
                self.assertEqual(reduction['latest']['after_stage'],1)
                for chi in (2,3):
                    folder=root/f'measurements/chi{chi}'
                    for name in ('observables.json','site_magnetization.csv','bond_energies.csv','triangle_observables.csv','schmidt_spectra.csv'):
                        self.assertTrue((folder/name).exists(),name)
            paths=list(out.glob('seed*/final.pt'))+list(out.glob('seed*/measurements/chi*/observables.json'))
            hashes={p:sha256(p) for p in paths}
            with (patch.object(protocol,'sweep',side_effect=AssertionError('Completed run evolved again')),
                  patch.object(BlockState,'random',side_effect=AssertionError('Completed run randomized again'))):
                self.assertTrue(protocol.run(cfg,out))
            self.assertEqual(hashes,{p:sha256(p) for p in paths})
            with self.assertRaises(ValueError):protocol.run(cfg|dict(field=.41),out)

    def test_atomic_interruption_resume_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.config();out=Path(tmp)/'resumed';original=protocol.sweep;calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt('test interruption during SU')
                return original(*args,**kwargs)
            with patch.object(protocol,'sweep',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):protocol.run(cfg,out)
            with patch.object(BlockState,'random',side_effect=AssertionError('Resume changed initial state')):
                self.assertTrue(protocol.run(cfg,out))
            baseline=Path(tmp)/'baseline';self.assertTrue(protocol.run(cfg,baseline))
            a=BlockState.load(out/'seed9101/final.pt');b=BlockState.load(baseline/'seed9101/final.pt')
            for x,y in zip(a.A+a.B,b.A+b.B):self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=0))
            self.assertEqual(a.steps,b.steps);self.assertEqual(a.c3,b.c3)

    def test_measurement_interrupt_retries_same_final_state_without_SU(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';cfg=self.config();original=protocol.measure_snapshot
            def interrupt(config,*args,**kwargs):
                if config['chi']==3:raise KeyboardInterrupt('test interrupted second chi')
                return original(config,*args,**kwargs)
            with patch.object(protocol,'measure_snapshot',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):protocol.run(cfg,out)
            final=out/'seed9101/final.pt';digest=sha256(final)
            first=out/'seed9101/measurements/chi2/observables.json';first_hash=sha256(first)
            with patch.object(protocol,'sweep',side_effect=AssertionError('Measurement retry ran SU')):
                self.assertTrue(protocol.run(cfg,out))
            self.assertEqual(sha256(final),digest);self.assertEqual(sha256(first),first_hash)

    def test_workflow_accepts_only_prepared_close_C3_case(self):
        original=BlockState.random
        def prepared(D,field,seed,c3=False):
            s=original(D,field,seed,c3=True);s.c3=False;return s
        with tempfile.TemporaryDirectory() as tmp,patch.object(BlockState,'random',side_effect=prepared):
            out=Path(tmp)/'run';cfg=self.config()
            self.assertTrue(protocol.run(cfg,out))
            done=json.loads((out/'seed9101/completed.json').read_text())
            self.assertTrue(done['c3_final'],done)
            self.assertTrue(done['c3_reduction']['accepted'])

    def test_changed_source_is_reported_failed_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';cfg=self.config()
            self.assertTrue(protocol.run(cfg,out))
            source=out/'seed9101/initial.pt'
            with source.open('ab') as f:f.write(b'changed source')
            self.assertFalse(protocol.run(cfg,out))
            self.assertEqual(json.loads((out/'batch_status.json').read_text())['status'],'finished_with_errors')


if __name__=='__main__':unittest.main()
