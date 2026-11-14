"""Tiny CPU checks only: python -m kagome_hexagon_triangle.dmi.tests.test_small."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
from ..src.model import (Couplings,two_spin_hamiltonian,physical_gate,arrow_sign,bond_terms,
                    complex_state,product_120,save_state,load_state,SZ,SP,SM)
from ..src.simple_update import sweep,cross_factors
from ..src.observables import environment,measure,ComplexFaceMeasurement
from ...simple_update import BlockState,embed
from ...honeycomb_ctm import layers,double_layer
from ...paper_method.physical_nn_su import sweep as old_sweep


def main():
    torch.set_num_threads(2);report={}
    h=two_spin_hamiltonian(.7,.2,1);z=torch.kron(SZ,torch.eye(2))+torch.kron(torch.eye(2),SZ)
    assert torch.linalg.norm(h-h.mH)<1e-14
    assert torch.linalg.norm(h@z.to(h)-z.to(h)@h)<1e-14
    swap=torch.eye(4,dtype=h.dtype)[[0,2,1,3]]
    assert torch.linalg.norm(swap@h@swap-two_spin_hamiltonian(.7,.2,-1))<1e-14
    for kind,k,i,j in bond_terms():arrow_sign(kind,k,i,j)
    for sign in (-1,1):
        a,b=cross_factors(.02,0.,.7,sign)
        reconstructed=torch.einsum('aij,akl->ikjl',a,b).reshape(4,4)
        assert torch.linalg.norm(reconstructed-physical_gate(.02,0.,.7,sign))<1e-13
    # Dz=0: compare one complete step against the unchanged real NN kernel.
    for c3 in (False,True):
        a=BlockState.random(2,field=.38,seed=18,c3=c3);b=copy.deepcopy(a)
        old_sweep(a,.003);sweep(b,.003,Dz=0)
        # SVD phases/gauges differ for complex vs real; compare observables.
        ra=measure(a,environment(a,chi=2,max_iterations=2),Dz=0)
        rb=measure(b,environment(b,chi=2,max_iterations=2),Dz=0)
        err=abs(ra['energy_total']-rb['energy_total']);assert err<1e-9,err
        report[f'zero_Dz_step_c3_{c3}_energy_error']=err
    for chirality in (1,-1):
        s=product_120(chirality=chirality);env=environment(s,chi=2,max_iterations=8,min_iterations=7)
        r=measure(s,env,J=0,Dz=1);expected=-chirality*np.sqrt(3)/4
        assert abs(r['energy_total']-expected)<1e-12,(r['energy_total'],expected)
        assert max(abs(b['vector_chirality_z_ccw']+chirality*np.sqrt(3)/8) for b in r['bonds'])<1e-12
        report[f'product_120_chirality_{chirality}_energy']=r['energy_total']
    polarized=product_120(field=.38)
    for group in (polarized.A,polarized.B):
        for t in group:t.zero_();t[0,0,0,0]=1
    r=measure(polarized,environment(polarized,chi=2,max_iterations=2),J=.7,Dz=.3)
    assert abs(r['energy_total']-(.7/2-.38/2))<1e-12
    # Complex response AD vs explicitly rebuilding each corner, including
    # a non-Hermitian Sp insertion whose expectation has an imaginary part.
    local=layers(s);fm=ComplexFaceMeasurement(env.faces[0],local,0)
    op=embed(SP,0,6);replacement={('A',0):double_layer(s.dressed('A',0),op)}
    actual=fm.expectation(replacement);cc=env.faces[0].corners(local|replacement,0)
    exact=complex(torch.trace(cc[0]@cc[1]@cc[2])/fm.norm)
    assert abs(actual-exact)<1e-12
    # Finite Dz, J=0 must actually evolve and stay finite at D=2.
    s=complex_state(BlockState.random(2,field=0,seed=91));before=s.A[0].clone()
    for _ in range(2):sweep(s,.01,J=0,Dz=.3)
    assert not torch.allclose(before,s.A[0])
    r=measure(s,environment(s,chi=2,max_iterations=2),J=0,Dz=.3)
    assert np.isfinite(r['energy_total']);report['D2_pure_DMI_short_run_energy']=r['energy_total']
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'state.pt';save_state(path,s,Couplings(J=0,Dz=.3,h=0))
        restored,couplings,_=load_state(path)
        assert couplings.Dz==.3 and torch.equal(s.A[0],restored.A[0])
    if torch.cuda.is_available():
        from ...server.device_support import measurement_state
        gpu=measurement_state(product_120(),torch.device('cuda:0')) # LOCAL tiny check only
        try:
            r=measure(gpu,environment(gpu,chi=2,max_iterations=2),J=0,Dz=1)
        except RuntimeError as exc:
            if 'failed to open nvrtc-builtins' not in str(exc):raise
            report['GPU_check_skipped']='Local CUDA runtime missing nvrtc-builtins64_118.dll'
        else:
            assert abs(r['energy_total']+np.sqrt(3)/4)<1e-12
            report['GPU_D1_product_energy']=r['energy_total']
    from ..server.run import run,OPTIONS
    with tempfile.TemporaryDirectory() as directory:
        cfg=OPTIONS|dict(D=1,J=0.,Dz=.3,h=0.,measurement_device='cpu',threads=2,
            schedule=[(.01,.01)],chi=2,ctm_max=2,out=str(Path(directory)/'run'))
        r=run(cfg)
        assert np.isfinite(r['energy_total'])
        assert all((Path(cfg['out'])/name).is_file() for name in ('final.pt','observables.json','bond_energies.csv','site_magnetization.csv','summary.csv'))
    print(json.dumps(report,indent=2));print('ALL SMALL DMI CHECKS PASSED')


if __name__=='__main__':main()
