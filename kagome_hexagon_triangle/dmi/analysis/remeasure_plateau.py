"""Remeasure rejected saved kets, preserving the original rejection records.

Two chi values and full imaginary diagnostics; no SU, no changes to old result.
Acceptance is explicitly 1e-6 imaginary/face error and 1e-5 chi stability, rather
than pretending the original absolute 1e-8 Hermiticity guard passed.
"""
import json,time,traceback
from pathlib import Path
import torch
from ..src.model import load_state
from ..src.observables import environment,measure
from ..src.plateau_explore import OPTIONS
from ...server.workflow import atomic_json


def run():
    torch.set_num_threads(10);root=Path(OPTIONS['out'])
    for p in sorted(root.glob('D*_Dz*_*/*/observables.json')):
        original=json.loads(p.read_text())
        if original['valid_measurement']:continue
        target=p.with_name('observables_recheck.json')
        if target.exists():continue
        state,cp,_=load_state(p.with_name('final.pt'));checks=[]
        try:
            for chi in (12,16):
                start=time.perf_counter()
                env=environment(state,chi=chi,max_iterations=100,tolerance=1e-9,min_iterations=32)
                r=measure(state,env,J=cp.J,Dz=cp.Dz,strict=False)
                r.update(chi=chi,measurement_seconds=time.perf_counter()-start)
                atomic_json(p.with_name(f'diagnostic_chi{chi}.json'),r);checks.append(r)
            a,b=checks;de=abs(a['energy_total']-b['energy_total']);dm=abs(a['magnetization_normalized']-b['magnetization_normalized'])
            ok=all(v['ctm_converged'] and v['max_hermitian_imaginary_part']<1e-6 and v['face_consistency_max']<1e-6 for v in checks) and de<1e-5 and dm<1e-5
            r=b|dict(valid_measurement=ok,m_normalized=b['magnetization_normalized'],
                branch=original['branch'],h_physical=original['h_physical'],
                recheck_acceptance='imaginary<1e-6, face spread<1e-6, converged CTM, chi energy/m stability<1e-5',
                chi_energy_difference=de,chi_m_difference=dm,original_rejection=original['error'])
            atomic_json(target,r)
            print('RECHECK',p.parent.parent.name,p.parent.name,'valid',ok,'E',r['energy_total'],'m',r['m_normalized'],'imag',r['max_hermitian_imaginary_part'],'dE',de,'dm',dm,flush=True)
        except Exception:
            p.with_name('recheck_error.txt').write_text(traceback.format_exc());print('RECHECK ERROR',p,flush=True)
        from .plateau_report import run as report
        report(quiet=True)


if __name__=='__main__':run()
