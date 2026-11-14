"""Small pure-DMI 120-degree SU test. All options are below; CPU only.

One random seed plus an independently embedded coherent reference at each D.
No spin/spatial symmetry projection. Only final CTM measurements per case/chi.
"""
from pathlib import Path
import json,time,copy
import numpy as np
import torch
from .model import Couplings,complex_state,product_120,save_state,load_state
from .simple_update import sweep
from .observables import environment,measure
from ...simple_update import BlockState
from ...server.workflow import expand_state,atomic_json,write_csv

OPTIONS=dict(dims=[2,3,4],branches=['random','ordered120'],seed=20261113,
    J=0.,Dz=1.,h=0.,threads=10,
    schedule=[(.05,8.),(.02,4.),(.005,1.),(.001,.2)],
    chi=12,check_chi=20,ctm_max=100,ctm_tol=1e-6,
    out=str(Path(__file__).resolve().parents[2]/'results/pure_DMI_D2_D4_order_20261113'))


def initial(D,branch,seed):
    if branch=='ordered120':
        # Exact padding, with NO enforced C3 or spin symmetry. SU gates may
        # populate new Schmidt channels; the actual spectrum is saved below.
        return expand_state(product_120(),D)
    if branch!='random':raise ValueError(branch)
    s=complex_state(BlockState.random(D,field=0.,seed=seed,c3=False))
    gen=torch.Generator().manual_seed(seed)
    for group in (s.A,s.B):
        for i,t in enumerate(group):
            group[i]=torch.randn(t.shape,generator=gen,dtype=torch.complex128)
            group[i]/=group[i].norm()
    return s


def run_case(cfg,D,branch):
    out=Path(cfg['out'])/f'D{D}_{branch}';out.mkdir(parents=True,exist_ok=True)
    cp=Couplings(J=cfg['J'],Dz=cfg['Dz'],h=cfg['h'])
    final=out/'final.pt';completed=0;history=[]
    if final.exists():s,_,meta=load_state(final);completed=meta['completed'];history=meta['history']
    else:
        s=initial(D,branch,cfg['seed']+D)
        save_state(out/'initial.pt',s,cp)
    for stage,(dt,tau) in enumerate(cfg['schedule'],1):
        if stage<=completed:continue
        steps=round(tau/dt);start=time.perf_counter();last=[]
        for step in range(1,steps+1):
            before=[w.clone() for row in s.weights for w in row]
            change,discard=sweep(s,dt,J=cfg['J'],Dz=cfg['Dz'])
            drift=max(float((a-b).abs().max()) for a,b in zip(before,[w for row in s.weights for w in row]))
            if step>steps-20:last.append(drift/dt)
            if step%100==0 or step==steps:
                print(f'PY D={D} {branch} stage={stage} step={step}/{steps} weight_drift_per_tau={drift/dt:.3g} seconds={time.perf_counter()-start:.1f}',flush=True)
        history.append(dict(stage=stage,dt=dt,steps=steps,tau=dt*steps,
            seconds=time.perf_counter()-start,weight_drift_per_tau_last20=max(last)))
        save_state(final,s,cp,dict(completed=stage,history=history))
        atomic_json(out/'su_progress.json',dict(completed=stage,history=history))
    spectra=[[w.tolist() for w in row] for row in s.weights]
    atomic_json(out/'schmidt_spectra.json',dict(weights=spectra))
    requested_chis=[cfg['chi'],cfg['check_chi']] if D==max(cfg['dims']) else [cfg['chi']]
    for chi in requested_chis:
        target=out/f'observables_chi{chi}.json'
        if target.exists():continue
        start=time.perf_counter();env=environment(s,chi=chi,max_iterations=cfg['ctm_max'],tolerance=cfg['ctm_tol'])
        strict_recheck=False
        try:
            result=measure(s,env,J=cfg['J'],Dz=cfg['Dz'])
        except ArithmeticError as exc:
            # Keep the SU checkpoint. A too-small CTM approximation can leave
            # complex Hermitian expectations; do not lower its accuracy check.
            atomic_json(out/f'rejected_chi{chi}.json',dict(error=str(exc),chi=chi,
                ctm_converged=env.converged,ctm_iterations=env.iterations,
                ctm_residual=env.residual if np.isfinite(env.residual) else None))
            print(f'PY RETRY STRICT CTM D={D} {branch} chi={chi}: {exc}',flush=True)
            # A corner-spectrum stop at 1e-6 does not guarantee 1e-8 accuracy
            # for every observable. Tighten CTM convergence, not the observable
            # test; reuse the converged SU ket without another SU stage.
            env=environment(s,chi=chi,max_iterations=cfg['ctm_max'],
                tolerance=min(cfg['ctm_tol'],1e-10),min_iterations=32)
            try:
                result=measure(s,env,J=cfg['J'],Dz=cfg['Dz'])
            except ArithmeticError as strict_exc:
                atomic_json(out/f'rejected_strict_chi{chi}.json',dict(error=str(strict_exc),chi=chi,
                    ctm_converged=env.converged,ctm_iterations=env.iterations,
                    ctm_tolerance_used=min(cfg['ctm_tol'],1e-10),
                    ctm_residual=env.residual if np.isfinite(env.residual) else None))
                if D==max(cfg['dims']) and chi<28:
                    next_chi=chi+4
                    if next_chi not in requested_chis:requested_chis.append(next_chi)
                    print(f'PY INCREASE CHI D={D} {branch}: {chi} -> {next_chi}; strict Hermiticity check retained',flush=True)
                    continue
                if (out/f"observables_chi{cfg['chi']}.json").exists():
                    atomic_json(out/'refinement_incomplete.json',dict(
                        baseline_chi=cfg['chi'],rejected_chi=chi,
                        reason='Higher-chi Hermiticity guard still fails; baseline retained, refinement NOT certified',
                        error=str(strict_exc)))
                    print(f'PY REFINEMENT NOT CERTIFIED D={D} {branch}; valid chi={cfg["chi"]} baseline retained',flush=True)
                    continue
                raise
            strict_recheck=True
        result.update(branch=branch,chi=chi,seed=cfg['seed']+D,su_history=history,
            strict_ctm_recheck=strict_recheck,
            ctm_tolerance_used=min(cfg['ctm_tol'],1e-10) if strict_recheck else cfg['ctm_tol'],
            measurement_seconds=time.perf_counter()-start)
        atomic_json(target,result)
        write_csv(out/f'sites_chi{chi}.csv',result['sites']);write_csv(out/f'bonds_chi{chi}.csv',result['bonds'])
        print(f"PY FINAL D={D} {branch} chi={chi} E={result['energy_total']:.12g} CTM={env.converged} residual={env.residual:.3g}",flush=True)


def run_all(cfg=OPTIONS):
    torch.set_num_threads(cfg['threads']);root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    target=root/'settings.json'
    if target.exists():
        if json.loads(target.read_text())!=json.loads(json.dumps(cfg)):raise ValueError('Settings changed; use a new output')
    else:atomic_json(target,cfg)
    for D in cfg['dims']:
        for branch in cfg['branches']:run_case(cfg,D,branch)
    print('PY ORDER PROBE COMPLETE',flush=True)


if __name__=='__main__':run_all()
