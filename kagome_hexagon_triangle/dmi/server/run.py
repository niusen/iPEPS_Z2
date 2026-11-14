"""Edit OPTIONS below; python -m kagome_hexagon_triangle.dmi.server.run.

CPU SU, one final CPU/CUDA measurement. No intermediate energy calculations.
The checkpoint contains J,Dz,h and cannot be mistaken for the old model.
"""
from pathlib import Path
import copy,time
import torch
from ..src.model import Couplings,complex_state,load_state,save_state
from ..src.simple_update import sweep
from ..src.observables import environment,measure
from ...simple_update import BlockState
from ...server.workflow import writer_lock,atomic_json,write_csv
from ...server.device_support import resolve_measurement_device,measurement_state,synchronize

OPTIONS=dict(
    J=1.,Dz=0.,h=.38, # Set J=0,Dz=1,h=0 for the pure-DMI benchmark.
    D=2,seed=91,c3=False, # Do not impose spatial C3 on transverse order.
    input='', # Existing old/DMI state, same D. A fresh output folder is required.
    out=str(Path(__file__).resolve().parents[2]/'results/dmi_D2'),
    schedule=[(.03,40.),(.01,20.),(.003,5.),(.001,2.),(.0005,1.)], # dt,tau
    threads=10,progress_every=100,measurement_device='cuda:1',
    chi=40,ctm_max=100,ctm_tol=1e-6,
)


def run(cfg):
    cfg=copy.deepcopy(cfg);couplings=Couplings(**{k:cfg[k] for k in ('J','Dz','h')})
    torch.set_num_threads(cfg['threads']);device=resolve_measurement_device(cfg['measurement_device'])
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    with writer_lock(out):
        if (out/'settings.json').exists():
            raise FileExistsError('Choose a new out; resume explicitly from latest.pt in input.')
        atomic_json(out/'settings.json',cfg)
        if cfg['input']:
            state,_,_=load_state(cfg['input']);state.field=cfg['h']
            if state.D!=cfg['D'] or state.c3!=cfg['c3']:
                raise ValueError('Input D/c3 must match options; no silent padding/symmetry change')
            complex_state(state)
        else:
            state=complex_state(BlockState.random(cfg['D'],field=cfg['h'],seed=cfg['seed'],c3=cfg['c3']))
            # General complex initial ket, not a real/collinear restriction.
            gen=torch.Generator().manual_seed(cfg['seed']+100000)
            for group in (state.A,state.B):
                for i,t in enumerate(group):
                    group[i]=t+1j*torch.randn(t.shape,generator=gen,dtype=torch.float64)
                    group[i]/=group[i].norm()
            if state.c3:
                from ...simple_update import copy_rotation_orbit
                copy_rotation_orbit(state,'A',0);copy_rotation_orbit(state,'B',0)
        save_state(out/'initial.pt',state,couplings);start=time.perf_counter()
        for stage,(dt,tau) in enumerate(cfg['schedule'],1):
            if dt<=0 or tau<0:raise ValueError('Invalid dt/tau')
            steps=round(tau/dt)
            for step in range(1,steps+1):
                change,discard=sweep(state,dt,J=cfg['J'],Dz=cfg['Dz'])
                if step%cfg['progress_every']==0 or step==steps:
                    print(f'SU stage={stage} step={step}/{steps} weight_change={change:.3g} discarded={discard:.3g} seconds={time.perf_counter()-start:.1f}',flush=True)
                    save_state(out/'latest.pt',state,couplings,dict(stage=stage,step=step,complete=False))
        su_seconds=time.perf_counter()-start;save_state(out/'final.pt',state,couplings)
        measured=measurement_state(state,device);synchronize(device);start=time.perf_counter()
        env=environment(measured,chi=cfg['chi'],max_iterations=cfg['ctm_max'],tolerance=cfg['ctm_tol'])
        result=measure(measured,env,J=cfg['J'],Dz=cfg['Dz']);synchronize(device)
        result.update(su_seconds=su_seconds,measurement_seconds=time.perf_counter()-start)
        # A capped CTMRG run remains explicitly flagged as unconverged.
        atomic_json(out/'observables.json',result)
        write_csv(out/'site_magnetization.csv',result['sites']);write_csv(out/'bond_energies.csv',result['bonds'])
        write_csv(out/'summary.csv',[{k:v for k,v in result.items() if k not in ('sites','bonds')}])
        print(f"FINAL E={result['energy_total']:.12g} m={result['magnetization_normalized']:.9g} CTM_converged={env.converged}",flush=True)
        return result


if __name__=='__main__':run(OPTIONS)
