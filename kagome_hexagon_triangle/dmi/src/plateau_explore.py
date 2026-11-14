"""Bounded overnight dense SU exploration. Options in code, CPU, no U1/C3.

Two independent random endpoints, ascending/descending sweeps. Checkpoints at
stage boundaries; one final CTM per field point. Ordinary process, no watchdog.
"""
from pathlib import Path
import time,json,copy,traceback,hashlib
import torch
from .model import Couplings,save_state,load_state
from .order_probe import initial
from .simple_update import sweep
from .observables import environment,measure
from ...server.workflow import atomic_json,write_csv,writer_lock

OPTIONS=dict(dims=[2,3,4],Dz=[.02,.05,.1],fields=[.28,.32,.36,.38,.40,.44,.48],
    fresh=[(.03,40.),(.01,20.),(.003,5.),(.001,2.),(.0005,1.)],
    warm=[(.03,10.),(.01,5.),(.003,1.),(.001,.5),(.0005,.2)],
    threads=10,chi=12,ctm_max=100,ctm_tol=1e-6,seed=20261114,
    out=str(Path(__file__).resolve().parents[2]/'results/DMI_plateau_dense_D2_D4_20261113'))


def point(cfg,D,Dz,h,branch,source,fresh):
    root=Path(cfg['out']);folder=root/f'D{D}_Dz{Dz:.3f}_{branch}'/f'h{h:.3f}';folder.mkdir(parents=True,exist_ok=True)
    result=folder/'observables.json';final=folder/'final.pt';progress=folder/'stage.pt'
    if result.exists():return final,json.loads(result.read_text())
    completed=0;history=[]
    if final.exists():s,_,meta=load_state(final);history=meta['history'];completed=999
    elif progress.exists():s,_,meta=load_state(progress);completed=meta['completed'];history=meta['history']
    elif source:s,_,_=load_state(source)
    else:s=initial(D,'random',cfg['seed']+D+round(Dz*1000)+(0 if branch=='up' else 10000))
    s.field=h;assert not s.c3
    cp=Couplings(J=1.,Dz=Dz,h=h);schedule=cfg['fresh'] if fresh else cfg['warm']
    if not (folder/'initial.pt').exists():save_state(folder/'initial.pt',s,cp)
    for k,(dt,tau) in enumerate(schedule,1):
        if k<=completed:continue
        start=time.perf_counter();n=round(tau/dt);last=0.
        for step in range(1,n+1):
            before=[w.clone() for row in s.weights for w in row]
            drift,discard=sweep(s,dt,J=1.,Dz=Dz)
            last=max(float((a-b).abs().max()) for a,b in zip(before,[w for row in s.weights for w in row]))/dt
            if step%250==0 or step==n:
                print(f'PY D={D} Dz={Dz} h={h} {branch} stage={k} {step}/{n} drift/tau={last:.3g} seconds={time.perf_counter()-start:.1f}',flush=True)
        history.append(dict(stage=k,dt=dt,steps=n,tau=n*dt,last_full_step_weight_drift_per_tau=last,seconds=time.perf_counter()-start))
        save_state(progress,s,cp,dict(completed=k,history=history,source=str(source or 'random')))
    save_state(final,s,cp,dict(history=history,source=str(source or 'random')))
    start=time.perf_counter()
    try:
        from ...server.device_support import resolve_measurement_device,measurement_state
        device=resolve_measurement_device(cfg.get('measurement_device','cpu'))
        measured=measurement_state(s,device)
        env=environment(measured,chi=cfg['chi'],max_iterations=cfg['ctm_max'],tolerance=cfg['ctm_tol'])
        r=measure(measured,env,J=1.,Dz=Dz)
        r.update(valid_measurement=True,Dz=Dz,h_physical=h,branch=branch,
            m_normalized=2*r['mean_sz'],measurement_seconds=time.perf_counter()-start,history=history,
            source=str(source or 'independent random dense'),symmetry='none',chi=cfg['chi'],measurement_device=str(device))
        write_csv(folder/'sites.csv',r['sites']);write_csv(folder/'bonds.csv',r['bonds'])
    except Exception as exc:
        r=dict(D=D,Dz=Dz,h_physical=h,branch=branch,valid_measurement=False,error=str(exc),history=history)
        (folder/'measurement_error.txt').write_text(traceback.format_exc())
    atomic_json(result,r)
    print('PY POINT',D,Dz,h,branch,r.get('energy_total'),r.get('m_normalized'),r.get('ctm_converged'),flush=True)
    return final,r


def collect(root):
    rows=[]
    for p in Path(root).glob('D*_Dz*_*/*/observables.json'):
        r=json.loads(p.read_text());rows.append({k:v for k,v in r.items() if k not in ('sites','bonds','history')})
    write_csv(Path(root)/'progress.csv',rows)


def run(cfg=OPTIONS):
    torch.set_num_threads(cfg['threads']);root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    with writer_lock(root):
        settings=root/'settings.json'
        if settings.exists():
            if json.loads(settings.read_text())!=json.loads(json.dumps(cfg)):raise ValueError('Settings changed')
        else:atomic_json(settings,cfg)
        for D in cfg['dims']:
            for Dz in cfg['Dz']:
                for branch,fields in (('up',cfg['fields']),('down',cfg['fields'][::-1])):
                    source=None
                    for i,h in enumerate(fields):
                        source,_=point(cfg,D,Dz,h,branch,source,i==0);collect(root)
        (root/'COMPLETED.txt').write_text('All bounded SU scans finished. Inspect convergence/energy branches before inferring a plateau.\n')
        print('PY DMI PLATEAU EXPLORATION COMPLETE',flush=True)


if __name__=='__main__':run()
