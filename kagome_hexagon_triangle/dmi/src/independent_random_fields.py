"""Every (D,Dz,h,trial) is a NEW unrestricted complex random ket.

No field continuation, no D continuation, no projected seed. Resume is allowed
only within the identical field point. This numerical job saves data only;
plotting is a separate laptop task and is never imported here.
"""
from pathlib import Path
import itertools,json,hashlib
import torch
from .plateau_explore import point
from .model import load_state
from .observables import environment,measure
from ...server.workflow import atomic_json,writer_lock,write_csv

OPTIONS=dict(dims=[2,3,4],Dz=[.02,.05,.1],fields=[.28,.32,.36,.38,.40,.44,.48],trials=[1,2,3],
    seed_base=202611140000,threads=10,chi=12,ctm_max=100,ctm_tol=1e-6,
    fresh=[(.03,40.),(.01,20.),(.003,5.),(.001,2.),(.0005,1.)],
    warm=[],out=str(Path(__file__).resolve().parents[2]/'results/DMI_independent_random3_D2_D4_20261114'))


def seed_for(cfg,D,Dz,h,trial):
    # Integer grid encoding is injective on this plan; assert uniqueness below.
    return cfg['seed_base']+trial*10000000+D*1000000+round(Dz*1000)*1000+round(h*1000)


def folder_for(cfg,D,Dz,h,trial):
    return Path(cfg['out'])/f'D{D}_Dz{Dz:.3f}_random{trial}'/f'h{h:.3f}'


def collect(root):
    """Write scalar progress with no plotting dependencies, including cached points."""
    keys=('D','Dz','h_physical','trial','seed','energy_total','energy_exchange',
          'energy_dmi','energy_field','m_normalized','mean_sz','valid_measurement',
          'ctm_converged','ctm_iterations','ctm_residual','chi','measurement_seconds',
          'measurement_device','warm_start','measurement_basis','error','recheck_error')
    rows=[]
    for p in sorted(Path(root).glob('D*_Dz*_*/*/independent_result.json')):
        r=json.loads(p.read_text())
        row={k:r.get(k) for k in keys}
        row['accepted']=bool(r.get('valid_measurement') and r.get('ctm_converged'))
        row['source']=str(p)
        rows.append(row)
    write_csv(Path(root)/'progress.csv',rows)


def recheck(final,raw,cfg,folder):
    if raw['valid_measurement'] and raw.get('ctm_converged',False):return raw
    state,cp,_=load_state(final);values=[]
    from ...server.device_support import resolve_measurement_device,measurement_state
    state=measurement_state(state,resolve_measurement_device(cfg.get('measurement_device','cpu')))
    for chi in (12,16):
        # Use the same CTMRG stopping tolerance for the original and diagnostic
        # measurements; quality/chi checks below remain separate diagnostics.
        env=environment(state,chi=chi,max_iterations=cfg['ctm_max'],tolerance=cfg['ctm_tol'])
        r=measure(state,env,J=cp.J,Dz=cp.Dz,strict=False);r['chi']=chi
        atomic_json(folder/f'diagnostic_chi{chi}.json',r);values.append(r)
    a,b=values;de=abs(a['energy_total']-b['energy_total']);dm=abs(a['magnetization_normalized']-b['magnetization_normalized'])
    valid=all(v['ctm_converged'] and v['max_hermitian_imaginary_part']<1e-6 and v['face_consistency_max']<1e-6 for v in values) and de<1e-5 and dm<1e-5
    return b|dict(valid_measurement=valid,m_normalized=b['magnetization_normalized'],
        chi_energy_difference=de,chi_m_difference=dm,original_error=raw.get('error','CTM not converged'),
        measurement_basis='recheck: imaginary/face<1e-6, both CTMs converged, chi energy/m differences<1e-5')


def run(cfg=OPTIONS):
    from ...server.device_support import resolve_measurement_device
    resolve_measurement_device(cfg.get('measurement_device','cpu')) # fail before any SU if CUDA is unavailable
    torch.set_num_threads(cfg['threads']);root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    cases=list(itertools.product(cfg['dims'],cfg['Dz'],cfg['fields'],cfg['trials']))
    seeds=[seed_for(cfg,*v) for v in cases];assert len(set(seeds))==len(seeds)
    with writer_lock(root):
        settings=root/'settings.json'
        if settings.exists():
            if json.loads(settings.read_text())!=json.loads(json.dumps(cfg)):raise ValueError('Settings changed; choose new out')
        else:atomic_json(settings,cfg)
        atomic_json(root/'plan.json',[dict(D=D,Dz=dz,h=h,trial=t,seed=s) for (D,dz,h,t),s in zip(cases,seeds)])
        collect(root)
        for (D,Dz,h,trial),seed in zip(cases,seeds):
            folder=folder_for(cfg,D,Dz,h,trial);folder.mkdir(parents=True,exist_ok=True)
            done=folder/'independent_result.json'
            if done.exists():
                print(f'SKIP saved D={D} Dz={Dz} h={h} random{trial}',flush=True)
                continue
            manifest=dict(D=D,Dz=Dz,h=h,trial=trial,seed=seed,initialization='fresh independent complex Gaussian',
                previous_field_source=None,previous_D_source=None,imposed_U1=False,imposed_C3=False)
            p=folder/'initialization.json'
            if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('Initialization metadata mismatch')
            atomic_json(p,manifest)
            # Reuse the checked kernel's initializer. Its branch!=up seed rule
            # adds D+round(1000Dz)+10000; compensate to obtain EXACTLY our seed.
            local=cfg|dict(seed=seed-D-round(Dz*1000)-10000)
            final,raw=point(local,D,Dz,h,f'random{trial}',source=None,fresh=True)
            start,_,_=load_state(folder/'initial.pt');assert start.seed==seed and start.D==D and not start.c3
            manifest2=manifest|dict(initial_checkpoint_sha256=hashlib.sha256((folder/'initial.pt').read_bytes()).hexdigest())
            atomic_json(folder/'initialization_audit.json',manifest2)
            try:r=recheck(final,raw,cfg,folder)
            except Exception as exc:r=raw|dict(valid_measurement=False,recheck_error=str(exc))
            r.update(D=D,Dz=Dz,h_physical=h,trial=trial,seed=seed,branch=f'random{trial}',
                initialization='fresh independent random',source=None,warm_start=False,
                measurement_device=cfg.get('measurement_device','cpu'),
                m_normalized=r.get('m_normalized',r.get('magnetization_normalized')))
            atomic_json(done,r)
            collect(root)
        (root/'COMPLETED.txt').write_text('All independent random field points finished. Inspect measurement validity before drawing phase conclusions.\n')


if __name__=='__main__':run()
