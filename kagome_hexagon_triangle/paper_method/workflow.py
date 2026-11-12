"""Fixed-budget six-tensor -> conditional-C3 SU, then same-state chi checks.

Reuses original checkpoint locks, BlockState, SU proxy and full-observable
measurement transport. Never imports author magnetization/pattern data.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import torch
from ..simple_update import BlockState
from ..server import workflow as io
from ..server.workflow_random_seeds import measure_snapshot,ARRAYS
from ..server.device_support import resolve_measurement_device,immutable_settings
from .physical_nn_su import sweep
from .c3_reduction import try_reduce

VERSION='physical_NN_six_then_conditional_C3_v1'


def validate(cfg):
    if not math.isfinite(cfg['field']):raise ValueError('Field must be finite')
    if type(cfg['D']) is not int or cfg['D']<1 or not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):
        raise ValueError('Invalid D or seeds')
    if any(type(s) is not int or not 0<=s<2**63 for s in cfg['seeds']):raise ValueError('Invalid seeds')
    if cfg['c3_policy'] not in ('if_close','never'):raise ValueError('c3_policy must be if_close or never')
    if cfg['c3_policy']=='if_close' and not 1<=cfg['reduce_after_stage']<len(cfg['schedule']):
        raise ValueError('C3 reduction must be followed by further SU stages')
    if not cfg['schedule'] or any(s['dt']<=0 or s['tau']<=0 or round(s['tau']/s['dt'])<1 for s in cfg['schedule']):
        raise ValueError('Invalid physical evolution schedule')
    if not cfg['chis'] or sorted(set(cfg['chis']))!=cfg['chis'] or min(cfg['chis'])<1:raise ValueError('Invalid chi list')
    if not 1<=cfg['ctm_min']<=cfg['ctm_max']<=50:raise ValueError('CTMRG must not exceed 50 iterations')
    if min(cfg[k] for k in ('threads','checkpoint_steps','monitor_steps','stable_checks','max_corner_cache_mb'))<1:
        raise ValueError('Invalid counts')
    if min(cfg[k] for k in ('ctm_tolerance','lambda_drift_tolerance','rho_drift_tolerance',
        'chi_energy_tolerance','chi_site_tolerance','chi_bond_tolerance'))<=0:raise ValueError('Invalid tolerance')
    if min(cfg['reduction'].values())<=0:raise ValueError('Invalid reduction bounds')


def fingerprint(cfg):
    root=Path(__file__).parent
    physics={name:io.sha256(root/name) for name in ('physical_nn_su.py','c3_reduction.py','workflow.py')}
    return hashlib.sha256(io.canonical_json(dict(version=VERSION,settings=immutable_settings(cfg),physics=physics)).encode()).hexdigest()


def atomic_state(path,state,**metadata):
    tmp=path.with_name(path.name+'.tmp');state.save(tmp,**metadata);tmp.replace(path)


def evolve(cfg,seed,folder,digest):
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/'initial.pt';checkpoint=folder/'checkpoint.pt';final=folder/'final.pt'
    if not source.exists():
        state=BlockState.random(cfg['D'],cfg['field'],seed,c3=False)
        atomic_state(source,state,settings_hash=digest,algorithm=VERSION)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload['metadata'].get('settings_hash')!=digest or payload['metadata'].get('algorithm')!=VERSION:
        raise ValueError('Initial random source/settings changed')
    source_sha=io.sha256(source)
    if checkpoint.exists():
        state,p,lineage=io.load_checkpoint(checkpoint,digest)
        if lineage['initial_sha256']!=source_sha:raise ValueError('Random source changed')
    else:
        state=BlockState.load(source);p=io.initialise_progress()
        p.update(reduction_done=False,reduction=None,reduction_attempts=[])
        lineage=dict(initial_sha256=source_sha,seed=seed,initialization='independent_six_tensor_random',algorithm=VERSION)
        io.save_checkpoint(checkpoint,state,digest,p,lineage)
    if state.D!=cfg['D'] or state.field!=cfg['field'] or state.seed!=seed:raise ValueError('State convention changed')
    start=time.perf_counter();elapsed=p['elapsed_su_seconds']
    def save():
        p['elapsed_su_seconds']=elapsed+time.perf_counter()-start
        io.save_checkpoint(checkpoint,state,digest,p,lineage)
    for stage,spec in enumerate(cfg['schedule']):
        if stage<p['stage']:continue
        if (not p['reduction_done'] and stage>=cfg['reduce_after_stage'] and
                stage not in [a['after_stage'] for a in p['reduction_attempts']]):
            atomic_state(folder/f'before_C3_stage{stage}.pt',state,settings_hash=digest)
            if cfg['c3_policy']=='if_close':
                state,info=try_reduce(state,**cfg['reduction'])
            else:info=dict(accepted=False,reason='disabled_keep_six')
            info=dict(after_stage=stage,**info)
            p['reduction_attempts'].append(info)
            p.update(reduction_done=info['accepted'] or cfg['c3_policy']=='never',reduction=info)
            # State, decision and stage counter commit in ONE atomic checkpoint.
            save();io.atomic_json(folder/'c3_reduction.json',dict(latest=info,attempts=p['reduction_attempts']))
            print(f'seed={seed} C3 reduction: {info}',flush=True)
        dt=float(spec['dt']);steps=round(spec['tau']/dt)
        if p['previous_weights'] is None:
            p['previous_weights'],p['previous_rho']=io.su_proxy(state)
            p['previous_steps']=p['stage_steps']
        discard=0.
        while p['stage_steps']<steps:
            _,discard=sweep(state,dt)
            p['stage_steps']+=1;p['d_steps']+=1;n=p['stage_steps']
            if n-p['previous_steps']>=cfg['monitor_steps'] or n==steps:
                w,rho=io.su_proxy(state);tau=dt*(n-p['previous_steps'])
                wrate=float((w-p['previous_weights']).abs().max())/tau
                rrate=float((rho-p['previous_rho']).norm())/tau
                stable=wrate<=cfg['lambda_drift_tolerance'] and rrate<=cfg['rho_drift_tolerance']
                p['stage_stable_checks']=p['stage_stable_checks']+1 if stable else 0
                p['last_rates']=dict(lambda_drift_per_tau=wrate,rho_drift_per_tau=rrate)
                p.update(previous_weights=w,previous_rho=rho,previous_steps=n)
                row=dict(seed=seed,stage=stage+1,dt=dt,stage_steps=n,tau=n*dt,c3=state.c3,
                    **p['last_rates'],discarded_weight_max=discard)
                io.append_csv(folder/'su_history.csv',row)
                print(f'seed={seed} stage={stage+1}/{len(cfg["schedule"])} step={n}/{steps} '
                      f'C3={state.c3} lambda/tau={wrate:.3e} rho/tau={rrate:.3e}',flush=True)
                save()
            elif p['d_steps']%cfg['checkpoint_steps']==0:save()
        report=dict(stage=stage+1,dt=dt,steps=steps,tau=steps*dt,c3=state.c3,
            proxy_converged=p['stage_stable_checks']>=cfg['stable_checks'],**p['last_rates'])
        p['stage_reports'].append(report)
        p.update(stage=stage+1,stage_steps=0,stage_stable_checks=0,previous_weights=None,
                 previous_rho=None,previous_steps=0,last_rates=None)
        save();io.atomic_json(folder/'su_stages.json',p['stage_reports'])
    if not final.exists():io.save_checkpoint(final,state,digest,p,lineage)
    else:
        saved,fp,fl=io.load_checkpoint(final,digest)
        if fp['stage']!=len(cfg['schedule']) or fp['d_steps']!=p['d_steps'] or fl!=lineage:
            raise ValueError('Final checkpoint/progress mismatch')
        state=saved
    return state,p,lineage


def chi_difference(a,b):
    def delta(key,identity,value):
        old={identity(r):r[value] for r in a[key]}
        if set(old)!={identity(r) for r in b[key]}:raise ValueError('Observable labels changed across chi')
        return max(abs(r[value]-old[identity(r)]) for r in b[key])
    return dict(delta_energy=abs(a['energy_total']-b['energy_total']),
        delta_exchange_energy=abs(a['energy_exchange']-b['energy_exchange']),
        delta_sz_max=delta('sites',lambda r:r['site_id'],'sz'),
        delta_bond_max=delta('bonds',lambda r:tuple(sorted((r['site_id_1'],r['site_id_2']))),'energy'))


def process_seed(cfg,seed,folder,digest,device):
    marker=folder/'completed.json'
    if marker.exists():
        done=json.loads(marker.read_text())
        if (done['settings_hash']!=digest or done['state_sha256']!=io.sha256(folder/'final.pt') or
                done['source_sha256']!=io.sha256(folder/'initial.pt')):
            raise ValueError('Completed result/state changed')
        for chi in cfg['chis']:
            m=folder/f'measurements/chi{chi}/observables.json'
            if not m.exists() or json.loads(m.read_text())['measurement_state_sha256']!=done['state_sha256']:
                raise ValueError('Completed measurement missing or changed')
        print(f'SKIP seed={seed}: fixed protocol and chi measurements complete',flush=True);return done
    state,p,lineage=evolve(cfg,seed,folder,digest)
    state_hash=io.sha256(folder/'final.pt');su_ok=p['stage_reports'][-1]['proxy_converged']
    previous=None;checks=[];ctm_ok=True
    for chi in cfg['chis']:
        target=folder/f'measurements/chi{chi}';target.mkdir(parents=True,exist_ok=True)
        obs=measure_snapshot(cfg|dict(chi=chi),state,target,device,state_hash,f'{digest}:chi={chi}',su_ok)
        ctm_ok=ctm_ok and obs['ctm_converged']
        if previous is not None:
            diff=chi_difference(previous,obs)
            checks.append(dict(chi=chi,previous_chi=previous['chi'],**diff,
                passed=bool(previous['ctm_converged'] and obs['ctm_converged'] and
                    diff['delta_energy']<=cfg['chi_energy_tolerance'] and
                    diff['delta_exchange_energy']<=cfg['chi_energy_tolerance'] and
                    diff['delta_sz_max']<=cfg['chi_site_tolerance'] and diff['delta_bond_max']<=cfg['chi_bond_tolerance'])))
        previous=obs
    chi_ok=bool(checks and checks[-1]['passed'])
    io.atomic_json(folder/'chi_convergence.json',dict(checks=checks,passed=chi_ok,
        note='Two chi values passing is a finite numerical check, not proof of the chi-infinity limit.'))
    done=dict(seed=seed,D=cfg['D'],settings_hash=digest,state_sha256=state_hash,
        source_sha256=lineage['initial_sha256'],status='protocol_complete',steps=state.steps,
        su_proxy_converged=su_ok,ctm_converged=bool(ctm_ok),chi_check_passed=chi_ok,
        c3_final=state.c3,c3_reduction=p['reduction'],su_seconds=p['elapsed_su_seconds'],
        total_tau=sum(r['tau'] for r in p['stage_reports']),
        numerically_checked=bool(su_ok and ctm_ok and chi_ok))
    io.atomic_json(marker,done)
    print(f'SEED DONE {seed}: E={previous["energy_total"]:.10f} m={previous["magnetization_normalized"]:.8f} '
          f'SU={su_ok} CTM={ctm_ok} chi_check={chi_ok} C3={state.c3}',flush=True)
    return done


def collect(cfg,out):
    summary=[];arrays={name:[] for _,name in ARRAYS};all_chi=[]
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';row=dict(seed=seed,D=cfg['D'],field=cfg['field'],status='not_started')
        if (folder/'error.json').exists():
            row.update(status='failed',error=json.loads((folder/'error.json').read_text())['error'])
        elif (folder/'completed.json').exists():
            done=json.loads((folder/'completed.json').read_text())
            row.update({k:v for k,v in done.items() if k!='c3_reduction'})
            row['reduction_reason']=(done.get('c3_reduction') or {}).get('reason','not_attempted')
            for chi in cfg['chis']:
                obs=json.loads((folder/f'measurements/chi{chi}/observables.json').read_text())
                scalar={k:v for k,v in obs.items() if k not in io.ARRAY_KEYS}
                all_chi.append(scalar|dict(seed=seed))
                for key,name in ARRAYS:arrays[name].extend(dict(seed=seed,D=cfg['D'],field=cfg['field'],chi=chi,**r) for r in obs[key])
            row.update(scalar)
            row['measurement_settings_hash']=row.pop('settings_hash')
            row['settings_hash']=done['settings_hash']
            row['ctm_converged']=done['ctm_converged']
        elif folder.exists():row['status']='in_progress'
        summary.append(row)
    io.write_csv(out/'summary.csv',summary);io.write_csv(out/'all_chi_measurements.csv',all_chi)
    for name,values in arrays.items():io.write_csv(out/f'{name}.csv',values)
    lines=['Physical-NN SU; random SIX tensors; conditional C3 only; no target magnetization/pattern.',
           'protocol_complete is a budget/completion flag, not a convergence claim.']
    for r in summary:
        lines.append(f"seed={r['seed']} {r['status']}"+(f" Eex={r['energy_exchange']:.10f} m={r['magnetization_normalized']:.9f} "
            f"SU={r['su_proxy_converged']} CTM={r['ctm_converged']} chi={r['chi_check_passed']} C3={r['c3_final']}"
            if 'energy_exchange' in r else ''))
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run(cfg,out):
    validate(cfg);device=resolve_measurement_device(cfg['measurement_device'])
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(cfg['threads']);torch.set_grad_enabled(False)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);digest=fingerprint(cfg)
    with io.writer_lock(out):
        settings=out/'settings.json'
        if settings.exists() and immutable_settings(json.loads(settings.read_text()))!=immutable_settings(cfg):
            raise ValueError('Settings changed; choose a new output directory')
        io.atomic_json(settings,cfg)
        provenance=out/'provenance.json'
        if provenance.exists() and json.loads(provenance.read_text())['fingerprint']!=digest:
            raise ValueError('Algorithm changed since this run; use a new output directory')
        io.atomic_json(provenance,dict(version=VERSION,fingerprint=digest,torch=torch.__version__,
            constraint='No fixed Sz/magnetization/mirror; SIX independent real tensors initially',
            author_code=False,unpublished_choices='Gate order, stage schedule, lambda-compatible gauge reduction'))
        print(f'{VERSION}: D={cfg["D"]}, seeds={cfg["seeds"]}, h={cfg["field"]}, '
              f'chis={cfg["chis"]}, SU=cpu, measurement={device}',flush=True)
        print('Fixed full steps per seed:',sum(round(s['tau']/s['dt']) for s in cfg['schedule']),flush=True)
        collect(cfg,out)
        for seed in cfg['seeds']:
            folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
            io.atomic_json(out/'batch_status.json',dict(status='running',seed=seed))
            try:
                process_seed(cfg,seed,folder,digest,device);(folder/'error.json').unlink(missing_ok=True)
            except Exception as exc:
                io.atomic_json(folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
                print(f'SEED FAILED {seed}: {exc}; checkpoint preserved',flush=True)
                if device.type=='cuda':torch.cuda.empty_cache()
            collect(cfg,out)
        rows=collect(cfg,out);success=all(r['status']=='protocol_complete' for r in rows)
        io.atomic_json(out/'batch_status.json',dict(status='protocol_complete' if success else 'finished_with_errors',
            complete=sum(r['status']=='protocol_complete' for r in rows),total=len(rows),
            numerically_checked=sum(bool(r.get('numerically_checked')) for r in rows)))
        return success
