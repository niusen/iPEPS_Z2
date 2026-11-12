"""Read an old stopped chain and continue every D with short SU then measurement.

The existing five-spin SU kernel is retained for this existing branch.
No paper_method/NN kernel is substituted into an already evolved state.
The old output is read only; completed observations and the selected state
are frozen in a NEW output before any evolution. No automatic fine chunks.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import torch
from ..simple_update import BlockState
from . import workflow as base
from .workflow_from_state import fingerprint as old_fingerprint
from .workflow_random_seeds import measure_snapshot,atomic_copy,ARRAYS
from .device_support import immutable_settings,resolve_measurement_device

VERSION='bounded_resume_existing_five_spin_chain_v1'
PACKAGE=Path(__file__).resolve().parents[1]


def source_root(cfg):
    path=Path(cfg['old_output'])
    return path.resolve() if path.is_absolute() else (PACKAGE/path).resolve()


def validate(cfg,out):
    if not cfg['dims'] or cfg['dims']!=list(range(cfg['source_D'],cfg['dims'][-1]+1)):
        raise ValueError('dims must begin at source_D and increase consecutively')
    if cfg['source_D']<1 or not cfg['stages'] or not cfg['source_stages']:raise ValueError('Invalid source or stages')
    if not math.isfinite(cfg['field']):raise ValueError('Invalid field')
    for s in cfg['stages']+cfg['source_stages']:
        if not (0<s['dt'] and 0<s['min_tau']<=s['max_tau']):raise ValueError('Invalid SU budget')
    if not 7<=cfg['ctm_min']<=cfg['ctm_max']<=50:raise ValueError('Use 7..50 CTM iterations')
    if min(cfg[k] for k in ('chi','threads','checkpoint_steps','stable_checks','monitor_min_steps'))<1:
        raise ValueError('Invalid counts')
    if min(cfg[k] for k in ('monitor_tau','lambda_drift_tolerance','rho_drift_tolerance','ctm_tolerance'))<=0:
        raise ValueError('Invalid tolerances')
    old,new=source_root(cfg),Path(out).resolve()
    if old==new or old in new.parents or new in old.parents:
        raise ValueError('Old and new output trees must be separate')


def fingerprint(cfg):
    return hashlib.sha256(base.canonical_json(dict(version=VERSION,settings=immutable_settings(cfg))).encode()).hexdigest()


@contextmanager
def old_reader_lock(root):
    """Do not create/change old lock/owner files; reject a live old writer."""
    path=root/'.writer.lock'
    if not path.exists():
        yield;return
    with path.open('rb') as handle:
        try:
            if __import__('os').name=='nt':
                import msvcrt
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBRLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Old task is still writing. Stop its PID first; do not delete its lock.') from exc
        try:yield
        finally:
            if __import__('os').name=='nt':
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle,fcntl.LOCK_UN)


def inspect_old(cfg):
    root=source_root(cfg)
    old=json.loads((root/'settings.json').read_text())
    for key in ('field','seed','c3','second_order'):
        if old[key]!=cfg[key]:raise ValueError(f'Old state uses a different {key}')
    if old['initial_state_sha256']!=cfg['expected_initial_sha256']:
        raise ValueError('Old chain is from a different original D5 source')
    old_hash=old_fingerprint(old)
    options=[]
    for name in ('final.pt','checkpoint.pt'):
        path=root/f'D{cfg["source_D"]}'/name
        if not path.exists():continue
        payload=torch.load(path,map_location='cpu',weights_only=False)
        m=payload.get('metadata',{});p=m.get('progress',{})
        if m.get('server_version')!=base.VERSION or m.get('settings_hash')!=old_hash:
            raise ValueError(f'Old checkpoint/settings mismatch: {path}')
        if (payload['A'][0].shape[1]!=cfg['source_D'] or payload['field']!=cfg['field'] or
                payload['seed']!=cfg['seed'] or payload['c3']!=cfg['c3']):
            raise ValueError('Old tensor D/field/seed/C3 does not match this continuation')
        tensors=payload['A']+payload['B']+[w for row in payload['weights'] for w in row]
        if any(not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Old checkpoint has nonfinite tensors')
        options.append((int(p['d_steps']),name=='final.pt',path,p,m.get('lineage',{})))
    if not options:raise FileNotFoundError(f'No saved D={cfg["source_D"]} state in {root}')
    _,_,path,p,lineage=max(options,key=lambda a:(a[0],a[1]))
    prior=[]
    for D in old['dims']:
        if D>=cfg['source_D']:continue
        folder=root/f'D{D}';done_file=folder/'completed.json'
        if not done_file.exists():continue
        done=json.loads(done_file.read_text())
        if not done.get('ready_for_next_D'):continue
        if done['settings_hash']!=old_hash or base.sha256(folder/'final.pt')!=done['state_sha256']:
            raise ValueError(f'Prior D={D} result integrity failed')
        if not (folder/'observables.json').exists():raise ValueError('Completed old observation is missing')
        prior.append(D)
    finished=(p.get('stage_steps',0)==0 and bool(p['stage_reports']))
    su_ok=bool(p['stage_reports'][-1]['proxy_converged']) if finished else False
    return dict(path=path,sha256=base.sha256(path),steps=int(p['d_steps']),
        old_settings_hash=old_hash,old_settings=old,prior_dims=prior,su_proxy_converged=su_ok,
        stage_reports=p['stage_reports'],elapsed_su_seconds=float(p['elapsed_su_seconds']),lineage=lineage)


def freeze_source(cfg,out,digest):
    marker=out/'source_manifest.json';snapshot=out/'source_snapshot.pt'
    if marker.exists():
        manifest=json.loads(marker.read_text())
        if manifest['settings_hash']!=digest or base.sha256(snapshot)!=manifest['source_sha256']:
            raise ValueError('Frozen source/settings changed')
        for item in manifest['archived_files']:
            if base.sha256(out/item['file'])!=item['sha256']:raise ValueError('Archived result changed')
        return manifest
    with old_reader_lock(source_root(cfg)):
        old=inspect_old(cfg)
        atomic_copy(old['path'],snapshot)
        if base.sha256(snapshot)!=old['sha256'] or base.sha256(old['path'])!=old['sha256']:
            raise ValueError('Source changed while being copied')
        archived=[]
        for D in old['prior_dims']:
            target=out/f'prior_results/D{D}';target.mkdir(parents=True,exist_ok=True)
            for name in ('completed.json','observables.json'):
                original=source_root(cfg)/f'D{D}'/name
                atomic_copy(original,target/name)
                archived.append(dict(file=(target/name).relative_to(out).as_posix(),sha256=base.sha256(target/name)))
        manifest=dict(settings_hash=digest,source_D=cfg['source_D'],source_file=str(old['path']),
            source_sha256=old['sha256'],inherited_steps=old['steps'],
            source_su_proxy_converged=old['su_proxy_converged'],old_settings_hash=old['old_settings_hash'],
            source_stage_reports=old['stage_reports'],inherited_su_seconds=old['elapsed_su_seconds'],
            source_lineage=old['lineage'],prior_dims=old['prior_dims'],archived_files=archived,
            original_D5_sha256=cfg['expected_initial_sha256'])
        base.atomic_json(marker,manifest) # all source copies committed before this marker
    return manifest


def completed_check(folder,digest,parent_hash):
    path=folder/'completed.json'
    if not path.exists():return None
    done=json.loads(path.read_text());obs_file=folder/'observables.json'
    if (done['settings_hash']!=digest or done['parent_sha256']!=parent_hash or
            done['state_sha256']!=base.sha256(folder/'final.pt') or not obs_file.exists()):
        raise ValueError('Completed continuation state/parent/settings changed')
    obs=json.loads(obs_file.read_text())
    if (obs['measurement_state_sha256']!=done['state_sha256'] or
            obs['settings_hash']!=f'{digest}:D={done["D"]}'):
        raise ValueError('Measurement/state/settings mismatch')
    return done


def evolve_bounded(cfg,folder,digest,parent,source,source_dimension):
    stages=cfg['source_stages'] if source_dimension else cfg['stages']
    final=folder/'final.pt'
    parent_hash=base.sha256(parent) if parent else None
    if final.exists():
        state,p,lineage=base.load_checkpoint(final,digest)
        if (p['stage']!=len(stages) or p['stage_steps']!=0 or lineage.get('parent_sha256')!=parent_hash or
            (source_dimension and lineage.get('initial_sha256')!=base.sha256(source))):
            raise ValueError('Final state/progress/lineage changed')
        print(f'REUSE finished SU at {folder.name}; measurement only, no extra steps',flush=True)
        return state,p,lineage
    runtime=cfg|dict(stages=stages,initial_D4_checkpoint=str(source))
    return base.evolve_dimension(runtime,folder,digest,parent)


def collect(out):
    rows=[];arrays={name:[] for _,name in ARRAYS}
    folders=list((out/'prior_results').glob('D*'))+list(out.glob('D*'))
    for folder in sorted(folders,key=lambda p:int(p.name[1:])):
        if not (folder/'completed.json').exists():continue
        done=json.loads((folder/'completed.json').read_text())
        obs=json.loads((folder/'observables.json').read_text())
        row={k:v for k,v in obs.items() if k not in base.ARRAY_KEYS}
        row.update(done)
        row['result_origin']='previous_completed' if folder.parent.name=='prior_results' else 'bounded_resume'
        rows.append(row)
        for key,name in ARRAYS:
            arrays[name].extend(dict(D=obs['D'],field=obs['field'],chi=obs['chi'],**r) for r in obs[key])
    base.write_csv(out/'summary.csv',rows)
    for name,data in arrays.items():base.write_csv(out/f'{name}.csv',data)
    lines=['Old five-spin SU branch; short bounded continuation, no intermediate CTMRG/energy/magnetization.',
           'Budget completion is NOT convergence; SU/CTM flags are recorded independently.']
    for r in rows:
        lines.append(f'D={r["D"]} Eex={r["energy_exchange"]:.10f} Etot={r["energy_total"]:.10f} '
            f'm={r["magnetization_normalized"]:.9f} SU={r["su_proxy_converged"]} '
            f'CTM={r["ctm_converged"]} added_steps={r.get("added_steps_this_job",0)} origin={r["result_origin"]}')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


def preflight(cfg,out):
    validate(cfg,out);out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    with base.writer_lock(out):
        if (out/'settings.json').exists() and immutable_settings(json.loads((out/'settings.json').read_text()))!=immutable_settings(cfg):
            raise ValueError('Settings changed; choose a new OUT')
        if (out/'source_manifest.json').exists():
            return freeze_source(cfg,out,fingerprint(cfg))
        with old_reader_lock(source_root(cfg)):
            source=inspect_old(cfg)
            return dict(source_file=str(source['path']),inherited_steps=source['steps'],source_sha256=source['sha256'])


def run(cfg,out):
    validate(cfg,out);device=resolve_measurement_device(cfg['measurement_device'])
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(cfg['threads']);torch.set_grad_enabled(False)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);digest=fingerprint(cfg)
    with base.writer_lock(out):
        settings=out/'settings.json'
        if settings.exists() and immutable_settings(json.loads(settings.read_text()))!=immutable_settings(cfg):
            raise ValueError('New continuation settings changed; use a new OUT')
        manifest=freeze_source(cfg,out,digest);base.atomic_json(settings,cfg)
        print(f'{VERSION}: import D={cfg["source_D"]} at {manifest["inherited_steps"]} old steps; '
            f'short fine continuation, then D={cfg["dims"][1:]} from each preceding D.',flush=True)
        print('Source-D additional-step cap:',sum(math.ceil(s['max_tau']/s['dt']-1e-10) for s in cfg['source_stages']),flush=True)
        print('Per-new-D maximum steps:',sum(math.ceil(s['max_tau']/s['dt']-1e-10) for s in cfg['stages']),
            '; no repeated fine chunks; measurement=',device,flush=True)
        collect(out);parent=None
        for D in cfg['dims']:
            folder=out/f'D{D}';folder.mkdir(exist_ok=True)
            parent_hash=base.sha256(parent) if parent else manifest['source_sha256']
            if completed_check(folder,digest,parent_hash):
                print(f'SKIP D={D}: budget and final measurement already complete',flush=True)
                parent=folder/'final.pt';continue
            print(f'D={D}: '+('continue saved same-D tensors at fine dt' if D==cfg['source_D'] else
                  f'load D={D-1}, exact zero padding')+'; short SU begins',flush=True)
            state,p,lineage=evolve_bounded(cfg,folder,digest,parent,out/'source_snapshot.pt',D==cfg['source_D'])
            su_ok=p['stage_reports'][-1]['proxy_converged'];added=p['d_steps']
            reports=p['stage_reports'];elapsed=p['elapsed_su_seconds'];final=folder/'final.pt'
            state_hash=base.sha256(final)
            print(f'D={D}: evolution finished; SU_proxy={su_ok}. ONE final CTMRG/complete measurement.',flush=True)
            obs=measure_snapshot(cfg,state,folder,device,state_hash,f'{digest}:D={D}',su_ok)
            done=dict(D=D,status='fixed_budget_measured',settings_hash=digest,state_sha256=state_hash,
                parent_sha256=parent_hash,source_state_sha256=manifest['source_sha256'],
                original_D5_sha256=manifest['original_D5_sha256'],su_proxy_converged=su_ok,
                added_steps_this_job=added,inherited_source_steps=manifest['inherited_steps'] if D==cfg['source_D'] else 0,
                steps=(manifest['inherited_steps'] if D==cfg['source_D'] else 0)+added,
                su_seconds_this_job=elapsed,ctm_converged=bool(obs['ctm_converged']),
                numerically_checked=bool(su_ok and obs['ctm_converged']),
                actual_tau=sum(r['tau'] for r in reports))
            base.atomic_json(folder/'completed.json',done);collect(out)
            print(f'DONE D={D}: added_steps={added} Eex={obs["energy_exchange"]:.10f} '
                f'Etot={obs["energy_total"]:.10f} m/ms={obs["magnetization_normalized"]:.9f} '
                f'SU_proxy={su_ok} CTM={obs["ctm_converged"]}',flush=True)
            base.atomic_json(out/'chain_status.json',dict(status='dimension_measured',D=D,
                su_proxy_converged=su_ok,ctm_converged=obs['ctm_converged']))
            parent=final
        base.atomic_json(out/'chain_status.json',dict(status='fixed_protocol_complete',dims=cfg['dims']))
    return True
