"""Serial random-start experiments; existing numerical kernels are reused.

Atomic checkpoints resume SU. A pending-measurement record retries exactly
the saved candidate after an interruption, without evolving it again.
Previous measured candidates are kept with their exact state snapshots.
No author data, reference pattern, or previous optimized state is loaded.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import torch
from ..simple_update import BlockState
from ..honeycomb_ctm import layers
from ..geometry import audit
from . import workflow as base
from .workflow_from_state import evolve_to_convergence,validate_config
from .observables import measure_complete
from .device_support import (immutable_settings,resolve_measurement_device,
                             measurement_state,initial_environment,synchronize)

VERSION='blocked_honeycomb_independent_random_batch_v1'
ARRAYS=(('sites','site_magnetization'),('bonds','bond_energies'),
        ('triangles','triangle_observables'),('schmidt_spectra','schmidt_spectra'))


def runtime_settings(config,seed):
    return config | dict(dims=[config['D']],seed=seed,
                         first_D_warmup=config['warmup'],next_D_warmup=[])


def validate_batch(config):
    if not isinstance(config['D'],int) or config['D']<1:
        raise ValueError('D must be a positive integer')
    seeds=config['seeds']
    if (not seeds or len(seeds)!=len(set(seeds)) or
            any(type(s) is not int or not 0<=s<2**63 for s in seeds)):
        raise ValueError('Provide distinct integer seeds in [0, 2**63)')
    validate_config(runtime_settings(config,seeds[0]))
    if config.get('evolution_mode','adaptive') not in ('adaptive','fixed_screen'):
        raise ValueError('Unknown evolution_mode')
    if config.get('evolution_mode')=='fixed_screen':
        if not config.get('fixed_stages'):
            raise ValueError('Fixed screening needs a stage list')
        for stage in config['fixed_stages']:
            if not 0<stage['dt'] or stage['min_tau']!=stage['max_tau'] or stage['min_tau']<=0:
                raise ValueError('Fixed stages must have positive dt and equal min/max tau')


def batch_fingerprint(config):
    return hashlib.sha256(base.canonical_json(dict(version=VERSION,
                         settings=immutable_settings(config))).encode()).hexdigest()


def seed_fingerprint(batch_hash,seed):
    return hashlib.sha256(f'{batch_hash}:seed={seed}'.encode()).hexdigest()


def atomic_copy(source,target):
    temp=target.with_name(target.name+'.tmp')
    shutil.copyfile(source,temp)
    os.replace(temp,target)


def reuse_paths(config,seed,root):
    """Only READ the old experiment. Copy its immutable random source once.

    The old SU checkpoint is atomically replaced by its writer, so copying
    the named checkpoint gives one complete saved version, not live RAM.
    Stop the old job before switching launchers to avoid wasting compute.
    """
    if config.get('evolution_mode')!='fixed_screen' or not config.get('reuse_long_run'):
        return None
    if ((root/'completed.json').exists() or (root/'pending_measurement.json').exists()
            or (root/f'D{config["D"]}'/'checkpoint.pt').exists()):
        return None  # committed screen state is independent of the old folder
    old=Path(config['reuse_long_run'])
    if not old.is_absolute():
        old=Path(__file__).resolve().parents[1]/old
    settings_file=old/'settings.json'
    if not settings_file.exists():
        return None
    old_config=json.loads(settings_file.read_text())
    for key in ('D','field','c3','second_order'):
        if old_config[key]!=config[key]:
            raise ValueError(f'Old experiment has different {key}; cannot reuse its states')
    if seed not in old_config['seeds']:
        return None
    old_seed=old/f'seed{seed}'
    source=old_seed/'random_source.pt'
    if not source.exists():
        return None
    record=json.loads((old_seed/'initialization.json').read_text())
    old_hash=seed_fingerprint(batch_fingerprint(old_config),seed)
    if record['settings_hash']!=old_hash or base.sha256(source)!=record['source_sha256']:
        raise ValueError('Old random source does not match its recorded fingerprint')
    new_source=root/'random_source.pt'
    if not new_source.exists():
        atomic_copy(source,new_source)
    if base.sha256(new_source)!=record['source_sha256']:
        if (root/'initialization.json').exists():
            return None  # this screen already committed its own random source
        raise ValueError('Old and new random source differ')
    return old_seed,old_hash


def fixed_evolution(config,seed,root,folder,settings_hash,source,reuse):
    """Fixed number of fresh steps; optionally measure a longer saved state.

    Imported candidates retain their actual old steps/tau and are explicitly
    marked. They are never relabelled as a fresh 9000-step state.
    """
    requested=sum(round(s['max_tau']/s['dt']) for s in config['fixed_stages'])
    if not (folder/'checkpoint.pt').exists() and reuse is not None:
        old_seed,old_hash=reuse
        old_checkpoint=old_seed/f'D{config["D"]}'/'checkpoint.pt'
        if old_checkpoint.exists():
            copied=folder/'imported_checkpoint.pt'
            atomic_copy(old_checkpoint,copied)
            state,progress,lineage=base.load_checkpoint(copied,old_hash)
            if lineage.get('initial_sha256')!=base.sha256(source):
                raise ValueError('Old checkpoint descends from a different random source')
            if progress['d_steps']>=requested:
                if not all(bool(torch.isfinite(t).all()) for t in state.A+state.B+
                           [w for row in state.weights for w in row]):
                    raise ValueError('Imported checkpoint contains nonfinite tensors')
                # Include an unfinished fine segment in the recorded tau.
                reports=[dict(r) for r in progress['stage_reports']]
                partial=progress['stage_steps']
                su_ok=bool(reports and reports[-1]['proxy_converged'] and partial==0 and
                           progress['stage']>len(json.loads((old_seed.parent/'settings.json').read_text())['warmup']))
                if partial:
                    old_config=json.loads((old_seed.parent/'settings.json').read_text())
                    stages=old_config['warmup']
                    dt=stages[progress['stage']]['dt'] if progress['stage']<len(stages) else old_config['fine_stage']['dt']
                    reports.append(dict(stage=progress['stage']+1,dt=dt,steps=partial,tau=partial*dt,
                                        proxy_converged=False,**(progress.get('last_rates') or {})))
                progress=progress | dict(stage_reports=reports,imported_steps=progress['d_steps'],
                                         screening_added_steps=0)
                atomic_copy(copied,folder/'final.pt')
                print(f'seed={seed}: reuse saved {progress["d_steps"]} steps; NO more SU, measure now',flush=True)
                return state,progress,lineage,su_ok
    state,progress,lineage=base.evolve_dimension(runtime_settings(config,seed) |
        dict(initial_D4_checkpoint=str(source),stages=config['fixed_stages']),folder,settings_hash,None)
    progress=progress | dict(imported_steps=0,screening_added_steps=progress['d_steps'])
    return state,progress,lineage,bool(progress['stage_reports'][-1]['proxy_converged'])


def random_source(config,seed,root,settings_hash):
    source=root/'random_source.pt'
    record=root/'initialization.json'
    if record.exists() and not source.exists():
        raise FileNotFoundError(f'Original random initial state is missing: {source}')
    if not source.exists():
        state=BlockState.random(config['D'],config['field'],seed,c3=config['c3'])
        temp=source.with_name(source.name+'.tmp')
        state.save(temp,initialization='independent_random_at_target_D',settings_hash=settings_hash)
        os.replace(temp,source)
    digest=base.sha256(source)
    if record.exists():
        info=json.loads(record.read_text())
        if info['settings_hash']!=settings_hash or info['source_sha256']!=digest:
            raise ValueError('Random source or experiment settings changed')
    state=BlockState.load(source)
    if (state.D!=config['D'] or state.seed!=seed or state.field!=config['field']
            or state.c3!=config['c3'] or state.steps!=0):
        raise ValueError('Random initial state has incompatible metadata')
    if not record.exists():
        base.atomic_json(record,dict(seed=seed,D=state.D,settings_hash=settings_hash,
            source_sha256=digest,initialization='independent_random_at_target_D',
            c3_imposed=state.c3,magnetization_fixed=False,mirror_imposed=False))
    return source,digest


def measure_snapshot(config,state,folder,device,state_hash,settings_hash,su_converged):
    marker=folder/'observables.json'
    if marker.exists():
        obs=json.loads(marker.read_text())
        if obs['measurement_state_sha256']!=state_hash or obs['settings_hash']!=settings_hash:
            raise ValueError('Saved measurement belongs to a different state/settings')
        return obs
    print(f'seed={state.seed}: ONE final CTMRG + complete observables; '
          f'SU_proxy={su_converged}',flush=True)
    synchronize(device)
    start=time.perf_counter()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    measured=measurement_state(state,device)
    env=initial_environment(state.D,device)
    history=[]
    def callback(row):
        row=row | dict(residual=row['residual'] if math.isfinite(row['residual']) else None)
        history.append(row)
        base.write_csv(folder/'ctm_history.csv',history)
        print(f'seed={state.seed} CTMRG {row["iteration"]}/{config["ctm_max"]} '
              f'residual={row["residual"]}',flush=True)
    env.run(layers(measured),config['chi'],config['ctm_max'],config['ctm_tolerance'],
            callback,min_iterations=config['ctm_min'])
    obs=measure_complete(measured,env,lambda msg:print(f'seed={state.seed}: {msg}',flush=True),
                         max_corner_cache_bytes=config['max_corner_cache_mb']*1024**2)
    synchronize(device)
    obs.update(chi=config['chi'],measurement_seconds=time.perf_counter()-start,
               measurement_state_sha256=state_hash,settings_hash=settings_hash,
               su_device='cpu',measurement_device=str(device),
               gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.)
    del measured,env
    if device.type=='cuda':
        torch.cuda.empty_cache()
    for key,name in ARRAYS:
        base.write_csv(folder/f'{name}.csv',obs[key])
    base.atomic_json(marker,obs)  # complete marker written LAST
    return obs


def process_seed(config,seed,root,settings_hash,device):
    root.mkdir(exist_ok=True)
    reuse=reuse_paths(config,seed,root)
    source,source_hash=random_source(config,seed,root,settings_hash)
    folder=root/f'D{config["D"]}'
    folder.mkdir(exist_ok=True)
    complete=root/'completed.json'
    pending_file=root/'pending_measurement.json'
    if complete.exists():
        previous=json.loads(complete.read_text())
        if previous['settings_hash']!=settings_hash or previous['source_sha256']!=source_hash:
            raise ValueError('Completed seed source/settings mismatch')
        snapshot=root/previous['measurement_folder']/'state.pt'
        if base.sha256(snapshot)!=previous['state_sha256']:
            raise ValueError('Previously measured state snapshot changed')
        # A crash can leave pending after the completed commit. It refers
        # to an already committed attempt and must not trigger remeasurement.
        if pending_file.exists():
            pending=json.loads(pending_file.read_text())
            if pending['measurement_folder']==previous['measurement_folder']:
                pending_file.unlink()
        if previous['status'] in ('complete','screened','screened_CTM_unconverged'):
            if base.sha256(folder/'final.pt')!=previous['state_sha256']:
                raise ValueError('Completed final state changed')
            print(f'SKIP seed={seed}: recorded protocol and measurement already complete ({previous["status"]})',flush=True)
            return previous
    if pending_file.exists():
        pending=json.loads(pending_file.read_text())
        if pending['settings_hash']!=settings_hash or pending['source_sha256']!=source_hash:
            raise ValueError('Pending measurement source/settings mismatch')
        print(f'RESUME seed={seed}: unfinished measurement only; SU is not repeated',flush=True)
    else:
        if config.get('evolution_mode')=='fixed_screen':
            state,progress,lineage,su_ok=fixed_evolution(config,seed,root,folder,settings_hash,source,reuse)
        else:
            state,progress,lineage,su_ok=evolve_to_convergence(runtime_settings(config,seed),
                                                             folder,settings_hash,None,source)
        if lineage.get('initial_sha256')!=source_hash:
            raise ValueError('Evolved state does not descend from this seed initial state')
        attempts=folder/'measurements'
        attempts.mkdir(exist_ok=True)
        index=max([int(p.name[7:]) for p in attempts.glob('attempt*') if p.is_dir()]+[0])+1
        measured_folder=attempts/f'attempt{index:04d}'
        measured_folder.mkdir()
        atomic_copy(folder/'final.pt',measured_folder/'state.pt')
        pending=dict(seed=seed,D=config['D'],settings_hash=settings_hash,source_sha256=source_hash,
                     state_sha256=base.sha256(measured_folder/'state.pt'),
                     measurement_folder=measured_folder.relative_to(root).as_posix(),attempt=index,
                     su_proxy_converged=su_ok,su_seconds=progress['elapsed_su_seconds'],
                     total_tau=sum(r['tau'] for r in progress['stage_reports']),
                     last_SU=progress['stage_reports'][-1])
        if config.get('evolution_mode')=='fixed_screen':
            pending.update(evolution_policy='fixed_screen',imported_steps=progress['imported_steps'],
                           screening_added_steps=progress['screening_added_steps'])
        base.atomic_json(measured_folder/'candidate.json',pending)
        base.atomic_json(pending_file,pending)
    measured_folder=root/pending['measurement_folder']
    snapshot=measured_folder/'state.pt'
    if base.sha256(snapshot)!=pending['state_sha256']:
        raise ValueError('Pending measured snapshot changed')
    state=BlockState.load(snapshot)
    obs=measure_snapshot(config,state,measured_folder,device,pending['state_sha256'],
                         settings_hash,pending['su_proxy_converged'])
    if config.get('evolution_mode')=='fixed_screen':
        status='screened' if obs['ctm_converged'] else 'screened_CTM_unconverged'
    else:
        status=('complete' if obs['ctm_converged'] else 'needs_CTM_check') if pending['su_proxy_converged'] else 'needs_more_SU'
    done=pending | dict(status=status,ctm_converged=bool(obs['ctm_converged']))
    base.atomic_json(complete,done)
    pending_file.unlink(missing_ok=True)
    print(f'SEED DONE {seed}: status={status}; Eex={obs["energy_exchange"]:.10f}; '
          f'm/ms={obs["magnetization_normalized"]:.9f}',flush=True)
    return done


def collect_batch(config,out):
    summary=[]
    arrays={name:[] for _,name in ARRAYS}
    history=[]
    for seed in config['seeds']:
        root=out/f'seed{seed}'
        complete=root/'completed.json'
        error=root/'error.json'
        row=dict(seed=seed,D=config['D'],field=config['field'],chi=config['chi'],status='not_started')
        if complete.exists():
            done=json.loads(complete.read_text())
            obs=json.loads((root/done['measurement_folder']/'observables.json').read_text())
            row.update({k:v for k,v in obs.items() if k not in base.ARRAY_KEYS})
            row.update({k:v for k,v in done.items() if k!='last_SU'})
            row.update({f'last_SU_{k}':v for k,v in done['last_SU'].items()})
            for key,name in ARRAYS:
                arrays[name].extend(dict(seed=seed,D=config['D'],field=config['field'],chi=config['chi'],
                                         candidate_status=done['status'],**r) for r in obs[key])
        elif error.exists():
            row.update(status='failed',error=json.loads(error.read_text())['error'])
        elif root.exists():
            row['status']='in_progress'
        if error.exists() and complete.exists():
            row['latest_retry_error']=json.loads(error.read_text())['error']
        summary.append(row)
        for marker in sorted(root.glob(f'D{config["D"]}/measurements/attempt*/observables.json')):
            obs=json.loads(marker.read_text())
            candidate=json.loads((marker.parent/'candidate.json').read_text())
            history_row={k:v for k,v in obs.items() if k not in base.ARRAY_KEYS}
            history_row.update({k:v for k,v in candidate.items() if k!='last_SU'})
            history.append(history_row)
    base.write_csv(out/'summary.csv',summary)
    for name,rows in arrays.items():
        base.write_csv(out/f'{name}.csv',rows)
    base.write_csv(out/'all_measurements.csv',history)
    lines=['Independent random starts at fixed D; no target magnetization or mirror imposed.',
           'SU/CTM convergence flags must be checked before comparing energies or patterns.']
    for row in summary:
        values=(f'Eex={row["energy_exchange"]:.10f} m={row["magnetization_normalized"]:.9f} '
                f'SU={row["su_proxy_converged"]} CTM={row["ctm_converged"]}' if 'energy_exchange' in row else '')
        lines.append(f'seed={row["seed"]} {row["status"]} {values}')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run_random_batch(config,out):
    validate_batch(config)
    device=resolve_measurement_device(config['measurement_device'])
    if device.type=='cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(config['threads'])
    torch.set_grad_enabled(False)
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    digest=batch_fingerprint(config)
    with base.writer_lock(out):
        settings_file=out/'settings.json'
        if settings_file.exists() and immutable_settings(json.loads(settings_file.read_text()))!=immutable_settings(config):
            raise ValueError('Batch settings changed: choose a new OUT (measurement device alone may change)')
        base.atomic_json(settings_file,config)
        package=Path(__file__).resolve().parents[1]
        base.atomic_json(out/'provenance.json',dict(version=VERSION,torch_version=torch.__version__,
            batch_hash=digest,initialization='independent_random_at_target_D',
            sources={str(p.relative_to(package)):base.sha256(p) for p in
                     list(package.glob('*.py'))+list((package/'server').glob('*.py'))},
            convention='H=sum S.S-h sum Sz; real float64; m/ms=2 mean Sz',
            c3_imposed=config['c3'],mirror_imposed=False,magnetization_fixed=False))
        print(f'{VERSION}: D={config["D"]}, seeds={config["seeds"]}, h={config["field"]}, '
              f'chi={config["chi"]}, SU=cpu, measurements={device}',flush=True)
        print(json.dumps(audit()),flush=True)
        collect_batch(config,out)
        for i,seed in enumerate(config['seeds'],1):
            root=out/f'seed{seed}'
            base.atomic_json(out/'batch_status.json',dict(status='running',current_seed=seed,index=i,
                                                         total=len(config['seeds'])))
            print(f'\n===== seed={seed} ({i}/{len(config["seeds"])}) =====',flush=True)
            try:
                process_seed(config,seed,root,seed_fingerprint(digest,seed),device)
                (root/'error.json').unlink(missing_ok=True)
            except Exception as exc:
                root.mkdir(exist_ok=True)
                base.atomic_json(root/'error.json',dict(seed=seed,error=str(exc),traceback=traceback.format_exc()))
                print(f'SEED FAILED {seed}: {exc}; saved error, continuing other seeds',flush=True)
                if device.type=='cuda':
                    torch.cuda.empty_cache()
            collect_batch(config,out)
        rows=collect_batch(config,out)
        finished=('screened','screened_CTM_unconverged') if config.get('evolution_mode')=='fixed_screen' else ('complete',)
        success=all(r['status'] in finished and 'latest_retry_error' not in r for r in rows)
        base.atomic_json(out/'batch_status.json',dict(status='all_complete' if success else 'finished_with_unfinished_seeds',
            protocol=config.get('evolution_mode','adaptive'),total=len(rows),complete=sum(r['status'] in finished for r in rows),
            unfinished=[r['seed'] for r in rows if r['status'] not in finished or 'latest_retry_error' in r]))
        label='all_screened' if config.get('evolution_mode')=='fixed_screen' else 'all_converged'
        print(f'BATCH FINISHED: {out}/summary.csv; {label}={success}',flush=True)
        return success
