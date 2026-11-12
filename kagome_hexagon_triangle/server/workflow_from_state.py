"""Convergence-gated chain; new orchestration, unchanged numerical kernels.

The existing checkpoint/evolution routine is called with progressively
longer stage prefixes. Every call resumes the same atomic checkpoint; a
new fine stage is appended only when the previous one reached its cap.
This preserves the complete SU monitor history without rerunning warm-up.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from ..simple_update import BlockState
from ..honeycomb_ctm import layers
from ..geometry import audit
from . import workflow as base
from .observables import measure_complete
from .device_support import (resolve_measurement_device,immutable_settings,
                             measurement_state,initial_environment,synchronize)

VERSION = 'blocked_honeycomb_seeded_convergence_chain_v1'


def validate_source(config):
    path = Path(config['initial_state_file'])
    if not path.is_absolute():
        path = Path(__file__).parent/path
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved initial state: {path}. Upload server/initial_states/.')
    digest = base.sha256(path)
    if digest != config['initial_state_sha256']:
        raise ValueError(f'Initial-state SHA256 mismatch: {path}. Refusing a different branch.')
    state = BlockState.load(path)
    if (state.D != config['dims'][0] or state.field != config['field']
            or state.c3 != config['c3'] or state.seed != config['seed']):
        raise ValueError('Saved initial state has incompatible D, field, C3 or seed')
    for group,p in ((state.A,64),(state.B,8)):
        if len(group) != 3:
            raise ValueError('Initial state must contain three tensors of each block kind')
        for t in group:
            if (tuple(t.shape) != (p,state.D,state.D,state.D) or t.dtype != torch.float64
                    or not torch.isfinite(t).all()):
                raise ValueError('Invalid initial tensor shape, precision or entries')
    if len(state.weights) != 3 or any(len(row) != 3 for row in state.weights):
        raise ValueError('Invalid initial weight layout')
    for row in state.weights:
        for w in row:
            if (tuple(w.shape) != (state.D,) or not torch.isfinite(w).all()
                    or bool((w < 0).any()) or abs(float(w.square().sum())-1) > 1e-10):
                raise ValueError('Invalid initial Schmidt weights')
    return path.resolve(),digest


def validate_config(config):
    dims = config['dims']
    if not dims or any(b != a+1 for a,b in zip(dims,dims[1:])):
        raise ValueError('dims must be consecutive increasing D values')
    if (not 1 <= config['ctm_min'] <= config['ctm_max'] <= 50
            or min(config['checkpoint_steps'],config['monitor_min_steps'],config['stable_checks'],
                   config['threads'],config['chi'],config['max_fine_chunks']) < 1
            or config['monitor_tau'] <= 0):
        raise ValueError('Invalid settings; CTMRG max must not exceed 50')
    for stage in config['first_D_warmup']+config['next_D_warmup']+[config['fine_stage']]:
        if not (0 < stage['dt'] and 0 < stage['min_tau'] <= stage['max_tau']):
            raise ValueError('Invalid SU stage')
    if min(config['lambda_drift_tolerance'],config['rho_drift_tolerance']) < 0:
        raise ValueError('Negative SU convergence tolerance')


def fingerprint(config):
    return hashlib.sha256(base.canonical_json(dict(version=VERSION,
                         settings=immutable_settings(config))).encode()).hexdigest()


def evolve_to_convergence(config,folder,settings_hash,parent,source):
    warm = config['first_D_warmup'] if parent is None else config['next_D_warmup']
    runtime = config | dict(initial_D4_checkpoint=str(source))
    checkpoint = folder/'checkpoint.pt'
    if checkpoint.exists():
        state,progress,lineage = base.load_checkpoint(checkpoint,settings_hash)
        expected_parent = base.sha256(parent) if parent else None
        if lineage.get('parent_sha256') != expected_parent:
            raise ValueError('Checkpoint parent state changed')
    else:
        state,progress,lineage = base.evolve_dimension(runtime | dict(stages=warm),folder,settings_hash,parent)
    if progress['stage'] < len(warm):
        state,progress,lineage = base.evolve_dimension(runtime | dict(stages=warm),folder,settings_hash,parent)
    if progress['stage'] > len(warm) and progress['stage_reports'][-1]['proxy_converged']:
        return state,progress,lineage,True
    for attempt in range(config['max_fine_chunks']):
        # stage is the number already FINISHED. A partially executed next
        # stage remains in stage_steps and is resumed, not restarted.
        total_stages = progress['stage']+1
        stages = warm+[config['fine_stage']]*(total_stages-len(warm))
        print(f'D={folder.name[1:]} fine continuation {attempt+1}/{config["max_fine_chunks"]} '
              f'this submission; overall fine stage {total_stages-len(warm)}',flush=True)
        state,progress,lineage = base.evolve_dimension(runtime | dict(stages=stages),folder,settings_hash,parent)
        if progress['stage_reports'][-1]['proxy_converged']:
            return state,progress,lineage,True
    return state,progress,lineage,False


def measure_final(config,state,folder,device):
    D = state.D
    print(f'D={D}: SU proxy passed; begin ONE final CTMRG and complete measurement',flush=True)
    synchronize(device)
    t0 = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    measured_state = measurement_state(state,device)
    env = initial_environment(D,device)
    history = []
    def callback(row):
        row = row | dict(residual=row['residual'] if math.isfinite(row['residual']) else None)
        history.append(row)
        base.write_csv(folder/'ctm_history.csv',history)
        print(f'D={D} CTMRG {row["iteration"]}/{config["ctm_max"]} residual={row["residual"]}',flush=True)
    env.run(layers(measured_state),config['chi'],config['ctm_max'],config['ctm_tolerance'],
            callback,min_iterations=config['ctm_min'])
    obs = measure_complete(measured_state,env,lambda msg:print(f'D={D}: {msg}',flush=True),
                           max_corner_cache_bytes=config['max_corner_cache_mb']*1024**2)
    synchronize(device)
    obs.update(chi=config['chi'],measurement_seconds=time.perf_counter()-t0,
               su_device='cpu',measurement_device=str(device),
               gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.)
    del measured_state,env
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    base.atomic_json(folder/'observables.json',obs)
    for key,name in (('sites','site_magnetization'),('bonds','bond_energies'),
                     ('triangles','triangle_observables'),('schmidt_spectra','schmidt_spectra')):
        base.write_csv(folder/f'{name}.csv',obs[key])
    return obs


def run_seeded_chain(config,out):
    validate_config(config)
    source,source_hash = validate_source(config)
    device = resolve_measurement_device(config.get('measurement_device','cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(config['threads'])
    torch.set_grad_enabled(False)
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    settings_hash = fingerprint(config)
    with base.writer_lock(out):
        saved = out/'settings.json'
        if saved.exists() and immutable_settings(json.loads(saved.read_text())) != immutable_settings(config):
            raise ValueError('Run settings changed: choose a new OUT; execution device alone may change')
        base.atomic_json(saved,config)
        root = Path(__file__).resolve().parents[1]
        base.atomic_json(out/'provenance.json',dict(version=VERSION,torch_version=torch.__version__,
            settings_hash=settings_hash,initial_state_sha256=source_hash,
            initial_state_file=config['initial_state_file'],
            sources={str(p.relative_to(root)):base.sha256(p) for p in
                     list(root.glob('*.py'))+list((root/'server').glob('*.py'))},
            su_device='cpu',measurement_device=str(device),
            convention='H=sum S.S-h sum Sz; m/ms=2 mean Sz; real float64; C3 imposed'))
        print(f'{VERSION}; D={config["dims"]}; h={config["field"]}; chi={config["chi"]}; '
              f'SU=cpu; measurements={device}; output={out}',flush=True)
        print(f'INITIAL SAVED STATE {source}; SHA256={source_hash}; NO random initialization',flush=True)
        print(json.dumps(audit()),flush=True)
        parent = None
        for D in config['dims']:
            folder = out/f'D{D}'
            folder.mkdir(exist_ok=True)
            completed = folder/'completed.json'
            if completed.exists():
                done = json.loads(completed.read_text())
                if (done['settings_hash'] != settings_hash
                        or done['state_sha256'] != base.sha256(folder/'final.pt')
                        or done.get('parent_sha256') != (base.sha256(parent) if parent else None)
                        or done.get('source_state_sha256') != source_hash):
                    raise ValueError(f'Completed-state lineage/settings mismatch: {folder}')
                if done.get('ready_for_next_D'):
                    print(f'SKIP D={D}: converged SU proxy and final CTM/observables already saved',flush=True)
                    parent = folder/'final.pt'
                    continue
            state,progress,lineage,converged = evolve_to_convergence(config,folder,settings_hash,parent,source)
            if not converged:
                status = dict(status='needs_more_SU',D=D,ready_for_next_D=False,
                              source_state_sha256=source_hash,last_SU=progress['stage_reports'][-1],
                              message='Saved checkpoint. Rerun the same sh to continue fine dt; no later D started.')
                base.atomic_json(out/'chain_status.json',status)
                print(f'STOP D={D}: fine-stage budget exhausted without SU proxy convergence. '
                      'Checkpoint preserved. Rerun the same sh for further fine evolution. No larger D started.',flush=True)
                return False
            obs = measure_final(config,state,folder,device)
            ready = bool(obs['ctm_converged'])
            done = dict(settings_hash=settings_hash,state_sha256=base.sha256(folder/'final.pt'),
                        parent_sha256=lineage.get('parent_sha256'),parent_D=lineage.get('parent_D'),
                        source_state_sha256=source_hash,su_proxy_converged=True,
                        ready_for_next_D=ready,su_seconds=progress['elapsed_su_seconds'],
                        total_tau=sum(r['tau'] for r in progress['stage_reports']))
            base.atomic_json(completed,done)
            base.refresh_tables(out)
            print(f'DONE D={D}: Eex={obs["energy_exchange"]:.10f} Etot={obs["energy_total"]:.10f} '
                  f'm/ms={obs["magnetization_normalized"]:.9f}; SU_proxy=True; CTM={ready}',flush=True)
            if not ready:
                base.atomic_json(out/'chain_status.json',dict(status='needs_CTM_check',D=D,
                    ready_for_next_D=False,message='All observables saved but CTMRG did not converge within 50 iterations. '
                    'No larger D started; rerun retries only measurement.'))
                print(f'STOP D={D}: CTMRG not converged; observables saved. No larger D started.',flush=True)
                return False
            parent = folder/'final.pt'
            base.atomic_json(out/'chain_status.json',dict(status='dimension_completed',D=D,ready_for_next_D=True))
        base.refresh_tables(out)
        base.atomic_json(out/'chain_status.json',dict(status='all_completed',dims=config['dims'],ready_for_next_D=True))
        return True
