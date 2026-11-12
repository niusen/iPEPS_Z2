"""Atomic checkpoints, exact D padding, long SU and one final CTM per D.

Only imports the already checked kernels; all server orchestration lives
here. No signal handlers, subprocesses, workers, watchdogs or respawning.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
import torch
from ..simple_update import BlockState, sweep
from ..honeycomb_ctm import layers
from ..geometry import audit
from .observables import measure_complete
from .device_support import (resolve_measurement_device, measurement_state, initial_environment,
                             synchronize, immutable_settings)

VERSION = 'blocked_honeycomb_server_chain_v1'
ARRAY_KEYS = ('sites', 'bonds', 'triangles', 'schmidt_spectra')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def append_csv(path, row):
    existed = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(row))
        if not existed:
            w.writeheader()
        w.writerow(row)
        f.flush()


@contextmanager
def writer_lock(directory):
    """Kernel lock, not an existence lock: a leftover FILE never blocks resume.

    The owning PID/host are printed on a conflict. Killing the actual owner
    releases the OS lock; deleting this file is unnecessary and unsafe.
    """
    path = directory/'.writer.lock'
    owner_path = directory/'.writer_owner.json'
    handle = path.open('a+b')
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b' ')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(f'OS output lock unavailable for {directory}: {exc}') from exc
            # Windows denies reading the locked byte through a different
            # handle, so read the separate human-readable owner record.
            try:
                owner = owner_path.read_text(encoding='utf-8').strip()
            except OSError:
                owner = 'PID/host record is not available yet'
            raise RuntimeError(f'Output is already owned by a live process: {owner}. '
                               f'Folder: {directory}. Do not delete .writer.lock.') from exc
        owner = dict(pid=os.getpid(), hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat())
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(owner).encode())
        handle.flush()
        atomic_json(owner_path, owner)
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


@torch.no_grad()
def expand_state(source, D):
    """Exact embedding, no random perturbation or nonzero new lambda floor.

    All old tensor entries and Schmidt weights are retained, new entries
    are zero. The represented wavefunction is unchanged BEFORE evolution.
    A subsequent two-block gate and rank-D SVD can create new Schmidt values.
    """
    if D <= source.D:
        raise ValueError(f'Target D={D} must exceed source D={source.D}')
    old = source.D
    groups = []
    for group in (source.A, source.B):
        padded = []
        for t in group:
            x = t.new_zeros((t.shape[0], D, D, D))
            x[:, :old, :old, :old] = t
            padded.append(x)
        groups.append(padded)
    ws = []
    for group in source.weights:
        row = []
        for w in group:
            p = w.new_zeros(D)
            p[:old] = w
            row.append(p)
        ws.append(row)
    return BlockState(*groups, ws, source.field, source.seed, steps=0, c3=source.c3)


@torch.no_grad()
def su_proxy(state):
    """Cheap full-step diagnostics, not a CTMRG energy or a proof of convergence.

    Keep normalized Schmidt values and product-environment physical density
    matrices. Consecutive snapshots are taken at the same full Trotter-step
    boundary, so the sub-gate O(dt) oscillation is not mistaken for drift.
    """
    weights = torch.cat([w for row in state.weights for w in row]).clone()
    rhos = []
    colours = (0,) if state.c3 else range(3)
    for kind in ('A', 'B'):
        for c in colours:
            # Full lambda, not sqrt(lambda): SU's product environment.
            t = state.dressed(kind, c, power=1.).flatten(1)
            rho = t@t.T
            rho /= torch.trace(rho)
            rhos.append(rho.flatten())
    return weights, torch.cat(rhos)


def save_checkpoint(path, state, settings_hash, progress, lineage):
    tmp = path.with_name(path.name+'.tmp')
    state.save(tmp, server_version=VERSION, settings_hash=settings_hash,
               progress=progress, lineage=lineage)
    # One atomic file holds BOTH tensors and their exact progress. A crash
    # cannot pair newer tensors with an older JSON stage counter.
    with tmp.open('r+b') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, expected_hash):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = payload.get('metadata', {})
    if metadata.get('server_version') != VERSION or metadata.get('settings_hash') != expected_hash:
        raise ValueError(f'Checkpoint/settings mismatch: {path}. Use a new output directory for changed settings.')
    return BlockState.load(path), metadata['progress'], metadata['lineage']


def initialise_progress():
    return dict(stage=0, stage_steps=0, d_steps=0, stage_stable_checks=0,
                previous_weights=None, previous_rho=None, previous_steps=0,
                stage_reports=[], last_rates=None, elapsed_su_seconds=0.)


def elapsed_progress(progress, start, base_elapsed):
    progress['elapsed_su_seconds'] = base_elapsed+time.perf_counter()-start


def evolve_dimension(config, folder, settings_hash, parent):
    checkpoint = folder/'checkpoint.pt'
    D = int(folder.name[1:])
    if checkpoint.exists():
        state, progress, lineage = load_checkpoint(checkpoint, settings_hash)
        if lineage.get('parent_sha256') != (sha256(parent) if parent else None):
            raise ValueError(f'Parent state changed for {folder}; refusing to mix continuation branches')
        print(f'RESUME D={D} stage={progress["stage"]+1} steps_at_D={progress["d_steps"]}', flush=True)
    else:
        if parent:
            source = BlockState.load(parent)
            state = expand_state(source, D)
            lineage = dict(initialization='exact_zero_padding', parent_D=source.D,
                           parent_file=str(parent.relative_to(folder.parent)), parent_sha256=sha256(parent))
        elif config.get('initial_D4_checkpoint'):
            initial = Path(config['initial_D4_checkpoint'])
            state = BlockState.load(initial)
            if state.D != D:
                raise ValueError('Initial checkpoint must have the first requested D')
            state.steps = 0
            lineage = dict(initialization='external_checkpoint', parent_D=None, parent_sha256=None,
                           initial_sha256=sha256(initial))
        else:
            state = BlockState.random(D, config['field'], config['seed'], c3=config['c3'])
            lineage = dict(initialization='random', parent_D=None, parent_sha256=None)
        progress = initialise_progress()
        state.save(folder/'initial.pt', lineage=lineage)
        save_checkpoint(checkpoint, state, settings_hash, progress, lineage)
    if state.D != D or state.field != config['field'] or state.c3 != config['c3']:
        raise ValueError('Loaded state has incompatible D, field, or C3 convention')
    start, base_elapsed = time.perf_counter(), progress['elapsed_su_seconds']
    for stage_index, spec in enumerate(config['stages']):
        if stage_index < progress['stage']:
            continue
        dt = float(spec['dt'])
        min_steps = math.ceil(spec['min_tau']/dt-1e-10)
        max_steps = math.ceil(spec['max_tau']/dt-1e-10)
        monitor_steps = max(config['monitor_min_steps'], math.ceil(config['monitor_tau']/dt-1e-10))
        if progress['previous_weights'] is None:
            progress['previous_weights'], progress['previous_rho'] = su_proxy(state)
            progress['previous_steps'] = progress['stage_steps']
        final_rates = progress.get('last_rates') or dict(lambda_drift_per_tau=None, rho_drift_per_tau=None)
        proxy_converged = (progress['stage_steps'] >= min_steps and
                           progress['stage_stable_checks'] >= config['stable_checks'])
        while progress['stage_steps'] < max_steps:
            _, discard = sweep(state, dt, config['second_order'])
            progress['stage_steps'] += 1
            progress['d_steps'] += 1
            n = progress['stage_steps']
            monitored = n-progress['previous_steps'] >= monitor_steps or n == max_steps
            if monitored:
                weights, rho = su_proxy(state)
                delta_tau = dt*(n-progress['previous_steps'])
                wrate = float((weights-progress['previous_weights']).abs().max())/delta_tau
                rrate = float(torch.linalg.vector_norm(rho-progress['previous_rho']))/delta_tau
                final_rates = dict(lambda_drift_per_tau=wrate, rho_drift_per_tau=rrate)
                progress['last_rates'] = final_rates
                stable = wrate <= config['lambda_drift_tolerance'] and rrate <= config['rho_drift_tolerance']
                progress['stage_stable_checks'] = progress['stage_stable_checks']+1 if stable else 0
                progress['previous_weights'], progress['previous_rho'], progress['previous_steps'] = weights, rho, n
                proxy_converged = n >= min_steps and progress['stage_stable_checks'] >= config['stable_checks']
                row = dict(D=D, stage=stage_index+1, dt=dt, stage_steps=n, stage_tau=n*dt,
                           lambda_drift_per_tau=wrate, rho_drift_per_tau=rrate,
                           stable_checks=progress['stage_stable_checks'], discarded_weight_max=discard,
                           elapsed_seconds=base_elapsed+time.perf_counter()-start)
                append_csv(folder/'su_history.csv', row)
                print(f'D={D} SU stage={stage_index+1}/{len(config["stages"])} '
                      f'dt={dt:g} tau={n*dt:.3f} [{spec["min_tau"]:g},{spec["max_tau"]:g}] '
                      f'full_step_lambda/tau={wrate:.3e} rho_proxy/tau={rrate:.3e} '
                      f'stable={progress["stage_stable_checks"]}/{config["stable_checks"]}', flush=True)
            if monitored or progress['d_steps'] % config['checkpoint_steps'] == 0:
                elapsed_progress(progress, start, base_elapsed)
                save_checkpoint(checkpoint, state, settings_hash, progress, lineage)
            if proxy_converged:
                break
        report = dict(stage=stage_index+1, dt=dt, steps=progress['stage_steps'], tau=progress['stage_steps']*dt,
                      proxy_converged=proxy_converged, **final_rates)
        progress['stage_reports'].append(report)
        progress.update(stage=stage_index+1, stage_steps=0, stage_stable_checks=0,
                        previous_weights=None, previous_rho=None, previous_steps=0, last_rates=None)
        elapsed_progress(progress, start, base_elapsed)
        save_checkpoint(checkpoint, state, settings_hash, progress, lineage)
        atomic_json(folder/'su_stages.json', progress['stage_reports'])
        if not proxy_converged:
            print(f'D={D} stage {stage_index+1}: reached maximum tau; convergence NOT asserted', flush=True)
    save_checkpoint(folder/'final.pt', state, settings_hash, progress, lineage)
    return state, progress, lineage


def refresh_tables(out):
    rows, sites, bonds, triangles, spectra = [], [], [], [], []
    for path in sorted(out.glob('D*/completed.json'), key=lambda p:int(p.parent.name[1:])):
        completed = json.loads(path.read_text())
        obs = json.loads((path.parent/'observables.json').read_text())
        row = {k:v for k,v in obs.items() if k not in ARRAY_KEYS}
        row.update(completed)
        rows.append(row)
        for key, dest in (('sites',sites), ('bonds',bonds), ('triangles',triangles), ('schmidt_spectra',spectra)):
            dest.extend(dict(D=obs['D'], field=obs['field'], chi=obs['chi'], **r) for r in obs[key])
    for name, data in (('summary',rows), ('site_magnetization',sites),
                       ('bond_energies',bonds), ('triangle_observables',triangles), ('schmidt_spectra',spectra)):
        write_csv(out/f'{name}.csv', data)
    lines = ['H=sum S.S-h sum Sz; energies per physical spin. C3 imposed; real dense tensors.',
             'SU proxy convergence is not proof of CTM observable convergence or a global minimum.']
    for r in rows:
        lines.append(f'D={r["D"]:2d} Eex={r["energy_exchange"]:.10f} Etot={r["energy_total"]:.10f} '
                     f'm={r["magnetization_normalized"]:.9f} max|Sx|={r["max_abs_sx"]:.3e} '
                     f'SU_proxy={r["su_proxy_converged"]} CTM={r["ctm_converged"]} '
                     f'face_error={r["face_consistency_max"]:.2e}')
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


def run_chain(config, out, make_plots=True):
    device = resolve_measurement_device(config.get('measurement_device', 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    dims = config['dims']
    if not dims or any(b != a+1 for a,b in zip(dims,dims[1:])):
        raise ValueError('dims must be a nonempty consecutive increasing list')
    if (not config['stages'] or not 1 <= config['ctm_min'] <= config['ctm_max'] <= 50
            or min(config['checkpoint_steps'],config['monitor_min_steps'],config['stable_checks'],
                   config['threads'],config['chi']) < 1 or config['monitor_tau'] <= 0):
        raise ValueError('Invalid run settings (CTMRG must have 1 <= min <= max <= 50)')
    for s in config['stages']:
        if not (0 < s['dt'] and 0 < s['min_tau'] <= s['max_tau']):
            raise ValueError('Invalid evolution stage')
    torch.set_num_threads(config['threads'])
    torch.set_grad_enabled(False)
    settings_hash = hashlib.sha256(canonical_json(immutable_settings(config)).encode()).hexdigest()
    with writer_lock(out):
        settings_path = out/'settings.json'
        if settings_path.exists() and immutable_settings(json.loads(settings_path.read_text())) != immutable_settings(config):
            raise ValueError('Settings changed in this output folder. Use a new OUT for a different experiment.')
        atomic_json(settings_path, config)
        print(f'{VERSION}; SU=cpu; CTMRG/observables={device}; CPU threads={config["threads"]}; D={dims}; h={config["field"]}; '
              f'chi={config["chi"]}; CTM max={config["ctm_max"]}; out={out}', flush=True)
        print(json.dumps(audit()), flush=True)
        atomic_json(out/'provenance.json', dict(version=VERSION, torch_version=torch.__version__,
            settings_hash=settings_hash, sources={str(p.relative_to(root)):sha256(p) for p in
            list(root.glob('*.py'))+list((root/'server').glob('*.py'))},
            su_device='cpu', measurement_device=str(device),
            measurement_device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            convention='H=sum S.S-h sum Sz; m/ms=2 mean Sz; real float64 tensors; CPU checkpoints'))
        parent = None
        for D in dims:
            folder = out/f'D{D}'
            folder.mkdir(exist_ok=True)
            completed = folder/'completed.json'
            if completed.exists():
                done = json.loads(completed.read_text())
                if done['settings_hash'] != settings_hash or done['state_sha256'] != sha256(folder/'final.pt'):
                    raise ValueError(f'Completed-state fingerprint mismatch: {folder}')
                if done.get('parent_sha256') != (sha256(parent) if parent else None):
                    raise ValueError('Completed-state parent changed')
                print(f'SKIP D={D}: state and complete observables already saved', flush=True)
            else:
                state, progress, lineage = evolve_dimension(config, folder, settings_hash, parent)
                print(f'D={D}: SU finished; begin ONE final native CTMRG and full measurement', flush=True)
                synchronize(device)
                t0 = time.perf_counter()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                measured_state = measurement_state(state, device)
                env = initial_environment(D, device)
                ctm_log = []
                def ctm_progress(row):
                    # JSON disallows infinity; early iterations have no residual yet.
                    row = row | dict(residual=row['residual'] if math.isfinite(row['residual']) else None)
                    ctm_log.append(row)
                    write_csv(folder/'ctm_history.csv', ctm_log)
                    print(f'D={D} CTMRG {row["iteration"]}/{config["ctm_max"]} residual={row["residual"]}', flush=True)
                env.run(layers(measured_state), config['chi'], config['ctm_max'], config['ctm_tolerance'],
                        ctm_progress, min_iterations=config['ctm_min'])
                obs = measure_complete(measured_state, env, lambda msg:print(f'D={D}: {msg}', flush=True),
                                       max_corner_cache_bytes=config['max_corner_cache_mb']*1024**2)
                synchronize(device)
                obs.update(chi=config['chi'], measurement_seconds=time.perf_counter()-t0,
                           su_device='cpu', measurement_device=str(device),
                           gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else 0.)
                del measured_state, env
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                atomic_json(folder/'observables.json', obs)
                for key, name in (('sites','site_magnetization'), ('bonds','bond_energies'),
                                  ('triangles','triangle_observables'), ('schmidt_spectra','schmidt_spectra')):
                    write_csv(folder/f'{name}.csv', obs[key])
                done = dict(settings_hash=settings_hash, state_sha256=sha256(folder/'final.pt'),
                            parent_sha256=lineage.get('parent_sha256'), parent_D=lineage.get('parent_D'),
                            su_proxy_converged=progress['stage_reports'][-1]['proxy_converged'],
                            su_seconds=progress['elapsed_su_seconds'], total_tau=sum(r['tau'] for r in progress['stage_reports']))
                atomic_json(completed, done)
                print(f'DONE D={D}: Eex={obs["energy_exchange"]:.10f} Etot={obs["energy_total"]:.10f} '
                      f'm/ms={obs["magnetization_normalized"]:.9f} SU_proxy={done["su_proxy_converged"]} '
                      f'CTM={obs["ctm_converged"]}', flush=True)
            parent = folder/'final.pt'
            refresh_tables(out)
            if make_plots:
                try:
                    from .report import make_report
                    make_report(out)
                except ImportError as exc:
                    print(f'Plotting dependency unavailable ({exc}); all numerical CSV data are saved.', flush=True)
                except Exception as exc:
                    # Graphics are optional; a plotting problem must not
                    # prevent later D values from being evolved/measured.
                    with (out/'plot_errors.log').open('a', encoding='utf-8') as f:
                        f.write(traceback.format_exc()+'\n')
                    print(f'Plotting failed ({exc}); numerical data saved, continuing. See plot_errors.log.', flush=True)
        return refresh_tables(out)
