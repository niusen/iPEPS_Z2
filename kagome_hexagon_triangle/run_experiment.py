"""Settings are here, not command-line arguments. Run from package root:

    python -m kagome_hexagon_triangle.run_experiment

This file starts the small pilot by default. Increase RUNS/SCHEDULE only
after examining its timings and checks. Existing kagome files are unused.
"""
import csv
import json
import math
import time
from pathlib import Path
import torch
from .simple_update import BlockState, sweep
from .honeycomb_ctm import HoneycombEnvironment, layers
from .observables import measure
from .geometry import audit

THREADS = 10
FIELD = .38
RUNS = [(2, 1701)]
# Preliminary timing/physics pilot; NOT a converged production schedule.
SCHEDULE = [(.03, 1.2), (.01, .4), (.003, .12)]
SECOND_ORDER = True
C3_ROTATED_COPIES = False
CHI_VALUES = [8, 12]
CTM_MAX_ITERATIONS = 50
CTM_TOLERANCE = 1e-6
OUT = Path(__file__).parent/'results'/'pilot'


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_one(D, seed, schedule=SCHEDULE, out=OUT, chis=CHI_VALUES, c3=C3_ROTATED_COPIES,
            field=FIELD, ctm_max=CTM_MAX_ITERATIONS, ctm_tol=CTM_TOLERANCE, second_order=SECOND_ORDER):
    folder = out/f'D{D}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder/'latest.pt'
    progress_path = folder/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(stage=0, step=0)
    state = BlockState.load(latest) if latest.exists() else BlockState.random(D, field, seed, c3=c3)
    if state.D != D or state.field != field or state.c3 != c3:
        raise ValueError('Checkpoint/settings mismatch')
    settings = dict(D=D, seed=seed, field=field, schedule=schedule, second_order=second_order, c3=c3,
                    chis=chis, ctm_max=ctm_max, ctm_tol=ctm_tol,
                    convention='H=J sum S.S - h sum Sz; m_normalized=2 mean Sz')
    settings_file = folder/'settings.json'
    if settings_file.exists() and json.loads(settings_file.read_text()) != json.loads(json.dumps(settings)):
        raise ValueError('Settings changed in an existing run: choose a new output directory')
    settings_file.write_text(json.dumps(settings, indent=2)+'\n')
    start = time.perf_counter()
    for stage, (dt, tau) in enumerate(schedule):
        n = int(math.ceil(tau/dt-1e-10))
        if stage < progress['stage']:
            continue
        begin = progress['step'] if stage == progress['stage'] else 0
        for step in range(begin, n):
            change, discard = sweep(state, dt, second_order)
            if (step+1) % 20 == 0 or step+1 == n:
                print(f'D={D} seed={seed} stage={stage+1}/{len(schedule)} step={step+1}/{n} '
                      f'dt={dt} lambda_delta={change:.3e} discarded={discard:.3e} '
                      f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
                state.save(folder/'latest.tmp')
                (folder/'latest.tmp').replace(latest)
                progress_path.write_text(json.dumps(dict(stage=stage, step=step+1)))
        progress_path.write_text(json.dumps(dict(stage=stage+1, step=0)))
    state.save(folder/'final.pt')
    records = []
    local = layers(state)
    for chi in chis:
        print(f'D={D} native honeycomb CTMRG chi={chi}', flush=True)
        env = HoneycombEnvironment.initial(D)
        ctm_log = []
        t0 = time.perf_counter()
        def report(row):
            ctm_log.append(row)
            if row['iteration'] % 10 == 0:
                print(f'CTM {row} elapsed={time.perf_counter()-t0:.1f}s', flush=True)
        env.run(local, chi, ctm_max, ctm_tol, report)
        obs = measure(state, env)
        obs['chi'] = chi
        obs['measurement_seconds'] = time.perf_counter()-t0
        csv_write(folder/f'sites_chi{chi}.csv', obs['sites'])
        csv_write(folder/f'bonds_chi{chi}.csv', obs['bonds'])
        csv_write(folder/f'ctm_chi{chi}.csv', ctm_log)
        (folder/f'observables_chi{chi}.json').write_text(json.dumps(obs, indent=2)+'\n')
        records.append({k:v for k, v in obs.items() if k not in ('sites', 'bonds')})
        print(json.dumps(records[-1]), flush=True)
    csv_write(folder/'summary.csv', records)
    return records


def main():
    torch.set_num_threads(THREADS)
    torch.set_grad_enabled(False)
    print(json.dumps(audit()), flush=True)
    records = []
    for D, seed in RUNS:
        records.extend(run_one(D, seed))
    csv_write(OUT/'summary.csv', records)


if __name__ == '__main__':
    main()
