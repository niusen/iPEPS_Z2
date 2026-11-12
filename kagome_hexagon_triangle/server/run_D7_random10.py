"""Ten independent D7 random starts. Edit all options here; no CLI physics.

No calculation runs at import. Uses the existing real blocked iPEPS SU,
native honeycomb CTMRG, and complete-observable kernels unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.server.workflow_random_seeds import run_random_batch

D = 7
SEEDS = list(range(7101,7111))       # 10 genuinely independent random starts
FIELD = 0.4                        # paper's public D7 local-Sz data use h=0.4
THREADS = 10
MEASUREMENT_DEVICE = 'cuda:1'       # SU on CPU; CTMRG + observables on GPU
C3_ROTATED_COPIES = True           # matches reduced paper ansatz; NO mirror
SECOND_ORDER = True

WARMUP = [
    dict(dt=0.03, min_tau=80., max_tau=320.),
    dict(dt=0.01, min_tau=40., max_tau=160.),
    dict(dt=0.003,min_tau=10., max_tau=40.),
    dict(dt=0.001,min_tau=4.,  max_tau=16.),
]
FINE_STAGE = dict(dt=0.0005,min_tau=2.,max_tau=32.)
MAX_FINE_CHUNKS_PER_SEED_PER_SUBMISSION = 16
MONITOR_TAU = 1.0
MONITOR_MIN_STEPS = 100
LAMBDA_DRIFT_PER_TAU = 1e-7
RHO_PROXY_DRIFT_PER_TAU = 1e-7
STABLE_CHECKS = 3
CHECKPOINT_STEPS = 200

CHI = 40
CTMRG_MAX_ITERATIONS = 50
CTMRG_MIN_ITERATIONS = 20
CTMRG_TOLERANCE = 1e-6
MAX_CORNER_CACHE_MB = 256
# One final measurement per seed per submission, even if SU hits its cap.
# Flags distinguish convergence from a measured but unfinished candidate.
# Rerunning skips successful seeds and continues only unfinished ones.
_field_tag=f'{FIELD:.2f}'.replace('.','')
OUT = Path(__file__).resolve().parents[1]/'results'/(
    f'server_D{D}_random{len(SEEDS)}_h{_field_tag}_'+('C3' if C3_ROTATED_COPIES else 'six_tensors'))


def settings():
    return dict(D=D,seeds=SEEDS,field=FIELD,threads=THREADS,
                measurement_device=MEASUREMENT_DEVICE,c3=C3_ROTATED_COPIES,
                second_order=SECOND_ORDER,warmup=WARMUP,fine_stage=FINE_STAGE,
                max_fine_chunks=MAX_FINE_CHUNKS_PER_SEED_PER_SUBMISSION,
                monitor_tau=MONITOR_TAU,monitor_min_steps=MONITOR_MIN_STEPS,
                lambda_drift_tolerance=LAMBDA_DRIFT_PER_TAU,
                rho_drift_tolerance=RHO_PROXY_DRIFT_PER_TAU,
                stable_checks=STABLE_CHECKS,checkpoint_steps=CHECKPOINT_STEPS,
                chi=CHI,ctm_max=CTMRG_MAX_ITERATIONS,ctm_min=CTMRG_MIN_ITERATIONS,
                ctm_tolerance=CTMRG_TOLERANCE,max_corner_cache_mb=MAX_CORNER_CACHE_MB)


if __name__=='__main__':
    raise SystemExit(0 if run_random_batch(settings(),OUT) else 2)
