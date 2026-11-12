"""Server entry point. Edit the options HERE; no command-line arguments.

Run `sh run_D4_D10.sh` in this directory, or run this Python file directly.
No computation runs at import time. Default is CPU, not CUDA.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.server.workflow import run_chain

FIELD = 0.38
D_VALUES = list(range(4, 11))
SEED = 1701                       # only the FIRST D is randomly initialized
INITIAL_D4_CHECKPOINT = None       # optional path to THIS ansatz's D4 .pt file
THREADS = 10
MEASUREMENT_DEVICE = 'cpu'         # set 'cuda:1' for CTMRG + ALL observables
C3_ROTATED_COPIES = True
SECOND_ORDER = True

# Minimum tau is twice the old schedule. Continue each stage if the cheap
# full-step diagnostics are still drifting, up to the stated maximum tau.
# No CTMRG is run during these stages. Reaching a maximum is explicitly
# recorded as unconverged, rather than silently claiming convergence.
STAGES = [
    dict(dt=0.03,   min_tau=80., max_tau=320.),
    dict(dt=0.01,   min_tau=40., max_tau=160.),
    dict(dt=0.003,  min_tau=10., max_tau=40.),
    dict(dt=0.001,  min_tau=4.,  max_tau=16.),
    dict(dt=0.0005, min_tau=2.,  max_tau=8.),
]
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
MAKE_PLOTS = False                 # server default: numerical computation only
OUT = Path(__file__).resolve().parents[1]/'results'/'server_D4_D10_h038_C3_chain'


def settings():
    return dict(dims=D_VALUES, field=FIELD, seed=SEED, threads=THREADS, measurement_device=MEASUREMENT_DEVICE,
                initial_D4_checkpoint=str(Path(INITIAL_D4_CHECKPOINT).resolve()) if INITIAL_D4_CHECKPOINT else None,
                c3=C3_ROTATED_COPIES, second_order=SECOND_ORDER, stages=STAGES,
                monitor_tau=MONITOR_TAU, monitor_min_steps=MONITOR_MIN_STEPS,
                lambda_drift_tolerance=LAMBDA_DRIFT_PER_TAU, rho_drift_tolerance=RHO_PROXY_DRIFT_PER_TAU,
                stable_checks=STABLE_CHECKS, checkpoint_steps=CHECKPOINT_STEPS,
                chi=CHI, ctm_max=CTMRG_MAX_ITERATIONS, ctm_min=CTMRG_MIN_ITERATIONS,
                ctm_tolerance=CTMRG_TOLERANCE, max_corner_cache_mb=MAX_CORNER_CACHE_MB)


if __name__ == '__main__':
    run_chain(settings(), OUT, make_plots=MAKE_PLOTS)
