"""New, independent server chain from the saved paper-like local D5 state.

All user options live here. No calculation runs on import. This reuses the
existing SU, native honeycomb CTMRG and complete observable functions.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.server.workflow_from_state import run_seeded_chain

FIELD = 0.38
D_VALUES = list(range(5, 11))
INITIAL_STATE_FILE = 'initial_states/D5_local_paperlike.pt'  # relative to server/
INITIAL_STATE_SHA256 = 'ac0e8cbc2d9141f6f2574bf830957949d310f90fcd090a98d5f750636734f6bf'
SEED = 1701                       # provenance only: no random initialization
THREADS = 10
MEASUREMENT_DEVICE = 'cuda:1'      # SU=cpu; CTMRG and all observables=cuda:1
C3_ROTATED_COPIES = True
SECOND_ORDER = True

# The saved D5 state already completed dt=0.0005 evolution. Refine it with
# small steps rather than reheating it with dt=0.03. Larger D uses the old
# warm-up schedule, then stays at the final dt until the SU proxy passes.
FIRST_D_WARMUP = [
    dict(dt=0.003, min_tau=10., max_tau=40.),
    dict(dt=0.001, min_tau=4., max_tau=16.),
]
NEXT_D_WARMUP = [
    dict(dt=0.03, min_tau=80., max_tau=320.),
    dict(dt=0.01, min_tau=40., max_tau=160.),
    dict(dt=0.003, min_tau=10., max_tau=40.),
    dict(dt=0.001, min_tau=4., max_tau=16.),
]
FINE_STAGE = dict(dt=0.0005, min_tau=2., max_tau=32.)
MAX_FINE_CHUNKS_PER_D_PER_SUBMISSION = 16
# If all chunks hit max_tau, save and STOP before increasing D. Rerunning
# the same sh continues at the same fine dt for up to 16 further chunks.
# Earlier warm-up stages are not repeated. No automatic relaunch loop.
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
OUT = Path(__file__).resolve().parents[1]/'results'/'server_from_local_D5_D5_D10_h038_C3_chain'


def settings():
    return dict(dims=D_VALUES,field=FIELD,seed=SEED,threads=THREADS,
                measurement_device=MEASUREMENT_DEVICE,
                initial_state_file=INITIAL_STATE_FILE,initial_state_sha256=INITIAL_STATE_SHA256,
                c3=C3_ROTATED_COPIES,second_order=SECOND_ORDER,
                first_D_warmup=FIRST_D_WARMUP,next_D_warmup=NEXT_D_WARMUP,
                fine_stage=FINE_STAGE,max_fine_chunks=MAX_FINE_CHUNKS_PER_D_PER_SUBMISSION,
                monitor_tau=MONITOR_TAU,monitor_min_steps=MONITOR_MIN_STEPS,
                lambda_drift_tolerance=LAMBDA_DRIFT_PER_TAU,
                rho_drift_tolerance=RHO_PROXY_DRIFT_PER_TAU,
                stable_checks=STABLE_CHECKS,checkpoint_steps=CHECKPOINT_STEPS,
                chi=CHI,ctm_max=CTMRG_MAX_ITERATIONS,ctm_min=CTMRG_MIN_ITERATIONS,
                ctm_tolerance=CTMRG_TOLERANCE,max_corner_cache_mb=MAX_CORNER_CACHE_MB)


if __name__ == '__main__':
    raise SystemExit(0 if run_seeded_chain(settings(),OUT) else 2)
