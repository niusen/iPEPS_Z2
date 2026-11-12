"""Resume the existing D5-origin branch; all options here, no CLI physics."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.server.workflow_bounded_resume import run

SOURCE_D=9
D_VALUES=list(range(9,16))
OLD_OUTPUT='results/server_from_local_D5_D5_D10_h038_C3_chain'
EXPECTED_D5_INITIAL_SHA256='ac0e8cbc2d9141f6f2574bf830957949d310f90fcd090a98d5f750636734f6bf'
FIELD=.38
SEED=1701
C3_ROTATED_COPIES=True
SECOND_ORDER=True
THREADS=10
MEASUREMENT_DEVICE='cuda:1'
# Hard upper bounds: at most 9000 full SU steps per NEW D, never 16 fine chunks.
# Stop a stage early only after both cheap SU proxies pass three checks.
# These are SU-internal diagnostics, not CTMRG/energy/magnetization measurements.
STAGES=[dict(dt=.03,min_tau=10.,max_tau=1333*.03),
        dict(dt=.01,min_tau=5.,max_tau=20.),
        dict(dt=.003,min_tau=1.,max_tau=1667*.003),
        dict(dt=.001,min_tau=.5,max_tau=2.),
        dict(dt=.0005,min_tau=.25,max_tau=1.)]
# D9 has already evolved for a long time: keep its existing tensors and use
# ONLY the final fine dt. At most 2000 further full steps; no warm-up replay.
SOURCE_STAGES=[dict(dt=.0005,min_tau=.25,max_tau=1.)]
LAMBDA_DRIFT_PER_TAU=1e-5
RHO_PROXY_DRIFT_PER_TAU=1e-5
STABLE_CHECKS=3
MONITOR_TAU=.1
MONITOR_MIN_STEPS=100
CHECKPOINT_STEPS=100
CHI=40
CTMRG_MAX_ITERATIONS=50
CTMRG_MIN_ITERATIONS=20
CTMRG_TOLERANCE=1e-6
MAX_CORNER_CACHE_MB=256
OUT=Path(__file__).resolve().parents[1]/'results/server_from_local_D5_D9_D15_h038_C3_bounded'


def settings():
    return dict(source_D=SOURCE_D,dims=D_VALUES,old_output=OLD_OUTPUT,
        expected_initial_sha256=EXPECTED_D5_INITIAL_SHA256,field=FIELD,seed=SEED,
        c3=C3_ROTATED_COPIES,second_order=SECOND_ORDER,threads=THREADS,
        measurement_device=MEASUREMENT_DEVICE,stages=STAGES,
        source_stages=SOURCE_STAGES,
        lambda_drift_tolerance=LAMBDA_DRIFT_PER_TAU,rho_drift_tolerance=RHO_PROXY_DRIFT_PER_TAU,
        stable_checks=STABLE_CHECKS,monitor_tau=MONITOR_TAU,monitor_min_steps=MONITOR_MIN_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,chi=CHI,ctm_max=CTMRG_MAX_ITERATIONS,
        ctm_min=CTMRG_MIN_ITERATIONS,ctm_tolerance=CTMRG_TOLERANCE,
        max_corner_cache_mb=MAX_CORNER_CACHE_MB)


if __name__=='__main__':raise SystemExit(0 if run(settings(),OUT) else 2)
