"""All physics/settings here. Server job only; import does not launch work."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.paper_method.workflow import run

D=7
SEEDS=list(range(7101,7111))
FIELD=.4
THREADS=10
MEASUREMENT_DEVICE='cuda:1'          # CPU SU; GPU CTMRG + complete observables
SCHEDULE=[dict(dt=.03,tau=40.),dict(dt=.01,tau=20.),dict(dt=.003,tau=5.),
          dict(dt=.001,tau=2.),dict(dt=.0005,tau=1.)]
C3_POLICY='if_close'                 # 'never' keeps all six tensors throughout
REDUCE_AFTER_STAGE=2                 # then retry after stages 3 and 4 if needed
REDUCTION=dict(tensor_tolerance=1e-3,weight_tolerance=1e-3,rho_tolerance=1e-4,
               degeneracy_tolerance=1e-10)
CHIS=[40,60]                        # Same final state; independent fresh CTM starts
CTMRG_MAX_ITERATIONS=50
CTMRG_MIN_ITERATIONS=20
CTMRG_TOLERANCE=1e-6
CHI_ENERGY_TOLERANCE=1e-6
CHI_SITE_TOLERANCE=1e-5
CHI_BOND_TOLERANCE=1e-5
CHECKPOINT_STEPS=100
MONITOR_STEPS=200
LAMBDA_DRIFT_PER_TAU=1e-7
RHO_DRIFT_PER_TAU=1e-7
STABLE_CHECKS=3
MAX_CORNER_CACHE_MB=256
OUT=Path(__file__).resolve().parents[1]/'results/server_D7_random10_h040_physicalNN_six_then_C3'


def settings():
    return dict(D=D,seeds=SEEDS,field=FIELD,threads=THREADS,measurement_device=MEASUREMENT_DEVICE,
        schedule=SCHEDULE,c3_policy=C3_POLICY,reduce_after_stage=REDUCE_AFTER_STAGE,reduction=REDUCTION,
        chis=CHIS,ctm_max=CTMRG_MAX_ITERATIONS,ctm_min=CTMRG_MIN_ITERATIONS,ctm_tolerance=CTMRG_TOLERANCE,
        chi_energy_tolerance=CHI_ENERGY_TOLERANCE,chi_site_tolerance=CHI_SITE_TOLERANCE,
        chi_bond_tolerance=CHI_BOND_TOLERANCE,checkpoint_steps=CHECKPOINT_STEPS,monitor_steps=MONITOR_STEPS,
        lambda_drift_tolerance=LAMBDA_DRIFT_PER_TAU,rho_drift_tolerance=RHO_DRIFT_PER_TAU,
        stable_checks=STABLE_CHECKS,max_corner_cache_mb=MAX_CORNER_CACHE_MB)


if __name__=='__main__':raise SystemExit(0 if run(settings(),OUT) else 2)
