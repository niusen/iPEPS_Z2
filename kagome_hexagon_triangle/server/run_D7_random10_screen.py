"""Fast random-start SCREENING: fixed 9000 SU steps, one measurement.

Saved candidates from the previous long job are reused for measurement
when available. All options are here; existing physics kernels unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kagome_hexagon_triangle.server.run_D7_random10 import settings as previous_settings
from kagome_hexagon_triangle.server.workflow_random_seeds import run_random_batch

D=7
SEEDS=list(range(7101,7111))
FIELD=0.4
THREADS=10
MEASUREMENT_DEVICE='cuda:1'
C3_ROTATED_COPIES=True
CHI=40
CTMRG_MAX_ITERATIONS=50
CTMRG_TOLERANCE=1e-6
SCHEDULE=[dict(dt=.03,tau=40.),dict(dt=.01,tau=20.),dict(dt=.003,tau=5.),
          dict(dt=.001,tau=2.),dict(dt=.0005,tau=1.)]
# Relative to kagome_hexagon_triangle/. Set '' to start entirely fresh.
# Old files are only read. Candidates already past 9000 steps are measured
# directly; incomplete/failed measurements never cause more SU on restart.
REUSE_LONG_RUN='results/server_D7_random10_h040_C3'
OUT=Path(__file__).resolve().parents[1]/'results'/'server_D7_random10_h040_C3_screen9000'


def settings():
    cfg=previous_settings()
    stages=[dict(dt=s['dt'],min_tau=round(s['tau']/s['dt'])*s['dt'],
                 max_tau=round(s['tau']/s['dt'])*s['dt'],requested_tau=s['tau']) for s in SCHEDULE]
    cfg.update(D=D,seeds=SEEDS,field=FIELD,threads=THREADS,measurement_device=MEASUREMENT_DEVICE,
               c3=C3_ROTATED_COPIES,chi=CHI,ctm_max=CTMRG_MAX_ITERATIONS,
               ctm_tolerance=CTMRG_TOLERANCE,evolution_mode='fixed_screen',fixed_stages=stages,
               reuse_long_run=REUSE_LONG_RUN)
    return cfg


if __name__=='__main__':
    raise SystemExit(0 if run_random_batch(settings(),OUT) else 2)
