"""Edit server settings HERE. Save data only; draw reports later on the laptop."""
from pathlib import Path
from ..src.independent_random_fields import OPTIONS,run

SERVER_OPTIONS=OPTIONS|dict(
    dims=[2,3,4],Dz=[.02,.05,.1],fields=[.28,.32,.36,.38,.40,.44,.48],trials=[1,2,3],
    measurement_device='cuda:1', # SU stays on CPU; use 'cpu' if desired.
    threads=10,chi=12,ctm_max=100,ctm_tol=1e-6,
    out=str(Path(__file__).resolve().parents[2]/'results/DMI_independent_random3_D2_D4_server'),
)

if __name__=='__main__':
    count=1
    for k in ('dims','Dz','fields','trials'):count*=len(SERVER_OPTIONS[k])
    steps=sum(round(tau/dt) for dt,tau in SERVER_OPTIONS['fresh'])
    print(f'{count} independent random field points; {steps} SU steps EACH; CPU SU, measurement='+SERVER_OPTIONS['measurement_device'],flush=True)
    print('Data only: checkpoints, full observations and progress.csv; no plotting.',flush=True)
    run(SERVER_OPTIONS)
