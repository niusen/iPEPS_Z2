"""D=6/8, Dz=0..0.7: edit server settings HERE; save data only.

Each (D,Dz,h,trial) is a fresh complex random ket. No D/Dz/field continuation
or old plateau seed is used. The physical grid matches the Julia D6/D8 job;
the Python 27-spin hexagon-triangle ansatz and numerical kernels are reused.
"""
from pathlib import Path
from ..src.independent_random_fields import OPTIONS,run

SERVER_OPTIONS=OPTIONS|dict(
    dims=[6,8],
    Dz=[0.,.02,.03,.05,.10,.15,.20,.30,.40,.50,.60,.70],
    fields=[.28,.32,.36,.38,.40,.44,.48,.52,.56,.60,.64],
    trials=[1,2,3],seed_base=202611170000,
    measurement_device='cuda:1', # SU stays on CPU; use 'cpu' if desired.
    threads=10,chi=12,ctm_max=100,ctm_tol=1e-6,
    out=str(Path(__file__).resolve().parents[2]/'results/DMI_independent_random3_D6_D8_positive_Dz070_h028_064_server'),
)

if __name__=='__main__':
    count=1
    for k in ('dims','Dz','fields','trials'):count*=len(SERVER_OPTIONS[k])
    steps=sum(round(tau/dt) for dt,tau in SERVER_OPTIONS['fresh'])
    print(f'{count} independent random field points; {steps} SU steps EACH; CPU SU, measurement='+SERVER_OPTIONS['measurement_device'],flush=True)
    print(f"D={SERVER_OPTIONS['dims']}; Dz={SERVER_OPTIONS['Dz']}; h={SERVER_OPTIONS['fields']}; {len(SERVER_OPTIONS['trials'])} independent trials per point",flush=True)
    print('Data only: checkpoints, full observations and progress.csv; no plotting.',flush=True)
    run(SERVER_OPTIONS)
