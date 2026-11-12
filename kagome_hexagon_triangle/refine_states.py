"""Continuation checks of saved states, preserving every initial run.

The small-dt tail alone does not prove imaginary-time convergence. This
adds 22 units of evolution and measures again; it is not another seed.
"""
from pathlib import Path
import shutil
import torch
from .run_experiment import run_one, csv_write

THREADS = 10
ROOT = Path(__file__).parent/'results'
# D5 has already undergone this continuation; D6 is the next selected case.
RUNS = [(6, 1701)]
SCHEDULE = [(.01, 20.), (.003, 1.), (.001, .5), (.0005, .5)]
CHIS = [16, 24]


def main():
    torch.set_num_threads(THREADS)
    torch.set_grad_enabled(False)
    for D, seed in RUNS:
        out = ROOT/f'D{D}_continuation_h038'
        source = ROOT/'D5_D6_c3_h038'/f'D{D}_seed{seed}'/'final.pt'
        folder = out/f'D{D}_seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder/'latest.pt').exists():
            shutil.copy2(source, folder/'latest.pt')
        records = run_one(D, seed, SCHEDULE, out, CHIS, c3=True)
        csv_write(out/'summary.csv', records)


if __name__ == '__main__':
    main()
