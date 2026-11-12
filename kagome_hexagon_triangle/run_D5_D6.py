"""Local modest-D experiment. All editable options are below.

From iPEPS_Z2: python -m kagome_hexagon_triangle.run_D5_D6
Separate output folder; no old iPESS files or data are overwritten.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from kagome_hexagon_triangle.run_experiment import run_one, csv_write

THREADS = 10
FIELD = .38
RUNS = [(5, 1701), (6, 1701)]
# User's established dense SU schedule. Every step is a SECOND-ORDER
# palindrome of the three C3-related disjoint-bond matchings.
SCHEDULE = [(.03, 40.), (.01, 20.), (.003, 5.), (.001, 2.), (.0005, 1.)]
CHI_VALUES = [16, 24]
CTM_MAX_ITERATIONS = 50
CTM_TOLERANCE = 1e-6
C3_ROTATED_COPIES = True
SECOND_ORDER = True
OUT = Path(__file__).parent/'results'/'D5_D6_c3_h038'


def main():
    torch.set_num_threads(THREADS)
    torch.set_grad_enabled(False)
    records = []
    for D, seed in RUNS:
        records.extend(run_one(D, seed, SCHEDULE, OUT, CHI_VALUES, c3=C3_ROTATED_COPIES,
                               field=FIELD, ctm_max=CTM_MAX_ITERATIONS,
                               ctm_tol=CTM_TOLERANCE, second_order=SECOND_ORDER))
        csv_write(OUT/'summary.csv', records)


if __name__ == '__main__':
    main()
