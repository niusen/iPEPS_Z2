"""Laptop plotting only. Edit paths below; run from the Python repository root.

python -m kagome_hexagon_triangle.dmi.analysis.run_local_report
No SU or CTMRG calculation is performed.
"""
from pathlib import Path
from .independent_random_report import run

PYTHON_RESULTS=Path(__file__).resolve().parents[2]/'results/DMI_independent_random3_D2_D4_server'
JULIA_RESULTS=None # Optional downloaded Julia result directory; None plots Python only.
OUTPUT=PYTHON_RESULTS/'analysis'

if __name__=='__main__':
    if not (Path(PYTHON_RESULTS)/'plan.json').is_file():
        raise FileNotFoundError(f'Download results first, then edit PYTHON_RESULTS: {PYTHON_RESULTS}')
    run(python_root=PYTHON_RESULTS,julia_root=JULIA_RESULTS,out=OUTPUT)
