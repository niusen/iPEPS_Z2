"""Run a small end-to-end optimization of the triangular J1-Jchi model.

With no command-line arguments this uses a fast CPU-friendly configuration.
Any arguments supplied by the user are forwarded to ``triangle_spin_opt.py``.
"""

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from triangle_spin_opt import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--J1", "1.0",
                "--Jchi", "0.2",
                "--D", "2",
                "--chi", "4",
                "--Lx", "1",
                "--Ly", "1",
                "--device", "cpu",
                "--threads", "4",
                "--maxiter", "2",
                "--ad-ctm-iter", "1",
                "--ls-ctm-iter", "2",
                "--line-search", "backtracking",
                "--step0", "0.05",
                "--ls-maxiter", "4",
                "--max-grad-norm", "1.0",
                "--save-prefix", os.path.join(HERE, "smoke_J1_1_Jchi_0p2"),
            ]
        )
    main()
