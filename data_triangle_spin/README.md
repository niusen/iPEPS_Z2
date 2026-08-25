# Triangular-lattice spin J1-Jchi runs

The model is

`H = J1 sum_<ij> Si.Sj + Jchi sum_triangle Si.(Sj x Sk)`,

with the same oriented chirality on up and down triangles.

Start with the local operator regression test:

```bash
python verify_J1_Jchi_operators.py
```

Then run the small end-to-end CPU optimization:

```bash
python smoke_opt_J1_Jchi.py
```

The default smoke run uses `D=2`, `chi=4`, a `1x1` cell, two L-BFGS
iterations, and writes its final state next to the run file. It tests CTMRG,
energy evaluation, automatic differentiation, line search, and state saving.

For a server calculation, edit all parameters directly near the top of
`optimize_triangle_spin_iPESS.py`.  The server submission script contains no
model parameters; submit with

```bash
bash run_triangle_spin_server.sh
```

Output and error logs are written under `logs/`. The single optimized-state
JSON is written next to `optimize_triangle_spin_iPESS.py`; its name is built
from `J1`, `Jchi`, `D`, `chi`, `Lx`, and `Ly`.

When a loaded JSON has a smaller bond dimension than `D`, the runner embeds it
into the requested dimension with zero padding. The single `Noise` setting is
then applied to the complete state, exactly as for a state that was not
expanded. The log prints both the loaded and actual optimization dimensions.
A loaded state larger than `D` is rejected rather than silently truncated.
