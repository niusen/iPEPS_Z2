# Z2 fermionic iPESS for the triangular J1-Jchi spin model

This workflow keeps the old fermionic iPESS tensor ordering, swap gates, Z2
virtual sectors, double layer, and CTMRG.  Its physical leg is exactly
`t=(1,), D=(2,)`, i.e. only `|up>` and `|down>`.

1. Edit and run `project_gutzwiller_d4_to_d2.py` once. It reads the old d=4
   Hofstadter-Hubbard JSON and writes an independent projected d=2 JSON.
2. Edit `optimize_triangle_spin_fermionic_iPESS.py`, especially the source
   path, input state, J1, Jchi, D, chi, device, and cell size.
3. Submit with `bash run_triangle_spin_fermionic_server.sh`.

To evaluate an optimized d=2 state without changing it, edit
`compute_triangle_spin_observables.py` and submit with
`bash run_triangle_spin_observables_server.sh`. The observable runner infers
`Lx`, `Ly`, and the actual virtual bond dimension directly from the JSON.

The optimization file only loads d=2 states. It does not project during an
optimization step. Accepted line-search steps print both triangle
chiralities, three bond energies, and Sx/Sy/Sz/total magnetization. Gradient
evaluations omit magnetization.

The conversion is a physical Gutzwiller projection, not merely a JSON-format
conversion. Empty and doubly occupied amplitudes contribute to the norm and
particle-number fluctuations of the original d=4 state and are removed.
Consequently, spin expectation values before and after projection generally
differ. The d=2 representation agrees only with the d=4 state *after* the same
local projector has been applied and the projected wavefunction normalized.
