# Triangular chiral-pair spin iPESS

This directory implements two decoupled triangular J1-Jchi layers with
`Jchi_antichiral = -Jchi_chiral`.  The onsite d=4 leg is split into two d=2
legs by one fixed unitary with basis convention `p=2*a+b`.  That unitary and
all derived operators are constructed once and reused throughout gradients,
line searches, and observable evaluations.

The optimization objective and the primary reported energy are the average
per layer, `(E_chiral + E_antichiral) / 2`. The two individual layer energies
are also printed separately.

To construct the exact product initial state `|psi> x |psi*>`, edit and run
`build_chiral_pair_from_single.py`. A single-layer bond dimension D becomes
pair bond dimension D squared.

For optimization, edit `optimize_triangle_spin_pair_iPESS.py`, point
`initial_state_file` to the generated d=4 state, set `load_initial_state=True`,
and submit with `bash run_triangle_spin_pair_server.sh`.

For observables without gradients, edit
`compute_triangle_spin_pair_observables.py` and submit with
`bash run_triangle_spin_pair_observables_server.sh`.
