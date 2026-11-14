# Nonzero DMI: unrestricted dense SU exploration

Physical Hamiltonian: H=J sum SS+Dz sum_CCW kappa-h sum Sz, J=1.
Dz/J=0.02,0.05,0.1; D=2,3,4; h/J=0.28,0.32,0.36,0.38,0.40,0.44,0.48.
No U1 or pure spatial C3 constraint. No projected fixed-charge plateau seed.

Each (D,Dz) has independent complex random low-field and high-field endpoints.
The ascending and descending branches inherit their own previous field's ket.
Endpoint schedule dt=(.03,.01,.003,.001,.0005), tau=(40,20,5,2,1): 9000 steps.
Warm schedule has tau=(10,5,1,.5,.2) at the same dt: 2066 steps.
One final CTM per point, CPU threads10, chi12; caps Python100/Julia50, tol1e-6.
There are 126 points per ansatz. These are initial finite-D scans, not an
exhaustive random search or high-precision phase-boundary determination.

Python entry/options: `src/plateau_explore.py`; run as
`python -m kagome_hexagon_triangle.dmi.src.plateau_explore` from the repository root.
Julia entry/options:
`SU2_AD/Kagome/simple_update/multicell/dmi/server/run_plateau_dense_D2_D4.jl`.
Julia functional helper: `SU2_AD/src/bosonic/kagome/dmi/plateau_explore.jl`.
Julia uses h_parameter=-h_physical to match Python's -hSz Hamiltonian.
Reported magnetization is always +2<Sz>, despite the legacy Julia API convention.

Stage checkpoints, initial/final kets, full observations, errors, site/bond
data and aggregate progress CSV are saved in each package's
`results/DMI_plateau_dense_D2_D4_20261113/`. On restart completed stages and
points are reused. A partially completed stage repeats from its saved start;
there is no false claim of resuming inside an interrupted Trotter step.
Early Python stage records named last_weight_drift_per_tau contain a subgate
diagnostic; later explicit last_full_step_weight_drift_per_tau records compare
whole steps. These different diagnostics must not be directly equated.

Do not launch a second writer for the same output. Jobs are ordinary finite
processes, without a watchdog or resurrection loop. The local Python PID is in
`dmi/data/logs/plateau_explore_20261113.pid` (historical record); Julia's historical PID record is also in its **dmi/data/logs/** directory. Terminating a job does not remove completed stage checkpoints.

`analysis/plateau_report.py` reads both independently computed outputs; it does not run
Julia from Python. The running Python job updates the report after each saved
point. It can also be run independently to refresh Julia-only progress.
Report directory: `iPEPS_Z2/kagome_DMI_plateau_review_20261113/`.
Only valid, CTM-converged observations are drawn. Failed measurements remain
flagged and their final SU states can be remeasured later.

Inspect both branches and their energies. A flat continuation branch can be
metastable even without imposing U1; a single point close to 1/9 is not a plateau.
Potential plateaux require fresh center-field seeds, longer evolution, chi
checks and checking competing lower-energy branches. The best_points files
choose the lower measured energy only within the same ansatz/D/Dz/field, not
between different ansatz at a nominally equal D.
