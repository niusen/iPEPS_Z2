# Server jobs: independent random initial state at every magnetic field

Both jobs use J=1, Dz/J=0.02/0.05/0.1, D=2/3/4 and
h/J=0.28/0.32/0.36/0.38/0.40/0.44/0.48. Each physical point has THREE independent
complex Gaussian random kets (`trials=[1,2,3]`), each with its own recorded seed. No U1/spatial C3 projection, no
neighbor-field state, no previous-D seed, no old plateau state. Own-point stage
checkpoints may be resumed. Three seeds reduce initialization bias but do not
guarantee a global minimum. Add seeds near competing branches or field boundaries.
Old lower-energy states remain useful energy references.

Each job has 63 physical points x 3 seeds = 189 independent optimizations
(378 across both ansatzes), about three times the single-seed work.
EVERY trial runs the full five-stage SU schedule:
dt=(.03,.01,.003,.001,.0005), tau=(40,20,5,2,1), 9000 complete steps.
There is one final measurement; rejected measurements may have explicit chi
remeasurements with saved error diagnostics. This is more costly than continuing
from neighboring fields. Only small-D short checks were run locally.

## Python

Edit **dmi/server/run_independent_random_server.py**, then in **dmi/server/** run
`sh run_independent_random_server.sh`. The script selects the existing pytorch
conda environment, starts ONE background process and prints log/PID paths.
Logs and PID records go to **dmi/data/logs/**.
It does not restart a killed process. Stop with the printed `kill -TERM PID`.
Re-submit unchanged settings to resume completed stages/points. An interrupted
stage repeats from its saved start; this is not exact within-step resumption.

Default SU=CPU, threads=10; CTMRG and observables=`cuda:1`, chi=12, max100,
tol1e-6, including diagnostic remeasurements. Change `measurement_device` to `cpu` in the settings file if needed.
The preflight rejects missing CUDA/device1 BEFORE any SU. Exact GPU thin SVD
uses gesvd, following the existing checked driver convention. No randomized SVD.

Upload/update **kagome_hexagon_triangle source code**, excluding `results/`,
`logs/`, `dmi/data/`, `__pycache__/` and *.log/*.pid. Keep its original directory structure
inside the Python repository; the job imports its existing geometry, SU, CTM,
paper_method and server helpers. Do not upload only the .sh and .py entry files.
Use the server's existing torch/numpy/opt_einsum environment; no vendor or old
result/seed folders are needed.

Output: `kagome_hexagon_triangle/results/DMI_independent_random3_D2_D4_server/`.
It contains settings, seed plan, per-field initialization records and audits,
initial/stage/final.pt, full observations and diagnostics. **analysis/** contains
the auto-updated HTML/PNG, all_points.csv, best_points.csv and site_magnetization.csv.
All seed results are retained; curves select the lowest energy among accepted
seeds at each (D,Dz,h), without averaging magnetization across different branches.
Light markers show other accepted seeds. An incomplete set of trials is provisional.

## Pure Julia TensorKit

Settings are in
`Kagome/simple_update/multicell/dmi/server/run_independent_random_D2_D4.jl`.
From that **dmi/server/** directory run **sh run_independent_random_D2_D4.sh**.
Default CPU throughout, 10 threads, chi12, CTMRG max50, tol1e-6; an unconverged
measurement can get a chi24 fixed-ket check. No Python numerical process is
called. Reuses the existing multicell Project.toml/Manifest.toml, compatible SU
kernels and factorized CTMRG. All initialized virtual spaces are ComplexSpace;
the SU2/U1 module names in reused code do not impose symmetry on these states.

Upload the new `src/bosonic/kagome/dmi/independent_random_fields.jl` and the
run .jl/.sh in **dmi/server/**. If the server has not received the DMI implementation, synchronize
the whole **src/** source directory first, including kagome/dmi and
kagome/native_measurement dependencies. Keep the existing Project/Manifest at
the parent multicell level. A parent renamed `u1_multicell` also works with the
same relative directory depth. Logs/PID records are now in **dmi/data/logs/**. No old U1 or SU2 optimized seed is required.

Output: parent multicell `results/DMI_independent_random3_D2_D4_20261114/`, with
plan.json/settings.toml, initialization.toml/audit, initial/stage/final.jld2,
independent_result.json/.toml, diagnostics and **progress.csv**.
Julia uses internal h=-h_physical; output magnetization is +2<Sz> like Python.

Download the results for joint plotting with **dmi/analysis/independent_random_report.py**;
its run() accepts explicit python_root, julia_root and output paths. Comparing
old results is optional analysis only and never supplies an initial ket.

These new three-seed jobs use new output directories so the old single-seed
settings/checkpoints are untouched. Do not change trials in an already started
output directory: the immutable-settings check will reject it. The same .sh
filenames are used; update the settings files and Python report/recheck code.

## Reproducibility and validity

Local two-field D2 tests verified distinct actual initial tensors, distinct seeds,
and own-field resumption in BOTH implementations. Julia additionally regenerates
the initial B tensor from the recorded seed and checks it after every point;
Python verifies the saved seed and hashes the initial checkpoint. Raw failures
are retained. Measurement validity and CTM convergence are separate flags.
Some Python remeasurements use explicitly labelled 1e-6 imaginary/face and 1e-5
chi-stability criteria rather than pretending the original 1e-8 guard passed.
Skipped/invalid points must not be interpreted as absence of a plateau.
