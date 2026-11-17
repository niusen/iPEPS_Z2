# Out-of-plane kagome DMI

New, independent files; the existing Heisenberg workflows and kernels are unchanged.
The tensor network is the existing 27-spin hexagon–triangle ansatz, with optional
spatial C3. Complex dense tensors are used; set `c3=False` for unrestricted patterns.

`H = J sum_<ij> Si.Sj + Dz sum_CCW (Sxi Syj - Syi Sxj) - h sum_i Szi`.
Every elementary triangle has counterclockwise arrows in the Cartesian coordinates
of `geometry.py`, for both up/down triangles. `J` and `Dz` are independent; pure
DMI (`J=0`) is supported. `Dz` defaults to zero. The directed term is
`(i Dz/2)(Sp_i Sm_j - Sm_i Sp_j)`; it conserves total Sz and breaks SU2.

## Use

Files are separated by purpose:

```text
dmi/
  src/        model, SVD, simple update, CTMRG/observables, scan workflows
  server/     editable run settings and background .sh launchers
  analysis/   reports, plots and fixed-state remeasurements
  tests/      D=1/2 checks and historical local postprocessing helper
  data/logs/  local/server logs and PID records
  docs/       experiment and server-upload instructions
```

The current three-seed server scan is configured in
[server/run_independent_random_server.py](server/run_independent_random_server.py).
From **dmi/server/** run `sh run_independent_random_server.sh`.
The current server settings use D=6,8, twelve nonnegative Dz values from 0 to
0.7, eleven fields from h=0.28 to 0.64, and three independent random trials
per point (792 optimizations). CPU SU and cuda:1 CTMRG/observables are retained;
chi=12, CTMRG max100, tolerance=1e-6. The new D6/D8 output folder preserves
the earlier D2--4 results. No saved platform state is used.
See [server instructions](docs/SERVER_INDEPENDENT_RANDOM.md) for both packages.
The server saves data only, including **progress.csv** and complete per-point
JSON/checkpoints; matplotlib is not required there. After downloading results,
edit paths in **analysis/run_local_report.py** and run
`python -m kagome_hexagon_triangle.dmi.analysis.run_local_report` on the laptop.

For a single run, edit **server/run.py**, then from the repository root run
`python -m kagome_hexagon_triangle.dmi.server.run`, or run `sh run.sh` inside
**dmi/server/**.
The shell uses the same `pytorch` conda selection as the existing jobs, runs in
the background, and writes its log and PID to **dmi/data/logs/**. There is no restart loop.

Numerical results/checkpoints retain their existing locations under
**kagome_hexagon_triangle/results/**. Moving the source files does not change
output directories or restart settings. The old flat module names have moved;
upload all current DMI source subfolders rather than just the entry files.

- CPU SU, optional GPU CTMRG/observables (`cuda:1` in the server example).
- Five stages use the previous dt/tau schedule, with **one final measurement**.
- `input` accepts an old or DMI checkpoint at the same D. Set a fresh `out`.
  An explicit restart from `latest.pt` starts the configured schedule again;
  it does not pretend to continue from the exact interrupted Trotter phase.
- State files record J,Dz,h and a separate DMI format. Do not send these to the
  existing Heisenberg-only energy-variance workflow.
- Output: `initial/latest/final.pt`, `observables.json`, `summary.csv`,
  `site_magnetization.csv`, `bond_energies.csv`. Each physical bond includes
  exchange correlation, CCW vector chirality, exchange energy and DMI energy.
- CUDA uses exact thin SVD with `driver='gesvd'`, matching the checked CTM
  convention; CPU omits the driver. No subspace/randomized SVD is introduced.

## Small checks

`python -m kagome_hexagon_triangle.dmi.tests.test_small` runs D=1/2 checks only:
Hermiticity, U1 conservation, reversing a bond, operator-Schmidt reconstruction,
Dz=0 old/new SU energy, complex CTM insertions, field counting, checkpoint and
output roundtrips, and a short pure-DMI D2 evolution.

For the spin-1/2 q=0 planar product state with sublattice angles 0,120,240 degrees,
every CCW bond has kappa_z=-sqrt(3)/8. Therefore at J=h=0 its per-spin energy is
`-sqrt(3)*Dz/4 = -0.4330127018922193*Dz`; reversing chirality reverses its energy.
This is a geometry/sign benchmark, **not an exact quantum ground-state energy**.
The test also attempts a tiny GPU check; an explicitly reported missing local
NVRTC DLL is an environment limitation, not evidence of GPU validation.

Julia retains the user's original **+h Sz** convention and m=-2<Sz>; Python uses
**-h Sz** and m=+2<Sz>. Compare h=0 directly, or reverse the h parameter when
measuring the same ket across implementations. DMI arrows/sign are identical.
