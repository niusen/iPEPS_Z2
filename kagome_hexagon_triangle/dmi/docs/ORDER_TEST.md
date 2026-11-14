# Pure-DMI 120-degree order probe

`src/order_probe.py` runs CPU dense SU at J=0, Dz=1, h=0, D=2,3,4. From the
repository root run `python -m kagome_hexagon_triangle.dmi.src.order_probe`. It uses one
general complex random seed per D and one independently embedded q=0 coherent
120-degree reference per D. No U1 or pure spatial C3 projection is applied.
The reference is an initial ket, not a constraint; its Schmidt channels may grow.

All options are in the source. The four stages `(dt,tau)` are
`(.05,8), (.02,4), (.005,1), (.001,.2)`: 760 complete steps, total tau=13.2.
These are bounded small-D tests, not a global-energy optimization campaign.
Only final CTMRG measurements are taken, at chi=12; D=4 checks chi=20 too.
If the strict 1e-8 Hermiticity check fails, the same saved ket is measured with
a tighter CTMRG stopping rule and, if needed, chi=24/28. Failed checks are saved
explicitly and never published as successful observables. A valid baseline may
be retained while a higher-chi refinement remains uncertified.

Every case saves initial/final states, all stages' timing/weight drift, Schmidt
spectra and site/bond observables. A restarted probe resumes completed SU stages
and existing measurements. There is no random-seed search or auto-restart loop.

Julia's corresponding CPU run file is
`SU2_AD/Kagome/simple_update/multicell/dmi/experiments/test_120_D2_D4.jl`; its functional
helper is `SU2_AD/src/bosonic/kagome/dmi/order_probe.jl`. It uses the same schedule
with dense iPESS and the nine-spin sqrt3 cell. Python uses the 27-spin
hexagon–triangle ansatz. Equal D does not imply equal variational spaces.

`analysis/analyze_order_probe.py` only reads the separately computed JSON files and makes
a common report. Its local paths are at the top. Output is
`iPEPS_Z2/kagome_dmi_order_review_20261113/`, including HTML, PNG/PDF, metric CSV,
all site moments and all bond DMI correlations.

The diagnostic is M120 = |mean[(Sx+iSy) exp(-2pi i b/3)]| for sublattices b=0,1,2.
Its ratio to the mean transverse moment is the q=0 coherence. Coherence near one
AND a nonzero amplitude show the three-sublattice 120-degree pattern. All plots
remove one common global spin phase per ket; sites are not individually rotated.
Nearest-neighbor angles and residual Sz are also checked. The analytic product
energy -sqrt(3)/4 is only a reference, not the quantum ground-state energy.
