"""Bosonic/dense CTMRG for triangular-lattice iPESS unit cells.

The double layer is kept factorized as two rank-3 tensors,
``B_double=(L,M,U)`` and ``T_double=(D,R,M)``, throughout the unit cell.
The mature directional CTM update kernels are shared with the fermionic
implementation; fermionic double-layer construction is deliberately kept in
``Fermionic_CTMRG_unitcell_iPESS.py``.
"""

from collections import OrderedDict

import numpy
import torch
import yastn
from torch.utils.checkpoint import checkpoint

import ctmrg.Fermionic_CTMRG_unitcell_iPESS as _ctm_core


def _key(cx, cy):
    return str(cx) + "," + str(cy)


def build_double_layer_Tm(Ap, A, with_physical=False):
    """Build rank-3 ``B_double=(L,M,U)`` without graded swaps."""
    if A.config.fermionic:
        return _ctm_core.build_double_layer_swap_Tm(Ap, A, with_physical)
    if with_physical:
        raise ValueError("The B simplex tensor must not carry a physical leg")
    if Ap.get_rank() != 3 or A.get_rank() != 3:
        raise ValueError("The B simplex tensor must have rank 3")
    double = yastn.ncon([Ap, A], [[-1, -3, -5], [-2, -4, -6]])
    return double.fuse_legs(((0, 1), (4, 5), (2, 3)), mode="hard")


def build_double_layer_Bm(Ap, A, with_physical=True):
    """Build rank-3 ``T_double=(D,R,M)`` by contracting the spin leg."""
    if A.config.fermionic:
        return _ctm_core.build_double_layer_swap_Bm(Ap, A, with_physical)
    if not with_physical:
        raise ValueError("The T site tensor must carry one physical leg")
    if Ap.get_rank() != 4 or A.get_rank() != 4:
        raise ValueError("The T site tensor must have rank 4: (M,s,R,D)")
    double = yastn.ncon([Ap, A], [[-1, 1, -3, -5], [-2, 1, -4, -6]])
    return double.fuse_legs(((4, 5), (2, 3), (0, 1)), mode="hard")


# The observable code historically uses these names.  In this neutral module
# they dispatch to the exact fermionic implementation for graded tensors and
# to the bosonic implementation above for dense tensors.
build_double_layer_swap_Tm = build_double_layer_Tm
build_double_layer_swap_Bm = build_double_layer_Bm


def build_doublelayer_iPESS(B_set, T_set, pos):
    key = _key(pos[0], pos[1])
    B = B_set[key]
    T = T_set[key]
    B_double = build_double_layer_Tm(B.conj(), B, with_physical=False)
    T_double = build_double_layer_Bm(T.conj(), T, with_physical=True)
    return T_double, B_double


def init_CTM_cell(B_set, T_set, ctm_setting, global_args):
    if ctm_setting.CTM_ite_info:
        print("initialize CTM from bosonic iPESS")
    Cset_cell = _ctm_core.initial_cell(global_args.Lx, global_args.Ly)
    Tset_cell = _ctm_core.initial_cell(global_args.Lx, global_args.Ly)
    for cx in range(1, global_args.Lx + 1):
        for cy in range(1, global_args.Ly + 1):
            T_double, B_double = build_doublelayer_iPESS(B_set, T_set, (cx, cy))
            Cset, Tset = _ctm_core.init_CTM_swap(T_double, B_double)
            Cset_cell[_key(cx, cy)] = Cset
            Tset_cell[_key(cx, cy)] = Tset
    return OrderedDict((('Cset', Cset_cell), ('Tset', Tset_cell)))


def initial_trivial_ctm(Cset_cell, Tset_cell, ctm_setting, global_args):
    numpy.random.seed(123)
    torch.manual_seed(123)
    config = Cset_cell['1,1']['C1'].config
    dim = ctm_setting.chi
    for cx in range(1, global_args.Lx + 1):
        for cy in range(1, global_args.Ly + 1):
            key = _key(cx, cy)
            for direction in (1, 2, 3, 4):
                C = Cset_cell[key]['C' + str(direction)]
                legs = C.get_legs()
                clegs = [
                    yastn.Leg(config, s=legs[0].s, D=(dim,)),
                    yastn.Leg(config, s=legs[1].s, D=(dim,)),
                ]
                Cset_cell[key]['C' + str(direction)] = yastn.rand(config=config, legs=clegs)

                T = Tset_cell[key]['T' + str(direction)]
                legs = T.get_legs()
                tlegs = [
                    yastn.Leg(config, s=legs[0].s, D=(dim,)),
                    legs[1],
                    yastn.Leg(config, s=legs[2].s, D=(dim,)),
                ]
                Tset_cell[key]['T' + str(direction)] = yastn.rand(config=config, legs=tlegs)
    return Cset_cell, Tset_cell


def CTMRG_cell_iPESS(B_set, T_set, init, CTM0, ctm_setting, global_args):
    """Run bosonic CTMRG while retaining factorized rank-3 double layers."""
    if B_set['1,1'].config.fermionic or T_set['1,1'].config.fermionic:
        raise ValueError(
            "CTMRG_cell_iPESS is the bosonic implementation; use "
            "Fermionic_CTMRG_cell_iPESS for graded tensors"
        )
    if not init.reconstruct_AA:
        raise ValueError("Bosonic CTMRG currently requires reconstruct_AA=True")

    Lx, Ly = global_args.Lx, global_args.Ly
    chi = ctm_setting.chi
    double_B_cell = _ctm_core.initial_cell(Lx, Ly)
    double_T_cell = _ctm_core.initial_cell(Lx, Ly)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            T_double, B_double = build_doublelayer_iPESS(B_set, T_set, (cx, cy))
            double_B_cell[_key(cx, cy)] = B_double
            double_T_cell[_key(cx, cy)] = T_double
    if ctm_setting.doublelayer_on_cpu:
        double_B_cell = _ctm_core.Cell_to_device(double_B_cell, 'cpu', global_args)
        double_T_cell = _ctm_core.Cell_to_device(double_T_cell, 'cpu', global_args)

    if init.reconstruct_CTM:
        CTM_cell = init_CTM_cell(B_set, T_set, ctm_setting, global_args)
    else:
        CTM_cell = _ctm_core.CTM_copy(CTM0, global_args)
    Cset_cell, Tset_cell = CTM_cell['Cset'], CTM_cell['Tset']

    if ctm_setting.trivial_initial_CTM:
        Cset_cell = _ctm_core.Cset_detach(Cset_cell, global_args)
        Tset_cell = _ctm_core.Tset_detach(Tset_cell, global_args)
        with torch.no_grad():
            Cset_cell, Tset_cell = initial_trivial_ctm(
                Cset_cell, Tset_cell, ctm_setting, global_args
            )
        Cset_cell = _ctm_core.Cset_requires_grad_(Cset_cell, global_args)
        Tset_cell = _ctm_core.Tset_requires_grad_(Tset_cell, global_args)

    old = [
        torch.ones((Lx, Ly, chi * 2), dtype=torch.float64, device=B_set['1,1'].device)
        for _ in range(4)
    ]
    new = [torch.zeros_like(value) for value in old]
    errors = [torch.zeros((Lx, Ly)) for _ in range(4)]
    errors[3].fill_(1)

    if ctm_setting.CTM_ite_info:
        print("start CTM iterations:")
    ite_num, ite_err = 0, 1.0
    for ci in range(1, ctm_setting.CTM_ite_nums + 1):
        ite_num = ci
        for direction in (3, 4, 1, 2):
            Cset_cell, Tset_cell = checkpoint(
                _ctm_core.CTM_ite_cell_continuous_update,
                Cset_cell,
                Tset_cell,
                double_B_cell,
                double_T_cell,
                direction,
                ctm_setting,
                global_args,
                use_reentrant=False,
            )

        with torch.no_grad():
            for cx in range(1, Lx + 1):
                for cy in range(1, Ly + 1):
                    key = _key(cx, cy)
                    for corner in range(4):
                        error, spectrum = _ctm_core.spectrum_conv_check(
                            old[corner][cx - 1, cy - 1, :],
                            Cset_cell[key]['C' + str(corner + 1)],
                        )
                        errors[corner][cx - 1, cy - 1] = error
                        new[corner][cx - 1, cy - 1, :].zero_()
                        new[corner][cx - 1, cy - 1, :len(spectrum)] = spectrum

            ite_err = max(torch.max(error).item() for error in errors)
            if ctm_setting.CTM_ite_info:
                print("CTMRG iteration: " + str(ci) + ", CTMRG err: " + str(ite_err))
            if ite_err < ctm_setting.CTM_conv_tol:
                break
            old = [value.clone() for value in new]

    CTM_cell = OrderedDict((('Cset', Cset_cell), ('Tset', Tset_cell)))
    return CTM_cell, double_B_cell, double_T_cell, ite_num, ite_err


Bosonic_CTMRG_cell_iPESS = CTMRG_cell_iPESS

# Statistics-independent helpers used by optimization and measurements.
Cell_to_device = _ctm_core.Cell_to_device
CTM_detach = _ctm_core.CTM_detach
