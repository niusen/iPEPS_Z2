import copy
from collections import OrderedDict

import numpy
import torch
import yastn
from torch.utils.checkpoint import checkpoint

from ansatz.square_iPEPS import (
    CTM_copy,
    Cset_detach,
    Cset_requires_grad_,
    Tset_detach,
    Tset_requires_grad_,
    build_double_layer,
)
from config.settings import *


def spectrum_conv_check(ss_old, C_new):
    U, spec, V = yastn.linalg.svd(C_new, svd_on_cpu=True)
    spec = spec.to_dense()
    ss_new = torch.diag(spec / spec[0, 0])
    ss_new = ss_new.to(ss_old.device)

    if len(ss_old) > len(ss_new):
        dss = copy.deepcopy(ss_old)
        siz = len(ss_new)
    else:
        dss = copy.copy(ss_new)
        siz = len(ss_old)

    dss[0:siz] = ss_old[0:siz] - ss_new[0:siz]
    er = torch.Tensor.norm(dss)
    return er, ss_new


def _cell_key(cx, cy):
    return str(cx) + "," + str(cy)


def _get_lattice_size(global_args):
    if isinstance(global_args, dict):
        return global_args["Lx"], global_args["Ly"]
    return global_args.Lx, global_args.Ly


def _get_device(global_args):
    if isinstance(global_args, dict):
        return global_args["default_device"]
    return global_args.device


def build_doublelayer_iPEPS(A_set, pos):
    c1 = pos[0]
    c2 = pos[1]
    A = A_set[_cell_key(c1, c2)]
    return build_double_layer(A.conj(), A)


def build_doublelayer_iPEPS_cell(A_set, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    double_A_cell = initial_cell(Lx, Ly)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            double_A_cell[_cell_key(cx, cy)] = build_doublelayer_iPEPS(A_set, (cx, cy))
    return double_A_cell


def convert_cell_posit(cx, cy, dx, dy, direction, Lx, Ly):
    if direction == 1:
        posit = (mod1(cx + dx, Lx), mod1(cy + dy, Ly))
    elif direction == 2:
        posit = (mod1(cy - dy, Lx), mod1(cx + dx, Ly))
    elif direction == 3:
        posit = (mod1(cx - dx, Lx), mod1(cy - dy, Ly))
    elif direction == 4:
        posit = (mod1(cy + dy, Lx), mod1(cx - dx, Ly))
    return posit


def rotate_AA_direction(AA_fused, direction):
    return yastn.transpose(
        AA_fused,
        axes=(mod1(2 - direction, 4) - 1, mod1(3 - direction, 4) - 1, mod1(4 - direction, 4) - 1, mod1(1 - direction, 4) - 1),
    )


def get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args):
    AA = double_A_cell[_cell_key(pos[0], pos[1])]
    device = _get_device(global_args)
    if ctm_setting.doublelayer_on_cpu and device != "cpu":
        AA = AA.to(device)
    AA = AA.switch_signature(axes=(2, 3))
    return rotate_AA_direction(AA, direction)


def init_CTM_iPEPS(AA):
    Cset = initial_Cset()
    Tset = initial_Tset()

    AA_LU = AA.unfuse_legs(axes=(0, 3))
    C1 = yastn.ncon([AA_LU], [[1, 1, -1, -2, 2, 2]]).switch_signature(axes=(1,))

    AA_RU = AA.unfuse_legs(axes=(2, 3))
    C2 = yastn.ncon([AA_RU], [[-1, -2, 1, 1, 2, 2]])

    AA_DR = AA.unfuse_legs(axes=(1, 2))
    C3 = yastn.ncon([AA_DR], [[-2, 1, 1, 2, 2, -1]]).switch_signature(axes=(0,))

    AA_LD = AA.unfuse_legs(axes=(0, 1))
    C4 = yastn.ncon([AA_LD], [[1, 1, 2, 2, -1, -2]]).switch_signature(axes=(0, 1))

    AA_L = AA.unfuse_legs(axes=(0,))
    T4 = yastn.ncon([AA_L], [[1, 1, -1, -2, -3]]).switch_signature(axes=(1, 2))

    AA_U = AA.unfuse_legs(axes=(3,))
    T1 = yastn.ncon([AA_U], [[-1, -2, -3, 1, 1]]).switch_signature(axes=(2,))

    AA_R = AA.unfuse_legs(axes=(2,))
    T2 = yastn.ncon([AA_R], [[-2, -3, 1, 1, -1]]).switch_signature(axes=(0,))

    AA_D = AA.unfuse_legs(axes=(1,))
    T3 = yastn.ncon([AA_D], [[-3, 1, 1, -1, -2]]).switch_signature(axes=(0, 1))

    Cset["C1"] = C1
    Cset["C2"] = C2
    Cset["C3"] = C3
    Cset["C4"] = C4
    Tset["T1"] = T1
    Tset["T2"] = T2
    Tset["T3"] = T3
    Tset["T4"] = T4
    return Cset, Tset


def init_CTM_cell(A_set, ctm_setting, global_args):
    if ctm_setting.CTM_ite_info:
        print("initialize CTM from square iPEPS")

    Lx, Ly = _get_lattice_size(global_args)
    Cset_cell = initial_cell(Lx, Ly)
    Tset_cell = initial_cell(Lx, Ly)

    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            AA = build_doublelayer_iPEPS(A_set, (cx, cy))
            Cset, Tset = init_CTM_iPEPS(AA)
            Cset_cell[_cell_key(cx, cy)] = Cset
            Tset_cell[_cell_key(cx, cy)] = Tset

    CTM_cell = OrderedDict()
    CTM_cell["Cset"] = Cset_cell
    CTM_cell["Tset"] = Tset_cell
    return CTM_cell


def initial_trivial_ctm(Cset_cell, Tset_cell, ctm_setting, global_args):
    seed_value = 123
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    Lx, Ly = _get_lattice_size(global_args)
    config_dense = Cset_cell["1,1"]["C1"].config
    dim = ctm_setting.chi

    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = _cell_key(cx, cy)
            for direction in [1, 2, 3, 4]:
                C_ = Cset_cell[key]["C" + str(direction)]
                legs = C_.get_legs()
                leg1 = yastn.Leg(config_dense, s=legs[0].s, D=(dim,))
                leg2 = yastn.Leg(config_dense, s=legs[1].s, D=(dim,))
                Cset_cell[key]["C" + str(direction)] = yastn.rand(config=config_dense, legs=[leg1, leg2])

                T_ = Tset_cell[key]["T" + str(direction)]
                legs = T_.get_legs()
                leg1 = yastn.Leg(config_dense, s=legs[0].s, D=(dim,))
                leg3 = yastn.Leg(config_dense, s=legs[2].s, D=(dim,))
                Tset_cell[key]["T" + str(direction)] = yastn.rand(config=config_dense, legs=[leg1, legs[1], leg3])

    return Cset_cell, Tset_cell


def build_corner_MMup(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    pos = convert_cell_posit(coord[0], coord[1], 1, 1, direction, Lx, Ly)
    AA = get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args)
    pos = convert_cell_posit(coord[0], coord[1], 0, 0, direction, Lx, Ly)
    C1 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 1, 0, direction, Lx, Ly)
    T1 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 0, 1, direction, Lx, Ly)
    T4 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 1, 4))]
    return yastn.ncon([C1, T1, T4, AA], [[1, 2], [2, 3, -3], [-1, 4, 1], [4, -2, -4, 3]])


def build_corner_MMlow(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    pos = convert_cell_posit(coord[0], coord[1], 1, 2, direction, Lx, Ly)
    AA = get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args)
    pos = convert_cell_posit(coord[0], coord[1], 0, 2, direction, Lx, Ly)
    T4 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 1, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 0, 3, direction, Lx, Ly)
    C4 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction - 1, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 1, 3, direction, Lx, Ly)
    T3 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 2, 4))]
    return yastn.ncon([T4, AA, C4, T3], [[1, 3, -1], [3, 4, -4, -2], [2, 1], [-3, 4, 2]])


def build_corner_MMup_reflect(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    pos = convert_cell_posit(coord[0], coord[1], 2, 1, direction, Lx, Ly)
    AA = get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args)
    pos = convert_cell_posit(coord[0], coord[1], 2, 0, direction, Lx, Ly)
    T1 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 3, 0, direction, Lx, Ly)
    C2 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction + 1, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 3, 1, direction, Lx, Ly)
    T2 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction + 1, 4))]
    return yastn.ncon([T1, C2, AA, T2], [[-1, 3, 1], [1, 2], [-2, -4, 4, 3], [2, 4, -3]])


def build_corner_MMlow_reflect(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    pos = convert_cell_posit(coord[0], coord[1], 2, 2, direction, Lx, Ly)
    AA = get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args)
    pos = convert_cell_posit(coord[0], coord[1], 3, 2, direction, Lx, Ly)
    T2 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction + 1, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 2, 3, direction, Lx, Ly)
    T3 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 2, 4))]
    pos = convert_cell_posit(coord[0], coord[1], 3, 3, direction, Lx, Ly)
    C3 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction - 2, 4))]
    MMlow_reflect = yastn.ncon([T2, T3, C3], [[-4, -3, 2], [1, -2, -1], [2, 1]])
    return yastn.ncon([MMlow_reflect, AA], [[-1, 1, 2, -3], [-2, 1, 2, -4]])


def get_M(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args):
    MMup = build_corner_MMup(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args)
    MMlow = build_corner_MMlow(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args)
    MMup_reflect = build_corner_MMup_reflect(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args)
    MMlow_reflect = build_corner_MMlow_reflect(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args)

    RMup = yastn.ncon([MMup, MMup_reflect], [[-3, -4, 1, 2], [1, 2, -1, -2]])
    RMlow = yastn.ncon([MMlow, MMlow_reflect], [[-1, -2, 1, 2], [1, 2, -3, -4]])

    RMlow = RMlow / yastn.linalg.norm(RMlow)
    RMup = RMup / yastn.linalg.norm(RMup)

    M = yastn.ncon([RMup, RMlow], [[-1, -2, 1, 2], [1, 2, -3, -4]])
    M = M / yastn.linalg.norm(M)

    if ctm_setting.doublelayer_on_cpu:
        return M.to("cpu"), RMup.to("cpu"), RMlow.to("cpu")
    return M, RMup, RMlow


def final_CTM_update(Cset_cell, Tset_cell, M1tem_cell, M5tem_cell, M7tem_cell, coord, direction, Lx, Ly):
    pos = convert_cell_posit(coord[0], coord[1], 1, 0, direction, Lx, Ly)
    Cset_new = update_CTM_C(Cset_cell[_cell_key(pos[0], pos[1])], M1tem_cell[_cell_key(pos[0], pos[1])], mod1(direction, 4))
    Cset_cell = update_cell(Cset_cell, Cset_new, pos[0], pos[1], Lx, Ly)

    pos = convert_cell_posit(coord[0], coord[1], 1, 2, direction, Lx, Ly)
    Tset_new = update_CTM_T(Tset_cell[_cell_key(pos[0], pos[1])], M5tem_cell[_cell_key(pos[0], pos[1])], mod1(direction - 1, 4))
    Tset_cell = update_cell(Tset_cell, Tset_new, pos[0], pos[1], Lx, Ly)

    pos = convert_cell_posit(coord[0], coord[1], 1, 3, direction, Lx, Ly)
    Cset_new = update_CTM_C(Cset_cell[_cell_key(pos[0], pos[1])], M7tem_cell[_cell_key(pos[0], pos[1])], mod1(direction - 1, 4))
    Cset_cell = update_cell(Cset_cell, Cset_new, pos[0], pos[1], Lx, Ly)
    return Cset_cell, Tset_cell


def ctm_update_single_cx(cx, cy_max, Cset_cell, Tset_cell, double_A_cell, direction, ctm_setting, global_args):
    chi = ctm_setting.chi

    def truncation_f(S):
        return yastn.linalg.truncation_mask_multiplets(
            S, keep_multiplets=True, D_total=chi, tol=ctm_setting.CTM_trun_tol, tol_block=0.0, eps_multiplet=1.0e-8
        )

    Lx, Ly = _get_lattice_size(global_args)
    device = _get_device(global_args)
    PM_cell = initial_cell(Lx, Ly)
    PM_inv_cell = initial_cell(Lx, Ly)
    M1tem_cell = initial_cell(Lx, Ly)
    M5tem_cell = initial_cell(Lx, Ly)
    M7tem_cell = initial_cell(Lx, Ly)

    for cy in range(1, cy_max + 1):
        coord = [cx, cy]
        if ctm_setting.use_sub_checkpoint:
            M, RMup, RMlow = checkpoint(get_M, coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args, use_reentrant=False)
        else:
            M, RMup, RMlow = get_M(coord, direction, double_A_cell, Cset_cell, Tset_cell, ctm_setting, global_args)

        chi_extra = 3
        uM, sM, vM = yastn.linalg.svd_with_truncation(
            M.to(device),
            axes=((0, 1), (2, 3)),
            D_total=chi + chi_extra,
            svd_on_cpu=ctm_setting.svd_on_cpu,
            tol=ctm_setting.CTM_trun_tol,
            mask_f=truncation_f,
        )
        if ctm_setting.doublelayer_on_cpu:
            uM = uM.to(device)
            sM = sM.to(device)
            vM = vM.to(device)

        sM = sM / yastn.linalg.norm(sM)
        sM_inv_sqrt = sM.rsqrt(cutoff=ctm_setting.CTM_trun_tol)

        vMp = vM.conj()
        if ctm_setting.doublelayer_on_cpu:
            PM_inv = yastn.ncon([RMlow.to(device), vMp, sM_inv_sqrt], [[-1, -2, 1, 2], [3, 1, 2], [3, -3]])
        else:
            PM_inv = yastn.ncon([RMlow, vMp, sM_inv_sqrt], [[-1, -2, 1, 2], [3, 1, 2], [3, -3]])

        uMp = uM.conj()
        if ctm_setting.doublelayer_on_cpu:
            PM = yastn.ncon([sM_inv_sqrt, uMp, RMup.to(device)], [[-3, 3], [1, 2, 3], [1, 2, -1, -2]])
        else:
            PM = yastn.ncon([sM_inv_sqrt, uMp, RMup], [[-3, 3], [1, 2, 3], [1, 2, -1, -2]])

        pos = convert_cell_posit(coord[0], coord[1], 0, 2, direction, Lx, Ly)
        PM_cell[_cell_key(pos[0], pos[1])] = PM
        pos = convert_cell_posit(coord[0], coord[1], 0, 1, direction, Lx, Ly)
        PM_inv_cell[_cell_key(pos[0], pos[1])] = PM_inv

    for cy in range(1, cy_max + 1):
        coord = [cx, cy]
        pos = convert_cell_posit(coord[0], coord[1], 1, 2, direction, Lx, Ly)
        AA = get_AA_direction(double_A_cell, direction, pos, ctm_setting, global_args)
        pos = convert_cell_posit(coord[0], coord[1], 0, 2, direction, Lx, Ly)
        T4 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 1, 4))]
        pos = convert_cell_posit(coord[0], coord[1], 1, 0, direction, Lx, Ly)
        T1 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction, 4))]
        pos = convert_cell_posit(coord[0], coord[1], 1, 3, direction, Lx, Ly)
        T3 = Tset_cell[_cell_key(pos[0], pos[1])]["T" + str(mod1(direction - 2, 4))]
        pos = convert_cell_posit(coord[0], coord[1], 0, 0, direction, Lx, Ly)
        C1 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction, 4))]
        pos = convert_cell_posit(coord[0], coord[1], 0, 3, direction, Lx, Ly)
        C4 = Cset_cell[_cell_key(pos[0], pos[1])]["C" + str(mod1(direction - 1, 4))]

        posa = convert_cell_posit(coord[0], coord[1], 0, 2, direction, Lx, Ly)
        posb = convert_cell_posit(coord[0], coord[1], 0, 2, direction, Lx, Ly)
        M5tem = yastn.ncon(
            [T4, AA, PM_inv_cell[_cell_key(posa[0], posa[1])], PM_cell[_cell_key(posb[0], posb[1])]],
            [[4, 3, 1], [3, 5, -2, 2], [4, 5, -1], [1, 2, -3]],
        )
        pos = convert_cell_posit(coord[0], coord[1], 0, 0, direction, Lx, Ly)
        M1tem = yastn.ncon([C1, T1, PM_inv_cell[_cell_key(pos[0], pos[1])]], [[1, 2], [2, 3, -2], [1, 3, -1]])
        pos = convert_cell_posit(coord[0], coord[1], 0, 3, direction, Lx, Ly)
        M7tem = yastn.ncon([C4, T3, PM_cell[_cell_key(pos[0], pos[1])]], [[1, 2], [-1, 3, 1], [2, 3, -2]])

        M5tem = M5tem / yastn.linalg.norm(M5tem)
        M1tem = M1tem / yastn.linalg.norm(M1tem)
        M7tem = M7tem / yastn.linalg.norm(M7tem)

        pos = convert_cell_posit(coord[0], coord[1], 1, 2, direction, Lx, Ly)
        M5tem_cell[_cell_key(pos[0], pos[1])] = M5tem
        pos = convert_cell_posit(coord[0], coord[1], 1, 0, direction, Lx, Ly)
        M1tem_cell[_cell_key(pos[0], pos[1])] = M1tem
        pos = convert_cell_posit(coord[0], coord[1], 1, 3, direction, Lx, Ly)
        M7tem_cell[_cell_key(pos[0], pos[1])] = M7tem

    for cy in range(1, cy_max + 1):
        coord = [cx, cy]
        Cset_cell, Tset_cell = checkpoint(final_CTM_update, Cset_cell, Tset_cell, M1tem_cell, M5tem_cell, M7tem_cell, coord, direction, Lx, Ly, use_reentrant=False)

    return Cset_cell, Tset_cell


def CTM_ite_cell_continuous_update(Cset_cell, Tset_cell, double_A_cell, direction, ctm_setting, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    cx_cy_matrix = numpy.array([[Lx, Ly], [Ly, Lx], [Lx, Ly], [Ly, Lx]], dtype=int)
    cx_max = cx_cy_matrix[direction - 1, 0]
    cy_max = cx_cy_matrix[direction - 1, 1]

    for cx in range(1, cx_max + 1):
        Cset_cell, Tset_cell = checkpoint(
            ctm_update_single_cx, cx, cy_max, Cset_cell, Tset_cell, double_A_cell, direction, ctm_setting, global_args, use_reentrant=False
        )
    return Cset_cell, Tset_cell


def Bosonic_CTMRG_cell_iPEPS(A_set, init, CTM0, ctm_setting, global_args):
    chi = ctm_setting.chi
    CTM_ite_info = ctm_setting.CTM_ite_info
    projector_strategy = ctm_setting.projector_strategy
    CTM_trun_svd = ctm_setting.CTM_trun_svd
    CTM_ite_nums = ctm_setting.CTM_ite_nums
    Lx, Ly = _get_lattice_size(global_args)
    device = _get_device(global_args)

    if (CTM_trun_svd is True) and (projector_strategy == "4x4"):
        print("Attention: truncated svd with 4x4 projector could give large error")

    if init.reconstruct_AA:
        double_A_cell = build_doublelayer_iPEPS_cell(A_set, global_args)
        if ctm_setting.doublelayer_on_cpu:
            for cx in range(1, Lx + 1):
                for cy in range(1, Ly + 1):
                    key = _cell_key(cx, cy)
                    double_A_cell[key] = double_A_cell[key].to("cpu")
    else:
        double_A_cell = None

    if init.reconstruct_CTM:
        CTM_cell = init_CTM_cell(A_set, ctm_setting, global_args)
    else:
        CTM_cell = CTM_copy(CTM0, global_args)

    Cset_cell = CTM_cell["Cset"]
    Tset_cell = CTM_cell["Tset"]

    if ctm_setting.trivial_initial_CTM:
        Cset_cell = Cset_detach(Cset_cell, global_args)
        Tset_cell = Tset_detach(Tset_cell, global_args)
        with torch.no_grad():
            Cset_cell_trivial, Tset_cell_trivial = initial_trivial_ctm(Cset_cell, Tset_cell, ctm_setting, global_args)
        Cset_cell = Cset_requires_grad_(Cset_cell_trivial, global_args)
        Tset_cell = Tset_requires_grad_(Tset_cell_trivial, global_args)

    ss_old1_cell = torch.ones((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_old2_cell = torch.ones((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_old3_cell = torch.ones((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_old4_cell = torch.ones((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_new1_cell = torch.zeros((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_new2_cell = torch.zeros((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_new3_cell = torch.zeros((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    ss_new4_cell = torch.zeros((Lx, Ly, chi * 2), dtype=torch.float64, device=device)
    er1_cell = torch.zeros((Lx, Ly))
    er2_cell = torch.zeros((Lx, Ly))
    er3_cell = torch.zeros((Lx, Ly))
    er4_cell = torch.ones((Lx, Ly))

    if CTM_ite_info:
        print("start CTM iterations:")

    ite_num = 0
    ite_err = 1
    err_set = (1,)
    for ci in range(1, CTM_ite_nums + 1):
        ite_num = ci
        direction_order = [3, 4, 1, 2]
        for direction in direction_order:
            Cset_cell, Tset_cell = checkpoint(
                CTM_ite_cell_continuous_update,
                Cset_cell,
                Tset_cell,
                double_A_cell,
                direction,
                ctm_setting,
                global_args,
                use_reentrant=False,
            )

        with torch.no_grad():
            for cx in range(1, Lx + 1):
                for cy in range(1, Ly + 1):
                    key = _cell_key(cx, cy)
                    er1, ss_new1 = spectrum_conv_check(ss_old1_cell[cx - 1, cy - 1, :], Cset_cell[key]["C1"])
                    er2, ss_new2 = spectrum_conv_check(ss_old2_cell[cx - 1, cy - 1, :], Cset_cell[key]["C2"])
                    er3, ss_new3 = spectrum_conv_check(ss_old3_cell[cx - 1, cy - 1, :], Cset_cell[key]["C3"])
                    er4, ss_new4 = spectrum_conv_check(ss_old4_cell[cx - 1, cy - 1, :], Cset_cell[key]["C4"])

                    er1_cell[cx - 1, cy - 1] = er1
                    er2_cell[cx - 1, cy - 1] = er2
                    er3_cell[cx - 1, cy - 1] = er3
                    er4_cell[cx - 1, cy - 1] = er4

                    ss_new1_cell[cx - 1, cy - 1, :] *= 0
                    ss_new2_cell[cx - 1, cy - 1, :] *= 0
                    ss_new3_cell[cx - 1, cy - 1, :] *= 0
                    ss_new4_cell[cx - 1, cy - 1, :] *= 0
                    ss_new1_cell[cx - 1, cy - 1, range(0, len(ss_new1))] = ss_new1
                    ss_new2_cell[cx - 1, cy - 1, range(0, len(ss_new2))] = ss_new2
                    ss_new3_cell[cx - 1, cy - 1, range(0, len(ss_new3))] = ss_new3
                    ss_new4_cell[cx - 1, cy - 1, range(0, len(ss_new4))] = ss_new4

            er = max((torch.max(er1_cell.flatten()), torch.max(er2_cell.flatten()), torch.max(er3_cell.flatten()), torch.max(er4_cell.flatten())))
            er = er.item()
            err_set = err_set + (er,)
            ite_err = er
            if CTM_ite_info:
                print("CTMRG iteration: " + str(ci) + ", CTMRG err: " + str(er))
            if er < ctm_setting.CTM_conv_tol:
                break

            ss_old1_cell = ss_new1_cell
            ss_old2_cell = ss_new2_cell
            ss_old3_cell = ss_new3_cell
            ss_old4_cell = ss_new4_cell

    CTM_cell = OrderedDict()
    CTM_cell["Cset"] = Cset_cell
    CTM_cell["Tset"] = Tset_cell
    return CTM_cell, double_A_cell, ite_num, ite_err
