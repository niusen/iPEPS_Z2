import numpy
import torch
import yastn
from collections import OrderedDict
from scipy.io import savemat

from config.settings import mod1
from model.fermion_ob_iPESS import fill_Z2_Ham, hopping_spin_resolved_Z2, ob_onsite_iPESS
from model.iPESS_correl_cell import (
    Cell_take_cy,
    build_AA_hop,
    evaluate_correl,
    solve_correl_length_simple,
)


def _mat_value(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _single_channel_operator(op, source_channel, config_kwargs):
    config_Z2 = yastn.make_config(sym="Z2", fermionic=True, **config_kwargs)
    dense = op.to_dense()[source_channel : source_channel + 1, :, :]
    legs = op.get_legs()
    dummy = yastn.Leg(config_Z2, s=legs[0].s, t=[1], D=[1])
    out = yastn.zeros(config=config_Z2, legs=[dummy, legs[1], legs[2]])
    return fill_Z2_Ham(dense, out)


def _single_channel_string(C, config_kwargs):
    config_Z2 = yastn.make_config(sym="Z2", fermionic=True, **config_kwargs)
    dummy = yastn.Leg(config_Z2, s=C.get_legs(axes=0).s, t=[1], D=[1])
    return yastn.eye(config=config_Z2, legs=dummy, isdiag=False)


def _single_channel_cross_hopping_ops(Cdagup, Cup, Cdagdn, Cdn, config_kwargs):
    Cdagup_1 = _single_channel_operator(Cdagup, 0, config_kwargs)
    Cup_1 = _single_channel_operator(Cup, 0, config_kwargs)
    Cdagdn_1 = _single_channel_operator(Cdagdn, 1, config_kwargs)
    Cdn_1 = _single_channel_operator(Cdn, 1, config_kwargs)
    return Cdagup_1, Cup_1, Cdagdn_1, Cdn_1, _single_channel_string(Cup_1, config_kwargs)


def _spin_resolved_density_ops(config_kwargs):
    config_Z2 = yastn.make_config(sym="Z2", fermionic=True, **config_kwargs)
    device = config_kwargs["default_device"]
    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))

    Id = torch.tensor([[1.0, 0], [0, 1.0]], device=device)
    sm = torch.tensor([[0, 1.0], [0, 0]], device=device)
    sp = torch.tensor([[0, 0], [1.0, 0]], device=device)
    occu = torch.matmul(sp, sm)

    order = (1 - 1, 4 - 1, 3 - 1, 2 - 1)
    n_up = torch.kron(occu, Id)
    n_dn = torch.kron(Id, occu)
    n_up = n_up[order, :]
    n_up = n_up[:, order]
    n_dn = n_dn[order, :]
    n_dn = n_dn[:, order]

    n_up_yast = yastn.zeros(config=config_Z2, legs=[Vp, Vp_conj])
    n_dn_yast = yastn.zeros(config=config_Z2, legs=[Vp, Vp_conj])
    return fill_Z2_Ham(n_up, n_up_yast), fill_Z2_Ham(n_dn, n_dn_yast)


def _onsite_cross_hopping_ops(config_kwargs):
    config_Z2 = yastn.make_config(sym="Z2", fermionic=True, **config_kwargs)
    device = config_kwargs["default_device"]
    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))

    Id = torch.tensor([[1.0, 0], [0, 1.0]], device=device)
    sm = torch.tensor([[0, 1.0], [0, 0]], device=device)
    sp = torch.tensor([[0, 0], [1.0, 0]], device=device)
    order = (1 - 1, 4 - 1, 3 - 1, 2 - 1)

    cdagup_cdn = torch.kron(sp, sm)
    cdagdn_cup = torch.kron(sm, sp)
    cdagup_cdn = cdagup_cdn[order, :]
    cdagup_cdn = cdagup_cdn[:, order]
    cdagdn_cup = cdagdn_cup[order, :]
    cdagdn_cup = cdagdn_cup[:, order]

    cdagup_cdn_yast = yastn.zeros(config=config_Z2, legs=[Vp, Vp_conj])
    cdagdn_cup_yast = yastn.zeros(config=config_Z2, legs=[Vp, Vp_conj])
    return fill_Z2_Ham(cdagup_cdn, cdagup_cdn_yast), fill_Z2_Ham(cdagdn_cup, cdagdn_cup_yast)


def _onsite_density_product_op(config_kwargs):
    n_up, n_dn = _spin_resolved_density_ops(config_kwargs)
    dense = n_up.to_dense() @ n_dn.to_dense()
    return fill_Z2_Ham(dense, n_up * 0)


def _build_AA_density(O, B_set, T_set, ca, cb):
    B0 = B_set[str(ca) + "," + str(cb)]
    T0 = T_set[str(ca) + "," + str(cb)]
    T_new = yastn.ncon([T0, O], [[-1, 1, -3, -4], [-2, 1]])
    B_double = build_AA_density_B(B0)
    T_double = build_AA_density_T(T0, T_new)
    return B_double, T_double


def build_AA_density_B(B0):
    from ctmrg.Fermionic_CTMRG_unitcell_iPESS import build_double_layer_swap_Tm

    return build_double_layer_swap_Tm(B0.conj(), B0, False)


def build_AA_density_T(T0, T_new):
    from ctmrg.Fermionic_CTMRG_unitcell_iPESS import build_double_layer_swap_Bm

    return build_double_layer_swap_Bm(T0.conj(), T_new, True)


def cal_onsite_cross_copy(
    CTM_cell,
    B_set,
    T_set,
    B_double_set,
    T_double_set,
    D,
    chi,
    config_kwargs,
    global_args,
    cx=1,
    cy=1,
    mat_prefix=None,
):
    Lx = global_args.Lx
    Ly = global_args.Ly
    Cdagup_Cdn, Cdagdn_Cup = _onsite_cross_hopping_ops(config_kwargs)
    NupNdn = _onsite_density_product_op(config_kwargs)

    CdagC_up_dn_onsite = ob_onsite_iPESS(
        CTM_cell, Cdagup_Cdn, B_set, T_set, B_double_set, T_double_set, cx, cy, Lx, Ly
    )
    CdagC_dn_up_onsite = ob_onsite_iPESS(
        CTM_cell, Cdagdn_Cup, B_set, T_set, B_double_set, T_double_set, cx, cy, Lx, Ly
    )
    NupNdn_onsite = ob_onsite_iPESS(
        CTM_cell, NupNdn, B_set, T_set, B_double_set, T_double_set, cx, cy, Lx, Ly
    )

    mat_filenm = mat_prefix or ("onsite_spinHall_cross_D" + str(D) + "_chi" + str(chi))
    datadic = {
        "Lx": Lx,
        "Ly": Ly,
        "cx": cx,
        "cy": cy,
        "CdagC_up_dn_onsite": _mat_value(CdagC_up_dn_onsite),
        "CdagC_dn_up_onsite": _mat_value(CdagC_dn_up_onsite),
        "NupNdn_onsite": _mat_value(NupNdn_onsite),
        "NdnNup_onsite": _mat_value(NupNdn_onsite),
    }
    savemat(mat_filenm + ".mat", datadic)
    return CdagC_up_dn_onsite, CdagC_dn_up_onsite, NupNdn_onsite


def _cal_hopping_cross(
    x_range,
    y_range,
    distance,
    direction,
    Lx,
    Ly,
    CTM_cell,
    config_kwargs,
    partly,
    B_set,
    T_set,
    B_double_set,
    T_double_set,
    Cdag,
    C,
    CdagC_string,
    global_args,
):
    CdagC_ob_set = numpy.zeros((len(x_range), len(y_range), distance), dtype=numpy.complex128)
    if direction != "x":
        raise NotImplementedError("cross-copy hopping correlation is currently implemented only for x direction")

    for cb in y_range:
        double_B_CdagC_L_set = OrderedDict()
        double_T_CdagC_L_set = OrderedDict()
        double_B_CdagC_R_set = OrderedDict()
        double_T_CdagC_R_set = OrderedDict()
        double_B_CdagC_mid_set = OrderedDict()
        double_T_CdagC_mid_set = OrderedDict()

        for ca in range(1, Lx + 1):
            (
                B_double_CdagC_L,
                T_double_CdagC_L,
                B_double_CdagC_mid,
                T_double_CdagC_mid,
                B_double_CdagC_R,
                T_double_CdagC_R,
            ) = build_AA_hop(Cdag, C, CdagC_string, B_set, T_set, ca, cb, Lx)
            double_B_CdagC_L_set.update({str(ca): B_double_CdagC_L})
            double_T_CdagC_L_set.update({str(ca): T_double_CdagC_L})
            double_B_CdagC_mid_set.update({str(mod1(ca + 1, Lx)): B_double_CdagC_mid})
            double_T_CdagC_mid_set.update({str(mod1(ca + 1, Lx)): T_double_CdagC_mid})
            double_B_CdagC_R_set.update({str(mod1(ca + 2, Lx)): B_double_CdagC_R})
            double_T_CdagC_R_set.update({str(mod1(ca + 2, Lx)): T_double_CdagC_R})

        for ca in x_range:
            norms = evaluate_correl(
                [ca, cb],
                1,
                "x",
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                CTM_cell,
                distance,
                global_args,
            )
            norm_coe = (norms[4 + Lx] / norms[4]) ** (1 / Lx)
            norms = evaluate_correl(
                [ca, cb],
                1 / norm_coe,
                "x",
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                Cell_take_cy(B_double_set, cb, Lx, Ly),
                Cell_take_cy(T_double_set, cb, Lx, Ly),
                CTM_cell,
                distance,
                global_args,
            )
            hopping_ob = evaluate_correl(
                [ca, cb],
                1 / norm_coe,
                "x",
                double_B_CdagC_mid_set,
                double_T_CdagC_mid_set,
                double_B_CdagC_L_set,
                double_T_CdagC_L_set,
                double_B_CdagC_R_set,
                double_T_CdagC_R_set,
                CTM_cell,
                distance,
                global_args,
            )

            CdagC_ob_set[ca - 1, cb - 1, :] = hopping_ob / norms

    return CdagC_ob_set


def _cal_density_cross(
    x_range,
    y_range,
    distance,
    direction,
    Lx,
    Ly,
    CTM_cell,
    B_set,
    T_set,
    B_double_set,
    T_double_set,
    O_left,
    O_right,
    global_args,
):
    density_ob_set = numpy.zeros((len(x_range), len(y_range), distance), dtype=numpy.complex128)
    if direction != "x":
        raise NotImplementedError("cross-copy density correlation is currently implemented only for x direction")

    for cb in y_range:
        double_B_left_set = OrderedDict()
        double_T_left_set = OrderedDict()
        double_B_right_set = OrderedDict()
        double_T_right_set = OrderedDict()

        for ca in range(1, Lx + 1):
            B_double_left, T_double_left = _build_AA_density(O_left, B_set, T_set, ca, cb)
            B_double_right, T_double_right = _build_AA_density(O_right, B_set, T_set, ca, cb)
            double_B_left_set.update({str(ca): B_double_left})
            double_T_left_set.update({str(ca): T_double_left})
            double_B_right_set.update({str(ca): B_double_right})
            double_T_right_set.update({str(ca): T_double_right})

        for ca in x_range:
            normal_B = Cell_take_cy(B_double_set, cb, Lx, Ly)
            normal_T = Cell_take_cy(T_double_set, cb, Lx, Ly)
            norms = evaluate_correl(
                [ca, cb],
                1,
                "x",
                normal_B,
                normal_T,
                normal_B,
                normal_T,
                normal_B,
                normal_T,
                CTM_cell,
                distance,
                global_args,
            )
            norm_coe = (norms[4 + Lx] / norms[4]) ** (1 / Lx)
            norms = evaluate_correl(
                [ca, cb],
                1 / norm_coe,
                "x",
                normal_B,
                normal_T,
                normal_B,
                normal_T,
                normal_B,
                normal_T,
                CTM_cell,
                distance,
                global_args,
            )
            density_ob = evaluate_correl(
                [ca, cb],
                1 / norm_coe,
                "x",
                normal_B,
                normal_T,
                double_B_left_set,
                double_T_left_set,
                double_B_right_set,
                double_T_right_set,
                CTM_cell,
                distance,
                global_args,
            )

            density_ob_set[ca - 1, cb - 1, :] = density_ob / norms

    return density_ob_set


def cal_correl_density_cross_copy(
    CTM_cell,
    B_set,
    T_set,
    B_double_set,
    T_double_set,
    D,
    chi,
    direction,
    distance,
    config_kwargs,
    global_args,
    partly,
    mat_prefix=None,
):
    Lx = global_args.Lx
    Ly = global_args.Ly
    n_up, n_dn = _spin_resolved_density_ops(config_kwargs)

    if partly:
        x_range = range(1, 2)
        y_range = range(1, 2)
    else:
        x_range = range(1, Lx + 1)
        y_range = range(1, Ly + 1)

    n_values = 10
    eu_x_cell, Q_set = solve_correl_length_simple(n_values, CTM_cell, direction, Lx, Ly, config_kwargs, partly)

    NupNdn_set = _cal_density_cross(
        x_range,
        y_range,
        distance,
        direction,
        Lx,
        Ly,
        CTM_cell,
        B_set,
        T_set,
        B_double_set,
        T_double_set,
        n_up,
        n_dn,
        global_args,
    )
    NdnNup_set = _cal_density_cross(
        x_range,
        y_range,
        distance,
        direction,
        Lx,
        Ly,
        CTM_cell,
        B_set,
        T_set,
        B_double_set,
        T_double_set,
        n_dn,
        n_up,
        global_args,
    )

    mat_filenm = mat_prefix or ("correl_spinHall_cross_density_D" + str(D) + "_chi" + str(chi))
    mat_filenm = mat_filenm + ("_part" if partly else "_full")
    datadic = {
        "Lx": Lx,
        "Ly": Ly,
        "NupNdn_set": NupNdn_set,
        "NdnNup_set": NdnNup_set,
        "eu_x_cell": eu_x_cell,
        "Q_set": Q_set,
    }
    savemat(mat_filenm + ".mat", datadic)
    return NupNdn_set, NdnNup_set


def cal_correl_spin_resolved_cross_copy(
    CTM_cell,
    B_set,
    T_set,
    B_double_set,
    T_double_set,
    D,
    chi,
    direction,
    distance,
    config_kwargs,
    global_args,
    partly,
    mat_prefix=None,
):
    Lx = global_args.Lx
    Ly = global_args.Ly
    Cdagup, Cup, CdagC_up_string, Cdagdn, Cdn, CdagC_dn_string = hopping_spin_resolved_Z2(config_kwargs)
    Cdagup_1, Cup_1, Cdagdn_1, Cdn_1, CdagC_string_1 = _single_channel_cross_hopping_ops(
        Cdagup, Cup, Cdagdn, Cdn, config_kwargs
    )

    if partly:
        x_range = range(1, 2)
        y_range = range(1, 2)
    else:
        x_range = range(1, Lx + 1)
        y_range = range(1, Ly + 1)

    n_values = 10
    eu_x_cell, Q_set = solve_correl_length_simple(n_values, CTM_cell, direction, Lx, Ly, config_kwargs, partly)

    # up_dn is <c_up^\dagger(0) c_down(r)>; dn_up is <c_down^\dagger(0) c_up(r)>.
    CdagC_up_dn_set = _cal_hopping_cross(
        x_range,
        y_range,
        distance,
        direction,
        Lx,
        Ly,
        CTM_cell,
        config_kwargs,
        partly,
        B_set,
        T_set,
        B_double_set,
        T_double_set,
        Cdagup_1,
        Cdn_1,
        CdagC_string_1,
        global_args,
    )
    CdagC_dn_up_set = _cal_hopping_cross(
        x_range,
        y_range,
        distance,
        direction,
        Lx,
        Ly,
        CTM_cell,
        config_kwargs,
        partly,
        B_set,
        T_set,
        B_double_set,
        T_double_set,
        Cdagdn_1,
        Cup_1,
        CdagC_string_1,
        global_args,
    )

    mat_filenm = mat_prefix or ("correl_spinHall_cross_D" + str(D) + "_chi" + str(chi))
    mat_filenm = mat_filenm + ("_part" if partly else "_full")
    datadic = {
        "Lx": Lx,
        "Ly": Ly,
        "CdagC_up_dn_set": CdagC_up_dn_set,
        "CdagC_dn_up_set": CdagC_dn_up_set,
        "eu_x_cell": eu_x_cell,
        "Q_set": Q_set,
    }
    savemat(mat_filenm + ".mat", datadic)
    return CdagC_up_dn_set, CdagC_dn_up_set
