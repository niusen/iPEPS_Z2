"""J1-Jchi observables for a dense spin-1/2 triangular iPESS."""

import torch
import yastn
from torch.utils.checkpoint import checkpoint

from ansatz.bosonic_triangle_iPESS import dense_config
from model.Spin_ob_iPESS import (
    hopping_diagonala_iPESS_no_sign,
    hopping_x_iPESS_no_sign,
    hopping_y_iPESS_no_sign,
    ob_dn_triangle_iPESS,
    ob_onsite_iPESS,
    ob_up_triangle_iPESS,
)


def _matrix_to_yastn(matrix, config_kwargs):
    config = dense_config(config_kwargs)
    leg_out = yastn.Leg(config, s=1, D=(matrix.shape[0],))
    leg_in = yastn.Leg(config, s=-1, D=(matrix.shape[1],))
    operator = yastn.zeros(config=config, legs=[leg_out, leg_in])
    operator.set_block(ts=(), val=matrix)
    return operator


def spin_half_components_dense(config_kwargs):
    """Return Sx, Sy, Sz for the dense two-state spin-1/2 space."""
    device = config_kwargs.get("default_device", "cpu")
    dtype_name = config_kwargs.get("default_dtype", "complex128")
    dtype = torch.complex64 if dtype_name == "complex64" else torch.complex128
    sx = 0.5 * torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = 0.5 * torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    Sx = _matrix_to_yastn(sx, config_kwargs)
    Sy = _matrix_to_yastn(sy, config_kwargs)
    Sz = _matrix_to_yastn(sz, config_kwargs)
    return Sx, Sy, Sz


def factorized_spin_operators_dense(Sx, Sy, Sz):
    """Factorize S.S and S.(S x S) for three pre-built spin components."""
    SS = (
        yastn.ncon([Sx, Sx], [[-1, -2], [-3, -4]])
        + yastn.ncon([Sy, Sy], [[-1, -2], [-3, -4]])
        + yastn.ncon([Sz, Sz], [[-1, -2], [-3, -4]])
    )
    u, s, v = yastn.linalg.svd_with_truncation(
        SS, axes=((0, 1), (2, 3)), svd_on_cpu=True, tol=1.0e-14
    )
    Sa = yastn.ncon([u, s], [[-2, -3, 1], [1, -1]])
    Sb = v
    SS_string = yastn.eye(config=SS.config, legs=Sb.get_legs(axes=0), isdiag=False)

    # epsilon_abc S1^a S2^b S3^c = S1 . (S2 x S3)
    xyz = yastn.ncon([Sx, Sy, Sz], [[-1, -4], [-2, -5], [-3, -6]])
    xzy = yastn.ncon([Sx, Sz, Sy], [[-1, -4], [-2, -5], [-3, -6]])
    yzx = yastn.ncon([Sy, Sz, Sx], [[-1, -4], [-2, -5], [-3, -6]])
    yxz = yastn.ncon([Sy, Sx, Sz], [[-1, -4], [-2, -5], [-3, -6]])
    zxy = yastn.ncon([Sz, Sx, Sy], [[-1, -4], [-2, -5], [-3, -6]])
    zyx = yastn.ncon([Sz, Sy, Sx], [[-1, -4], [-2, -5], [-3, -6]])
    chirality = xyz - xzy + yzx - yxz + zxy - zyx

    u, s, v = yastn.linalg.svd_with_truncation(
        chirality,
        axes=((0, 3), (1, 2, 4, 5)),
        svd_on_cpu=True,
        tol=1.0e-14,
    )
    chirality_S1 = u
    S2S3 = yastn.ncon([s, v], [[-1, 1], [1, -2, -3, -4, -5]])
    u, s, v = yastn.linalg.svd_with_truncation(
        S2S3, axes=((0, 1, 3), (2, 4)), svd_on_cpu=True, tol=1.0e-14
    )
    chirality_S2 = u
    chirality_S3 = yastn.ncon([s, v], [[-1, 1], [1, -2, -3]])
    string12 = yastn.eye(
        config=chirality.config, legs=chirality_S2.get_legs(axes=0), isdiag=False
    )
    string23 = yastn.eye(
        config=chirality.config, legs=chirality_S3.get_legs(axes=0), isdiag=False
    )
    return Sa, Sb, SS_string, chirality_S1, chirality_S2, chirality_S3, string12, string23


def spin_half_operators_dense(config_kwargs):
    """Return factorized S.S and S.(S x S) operators for the 2-state spin space."""
    return factorized_spin_operators_dense(*spin_half_components_dense(config_kwargs))


def evaluate_triangle_spin_energy(
    parameters,
    B_set,
    T_set,
    double_B_set,
    double_T_set,
    CTM_cell,
    config_kwargs,
    global_args,
    return_observables=False,
):
    """Evaluate energy per site for H=J1 sum_<ij> Si.Sj + Jchi sum_triangle chi."""
    J1 = parameters.get("J1", 1.0)
    Jchi = parameters.get("Jchi", parameters.get("J_chi", 0.0))
    operators = spin_half_operators_dense(config_kwargs)
    Sa, Sb, SS_string, S1, S2, S3, string12, string23 = operators
    Lx, Ly = global_args.Lx, global_args.Ly
    values = {"chi_up": [], "chi_down": [], "SS_x": [], "SS_y": [], "SS_diagonal": []}
    if return_observables:
        values.update({"Sx": [], "Sy": [], "Sz": []})
        Sx_operator, Sy_operator, Sz_operator = spin_half_components_dense(config_kwargs)
    energy = 0.0

    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            chi_up = checkpoint(
                ob_up_triangle_iPESS,
                CTM_cell, S1, S2, S3, string12, string23,
                B_set, T_set, double_B_set, double_T_set, cx, cy, Lx, Ly,
                use_reentrant=False,
            )
            # The down-triangle routine uses the opposite cyclic site order.
            chi_down = -checkpoint(
                ob_dn_triangle_iPESS,
                CTM_cell, S1, S2, S3, string12, string23,
                B_set, T_set, double_B_set, double_T_set, cx, cy, Lx, Ly,
                use_reentrant=False,
            )
            SS_x = checkpoint(
                hopping_x_iPESS_no_sign,
                CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                double_B_set, double_T_set, cx, cy, Lx, Ly,
                use_reentrant=False,
            )
            SS_y = checkpoint(
                hopping_y_iPESS_no_sign,
                CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                double_B_set, double_T_set, cx, cy, Lx, Ly,
                use_reentrant=False,
            )
            SS_diagonal = checkpoint(
                hopping_diagonala_iPESS_no_sign,
                CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                double_B_set, double_T_set, cx, cy, Lx, Ly,
                use_reentrant=False,
            )
            energy = energy + J1 * (SS_x + SS_y + SS_diagonal)
            energy = energy + Jchi * (chi_up + chi_down)
            if return_observables:
                with torch.no_grad():
                    Sx_value = ob_onsite_iPESS(
                        CTM_cell, Sx_operator, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly
                    )
                    Sy_value = ob_onsite_iPESS(
                        CTM_cell, Sy_operator, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly
                    )
                    Sz_value = ob_onsite_iPESS(
                        CTM_cell, Sz_operator, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly
                    )
                for name, value in (
                    ("chi_up", chi_up), ("chi_down", chi_down),
                    ("SS_x", SS_x), ("SS_y", SS_y),
                    ("SS_diagonal", SS_diagonal),
                    ("Sx", Sx_value), ("Sy", Sy_value), ("Sz", Sz_value),
                ):
                    values[name].append(value)

    energy = torch.real(energy / (Lx * Ly))
    if not return_observables:
        return energy
    return energy, {
        name: torch.stack(tuple(value_list)).reshape(Lx, Ly)
        for name, value_list in values.items()
    }
