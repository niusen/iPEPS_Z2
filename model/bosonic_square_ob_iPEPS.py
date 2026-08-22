import itertools
import math

import torch
import yastn

from ansatz.square_iPEPS import build_double_layer, dense_config
from config.settings import mod1


def _cell_key(cx, cy):
    return str(cx) + "," + str(cy)


def _get_lattice_size(global_args):
    if isinstance(global_args, dict):
        return global_args["Lx"], global_args["Ly"]
    return global_args.Lx, global_args.Ly


def _to_number(T):
    return T.to_number() if hasattr(T, "to_number") else T


def dense_matrix_to_yastn(mat, config_kwargs):
    config_dense = dense_config(config_kwargs)
    dtype = torch.complex128 if config_kwargs.get("default_dtype", "complex128") == "complex128" else torch.complex64
    mat = torch.as_tensor(mat, dtype=dtype, device=config_kwargs.get("default_device", "cpu"))
    leg_out = yastn.Leg(config_dense, s=1, D=(mat.shape[0],))
    leg_in = yastn.Leg(config_dense, s=-1, D=(mat.shape[1],))
    op = yastn.zeros(config=config_dense, legs=[leg_out, leg_in])
    op.set_block(ts=(), val=mat)
    return op


def spin_operators_dense(config_kwargs):
    device = config_kwargs.get("default_device", "cpu")
    dtype = torch.complex128 if config_kwargs.get("default_dtype", "complex128") == "complex128" else torch.complex64
    Id, sx, sy, sz = spin_matrices(device=device, dtype=dtype)
    return (
        dense_matrix_to_yastn(Id, config_kwargs),
        dense_matrix_to_yastn(sx, config_kwargs),
        dense_matrix_to_yastn(sy, config_kwargs),
        dense_matrix_to_yastn(sz, config_kwargs),
    )


def spin_matrices(device="cpu", dtype=torch.complex128):
    Id = torch.eye(2, dtype=dtype, device=device)
    sx = 0.5 * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    sy = 0.5 * torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    sz = 0.5 * torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    return Id, sx, sy, sz


def pauli_matrix(name, device="cpu", dtype=torch.complex128):
    name = str(name).lower()
    if name in ("sigmax", "sigma_x", "x", "sx"):
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    if name in ("sigmay", "sigma_y", "y", "sy"):
        return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    raise ValueError("unknown checkerboard Pauli transform: " + str(name))


def checkerboard_transform_name(config_kwargs):
    transform = config_kwargs.get("checkerboard_spin_transform", None)
    if transform is None or str(transform).lower() in ("none", "false", "0"):
        return None
    return transform


def transform_operator_for_site(O, abs_pos, config_kwargs=None):
    if config_kwargs is None:
        config_kwargs = {}
    transform = checkerboard_transform_name(config_kwargs)
    if transform is None or (abs_pos[0] + abs_pos[1]) % 2 == 0:
        return O
    dtype = torch.complex128 if config_kwargs.get("default_dtype", "complex128") == "complex128" else torch.complex64
    device = config_kwargs.get("default_device", "cpu")
    U = pauli_matrix(transform, device=device, dtype=dtype)
    O_dense = O.to_dense()
    O_transformed = U.conj().T @ O_dense @ U
    return dense_matrix_to_yastn(O_transformed, config_kwargs)


OPERATOR_SVD_TOL = 1.0e-10
RING_EXCHANGE_CHIRALITY_FACTOR = 2.0


def ring_exchange_to_chirality_coupling(lambda_ring, chirality_sign=1.0):
    """Convert i*lambda*(P_1234 - P_4321) to the scalar-chirality coupling.

    The plaquette orientation matches the four triangle terms below:
    LD->RD->RU->LU. For spin-1/2,
    i(P_1234 - P_4321) = 2(chi_123 + chi_234 + chi_341 + chi_412).
    """
    return chirality_sign * RING_EXCHANGE_CHIRALITY_FACTOR * lambda_ring


def prl_129_177201_square_csl_parameters(chirality_sign=1.0):
    theta = 0.06 * math.pi
    phi = 0.14 * math.pi
    J1 = 2.0 * math.cos(theta) * math.cos(phi)
    J2 = 2.0 * math.cos(theta) * math.sin(phi)
    lambda_ring = 2.0 * math.sin(theta)
    K = ring_exchange_to_chirality_coupling(lambda_ring, chirality_sign=chirality_sign)
    return {
        "J": J1,
        "J2a": J2,
        "J2b": J2,
        "K_no_LU": K,
        "K_no_LD": K,
        "K_no_RD": K,
        "K_no_RU": K,
    }


def chirality_svd_operators_dense(config_kwargs):
    device = config_kwargs.get("default_device", "cpu")
    dtype = torch.complex128 if config_kwargs.get("default_dtype", "complex128") == "complex128" else torch.complex64
    _, sx, sy, sz = spin_matrices(device=device, dtype=dtype)
    spins = (sx, sy, sz)

    H = 0
    for a, b, c in itertools.product(range(3), repeat=3):
        eps = levi_civita(a, b, c)
        if eps != 0:
            H = H + eps * torch.einsum("ad,be,cf->abcdef", spins[a], spins[b], spins[c])

    left = H.permute(0, 3, 1, 2, 4, 5).reshape(4, 16)
    u1, s1, vh1 = torch.linalg.svd(left, full_matrices=False)
    keep1 = s1 > OPERATOR_SVD_TOL
    u1 = u1[:, keep1].reshape(2, 2, -1)
    s1 = s1[keep1]
    vh1 = vh1[keep1, :]

    s2s3 = (s1[:, None] * vh1).reshape(len(s1), 2, 2, 2, 2)
    middle = s2s3.permute(0, 1, 3, 2, 4).reshape(len(s1) * 4, 4)
    u2, s2, vh2 = torch.linalg.svd(middle, full_matrices=False)
    keep2 = s2 > OPERATOR_SVD_TOL
    u2 = u2[:, keep2].reshape(len(s1), 2, 2, -1)
    s3 = (s2[keep2, None] * vh2[keep2, :]).reshape(sum(keep2).item(), 2, 2)

    terms = []
    for bond12 in range(u1.shape[2]):
        for bond23 in range(u2.shape[3]):
            op1 = dense_matrix_to_yastn(u1[:, :, bond12], config_kwargs)
            op2 = dense_matrix_to_yastn(u2[bond12, :, :, bond23], config_kwargs)
            op3 = dense_matrix_to_yastn(s3[bond23, :, :], config_kwargs)
            terms.append((op1, op2, op3))
    return terms


def ss_svd_operators_dense(config_kwargs):
    device = config_kwargs.get("default_device", "cpu")
    dtype = torch.complex128 if config_kwargs.get("default_dtype", "complex128") == "complex128" else torch.complex64
    _, sx, sy, sz = spin_matrices(device=device, dtype=dtype)
    H = torch.einsum("ac,bd->abcd", sx, sx)
    H = H + torch.einsum("ac,bd->abcd", sy, sy)
    H = H + torch.einsum("ac,bd->abcd", sz, sz)

    mat = H.permute(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    keep = s > OPERATOR_SVD_TOL
    u = u[:, keep].reshape(2, 2, -1)
    sv = (s[keep, None] * vh[keep, :]).reshape(sum(keep).item(), 2, 2)

    terms = []
    for bond in range(u.shape[2]):
        op1 = dense_matrix_to_yastn(u[:, :, bond], config_kwargs)
        op2 = dense_matrix_to_yastn(sv[bond, :, :], config_kwargs)
        terms.append((op1, op2))
    return terms


def build_double_layer_with_operator(Ap, A, O):
    A_op = yastn.ncon([A, O], [[-1, -2, -3, -4, 1], [-5, 1]])
    return build_double_layer(Ap, A_op)


def canonical_AA(AA):
    return AA.switch_signature(axes=(2, 3))


def get_AA_simple(double_A_cell, pos):
    return canonical_AA(double_A_cell[_cell_key(pos[0], pos[1])])


def get_AA_with_operator(A_set, pos, O):
    A = A_set[_cell_key(pos[0], pos[1])]
    return canonical_AA(build_double_layer_with_operator(A.conj(), A, O))


def build_MM_LU(Cset, Tset, AA_LU, cx, cy, Lx, Ly):
    return yastn.ncon(
        [
            Cset[_cell_key(mod1(cx, Lx), mod1(cy, Ly))]["C1"],
            Tset[_cell_key(mod1(cx + 1, Lx), mod1(cy, Ly))]["T1"],
            Tset[_cell_key(mod1(cx, Lx), mod1(cy + 1, Ly))]["T4"],
            AA_LU,
        ],
        [[1, 2], [2, 3, -3], [-1, 4, 1], [4, -2, -4, 3]],
    )


def build_MM_RU(Cset, Tset, AA_RU, cx, cy, Lx, Ly):
    return yastn.ncon(
        [
            Tset[_cell_key(mod1(cx + 2, Lx), mod1(cy, Ly))]["T1"],
            Cset[_cell_key(mod1(cx + 3, Lx), mod1(cy, Ly))]["C2"],
            AA_RU,
            Tset[_cell_key(mod1(cx + 3, Lx), mod1(cy + 1, Ly))]["T2"],
        ],
        [[-1, 3, 1], [1, 2], [-2, -4, 4, 3], [2, 4, -3]],
    )


def build_MM_LD(Cset, Tset, AA_LD, cx, cy, Lx, Ly):
    return yastn.ncon(
        [
            Tset[_cell_key(mod1(cx, Lx), mod1(cy + 2, Ly))]["T4"],
            AA_LD,
            Cset[_cell_key(mod1(cx, Lx), mod1(cy + 3, Ly))]["C4"],
            Tset[_cell_key(mod1(cx + 1, Lx), mod1(cy + 3, Ly))]["T3"],
        ],
        [[1, 3, -2], [3, 4, -5, -3], [2, 1], [-4, 4, 2]],
    )


def build_MM_RD(Cset, Tset, AA_RD, cx, cy, Lx, Ly):
    MM_RD = yastn.ncon(
        [
            Tset[_cell_key(mod1(cx + 3, Lx), mod1(cy + 2, Ly))]["T2"],
            Tset[_cell_key(mod1(cx + 2, Lx), mod1(cy + 3, Ly))]["T3"],
            Cset[_cell_key(mod1(cx + 3, Lx), mod1(cy + 3, Ly))]["C3"],
        ],
        [[-4, -3, 2], [1, -2, -1], [2, 1]],
    )
    return yastn.ncon([MM_RD, AA_RD], [[-1, 1, 2, -3], [-2, 1, 2, -4]])


def ob_2x2_iPEPS(CTM, AA_LU, AA_RU, AA_LD, AA_RD, cx, cy, Lx, Ly):
    Cset = CTM["Cset"]
    Tset = CTM["Tset"]
    MM_LU = build_MM_LU(Cset, Tset, AA_LU, cx, cy, Lx, Ly)
    MM_RU = build_MM_RU(Cset, Tset, AA_RU, cx, cy, Lx, Ly)
    MM_LD = build_MM_LD(Cset, Tset, AA_LD, cx, cy, Lx, Ly)
    MM_RD = build_MM_RD(Cset, Tset, AA_RD, cx, cy, Lx, Ly)
    return yastn.ncon([MM_LU, MM_RU, MM_LD, MM_RD], [[5, 6, 1, 2], [1, 2, 7, 8], [5, 6, 3, 4], [3, 4, 7, 8]])


def plaquette_positions(cx, cy, Lx, Ly):
    return {
        "LU": [mod1(cx + 1, Lx), mod1(cy + 1, Ly)],
        "RU": [mod1(cx + 2, Lx), mod1(cy + 1, Ly)],
        "LD": [mod1(cx + 1, Lx), mod1(cy + 2, Ly)],
        "RD": [mod1(cx + 2, Lx), mod1(cy + 2, Ly)],
    }


def plaquette_absolute_positions(cx, cy):
    return {
        "LU": [cx + 1, cy + 1],
        "RU": [cx + 2, cy + 1],
        "LD": [cx + 1, cy + 2],
        "RD": [cx + 2, cy + 2],
    }


def plaquette_AA(double_A_cell, positions):
    return {name: get_AA_simple(double_A_cell, pos) for name, pos in positions.items()}


def plaquette_norm(CTM, double_A_cell, cx, cy, Lx, Ly):
    pos = plaquette_positions(cx, cy, Lx, Ly)
    AA = plaquette_AA(double_A_cell, pos)
    return ob_2x2_iPEPS(CTM, AA["LU"], AA["RU"], AA["LD"], AA["RD"], cx, cy, Lx, Ly)


def ob_factorized_2x2_raw(CTM, A_set, double_A_cell, ops_by_corner, cx, cy, Lx, Ly, config_kwargs=None):
    if config_kwargs is None:
        config_kwargs = {}
    pos = plaquette_positions(cx, cy, Lx, Ly)
    abs_pos = plaquette_absolute_positions(cx, cy)
    AA = plaquette_AA(double_A_cell, pos)
    for corner, op in ops_by_corner.items():
        op_site = transform_operator_for_site(op, abs_pos[corner], config_kwargs)
        AA[corner] = get_AA_with_operator(A_set, pos[corner], op_site)
    return ob_2x2_iPEPS(CTM, AA["LU"], AA["RU"], AA["LD"], AA["RD"], cx, cy, Lx, Ly)


def ob_factorized_2x2(CTM, A_set, double_A_cell, ops_by_corner, cx, cy, Lx, Ly, config_kwargs=None):
    if config_kwargs is None:
        config_kwargs = {}
    ob = ob_factorized_2x2_raw(CTM, A_set, double_A_cell, ops_by_corner, cx, cy, Lx, Ly, config_kwargs)
    norm = plaquette_norm(CTM, double_A_cell, cx, cy, Lx, Ly)
    return _to_number(ob) / _to_number(norm)


def ob_onsite_iPEPS(CTM, O, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None, corner="LU"):
    return ob_factorized_2x2(CTM, A_set, double_A_cell, {corner: O}, cx, cy, Lx, Ly, config_kwargs)


def nearest_neighbor_x_iPEPS(CTM, O1, O2, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None):
    return ob_factorized_2x2(CTM, A_set, double_A_cell, {"LU": O1, "RU": O2}, cx, cy, Lx, Ly, config_kwargs)


def nearest_neighbor_y_iPEPS(CTM, O1, O2, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None):
    return ob_factorized_2x2(CTM, A_set, double_A_cell, {"RU": O1, "RD": O2}, cx, cy, Lx, Ly, config_kwargs)


def next_nearest_neighbor_diag_a_iPEPS(CTM, O1, O2, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None):
    return ob_factorized_2x2(CTM, A_set, double_A_cell, {"LU": O1, "RD": O2}, cx, cy, Lx, Ly, config_kwargs)


def next_nearest_neighbor_diag_b_iPEPS(CTM, O1, O2, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None):
    return ob_factorized_2x2(CTM, A_set, double_A_cell, {"LD": O1, "RU": O2}, cx, cy, Lx, Ly, config_kwargs)


def ob_two_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, corners, svd_terms, config_kwargs=None):
    if config_kwargs is None:
        config_kwargs = {}
    ob = None
    for op1, op2 in svd_terms:
        term = ob_factorized_2x2_raw(CTM, A_set, double_A_cell, {corners[0]: op1, corners[1]: op2}, cx, cy, Lx, Ly, config_kwargs)
        ob = term if ob is None else ob + term
    norm = plaquette_norm(CTM, double_A_cell, cx, cy, Lx, Ly)
    return _to_number(ob) / _to_number(norm)


def ob_three_site_iPEPS(CTM, O1, O2, O3, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs=None, corners=("LD", "RD", "RU")):
    return ob_factorized_2x2(
        CTM,
        A_set,
        double_A_cell,
        {corners[0]: O1, corners[1]: O2, corners[2]: O3},
        cx,
        cy,
        Lx,
        Ly,
        config_kwargs,
    )


def ob_three_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, corners, svd_terms, config_kwargs=None):
    if config_kwargs is None:
        config_kwargs = {}
    ob = None
    for op1, op2, op3 in svd_terms:
        term = ob_factorized_2x2_raw(
            CTM,
            A_set,
            double_A_cell,
            {corners[0]: op1, corners[1]: op2, corners[2]: op3},
            cx,
            cy,
            Lx,
            Ly,
            config_kwargs,
        )
        ob = term if ob is None else ob + term
    norm = plaquette_norm(CTM, double_A_cell, cx, cy, Lx, Ly)
    return _to_number(ob) / _to_number(norm)


def levi_civita(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    perm = [a, b, c]
    inversions = sum(1 for i, j in itertools.combinations(range(3), 2) if perm[i] > perm[j])
    return -1 if inversions % 2 else 1


def ss_nearest_neighbor_x_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=None):
    if svd_terms is None:
        svd_terms = ss_svd_operators_dense(config_kwargs)
    return ob_two_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, ("LU", "RU"), svd_terms, config_kwargs)


def ss_nearest_neighbor_y_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=None):
    if svd_terms is None:
        svd_terms = ss_svd_operators_dense(config_kwargs)
    return ob_two_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, ("RU", "RD"), svd_terms, config_kwargs)


def ss_next_nearest_neighbor_diag_a_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=None):
    if svd_terms is None:
        svd_terms = ss_svd_operators_dense(config_kwargs)
    return ob_two_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, ("LU", "RD"), svd_terms, config_kwargs)


def ss_next_nearest_neighbor_diag_b_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=None):
    if svd_terms is None:
        svd_terms = ss_svd_operators_dense(config_kwargs)
    return ob_two_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, ("LD", "RU"), svd_terms, config_kwargs)


def chirality_triangle_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, corners=("LD", "RD", "RU"), svd_terms=None):
    if svd_terms is None:
        svd_terms = chirality_svd_operators_dense(config_kwargs)
    return ob_three_site_terms_iPEPS(CTM, A_set, double_A_cell, cx, cy, Lx, Ly, corners, svd_terms, config_kwargs)


def evaluate_spin_cell_iPEPS(A_set, double_A_cell, CTM_cell, config_kwargs, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    _, sx, sy, sz = spin_operators_dense(config_kwargs)
    device = config_kwargs.get("default_device", "cpu")
    sx_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    sy_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    sz_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            sx_set[cx - 1, cy - 1] = ob_onsite_iPEPS(CTM_cell, sx, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs)
            sy_set[cx - 1, cy - 1] = ob_onsite_iPEPS(CTM_cell, sy, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs)
            sz_set[cx - 1, cy - 1] = ob_onsite_iPEPS(CTM_cell, sz, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs)
    return sx_set, sy_set, sz_set


def evaluate_spin_ob_cell_iPEPS(A_set, double_A_cell, CTM_cell, config_kwargs, global_args):
    Lx, Ly = _get_lattice_size(global_args)
    device = config_kwargs.get("default_device", "cpu")
    SS_x_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    SS_y_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    SS_diag_a_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    SS_diag_b_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    triangle_no_LU_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    triangle_no_LD_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    triangle_no_RD_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    triangle_no_RU_set = torch.zeros(Lx, Ly, dtype=torch.complex128, device=device)
    ss_terms = ss_svd_operators_dense(config_kwargs)
    chirality_terms = chirality_svd_operators_dense(config_kwargs)
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            SS_x_set[cx - 1, cy - 1] = ss_nearest_neighbor_x_iPEPS(CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=ss_terms)
            SS_y_set[cx - 1, cy - 1] = ss_nearest_neighbor_y_iPEPS(CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=ss_terms)
            SS_diag_a_set[cx - 1, cy - 1] = ss_next_nearest_neighbor_diag_a_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=ss_terms
            )
            SS_diag_b_set[cx - 1, cy - 1] = ss_next_nearest_neighbor_diag_b_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, svd_terms=ss_terms
            )
            triangle_no_LU_set[cx - 1, cy - 1] = chirality_triangle_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, corners=("LD", "RD", "RU"), svd_terms=chirality_terms
            )
            triangle_no_LD_set[cx - 1, cy - 1] = chirality_triangle_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, corners=("RD", "RU", "LU"), svd_terms=chirality_terms
            )
            triangle_no_RD_set[cx - 1, cy - 1] = chirality_triangle_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, corners=("RU", "LU", "LD"), svd_terms=chirality_terms
            )
            triangle_no_RU_set[cx - 1, cy - 1] = chirality_triangle_iPEPS(
                CTM_cell, A_set, double_A_cell, cx, cy, Lx, Ly, config_kwargs, corners=("LU", "LD", "RD"), svd_terms=chirality_terms
            )
    return (
        triangle_no_LU_set,
        triangle_no_LD_set,
        triangle_no_RD_set,
        triangle_no_RU_set,
        SS_x_set,
        SS_y_set,
        SS_diag_a_set,
        SS_diag_b_set,
    )


def evaluate_ob_cell_iPEPS(parameters, A_set, double_A_cell, CTM_cell, config_kwargs, global_args):
    (
        triangle_no_LU_set,
        triangle_no_LD_set,
        triangle_no_RD_set,
        triangle_no_RU_set,
        SS_x_set,
        SS_y_set,
        SS_diag_a_set,
        SS_diag_b_set,
    ) = evaluate_spin_ob_cell_iPEPS(
        A_set, double_A_cell, CTM_cell, config_kwargs, global_args
    )
    Jx = parameters.get("Jx", parameters.get("J", 1.0))
    Jy = parameters.get("Jy", parameters.get("J", 1.0))
    J2a = parameters.get("J2a", parameters.get("J2", 0.0))
    J2b = parameters.get("J2b", parameters.get("J2", 0.0))
    K_no_LU = parameters.get("K_no_LU", parameters.get("K", 0.0))
    K_no_LD = parameters.get("K_no_LD", parameters.get("K", 0.0))
    K_no_RD = parameters.get("K_no_RD", parameters.get("K", 0.0))
    K_no_RU = parameters.get("K_no_RU", parameters.get("K", 0.0))

    energy_set = (
        Jx * SS_x_set
        + Jy * SS_y_set
        + J2a * SS_diag_a_set
        + J2b * SS_diag_b_set
        + K_no_LU * triangle_no_LU_set
        + K_no_LD * triangle_no_LD_set
        + K_no_RD * triangle_no_RD_set
        + K_no_RU * triangle_no_RU_set
    )
    E_total = torch.mean(energy_set)
    return (
        E_total,
        SS_x_set,
        SS_y_set,
        SS_diag_a_set,
        SS_diag_b_set,
        triangle_no_LU_set,
        triangle_no_LD_set,
        triangle_no_RD_set,
        triangle_no_RU_set,
    )
