"""Generic momentum-resolved EH with a streamed periodic chi trace.

This contraction is ported essentially verbatim from
``Juraj/tn-torch_dev@9e36bd9:examples/j1j2/generic_eh_momentum_stream.py``.
The state and environment adapters live in ``triangle_ipess_adapter.py`` so
that the algorithm and its index conventions remain isolated from iPESS_Z2.
"""

import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigs


def _translate_boundary_tensor(V, inverse=False):
    if inverse:
        return V.permute([V.dim() - 1] + list(range(V.dim() - 1))).contiguous()
    return V.permute(list(range(1, V.dim())) + [0]).contiguous()


def _boundary_momentum_phase(V):
    V_shifted = _translate_boundary_tensor(V)
    denom = torch.vdot(V.reshape(-1), V.reshape(-1))
    if torch.abs(denom) < 1.0e-30:
        return torch.zeros((), dtype=V.dtype, device=V.device)
    return torch.vdot(V_shifted.reshape(-1), V.reshape(-1)) / denom


def _boundary_momentum_phase_inverse(V):
    V_shifted = _translate_boundary_tensor(V, inverse=True)
    denom = torch.vdot(V.reshape(-1), V.reshape(-1))
    if torch.abs(denom) < 1.0e-30:
        return torch.zeros((), dtype=V.dtype, device=V.device)
    return torch.vdot(V_shifted.reshape(-1), V.reshape(-1)) / denom


def _eh_geometry(L, coord, direction, state, env):
    if direction != (1, 0):
        raise ValueError(
            "Only x-transfer EH is supported: direction must be (1, 0), and L is Ly."
        )

    chi = env.chi
    dir_to_ind = {(0, -1): 1, (-1, 0): 2, (0, 1): 3, (1, 0): 4}
    ind_to_dir = dict(zip(dir_to_ind.values(), dir_to_ind.keys()))
    d_grow = ind_to_dir[
        dir_to_ind[direction]
        - 1
        + ((4 - dir_to_ind[direction] + 1) // 4) * 4
    ]
    d_opp = (-direction[0], -direction[1])

    def site_path(start_coord):
        return [
            state.vertexToSite(
                (start_coord[0] + i * d_grow[0], start_coord[1] + i * d_grow[1])
            )
            for i in range(L)
        ]

    if state.vertexToSite(
        (coord[0] + L * d_grow[0], coord[1] + L * d_grow[1])
    ) != state.vertexToSite(coord):
        raise ValueError(
            f"EH boundary of length L={L} does not close on the unit cell from "
            f"coord={coord} with growth direction {d_grow}. Choose L compatible "
            "with the unit-cell period."
        )

    ads = [state.site(c).size(dir_to_ind[direction]) for c in site_path(coord)]
    return chi, dir_to_ind, d_opp, site_path, ads


def _get_and_transform_T(c, d, rows, chi, dir_to_ind, state, env):
    if rows not in ("ket", "bra"):
        raise ValueError(f"Invalid T unfuse row convention {rows}")

    def as_operator(t):
        # Generic CTMRG fused auxiliary legs are in (ket, bra) order.
        return t if rows == "ket" else t.permute(0, 1, 3, 2).contiguous()

    if d == (0, -1):
        return as_operator(
            env.T[(c, (0, -1))]
            .permute(0, 2, 1)
            .contiguous()
            .view(
                [chi] * 2
                + [state.site(c).size(dir_to_ind[(0, -1)])] * 2
            )
        )
    if d == (-1, 0):
        return as_operator(
            env.T[(c, (-1, 0))].view(
                [chi] * 2 + [state.site(c).size(dir_to_ind[(-1, 0)])] * 2
            )
        )
    if d == (0, 1):
        return as_operator(
            env.T[(c, (0, 1))]
            .permute(1, 2, 0)
            .contiguous()
            .view(
                [chi] * 2 + [state.site(c).size(dir_to_ind[(0, 1)])] * 2
            )
        )
    if d == (1, 0):
        return as_operator(
            env.T[(c, (1, 0))]
            .permute(0, 2, 1)
            .contiguous()
            .view(
                [chi] * 2 + [state.site(c).size(dir_to_ind[(1, 0)])] * 2
            )
        )
    raise ValueError(f"Invalid direction {d}")


def mv_sigma_dense(V, L, d_sigma, start_coord, rows, chi, dir_to_ind, site_path, state, env):
    path = site_path(start_coord)
    c = path[0]
    T = _get_and_transform_T(c, d_sigma, rows, chi, dir_to_ind, state, env)
    V = torch.tensordot(T, V, ([3], [0]))
    V = V.permute([1, 2] + list(range(3, L - 1 + 3)) + [0])

    for i in range(1, L - 1):
        c = path[i]
        T = _get_and_transform_T(c, d_sigma, rows, chi, dir_to_ind, state, env)
        V = torch.tensordot(T, V, ([0, 3], [0, i + 1]))

    c = path[-1]
    T = _get_and_transform_T(c, d_sigma, rows, chi, dir_to_ind, state, env)
    V = torch.tensordot(T, V, ([0, 3, 1], [0, L - 1 + 1, L - 1 + 2]))
    return V.permute(list(range(L - 1, -1, -1))).contiguous()


def mv_sigma_stream_chi(
    V, L, d_sigma, start_coord, rows, chi, dir_to_ind, site_path, state, env
):
    """Apply one periodic T-string while keeping only one open chi leg.

    This is algebraically the same contraction as ``mv_sigma_dense`` but expands
    the trace over the periodic chi leg into ``sum_alpha``. The peak tensor is
    O(chi * D**L) instead of O(chi**2 * D**L).
    """
    path = site_path(start_coord)
    out = None
    for alpha in range(chi):
        c = path[0]
        T0 = _get_and_transform_T(c, d_sigma, rows, chi, dir_to_ind, state, env)[
            alpha
        ]
        W = torch.tensordot(T0, V, ([2], [0]))

        for i in range(1, L - 1):
            c = path[i]
            T = _get_and_transform_T(c, d_sigma, rows, chi, dir_to_ind, state, env)
            W = torch.tensordot(T, W, ([0, 3], [0, i + 1]))

        c = path[-1]
        Tlast = _get_and_transform_T(
            c, d_sigma, rows, chi, dir_to_ind, state, env
        )[:, alpha]
        W = torch.tensordot(Tlast, W, ([0, 2], [0, L]))
        W = W.permute(list(range(L - 1, -1, -1))).contiguous()
        out = W if out is None else out + W
    return out


def apply_expEH_once(v0, L, coord, direction, state, env, stream_chi=True):
    chi, dir_to_ind, d_opp, site_path, ads = _eh_geometry(
        L, coord, direction, state, env
    )
    V = torch.as_tensor(v0, dtype=env.dtype, device=env.device).view(ads)
    mv_sigma = mv_sigma_stream_chi if stream_chi else mv_sigma_dense
    if state.lX == state.lY == 1:
        V = mv_sigma(
            V, L, direction, coord, "ket", chi, dir_to_ind, site_path, state, env
        )
        V = mv_sigma(
            V, L, d_opp, coord, "bra", chi, dir_to_ind, site_path, state, env
        )
    else:
        V = mv_sigma(
            V, L, direction, coord, "ket", chi, dir_to_ind, site_path, state, env
        )
        V = mv_sigma(
            V,
            L,
            d_opp,
            (coord[0] + direction[0], coord[1] + direction[1]),
            "bra",
            chi,
            dir_to_ind,
            site_path,
            state,
            env,
        )
    return V.reshape(int(np.prod(ads)))


def get_EH_spec_Ttensor_momentum(
    n,
    L,
    coord,
    direction,
    state,
    env,
    return_momentum=False,
    full_diag_threshold=4096,
    stream_chi=True,
    print_each_matvec=False,
):
    assert L > 1, "L must be larger than 1"
    chi, _, _, _, ads = _eh_geometry(L, coord, direction, state, env)
    if len(set(ads)) != 1 and return_momentum:
        raise ValueError("Momentum labels require equal boundary dimensions on all sites")
    dim = int(np.prod(ads))
    if dim <= n:
        return None

    mv_count = {"n": 0}

    def mv(v0):
        started = time.time()
        y = apply_expEH_once(
            v0, L, coord, direction, state, env, stream_chi=stream_chi
        )
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize(env.device)
        y_np = y.cpu().numpy()
        mv_count["n"] += 1
        if print_each_matvec:
            peak_mem = _max_memory_mb(env.device)
            print(
                f"EH_MATVEC_DONE L={L} count={mv_count['n']} "
                f"stream_chi={stream_chi} "
                f"time={time.time() - started:.6f} peak_mem_MB={peak_mem}",
                flush=True,
            )
        return y_np

    test_T = torch.zeros(1, dtype=env.dtype)
    expEH = LinearOperator(
        (dim, dim),
        matvec=mv,
        dtype="complex128" if test_T.is_complex() else "float64",
    )

    use_full_diag = (
        full_diag_threshold is not None
        and full_diag_threshold > 0
        and dim <= full_diag_threshold
    )
    if use_full_diag:
        mat = np.empty(
            (dim, dim), dtype=np.complex128 if test_T.is_complex() else np.float64
        )
        eye = np.eye(dim, dtype=mat.dtype)
        for j in range(dim):
            mat[:, j] = mv(eye[:, j])
        vals, vecs = np.linalg.eig(mat)
    elif return_momentum:
        vals, vecs = eigs(expEH, k=n, v0=None, return_eigenvectors=True)
    else:
        vals = eigs(expEH, k=n, v0=None, return_eigenvectors=False)

    ind_sorted = np.argsort(np.abs(vals))[::-1]
    vals = vals[ind_sorted]
    if use_full_diag:
        norm = np.sum(vals)
        if np.abs(norm) > 1.0e-300:
            vals = vals / norm
    else:
        vals = (1.0 / np.abs(vals[0])) * vals
    vals = vals[:n]

    spec = torch.zeros((n, 2), dtype=torch.float64, device=state.device)
    spec[:, 0] = torch.as_tensor(np.real(vals), device=state.device)
    spec[:, 1] = torch.as_tensor(np.imag(vals), device=state.device)
    if not return_momentum:
        return spec

    vecs = vecs[:, ind_sorted[:n]]
    vec_dtype = env.dtype if test_T.is_complex() else torch.complex128
    k_phases = torch.zeros((n, 2), dtype=torch.float64, device=state.device)
    k_phases_inverse = torch.zeros(
        (n, 2), dtype=torch.float64, device=state.device
    )
    for i in range(n):
        V = torch.as_tensor(vecs[:, i], dtype=vec_dtype, device=env.device).view(ads)
        phase = _boundary_momentum_phase(V)
        phase_inverse = _boundary_momentum_phase_inverse(V)
        k_phases[i, 0] = phase.real.to(device=state.device)
        k_phases[i, 1] = phase.imag.to(device=state.device)
        k_phases_inverse[i, 0] = phase_inverse.real.to(device=state.device)
        k_phases_inverse[i, 1] = phase_inverse.imag.to(device=state.device)
    return spec, k_phases, k_phases_inverse


def _max_memory_mb(device):
    if torch.device(device).type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
