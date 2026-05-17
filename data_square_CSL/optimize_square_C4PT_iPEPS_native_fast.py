import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import dense_tensor_from_array, load_square_iPEPS, save_square_iPEPS
from optimization.lbfgs_modified import LBFGS_MOD


PRL_J1 = 1.7776001555035785
PRL_J2 = 0.8364751394573281
PRL_LMBD = 0.3747626291714492


def square_to_c4v(A_square):
    # iPEPS_Z2 square: [l,d,r,u,p]. Native C4/PT fast path: [p,u,l,d,r].
    return numpy.transpose(A_square, (4, 3, 0, 1, 2))


def c4v_to_square(A_c4v):
    return numpy.transpose(A_c4v, (2, 3, 4, 1, 0))


def load_square_as_c4v_tensor(square_prefix, config_kwargs, dtype, device):
    A_set = load_square_iPEPS(square_prefix, config_kwargs)
    A_np = numpy.array(A_set["1,1"].to_dense().to("cpu"))
    A = torch.as_tensor(square_to_c4v(A_np), dtype=dtype, device=device).contiguous()
    return A / torch.linalg.norm(A)


def save_c4v_tensor_as_square(A_c4v, square_prefix, config_kwargs):
    A_square = c4v_to_square(A_c4v.detach().cpu().numpy())
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    save_square_iPEPS(A_set, square_prefix, config_kwargs)


def make_c4v_a1(A):
    A = 0.5 * (A + A.permute(0, 1, 4, 3, 2))
    A = 0.5 * (A + A.permute(0, 3, 2, 1, 4))
    A = 0.5 * (A + A.permute(0, 4, 1, 2, 3))
    A = 0.5 * (A + A.permute(0, 2, 3, 4, 1))
    return A


def make_c4v_a2(A):
    A = 0.5 * (A - A.permute(0, 1, 4, 3, 2))
    A = 0.5 * (A - A.permute(0, 4, 3, 2, 1))
    A = 0.5 * (A + A.permute(0, 4, 1, 2, 3))
    A = 0.5 * (A + A.permute(0, 3, 4, 1, 2))
    return A


def project_c4pt(A):
    A = make_c4v_a1(A.real) + 1.0j * make_c4v_a2(A.imag)
    return A / torch.linalg.norm(A)


def safe_inverse(x, epsilon=1.0e-12):
    return x / (x * x + epsilon)


class HermitianEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, ad_decomp_reg):
        D, U = torch.linalg.eigh(M)
        _, order = torch.sort(torch.abs(D), descending=True)
        D = D[order]
        U = U[:, order]
        ctx.save_for_backward(D, U, ad_decomp_reg)
        return D, U

    @staticmethod
    def backward(ctx, dD, dU):
        D, U, ad_decomp_reg = ctx.saved_tensors
        Uh = U.t().conj()
        F = safe_inverse(D - D[:, None], epsilon=ad_decomp_reg)
        F.diagonal().fill_(0)
        dM = U @ (torch.diag(dD.to(U.dtype)) + F.to(U.dtype) * (Uh @ dU)) @ Uh
        dM = 0.5 * (dM + dM.conj().t())
        return dM, None


def truncated_eig_sym(M, chi):
    M = 0.5 * (M + M.conj().t())
    reg = torch.as_tensor(1.0e-12, dtype=M.real.dtype, device=M.device)
    D, U = HermitianEigh.apply(M, reg)
    D = D[: min(chi, M.size(0))]
    U = U[:, : D.numel()]
    if U.size(1) < chi:
        pad_cols = chi - U.size(1)
        U = torch.nn.functional.pad(U, (0, pad_cols))
        D = torch.nn.functional.pad(D, (0, pad_cols))
    return D, U


def _normalize_ctm(C, T, norm_type="inf"):
    if norm_type == "inf":
        c_norm = C.abs().max().clamp_min(1.0e-30)
        t_norm = T.abs().max().clamp_min(1.0e-30)
    else:
        c_norm = torch.linalg.norm(C).clamp_min(1.0e-30)
        t_norm = torch.linalg.norm(T).clamp_min(1.0e-30)
    return C / c_norm, T / t_norm


@dataclass
class EnvC4V:
    chi: int
    C: torch.Tensor
    T: torch.Tensor

    def detach_(self):
        self.C = self.C.detach()
        self.T = self.T.detach()
        return self


def init_env_ctmrg(a, chi):
    dtype = a.dtype
    device = a.device
    d = a.size()
    dk = [d[i + 1] * d[i + 1] for i in range(4)]

    corner = torch.einsum("mijef,mijab->eafb", a, a.conj()).contiguous().view(dk[2], dk[3])
    corner = corner / corner.abs().max().clamp_min(1.0e-30)
    D, U = truncated_eig_sym(corner, corner.size(0))
    diag_corner = torch.diag(D.to(dtype))

    C = torch.zeros((chi, chi), dtype=dtype, device=device)
    c0 = min(chi, diag_corner.size(0))
    C[:c0, :c0] = diag_corner[:c0, :c0]

    edge = torch.einsum("meifg,maibc->eafbgc", a, a.conj()).contiguous().view(dk[0], dk[2], dk[3])
    edge = edge / edge.abs().max().clamp_min(1.0e-30)
    edge = torch.einsum("ai,abs,bj->ijs", U, edge, U.conj())

    T = torch.zeros((chi, chi, dk[3]), dtype=dtype, device=device)
    t0 = min(chi, edge.size(0))
    t1 = min(chi, edge.size(1))
    T[:t0, :t1, :] = edge[:t0, :t1, :]
    C, T = _normalize_ctm(C, T)
    return EnvC4V(chi=chi, C=C, T=T)


def c2x2_sl(a, C, T):
    C2x2 = torch.tensordot(C, T, ([1], [1]))
    C2x2 = torch.tensordot(C2x2, T, ([0], [0]))
    C2x2 = C2x2.view(C2x2.size(0), a.size(1), a.size(1), C2x2.size(2), a.size(2), a.size(2))
    C2x2 = torch.tensordot(C2x2, a, ([1, 4], [1, 2]))
    C2x2 = torch.tensordot(C2x2, a.conj(), ([1, 3, 4], [1, 2, 0]))
    C2x2 = C2x2.permute(1, 2, 4, 0, 3, 5).contiguous().view(
        C2x2.size(1) * a.size(3) * a.size(3),
        C2x2.size(0) * a.size(4) * a.size(4),
    )
    return C2x2


def ctm_move_sl(a, env):
    C2X2 = c2x2_sl(a, env.C, env.T)
    D, P = truncated_eig_sym(C2X2, env.chi)
    new_C = torch.diag(D.to(a.dtype))

    P = P.view(env.chi, env.T.size(2), env.chi)
    new_T = torch.tensordot(P, env.T, ([0], [0]))
    new_T = new_T.view(a.size(1), a.size(1), new_T.size(1), new_T.size(2), a.size(2), a.size(2))
    new_T = torch.tensordot(new_T, a, ([0, 4], [1, 2]))
    new_T = torch.tensordot(new_T, a.conj(), ([0, 3, 4], [1, 2, 0]))
    new_T = new_T.permute(0, 1, 2, 4, 3, 5).contiguous().view(
        new_T.size(0), new_T.size(1), a.size(3) * a.size(3), a.size(4) * a.size(4)
    )
    new_T = torch.tensordot(new_T, P.conj(), ([1, 2], [0, 1]))
    new_T = new_T.permute(0, 2, 1).contiguous()
    new_T = 0.5 * (new_T + new_T.conj().permute(1, 0, 2))

    new_C, new_T = _normalize_ctm(new_C, new_T)
    env.C = new_C
    env.T = new_T
    return env


def sym_pos_def_rdm(rdm, sym_pos_def=False):
    nsites = len(rdm.size()) // 2
    orig_shape = tuple(rdm.size())
    left_dim = math.prod(orig_shape[:nsites])
    rdm_m = rdm.reshape(left_dim, -1)
    rdm_m = 0.5 * (rdm_m + rdm_m.conj().t())
    if sym_pos_def:
        with torch.no_grad():
            D, U = torch.linalg.eigh(rdm_m)
            D = torch.clamp(D, min=0)
            rdm_m = U @ torch.diag(D.to(rdm_m.dtype)) @ U.conj().t()
    norm = rdm_m.diagonal().sum().real
    return (rdm_m / norm).reshape(orig_shape)


def rdm2x1_sl(a, env):
    C = env.C
    T = env.T
    C2x1 = torch.tensordot(C, T, ([1], [1]))
    C2x2 = torch.tensordot(C2x1, T, ([0], [0]))
    C2x2 = C2x2.view(C2x2.size(0), a.size(1), a.size(1), C2x2.size(2), a.size(2), a.size(2))
    C2x2 = torch.tensordot(C2x2, a, ([1, 4], [1, 2]))
    C2x2 = torch.tensordot(C2x2, a.conj(), ([1, 3], [1, 2]))
    C2x2 = C2x2.permute(1, 3, 6, 0, 4, 7, 2, 5).contiguous().view(
        C2x2.size(1), a.size(3) * a.size(3), C2x2.size(0), a.size(4) * a.size(4), a.size(0), a.size(0)
    )
    C2x1 = torch.tensordot(C, T, ([1], [0]))
    left_half = torch.tensordot(C2x1, C2x2, ([0, 2], [0, 1]))
    rdm = torch.tensordot(left_half, left_half, ([0, 1, 2], [1, 0, 2]))
    rdm = rdm.permute(0, 2, 1, 3).contiguous()
    return sym_pos_def_rdm(rdm)


def run_ctmrg(a, chi, max_iter=80, conv_tol=1.0e-8, min_iter=1, conv_check=True):
    env = init_env_ctmrg(a, chi)
    history = []
    old_rdm = None
    for step in range(1, max_iter + 1):
        env = ctm_move_sl(a, env)
        if conv_check:
            rdm = rdm2x1_sl(a, env)
            dist = float("inf") if old_rdm is None else torch.dist(rdm, old_rdm, p=2).item()
            history.append(dist)
            old_rdm = rdm
            if step >= min_iter and dist < conv_tol:
                break
    return env, history


def open_c2x2_lu_dl(C, T, a):
    d = a.size()
    A = torch.einsum("mefgh,nabcd->eafbgchdmn", a, a.conj()).contiguous().view(
        d[1] ** 2, d[2] ** 2, d[3] ** 2, d[4] ** 2, d[0], d[0]
    )
    C2x2 = torch.tensordot(C, T, ([1], [1]))
    C2x2 = torch.tensordot(C2x2, T, ([0], [0]))
    C2x2 = torch.tensordot(C2x2, A, ([1, 3], [0, 1]))
    C2x2 = C2x2.permute(1, 2, 0, 3, 4, 5).contiguous().view(
        T.size(1) * A.size(2), T.size(1) * A.size(3), d[0], d[0]
    )
    return C2x2


def rdm2x2(a, env):
    C2x2 = open_c2x2_lu_dl(env.C, env.T, a)
    upper_half = torch.tensordot(C2x2, C2x2, ([1], [0]))
    upper_half = upper_half.permute(0, 3, 1, 2, 4, 5)
    rdm = torch.tensordot(upper_half, upper_half, ([0, 1], [1, 0]))
    rdm = rdm.permute(0, 2, 6, 4, 1, 3, 7, 5).contiguous()
    return sym_pos_def_rdm(rdm)


def spin_ops(dtype, device):
    I = torch.eye(2, dtype=dtype, device=device)
    SZ = torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=dtype, device=device)
    SP = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    SM = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)
    ROT = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype, device=device)
    return I, SZ, SP, SM, ROT


def build_hamiltonian(dtype, device, j1=PRL_J1, j2=PRL_J2, lmbd=PRL_LMBD):
    I, SZ, SP, SM, ROT = spin_ops(dtype, device)
    id2 = torch.eye(4, dtype=dtype, device=device).view(2, 2, 2, 2)
    expr = "ij,ab->iajb"
    SS = torch.einsum(expr, SZ, SZ) + 0.5 * (
        torch.einsum(expr, SP, SM) + torch.einsum(expr, SM, SP)
    )
    h2x2_SS = torch.einsum("ijab,klcd->ijklabcd", SS, id2)
    hp = (
        0.5
        * j1
        * (
            h2x2_SS
            + h2x2_SS.permute(0, 2, 1, 3, 4, 6, 5, 7)
            + h2x2_SS.permute(2, 3, 0, 1, 6, 7, 4, 5)
            + h2x2_SS.permute(3, 1, 2, 0, 7, 5, 6, 4)
        )
        + j2
        * (
            h2x2_SS.permute(0, 3, 2, 1, 4, 7, 6, 5)
            + h2x2_SS.permute(2, 1, 0, 3, 6, 5, 4, 7)
        )
    )
    hp = torch.einsum("xj,yk,ixylauvd,ub,vc->ijklabcd", ROT, ROT, hp, ROT, ROT)

    P12 = torch.as_tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device
    ).view(2, 2, 2, 2)
    P12II = torch.einsum("abij,cdkl->abcdijkl", P12, id2)
    PI12I = P12II.permute(3, 0, 1, 2, 7, 4, 5, 6).contiguous()
    PII12 = P12II.permute(2, 3, 0, 1, 6, 7, 4, 5).contiguous()
    P4 = torch.tensordot(PI12I, P12II, ([4, 5, 6, 7], [0, 1, 2, 3]))
    P4 = torch.tensordot(PII12, P4, ([4, 5, 6, 7], [0, 1, 2, 3]))
    chiral = 1.0j * (P4 - P4.view(16, 16).t().view(2, 2, 2, 2, 2, 2, 2, 2))
    chiral = chiral.permute(0, 1, 3, 2, 4, 5, 7, 6)
    chiral = torch.einsum("xj,yk,ixylauvd,ub,vc->ijklabcd", ROT, ROT, chiral, ROT, ROT)
    return (hp + lmbd * chiral).contiguous(), SS.contiguous()


def energy_and_env(raw_A, h2x2, chi, ctm_max_iter, ctm_conv_tol, conv_check=True):
    a = project_c4pt(raw_A)
    env, history = run_ctmrg(a, chi, max_iter=ctm_max_iter, conv_tol=ctm_conv_tol, conv_check=conv_check)
    rho = rdm2x2(a, env)
    energy = torch.einsum("ijklabcd,ijklabcd", rho, h2x2).real
    return energy, env, history, a


def eval_state(raw_A, h2x2, ss_op, args):
    start = time.perf_counter()
    energy, env, history, a = energy_and_env(
        raw_A,
        h2x2,
        args.chi,
        args.ctm_max_iter,
        args.ctm_conv_tol,
        conv_check=not args.no_conv_check,
    )
    rho21 = rdm2x1_sl(a, env)
    ss = torch.einsum("ijab,ijab", rho21, ss_op).real
    wall = time.perf_counter() - start
    return energy, ss, history, wall


def main():
    parser = argparse.ArgumentParser(
        description="Native iPEPS_Z2 fast C4/PT CSL optimizer using a one-site C4v single-layer CTMRG."
    )
    parser.add_argument("--mode", choices=("eval", "opt"), default="eval")
    parser.add_argument("--square-instate-prefix", default=None)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default="data_square_CSL/native_fast_C4PT")
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--no-conv-check", action="store_true")
    parser.add_argument("--opt-max-iter", type=int, default=20)
    parser.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", choices=("backtracking", "strong_wolfe", "none"), default="backtracking")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--target-energy", type=float, default=None)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("complex128", "complex64"), default="complex128")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    device = torch.device(args.device)
    square_config = {
        "backend": "torch",
        "default_dtype": args.dtype,
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }
    if args.random_init:
        torch.manual_seed(args.seed)
        raw_re = torch.randn((2, args.D, args.D, args.D, args.D), dtype=torch.float64, device=device)
        raw_im = torch.randn((2, args.D, args.D, args.D, args.D), dtype=torch.float64, device=device)
        raw_A = make_c4v_a1(raw_re) + 1.0j * make_c4v_a2(raw_im)
        raw_A = raw_A.to(dtype=dtype)
        raw_A = raw_A / torch.linalg.norm(raw_A)
        print("random_init_seed:", args.seed)
    else:
        if args.square_instate_prefix is None:
            raise ValueError("provide --square-instate-prefix or use --random-init")
        raw_A = load_square_as_c4v_tensor(args.square_instate_prefix, square_config, dtype, device)
    raw_A = torch.nn.Parameter(raw_A)
    h2x2, ss_op = build_hamiltonian(dtype, device)

    energy, ss, history, wall = eval_state(raw_A, h2x2, ss_op, args)
    print("initial_energy:", f"{energy.item():.16g}")
    print("initial_SS2x1:", f"{ss.item():.16g}")
    print("initial_ctm_steps:", len(history))
    print("initial_ctm_last_err:", history[-1] if history else "not_checked")
    print("initial_eval_wall_s:", f"{wall:.6g}")

    if args.mode == "eval":
        return

    if args.optimizer == "lbfgs":
        optimizer = LBFGS_MOD(
            [raw_A],
            lr=args.lr,
            max_iter=1,
            history_size=args.history_size,
            line_search_fn=None if args.line_search == "none" else args.line_search,
            tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
            line_search_eps=1.0e-8,
        )
    else:
        optimizer = torch.optim.Adam([raw_A], lr=args.lr)

    losses = []
    post_energies = []
    step_times = []
    grad_norms = []
    elapsed_times = []

    def closure(linesearching=False):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = energy_and_env(
            raw_A,
            h2x2,
            args.chi,
            args.ctm_max_iter,
            args.ctm_conv_tol,
            conv_check=not args.no_conv_check,
        )
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([raw_A], args.max_grad_norm)
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        loss, _, _, _ = energy_and_env(
            raw_A,
            h2x2,
            args.chi,
            args.ctm_max_iter,
            args.ctm_conv_tol,
            conv_check=not args.no_conv_check,
        )
        return loss

    opt_start = time.perf_counter()
    reached_target = False
    for step in range(1, args.opt_max_iter + 1):
        step_start = time.perf_counter()
        if args.optimizer == "lbfgs":
            loss_tensor = optimizer.step_2c(closure, closure_linesearch)
        else:
            loss_tensor = closure()
            optimizer.step()
        with torch.no_grad():
            raw_A.copy_(project_c4pt(raw_A))
        step_wall = time.perf_counter() - step_start
        grad_norm = raw_A.grad.norm().item() if raw_A.grad is not None else float("nan")
        loss_value = float(loss_tensor.detach().cpu())
        post_energy_value = float("nan")
        if args.check_every > 0 and (step % args.check_every == 0 or step == args.opt_max_iter):
            with torch.no_grad():
                post_energy, _, _, _ = energy_and_env(
                    raw_A,
                    h2x2,
                    args.chi,
                    args.ctm_max_iter,
                    args.ctm_conv_tol,
                    conv_check=not args.no_conv_check,
                )
                post_energy_value = float(post_energy.detach().cpu())
        elapsed = time.perf_counter() - opt_start
        losses.append(loss_value)
        post_energies.append(post_energy_value)
        grad_norms.append(grad_norm)
        step_times.append(step_wall)
        elapsed_times.append(elapsed)
        print(
            f"step,{step},loss,{loss_value:.16g},post_energy,{post_energy_value:.16g},"
            f"grad_norm,{grad_norm:.16g},step_s,{step_wall:.6g},elapsed_s,{elapsed:.6g}"
        )
        if args.target_energy is not None and post_energy_value <= args.target_energy:
            print(
                f"target_reached,step,{step},post_energy,{post_energy_value:.16g},"
                f"target,{args.target_energy:.16g},elapsed_s,{elapsed:.6g}"
            )
            reached_target = True
            break

    final_energy, final_ss, final_history, final_wall = eval_state(raw_A, h2x2, ss_op, args)
    final_A = project_c4pt(raw_A)
    print("final_energy:", f"{final_energy.item():.16g}")
    print("final_SS2x1:", f"{final_ss.item():.16g}")
    print("final_ctm_steps:", len(final_history))
    print("final_ctm_last_err:", final_history[-1] if final_history else "not_checked")
    print("final_eval_wall_s:", f"{final_wall:.6g}")
    save_c4v_tensor_as_square(final_A, args.out_prefix, square_config)
    timing_path = args.out_prefix + "_native_fast_timing.npz"
    os.makedirs(os.path.dirname(os.path.abspath(timing_path)), exist_ok=True)
    numpy.savez(
        timing_path,
        losses=numpy.array(losses),
        post_energies=numpy.array(post_energies),
        grad_norms=numpy.array(grad_norms),
        step_times=numpy.array(step_times),
        elapsed_times=numpy.array(elapsed_times),
        target_reached=numpy.array(reached_target),
    )
    print("saved_square_state:", args.out_prefix + ".json")
    print("saved_timing:", timing_path)


if __name__ == "__main__":
    main()
