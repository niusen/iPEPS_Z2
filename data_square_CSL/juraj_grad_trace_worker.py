import argparse
import os
import sys
import time

import numpy
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--juraj-repo", default=r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev")
    parser.add_argument("--instate", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--chi", type=int, default=24)
    parser.add_argument("--ctm-max-iter", type=int, default=60)
    parser.add_argument("--opt-max-iter", type=int, default=3)
    parser.add_argument("--j1", type=float, required=True)
    parser.add_argument("--j2", type=float, required=True)
    parser.add_argument("--lmbd", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", default="backtracking")
    parser.add_argument("--target-energy", type=float, default=None)
    parser.add_argument("--check-every", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.juraj_repo)
    os.chdir(args.juraj_repo)

    import config as cfg
    from ctm.one_site_c4v.env_c4v import ENV_C4V, init_env
    from ctm.one_site_c4v import ctmrg_c4v
    from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
    from ipeps.ipeps_c4v import read_ipeps_c4v, to_ipeps_c4v, make_c4v_symm
    from models import j1j2lambda
    from optim.lbfgs_modified import LBFGS_MOD

    class Args:
        pass

    cfg_args = Args()
    cfg_args.out_prefix = os.path.splitext(args.out_npz)[0] + "_juraj_worker"
    # Minimal subset used by cfg.configure and the modules touched here.
    cfg_args.GLOBALARGS_dtype = "complex128"
    cfg_args.GLOBALARGS_device = "cpu"
    cfg_args.GLOBALARGS_offload_to_gpu = None
    cfg_args.GLOBALARGS_tensor_io_format = "legacy"
    cfg_args.PEPSARGS_build_dl = True
    cfg_args.PEPSARGS_build_dl_open = False
    cfg_args.PEPSARGS_quasi_gauge_max_iter = 1000000
    cfg_args.PEPSARGS_quasi_gauge_tol = 1.0e-8
    cfg_args.CTMARGS_ctm_max_iter = args.ctm_max_iter
    cfg_args.CTMARGS_ctm_env_init_type = "CTMRG"
    cfg_args.CTMARGS_ctm_conv_tol = 1.0e-8
    cfg_args.CTMARGS_ctm_absorb_normalization = "inf"
    cfg_args.CTMARGS_fpcm_init_iter = 1
    cfg_args.CTMARGS_fpcm_freq = -1
    cfg_args.CTMARGS_fpcm_isogauge_tol = 1.0e-14
    cfg_args.CTMARGS_fpcm_fpt_tol = 1.0e-8
    cfg_args.CTMARGS_conv_check_cpu = False
    cfg_args.CTMARGS_projector_method = "4X4"
    cfg_args.CTMARGS_projector_svd_method = "DEFAULT"
    cfg_args.CTMARGS_projector_svd_reltol = 1.0e-8
    cfg_args.CTMARGS_projector_svd_reltol_block = 0.0
    cfg_args.CTMARGS_projector_eps_multiplet = 1.0e-8
    cfg_args.CTMARGS_projector_multiplet_abstol = 1.0e-14
    cfg_args.CTMARGS_ad_decomp_reg = 1.0e-12
    cfg_args.CTMARGS_ctm_move_sequence = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    cfg_args.CTMARGS_ctm_force_dl = False
    cfg_args.CTMARGS_ctm_logging = False
    cfg_args.CTMARGS_verbosity_initialization = 0
    cfg_args.CTMARGS_verbosity_ctm_convergence = 0
    cfg_args.CTMARGS_verbosity_projectors = 0
    cfg_args.CTMARGS_verbosity_ctm_move = 0
    cfg_args.CTMARGS_verbosity_fpcm_move = 0
    cfg_args.CTMARGS_verbosity_rdm = 0
    cfg_args.CTMARGS_fwd_checkpoint_c2x2 = False
    cfg_args.CTMARGS_fwd_checkpoint_halves = False
    cfg_args.CTMARGS_fwd_checkpoint_projectors = False
    cfg_args.CTMARGS_fwd_checkpoint_absorb = False
    cfg_args.CTMARGS_fwd_checkpoint_move = False
    cfg_args.CTMARGS_fwd_checkpoint_loop_rdm = False
    cfg.configure(cfg_args)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    model = j1j2lambda.J1J2LAMBDA_C4V_BIPARTITE(j1=args.j1, j2=args.j2, lmbd=args.lmbd)
    state = read_ipeps_c4v(args.instate)
    ctm_env = ENV_C4V(args.chi, to_ipeps_c4v(state))
    init_env(to_ipeps_c4v(state), ctm_env)

    def ctmrg_conv_f(state_sym, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = {"log": []}
        rdm2x1 = rdm2x1_sl(state_sym, env, force_cpu=ctm_args.conv_check_cpu)
        dist = float("inf")
        if len(history["log"]) > 0:
            dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"] = rdm2x1
        history["log"].append(dist)
        return dist < ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter, history

    params = state.get_parameters()
    for param in params:
        param.requires_grad_(True)
    optimizer = LBFGS_MOD(
        params,
        max_iter=1,
        lr=args.lr,
        tolerance_grad=1.0e-5,
        tolerance_change=1.0e-9,
        history_size=args.history_size,
        line_search_fn=args.line_search,
        line_search_eps=1.0e-8,
    )

    traces = {
        "loss": [],
        "grad": [],
        "param": [],
        "ls_loss": [],
        "post_energy": [],
        "step_time": [],
        "elapsed_time": [],
    }
    current_env = [ctm_env]

    def loss_for_state(line_searching=False):
        state_sym = to_ipeps_c4v(state, normalize=True)
        init_env(state_sym, current_env[0])
        env, history, *_ = ctmrg_c4v.run(state_sym, current_env[0], conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)
        current_env[0] = env
        return model.energy_1x1(state_sym, env, force_cpu=True)

    def closure(linesearching=False):
        optimizer.zero_grad()
        loss = loss_for_state(line_searching=linesearching)
        loss.backward()
        current_env[0].detach_()
        if not linesearching:
            traces["loss"].append(float(loss.detach().cpu()))
            traces["grad"].append(state.site().grad.detach().cpu().numpy().copy())
            traces["param"].append(state.site().detach().cpu().numpy().copy())
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        loss = loss_for_state(line_searching=linesearching)
        traces["ls_loss"].append(float(loss.detach().cpu()))
        return loss

    opt_start = time.perf_counter()
    target_reached = False
    for step in range(1, args.opt_max_iter + 1):
        step_start = time.perf_counter()
        optimizer.step_2c(closure, closure_linesearch)
        step_time = time.perf_counter() - step_start
        post_energy = float("nan")
        if args.check_every > 0 and (step % args.check_every == 0 or step == args.opt_max_iter):
            with torch.no_grad():
                post_energy = float(loss_for_state().detach().cpu())
        elapsed = time.perf_counter() - opt_start
        traces["post_energy"].append(post_energy)
        traces["step_time"].append(step_time)
        traces["elapsed_time"].append(elapsed)
        loss = traces["loss"][-1] if traces["loss"] else float("nan")
        grad_norm = state.site().grad.norm().item() if state.site().grad is not None else float("nan")
        print(
            f"step,{step},loss,{loss:.16g},post_energy,{post_energy:.16g},"
            f"grad_norm,{grad_norm:.16g},step_s,{step_time:.6g},elapsed_s,{elapsed:.6g}"
        )
        if args.target_energy is not None and post_energy <= args.target_energy:
            target_reached = True
            print(
                f"target_reached,step,{step},post_energy,{post_energy:.16g},"
                f"target,{args.target_energy:.16g},elapsed_s,{elapsed:.6g}"
            )
            break

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    numpy.savez(
        args.out_npz,
        losses=numpy.array(traces["loss"]),
        grads=numpy.array(traces["grad"]),
        params=numpy.array(traces["param"]),
        ls_losses=numpy.array(traces["ls_loss"]),
        post_energies=numpy.array(traces["post_energy"]),
        step_times=numpy.array(traces["step_time"]),
        elapsed_times=numpy.array(traces["elapsed_time"]),
        target_reached=numpy.array(target_reached),
    )
    print("wrote", args.out_npz)


if __name__ == "__main__":
    main()
