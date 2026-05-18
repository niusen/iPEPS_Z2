import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import dense_tensor_from_array, load_square_iPEPS, save_square_iPEPS


JURAJ_REPO = r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev"
PRL_J1 = 1.7776001555035785
PRL_J2 = 0.8364751394573281
PRL_LMBD = 0.3747626291714492


def square_to_juraj(A_square):
    # iPEPS_Z2 square: [l,d,r,u,p]. Juraj/peps-torch C4v: [p,u,l,d,r].
    return numpy.transpose(A_square, (4, 3, 0, 1, 2))


def juraj_to_square(A_juraj):
    return numpy.transpose(A_juraj, (2, 3, 4, 1, 0))


def load_square_as_juraj_tensor(square_prefix, config_kwargs):
    A_set = load_square_iPEPS(square_prefix, config_kwargs)
    return torch.as_tensor(square_to_juraj(numpy.array(A_set["1,1"].to_dense().to("cpu")))).contiguous()


def save_juraj_tensor_as_square(A_juraj, square_prefix, config_kwargs):
    A_square = juraj_to_square(A_juraj.detach().cpu().numpy())
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    save_square_iPEPS(A_set, square_prefix, config_kwargs)


def configure_juraj(args):
    sys.path.insert(0, args.juraj_repo)
    import config as cfg

    class CfgArgs:
        pass

    cfg_args = CfgArgs()
    cfg_args.out_prefix = args.out_prefix + "_direct_fast"
    cfg_args.GLOBALARGS_dtype = "complex128"
    cfg_args.GLOBALARGS_device = args.device
    cfg_args.GLOBALARGS_offload_to_gpu = None
    cfg_args.GLOBALARGS_tensor_io_format = "legacy"
    cfg_args.PEPSARGS_build_dl = True
    cfg_args.PEPSARGS_build_dl_open = False
    cfg_args.PEPSARGS_quasi_gauge_max_iter = 1000000
    cfg_args.PEPSARGS_quasi_gauge_tol = 1.0e-8
    cfg_args.CTMARGS_ctm_max_iter = args.ctm_max_iter
    cfg_args.CTMARGS_ctm_env_init_type = "CTMRG"
    cfg_args.CTMARGS_ctm_conv_tol = args.ctm_conv_tol
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
    cfg_args.CTMARGS_verbosity_ctm_convergence = args.verbosity_ctm_convergence
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
    return cfg


def ctmrg_conv_checker(cfg, rdm2x1_sl):
    def ctmrg_conv_f(state_sym, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history = {"log": []}
        rdm2x1 = rdm2x1_sl(state_sym, env, force_cpu=ctm_args.conv_check_cpu)
        dist = float("inf")
        if len(history["log"]) > 0:
            dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"] = rdm2x1
        history["log"].append(dist)
        converged = dist < ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter
        return converged, history

    return ctmrg_conv_f


def evaluate_state(state, model, cfg, chi, ctmrg_c4v, ENV_C4V, init_env, ctmrg_conv_f, force_cpu=True):
    state_sym = state if state.__class__.__name__ == "IPEPS_C4V" else state
    env = ENV_C4V(chi, state_sym)
    init_env(state_sym, env)
    env, history, t_ctm, t_check = ctmrg_c4v.run(state_sym, env, conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args)
    energy = model.energy_1x1(state_sym, env, force_cpu=force_cpu)
    obs_values, obs_labels = model.eval_obs(state_sym, env, force_cpu=force_cpu)
    return energy, obs_labels, obs_values, history, t_ctm, t_check


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser(
        description="In-process fast C4/PT square CSL optimizer for iPEPS_Z2 states using Juraj's one-site C4v backend."
    )
    parser.add_argument("--mode", choices=("eval", "opt"), default="eval")
    parser.add_argument("--juraj-repo", default=JURAJ_REPO)
    parser.add_argument("--square-instate-prefix", required=True)
    parser.add_argument("--out-prefix", default="data_square_CSL/direct_fast_C4PT")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--opt-max-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", default="backtracking")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--verbosity-ctm-convergence", type=int, default=0)
    args = parser.parse_args()

    cfg = configure_juraj(args)

    from ctm.one_site_c4v.env_c4v import ENV_C4V, init_env
    from ctm.one_site_c4v import ctmrg_c4v
    from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
    from ipeps.ipeps_c4v import IPEPS_C4V, to_ipeps_c4v
    from models import j1j2lambda
    from optim.lbfgs_modified import LBFGS_MOD

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    square_config = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }
    A_juraj = load_square_as_juraj_tensor(args.square_instate_prefix, square_config).to(
        dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device
    )
    state = IPEPS_C4V(A_juraj)
    model = j1j2lambda.J1J2LAMBDA_C4V_BIPARTITE(j1=PRL_J1, j2=PRL_J2, lmbd=PRL_LMBD)
    ctmrg_conv_f = ctmrg_conv_checker(cfg, rdm2x1_sl)
    force_cpu = args.force_cpu or str(args.device) == "cpu"

    eval_start = time.perf_counter()
    state_sym = to_ipeps_c4v(state, normalize=True)
    energy, obs_labels, obs_values, history, t_ctm, t_check = evaluate_state(
        state_sym, model, cfg, args.chi, ctmrg_c4v, ENV_C4V, init_env, ctmrg_conv_f, force_cpu=force_cpu
    )
    eval_wall = time.perf_counter() - eval_start
    print("initial_energy:", energy.item())
    print("initial_ctm_history_len:", len(history["log"]))
    print("initial_eval_wall_s:", eval_wall)
    print("initial_obs:", ", ".join([f"{label}={value}" for label, value in zip(obs_labels, obs_values)]))

    if args.mode == "eval":
        return

    env = ENV_C4V(args.chi, state_sym)
    init_env(state_sym, env)
    current_env = [env]
    traces = {"loss": [], "step_time": [], "ls_loss": []}

    params = list(state.get_parameters())
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

    def loss_for_state():
        state_sym_local = to_ipeps_c4v(state, normalize=True)
        init_env(state_sym_local, current_env[0])
        env_out, history_out, *_ = ctmrg_c4v.run(
            state_sym_local, current_env[0], conv_check=ctmrg_conv_f, ctm_args=cfg.ctm_args
        )
        current_env[0] = env_out
        return model.energy_1x1(state_sym_local, env_out, force_cpu=force_cpu)

    def closure(linesearching=False):
        optimizer.zero_grad()
        loss = loss_for_state()
        loss.backward()
        current_env[0].detach_()
        if not linesearching:
            traces["loss"].append(float(loss.detach().cpu()))
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        loss = loss_for_state()
        traces["ls_loss"].append(float(loss.detach().cpu()))
        return loss

    for step in range(1, args.opt_max_iter + 1):
        step_start = time.perf_counter()
        optimizer.step_2c(closure, closure_linesearch)
        step_time = time.perf_counter() - step_start
        traces["step_time"].append(step_time)
        loss = traces["loss"][-1] if traces["loss"] else float("nan")
        grad_norm = params[0].grad.norm().item() if params[0].grad is not None else float("nan")
        print(f"step,{step},loss,{loss:.16g},grad_norm,{grad_norm:.16g},step_s,{step_time:.6g}")

    final_state = to_ipeps_c4v(state, normalize=True)
    final_start = time.perf_counter()
    final_energy, final_obs_labels, final_obs_values, final_history, *_ = evaluate_state(
        final_state, model, cfg, args.chi, ctmrg_c4v, ENV_C4V, init_env, ctmrg_conv_f, force_cpu=force_cpu
    )
    final_wall = time.perf_counter() - final_start
    print("final_energy:", final_energy.item())
    print("final_ctm_history_len:", len(final_history["log"]))
    print("final_eval_wall_s:", final_wall)
    print("final_obs:", ", ".join([f"{label}={value}" for label, value in zip(final_obs_labels, final_obs_values)]))

    save_juraj_tensor_as_square(final_state.site(), args.out_prefix, square_config)
    print("saved_square_state:", args.out_prefix + ".json")
    timing_path = args.out_prefix + "_timing.npz"
    os.makedirs(os.path.dirname(os.path.abspath(timing_path)), exist_ok=True)
    numpy.savez(timing_path, losses=numpy.array(traces["loss"]), step_times=numpy.array(traces["step_time"]))
    print("saved_timing:", timing_path)


if __name__ == "__main__":
    main()
