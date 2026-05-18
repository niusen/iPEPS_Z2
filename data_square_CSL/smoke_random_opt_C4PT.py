import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS, random_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, LINESEARCH
from model.bosonic_square_ob_iPEPS import prl_129_177201_square_csl_parameters
from optimization.optimize_bosonic_square_iPEPS import optimize_bosonic_square_iPEPS


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--ad-ctm-iters", type=int, default=4)
    parser.add_argument("--ls-ctm-iters", type=int, default=12)
    parser.add_argument("--ad-ctm-conv-tol", type=float, default=0.0)
    parser.add_argument("--ls-ctm-conv-tol", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--instate-prefix", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--step0", type=float, default=0.25)
    parser.add_argument("--ls-maxiter", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--line-search", choices=("backtracking", "hager_zhang"), default="backtracking")
    parser.add_argument("--target-energy", type=float, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    save_prefix = args.out_prefix
    if save_prefix is None:
        save_prefix = f"data_square_CSL/random_C4PT_smoke_D{args.D}_chi{args.chi}_seed{args.seed}"

    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
        "save_file_prefix": save_prefix,
    }

    global_args = GLOBALARGS()
    global_args.Lx = 1
    global_args.Ly = 1
    global_args.device = args.device

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    if args.instate_prefix is None:
        A_set = random_square_iPEPS(args.D, 2, config_kwargs)
    else:
        A_set = load_square_iPEPS(args.instate_prefix, config_kwargs)
    state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    state.require_grad(False)
    state.to_device(args.device)
    state.normalize()

    ad_ctm_args = CTMARGS()
    ad_ctm_args.chi = args.chi
    ad_ctm_args.CTM_ite_nums = args.ad_ctm_iters
    ad_ctm_args.CTM_trun_tol = 1.0e-8
    ad_ctm_args.CTM_conv_tol = args.ad_ctm_conv_tol
    ad_ctm_args.CTM_ite_info = False
    ad_ctm_args.doublelayer_on_cpu = False
    ad_ctm_args.use_sub_checkpoint = False

    ls_ctm_args = CTMARGS()
    ls_ctm_args.chi = args.chi
    ls_ctm_args.CTM_ite_nums = args.ls_ctm_iters
    ls_ctm_args.CTM_trun_tol = 1.0e-8
    ls_ctm_args.CTM_conv_tol = args.ls_ctm_conv_tol
    ls_ctm_args.CTM_ite_info = False
    ls_ctm_args.doublelayer_on_cpu = False
    ls_ctm_args.use_sub_checkpoint = False

    ls = LINESEARCH()
    ls.method = "lbfgs"
    ls.line_search = args.line_search
    ls.maxiter = args.maxiter
    ls.ls_maxiter = args.ls_maxiter
    ls.step0 = args.step0
    ls.alpha = 0.5
    ls.gtol = 1.0e-5
    ls.history_size = args.history_size
    ls.print_observables = False
    ls.max_grad_norm = args.max_grad_norm
    ls.target_energy = args.target_energy

    start = time.perf_counter()
    print(
        f"random C4/PT smoke opt: D={args.D}, chi={args.chi}, maxiter={args.maxiter}, "
        f"ad_ctm_iters={args.ad_ctm_iters}, ls_ctm_iters={args.ls_ctm_iters}, seed={args.seed}, "
        f"ad_ctm_conv_tol={args.ad_ctm_conv_tol}, ls_ctm_conv_tol={args.ls_ctm_conv_tol}, "
        f"max_grad_norm={args.max_grad_norm}, step0={args.step0}, ls_maxiter={args.ls_maxiter}, "
        f"history_size={args.history_size}, line_search={args.line_search}, "
        f"target_energy={args.target_energy}, save_prefix={save_prefix}"
    )
    optimize_bosonic_square_iPEPS(parameters, args.D, args.chi, state, ad_ctm_args, ls_ctm_args, None, global_args, config_kwargs, ls)
    elapsed = time.perf_counter() - start
    print("elapsed_wall_s:", "{:.6g}".format(elapsed))


if __name__ == "__main__":
    main()
