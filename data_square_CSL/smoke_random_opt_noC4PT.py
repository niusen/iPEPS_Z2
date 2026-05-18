import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE, load_square_iPEPS, random_square_iPEPS, save_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, LINESEARCH
from model.bosonic_square_ob_iPEPS import prl_129_177201_square_csl_parameters
from optimization.lbfgs_modified import LBFGS_MOD
from optimization.optimize_bosonic_square_iPEPS import cost_fun, optimize_bosonic_square_iPEPS


def to_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def make_state_from_params(template_state, params, detach=False):
    A_set = OrderedDict()
    for key in template_state.A_set:
        data = params[key].detach().clone() if detach else params[key]
        A_set[key] = template_state.A_set[key]._replace(data=data)
    state = IPEPS_SQUARE(A_set, template_state.global_args)
    state.normalize()
    return state


def optimize_with_lbfgsmod(parameters, D, chi, template_state, ad_ctm_args, ls_ctm_args, global_args, config_kwargs, args):
    params = OrderedDict()
    for key, tensor in template_state.A_set.items():
        params[key] = torch.nn.Parameter(tensor._data.detach().clone())

    line_search = None if args.line_search == "none" else args.line_search
    optimizer = LBFGS_MOD(
        list(params.values()),
        max_iter=1,
        lr=args.lr,
        tolerance_grad=1.0e-5,
        tolerance_change=1.0e-9,
        history_size=args.history_size,
        line_search_fn=line_search,
        line_search_eps=1.0e-8,
    )

    best_energy = numpy.inf
    start = time.perf_counter()

    def grad_norm():
        total = 0.0
        for param in params.values():
            if param.grad is not None:
                total += float(torch.sum(torch.abs(param.grad) ** 2).detach().cpu())
        return total ** 0.5

    def closure(linesearching=False):
        optimizer.zero_grad()
        state = make_state_from_params(template_state, params)
        loss, _ctm = cost_fun(parameters, state, ad_ctm_args, None, global_args, config_kwargs)
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(params.values()), args.max_grad_norm)
        norm = grad_norm()
        print("norm of grad:" + str(norm))
        if not numpy.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm")
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        state = make_state_from_params(template_state, params)
        loss, _ctm = cost_fun(parameters, state, ls_ctm_args, None, global_args, config_kwargs)
        return loss

    for step in range(1, args.maxiter + 1):
        step_start = time.perf_counter()
        loss_tensor = optimizer.step_2c(closure, closure_linesearch)
        step_s = time.perf_counter() - step_start
        loss_value = to_float(loss_tensor)
        post_energy_value = float("nan")
        with torch.no_grad():
            post_state = make_state_from_params(template_state, params)
            post_energy, _ctm = cost_fun(parameters, post_state, ls_ctm_args, None, global_args, config_kwargs)
            post_energy_value = to_float(post_energy)
            if post_energy_value < best_energy:
                best_energy = post_energy_value
                save_state = make_state_from_params(template_state, params, detach=True)
                save_square_iPEPS(save_state.A_set, config_kwargs["save_file_prefix"], config_kwargs)
        elapsed = time.perf_counter() - start
        print(
            f"step,{step},loss,{loss_value:.16g},post_energy,{post_energy_value:.16g},"
            f"grad_norm,{grad_norm():.16g},step_s,{step_s:.6g},elapsed_s,{elapsed:.6g}"
        )
        if args.target_energy is not None and post_energy_value <= args.target_energy:
            print(
                f"target reached: E={post_energy_value}, target={args.target_energy}, "
                f"step={step}, elapsed_s={elapsed:.6g}"
            )
            break

    final_state = make_state_from_params(template_state, params, detach=True)
    save_square_iPEPS(final_state.A_set, config_kwargs["save_file_prefix"], config_kwargs)
    return final_state


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--Lx", type=int, default=1)
    parser.add_argument("--Ly", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--ad-ctm-iters", type=int, default=4)
    parser.add_argument("--ls-ctm-iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--instate-prefix", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--step0", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--ls-maxiter", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--line-search", choices=("backtracking", "strong_wolfe", "none"), default="backtracking")
    parser.add_argument("--optimizer", choices=("old", "lbfgsmod"), default="old")
    parser.add_argument("--target-energy", type=float, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    save_prefix = args.out_prefix
    if save_prefix is None:
        save_prefix = (
            f"data_square_CSL/random_noC4PT_smoke_D{args.D}_chi{args.chi}_"
            f"Lx{args.Lx}_Ly{args.Ly}_seed{args.seed}"
        )

    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": args.Lx,
        "Ly": args.Ly,
        "checkerboard_spin_transform": "sigmay",
        "save_file_prefix": save_prefix,
    }

    global_args = GLOBALARGS()
    global_args.Lx = args.Lx
    global_args.Ly = args.Ly
    global_args.device = args.device

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    if args.instate_prefix is None:
        A_set = random_square_iPEPS(args.D, 2, config_kwargs)
    else:
        A_set = load_square_iPEPS(args.instate_prefix, config_kwargs)

    state = IPEPS_SQUARE(A_set, config_kwargs)
    state.require_grad(False)
    state.to_device(args.device)
    state.normalize()

    ad_ctm_args = CTMARGS()
    ad_ctm_args.chi = args.chi
    ad_ctm_args.CTM_ite_nums = args.ad_ctm_iters
    ad_ctm_args.CTM_trun_tol = 1.0e-8
    ad_ctm_args.CTM_conv_tol = 0.0
    ad_ctm_args.CTM_ite_info = False
    ad_ctm_args.doublelayer_on_cpu = False
    ad_ctm_args.use_sub_checkpoint = False

    ls_ctm_args = CTMARGS()
    ls_ctm_args.chi = args.chi
    ls_ctm_args.CTM_ite_nums = args.ls_ctm_iters
    ls_ctm_args.CTM_trun_tol = 1.0e-8
    ls_ctm_args.CTM_conv_tol = 1.0e-6
    ls_ctm_args.CTM_ite_info = False
    ls_ctm_args.doublelayer_on_cpu = False
    ls_ctm_args.use_sub_checkpoint = False

    ls = LINESEARCH()
    ls.method = "lbfgs"
    ls.line_search = "backtracking"
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
        f"random no-C4/PT smoke opt: D={args.D}, chi={args.chi}, Lx={args.Lx}, Ly={args.Ly}, "
        f"maxiter={args.maxiter}, ad_ctm_iters={args.ad_ctm_iters}, "
        f"ls_ctm_iters={args.ls_ctm_iters}, seed={args.seed}, "
        f"max_grad_norm={args.max_grad_norm}, step0={args.step0}, "
        f"lr={args.lr}, ls_maxiter={args.ls_maxiter}, history_size={args.history_size}, "
        f"line_search={args.line_search}, optimizer={args.optimizer}, "
        f"target_energy={args.target_energy}, save_prefix={save_prefix}"
    )
    if args.optimizer == "lbfgsmod":
        optimize_with_lbfgsmod(parameters, args.D, args.chi, state, ad_ctm_args, ls_ctm_args, global_args, config_kwargs, args)
    else:
        optimize_bosonic_square_iPEPS(
            parameters,
            args.D,
            args.chi,
            state,
            ad_ctm_args,
            ls_ctm_args,
            None,
            global_args,
            config_kwargs,
            ls,
        )
    elapsed = time.perf_counter() - start
    print("elapsed_wall_s:", "{:.6g}".format(elapsed))


if __name__ == "__main__":
    main()
