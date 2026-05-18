import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS, random_square_iPEPS, save_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import evaluate_ob_cell_iPEPS, prl_129_177201_square_csl_parameters
from optimization.lbfgs_modified import LBFGS_MOD
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables


def to_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def format_float(x):
    return "{:.16g}".format(to_float(x))


def make_state_from_param(template_tensor, param, config_kwargs):
    A = template_tensor._replace(data=param)
    state = IPEPS_SQUARE_C4_PT(OrderedDict({"1,1": A}), config_kwargs, impose_c4=True, impose_pt=True)
    state.normalize()
    return state


def evaluate_energy(parameters, state, ctm_args, global_args, config_kwargs):
    init = INITCTMARGS()
    CTM_cell, double_A_cell, _ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, init, None, ctm_args, global_args
    )
    double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
    E_total, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    E_total = torch.real(E_total)
    print("E=" + format_float(E_total))
    return E_total, ite_err


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--ctm-iters", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--instate-prefix", default=None)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", choices=("backtracking", "strong_wolfe", "none"), default="backtracking")
    parser.add_argument("--target-energy", type=float, default=None)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--ctm-trun-tol", type=float, default=1.0e-8)
    parser.add_argument("--projector-min-singular-cutoff", type=float, default=1.0e-8)
    parser.add_argument("--svd-ad-decomp-reg", type=float, default=1.0e-8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    save_prefix = args.out_prefix
    if save_prefix is None:
        save_prefix = f"general_C4PT_ansatz_lbfgsmod_D{args.D}_chi{args.chi}_seed{args.seed}"

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
        init_source = "random_seed_" + str(args.seed)
    else:
        A_set = load_square_iPEPS(args.instate_prefix, config_kwargs)
        init_source = args.instate_prefix + ".json"

    initial_state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    initial_state.to_device(args.device)
    initial_state.normalize()
    template_tensor = initial_state.A_set["1,1"].copy()
    param = torch.nn.Parameter(initial_state.A_set["1,1"]._data.detach().clone())

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_iters
    ctm_args.CTM_trun_tol = args.ctm_trun_tol
    ctm_args.CTM_conv_tol = args.ctm_conv_tol
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False
    ctm_args.projector_min_singular_cutoff = args.projector_min_singular_cutoff
    ctm_args.svd_ad_decomp_reg = args.svd_ad_decomp_reg

    print(
        f"ctm params: trun_tol={ctm_args.CTM_trun_tol}, "
        f"projector_min_singular_cutoff={ctm_args.projector_min_singular_cutoff}, "
        f"svd_ad_decomp_reg={ctm_args.svd_ad_decomp_reg}"
    )

    line_search = None if args.line_search == "none" else args.line_search
    optimizer = LBFGS_MOD(
        [param],
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
    print(
        f"C4/PT LBFGS_MOD opt: D={args.D}, chi={args.chi}, maxiter={args.maxiter}, "
        f"ctm_iters={args.ctm_iters}, ctm_conv_tol={args.ctm_conv_tol}, seed={args.seed}, "
        f"lr={args.lr}, history_size={args.history_size}, line_search={args.line_search}, "
        f"max_grad_norm={args.max_grad_norm}, target_energy={args.target_energy}, "
        f"save_prefix={save_prefix}, init_source={init_source}"
    )

    def closure(linesearching=False):
        optimizer.zero_grad()
        state = make_state_from_param(template_tensor, param, config_kwargs)
        loss, _ite_err = evaluate_energy(parameters, state, ctm_args, global_args, config_kwargs)
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([param], args.max_grad_norm)
        grad_norm = param.grad.norm().item() if param.grad is not None else float("nan")
        if not numpy.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        state = make_state_from_param(template_tensor, param, config_kwargs)
        loss, _ite_err = evaluate_energy(parameters, state, ctm_args, global_args, config_kwargs)
        return loss

    for step in range(1, args.maxiter + 1):
        step_start = time.perf_counter()
        loss_tensor = optimizer.step_2c(closure, closure_linesearch)
        step_s = time.perf_counter() - step_start
        loss_value = to_float(loss_tensor)
        grad_norm = param.grad.norm().item() if param.grad is not None else float("nan")
        post_energy_value = float("nan")

        if args.check_every > 0 and (step % args.check_every == 0 or step == args.maxiter):
            with torch.no_grad():
                post_state = make_state_from_param(template_tensor, param, config_kwargs)
                post_energy, ite_err = evaluate_energy(parameters, post_state, ctm_args, global_args, config_kwargs)
                post_energy_value = to_float(post_energy)
                if post_energy_value < best_energy:
                    best_energy = post_energy_value
                    save_square_iPEPS(post_state.A_set, save_prefix, config_kwargs)

        elapsed = time.perf_counter() - start
        print(
            f"step,{step},loss,{loss_value:.16g},post_energy,{post_energy_value:.16g},"
            f"grad_norm,{grad_norm:.16g},step_s,{step_s:.6g},elapsed_s,{elapsed:.6g}"
        )
        if args.target_energy is not None and post_energy_value <= args.target_energy:
            print(
                f"target_reached,step,{step},post_energy,{post_energy_value:.16g},"
                f"target,{args.target_energy:.16g},elapsed_s,{elapsed:.6g}"
            )
            break

    with torch.no_grad():
        final_state = make_state_from_param(template_tensor, param, config_kwargs)
        final_energy, _ite_err = evaluate_energy(parameters, final_state, ctm_args, global_args, config_kwargs)
        save_square_iPEPS(final_state.A_set, save_prefix, config_kwargs)
    elapsed = time.perf_counter() - start
    print("final_energy:", format_float(final_energy))
    print("saved_square_state:", save_prefix + ".json")
    print("elapsed_wall_s:", "{:.6g}".format(elapsed))


if __name__ == "__main__":
    main()
