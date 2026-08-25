"""Optimize a dense spin-1/2 triangular iPESS for the J1-Jchi model.

Example:
    python triangle_spin_opt.py --Jchi 0.2 --D 6 --chi 72 --device cuda:0
"""

import argparse

import torch

from ansatz.bosonic_triangle_iPESS import (
    IPESS_TRIANGLE_DENSE,
    add_bosonic_noise,
    bosonic_triangle_iPESS_bond_dimension,
    expand_bosonic_triangle_iPESS,
    load_bosonic_triangle_iPESS,
    random_bosonic_triangle_iPESS,
    save_bosonic_triangle_iPESS,
)
from config.config import CTMARGS, GLOBALARGS, LINESEARCH, Square_Hubbard_Energy_settings
from ctmrg.CTMRG_unitcell_iPESS import CTMRG_cell_iPESS
from model.bosonic_spin_ob_iPESS import evaluate_triangle_spin_energy
from optimization.stochastic_opt import optimize_iPESS
from config.config import INITCTMARGS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--Jchi", type=float, default=0.0)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--Lx", type=int, default=2)
    parser.add_argument("--Ly", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--state", help="Dense iPESS JSON prefix to load (omit .json)")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--method", choices=("lbfgs", "cg", "stochastic"), default="lbfgs")
    parser.add_argument("--line-search", choices=("backtracking", "hager_zhang"), default="backtracking")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--gtol", type=float, default=1.0e-3)
    parser.add_argument("--step0", type=float, default=0.1)
    parser.add_argument("--ls-maxiter", type=int, default=6)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ad-ctm-iter", type=int, default=10)
    parser.add_argument("--ls-ctm-iter", type=int, default=50)
    parser.add_argument("--ctm-tol", type=float, default=1.0e-8)
    parser.add_argument("--save-prefix", default="triangle_spin")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config_kwargs = {
        "backend": "torch",
        "default_dtype": args.dtype,
        "default_device": args.device,
        "Lx": args.Lx,
        "Ly": args.Ly,
        "save_file_prefix": args.save_prefix,
    }
    global_args = GLOBALARGS()
    global_args.Lx, global_args.Ly = args.Lx, args.Ly
    global_args.device = args.device

    if args.state:
        B_set, T_set = load_bosonic_triangle_iPESS(args.state, config_kwargs)
        state = IPESS_TRIANGLE_DENSE(B_set, T_set, config_kwargs)
        loaded_D = bosonic_triangle_iPESS_bond_dimension(state)
        print("loaded state bond dimension: D=" + str(loaded_D))
        if loaded_D < args.D:
            state = expand_bosonic_triangle_iPESS(state, args.D)
        elif loaded_D > args.D:
            raise ValueError(
                "loaded state has D=" + str(loaded_D)
                + ", which is larger than requested D=" + str(args.D)
            )
    else:
        state = random_bosonic_triangle_iPESS(
            args.D, config_kwargs, physical_dim=2, seed=args.seed
        )
    actual_D = bosonic_triangle_iPESS_bond_dimension(state)
    if actual_D != args.D:
        raise RuntimeError(
            "actual state bond dimension D=" + str(actual_D)
            + " does not match requested D=" + str(args.D)
        )
    print("actual optimization bond dimension: D=" + str(actual_D))
    state.to_device(args.device)
    state = add_bosonic_noise(state, args.noise)
    state.normalize()
    state.require_grad(False)

    parameters = {"J1": args.J1, "Jchi": args.Jchi}
    energy_setting = Square_Hubbard_Energy_settings()
    energy_setting.model = "triangle_spin_J1_Jchi"

    AD_ctm_args = CTMARGS()
    AD_ctm_args.chi = args.chi
    AD_ctm_args.CTM_ite_nums = args.ad_ctm_iter
    AD_ctm_args.CTM_trun_tol = args.ctm_tol
    AD_ctm_args.CTM_ite_info = True

    ls_ctm_args = CTMARGS()
    ls_ctm_args.chi = args.chi
    ls_ctm_args.CTM_ite_nums = args.ls_ctm_iter
    ls_ctm_args.CTM_trun_tol = args.ctm_tol

    if args.evaluate_only:
        init = INITCTMARGS()
        CTM, double_B, double_T, ite_num, ite_err = CTMRG_cell_iPESS(
            state.B_set, state.T_set, init, None, ls_ctm_args, global_args
        )
        energy, observables = evaluate_triangle_spin_energy(
            parameters, state.B_set, state.T_set, double_B, double_T,
            CTM, config_kwargs, global_args, return_observables=True
        )
        print("CTMRG iterations:", ite_num, "error:", ite_err)
        print("energy per site:", energy.item())
        for name, values in observables.items():
            print(name + ":", values.tolist())
        return

    line_search = LINESEARCH()
    line_search.method = args.method
    line_search.line_search = args.line_search
    line_search.maxiter = args.maxiter
    line_search.gtol = args.gtol
    line_search.step0 = args.step0
    line_search.ls_maxiter = args.ls_maxiter
    line_search.history_size = args.history_size
    line_search.max_grad_norm = args.max_grad_norm
    optimized_state = optimize_iPESS(
        parameters, args.D, args.chi, state, AD_ctm_args, ls_ctm_args,
        energy_setting, global_args, config_kwargs, line_search
    )
    optimized_state.normalize()
    save_bosonic_triangle_iPESS(
        optimized_state.B_set, optimized_state.T_set, args.save_prefix, config_kwargs
    )
    print("saved final state:", args.save_prefix + ".json")


if __name__ == "__main__":
    main()
