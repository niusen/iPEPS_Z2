"""Optimize J1-Jchi with a d=2 Z2-symmetric fermionic triangular iPESS.

The initial JSON must first be created by
``project_gutzwiller_d4_to_d2.py``.  This program never uses the d=4 Fock
space and never performs projection during optimization.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HERE = os.path.dirname(os.path.abspath(__file__))

# Absolute source-code directory on the server.
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

import torch

from ansatz.fermionic_spin_triangle_iPESS import (
    add_fermionic_spin_noise,
    fermionic_spin_bond_dimension,
    load_fermionic_spin_triangle_iPESS,
    make_fermionic_spin_state,
    save_fermionic_spin_triangle_iPESS,
)
from config.config import CTMARGS, GLOBALARGS, LINESEARCH, Square_Hubbard_Energy_settings
from optimization.stochastic_opt import optimize_iPESS


########################
# Hamiltonian parameters
########################
J1 = 1.0
Jchi = 0.4
parameters = {"J1": J1, "Jchi": Jchi}


########################
# iPESS and CTMRG parameters
########################
Lx = 2
Ly = 2
D = 8
chi = 80

AD_ctm_ite_nums = 10
LS_ctm_ite_nums = 50
CTM_trun_tol = 1.0e-8
CTM_conv_tol = 1.0e-6
AD_doublelayer_on_cpu = True
LS_doublelayer_on_cpu = False
AD_use_sub_checkpoint = True
LS_use_sub_checkpoint = False


########################
# Optimization parameters
########################
# "lbfgs" and "cg" use line_search_method. "stochastic" does not.
opt_method = "lbfgs"
# For L-BFGS/CG: "hager_zhang" uses trial gradients; "backtracking" uses
# energy-only Armijo trials and is cheaper per line-search evaluation.
line_search_method = "hager_zhang"
maxiter = 1000
gtol = 1.0e-3
max_grad_norm = None
step0 = 1.0
ls_maxiter = 10
history_size = 8          # L-BFGS only
cg_beta = "PRP"           # CG only: "PRP" or "FR"
stochastic_delta0 = 1.0e-3
stochastic_alpha = 3.0 / 4.0


########################
# Runtime and state I/O
########################
device = "cuda:1"
default_dtype = "complex128"
n_cpu = 10
seed = 1234
Noise = 0.0

# d=2 file produced by project_gutzwiller_d4_to_d2.py; no .json suffix.
initial_state_file = "Z2_spin_Gutzwiller_D8_Lx2_Ly2"


def parameter_tag(value):
    text = format(value, ".12g") if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


save_file_name = (
    "triangle_spin_fermionic"
    + "_J1_" + parameter_tag(J1)
    + "_Jchi_" + parameter_tag(Jchi)
    + "_D" + str(D)
    + "_chi" + str(chi)
    + "_Lx" + str(Lx)
    + "_Ly" + str(Ly)
)


def show_settings(title, settings):
    print(title + ":")
    for name, value in settings.items():
        print("  " + name + " = " + repr(value))


def main():
    print("pid= " + str(os.getpid()))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    show_settings("Hamiltonian parameters", parameters)
    show_settings(
        "iPESS and CTMRG parameters",
        {
            "Lx": Lx, "Ly": Ly, "D": D, "physical_dim": 2, "chi": chi,
            "AD_ctm_ite_nums": AD_ctm_ite_nums,
            "LS_ctm_ite_nums": LS_ctm_ite_nums,
            "CTM_trun_tol": CTM_trun_tol,
            "CTM_conv_tol": CTM_conv_tol,
            "AD_doublelayer_on_cpu": AD_doublelayer_on_cpu,
            "LS_doublelayer_on_cpu": LS_doublelayer_on_cpu,
            "AD_use_sub_checkpoint": AD_use_sub_checkpoint,
            "LS_use_sub_checkpoint": LS_use_sub_checkpoint,
        },
    )
    optimization_settings = {
        "opt_method": opt_method, "maxiter": maxiter,
        "gtol": gtol, "max_grad_norm": max_grad_norm,
    }
    if opt_method == "lbfgs":
        optimization_settings.update(
            line_search_method=line_search_method, step0=step0,
            ls_maxiter=ls_maxiter, history_size=history_size,
        )
    elif opt_method == "cg":
        optimization_settings.update(
            line_search_method=line_search_method, step0=step0,
            ls_maxiter=ls_maxiter, cg_beta=cg_beta,
        )
    elif opt_method == "stochastic":
        optimization_settings.update(
            stochastic_delta0=stochastic_delta0,
            stochastic_alpha=stochastic_alpha,
        )
    else:
        raise ValueError("unknown opt_method: " + str(opt_method))
    show_settings("Optimization parameters", optimization_settings)
    initial_prefix = (
        initial_state_file if os.path.isabs(initial_state_file)
        else os.path.join(HERE, initial_state_file)
    )
    final_prefix = os.path.join(HERE, save_file_name)
    show_settings(
        "Runtime and state I/O",
        {
            "source_code_dir": source_code_dir,
            "device": device, "default_dtype": default_dtype,
            "n_cpu": n_cpu, "seed": seed, "Noise": Noise,
            "initial_state_file": initial_prefix + ".json",
            "save_file_name": final_prefix + ".json",
        },
    )

    torch.set_num_threads(n_cpu)
    torch.manual_seed(seed)
    config_kwargs = {
        "backend": "torch",
        "default_dtype": default_dtype,
        "default_device": device,
        "Lx": Lx,
        "Ly": Ly,
        "save_file_prefix": final_prefix,
    }
    global_args = GLOBALARGS()
    global_args.Lx, global_args.Ly, global_args.device = Lx, Ly, device

    B_set, T_set = load_fermionic_spin_triangle_iPESS(
        initial_prefix, config_kwargs
    )
    state = make_fermionic_spin_state(B_set, T_set, config_kwargs)
    actual_D = fermionic_spin_bond_dimension(state)
    if actual_D != D:
        raise ValueError(
            "projected state has D=" + str(actual_D)
            + ", but this run requests D=" + str(D)
        )
    print("actual optimization bond dimension: D=" + str(actual_D))
    state = add_fermionic_spin_noise(state, Noise)
    state.require_grad(False)
    state.to_device(device)
    state.normalize()

    energy_setting = Square_Hubbard_Energy_settings()
    energy_setting.model = "triangle_spin_J1_Jchi_fermionic_d2"

    AD_ctm_args = CTMARGS()
    AD_ctm_args.CTM_ite_info = True
    AD_ctm_args.chi = chi
    AD_ctm_args.CTM_ite_nums = AD_ctm_ite_nums
    AD_ctm_args.CTM_trun_tol = CTM_trun_tol
    AD_ctm_args.CTM_conv_tol = CTM_conv_tol
    AD_ctm_args.doublelayer_on_cpu = AD_doublelayer_on_cpu
    AD_ctm_args.use_sub_checkpoint = AD_use_sub_checkpoint
    print(AD_ctm_args)

    ls_ctm_args = CTMARGS()
    ls_ctm_args.CTM_ite_info = False
    ls_ctm_args.chi = chi
    ls_ctm_args.CTM_ite_nums = LS_ctm_ite_nums
    ls_ctm_args.CTM_trun_tol = CTM_trun_tol
    ls_ctm_args.CTM_conv_tol = CTM_conv_tol
    ls_ctm_args.doublelayer_on_cpu = LS_doublelayer_on_cpu
    ls_ctm_args.use_sub_checkpoint = LS_use_sub_checkpoint
    print(ls_ctm_args)

    ls = LINESEARCH()
    ls.method, ls.maxiter, ls.gtol = opt_method, maxiter, gtol
    ls.max_grad_norm = max_grad_norm
    if opt_method == "lbfgs":
        ls.line_search, ls.step0 = line_search_method, step0
        ls.ls_maxiter, ls.history_size = ls_maxiter, history_size
    elif opt_method == "cg":
        ls.line_search, ls.step0 = line_search_method, step0
        ls.ls_maxiter, ls.cg_beta = ls_maxiter, cg_beta
    elif opt_method == "stochastic":
        ls.delta0, ls.alpha = stochastic_delta0, stochastic_alpha

    optimized_state = optimize_iPESS(
        parameters, D, chi, state, AD_ctm_args, ls_ctm_args,
        energy_setting, global_args, config_kwargs, ls,
    )
    optimized_state.normalize()
    save_fermionic_spin_triangle_iPESS(
        optimized_state.B_set, optimized_state.T_set,
        final_prefix, config_kwargs,
    )
    print("saved final d=2 state:", final_prefix + ".json")


if __name__ == "__main__":
    main()
