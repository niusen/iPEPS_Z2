"""Server driver for the triangular spin J1-Jchi model.

Edit the parameters in this file, then submit it with
``run_triangle_spin_server.sh``.  This intentionally follows the original
hard-coded workflow used by ``stochastic_opt_iPESS.py``.
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HERE = os.path.dirname(os.path.abspath(__file__))

# Absolute path of the source-code package on the server.
source_code_dir = "/home/sniu/python_code/iPESS_triangle_spin"
sys.path.insert(0, source_code_dir)

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
D = 4
physical_dim = 2
chi = 40

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
# Choose one optimizer:
#   "lbfgs": limited-memory BFGS. Uses line_search_method and history_size.
#   "cg": nonlinear conjugate gradient. Uses line_search_method and cg_beta.
#   "stochastic": random-direction search. Does NOT use line_search_method;
#                 it uses stochastic_delta0 and stochastic_alpha instead.
opt_method = "lbfgs"

# Used only by opt_method="lbfgs" or "cg":
#   "hager_zhang": Wolfe-type search; evaluates trial-point gradients and is
#                   usually the more robust choice, but each trial is costly.
#   "backtracking": Armijo energy-only search; cheaper per trial.
line_search_method = "hager_zhang"

# Common stopping criteria.
maxiter = 1000
gtol = 1.0e-3
max_grad_norm = None

# Used by Hager-Zhang and backtracking searches for CG/L-BFGS.
step0 = 1.0
ls_maxiter = 10

# L-BFGS-only parameter.
history_size = 8

# CG-only parameter: "PRP" or "FR".
cg_beta = "PRP"

# Stochastic-only parameters.
stochastic_delta0 = 1.0e-3
stochastic_alpha = 3.0 / 4.0


########################
# Runtime and state I/O
########################
device = "cuda:1"
default_dtype = "complex128"
n_cpu = 10
seed = 1234
# Relative noise applied after loading, random initialization, or D expansion.
Noise = 0.0

load_initial_state = False
initial_state_file = "initial_triangle_spin_D6"


def parameter_tag(value):
    """Format a numeric parameter as a filesystem-safe, readable tag."""
    text = format(value, ".12g") if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


save_file_name = (
    "triangle_spin"
    + "_J1_" + parameter_tag(J1)
    + "_Jchi_" + parameter_tag(Jchi)
    + "_D" + str(D)
    + "_chi" + str(chi)
    + "_Lx" + str(Lx)
    + "_Ly" + str(Ly)
)


def show_settings(title, settings):
    """Print hard-coded inputs in a Julia ``show``-like readable form."""
    print(title + ":")
    for name, value in settings.items():
        print("  " + name + " = " + repr(value))


def main():
    print("pid= " + str(os.getpid()))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    show_settings(
        "Hamiltonian parameters",
        {
            "J1": J1,
            "Jchi": Jchi,
        },
    )
    show_settings(
        "iPESS and CTMRG parameters",
        {
            "Lx": Lx,
            "Ly": Ly,
            "D": D,
            "physical_dim": physical_dim,
            "chi": chi,
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
        "opt_method": opt_method,
        "maxiter": maxiter,
        "gtol": gtol,
        "max_grad_norm": max_grad_norm,
    }
    if opt_method == "lbfgs":
        optimization_settings.update(
            {
                "line_search_method": line_search_method,
                "step0": step0,
                "ls_maxiter": ls_maxiter,
                "history_size": history_size,
            }
        )
    elif opt_method == "cg":
        optimization_settings.update(
            {
                "line_search_method": line_search_method,
                "step0": step0,
                "ls_maxiter": ls_maxiter,
                "cg_beta": cg_beta,
            }
        )
    elif opt_method == "stochastic":
        optimization_settings.update(
            {
                "stochastic_delta0": stochastic_delta0,
                "stochastic_alpha": stochastic_alpha,
            }
        )
    else:
        raise ValueError("unknown opt_method: " + str(opt_method))
    show_settings("Optimization parameters", optimization_settings)
    show_settings(
        "Runtime and state I/O",
        {
            "device": device,
            "default_dtype": default_dtype,
            "source_code_dir": source_code_dir,
            "n_cpu": n_cpu,
            "seed": seed,
            "Noise": Noise,
            "load_initial_state": load_initial_state,
            "initial_state_file": initial_state_file,
            "save_file_name": save_file_name + ".json",
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
        "save_file_prefix": os.path.join(HERE, save_file_name),
    }
    global_args = GLOBALARGS()
    global_args.Lx = Lx
    global_args.Ly = Ly
    global_args.device = device

    if load_initial_state:
        initial_prefix = os.path.join(HERE, initial_state_file)
        B_set, T_set = load_bosonic_triangle_iPESS(initial_prefix, config_kwargs)
        state = IPESS_TRIANGLE_DENSE(B_set, T_set, config_kwargs)
        loaded_D = bosonic_triangle_iPESS_bond_dimension(state)
        print("loaded state bond dimension: D=" + str(loaded_D))
        if loaded_D < D:
            print(
                "expand state bond dimension: D=" + str(loaded_D)
                + " -> D=" + str(D)
            )
            state = expand_bosonic_triangle_iPESS(state, D)
        elif loaded_D > D:
            raise ValueError(
                "loaded state has D=" + str(loaded_D)
                + ", which is larger than requested D=" + str(D)
            )
    else:
        state = random_bosonic_triangle_iPESS(
            D, config_kwargs, physical_dim=physical_dim, seed=seed
        )
    actual_D = bosonic_triangle_iPESS_bond_dimension(state)
    if actual_D != D:
        raise RuntimeError(
            "actual state bond dimension D=" + str(actual_D)
            + " does not match requested D=" + str(D)
        )
    print("actual optimization bond dimension: D=" + str(actual_D))
    state = add_bosonic_noise(state, Noise)
    state.require_grad(False)
    state.to_device(device)
    state.normalize()

    energy_setting = Square_Hubbard_Energy_settings()
    energy_setting.model = "triangle_spin_J1_Jchi"

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
    ls.method = opt_method
    ls.maxiter = maxiter
    ls.gtol = gtol
    ls.max_grad_norm = max_grad_norm

    if opt_method == "lbfgs":
        ls.line_search = line_search_method
        ls.step0 = step0
        ls.ls_maxiter = ls_maxiter
        ls.history_size = history_size
    elif opt_method == "cg":
        ls.line_search = line_search_method
        ls.step0 = step0
        ls.ls_maxiter = ls_maxiter
        ls.cg_beta = cg_beta
    elif opt_method == "stochastic":
        ls.delta0 = stochastic_delta0
        ls.alpha = stochastic_alpha
    else:
        raise ValueError("unknown opt_method: " + str(opt_method))

    optimized_state = optimize_iPESS(
        parameters,
        D,
        chi,
        state,
        AD_ctm_args,
        ls_ctm_args,
        energy_setting,
        global_args,
        config_kwargs,
        ls,
    )
    optimized_state.normalize()
    final_prefix = os.path.join(HERE, save_file_name)
    save_bosonic_triangle_iPESS(
        optimized_state.B_set, optimized_state.T_set, final_prefix, config_kwargs
    )
    print("saved final state:", final_prefix + ".json")


if __name__ == "__main__":
    main()
