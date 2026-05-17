import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

sys.path.append("/home/sniu/python_code/iPEPS_Z2_test_codex/")
sys.path.append("D:/My Documents/Code/python_codes/iPEPS_Z2/")

import numpy
import torch

from ansatz.square_iPEPS import (
    IPEPS_SQUARE,
    add_noise,
    load_square_iPEPS,
    random_square_iPEPS,
)
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS, LINESEARCH, OPTARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import (
    evaluate_ob_cell_iPEPS,
    evaluate_spin_ob_cell_iPEPS,
    prl_129_177201_square_csl_parameters,
)
from optimization.optimize_bosonic_square_iPEPS import optimize_bosonic_square_iPEPS, stochastic_opt


def to_real_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


pid = os.getpid()
print("pid= " + str(pid))
n_cpu = 20
torch.set_num_threads(n_cpu)


# PRL 129, 177201 (2022): J1=2cos(0.06pi)cos(0.14pi),
# J2=2cos(0.06pi)sin(0.14pi), lambda=2sin(0.06pi).
# The paper's i*lambda*(P_ijkl-P_ijkl^-1) equals 2*lambda times
# the sum of our four oriented scalar-chirality triangles.
parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
print(parameters)


Noise = 0.0
print("Noise=" + str(Noise))

Lx = 2
Ly = 2
D = 3
d = 2
chi = 40


AD_ctm_args = CTMARGS()
AD_ctm_args.CTM_ite_info = True
AD_ctm_args.chi = chi
AD_ctm_args.CTM_ite_nums = 10
AD_ctm_args.CTM_trun_tol = 1e-8
AD_ctm_args.doublelayer_on_cpu = True
AD_ctm_args.use_sub_checkpoint = True
print(AD_ctm_args)

ls_ctm_args = CTMARGS()
ls_ctm_args.CTM_ite_info = False
ls_ctm_args.chi = chi
ls_ctm_args.CTM_ite_nums = 50
ls_ctm_args.CTM_trun_tol = 1e-8
print(ls_ctm_args)

opt_args = OPTARGS()
init = INITCTMARGS()


# device: "cpu", "cuda", "cuda:0", ...
device = "cuda:0"
config_kwargs = {
    "backend": "torch",
    "default_dtype": "complex128",
    "default_device": device,
    "Lx": Lx,
    "Ly": Ly,
}

global_args = GLOBALARGS()
global_args.Lx = Lx
global_args.Ly = Ly
global_args.device = config_kwargs["default_device"]


load_initial_state = False
initial_state_file = "square_iPEPS_D" + str(D) + "_chi" + str(chi)

if load_initial_state:
    A_set = load_square_iPEPS(initial_state_file, config_kwargs)
else:
    A_set = random_square_iPEPS(D, d, config_kwargs)

state = IPEPS_SQUARE(A_set, config_kwargs)
state = add_noise(state, Noise, config_kwargs)
state.require_grad(False)
state.to_device(config_kwargs["default_device"])
state.normalize()

A_set = state.A_set
print(A_set["1,1"].requires_grad)
print("A[1,1] device:", A_set["1,1"].device)


run_initial_observables = False
if run_initial_observables:
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(A_set, init, None, ls_ctm_args, global_args)
    print("CTM ite_num=" + str(ite_num) + ", ite_err=" + str(ite_err))
    E_total, SS_x_set, SS_y_set, SS_diag_a_set, SS_diag_b_set, triangle_no_LU_set, triangle_no_LD_set, triangle_no_RD_set, triangle_no_RU_set = (
        evaluate_ob_cell_iPEPS(parameters, A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    )
    print("E=" + "{:.16g}".format(to_real_float(E_total)))
    print("SS_x:")
    print(SS_x_set)
    print("SS_y:")
    print(SS_y_set)
    print("SS_diag_a:")
    print(SS_diag_a_set)
    print("SS_diag_b:")
    print(SS_diag_b_set)
    print("chirality no_LU/no_LD/no_RD/no_RU:")
    print(triangle_no_LU_set)
    print(triangle_no_LD_set)
    print(triangle_no_RD_set)
    print(triangle_no_RU_set)
    print(evaluate_spin_ob_cell_iPEPS(A_set, double_A_cell, CTM_cell, config_kwargs, global_args))


opt_method = "lbfgs"  # options: "stochastic", "cg", "lbfgs"

ls = LINESEARCH()
ls.method = opt_method
ls.maxiter = 100
ls.gtol = 1e-5
ls.print_observables = True
ls.max_grad_norm = 1.0

if opt_method == "stochastic":
    ls.delta0 = 1e-3
    ls.alpha = 3 / 4
    stochastic_opt(parameters, D, chi, state, AD_ctm_args, ls_ctm_args, None, global_args, config_kwargs, ls)
elif opt_method == "cg":
    ls.cg_beta = "PRP"  # options: "PRP", "FR"
    ls.line_search = "hager_zhang"  # options: "hager_zhang", "backtracking"
    optimize_bosonic_square_iPEPS(parameters, D, chi, state, AD_ctm_args, ls_ctm_args, None, global_args, config_kwargs, ls)
elif opt_method == "lbfgs":
    ls.line_search = "hager_zhang"  # options: "hager_zhang", "backtracking"
    optimize_bosonic_square_iPEPS(parameters, D, chi, state, AD_ctm_args, ls_ctm_args, None, global_args, config_kwargs, ls)
else:
    raise ValueError("unknown optimization method: " + str(opt_method))
