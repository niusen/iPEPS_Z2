import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/home/sniu/python_code/iPEPS_Z2_test_codex/")
sys.path.append("D:/My Documents/Code/python_codes/iPEPS_Z2/")

import torch

from ansatz.square_iPEPS import IPEPS_SQUARE, random_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, LINESEARCH
from model.bosonic_square_ob_iPEPS import prl_129_177201_square_csl_parameters
from optimization.optimize_bosonic_square_iPEPS import cost_fun, get_grad, optimize_bosonic_square_iPEPS


print("pid= " + str(os.getpid()))
torch.set_num_threads(4)

Lx = 2
Ly = 2
D = 3
d = 2
chi = 6
device = "cpu"

parameters = prl_129_177201_square_csl_parameters(chirality_sign=1.0)
print(parameters)

config_kwargs = {"backend": "torch", "default_dtype": "complex128", "default_device": device, "Lx": Lx, "Ly": Ly}

global_args = GLOBALARGS()
global_args.Lx = Lx
global_args.Ly = Ly
global_args.device = device

ctm_args = CTMARGS()
ctm_args.chi = chi
ctm_args.CTM_ite_nums = 1
ctm_args.CTM_trun_tol = 1e-8
ctm_args.CTM_ite_info = True
ctm_args.doublelayer_on_cpu = False
ctm_args.use_sub_checkpoint = False

ls_ctm_args = CTMARGS()
ls_ctm_args.chi = chi
ls_ctm_args.CTM_ite_nums = 1
ls_ctm_args.CTM_trun_tol = 1e-8
ls_ctm_args.doublelayer_on_cpu = False
ls_ctm_args.use_sub_checkpoint = False

A_set = random_square_iPEPS(D, d, config_kwargs)
state = IPEPS_SQUARE(A_set, config_kwargs)
state.require_grad(False)
state.to_device(device)
state.normalize()

print("running cost_fun")
E, CTM = cost_fun(parameters, state, ctm_args, None, global_args, config_kwargs)
print("cost OK:", E)

print("running get_grad")
grad, E_grad, CTM_grad = get_grad(parameters, state, ctm_args, None, global_args, config_kwargs)
print("grad OK:", grad.norm())

print("running one optimizer iteration")
ls = LINESEARCH()
ls.method = "lbfgs"
ls.line_search = "backtracking"
ls.maxiter = 1
ls.ls_maxiter = 1
ls.gtol = 1e-12
ls.step0 = 1e-3
ls.print_observables = False

state_opt = optimize_bosonic_square_iPEPS(parameters, D, chi, state, ctm_args, ls_ctm_args, None, global_args, config_kwargs, ls)
print("optimizer OK:", state_opt.norm())
