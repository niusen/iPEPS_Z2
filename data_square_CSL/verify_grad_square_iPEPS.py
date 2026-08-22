import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/home/sniu/python_code/iPEPS_Z2_test_codex/")
sys.path.append("D:/My Documents/Code/python_codes/iPEPS_Z2/")

import torch
import yastn

from ansatz.square_iPEPS import IPEPS_SQUARE, random_square_iPEPS
from config.config import CTMARGS, GLOBALARGS
from model.bosonic_square_ob_iPEPS import prl_129_177201_square_csl_parameters
from optimization.optimize_bosonic_square_iPEPS import cost_fun, get_grad


print("pid= " + str(os.getpid()))
torch.set_num_threads(4)

Lx = 2
Ly = 2
D = 3
d = 2
chi = 6
device = "cpu"
delta = 1e-6

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
ctm_args.doublelayer_on_cpu = False
ctm_args.use_sub_checkpoint = False

A_set = random_square_iPEPS(D, d, config_kwargs)
state = IPEPS_SQUARE(A_set, config_kwargs)
state.require_grad(False)
state.to_device(device)
state.normalize()

grad, E0, CTM = get_grad(parameters, state, ctm_args, None, global_args, config_kwargs)
print("E0:", E0)

coord = "1,1"
A0_1d, meta = yastn.Tensor.compress_to_1d(state.A_set[coord])
grad_1d, _ = yastn.Tensor.compress_to_1d(grad.A_set[coord])

test_indices = [0, min(1, len(A0_1d) - 1), min(3, len(A0_1d) - 1)]

for ind in test_indices:
    for part, step in (("real", delta), ("imag", 1j * delta)):
        A_plus = A0_1d.clone()
        A_minus = A0_1d.clone()
        A_plus[ind] = A_plus[ind] + step
        A_minus[ind] = A_minus[ind] - step

        state_plus = state.copy()
        state_minus = state.copy()
        state_plus.A_set[coord] = yastn.decompress_from_1d(A_plus, meta)
        state_minus.A_set[coord] = yastn.decompress_from_1d(A_minus, meta)

        E_plus, _ = cost_fun(parameters, state_plus, ctm_args, None, global_args, config_kwargs)
        E_minus, _ = cost_fun(parameters, state_minus, ctm_args, None, global_args, config_kwargs)
        finite_diff = (E_plus - E_minus) / (2 * delta)

        if part == "real":
            auto_grad = torch.real(grad_1d[ind])
        else:
            auto_grad = -torch.real(1j * grad_1d[ind])

        print(
            "index",
            ind,
            part,
            "finite_diff=",
            finite_diff.item(),
            "autograd=",
            auto_grad.item(),
            "abs_diff=",
            abs(finite_diff.item() - auto_grad.item()),
        )
